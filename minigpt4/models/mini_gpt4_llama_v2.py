import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from minigpt4.common.registry import registry
from minigpt4.models.blip2 import Blip2Base, disabled_train
# from minigpt4.models.modeling_llama_v2 import LlamaForCausalLM as llm_model
# minigpt4.models.modeling_mistral import MistralForCausalLM as llm_model
from minigpt4.conversation.conversation import Conversation, SeparatorStyle, StoppingCriteriaList, StoppingCriteriaSub

from transformers import LlamaTokenizer
from transformers import BitsAndBytesConfig
from transformers import AutoConfig, AutoTokenizer
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
import time
import json
import numpy as np
import os
from transformers import PretrainedConfig
from transformers import PreTrainedModel
from typing import List
class minigpt4_video_config(PretrainedConfig):
    model_type="minigpt4_video"
    PRETRAINED_MODEL_CONFIG_DICT = {
        "minigpt4_video": "configs/models/minigpt4.yaml",
    }
    def __init__(
        self,
        omg_config:dict = {},
        **kwargs,
    ):
        for key, value in omg_config.items():
            setattr(self, key, value)
        super().__init__(**kwargs)
        
    # def to_dict(self):
    #     output = super().to_dict()
    #     return output
        
@registry.register_model("mini_gpt4_llama_v2")
class MiniGPT4_Video(Blip2Base, PreTrainedModel):
    """
    BLIP2 GPT-LLAMA model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "minigpt4_video": "minigpt4/configs/models/minigpt4.yaml",
    }
    config_class=minigpt4_video_config

    def __init__(
        self,
        cfg={},
    ):
        ## loop through the config minigpt4_video_config object and set the attributes
        if isinstance(cfg, minigpt4_video_config):
            cfg = cfg.to_dict()
        
        for key, value in cfg.items():
            try:
                setattr(self, key, value)
            except:
                print(f"Error setting attribute {key} with value {value}")
        PreTrainedModel.__init__(self, minigpt4_video_config(cfg))
        Blip2Base.__init__(self)
        if "Mistral" in self.llama_model:
            from minigpt4.models.modeling_mistral import MistralForCausalLM as llm_model
            print("Mistral model")
            self.model_type = "Mistral"
        else:
            from minigpt4.models.modeling_llama_v2 import LlamaForCausalLM as llm_model
            print("Llama model")
            self.model_type = "Llama"
        self.tokenizer = self.init_tokenizer()

        print("token pooling", self.token_pooling)
        if self.freeze_vit:
            # self.vit_precision="fp32"
            print("vit precision", self.vit_precision)
            self.visual_encoder, self.ln_vision = self.init_vision_encoder(
                self.vit_model, self.img_size, self.drop_path_rate, self.use_grad_checkpoint, self.vit_precision
            )
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train
            logging.info("freeze vision encoder")
            print("freeze the vision encoder")

        else:
            self.vit_precision="fp32"
            self.visual_encoder, self.ln_vision = self.init_vision_encoder(
                self.vit_model, self.img_size, self.drop_path_rate, self.use_grad_checkpoint, self.vit_precision
            )

            print("unfreeze the vision encoder")
        print('Loading VIT Done')

        print('Loading LLAMA')
        
        self.B_SYS, self.E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        token=os.environ.get("HF_TKN")
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(self.llama_model,use_fast=False,token=token)  #
        self.llama_tokenizer.pad_token = "$$"
        print("self.low_resource",self.low_resource)
        if self.low_resource:
            self.llama_model = llm_model.from_pretrained(
                self.llama_model,
                torch_dtype=torch.float16,
                # torch_dtype = torch.bfloat16,
                load_in_8bit=True,
                # device_map = "balanced"
                # device_map="auto",
                # device_map={'':torch.cuda.current_device()},token=token
                device_map={'':f"cuda:{self.minigpt4_gpu_id}"},token=token
                
            )
        else:
            self.llama_model = llm_model.from_pretrained(
                self.llama_model,
                torch_dtype=torch.float16,token=token
            )
            
        # self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
        self.llama_model = prepare_model_for_int8_training(self.llama_model)
        loraconfig = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self.lora_target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.llama_model = get_peft_model(self.llama_model, loraconfig)

        self.llama_model.print_trainable_parameters()

        if self.use_grad_checkpoint_llm:
            self.llama_model.gradient_checkpointing_enable()
       
        print('Loading LLAMA Done')


        if self.token_pooling:
            self.llama_proj = nn.Linear(
                1408*4, self.llama_model.config.hidden_size
            )
        else:
            self.llama_proj = nn.Linear(
                1408, self.llama_model.config.hidden_size
            )
        if self.prompt_path:
            with open(self.prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt]
            self.prompt_list = [self.prompt_template.format(p) for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []

    def encode_img(self, image):
        device = image.device
        if len(image.shape) > 4: 
            image = image.reshape(-1, *image.shape[-3:]) # for video input flatten the batch and time dimension (4,50,3,224,224) -> (200,3,224,224)
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device) # (200,3,224,224) -> (200,257,1408)
            image_embeds = image_embeds[:,1:,:] # remove the first token (CLS) (200,256,1408)
            bs, pn, hs = image_embeds.shape
            if self.token_pooling: # concat the each 4 tokens into one token (200,64,5632)
                image_embeds = image_embeds.view(bs, int(pn/4), int(hs*4)) # (200,64,5632)

            inputs_llama = self.llama_proj(image_embeds) # project to llama input size (200,64,5632) -> (200,64,4096)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
        return inputs_llama, atts_llama

    def get_context_emb(self, prompt, img_list):
        img_device = img_list[0].device
        prompt_segs = prompt.split('<ImageHere>')
        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
        seg_tokens = [
            self.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i==0).to(img_device).input_ids  # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]

        seg_embs = [self.embed_tokens(seg_t) for seg_t in seg_tokens]

        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]

        mixed_embs = torch.cat(mixed_embs, dim=1)
        
        return mixed_embs

    def prompt_wrap(self, img_embeds, atts_img, prompts, lengths=None):
        if prompts is None or len(prompts) == 0:
            # prompts is not provided, just return the original image embedding
            return img_embeds, atts_img
        elif img_embeds is None:
            # prompt is provided but there is no image embedding. return the prompt embedding in right padding
            self.llama_tokenizer.padding_side = "right"
            prompt_tokens = self.llama_tokenizer(
                prompts,
                return_tensors="pt",
                padding="max_length",
                add_special_tokens=False
            ).to(self.device)
            prompt_embeds = self.embed_tokens(prompt_tokens.input_ids)
            atts_prompt = prompt_tokens.attention_mask
            return prompt_embeds, atts_prompt

        else:
            # return the multi-modal embedding in right padding
            emb_lists = []
            if type(prompts) == str:
                prompts = [prompts] * len(img_embeds)
            for idx, (each_img_embed, each_prompt) in enumerate(zip(img_embeds, prompts)):
                pn = each_img_embed.shape[-2]
                if lengths is not None:
                    each_img_embed = each_img_embed.reshape(-1, each_img_embed.shape[-1])
                    each_img_embed = each_img_embed[:lengths[idx] * pn]

                p_segs = each_prompt.split('<ImageHere>')
                interleave_emb = []
                for idx, seg in enumerate(p_segs[:-1]):
                    p_tokens = self.llama_tokenizer(seg, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
                    p_embed = self.embed_tokens(p_tokens.input_ids)
                    
                    interleave_emb.append(torch.cat([p_embed, each_img_embed[None][:, idx*pn:(idx+1)*pn]], dim=1))

                wrapped_emb = torch.cat(interleave_emb, dim=1)
                p_tokens = self.llama_tokenizer(p_segs[-1], return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
                p_embed = self.embed_tokens(p_tokens.input_ids)
                wrapped_emb = torch.cat([wrapped_emb,p_embed], dim=1)
                emb_lists.append(wrapped_emb)

            emb_lens = [emb.shape[1] for emb in emb_lists]
            pad_emb = self.embed_tokens(torch.tensor(self.llama_tokenizer.pad_token_id, device=img_embeds.device))

            # max_length = max(emb_lens) if max(emb_lens) < self.max_context_len else self.max_context_len
            max_length = self.max_context_len
            wrapped_embs = pad_emb.expand(len(emb_lens), max_length, -1).clone()
            wrapped_atts = torch.zeros([len(emb_lens), max_length], dtype=torch.int, device=img_embeds.device)

            for i, emb in enumerate(emb_lists):
                length = emb_lens[i] if emb_lens[i] < self.max_context_len else self.max_context_len
                wrapped_embs[i, :length] = emb[:, :length]
                wrapped_atts[i, :length] = 1

            return wrapped_embs, wrapped_atts

    def concat_emb_input_output(self, input_embs, input_atts, output_embs, output_atts):
        """
        Concatenate the batched input embedding and batched output embedding together.
        Both the input and the output embedding should be right padded.
        """

        input_lens = []
        cat_embs = []
        cat_atts = []

        for i in range(input_embs.size(0)):
            input_len = input_atts[i].sum()
            input_lens.append(input_len)

            cat_embs.append(
                torch.cat([
                    input_embs[i][:input_len],
                    output_embs[i],
                    input_embs[i][input_len:]
                ])
            )
            cat_atts.append(
                torch.cat([
                    input_atts[i][:input_len],
                    output_atts[i],
                    input_atts[i][input_len:]
                ])
            )

        cat_embs = torch.stack(cat_embs)
        cat_atts = torch.stack(cat_atts)
        return cat_embs, cat_atts, input_lens

    def get_conv_emb(self, conv_q, conv_a, conv_img):
        """concatenate conversation and make sure the model is only trained to regress the answer"""

        regress_embs_list = []
        targets_list = []

        batch_size = len(conv_q)
        for batch_idx in range(batch_size):
            questions, answers = conv_q[batch_idx], conv_a[batch_idx]
            assigned_imgs = conv_img[batch_idx]
            questions = [self.prompt_wrap(
                img_embeds=img,
                atts_img=None,
                prompts=[q],
                lengths=[img.shape[1]] if img is not None else None) for q, img in zip(questions, assigned_imgs)]
            q_embs = [emb for emb, _ in questions]

            answers = [self.llama_tokenizer(a, return_tensors="pt", add_special_tokens=False).to(self.device) for a in answers]
            cur_emb = []
            cur_target = []
            for i in range(len(questions)):
                cur_emb.append(q_embs[i])
                cur_target.append(torch.ones_like(q_embs[i][..., 0], dtype=torch.int) * -100)

                cur_emb.append(self.embed_tokens(answers[i].input_ids))
                cur_target.append(answers[i].input_ids)

            cur_emb = torch.cat(cur_emb, dim=1)
            cur_target = torch.cat(cur_target, dim=1)

            regress_embs_list.append(cur_emb)
            targets_list.append(cur_target)

        max_len = min(max([target.shape[1] for target in targets_list]), self.max_txt_len)

        regress_embeds = torch.zeros([batch_size, max_len, cur_emb.shape[-1]], device=self.device)
        regress_attn = torch.zeros([batch_size, max_len], dtype=torch.int, device=self.device)
        targets = torch.ones([batch_size, max_len], dtype=torch.long, device=self.device) * -100

        for batch_idx in range(batch_size):
            cur_len = regress_embs_list[batch_idx].shape[1]
            regress_embeds[batch_idx, :cur_len] = regress_embs_list[batch_idx][0, :max_len]
            regress_attn[batch_idx, :cur_len] = 1
            targets[batch_idx, :cur_len] = targets_list[batch_idx][0, :max_len]

        return regress_embeds, regress_attn, targets

    def preparing_embedding(self, samples):
        def remove_special_tokens(data):
            
            # if "instruction_input" in data:
            data = [instruct.replace(" [caption]","") for instruct in data]
            data = [instruct.replace(" [vqa]","") for instruct in data]
            data = [instruct.replace(" [grounding]","") for instruct in data]
            data = [instruct.replace(" [identify]","") for instruct in data]
            data = [instruct.replace(" [refer]","") for instruct in data]
            return data

        ### prepare input tokens
        if 'image' in samples:
            img_embeds, img_atts = self.encode_img(samples["image"])
        else:
            img_embeds = img_atts = None

        if 'conv_q' in samples:
            # handeling conversation datasets
            conv_q, conv_a = samples['conv_q'], samples['conv_a']

            connect_sym = samples['connect_sym'][0]
            conv_q = [q.split(connect_sym)for q in conv_q]
            conv_a = [a.split(connect_sym) for a in conv_a]
            conv_img = assign_imgs(conv_q, img_embeds)

            if self.chat_template:
                conv_q = [["[INST] " + item + "[/INST]" for item in items] for items in conv_q]

            regress_embeds, regress_atts, part_targets = self.get_conv_emb(conv_q, conv_a, conv_img)
            cond_embeds, cond_atts = regress_embeds[:, :0], regress_atts[:, :0]

        else:
            if "instruction_input" in samples:
                instruction = samples["instruction_input"]
            elif len(self.prompt_list) > 1:
                instruction = random.choice(self.prompt_list)
            else:
                instruction = None

            if self.remove_template:
                instruction = remove_special_tokens(instruction)
                
            if self.chat_template:
                instruction = ["[INST] " + instruct + "[/INST]" for instruct in instruction]

            if 'length' in samples:
                # the input is a image train (like videos)
                bsz, pn, hs = img_embeds.shape
                img_embeds = img_embeds.reshape(len(samples['image']), -1, pn, hs) # (200,64,4096) -> (4,50,64,4096)
                cond_embeds, cond_atts = self.prompt_wrap(img_embeds, img_atts, instruction, samples['length'])
            else:
                cond_embeds, cond_atts = self.prompt_wrap(img_embeds, img_atts, instruction)

            ### prepare target tokens
            self.llama_tokenizer.padding_side = "right"
            text = [t + self.end_sym for t in samples["answer"]]

            regress_tokens = self.llama_tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                add_special_tokens=False
            ).to(self.device)

            regress_token_ids = regress_tokens.input_ids
            regress_atts = regress_tokens.attention_mask
            part_targets = regress_token_ids.masked_fill(
                regress_token_ids == self.llama_tokenizer.pad_token_id, -100
            )

            regress_embeds = self.embed_tokens(regress_token_ids)

        return cond_embeds, cond_atts, regress_embeds, regress_atts, part_targets

    def forward(self, samples, reduction="mean"):
        # prepare the embedding to condition and the embedding to regress
        cond_embeds, cond_atts, regress_embeds, regress_atts, part_targets = \
            self.preparing_embedding(samples)

        # concat the embedding to condition and the embedding to regress
        inputs_embeds, attention_mask, input_lens = \
            self.concat_emb_input_output(cond_embeds, cond_atts, regress_embeds, regress_atts)
        # get bos token embedding
        bos = torch.ones_like(part_targets[:, :1]) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        bos_atts = attention_mask[:, :1]

        # add bos token at the begining
        inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=1)
        attention_mask = torch.cat([bos_atts, attention_mask], dim=1)

        targets = torch.ones([inputs_embeds.shape[0], inputs_embeds.shape[1]],
                             dtype=torch.long).to(self.device).fill_(-100)
        for i, target in enumerate(part_targets):
            targets[i, input_lens[i]+1:input_lens[i]+len(target)+1] = target  # plus 1 for bos

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
                reduction=reduction,
            )
        loss = outputs.loss

        return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        images,
        texts,
        use_nucleus_sampling=False,
        num_beams=1,
        max_new_tokens=20,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1,
        temperature=1,
        do_sample=False,
        stop_words_ids=[2],
        lengths=None,
        img_embeds=None,
    ):
        '''
            function for generate test use
        '''

        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(
            stops=[torch.tensor([i]).to(self.device) for i in stop_words_ids])])
        if img_embeds is None:
            img_embeds, atts_img = self.encode_img(images.to(self.device))
        else:
            # Use images features from the input(4,45,64,5632)
            img_embeds = img_embeds.reshape(-1, *img_embeds.shape[-2:])
            img_embeds= img_embeds.to(self.device)
            img_embeds = self.llama_proj(img_embeds) # project to llama input size (200,64,5632) -> (200,64,4096)
            atts_img = torch.ones(img_embeds.size()[:-1], dtype=torch.long).to(self.device)
            
        if lengths is not None:
            image_lists = []
            img_embeds = img_embeds.reshape(len(lengths), -1, img_embeds.shape[-2], img_embeds.shape[-1])
            for idx, img_embed in enumerate(img_embeds):
                image_lists.append([img_embed[i][None] for i in range(lengths[idx])])
        else:
            image_lists = [[image_emb[None]] for image_emb in img_embeds]
        assert len(texts) == len(image_lists)
        batch_embs = [self.get_context_emb(text, img_list) for text, img_list in zip(texts, image_lists)]

        batch_size = len(batch_embs)
        max_len = max([emb.shape[1] for emb in batch_embs])
        emb_dim = batch_embs[0].shape[2]
        dtype = batch_embs[0].dtype
        device = batch_embs[0].device
        embs = torch.zeros([batch_size, max_len, emb_dim], dtype=dtype, device=device)
        attn_mask = torch.zeros([batch_size, max_len], dtype=torch.int, device=device)
        for i, emb in enumerate(batch_embs):
            emb_len = emb.shape[1]
            embs[i, -emb_len:] = emb[0]
            attn_mask[i, -emb_len:] = 1
        # check if the input embedding tokens are in the range of the model cotext window (4096) and if it is not, then truncate it to the max context window
        if self.model_type == "Llama":
            context_window = 3700
        else:
            context_window = 7500
        if embs.shape[1] > context_window:
            embs = embs[:, -context_window:]
            attn_mask = attn_mask[:, -context_window:]
        with self.maybe_autocast():
            outputs = self.llama_model.generate(
                inputs_embeds=embs,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=do_sample,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                # stopping_criteria=stopping_criteria,
            )
        answers = []
        for output_token in outputs:
            if output_token[0] == 0:
                output_token = output_token[1:]
            output_texts = self.llama_tokenizer.decode(output_token, skip_special_tokens=True)
            output_texts = output_texts.split('</s>')[0]  # remove the stop sign </s>
            output_texts = output_texts.replace("<s>", "")
            output_texts = output_texts.split(r'[/INST]')[-1].strip()
            answers.append(output_texts)
        return answers

    @torch.no_grad()
    def generate_text_only(
        self,
        images,
        seg_tokens,
        use_nucleus_sampling=False,
        num_beams=1,
        max_new_tokens=20,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1,
        temperature=1,
        do_sample=False,
        stop_words_ids=[2],
        lengths=None,
        return_video_temporal_features=False,
        img_embeds=None,
    ):
        '''
            function for generate test use
        '''

        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(
            stops=[torch.tensor([i]).to(self.device) for i in stop_words_ids])])
        
        batch_embs = [torch.cat([self.embed_tokens(seg_t)]) for seg_t in seg_tokens]

        batch_size = len(batch_embs)
        max_len = max([emb.shape[1] for emb in batch_embs])
        emb_dim = batch_embs[0].shape[2]
        dtype = batch_embs[0].dtype
        device = batch_embs[0].device

        embs = torch.zeros([batch_size, max_len, emb_dim], dtype=dtype, device=device)
        attn_mask = torch.zeros([batch_size, max_len], dtype=torch.int, device=device)
        for i, emb in enumerate(batch_embs):
            emb_len = emb.shape[1]
            embs[i, -emb_len:] = emb[0]
            attn_mask[i, -emb_len:] = 1
        
        with self.maybe_autocast():
            outputs = self.llama_model.generate(
                inputs_embeds=embs,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=do_sample,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                # stopping_criteria=stopping_criteria,
            )

        answers = []
        for output_token in outputs:
            if output_token[0] == 0:
                output_token = output_token[1:]
            output_texts = self.llama_tokenizer.decode(output_token, skip_special_tokens=True)
            output_texts = output_texts.split('</s>')[0]  # remove the stop sign </s>
            output_texts = output_texts.replace("<s>", "")
            output_texts = output_texts.split(r'[/INST]')[-1].strip()
            answers.append(output_texts)
        return answers



    @torch.no_grad()
    def multi_select(self, images, texts, answers, num_cand=None):
        all_losses = []
        for answer in answers:
            choice_samples = {
                'image': images,
                'instruction_input': texts,
                'answer': answer
            }
            loss = self.forward(choice_samples, reduction='none')['loss'].reshape(-1, 1)
            all_losses.append(loss)
            torch.cuda.empty_cache()
        all_losses = torch.cat(all_losses, dim=-1)
        if num_cand is not None:
            for i in range(all_losses.shape[0]):
                all_losses[i, num_cand[i]:] = 9999
        output_class_ranks = torch.argsort(all_losses, dim=-1)
        return output_class_ranks.tolist()

    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=0,
        **kwargs
    ):
        '''
            function for open-ended VQA
        '''
        images = samples["image"].cuda()
        texts = samples["instruction_input"]

        output_text = self.generate(
            images=images,
            texts=texts,
            num_beams=num_beams,
            max_new_tokens=max_len,
            min_length=min_len,
            length_penalty=length_penalty
        )

        if "apply_lemmatizer" in samples.keys() and samples["apply_lemmatizer"]:
            output_text = self._lemmatize(output_text)

        return output_text

    def predict_class(
            self,
            samples,
            num_beams=5,
            inference_method="generate",
            max_len=10,
            min_len=1,
            num_ans_candidates=5,
            answer_list=None,
            prompt="",
            length_penalty=0,
            **kwargs
    ):
        '''
            function for multi-choice VQA
        '''

        image = samples["image"].cuda()
        instruction = samples['instruction_input']
        answers = samples["choices"]
        num_cand = samples["num_choices"]

        ranks = self.multi_select(image, instruction, answers, num_cand)

        pred_ans = []
        for i, rank in enumerate(ranks):
            pred = answers[rank[0]][i]
            pred_ans.append(pred)
        return pred_ans

    def embed_tokens(self, token_ids):
        try:
            embeds = self.llama_model.base_model.model.model.embed_tokens(token_ids)
        except AttributeError:
            embeds = self.llama_model.model.embed_tokens(token_ids)

        return embeds

    @classmethod
    def from_config(cls, cfg):
        model = cls(
            cfg=cfg,
        )
        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load Minigpt-4-LLM Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)  
        # push the model to the hub with its metadata and config file
        # model.push_to_hub("MiniGPT4-video-v2")
        # video_config = minigpt4_video_config(cfg)
        # video_config.save_pretrained("minigpt4_video_config")
        # print("Save Minigpt-4-LLM Config: minigpt4_video_config")
        # video_config.push_to_hub("MiniGPT4-video")
        return model


def assign_imgs(batched_instruct_list, batched_img_embeds):
    '''this function is used when the data is interleaved.
    the interlevaed data is separated, and this function assign
    corresponding image embeddings to each segment'''
    if len(batched_img_embeds.shape) == 3:
        batched_img_embeds = batched_img_embeds[:, None]

    batched_assigned = []

    for instruct_list, img_embeds in zip(batched_instruct_list, batched_img_embeds):
        img_idx = 0
        assigned_img = []
        n_assigned = []
        for instruct in instruct_list:
            n_img = instruct.count('<ImageHere>')
            if n_img > 0:  # this instruction include images.
                assigned_img.append(img_embeds[None, img_idx:img_idx+n_img])
                img_idx += n_img
                n_assigned.append(n_img)
            else:  # this instruction doesn't include images
                assigned_img.append(None)
                n_assigned.append(None)
        batched_assigned.append(assigned_img)

    return batched_assigned
