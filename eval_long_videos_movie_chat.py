import json 
from tqdm import tqdm
from goldfish_lv import GoldFish_LV,split_subtitles,time_to_seconds
import argparse
import os
import time
import json
import argparse
import torch
import cv2
import moviepy.editor as mp
import webvtt
import re
from typing import Dict, Tuple, Optional, List
from tqdm import tqdm
from PIL import Image
# from openai import OpenAI
from torchvision import transforms
from pytubefix import YouTube
from minigpt4.common.eval_utils import init_model
from minigpt4.conversation.conversation import CONV_VISION
from index import MemoryIndex  
import pysrt
import chardet
import pickle
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn
import shutil
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_arguments():
    parser = argparse.ArgumentParser(description="Inference parameters")
    parser.add_argument("--neighbours", type=int, default=-1)
    parser.add_argument("--neighbours_global", type=int, default=-1)
    parser.add_argument("--fps", type=float, default=0.5)
    parser.add_argument("--name", type=str,default="ckpt_92",help="name of the experiment")
    parser.add_argument("--add_unknown", action='store_true')
    parser.add_argument("--use_chatgpt", action='store_true')
    parser.add_argument("--use_choices_for_info", action='store_true')
    parser.add_argument("--use_gt_information", action='store_true')
    parser.add_argument("--inference_text", action='store_true')
    parser.add_argument("--use_gt_information_with_distraction", action='store_true')
    parser.add_argument("--num_distraction", type=int, default=2)
    parser.add_argument("--add_confidance_score", action='store_true')
    parser.add_argument("--use_original_video", action='store_true')
    parser.add_argument("--use_video_embedding", action='store_true')
    parser.add_argument("--use_clips_for_info", action='store_true')
    parser.add_argument("--use_GT_video", action='store_true')
    parser.add_argument("--use_gt_summary", action='store_true')
    parser.add_argument("--index_subtitles", action='store_true')
    parser.add_argument("--index_subtitles_together", action='store_true')
    
    parser.add_argument("--ask_the_question_early", action='store_true')
    parser.add_argument("--clip_in_ask_early", action='store_true')
    parser.add_argument("--summary_with_subtitles_only", action='store_true')
    parser.add_argument("--use_coherent_description", action='store_true')
    parser.add_argument("--v_sum_and_info", action='store_true')
    
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=100000, type=int)
    parser.add_argument("--exp_name", type=str,default="",help="name of eval folder")

    
    parser.add_argument("--cfg-path", default="test_configs/llama2_test_config.yaml")
    parser.add_argument("--ckpt", type=str, default="checkpoints/video_llama_checkpoint_last.pth")
    parser.add_argument("--add_subtitles", action='store_true')
    parser.add_argument("--eval_opt", type=str, default='all')
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--video_path", type=str, help="path to the video")
    parser.add_argument("--use_openai_embedding",type=str2bool, default=False)
    parser.add_argument("--dataset_videos_path", type=str, help="path to the dataset videos")
    parser.add_argument("--annotation_json_folder", type=str, help="path to the annotation folder")
    parser.add_argument("--options", nargs="+")
    return parser.parse_args()
def time_to_seconds(subrip_time):
    return subrip_time.hours * 3600 + subrip_time.minutes * 60 + subrip_time.seconds + subrip_time.milliseconds / 1000

def get_movie_time(subtitle_path):
    # read the subtitle file and detect the encoding
    with open(subtitle_path, 'rb') as f:
        result = chardet.detect(f.read())
    subtitles = pysrt.open(subtitle_path, encoding=result['encoding'])
    video_time=time_to_seconds(subtitles[-1].end)
    return video_time


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
import h5py
import torch
import os

def numerical_sort_key(filename):
    base_name = os.path.splitext(filename)[0]  
    return int(base_name) 

class MovieChatDataset(Dataset):
    def __init__(self, dataset_path, annotation_path,fps, transform=None,start=0,end=100000):
        self.dataset_path = dataset_path
        self.annotation_path=annotation_path
        self.transform = transform
        self.movie_name = os.listdir(dataset_path)
        self.movie_name = [file for file in self.movie_name if file != '.DS_Store']
        self.fps = fps
        self.len_clip = 45
        self.start=start
        self.end=end
    def load_frames(self, movie_name):
        filenames = sorted(os.listdir(os.path.join(self.dataset_path, movie_name)))
        
        filenames.sort(key=numerical_sort_key)
        # define torch tensor to store the frames of size(0,0,0)
        data = []
        for filename_number in tqdm(filenames,desc="Loading frames"):
            file_path = os.path.join(self.dataset_path, movie_name, filename_number)

            if not os.path.isfile(file_path):
                print(f"Did not find file: {filename_number}")
            try:
                with h5py.File(file_path, 'r') as h5_file:
                    image_embeds=torch.tensor(h5_file[f"frames_{filename_number[:-3]}"][:])
                    image_embeds = image_embeds[:,1:,:] # remove the first token (CLS) (200,256,1408)
                    # concate each 4 neighbours image tokens 
                    bs, pn, hs = image_embeds.shape
                    image_embeds = image_embeds.view(bs, int(pn/4), int(hs*4)) 
                    data.extend(image_embeds)

            except Exception as e:  
                print(f"Failed to process {filename_number}: {e}")
            
                
        frames=torch.stack(data)
        return frames

    def __len__(self):
        return len(self.movie_name)

    def _get_movie_questions(self,movie_annotations):
        global_questions=movie_annotations['global']
        local_questions=movie_annotations['breakpoint']
        return global_questions,local_questions
    def __getitem__(self, idx):
        if self.start<=idx<self.end:
            self.frames = self.load_frames(self.movie_name[idx])
            movie_name=self.movie_name[idx]
            with open(os.path.join(self.annotation_path,movie_name+".json"), 'r') as f:
                movie_annotations = json.load(f)
            global_questions,local_questions=self._get_movie_questions(movie_annotations)
            sampling_value = int(movie_annotations['info']['fps']/self.fps)
            clips_list=[]
            current_clip=[]
            for i in range(0,self.frames.shape[0], sampling_value):
                current_clip.append(self.frames[i])
                if len(current_clip) >= self.len_clip:
                    clips_list.append(torch.stack(current_clip))
                    current_clip=[]
            if len(current_clip) > 0:
                last_frame_current_clip = current_clip[-1]
                while len(current_clip) < self.len_clip:
                    current_clip.append(last_frame_current_clip)
                clips_list.append(torch.stack(current_clip))
            return clips_list, movie_name,global_questions,local_questions
        else:
            return [], self.movie_name[idx],[],[]


class MovieChat (GoldFish_LV):
    
    def __init__(self,args):
        super().__init__(args)
        self.args=args
        self.save_long_videos_path = "new_workspace/clips_summary/movie_chat/"
        if args.use_openai_embedding:
            self.save_embedding_path = "new_workspace/open_ai_embedding/movie_chat/"
        else:
            self.save_embedding_path = "new_workspace/embedding/movie_chat/"
        os.makedirs(self.save_long_videos_path, exist_ok=True)
        os.makedirs(self.save_embedding_path, exist_ok=True)
        self.max_sub_len=400
        self.max_num_images=45
    
    
    def _get_long_video_summaries(self,clips,save_path):
        batch=[]
        batch_instructions=[]
        preds={}
        clip_numbers=[]
        max_caption_index=0
        for i,clip_features in enumerate(clips):
            if len(clip_features)!=self.max_num_images:
                continue
            batch.append(clip_features)
            img_placeholder=""
            for j in range(len(clip_features)):
                img_placeholder+="<Img><ImageHere>"
            instruction = img_placeholder + '\n' + self.summary_instruction
            batch_instructions.append(instruction)
            clip_numbers.append(i)
            if len(batch)<args.batch_size:
                continue
            batch=torch.stack(batch)
            batch_pred= self.run_images_features(batch,batch_instructions)
            for j,pred in enumerate(batch_pred):
                max_caption_index += 1
                if pred !="":
                    preds[f'caption__clip_{str(clip_numbers[j]).zfill(2)}'] = pred 
            batch=[]
            clip_numbers=[]
            batch_instructions=[]
        if len(batch)>0:
            batch=torch.stack(batch)
            batch_pred= self.run_images_features(batch,batch_instructions)
            for j,pred in enumerate(batch_pred):
                max_caption_index += 1
                if pred !="":
                    preds[f'caption__clip_{str(clip_numbers[j]).zfill(2)}'] = pred
        with open(save_path, 'w') as file:
            json.dump(preds, file, indent=4)
        return preds
    def use_model_summary (self,qa_prompts,related_context_documents_list,related_context_keys_list,external_memory):
        related_context_documents_text_list=[]
        for related_context_documents,related_context_keys in zip(related_context_documents_list,related_context_keys_list): 
            related_information=""
            most_related_clips=self.get_most_related_clips_index(related_context_keys,external_memory)
            for clip_name in most_related_clips:
                general_sum=""
                clip_name=str(clip_name).zfill(2)
                for key in external_memory.documents.keys():
                    if clip_name in key and 'caption' in key:
                        general_sum="Clip Summary: "+external_memory.documents[key]
                        break
                related_information+=f"{general_sum}\n"
            related_context_documents_text_list.append(related_information)
                
        if args.use_chatgpt :
            batch_pred=self.inference_RAG_chatGPT(qa_prompts,related_context_documents_text_list)
        else:
            batch_pred=self.inference_RAG(qa_prompts,related_context_documents_text_list)
        return batch_pred, related_context_documents_text_list
    def answer_movie_questions_RAG(self,qa_list,information_RAG_path,embedding_path,q_type):
        if q_type=='local':
            external_memory=MemoryIndex(args.neighbours, use_openai=self.args.use_openai_embedding)
        else:  
            external_memory=MemoryIndex(args.neighbours_global, use_openai=self.args.use_openai_embedding)
        if os.path.exists(embedding_path):
            external_memory.load_embeddings_from_pkl(embedding_path)
        else:
            external_memory.load_documents_from_json(information_RAG_path,embedding_path)
        # get the most similar context from the external memory to this instruction 
        related_context_documents_list=[]
        related_context_keys_list=[]
        total_batch_pred=[]
        related_text=[]
        qa_prompts=[]
        for qa in qa_list:
            related_context_documents,related_context_keys = external_memory.search_by_similarity(qa['question'])
            related_context_documents_list.append(related_context_documents)
            related_context_keys_list.append(related_context_keys)
            prompt=self.prepare_prompt(qa)
            qa_prompts.append(prompt)
        if args.use_clips_for_info:
            batch_pred,related_context_keys_list=self.use_clips_for_info(qa_list,related_context_keys_list,external_memory)   
            total_batch_pred.extend(batch_pred)
            related_text.extend(related_context_keys_list)
        else:
            batch_pred, related_context_documents_text_list=self.use_model_summary (qa_prompts,
                        related_context_documents_list,related_context_keys_list,external_memory)
            total_batch_pred.extend(batch_pred)
            related_text.extend(related_context_documents_text_list)
        assert len(total_batch_pred)==len(qa_list)      
        assert len(total_batch_pred)==len(related_text)
        return total_batch_pred, related_text
    def get_most_related_clips_index(self,related_context_keys,external_memory):
        most_related_clips_index=[]
        for context_key in related_context_keys:
            # loop over memory keys to get the context key index
            for i,key in enumerate(external_memory.documents.keys()):
                if context_key in key:
                    most_related_clips_index.append(i)
                    break
            
        return most_related_clips_index

    
    def clip_inference(self,clips_idx,prompts):
        setup_seeds(seed)
        images_batch, instructions_batch = [], []
        for clip_idx, prompt in zip(clips_idx, prompts):
            clip_features=self.video_clips[clip_idx]
            img_placeholder=""
            for j in range(len(clip_features)):
                img_placeholder+='<Img><ImageHere>'
            instruction = img_placeholder + '\n' + prompt
            images_batch.append(clip_features)
            instructions_batch.append(instruction)
        # run inference for the batch
        images_batch=torch.stack(images_batch)
        batch_pred= self.run_images_features(images_batch,instructions_batch)
        return batch_pred
    def prepare_prompt(self,qa):
        prompt=qa["question"]
        return prompt
    def use_clips_for_info(self,qa_list,related_context_keys_list,external_memory):
        total_batch_pred=[]
        questions=[]
        related_information_list=[]
        related_context_keys_list_new=[]
        for qa,related_context_keys in zip(qa_list,related_context_keys_list):
            most_related_clips_index=self.get_most_related_clips_index(related_context_keys,external_memory)
            question=qa['question']
            prompt=f"From this video extract the related information to This question and provide an explaination for your answer and If you can't find any related information, say 'I DON'T KNOW' as option 5 because maybe the questoin is not related to the video content.\n the question is :\n {question}\n your answer :"
            batch_inference=[]
            all_info=[]
            for clip_idx in most_related_clips_index:
                batch_inference.append(clip_idx)
                if len(batch_inference)<args.batch_size:
                    continue    
                all_info.extend(self.clip_inference(batch_inference,[prompt]*len(batch_inference)))
                batch_inference=[]
            if len(batch_inference)>0:
                all_info.extend(self.clip_inference(batch_inference,[prompt]*len(batch_inference)))
            # all_info=self.clip_inference(most_related_clips_index,[prompt]*len(most_related_clips_index))      
            related_information=""
            for info,clip_name in zip(all_info,most_related_clips_index):
                general_sum=""
                clip_name=str(clip_name).zfill(2)
                for key in external_memory.documents.keys():
                    if clip_name in key and 'caption' in key:
                        general_sum="Clip Summary: "+external_memory.documents[key]
                if args.v_sum_and_info:
                    related_information+=f"{general_sum},question_related_information: {info}\n"  
                else:
                    related_information+=f"question_related_information: {info}\n"
            questions.append(question)
            related_information_list.append(related_information)
            related_context_keys.append(related_information)
            related_context_keys_list_new.append(related_context_keys)
            if len(questions)< args.batch_size:
                continue
            setup_seeds(seed)
            if args.use_chatgpt :
                batch_pred=self.inference_RAG_chatGPT(questions, related_information_list)
            else:
                batch_pred=self.inference_RAG(questions, related_information_list)
            
            for pred in batch_pred:
                total_batch_pred.append(pred)
            questions=[]
            related_information_list=[]
        
        if len(questions)>0:
            setup_seeds(seed)
            if args.use_chatgpt :
                batch_pred=self.inference_RAG_chatGPT(questions, related_information_list)
            else:
                batch_pred=self.inference_RAG(questions, related_information_list)
            for pred in batch_pred:
                total_batch_pred.append(pred)
        return total_batch_pred,related_context_keys_list_new
    def define_save_name(self):
        save_name="subtitles" if args.index_subtitles else "no_subtitles"
        save_name="subtitles_together" if args.index_subtitles_together else save_name
        save_name="summary_with_subtitles_only" if args.summary_with_subtitles_only else save_name
        save_name+="_unknown" if args.add_unknown else ""
        save_name+="_clips_for_info" if args.use_clips_for_info else ""
        save_name+="_chatgpt" if args.use_chatgpt else ""
        save_name+="_choices_for_info" if args.use_choices_for_info else ""
        save_name+="_v_sum_and_info" if args.v_sum_and_info else ""
        save_name+='fps_'+str(args.fps)
        save_dir=f"new_workspace/results/moviechat/{args.exp_name}/{save_name}_{args.neighbours_global}_neighbours"
        os.makedirs(save_dir, exist_ok=True)
        return save_dir
        
    def eval_moviechat(self):
        start=args.start
        end=args.end
        dataset_path = args.dataset_videos_path
        annotation_json_folder=args.annotation_json_folder
        dataset = MovieChatDataset(dataset_path,annotation_json_folder, fps=args.fps,start=start,end=end)
        # dataloader = DataLoader(dataset, batch_size=1,  shuffle=False)
        full_questions_result=[]
        save_dir=self.define_save_name()

        for i,(clips ,video_name,global_questions,local_questions) in enumerate(dataset):
            # code here
            if start<=i < end:
                print("video_name",video_name)
                self.video_clips=clips
                self.video_name=video_name
                file_path=os.path.join(self.save_long_videos_path,self.video_name+f"_fps{args.fps}.json")
                embedding_path=os.path.join(self.save_embedding_path,self.video_name+f"_fps{args.fps}.pkl")
                if os.path.exists(file_path):
                    print("Already processed")
                else:
                    self._get_long_video_summaries(clips,file_path)
                batch_questions=[]
                for qa in global_questions:
                    batch_questions.append(qa)
                    if len(batch_questions)<args.batch_size:
                        continue
                    model_answers, related_text=self.answer_movie_questions_RAG(batch_questions,file_path,embedding_path,q_type='global')
                    for qa,ans in zip(batch_questions,model_answers):
                        qa.update({'pred':ans})
                        qa['Q']=qa['question']
                        qa['A']=qa['answer']
                        qa.pop('question', None)
                        qa.pop('answer', None)
                    
                    batch_questions=[]
                if len(batch_questions)>0:
                    model_answers, related_text=self.answer_movie_questions_RAG(batch_questions,file_path,embedding_path,q_type='global')
                    for qa,ans in zip(batch_questions,model_answers):
                        qa.update({'pred':ans})
                        qa['Q']=qa['question']
                        qa['A']=qa['answer']
                        qa.pop('question', None)
                        qa.pop('answer', None)
                
                full_questions_result.extend(global_questions)
                print(f"Finished {i} out of {len(dataset)}")
                # save the results 
                with open(f"{save_dir}/{self.video_name}.json", 'w') as file:
                    # json.dump(global_questions+local_questions, file, indent=4)
                    json.dump(global_questions, file, indent=4)
                    
        with open(f"{save_dir}/full_pred_{start}_{end}.json", 'w') as fp:
            json.dump(full_questions_result, fp)
args=get_arguments()

def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

import yaml 
# read this file test_configs/llama2_test_config.yaml
with open('test_configs/llama2_test_config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
seed=config['run']['seed']
print("seed",seed)

if __name__ == "__main__":
    setup_seeds(seed)
    llama_vid_eval=MovieChat(args)
    llama_vid_eval.eval_moviechat()
    