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
    parser.add_argument("--name", type=str,default="ckpt_92",help="name of the experiment")
    parser.add_argument("--exp_name", type=str,default="",help="name of the experiment")
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
    parser.add_argument("--index_subtitles_together", action='store_true')
    
    parser.add_argument("--ask_the_question_early", action='store_true')
    parser.add_argument("--clip_in_ask_early", action='store_true')
    parser.add_argument("--use_coherent_description", action='store_true')
    
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=100000, type=int)
    
    parser.add_argument("--vision_only", action='store_true')
    parser.add_argument("--model_summary_only", action='store_true')
    parser.add_argument("--subtitles_only", action='store_true')
    parser.add_argument("--subtitles_only_after_retrieval", action='store_true')
    parser.add_argument("--info_only", action='store_true')
    
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
    parser.add_argument("--annotation_path", type=str, help="path to the annotation file")
    parser.add_argument("--videos_frames", type=str, help="path to the dataset extracted frames")
    parser.add_argument("--tvqa_json_subtitles", type=str, help="path to the tvqa json subtitles")
    parser.add_argument("--tvqa_clips_subtitles", type=str, help="path to the tvqa json")  
    parser.add_argument("--options", nargs="+")
    return parser.parse_args()
   
def clean_text(subtitles_text):
    # Remove unwanted characters except for letters, digits, and single quotes
    subtitles_text = re.sub(r'[^a-zA-Z0-9\s\']', '', subtitles_text)
    # Replace multiple spaces with a single space
    subtitles_text = re.sub(r'\s+', ' ', subtitles_text)
    return subtitles_text.strip()
                     
class TVQAEVAL (GoldFish_LV):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.tv_shows_mapping={"Grey's Anatomy":"grey_frames", 'How I Met You Mother':"met_frames", 'Friends':"friends_frames", 'The Big Bang Theory':"bbt_frames", 'House M.D.':"house_frames", 'Castle':"castle_frames"} 
        self.save_long_videos_path = f"new_workspace/clips_summary/tvqa"
        if args.use_openai_embedding:
            self.save_embedding_path = f"new_workspace/open_ai_embedding/tvqa"
        else:
            self.save_embedding_path = f"new_workspace/embedding/tvqa"
        os.makedirs(self.save_long_videos_path, exist_ok=True)
        self.max_sub_len=400
        self.max_num_images=45
        self.fps=3
        with open(args.tvqa_json_subtitles) as f:
            self.subtitles_list=json.load(f)
        self.subtitles={}
        for sub in self.subtitles_list:
            self.subtitles[sub["vid_name"]]=sub["sub"]
     
    def _get_TVs_data(self):
        json_file_path=args.annotation_path
        frames_path=args.videos_frames
        subtitle_path=args.tvqa_clips_subtitles
        with open (json_file_path) as f:
            tv_shows_data=json.load(f)
        return tv_shows_data,frames_path,subtitle_path
    def _get_shows_subtitles(self,clip_subtitles_path):
        try :
            with open(clip_subtitles_path, 'rb') as f:
                result = chardet.detect(f.read())
            clip_subtitles = pysrt.open(clip_subtitles_path, encoding=result['encoding'])
            return clip_subtitles
        except:
            print("No subtitles found")
            return []
    def episode_inference(self,clips,folder_name,use_subtitles):
        max_caption_index = 0
        max_subtitle_index = 0
        preds={}
        important_data = {}
        videos_summaries=[]
        batch_size=args.batch_size
        batch_images=[]
        batch_instructions=[]
        conversations=[]
        clips_names=[]
        for clip_name in tqdm(clips,desc="Inference Episode clips"):
            conversation=""
            try:
                for subtitle in self.subtitles[clip_name]:
                    conversation+=subtitle['text']+" "
            except:
                pass
            conversations.append(clean_text(conversation))               
            images,img_placeholder=self.prepare_input_images(clip_name,folder_name,use_subtitles)
            instruction = img_placeholder + '\n' + self.summary_instruction
            batch_images.append(images)
            batch_instructions.append(instruction)
            clips_names.append(clip_name)
            if len(batch_images) < batch_size:
                continue
            batch_images = torch.stack(batch_images)
            batch_pred=self.run_images(batch_images,batch_instructions)
            for i,pred in enumerate(batch_pred):
                max_caption_index += 1
                videos_summaries.append(pred)
                if args.use_coherent_description:
                    preds[f'caption_{max_caption_index}__{clips_names[i]}'] = f"model_summary :{pred}\nVideo conversation :{conversations[i]}"
                else:
                    if args.index_subtitles_together and use_subtitles:
                        if conversations[i] != "":
                            max_subtitle_index+=1
                            important_data.update({f"subtitle_{max_subtitle_index}__{clips_names[i]}": conversations[i]})   
                    preds[f'caption_{max_caption_index}__{clips_names[i]}'] = pred
            
            batch_images=[]
            batch_instructions=[]
            clips_names=[]
            conversations=[]
        # run inference for the last batch
        if len(batch_images)>0:
            batch_images = torch.stack(batch_images)
            batch_pred=self.run_images(batch_images,batch_instructions)
            for i,pred in enumerate(batch_pred):
                max_caption_index += 1
                videos_summaries.append(pred)
                if args.use_coherent_description:
                    preds[f'caption_{max_caption_index}__{clips_names[i]}'] = f"model_summary :{pred}\nVideo conversation :{conversations[i]}"
                else:
                    if args.index_subtitles_together and use_subtitles:
                        if conversations[i] != "":
                            max_subtitle_index+=1
                            important_data.update({f"subtitle_{max_subtitle_index}__{clips_names[i]}": conversations[i]})   
                    preds[f'caption_{max_caption_index}__{clips_names[i]}'] = pred
            batch_images=[]
            batch_instructions=[]
            clips_names=[]
        return preds,important_data  
    
    def episode_inference_only_subtitles(self,clips,tv_images_path,subtitle_path):
        max_subtitle_index = 0
        important_data = {}
        for c_name in tqdm(clips,desc="Inference Episode clips"):
            clip_subtitles_path=os.path.join(subtitle_path,c_name+".srt")
            clip_subtitles=self._get_shows_subtitles(clip_subtitles_path)
            conversation=""
            if args.index_subtitles_together:
                if self.subtitles.get(c_name,False):
                    for subtitle in self.subtitles[c_name]:
                        conversation+=subtitle['text']+" "
                    conversation=clean_text(conversation)
                    if conversation != "":
                        max_subtitle_index+=1
                        important_data.update({f"subtitle_{max_subtitle_index}__{c_name}": conversation})
        return important_data 
    def prepare_input_images(self,clip_name,folder_name,use_subtitles):
        tv_shows_data,frames_path,subtitle_path=self._get_TVs_data()
        tv_images_path =os.path.join(frames_path,folder_name)
        clip_path=os.path.join(tv_images_path,clip_name)
        total_frames=len(os.listdir(clip_path))
        sampling_interval=int(total_frames//self.max_num_images)
        if sampling_interval==0:
            sampling_interval=1
        images=[]
        img_placeholder = ""
        video_frames_path = os.path.join(frames_path,folder_name,clip_name)
        total_num_frames=len(os.listdir(video_frames_path))
        sampling_interval = round(total_num_frames / self.max_num_images)
        if sampling_interval == 0:
            sampling_interval = 1
        subtitle_text_in_interval = ""
        history_subtitles = {}
        number_of_sub_words=0
        for i,frame in enumerate(sorted(os.listdir(video_frames_path))):
            # Find the corresponding subtitle for the frame and combine the interval subtitles into one subtitle
            # we choose 1 frame for every 2 seconds,so we need to combine the subtitles in the interval of 2 seconds
            if self.subtitles.get(clip_name,False) and use_subtitles:
                for subtitle in self.subtitles[clip_name]:
                    if (subtitle['start'] <= (i / self.fps) <= subtitle['end']) and subtitle['text'] not in subtitle_text_in_interval:
                        if not history_subtitles.get(subtitle['text'],False):
                            subtitle_text_in_interval+=subtitle['text']+" "
                        history_subtitles[subtitle['text']]=True
                        break
            if i % sampling_interval == 0:
                frame = Image.open(os.path.join(video_frames_path,frame)).convert("RGB") 
                frame = self.vis_processor(frame)
                images.append(frame)
                img_placeholder += '<Img><ImageHere>'
                if number_of_sub_words<self.max_sub_len and use_subtitles:
                    if subtitle_text_in_interval != "":
                        subtitle_text_in_interval=clean_text(subtitle_text_in_interval)
                        img_placeholder+=f'<Cap>{subtitle_text_in_interval}'
                        number_of_sub_words+=len(subtitle_text_in_interval.split(' '))
                        subtitle_text_in_interval = ""
            if len(images) >= self.max_num_images:
                break
        if len(images) ==0:
            print("Video not found",video_frames_path)
            
        if 0 <len(images) < self.max_num_images:
            last_item = images[-1]
            while len(images) < self.max_num_images:
                images.append(last_item)
                img_placeholder += '<Img><ImageHere>'
        images = torch.stack(images)
        return images,img_placeholder
    def clip_inference(self,clips_name,folders_name,prompts):
        setup_seeds(seed)
        images_batch, instructions_batch = [], []
        for clip_name,folder_name, prompt in zip(clips_name,folders_name, prompts):
            images,img_placeholder=self.prepare_input_images(clip_name,folder_name,use_subtitles=not args.vision_only)
            instruction = img_placeholder + '\n' + prompt
            images_batch.append(images)
            instructions_batch.append(instruction)
        # run inference for the batch
        images_batch=torch.stack(images_batch)
        batch_pred=self.run_images(images_batch,instructions_batch)
        return batch_pred
    def prepare_prompt(self,qa):
        prompt=qa["q"]+" \n\n As you watched in this video Choose ONE suitable answer from these mutiple choices \n"
        for i,choice in enumerate(["a0","a1","a2","a3","a4"]):
            prompt+=f"option {i}: {qa[choice]} \n"
        if args.add_unknown and args.add_confidance_score:
            # Add unknown option
            prompt+=f"option 5: Can't answer based on the provided information\n"
            prompt+="\n Your output should be THE NUMBER OF THE CORRECT ANSWER FROM THE CHOICES FROM 0 TO 5 INCLUSIVE and aslo output a CONFIDANCE SCORE FROM 0 TO 5 representing how confident you are with your answer where 0 is the least confident and 5 is the most confident"
        elif args.add_unknown:
            prompt+=f"option 5: Can't answer based on the provided information\n"
            prompt+="\n Your output should be THE NUMBER OF THE CORRECT ANSWER FROM THE CHOICES FROM 0 TO 5 INCLUSIVE"
        elif args.add_confidance_score:
            prompt+="\n Your output should be THE NUMBER OF THE CORRECT ANSWER FROM THE CHOICES FROM 0 TO 4 INCLUSIVE and aslo output a CONFIDANCE SCORE FROM 0 TO 5 representing how confident you are with your answer where 0 is the least confident and 5 is the most confident"
        else:
            prompt+="\n Your output should be THE NUMBER OF THE CORRECT ANSWER FROM THE CHOICES FROM 0 TO 4 INCLUSIVE"
        return prompt
    def get_most_related_clips(self,qa,related_context_keys):
        if args.use_gt_information:
            most_related_clips=[qa['vid_name']]
        elif args.use_gt_information_with_distraction:
            most_related_clips=[qa['vid_name']]
            for context_key in related_context_keys:
                if len(context_key.split('__'))>1:
                    most_related_clips.append(context_key.split('__')[1])
                if len(most_related_clips)==args.num_distraction+1:
                    break
        else:
            most_related_clips=[]
            for context_key in related_context_keys:
                if len(context_key.split('__'))>1:
                    most_related_clips.append(context_key.split('__')[1])
                if len(most_related_clips)==args.neighbours:
                    break
        assert len(most_related_clips)!=0, f"No related clips found {related_context_keys}"
        return most_related_clips
    def use_clips_for_info(self,qa_list,related_context_keys_list,external_memory):
        total_batch_pred=[]
        questions=[]
        related_information_list=[]
        related_context_keys_list_new=[]
        for qa,related_context_keys in zip(qa_list,related_context_keys_list):
            most_related_clips=self.get_most_related_clips(qa,related_context_keys)
            folder_name=self.tv_shows_mapping[qa['show_name']]
            question=qa['q']+ "\nand these are the choices :\n"
            for i,choice in enumerate(["a0","a1","a2","a3","a4"]):
                question+=f"option {i}: {qa[choice]} \n"
            if args.add_unknown:    
                question+= "option 5: Can't answer based on the provided information\n"
                question+="\n Your output should be THE NUMBER OF THE CORRECT ANSWER FROM THE CHOICES FROM 0 TO 5 INCLUSIVE"
            else:
                question+="\n Your output should be THE NUMBER OF THE CORRECT ANSWER FROM THE CHOICES FROM 0 TO 4 INCLUSIVE"
            if args.use_choices_for_info:
                # prompt=self.prepare_prompt(qa)
                # prompt+=" and also provide an EXPLAINATION for your answer and If you don't know the answer, say that you don't know.\n\n"
                prompt=f"From this video extract the related information to This multichioce question and provide an explaination for your answer and If you don't know the answer, say 'I DON'T KNOW' as option 5 because maybe the questoin is not related to the video content.\n the question is :\n {question}\n your answer :"

            else:
                prompt=f"As you watched in this video answer this {qa['q']}\n\n and also provide an EXPLAINATION for your answer and If you don't know the answer, say that you don't know.\n\n"
            all_info=self.clip_inference(most_related_clips,[folder_name]*len(most_related_clips),[prompt]*len(most_related_clips))
            # concatinate all the information together
            related_information=""
            for info,clip_name in zip(all_info,most_related_clips):
                clip_conversation=""
                general_sum=""
                for key in external_memory.documents.keys():
                    if clip_name in key and 'caption' in key:
                        general_sum="Clip Summary: "+external_memory.documents[key]
                    if clip_name in key and 'subtitle' in key:
                        clip_conversation="Clip Subtitles: "+external_memory.documents[key]
                
                if args.use_coherent_description:
                    related_information+=f"question_related_information: {info},{general_sum}\n"
                else:
                    # related_information+=f"{general_sum},{clip_conversation},question_related_information: {info}\n"  
                    # related_information+=f"question_related_information: {info},{clip_conversation}\n"
                    if args.model_summary_only:
                            related_information+=f"{general_sum},question_related_information: {info}\n"
                    elif args.info_only:
                        related_information+=f"question_related_information: {info}\n"
                    elif args.subtitles_only:
                        related_information+=f"{clip_conversation},question_related_information: {info}\n"
                    elif args.subtitles_only_after_retrieval:
                        related_information+=f"{clip_conversation},question_related_information: {info}\n"
                    else:
                        related_information+=f"{general_sum},{clip_conversation},question_related_information: {info}\n" 
                    
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
    def answer_TV_questions_RAG(self,qa_list,external_memory,episode_clips,episode_name):
        related_context_keys_list,related_context_documents_list=[],[]
        setup_seeds(seed)
        for qa in qa_list:
            question_choices=qa['q']+ "\n and these are the options for the question\n\n"
            for i,choice in enumerate(["a0","a1","a2","a3","a4"]):
                question_choices+=f"option {i}: {qa[choice]} \n\n"
            related_context_documents,related_context_keys = external_memory.search_by_similarity(question_choices)
            
            related_context_documents_list.append(related_context_documents)
            related_context_keys_list.append(related_context_keys) 

        if args.use_clips_for_info:
            batch_pred,related_context_keys_list=self.use_clips_for_info(qa_list,related_context_keys_list,external_memory)
        else:
            prompts=[]
            related_context_documents_text_list=[]
            for qa,related_context_documents,related_context_keys in zip(qa_list,related_context_documents_list,related_context_keys_list): 

                related_information=""
                most_related_clips=self.get_most_related_clips(qa,related_context_keys)
                for clip_name in most_related_clips:
                    clip_conversation=""
                    general_sum=""
                    for key in external_memory.documents.keys():
                        if clip_name in key and 'caption' in key:
                            general_sum="Clip Summary: "+external_memory.documents[key]
                        if clip_name in key and 'subtitle' in key:
                            clip_conversation="Clip Subtitles: "+external_memory.documents[key]
                    # related_information+=f"{general_sum},{clip_conversation}\n"
                    if args.use_coherent_description:
                        related_information+=f"{general_sum}\n"
                    else:
                        if args.model_summary_only:
                            related_information+=f"{general_sum}\n"
                        elif args.subtitles_only:
                            related_information+=f"{clip_conversation}\n"
                        else:
                            related_information+=f"{general_sum},{clip_conversation}\n"
                    
                prompt=self.prepare_prompt(qa)
                prompts.append(prompt)
                related_context_documents_text_list.append(related_information)
                
            setup_seeds(seed)
            if args.use_chatgpt:
                batch_pred=self.inference_RAG_chatGPT(prompts, related_context_documents_text_list)
            else:
                batch_pred=self.inference_RAG(prompts, related_context_documents_text_list)
        return batch_pred ,related_context_keys_list 
    def answer_episode_questions(self,questions,information_RAG_path,embedding_path,episode_clips):
        external_memory=MemoryIndex(args.neighbours, use_openai=args.use_openai_embedding)
        if os.path.exists(embedding_path):
            external_memory.load_embeddings_from_pkl(embedding_path)
        else:
            external_memory.load_documents_from_json(information_RAG_path,embedding_path)    
        episode_name=information_RAG_path.split('/')[-1].split('.')[0]
        pred_json=[]
        batch_questions=[]
        for qa in tqdm(questions,desc="Answering questions"):
            batch_questions.append(qa)
            if len(batch_questions)<args.batch_size:
                continue
            batch_pred,batch_related_context_keys = self.answer_TV_questions_RAG(batch_questions,external_memory,episode_clips,episode_name)
            for pred,related_context_keys,qa in zip(batch_pred,batch_related_context_keys,batch_questions):
                qa['pred']=pred
                qa['related_context_keys']=related_context_keys
                pred_json.append(qa)
            batch_questions=[]
        if len(batch_questions)>0:
            batch_pred,batch_related_context_keys = self.answer_TV_questions_RAG(batch_questions,external_memory,episode_clips,episode_name)
            for pred,related_context_keys,qa in zip(batch_pred,batch_related_context_keys,batch_questions):
                qa['pred']=pred
                qa['related_context_keys']=related_context_keys
                pred_json.append(qa)
        return pred_json
             
    def eval_tv_shows(self,):
        tv_shows_data,frames_path,subtitle_path=self._get_TVs_data()
        full_questions_result=[]
        number_of_episodes=0
        start=args.start
        end=args.end
        for show in tqdm(tv_shows_data,desc="Inference TV shows"):
            for season in tqdm(tv_shows_data[show],desc=f"Inference {show} seasons"):
                for episode in tqdm(tv_shows_data[show][season],desc=f"Inference {show} {season} episodes"):
                    # Generate clips summary and store the important data (summary and subtitles) in json file
                    if start<=number_of_episodes<end: 
                        folder_name=self.tv_shows_mapping[show]
                        tv_images_path =os.path.join(frames_path,folder_name)
                        os.makedirs(self.save_long_videos_path, exist_ok=True)
                        save_name="" if args.index_subtitles_together else "no_subtitles_"
                        save_name="subtitles_only" if args.subtitles_only else save_name
                        save_name="use_coherent_description" if args.use_coherent_description else save_name
                        file_path=os.path.join(self.save_long_videos_path,save_name+folder_name+"_"+season+"_"+episode+".json")
                        embedding_path=os.path.join(self.save_embedding_path,save_name+folder_name+"_"+season+"_"+episode+".pkl")
                        # options don't require rerunning the inference
                        save_name+="_unknown" if args.add_unknown else ""
                        save_name+="_clips_for_info" if args.use_clips_for_info else ""
                        save_name+="_chatgpt" if args.use_chatgpt else ""
                        save_name+="_choices_for_info" if args.use_choices_for_info else ""
                        save_name+="_info_only" if args.info_only else ""
                        save_name+="_subtitles_only" if args.subtitles_only else ""
                        save_name+="_subtitles_only_after_retrieval" if args.subtitles_only_after_retrieval else ""
                        if os.path.exists(file_path):
                            with open(file_path, 'r') as file:
                                important_data = json.load(file)
                            print("Already processed")
                        else:
                            episode_clips=tv_shows_data[show][season][episode]['clips']
                            if args.subtitles_only :
                                important_data=self.episode_inference_only_subtitles(episode_clips,tv_images_path,subtitle_path)
                            else:
                                preds,important_data=self.episode_inference(episode_clips,folder_name,use_subtitles=not args.vision_only)
                                important_data.update(preds)
                            # if not args.subtitles_only :
                            #     summary = self.compine_summaries(important_data)
                            #     preds['summary'] = summary
                            #     important_data["summary"]=summary
                            with open(file_path, 'w') as file:
                                json.dump(important_data, file, indent=4)
                        # Answer questions 
                        questions=tv_shows_data[show][season][episode]['questions']
                        episode_clips=tv_shows_data[show][season][episode]['clips']
                        episode_name=file_path.split('/')[-1].split('.')[0]
                        pred_json=self.answer_episode_questions(questions,file_path,embedding_path,episode_clips)
                        full_questions_result.extend(pred_json)
                        save_dir=f"new_workspace/results/tvqa/{args.exp_name}/{save_name}_{args.neighbours}_neighbours"
                        os.makedirs(save_dir, exist_ok=True)
                        with open(f"{save_dir}/{episode_name}.json", 'w') as fp:
                            json.dump(pred_json, fp)
                        print(f"Episode {episode_name} prediction saved to {save_dir}/{episode_name}_pred_{args.neighbours}.json")
                    number_of_episodes+=1
        with open(f"{save_dir}/full_pred_{start}_{end}.json", 'w') as fp:
            json.dump(full_questions_result, fp)
        print(f"TV shows prediction saved to {save_dir}/full_pred_{start}{end}.json")
args=get_arguments()

def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

import yaml 
with open('test_configs/llama2_test_config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
seed=config['run']['seed']
print("seed",seed)

if __name__ == "__main__":
    setup_seeds(seed)
    tvqa_eval=TVQAEVAL(args)
    tvqa_eval.eval_tv_shows()