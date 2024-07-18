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
    parser.add_argument("--neighbours", type=int, default=3)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=100000, type=int)
    parser.add_argument("--use_openai_embedding",type=str2bool, default=False)
    parser.add_argument("--skill_path",default="/ibex/project/c2106/kirolos/Long_video_Bench/benchmark/final/concatenated/summarization.json")
    parser.add_argument("--cfg-path", default="test_configs/llama2_test_config.yaml")
    parser.add_argument("--add_subtitles", action='store_true')
    parser.add_argument("--ckpt", type=str, default="checkpoints/video_llama_checkpoint_last.pth")
    parser.add_argument("--eval_opt", type=str, default='all')
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--options", nargs="+")
    return parser.parse_args()
   
def clean_text(subtitles_text):
    # Remove unwanted characters except for letters, digits, and single quotes
    subtitles_text = re.sub(r'[^a-zA-Z0-9\s\']', '', subtitles_text)
    # Replace multiple spaces with a single space
    subtitles_text = re.sub(r'\s+', ' ', subtitles_text)
    return subtitles_text.strip()
                     
class TVQAEVAL (GoldFish_LV):
    def __init__(self, args: argparse.Namespace,use_openai_embedding) -> None:
        super().__init__(args)
        self.use_openai_embedding=use_openai_embedding
        self.tv_shows_mapping={"Grey's Anatomy":"grey_frames", 'How I Met You Mother':"met_frames", 'Friends':"friends_frames", 'The Big Bang Theory':"bbt_frames", 'House M.D.':"house_frames", 'Castle':"castle_frames"} 
        self.save_clips_summary = f"new_workspace/clips_summary/tvqa/"
        if self.use_openai_embedding :
            self.save_embedding = f"new_workspace/open_ai_embedding/tvqa/"
        else:
            self.save_embedding = f"new_workspace/embedding/tvqa/"
        os.makedirs(self.save_clips_summary, exist_ok=True)
        os.makedirs(self.save_embedding, exist_ok=True)
        self.max_sub_len=400
        self.max_num_images=45
        
        self.fps=3
        self.frames_path="/ibex/project/c2090/datasets/TVR_dataset/videos/video_files/frames_hq/"
        self.subtitle_path="/ibex/project/c2090/datasets/TVR_dataset/videos/tvqa_subtitles" 
        with open("datasets/evaluation_datasets/long_video_datasets/tvqa/tvqa_preprocessed_subtitles.json") as f:
            self.subtitles_list=json.load(f)
        self.tvqa_subtitles={}
        for sub in self.subtitles_list:
            self.tvqa_subtitles[sub["vid_name"]]=sub["sub"]
     
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
                for subtitle in self.tvqa_subtitles[clip_name]:
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
                if use_subtitles:
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
                if use_subtitles:
                    if conversations[i] != "":
                        max_subtitle_index+=1
                        important_data.update({f"subtitle_{max_subtitle_index}__{clips_names[i]}": conversations[i]})   
                preds[f'caption_{max_caption_index}__{clips_names[i]}'] = pred
            batch_images=[]
            batch_instructions=[]
            clips_names=[]
        return preds,important_data  
    
    def prepare_input_images(self,clip_name,folder_name,use_subtitles):
        tv_images_path =os.path.join(self.frames_path,folder_name)
        clip_path=os.path.join(tv_images_path,clip_name)
        total_frames=len(os.listdir(clip_path))
        sampling_interval=int(total_frames//self.max_num_images)
        if sampling_interval==0:
            sampling_interval=1
        images=[]
        img_placeholder = ""
        video_frames_path = os.path.join(self.frames_path,folder_name,clip_name)
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
            if self.tvqa_subtitles.get(clip_name,False) and use_subtitles:
                for subtitle in self.tvqa_subtitles[clip_name]:
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

    def prepare_prompt(self,qa):
        if qa.get('options',False):
            prompt=qa['question']+ "\n and these are the options for the question\n\n"
            for i,choice in enumerate(qa['options']):
                prompt+=f"option {i}: {choice} \n\n"
            prompt+="\n Your output should be THE NUMBER OF THE CORRECT ANSWER FROM THE CHOICES FROM 0 TO 4 INCLUSIVE"
        else:
            prompt=qa['question']
        return prompt
    def get_most_related_clips(self,qa,related_context_keys):
        most_related_clips=[]
        for context_key in related_context_keys:
            if len(context_key.split('__'))>1:
                most_related_clips.append(context_key.split('__')[1])
            if len(most_related_clips)==args.neighbours:
                break
        assert len(most_related_clips)!=0, f"No related clips found {related_context_keys}"
        return most_related_clips
    def answer_TV_questions_RAG(self,qa_list,external_memory,episode_clips,episode_name):
        related_context_keys_list,related_context_documents_list=[],[]
        setup_seeds(seed)
        for qa in qa_list:
            if qa.get('options',False):
                question_choices=qa['question']+ "\n and these are the options for the question\n\n"
                for i,choice in enumerate(qa['options']):
                    question_choices+=f"option {i}: {choice} \n\n"
                related_context_documents,related_context_keys = external_memory.search_by_similarity(question_choices)
            else:
                question_open_ended=qa['question']
                related_context_documents,related_context_keys = external_memory.search_by_similarity(question_open_ended)
            
            related_context_documents_list.append(related_context_documents)
            related_context_keys_list.append(related_context_keys) 

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
                related_information+=f"{general_sum},{clip_conversation}\n"
              
            prompt=self.prepare_prompt(qa)
            prompts.append(prompt)
            related_context_documents_text_list.append(related_information)
            setup_seeds(seed)
            batch_pred=self.inference_RAG(prompts, related_context_documents_text_list)
        return batch_pred ,related_context_keys_list 
    def answer_episode_questions(self,questions,information_RAG_path,file_embedding_path,episode_clips):
        episode_name=information_RAG_path.split('/')[-1].split('.')[0]
        external_memory=MemoryIndex(args.neighbours, use_openai=self.use_openai_embedding)
        if os.path.exists(file_embedding_path):
            print("Loading embeddings from pkl file")
            external_memory.load_embeddings_from_pkl(file_embedding_path)
        else:
            # will embed the information and save it in the pkl file
            external_memory.load_documents_from_json(information_RAG_path,file_embedding_path)
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
             
    def eval(self,video_path,qa_list,episode_clips):   
        os.makedirs(self.save_clips_summary, exist_ok=True)
        season=qa_list[0]['season']
        episode_name=qa_list[0]['episode']
        folder_name=qa_list[0]['show']+"_frames"
        file_path=os.path.join(self.save_clips_summary+folder_name+"_"+season+"_"+episode_name+".json")
        file_embedding_path=os.path.join(self.save_embedding+folder_name+"_"+season+"_"+episode_name+".pkl")
        print("file_embedding_path",file_embedding_path)
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                important_data = json.load(file)
            print("Already processed")
        else:
            preds,important_data=self.episode_inference(episode_clips,folder_name,use_subtitles=True)
            important_data.update(preds)
            summary = self.compine_summaries(important_data)
            preds['summary'] = summary
            important_data["summary"]=summary
            with open(file_path, 'w') as file:
                json.dump(important_data, file, indent=4)
        # Answer questions 
        pred_json=self.answer_episode_questions(qa_list,file_path,file_embedding_path,episode_clips)
        return pred_json


class MovienetEval (GoldFish_LV):
    
    def __init__(self,args,use_openai_embedding=True):
        super().__init__(args)
        self.use_openai_embedding=use_openai_embedding
        self.movienet_summaries_save_path = "new_workspace/clips_summary/movienet/"
        if self.use_openai_embedding :
            self.movienet_embedding_save_path = "new_workspace/open_ai_embedding/movienet"
        else:
            self.movienet_embedding_save_path = "new_workspace/embedding/movienet"
        os.makedirs(self.movienet_summaries_save_path, exist_ok=True)
        os.makedirs(self.movienet_embedding_save_path, exist_ok=True)
        
        self.max_sub_len=400
        self.max_num_images=45
    
    def _get_movie_data(self,videoname):
        video_images_path =f"/ibex/project/c2106/kirolos/MovieNet/240_frames/{videoname}"
        movie_clips_path =f"/ibex/project/c2106/kirolos/MovieNet/240_clips/{videoname}"
        subtitle_path = f"/ibex/project/c2106/kirolos/Movie_QA/subtitle/{videoname}.srt"
        annotation_file=f"/ibex/ai/reference/videos/MoiveNet/MovieNet/raw/files/annotation/{videoname}.json"
        # load the annotation file 
        with open(annotation_file, 'r') as f:
            movie_annotation = json.load(f)
        return video_images_path,subtitle_path,movie_annotation,movie_clips_path
    def _store_subtitles_paragraphs(self,subtitle_path,important_data,number_of_paragraphs):
        paragraphs=[]
        movie_name=subtitle_path.split('/')[-1].split('.')[0]
        # if there is no story, split the subtitles into paragraphs 
        paragraphs = split_subtitles(subtitle_path, number_of_paragraphs)  
        for i,paragraph in enumerate(paragraphs):
            paragraph=clean_text(paragraph)
            important_data.update({f"subtitle_{i}__{movie_name}_clip_{str(i).zfill(2)}": paragraph})  
        return important_data
    def _get_shots_subtitles(self,movie_annotation):
        shots_subtitles={}
        if movie_annotation['story'] is not None:
            for section in movie_annotation['story']:
                for shot in section['subtitle']:
                    shot_number=shot['shot']
                    shot_subtitle=' '.join(shot['sentences'])
                    shots_subtitles[shot_number]=clean_text(shot_subtitle)
                
    
        return shots_subtitles
    
    def prepare_input_images(self,clip_path,shots_subtitles,use_subtitles):
        total_frames=len(os.listdir(clip_path))
        sampling_interval=int(total_frames//self.max_num_images)
        if sampling_interval==0:
            sampling_interval=1
        images=[]
        img_placeholder = ""
        video_frames_path = os.path.join(clip_path)
        total_num_frames=len(os.listdir(video_frames_path))
        sampling_interval = round(total_num_frames / self.max_num_images)
        if sampling_interval == 0:
            sampling_interval = 1
        number_of_words=0
        video_images_list=sorted(os.listdir(video_frames_path))
        for i,frame in enumerate(video_images_list):
            if i % sampling_interval == 0:
                frame = Image.open(os.path.join(video_frames_path,frame)).convert("RGB") 
                frame = self.vis_processor(frame)
                images.append(frame)
                img_placeholder += '<Img><ImageHere>'
                shot_num=video_images_list[i].split('_')[1]
                if shots_subtitles.get(shot_num) is not None:
                    sub=clean_text(shots_subtitles[shot_num])
                    number_of_words+=len(sub.split(' '))
                    if number_of_words<= self.max_sub_len and use_subtitles:
                        img_placeholder+=f'<Cap>{sub}'
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

    def _get_movie_summaries(self,video_images_path,use_subtitles,shots_subtitles,movie_clips_path):
        video_images_list=sorted(os.listdir(video_images_path))
        max_caption_index = 0
        preds = {}
        movie_name=movie_clips_path.split('/')[-1]
        videos_summaries=[]
        previous_caption=""
        batch_size=args.batch_size
        batch_images=[]
        batch_instructions=[]
        clip_numbers=[]
        clip_number=0
        conversations=[]
        for i in tqdm(range(0,len(video_images_list),135), desc="Inference video clips", total=len(video_images_list)/135):
            images=[]
            img_placeholder = ""
            number_of_words=0
            clip_number_str=str(clip_number).zfill(2)
            clip_path=os.path.join(movie_clips_path,f"{movie_name}_clip_{clip_number_str}")
            
            conversation=""
            for j in range(i,i+135,3):
                if j >= len(video_images_list):
                    break
                image_path = os.path.join(video_images_path, video_images_list[j])
                # copy the images to clip folder 
                if not os.path.exists(clip_path):
                    os.makedirs(clip_path, exist_ok=True)
                    shutil.copy(image_path,clip_path)
                img=Image.open(image_path)
                images.append(self.vis_processor(img))
                img_placeholder += '<Img><ImageHere>'
                shot_num=int(video_images_list[j].split('_')[1])
                if use_subtitles:
                    if shots_subtitles.get(shot_num) is not None:
                        sub=clean_text(shots_subtitles[shot_num])
                        number_of_words+=len(sub.split(' '))
                        if number_of_words<= self.max_sub_len :
                            img_placeholder+=f'<Cap>{sub}'
                        conversation+=sub+" " 
                if len(images) >= self.max_num_images:
                    break
            if len(images) ==0:
                print("Video not found",video_images_path)
                continue
            if 0 <len(images) < self.max_num_images:
                last_item = images[-1]
                while len(images) < self.max_num_images:
                    images.append(last_item)
                    img_placeholder += '<Img><ImageHere>'
                    
            images = torch.stack(images)
            print(images.shape)
            clip_numbers.append(clip_number_str)
            clip_number+=1
            conversations.append(clean_text(conversation))
            instruction = img_placeholder + '\n' + self.summary_instruction
            batch_images.append(images)
            batch_instructions.append(instruction)
            if len(batch_images) < batch_size:
                continue
            # run inference for the batch
            batch_images = torch.stack(batch_images)
            batch_pred=self.run_images(batch_images,batch_instructions)
            for i,pred in enumerate(batch_pred):
                max_caption_index += 1
                videos_summaries.append(pred)
                preds[f'caption_{max_caption_index}__{movie_name}_clip_{clip_numbers[i]}'] = pred 
                if conversations[i]!="" and use_subtitles:
                    preds[f'subtitle_{max_caption_index}__{movie_name}_clip_{clip_numbers[i]}'] = conversations[i]
                   
            batch_images=[]
            batch_instructions=[]
            clip_numbers=[]
            conversations=[]

        # run inference for the last batch
        if len(batch_images)>0:
            batch_images = torch.stack(batch_images)
            batch_pred=self.run_images(batch_images,batch_instructions)
            for k,pred in enumerate(batch_pred):
                max_caption_index += 1
                videos_summaries.append(pred)
                preds[f'caption_{max_caption_index}__{movie_name}_clip_{clip_numbers[k]}'] = pred 
                if conversations[k]!="" and use_subtitles:
                    preds[f'subtitle_{max_caption_index}__{movie_name}_clip_{clip_numbers[k]}'] = conversations[k] 
            batch_images=[]
            batch_instructions=[]
        return preds
    def movie_inference(self,videoname,use_subtitles):
        file_path=self.movienet_summaries_save_path+videoname+".json"
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                important_data = json.load(file)
            print("Already processed")
            # return important_data['summary'] ,file_path
            return "" ,file_path

        important_data = {}
        video_images_path,subtitle_path,movie_annotation,movie_clips_path=self._get_movie_data(videoname)
        shots_subtitles={}   
        if use_subtitles: 
            if movie_annotation['story'] is not None:
                shots_subtitles=self._get_shots_subtitles(movie_annotation) 

        preds=self._get_movie_summaries(video_images_path,use_subtitles,shots_subtitles,movie_clips_path)
        important_data.update(preds)
        summary = self.compine_summaries(important_data)
        important_data['summary'] = summary
        
        with open(file_path, 'w') as file:
            json.dump(important_data, file, indent=4)
        return summary,file_path
    def answer_movie_questions_RAG(self,qa_list,external_memory):
        # get the most similar context from the external memory to this instruction 
        related_context_keys_list=[]
        related_context_documents_list=[]
        related_text=[]
        questions=[]
        prompts=[]
        for qa in qa_list:
            related_context_documents,related_context_keys = external_memory.search_by_similarity(qa['question'])
            related_context_documents_list.append(related_context_documents)
            related_context_keys_list.append(related_context_keys)
            questions.append(qa)
            prompt=self.prepare_prompt(qa)
            prompts.append(prompt)
        related_context_documents_text_list=[]
        for related_context_documents,related_context_keys in zip(related_context_documents_list,related_context_keys_list): 
            related_information=""
            most_related_clips=self.get_most_related_clips(related_context_keys)
            for clip_name in most_related_clips:
                clip_conversation=""
                general_sum=""
                for key in external_memory.documents.keys():
                    if clip_name in key and 'caption' in key:
                        general_sum="Clip Summary: "+external_memory.documents[key]
                    if clip_name in key and 'subtitle' in key:
                        clip_conversation="Clip Subtitles: "+external_memory.documents[key]
                related_information+=f"{general_sum},{clip_conversation}\n"
                
            related_context_documents_text_list.append(related_information)
            batch_pred=self.inference_RAG(prompts,related_context_documents_text_list)
            related_text.extend(related_context_documents_text_list)
        return batch_pred ,related_text
    def get_most_related_clips(self,related_context_keys):
        most_related_clips=[]
        for context_key in related_context_keys:
            if len(context_key.split('__'))>1:
                most_related_clips.append(context_key.split('__')[1])
            if len(most_related_clips)==args.neighbours:
                break
        assert len(most_related_clips)!=0, f"No related clips found {related_context_keys}"
        return most_related_clips 
    
   
    def prepare_prompt(self,qa):
        if qa.get('options',False):
            prompt=qa['question']+ "\n and these are the options for the question\n\n"
            for i,choice in enumerate(qa['options']):
                prompt+=f"option {i}: {choice} \n\n"
            prompt+="\n Your output should be THE NUMBER OF THE CORRECT ANSWER FROM THE CHOICES FROM 0 TO 4 INCLUSIVE"
        else:
            prompt=qa['question']
        return prompt


    def eval_movienet_video(self,movie,qa_list):
        use_subtitles_while_generating_summary=True
        movie_full_summary,information_RAG_path=self.movie_inference(movie,use_subtitles_while_generating_summary)
        external_memory=MemoryIndex(args.neighbours, use_openai=self.use_openai_embedding)
        if os.path.exists(f"{self.movienet_embedding_save_path}/{movie}.pkl"):
            print("Loading embeddings from pkl file")
            external_memory.load_embeddings_from_pkl(f"{self.movienet_embedding_save_path}/{movie}.pkl")
        else:
            # will embed the information and save it in the pkl file
            external_memory.load_documents_from_json(information_RAG_path,f"{self.movienet_embedding_save_path}/{movie}.pkl")
        pred_json=[]
        batch_questions=[]
        for qa in tqdm(qa_list):
            batch_questions.append(qa)
            if len(batch_questions)<args.batch_size:
                continue
            model_ans,related_text=self.answer_movie_questions_RAG(batch_questions,external_memory)
            for qa,ans,related_info in zip(batch_questions,model_ans,related_text):
                qa.update({'pred':ans})
                qa.update({'related_info':related_info})
                pred_json.append(qa)
            batch_questions=[]
        if len(batch_questions)>0:
            model_ans,related_text=self.answer_movie_questions_RAG(batch_questions,external_memory)
            for qa,ans,related_info in zip(batch_questions,model_ans,related_text):
                qa.update({'pred':ans})
                qa.update({'related_info':related_info})
                pred_json.append(qa)
        
        return pred_json





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
    print("Use openai embedding",args.use_openai_embedding)
    tvqa_eval=TVQAEVAL(args,use_openai_embedding=args.use_openai_embedding)
    movienet_eval= MovienetEval(args,use_openai_embedding=args.use_openai_embedding)
    tvqa_video_clips_mapping_path="/ibex/project/c2090/datasets/TVR_dataset/tvqa_qa_release/tvqa_video_clips_mapping.json"
    tvqa_video_clips_mapping=json.load(open(tvqa_video_clips_mapping_path))
    skill_path=args.skill_path
    args.neighbours=3
    skill_data=json.load(open(skill_path))
    skill_name=skill_path.split('/')[-1].replace('.json','')
    save_results_path=f"new_workspace/results"
    result=[]
    os.makedirs(save_results_path,exist_ok=True)
    for i,video_path in tqdm (enumerate(skill_data)):
        if args.start <= i <args.end:
            if skill_data[video_path][0]['source']=="tvqa":
                print("video_path_tvqa",video_path)
                qa_list=skill_data[video_path]
                episode_clips=tvqa_video_clips_mapping[video_path.replace('.mp4','')]
                pred_json=tvqa_eval.eval(video_path,qa_list,episode_clips)
                result.extend(pred_json)
            else:
                # MovieNet video
                print("video_path_movienet",video_path)
                qa_list=skill_data[video_path]
                pred_json=movienet_eval.eval_movienet_video(video_path.replace(".mp4","").replace('/',""),qa_list)
                result.extend(pred_json)
    with open(f"{save_results_path}/{skill_name}_{args.start}_{args.end}.json", 'w') as file:
        json.dump(result, file, indent=4)