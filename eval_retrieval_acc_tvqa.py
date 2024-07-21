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
from pytube import YouTube
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
    parser.add_argument("--options", nargs="+")
    return parser.parse_args()
   
def clean_text(subtitles_text):
    # Remove unwanted characters except for letters, digits, and single quotes
    subtitles_text = re.sub(r'[^a-zA-Z0-9\s\']', '', subtitles_text)
    # Replace multiple spaces with a single space
    subtitles_text = re.sub(r'\s+', ' ', subtitles_text)
    return subtitles_text.strip()
                     
class TVQAEVALRetrieval (GoldFish_LV):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.tv_shows_mapping={"Grey's Anatomy":"grey_frames", 'How I Met You Mother':"met_frames", 'Friends':"friends_frames", 'The Big Bang Theory':"bbt_frames", 'House M.D.':"house_frames", 'Castle':"castle_frames"} 
        self.save_long_videos_path = f"workspace/results/tv_shows/{args.name}"
        os.makedirs(self.save_long_videos_path, exist_ok=True)
        self.max_sub_len=400
        self.max_num_images=45
        self.fps=3
        with open("datasets/evaluation_datasets/goldfish_eval_datasets/tvqa/tvqa_preprocessed_subtitles.json") as f:
            self.subtitles_list=json.load(f)
        self.subtitles={}
        for sub in self.subtitles_list:
            self.subtitles[sub["vid_name"]]=sub["sub"]
     
    def _get_TVs_data(self):
        json_file_path="datasets/evaluation_datasets/long_video_datasets/tvqa/tvqa_val_edited.json"
        frames_path="/ibex/project/c2090/datasets/TVR_dataset/videos/video_files/frames_hq/"
        subtitle_path="/ibex/project/c2090/datasets/TVR_dataset/videos/tvqa_subtitles" 
        with open (json_file_path) as f:
            tv_shows_data=json.load(f)
        return tv_shows_data,frames_path,subtitle_path
        
        return vision_questions,subtitle_questions,frames_path
    def episode_inference(self,video_frames_path,qa,use_subtitles):
        batch_prepared_images,batch_img_placeholder,gt_clip_numbers=self.prepare_input_images(video_frames_path,qa,use_subtitles,n_clips=10)
        preds={}
        batch_instructions=[]
        batch_images=[]
        important_data = {}
        conversations=[]
        clips_numbers=[]
        for clip_number,images,img_placeholder in zip(range(len(batch_prepared_images)),batch_prepared_images,batch_img_placeholder):
            instruction = img_placeholder + '\n' + self.summary_instruction
            batch_images.append(images)
            batch_instructions.append(instruction)
            conv=img_placeholder.replace('<Img><ImageHere>','')
            conv=conv.replace('<Cap>',' ')
            conversations.append(conv.strip())
            clips_numbers.append(clip_number)
            if len(batch_images) < args.batch_size:
                continue
            batch_images = torch.stack(batch_images)
            setup_seeds(seed)
            batch_pred=self.run_images(batch_images,batch_instructions)
            for i,pred in enumerate(batch_pred):
                if args.use_coherent_description:
                    preds[f'caption__{clips_numbers[i]}'] = f"model_summary :{pred}\nVideo conversation :{conversations[i]}"
                else:
                    if use_subtitles:
                        if conversations[i] != "":
                            important_data.update({f"subtitle__{clips_numbers[i]}": conversations[i]})   
                    preds[f'caption__{clips_numbers[i]}'] = pred
            
            batch_images=[]
            batch_instructions=[]
            conversations=[]
            clips_numbers=[]
        # run inference for the last batch
        if len(batch_images)>0:
            batch_images = torch.stack(batch_images)
            batch_pred=self.run_images(batch_images,batch_instructions)
            for i,pred in enumerate(batch_pred):
                if args.use_coherent_description:
                    preds[f'caption__{clips_numbers[i]}'] = f"model_summary :{pred}\nVideo conversation :{conversations[i]}"
                else:
                    if use_subtitles:
                        if conversations[i] != "":
                            important_data.update({f"subtitle__{clips_numbers[i]}": conversations[i]})  
                    preds[f'caption__{clips_numbers[i]}'] = pred
            batch_images=[]
            batch_instructions=[]
            clips_numbers=[]
        return preds,important_data ,gt_clip_numbers
    
    def episode_inference_only_subtitles(self,video_frames_path,qa):
        use_subtitles=True
        batch_prepared_images,batch_img_placeholder,gt_clip_numbers=self.prepare_input_images(video_frames_path,qa,use_subtitles,n_clips=10)
        important_data = {}
        for clip_number,img_placeholder in enumerate(batch_img_placeholder) :
            conv=img_placeholder.replace('<Img><ImageHere>','')
            conv=conv.replace('<Cap>',' ')
            conversation=conv.strip()
            conversation=clean_text(conversation)
            if conversation != "":
                important_data.update({f"subtitle__{clip_number}": conversation})
        return important_data ,gt_clip_numbers
    def prepare_input_images(self,video_frames_path,qa,use_subtitles,n_clips=10):
        batch_images=[]
        batch_img_placeholder = []
        clip_name=video_frames_path.split('/')[-1]
        images=[]
        img_placeholders = []
        gt_clip_numbers = set()
        gt_start_time=qa['ts'][0]
        gt_end_time=qa['ts'][1]
        total_num_frames=len(os.listdir(video_frames_path))
        subtitle_text_in_interval = ""
        history_subtitles = {}
        number_of_sub_words=0
        # samples_per_clip = total_num_frames // n_clips
        samples_per_clip=45
        clip_num=0
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
            if gt_start_time<=(i/self.fps)<= gt_end_time:
                    gt_clip_numbers.add(clip_num)
            if i % samples_per_clip == 0 and i != 0:
                # here we have one clip , let's sample 45 frames from images array
                sample_value=len(images)//self.max_num_images
                if sample_value==0:
                    sample_value=1
                frames_indices = [i for i in range(0, len(images), sample_value)]
                samples_images=[]
                img_placeholder=''
                for j in frames_indices:
                    samples_images.append(images[j])
                    img_placeholder+=img_placeholders[j]
                    if len(samples_images) >= self.max_num_images:
                        break
                if 0 <len(samples_images) < self.max_num_images:
                    last_item = samples_images[-1]
                    while len(samples_images) < self.max_num_images:
                        samples_images.append(last_item)
                        img_placeholder += '<Img><ImageHere>'
                samples_images = torch.stack(samples_images)
                batch_images.append(samples_images)
                batch_img_placeholder.append(img_placeholder)
                img_placeholders =[] 
                images = []
                clip_num+=1
                
            frame = Image.open(os.path.join(video_frames_path,frame)).convert("RGB") 
            frame = self.vis_processor(frame)
            images.append(frame)
            img_placeholder = '<Img><ImageHere>'
            if number_of_sub_words<self.max_sub_len and use_subtitles:
                if subtitle_text_in_interval != "":
                    subtitle_text_in_interval=clean_text(subtitle_text_in_interval)
                    img_placeholder+=f'<Cap>{subtitle_text_in_interval}'
                    number_of_sub_words+=len(subtitle_text_in_interval.split(' '))
                    subtitle_text_in_interval = ""
            img_placeholders.append(img_placeholder)
        return batch_images,batch_img_placeholder,list(gt_clip_numbers)

    def test_retrieval(self,indexed_data_path,qa,gt_clip_numbers):  
        external_memory=MemoryIndex(args.neighbours, use_openai=True)
        external_memory.load_documents_from_json(indexed_data_path)
        question=qa['desc']
        related_context_documents,related_context_keys = external_memory.search_by_similarity(question)
        print(f"related_context_keys {related_context_keys}")
        print(f"gt_clip_numbers {gt_clip_numbers}")
        for key in related_context_keys:
            clip_idx=int(key.split('__')[-1])
            if clip_idx in gt_clip_numbers:
                return True
        return False
    
    def get_ground_truth_clip(self,video_frames_path,qa):
        gt_clip_numbers = set()
        gt_start_time=qa['ts'][0]
        gt_end_time=qa['ts'][1]
        samples_per_clip=45
        clip_num=0
        for i in range(len(os.listdir(video_frames_path))):
            if gt_start_time<=(i/self.fps)<= gt_end_time:
                    gt_clip_numbers.add(clip_num)
            if i % samples_per_clip == 0 and i != 0:
                clip_num+=1  
        return list(gt_clip_numbers)
                
    def eval_tv_shows(self,):
        vision_questions,subtitle_questions,frames_path=self._get_TVs_data()
        number_of_videos=0
        start=args.start
        end=args.end
        if args.exp_name=="vision":
            questions=vision_questions
        else:
            questions=subtitle_questions
        correct_retrieval=0
        wrong_retrieval=0
        for qa in questions:
            # Generate clips summary and store the important data (summary and subtitles) in json file
            if start<=number_of_videos<end: 
                show_name=qa['vid_name'].split('_')[0]
                if self.tv_shows_mapping.get(show_name,False):
                    folder_name=self.tv_shows_mapping[show_name]
                else:
                    folder_name=self.tv_shows_mapping['bbt']
                
                clip_frames_path =os.path.join(frames_path,folder_name,qa['vid_name'])
                save_name="subtitles_only" if args.subtitles_only else "vision_only" if args.vision_only else "vision_subtitles"
                indexed_data_path=os.path.join(self.save_long_videos_path,f"{qa['vid_name']}_{args.exp_name}_{save_name}_num_{number_of_videos}.json")
                if not os.path.exists(indexed_data_path):
                    if args.subtitles_only :
                        # TODO
                        important_data,gt_clip_numbers=self.episode_inference_only_subtitles(clip_frames_path,qa)
                    else:
                        preds,important_data ,gt_clip_numbers=self.episode_inference(clip_frames_path,qa,use_subtitles=not args.vision_only)
                        important_data.update(preds)
                    with open(indexed_data_path, 'w') as file:
                        json.dump(important_data, file, indent=4)
                else:
                    gt_clip_numbers=self.get_ground_truth_clip(clip_frames_path,qa)
                retrieval_res=self.test_retrieval(indexed_data_path,qa,gt_clip_numbers)
                if retrieval_res==True:
                    correct_retrieval+=1
                else:
                    wrong_retrieval+=1
            number_of_videos+=1
        
        save_dir=f"workspace/eval/retrieval/{args.exp_name}_neighbors_{args.neighbours}"
        save_dir+="_subtitles_only" if args.subtitles_only else "_vision_only" if args.vision_only else "_vision_subtitles"
        os.makedirs(save_dir,exist_ok=True)
        with open(f"{save_dir}/s{start}_end{end}.json", 'w') as fp:
            json.dump({"correct":correct_retrieval,"wrong":wrong_retrieval}, fp)
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
    tvqa_eval=TVQAEVALRetrieval(args)
    tvqa_eval.eval_tv_shows()