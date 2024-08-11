"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict
import sys
sys.path.append('/ibex/project/c2090/kirolos/MiniGPT4-video-llama3')
from minigpt4.datasets.datasets.base_dataset import BaseDataset
from PIL import Image
import random
import json

import cv2
import torch
import torchvision.transforms as transforms

import numpy as np
import webvtt
import math
from moviepy.editor import VideoFileClip
from minigpt4.processors.blip_processors import Blip2ImageTrainProcessor,BlipCaptionProcessor
import pickle
import time
from decord import VideoReader, cpu, gpu
from tqdm import tqdm
import pysrt
import chardet
import re 
import whisper
from datetime import timedelta
# Function to format timestamps for VTT
def format_timestamp(seconds):
    td = timedelta(seconds=seconds)
    return str(td)

def duration_to_seconds(duration_str):
    duration_str = duration_str[2:]  # Removing 'PT' prefix
    seconds = 0
    if 'H' in duration_str:
        hours_str = duration_str.split('H')[0]
        seconds += int(hours_str) * 3600
        duration_str = duration_str.split('H')[1]
    if 'M' in duration_str:
        minutes_str = duration_str.split('M')[0]
        seconds += int(minutes_str) * 60
        duration_str = duration_str.split('M')[1]
    if 'S' in duration_str:
        seconds_str = duration_str.split('S')[0]
        seconds += int(seconds_str)
    return seconds

def extract_audio(video_path, audio_path):
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(audio_path, codec="libmp3lame", bitrate="320k")
    
def generate_subtitles(video_path,existed_subtitles,whisper_model):
    video_id=video_path.split('/')[-1].split('.')[0]
    subtitle_dir="workspace/misssing_eval_subtitles"
    audio_dir="workspace/misssing_eval_subtitles/mp3"
    os.makedirs(subtitle_dir,exist_ok=True)
    os.makedirs(audio_dir,exist_ok=True)
    audio_path = f"{audio_dir}/{video_id}"+'.mp3'
    if existed_subtitles.get(video_id,False):
        print("subtitle already generated")
        return f"{subtitle_dir}/{video_id}"+'.vtt'
    try:
        extract_audio(video_path,audio_path)
        print("successfully extracted")
        subtitle_path=f"{subtitle_dir}/{video_id}"+'.vtt'
        result = whisper_model.transcribe(audio_path,language="en") 
        # Create VTT file
        with open(subtitle_path, "w", encoding="utf-8") as vtt_file:
            vtt_file.write("WEBVTT\n\n")
            for segment in result['segments']:
                start = format_timestamp(segment['start'])
                end = format_timestamp(segment['end'])
                text = segment['text']
                vtt_file.write(f"{start} --> {end}\n{text}\n\n")
        # remove the audio file
        os.system(f"rm {audio_path}")
        print("subtitle successfully generated")  
        return subtitle_path
    except Exception as e:
        print("error",video_path ,e)
        return None

def read_subtitles(subtitle_path):
    # read the subtitle file and detect the encoding
    try:
        with open(subtitle_path, 'rb') as f:
            result = chardet.detect(f.read())
        subs = pysrt.open(subtitle_path, encoding=result['encoding']) 
        return subs
    except:
        return []
    
                  
def srt_time_to_seconds(time):
    return time.hours * 3600 + time.minutes * 60 + time.seconds + time.milliseconds / 1000
      

class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "caption": ann["caption"],
                "image": sample["image"],
            }
        )


class CMDVideoDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, subtitles_path,model_name='llama2'):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.instruction_pool = [
            'Describe this video.',
            'Provide a concise depiction of this video.',
            'Present a description of this video.',
            'Summarize this video.',
            'Generate video caption:',
            'Generate video description:',
            'Write a description for the video.',
            'Provide a description of what is presented in the video.',
            'Describe the content of the video.',
            'Can you explain what you see in the video?',
            'Could you describe what you perceive in the video?',
            'Please provide a depiction of the video.',
            'Illustrate what is happening in the video.',
        ]

        self.model_name=model_name
        if self.model_name =='mistral':
            self.length = 90
            self.max_sub_len = 800
        else:
            self.length = 45
            self.max_sub_len = 400
        
        self.subtitle_folder = subtitles_path
        self.videos_has_subtitles={}
        for sub in os.listdir(self.subtitle_folder):
            video_id = sub.split('.')[0]
            self.videos_has_subtitles[video_id] = True
        self.transform = transforms.Compose([
                transforms.ToPILImage(),
            ])
        

    def __getitem__(self, index):
        ann = self.annotation[index]
        video_id = ann["image_id"]
        answer =ann['caption']
        instruction = random.choice(self.instruction_pool)
        has_subtitles = self.videos_has_subtitles.get(video_id, False)
        if has_subtitles:
            subtitle_path = os.path.join(self.subtitle_folder, f'{video_id}.en.vtt')
            # Load the VTT subtitle file
            vtt_file = webvtt.read(subtitle_path)
        video_path = os.path.join(self.vis_root, f'{video_id}.mp4')
        clip = VideoFileClip(video_path)
        total_num_frames = int(clip.duration * clip.fps)
        clip.close()
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        sampling_interval = int(total_num_frames / self.length)
        if sampling_interval == 0:
            sampling_interval = 1
        img_placeholder = ""
        subtitle_text_in_interval = ""
        number_of_sub_words=0
        images=[]
        history_subtitles = {}
        previous_sub = ""
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Find the corresponding subtitle for the each frame and combine the interval subtitles into one subtitle
            if has_subtitles:
                for subtitle in vtt_file:
                    sub=subtitle.text.replace('\n',' ')
                    if (subtitle.start_in_seconds <= (frame_count / int(clip.fps)) <= subtitle.end_in_seconds):
                        if not history_subtitles.get(sub,False):
                            for word in sub.split(' '):
                                if word not in subtitle_text_in_interval and word not in previous_sub:
                                    subtitle_text_in_interval+=word+" "
                        history_subtitles[sub]=True
            if frame_count % sampling_interval == 0:
                frame = self.transform(frame[:,:,::-1])# BGR to RGB 
                frame = self.vis_processor(frame)
                images.append(frame)
                img_placeholder += '<Img><ImageHere>'
                if has_subtitles and number_of_sub_words<self.max_sub_len:
                    if subtitle_text_in_interval != "":
                        img_placeholder+=f'<Cap>{subtitle_text_in_interval}'
                        number_of_sub_words+=len(subtitle_text_in_interval.split(' '))
                        previous_sub = subtitle_text_in_interval
                        subtitle_text_in_interval = ""
            frame_count += 1
            if len(images) >= self.length:
                break
        cap.release()
        if len(images) ==0:
            print("Video not found",video_path)
            
        if 0 <len(images) < self.length:
            last_item = images[-1]
            while len(images) < self.length:
                images.append(last_item)
                img_placeholder += '<Img><ImageHere>'
        images = torch.stack(images)
        instruction = img_placeholder + '\n' + instruction
        return{
            "image": images,
            "answer": answer,
            "image_id": video_id,
            "instruction_input": instruction,
            "length": self.length,
        }


class WebVidDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths,subtitles_path,model_name,add_subtitles=False):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.instruction_pool = [
            'Describe this video.',
            'Provide a concise depiction of this video.',
            'Present a description of this video.',
            'Summarize this video.',
            'Generate video caption:',
            'Generate video description:',
            'Write a description for the video.',
            'Provide a description of what is presented in the video.',
            'Describe the content of the video.',
            'Can you explain what you see in the video?',
            'Could you describe what you perceive in the video?',
            'Please provide a depiction of the video.',
            'Illustrate what is happening in the video.',
        ]
        self.model_name=model_name
        if self.model_name =='mistral':
            self.length = 90
            self.max_sub_len = 800
        else:
            self.length = 45
            self.max_sub_len = 400
        self.add_subtitles = add_subtitles
        self.videos_has_subtitles = {}
        if self.add_subtitles:
            self.subtitle_folder = os.path.join(subtitles_path)
            for sub in os.listdir(self.subtitle_folder):
                video_id = sub.split('.')[0]
                self.videos_has_subtitles[video_id] = True
        self.transform = transforms.Compose([
                transforms.ToPILImage(),
            ])

    def __getitem__(self, index):
        ann = self.annotation[index]
        video_id = ann["videoid"]
        images = []
        caption = ann["name"].split('-')[-1].split(':')[-1]
        # caption = self.text_processor(caption)
        video_path = os.path.join(self.vis_root, ann['page_dir'], f'{video_id}.mp4')
        has_subtitles = self.videos_has_subtitles.get(video_id, False)
        if self.add_subtitles and has_subtitles:
            subtitle_path = os.path.join(self.subtitle_folder, f'{video_id}.vtt')
            # Load the VTT subtitle file
            vtt_file = webvtt.read(subtitle_path)
                 
        cap = cv2.VideoCapture(video_path)
        clip = VideoFileClip(video_path)
        total_num_frames = int(clip.duration * clip.fps)
        clip.close()
        cap = cv2.VideoCapture(video_path)
        images = []
        frame_count = 0
        sampling_interval = int(total_num_frames /self.length)
        if sampling_interval == 0:
            sampling_interval = 1
        img_placeholder = ""
        subtitle_text_in_interval = ""
        history_subtitles = {}
        number_of_sub_words=0
        previous_sub = ""
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Find the corresponding subtitle for the each frame and combine the interval subtitles into one subtitle

            if self.add_subtitles and has_subtitles:
                for subtitle in vtt_file:
                    sub=subtitle.text.replace('\n',' ')
                    if (subtitle.start_in_seconds <= (frame_count / int(clip.fps)) <= subtitle.end_in_seconds):
                        if not history_subtitles.get(sub,False):
                            for word in sub.split(' '):
                                if word not in subtitle_text_in_interval and word not in previous_sub:
                                    subtitle_text_in_interval+=word+" "
                        history_subtitles[sub]=True
            if frame_count % sampling_interval == 0:
                frame = self.transform(frame[:,:,::-1])
                frame = self.vis_processor(frame)
                images.append(frame)
                img_placeholder += '<Img><ImageHere>'
                if self.add_subtitles and has_subtitles and subtitle_text_in_interval != "" and number_of_sub_words<self.max_sub_len:
                    img_placeholder+=f'<Cap>{subtitle_text_in_interval}'
                    number_of_sub_words+=len(subtitle_text_in_interval.split(' '))
                    previous_sub = subtitle_text_in_interval
                    subtitle_text_in_interval = ""
            frame_count += 1
            if len(images) >= self.length:
                break
        cap.release()

        if len(images) < self.length:
            last_item = images[-1]
            while len(images) < self.length:
                images.append(last_item)
                img_placeholder += '<Img><ImageHere>'

        images = torch.stack(images)
        instruction = random.choice(self.instruction_pool)
        instruction = img_placeholder + '\n' + instruction
        return {
            "image": images,
            "answer": caption,
            "image_id": video_id,
            "instruction_input": instruction,
            "length": self.length,
        }

class VideoChatGPTDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths,subtitles_path,model_name='llama2',add_subtitles=True):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.img_ids = {}
        n=0
        self.model_name=model_name
        if self.model_name =='mistral':
            self.length = 90
            self.max_sub_len = 800
        else:
            self.length = 45
            self.max_sub_len = 400
        self.add_subtitles = add_subtitles
        self.videos_has_subtitles = {}
        if self.add_subtitles:
            self.subtitle_folder = subtitles_path
            for sub in os.listdir(self.subtitle_folder):
                video_id = sub.split('.')[0]
                self.videos_has_subtitles[video_id] = True
        for ann in self.annotation:
            img_id = ann["video_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n+= 1
            
        self.videos_extension={}
        for video in os.listdir(self.vis_root):
            self.videos_extension[video.split('.')[0]]=video.split('.')[1]

        self.transform = transforms.Compose([
                transforms.ToPILImage(),
            ])
    def __len__(self):
        return len(self.annotation)
    def __getitem__(self, index):
        ann = self.annotation[index]
        video_id = ann["video_id"]
        answer=ann["a"]
        instruction=ann["q"]
        images=[]
        img_placeholder = ""
        has_subtitles = self.videos_has_subtitles.get(video_id, False)
        if self.add_subtitles and has_subtitles:
            subtitle_path = os.path.join(self.subtitle_folder, f'{video_id}.vtt')
            # Load the VTT subtitle file
            vtt_file = webvtt.read(subtitle_path)
                
        video_path = os.path.join(self.vis_root,f'{video_id}.{self.videos_extension[video_id]}')
        clip = VideoFileClip(video_path)
        total_num_frames = int(clip.duration * clip.fps)
        clip.close()
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        sampling_interval = int(total_num_frames / self.length)
        if sampling_interval == 0:
            sampling_interval = 1
        img_placeholder = ""
        subtitle_text_in_interval = ""
        history_subtitles = {}
        number_of_sub_words=0
        previous_sub = ""
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Find the corresponding subtitle for the each frame and combine the interval subtitles into one subtitle

            if self.add_subtitles and has_subtitles:
                for subtitle in vtt_file:
                    sub=subtitle.text.replace('\n',' ')
                    if (subtitle.start_in_seconds <= (frame_count / int(clip.fps)) <= subtitle.end_in_seconds):
                        if not history_subtitles.get(sub,False):
                            for word in sub.split(' '):
                                if word not in subtitle_text_in_interval and word not in previous_sub:
                                    subtitle_text_in_interval+=word+" "
                        history_subtitles[sub]=True
            if frame_count % sampling_interval == 0:
                frame = self.transform(frame[:,:,::-1])# BGR to RGB 
                frame = self.vis_processor(frame)
                images.append(frame)
                img_placeholder += '<Img><ImageHere>'
                if self.add_subtitles and has_subtitles and number_of_sub_words<self.max_sub_len:
                    if subtitle_text_in_interval != "":
                        img_placeholder+=f'<Cap>{subtitle_text_in_interval}'
                        number_of_sub_words+=len(subtitle_text_in_interval.split(' '))
                        previous_sub = subtitle_text_in_interval
                        subtitle_text_in_interval = ""
            frame_count += 1
            if len(images) >= self.length:
                break
        cap.release()
        if len(images) ==0:
            print("Video not found",video_path)
            
        if 0 <len(images) < self.length:
            last_item = images[-1]
            while len(images) < self.length:
                images.append(last_item)
                img_placeholder += '<Img><ImageHere>'
        images = torch.stack(images)
        instruction = img_placeholder + '\n' + instruction
        return{
            "image": images,
            "answer": answer,
            "image_id": video_id,
            "instruction_input": instruction,
            "length": self.length,
        }

    
class VideoChatGPTEvalDataset(torch.utils.data.Dataset):
    def __init__(self, vis_processor, videos_path, ann_path,subtitles_path,annotations_keys,add_subtitles=True,llm_name="llama2"):
        if llm_name=="llama2":
            self.length = 45
            self.max_sub_len = 400
        else:
            self.length = 90
            self.max_sub_len = 800
        self.add_subtitles = add_subtitles
        if subtitles_path=="":
            self.add_subtitles=False
        self.vis_processor=vis_processor
        self.videos_path=videos_path
        self.question_key=annotations_keys[0]
        self.answer_key=annotations_keys[1]
        self.video_name_key=annotations_keys[2]
        self.videos_extension={}
        for video in os.listdir(self.videos_path):
            self.videos_extension[video.split('.')[0]]=video.split('.')[1]
        self.annotation=json.load(open(ann_path,'r'))
        self.videos_has_subtitles = {}
        if self.add_subtitles:
            self.subtitle_folder = subtitles_path
            for sub in os.listdir(self.subtitle_folder):
                video_id = sub.split('.')[0]
                self.videos_has_subtitles[video_id] = True
        self.transform = transforms.Compose([
                transforms.ToPILImage(),
            ])
        self.whisper_model=whisper.load_model("large",device=f"cuda:0")
        
    def __len__(self):
        return len(self.annotation)
    def __getitem__(self, index):
        ann = self.annotation[index]
        video_id = ann[self.video_name_key]
        answer=ann[self.answer_key]
        instruction=ann[self.question_key]
        images=[]
        img_placeholder = ""
        video_path = os.path.join(self.videos_path,f'{video_id}.{self.videos_extension[video_id]}')
        cap = cv2.VideoCapture(video_path)
        clip = VideoFileClip(video_path)
        total_num_frames = int(clip.duration * clip.fps)
        clip.close()
        frame_count = 0
        sampling_interval = int(total_num_frames / self.length)
        if sampling_interval == 0:
            sampling_interval = 1
        subtitle_path=None
        if self.add_subtitles :
            subtitle_path = generate_subtitles(video_path,self.videos_has_subtitles,self.whisper_model)
            if  subtitle_path is not None:
                # Load the VTT subtitle file
                vtt_file = webvtt.read(subtitle_path)
        subtitle_text_in_interval = ""
        history_subtitles = {}
        number_of_sub_words=0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Find the corresponding subtitle for the each frame and combine the interval subtitles into one subtitle

            if self.add_subtitles and subtitle_path is not None:
                for subtitle in vtt_file:
                    sub=subtitle.text.replace('\n',' ')
                    if (subtitle.start_in_seconds <= (frame_count / int(cap.get(cv2.CAP_PROP_FPS))) <= subtitle.end_in_seconds) and sub not in subtitle_text_in_interval:
                        if not history_subtitles.get(sub,False):
                            subtitle_text_in_interval+=sub+" "
                        history_subtitles[sub]=True
                        break
            if frame_count % sampling_interval == 0:
                frame = self.transform(frame[:,:,::-1])
                frame = self.vis_processor(frame)
                images.append(frame)
                img_placeholder += '<Img><ImageHere>'
                if self.add_subtitles and subtitle_path is not None and number_of_sub_words<self.max_sub_len and subtitle_text_in_interval != "":
                    img_placeholder+=f'<Cap>{subtitle_text_in_interval}'
                    number_of_sub_words+=len(subtitle_text_in_interval.split(' '))
                    subtitle_text_in_interval = ""
            frame_count += 1
            if len(images) >= self.length:
                break
        cap.release()
        if len(images) == 0:
            print("Video not found")
            print('Video path',video_path)
            return None,None,None,None,None
        if  0 <len(images) < self.length:
            last_image = images[-1]
            while len(images) < self.length:
                images.append(last_image)
                img_placeholder += '<Img><ImageHere>'
        images = torch.stack(images)
        instruction = img_placeholder + '\n' + instruction
        return images,instruction,answer,self.length,video_id

class Video_validation_Dataset(torch.utils.data.Dataset):
    def __init__(self, vis_processor, videos_path, ann_path,subtitles_path,annotations_keys,add_subtitles=True,llm_name="llama2"):
        if llm_name=="llama2":
            self.length = 45
            self.max_sub_len = 400
        else:
            self.length = 90
            self.max_sub_len = 800
        self.add_subtitles = add_subtitles
        self.vis_processor=vis_processor
        self.videos_path=videos_path
        self.question_key=annotations_keys[0]
        self.answer_key=annotations_keys[1]
        self.video_name_key=annotations_keys[2]
        self.videos_extension={}
        for video in os.listdir(self.videos_path):
            self.videos_extension[video.split('.')[0]]=video.split('.')[1]
        self.annotation=json.load(open(ann_path,'r'))
        self.videos_has_subtitles = {}
        if self.add_subtitles:
            self.subtitle_folder = subtitles_path
            for sub in os.listdir(self.subtitle_folder):
                video_id = sub.split('.')[0]
                self.videos_has_subtitles[video_id] = True
        self.transform = transforms.Compose([
                transforms.ToPILImage(),
            ])
        self.whisper_model=whisper.load_model("large",device=f"cuda:0")
        
    def __len__(self):
        return len(self.annotation)
    def __getitem__(self, index):
        ann = self.annotation[index]
        video_id = ann[self.video_name_key]
        answer=ann[self.answer_key]
        instruction=ann[self.question_key]
        video_path = os.path.join(self.videos_path,f'{video_id}.{self.videos_extension[video_id]}')
        images=[]
        img_placeholder = ""
        cap = cv2.VideoCapture(video_path)
        clip = VideoFileClip(video_path)
        total_num_frames = int(clip.duration * clip.fps)
        clip.close()
        frame_count = 0
        sampling_interval = int(total_num_frames / self.length)
        if sampling_interval == 0:
            sampling_interval = 1
        subtitle_path=None
        if self.add_subtitles :
            subtitle_path = generate_subtitles(video_path,self.videos_has_subtitles,self.whisper_model)
            if  subtitle_path is not None:
                # Load the VTT subtitle file
                vtt_file = webvtt.read(subtitle_path)
        subtitle_text_in_interval = ""
        history_subtitles = {}
        number_of_sub_words=0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Find the corresponding subtitle for the each frame and combine the interval subtitles into one subtitle

            if self.add_subtitles and subtitle_path is not None:
                for subtitle in vtt_file:
                    sub=subtitle.text.replace('\n',' ')
                    if (subtitle.start_in_seconds <= (frame_count / int(cap.get(cv2.CAP_PROP_FPS))) <= subtitle.end_in_seconds) and sub not in subtitle_text_in_interval:
                        if not history_subtitles.get(sub,False):
                            subtitle_text_in_interval+=sub+" "
                        history_subtitles[sub]=True
                        break
            if frame_count % sampling_interval == 0:
                frame = self.transform(frame[:,:,::-1])
                frame = self.vis_processor(frame)
                images.append(frame)
                img_placeholder += '<Img><ImageHere>'
                if self.add_subtitles and subtitle_path is not None and number_of_sub_words<self.max_sub_len and subtitle_text_in_interval != "":
                    img_placeholder+=f'<Cap>{subtitle_text_in_interval}'
                    number_of_sub_words+=len(subtitle_text_in_interval.split(' '))
                    subtitle_text_in_interval = ""
            frame_count += 1
            if len(images) >= self.length:
                break
        cap.release()
        if len(images) == 0:
            print("Video not found")
            print('Video path',video_path)
            return None,None,None,None,None
        if  0 <len(images) < self.length:
            last_image = images[-1]
            while len(images) < self.length:
                images.append(last_image)
                img_placeholder += '<Img><ImageHere>'
        images = torch.stack(images)
        instruction = img_placeholder + '\n' + instruction
        return images,instruction,answer,self.length,video_id


class VideoChatGPTEval_consistancy(torch.utils.data.Dataset):
    def __init__(self, vis_processor, videos_path, ann_path,subtitles_path,annotations_keys,add_subtitles=True,llm_name="llama2"):
        if llm_name=="llama2":
            self.length = 45
            self.max_sub_len = 400
        else:
            self.length = 90
            self.max_sub_len = 800
        self.add_subtitles = add_subtitles
        self.vis_processor=vis_processor
        self.videos_path=videos_path
        self.question1_key=annotations_keys[0][0]
        self.question2_key=annotations_keys[0][1]
        self.answer_key=annotations_keys[1]
        self.video_name_key=annotations_keys[2]
        self.videos_extension={}
        for video in os.listdir(self.videos_path):
            self.videos_extension[video.split('.')[0]]=video.split('.')[1]
        self.annotation=json.load(open(ann_path,'r'))
        self.videos_has_subtitles = {}
        if self.add_subtitles:
            self.subtitle_folder = subtitles_path
            for sub in os.listdir(self.subtitle_folder):
                video_id = sub.split('.')[0]
                self.videos_has_subtitles[video_id] = True
        self.transform = transforms.Compose([
                transforms.ToPILImage(),
            ])
        self.whisper_model=whisper.load_model("large",device=f"cuda:0")
    def __len__(self):
        return len(self.annotation)
    def __getitem__(self, index):
        ann = self.annotation[index]
        video_id = ann[self.video_name_key]
        answer=ann[self.answer_key]
        instruction_1=ann[self.question1_key]
        instruction_2=ann[self.question2_key]
        video_path = os.path.join(self.videos_path,f'{video_id}.{self.videos_extension[video_id]}')
        cap = cv2.VideoCapture(video_path)
        clip = VideoFileClip(video_path)
        total_num_frames = int(clip.duration * clip.fps)
        clip.close()
        images = []
        frame_count = 0
        sampling_interval = int(total_num_frames / self.length)
        if sampling_interval == 0:
            sampling_interval = 1
        subtitle_path=None
        if self.add_subtitles :
            subtitle_path = generate_subtitles(video_path,self.videos_has_subtitles,self.whisper_model)
            if  subtitle_path is not None:
                # Load the VTT subtitle file
                vtt_file = webvtt.read(subtitle_path)
        img_placeholder = ""
        subtitle_text_in_interval = ""
        history_subtitles = {}
        number_of_sub_words=0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Find the corresponding subtitle for the each frame and combine the interval subtitles into one subtitle

            if self.add_subtitles and subtitle_path is not None:
                for subtitle in vtt_file:
                    sub=subtitle.text.replace('\n',' ')
                    if (subtitle.start_in_seconds <= (frame_count / int(cap.get(cv2.CAP_PROP_FPS))) <= subtitle.end_in_seconds) and sub not in subtitle_text_in_interval:
                        if not history_subtitles.get(sub,False):
                            subtitle_text_in_interval+=sub+" "
                        history_subtitles[sub]=True
                        break
            if frame_count % sampling_interval == 0:
                frame = self.transform(frame[:,:,::-1])
                frame = self.vis_processor(frame)
                images.append(frame)
                img_placeholder += '<Img><ImageHere>'
                if self.add_subtitles and subtitle_path is not None and number_of_sub_words<self.max_sub_len and subtitle_text_in_interval != "":
                    img_placeholder+=f'<Cap>{subtitle_text_in_interval}'
                    number_of_sub_words+=len(subtitle_text_in_interval.split(' '))
                    subtitle_text_in_interval = ""
            frame_count += 1
            if len(images) >= self.length:
                break
        cap.release()
        if len(images) == 0:
            print("Video not found")
            print('Video path',video_path)
            return None,None,None,None,None
        if  0 <len(images) < self.length:
            last_image = images[-1]
            while len(images) < self.length:
                images.append(last_image)
                img_placeholder += '<Img><ImageHere>'
        images = torch.stack(images)
        instruction_1 = img_placeholder + '\n' + instruction_1
        instruction_2 = img_placeholder + '\n' + instruction_2
        return images,instruction_1,instruction_2,answer,self.length,video_id


                        
class TVQAEVAL (torch.utils.data.Dataset):
    def __init__(self, vis_processor, videos_path, ann_path,subtitles_path,add_subtitles=True,llm_name="llama2"):
        self.tv_shows_mapping={"Grey's Anatomy":"grey_frames", 'How I Met You Mother':"met_frames", 'Friends':"friends_frames", 'The Big Bang Theory':"bbt_frames", 'House M.D.':"house_frames", 'Castle':"castle_frames"} 
        self.fps=3
        if llm_name=="llama2":
            self.length = 45
            self.max_sub_len = 400
        else:
            self.length = 90
            self.max_sub_len = 800
        self.add_subtitles = add_subtitles
        self.vis_processor=vis_processor
        self.videos_path=videos_path
        with open(ann_path,'r') as f:
            self.annotation=json.load(f)
        with open(subtitles_path,'r') as f:
            self.subtitles_list=json.load(f)
        self.subtitles={}
        for sub in self.subtitles_list:
            self.subtitles[sub["vid_name"]]=sub["sub"]
        self.transform = transforms.Compose([
                transforms.ToPILImage(),
            ])
        
    def __len__(self):
        return len(self.annotation)
    def __getitem__(self, index):
        ann = self.annotation[index]
        video_id = ann["vid_name"]
        answer=str(ann['answer_idx'])
        folder_name=self.tv_shows_mapping[ann["show_name"]]
        instruction=ann["q"]+" \n\n As you watched in this video Choose ONE suitable answer from these mutiple choices \n\n"
        for i in range(5):
            ans=ann[f"a{i}"]
            instruction+=f"option {i}: {ans} \n\n"
        instruction+="\n Your output should be THE NUMBER OF THE CORRECT ANSWER FROM THE CHOICES FROM 0 TO 4 INCLUSIVE"
        images=[]
        img_placeholder = ""

        video_frames_path = os.path.join(self.videos_path,folder_name,video_id)
        total_num_frames=len(os.listdir(video_frames_path))
        sampling_interval = round(total_num_frames / self.length)
        if sampling_interval == 0:
            sampling_interval = 1
        subtitle_text_in_interval = ""
        history_subtitles = {}
        number_of_sub_words=0
        for i,frame in enumerate(sorted(os.listdir(video_frames_path))):
            # Find the corresponding subtitle for the each frame and combine the interval subtitles into one subtitle

            if self.add_subtitles:
                for subtitle in self.subtitles[video_id]:
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
                if self.add_subtitles and number_of_sub_words<self.max_sub_len:
                    if subtitle_text_in_interval != "":
                        img_placeholder+=f'<Cap>{subtitle_text_in_interval}'
                        number_of_sub_words+=len(subtitle_text_in_interval.split(' '))
                        subtitle_text_in_interval = ""
            if len(images) >= self.length:
                break
        if len(images) ==0:
            print("Video not found",video_frames_path)
            
        if 0 <len(images) < self.length:
            last_item = images[-1]
            while len(images) < self.length:
                images.append(last_item)
                img_placeholder += '<Img><ImageHere>'
        images = torch.stack(images)
        instruction = img_placeholder + '\n' + instruction
        return images,instruction,answer,self.length,video_id  






class Video_loader_template(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths,subtitles_path,model_name='llama2',add_subtitles=True):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.model_name=model_name
        if self.model_name =='mistral':
            self.length = 90
            self.max_sub_len = 800
        else:
            self.length = 45
            self.max_sub_len = 400
        self.add_subtitles = add_subtitles
        self.videos_has_subtitles = {}
        if self.add_subtitles:
            self.subtitle_folder = subtitles_path
            for sub in os.listdir(self.subtitle_folder):
                video_id = sub.split('.')[0]
                self.videos_has_subtitles[video_id] = True
        self.videos_extension={}
        for video in os.listdir(os.path.join(self.vis_root,'videos')):
            self.videos_extension[video.split('.')[0]]=video.split('.')[1]
        self.transform = transforms.Compose([
                transforms.ToPILImage(),
            ])
    def __len__(self):
        return len(self.annotation)
    def __getitem__(self, index):
        ann = self.annotation[index]
        video_id = ann["video_id"] # video_id
        answer=ann["a"] # answer (ground truth)
        instruction=ann["q"] # question (instruction)
        images=[]
        img_placeholder = ""
        has_subtitles = self.videos_has_subtitles.get(video_id, False)
        if self.add_subtitles and has_subtitles:
            subtitle_path = os.path.join(self.subtitle_folder, f'{video_id}.vtt')
            # Load the VTT subtitle file
            vtt_file = webvtt.read(subtitle_path)
                
        video_path = os.path.join(self.vis_root,'videos',f'{video_id}.{self.videos_extension[video_id]}')
        clip = VideoFileClip(video_path)
        total_num_frames = int(clip.duration * clip.fps)
        clip.close()
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        # Choose sampling interval based on the total number of frames in the video and the desired length of the video
        sampling_interval = int(total_num_frames / self.length)
        if sampling_interval == 0:
            sampling_interval = 1
        img_placeholder = ""
        subtitle_text_in_interval = ""
        history_subtitles = {}
        number_of_sub_words=0
        # Iterate through the video frames and extract the frames based on the sampling interval and add the subtitles if needed
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Find the corresponding subtitle for the each frame and combine the interval subtitles into one subtitle
            if self.add_subtitles and has_subtitles:
                for subtitle in vtt_file:
                    sub=subtitle.text.replace('\n',' ')
                    if (subtitle.start_in_seconds <= (frame_count / int(clip.fps)) <= subtitle.end_in_seconds) and sub not in subtitle_text_in_interval:
                        if not history_subtitles.get(sub,False):
                            subtitle_text_in_interval+=sub+" "
                        history_subtitles[sub]=True
                        break
            if frame_count % sampling_interval == 0:
                frame = self.transform(frame[:,:,::-1])# BGR to RGB 
                frame = self.vis_processor(frame)
                images.append(frame)
                img_placeholder += '<Img><ImageHere>'
                if self.add_subtitles and has_subtitles and number_of_sub_words<self.max_sub_len:
                    if subtitle_text_in_interval != "":
                        img_placeholder+=f'<Cap>{subtitle_text_in_interval}'
                        number_of_sub_words+=len(subtitle_text_in_interval.split(' '))
                        subtitle_text_in_interval = ""
            frame_count += 1
            if len(images) >= self.length:
                break
        cap.release()
        if len(images) ==0:
            print("Video not found",video_path)
            
        if 0 <len(images) < self.length:
            last_item = images[-1]
            while len(images) < self.length:
                images.append(last_item)
                img_placeholder += '<Img><ImageHere>'
        images = torch.stack(images)
        # Combine the images and the instruction
        instruction = img_placeholder + '\n' + instruction
        # Return the images, instruction, answer, video_id, and the length of the video
        return{
            "image": images,
            "answer": answer,
            "image_id": video_id,
            "instruction_input": instruction,
            "length": self.length,
        }
             