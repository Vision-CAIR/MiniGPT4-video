#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
from torchvision import transforms
from pytubefix import YouTube
from minigpt4.common.eval_utils import init_model
from minigpt4.conversation.conversation import CONV_VISION
from index import MemoryIndex  
import pysrt
import chardet
from openai import OpenAI
if os.getenv("OPENAI_API_KEY") is not None:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
else:
    client = OpenAI(api_key="")
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from transformers import BitsAndBytesConfig



def time_to_seconds(subrip_time):
    return subrip_time.hours * 3600 + subrip_time.minutes * 60 + subrip_time.seconds + subrip_time.milliseconds / 1000

def split_subtitles_old(subtitle_path, max_duration=180):
    # read the subtitle file and detect the encoding
    with open(subtitle_path, 'rb') as f:
        result = chardet.detect(f.read())
    subtitles = pysrt.open(subtitle_path, encoding=result['encoding'])
    # split the subtitle file into paragraphs 
    current_paragraph = []
    current_duration = 0
    paragraphs = []

    for subtitle in subtitles:
        subtitle_duration = time_to_seconds(subtitle.end - subtitle.start)
        if current_duration + subtitle_duration > max_duration:
            paragraphs.append(' '.join(current_paragraph) + '.')
            current_paragraph = []
            current_duration = 0

        current_paragraph.append(subtitle.text)
        current_duration += subtitle_duration

    if current_paragraph:
        paragraphs.append(' '.join(current_paragraph) + '.')

    return paragraphs

def split_subtitles(subtitle_path, n):
    # read the subtitle file and detect the encoding
    with open(subtitle_path, 'rb') as f:
        result = chardet.detect(f.read())
    subs = pysrt.open(subtitle_path, encoding=result['encoding'])

    total_subs = len(subs)

    if n <= 0 or n > total_subs:
        print("Invalid value for n. It should be a positive integer less than or equal to the total number of subtitles.")
        return None

    subs_per_paragraph = total_subs // n
    remainder = total_subs % n

    paragraphs = []

    current_index = 0

    for i in range(n):
        num_subs_in_paragraph = subs_per_paragraph + (1 if i < remainder else 0)

        paragraph_subs = subs[current_index:current_index + num_subs_in_paragraph]
        current_index += num_subs_in_paragraph

        # Join subtitles using pysrt's built-in method for efficient formatting
        paragraph = pysrt.SubRipFile(items=paragraph_subs).text
        paragraphs.append(paragraph)

    return paragraphs

def clean_text(subtitles_text):
    # Remove unwanted characters except for letters, digits, spaces, periods, commas, exclamation marks, and single quotes
    subtitles_text = re.sub(r'[^a-zA-Z0-9\s\']', '', subtitles_text)
    # Replace multiple spaces with a single space
    subtitles_text = re.sub(r'\s+', ' ', subtitles_text)
    return subtitles_text.strip()
class GoldFish_LV:
    """
    'GoldFish_LV' class is to handle long video processing and subtitle management with MiniGPT-V base model.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.model, self.vis_processor = init_model(args)
        self.original_llama_model,self.original_llama_tokenizer=self.load_original_llama_model()
        # self.summary_instruction="Generate a description of this video .Pay close attention to the objects, actions, emotions portrayed in the video,providing a vivid description of key moments.Specify any visual cues or elements that stand out."
        self.summary_instruction="I'm a blind person, please provide me with a detailed summary of the video content and try to be as descriptive as possible. The videos are created from Nexar's road camera"
    def load_original_llama_model(self):
        model_name="meta-llama/Llama-2-7b-chat-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token = "[PAD]"
        tokenizer.padding_side = "left"
        bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
        llama_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map={'':torch.cuda.current_device()},
                quantization_config=bnb_config,
            )
        return llama_model,tokenizer
    def _load_existing_subtitles(self) -> Dict[str, bool]:

        subtitles_folder = f"workspace/subtitles/{self.video_name}"
        os.makedirs(subtitles_folder, exist_ok=True)
        existing_subtitles = {}
        for subtitle_file in os.listdir(subtitles_folder):
            subtitle_name, _ = os.path.splitext(subtitle_file)
            existing_subtitles[subtitle_name] = True
        return existing_subtitles

    def _youtube_download(self, url: str) -> str:

        try:
            video_id = url.split('v=')[-1].split('&')[0]
            video_id = video_id.strip()
            youtube = YouTube(f"https://www.youtube.com/watch?v={video_id}")
            video_stream = youtube.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
            if not video_stream:
                raise ValueError("No suitable video stream found.")
            output_path = f"workspace/tmp/{video_id}.mp4"
            self.video_id=video_id
            video_stream.download(output_path="workspace/tmp", filename=f"{video_id}.mp4")
            return output_path
        except Exception as e:
            print(f"Error downloading video: {e}")
            return url

    @staticmethod
    def is_youtube_url(url: str) -> bool:

        youtube_regex = (
            r'(https?://)?(www\.)?'
            '(youtube|youtu|youtube-nocookie)\.(com|be)/'
            '(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
        )
        return bool(re.match(youtube_regex, url))


    def find_max_index(self, prefix, existing_data):
        max_index = 0
        for key in existing_data.keys():
            if key.startswith(f"{prefix}_"):
                index = int(key.split("_")[1])
                max_index = max(max_index, index)
        return max_index

    def process_video_url(self, video_path: str) -> str:
        
        if self.is_youtube_url(video_path):
            return self._youtube_download(video_path)
        else:
            return video_path

    def create_video_grid(self, images: list, rows: int, cols: int, save_path: str) -> Image.Image:

        image_width, image_height = images[0].size
        grid_width = cols * image_width
        grid_height = rows * image_height
        new_image = Image.new("RGB", (grid_width, grid_height))

        for i in range(rows):
            for j in range(cols):
                index = i * cols + j
                if index < len(images):
                    image = images[index]
                    x_offset = j * image_width
                    y_offset = i * image_height
                    new_image.paste(image, (x_offset, y_offset))

        new_image.save(save_path)

        return new_image

    def get_subtitles(self, video_path: str) -> Optional[str]:
        video_name=video_path.split('/')[-2]
        video_id=video_path.split('/')[-1].split('.')[0]
        audio_dir = f"workspace/audio/{video_name}"
        subtitle_dir = f"workspace/subtitles/{video_name}"
        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(subtitle_dir, exist_ok=True)
        # if the subtitles are already generated, return the path of the subtitles
        subtitle_path = f"{subtitle_dir}/{video_id}"+'.vtt'
        if os.path.exists(subtitle_path):
            return f"{subtitle_dir}/{video_id}"+'.vtt'
        audio_path = f"{audio_dir}/{video_id}"+'.mp3'
        subtitle_path = f"{subtitle_dir}/{video_id}"+'.vtt'
        # self.existed_subtitles = self._load_existing_subtitles()

        # if self.existed_subtitles.get(video_id, False):
        #     print("Subtitle already generated")
        #     return subtitle_path
        
        try:
            self.extract_audio(video_path, audio_path)
            print("Successfully extracted audio")

            os.system(f"whisper {audio_path}  --language English --model medium --output_format vtt --output_dir {subtitle_dir}")
            os.system(f"rm {audio_path}")
            # os.system(f"rm -r {audio_dir}")
            print("Subtitle successfully generated")
            return subtitle_path
        
        except:
            print(f"Error during subtitle generation for {video_path}")
            return None
            
    def prepare_input(self, 
                    video_path: str,
                    subtitle_path: Optional[str],
                    instruction: str,previous_caption="") -> Tuple[Optional[torch.Tensor], Optional[str]]:
        # If a subtitle path is provided, read the VTT (Web Video Text Tracks) file, else set to an empty list
        conversation=""
        if subtitle_path:
            vtt_file = webvtt.read(subtitle_path)
            print("Subtitle loaded successfully")
            try:
                for subtitle in vtt_file:
                    sub = subtitle.text.replace('\n',' ')
                    conversation+=sub
            except:
                pass

        max_images_length = 45
        max_sub_len = 400

        # Load the video file using moviepy and calculate the total number of frames
        clip = mp.VideoFileClip(video_path)
        total_num_frames = int(clip.duration * clip.fps)
        clip.close()

        # Calculate how often to sample a frame based on the total number of frames and the maximum images length
        cap = cv2.VideoCapture(video_path)
        images = []
        frame_count = 0
        sampling_interval = int(total_num_frames / max_images_length)

        if sampling_interval == 0:
            sampling_interval = 1

        # Initialize variables to hold image placeholders, current subtitle text, and subtitle history
        if previous_caption != "":
            img_placeholder = previous_caption+" "
        else:
            img_placeholder = ""
        subtitle_text_in_interval = ""
        history_subtitles = {}
        raw_frames=[]
        number_of_words=0
        transform=transforms.Compose([
                    transforms.ToPILImage(),
                ])

        
        # Loop through each frame in the video
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # TODO: we need to add subtitles in external memory either
            if subtitle_path is not None: 
                for i, subtitle in enumerate(vtt_file):
                    sub = subtitle.text.replace('\n',' ')
                    if (subtitle.start_in_seconds <= (frame_count / int(clip.fps)) <= subtitle.end_in_seconds) and sub not in subtitle_text_in_interval:

                        if not history_subtitles.get(sub, False):
                            subtitle_text_in_interval += sub + " "

                        history_subtitles[sub] = True
                        break
            # Process and store the frame at specified intervals
            if frame_count % sampling_interval == 0:
                raw_frames.append(Image.fromarray(cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)))
                frame = transform(frame[:,:,::-1]) # convert to RGB
                frame = self.vis_processor(frame)
                images.append(frame)
                img_placeholder += '<Img><ImageHere>'
                if subtitle_path is not None and subtitle_text_in_interval != "" and number_of_words< max_sub_len: 
                    img_placeholder+=f'<Cap>{subtitle_text_in_interval}'
                    number_of_words+=len(subtitle_text_in_interval.split(' '))
                    subtitle_text_in_interval = ""
            frame_count += 1

            # Break the loop if the maximum number of images is reached
            if len(images) >= max_images_length:
                break

        cap.release()
        cv2.destroyAllWindows()

        # Return None if no images are extracted
        if len(images) == 0:
            return None, None
        
        # Save the concatenated image grid to a specified path
        # save_concat_img_path=f"workspace/imgs/{self.video_name}"
        # os.makedirs(save_concat_img_path, exist_ok=True)        
        # self.create_video_grid(raw_frames,8,len(raw_frames)//8, f"{save_concat_img_path}/concatenated.jpg")
        images = torch.stack(images)

        print("Input instruction length",len(instruction.split(' ')))
        instruction = img_placeholder + '\n' + instruction
        print("number of words",number_of_words)
        print("number of images",len(images))

        return images, instruction,conversation

    def extract_audio(self, video_path: str, audio_path: str) -> None:

        video_clip = mp.VideoFileClip(video_path)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(audio_path, codec="libmp3lame", bitrate="320k")
    
    def split_video(self, input_path, output_folder, clip_duration=80):

        if len(os.listdir(output_folder)) > 0:
            return
        video = mp.VideoFileClip(input_path)
        total_duration = video.duration
        num_clips = int(total_duration / clip_duration)

        for i in tqdm (range(num_clips+1), desc="Splitting video"):
            start_time = i * clip_duration
            if i == num_clips:
                end_time=total_duration
            else:
                end_time = (i + 1) * clip_duration
            clip = video.subclip(start_time, end_time)
            save_name=f"{i + 1}".zfill(5)
            output_path = os.path.join(output_folder, f"{save_name}.mp4")
            clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

        video.close()


    def short_video_inference (self,video_path,instruction,gen_subtitles=True):
        if gen_subtitles:
            subtitle_path=self.get_subtitles(video_path)
        else :
            subtitle_path=None
        prepared_images,prepared_instruction,video_conversation=self.prepare_input(video_path,subtitle_path,instruction)
        if prepared_images is None:
            return "Video cann't be open ,check the video path again"
        length=len(prepared_images)
        prepared_images=prepared_images.unsqueeze(0)
        conv = CONV_VISION.copy()
        conv.system = ""
        # if you want to make conversation comment the 2 lines above and make the conv is global variable
        conv.append_message(conv.roles[0], prepared_instruction)
        conv.append_message(conv.roles[1], None)
        prompt = [conv.get_prompt()]
        answers = self.model.generate(prepared_images, prompt, max_new_tokens=512, do_sample=False, lengths=[length],num_beams=1)
        return answers[0]

    def split_long_video_into_clips(self,video_path):
        # Split the video into 90 seconds clips and make a queue of the videos and run the inference on each video 
        self.video_name=video_path.split('/')[-1].split('.')[0]
        tmp_save_path=f"workspace/tmp/{self.video_name}"
        os.makedirs(tmp_save_path, exist_ok=True)
        print("tmp_save_path",tmp_save_path)

        # self.split_video(video_path, tmp_save_path)
        if len(os.listdir(tmp_save_path)) == 0:
            print("Splitting Long video")
            os.system(f"python split_long_video_in_parallel.py --video_path {video_path} --output_folder {tmp_save_path}")
        videos_list = sorted(os.listdir(tmp_save_path))
        return videos_list,tmp_save_path
    def long_inference_video(self, videos_list: List[str], tmp_save_path: str, subtitle_paths: Optional[List[str]] = None, use_subtitles: bool = False) -> Optional[dict]:
        save_long_videos_path = "new_workspace/clips_summary/demo"
        os.makedirs(save_long_videos_path, exist_ok=True)
        file_path = f'{save_long_videos_path}/{self.video_name}.json'

        if os.path.exists(file_path):
            print("Clips inference already done")
            with open(file_path, 'r') as file:
                video_information = json.load(file)
            return video_information

        video_information = {}
        video_captions = []
        batch_video_paths, batch_instructions, batch_subtitles = [], [], []

        for i, video in tqdm(enumerate(videos_list), desc="Inference video clips", total=len(videos_list)):
            clip_path = os.path.join(tmp_save_path, video)
            batch_video_paths.append(clip_path)
            batch_instructions.append(self.summary_instruction)
            if use_subtitles and subtitle_paths:
                batch_subtitles.append(subtitle_paths[i])
            else:
                batch_subtitles.append(None)  # Handling for no subtitle case

            if len(batch_video_paths) % 2 == 0:  # Assuming batch_size is 2
                batch_preds, videos_conversation = self.run_batch(batch_video_paths, batch_instructions, batch_subtitles)
                for pred, subtitle in zip(batch_preds, videos_conversation):
                    video_number = i + 1  # Better scope handling for video number
                    save_name = f"{video_number}".zfill(5)
                    video_information[f'caption__{save_name}'] = pred
                    video_information[f'subtitle__{save_name}'] = subtitle
                    video_captions.append(pred)
                batch_video_paths, batch_instructions, batch_subtitles = [], [], []

        # Process any remaining videos in the last batch
        if batch_video_paths:
            batch_preds, videos_conversation = self.run_batch(batch_video_paths, batch_instructions, batch_subtitles)
            for pred, subtitle in zip(batch_preds, videos_conversation):
                video_number = len(videos_list) - len(batch_video_paths) + len(batch_preds)
                save_name = f"{video_number}".zfill(5)
                video_information[f'caption__{save_name}'] = pred
                video_information[f'subtitle__{save_name}'] = subtitle
                video_captions.append(pred)

        video_information['summary'] = "summary"  # Example summary
        with open(file_path, 'w') as file:
            json.dump(video_information, file, indent=4)
        print("Clips inference done")

        return video_information

    def compine_summaries(self, text: str, rag: str = False) -> str:
        concate_instruction = (
            "<s>[INST]Task: Generate video summary by "
            "merging the following list of video clip summaries and the subtitles Ensure the "
            "final description is detailed, intelligible, and captures the essence "
            "of the entire video content and DON'T MENTION THE NUMBER OF CLIPS in the output\n\n"
            f"List of Video Clip Summaries and subtitles:\n{str(text)}\n\n"
            "Comprehensive Video Description:[/INST]"

        )
        inputs = self.original_llama_tokenizer(concate_instruction, return_tensors="pt", padding=True, truncation=True,max_length=3500)

        with torch.no_grad():
            summary_ids = self.original_llama_model.generate(inputs.input_ids,max_new_tokens=512)
        answers=[]
        for i in range(len(summary_ids)):
            output_text=self.original_llama_tokenizer.decode(summary_ids[i], skip_special_tokens=True)
            output_text = output_text.split('</s>')[0]  # remove the stop sign </s>
            output_text = output_text.replace("<s>", "")
            output_text = output_text.split(r'[/INST]')[-1].strip()
            answers.append(output_text)
        
        return answers[0]
    def compine_summaries_gpt4(self, text_str: str, rag: str = False) -> str:
        concate_instruction = (
            "Task: Generate video summary by "
            "merging the following list of video clip summaries and the subtitles Ensure the "
            "final description is detailed, intelligible, and captures the essence "
            "of the entire video content and DON'T MENTION THE NUMBER OF CLIPS in the output\n\n"
            f"List of Video Clip Summaries and subtitles:\n{text_str}\n\n"
            "Comprehensive Video Description:"
        )
        try:
            response = client.ChatCompletion.create(
                model="gpt-4",
                messages=[
                        {
                        "role": "user",
                        "content": concate_instruction
                        }],
            )
            answer=response.choices[0].message['content']
        except Exception as e:
            print("chat gpt error",e)
        return answer
    def language_infernece(self, texts):
        prepared_images=[]
        prompts=[]
        lengths=[]
        for text in texts:
            placeholder_image = torch.zeros(3, 224, 224).unsqueeze(0)
            prepared_images.append(placeholder_image)
            prepared_instruction = '<Img><ImageHere> ' + text

            # Initialize conversation for the model
            model_conversation = CONV_VISION.copy()
            model_conversation.system = ""
            model_conversation.append_message(model_conversation.roles[0], prepared_instruction)
            model_conversation.append_message(model_conversation.roles[1], None)

            prompts.append(model_conversation.get_prompt())
            lengths.append(1)
        try:
            prepared_images=torch.stack(prepared_images)
            answers = self.model.generate(prepared_images, prompts, max_new_tokens=self.args.max_new_tokens, do_sample=False, lengths=lengths, num_beams=1)
            return answers
        except Exception as e:
            # Handle exceptions and errors
            print(f"Error during text inference: {e}")
            return [f"Error during text inference: {e}"]*len(texts) 
    
    # def inference_RAG(self, instructions: str, context_list) -> str:
    #     prepared_images=[]
    #     prompts=[]
    #     lengths=[]
    #     for instruction,context in zip(instructions,context_list):
    #         placeholder_image = torch.zeros(3, 224, 224).unsqueeze(0)
    #         prepared_images.append(placeholder_image)
    #         inference_format="Your task is to answer questions for long video \n\n Given these related information from the most related clips: \n "+context +"\n\n" +"Answer this question: "+instruction
    #         prepared_instruction = '<Img><ImageHere> ' + inference_format 

    #         # Initialize conversation for the model
    #         model_conversation = CONV_VISION.copy()
    #         model_conversation.system = ""
    #         model_conversation.append_message(model_conversation.roles[0], prepared_instruction)
    #         model_conversation.append_message(model_conversation.roles[1], None)

    #         prompts.append(model_conversation.get_prompt())
    #         lengths.append(1)

    #     try:
    #         prepared_images=torch.stack(prepared_images)
    #         answers = self.model.generate(prepared_images, prompts, max_new_tokens=self.args.max_new_tokens, do_sample=False, lengths=lengths, num_beams=1)
    #         return answers
    #     except Exception as e:
    #         # Handle exceptions and errors
    #         print(f"Error during text inference: {e}")
    #         return [f"Error during text inference: {e}"]*len(instructions)
   

    

    def inference_RAG(self, instructions, context_list):
        context_promots=[]
        questions_prompts=[]
        try:
            for instruction,context in zip(instructions,context_list):
                context=clean_text(context)
                context_prompt=f"<s>[INST] Your task is to answer questions for one long video which is split into multiple clips.\nGiven these related information from the most related clips: \n{context}\n"
                question_prompt=f"\nAnswer this question :{instruction} \n your answer is: [/INST]"
                context_promots.append(context_prompt)
                questions_prompts.append(question_prompt)
            context_inputs = self.original_llama_tokenizer(context_promots, return_tensors="pt", padding=True, truncation=True,max_length=3500)
            # print(context_inputs.keys())
            print("context_inputs shape",context_inputs['input_ids'].shape)
            question_inputs = self.original_llama_tokenizer(questions_prompts, return_tensors="pt", padding=True, truncation=True,max_length=300)
            print("question_inputs shape",question_inputs['input_ids'].shape)
            # concate the context and the question together 
            inputs_ids=torch.cat((context_inputs['input_ids'],question_inputs['input_ids']),dim=1).to('cuda')
            print("inputs shape",inputs_ids.shape)
        except Exception as e:
            print("error while tokenization",e)
            return self.inference_RAG_batch_size_1(instructions, context_list)
        with torch.no_grad():
            summary_ids = self.original_llama_model.generate(inputs_ids,max_new_tokens=512)
        answers=[]
        for i in range(len(summary_ids)):
            output_text=self.original_llama_tokenizer.decode(summary_ids[i], skip_special_tokens=True)
            output_text = output_text.split('</s>')[0]  # remove the stop sign </s>
            output_text = output_text.replace("<s>", "")
            output_text = output_text.split(r'[/INST]')[-1].strip()
            answers.append(output_text)
        
        return answers
       
        
    def inference_RAG_batch_size_1(self, instructions, context_list):
        answers=[]
        for instruction,context in zip(instructions,context_list):
            context=clean_text(context)
            context_prompt=f"<s>[INST] Your task is to answer questions for one long video which is split into multiple clips.\nGiven these related information from the most related clips: \n{context}\n"
            question_prompt=f"\nAnswer this question :{instruction} \n your answer is: [/INST]"
            context_inputs=self.original_llama_tokenizer([context_prompt], return_tensors="pt", padding=True, truncation=True,max_length=3500)['input_ids']
            question_inputs=self.original_llama_tokenizer([question_prompt], return_tensors="pt", padding=True, truncation=True,max_length=300)['input_ids']
            
            inputs_ids=torch.cat((context_inputs,question_inputs),dim=1).to('cuda')
            with torch.no_grad():
                summary_ids = self.original_llama_model.generate(inputs_ids,max_new_tokens=512,)
            
            output_text=self.original_llama_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            output_text = output_text.split('</s>')[0]  # remove the stop sign </s>
            output_text = output_text.replace("<s>", "")
            output_text = output_text.split(r'[/INST]')[-1].strip()
            answers.append(output_text)
            
        return answers
        
    # def inference_RAG_text_only(self, instructions, context_list):
    #     # Use VideoLLM as the answer module
    #     seg_tokens=[]
    #     for instruction,context in zip(instructions,context_list):
    #         context=clean_text(context)
    #         context_prompt=f"<s>[INST] Your task is to answer questions for one long video which is split into multiple clips.\nGiven these related information from the most related clips: \n{context}\n"
    #         question_prompt=f"\nAnswer this question :{instruction} \n your answer is: [/INST]"
    #         context_inputs = self.model.llama_tokenizer(context_prompt,add_special_tokens=True, return_tensors="pt", padding=True, truncation=True,max_length=3500)
    #         question_inputs = self.model.llama_tokenizer(question_prompt, return_tensors="pt", padding=True, truncation=True,max_length=300)
    #         # concate the context and the question together 
    #         inputs_ids=torch.cat((context_inputs['input_ids'],question_inputs['input_ids']),dim=1).to('cuda')
    #         seg_tokens.append(inputs_ids)
    #     with torch.no_grad():
    #         answers = self.model.generate_text_only(images=None,seg_tokens=seg_tokens,max_new_tokens=512)        
    #     return answers
        
             
    def inference_RAG_chatGPT(self, instructions: str, context_list) -> str:
        batch_preds=[]
        for context,instruction in zip(context_list,instructions):
            prompt="Your task is to answer questions for long video \n\n Given these related information from the most related clips: \n "+context +"\n\n" +"Answer this question: "+instruction
            while True:
                try:
                    response = client.ChatCompletion.create(
                        model="gpt-4o",
                        messages=[
                                {
                                "role": "user",
                                "content": prompt
                                }],
                    )
                    answer=response.choices[0].message['content']
                    batch_preds.append(answer)
                    break
                except Exception as e:
                    print("chat gpt error",e)
                    time.sleep(50)
        
        return batch_preds
     
    def get_most_related_clips(self,related_context_keys):
        most_related_clips=[]
        for context_key in related_context_keys:
            if len(context_key.split('__'))>1:
                most_related_clips.append(context_key.split('__')[1])
            if len(most_related_clips)==self.args.neighbours:
                break
        assert len(most_related_clips)!=0, f"No related clips found {related_context_keys}"
        return most_related_clips   
    def get_related_context(self, external_memory,related_context_keys):
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
        return related_information              
    def inference(self, video_path, use_subtitles=True, instruction=""):
        start_time = time.time()
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"Video name: {video_name}")
        print("Video path",video_path)
        video_duration = mp.VideoFileClip(video_path).duration
        print(f"Video duration: {video_duration:.2f} seconds")
        # if the video duration is more than 2 minutes we need to run the long inference 
        if video_duration > 180 :
            print("Long video")
            # if the video data is already stored in the external memory, we can use it directly else we need to run the long inference
            file_path=f'new_workspace/clips_summary/demo/{video_name}.json'
            if not os.path.exists(file_path):
                print("Clips summary is not ready")
                videos_list,tmp_save_path=self.split_long_video_into_clips(video_path)
                clips_summary = self.long_inference_video(videos_list,tmp_save_path, use_subtitles=use_subtitles)
            else: 
                print("External memory is ready")
            os.makedirs("new_workspace/embedding/demo", exist_ok=True)
            os.makedirs("new_workspace/open_ai_embedding/demo", exist_ok=True)
            if self.args.use_openai_embedding:
                embedding_path=f"new_workspace/open_ai_embedding/demo/{video_name}.pkl"
            else:
                embedding_path=f"new_workspace/embedding/demo/{video_name}.pkl"
            external_memory=MemoryIndex(self.args.neighbours,use_openai=self.args.use_openai_embedding)
            if os.path.exists(embedding_path):
                print("Loading embeddings from pkl file")
                external_memory.load_embeddings_from_pkl(embedding_path)
            else:
                # will embed the information and save it in the pkl file
                external_memory.load_documents_from_json(file_path,embedding_path)
            # get the most similar context from the external memory to this instruction 
        
            related_context_documents,related_context_keys = external_memory.search_by_similarity(instruction)
            related_information=self.get_related_context(external_memory,related_context_keys)
            pred=self.inference_RAG([instruction], [related_information])[0] 
            # remove stored data 
            # os.remove(file_path)
            # os.system(f"rm -r workspace/tmp/{self.video_name}")
            # os.system(f"rm -r workspace/subtitles/{self.video_name}")
            # os.system(f"rm workspace/tmp/{self.video_id}.mp4") 
        else:
            print("Short video")
            self.video_name=video_path.split('/')[-1].split('.')[0]
            clips_summary={}
            clips_summary['summary'] = self.short_video_inference(video_path, self.summary_instruction,use_subtitles)
            pred=self.short_video_inference(video_path,instruction,use_subtitles)
        processing_time = time.time() - start_time
        print(f"Processing time: {processing_time:.2f} seconds")
        return {
            'video_name': os.path.splitext(os.path.basename(video_path))[0],
            'pred': pred,
            'clips_summary': "clips_summary['summary']",
        }

    
    def run_batch(self, video_paths, instructions,subtitle_paths,previous_caption="") -> List[str]:

        prepared_images_batch = []
        prepared_instructions_batch = []
        lengths_batch = []
        videos_conversations=[]

        for i,video_path, instruction in zip(range(len(video_paths)),video_paths, instructions):

            subtitle_path = subtitle_paths[i]
            prepared_images, prepared_instruction,video_conversation = self.prepare_input( video_path, subtitle_path, instruction,previous_caption)
            
            if prepared_images is None:
                print(f"Error: Unable to open video at {video_path}. Check the path and try again.")
                continue
            videos_conversations.append(video_conversation)
            conversation = CONV_VISION.copy()
            conversation.system = ""
            conversation.append_message(conversation.roles[0], prepared_instruction)
            conversation.append_message(conversation.roles[1], None)
            prepared_instructions_batch.append(conversation.get_prompt())
            prepared_images_batch.append(prepared_images)
            lengths_batch.append(len(prepared_images))

        if not prepared_images_batch:
            return []

        prepared_images_batch = torch.stack(prepared_images_batch)
        answers=self.model.generate(prepared_images_batch, prepared_instructions_batch, max_new_tokens=self.args.max_new_tokens, do_sample=False, lengths=lengths_batch, num_beams=1)
        return answers , videos_conversations

    def run_images_features (self,img_embeds,prepared_instruction,return_embedding=False):        
        lengths=[]
        prompts=[]
        for i in range(img_embeds.shape[0]):    
            conv = CONV_VISION.copy()
            conv.system = ""
            conv.append_message(conv.roles[0], prepared_instruction[i])
            conv.append_message(conv.roles[1], None)
            prompts.append(conv.get_prompt())
            lengths.append(len(img_embeds[i]))
        if return_embedding: 
            answers,videos_embedding = self.model.generate(images=None,img_embeds=img_embeds,texts=prompts, max_new_tokens=300, do_sample=False, lengths=lengths,num_beams=1, 
                                                       return_video_temporal_features=return_embedding)
            return answers ,videos_embedding
        else:
            answers = self.model.generate(images=None,img_embeds=img_embeds,texts=prompts, max_new_tokens=300, do_sample=False, lengths=lengths,num_beams=1, 
                                                       return_video_temporal_features=return_embedding)
            # for ans, p in zip(answers, prompts):
            #     print(f"Prompt: {p}")
            #     print(f"Answer: {ans}")
            return answers
        
    def run_images (self,prepared_images,prepared_instruction,return_embedding=False):        
        lengths=[]
        prompts=[]
        for i in range(prepared_images.shape[0]):    
            conv = CONV_VISION.copy()
            conv.system = ""
            conv.append_message(conv.roles[0], prepared_instruction[i])
            conv.append_message(conv.roles[1], None)
            prompts.append(conv.get_prompt())
            lengths.append(len(prepared_images[i]))
        if return_embedding: 
            answers,videos_embedding = self.model.generate(prepared_images, prompts, max_new_tokens=300, do_sample=False, lengths=lengths,num_beams=1, 
                                                       return_video_temporal_features=return_embedding)
            return answers ,videos_embedding
        else:
            answers = self.model.generate(prepared_images, prompts, max_new_tokens=300, do_sample=False, lengths=lengths,num_beams=1, 
                                                       return_video_temporal_features=return_embedding)
            return answers



# def get_arguments():

#     parser = argparse.ArgumentParser(description="Inference parameters")
#     parser.add_argument("--cfg-path", default="test_configs/llama2_test_config.yaml")
#     parser.add_argument("--ckpt", type=str, default="checkpoints/video_llama_checkpoint_last.pth")
#     parser.add_argument("--neighbours", type=int, default=3)
#     parser.add_argument("--start", default=0, type=int)
#     parser.add_argument("--end", default=100000, type=int)
#     parser.add_argument("--skill_path",default="/ibex/project/c2106/kirolos/Long_video_Bench/benchmark/final/concatenated/summarization.json")
#     parser.add_argument("--add_subtitles", action='store_true')
#     parser.add_argument("--use_openai_embedding", action="store_true")
#     parser.add_argument("--eval_opt", type=str, default='all')
#     parser.add_argument("--max_new_tokens", type=int, default=512)
#     parser.add_argument("--batch_size", type=int, default=4)
#     parser.add_argument("--lora_r", type=int, default=64)
#     parser.add_argument("--lora_alpha", type=int, default=16)
#     parser.add_argument("--video_path", type=str, help="Path to the video file")
#     parser.add_argument("--options", nargs="+")
#     return parser.parse_args()
# args = get_arguments()  
# if __name__ == "__main__":
#     minigpt_lv = GoldFish_LV(args)
#     # processed_video_path = minigpt_lv.process_video_url(args.video_path)
#     # minigpt_lv.inference(video_path=processed_video_path)
#     with open ('/home/ataallka/minigpt_video/minigpt_multi_img/workspace/eval/llama_vid_qa/ckpt_92/output.txt','r') as f:
#         context=f.read()
#     # print(clean_text(context))
#     print("start inference")
#     t1=time.time()
#     bs=1
#     contexts=[]
#     questions=[]
#     for i in range(1):
#         contexts.append(context)
#         questions.append("Summarize this movie")
#         if len(contexts)==bs:
#             print(minigpt_lv.inference_RAG(questions,contexts))
#             contexts=[]
#             questions=[]
    
#     if len(contexts)>0:
#         print(minigpt_lv.inference_RAG(questions,contexts))
    
#     print("time for 18 clip",time.time()-t1)