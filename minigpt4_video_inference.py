import torch
import webvtt
import os
import cv2 
from minigpt4.common.eval_utils import prepare_texts, init_model
from minigpt4.conversation.conversation import CONV_VISION
from torchvision import transforms
import json
from tqdm import tqdm
import soundfile as sf
import argparse
import moviepy.editor as mp
import gradio as gr
from pytubefix import YouTube
import shutil
from PIL import Image
from moviepy.editor import VideoFileClip
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn 
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from threading import Thread
import cv2
import webvtt
from bisect import bisect_left
import whisper
import time
from datetime import timedelta
# Function to format timestamps for VTT
def format_timestamp(seconds):
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    milliseconds = int(td.microseconds / 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"
def extract_video_info(video_path,max_images_length):
    clip = VideoFileClip(video_path)
    total_num_frames = int(clip.duration * clip.fps)
    clip.close()
    sampling_interval = int(total_num_frames / max_images_length)
    if sampling_interval == 0:
        sampling_interval = 1
    return sampling_interval,clip.fps
def time_to_milliseconds(time_str):
    # Convert time format "hh:mm:ss.sss" to milliseconds
    h, m, s = map(float, time_str.split(':'))
    return int((h * 3600 + m * 60 + s) * 1000)
def extract_subtitles(subtitle_path):
    # Parse the VTT file into a list of (start_time_ms, end_time_ms, text)
    subtitles = []
    for caption in webvtt.read(subtitle_path):
        start_ms = time_to_milliseconds(caption.start)
        end_ms = time_to_milliseconds(caption.end)
        text = caption.text.strip().replace('\n', ' ')
        subtitles.append((start_ms, end_ms, text))
    return subtitles
def find_subtitle(subtitles, frame_count, fps):
    frame_time = (frame_count / fps)*1000

    left, right = 0, len(subtitles) - 1
    
    while left <= right:
        mid = (left + right) // 2
        start, end, subtitle_text = subtitles[mid]
        # print("Mid start end sub ",mid,start,end,subtitle_text)
        if start <= frame_time <= end:
            return subtitle_text
        elif frame_time < start:
            right = mid - 1
        else:
            left = mid + 1
    
    return None  # If no subtitle is found
def match_frames_and_subtitles(video_path,subtitles,sampling_interval,max_sub_len,fps,max_frames):  
    cap = cv2.VideoCapture(video_path)
    images = []
    frame_count = 0
    img_placeholder = ""
    subtitle_text_in_interval = ""
    history_subtitles = {}
    number_of_words=0
    transform=transforms.Compose([
                transforms.ToPILImage(),
            ])
    # first_frame=cap.read()[1]
    # video_out=cv2.VideoWriter("old_prepare_input.mp4",cv2.VideoWriter_fourcc(*'mp4v'), 1, (first_frame.shape[1],first_frame.shape[0]))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if len (subtitles) > 0: 
            # use binary search to find the subtitle for the current frame which the frame time is between the start and end time of the subtitle 
            frame_subtitle=find_subtitle(subtitles, frame_count, fps)
            if frame_subtitle and not history_subtitles.get(frame_subtitle,False):
                subtitle_text_in_interval+=frame_subtitle+" "
                history_subtitles[frame_subtitle]=True
        if frame_count % sampling_interval == 0:
            # raw_frame=frame.copy()
            frame = transform(frame[:,:,::-1]) # convert to RGB
            frame = vis_processor(frame)
            images.append(frame)
            img_placeholder += '<Img><ImageHere>'
            if subtitle_text_in_interval != "" and number_of_words< max_sub_len: 
                img_placeholder+=f'<Cap>{subtitle_text_in_interval}'
                # write the subtitle on the frame 
                # cv2.putText(raw_frame,subtitle_text_in_interval,(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
                number_of_words+=len(subtitle_text_in_interval.split(' '))
                subtitle_text_in_interval = ""
            # video_out.write(raw_frame)
        frame_count += 1
        if len(images) >= max_frames:
            break
    cap.release()
    cv2.destroyAllWindows()
    # video_out.release()
    if len(images) == 0:
        # skip the video if no frame is extracted
        return None,None
    images = torch.stack(images)
    return images,img_placeholder

def prepare_input(video_path, subtitle_path,instruction):
    if "mistral" in args.ckpt :
        max_frames=90
        max_sub_len = 800
    else:
        max_frames = 45
        max_sub_len = 400
    sampling_interval,fps = extract_video_info(video_path, max_frames)
    subtitles = extract_subtitles(subtitle_path)
    frames_features,input_placeholder = match_frames_and_subtitles(video_path,subtitles,sampling_interval,max_sub_len,fps,max_frames)
    input_placeholder+="\n"+instruction
    return frames_features, input_placeholder


def extract_audio(video_path, audio_path):
    video_clip = mp.VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(audio_path, codec="libmp3lame", bitrate="320k")

def get_subtitles(video_path) :
    audio_dir="workspace/inference_subtitles/mp3"
    subtitle_dir="workspace/inference_subtitles"
    os.makedirs(subtitle_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)
    video_id=video_path.split('/')[-1].split('.')[0]
    audio_path = f"workspace/inference_subtitles/mp3/{video_id}"+'.mp3'
    subtitle_path = f"{subtitle_dir}/{video_id}"+'.vtt'
    # if the subtitles are already generated, return the path of the subtitles
    if os.path.exists(subtitle_path):
        return f"{subtitle_dir}/{video_id}"+'.vtt'
    audio_path = f"{audio_dir}/{video_id}"+'.mp3'
    try:
        extract_audio(video_path, audio_path)
        result = whisper_model.transcribe(audio_path,language="en") 
        # Create VTT file
        with open(subtitle_path, "w", encoding="utf-8") as vtt_file:
            vtt_file.write("WEBVTT\n\n")
            for segment in result['segments']:
                start = format_timestamp(segment['start'])
                end = format_timestamp(segment['end'])
                text = segment['text']
                vtt_file.write(f"{start} --> {end}\n{text}\n\n")
        return subtitle_path
    except Exception as e:
        print(f"Error during subtitle generation for {video_path}: {e}")
        return None
        
     
def stream_answer(generation_kwargs):
    streamer = TextIteratorStreamer(model.llama_tokenizer, skip_special_tokens=True)
    generation_kwargs['streamer'] = streamer
    thread = Thread(target=model_generate, kwargs=generation_kwargs)
    thread.start()
    return streamer
def escape_markdown(text):
    # List of Markdown special characters that need to be escaped
    md_chars = ['<', '>']
    # Escape each special character
    for char in md_chars:
        text = text.replace(char, '\\' + char)
    return text
def model_generate(*args, **kwargs):
    # for 8 bit and 16 bit compatibility
    with model.maybe_autocast():
        output = model.llama_model.generate(*args, **kwargs)
    return output

def generate_prediction (video_path,instruction,gen_subtitles=True,stream=True):
    if gen_subtitles:
        subtitle_path=get_subtitles(video_path)
    else :
        subtitle_path=None
    prepared_images,prepared_instruction=prepare_input(video_path,subtitle_path,instruction)
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
    # print("prompt",prompt)
    if stream:
        generation_kwargs = model.answer_prepare_for_streaming(prepared_images, prompt, max_new_tokens=args.max_new_tokens, do_sample=True, lengths=[length],num_beams=1)
        streamer=stream_answer(generation_kwargs)
        print("Streamed answer:",end='')
        for a in streamer:
            print(a,end='')
        print()
    else:
        setup_seeds(50)
        answers = model.generate(prepared_images, prompt, max_new_tokens=args.max_new_tokens, do_sample=True, lengths=[length],num_beams=1)
        print("Generated_answer :",answers[0])
  
def get_arguments():
    parser = argparse.ArgumentParser(description="Inference parameters")
    parser.add_argument("--cfg-path", help="path to configuration file.",default="test_configs/llama2_test_config.yaml")
    parser.add_argument("--ckpt", type=str,default='checkpoints/video_llama_checkpoint_last.pth', help="path to checkpoint")
    parser.add_argument("--add_subtitles",action= 'store_true',help="whether to add subtitles")
    parser.add_argument("--stream",action= 'store_true',help="whether to stream the answer")
    parser.add_argument("--question", type=str, help="question to ask")
    parser.add_argument("--video_path", type=str, help="Path to the video file")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="max number of generated tokens")
    parser.add_argument("--lora_r", type=int, default=64, help="lora rank of the model")
    parser.add_argument("--lora_alpha", type=int, default=16, help="lora alpha")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
                "in xxx=yyy format will be merged into config file (deprecate), "
                "change to --cfg-options instead.",
    )
    return parser.parse_args()
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
model, vis_processor,whisper_gpu_id,minigpt4_gpu_id,answer_module_gpu_id = init_model(args)
whisper_model=whisper.load_model("large").to(f"cuda:{whisper_gpu_id}")
conv = CONV_VISION.copy()
conv.system = ""
if __name__ == "__main__":
    video_path=args.video_path
    instruction=args.question
    add_subtitles=args.add_subtitles
    stream=args.stream
    setup_seeds(50)
    t1=time.time()
    generate_prediction(video_path,instruction,gen_subtitles=add_subtitles,stream=stream)
    print("Time taken for inference",time.time()-t1)