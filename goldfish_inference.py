#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import gradio as gr
from goldfish_lv import GoldFish_LV 
from theme import minigptlv_style
import time
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
    parser.add_argument("--cfg-path", default="test_configs/llama2_test_config.yaml")
    parser.add_argument("--neighbours", type=int, default=3)
    parser.add_argument("--ckpt", type=str, default="checkpoints/video_llama_checkpoint_last.pth")
    parser.add_argument("--add_subtitles", action='store_true')
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--use_openai_embedding",type=str2bool, default=False)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--video_path", type=str,default="path for video.mp4", help="Path to the video file or youtube url")
    parser.add_argument("--question", type=str, default="Why rachel is wearing a wedding dress?")
    parser.add_argument("--options", nargs="+")
    return parser.parse_args()

def download_video(youtube_url):
    processed_video_path = goldfish_lv.process_video_url(youtube_url)
    return processed_video_path

def process_video(video_path, has_subtitles, instruction="", number_of_neighbours=3):
    result = goldfish_lv.inference(video_path, has_subtitles, instruction,number_of_neighbours)
    pred = result["pred"]
    return pred

def return_video_path(youtube_url):
    video_id = youtube_url.split("https://www.youtube.com/watch?v=")[-1].split('&')[0]
    if video_id:
        return os.path.join("workspace", "tmp", f"{video_id}.mp4")
    else:
        raise ValueError("Invalid YouTube URL provided.")

args=get_arguments()
if __name__ == "__main__":
    t1=time.time()
    print("using openai: ", args.use_openai_embedding)
    goldfish_lv = GoldFish_LV(args)
    t2=time.time()
    print("Time taken to load model: ", t2-t1)
    processed_video_path = goldfish_lv.process_video_url(args.video_path)
    pred=process_video(processed_video_path, args.add_subtitles, args.question)      
    print("Question answer: ", pred)
    print(f"Time taken for inference: ", time.time()-t2)