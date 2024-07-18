import os
import moviepy.editor as mp
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import time
def split_clip(args):
    i, clip_duration, total_duration, input_path, output_folder = args
    start_time = i * clip_duration
    end_time = min((i + 1) * clip_duration, total_duration)
    clip = mp.VideoFileClip(input_path).subclip(start_time, end_time)
    save_name = f"{i + 1}".zfill(5)
    output_path = os.path.join(output_folder, f"{save_name}.mp4")
    clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
    clip.close()

def split_video(input_path, output_folder, clip_duration=80):
    os.makedirs(output_folder, exist_ok=True)
    if len(os.listdir(output_folder)) > 0:
        return

    video = mp.VideoFileClip(input_path)
    total_duration = video.duration
    num_clips = int(total_duration / clip_duration)
    if total_duration % clip_duration != 0:
        num_clips += 1

    args_list = [(i, clip_duration, total_duration, input_path, output_folder) for i in range(num_clips)]

    with Pool(processes=cpu_count()) as pool:
        list(tqdm(pool.imap(split_clip, args_list), total=num_clips, desc="Splitting video"))

    video.close()

def split_video_seq(input_path, output_folder, clip_duration=80):
    os.makedirs(output_folder, exist_ok=True)
    if len(os.listdir(output_folder)) > 0:
        return
    video = mp.VideoFileClip(input_path)
    total_duration = video.duration
    num_clips = int(total_duration / clip_duration)
    if total_duration % clip_duration != 0:
        num_clips += 1

    for i in tqdm (range(num_clips), desc="Splitting video"):
        start_time = i * clip_duration
        end_time = min((i + 1) * clip_duration, total_duration)
        clip = video.subclip(start_time, end_time)
        save_name=f"{i + 1}".zfill(5)
        output_path = os.path.join(output_folder, f"{save_name}.mp4")
        clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

    video.close()

import argparse
parser = argparse.ArgumentParser(description="Split video")
parser.add_argument("--video_path", type=str,default="/ibex/project/c2133/minigpt4_v2_dataset/Friends/S01E01.mp4", help="Path to the video file or youtube url")
parser.add_argument("--output_folder", type=str,default="workspace/tmp/clips", help="Path to the output folder")
args = parser.parse_args()
t1 = time.time()
split_video(args.video_path, args.output_folder)
print("Time taken to split video from test parallel: ", time.time()-t1)