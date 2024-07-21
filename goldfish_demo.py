#!/usr/bin/env python
# -*- coding: utf-8 -*-
import spaces
import os
import argparse
import gradio as gr
from goldfish_lv import GoldFish_LV 
from theme import minigptlv_style, custom_css,text_css
import re
from huggingface_hub import login, hf_hub_download
import time
import moviepy.editor as mp
from index import MemoryIndex  


# hf_token = os.environ.get('HF_TKN')
# login(token=hf_token)
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
    parser.add_argument("--name", type=str, default='test')
    parser.add_argument("--ckpt", type=str, default="checkpoints/video_llama_checkpoint_last.pth")
    parser.add_argument("--add_subtitles", action='store_true')
    parser.add_argument("--neighbours", type=int, default=3)
    parser.add_argument("--eval_opt", type=str, default='all')
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--use_openai_embedding",type=str2bool, default=False)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--video_path", type=str, help="Path to the video file")
    parser.add_argument("--options", nargs="+")
    return parser.parse_args()

def download_video(youtube_url, download_finish):
    if is_youtube_url(youtube_url):
        processed_video_path = goldfish_obj.process_video_url(youtube_url)
        download_finish = gr.State(value=True)
        return processed_video_path, download_finish
    else:
        return None, download_finish
def is_youtube_url(url: str) -> bool:
    youtube_regex = (
        r'(https?://)?(www\.)?'
        '(youtube|youtu|youtube-nocookie)\.(com|be)/'
        '(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
    )
    return bool(re.match(youtube_regex, url))

@spaces.GPU(duration=60*5)
def gradio_long_inference_video(videos_list,tmp_save_path,subtitle_paths, use_subtitles=True):
    clips_summary = goldfish_obj.long_inference_video(videos_list,tmp_save_path,subtitle_paths)
    return clips_summary

@spaces.GPU(duration=60*3)
def gradio_short_inference_video(video_path, instruction, use_subtitles=True):
    pred = goldfish_obj.short_video_inference(video_path, instruction, use_subtitles)
    return pred

@spaces.GPU(duration=60*3)
def gradio_inference_RAG (instruction,related_information):
    pred=goldfish_obj.inference_RAG([instruction], [related_information])[0] 
    return pred
def inference(video_path, use_subtitles=True, instruction="", number_of_neighbours=3):
    start_time = time.time()
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    goldfish_obj.args.neighbours = number_of_neighbours
    print(f"Video name: {video_name}")
    video_duration = mp.VideoFileClip(video_path).duration
    print(f"Video duration: {video_duration:.2f} seconds")
    # if the video duration is more than 2 minutes we need to run the long inference 
    if video_duration > 180 :
        print("Long video")
        # if the video data is already stored in the external memory, we can use it directly else we need to run the long inference
        file_path=f'new_workspace/clips_summary/demo/{video_name}.json'
        if not os.path.exists(file_path):
            print("Clips summary is not ready")
            videos_list,tmp_save_path=goldfish_obj.split_long_video_into_clips(video_path)
            subtitle_paths = []
            for video_p in videos_list:
                clip_path = os.path.join(tmp_save_path, video_p)
                subtitle_path = goldfish_obj.get_subtitles(clip_path) if use_subtitles else None
                subtitle_paths.append(subtitle_path)
            gradio_long_inference_video(videos_list,tmp_save_path,subtitle_paths, use_subtitles=use_subtitles)
        else: 
            print("External memory is ready")
        os.makedirs("new_workspace/embedding/demo", exist_ok=True)
        os.makedirs("new_workspace/open_ai_embedding/demo", exist_ok=True)
        if goldfish_obj.args.use_openai_embedding:
            embedding_path=f"new_workspace/open_ai_embedding/demo/{video_name}.pkl"
        else:
            embedding_path=f"new_workspace/embedding/demo/{video_name}.pkl"
        external_memory=MemoryIndex(goldfish_obj.args.neighbours,use_openai=goldfish_obj.args.use_openai_embedding)
        if os.path.exists(embedding_path):
            print("Loading embeddings from pkl file")
            external_memory.load_embeddings_from_pkl(embedding_path)
        else:
            # will embed the information and save it in the pkl file
            external_memory.load_documents_from_json(file_path,embedding_path)
        # get the most similar context from the external memory to this instruction 
    
        related_context_documents,related_context_keys = external_memory.search_by_similarity(instruction)
        related_information=goldfish_obj.get_related_context(external_memory,related_context_keys)
        pred=gradio_inference_RAG(instruction,related_information)
        # remove stored data 
        # os.remove(file_path)
        # os.system(f"rm -r workspace/tmp/{self.video_name}")
        # os.system(f"rm -r workspace/subtitles/{self.video_name}")
        # os.system(f"rm workspace/tmp/{self.video_id}.mp4") 
    else:
        print("Short video")
        goldfish_obj.video_name=video_path.split('/')[-1].split('.')[0]
        pred=gradio_short_inference_video(video_path,instruction,use_subtitles)
    processing_time = time.time() - start_time
    print(f"Processing time: {processing_time:.2f} seconds")
    return {
        'video_name': os.path.splitext(os.path.basename(video_path))[0],
        'pred': pred,
    }


def process_video(path_url, has_subtitles, instruction, number_of_neighbours):
    if is_youtube_url(path_url):
        video_path = return_video_path(path_url)
    else:
        video_path = path_url
    result = inference(video_path, has_subtitles, instruction, number_of_neighbours)    
    pred = result["pred"]
    video_name = result["video_name"]
    # pred="mmmmm"
    return pred

def return_video_path(youtube_url):
    video_id = youtube_url.split("https://www.youtube.com/watch?v=")[-1].split('&')[0]
    if video_id:
        return os.path.join("workspace", "tmp", f"{video_id}.mp4")
    else:
        raise ValueError("Invalid YouTube URL provided.")

def run_gradio():
    title = """<h1 align="center">Goldfish Demo </h1>"""
    description = """<h5>[ECCV 2024 Accepted]Goldfish: Vision-Language Understanding of Arbitrarily Long Videos</h5>"""
    project_page = """<p><a href='https://vision-cair.github.io/MiniGPT4-video/'><img src='https://img.shields.io/badge/Project-Page-Green'></a></p>"""
    code_link="""<p><a href='https://github.com/Vision-CAIR/MiniGPT4-video'><img src='repo_imgs/goldfishai_png.png'></a></p>"""
    paper_link="""<p><a href=''><img src='https://img.shields.io/badge/Paper-PDF-red'></a></p>"""
    with gr.Blocks(title="Goldfish demo",css=text_css ) as demo :
        gr.Markdown(title)
        gr.Markdown(description)        
        with gr.Tab("Youtube videos") as youtube_tab:
            with gr.Row():
                with gr.Column():
                    youtube_link = gr.Textbox(label="YouTube link", placeholder="Paste YouTube URL here")
                    video_player = gr.Video(autoplay=False)
                    download_finish = gr.State(value=False)
                    youtube_link.change(
                        fn=download_video,
                        inputs=[youtube_link, download_finish], 
                        outputs=[video_player, download_finish]
                    )

            with gr.Row():
                with gr.Column(scale=2) :
                    youtube_question = gr.Textbox(label="Your Question", placeholder="Default: What's this video talking about?")
                    youtube_has_subtitles = gr.Checkbox(label="Use subtitles", value=True)
                    youtube_input_note = """<p>For the global questions set the number of neighbours=-1 otherwise use 3 as the defualt.</p>"""
                    gr.Markdown(youtube_input_note)
                    # input number 
                    youtube_number_of_neighbours=gr.Number(label="Number of Neighbours",interactive=True,value=3)
                    youtube_process_button = gr.Button("⛓️ Answer the Question (QA)")
                with gr.Column(scale=3):
                    youtube_answer = gr.Textbox(label="Answer of the question", lines=8, interactive=True, placeholder="Answer of the question will show up here.")
                youtube_process_button.click(fn=process_video, inputs=[youtube_link, youtube_has_subtitles, youtube_question,youtube_number_of_neighbours], outputs=[youtube_answer])
        with gr.Tab("Local videos") as local_tab:
            with gr.Row():
                with gr.Column():
                    local_video_player = gr.Video(sources=["upload"])
            with gr.Row():
                with gr.Column(scale=2):
                    local_question = gr.Textbox(label="Your Question", placeholder="Default: What's this video talking about?")
                    local_has_subtitles = gr.Checkbox(label="Use subtitles", value=True)
                    local_input_note = """<p>For the global questions set the number of neighbours=-1 otherwise use 3 as the defualt.</p>"""
                    gr.Markdown(local_input_note)
                    local_number_of_neighbours=gr.Number(label="Number of Neighbours",interactive=True,value=3)
                    local_process_button = gr.Button("⛓️ Answer the Question (QA)")
                with gr.Column(scale=3):
                    local_answer = gr.Textbox(label="Answer of the question", lines=8, interactive=True, placeholder="Answer of the question will show up here.")
                local_process_button.click(fn=process_video, inputs=[local_video_player, local_has_subtitles, local_question,local_number_of_neighbours], outputs=[local_answer])
                 
    demo.queue(max_size=10).launch(share=True,show_error=True, show_api=False)

if __name__ == "__main__":
    args=get_arguments()
    goldfish_obj = GoldFish_LV(args)
    run_gradio()