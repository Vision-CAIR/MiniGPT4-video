import torch
import webvtt
import os
import cv2 
from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser, eval_bleu,eval_cider,chat_gpt_eval
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
from theme import minigptlv_style, custom_css,text_css
def create_video_grid(images, rows, cols,save_path):
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
    # new_image.save(save_path)
    return new_image

def prepare_input(vis_processor,video_path,subtitle_path,instruction):  
    cap = cv2.VideoCapture(video_path)
    if subtitle_path is not None: 
        # Load the VTT subtitle file
        vtt_file = webvtt.read(subtitle_path) 
        print("subtitle loaded successfully")  
        clip = VideoFileClip(video_path)
        total_num_frames = int(clip.duration * clip.fps)
        # print("Video duration = ",clip.duration)
        clip.close()
    else :
        # calculate the total number of frames in the video using opencv        
        total_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    if "mistral" in args.ckpt :
        max_images_length=90
        max_sub_len = 800
    else:
        max_images_length = 50
        max_sub_len = 400
    images = []
    frame_count = 0
    sampling_interval = int(total_num_frames / max_images_length)
    if sampling_interval == 0:
        sampling_interval = 1
    img_placeholder = ""
    subtitle_text_in_interval = ""
    history_subtitles = {}
    # raw_frames=[]
    number_of_words=0
    transform=transforms.Compose([
                transforms.ToPILImage(),
            ])
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Find the corresponding subtitle for the frame and combine the interval subtitles into one subtitle
        # we choose 1 frame for every 2 seconds,so we need to combine the subtitles in the interval of 2 seconds
        if subtitle_path is not None: 
            for subtitle in vtt_file:
                sub=subtitle.text.replace('\n',' ')
                if (subtitle.start_in_seconds <= (frame_count / int(clip.fps)) <= subtitle.end_in_seconds) and sub not in subtitle_text_in_interval:
                    if not history_subtitles.get(sub,False):
                        subtitle_text_in_interval+=sub+" "
                    history_subtitles[sub]=True
                    break
        if frame_count % sampling_interval == 0:
            # raw_frames.append(Image.fromarray(cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)))
            frame = transform(frame[:,:,::-1]) # convert to RGB
            frame = vis_processor(frame)
            images.append(frame)
            img_placeholder += '<Img><ImageHere>'
            if subtitle_path is not None and subtitle_text_in_interval != "" and number_of_words< max_sub_len: 
                img_placeholder+=f'<Cap>{subtitle_text_in_interval}'
                number_of_words+=len(subtitle_text_in_interval.split(' '))
                subtitle_text_in_interval = ""
        frame_count += 1

        if len(images) >= max_images_length:
            break
    cap.release()
    cv2.destroyAllWindows()
    if len(images) == 0:
        # skip the video if no frame is extracted
        return None,None
    # video_grid_image=create_video_grid(raw_frames,8,len(raw_frames)//8,"concatenated.jpg")
    images = torch.stack(images)
    instruction = img_placeholder + '\n' + instruction
    return images,instruction
def extract_audio(video_path, audio_path):
    video_clip = mp.VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(audio_path, codec="libmp3lame", bitrate="320k")
    
def generate_subtitles(video_path):
    video_id=video_path.split('/')[-1].split('.')[0]
    audio_path = f"workspace/inference_subtitles/mp3/{video_id}"+'.mp3'
    os.makedirs("workspace/inference_subtitles/mp3",exist_ok=True)
    if existed_subtitles.get(video_id,False):
        return f"workspace/inference_subtitles/{video_id}"+'.vtt'
    try:
        extract_audio(video_path,audio_path)
        print("successfully extracted")
        os.system(f"whisper {audio_path}  --language English --model large --output_format vtt --output_dir workspace/inference_subtitles")
        # remove the audio file
        os.system(f"rm {audio_path}")
        print("subtitle successfully generated")  
        return f"workspace/inference_subtitles/{video_id}"+'.vtt'
    except Exception as e:
        print("error",e)
        print("error",video_path)
        return None
    

def run (video_path,instruction,model,vis_processor,gen_subtitles=True):
    if gen_subtitles:
        subtitle_path=generate_subtitles(video_path)
    else :
        subtitle_path=None
    prepared_images,prepared_instruction=prepare_input(vis_processor,video_path,subtitle_path,instruction)
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
    answers = model.generate(prepared_images, prompt, max_new_tokens=args.max_new_tokens, do_sample=True, lengths=[length],num_beams=2)
    # remove the subtitle file and the video file
    if subtitle_path:
        os.system(f"rm {subtitle_path}")
    if video_path.split('.')[-1] == 'mp4' or video_path.split('.')[-1] == 'mkv' or video_path.split('.')[-1] == 'avi':
        os.system(f"rm {video_path}")
    return answers[0]

def run_single_image (image_path,instruction,model,vis_processor):
    image=Image.open(image_path)
    image = vis_processor(image)
    prepared_images=torch.stack([image])
    prepared_instruction='<Img><ImageHere>'+instruction
    length=len(prepared_images)
    prepared_images=prepared_images.unsqueeze(0)
    conv = CONV_VISION.copy()
    conv.system = ""
    # if you want to make conversation comment the 2 lines above and make the conv is global variable
    conv.append_message(conv.roles[0], prepared_instruction)
    conv.append_message(conv.roles[1], None)
    prompt = [conv.get_prompt()]
    answers = model.generate(prepared_images, prompt, max_new_tokens=args.max_new_tokens, do_sample=False, lengths=[length],num_beams=1)
    return answers[0]
   
def get_video_url(url,has_subtitles):
    # get video id from url
    video_id=url.split('v=')[-1].split('&')[0]
    # Create a YouTube object
    youtube = YouTube(url)
    # Get the best available video stream
    video_stream = youtube.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
    if has_subtitles:
        # Download the video to the workspace folder
        print('Downloading video')
        video_stream.download(output_path="workspace",filename=f"{video_id}.mp4")
        print('Video downloaded successfully')
        return f"workspace/{video_id}.mp4"
    else:
        return video_stream.url 
    
  
def get_arguments():
    parser = argparse.ArgumentParser(description="Inference parameters")
    parser.add_argument("--cfg-path", help="path to configuration file.",default="test_configs/llama2_test_config.yaml")
    parser.add_argument("--ckpt", type=str,default='checkpoints/video_llama_checkpoint_last.pth', help="path to checkpoint")
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
model, vis_processor = init_model(args)
conv = CONV_VISION.copy()
conv.system = ""
inference_subtitles_folder="workspace/inference_subtitles"
os.makedirs(inference_subtitles_folder,exist_ok=True)
existed_subtitles={}
for sub in os.listdir(inference_subtitles_folder):
    existed_subtitles[sub.split('.')[0]]=True

def gradio_demo_local(video_path,has_sub,instruction):
    pred=run(video_path,instruction,model,vis_processor,gen_subtitles=has_sub)
    return pred

def gradio_demo_youtube(youtube_url,has_sub,instruction):
    video_path=get_video_url(youtube_url,has_sub)
    pred=run(video_path,instruction,model,vis_processor,gen_subtitles=has_sub)
    return pred
    

title = """<h1 align="center">MiniGPT4-video 🎞️</h1>"""
description = """<h5>This is the demo of MiniGPT4-video Model.</h5>"""
# article = """<p><a href='https://minigpt-4.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a></p><p><a href='https://github.com/Vision-CAIR/MiniGPT-4'><img src='https://img.shields.io/badge/Github-Code-blue'></a></p><p><a href='https://raw.githubusercontent.com/Vision-CAIR/MiniGPT-4/main/MiniGPT_4.pdf'><img src='https://img.shields.io/badge/Paper-PDF-red'></a></p>"""

with gr.Blocks(title="MiniGPT4-video 🎞️",css=text_css ) as demo :
    gr.Markdown(title)
    gr.Markdown(description)
    # gr.Markdown(article)
    gr.Interface(
        fn=gradio_demo_youtube,
        inputs=[gr.Textbox(label="Enter the youtube link"),gr.Checkbox(label='Use subtitles'),gr.Textbox(label="Write any Question")],
        outputs=["text",
                 ],
        title="<h2>YouTube videos</h2>",
        description="Videos length should be from one to two minutes",
        # examples=[
        #     ["https://www.youtube.com/watch?v=dQw4w9WgXcQ", True, "What is the main idea of this video?"],
        #     ["https://www.youtube.com/watch?v=HmGH4LxXS_Y", False, "What are the key points discussed here?"]
        # ],
        css=custom_css,  # Apply custom CSS
        allow_flagging='auto'
        
    )
    gr.Interface(
        fn=gradio_demo_local,
        inputs=[gr.Video(sources=["upload"]),gr.Checkbox(label='Use subtitles'),gr.Textbox(label="Write any Question")],
        outputs=["text",
                 ],
        
        title="<h2>Local videos</h2>",
        description="Upload your videos with length from one to two minutes",
        css=custom_css,  # Apply custom CSS
        allow_flagging='auto'
    )

if __name__ == "__main__":
    demo.queue().launch(share=True,show_error=True)

    
    