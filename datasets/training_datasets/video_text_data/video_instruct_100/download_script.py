import json 
from tqdm import tqdm
from pytubefix import YouTube

import xml.etree.ElementTree as ET
import os

with open ('VideoInstruct100K.json','r') as f :
    data=json.load(f)

# Usage
existed_video_id={}
for video_name in os.listdir('videos'):
    video_id = video_name.split('.')[0]
    existed_video_id[video_id]=True 



def download_video_with_subtitles(video_id):
    # Create a YouTube object.
    yt = YouTube(f'https://www.youtube.com/watch?v={video_id}')

    video_filename = f"{video_id}.mp4"
    video_downloaded=False
    try :
        # Get the video stream with the highest resolution and download the video.
        stream = yt.streams.get_highest_resolution()
        stream.download(output_path='videos', filename=video_filename)
        video_downloaded=True
    except Exception as e:
        print(f"Error downloading video {video_id}: {str(e)}")
        video_downloaded=False
    if not video_downloaded:
        return False,False

    # Get the video's available captions (subtitles).
    captions = yt.captions.all()

    # Download the captions if available in xml format.
    caption_downloaded = False
    for caption in captions:
        caption_code = caption.code
        # select only english captions
        if 'en' in caption_code:
            caption.download(title=f"{video_id}", output_path='subtitles_xml',srt=False)
            caption_downloaded = True
    return video_downloaded,caption_downloaded
def convert_xml_vtt(xml_path, vtt_path):
    # Parse the XML subtitle file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Initialize a list to store VTT subtitle entries
    vtt_subtitle = []

    # Function to convert time in milliseconds to WebVTT format
    def ms_to_vtt_time(milliseconds):
        seconds, milliseconds = divmod(milliseconds, 1000)
        minutes, seconds = divmod(seconds, 60)
        return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

    # Iterate through subtitle elements
    toggle = True
    for p in root.findall(".//p"):
        if toggle:
            start_time = int(p.get("t"))
            subtitle_text = " ".join(s.text.strip() for s in p.findall(".//s"))
        # duration = int(p.get("d")) if p.get("d") is not None else 0
        if not toggle:
            end_time = int(p.get("t"))
            # Format and append the VTT entry to the list
            vtt_subtitle.append(f"{ms_to_vtt_time(start_time)} --> {ms_to_vtt_time(end_time)}\n{subtitle_text}\n")
        toggle = not toggle
    # Join the VTT entries into a single string
    vtt_content = "WEBVTT\n\n" + "\n".join(vtt_subtitle)

    # Save the VTT content to a file
    with open(vtt_path, "w", encoding="utf-8") as vtt_file:
        vtt_file.write(vtt_content)
import os
os.makedirs('videos', exist_ok=True)
os.makedirs('subtitles_vtt', exist_ok=True)
os.makedirs('subtitles_xml', exist_ok=True)
for video_path in tqdm(data,desc='Downloading videos') :
    video_id=video_path.split('/')[-1].split('.')[0]
    if existed_video_id.get(video_id,False):
        continue
    video_downloaded,caption_downloaded=download_video_with_subtitles(video_id)
    if caption_downloaded:
        # convert xml to vtt
        xml_file_path=f'subtitles_xml/{video_id} (a.en).xml'
        convert_xml_vtt(xml_file_path,f'subtitles_vtt/{video_id}.vtt')
        
    

