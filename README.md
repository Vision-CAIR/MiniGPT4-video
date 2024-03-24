# Minigpt4-Video : Video question answering
<!-- technical report link  -->
<!-- demo link  -->
## Overview
MiniGPT4-Video is an innovative model designed for video question answering, adept at comprehending both visual stimuli and conversational content within the video domain. This model undergoes training on extensive video-text and image-text corpora. The architectural framework of MiniGPT4-Video is visually depicted in the accompanying figure. During the frame encoding phase, EVA-CLIP is employed, incorporating a projection layer to facilitate the mapping of visual features onto the textual domain of the Large Language Model (LLM). Similar to MiniGPT-v2, we condense each group of four adjacent visual tokens into a singular token, resulting in a 75% reduction in token count per image, from 256 to 64. Throughout the training process, the LLM assimilates the fusion of video frame features with subtitles, thereby comprehending the temporal dynamics inherent within the video content. During inference, the Whisper model is utilized to generate subtitles for the video. Then, both the video and the subtitle are input to the MiniGPT4-Video model.
![methodology](repo_imgs/short_video_figure_new.jpg)

## How to run the demo: 
**1. Clone the repository** <br>
```bash
git clone https://github.com/Vision-CAIR/MiniGPT4-video.git
```
**2. Set up the environment** <br>
```bash
conda env create -f environment.yml
```
**3. Download the checkpoints**

[checkpoints](https://huggingface.co/Vision-CAIR/MiniGPT4-Video/tree/main/checkpoints)  <br>

**4. Run the demo** <br>
```bash
python minigpt4_video_demo.py --ckpt path_to_video_checkpoint
```
## How to use the model for inference:
Do the previous steps and replace step 4 with this step 
```bash
python minigpt4_video_inference.py --ckpt path_to_video_checkpoint --video_path path_to_video --question "Your question here" 
```
## Training datasets :
You can find the datasets annotation files [datasets](https://huggingface.co/Vision-CAIR/MiniGPT4-Video/tree/main/datasets) <br>:
We trained the MiniGPT4-video model on the following datasets:
We used the MiniGPT-v2 architecture and train it with the following datasets:
For image text training we used the following datasets (these are the datasets used for Minigpt-4 v1): <br>
+ LAION <br>
+ Conceptual Captions <br>
+ SBU <br>
For video text training we extend the architecture to accept mutiple frames then trained  we used the following datasets:<br>
You can find the datasets annotation files in the datasets folder and can download the videos from the following links: <br>
+ [CMD](https://www.robots.ox.ac.uk/~vgg/data/condensed-movies/) <br>
+ Webvid <br> <!-- -[Webvid](https://huggingface.co/datasets/TempoFunk/webvid-10M?row=2)  -->
+ [Video Instructional Dataset 100K](https://huggingface.co/datasets/MBZUAI/VideoInstruct-100K) <br>

## Model training
Prepare the configrations in the train_configs then run the training script <br>
You should Edit the number of gpus in the parallel_training.sh script
```bash
bash jobs_video/train/parallel_training.sh 
```
or 
``` bash
torchrun --master-port port number --nproc-per-node number_of_gpus train.py --job_name "job_name" --cfg-path "train cfg_path"
```

## Model evaluation
![short_results](repo_imgs/short_results.PNG)
we used the same evaluation as videochatgpt evaluation [here](https://mbzuai-oryx.github.io/Video-ChatGPT/)<br>
### Download datasets for evaluation
1- [MSVD](https://www.cs.utexas.edu/users/ml/clamp/videoDescription/) <br>
2- [MSRVTT](https://cove.thecvf.com/datasets/839) <br>
3- [TGIF](https://github.com/YunseokJANG/tgif-qa/blob/master/dataset/README.md) <br>
4- [ActivityNet](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hanoona_bangalath_mbzuai_ac_ae/ESa302OCJMNHsMk7wuBbQc8BZH5CqlcdCWiSpXynQZDfAQ?e=CrOPbm) <br>
5- [TVQA](https://tvqa.cs.unc.edu/) <br>
6-[Videochatgpt benchmark](https://mbzuai-oryx.github.io/Video-ChatGPT/) <br>

### Run evaluation script
Edit the evaluation script to include the path to the checkpoints and the dataset name 
```bash
bash jobs_video/eval/evalualtion.sh
```
then Use GPT3.5 turbo to compare the predictions with the ground truth and generate the accuracy and scores <br>
To evaluate [videochatgpt benchmark] run the following script <br>
```bash
bash test_benchmark/quantitative_evaluation/evaluate_benchmark.sh
```
To evaluate open ended questions run the following script <br>
```bash
bash test_benchmark/quantitative_evaluation/evaluate_zeroshot.sh
```
