# MiniGPT-Video
<!-- technical report link  -->
<!-- demo link  -->
## Overview
MiniGPT4-Video is an innovative model designed for video question answering, adept at comprehending both visual stimuli and conversational content within the video domain. This model undergoes training on extensive video-text and image-text corpora. The architectural framework of MiniGPT4-Video is visually depicted in the accompanying figure. During the frame encoding phase, EVA-CLIP is employed, incorporating a projection layer to facilitate the mapping of visual features onto the textual domain of the Large Language Model (LLM). Similar to MiniGPT-v2, we condense each group of four adjacent visual tokens into a singular token, resulting in a 75% reduction in token count per image, from 256 to 64. Throughout the training process, the LLM assimilates the fusion of video frame features with subtitles, thereby comprehending the temporal dynamics inherent within the video content. During inference, the Whisper model is utilized to generate subtitles for the video. Then, both the video and the subtitle are input to the MiniGPT4-Video model.
![methodology](repo_imgs/short_video_figure_new.jpg)

## :rocket: Demo
**1. Clone the repository** <br>
```bash
git clone https://github.com/Vision-CAIR/MiniGPT4-video.git
```
**2. Set up the environment** <br>
```bash
conda env create -f environment.yml
```
**3. Download the checkpoints**

+ prepare pretrained LLM weights

|                            Llama 2 Chat 7B                             |                                           Mistral 7B Instruct v0.2                  |
:------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------:
| [Download](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/tree/main) | [Downlad](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) |

+ download pretrained checkpoints

| MiniGPT-Video (Llama2 Chat 7B) | MiniGPT-Video (Mistral 7B) |
:------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------:
| [Download](https://huggingface.co/Vision-CAIR/MiniGPT4-Video/blob/main/checkpoints/video_llama_checkpoint_best.pth) | [Download](https://huggingface.co/Vision-CAIR/MiniGPT4-Video/blob/main/checkpoints/video_mistral_all_checkpoint_last.pth) |

**4. Run the demo** <br>
#### Llama2
```bash
python minigpt4_video_demo.py --ckpt path_to_video_checkpoint --cfg-path test_configs/llama2_test_config.yaml
```
#### Mistral
```bash
python minigpt4_video_demo.py --ckpt path_to_video_checkpoint --cfg-path test_configs/mistral_test_config.yaml
```
### Inference
Do the previous steps and replace step 4 with this step 
#### Llama2
```bash
python minigpt4_video_inference.py --ckpt path_to_video_checkpoint --cfg-path test_configs/llama2_test_config.yaml --video_path path_to_video --question "Your question here" 
```
#### Mistral
```bash
python minigpt4_video_inference.py --ckpt path_to_video_checkpoint --cfg-path test_configs/mistral_test_config.yaml --video_path path_to_video --question "Your question here" 
```
## :fire: Training

### Training datasets

Image text training<br>
+ LAION <br>
+ Conceptual Captions <br>
+ SBU <br>

Video text training:<br>

+ [CMD](https://www.robots.ox.ac.uk/~vgg/data/condensed-movies/) <br>
+ Webvid <br> <!-- -[Webvid](https://huggingface.co/datasets/TempoFunk/webvid-10M?row=2)  -->
+ [Video Instructional Dataset 100K](https://huggingface.co/datasets/MBZUAI/VideoInstruct-100K) <br>

You can find the datasets annotation files [download](https://huggingface.co/Vision-CAIR/MiniGPT4-Video/tree/main/datasets/training_datasets) <br>

### Model training: 
You can edit the number of gpus in the script.sh below<br>
#### Stage 1 (image text pretraining)
#### For Mistral
Set the cfg-path in the script to ---- <br>
#### For Llama2
Set the cfg-path in the script to ---- <br>
```bash
```
#### Stage 2 (video captioning pretraining)

#### For Mistral
Set the cfg-path in the script to train_configs/224_v2_mistral_video_stage_2.yaml <br>
#### For Llama2
Set the cfg-path in the script to train_configs/224_v2_llama2_video_stage_2.yaml <br>

```bash
bash jobs_video/train/stage_2.sh
```
#### Stage 3 (video Instruction finetuning)
#### For Mistral
Set the cfg-path in the script to train_configs/224_v2_mistral_video_stage_3.yaml <br>
#### For Llama2
Set the cfg-path in the script to train_configs/224_v2_llama2_video_stage_3.yaml <br>

```bash
bash jobs_video/train/stage_3.sh
```

## :zap: Evaluation
We used the same evaluation as [Video-ChatGPT](https://mbzuai-oryx.github.io/Video-ChatGPT/)<br>
![short_results](repo_imgs/short_results.PNG)

### Download datasets for evaluation
+ [MSVD](https://www.cs.utexas.edu/users/ml/clamp/videoDescription/) <br>
+ [MSRVTT](https://cove.thecvf.com/datasets/839) <br>
+ [TGIF](https://github.com/YunseokJANG/tgif-qa/blob/master/dataset/README.md) <br>
+ [ActivityNet](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hanoona_bangalath_mbzuai_ac_ae/ESa302OCJMNHsMk7wuBbQc8BZH5CqlcdCWiSpXynQZDfAQ?e=CrOPbm) <br>
+ [TVQA](https://tvqa.cs.unc.edu/) <br>
+ [Video-ChatGPT benchmark](https://mbzuai-oryx.github.io/Video-ChatGPT/) <br>

You can find the datasets annotation files [download](https://huggingface.co/Vision-CAIR/MiniGPT4-Video/tree/main/datasets/evaluation_datasets) <br>

### Run evaluation script
Edit the evaluation script to include the path to the checkpoints, the dataset name and whether to use subtitles or not <br> 
#### For Mistral
```bash
bash jobs_video/eval/mistral_evalualtion.sh
```
#### For Llama 2 
```bash
bash jobs_video/eval/llama2_evaluation.sh
```
Then Use GPT3.5 turbo to compare the predictions with the ground truth and generate the accuracy and scores <br>
To evaluate [videochatgpt benchmark] run the following script <br>
```bash
bash test_benchmark/quantitative_evaluation/evaluate_benchmark.sh
```
To evaluate open ended questions run the following script <br>
```bash
bash test_benchmark/quantitative_evaluation/evaluate_zeroshot.sh
```
