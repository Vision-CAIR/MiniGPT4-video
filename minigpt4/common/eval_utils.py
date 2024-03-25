import argparse
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
import sys 
sys.path.append('/home/ataallka/minigpt_video/minigpt_multi_img')
from minigpt4.common.registry import registry
from minigpt4.common.config import Config

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
# from minigpt4.runners import *
from minigpt4.tasks import *
from pycocoevalcap.cider.cider import Cider
import os
import openai
from tqdm import tqdm
import json
import ast
import time

def eval_parser():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", help="path to configuration file.",default="test_configs/llama2_test_config.yaml")
    parser.add_argument("--ckpt", type=str,default='checkpoints/video_llama_checkpoint_last.pth', help="path to checkpoint")
    parser.add_argument("--eval_opt", type=str, default='all', help="path to configuration file.")
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
    return parser


def prepare_texts(texts, conv_temp, template='<Img><ImageHere></Img>', lengths=None):
    convs = [conv_temp.copy() for _ in range(len(texts))]
    if lengths is None:
        [conv.append_message(conv.roles[0], '{} {}'.format(template, text)) for conv, text in zip(convs, texts)]
    else:
        templates = [template * length for length in lengths]
        [conv.append_message(conv.roles[0], '{} {}'.format(template, text)) for template, conv, text in zip(templates, convs, texts)]
    [conv.append_message(conv.roles[1], None) for conv in convs]
    texts = [conv.get_prompt() for conv in convs]
    return texts


def init_model(args):
    print('Initialization Model')
    cfg = Config(args)
    cfg.model_cfg.ckpt = args.ckpt
    cfg.model_cfg.lora_r = args.lora_r
    cfg.model_cfg.lora_alpha = args.lora_alpha

    model_config = cfg.model_cfg
    model_config.low_resource = True
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:0')

#     import pudb; pudb.set_trace()
    key = list(cfg.datasets_cfg.keys())[0]
    vis_processor_cfg = cfg.datasets_cfg.get(key).vis_processor.train
    print(vis_processor_cfg)
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    print('Initialization Finished')
    return model, vis_processor

def computeIoU(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    intersection_x1 = max(x1, x3)
    intersection_y1 = max(y1, y3)
    intersection_x2 = min(x2, x4)
    intersection_y2 = min(y2, y4)
    intersection_area = max(0, intersection_x2 - intersection_x1 + 1) * max(0, intersection_y2 - intersection_y1 + 1)
    bbox1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    bbox2_area = (x4 - x3 + 1) * (y4 - y3 + 1)
    union_area = bbox1_area + bbox2_area - intersection_area
    iou = intersection_area / union_area
    return iou

def eval_bleu(results):
    bleus1,bleus2,bleus3,bleus4 = [],[],[],[]
    for result in tqdm (results,desc="bleu_eval"):
        gt = result['gt']
        pred = result['pred']
        bleus1.append(sentence_bleu([gt.split()], pred.split(), weights=(1,0,0,0)))
        bleus2.append(sentence_bleu([gt.split()], pred.split(), weights=(0.5,0.5,0,0)))
        bleus3.append(sentence_bleu([gt.split()], pred.split(), weights=(0.33,0.33,0.33,0)))
        bleus4.append(sentence_bleu([gt.split()], pred.split()))
    # print(np.mean(bleus1),np.mean(bleus2),np.mean(bleus3),np.mean(bleus4),flush=True)
    return {'bleu1':np.mean(bleus1),'bleu2':np.mean(bleus2),'bleu3':np.mean(bleus3),'bleu4':np.mean(bleus4)}

# Create a Cider object
cider_scorer = Cider()
def eval_cider(pred_result,gt_result):
    # Compute CIDEr scores
    mean_cider_scores, cider_scores = cider_scorer.compute_score(gt_result, pred_result)
    cider_scores_dict={}
    for score,pred_vid_id,gt_vid_id in tqdm(zip(cider_scores.tolist(),pred_result,gt_result),desc="cider_eval") :
        assert pred_vid_id==gt_vid_id
        cider_scores_dict[pred_vid_id] = score
    return {'mean_cider_scores':mean_cider_scores,'cider_scores':cider_scores_dict}


openai.api_key_path = "/home/ataallka/chatgpt_api.txt"


def chat_gpt_eval(results,output_path):
    trial=0
    gpt_results=[]
    avg_chatgpt_score=0
    existed_files={}
    # read previous results from output path
    for file in os.listdir(output_path):
        if file.endswith(".json"):
            with open(f'{output_path}/{file}') as json_file:
                data = json.load(json_file)
                gpt_results.append(data[0])
                avg_chatgpt_score+=float(data[0]['chatgpt_score'])
                existed_files[data[0]['video_name']]=True
    length_output_path=len(os.listdir(output_path))
    while len (results)!= length_output_path:
        for res in tqdm(results,desc="chatgpt_eval"):
            if existed_files.get(res['video_name'],False):
                continue
            video_name=res['video_name']
            sentence_1=res['A']
            sentence_2=res['pred']
            try:
                # prompt=f"given these 2 sentences the first one is the ground truth text and the second sentence is the generated text ,give me a score from 0 to 1 to evaluate how much they are similar to each other, and have the same context and related to each other to evaluate the quality of this generated text.the output should be only the score float number without any additional information\nfirst sentence: {sentence_1}\nsecond sentence: {sentence_2}\nscore:"
                prompt=f"given these 2 sentences the first one is the ground truth descrption of a video and the second sentence is the generated text from a video summarization model,give it a score from 0 to 5 to evaluate the model summarization performance.the output should be only the score number without any additional information\nfirst sentence: {sentence_1}\nsecond sentence: {sentence_2}\nscore:"        
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                            {
                            "role": "user",
                            "content": prompt
                            }],
                )
                res['chatgpt_score']=response.choices[0].message['content']
                out={'video_name':video_name,'chatgpt_score':response.choices[0].message['content']}
                gpt_results.append(out)
                # save each video result in a json file
                with open(f'{output_path}/{video_name}.json', 'w') as f:
                    json.dump([out], f)
                avg_chatgpt_score+=float(response.choices[0].message['content'])
            except Exception as e:
                print("chat gpt error",e)
        print ("Finished chat gpt evaluation in trial",trial)
        trial+=1
        length_output_path=len(os.listdir(output_path))
    return results,avg_chatgpt_score/len(results)
def GPT4_answer(question, answer,pred):
        try:
            # Compute the correctness score
            completion = openai.ChatCompletion.create(
                # model="gpt-3.5-turbo",
                model='gpt-4',
                messages=[
                    {
                        "role": "system",
                        "content": 
                            "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                            "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                            "------"
                            "##INSTRUCTIONS: "
                            "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                            "- Consider synonyms or paraphrases as valid matches.\n"
                            "- Evaluate the correctness of the prediction compared to the answer."
                    },
                    {
                        "role": "user",
                        "content":
                            "Please evaluate the following video-based question-answer pair:\n\n"
                            f"Question: {question}\n"
                            f"Correct Answer: {answer}\n"
                            f"Predicted Answer: {pred}\n\n"
                            "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                            "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
                            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                            "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}."
                    }
                ]
            )
            # Convert response to a Python dictionary.
            response_message = completion["choices"][0]["message"]["content"]
            response_dict = ast.literal_eval(response_message)
            return response_dict
        except Exception as e:
            print(f"Error : {e}")
            return None
def GPT4_evaluation(val_result):
    scores=[]
    yes_count=0
    no_count=0
    for res in val_result:
        gpt_response=GPT4_answer(res['Q'],res['A'],res['pred'])
        if gpt_response is None:
            continue
        try:
            scores.append(float(gpt_response['score']))
            if 'yes' in gpt_response['pred'].lower():
                yes_count+=1
            elif 'no' in gpt_response['pred'].lower():
                no_count+=1
        except:
            continue
    avg_score=sum(scores)/len(scores)
    accuracy=(yes_count/(yes_count+no_count))*100
    print(f"chatgpt score: {avg_score} accuracy: {accuracy}")
    return avg_score,accuracy

# with open('results/ckpt_15_res89_res32_Video_validation_Dataset_subtitles.json','r') as f:
#     results = json.load(f)
# t1=time.time()
# avg_score,accuracy=GPT4_evaluation(results)
# print(f"chatgpt score: {avg_score} accuracy: {accuracy}")
# print(f"Time taken: {time.time()-t1}")