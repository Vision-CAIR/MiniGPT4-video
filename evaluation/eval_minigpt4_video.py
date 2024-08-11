import os
import json
from tqdm import tqdm
import sys 
project_dir = os.getcwd()
sys.path.append(project_dir)
from torch.utils.data import DataLoader
from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser
from minigpt4.conversation.conversation import CONV_VISION
from minigpt4.processors.blip_processors import Blip2ImageTrainProcessor,BlipCaptionProcessor
from minigpt4.datasets.datasets.video_datasets import VideoChatGPTEvalDataset,VideoChatGPTEval_consistancy,Video_validation_Dataset,TVQAEVAL

parser = eval_parser()
parser.add_argument("--dataset", type=str, default='msvd', help="dataset to evaluate")
parser.add_argument("--add_subtitles",action='store_true',help="whether to add subtitles to the video")
parser.add_argument("--name", type=str, default='test', help="evaluation name")
parser.add_argument("--videos_path", type=str, default='videos path', help="path to videos")
parser.add_argument("--subtitles_path", type=str, default='subtitles path', help="path to subtitles")
parser.add_argument("--ann_path", type=str, default='annotations path', help="path to annotations")

parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--start", type=int, default=0, help="start from video number")
parser.add_argument("--end", type=int, default=10000000, help="end at video number")
args = parser.parse_args()

print(args.ckpt)
print(args.name)
print(args.cfg_path)
if "test_configs/mistral_test_config.yaml" == args.cfg_path: 
    llm_name="mistral"
else:   
    llm_name="llama2"
print("using captions",args.add_subtitles)
model, vis_processor,whisper_gpu_id,minigpt4_gpu_id,answer_module_gpu_id = init_model(args)
conv_temp = CONV_VISION.copy()
conv_temp.system = ""
if args.dataset == 'video_chatgpt_generic':
    ann_path=args.ann_path
    videos_path= args.videos_path
    subtitles_path=args.subtitles_path
    annotations_keys=['Q','A','video_name']
    data = VideoChatGPTEvalDataset(vis_processor, videos_path, ann_path,subtitles_path,annotations_keys, add_subtitles=args.add_subtitles,llm_name=llm_name)
elif args.dataset == 'video_chatgpt_temporal':
    ann_path=args.ann_path
    videos_path= args.videos_path
    subtitles_path=args.subtitles_path
    annotations_keys=['Q','A','video_name']
    data = VideoChatGPTEvalDataset(vis_processor, videos_path, ann_path,subtitles_path,annotations_keys, add_subtitles=args.add_subtitles,llm_name=llm_name)
elif args.dataset == 'video_chatgpt_consistency':
    ann_path=args.ann_path
    videos_path= args.videos_path
    subtitles_path=args.subtitles_path
    annotations_keys=[['Q1','Q2'],'A','video_name']
    data = VideoChatGPTEval_consistancy(vis_processor, videos_path, ann_path,subtitles_path,annotations_keys, add_subtitles=args.add_subtitles,llm_name=llm_name)
    
elif args.dataset == 'msrvtt':
    ann_path=args.ann_path
    videos_path= args.videos_path
    subtitles_path=args.subtitles_path
    annotations_keys=['question','answer','video_id']
    data = VideoChatGPTEvalDataset(vis_processor, videos_path, ann_path,subtitles_path,annotations_keys, add_subtitles=args.add_subtitles,llm_name=llm_name)

elif args.dataset == 'msvd':
    ann_path=args.ann_path
    videos_path= args.videos_path
    subtitles_path="" # no subtitles for msvd as these videos don't have audio 
    annotations_keys=['question','answer','video_id']
    data = VideoChatGPTEvalDataset(vis_processor, videos_path, ann_path,subtitles_path,annotations_keys, add_subtitles=args.add_subtitles,llm_name=llm_name)
elif args.dataset == 'activitynet':
    ann_path=args.ann_path
    videos_path= args.videos_path
    subtitles_path=args.subtitles_path
    annotations_keys=['question','answer','video_id']
    data = VideoChatGPTEvalDataset(vis_processor, videos_path, ann_path,subtitles_path,annotations_keys, add_subtitles=args.add_subtitles,llm_name=llm_name)
elif args.dataset == 'tgif':
    ann_path="datasets/evaluation_datasets/tgif/Test_frameqa_question.json"
    videos_path= args.videos_path
    subtitles_path="" # no subtitles for TGIF as these videos don't have audio
    annotations_keys=['question','answer','gif_name']
    data = VideoChatGPTEvalDataset(vis_processor, videos_path, ann_path,subtitles_path,annotations_keys, add_subtitles=False,llm_name=llm_name)
elif args.dataset == 'tvqa':
    # TVQA dataset
    ann_path="datasets/evaluation_datasets/tvqa_short/tvqa_val.json"
    videos_path= args.videos_path
    subtitles_path=args.subtitles_path
    data = TVQAEVAL(vis_processor, videos_path, ann_path,subtitles_path,add_subtitles=args.add_subtitles,llm_name=llm_name)

eval_dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=False)

minigpt4_predict = []
sub="subtitles" if args.add_subtitles else "no_subtitles"
if args.start == 0 and args.end == 10000000:
    save_path = f'results/{args.name}_{args.dataset}_{sub}.json'
else:
    print("start from video number",args.start)
    print("end at video number",args.end)
    save_path = f'results/{args.name}_{args.dataset}_{sub}_{args.start}_{args.end}.json'

os.makedirs("results", exist_ok=True)
c=0
pred_result = {}
gt_result = {}
if args.dataset == 'video_chatgpt_consistency':
    for images, texts_1,texts_2, gt_answers, lengths,videos_ids in tqdm(eval_dataloader,desc=f"Eval {args.dataset}"):
        if args.start<= c <args.end :
            texts_q1 = prepare_texts(texts_1, conv_temp, template='', lengths=lengths)  # warp the texts with conversation template
            texts_q2 = prepare_texts(texts_2, conv_temp, template='', lengths=lengths)  # warp the texts with conversation template
            models_answers_q1 = model.generate(images, texts_q1, max_new_tokens=args.max_new_tokens, do_sample=False, lengths=lengths,num_beams=1)
            models_answers_q2 = model.generate(images, texts_q2, max_new_tokens=args.max_new_tokens, do_sample=False, lengths=lengths,num_beams=1)
            for video_id,model_answer_q1,model_answer_q2, gt_answer,text_q1,text_q2 in zip(videos_ids,models_answers_q1,models_answers_q2, gt_answers,texts_q1,texts_q2):
                result = dict()
                result['video_name'] = video_id
                result['Q1'] = text_q1.split('\n')[-1].replace('[/INST]','')
                result['Q2'] = text_q2.split('\n')[-1].replace('[/INST]','')
                result['A'] = gt_answer
                result['pred1'] = model_answer_q1
                result['pred2'] = model_answer_q2
                pred_result[video_id] = [model_answer_q1,model_answer_q2]
                gt_result[video_id] = [gt_answer]
                minigpt4_predict.append(result)
            # save results every 100 videos to avoid losing results
            if c%100==0:
                with open(save_path, 'w') as f:
                    json.dump(minigpt4_predict, f)
        if c >= args.end :
            break
        c+=1

elif args.dataset == 'tvr':
    for images, texts, gt_answers, lengths,videos_ids in tqdm(eval_dataloader,desc=f"Eval {args.dataset}"):
        if args.start<= c <args.end :
            texts = prepare_texts(texts, conv_temp, template='', lengths=lengths)  # warp the texts with conversation template
            models_answers = model.generate(images, texts, max_new_tokens=args.max_new_tokens, do_sample=False, lengths=lengths,num_beams=1)
            for video_id,model_answer, gt_answer,text in zip(videos_ids,models_answers, gt_answers,texts):
                result = dict()
                result['video_name'] = video_id
                result['Q'] = text.split('\n')[-1].replace('[/INST]','')
                result['A'] = gt_answer
                result['pred'] = model_answer
                pred_result[video_id] = [model_answer]
                gt_result[video_id] = [gt_answer]
                minigpt4_predict.append(result)
            # save results every 100 videos to avoid losing results
            if c%100==0:
                with open(save_path, 'w') as f:
                    json.dump(minigpt4_predict, f)
        if c >= args.end :
            break
        c+=1
elif args.dataset == 'ego_schema' or args.dataset == 'tvqa' or args.dataset == 'tvqa_long_videos':
    for images, texts, gt_answers, lengths,videos_ids in tqdm(eval_dataloader,desc=f"Eval {args.dataset}"):
        if args.start<= c <args.end :
            texts = prepare_texts(texts, conv_temp, template='', lengths=lengths)  # warp the texts with conversation template
            models_answers = model.generate(images, texts, max_new_tokens=args.max_new_tokens, do_sample=False, lengths=lengths,num_beams=1)
            for video_id,model_answer, gt_answer,text in zip(videos_ids,models_answers, gt_answers,texts):
                result = dict()
                result['video_name'] = video_id
                if args.dataset == 'tvqa_long_videos':
                    result['Q'] = text.split('\n\n')[1:]
                else:
                    result['Q'] = text.split('\n')[1:]
                result['A'] = gt_answer
                result['pred'] = model_answer
                pred_result[video_id] = [model_answer]
                gt_result[video_id] = [gt_answer]
                minigpt4_predict.append(result)
            # save results every 100 videos to avoid losing results
            if c%100==0:
                with open(save_path, 'w') as f:
                    json.dump(minigpt4_predict, f)
        if c >= args.end :
            break
        c+=1
else:
    for images, texts, gt_answers, lengths,videos_ids in tqdm(eval_dataloader,desc=f"Eval {args.dataset}"):
        if args.start<= c <args.end :
            texts = prepare_texts(texts, conv_temp, template='', lengths=lengths)  # warp the texts with conversation template
            models_answers = model.generate(images, texts, max_new_tokens=args.max_new_tokens, do_sample=False, lengths=lengths,num_beams=1)
            for video_id,model_answer, gt_answer,text in zip(videos_ids,models_answers, gt_answers,texts):
                result = dict()
                result['video_name'] = video_id
                result['Q'] = text.split('\n')[-1].replace('[/INST]','')
                result['A'] = gt_answer
                result['pred'] = model_answer
                pred_result[video_id] = [model_answer]
                gt_result[video_id] = [gt_answer]
                minigpt4_predict.append(result)
            # save results every 100 videos to avoid losing results
            if c%100==0:
                with open(save_path, 'w') as f:
                    json.dump(minigpt4_predict, f)
        if c >= args.end :
            break
        c+=1

with open(save_path, 'w') as f:
    json.dump(minigpt4_predict, f)
print("saved results to",save_path)



