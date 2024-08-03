import sys 
import os
project_dir = os.getcwd()
sys.path.append(project_dir)
import json 
from tqdm import tqdm
from goldfish_lv import GoldFish_LV,split_subtitles,time_to_seconds
import argparse
import json
import torch
import re
from tqdm import tqdm
from PIL import Image
from index import MemoryIndex  
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn
import shutil
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
    parser.add_argument("--neighbours", type=int, default=-1)
    parser.add_argument("--name", type=str,default="ckpt_92",help="name of the experiment")
    parser.add_argument("--add_unknown", action='store_true')
    parser.add_argument("--use_chatgpt", action='store_true')
    parser.add_argument("--use_choices_for_info", action='store_true')
    parser.add_argument("--use_gt_information", action='store_true')
    parser.add_argument("--inference_text", action='store_true')
    parser.add_argument("--use_gt_information_with_distraction", action='store_true')
    parser.add_argument("--num_distraction", type=int, default=2)
    parser.add_argument("--add_confidance_score", action='store_true')
    parser.add_argument("--use_original_video", action='store_true')
    parser.add_argument("--use_video_embedding", action='store_true')
    parser.add_argument("--use_clips_for_info", action='store_true')
    parser.add_argument("--use_GT_video", action='store_true')
    parser.add_argument("--use_gt_summary", action='store_true')
    parser.add_argument("--index_subtitles", action='store_true')
    parser.add_argument("--index_subtitles_together", action='store_true')
    
    parser.add_argument("--ask_the_question_early", action='store_true')
    parser.add_argument("--clip_in_ask_early", action='store_true')
    parser.add_argument("--summary_with_subtitles_only", action='store_true')
    parser.add_argument("--use_coherent_description", action='store_true')
    
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=100000, type=int)
    parser.add_argument("--exp_name", type=str,default="",help="name of eval folder")
    
    
    parser.add_argument("--vision_only", action='store_true')
    parser.add_argument("--model_summary_only", action='store_true')
    parser.add_argument("--subtitles_only", action='store_true')
    parser.add_argument("--info_only", action='store_true')
    
    parser.add_argument("--cfg-path", default="test_configs/llama2_test_config.yaml")
    parser.add_argument("--ckpt", type=str, default="checkpoints/video_llama_checkpoint_last.pth")
    parser.add_argument("--add_subtitles", action='store_true')
    parser.add_argument("--eval_opt", type=str, default='all')
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--video_path", type=str, help="path to the video")
    parser.add_argument("--use_openai_embedding",type=str2bool, default=False)
    parser.add_argument("--annotation_path", type=str, help="path to the annotation file")
    parser.add_argument("--videos_path", type=str, help="path to the videos directory")
    parser.add_argument("--subtitle_path", type=str, help="path to the subtitles directory")
    parser.add_argument("--movienet_annotations_dir", type=str, help="path to the movienet annotations directory")
    parser.add_argument("--video_clips_saving_path", type=str, help="path to save the splitted small video clips")
    
    parser.add_argument("--save_path", type=str, help="path to save the results")
    
    parser.add_argument("--options", nargs="+")
    return parser.parse_args()
def time_to_seconds(subrip_time):
    return subrip_time.hours * 3600 + subrip_time.minutes * 60 + subrip_time.seconds + subrip_time.milliseconds / 1000

def clean_text(subtitles_text):
    # Remove unwanted characters except for letters, digits, and single quotes
    subtitles_text = re.sub(r'[^a-zA-Z0-9\s\']', '', subtitles_text)
    # Replace multiple spaces with a single space
    subtitles_text = re.sub(r'\s+', ' ', subtitles_text)
    return subtitles_text.strip()
        
class LlamaVidQAEval (GoldFish_LV):
    
    def __init__(self,args):
        super().__init__(args)
        self.save_json_path = "new_workspace/clips_summary/movienet"
        if args.use_openai_embedding:
            self.save_pkls_path = "new_workspace/open_ai_embedding/movienet"
        else:
            self.save_pkls_path = "new_workspace/embedding/movienet"
        os.makedirs(self.save_json_path, exist_ok=True)
        annotation_path=args.annotation_path
        with open(annotation_path, 'r') as f:
            self.movies_dict = json.load(f)
        self.max_sub_len=400
        self.max_num_images=45
    

    def _get_movie_data(self,videoname):
        video_images_path =f"{args.videos_path}/{videoname}"
        movie_clips_path =f"{args.video_clips_saving_path}/{videoname}"
        subtitle_path = f"{args.subtitle_path}/{videoname}.srt"
        annotation_file=f"{args.movienet_annotations_dir}/{videoname}.json"
        # load the annotation file 
        with open(annotation_file, 'r') as f:
            movie_annotation = json.load(f)
        return video_images_path,subtitle_path,movie_annotation,movie_clips_path
    def _store_subtitles_paragraphs(self,subtitle_path,important_data,number_of_paragraphs):
        paragraphs=[]
        movie_name=subtitle_path.split('/')[-1].split('.')[0]
        # if there is no story, split the subtitles into paragraphs 
        paragraphs = split_subtitles(subtitle_path, number_of_paragraphs)  
        for i,paragraph in enumerate(paragraphs):
            paragraph=clean_text(paragraph)
            important_data.update({f"subtitle_{i}__{movie_name}_clip_{str(i).zfill(2)}": paragraph})  
        return important_data
    def _get_shots_subtitles(self,movie_annotation):
        shots_subtitles={}
        if movie_annotation['story'] is not None:
            for section in movie_annotation['story']:
                for shot in section['subtitle']:
                    shot_number=shot['shot']
                    shot_subtitle=' '.join(shot['sentences'])
                    shots_subtitles[shot_number]=clean_text(shot_subtitle)
                
        return shots_subtitles
    
    def prepare_input_images(self,clip_path,shots_subtitles,use_subtitles):
        total_frames=len(os.listdir(clip_path))
        movie_name=clip_path.split('/')[-2]
        clip_name=clip_path.split('/')[-1]
        sampling_interval=int(total_frames//self.max_num_images)
        if sampling_interval==0:
            sampling_interval=1
        use_subtitles_save_name="subtitles" if use_subtitles else "no_subtitles"
        video_frames_path = os.path.join(clip_path)
        total_num_frames=len(os.listdir(video_frames_path))
        sampling_interval = round(total_num_frames / self.max_num_images)
        if sampling_interval == 0:
            sampling_interval = 1
        number_of_words=0
        video_images_list=sorted(os.listdir(video_frames_path))
        images = []
        img_placeholder = ""
        for i,frame in enumerate(video_images_list):
            if i % sampling_interval == 0:
                frame = Image.open(os.path.join(video_frames_path,frame)).convert("RGB") 
                frame = self.vis_processor(frame)
                images.append(frame)
                img_placeholder += '<Img><ImageHere>'
                shot_num=video_images_list[i].split('_')[1]
                if shots_subtitles.get(shot_num) is not None:
                    sub=clean_text(shots_subtitles[shot_num])
                    number_of_words+=len(sub.split(' '))
                    if number_of_words<= self.max_sub_len and use_subtitles:
                        img_placeholder+=f'<Cap>{sub}'
            if len(images) >= self.max_num_images:
                break
        if len(images) ==0:
            print("Video not found",video_frames_path)
            
        if 0 <len(images) < self.max_num_images:
            last_item = images[-1]
            while len(images) < self.max_num_images:
                images.append(last_item)
                img_placeholder += '<Img><ImageHere>'
        images = torch.stack(images)

        return images,img_placeholder

    def _get_movie_summaries(self,video_images_path,use_subtitles,shots_subtitles,movie_clips_path):
        video_images_list=sorted(os.listdir(video_images_path))
        max_caption_index = 0
        preds = {}
        movie_name=movie_clips_path.split('/')[-1]
        videos_summaries=[]
        previous_caption=""
        batch_size=args.batch_size
        batch_images=[]
        batch_instructions=[]
        clip_numbers=[]
        clip_number=0
        conversations=[]
        for i in tqdm(range(0,len(video_images_list),135), desc="Inference video clips", total=len(video_images_list)/120):
            images=[]
            # Add the previous caption to the new video clip 
            # if batch_size==1:
            #     previous_caption="You are analysing a one long video of mutiple clips and this is the summary from all previous clips :"+videos_summaries[-1] +"\n\n"if len(videos_summaries)>0 else ""
            if previous_caption != "":
                img_placeholder = previous_caption+" "
            else:
                img_placeholder = ""
            number_of_words=0
            max_num_words=400
            max_num_images=45
            clip_number_str=str(clip_number).zfill(2)
            clip_path=os.path.join(movie_clips_path,f"{movie_name}_clip_{clip_number_str}")
            os.makedirs(clip_path, exist_ok=True)
            conversation=""
            for j in range(i,i+135,3):
                if j >= len(video_images_list):
                    break
                image_path = os.path.join(video_images_path, video_images_list[j])
                # copy the images to clip folder 
                # if the image is already copied, skip it 
                if not os.path.exists(os.path.join(clip_path,video_images_list[j])):
                    shutil.copy(image_path,clip_path)
                img=Image.open(image_path)
                images.append(self.vis_processor(img))
                img_placeholder += '<Img><ImageHere>'
                shot_num=int(video_images_list[j].split('_')[1])
                if use_subtitles:
                    if shots_subtitles.get(shot_num) is not None:
                        sub=clean_text(shots_subtitles[shot_num])
                        number_of_words+=len(sub.split(' '))
                        if number_of_words<= max_num_words and use_subtitles:
                            img_placeholder+=f'<Cap>{sub}'
                        conversation+=sub+" "        
                if len(images) >= max_num_images:
                    break
            if len(images) ==0:
                print("Video not found",video_images_path)
                continue
            if 0 <len(images) < max_num_images:
                last_item = images[-1]
                while len(images) < max_num_images:
                    images.append(last_item)
                    img_placeholder += '<Img><ImageHere>'
            images = torch.stack(images)
            print(images.shape)
            clip_numbers.append(clip_number_str)
            clip_number+=1
            conversations.append(clean_text(conversation))
            instruction = img_placeholder + '\n' + self.summary_instruction
            batch_images.append(images)
            batch_instructions.append(instruction)
            if len(batch_images) < batch_size:
                continue
            # run inference for the batch
            batch_images = torch.stack(batch_images)
            batch_pred=self.run_images(batch_images,batch_instructions)
            for i,pred in enumerate(batch_pred):
                max_caption_index += 1
                videos_summaries.append(pred)
                if args.use_coherent_description:
                    preds[f'caption_{max_caption_index}__{movie_name}_clip_{clip_numbers[i]}'] = f"model_summary :{pred}\nVideo conversation :{conversations[i]}"
                else:
                    preds[f'caption_{max_caption_index}__{movie_name}_clip_{clip_numbers[i]}'] = pred 
                    if conversations[i]!="" and use_subtitles:
                        preds[f'subtitle_{max_caption_index}__{movie_name}_clip_{clip_numbers[i]}'] = conversations[i]
                   
            batch_images=[]
            batch_instructions=[]
            clip_numbers=[]
            conversations=[]
        
        # run inference for the last batch
        if len(batch_images)>0:
            batch_images = torch.stack(batch_images)
            batch_pred=self.run_images(batch_images,batch_instructions)
            for k,pred in enumerate(batch_pred):
                max_caption_index += 1
                videos_summaries.append(pred)
                if args.use_coherent_description:
                    preds[f'caption_{max_caption_index}__{movie_name}_clip_{clip_numbers[k]}'] = f"model_summary :{pred}\nVideo conversation :{conversations[k]}"
                else:
                    preds[f'caption_{max_caption_index}__{movie_name}_clip_{clip_numbers[k]}'] = pred 
                    if conversations[k]!="" and use_subtitles:
                        preds[f'subtitle_{max_caption_index}__{movie_name}_clip_{clip_numbers[k]}'] = conversations[k]
                    
            batch_images=[]
            batch_instructions=[]
        return preds
    def movie_inference(self,videoname,use_subtitles):
        embedding_path=os.path.join(self.save_pkls_path,f"{videoname}.pkl")
        if args.index_subtitles_together:
            file_path=os.path.join(self.save_json_path,f"{videoname}.json")
            embedding_path=os.path.join(self.save_pkls_path,f"{videoname}.pkl")
        else:
            file_path=os.path.join(self.save_json_path,f"no_subtiltles_{videoname}.json")
            embedding_path=os.path.join(self.save_pkls_path,f"no_subtiltles_{videoname}.pkl")
        
        if args.subtitles_only:
            file_path=os.path.join(self.save_json_path,f"subtiltles_only_{videoname}.json")
            embedding_path=os.path.join(self.save_pkls_path,f"subtiltles_only_{videoname}.pkl")
           
        if os.path.exists(file_path):
            print("Already processed")
            return file_path,embedding_path
        important_data = {}
        video_images_path,subtitle_path,movie_annotation,movie_clips_path=self._get_movie_data(videoname)
        shots_subtitles={}
        if use_subtitles: 
            if movie_annotation['story'] is not None:
                shots_subtitles=self._get_shots_subtitles(movie_annotation) 
        if args.subtitles_only:
            number_of_paragraphs=20
            important_data=self._store_subtitles_paragraphs(subtitle_path,important_data,number_of_paragraphs)
        else:
            preds=self._get_movie_summaries(video_images_path,use_subtitles,shots_subtitles,movie_clips_path)
            if len(shots_subtitles)==0 and use_subtitles:
                number_of_paragraphs=len(preds)
                important_data=self._store_subtitles_paragraphs(subtitle_path,important_data,number_of_paragraphs)
            important_data.update(preds)
        with open(file_path, 'w') as file:
            json.dump(important_data, file, indent=4)
        return file_path,embedding_path
    def answer_movie_questions_RAG(self,qa_list,information_RAG_path,embedding_path):
        QA_external_memory=MemoryIndex(args.neighbours, use_openai=args.use_openai_embedding)
        if os.path.exists(embedding_path):
            QA_external_memory.load_embeddings_from_pkl(embedding_path)
        else:
            QA_external_memory.load_documents_from_json(information_RAG_path,embedding_path)
        summarization_external_memory=MemoryIndex(-1, use_openai=args.use_openai_embedding)
        if os.path.exists(embedding_path):
            summarization_external_memory.load_embeddings_from_pkl(embedding_path)
        else:
            summarization_external_memory.load_documents_from_json(information_RAG_path,embedding_path)
    
        # get the most similar context from the external memory to this instruction 
        general_related_context_keys_list=[]
        general_related_context_documents_list=[]
        summary_related_context_documents_list=[]
        summary_related_context_keys_list=[]
        total_batch_pred=[]
        related_text=[]
        qa_genearl_prompts=[]
        qa_summary_prompts=[]
        qa_general=[]
        qa_summary=[]
        for qa in qa_list:
            if qa['q_type']=='summary':
                related_context_documents,related_context_keys = summarization_external_memory.search_by_similarity(qa['Q'])
                summary_related_context_documents_list.append(related_context_documents)
                summary_related_context_keys_list.append(related_context_keys)
                prompt=self.prepare_prompt(qa)
                qa_summary_prompts.append(prompt)
                qa_summary.append(qa)
            else:
                related_context_documents,related_context_keys = QA_external_memory.search_by_similarity(qa['Q'])
                general_related_context_keys_list.append(related_context_keys)
                general_related_context_documents_list.append(related_context_documents)
                prompt=self.prepare_prompt(qa)
                qa_genearl_prompts.append(prompt)
                qa_general.append(qa)  
        # if I have summary questions answer first, without the need to use clips for information 
        if len(qa_summary_prompts)>0:
            # Here the retrieved clips are all movie clips
            context_information_list=[]
            for related_context_keys in summary_related_context_keys_list:
                most_related_clips=self.get_most_related_clips(related_context_keys)
                context_information=""
                for clip_name in most_related_clips:
                    clip_conversation=""
                    general_sum=""
                    for key in related_context_keys:
                        if clip_name in key and 'caption' in key:
                            general_sum="Clip Summary: "+summarization_external_memory.documents[key]
                        if clip_name in key and 'subtitle' in key:
                            clip_conversation="Clip Subtitles: "+summarization_external_memory.documents[key]
                    
                    if args.use_coherent_description:
                        context_information+=f"{general_sum}\n"
                    else:
                        if args.model_summary_only:
                            context_information+=f"{general_sum}\n"
                        elif args.subtitles_only:
                            context_information+=f"{clip_conversation}\n"
                        else:
                            context_information+=f"{general_sum},{clip_conversation}\n"
                context_information_list.append(context_information) 
            if args.use_chatgpt :
                batch_pred=self.inference_RAG_chatGPT(qa_summary_prompts,context_information_list)
            else:
                batch_pred=self.inference_RAG(qa_summary_prompts,context_information_list) 
            total_batch_pred.extend(batch_pred)
            related_text.extend(context_information_list)
        
        if args.use_clips_for_info:
            batch_pred,general_related_context_keys_list=self.use_clips_for_info(qa_general,general_related_context_keys_list,QA_external_memory)   
            total_batch_pred.extend(batch_pred)
            related_text.extend(general_related_context_keys_list)
        else:   
            related_context_documents_text_list=[]
            for related_context_documents,related_context_keys in zip(general_related_context_documents_list,general_related_context_keys_list): 
                related_information=""
                most_related_clips=self.get_most_related_clips(related_context_keys)
                for clip_name in most_related_clips:
                    clip_conversation=""
                    general_sum=""
                    for key in QA_external_memory.documents.keys():
                        if clip_name in key and 'caption' in key:
                            general_sum="Clip Summary: "+QA_external_memory.documents[key]
                        if clip_name in key and 'subtitle' in key:
                            clip_conversation="Clip Subtitles: "+QA_external_memory.documents[key]
                    if args.use_coherent_description:
                        related_information+=f"{general_sum}\n"
                    else:
                        if args.model_summary_only:
                            related_information+=f"{general_sum}\n"
                        elif args.subtitles_only:
                            related_information+=f"{clip_conversation}\n"
                        else:
                            related_information+=f"{general_sum},{clip_conversation}\n"
                    
                related_context_documents_text_list.append(related_information)
            
            if len (qa_genearl_prompts) >0 and args.use_chatgpt :
                batch_pred=self.inference_RAG_chatGPT(qa_genearl_prompts,related_context_documents_text_list)
            elif len (qa_genearl_prompts) >0:
                batch_pred=self.inference_RAG(qa_genearl_prompts,related_context_documents_text_list)
                total_batch_pred.extend(batch_pred)
                related_text.extend(related_context_documents_text_list)
        assert len(total_batch_pred)==len(related_text)
        return total_batch_pred, related_text
    def get_most_related_clips(self,related_context_keys):
        most_related_clips=[]
        for context_key in related_context_keys:
            if len(context_key.split('__'))>1:
                most_related_clips.append(context_key.split('__')[1])
            if len(most_related_clips)==args.neighbours:
                break
        assert len(most_related_clips)!=0, f"No related clips found {related_context_keys}"
        return most_related_clips 
    
    def clip_inference(self,clips_name,prompts):
        setup_seeds(seed)
        images_batch, instructions_batch = [], []
        for clip_name, prompt in zip(clips_name, prompts):
            movie_name=clip_name.split('_')[0]
            video_images_path,subtitle_path,movie_annotation,movie_clips_path=self._get_movie_data(movie_name)
            clip_path=os.path.join(movie_clips_path,clip_name)
            if movie_annotation['story'] is not None:
                shots_subtitles=self._get_shots_subtitles(movie_annotation) 
            else:
                shots_subtitles={}
            images,img_placeholder=self.prepare_input_images(clip_path,shots_subtitles,use_subtitles=not args.vision_only)
            instruction = img_placeholder + '\n' + prompt
            images_batch.append(images)
            instructions_batch.append(instruction)
        # run inference for the batch
        images_batch=torch.stack(images_batch)
        batch_pred=self.run_images(images_batch,instructions_batch)
        return batch_pred
    def prepare_prompt(self,qa):
        prompt=qa["Q"]
        return prompt
    def use_clips_for_info(self,qa_list,related_context_keys_list,external_memory):
            total_batch_pred=[]
            questions=[]
            related_information_list=[]
            related_context_keys_list_new=[]
            for qa,related_context_keys in zip(qa_list,related_context_keys_list):
                most_related_clips=self.get_most_related_clips(related_context_keys)
                question=qa['Q']
                # prompt=self.prepare_prompt(qa)
                # prompt+=" and also provide an EXPLAINATION for your answer and If you don't know the answer, say that you don't know.\n\n"
                prompt=f"From this video extract the related information to This question and provide an explaination for your answer and If you can't find related information, say 'I DON'T KNOW' as option 5 because maybe the questoin is not related to the video content.\n the question is :\n {question}\n your answer :"
                # all_info=self.clip_inference(most_related_clips,[prompt]*len(most_related_clips))
                # make the most_related_clips has unique elements (if retrival from vision summary and conversations)
                most_related_clips=list(set(most_related_clips))
                batch_inference=[]
                all_info=[]
                for related_clip in most_related_clips:
                    batch_inference.append(related_clip)
                    if len(batch_inference)<args.batch_size:
                        continue    
                    all_info.extend(self.clip_inference(batch_inference,[prompt]*len(batch_inference)))
                    batch_inference=[]
                if len(batch_inference)>0:
                    all_info.extend(self.clip_inference(batch_inference,[prompt]*len(batch_inference)))
           
                related_information=""
                for info,clip_name in zip(all_info,most_related_clips):
                    clip_conversation=""
                    general_sum=""
                    for key in external_memory.documents.keys():
                        if clip_name in key and 'caption' in key:
                            general_sum="Clip Summary: "+external_memory.documents[key]
                        if clip_name in key and 'subtitle' in key:
                            clip_conversation="Clip Subtitles: "+external_memory.documents[key]
                    
                    if args.use_coherent_description:
                        related_information+=f"question_related_information: {info},{general_sum}\n"
                    else:
                        if args.model_summary_only:
                            related_information+=f"{general_sum},question_related_information: {info}\n"
                        elif args.info_only:
                            related_information+=f"question_related_information: {info}\n"
                        elif args.subtitles_only:
                            related_information+=f"{clip_conversation},question_related_information: {info}\n"
                        else:
                            related_information+=f"{general_sum},{clip_conversation},question_related_information: {info}\n" 
                        
                         
                        # related_information+=f"question_related_information: {info},{clip_conversation}\n"
                questions.append(question)
                related_information_list.append(related_information)
                related_context_keys.append(related_information)
                related_context_keys_list_new.append(related_context_keys)
                if len(questions)< args.batch_size:
                    continue
                setup_seeds(seed)
                if args.use_chatgpt :
                    batch_pred=self.inference_RAG_chatGPT(questions, related_information_list)
                else:
                    batch_pred=self.inference_RAG(questions, related_information_list)
                
                for pred in batch_pred:
                    total_batch_pred.append(pred)
                questions=[]
                related_information_list=[]
            
            if len(questions)>0:
                setup_seeds(seed)
                if args.use_chatgpt :
                    batch_pred=self.inference_RAG_chatGPT(questions, related_information_list)
                else:
                    batch_pred=self.inference_RAG(questions, related_information_list)
                for pred in batch_pred:
                    total_batch_pred.append(pred)
            return total_batch_pred,related_context_keys_list_new
    def define_save_name(self):
        save_name="subtitles" if args.index_subtitles_together else "no_subtitles"
        save_name+="_clips_for_info" if args.use_clips_for_info else ""
        save_name+="_chatgpt" if args.use_chatgpt else ""
        save_name+="_vision_only" if args.vision_only else ""
        save_name+="_model_summary_only" if args.model_summary_only else ""
        save_name+="_subtitles_only" if args.subtitles_only else ""
        save_name+="_info_only" if args.info_only else ""
        print("save_name",save_name)
        return save_name
    def eval_llama_vid(self):
        ## LLAMa vid QA evaluation
        full_questions_result=[]
        movie_number=0
        start=args.start
        end=args.end
        save_name=self.define_save_name()
        for movie in tqdm(self.movies_dict.keys()):
            if args.start <=movie_number < args.end:
                save_dir=f"new_workspace/results/llama_vid/{args.exp_name}/{save_name}_{args.neighbours}_neighbours"
                if os.path.exists( f"{save_dir}/{movie}.json" ):
                    print(f"Movie {movie} already processed")
                    with open(f"{save_dir}/{movie}.json", 'r') as f:
                        pred_json = json.load(f)
                    full_questions_result.extend(pred_json)
                    continue
                use_subtitles_while_generating_summary=not args.vision_only
                information_RAG_path,embedding_path=self.movie_inference(movie,use_subtitles_while_generating_summary)
                external_memory=MemoryIndex(args.neighbours, use_openai=args.use_openai_embedding)
                if os.path.exists(embedding_path):
                    external_memory.load_embeddings_from_pkl(embedding_path)
                else:
                    external_memory.load_documents_from_json(information_RAG_path,emdedding_path=embedding_path)
                save_dir=f"new_workspace/results/llama_vid/{args.exp_name}/{save_name}_{args.neighbours}_neighbours"
                os.makedirs(save_dir, exist_ok=True)
                pred_json=[]
                batch_questions=[]
                for qa in tqdm(self.movies_dict[movie],desc="Inference questions"):
                    batch_questions.append(qa)
                    if len(batch_questions)<args.batch_size:
                        continue
                    model_ans,related_text=self.answer_movie_questions_RAG(batch_questions,information_RAG_path,embedding_path)
                    for qa,ans,related_info in zip(batch_questions,model_ans,related_text):
                        qa.update({'pred':ans})
                        qa.update({'related_info':related_info})
                        pred_json.append(qa)
                    batch_questions=[]
                if len(batch_questions)>0:
                    model_ans,related_text=self.answer_movie_questions_RAG(batch_questions,information_RAG_path,embedding_path)
                    for qa,ans,related_info in zip(batch_questions,model_ans,related_text):
                        qa.update({'pred':ans})
                        qa.update({'related_info':related_info})
                        pred_json.append(qa)
                full_questions_result.extend(pred_json)
                with open(f"{save_dir}/{movie}.json", 'w') as fp:
                    json.dump(pred_json, fp)
                print(f"Movie {movie} prediction saved to {save_dir}/{movie}.json")
            movie_number+=1
        with open(f"{save_dir}/full_pred_s{start}_end{end}.json", 'w') as fp:
            json.dump(full_questions_result, fp)
args=get_arguments()

def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

import yaml 
# read this file test_configs/llama2_test_config.yaml
with open('test_configs/llama2_test_config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
seed=config['run']['seed']
print("seed",seed)

if __name__ == "__main__":
    setup_seeds(seed)
    llama_vid_eval=LlamaVidQAEval(args)
    llama_vid_eval.eval_llama_vid()