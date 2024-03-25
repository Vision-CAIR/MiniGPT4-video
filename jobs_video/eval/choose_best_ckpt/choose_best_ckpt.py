import os 
import shutil 
ckpt_dir = '/ibex/ai/home/ataallka/Weights_folder/minigpt4_video/ckpt_mistral/cmd_webvid_video_instruct/202403171707'
print(f'number of ckpts: {len(os.listdir(ckpt_dir))}')
for ckpt in sorted(os.listdir(ckpt_dir)):
    if not ckpt.endswith('.pth'):
        continue
    ckpt_path = os.path.join(ckpt_dir,ckpt)
    job_name="cmd_webvid_video_instruct_"+ckpt.split(".")[0]
    # submit a job with this ckpt file
    os.system(f'sbatch ./evalualtion_ckpt.sh {ckpt_path} {job_name}')
    # print(f'sbatch ./evalualtion_ckpt.sh {ckpt_path} {job_name}')
    # print(f'job {job_name} submitted')
    # break
