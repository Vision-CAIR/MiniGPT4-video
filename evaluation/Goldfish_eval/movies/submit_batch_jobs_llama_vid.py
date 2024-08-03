import os

# bash_script = 'eval_q_related_info_llama_vid.sh'

bash_script = 'eval_model_summary_llama_vid.sh'
start=0
end=45
step=11
for i in range(start, end, step):
    # print(i, i+step, job_id)
    # job_id+=1
    cmd=f'sbatch {bash_script} {str(i)} {str(i+step)}'
    # print(cmd)
    os.system(cmd)