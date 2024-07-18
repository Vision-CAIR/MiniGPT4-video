import os

bash_script = 'eval_q_related_info_movie_chat.sh'

# bash_script = 'eval_model_summary_movie_chat.sh'
start=0
end=101
step=26
for i in range(start, end, step):
    # print(i, i+step, job_id)
    # job_id+=1
    cmd=f'sbatch {bash_script} {str(i)} {str(i+step)}'
    # print(cmd)
    os.system(cmd)