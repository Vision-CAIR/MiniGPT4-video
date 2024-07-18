import os 
import sys 

bash_script = 'eval_model_summary_movie_qa.sh'
# bash_script = 'eval_q_related_info_movie_qa.sh'
start=0
end=30
step=4
for i in range(start, end, step):
    # print(i, i+step, job_id)
    # job_id+=1
    cmd=f'sbatch {bash_script} {str(i)} {str(i+step)}'
    # print(cmd)
    os.system(cmd)
    
    
