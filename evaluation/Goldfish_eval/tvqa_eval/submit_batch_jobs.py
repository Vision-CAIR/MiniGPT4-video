import os 
import sys 

bash_script = 'RAG_summary.sh'
# bash_script = 'RAG.sh'

# general
start=0
end=850
step=60


# bash_script="RAG_summary_R_ablations.sh"
# sample 50 
# start=0
# end=52
# step=6


# job_id=32434597
for i in range(start, end, step):
    # print(i, i+step, job_id)
    # job_id+=1
    cmd=f'sbatch {bash_script} {str(i)} {str(i+step)}'
    os.system(cmd)