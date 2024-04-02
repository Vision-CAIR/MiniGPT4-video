import os 
import shutil
import sys

start=0
end=7800
step=800

# Mistral
for i in range(start,end,step):
    cmd=f'sbatch ./mistral_evalualtion.sh {i} {i+step}'
    # print(cmd)
    os.system(cmd)
  
# Llama 2  
# for i in range(start,end,step):
#     cmd=f'sbatch ./llama2_evalualtion.sh {i} {i+step}'
#     # print(cmd)
#     os.system(cmd)