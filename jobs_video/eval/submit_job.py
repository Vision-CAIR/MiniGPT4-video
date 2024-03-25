import os 
import shutil
import sys

start=0
end=7650
step=1000

for i in range(start,end,step):
    cmd=f'sbatch ./mistral_evalualtion.sh {i} {i+step}'
    # print(cmd)
    os.system(cmd)