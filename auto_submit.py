import subprocess
import time

USER_NAME = 'rio-hada'

MAX_USE = 8 # ノードを最大で何個使用するか

JOBS = '''
sbatch run nasbowl2.py acc -T 1500 --trials 100 --eval_length 1 --name base
sbatch run nasbowl2.py acc -T 1500 --trials 100 --eval_length 1 --k_size_max 800
sbatch run nasbowl2.py acc -T 1500 --trials 100 --eval_length 1 --name base
sbatch run nasbowl2.py acc -T 1500 --trials 100 --eval_length 1 --k_size_max 800
'''


job_list = [j for j in JOBS.split('\n') if len(j) > 0]

while True:
    time.sleep(5)
    
    squeue_out = subprocess.check_output('squeue').decode()
    lines = squeue_out.split('\n')[1:]
    
    count = 0
    try:
        for line in lines:
            tokens = line.split()
            if len(tokens) >= 8:
                if tokens[-5] == USER_NAME:
                    count += int(tokens[-2])
    except:
        print(f'Can\'t parse \'line\'')
        continue
    
    if count < MAX_USE: # 空いているノードが十分にあれば
        job = job_list.pop(0)
        subprocess.call(job.split())
        print(f'Remaining {len(job_list)} jobs.')
        if len(job_list) == 0:
            break
        
print('All jobs submitted.')