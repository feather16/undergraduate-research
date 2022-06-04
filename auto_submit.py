import subprocess
import time

USER_NAME = 'rio-hada'

MAX_USE = 4

'''
sbatch spy srcc.py -T 1500 --trials 100 --eval_freq 10 --name base
sbatch spy srcc.py -T 1500 --trials 100 --k_size_max 200 --eval_freq 10
sbatch spy srcc.py -T 1500 --trials 100 --k_size_max 300 --eval_freq 10
sbatch spy srcc.py -T 1500 --trials 100 --k_size_max 400 --eval_freq 10
sbatch spy srcc.py -T 1500 --trials 100 --k_size_max 500 --eval_freq 10
'''

JOBS = '''
ls
'''

'''
sbatch spy srcc.py -T 1500 --trials 100 --k_size_max 600 --eval_freq 10
sbatch spy srcc.py -T 1500 --trials 100 --k_size_max 700 --eval_freq 10
sbatch spy srcc.py -T 1500 --trials 100 --k_size_max 800 --eval_freq 10
sbatch spy srcc.py -T 1500 --trials 100 --k_size_max 900 --eval_freq 10
sbatch spy srcc.py -T 1500 --trials 100 --k_size_max 900 --eval_freq 10 --select_mode similarity
sbatch spy srcc.py -T 1500 --trials 100 --k_size_max 800 --eval_freq 10 --select_mode similarity
sbatch spy srcc.py -T 1500 --trials 100 --k_size_max 700 --eval_freq 10 --select_mode similarity
sbatch spy srcc.py -T 1500 --trials 100 --k_size_max 600 --eval_freq 10 --select_mode similarity
sbatch spy srcc.py -T 1500 --trials 100 --k_size_max 500 --eval_freq 10 --select_mode similarity
sbatch spy srcc.py -T 1500 --trials 100 --k_size_max 400 --eval_freq 10 --select_mode similarity
sbatch spy srcc.py -T 1500 --trials 100 --k_size_max 300 --eval_freq 10 --select_mode similarity
sbatch spy srcc.py -T 1500 --trials 100 --k_size_max 200 --eval_freq 10 --select_mode similarity
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
    
    if count < MAX_USE:
        job = job_list.pop(0)
        subprocess.call(job.split())
        print(f'Remaining {len(job_list)} jobs.')
        if len(job_list) == 0:
            break
        
print('All jobs submitted.')