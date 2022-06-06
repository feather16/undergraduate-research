import sys
import os
import time
import requests
import json
import subprocess
from typing import List, Tuple, Dict, Union, Optional, Callable, Any
import yaml

WEBHOOK_URL = open(f'{os.path.dirname(__file__)}/WEBHOOK_URL').read()
USER_NAME = 'rio-hada'

# jobidのコマンドを取得
def get_command(jobid: int) -> str:
    cmd = 'Unknown'
    try:
        with open(f'result/log/out_{jobid}.yaml') as f:
            cmd = yaml.safe_load(f)['command']
    except:
        pass
    return cmd

# 標準エラーが出力されるファイルにエラーが含まれるかチェック
def check_error(jobid: int) -> str:
    error = 'Unknown'
    try:
        with open(f'result/log/err_{jobid}.out') as f:
            content = f.read().lower()
            if 'error' in content:
                error = 'True'
            else:
                error = 'False'
    except:
        pass
    return error

# squeue noticeをSlackに送信すべき時間かどうか
def is_squeue_notice_time() -> bool:
    tmp = int(time.time() % (24 * 3600))
    notice_hours = [0, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    return tmp % 3600 == 0 and ((tmp // 3600) + 9) % 24 in notice_hours
        
# squeueの内容をSlackにポストする
def send_squeue_message() -> None:
    squeue_out = subprocess.check_output('squeue').decode()
    text = f'```{squeue_out}```'
    icon = ':page_facing_up:'
    username = 'Squeue Notice'
    requests.post(WEBHOOK_URL, data=json.dumps({
        'text': text,
        'icon_emoji': icon,
        'username': username
    }))

# jobの内容をSlackにポストする
def send_job_message(jobid: int, t: int) -> None:
    time_format: str = ''
    cmd = 'Unknown'
    
    cmd = get_command(jobid)
    error = check_error(jobid)

    sec = t % 60
    min = (t // 60) % 60
    hour = t // 3600
    if hour > 0:
        time_format += f'{hour}h'
    if min > 0:
        time_format += f'{min}m'
    time_format += f'{sec}s'

    squeue_out: str = subprocess.check_output('squeue').decode()
    remaining_tasks = squeue_out.count(USER_NAME)

    text_list: List[str] = []
    text_list.append(f'Job *{jobid}* done in *{time_format}*.')
    if error == 'True':
        text_list.append(f'*Error!*')
    elif error == 'False':
        pass
    else:
        text_list.append(f'Error = *Unknown*')
        
    text_list.append(f'`{cmd}`')
    text_list.append(f'Remaining *{remaining_tasks}* tasks.')

    text = '\n'.join(text_list)
    icon = ':done:'
    username = 'Python Job Notice'

    requests.post(WEBHOOK_URL, data=json.dumps({
        'text': text,
        'icon_emoji': icon,
        'username': username
    }))
    
class Job:
    def __init__(self, tokens: List[str]):
        try:
            self.id = int(tokens[0])
            self.partition = tokens[1]
            self.name = ' ' .join(tokens[2:-5])
            self.user = tokens[-5]
            self.status = tokens[-4]
            self.time_s = tokens[-3]
            self.nodes = int(tokens[-2])
            self.nodelist = tokens[-1]
        except ValueError as e:
            print(f'tokens = {tokens}')
            print(e)
        
        self.time: int = 0
        try:
            tmp = self.time_s.split(':')
            if len(tmp) >= 3:
                tp = tuple(tmp[-3].split('-'))
                d = tp[-2] if len(tp) == 2 else 0
                h = tp[-1]
                self.time += 24 * 3600 * int(d) + 3600 * int(h)
            if len(tmp) >= 2:
                self.time += 60 * int(tmp[-2])
            self.time += int(tmp[-1])
        except:
            self.time = -1

old_dict: Dict[int, Job] = {}

while True:
    squeue_out = subprocess.check_output('squeue').decode()
    lines = squeue_out.split('\n')[1:]
    jobs: List[Job] = []
    for line in lines:
        tokens = line.split()
        if len(tokens) > 0:
            jobs.append(Job(tokens))
            
    new_dict: Dict[int, Job] = {}
    for job in jobs:
        if job.user == USER_NAME:
            new_dict[job.id] = job
            
    for jobid, job in old_dict.items():
        if jobid not in new_dict:
            print(f'Job {jobid} done.')
            send_job_message(jobid, job.time)
            
    old_dict = new_dict
    
    # squeueの内容を送信
    if is_squeue_notice_time():
        send_squeue_message()
        time.sleep(0.06)
    
    time.sleep(0.95)