from typing import Callable, List, Optional, Tuple, Dict, Union, Any
from matplotlib import pyplot as plt
import numpy as np
import time
import random
from argparse import ArgumentParser
from util import *
import yaml

parser = ArgumentParser()
parser.add_argument('objective', type=str, choices=['acc', 'srcc', 'time'])
parser.add_argument('-T', type=int, required=True, help='loop count')
parser.add_argument('-P', type=int, default=16, help='pool size')
parser.add_argument('-B', type=int, default=2, help='batch size')
parser.add_argument('-D', type=int, default=100, help='initial size of data')
parser.add_argument('--recalc', type=int, default=1, help='recalc')
parser.add_argument('--k_size_max', type=int, help='K size max')
parser.add_argument('--select_mode', type=str, default='random', choices=['random', 'similarity'], help='random or similarity')
parser.add_argument('--mean0', action='store_true', help='let mean 0')
parser.add_argument('--eval_length', type=int, default=5, help='for \'acc\'')
parser.add_argument('--minT', type=int, default=1, help='min T for \'acc\'')
parser.add_argument('--maxT', type=int, help='max T for \'acc\'')
parser.add_argument('--seed', type=int, help='random seed')
parser.add_argument('--trials', type=int, default=1)
parser.add_argument('--name', type=str, help='name')
parser.add_argument('--id', type=int, help='id')
parser.add_argument('--gpu', action='store_true', help='use GPU')
parser.add_argument('--eval_freq', type=int, default=50, help='evaluation frequency for \'srcc\'')
parser.add_argument('--eval_archs', type=int, default=100, help='evaluated architectures for \'srcc\'')
parser.add_argument('--suppress_print_arrays', action='store_true', help='suppress to print arrays for \'time\'')

args = parser.parse_args()
print('args:', vars(args))

id = args.id if args.id != None else "unknownID"
save_path = f'result/image/{id}.png'
print(f'imagePath: {save_path}')

if args.maxT == None:
    args.maxT = args.T

wrapper = NATSBenchWrapper()
wrapper.load_from_csv('data/tss.csv')

T: int = args.T
P: int = args.P
B: int = args.B
D: int = args.D
eval_length: int = args.eval_length
trials: int = args.trials
eval_freq: int = args.eval_freq
eval_archs: int = args.eval_archs
hparam = HyperParam(
    T, P, B, D, 
    gpu=args.gpu, 
    recalc_freq=args.recalc,
    k_size_max=args.k_size_max, 
    select_mode=args.select_mode, 
    mean0=args.mean0, 
    eval_length=eval_length, 
    eval_freq_srcc=eval_freq, 
    eval_archs_srcc=eval_archs, 
    seed=args.seed
)

objective: str = args.objective

# アーキテクチャの精度を計測
def acc_task():
    results: Dict[str, np.ndarray] = {}
    stat_results: Dict[str, List[np.ndarray]] = {}
    print('timeTrial:')
    for trial in range(trials):
        start_t = time.time()
        seed = args.seed + trial if args.seed != None else None
        result = accuracy_compare(wrapper, hparam)
        for key, value in result.items():
            if key not in results:
                results[key] = np.zeros((len(value),))
                stat_results[key] = [None] * trials
            results[key] += np.array(value)
            stat_results[key][trial] = np.array(value)
        print(f'  - {time.time() - start_t}')
    for key, value in results.items():
        results[key] /= trials

    # stat_test とりあえずコメントアウト
    #for key, value in results.items():
    #    print(key)
    #    print(np.mean(stat_results[key], 0).tolist())
    #    print(np.std(stat_results[key], 0).tolist())
        
    plt.xlabel('T')
    plt.ylabel('accuracy (%) (avg of top10)')
    for key, value in results.items():
        plt.plot(range(args.minT, args.maxT + 1), value[args.minT - 1 : args.maxT], label=key)
    plt.legend()
    plt.savefig(save_path)

    results_arr: Dict[List[float]] = {}
    for key, arr in results.items():
        results_arr[key] = arr.tolist()
    print('result:', results_arr)

# スピアマンの順位相関係数を計測
def srcc_task():
    results_srcc: numpy.ndarray = numpy.zeros((T // eval_freq,))
    results_acc: numpy.ndarray = numpy.zeros((T // eval_freq,))
    print('timeTrial:')
    for trial in range(trials):
        start_t = time.time()
        seed = args.seed + trial if args.seed != None else None
        result = srcc_eval(wrapper, hparam)
        results_srcc += result['srcc']
        results_acc += result['acc']
        print(f'  - {time.time() - start_t}')
    results_srcc /= trials
    results_acc /= trials
    plt.xlabel('T')
    plt.ylabel('Spearman\'s rank correlation coefficient')
    plt.plot(range(eval_freq, T + 1, eval_freq), results_srcc)
    plt.savefig(save_path)

    print('result:', {'srcc': results_srcc.tolist(), 'acc': results_acc.tolist()})

# 実行時間を計測
def time_task():
    result: Dict[str, np.ndarray] = time_compare(wrapper, hparam)

    plt.xlabel('T')
    plt.ylabel('time (s)')

    for key, values in result.items():
        plt.plot(range(1, T + 1), values, label=key)

    plt.legend()

    for key, values in result.items():
        print('time' + key + ':', values[-1])
    if not args.suppress_print_arrays:
        results_arr = {}
        for key, arr in result.items():
            results_arr[key] = arr.tolist()
        print('result:', results_arr)

    plt.savefig(save_path)
    
if objective == 'acc':
    acc_task()
elif objective == 'srcc':
    srcc_task()
elif objective == 'time':
    time_task()
