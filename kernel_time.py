# 偏りが大きいので使えなさそう 

from typing import Callable, List, Optional, Tuple, Dict, Union, Any
import time
from argparse import ArgumentParser
from util import *

parser = ArgumentParser()
parser.add_argument('-N', type=int, required=True, help='how many kernels computed')

args = parser.parse_args()
print(args)

N: int = args.N

wrapper = NATSBenchWrapper()
wrapper.load_from_csv('data/tss.csv')

size = len(wrapper)

start_t = time.time()

for i in range(N):
    i = np.random.randint(size)
    j = np.random.randint(size)
    wl_kernel(wrapper[i], wrapper[j])
    
print(f'Done (normal) {time.time() - start_t}')

start_t = time.time()

for i in range(N):
    i = np.random.randint(size)
    wl_kernel(wrapper[i], wrapper[i])
    
print(f'Done (same) {time.time() - start_t}')

start_t = time.time()

for i in range(N):
    wl_kernel(wrapper[0], wrapper[0])
    
print(f'Done (0, 0) {time.time() - start_t}')