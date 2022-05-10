from typing import Callable, List, Optional, Tuple, Dict, Union, Any
from matplotlib import pyplot as plt
import numpy as np
import time
import random
from argparse import ArgumentParser
from collections import OrderedDict
from util import *

wrapper = NATSBenchWrapper()
wrapper.load_from_csv('data/tss.csv')

H = 2
LENGTH = 15625

wl_kernel_cache: np.ndarray = np.full((LENGTH, LENGTH), -1)
start_t = time.time()
for i in range(LENGTH):
    for j in range(LENGTH):
        cell1 = wrapper[i]
        cell2 = wrapper[j]
        if wl_kernel_cache[i][j] == -1:
            wl_kernel_cache[i][j] = wl_kernel(cell1, cell2, H)

with open('data/wl_kernel_H=2.txt', 'w') as f:
    wl_kernel_cache_list: List[List] = wl_kernel_cache.tolist()
    for i in range(LENGTH):
        line = list(map(lambda x: str(x), wl_kernel_cache_list[i][i:]))
        print(','.join(line), file=f)
            
print(f'Done {time.time() - start_t}')