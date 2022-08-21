import sys
from typing import List, Tuple, Dict, Union, Optional, Callable, Any
import numpy
import numpy as np
#from scipy import linalg as sp_linalg
import math
import csv
import time
import copy
import random
import statistics
from tqdm import tqdm

from nats_bench import create
from nats_bench.api_topology import NATStopology
from nats_bench.api_utils import ArchResults

from cython_wl_kernel import cython_wl_kernel_ as wl_kernel

import sys
#sys.path.append('/home/rio-hada')
#import workspace.util.debug as my

DATASET = 'ImageNet' # とりあえず定数

# 未使用
ADJACENT_MATRIX: np.ndarray = np.array([
    [0, 1, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0],
], dtype='u1')
NEXT_NODES: List[List[int]] = [
    [1, 2, 4],
    [3, 5],
    [6],
    [6],
    [7],
    [7],
    [7],
    []
]

class HyperParam:
    def __init__(
            self,
            T: int,
            P: int,
            B: int,
            D: int,
            gpu: bool = False,
            recalc_freq: int = 1,
            k_size_max: Optional[int] = None,
            select_mode: str = 'random',
            mean0: bool = False,
            eval_length: Optional[int] = None,
            seed: Optional[int] = None,
            eval_freq_srcc: Optional[int] = None,
            eval_archs_srcc: Optional[int] = None
            ):
        self.T = T
        self.P = P
        self.B = B
        self.D = D
        self.gpu = gpu
        self.recalc_freq = recalc_freq 
        self.k_size_max = k_size_max
        self.select_mode = select_mode
        self.mean0 = mean0
        self.eval_length = eval_length
        self.seed = seed
        self.eval_freq_srcc = eval_freq_srcc
        self.eval_archs_srcc = eval_archs_srcc

class Cell:
    # (i, j) <=> (Node_i -> Node_j)
    
    OPS = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]
    OP_TO_INDEX: Dict[str, int] = dict(map(lambda kv: kv[::-1], enumerate(OPS)))
    
    def __init__(self, arch_str: str, accuracy: Dict[str, float], flops: Dict[str, float], index: int):
        self.arch_str = arch_str
        self.arch_matrix: np.ndarray = NATStopology.str2matrix(self.arch_str).astype('u1')
        self.accuracy = accuracy
        self.main_accuracy = accuracy[DATASET]
        self.flops = flops
        self.index = index
        self.label_list: List[int] = [
            0,
            self.arch_matrix[1, 0] + 1,
            self.arch_matrix[2, 0] + 1,
            self.arch_matrix[2, 1] + 1,
            self.arch_matrix[3, 0] + 1,
            self.arch_matrix[3, 1] + 1,
            self.arch_matrix[3, 2] + 1,
            6
        ]
        
    # 良くない指標なので変えたい
    #def get_avg_accuracy(self) -> float:
    #    return sum(self.accuracy.values()) / len(self.accuracy)
    
    def to_label_list(self) -> List[int]:
        ret = [-1] * 8
        ret[0] = 0
        ret[1] = self.arch_matrix[1, 0] + 1
        ret[2] = self.arch_matrix[2, 0] + 1
        ret[3] = self.arch_matrix[2, 1] + 1
        ret[4] = self.arch_matrix[3, 0] + 1
        ret[5] = self.arch_matrix[3, 1] + 1
        ret[6] = self.arch_matrix[3, 2] + 1
        ret[7] = 6
        return ret
    
    def __str__(self) -> str:
        return f'Cell({self.arch_str}, {self.arch_matrix}, {self.accuracy}, {self.flops}, {self.index})'
    
    def __repr__(self) -> str:
        return f'Cell(\'{self.arch_str}\', {self.arch_matrix}, {self.accuracy}, {self.flops}, {self.index})'
    
class NATSBenchWrapper:
    def __init__(self):
        self.cells: List[Cell] = []
            
    # アーカイブファイルからアーキテクチャの精度などを読み込む(低速)
    def load_from_archive(self, data_path: str) -> None:
        nats_bench: NATStopology = create(data_path, search_space='topology', fast_mode=True, verbose=False)
        self.num_archs: int = len(nats_bench)
        
        for i in tqdm(range(self.num_archs)):
            arch_results: ArchResults = nats_bench.query_by_index(i, hp='200')
            arch_str: str = arch_results.arch_str
            accuracy_dict = {}
            flops_dict = {} # 今は使っていない
            for dataset_key, dataset_name in [('cifar10-valid', 'cifar10'), ('cifar100', 'cifar100'), ('ImageNet16-120', 'ImageNet')]:
                # 精度(%)
                more_info = nats_bench.get_more_info(i, dataset_key, hp='200', is_random=False)
                accuracy: float = more_info['valid-accuracy']
                flops: float = arch_results.get_compute_costs(dataset_key)['flops']
                accuracy_dict[dataset_name] = accuracy
                flops_dict[dataset_name] = flops
            cell = Cell(arch_str, accuracy_dict, flops_dict, i)
            self.cells.append(cell)
            
    # csvファイルからアーキテクチャの精度などを読み込む(高速)
    def load_from_csv(self, csv_path: str) -> None:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            i = 0
            for dic in reader:
                dataset_keys = ['cifar10', 'cifar100', 'ImageNet']
                arch_str = dic['arch_str']
                accuracy, flops = {}, {}
                for dataset in dataset_keys:
                    accuracy[dataset] = float(dic[f'acc-{dataset}'])
                    flops[dataset] = float(dic[f'flops-{dataset}'])
                cell = Cell(arch_str, accuracy, flops, i)
                self.cells.append(cell)
                i += 1
        self.num_archs: int = len(self.cells)
    
    # アーキテクチャの精度をcsvファイルに保存
    def save_to_csv(self, csv_path: str) -> None:
        with open(csv_path, mode='w') as f:
            writer = csv.writer(f)
            writer.writerow([
                'arch_str', 
                'acc-cifar10', 'acc-cifar100', 'acc-ImageNet', 
                'flops-cifar10', 'flops-cifar100', 'flops-ImageNet'])
            for i, cell in enumerate(self.cells):
                writer.writerow([
                    cell.arch_str, 
                    cell.accuracy['cifar10'], 
                    cell.accuracy['cifar100'],
                    cell.accuracy['ImageNet'],
                    cell.flops['cifar10'], 
                    cell.flops['cifar100'],
                    cell.flops['ImageNet'],
                ])

    def __getitem__(self, key) -> Cell:
        return self.cells[key]
    
    def __len__(self) -> int:
        return self.num_archs
    
# 変数
wl_kernel_cache: Dict[Tuple[int, int], float] = {}
K_cache: np.ndarray = np.array([])
wl_kernel_time: float = 0
matrix_inv_time: float = 0
matrix_mult_time: float = 0
dropping_out_time: float = 0
K_inv_cache: np.ndarray = np.array([]) #
K_inv_cache_count: int = 0

''' 関数呼び出しのオーバーヘッドが大きいので未使用
def wl_kernel(cell1: Cell, cell2: Cell, H: int = 2) -> float:
    global wl_kernel_cache
    key = (cell1.index, cell2.index) if cell1.index < cell2.index else (cell2.index, cell1.index)
    if key in wl_kernel_cache:
        return wl_kernel_cache[key]
    wl_kernel_cache[key] = result = wl_kernel_(cell1.label_list, cell2.label_list, H)
    return result
'''

# 平均と分散を推定
# mu = k.T * K^-1 * y
# sigma^2 = kernel(x, x) - k.T * K_inv * k
def acquisition_gp_with_wl_kernel(
        x: Cell, 
        data: List[Cell], 
        K_inv: np.ndarray, # K^-1
        K_inv_y: np.ndarray, # K^-1 * y
        mean_acc: float
        #coeff: float
    ) -> Tuple[float, float]:
    
    global wl_kernel_cache
    global wl_kernel_time
    global matrix_mult_time
    
    t = len(data)
    
    # kernel(x, x)
    xx_kernel: float
    key = (x.index, x.index)
    if key in wl_kernel_cache:
        xx_kernel = wl_kernel_cache[key]# / coeff
    else:
        start_t = time.time()
        kernel_value = wl_kernel(x.label_list, x.label_list)
        xx_kernel = kernel_value# / coeff
        wl_kernel_cache[key] = kernel_value
        wl_kernel_time += time.time() - start_t
    
    k: np.ndarray = np.empty((t,))
    for i in range(t):
        c = data[i]
        key = (x.index, c.index) if x.index < c.index else (c.index, x.index)
        if key in wl_kernel_cache:
            k[i] = wl_kernel_cache[key]# / coeff
        else:
            start_t = time.time()
            kernel_value = wl_kernel(x.label_list, c.label_list)
            wl_kernel_time += time.time() - start_t
            k[i] = kernel_value# / coeff
            wl_kernel_cache[key] = kernel_value
    
    # 行列演算
    start_t = time.time()
    mu: np.ndarray = mean_acc + k @ K_inv_y
    k_K_inv: np.ndarray = k @ K_inv
    var: np.ndarray = xx_kernel - k_K_inv @ k.T
    matrix_mult_time += time.time() - start_t
    
    #
    '''
    if mu < -10:
        k_dot_K_inv_y: np.ndarray = k * K_inv_y
        k_dot_K_inv_y = np.array(sorted(k_dot_K_inv_y, key=lambda x: abs(x)))
        
        x_sum = 0
        for x in k_dot_K_inv_y:
            x_sum += x
                    
        print(f'K.shape[0] = {K_inv.shape[0]}')
        print('k =')
        print(k[:10])
        print('K_inv_y =')
        print(K_inv_y[:10])
        print('k * K_inv_y =')
        print((k * K_inv_y)[:10])
        print('sorted(k * K_inv_y) =')
        print(k_dot_K_inv_y[:10])
        print('mu =', mu)
        print('mu\' =', sum(k_dot_K_inv_y))
        print('x_sum =', x_sum)
        print('var =', var)
        print('')
    '''
    
    return mu, math.sqrt(max(var, 0))

def random_sampler(search_space: List[Cell], sample_indices: List[int], data: List[Cell], hparam: HyperParam):
    return random.sample(sample_indices, hparam.B)

def compose_K(data: List[Cell], t: int, B: int) -> np.ndarray:
    global wl_kernel_cache
    global wl_kernel_time
    global K_cache
    cached = False
    if K_cache.shape[0] == t - B:
        K = K_cache
        L = np.empty((t - B, B))
        LM = np.empty((B, t))
        K = np.concatenate([K, L], axis=1)
        K = np.concatenate([K, LM], axis=0)
        cached = True
    else:
        K: np.ndarray = np.empty((t, t))
    for i in range(t):
        j0 = i
        if cached and t - B > i: j0 = t - B
        for j in range(j0, t):
            c1, c2 = data[i], data[j]
            key = (c1.index, c2.index)
            if key in wl_kernel_cache:
                K[i, j] = K[j, i] = wl_kernel_cache[key]
            else:
                start_t = time.time()
                kernel_value = wl_kernel(c1.label_list, c2.label_list)
                wl_kernel_time += time.time() - start_t
                K[i, j] = K[j, i] = wl_kernel_cache[key] = kernel_value
    K_cache = K.copy()
    return K

def compose_K_inv(K: np.ndarray, t: int, B: int, is_dropped: bool, recalc_freq: int) -> np.ndarray:
    global K_inv_cache
    global K_inv_cache_count
    global K_INV_RECALC_FREQ
    global matrix_inv_time
    K_inv: np.ndarray
    cached = False
    start_t = time.time()  
    
    if not is_dropped and K_inv_cache.shape[0] == t:
        K_inv = K_inv_cache
        cached = True
    
    if K_inv_cache.shape[0] == t - B:
        if K_inv_cache_count < recalc_freq - 1:
            K_inv = reuse_inverse(K, K_inv_cache, t, B)
            K_inv_cache_count += 1
            cached = True
    if not cached:
        K_inv_cache_count = 0
        try:
            K_inv = np.linalg.inv(K)
        except:
            print(f'pinv: t = {t}', file=sys.stderr)
            K_inv = np.linalg.pinv(K)
    K_inv_cache = K_inv
    matrix_inv_time += time.time() - start_t
    return K_inv

# Kよりも一回り小さい行列の逆行列を利用して、Kの逆行列を計算
# 計算誤差が大きいので不採用
# 参考: https://ja.wikipedia.org/wiki/区分行列
def reuse_inverse(K: np.ndarray, K_inv_cache: np.ndarray, t: int, B: int) -> np.ndarray:
    K_t_1_inv: np.ndarray = K_inv_cache
    L: np.ndarray = K[:t - B, t - B:]
    M: np.ndarray = K[t - B:, t - B:]
    
    K_t_1_inv_L: np.ndarray = K_t_1_inv @ L
    S: np.ndarray = M - L.T @ K_t_1_inv_L
    S_inv: np.ndarray = np.linalg.inv(S)
    K_t_1_inv_L_S_inv: np.ndarray = K_t_1_inv_L @ S_inv
    
    K_inv: np.ndarray = np.empty((t, t))
    K_inv[:t - B, :t - B] = K_t_1_inv + K_t_1_inv_L_S_inv @ K_t_1_inv_L.T
    K_inv[:t - B, t - B:] = -K_t_1_inv_L_S_inv
    K_inv[t - B:, :t - B] = -K_t_1_inv_L_S_inv.T
    K_inv[t - B:, t - B:] = S_inv
    
    return K_inv

def gp_with_wl_kernel(
        search_space: List[Cell], 
        sample_indices: List[int], # search_spaceのインデックス
        data: List[Cell], 
        hparam: HyperParam
        ) -> List[Tuple[float, float]]:
    t = len(data) # Kのサイズ
    B = hparam.B
    
    global matrix_mult_time
    global dropping_out_time
    #global K_SIZE_MAX
    k_size_max: int = hparam.k_size_max if hparam.k_size_max != None else 20000
    
    musigma_tuples_list: List[List[Tuple[float, float]]] = []
    
    SELECTED_RATE = 0.99
    samples: int
    if t > k_size_max and hparam.select_mode == 'random':
        samples = math.ceil(math.log(1 - SELECTED_RATE) / math.log(1 - k_size_max / t))
    else:
        samples = 1
    
    K_base: np.ndarray = compose_K(data, t, B)
    if hparam.mean0:
        mean_acc = 0
    else:
        mean_acc = statistics.mean([data[i].main_accuracy for i in range(t)])
    y_base: np.ndarray = np.array([data[i].main_accuracy - mean_acc for i in range(t)]) 
    
    for n in range(samples):
        # Kの構成とキャッシュ化
        K = K_base # ファンシーインデックスはコピーが作成されるので、ビューの代入でOK
        y = y_base # ファンシーインデックスはコピーが作成されるので、ビューの代入でOK
        sub_data: List[Cell] = copy.copy(data)
        
        if t > k_size_max and hparam.select_mode == 'random':
            start_t = time.time()
            sorted_remaining_indices: np.ndarray = np.sort(np.random.choice(range(t), k_size_max, replace=False))
            
            K = K[sorted_remaining_indices, :][:, sorted_remaining_indices]
            y = y[sorted_remaining_indices]
            for i, remainig_index in enumerate(sorted_remaining_indices):
                sub_data[i], sub_data[remainig_index] = sub_data[remainig_index], sub_data[i]
            sub_data = sub_data[:k_size_max]
            dropping_out_time += time.time() - start_t
                
        # 逆行列
        K_inv: np.ndarray = compose_K_inv(K, t, B, t >= k_size_max, hparam.recalc_freq)
                               
        # 行列演算
        start_t = time.time()
        K_inv_y: np.ndarray = K_inv @ y # オリジナル
        #K_inv_y: np.ndarray = np.linalg.solve(K, y) # 誤差が少ない? srccはあまり変わらず
        matrix_mult_time += time.time() - start_t
        
        musigma_tuples: List[Tuple[float, float]] = []
    
        for sample_index in sample_indices:
            mu, sigma = acquisition_gp_with_wl_kernel(search_space[sample_index], sub_data, K_inv, K_inv_y, mean_acc)#, coeff)
            musigma_tuples.append((mu, sigma))
            
        musigma_tuples_list.append(musigma_tuples)
        
    # medianの効果検証
    std_of_mean_list = []
    std_of_std_list = []
    
    ret: List[Tuple[float, float]] = []
    for i in range(len(sample_indices)):
        mu = statistics.median([musigma_tuples_list[j][i][0] for j in range(samples)])
        sigma = statistics.median([musigma_tuples_list[j][i][1] for j in range(samples)])
        ret.append((mu, sigma))
        
        # medianの効果検証
        #if len(sample_indices) == 100 and t in [1100, 1600, 2100, 2600, 3100]:
            #mean_list = [musigma_tuples_list[j][i][0] for j in range(samples)]
            #std_list = [musigma_tuples_list[j][i][1] for j in range(samples)]
            #print(i, statistics.mean(mean_list), statistics.stdev(mean_list), statistics.median(mean_list), min(mean_list), max(mean_list))
            #std_of_mean_list.append(statistics.stdev(mean_list))
            #std_of_std_list.append(statistics.stdev(std_list))
            
    # medianの効果検証
    #print(t, len(sample_indices))
    #if len(sample_indices) == 100 and t in [1100, 1600, 2100, 2600, 3100]:#t == 3100:
        #print('mean')
        #print(t - 100, statistics.mean(std_of_mean_list))
        #print(statistics.stdev(std_of_mean_list))
        #print(statistics.median(std_of_mean_list))
        #print(min(std_of_mean_list))
        #print(max(std_of_mean_list))
        #print('std')
        #print(statistics.mean(std_of_std_list))
        #print(statistics.stdev(std_of_std_list))
        #print(statistics.median(std_of_std_list))
        #print(min(std_of_std_list))
        #print(max(std_of_std_list))

    return ret

def gp_with_wl_kernel_sampler(
        search_space: List[Cell], 
        sample_indices: List[int], # search_spaceのインデックス
        data: List[Cell], 
        hparam: HyperParam
        ) -> List[int]:
    itr = (len(data) - hparam.D) // hparam.B # イテレーション回数
    gamma = 3 * math.sqrt(1/2 * math.log(2 * (itr + 1)))
    
    musigma_tuples = gp_with_wl_kernel(search_space, sample_indices, data, hparam)
    index_musigma_tuples = list(zip(sample_indices, musigma_tuples))
    index_musigma_tuples = sorted(index_musigma_tuples, key=lambda x: x[1][0] + gamma * x[1][1], reverse=True)[:hparam.B]
    ret = [t[0] for t in index_musigma_tuples]
    return ret

def search(
    sampler: Callable[[Any], List[int]],
    wrapper: NATSBenchWrapper, data: List[Cell], search_space: List[Cell], 
    hparam: HyperParam
    ) -> List[float]:
    
    global wl_kernel_time
    global dropping_out_time

    for t in range(hparam.T):
        sample_indices: List[int] = random.sample(range(len(search_space)), hparam.P) # search_spaceのインデックス
        trained_indices: List[int] = sampler(search_space, sample_indices, data, hparam) # search_spaceのインデックス
        
        # データに追加
        for index in trained_indices:
            data.append(search_space[index])
            
            # 提案手法のうちの1つ
            # 類似性に基づいて行列サイズを抑え、過学習を抑制
            
            SIM_REVERSE = False # 類似度の低いものを取り除く場合
            SIM_DONT_CARE = False # ランダムに取り除く場合
            
            if hparam.select_mode == 'similarity' and hparam.k_size_max != None and len(data) > hparam.k_size_max:
                start_t = time.time()
                while len(data) > hparam.k_size_max:
                    
                    if SIM_REVERSE:
                        min_diff: float = 1000000
                        min_diff_indices: Tuple[int, int] = (-1, -1)
                        for i in range(len(data)):
                            for j in range(i + 1, len(data)):
                                # i < j の組み合わせ全て
                                c1, c2 = data[i], data[j]
                                key = (c1.index, c2.index)
                                if key in wl_kernel_cache:
                                    diff = wl_kernel_cache[key]
                                else:
                                    start_t1 = time.time()
                                    kernel_value = wl_kernel(c1.label_list, c2.label_list)
                                    wl_kernel_time += time.time() - start_t1
                                    dropping_out_time -= time.time() - start_t1
                                    diff = wl_kernel_cache[key] = kernel_value
                                diff += random.random() / 16
                                if diff < min_diff:
                                    min_diff = diff
                                    min_diff_indices = (i, j)
                        
                        sum_diff = [random.random() / 16, random.random() / 16]
                        for j in range(2):
                            for i in range(len(data)):
                                c1, c2 = data[min_diff_indices[j]], data[i]
                                key = (c1.index, c2.index) if c1.index < c2.index else (c2.index, c1.index)
                                if key in wl_kernel_cache:
                                    sum_diff[j] += wl_kernel_cache[key]
                                else:
                                    start_t1 = time.time()
                                    kernel_value = wl_kernel(c1.label_list, c2.label_list)
                                    wl_kernel_time += time.time() - start_t1
                                    dropping_out_time -= time.time() - start_t1
                                    wl_kernel_cache[key] = kernel_value
                                    sum_diff[j] += kernel_value
                        dropped_index = min_diff_indices[0] if sum_diff[0] < sum_diff[1] else min_diff_indices[1]
                        data.pop(dropped_index)
                        continue
                        
                        
                    if SIM_DONT_CARE:
                        data.pop(random.randrange(len(data)))
                        continue
                        
                    
                    max_diff: float = 0.
                    max_diff_indices: Tuple[int, int] = (-1, -1)
                    for i in range(len(data)):
                        for j in range(i + 1, len(data)):
                            # i < j の組み合わせ全て
                            c1, c2 = data[i], data[j]
                            key = (c1.index, c2.index)
                            if key in wl_kernel_cache:
                                diff = wl_kernel_cache[key]
                            else:
                                start_t1 = time.time()
                                kernel_value = wl_kernel(c1.label_list, c2.label_list)
                                wl_kernel_time += time.time() - start_t1
                                dropping_out_time -= time.time() - start_t1
                                diff = wl_kernel_cache[key] = kernel_value
                            diff += random.random() / 16
                            if diff > max_diff:
                                max_diff = diff
                                max_diff_indices = (i, j)
                    
                    sum_diff = [random.random() / 16, random.random() / 16]
                    for j in range(2):
                        for i in range(len(data)):
                            c1, c2 = data[max_diff_indices[j]], data[i]
                            key = (c1.index, c2.index) if c1.index < c2.index else (c2.index, c1.index)
                            if key in wl_kernel_cache:
                                sum_diff[j] += wl_kernel_cache[key]
                            else:
                                start_t1 = time.time()
                                kernel_value = wl_kernel(c1.label_list, c2.label_list)
                                wl_kernel_time += time.time() - start_t1
                                dropping_out_time -= time.time() - start_t1
                                wl_kernel_cache[key] = kernel_value
                                sum_diff[j] += kernel_value
                    dropped_index = max_diff_indices[0] if sum_diff[0] > sum_diff[1] else max_diff_indices[1]
                    data.pop(dropped_index)
                dropping_out_time += time.time() - start_t
            
        # 学習したインデックスを大きい順にソート
        trained_indices.sort(reverse=True)
        # search_spaceから学習したものを取り除く
        for index in trained_indices:
            search_space.pop(index)

    ret = sorted([cell.main_accuracy for cell in data[hparam.D:]], reverse=True) # これの計算時間は問題にならない
    return ret

def accuracy_compare(wrapper: NATSBenchWrapper, hparam: HyperParam) -> Dict[str, List[float]]:
    '''
    ランダムとWLカーネルのGPに対応
    '''
    
    hparam = copy.copy(hparam)
    
    if hparam.gpu:
        global np
        import cupy
        np = cupy    
    
    T = hparam.T
    hparam.T = 1
    
    random_results = []
    gpwl_results = []

    if hparam.seed is not None:
        random.seed(hparam.seed)
        
    random.shuffle(wrapper.cells)
    data = wrapper[:hparam.D]
    search_space = wrapper[hparam.D:]
    
    for t in range(T):
        r = search(random_sampler, wrapper, data, search_space, hparam)
        random_results.append(sum(r[:hparam.eval_length]) / len(r[:hparam.eval_length]))
    
    if hparam.seed is not None:
        random.seed(hparam.seed)
        
    random.shuffle(wrapper.cells)
    data = wrapper[:hparam.D]
    search_space = wrapper[hparam.D:]
    
    for t in range(T):
        r = search(gp_with_wl_kernel_sampler, wrapper, data, search_space, hparam)
        
        # 以下は上位hparam.eval_length個のアーキテクチャの平均精度を記録する場合のコード
        #gpwl_results.append(sum(r[:hparam.eval_length]) / len(r[:hparam.eval_length]))
        
        # 以下は、上位hparam.eval_length番目のアーキテクチャの精度を記録する場合のコード
        if len(r) >= hparam.eval_length:
            gpwl_results.append(r[hparam.eval_length - 1])
        else:
            gpwl_results.append(0)
        
        
    return {'Random': random_results, 'GP with WL-Kernel': gpwl_results}

def time_compare(wrapper: NATSBenchWrapper, hparam: HyperParam) -> Dict[str, np.ndarray]:
    '''
    WLカーネルのGP
    '''
    
    hparam = copy.copy(hparam)
    
    if hparam.gpu:
        global np
        import cupy
        np = cupy    
    
    T = hparam.T
    hparam.T = 1
    
    ret: Dict[str, List[float]] = {}
    keys = ['Total', 'WLKernel', 'MatrixMult', 'MatrixInv']
    if hparam.k_size_max != None:
        keys.append('DroppingOut')
    keys.append('Others')
    for key in keys:
        ret[key] = []
    
    if hparam.seed is not None:
        random.seed(hparam.seed)

    random.shuffle(wrapper.cells)
    data = wrapper[:hparam.D]
    search_space = wrapper[hparam.D:]
    
    for t in range(T):
        start_t = time.time()
        _ = search(gp_with_wl_kernel_sampler, wrapper, data, search_space, hparam)
        ret['Total'].append(time.time() - start_t)
        ret['WLKernel'].append(wl_kernel_time)
        ret['MatrixMult'].append(matrix_mult_time)
        ret['MatrixInv'].append(matrix_inv_time)
        if hparam.k_size_max != None:
            ret['DroppingOut'].append(dropping_out_time)
    
    for key in filter(lambda k: k != 'Others', keys):
        ret[key] = numpy.array(ret[key])
        
    ret['Total'] = numpy.cumsum(ret['Total'])
    
    ret['Others'] = ret['Total'].copy()
    for key in filter(lambda k: k != 'Total' and k != 'Others', keys):
        ret['Others'] -= ret[key]

    return ret

def get_ranks(array: List[float]) -> List[int]:
    tmp = np.array(array).argsort()
    ranks = np.empty_like(tmp)
    ranks[tmp] = np.arange(len(array))
    ranks = len(array) - ranks
    return ranks.tolist()

def spearman_rcc(values1: List[float], values2: List[float]) -> float:
    ranks1 = get_ranks(values1)
    ranks2 = get_ranks(values2)
    d2 = 0
    for rank1, rank2 in zip(ranks1, ranks2):
        d2 += (rank1 - rank2) ** 2
    N = len(values1)
    
    return 1 - 6 * d2 / (N * N * N - N)

def srcc_eval(wrapper: NATSBenchWrapper, hparam: HyperParam) -> Dict[str, numpy.ndarray]:
    '''
    ランダムとWLカーネルのGPに対応
    '''
    
    hparam = copy.copy(hparam)
    
    if hparam.gpu:
        global np
        import cupy
        np = cupy    
    
    T = hparam.T
    eval_freq: int = hparam.eval_freq_srcc
    eval_archs: int = hparam.eval_archs_srcc
    search_loops = T // eval_freq
    hparam.T = eval_freq
    
    srcc_list: numpy.ndarray = numpy.zeros((search_loops,))
    top_acc: numpy.ndarray = numpy.zeros((search_loops,))
    
    random.shuffle(wrapper.cells)
    data: List[Cell] = wrapper[:hparam.D]
    search_space: List[Cell] = wrapper[hparam.D:]
    
    for t in range(search_loops):
        _ = search(gp_with_wl_kernel_sampler, wrapper, data, search_space, hparam)
        
        # 探索空間からeval_archs個取り出す
        sample_indices: List[int] = random.sample(range(len(search_space)), eval_archs) # search_spaceのインデックス
        musigma_tuples = gp_with_wl_kernel(search_space, sample_indices, data, hparam)
        true_accs = [search_space[sample_index].main_accuracy for sample_index in sample_indices]
        pred_accs = [tp[0] for tp in musigma_tuples]
        srcc_list[t] = spearman_rcc(true_accs, pred_accs)
        
        list_of_tuple = sorted(zip(pred_accs, true_accs), reverse=True) # 精度が高そうな順に並び変え
        expected_accs = list(list(zip(*list_of_tuple))[1]) # 精度が高そうなもの順に，真の精度を並び替え
        acc = statistics.mean(expected_accs[:10]) #  精度が高そうなアーキテクチャ上位10個の真の精度の平均
        top_acc[t] = acc
    
    return {'srcc': srcc_list, 'acc': top_acc}
