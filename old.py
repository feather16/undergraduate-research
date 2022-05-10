from collections import Counter
from util import *

'''  
Mに対するノイズ
for i in range(t - B, t):
    for j in range(i, t):
        noise = random.randint(-n_range, n_range) * delta
        K[i, j] += noise
        if i != j:
            K[j, i] += noise
            
Lに対するノイズ
for i in range(0, t - B):
    for j in range(t - B, t):
        noise = random.randint(-n_range, n_range) * delta
        K[i, j] += noise
        K[j, i] += noise
        
# ノイズを用いた手法
count = 0
det_K: float = np.linalg.det(K)
while not sys.float_info.min < abs(det_K):
    delta = 2 ** (-15)
    n_range = 2 ** 6
    noise: np.ndarray = np.random.randint(-n_range, n_range + 1, (K.shape)) * delta
    K += noise + noise.T
    count += 1
    det_K = np.linalg.det(K)
if count > 0:
    print(f'count = {count}, K.shape[0] = {K.shape[0]}')
    
#print(K_inv_cache.shape[0], t)

if USE_K_INV_CACHE and K_inv_cache.shape[0] == t - B: # 誤差が生じて上手くいかないので保留中
    K_t_1_inv: np.ndarray = K_inv_cache
    L: np.ndarray = K[:t - B, t - B:]
    M: np.ndarray = K[t - B:, t - B:]
    K_t_1_inv_L: np.ndarray = K_t_1_inv @ L
    S: np.ndarray = M - L.T @ K_t_1_inv_L
    S_inv: np.ndarray = np.linalg.inv(S)
    K_t_1_inv_L_S_inv: np.ndarray = K_t_1_inv_L @ S_inv
    K_inv_p1: np.ndarray = K_t_1_inv + K_t_1_inv_L_S_inv @ L.T @ K_t_1_inv # K_t_1_inv_Lを再利用した場合と比較
    K_inv_p2: np.ndarray = -K_t_1_inv_L_S_inv
    K_inv_p3: np.ndarray = K_inv_p2.T
    K_inv_p4: np.ndarray = S_inv
    K_inv_p3p4: np.ndarray = np.concatenate([K_inv_p3, K_inv_p4], axis=1)
    K_inv: np.ndarray = K_inv_p1
    K_inv = np.concatenate([K_inv, K_inv_p2], axis=1)
    K_inv = np.concatenate([K_inv, K_inv_p3p4], axis=0)
    
    # debug
    #print(f't = {t}, B = {B}')
    #print('@K_{t-1}')
    #print(K[:t-B,:t-B])
    #print('@K_t')
    #print(K)
    #print('@L');print(L)
    #print('@M');print(M)
    #print('@S');print(S)
    #print('@K_inv');print(K_inv)
    #print('@np.linalg.inv(K)');print(np.linalg.inv(K))
    #exit()
    
    # debug
    #print('M');print(M)
    #print('?');print(L.T @ K_t_1_inv_L)
    
    # debug
    #true_inv = np.linalg.inv(K)
    #print(f't = {t}, B = {B}')
    #print('part1');print(K_inv_p1);print(true_inv[:t-B,:t-B])
    #print('part2');print(K_inv_p2);print(true_inv[:t-B,t-B:])
    #print('part3');print(K_inv_p3);print(true_inv[t-B:,:t-B])
    #print('part4');print(K_inv_p4);print(true_inv[t-B:,t-B:])
    #print('')
    #if np.sum(np.abs(K_inv_p4 - true_inv[t-B:,t-B:])) > 1:
    #    exit()
elif K_inv_cache.shape[0] == t:
    K_inv = K_inv_cache
else:
    K_inv = np.linalg.inv(K)
'''
    
def cython_wl_kernel_old_(cell1: list, cell2: list, H: int = 2) -> float:
    '''
    cdef:
        list NEXT_NODES
        list cell_labels1, cell_labels2
        int h
        int i, j
        int s_index
        list nb_labels
        dict dic1, dic2
        set keys
    '''
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
    cell_labels1 = []
    cell_labels2 = []
    
    cell_labels1.extend(cell1)
    cell_labels2.extend(cell2)
    
    for h in range(H):
        s_index = 7 * h + 1 if h > 0 else 0
        
        for i, label in enumerate(cell_labels1[s_index:]):
            nb_labels = [cell_labels1[s_index + j] if j != 7 else cell_labels1[7] for j in NEXT_NODES[i]]
            nb_labels.sort()
            nb_labels.append(label)
            if len(nb_labels) >= 2:
                cell_labels1.append(tuple(nb_labels))
            
        for i, label in enumerate(cell_labels2[s_index:]):
            nb_labels = [cell_labels2[s_index + j] if j != 7 else cell_labels2[7] for j in NEXT_NODES[i]]
            nb_labels.sort()
            nb_labels.append(label)
            if len(nb_labels) >= 2:
                cell_labels2.append(tuple(nb_labels))

    dic1 = dict(Counter(cell_labels1))
    dic2 = dict(Counter(cell_labels2))
    
    keys = set(dic1.keys()) & set(dic2.keys())
    
    return sum(dic1[key] * dic2[key] for key in keys)

def wl_kernel_older_(cell1: List[int], cell2: List[int], H: int = 2) -> float:
    # 周囲のラベルを集める
    def collects(label_list: List[int]) -> List[List[int]]:
        labels = []
        for i, label in enumerate(label_list):
            neighbourhood_labels = [label_list[j] for j in Cell.NEXT_NODES[i]]
            neighbourhood_labels.sort()
            neighbourhood_labels.append(label) # 自身のラベルを後で追加(これはソート対象外)
            labels.append(neighbourhood_labels)
        return labels
    
    def relabel(labels1: List[List[int]], labels2: List[List[int]], start_new_label: int
            ) -> Tuple[List[int], List[int]]:

        new_label_dict: Dict[tuple, int] = {}
        new_label: int = start_new_label
        
        ret1: List[int] = [0] * len(labels1)
        ret2: List[int] = [0] * len(labels2)
        for i, label_set in enumerate(labels1):
            key = tuple(label_set)
            if len(label_set) == 1:
                new_label_dict[key] = label_set[0]
                ret1[i] = label_set[0]
            elif key not in new_label_dict:
                new_label_dict[key] = new_label
                ret1[i] = new_label
                new_label += 1
            else:
                ret1[i] = new_label_dict[key]
        for i, label_set in enumerate(labels2):
            key = tuple(label_set)
            if len(label_set) == 1:
                new_label_dict[key] = label_set[0]
                ret2[i] = label_set[0]
            elif key not in new_label_dict:
                new_label_dict[key] = new_label
                ret2[i] = new_label
                new_label += 1
            else:
                ret2[i] = new_label_dict[key]

        return ret1, ret2
    
    phi1 = [0] * (8 * 2 * (H + 1))
    phi2 = [0] * (8 * 2 * (H + 1))
    
    for label in cell1:
        phi1[label] += 1
    for label in cell2:
        phi2[label] += 1
    
    for h in range(H):
        label_max = max(cell1 + cell2)
        
        cell_labels_set1: List[List[int]] = collects(cell1)
        cell_labels_set2: List[List[int]] = collects(cell2)
        
        cell1, cell2 = relabel(cell_labels_set1, cell_labels_set2, label_max + 1)
               
        for label in cell1:
            if label > label_max:
                phi1[label] += 1
        for label in cell2:
            if label > label_max:
                phi2[label] += 1

    return float(np.dot(phi1, phi2))

def load_wl_kernel_cache(cache_file_path: str) -> None:
    '''
    空間計算量が大きすぎるので却下
    '''
    with open(cache_file_path) as f:
        i = 0
        for line in f:
            line = line.strip()
            values = [float(s) for s in line.split(',')]
            for j, value in enumerate(values):
                wl_kernel_cache[(i, i + j)] = value
            i += 1
            
def load_kernel_cache(path: str) -> None:
    '''
    対角成分のみ
    速くならないので却下
    '''
    with open(path) as f:
        lines = f.readlines()
    kernel_values = [float(line) for line in lines if len(line) > 0]
    global wl_kernel_cache
    for i, kernel_value in enumerate(kernel_values):
        wl_kernel_cache[(i, i)] = kernel_value
        
'''
def parse(text: str) -> Any:
    return eval(text.replace('array([', 'np.array(['))

def parse_file(file_path: str) -> Any:
    return parse(open(file_path).read())

def parse_out_file(file_path: str) -> Any:
    content = open(file_path).read()
    content = content[content.find('{') : content.rfind('}') + 1]
    return parse(content)

def parse_ssv_list(text: str) -> List[float]:
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = text[1:-1] # []を削除
    return [float(s) for s in text.split()]
    
def parse_ssv_list_from_out_file(file_path: str) -> List[float]:
    content = open(file_path).read()
    content = content[content.find('[') : content.rfind(']') + 1]
    return parse_ssv_list(content)
'''