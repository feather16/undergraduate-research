from distutils import command
from nats_bench.api_utils import NASBenchMetaAPI
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter
from util import *
import yaml
plt.rcParams['font.family'] = 'WenQuanYi Micro Hei'

def dataset_accuracy_distribution():
    cifar10 = []
    cifar100 = []
    ImageNet = []

    with open('data/tss.csv') as f:
        reader = csv.DictReader(f)
        i = 0
        for dic in reader:
            cifar10.append(float(dic[f'acc-cifar10']))
            cifar100.append(float(dic[f'acc-cifar100']))
            ImageNet.append(float(dic[f'acc-ImageNet']))
            
    cifar10.sort()
    cifar100.sort()
    ImageNet.sort()        
            
    size = 15625
    cifar10 = cifar10[500:size-500]
    cifar100 = cifar100[500:size-500]
    #ImageNet = ImageNet[500:size-500]
    
    def plot_(data: list, dataset: str):
        plt.clf()
        plt.xlabel('Accuracy (%)')
        plt.ylabel('Count')
        plt.hist(data, bins = 64, label=dataset)
        plt.legend()
        plt.savefig(f'result/image/distribution_{dataset}.png')
                
    plot_(ImageNet, 'ImageNet')
    
    print(np.mean(cifar10), math.sqrt(np.var(cifar10)))
    print(np.mean(cifar100), math.sqrt(np.var(cifar100)))
    print(np.mean(ImageNet), math.sqrt(np.var(ImageNet)))

# 現在未使用
def wl_kernel_old_(cell1: List[int], cell2: List[int], H: int = 2) -> float:
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
  
def kernel_test(wrapper: NATSBenchWrapper):  
    for i in range(4):
        for j in range(4):
            c1 = wrapper[i].label_list
            c2 = wrapper[j].label_list
            k1 = wl_kernel(c1, c2)
            k2 = wl_kernel_old_(c1, c2)
            print(i, j, k1, k2)

wrapper = NATSBenchWrapper()
wrapper.load_from_csv('data/tss.csv')

def plot_simple(ids: list, n: int, key: str):
    res = []
    length = len(ids)
    
    for i in range(length):
        res.append(yaml.safe_load(open(f'result/log/out_{ids[i]}.yaml'))['result'][key])
    for i in range(length):
        values = [statistics.mean(res[i][max(j-n+1,0):j+1]) for j in range(len(res[i]))]
        plt.plot(range(1, len(values) + 1), values, label=f'{i}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig('result/image/tmp.png')
#plot_simple([8623, 8629, 8630], 8, 'srcc')

### ここまで確認済み

def load_label(command: str):
    tokens = command.split(' ')
    try:
        dmax = tokens[tokens.index('--k_size_max') + 1]
    except:
        return ''
    return r'$D_{max} = $' + dmax # 仮

def plot_srcc(id_to_label: dict, n: int):    
    log_dir = 'result/log'
    B = 2
    T = 3000
    eval_freq = 10
    
    srcc = []
    
    for id, label in id_to_label.items():
        yaml_dict = yaml.safe_load(open(f'{log_dir}/out_{id}.yaml'))
        srcc_values: list = yaml_dict['result']['srcc']
        command: str = yaml_dict['command']
        if label == '':
            id_to_label[id] = load_label(command)
        srcc_values = srcc_values[:(T // eval_freq)]
        srcc.append(srcc_values)
    
    for i, (id, label) in enumerate(id_to_label.items()):
        length = len(srcc[i])
        values = [statistics.mean(srcc[i][max(j-n+1,0):j+1]) for j in range(len(srcc[i]))]
        print(f'argmax = {np.argmax(values)}, max = {np.max(values)}, label = {label}')
        plt.plot(range(eval_freq * B, (length + 1) * eval_freq * B, eval_freq * B), values, label=label)
    plt.xlabel('訓練アーキテクチャ数')
    plt.ylabel('スピアマンの順位相関係数')
    plt.legend()
    plt.ylim(bottom=0.62, top=0.835)
    plt.savefig('result/image/tmp.png')

'''
plot_srcc({
    8621: '既存手法',
    8622: 'バギング',
    8623: '類似度の高いものを削除', 
    8629: '類似度の低いものを削除', 
    8630: 'ランダムに削除'
}, 8)
'''
    
def plot_srcc_old():
    id_to_label_random = {
        8292: r'$D_{max} = \infty$',
        8293: r'$D_{max} = 200$',
        8294: r'$D_{max} = 300$',
        8295: r'$D_{max} = 400$',
        8296: r'$D_{max} = 500$',
        8309: r'$D_{max} = 600$',
        8310: r'$D_{max} = 700$',
        8311: r'$D_{max} = 800$',
        8312: r'$D_{max} = 900$',
    }
    
    id_to_label_sim = {
        8292: r'$D_{max} = \infty$',
        8343: r'$D_{max} = 200$',
        8342: r'$D_{max} = 300$',
        8341: r'$D_{max} = 400$',
        8340: r'$D_{max} = 500$',
        8322: r'$D_{max} = 600$',
        8321: r'$D_{max} = 700$',
        8320: r'$D_{max} = 800$',
        8318: r'$D_{max} = 900$',
    }
    
    id_to_label = id_to_label_sim
    
    log_dir = 'result/log'
    n = 16
    B = 2
    T = 3000
    eval_freq = 10
    
    srcc = []
    
    for id in id_to_label.keys():
        dic = parse_out_file(f'{log_dir}/out_{id}.out')
        srcc_values = dic['srcc'][:(T // eval_freq)]
        srcc.append(srcc_values)
    
    for i, (id, label) in enumerate(id_to_label.items()):
        length = len(srcc[i])
        values = [statistics.mean(srcc[i][max(j-n+1,0):j+1]) for j in range(len(srcc[i]))]
        print(f'argmax = {np.argmax(values)}, max = {np.max(values)}, label = {label}')
        plt.plot(range(eval_freq * B, (length + 1) * eval_freq * B, eval_freq * B), values, label=label)
    plt.xlabel('Trained Architectures')
    plt.ylabel('Spearman\'s rank correlation coefficient')
    plt.legend()
    plt.ylim(bottom=0.65, top=0.835)
    plt.savefig('tmp.png')
    
def multi_plot_srcc_old():
    id_to_label_random: Dict[Tuple[int, int], str] = {
        (8027, 8292, 8396): r'$D_{max} = \infty$',
        (8038, 8293, 8397): r'$D_{max} = 200$',
        (8051, 8294, 8398): r'$D_{max} = 300$',
        (8078, 8295, 8399): r'$D_{max} = 400$',
        (8106, 8296, 8400): r'$D_{max} = 500$',
        (8113, 8309, 8401): r'$D_{max} = 600$',
        (8124, 8310, 8403): r'$D_{max} = 700$',
        (8151, 8311, 8404): r'$D_{max} = 800$',
        (8153, 8312, 8405): r'$D_{max} = 900$',
    }
    
    id_to_label_sim: Dict[Tuple[int, int], str] = {
        (8027, 8292, 8396): r'$D_{max} = \infty$',
        (8169, 8343, 8424): r'$D_{max} = 200$',
        (8180, 8342, 8423): r'$D_{max} = 300$',
        (8184, 8341, 8422): r'$D_{max} = 400$',
        (8185, 8340, 8421): r'$D_{max} = 500$',
        (8187, 8322, 8417): r'$D_{max} = 600$',
        (8188, 8321, 8416): r'$D_{max} = 700$',
        (8231, 8320, 8415): r'$D_{max} = 800$',
        (8255, 8318, 8407): r'$D_{max} = 900$',
    }
    
    id_to_label = id_to_label_random
    
    log_dir = 'result/log'
    n = 16
    B = 2
    T = 1500
    eval_freq = 10
    
    srcc: List[np.ndarray] = []
    
    for ids in id_to_label.keys():
        #ids = [ids[2]]
        srcc_values = np.zeros((T // eval_freq,))
        for id in ids:
            arr = parse_srcc(f'{log_dir}/out_{id}.out')
            srcc_values += arr[:(T // eval_freq)]
        srcc_values /= len(ids)
        srcc.append(srcc_values)
    
    for i, (ids, label) in enumerate(id_to_label.items()):
        length = len(srcc[i])
        values = [statistics.mean(srcc[i][max(j-n+1,0):j+1]) for j in range(len(srcc[i]))]
        print(f'argmax = {np.argmax(values)}, max = {np.max(values)}, label = {label}')
        plt.plot(range(eval_freq * B, (length + 1) * eval_freq * B, eval_freq * B), values, label=label)
    plt.xlabel('Trained Architectures')
    plt.ylabel('Spearman\'s rank correlation coefficient')
    plt.legend()
    #plt.ylim(bottom=0.675, top=0.836)
    plt.savefig('tmp.png')    
    
def plot_time_old():
    id_to_label_random = {
        8345: 'D_max = inf',
        8346: 'D_max = 200',
        8347: 'D_max = 300',
        8348: 'D_max = 400',
        8349: 'D_max = 500',
        8350: 'D_max = 600',
        8351: 'D_max = 700',
        8352: 'D_max = 800',
        8353: 'D_max = 900',
    }
    
    id_to_label_sim = {
        8345: 'D_max = inf',
        8354: 'D_max = 200',
        8355: 'D_max = 300',
        8356: 'D_max = 400',
        8357: 'D_max = 500',
        8358: 'D_max = 600',
        8359: 'D_max = 700',
        8360: 'D_max = 800',
        8361: 'D_max = 900',
    }
    
    id_to_label = id_to_label_random
    
    log_dir = 'result/log'
    B = 2
    T = 1500
    
    res = []
    
    for id in id_to_label.keys():
        res.append(parse_out_file(f'{log_dir}/out_{id}.out')['Total'][:T])
    
    for i, (id, label) in enumerate(id_to_label.items()):
        values = res[i]
        print(label, values[-1] / 60)
        plt.plot(range(B, (T + 1) * B, B), values, label=label)
    plt.xlabel('Trained Architectures')
    plt.ylabel('Time (s)')
    plt.legend()

    plt.savefig('tmp.png')
    
def plot_time_one_old():
    id = 7813
    log_dir = 'result/log'
    B = 2
    T = 3000
    
    res = parse_out_file(f'{log_dir}/out_{id}.out')
    
    for label, values in res.items():
        plt.plot(range(B, (T + 1) * B, B), values, label=label)
    plt.xlabel('Trained Architectures')
    plt.ylabel('Time (s)')
    plt.legend()

    plt.savefig('tmp.png')

method1 = yaml.safe_load(open('result/log/out_8626.yaml'))['result']['Total']
method2 = yaml.safe_load(open('result/log/out_8627.yaml'))['result']['Total']
#print(len(method1), len(method2), method1[-1], method2[-1])
for iter in [75, 150, 300, 600, 1500, 3000]:
    index = (iter + 1) // 2 - 1
    print('%4d %f %f' % (iter, method1[index], method2[index]))