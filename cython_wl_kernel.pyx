from typing import List, Tuple, Dict, Union, Optional, Callable, Any
from collections import Counter

from libcpp.vector cimport vector

cdef extern from "wl_kernel.hpp":
    cdef int wl_kernel_c_(vector[int] cell1, vector[int] cell2, int H)

def cython_wl_kernel_(list cell1, list cell2, int H = 2) -> float:
    return float(wl_kernel_c_(cell1, cell2, H))

def cython_wl_kernel_written_in_cython(list cell1, list cell2, int H = 2) -> float:
    cdef:
        vector[vector[int]] NEXT_NODES = [
            [1, 2, 4],
            [3, 5],
            [6],
            [6],
            [7],
            [7],
            [7],
            []
        ]
        vector[int] N_NEXT_NODES = [3, 2, 1, 1, 1, 1, 1];
        str LABEL_MAX_S = '6'
        int N_NODES = 8
        
        list cell_labels1 = []
        list cell_labels2 = []

        int h, s_index

        list nb_labels
        str s

        int ret
        dict counter1 = {}

    for i in range(N_NODES - 1):
        cell_labels1.append(str(cell1[i]))
        cell_labels2.append(str(cell2[i]))

    for h in range(H):
        s_index = (N_NODES - 1) * h

        for i in range(N_NODES - 1):
            nb_labels = []
            for j in NEXT_NODES[i]:
                nb_labels.append(cell_labels1[s_index + j] if j != N_NODES - 1 else LABEL_MAX_S)
            nb_labels.sort()
            s = '['
            for k in range(N_NEXT_NODES[i]):
                s += nb_labels[k]
            s += cell_labels1[s_index + i]
            s += ']'
            cell_labels1.append(s)

        for i in range(N_NODES - 1):
            nb_labels = []
            for j in NEXT_NODES[i]:
                nb_labels.append(cell_labels2[s_index + j] if j != N_NODES - 1 else LABEL_MAX_S)
            nb_labels.sort()
            s = '['
            for k in range(N_NEXT_NODES[i]):
                s += nb_labels[k]
            s += cell_labels2[s_index + i]
            s += ']'
            cell_labels2.append(s)

    ret = 1

    #print('@1')
    #print(cell_labels1)
    #print('@2')
    #print(cell_labels2)
    #print('')

    for label in cell_labels1:
        if label in counter1:
            counter1[label] += 1
        else:
            counter1[label] = 1
    for label in cell_labels2:
        if label in counter1:
            ret += counter1[label]
            #print('+', ret, counter1[label], label)
    
    return float(ret) / 2