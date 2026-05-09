import numpy as np
import torch
from typing import Union
from copy import deepcopy


#待检查
def rand_uniform(low, high, size=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(low, high, size)


#待检查
def rand_normal(avg, stdv, size=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return np.random.normal(avg, stdv, size)


def shuffle_array(arr: Union[np.ndarray, torch.Tensor], ret_inds: bool = True, seed=None):
    if seed is not None:
        np.random.seed(seed)

    shuffled_inds = np.random.permutation(len(arr)) # np.random.shuffle only works along the 1st axis
    if not ret_inds:
        return deepcopy(arr[shuffled_inds])
    return deepcopy(arr[shuffled_inds]), shuffled_inds
