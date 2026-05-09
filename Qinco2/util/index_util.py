import numpy as np
from copy import deepcopy


# 待检查
def ind_multi_to_1D(inds_by_dim, dim_sizes):
    """
    Convert multi-dimensional coordinates to 1D coordinate in flattened array.
    :param inds_by_dim: 【从外层到内层】存放的各个维度上的下标值
    :param dim_sizes: 【从外层到内层】存放的各个维度的尺寸，可以包括也可以不包括最外层的尺寸（因为没用）
    :return: 1D坐标
    """

    ndims, ndims_ = len(inds_by_dim), len(dim_sizes)
    rev_dim_sizes = deepcopy(dim_sizes[1:]) if ndims == ndims_ else deepcopy(dim_sizes)
    rev_inds_by_dim, rev_dim_sizes = inds_by_dim[::-1], rev_dim_sizes[::-1]  # 转成由内到外
    ndims -= 1  # 省一层

    ret, multiplier = rev_inds_by_dim[0], 1
    for i in range(ndims):
        ind, dim_size = rev_inds_by_dim[i + 1], rev_dim_sizes[i]
        multiplier *= dim_size
        ret += ind * multiplier
    return ret


# 待检查
def ind_1D_to_multi(ind_1D, dim_sizes):
    """
    Convert 1D coordinate in flattened array to multi-dimensional coordinates.
    :param ind_1D: 1D坐标
    :param dim_sizes: 【从外层到内层】存放的各个维度的尺寸，必须是完整的！(因为即使最内层没用，但是要通过dim_sizes判断ndims）
    :return: 【从外层到内层】存放的多维坐标。如果只有一维，返回标量值（即ind_1D自身）
    """

    ndims = len(dim_sizes)
    if ndims == 1:
        return np.array([ind_1D])

    dividends = np.empty(ndims - 1, dtype=int)  # 最外层不需要除
    dividends[-1] = dim_sizes[-1]
    for i in range(ndims - 2, 0, -1):  # 倒着来，不考虑最外层
        dividends[i - 1] = dividends[i] * dim_sizes[i]

    ret = np.empty(ndims, dtype=int)
    for i, dividend in enumerate(dividends):  # 这里要正着来
        ret[i] = ind_1D // dividend
        ind_1D %= dividend
    ret[-1] = ind_1D
    print(ret, '***')
    return ret

# all_dim_sizes = [
#     [3], [2, 3], [3, 4, 5, 3]
# ]
#
# all_inds_by_dim = [
#     [2], [1, 0], [1, 3, 4, 2]
# ]
#
# for i in range(len(all_dim_sizes)):
#     dim_sizes, inds_by_dim = all_dim_sizes[i], all_inds_by_dim[i]
#     ind_1D = ind_multi_to_1D(inds_by_dim, dim_sizes)
#     print(ind_1D, ind_1D_to_multi(ind_1D, dim_sizes))