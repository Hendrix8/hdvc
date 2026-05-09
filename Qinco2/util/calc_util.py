import numpy as np
import datetime
from scipy import stats
from util.dtype_util import to_iterable, is_iterable
# from numba import njit
from scipy.signal import welch
from copy import deepcopy


#待检查
#返回比val大的下一个整数，eq用来规定是否允许等于
def next_integer(val, eq):
    if float(val).is_integer():
        return int(val) if eq else int(val) + 1
    else:
        return int(np.ceil(val))


#待检查
#返回比val小的上一个整数，eq用来规定是否允许等于
def last_integer(val, eq):
    if float(val).is_integer():
        return int(val) if eq else int(val) - 1
    else:
        return int(np.floor(val))


#待检查
def toNpFloat32(val):
    return np.float32(val)


#待检查
def toNpInt32(val):
    return np.int32(val)


#待检查
def ceil2int(val):
    return int(np.ceil(val))


#待检查
def floor2int(val):
    return int(np.floor(val))


#待检查
def round2int(val):
    return int(np.round(val))


#待检查
def cumsum(arr, pad_0, dtype=float):
    if pad_0:
        return np.cumsum([0] + list(arr), dtype=dtype)
    else:
        return np.cumsum(arr, dtype=dtype)


# 待检查
def intersect1d(*all_arr):
    ret = all_arr[0]
    for arr in all_arr[1:]:
        ret = np.intersect1d(ret, np.array(arr))
    return ret


# 待检查
def union1d(*all_arr):
    ret = all_arr[0]
    for arr in all_arr[1:]:
        ret = np.union1d(ret, np.array(arr))
    return ret


# 待检查
def degree_to_radian(angles):
    return np.array(angles) * np.pi / 180


# 待检查
def radian_to_degree(angles):
    return np.array(angles) * 180 / np.pi


# 待检查
def circular_mean(angles, low, high, input_in_degree=True, output_in_degree=True):
    # return np.arctan2(
    #     np.sum(np.sin(angles)), np.sum(np.cos(angles))
    # )
    if low < -2 * np.pi or high > 2 * np.pi:
        print('Error! Must be bounded between -2 * pi to 2 * pi!')
        exit(1)
    input_ = degree_to_radian(angles) if input_in_degree else angles
    output = stats.circmean(input_, low=low, high=high)
    if output_in_degree:
        return radian_to_degree(output)
    return output


# 待检查
def select_kth(arr, k, min_k_is_0):
    if not min_k_is_0:
        k -= 1

    if k == 0:
        return np.min(arr)
    if k == len(arr):
        return np.max(arr)
    return np.partition(arr, k)[k]


# 待检查
def percentile(arr, p):
    if p > 1 or p < 0:
        print('Error! p must be in [0, 1]!')
        exit(1)

    n = len(arr)
    k = int(np.floor(1 / 3 + p * (n + 1 / 3)))
    if k < 1:
        return np.min(arr)
    if k + 1 > n:
        return np.max(arr)

    gamma = p * n - k + (p + 1) / 3
    return (1 - gamma) * select_kth(arr, k, False) + gamma * select_kth(arr, k + 1, False)


# 待检查
def var(sum1, sum2, num):
    return max(0, sum2 / num - (sum1 * sum1) / (num * num))


# 待检查
def std(sum1, sum2, num):
    return np.sqrt(var(sum1, sum2, num))


def time_difference_in_seconds(start_time, end_time, take_abs):

    '''
    :param start_time: start time in the format of (hour, min, sec)
    :param end_time: end time in the format of (hour, min, sec)
    :param take_abs: if True, return the absolute difference between the two timepoints
    :return: time difference in seconds

    Notes:
    1. This function assumes that the time format is in 24 (NOT 12) hours!
    2. If the input is in the format of (hour, min, sec), the two time points MUST be in the same day!
    '''

    if len(start_time) == len(end_time) == 3:
        start_seconds = (start_time[0] * 3600) + (start_time[1] * 60) + start_time[2]
        end_seconds = (end_time[0] * 3600) + (end_time[1] * 60) + end_time[2]
        ret = end_seconds - start_seconds
        return abs(ret) if take_abs else ret
    # if len(start_time) == len(end_time) == 6:  # 这个有问题！不能处理带小数点的情况！
    #     year1, month1, day1, hour1, minute1, second1 = start_time
    #     year2, month2, day2, hour2, minute2, second2 = end_time
    #     t1 = datetime.datetime(year1, month1, day1, hour1, minute1, second1)
    #     t2 = datetime.datetime(year2, month2, day2, hour2, minute2, second2)
    #     return (t2 - t1).total_seconds()
    raise Exception(f"Format error for {start_time} and {end_time}. They must have the same length of either 3 or 6.")


# 毫秒数 -> 点数; 如果msecs是array-like，那就返回int型的ndarray; 否则返回int
def msecs_to_npts(msecs, samp_rate):
    msecs_ = np.array(to_iterable(msecs))
    npts = np.rint(msecs_ * samp_rate / 1000).astype(int)
    if not is_iterable(msecs):
        npts = npts[0]
    return npts


# 秒数 -> 点数; 如果msecs是array-like，那就返回int型的ndarray; 否则返回int
def secs_to_npts(secs, samp_rate):
    secs_ = np.array(to_iterable(secs))
    npts = np.rint(secs_ * samp_rate).astype(int)
    if not is_iterable(secs):
        npts = npts[0]
    return npts


# 对给定的二维数组，分别对每一行做mean-norm
def mean_norm_2d_by_row(arr):
    mean = np.mean(arr, axis=1)
    return arr - mean[:, np.newaxis]


# 对给定的二维数组，分别对每一行做z-norm
def z_norm_2d_by_row(arr: np.ndarray):
    assert arr.ndim == 2
    mean = np.mean(arr, axis=1)
    std_ = np.std(arr, axis=1)
    std_[std_ == 0] = 1
    return (arr - mean[:, np.newaxis]) / std_[:, np.newaxis]


class KNNClassifier:
    def __init__(self, k=3, reduction='sum'):
        self.k = k
        self.train_data = None
        self.train_labels = None
        self.reduction='sum'

    def fit(self, train_data, train_labels):
        self.train_data = train_data
        self.train_labels = train_labels

    # @njit
    def euclidean_distance(self, x1: np.ndarray, x2: np.ndarray):
        assert x1.ndim == x2.ndim and x1.ndim in (1, 2)
        if x1.ndim == 1 or self.reduction == 'sum': # reduction是sum的情况下也应该这么算
            return np.sqrt(np.sum((x1 - x2) ** 2))
        else:
            raise NotImplementedError()

    # @staticmethod
    # # @njit
    # def euclidean_distance_ts_to_set(X1, x2):  # 这个在njit下反而比上面那个更慢，为什么？？？
    #     return np.sqrt(np.sum((X1 - x2) ** 2, axis=1))

    # @njit
    def predict(self, test_data):
        n_test = test_data.shape[0]
        predictions = np.zeros(n_test, dtype=self.train_labels.dtype)
        top_k_distances = np.zeros((n_test, self.k), dtype=np.float32)
        top_k_indices = np.zeros((n_test, self.k), dtype=int)

        for i in range(n_test):
            print(f'{i + 1} / {n_test}')
            distances = np.array([self.euclidean_distance(test_data[i], x) for x in self.train_data])
            # distances = self.euclidean_distance_ts_to_set(self.train_data, test_data[i])
            # assert (distances_1 == distances).all()
            sorted_indices = np.argsort(distances)
            k_indices = sorted_indices[:self.k]
            k_nearest_labels = self.train_labels[k_indices]

            top_k_distances[i] = distances[k_indices]
            top_k_indices[i] = k_indices
            unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
            predictions[i] = unique_labels[np.argmax(counts)]

        return predictions, top_k_distances, top_k_indices
    

# This only works for 1d array!
# https://stackoverflow.com/questions/30003068/how-to-get-a-list-of-all-indices-of-repeated-elements-in-a-numpy-array
def unique_with_all_indices_1d(array_1d):

    if array_1d.ndim != 1:
        raise Exception("unique_with_all_indices_1d only works for 1D ndarrays!")

    # creates an array of indices, sorted by unique element
    idx_sort = np.argsort(array_1d)

    # sorts records array so all unique elements are together
    sorted_array_1d = array_1d[idx_sort]

    # returns the unique values, the index of the first occurrence of a value, and the count for each element
    u_vals, idx_start, count = np.unique(sorted_array_1d, return_counts=True, return_index=True)

    # splits the indices into separate arrays
    indices = np.split(idx_sort, idx_start[1:])

    # # filter them with respect to their size, keeping only items occurring more than once
    # u_vals = u_vals[count > 1]
    # indices = filter(lambda x: x.size > 1, indices)

    return u_vals, indices

# bands = {'Delta (1-4 Hz)': (1, 4), 'Theta (4-8 Hz)': (4, 8), 'Alpha (8-12 Hz)': (8, 12),
#                               'Beta (12-30 Hz)': (12, 30), 'Gamma (30-45 Hz)': (30, 45)}


def get_psds_by_row(tss, samp_rate, nperseg=256, aggregate_by_eeg=False):
    freqs, psds_by_row = welch(tss, fs=samp_rate, nperseg=nperseg)
    # print(freqs)
    if aggregate_by_eeg:        # aggregate by the five eeg frequency bands
        bands = [(.5, 4), (4, 8), (8, 12), (12, 30), (30, 50)]
        n_bands = len(bands)

        inds = [
            np.where((freqs >= band[0]) & (freqs < band[1]))[0] for band in bands
        ]

        psds_by_row_ = deepcopy(psds_by_row)
        psds_by_row = np.empty((psds_by_row_.shape[0], n_bands))
        for i, cur_inds in enumerate(inds):
            # print(freqs[cur_inds])
            psds_by_row[:, i] = np.sum(psds_by_row_[:, cur_inds], axis=1)

        freqs = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]

    return freqs, psds_by_row