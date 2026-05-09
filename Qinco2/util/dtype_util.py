from collections import OrderedDict
import numpy as np
# import torch


# Bi Rico. https://stackoverflow.com/questions/10016352/convert-numpy-array-to-tuple
def to_tuple(a):
    try:
        return tuple(to_tuple(i) for i in a)
    except TypeError:
        return a


def is_iterable(obj, exclude_str=True, exclude_dict=True, exclude_tuple=False):
    try:
        iter(obj)
        ex_flg = (exclude_str and type(obj) == str) or \
                 (exclude_dict and type(obj) in (dict, OrderedDict)) or \
                 (exclude_tuple and type(obj) == tuple)
        return not ex_flg
    except:
        return False


def to_iterable(obj, exclude_str=True, exclude_dict=True, exclude_tuple=False, to_ndarray=True):
    return obj if is_iterable(obj, exclude_str=exclude_str, exclude_dict=exclude_dict,
                              exclude_tuple=exclude_tuple) else (np.array([obj]) if to_ndarray else [obj])


# def np2tensor(ndarray, device=None, to_float=True):
#     if not torch.cuda.is_available():
#         device = torch.device('cpu')
#     elif device is None:
#         device = torch.device("cuda:0")
#     elif isinstance(device, str):
#         device = torch.device(device)
#     elif isinstance(device, torch.device):
#         pass
#     else:
#         raise ValueError(f'device = {device} is not supported!')
#
#     return torch.from_numpy(ndarray).to(device) if not to_float \
#         else torch.from_numpy(ndarray).float().to(device)
#
#
# def tensor2np(tensor):
#     return tensor.detach().cpu().numpy().astype(np.float32)

# def mne_single_raw_to_epochs(raw, ret_type, event_id=1):
#     """
#     :param raw:
#     :param ret_type: either "Epochs" or "EpochsArray"
#     :param event_id:
#     :return:
#     """
#     if ret_type == "EpochsArray":
#         return mne.EpochsArray([raw.get_data()], raw.info, events=[[0, 0, event_id]])
#     elif ret_type == "Epochs":
#         raise Exception("Error: Raw to Epochs, rather than EpochsArray, is yet to be implemented.")
#     else:
#         raise Exception(f"Error: ret_type {ret_type} is not supported.")
#
#
# def mne_multi_raw_to_epochs(raws, event_times, ret_type, event_ids=None):
#
#     """
#     :param raws:
#     :param event_times:
#     :param ret_type: either "Epochs" or "EpochsArray"
#     :param event_ids: if None, event_ids is set to list(range(len(raws)))
#     :return:
#     """
#
#     if event_ids is None:
#         event_ids = list(range(len(raws)))
#     events = [[event_time, 0, event_id] for event_time, event_id in zip(event_times, event_ids)]
#
#     if ret_type == "EpochsArray":
#         return mne.EpochsArray([raw.get_data() for raw in raws], raws[0].info, events=events)
#     elif ret_type == "Epochs":
#         raise Exception("Error: Raws to Epochs, rather than EpochsArray, is yet to be implemented.")
#     else:
#         raise Exception(f"Error: ret_type {ret_type} is not supported.")

