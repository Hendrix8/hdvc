# -*- coding: UTF-8 -*-

import sys
sys.path.append('/lustre/fswork/projects/rech/thj/uth68ud/PycharmProjects/Qinco2')

# import pandas as pd
# from osgeo import gdal
import numpy as np
import sys
import re
import pickle
from collections import OrderedDict
from util.test_util import *
# from util.calc_util import intersect1d
import gc
import pathlib
import os
import shutil
import pandas as pd
import json
from typing import Union
import paramiko

class FileReader(object):

    @staticmethod
    def get_full_fname(fname, path=None):
        return fname if path is None else os.path.join(path, fname)

    # 待检查
    @staticmethod
    def read_file_to_str(fname, path=None):
        if path is not None:
            fname = os.path.join(path, fname)

        with open(fname, 'r') as f:
            ret = f.read()
        return ret


    # 改动，待检查
    @staticmethod
    def read_lines_from_file(fname, path=None, do_strip=True):

        #新增，待检查。由于fname是字符串，这里应该不会改变原先fname的值
        if path is not None:
            fname = os.path.join(path, fname)

        file = open(fname)
        lines = file.readlines()
        if do_strip:
            lines = [line.strip() for line in lines]
        file.close()
        return lines

    # 改动，待检查
    # 注意这个函数的局限性，强烈建议没把握的时候输出一下读入进来的东西
    @staticmethod
    def read_2dmat_from_file(fname, dtype, path=None, sep=r'[,;\s]\s*'): #这里默认的sep意思是，匹配形成如”A B“的所有字符串，其中A为逗号、分号、空白符，B为通配符，A和B之间有空白

        # 新增，待检查：直接粘下来的
        if path is not None:
            fname = os.path.join(path, fname)

        lines = FileReader.read_lines_from_file(fname)
        ret = []
        for line in lines:
            s_vals = re.split(sep, line.strip())
            s_vals = [s for s in s_vals if s != '']
            if dtype in (str, "string", "str"):
                pass
            elif dtype in (int, "integer", "int"):
                vals = list(map(int, s_vals))
            elif dtype in (float, "float"):
                vals = list(map(float, s_vals))
            else:
                print("Error! Dtype not supported!")
                exit(1)
            ret.append(vals)
        return np.array(ret)

    @staticmethod
    def load_pickle(fname, path=None):

        if path is not None:
            fname = os.path.join(path, fname)

        gc.disable()
        with open(fname, 'rb') as f:
            ret = pickle.load(f)
        gc.enable()
        return ret

    @staticmethod
    def load_dict_from_json(path):
        with open(path, 'r') as f:
            return json.load(f)

    @staticmethod
    def str_in_file(target_string: str, fname: str, path : str =None):
        if path is not None:
            fname = os.path.join(path, fname)
        if not os.path.exists(fname):
            return False

        with open(fname, 'r') as f:
            for line_number, line in enumerate(f, 1):
                if target_string in line:
                    return True

        return False


class PandasDFProcessor(object):

    @staticmethod
    def read_csv(fname, sep=",", path=None):
        fname = FileReader.get_full_fname(fname, path=path)
        try:
            return pd.read_csv(fname, sep=sep, engine="pyarrow")
        except:
            return pd.read_csv(fname, sep=sep)

    @staticmethod
    def df_to_numpy(df, dtype, copy=True):
        return df.to_numpy(dtype=dtype, copy=copy)

    @staticmethod
    def get_df_header(df):
        return df.columns.values.tolist()

    @staticmethod
    def get_df_content(df, dtype="str"):
        return PandasDFProcessor.df_to_numpy(df, dtype=dtype, copy=True)

    @staticmethod
    def get_df_header_and_content(df, content_dtype="str"):
        return PandasDFProcessor.get_df_header(df), PandasDFProcessor.get_df_content(df, dtype=content_dtype)

    @staticmethod
    def get_columns_by_name(df, column_names, dtype="str"):
        col_names = to_iterable(column_names)
        return PandasDFProcessor.df_to_numpy(df.loc[:, col_names], dtype=dtype, copy=True)


#检查过一遍，需检查第二遍
class FileWriter(object):

    # 待检查：如需使用请仔细检查
    # # 如果fname时None的话，直接打印到标准输出流
    # # mode：输出模式，默认是w
    # # 其他和print完全相同
    # @staticmethod
    # def write_to_file(*to_write, fname=None, mode='w', sep=' ', end='\n'):
    #     f = sys.stdout if fname is None else open(fname, mode)
    #     print(*to_write, sep=sep, end=end, file=f)
    #     if fname is not None:
    #         f.close()

    # 改动，待检查
    @staticmethod
    def write_vec_to_file(vec, fname=None, path=None, mode='w', sep=' ', dtype='float'):

        # 新增，待检查：直接粘下来的
        if path is not None:
            fname = os.path.join(path, fname)

        f = sys.stdout if fname is None else open(fname, mode)
        for v in vec:
            if dtype in ("str", "string", str):
                print(f'{v}{sep}', file=f, end='')
            elif dtype in ('int', 'integer', int):
                print(f'{int(v)}{sep}', file=f, end='')
            elif dtype in ('float', float):
                print(f'{float(v)}{sep}', file=f, end='')
            else:
                print('dtype not supported!')
                exit(1)
        print('\n', file=f, end='')
        if fname is not None:
            f.close()

    # 改动，待检查
    @staticmethod
    def write_2dmat_to_file(mat, fname=None, path=None, mode='w', sep=' ', dtype='float'):

        # 新增，待检查：直接粘下来的
        if path is not None:
            fname = os.path.join(path, fname)

        # 新增，待检查
        mat = np.array(mat)
        if mat.ndim == 1:
            FileWriter.write_vec_to_file(mat, fname=fname, path=path, mode=mode, sep=sep, dtype=dtype)
            return

        f = sys.stdout if fname is None else open(fname, mode)
        for vec in mat:
            for v in vec:
                if dtype in ("str", "string", str):
                    print(f'{v}{sep}', file=f, end='')
                elif dtype in ('int', 'integer', int):
                    print(f'{int(v)}{sep}', file=f, end='')
                elif dtype in ('float', float):
                    print(f'{float(v)}{sep}', file=f, end='')
                else:
                    print('dtype not supported!')
                    exit(1)
            print('\n', file=f, end='')
        if fname is not None:
            f.close()

    @staticmethod
    def save_dict_to_json(dict_: Union[dict, OrderedDict], path, excluded_keys=None):

        exclude = set(excluded_keys or [])
        filtered = {k: v for k, v in dict_.items() if k not in exclude}
        with open(path, 'w') as f:
            json.dump(filtered, f, indent=4)

    # 待检查
    @staticmethod
    def dump_pickle(obj, fname, path=None):

        # 新增，待检查：直接粘下来的
        if path is not None:
            fname = os.path.join(path, fname)

        gc.disable()
        with open(fname, 'wb') as f:
            pickle.dump(obj, f, protocol=-1)
        gc.enable()


class DirProcessor(object):

    @staticmethod
    def create_dir(path, recursive=True):
        path_ = pathlib.Path(path)
        path_.mkdir(parents=recursive, exist_ok=True)

    # https://stackoverflow.com/questions/185936/how-to-delete-the-contents-of-a-folder
    # 注意：这个连子文件夹的全部内容一起删除！
    @staticmethod
    def clear_dir(path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    @staticmethod
    def list_dir(path, list_full=False):
        if not list_full:
            return os.listdir(path)
        return [os.path.join(path, sub_path) for sub_path in os.listdir(path)]

    @staticmethod
    def list_remote_dir(remote_host, remote_path, remote_user_name, remote_password=None, list_full=False):
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(remote_host, username=remote_user_name, password=remote_password)

        sftp = client.open_sftp()
        sub_paths = sftp.listdir(remote_path)

        if list_full:
            sub_paths = [os.path.join(remote_path, sub_path) for sub_path in sub_paths]

        sftp.close()
        client.close()

        return sub_paths

    @staticmethod
    def recursive_get_all_fnames_under_path(path):

        """
        Get the absolute directories of all files under a path and all its subpaths (recursively)
        """
        return [str(p.resolve()) for p in pathlib.Path(path).rglob("*") if p.is_file()]



class FileNameProcessor(object):

    # 新增，待检查
    @staticmethod
    def var2str(var, primary_sep='_', secondary_sep='-'):

        simple_types = (str, int, float, bool, np.int32, np.int64, np.float32, np.float64)
        if type(var) in simple_types or var is None:
            return str(var)
        elif type(var) in (list, tuple, np.ndarray):
            for elm in var:
                if type(elm) not in simple_types or elm is None:
                    print(f'Error! Data type {type(elm)} of element {elm} currently not supported!')
                    exit(1)
            return primary_sep.join((FileNameProcessor.var2str(elm) for elm in var))
        elif type(var) == OrderedDict:  # 注意：不允许使用dict，否则无法判断顺序！
            return primary_sep.join((
                primary_sep.join(
                    (key,
                     secondary_sep.join((FileNameProcessor.var2str(elm) for elm in to_iterable(val, to_ndarry=False))))
                ) for key, val in var.items()
            ))
        else:
            print(f'Error! Data type {type(var)} currently not supported!')
            exit(1)

    #新增，待检查
    @staticmethod
    def create_fname(elements, ext=None, path=None, sep='_', primary_sep='_', secondary_sep='-'):
        fname = sep.join((FileNameProcessor.var2str(elm, primary_sep, secondary_sep)
                          for elm in elements if elm != ""))

        if ext is not None:
            if not ext.startswith('.'):
                ext = '.' + ext
            fname += ext
        if path is not None:
            fname = os.path.join(path, fname)
        return fname


if __name__ == '__main__':

    paths = [
        '/lustre/fsn1/projects/rech/thj/uth68ud/seeg_data/_non_SEANet_format_fully_supervised_all_ch',
        '/lustre/fsn1/projects/rech/thj/uth68ud/seeg_data/_non_SEANet_format_UAL_selection_results',
        '/lustre/fsn1/projects/rech/thj/uth68ud/seeg_data/_non_SEANet_format_Z2H_results',
        '/lustre/fsn1/projects/rech/thj/uth68ud/seeg_results/'
        ]
    for path in paths:
        fnames = DirProcessor.recursive_get_all_fnames_under_path(path)
        for fname in fnames:
            if fname.endswith('.pkl'):
                print(fname)
                FileReader.load_pickle(fname)

    # remote_host = 'andromache.mi.parisdescartes.fr'
    # remote_path = '/mnt/hddhelp/sliang/sEEG/puDistMat'
    # remote_user_name = 'sliang'
    # list_full = True
    #
    # sub_paths = DirProcessor.list_remote_dir(remote_host, remote_path, remote_user_name, list_full=list_full)
    # for sub_path in sub_paths:
    #     print(sub_path)
    


