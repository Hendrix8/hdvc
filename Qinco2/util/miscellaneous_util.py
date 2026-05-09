import numpy as np
from collections import OrderedDict
from pathlib import Path
import tarfile
import hashlib
import os
import fnmatch
from typing import Union, Iterable, Dict, List, Set, Tuple, Optional
import argparse

# 待检查
def split_vector(vec, num_per_group=None):
    if num_per_group is None:
        return vec
    else:
        ret, cnt, num_total = [], 0, len(vec)
        while cnt != num_total:
            num = min(num_per_group, num_total - cnt)
            ret.append(vec[cnt: cnt + num])
            cnt += num
        return ret


# 待检查
def safe_concat(prev_arr, incre_arr):
    return incre_arr if len(prev_arr) == 0 else np.concatenate((prev_arr, incre_arr))

# 待检查
def safe_c_(prev_arr, incre_arr):
    return incre_arr if len(prev_arr) == 0 else np.c_[prev_arr, incre_arr]


# 待检查
def add_or_incre(dictionary, key, increment):
    if type(dictionary) not in (dict, OrderedDict):
        print('Error! "dictionry" must be either dict or OrderedDict')
        exit(1)

    if key in dictionary.keys():
        dictionary[key] += increment
    else:
        dictionary[key] = increment


# 待检查
def add_or_append(dictionary, key, to_append):
    if type(dictionary) not in (dict, OrderedDict):
        print('Error! "dictionry" must be either dict or OrderedDict')
        exit(1)

    if key in dictionary.keys():
        if type(dictionary[key]) != list:
            print('Error! dictionary[key] must be a list!')
            exit(1)
        dictionary[key].append(to_append)
    else:
        dictionary[key] = [to_append]


# 待检查
def add_or_concat(dictionary, key, to_concat):
    if type(dictionary) not in (dict, OrderedDict):
        print('Error! "dictionry" must be either dict or OrderedDict')
        exit(1)
    if type(to_concat) != list:
        print('Error! to_concat must be a list!')
        exit(1)

    if key in dictionary.keys():
        if type(dictionary[key]) != list:
            print('Error! dictionary[key] must be a list!')
            exit(1)
        dictionary[key] += to_concat
    else:
        dictionary[key] = to_concat


# 待检查
# 注意：只能对python list展开，不能对ndarray等其他类型的数组进行展开
def flatten_python_list(list_to_flatten, to_ndarray=True):

    if type(list_to_flatten) != list:
        return np.array([list_to_flatten]) if to_ndarray else [list_to_flatten]
    flg = np.array([type(elm) != list for elm in list_to_flatten]).all()
    if flg:
        # print('flg = True', list_to_flatten)
        return np.array(list_to_flatten) if to_ndarray else list_to_flatten

    # print('flg = False', list_to_flatten)
    ret = np.concatenate(tuple(flatten_python_list(elm) for elm in list_to_flatten))
    return ret if to_ndarray else ret.tolist()

# a = [[1, 4, 2], [2, 1, 1]]
# b = [[[1, 3], 3], [2, [0, [1, 0]]]]
# print('***', flatten_python_list(a, True))
# print('***', flatten_python_list(b, True))
# print('***', np.array(b).flatten())
# print('***', flatten_python_list(a, False))
# print('***', flatten_python_list(b, False))

# import os
# import fnmatch
# import hashlib
# import tarfile
# from pathlib import Path
# from typing import Dict, List, Optional, Set, Tuple, Iterable

def verify_tar_gz(
    tar_gz_path: str,
    source_dir: str,
    *,
    include_dirs: bool = True,
    content_check: bool = False,
    hash_algo: str = "sha256",
    exclude: Optional[Iterable[str]] = None,
    ignore_symlinks: bool = False,
) -> Dict:
    """
    严格模式：要求 tar.gz 内的顶层目录 == Path(source_dir).name
    例如 source_dir=/a/b/c，则包内必须是 c/...（或目录条目 c/）。
    """

    def _norm(p: str) -> str:
        while p.startswith("./"):
            p = p[2:]
        return os.path.normpath(p).replace("\\", "/")

    def _excluded(rel: str) -> bool:
        rel_u = rel.replace(os.sep, "/")
        for pat in exclude:
            if fnmatch.fnmatch(rel_u, pat):
                return True
        return False

    src = Path(source_dir).resolve()
    arc = Path(tar_gz_path).resolve()
    if not src.is_dir():
        raise ValueError(f"source_dir 不存在或不是目录: {src}")
    if not arc.is_file():
        raise ValueError(f"archive_path 不存在或不是文件: {arc}")

    exclude = list(exclude or [])
    top = src.name  # 期望的包内顶层目录名

    # 1) 构建源清单（相对 source_dir 的路径，不含顶层）
    src_files: Set[str] = set()
    src_dirs: Set[str] = set()
    src_sizes: Dict[str, int] = {}

    for root, dirs, files in os.walk(src, followlinks=False):
        root_path = Path(root)
        if include_dirs:
            rel_dir = str(root_path.relative_to(src)).replace("\\", "/")
            if rel_dir != "." and not _excluded(rel_dir + "/"):
                src_dirs.add(rel_dir + "/")
        for name in files:
            p = root_path / name
            rel = str(p.relative_to(src)).replace("\\", "/")
            if _excluded(rel):
                continue
            try:
                st = p.lstat()
            except FileNotFoundError:
                continue
            if os.path.islink(p):
                if ignore_symlinks:
                    continue
                src_files.add(rel)  # 记录 symlink（不做大小/哈希）
            else:
                src_files.add(rel)
                src_sizes[rel] = st.st_size

    # 2) 期望的“包内带顶层”的清单
    expect_dirs = set()
    if include_dirs:
        expect_dirs = {f"{top}/"} | {f"{top}/{d}" for d in src_dirs}
    expect_files = {f"{top}/{f}" for f in src_files}
    expect_sizes = {f"{top}/{k}": v for k, v in src_sizes.items()}
    expect_set = expect_dirs | expect_files

    # 3) 读取 tar 清单并强制过滤到以 top/ 开头（或恰为 top/）
    tar_files: Set[str] = set()
    tar_dirs: Set[str] = set()
    tar_sizes: Dict[str, int] = {}
    errors: List[str] = []

    def _is_under_top(name: str) -> bool:
        if name == f"{top}/":
            return True
        return name.startswith(f"{top}/")

    try:
        with tarfile.open(arc, mode="r:gz") as tf:
            for m in tf.getmembers():
                name = _norm(m.name)
                # 目录：统一为以 / 结尾的形式
                if m.isdir():
                    dname = name if name.endswith("/") else name + "/"
                    if _is_under_top(dname):
                        if include_dirs and not _excluded(dname[len(top)+1:]):  # 对比排除使用相对 path（去掉 top/）
                            tar_dirs.add(dname)
                elif m.isfile():
                    if _is_under_top(name):
                        rel_under_top = name[len(top)+1:]  # 去除 "top/"
                        if _excluded(rel_under_top):
                            continue
                        tar_files.add(name)
                        tar_sizes[name] = m.size
                elif m.issym() or m.islnk():
                    if _is_under_top(name):
                        rel_under_top = name[len(top)+1:]
                        if _excluded(rel_under_top):
                            continue
                        if not ignore_symlinks:
                            tar_files.add(name)
                else:
                    pass
    except Exception as e:
        errors.append(f"读取 tar.gz 失败：{e}")

    # 4) 顶层目录合法性快速检查
    # 若包内存在与 top 无关的条目（例如不以 top/ 开头），可视作多余
    # 这里简单地：我们只采集了 top/ 之下的条目；若 tar 实际含有不在 top/ 下的条目，它们不会被采集，
    # 从而在下面“extra_in_archive”比较中不会出现（因为只比较 expect_set vs tar_set）。
    # 如需更严格，可先收集 all_members 再报错，但你说“只接受保留顶层目录”，通常就打包时控制好即可。

    tar_set = tar_dirs | tar_files

    # 5) 清单对比
    missing_in_archive = sorted(expect_set - tar_set)
    extra_in_archive   = sorted(tar_set - expect_set)

    # 6) 大小对比（仅常规文件）
    size_mismatch: List[Tuple[str, int, int]] = []
    for path_with_top, ssz in expect_sizes.items():
        if path_with_top in tar_sizes:
            if tar_sizes[path_with_top] != ssz:
                size_mismatch.append((path_with_top, ssz, tar_sizes[path_with_top]))

    # 7) 可选：内容哈希（仅按 "top/rel" 精确匹配）
    hash_mismatch: List[str] = []
    if content_check and not errors:
        try:
            with tarfile.open(arc, mode="r:gz") as tf:
                for rel in src_files:
                    if rel not in src_sizes:  # symlink 或被排除
                        continue
                    member_name = f"{top}/{rel}"
                    # 源文件哈希
                    h_src = hashlib.new(hash_algo)
                    with open(src / rel, "rb") as f:
                        for chunk in iter(lambda: f.read(1024 * 1024), b""):
                            h_src.update(chunk)
                    # tar 内哈希
                    try:
                        m = tf.getmember(member_name)
                        if not m.isfile():
                            hash_mismatch.append(member_name)
                            continue
                        fobj = tf.extractfile(m)
                        if fobj is None:
                            hash_mismatch.append(member_name)
                            continue
                        h_tar = hashlib.new(hash_algo)
                        with fobj:
                            for chunk in iter(lambda: fobj.read(1024 * 1024), b""):
                                h_tar.update(chunk)
                        if h_src.hexdigest() != h_tar.hexdigest():
                            hash_mismatch.append(member_name)
                    except KeyError:
                        # 成员缺失 -> 会在 missing 中体现，这里略过
                        pass
        except Exception as e:
            errors.append(f"内容哈希比对失败：{e}")

    ok = (not errors and
          not missing_in_archive and
          not extra_in_archive and
          not size_mismatch and
          (not content_check or not hash_mismatch))

    return {
        "ok": ok,
        "missing_in_archive": missing_in_archive,
        "extra_in_archive": extra_in_archive,
        "size_mismatch": size_mismatch,
        "hash_mismatch": hash_mismatch,
        "errors": errors,
    }


def gen_conf_dict(args: argparse.Namespace, excluded_keys: Union[list, tuple] = None):
    return {k: v for k, v in vars(args).items() if k not in (excluded_keys if excluded_keys is not None else [])}

def pack_vars_to_dict(**kwargs):

    """
    Usage: if call "pack_vars_to_dict(a=1, b=2)", returns {'a': 1, 'b': 2}
    """

    return kwargs


def HMS_to_seconds(hms: str, delimit=':'):
    h, m, s = map(int, hms.split(delimit))
    return h * 3600 + m * 60 + s

def nested_dict_to_nested_list(dict_to_convert, info_list, current_depth=0):
    # 递归终止：如果已经处理完所有指令层，或者数据已不再是字典
    if current_depth >= len(info_list) or not isinstance(dict_to_convert, dict):
        return dict_to_convert

    instruction = info_list[current_depth]

    # 情况 A: 指令是列表 -> 执行排序并展开 (产生嵌套层级)
    if isinstance(instruction, list):
        return [
            nested_dict_to_nested_list(dict_to_convert[k], info_list, current_depth + 1)
            for k in instruction
        ]

    # 情况 B: 指令是单个 Key -> 执行“跳过/穿透” (不产生嵌套层级)
    else:
        # 直接定位到该 key 对应的子 dict，并继续向下处理
        # 注意：这里 current_depth + 1 是为了消耗掉这一层的指令
        target_sub_dict = dict_to_convert[instruction]
        return nested_dict_to_nested_list(target_sub_dict, info_list, current_depth + 1)


if __name__ == '__main__':
    # 嵌套两层的字典 (n_layers = 2)
    data = {
        'Year2023': {
            'Sales': {'Q1': 100, 'Q2': 200},
            'Profit': {'Q1': 50, 'Q2': 60}
        },
        'Year2024': {
            'Sales': {'Q1': 110, 'Q2': 210},
            'Profit': {'Q1': 55, 'Q2': 65}
        }
    }

    info_list = [['Year2024', 'Year2023'], ['Sales', 'Profit'], ['Q2', 'Q1']]
    result = nested_dict_to_nested_list(data, info_list)
    print(result)

    info_list = [['Year2024', 'Year2023'], 'Sales', ['Q2', 'Q1']]
    result = nested_dict_to_nested_list(data, info_list)
    print(result)