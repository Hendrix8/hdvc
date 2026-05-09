# -*- coding: UTF-8 -*-

import sys
sys.path.append('/lustre/fswork/projects/rech/thj/uth68ud/PycharmProjects/Qinco2')

import os
import argparse
from os.path import basename, splitext
from util.test_util import *
from util.var_util import default_code_path


def compose_batch_submit_cmds(conf, job_id_pfx, python_script_full_fname, slurm_time='20:00:00', test=False, ):

    def expand_conf(conf: dict):  # 注意：这里confs不能省，因为里边要递归

        """

        conf里，一个key对应的value会对应多个options. 这个函数的作用就是把这个conf扩展成一个list of expanded_confs，其中每一个expanded_conf的key和conf相同，但是对应的value（被称为atom_v）是单个option
        其实就是对全部key的全部options做了一个product

        """

        first_key = list(conf.keys())[0]

        if len(conf) == 1:
            return [{first_key: atom_v} for atom_v in conf[first_key]]

        conf_ = deepcopy(conf)
        conf_.pop(first_key)
        expanded_confs_ = expand_conf(conf_)

        expanded_confs = []
        for atom_v in conf[first_key]:
            # exec(gen_cmd_print_variables('expanded_confs_'))
            cur_expanded_conf = [dict(**{first_key: atom_v}, **expanded_conf_) for expanded_conf_ in expanded_confs_]
            expanded_confs += deepcopy(cur_expanded_conf)

        return expanded_confs

    def compose_item(key, atom_v):

        """

        将一个expanded_conf中单个的(key, atom_v)键值对转化成对应的命令

        """

        if atom_v is None:
            return ''

        # exec(gen_cmd_print_variables('atom_v, type(atom_v)'))
        assert type(atom_v) in (str, int, float, bool, list)
        assert key is not None
        if type(atom_v) in (str, int, float):
            return f'--{key} {atom_v}'
        if isinstance(atom_v, bool):
            return f'--{key}' if atom_v else ''
        if isinstance(atom_v, list):
            ret = f'--{key}'
            for v in atom_v:
                ret = f'{ret} {v}'
            return ret

    # exec(gen_cmd_print_variables('conf'))
    # input("paused")

    expanded_confs = expand_conf(conf)

    # exec(gen_cmd_print_variables('expanded_confs[0]'))
    # input("paused")

    cmds = []
    for i_conf, expanded_conf in enumerate(expanded_confs):
        submit_script = os.path.join(default_code_path, 'slurm', 'submit_job_Qinco.py')  # 注意：这里不能直接去找当前文件名（因为会找到的是调用这个函数的那个文件名）
        cmd = f'python {submit_script} "{python_script_full_fname}'

        for k, v in expanded_conf.items():
            cmd = f'{cmd} {compose_item(k, v)}'

        if not test:
            cmd = f'{cmd}" --slurm_time {slurm_time}'
        else:
            cmd = f'{cmd}" --test'
        cmd = f'{cmd} --job_id {job_id_pfx}_{i_conf}'

        cmds.append(cmd)

    return cmds


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('python_cmd', help='In the form of "script.py --arg1 arg1 --arg2 arg2"')

    parser.add_argument('--multi_gpu', action='store_true') # 4 GPUs
    parser.add_argument('--job_id', default=None)
    parser.add_argument('--slurm_time', default='20:00:00', help='Slurm time limit, in the format of HH:MM:SS') # 注意：这个在--test的时候会被override
    parser.add_argument('--print_cmd_only', action='store_true')


    args = parser.parse_args()

    python_cmd = args.python_cmd
    multi_gpu = args.multi_gpu
    job_id = args.job_id
    slurm_time = args.slurm_time
    print_cmd_only = args.print_cmd_only

    slurm_script = 'submit_to_h100_Qinco_single_gpu.sh' if not multi_gpu else 'submit_to_h100_Qinco_multi_gpu.sh'

    bash_fname = os.path.join(os.path.dirname(os.path.abspath(__file__)), slurm_script)

    ####################################################
    cmd_elms = python_cmd.strip().split(' ')

    script = cmd_elms[0]
    assert script.endswith('.py' if not multi_gpu else '.sh')
    assert os.path.exists(script)
    job_id_pfx = splitext(basename(script))[0]

    abbv_map = {
        # 如果有需要再用
    }

    s_conf = ''
    for cmd_elm in cmd_elms[1:]:

        if cmd_elm == '--overwrite':
            continue

        if cmd_elm.startswith('--'):
            cmd_elm = cmd_elm[len('--'):]
        if cmd_elm in abbv_map.keys():
            cmd_elm = abbv_map[cmd_elm]
            if len(cmd_elm) == 0:
                continue

        s_conf += f'_{cmd_elm}'
    job_id = f'{job_id_pfx}{s_conf}' if job_id is None else job_id

    submit_cmd = f'{bash_fname} {job_id} \"{python_cmd}\" --time={slurm_time}'


    if not print_cmd_only:
        print(f'Executing command: {submit_cmd}')
        os.system(submit_cmd)
    else:
        print(f'Command to execute: {submit_cmd}')

