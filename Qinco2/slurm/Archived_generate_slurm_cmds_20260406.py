import sys
sys.path.append('/lustre/fswork/projects/rech/thj/uth68ud/PycharmProjects/Qinco2')

from util.var_util import *
import argparse
from util.file_util import FileWriter, FileReader, DirProcessor
import itertools
import os
import matplotlib.pyplot as plt
import numpy as np
import re
from util.test_util import *


def get_hyperparam(args):
    default_A = {'train': 16, 'test': 32}
    A = args.A if args.A is not None else default_A['train' if args.mode == 'train' else 'test']
    if args.mode == 'train':
        assert args.train_A is None
        train_A = A
    else:
        train_A = args.train_A if args.train_A is not None else default_A['train']

    default_B = {'train': 32, 'test': 64}
    B = args.B if args.B is not None else default_B['train' if args.mode == 'train' else 'test']
    if args.mode == 'train':
        assert args.train_B is None
        train_B = B
    else:
        train_B = args.train_B if args.train_B is not None else default_B['train']

    default_L = {'S': 2, 'M': 4, 'L': 16}
    L = args.L if args.L is not None else default_L[args.model_size]

    default_de = {'S': 128, 'M': 384, 'L': 384}
    de = args.de if args.de is not None else default_de[args.model_size]

    default_dh = {'S': 128, 'M': 384, 'L': 384}
    dh = args.dh if args.dh is not None else default_dh[args.model_size]

    return A, B, train_A, train_B, L, de, dh


def get_setting_full_path(conf: dict, parent_path):

    i_setting = 0
    while True:

        setting_path = f'setting_{i_setting}'
        setting_full_path = os.path.join(parent_path, setting_path)
        conf_fname_ = os.path.join(setting_full_path, f'conf_{i_setting}.json')
        if not os.path.exists(setting_full_path):
            DirProcessor.create_dir(setting_full_path)
            FileWriter.save_dict_to_json(conf, conf_fname_)
            print(f'No existing experiment found that uses the same configuration as this one. '
                  f'Creating new setting save path at {setting_full_path}')
            break

        if os.path.exists(conf_fname_):
            # exec(gen_cmd_print_variables('conf_fname_'))
            conf_ = FileReader.load_dict_from_json(conf_fname_)

            # # 打补丁
            if 'epoch_size' not in conf_:
                conf_['epoch_size'] = 10_000_000
            if 'lr' not in conf_:
                conf_['lr'] = 0.0008
            FileWriter.save_dict_to_json(conf_, conf_fname_)


            assert 'epoch_size' in conf_

            is_same = True
            for key, val in conf.items():  # 注意：这里为后续增加其它的key做好了准备，逻辑是，如果新的confd中某些keys在旧confs_里存在，而且这些keys对应的vals在二者中不一样，才认为二者是不一致的。不考虑只在conf里存在的新keys

                if key in conf_.keys() and conf_[key] != val:
                    is_same = False
                    break
            if is_same:
                print(f'Existing experiment found at {setting_full_path} that uses the same configuration as this one.')
                break

        i_setting += 1

    return setting_full_path, i_setting


def plot_training_curves(run_log_full_fname):
    losses = []
    val_metrics = []

    # 预编译正则表达式以提高效率
    loss_pattern = re.compile(r"All losses:.*mse_loss=([0-9\.\-e]+|nan)")
    val_pattern = re.compile(r"Validation metrics:.*MSE=([0-9\.\-e]+|nan)")

    training_started, lines_since_last_loss = False, 1
    with open(run_log_full_fname, 'r', encoding='utf-8') as f:
        for line in f:
            # 提取训练 Loss
            loss_match = loss_pattern.search(line)
            if loss_match:
                val_str = loss_match.group(1)
                losses.append(float(val_str))

                lines_since_last_loss = 1
                if training_started is False:
                    training_started = True
                continue

            # 提取验证指标
            val_match = val_pattern.search(line)
            if val_match and lines_since_last_loss == 1:
                val_str = val_match.group(1)
                val_metrics.append(float(val_str))

            if training_started:
                lines_since_last_loss += 1
    try:
        assert len(losses) == len(val_metrics) - 1
    except Exception as e:
        exec(gen_cmd_print_variables('run_log_full_fname'))
        input("paused")
    losses = [float('nan')] + losses

    dir_name = os.path.dirname(run_log_full_fname)
    setting_name, dir_name = os.path.basename(dir_name), os.path.dirname(dir_name)
    main_name, uniform_result_path = os.path.basename(dir_name), os.path.dirname(dir_name)

    # exec(gen_cmd_print_variables('main_name, setting_name, uniform_result_path'))
    # input("paused")

    # assert uniform_result_path == default_result_path



    """
    绘制带有双 Y 轴的学习曲线，并自动处理 NaN 断点。

    参数:
    losses: 1D ndarray, 训练损失
    val_metrics: 1D ndarray, 验证集指标 (例如 MSE)
    """
    epochs = np.arange(1, len(losses) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # --- 绘制左侧 Y 轴 (Loss) ---
    color_loss = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color_loss, fontsize=12)
    # 使用实线 '-'，颜色为红色
    line1, = ax1.plot(epochs, losses, label='Train Loss',
                      color=color_loss, linestyle='-', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color_loss)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- 绘制右侧 Y 轴 (Validation Metric) ---
    ax2 = ax1.twinx()  # 共享 X 轴
    color_val = 'tab:blue'
    ax2.set_ylabel('Val MSE', color=color_val, fontsize=12)
    # 使用虚线 '--'，颜色为蓝色
    line2, = ax2.plot(epochs, val_metrics, label='Val MSE',
                      color=color_val, linestyle='--', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color_val)

    # --- 合并图例 ---
    # 因为有两个 ax，直接调 plt.legend() 只会显示一个，需要手动合并
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')

    plt.title(f'{main_name} {setting_name}')
    fig.tight_layout()

    save_full_fname = os.path.join(uniform_result_path, f'learning_curves_{main_name}_{setting_name}.png')
    print(f'Saving learning curves to {save_full_fname}.')
    fig.savefig(save_full_fname, bbox_inches='tight')
    # input("paused")
    # plt.show()

    plt.close(fig)


if __name__ == '__main__':

    # 注意：这个script只能生成出来要跑的代码然后手动粘贴到shell（因为涉及到cd）
    # 基础用法：
    # 用于训练 python $VQ_CODE/slurm/generate_slurm_cmds.py --mode train --print_early_stop
    # 用于shen_search: python $VQ_CODE/slurm/generate_slurm_cmds.py --mode shen_search

    parser = argparse.ArgumentParser()

    # 注意：此为必须参数
    parser.add_argument('--mode', choices=['train', 'shen_search', 'ivf_centroids'], type=str, required=True)

    parser.add_argument('--all_datasets', nargs='+', default=list(default_data_info.keys()))

    parser.add_argument('--all_M', nargs='+', type=int, default=[6, 8, 16, 20])
    parser.add_argument('--all_K', nargs='+', type=int, default=[64, 256, 1024, 4096])

    parser.add_argument('--model_size', choices=['S', 'M', 'L'], default='L')
    parser.add_argument('--A', type=int, default=None)
    parser.add_argument('--B', type=int, default=None)
    parser.add_argument('--train_A', type=int, default=None)    # 注意：这两个参数只在mode不是'train'的时候使用，目的是指出要用哪一个train setting下的A和B来做testing; testing用的A和B通过"--A""--B"来设置
    parser.add_argument('--train_B', type=int, default=None)
    parser.add_argument('--L', type=int, default=None)
    parser.add_argument('--de', type=int, default=None)
    parser.add_argument('--dh', type=int, default=None)

    # parser.add_argument('--n_train', type=int, default=None)
    # parser.add_argument('--n_val', type=int, default=None)


    # train parameters
    parser.add_argument('--epoch_size', type=int, default=10_000_000)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--lr', type=float, default=0.0008)

    # shen search parameters
    parser.add_argument('--first_n_queries', type=int, default=1000)
    parser.add_argument('--first_n_db', type=int, default=10000)
    parser.add_argument('--shen_search_batch_size', type=int, default=64)

    # IVF centroids parameters
    parser.add_argument('--ivf_k', type=int, default=1048576)
    parser.add_argument('--ivf_centroid_trainset', type=int, default=100_000)
    parser.add_argument('--ivf_centroid_valset', type=int, default=10_000)

    parser.add_argument('--slurm_time', default='20:00:00')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--print_early_stop', action='store_true')  # just for training

    args = parser.parse_args()

    A, B, train_A, train_B, L, de, dh = get_hyperparam(args)

    cmds = []
    early_stop_logs, nan_logs = [], []
    for dataset, M, K in itertools.product(args.all_datasets, args.all_M, args.all_K):
        train_main_save_path = os.path.join(default_result_path, f'{dataset}_M_{M}_K_{K}')
        DirProcessor.create_dir(train_main_save_path)

        # Check if there is a pre-existing configuration
        train_conf = {
            'M': M, 'K': K,
            'Base_model': f'Qinco2-{args.model_size}',
            'A': train_A, 'B': train_B, 'L': L, 'de': de, 'dh': dh,
            'epoch_size': args.epoch_size, 'epochs': args.epochs,
            'lr': args.lr,
        }

        # cmd_train_val = ''
        # if args.n_train is not None:
        #     cmd_train_val += f'ds.trainset={args.n_train}'
        # if args.n_val is not None:
        #     cmd_train_val += f'ds.valset={args.n_val}'

        # Training
        if args.mode == 'train':
            save_path, i_setting = get_setting_full_path(train_conf, train_main_save_path)

            train_set_full_fname = os.path.join(
                default_data_info[dataset]['path'], default_data_info[dataset]['train_fname']
            )

            saved_model_fname = f'Qinco2{args.model_size}_{dataset}_M_{M}_K_{K}.pt'
            saved_model_full_fname = os.path.join(save_path, saved_model_fname)

            run_log_fname = 'run.log'
            run_log_full_fname = os.path.join(save_path, run_log_fname)

            if args.overwrite or not os.path.exists(saved_model_full_fname) or not os.path.exists(run_log_full_fname):
                run_option = 'from_scratch'
            else:
                # assert os.path.exists(run_log_full_fname)

                if FileReader.str_in_file('Task done', run_log_full_fname):
                    run_option = 'no_run'
                    plot_training_curves(run_log_full_fname)

                    if args.print_early_stop and FileReader.str_in_file(
                            'Val loss did not improve for 10 steps, stopping', run_log_full_fname):
                        early_stop_logs.append(run_log_full_fname)
                        if FileReader.str_in_file('nan', run_log_full_fname):
                            nan_logs.append(run_log_full_fname)

                else:
                    run_option = 'resume'

            if run_option == 'no_run':
                continue

            if run_option == 'from_scratch':
                resume = False
                resume_from = ''
            else:
                resume = True
                resume_from = f'model={saved_model_full_fname}' # 注意：model=那里等号两边不要加空格


            cmds.append(f'\ncd {save_path}')
            cmds.append(f'python $VQ_CODE/slurm/submit_job_Qinco.py '
                  f'"$VQ_CODE/run.sh '   # 注意：不要删掉末尾空格
                  f'task={args.mode} '
                  f'output={saved_model_full_fname} '
                  f'resume={resume} '
                  f'model_args=qinco2-{args.model_size} L={L} M={M} K={K} dh={dh} de={de} A={A} B={B} db={dataset} '
                  f'ds.loop={args.epoch_size} epochs={args.epochs} '      
                  f'trainset={train_set_full_fname} verbose=true {resume_from}" '
                  f'--multi_gpu --job_id Qinco2_{dataset}_M_{M}_K_{K}_setting_{i_setting} --slurm_time {args.slurm_time}')

        # Testing tasks
        # 注意：要判断清每一个testing task是否支持multi-gpu，甚至只能在cpu上跑
        else:

            ######### Make sure the training is done #############
            train_save_path, train_i_setting = get_setting_full_path(train_conf, train_main_save_path)

            train_set_full_fname = os.path.join(
                default_data_info[dataset]['path'], default_data_info[dataset]['train_fname']
            )

            saved_model_fname = f'Qinco2{args.model_size}_{dataset}_M_{M}_K_{K}.pt'
            saved_model_full_fname = os.path.join(train_save_path, saved_model_fname)

            train_log_fname = 'run.log'
            train_log_full_fname = os.path.join(train_save_path, train_log_fname)

            # 临时代码，记得删掉
            ######## testing result save path ###########
            test_main_save_path = os.path.join(train_save_path, args.mode)
            DirProcessor.create_dir(test_main_save_path)

            ####### testing data path ########
            test_full_fname = os.path.join(
                default_data_info[dataset]['path'], default_data_info[dataset]['test_fname']
            )
            query_full_fname = os.path.join(
                default_data_info[dataset]['path'], default_data_info[dataset]['query_fname']
            )

            if args.mode == 'shen_search':
                conf = deepcopy(train_conf)

                # Qinco2 uses different A and B for training and testing
                conf['train_A'] = conf['A']
                conf['train_B'] = conf['B']
                conf['A'] = A
                conf['B'] = B

                conf['first_n_queries'] = args.first_n_queries
                conf['first_n_db'] = args.first_n_db
                conf['shen_search_batch_size'] = args.shen_search_batch_size

                save_path, i_setting = get_setting_full_path(conf, test_main_save_path)

                cmds.append(f'\ncd {save_path}')
                cmds.append(f'python $VQ_CODE/slurm/submit_job_Qinco.py '
                            f'"$VQ_CODE/run.py '  # 注意：不要删掉末尾空格
                            f'task={args.mode} '
                            f'output={save_path} '
                            f'model={saved_model_full_fname} '
                            f'db={test_full_fname} queries={query_full_fname} '

                            # Preserve the training parameters just to be safe, but replace A and B with test settings!
                            f'model_args=qinco2-{args.model_size} L={L} M={M} K={K} dh={dh} de={de} A={A} B={B} '
                            f'ds.loop={args.epoch_size} epochs={args.epochs} '

                            f'shen_search.first_n_queries={args.first_n_queries} shen_search.first_n_db={args.first_n_db} '
                            f'batch={args.shen_search_batch_size}" '
                            # f'trainset={train_set_full_fname} verbose=true {resume_from}" '
                            f'--job_id Qinco2_{dataset}_M_{M}_K_{K}_train_{train_i_setting}_{args.mode}_{i_setting} --slurm_time {args.slurm_time}')


    print('\n\n\n******** All done: the complete commands are as follows ***********\n')
    for cmd in cmds:
        print(cmd)

    if args.print_early_stop:
        print('\n\n\n####### Early stopping recorded in the following logs #########\n')
        for log in early_stop_logs:
            print(log)

        print('\n\n\n!!!!!!! NaN recorded in the following logs !!!!!!!!!!\n')
        for log in nan_logs:
            print(log)


