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
from datetime import datetime, timedelta
from typing import Literal


def get_hyperparam(args, K):
    default_A = {'train': 16, 'test': 32}
    A = args.A if args.A is not None else default_A['train' if args.mode in 'train' else 'test']
    A = min(A, K) # 注意：A必须不能大于K!
    if args.mode in 'train':
        assert args.train_A is None
        train_A = A
    else:
        train_A = args.train_A if args.train_A is not None else default_A['train']
    train_A = min(train_A, K)

    default_B = {'train': 32, 'test': 64}
    B = args.B if args.B is not None else default_B['train' if args.mode == 'train' else 'test']
    B = min(B, K)  # 注意：B必须不能大于K!
    if args.mode == 'train':
        assert args.train_B is None
        train_B = B
    else:
        train_B = args.train_B if args.train_B is not None else default_B['train']
    train_B = min(train_B, K)

    default_L = {'S': 2, 'M': 4, 'L': 16}
    L = args.L if args.L is not None else default_L[args.model_size]

    default_de = {'S': 128, 'M': 384, 'L': 384}
    de = args.de if args.de is not None else default_de[args.model_size]

    default_dh = {'S': 128, 'M': 384, 'L': 384}
    dh = args.dh if args.dh is not None else default_dh[args.model_size]

    return A, B, train_A, train_B, L, de, dh


def get_setting_full_path(args, conf: dict, parent_path, verbose=True):

    result_path = args.result_path.replace('\\', '/').replace('//', '/')
    for k, v in conf.items():
        if isinstance(v, (float, np.floating)) and v > 1 and int(v) == v:
            conf[k] = int(v)
            continue

        v = str(v).replace('\\', '/').replace('//', '/')
        if result_path in v:
            conf[k] = v  # 注意：这个v已经改动过


    i_setting = 0
    while True:

        setting_path = f'setting_{i_setting}'
        setting_full_path = os.path.join(parent_path, setting_path)
        conf_fname_ = os.path.join(setting_full_path, f'conf_{i_setting}.json')
        if not os.path.exists(setting_full_path):
            DirProcessor.create_dir(setting_full_path)
            FileWriter.save_dict_to_json(conf, conf_fname_)
            if verbose:
                print(f'No existing experiment found that uses the same configuration as this one. '
                      f'Creating new setting save path at {setting_full_path}')
            break

        if os.path.exists(conf_fname_):
            # exec(gen_cmd_print_variables('conf_fname_'))
            conf_ = FileReader.load_dict_from_json(conf_fname_)

            # # 打补丁
            if 'une_ivf_centroid' in conf_:
                conf_['use_ivf_centroids'] = conf_['une_ivf_centroid']
                conf_.pop('une_ivf_centroid', None)
            FileWriter.save_dict_to_json(conf_, conf_fname_)


            for k_, v_ in conf_.items():
                v_ = str(v_).replace('\\', '/').replace('//', '/')
                if default_result_path in v_:
                    conf_[k_] = v_.replace(default_result_path, result_path).replace('\\', '/').replace('//', '/')


            # assert 'ds_loop' in conf_

            # is_same = True
            # for key, val in conf.items():  # 注意：这里为后续增加其它的key做好了准备，逻辑是，如果新的conf中某些keys在旧confs_里存在，而且这些keys对应的vals在二者中不一样，才认为二者是不一致的。不考虑只在conf里存在的新keys
            #
            #     if key in conf_.keys() and conf_[key] != val:
            #         is_same = False
            #         break
            # if is_same:

            if conf == conf_:
                if verbose:
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
        # input("paused")
    losses = [float('nan')] + losses

    dir_name = os.path.dirname(run_log_full_fname)
    setting_name, dir_name = os.path.basename(dir_name), os.path.dirname(dir_name)
    main_name, uniform_result_path = os.path.basename(dir_name), os.path.dirname(dir_name)

    # exec(gen_cmd_print_variables('main_name, setting_name, uniform_result_path'))
    # input("paused")

    # assert uniform_result_path == args.result_path



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


def get_all_train_checkpoints(train_save_path, return_format: Literal['base_fname', 'full_fname', 'epoch'] = 'epoch'):

    all_best, all_latest = [], []
    for fname in os.listdir(train_save_path):
        if not fname.endswith('.pt'):
            continue

        if fname.startswith('best'):
            all_best.append(fname)
            continue
        if fname.startswith('latest'):
            all_latest.append(fname)

    # sort by epochs
    all_best_epochs = [int(fname.split('.')[0].split('_')[-1]) for fname in all_best]
    all_latest_epochs = [int(fname.split('.')[0].split('_')[-1]) for fname in all_latest]
    order_best, order_latest = np.argsort(all_best_epochs), np.argsort(all_latest_epochs)

    all_best = [all_best[i] for i in order_best]
    all_latest = [all_latest[i] for i in order_latest]

    if return_format == 'base_fname':   # 注意：这个哪怕什么都不做也要保留（因为下面会raise Error）
        pass
    elif return_format == 'full_fname':
        all_best = [os.path.join(train_save_path, fname) for fname in all_best]
        all_latest = [os.path.join(train_save_path, fname) for fname in all_latest]
    elif return_format == 'epoch':
        all_best = np.array(all_best_epochs)[order_best]
        all_latest = np.array(all_latest_epochs)[order_latest]
    else:
        raise ValueError('Invalid return_format.')

    return all_best, all_latest


def generate_cmds(args, dataset, M, K):
    main_save_id = f'{dataset}_M_{M}_K_{K}'
    train_main_save_path = os.path.join(args.result_path, f'{dataset}_M_{M}_K_{K}')
    DirProcessor.create_dir(train_main_save_path)

    # Check if there is a pre-existing configuration
    A, B, train_A, train_B, L, de, dh = get_hyperparam(args, K)
    train_conf = {
        'M': M, 'K': K,
        'Base_model': f'Qinco2-{args.model_size}',
        'A': train_A, 'B': train_B, 'L': L, 'de': de, 'dh': dh,
        'ds_loop': args.ds_loop, 'max_epochs': args.max_epochs,
        'lr': args.lr,
    }
    if not args.use_ivf_centroids:
        train_conf['use_ivf_centroids'] = False
    else:
        raise NotImplementedError

    # add new parameters here. If these parameters are not present in the conf.json file, default values are used.
    if args.n_train is not None:
        assert args.n_val is not None
        train_conf['n_train'] = args.n_train
        train_conf['n_val'] = args.n_val

    # Training
    if args.mode == 'train':
        cur_early_stop_log, cur_nan_log = None, None
        save_path, i_setting = get_setting_full_path(args, train_conf, train_main_save_path)
        save_id = f'{main_save_id}_setting_{i_setting}'

        train_set_full_fname = os.path.join(
            default_data_info[dataset]['path'], default_data_info[dataset]['train_fname']
        )

        # saved_model_fname = f'Qinco2{args.model_size}_{dataset}_M_{M}_K_{K}.pt'
        # saved_model_full_fname = os.path.join(save_path, saved_model_fname)

        run_log_fname = 'run.log'
        run_log_full_fname = os.path.join(save_path, run_log_fname)

        resume_from = None
        if args.overwrite or not os.path.exists(run_log_full_fname):
            run_option = 'from_scratch'
        else:
            if FileReader.str_in_file('Forced quitting', run_log_full_fname):   # forced quitting
                run_option = 'no_run'
            else:
                _, all_latest_full_fnames = get_all_train_checkpoints(save_path, return_format='full_fname')
                if len(all_latest_full_fnames) == 0:
                    run_option = 'from_scratch'

                elif FileReader.str_in_file('Task done', run_log_full_fname):
                    run_option = 'no_run'
                    # plot_training_curves(run_log_full_fname)

                    if args.print_anomalies and FileReader.str_in_file(
                            'Val loss did not improve for 10 steps, stopping', run_log_full_fname):
                        cur_early_stop_log = run_log_full_fname
                        if FileReader.str_in_file('nan', run_log_full_fname):
                            cur_nan_log = run_log_full_fname
                else:
                    run_option = 'resume'
                    resume_from = all_latest_full_fnames[-1]

        if run_option == 'no_run':
            return None, cur_early_stop_log, cur_nan_log

        if run_option == 'from_scratch':
            resume = False
            resume_from = ''
        else:
            assert resume_from is not None
            resume = True
            resume_from = f'model={resume_from}'  # 注意：model=那里等号两边不要加空格

        cd_cmd = f'\ncd {save_path}'
        main_cmd = (f'python {args.code_path}/slurm/submit_job_Qinco.py '    # 注意：不要删掉结尾空格！
               f'"{args.code_path}/run.sh '
               f'task={args.mode} '
               f'output_dir={save_path} '
               f'save_id={save_id} '
               # f'output={saved_model_full_fname} '
               f'resume={resume} '
               f'model_args=qinco2-{args.model_size} L={L} M={M} K={K} dh={dh} de={de} A={A} B={B} db={dataset} '
               f'ds.loop={args.ds_loop} epochs={args.max_epochs} ')
        if args.n_train is not None:
            assert args.n_val is not None
            main_cmd += f'ds.trainset={args.n_train} ds.valset={args.n_val} '
        main_cmd += (f'trainset={train_set_full_fname} verbose=true {resume_from} '
                     f'max_sec_total={args.max_sec_total} max_sec_per_epoch={args.max_sec_per_epoch}" '
                     f'--multi_gpu --job_id Qinco2_{dataset}_M_{M}_K_{K}_setting_{i_setting} --slurm_time {args.slurm_time} '
                    )
        cur_cmds = [cd_cmd, main_cmd]

        return cur_cmds, cur_early_stop_log, cur_nan_log


    # Non-training tasks (Including the training of the pairwise encoder)
    # 注意：要判断清每一个testing task是否支持multi-gpu，甚至只能在cpu上跑

    ######### Find the training path #############
    train_save_path, train_i_setting = get_setting_full_path(args, train_conf, train_main_save_path)
    train_save_id = f'{main_save_id}_setting_{train_i_setting}'
    # train_set_full_fname = os.path.join(
    #     default_data_info[dataset]['path'], default_data_info[dataset]['train_fname']
    # )

    if args.mode in ('encode_train', 'encode_test', 'encode_queries'):
        conf = {
            'encode_tgt_epochs': args.encode_tgt_epochs,
            'train_A': train_conf['A'],
            'train_B': train_conf['B']
        }
        if args.mode == 'encode_train':
            A, B = train_A, train_B
        else:
            conf['A'], conf['B'] = A, B

        main_save_path = os.path.join(train_save_path, args.mode)
        DirProcessor.create_dir(main_save_path)
        save_path, i_setting = get_setting_full_path(args, conf, main_save_path)

        save_fname = f'{args.mode}_setting_{i_setting}'
        if args.encode_first_n is not None:
            save_fname += f'_0-{args.encode_first_n - 1}'   # 下标从0开始
        save_fname += '.npz'
        save_full_fname = os.path.join(save_path, save_fname)

        run_log_fname = 'run.log'
        run_log_full_fname = os.path.join(save_path, run_log_fname)

        if args.overwrite or not os.path.exists(run_log_full_fname) or not os.path.exists(save_full_fname):
            run_option = 'from_scratch'
        else:
            if FileReader.str_in_file('Task done', run_log_full_fname):
                run_option = 'no_run'
            else:
                run_option = 'from_scratch'

        if run_option == 'from_scratch':

            ############### locate data file
            if args.mode == 'encode_train':
                fname_key = 'train_fname'
            elif args.mode == 'encode_test':
                fname_key = 'test_fname'
            else:
                fname_key = 'query_fname'

            data_full_fname = os.path.join(
                default_data_info[dataset]['path'], default_data_info[dataset][fname_key]
            )

            ############### locate trained model
            print("****", os.path.join(train_save_path, f'latest_{train_save_id}_checkpoint_{args.encode_tgt_epochs}.pt'))
            cur_no_model = not FileReader.str_in_file('Task done', os.path.join(train_save_path, 'run.log'))
            if cur_no_model:
                cur_no_model_log = run_log_full_fname
                return None, cur_no_model_log
            all_best, _ = get_all_train_checkpoints(train_save_path)
            all_best = [best for best in all_best if best <= args.encode_tgt_epochs]
            model_full_fname = os.path.join(
                train_save_path, f'best_{train_save_id}_checkpoint_{all_best[-1]}.pt'
            )

            ############## generate the cmd
            cd_cmd = f'\ncd {save_path}'
            main_cmd = (f'python {args.code_path}/slurm/submit_job_Qinco.py '  # 注意：不要删掉结尾空格！
                        f'"{args.code_path}/run.sh '
                        f'task=encode '
                        f'output={save_full_fname} '
                        f'model={model_full_fname} '
                        f'trainset={data_full_fname} '
                        f'db={data_full_fname} '
                        f'encode_trainset={args.mode == "encode_train"} '
                        )
            if args.encode_first_n is not None:
                main_cmd += f'ds.db={args.encode_first_n} ds.trainset={args.encode_first_n} '
            main_cmd += (f'A={A} B={B}" '
                         f'--multi_gpu --job_id Qinco2_{args.mode}_{train_save_id}_encode_setting_{i_setting} --slurm_time {args.slurm_time} '
                         )
            cur_cmds = [cd_cmd, main_cmd]

            return cur_cmds, None
        else:
            return None, None

    if args.mode == 'train_qinco_aq':

        encode_train_conf = {
            'encode_tgt_epochs': args.encode_tgt_epochs,
            'train_A': train_conf['A'],
            'train_B': train_conf['B']
        }
        encode_train_save_path, encode_train_i_setting = get_setting_full_path(
            args, encode_train_conf, os.path.join(train_save_path, 'encode_train'))

        conf = {}
        if args.tqa_n_train is not None:
            assert args.tqa_n_val is not None
            conf['tqa_n_train'] = args.tqa_n_train
            conf['tqa_n_val'] = args.tqa_n_val

        tqa_main_save_path = os.path.join(encode_train_save_path, args.mode)
        DirProcessor.create_dir(tqa_main_save_path)
        save_path, i_setting = get_setting_full_path(args, conf, tqa_main_save_path)

        save_fname = f'{args.mode}_setting_{i_setting}.npz'
        save_full_fname = os.path.join(save_path, save_fname)

        run_log_fname = 'run.log'
        run_log_full_fname = os.path.join(save_path, run_log_fname)

        if args.overwrite or not os.path.exists(run_log_full_fname) or not os.path.exists(save_full_fname):
            run_option = 'from_scratch'
        else:
            if FileReader.str_in_file('Task done', run_log_full_fname):
                run_option = 'no_run'
            else:
                run_option = 'from_scratch'

        if run_option == 'from_scratch':
            ori_data_full_fname = os.path.join(
                default_data_info[dataset]['path'], default_data_info[dataset]['train_fname']
            )
            encode_data_full_fname = os.path.join(
                encode_train_save_path, f'encode_train_setting_{encode_train_i_setting}.npz'
                # 注意：这里必须找对整个训练集的encoding文件，而非一部分的
            )
            if not os.path.exists(encode_data_full_fname):
                cur_no_encode_log = run_log_full_fname
                # print("Checkpoint 2")
                return None, cur_no_encode_log

            cd_cmd = f'\ncd {save_path}'
            if not args.force_submit_to_slurm:
                main_cmd = (f'python {args.code_path}/run.py '
                            f'task=train_qinco_aq '
                            f'output={save_full_fname} '
                            f'trainset={ori_data_full_fname} '
                            f'encoded_trainset={encode_data_full_fname} '
                            )
                if args.tqa_n_train is not None:
                    assert args.tqa_n_val is not None
                    main_cmd += f'ds.trainset={args.tqa_n_train} ds.valset={args.tqa_n_val}'
            else:
                main_cmd = (f'python {args.code_path}/slurm/submit_job_Qinco.py '  # 注意：不要删掉结尾空格！
                            f'"{args.code_path}/run.py '
                            f'task=train_qinco_aq '
                            f'output={save_full_fname} '
                            f'trainset={ori_data_full_fname} '
                            f'encoded_trainset={encode_data_full_fname} '
                            )
                if args.tqa_n_train is not None:
                    assert args.tqa_n_val is not None
                    main_cmd += f'ds.trainset={args.tqa_n_train} ds.valset={args.tqa_n_val} '
                main_cmd += (f'" '
                             f'--job_id Qinco2_tqa_{train_save_id}_tqa_setting_{i_setting} --slurm_time {args.slurm_time} '
                             )
            cur_cmds = [cd_cmd, main_cmd]

            # print("Checkpoint 1")

            return cur_cmds, None

    if args.mode == 'qinco_aq_compute_distances':

        # locate trained pairwise decoder
        encode_train_conf = {
            'encode_tgt_epochs': args.encode_tgt_epochs,
            'train_A': train_conf['A'],
            'train_B': train_conf['B']
        }
        encode_train_save_path, encode_train_i_setting = get_setting_full_path(
            args, encode_train_conf, os.path.join(train_save_path, 'encode_train'))

        conf = {}
        if args.tqa_n_train is not None:
            assert args.tqa_n_val is not None
            conf['tqa_n_train'] = args.tqa_n_train
            conf['tqa_n_val'] = args.tqa_n_val

        tqa_main_save_path = os.path.join(encode_train_save_path, 'train_qinco_aq')
        DirProcessor.create_dir(tqa_main_save_path)
        tqa_save_path, tqa_i_setting = get_setting_full_path(args, conf, tqa_main_save_path)

        tqa_save_fname = f'train_qinco_aq_setting_{tqa_i_setting}.npz'
        tqa_save_full_fname = os.path.join(tqa_save_path, tqa_save_fname)

        # locate encoded database
        encode_test_conf = {
            'encode_tgt_epochs': args.encode_tgt_epochs,
            'train_A': train_conf['A'],
            'train_B': train_conf['B'],
            'A': A,
            'B': B
        }
        encode_test_main_save_path = os.path.join(train_save_path, 'encode_test')
        encode_test_save_path, encode_test_i_setting = get_setting_full_path(args, encode_test_conf,
                                                                             encode_test_main_save_path)

        encode_test_save_fname = f'encode_test_setting_{encode_test_i_setting}'
        if args.encode_first_n is not None:
            encode_test_save_fname += f'_0-{args.encode_first_n - 1}'  # 下标从0开始
        encode_test_save_fname += '.npz'
        encode_test_save_full_fname = os.path.join(encode_test_save_path, encode_test_save_fname)

        # Create save path
        conf = {
            'encode_test_save_full_fname': encode_test_save_full_fname
        }
        if args.n_db is not None:
            conf['n_db'] = args.n_db
        if args.n_queries is not None:
            conf['n_queries'] = args.n_queries
        main_save_path = os.path.join(tqa_save_path, args.mode)
        DirProcessor.create_dir(main_save_path)
        save_path, i_setting = get_setting_full_path(args, conf, main_save_path)

        save_fname = f'{args.mode}_setting_{i_setting}.npz'
        save_full_fname = os.path.join(save_path, save_fname)

        run_log_fname = 'run.log'
        run_log_full_fname = os.path.join(save_path, run_log_fname)

        if args.overwrite or not os.path.exists(run_log_full_fname) or not os.path.exists(save_full_fname):
            run_option = 'from_scratch'
        else:
            if FileReader.str_in_file('Task done', run_log_full_fname):
                run_option = 'no_run'
            else:
                run_option = 'from_scratch'

        print("!!!!!!!!!!!!!!!!!!")
        if run_option == 'from_scratch':
            cur_no_encode_log, cur_no_tqa_log = None, None
            if not os.path.exists(encode_test_save_full_fname):
                exec(gen_cmd_print_variables('encode_test_save_full_fname'))
                cur_no_encode_log = run_log_full_fname
            if not os.path.exists(tqa_save_full_fname):
                exec(gen_cmd_print_variables('tqa_save_full_fname'))
                cur_no_tqa_log = run_log_full_fname
            if (cur_no_tqa_log is not None) or (cur_no_encode_log is not None):
                return None, cur_no_encode_log, cur_no_tqa_log

            db_full_fname = os.path.join(
                default_data_info[dataset]['path'], default_data_info[dataset]['test_fname']
            )
            query_full_fname = os.path.join(
                default_data_info[dataset]['path'], default_data_info[dataset]['query_fname']
            )

            cd_cmd = f'\ncd {save_path}'
            if not args.force_submit_to_slurm:
                main_cmd = (f'python {args.code_path}/run.py '
                            f'task=qinco_aq_compute_distances '
                            f'output={save_full_fname} '
                            f'qinco_aq_codebooks={tqa_save_full_fname} '
                            f'encoded_db={encode_test_save_full_fname} '
                            f'db={db_full_fname} '
                            f'queries={query_full_fname} '
                            )
                if args.n_db is not None:
                    main_cmd += f'ds.db={args.n_db} '
                if args.n_queries is not None:
                    main_cmd += f'ds.query={args.n_queries} '
            else:
                # main_cmd = (f'python {args.code_path}/run.py '
                #             f'task=qinco_aq_compute_distances '
                #             f'output={save_full_fname} '
                #             f'qinco_aq_codebooks={tqa_save_full_fname} '
                #             f'encoded_db={encode_test_save_full_fname} '
                #             )

                main_cmd = (f'python {args.code_path}/slurm/submit_job_Qinco.py '  # 注意：不要删掉结尾空格！
                            f'"{args.code_path}/run.py '
                            f'task=qinco_aq_compute_distances '
                            f'output={save_full_fname} '
                            f'qinco_aq_codebooks={tqa_save_full_fname} '
                            f'encoded_db={encode_test_save_full_fname} '
                            f'db={db_full_fname} '
                            f'queries={query_full_fname} '
                            )
                if args.n_db is not None:
                    main_cmd += f'ds.db={args.n_db} '
                if args.n_queries is not None:
                    main_cmd += f'ds.query={args.n_queries} '
                main_cmd += (f'" '
                             f'--job_id Qinco2_qacd_{train_save_id}_tqa_{tqa_i_setting}_qacd_setting_{i_setting} --slurm_time {args.slurm_time} '
                             )
            cur_cmds = [cd_cmd, main_cmd]

            return cur_cmds, None, None
        else:
            return None, None, None

    if args.mode == 'train_pairwise_decoder':

        encode_train_conf = {
            'encode_tgt_epochs': args.encode_tgt_epochs,
            'train_A': train_conf['A'],
            'train_B': train_conf['B']
        }
        encode_train_save_path, encode_train_i_setting = get_setting_full_path(
            args, encode_train_conf, os.path.join(train_save_path, 'encode_train'))

        conf = {}
        if args.tpd_n_train is not None:
            assert args.tpd_n_val is not None
            conf['tpd_n_train'] = args.tpd_n_train
            conf['tpd_n_val'] = args.tpd_n_val

        tpd_main_save_path = os.path.join(encode_train_save_path, 'train_pairwise_decoder')
        DirProcessor.create_dir(tpd_main_save_path)
        save_path, i_setting = get_setting_full_path(args, conf, tpd_main_save_path)

        save_fname = f'{args.mode}_setting_{i_setting}.pt'
        save_full_fname = os.path.join(save_path, save_fname)

        run_log_fname = 'run.log'
        run_log_full_fname = os.path.join(save_path, run_log_fname)

        if args.overwrite or not os.path.exists(run_log_full_fname) or not os.path.exists(save_full_fname):
            run_option = 'from_scratch'
        else:
            if FileReader.str_in_file('Task done', run_log_full_fname):
                run_option = 'no_run'
            else:
                run_option = 'from_scratch'

        # print("Checkpoint 3")

        if run_option == 'from_scratch':
            ori_data_full_fname = os.path.join(
                default_data_info[dataset]['path'], default_data_info[dataset]['train_fname']
            )
            encode_data_full_fname = os.path.join(
                encode_train_save_path, f'encode_train_setting_{encode_train_i_setting}.npz'    # 注意：这里必须找对整个训练集的encoding文件，而非一部分的
            )
            if not os.path.exists(encode_data_full_fname):
                cur_no_encode_log = run_log_full_fname
                # print("Checkpoint 2")
                return None, cur_no_encode_log


            cd_cmd = f'\ncd {save_path}'
            main_cmd = (f'python {args.code_path}/slurm/submit_job_Qinco.py '  # 注意：不要删掉结尾空格！
                        f'"{args.code_path}/run.py '
                        f'task=train_pairwise_decoder '
                        f'output={save_full_fname} '
                        f'trainset={ori_data_full_fname} '
                        f'encoded_trainset={encode_data_full_fname} '
                        )
            if train_conf['use_ivf_centroids']:
                raise NotImplementedError
            if args.tpd_n_train is not None:
                assert args.tpd_n_val is not None
                main_cmd += f'ds.trainset={args.tpd_n_train} ds.valset={args.tpd_n_val} '
            main_cmd += (f'" '
                         f'--job_id Qinco2_tpd_{train_save_id}_tpd_setting_{i_setting} --slurm_time {args.slurm_time} '
                         )
            cur_cmds = [cd_cmd, main_cmd]

            # print("Checkpoint 1")

            return cur_cmds, None
        else:
            return None, None

    if args.mode == 'shen_build_indices':

        # locate trained pairwise decoder
        encode_train_conf = {
            'encode_tgt_epochs': args.encode_tgt_epochs,
            'train_A': train_conf['A'],
            'train_B': train_conf['B']
        }
        encode_train_save_path, encode_train_i_setting = get_setting_full_path(
            args, encode_train_conf, os.path.join(train_save_path, 'encode_train'))

        conf = {}
        if args.tpd_n_train is not None:
            assert args.tpd_n_val is not None
            conf['tpd_n_train'] = args.tpd_n_train
            conf['tpd_n_val'] = args.tpd_n_val

        tpd_main_save_path = os.path.join(encode_train_save_path, 'train_pairwise_decoder')
        DirProcessor.create_dir(tpd_main_save_path)
        tpd_save_path, tpd_i_setting = get_setting_full_path(args, conf, tpd_main_save_path)

        tpd_save_fname = f'train_pairwise_decoder_setting_{tpd_i_setting}.pt'
        tpd_save_full_fname = os.path.join(tpd_save_path, tpd_save_fname)

        # locate encoded database
        encode_test_conf = {
            'encode_tgt_epochs': args.encode_tgt_epochs,
            'train_A': train_conf['A'],
            'train_B': train_conf['B'],
            'A': A,
            'B': B
        }
        encode_test_main_save_path = os.path.join(train_save_path, 'encode_test')
        encode_test_save_path, encode_test_i_setting = get_setting_full_path(args, encode_test_conf, encode_test_main_save_path)

        encode_test_save_fname = f'encode_test_setting_{encode_test_i_setting}'
        if args.encode_first_n is not None:
            encode_test_save_fname += f'_0-{args.encode_first_n - 1}'  # 下标从0开始
        encode_test_save_fname += '.npz'
        encode_test_save_full_fname = os.path.join(encode_test_save_path, encode_test_save_fname)

        # Create save path
        conf = {
            'encode_test_save_full_fname': encode_test_save_full_fname
        }
        main_save_path = os.path.join(tpd_save_path, 'shen_build_indices')
        DirProcessor.create_dir(main_save_path)
        save_path, i_setting = get_setting_full_path(args, conf, main_save_path)

        save_fname = f'shen_build_indices_setting_{i_setting}.npz'
        save_full_fname = os.path.join(save_path, save_fname)

        run_log_fname = 'run.log'
        run_log_full_fname = os.path.join(save_path, run_log_fname)

        if args.overwrite or not os.path.exists(run_log_full_fname) or not os.path.exists(save_full_fname):
            run_option = 'from_scratch'
        else:
            if FileReader.str_in_file('Task done', run_log_full_fname):
                run_option = 'no_run'
            else:
                run_option = 'from_scratch'

        if run_option == 'from_scratch':
            cur_no_encode_log, cur_no_tpd_log = None, None
            if not os.path.exists(encode_test_save_full_fname):
                cur_no_encode_log = run_log_full_fname
            if not os.path.exists(tpd_save_full_fname):
                exec(gen_cmd_print_variables('tpd_save_full_fname'))
                cur_no_tpd_log = run_log_full_fname
            if (cur_no_tpd_log is not None) or (cur_no_encode_log is not None):
                return None, cur_no_encode_log, cur_no_tpd_log

            cd_cmd = f'\ncd {save_path}'
            main_cmd = (f'python {args.code_path}/slurm/submit_job_Qinco.py '  # 注意：不要删掉结尾空格！
                        f'"{args.code_path}/run.py '
                        f'task=shen_build_indices '
                        f'output={save_full_fname} '
                        f'pairwise_decoder={tpd_save_full_fname} '
                        f'encoded_db={encode_test_save_full_fname} '
                        )
            main_cmd += (f'" '
                         f'--job_id Qinco2_shen_build_indices_{train_save_id}_tpd_{tpd_i_setting}_sbi_setting_{i_setting} --slurm_time {args.slurm_time} '
                         )
            cur_cmds = [cd_cmd, main_cmd]

            return cur_cmds, None, None
        else:
            return None, None, None

    if args.mode == 'shen_compute_distances':

        # UNFINISHED

        # locate trained pairwise decoder
        encode_train_conf = {
            'encode_tgt_epochs': args.encode_tgt_epochs,
            'train_A': train_conf['A'],
            'train_B': train_conf['B']
        }
        encode_train_save_path, encode_train_i_setting = get_setting_full_path(
            args, encode_train_conf, os.path.join(train_save_path, 'encode_train'))

        conf = {}
        if args.tpd_n_train is not None:
            assert args.tpd_n_val is not None
            conf['tpd_n_train'] = args.tpd_n_train
            conf['tpd_n_val'] = args.tpd_n_val

        tpd_main_save_path = os.path.join(encode_train_save_path, 'train_pairwise_decoder')
        DirProcessor.create_dir(tpd_main_save_path)
        tpd_save_path, tpd_i_setting = get_setting_full_path(args, conf, tpd_main_save_path)

        tpd_save_fname = f'train_pairwise_decoder_setting_{tpd_i_setting}.pt'
        tpd_save_full_fname = os.path.join(tpd_save_path, tpd_save_fname)

        # locate encoded database
        encode_test_conf = {
            'encode_tgt_epochs': args.encode_tgt_epochs,
            'train_A': train_conf['A'],
            'train_B': train_conf['B'],
            'A': A,
            'B': B
        }
        encode_test_main_save_path = os.path.join(train_save_path, 'encode_test')
        encode_test_save_path, encode_test_i_setting = get_setting_full_path(args, encode_test_conf,
                                                                             encode_test_main_save_path)

        encode_test_save_fname = f'encode_test_setting_{encode_test_i_setting}'
        if args.encode_first_n is not None:
            encode_test_save_fname += f'_0-{args.encode_first_n - 1}'  # 下标从0开始
        encode_test_save_fname += '.npz'
        encode_test_save_full_fname = os.path.join(encode_test_save_path, encode_test_save_fname)

        # Create save path
        conf = {
            'encode_test_save_full_fname': encode_test_save_full_fname
        }
        main_save_path = os.path.join(tpd_save_path, 'shen_build_indices')
        DirProcessor.create_dir(main_save_path)
        save_path, i_setting = get_setting_full_path(args, conf, main_save_path)



if __name__ == '__main__':

    # 注意：这个script只能生成出来要跑的代码然后手动粘贴到shell（因为涉及到cd）
    # 基础用法：
    # 用于训练 python $VQ_CODE/slurm/generate_slurm_cmds.py --mode train --ds_loop 100000 --print_anomalies
    # 用于shen_search: python $VQ_CODE/slurm/generate_slurm_cmds.py --mode shen_search

    parser = argparse.ArgumentParser()

    # 注意：此为必须参数
    parser.add_argument('--mode', choices=['train', 'encode_train', 'encode_test',
                                           'train_qinco_aq', 'qinco_aq_compute_distances',
                                           'train_pairwise_decoder',
                                           'shen_build_indices', 'shen_compute_distances',
                                           'shen_search', 'ivf_centroids'], type=str, required=True)

    parser.add_argument('--all_datasets', nargs='+', default=list(default_data_info.keys()))

    parser.add_argument('--all_M', nargs='+', type=int, default=None)
    parser.add_argument('--all_K', nargs='+', type=int, default=[16, 64, 256, 1024, 4096])
    parser.add_argument('--min_M', type=int, default=1)
    parser.add_argument('--max_M', type=int, default=100_000)
    parser.add_argument('--max_K', type=int, default=100_000)

    parser.add_argument('--model_size', choices=['S', 'M', 'L'], default='L')
    parser.add_argument('--A', type=int, default=None)
    parser.add_argument('--B', type=int, default=None)
    parser.add_argument('--train_A', type=int, default=None)    # 注意：这两个参数只在mode不是'train'的时候使用，目的是指出要用哪一个train setting下的A和B来做testing; testing用的A和B通过"--A""--B"来设置
    parser.add_argument('--train_B', type=int, default=None)
    parser.add_argument('--L', type=int, default=None)
    parser.add_argument('--de', type=int, default=None)
    parser.add_argument('--dh', type=int, default=None)

    parser.add_argument('--n_train', type=float, default=None)
    parser.add_argument('--n_val', type=float, default=None)
    parser.add_argument('--n_db', type=float, default=None)
    parser.add_argument('--n_queries', type=float, default=None)

    # parameters for 'train'
    parser.add_argument('--ds_loop', type=int, default=10_000_000)
    parser.add_argument('--lr', type=float, default=0.0008)
    parser.add_argument('--max_epochs', type=int, default=60)   # 注意：这个值实际是要+10的
    parser.add_argument('--save_every_epochs', type=int, default=5)
    parser.add_argument('--use_ivf_centroids', action='store_true')
    parser.add_argument('--max_sec_per_epoch', type=int, default=3600)  # each epoch should not exceed one hour

    # parameters for 'encode_train' and 'encode_db'

    # 注意：encode_tgt_epochs如果不是None，会只对某一个特定的training epochs数做对应的checkpoint做，
    #  而且这个epoch是真实的epoch，而非需要另外+10的，因此应该用70而非60；
    #  如果是None, 对全部checkpoints都做
    parser.add_argument('--encode_tgt_epochs', type=int, default=70)
    parser.add_argument('--encode_first_n', type=int, default=None)

    # parameters for 'train_qinco_aq'
    parser.add_argument('--tqa_n_train', type=float, default=None)
    parser.add_argument('--tqa_n_val', type=float, default=None)

    # parameters for 'train_pairwise_decoder'
    parser.add_argument('--tpd_n_train', type=float, default=None)
    parser.add_argument('--tpd_n_val', type=float, default=None)



    # shen search parameters
    parser.add_argument('--first_n_queries', type=int, default=1000)
    parser.add_argument('--first_n_db', type=int, default=10000)
    parser.add_argument('--shen_search_batch_size', type=int, default=64)

    # IVF centroids parameters
    parser.add_argument('--ivf_k', type=int, default=1048576)
    parser.add_argument('--ivf_centroid_trainset', type=int, default=100_000)
    parser.add_argument('--ivf_centroid_valset', type=int, default=10_000)

    parser.add_argument('--max_sec_total', type=int, default=18000) # total time limited to 5 hours

    parser.add_argument('--slurm_time', default='20:00:00') # 注意：要故意把slurm_time设置得比max_sec_total更长一些
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--print_anomalies', action='store_true')  # just for training
    parser.add_argument('--force_submit_to_slurm', action='store_true') # for the CPU-based methods; just for testing purposes in case local memory is not enough. Don't use this!

    parser.add_argument('--code_path', default='$VQ_CODE')
    parser.add_argument('--result_path', default=default_result_path)
    parser.add_argument('--bash_full_fname', default=os.path.join(default_code_path, 'slurm', 'all_cmds.sh'))
    parser.add_argument('--overwrite_existing_bash', action='store_true')

    args = parser.parse_args()

    if args.mode == 'train':
        slurm_t = datetime.strptime(args.slurm_time, '%H:%M:%S')
        slurm_secs = timedelta(hours=slurm_t.hour, minutes=slurm_t.minute, seconds=slurm_t.second).total_seconds()
        if slurm_secs < args.max_sec_total * 1.1:
            raise ValueError('Set slurm time to at least 1.1 times of the total time limit to account for overhead.')

    # A, B, train_A, train_B, L, de, dh = get_hyperparam(args)

    all_M_by_dataset = {
        dataset: [M for M in (args.all_M if args.all_M is not None else default_data_info[dataset]['all_M'])
                  if M <= args.max_M and M >= args.min_M]
        for dataset in args.all_datasets
    }

    cmds = []
    early_stop_logs, nan_logs = [], []
    no_model_logs = []
    no_encode_logs = []
    no_tpd_logs = []
    no_tqa_logs = []
    for dataset, K in itertools.product(args.all_datasets, args.all_K):
        if K > args.max_K:
            continue
        all_M = all_M_by_dataset[dataset]
        for M in all_M:
            if args.mode == 'train':
                cur_cmds, cur_early_stop_log, cur_nan_log = generate_cmds(args, dataset, M, K)
                if cur_cmds is not None:
                    cmds += cur_cmds   # 注意不是append
                if cur_early_stop_log is not None:
                    early_stop_logs.append(cur_early_stop_log)
                if cur_nan_log is not None:
                    nan_logs.append(cur_nan_log)
            elif args.mode in ('encode_train', 'encode_test', 'encode_queries'):
                cur_cmds, cur_no_model_log = generate_cmds(args, dataset, M, K)
                if cur_cmds is not None:
                    cmds += cur_cmds   # 注意不是append
                if cur_no_model_log is not None:
                    no_model_logs.append(cur_no_model_log)
            elif args.mode in ('train_pairwise_decoder', 'train_qinco_aq'):
                cur_cmds, cur_no_encode_log = generate_cmds(args, dataset, M, K)
                if cur_cmds is not None:
                    cmds += cur_cmds   # 注意不是append
                if cur_no_encode_log is not None:
                    no_encode_logs.append(cur_no_encode_log)
            elif args.mode == 'shen_build_indices':
                cur_cmds, cur_no_encode_log, cur_no_tpd_log = generate_cmds(args, dataset, M, K)
                if cur_cmds is not None:
                    cmds += cur_cmds   # 注意不是append
                if cur_no_encode_log is not None:
                    no_encode_logs.append(cur_no_encode_log)
                if cur_no_tpd_log is not None:
                    no_tpd_logs.append(cur_no_tpd_log)
            elif args.mode == 'qinco_aq_compute_distances':
                cur_cmds, cur_no_encode_log, cur_no_tqa_log = generate_cmds(args, dataset, M, K)
                exec(gen_cmd_print_variables('cur_cmds, cur_no_encode_log, cur_no_tqa_log'))
                if cur_cmds is not None:
                    cmds += cur_cmds   # 注意不是append
                if cur_no_encode_log is not None:
                    no_encode_logs.append(cur_no_encode_log)
                if cur_no_tqa_log is not None:
                    no_tqa_logs.append(cur_no_tqa_log)
            else:
                raise NotImplementedError()


    print('\n\n\n******** All done: the complete commands are as follows ***********\n')
    for cmd in cmds:
        print(cmd)

    # save the commands to bash file
    if args.overwrite_existing_bash or (not os.path.exists(args.bash_full_fname)):
        with open(args.bash_full_fname, "w") as f:
            f.write("#!/bin/bash\n")
    with open(args.bash_full_fname, "a") as f:
        for cmd in cmds:
            f.write(f'{cmd}\n')


    if args.print_anomalies:
        if args.mode == 'train':
            print('\n\n\n####### Early stopping recorded in the following logs #########\n')
            for log in early_stop_logs:
                print(log)

            print('\n\n\n!!!!!!! NaN recorded in the following logs !!!!!!!!!!\n')
            for log in nan_logs:
                print(log)
        elif args.mode in ('encode_train', 'encode_test', 'encode_queries'):
            print('\n\n\n####### No trained model found in the following logs #########\n')
            for log in no_model_logs:
                print(log)
        elif args.mode in ('train_pairwise_decoder', 'train_qinco_aq'):
            print('\n\n\n####### No encoded training set found in the following logs #########\n')
            for log in no_encode_logs:
                print(log)
        elif args.mode == 'shen_build_indices':
            print('\n\n\n####### No encoded database found in the following logs #########\n')
            for log in no_encode_logs:
                print(log)

            print('\n\n\n####### No pairwise decoder found in the following logs #########\n')
            for log in no_tpd_logs:
                print(log)
        elif args.mode == 'qinco_aq_compute_distances':
            print('\n\n\n####### No encoded database found in the following logs #########\n')
            for log in no_encode_logs:
                print(log)

            print('\n\n\n####### No trained AQ decoder found in the following logs #########\n')
            for log in no_tqa_logs:
                print(log)


