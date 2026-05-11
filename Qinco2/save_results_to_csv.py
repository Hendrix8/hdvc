import os
import sys

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from util.var_util import *
import argparse
from util.file_util import FileWriter, FileReader, DirProcessor
import itertools
# import matplotlib.pyplot as plt
# import numpy as np
# import re
from util.test_util import *
# from datetime import datetime, timedelta
# from typing import Literal
from slurm.generate_slurm_cmds import get_hyperparam, get_setting_full_path
import pandas as pd


def collect_results(args, dataset, M, K, n_train):
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
    if n_train is not None:
        assert args.n_val is not None
        train_conf['n_train'] = n_train
        train_conf['n_val'] = args.n_val

    ######### Find the training path #############
    train_save_path, train_i_setting = get_setting_full_path(args, train_conf, train_main_save_path, verbose=False)
    train_save_id = f'{main_save_id}_setting_{train_i_setting}'

    if args.mode == 'qinco_aq_compute_distances':

        # locate trained pairwise decoder
        encode_train_conf = {
            'encode_tgt_epochs': args.encode_tgt_epochs,
            'train_A': train_conf['A'],
            'train_B': train_conf['B']
        }
        encode_train_save_path, encode_train_i_setting = get_setting_full_path(
            args, encode_train_conf, os.path.join(train_save_path, 'encode_train'), verbose=False)

        conf = {}
        if args.tqa_n_train is not None:
            assert args.tqa_n_val is not None
            conf['tqa_n_train'] = args.tqa_n_train
            conf['tqa_n_val'] = args.tqa_n_val

        tqa_main_save_path = os.path.join(encode_train_save_path, 'train_qinco_aq')
        DirProcessor.create_dir(tqa_main_save_path)
        tqa_save_path, tqa_i_setting = get_setting_full_path(args, conf, tqa_main_save_path, verbose=False)

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
                                                                             encode_test_main_save_path, verbose=False)

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
        save_path, i_setting = get_setting_full_path(args, conf, main_save_path, verbose=False)

        save_fname = f'{args.mode}_setting_{i_setting}.npz'
        save_full_fname = os.path.join(save_path, save_fname)

        run_log_fname = 'run.log'
        run_log_full_fname = os.path.join(save_path, run_log_fname)

        if not os.path.exists(run_log_full_fname) or not os.path.exists(save_full_fname):
            run_finished = False
        else:
            if FileReader.str_in_file('Task done', run_log_full_fname):
                run_finished = True
            else:
                run_finished = False

        if not run_finished:
            return None, None
        return np.load(save_full_fname), save_path
    else:
        raise NotImplementedError



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # required arguments
    parser.add_argument('--mode', choices=['qinco_aq_compute_distances',], type=str, required=True)

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

    parser.add_argument('--all_n_train', nargs='+', type=float, default=None)   # used for varying train sizes
    # parser.add_argument('--n_train', type=float, default=None)
    parser.add_argument('--n_val', type=float, default=None)
    parser.add_argument('--n_db', type=int, default=10000)
    parser.add_argument('--n_queries', type=int, default=1000)

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

    # # parameters for 'train_pairwise_decoder'
    # parser.add_argument('--tpd_n_train', type=float, default=None)
    # parser.add_argument('--tpd_n_val', type=float, default=None)

    # # shen search parameters
    # parser.add_argument('--first_n_queries', type=int, default=1000)
    # parser.add_argument('--first_n_db', type=int, default=10000)
    # parser.add_argument('--shen_search_batch_size', type=int, default=64)

    # IVF centroids parameters
    parser.add_argument('--ivf_k', type=int, default=1048576)
    parser.add_argument('--ivf_centroid_trainset', type=int, default=100_000)
    parser.add_argument('--ivf_centroid_valset', type=int, default=10_000)

    parser.add_argument('--result_path', default=default_result_path)
    parser.add_argument('--csv_save_path', default=None)    # if None, save to the current path


    args = parser.parse_args()

    if args.all_n_train is not None:
        assert len(args.all_n_train) > 1

    all_M_by_dataset = {
        dataset: [M for M in (args.all_M if args.all_M is not None else default_data_info[dataset]['all_M'])
                  if M <= args.max_M and M >= args.min_M]
        for dataset in args.all_datasets
    }

    n_train_by_dataset = {
        'bigann': 100_0000,
        'deep': 100_0000,
        'gist': 500_000,
        'msmarco': 100_0000,
        'openai': 100_0000,
    }
    nb_by_dataset = {
        'bigann': 100_0000,
        'deep': 100_0000,
        'gist': 100_0000,
        'msmarco': 100_0000,
        'openai': 100_0000,
    }
    dim_by_dataset = {
        'bigann': 128,
        'deep': 96,
        'gist': 960,
        'msmarco': 1024,
        'openai': 1536,
    }

    all_n_train = args.all_n_train if args.all_n_train is not None else [None]
    headers = [
        'method', 'dataset', 'experiment_folder', 'nq', 'nb', 'nb_sample', 'dim', 'n_subquantizers', 'nbits',
        'bits_per_vector', 'train_size', 'train_time_s', 'encoding_time_s', 'distance_table_time_s',
        'cdist_time_s', 'adc_time_s', 'rel_error_mean', 'rel_error_std',
    ]

    for dataset in args.all_datasets:

        df = pd.DataFrame(columns=headers)

        all_M = all_M_by_dataset[dataset]
        for M, K, n_train in itertools.product(all_M, args.all_K, all_n_train):
            if K > args.max_K:
                continue

            exec(gen_cmd_print_variables('dataset, M, K, n_train'))

            if args.mode == 'qinco_aq_compute_distances':
                if n_train == 1:
                    n_train = None

                results, experiment_folder = collect_results(args, dataset, M, K, n_train)
                if results is None:
                    continue

                if n_train is None:
                    n_train = 1.

                result_dict = {
                    'method': 'QINCo2',
                    'dataset': dataset,
                    'experiment_folder': experiment_folder,
                    'nq': args.n_queries,
                    'nb': nb_by_dataset[dataset],
                    'nb_sample': args.n_db,
                    'dim': dim_by_dataset[dataset],
                    'n_subquantizers': M,
                    'nbits': int(np.log2(K)),
                    'bits_per_vector': M * int(np.log2(K)),
                    'train_size': int(n_train) if n_train is not None and n_train > 1 else int(np.rint(n_train * n_train_by_dataset[dataset])),
                    'train_time_s': -1,
                    'encoding_time_s': -1,
                    'distance_table_time_s': np.round(results['total_distance_table_time_s'], 6),
                    'cdist_time_s': np.round(results['total_cdist_time_s'], 6),
                    'adc_time_s': np.round(results['total_adc_time_s'], 6),
                    'rel_error_mean': np.round(results['avg_relative_error'], 6),
                    'rel_error_std': np.round(results['std_relative_error'], 6),

                    # comment these out if these cause trouble plotting the figures
                    'recon_mse_mean': np.round(results['avg_recon_mse_no_scale'], 6),
                    'recon_mse_std': np.round(results['std_recon_mse_no_scale'], 6),
                }
                exec(gen_cmd_print_variables('df'))
                df.loc[len(df)] = result_dict
                exec(gen_cmd_print_variables('df'))

            else:
                raise NotImplementedError()

        save_full_fname = f'{dataset}_QINCo2_adc_vs_exact_eval.csv'
        if args.csv_save_path is not None:
            save_full_fname = os.path.join(args.csv_save_path, save_full_fname)
        df.to_csv(save_full_fname, index=False)

    print('All done!')