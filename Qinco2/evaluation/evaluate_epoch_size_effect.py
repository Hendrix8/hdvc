import itertools
import sys

import numpy as np

sys.path.append('/lustre/fswork/projects/rech/thj/uth68ud/PycharmProjects/Qinco2')
import os

from util.test_util import *
from util.var_util import *
import argparse
from util.file_util import FileReader, FileWriter, DirProcessor
# from util.evaluation_util import prf
from typing import Literal
from util.evaluation_util import MetricComparer
from scipy.stats import rankdata
import pickletools
from slurm.generate_slurm_cmds import get_hyperparam, get_setting_full_path


class Evaluator(object):

    def __init__(
            self, args,
    ):
        self.args = args

        self.metric_aliases = { # 用作ylabel
            'Average relative error': 'Relative error',
            'reconstruction mse': 'MSE',
            'Recall@1': 'R@1',
            'Recall@10': 'R@10',
            'Recall@100': 'R@100',
            'Per-batch ADC time': 'ADC time'
        }
        self.metric_abbvs = {   # 这个是专门用来放在文件名里的，中间没有空格
            'Average relative error': 'Rel_error',
            'reconstruction mse': 'recon_mse',
            'Recall@1': 'R@1',
            'Recall@10': 'R@10',
            'Recall@100': 'R@100',
            'Per-batch ADC time': 'ADC_time'
        }

    def load_results(
            self, dataset, M, K, epoch_size
    ):

        train_main_save_path = os.path.join(default_result_path, f'{dataset}_M_{M}_K_{K}')
        DirProcessor.create_dir(train_main_save_path)

        args = deepcopy(self.args)
        delattr(args, 'all_epoch_sizes')
        args.epoch_size = epoch_size

        A, B, train_A, train_B, L, de, dh = get_hyperparam(self.args)
        train_conf = {
            'M': M, 'K': K,
            'Base_model': f'Qinco2-{args.model_size}',
            'A': train_A, 'B': train_B, 'L': L, 'de': de, 'dh': dh,
            'epoch_size': args.epoch_size, 'epochs': args.epochs,
            'lr': args.lr,
        }
        train_save_path, train_i_setting = get_setting_full_path(train_conf, train_main_save_path)
        test_main_save_path = os.path.join(train_save_path, args.mode)

        conf = deepcopy(train_conf)
        # Qinco2 uses different A and B for training and testing
        if args.mode == 'shen_search':
            conf['train_A'] = conf['A']
            conf['train_B'] = conf['B']
            conf['A'] = A
            conf['B'] = B

            conf['first_n_queries'] = args.first_n_queries
            conf['first_n_db'] = args.first_n_db
            conf['shen_search_batch_size'] = args.shen_search_batch_size

            result_save_path, i_setting = get_setting_full_path(conf, test_main_save_path)
            results = FileReader.load_dict_from_json(os.path.join(result_save_path, 'metrics.json'))
        else:
            raise NotImplementedError

        return results

    def get_metric_tensor(self, metric, tgt_dataset=None, tgt_M=None, tgt_K=None):

        metric_tensor = [] # (M, K) -> dataset -> epoch_size
        for M, K in itertools.product(self.args.all_M, self.args.all_K):
            if tgt_M is not None and M != tgt_M:
                continue
            if tgt_K is not None and K != tgt_K:
                continue

            metric_mat = []
            for dataset in self.args.all_datasets:
                if tgt_dataset is not None and dataset != tgt_dataset:
                    continue

                metric_vec = []
                for epoch_size in self.args.all_epoch_sizes:
                    metric_vec.append(self.load_results(dataset, M, K, epoch_size)[metric])
                metric_mat.append(metric_vec)
            metric_tensor.append(metric_mat)
        metric_tensor = np.array(metric_tensor)
        if metric == 'reconstruction mse':
            metric_tensor /= np.max(metric_tensor, axis=-1, keepdims=True)
        return metric_tensor

    def plot_figure(
        self, metrics=['reconstruction mse', 'Average relative error', 'Recall@1', 'Recall@10', 'Recall@100', 'Per-batch ADC time'],
    ):

        metric_comparor = MetricComparer(self.args.all_datasets)

        for metric in metrics:
            metric_tensor = self.get_metric_tensor(metric)
            fig_save_path = os.path.join('/lustre/fswork/projects/rech/thj/uth68ud/PycharmProjects/Qinco2/results/figures', args.mode)
            DirProcessor.create_dir(fig_save_path)

            for i_MK, (M, K), in enumerate(itertools.product(self.args.all_M, self.args.all_K)):
                metric_mat = metric_tensor[i_MK][np.newaxis, :, :]  # (1, n_datasets, n_epoch_sizes)

                metric_comparor.plot_line_chart_from_metric_matrix(
                    metric_mat,

                    title=metric,

                    fig_height=8.5,
                    fig_width_per_subplot=22,
                    n_subplot_rows=1,
                    tight_layout_h_pad=2,
                    tight_layout_w_pad=22,
                    tight_layout_rect=(0, 0, 1, 1),

                    include_unified_legend=True,

                    xlabel='Examples per epoch',
                    xticks=np.arange(len(self.args.all_epoch_sizes)),
                    xtick_labels=[fr"$10^{{{int(np.log10(epoch_size))}}}$" for epoch_size in self.args.all_epoch_sizes],
                    ylabel=self.metric_aliases[metric],

                    linewidth=5,
                    markersize=30,
                    xtick_fontsize=70,
                    ytick_fontsize=70,
                    xlabel_fontsize=85,
                    ylabel_fontsize=85,
                    xlabel_pad=20,
                    ylabel_pad=20,
                    ylim=None,
                    title_fontsize=80,
                    title_pad=20,

                    legend_loc='upper center',
                    # See here for other options: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html
                    legend_bbox_to_anchor=(0.5, 1.6),
                    legend_n_cols=3,
                    legend_fontsize=80,
                    legend_columnspacing=1,
                    legend_frameon=False,
                    legend_title=None,  # 是的，legend也可以加title
                    legend_title_fontsize=55,

                    fig_save_fname=os.path.join(fig_save_path, f'{self.metric_abbvs[metric]}_varying_epoch_size_M_{M}_K_{K}.png')
                )


if __name__ == '__main__':

    # Usage: python $VQ_CODE/evaluation/evaluate_epoch_size_effect.py

    parser = argparse.ArgumentParser()

    # 注意：此为必须参数
    parser.add_argument('--mode', type=str, default='shen_search')

    parser.add_argument('--all_datasets', nargs='+', default=list(default_data_info.keys()))

    parser.add_argument('--all_M', type=int, nargs='+', default=[8])
    parser.add_argument('--all_K', type=int, nargs='+', default=[256])

    parser.add_argument('--model_size', choices=['S', 'M', 'L'], default='L')
    parser.add_argument('--A', type=int, default=None)
    parser.add_argument('--B', type=int, default=None)
    parser.add_argument('--train_A', type=int,
                        default=None)  # 注意：这两个参数只在mode不是'train'的时候使用，目的是指出要用哪一个train setting下的A和B来做testing; testing用的A和B通过"--A""--B"来设置
    parser.add_argument('--train_B', type=int, default=None)
    parser.add_argument('--L', type=int, default=None)
    parser.add_argument('--de', type=int, default=None)
    parser.add_argument('--dh', type=int, default=None)

    # train parameters
    parser.add_argument('--all_epoch_sizes', nargs='+', type=int, default=[100_000, 1_000_000, 10_000_000])
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--lr', type=float, default=0.0008)

    # shen search parameters
    parser.add_argument('--first_n_queries', type=int, default=1000)
    parser.add_argument('--first_n_db', type=int, default=10000)
    parser.add_argument('--shen_search_batch_size', type=int, default=64)

    parser.add_argument('--overwrite', action='store_true')

    args = parser.parse_args()


    evaluator = Evaluator(args)
    # evaluator.generate_main_tabular_body(num_selected_of_interest=num_selected_of_interest)
    # evaluator.gen_ssl_data(args, channel_id)
    evaluator.plot_figure()
