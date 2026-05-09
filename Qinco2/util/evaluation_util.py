# -*- coding: UTF-8 -*-

import sys

from ray import method

sys.path.append('/linkhome/rech/genlpd01/uth68ud/2-eeg-shen')

from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from util.test_util import *
from util.miscellaneous_util import nested_dict_to_nested_list

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from util.visualization_util import create_subplot_layout, row_major_legend

import operator
import math
from scipy.stats import wilcoxon
from scipy.stats import friedmanchisquare
import networkx
import pandas as pd
from scipy.stats import rankdata


def prf(real: np.ndarray, pred: np.ndarray):
    p = precision_score(real, pred)
    r = recall_score(real, pred)
    f1 = f1_score(real, pred)

    return p, r, f1


class MetricComparer(object):

    def __init__(self, all_method_ids, all_run_ids=None, metric_id='f1', all_indicators=None):
        self.all_method_ids = [str(method_id) for method_id in all_method_ids]
        self.all_run_ids = None if all_run_ids is None else [str(run_id) for run_id in all_run_ids]
        self.metric_id = metric_id

        self.n_methods = len(self.all_method_ids)

        factor_by_method_id = {
            'precision': 1,  # 1表示这个metric越大越好，否则越小越好
            'recall': 1,
            'f1': 1,
        }
        assert metric_id in factor_by_method_id.keys()
        self.method_factor = factor_by_method_id[metric_id]

        self.factor_by_indicator = {
            'avg_metric': 1,  # 1表示对于method_factor为1的metric, indicator越大越好，否则越小越好
            'avg_rank': -1,
            'n_wins': 1
        }
        self.calculator_by_indicator = {
            'avg_metric': lambda metric_mat: np.nanmean(self.method_factor * metric_mat, axis=1),
            'avg_rank': lambda metric_mat: np.nanmean(
                rankdata(-self.method_factor * metric_mat, method='min', axis=0),
                axis=1
            ),
            'n_wins': lambda metric_mat: np.bincount(np.nanargmax(self.method_factor * metric_mat, axis=0),
                                                     minlength=self.n_methods)
        }
        assert sorted(list(self.factor_by_indicator.keys())) == sorted(list(self.calculator_by_indicator.keys()))
        if all_indicators is not None:
            self.all_indicators = [all_indicators] if isinstance(all_indicators, str) else all_indicators
            assert all(indicator in self.factor_by_indicator.keys() for indicator in self.all_indicators)
        else:
            self.all_indicators = list(self.factor_by_indicator.keys())

        self.config_plotting()


    def config_plotting(self):
        self.linestyles = ['-', '--', ':',]
        self.markers = ['s', 'd', '>', '<', 'X']
        self.colors = np.array([
            [27, 158, 119],
            [217, 95, 2],
            [117, 112, 179],
            [102, 102, 102],
            [102, 166, 30],
            [230, 171, 2],
            [231, 41, 138],
            [166, 118, 29],
            [152, 78, 163],
            [55, 126, 184],
        ]) / 255

        self.n_linestyles = len(self.linestyles)
        self.n_markers = len(self.markers)
        self.n_colors = len(self.colors)

    def compare_from_metric_mat(
            self,
            metric_mat_: np.ndarray,    # (n_methods, n_runs)
            print_detailed_results=True,
            value_to_ignore=np.inf, # 默认没有要ignore的
    ):

        assert metric_mat_.ndim == 2
        assert len(metric_mat_) == self.n_methods
        n_methods, n_runs = metric_mat_.shape

        metric_mat = deepcopy(metric_mat_)
        metric_mat[metric_mat == value_to_ignore] = np.nan

        summary_results = {}  # summary_results: only include best results
        detailed_results = {} # 以每一个indicator为key, val是一个子dict: method_id -> 该method_id的indicator value（即score)

        for indicator in self.all_indicators:

            indicator_factor, indicator_calculator = (self.factor_by_indicator[indicator],
                                                      self.calculator_by_indicator[indicator])

            i_methods_to_ignore = np.isnan(metric_mat).all(axis=1)
            i_methods_to_keep = np.setdiff1d(np.arange(n_methods), i_methods_to_ignore, assume_unique=True)

            score_by_method = np.empty(n_methods)
            score_by_method[i_methods_to_ignore] = np.nan
            score_by_method[i_methods_to_keep] = indicator_calculator(metric_mat[i_methods_to_keep])
            detailed_results[indicator] = dict(zip(self.all_method_ids, score_by_method))

            i_best = np.nanargmax(indicator_factor * score_by_method)
            best_method_id, best_score = self.all_method_ids[i_best], score_by_method[i_best]
            summary_results[indicator] = {
                'i_best': i_best,
                'best_method_id': best_method_id,
                'best_score': best_score,
            }

            print(f'\nUnder indicator {indicator}, the best method is {best_method_id} with a score of {best_score}.', end=' ')
            if print_detailed_results:
                print(f'Details below ({"larger" if indicator_factor == 1 else "smaller"} is better):')
                for method_id, score in zip(self.all_method_ids, score_by_method):
                    print(f'{method_id}: {score}')

        print()
        return summary_results, detailed_results


    def compare_from_metric_matrix(
            self,
            metric_matrix: np.ndarray,  # 注意：必须是(n_settings, n_methods, n_runs), 比方说：(n_ied_nets, n_data_selection_methods, n_data_ids)
            all_setting_ids, # 要比较的全部settings
            value_to_ignore=np.inf,
            print_detailed_results=True,
            additional_header_info_in_final_output='',
    ):
        assert len(all_setting_ids) == metric_matrix.shape[0]
        assert self.n_methods == metric_matrix.shape[1]
        n_settings, n_methods, n_runs = metric_matrix.shape
        assert n_methods == self.n_methods

        run_lev_summary_results_by_setting = {} # setting_id -> 当前setting下各个runs的summary_results
        run_lev_detailed_results_by_setting = {}   # setting_id -> 当前setting下各个runs的detailed_results（即setting_id -> indicator -> method_id: 标量indicator_val
        for setting_id, metric_mat in zip(all_setting_ids, metric_matrix):

            print(f'\n******** Setting: {setting_id} **********')

            summary_results, detailed_results = self.compare_from_metric_mat(
                metric_mat, value_to_ignore=value_to_ignore, print_detailed_results=print_detailed_results)
            run_lev_summary_results_by_setting[setting_id] = summary_results
            run_lev_detailed_results_by_setting[setting_id] = detailed_results

        if additional_header_info_in_final_output != '':
            additional_header_info_in_final_output = f'({additional_header_info_in_final_output}) ' # 注意：结尾的空格不要删除
        print(f'\n********** Final setting level results {additional_header_info_in_final_output}**********')
        setting_lev_summary_results = {}    # indicator -> (i_best, best_method_id, best_score): 注意：这里的best是相对于setting的，是在全部setting中平均下来（其实也就是全部(setting, run)pairs中最好的那一个）
        setting_lev_detailed_results = {}   # indicator -> method_id -> setting_lev_score (每一个settings下得到的scores的均值）
        for indicator in self.all_indicators:
            setting_lev_score_mat = np.array(nested_dict_to_nested_list(
                run_lev_detailed_results_by_setting,
                [all_setting_ids, indicator, self.all_method_ids]
            )).T # 注意：一定要转置
            assert setting_lev_score_mat.shape == (n_methods, n_settings)

            setting_lev_scores_by_method = np.mean(setting_lev_score_mat, axis=1)
            setting_lev_detailed_results[indicator] = dict(zip(self.all_method_ids, setting_lev_scores_by_method))

            indicator_factor = self.factor_by_indicator[indicator]
            i_best = np.argmax(indicator_factor * setting_lev_scores_by_method)
            best_method_id, best_score = self.all_method_ids[i_best], setting_lev_scores_by_method[i_best]
            setting_lev_summary_results[indicator] = {
                'i_best': i_best,
                'best_method_id': best_method_id,
                'best_score': best_score,
            }

            print(f'\nUnder indicator {indicator}, the best method over all under all settings '
                  f'is {best_method_id} with a score of {best_score}.',
                  end=' ')
            if print_detailed_results:
                print(f'Details below ({"larger" if indicator_factor == 1 else "smaller"} is better):')
                for method_id, score in zip(self.all_method_ids, setting_lev_scores_by_method):
                    print(f'{method_id}: {score}')

        print()
        return (setting_lev_summary_results, setting_lev_detailed_results,
                run_lev_summary_results_by_setting, run_lev_detailed_results_by_setting)


    def plot_line_chart_from_metric_mat(
            self,
            ax,
            metric_mat: np.ndarray,    # (n_methods, n_runs). 横轴对应runs，每个method用一条折线
            std_mat=None,
            include_legend=True,
            title=None,

            # all_method_ids=None,

            xticks: list | tuple | np.ndarray = None,
            xtick_labels=None, # 对应runs. 注意：如果是None，会被default到self.all_run_ids上去
            xtick_label_rotation=0,
            xlabel=None,
            ylabel=None,

            linewidth=3,
            markersize=12,
            xtick_fontsize=20,
            ytick_fontsize=20,
            xlabel_fontsize=20,
            ylabel_fontsize=20,
            xlabel_pad=20,
            ylabel_pad=20,
            xlim=None,
            ylim=None,
            title_fontsize=25,
            title_pad=20,

            strawman_method_ids=None,   # 需要特殊画的，比方说fully_supervised
            strawman_linestyles=None,
            strawman_markers=None,
            strawman_colors=None,

            legend_loc='upper right',   # See here for other options: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html
            legend_bbox_to_anchor=None,
            legend_n_cols=1,
            legend_fontsize=35,
            legend_frameon=False,
            legend_columnspacing=2.,    # 用于控制每列legend之间的间距
            legend_title=None,  # 是的，legend也可以加title
            legend_title_fontsize=20,
    ):
        # exec(gen_cmd_print_variables('metric_mat.shape, len(self.all_method_ids), len(xticks)'))
        # if all_method_ids is None:
        #     all_method_ids = self.all_method_ids

        n_methods, n_runs = metric_mat.shape
        exec(gen_cmd_print_variables('n_methods, n_runs, len(xticks)'))
        assert len(xticks) == n_runs
        if xtick_labels is not None:
            assert len(xtick_labels) == n_runs

        # 注意：这里指的是，如果metric_mat有nan，那么nan必须集中在最后几行，
        # 且它的最后几行必须全部是nan，这是为了在多个subplots情况下，部分子图需要临时删掉若干行的时候准备的
        if np.any(np.isnan(metric_mat)):

            method_has_nan = np.isnan(metric_mat).any(axis=1)
            nan_i_method = np.where(method_has_nan)[0]

            assert np.all(np.isnan(metric_mat[method_has_nan]))
            first_nan_i_method = nan_i_method[0]
            assert np.array_equal(nan_i_method, np.arange(first_nan_i_method, n_methods))
        else:
            first_nan_i_method = n_methods

        if strawman_method_ids is not None:
            assert metric_mat.shape[0] == len(self.all_method_ids)
            assert (len(strawman_method_ids) == len(strawman_linestyles) ==
                    len(strawman_markers) == len(strawman_colors))
            exec(gen_cmd_print_variables('np.array(self.all_method_ids), strawman_method_ids'))
            all_i_strawman = np.array([
                np.where(np.array(self.all_method_ids) == strawman)[0][0] for strawman in strawman_method_ids
            ])
            # 注意：strawmans必须在最后几个不是nan的行上面
            exec(gen_cmd_print_variables('first_nan_i_method, len(all_i_strawman), all_i_strawman'))
            assert (np.sort(all_i_strawman) == np.arange(first_nan_i_method - len(all_i_strawman), first_nan_i_method)).all()
        else:
            all_i_strawman = []
        n_strawmans = len(all_i_strawman)


        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels if xtick_labels is not None else self.all_run_ids)
        for xtick_label in ax.get_xticklabels():
            xtick_label.set_rotation(xtick_label_rotation)

        ax.tick_params(axis='x', labelsize=xtick_fontsize)
        ax.tick_params(axis='y', labelsize=ytick_fontsize)
        exec(gen_cmd_print_variables('xlabel'))
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=xlabel_fontsize, labelpad=xlabel_pad)
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=ylabel_fontsize, labelpad=ylabel_pad)
        for i_method, (method_metrics, method_id) in enumerate(zip(metric_mat[:first_nan_i_method], self.all_method_ids[:first_nan_i_method])):
            if i_method < first_nan_i_method - n_strawmans:
                linestyle = self.linestyles[i_method % self.n_linestyles]
                marker = self.markers[i_method % self.n_markers]
                color = self.colors[i_method % self.n_colors]
            else:
                linestyle = strawman_linestyles[i_method - (first_nan_i_method - n_strawmans)]
                marker = strawman_markers[i_method - (first_nan_i_method - n_strawmans)]
                color = strawman_colors[i_method - (first_nan_i_method - n_strawmans)]

            if std_mat is None:
                ax.plot(xticks, method_metrics, linestyle=linestyle, linewidth=linewidth,
                        marker=marker, color=color, markersize=markersize, label=method_id)  # 注意：legend是method_id
            else:
                ax.errorbar(xticks, method_metrics, yerr=std_mat[i_method], linestyle=linestyle, linewidth=linewidth,
                        marker=marker, color=color, markersize=markersize, label=method_id)  # TODO：这个有可能要加参数
                # ax.plot(xticks, method_metrics, linestyle=self.linestyles[i_linestyle], linewidth=linewidth,
                #         marker=self.markers[i_marker], markersize=markersize, label=method_id)  # 注意：legend是method_id
                # ax.fill_between(xticks, method_metrics - std_mat[i_method], method_metrics + std_mat[i_method], alpha=0.2)

        if include_legend:
            handles, legend_labels = row_major_legend(ax, legend_n_cols)
            ax.legend(handles, legend_labels, loc=legend_loc, title=legend_title, title_fontsize=legend_title_fontsize,
                      bbox_to_anchor=legend_bbox_to_anchor, ncol=legend_n_cols,
                      fontsize=legend_fontsize, frameon=legend_frameon, columnspaceing=legend_columnspacing)

        if title is not None:
            ax.set_title(title, fontsize=title_fontsize, pad=title_pad)
        return ax, first_nan_i_method

    def plot_line_chart_from_metric_matrix(
            self,

            metric_matrix, # (n_settings, n_methods, n_runs). 每个setting一个subplot, 用plot_line_chart_from_metric_mat具体画。
            # 注意：对于不同的setting，n_methods和n_runs允许会有变化

            # 这几个参数只在需要为每个subplot定义不同的参数时使用。几个subplots公用的参数在**subplot_kwargs里
            subplot_xticks=None,
            subplot_xtick_labels=None,
            subplot_xtick_label_rotations=None,
            subplot_xlabels=None,
            subplot_xlims=None,
            subplot_titles=None,

            std_matrix=None,
            include_unified_legend = True,
            subplot_include_legend = False,

            fig_height=12,
            fig_width_per_subplot=12,
            n_subplot_rows = 1,
            tight_layout_h_pad=None,
            tight_layout_w_pad=None,
            tight_layout_rect=(0, 0, 1, 1),

            fig_save_fname=None,
            save_dpi=200,


            **subplot_kwargs    # plot_line_chart_from_metric_mat的可选参数（除了std_mat和include_legend）

    ):

        n_settings = len(metric_matrix)
        if std_matrix is not None:
            assert len(std_matrix) == len(metric_matrix)
        else:
            std_matrix = [None] * n_settings

        fig, axes = create_subplot_layout(
            n_settings, n_subplot_rows,
            fig_height, fig_width_per_subplot,
            save_dpi,
            tight_layout_h_pad=tight_layout_h_pad,
            tight_layout_w_pad=tight_layout_w_pad,
            tight_layout_rect=tight_layout_rect,
        )

        if subplot_xticks is not None:
            assert 'xticks' not in subplot_kwargs
        else:
            # assert subplot_kwargs['xticks'] is not None
            subplot_xticks = [subplot_kwargs.get('xticks', None)] * n_settings
        assert len(subplot_xticks) == n_settings

        if subplot_xtick_labels is not None:
            assert 'xtick_labels' not in subplot_kwargs
        else:
            # assert subplot_kwargs['xtick_labels'] is not None
            subplot_xtick_labels = [subplot_kwargs.get('xtick_labels', None)] * n_settings
        assert len(subplot_xtick_labels) == n_settings

        if subplot_xtick_label_rotations is not None:
            assert 'xtick_label_rotation' not in subplot_kwargs
        else:
            # assert subplot_kwargs['xtick_label_rotation'] is not None
            subplot_xtick_label_rotations = [subplot_kwargs.get('xtick_label_rotation', None)] * n_settings
        assert len(subplot_xtick_label_rotations) == n_settings

        if subplot_xlabels is not None:
            assert 'xlabel' not in subplot_kwargs
        else:
            # assert subplot_kwargs['xlabels'] is not None
            subplot_xlabels = [subplot_kwargs.get('xlabel', None)] * n_settings
        assert len(subplot_xlabels) == n_settings

        if subplot_xlims is not None:
            assert 'xlim' not in subplot_kwargs
        else:
            # assert subplot_kwargs['xlim'] is not None
            subplot_xlims = [subplot_kwargs.get('xlim', None)] * n_settings
        assert len(subplot_xlims) == n_settings

        if subplot_titles is not None:
            assert 'title' not in subplot_kwargs
        else:
            subplot_titles = [subplot_kwargs.get('title', None)] * n_settings
        assert len(subplot_titles) == n_settings

        assert 'include_legend' not in subplot_kwargs
        if include_unified_legend:
            subplot_include_legend = False  # 注意：如果用统一的legends，那么每个subplot就不用legend
        subplot_kwargs['include_legend'] = subplot_include_legend

        handles, legend_labels = None, None
        for metric_mat, std_mat, ax, title, xticks, xlim, xlabel, xtick_labels, xtick_label_rotation in (
                zip(metric_matrix, std_matrix, axes, subplot_titles, subplot_xticks, subplot_xlims, subplot_xlabels, subplot_xtick_labels, subplot_xtick_label_rotations)):
            cur_subplot_kwargs = deepcopy(subplot_kwargs)
            cur_subplot_kwargs['title'] = title
            cur_subplot_kwargs['xticks'] = xticks
            cur_subplot_kwargs['xlim'] = xlim
            cur_subplot_kwargs['xlabel'] = xlabel
            cur_subplot_kwargs['xtick_labels'] = xtick_labels
            cur_subplot_kwargs['xtick_label_rotation'] = xtick_label_rotation
            _, first_nan_i_method = (
                self.plot_line_chart_from_metric_mat(ax, metric_mat, std_mat=std_mat, **cur_subplot_kwargs))
            exec(gen_cmd_print_variables('first_nan_i_method'))
            if first_nan_i_method == self.n_methods:
                handles, legend_labels = row_major_legend(axes[0], cur_subplot_kwargs['legend_n_cols'])
        assert handles is not None and legend_labels is not None

        if include_unified_legend:
            fig.legend(handles, legend_labels, loc=subplot_kwargs['legend_loc'], title=subplot_kwargs['legend_title'],
                      title_fontsize=subplot_kwargs['legend_title_fontsize'],
                      bbox_to_anchor=subplot_kwargs['legend_bbox_to_anchor'], ncol=subplot_kwargs['legend_n_cols'],
                      fontsize=subplot_kwargs['legend_fontsize'], frameon=subplot_kwargs['legend_frameon'],
                        columnspacing=subplot_kwargs.get('legend_columnspacing', 2)
            )

        if fig_save_fname is not None:
            plt.savefig(fig_save_fname,bbox_inches='tight', dpi=save_dpi)
        return fig, axes

    # inspired from orange3 https://docs.orange.biolab.si/3/data-mining-library/reference/evaluation.cd.html
    def graph_ranks(self, ax, avranks, names, lowv=None, highv=None,
                    textspace=1, reverse=False, labels=False, ):
        """
        Draws a CD graph, which is used to display  the differences in methods'
        performance. See Janez Demsar, Statistical Comparisons of method_ids over
        Multiple Data Sets, 7(Jan):1--30, 2006.

        Needs matplotlib to work.

        The image is ploted on `plt` imported using
        `import matplotlib.pyplot as plt`.

        Args:
            avranks (list of float): average ranks of methods.
            names (list of str): names of methods.
            cd (float): Critical difference used for statistically significance of
                difference between methods.
            cdmethod (int, optional): the method that is compared with other methods
                If omitted, show pairwise comparison of methods
            lowv (int, optional): the lowest shown rank
            highv (int, optional): the highest shown rank
            width (int, optional): default width in inches (default: 6)
            textspace (int, optional): space on figure sides (in inches) for the
                method names (default: 1)
            reverse (bool, optional):  if set to `True`, the lowest rank is on the
                right (default: `False`)
            filename (str, optional): output file name (with extension). If not
                given, the function does not write a file.
            labels (bool, optional): if set to `True`, the calculated avg rank
            values will be displayed
        """
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_agg import FigureCanvasAgg
        except ImportError:
            raise ImportError("Function graph_ranks requires matplotlib.")

        # width = float(width)
        bbox = ax.get_window_extent()
        width_px = bbox.width
        dpi = ax.figure.dpi
        width = width_px / dpi

        exec(gen_cmd_print_variables('width'))

        textspace = float(textspace)

        def nth(l, n):
            """
            Returns only nth elemnt in a list.
            """
            n = lloc(l, n)
            return [a[n] for a in l]

        def lloc(l, n):
            """
            List location in list of list structure.
            Enable the use of negative locations:
            -1 is the last element, -2 second last...
            """
            if n < 0:
                return len(l[0]) + n
            else:
                return n

        def mxrange(lr):
            """
            Multiple xranges. Can be used to traverse matrices.
            This function is very slow due to unknown number of
            parameters.

            >>> mxrange([3,5])
            [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

            >>> mxrange([[3,5,1],[9,0,-3]])
            [(3, 9), (3, 6), (3, 3), (4, 9), (4, 6), (4, 3)]

            """
            if not len(lr):
                yield ()
            else:
                # it can work with single numbers
                index = lr[0]
                if isinstance(index, int):
                    index = [index]
                for a in range(*index):
                    for b in mxrange(lr[1:]):
                        yield tuple([a] + list(b))

        sums = avranks

        nnames = names
        ssums = sums

        if lowv is None:
            lowv = min(1, int(math.floor(min(ssums))))
        if highv is None:
            highv = max(len(avranks), int(math.ceil(max(ssums))))

        cline = 0.4

        k = len(sums)


        linesblank = 0
        scalewidth = width - 2 * textspace

        def rankpos(rank):
            if not reverse:
                a = rank - lowv
            else:
                a = highv - rank
            return textspace + scalewidth / (highv - lowv) * a

        distanceh = 0.25

        cline += distanceh

        # calculate height needed height of an image
        minnotsignificant = max(2 * 0.2, linesblank)
        height = cline + ((k + 1) / 2) * 0.2 + minnotsignificant

        # fig = plt.figure(figsize=(width, height))
        # fig.set_facecolor('white')
        # ax = fig.add_axes([0, 0, 1, 1])  # reverse y axis
        ax.set_axis_off()

        hf = 1. / height  # height factor
        wf = 1. / width

        def hfl(l):
            return [a * hf for a in l]

        def wfl(l):
            return [a * wf for a in l]

        # Upper left corner is (0,0).
        ax.plot([0, 1], [0, 1], c="w")
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)

        def line(l, color='k', **kwargs):
            """
            Input is a list of pairs of points.
            """
            ax.plot(wfl(nth(l, 0)), hfl(nth(l, 1)), color=color, **kwargs)

        def text(x, y, s, *args, **kwargs):
            ax.text(wf * x, hf * y, s, *args, **kwargs)

        line([(textspace, cline), (width - textspace, cline)], linewidth=2)

        bigtick = 0.3
        smalltick = 0.15
        linewidth = 2.0

        tick = None
        for a in list(np.arange(lowv, highv, 0.5)) + [highv]:
            tick = smalltick
            if a == int(a):
                tick = bigtick
            line([(rankpos(a), cline - tick / 2),
                  (rankpos(a), cline)],
                 linewidth=2)

        for a in range(lowv, highv + 1):
            text(rankpos(a), cline - tick / 2 - 0.05, str(a),
                 ha="center", va="bottom", size=16)

        k = len(ssums)

        def filter_names(name):
            return name

        space_between_names = 0.24

        for i in range(math.ceil(k / 2)):
            chei = cline + minnotsignificant + i * space_between_names
            line([(rankpos(ssums[i]), cline),
                  (rankpos(ssums[i]), chei),
                  (textspace - 0.1, chei)],
                 linewidth=linewidth)
            if labels:
                text(textspace + 0.3, chei - 0.075, format(ssums[i], '.4f'), ha="right", va="center", size=10)
            text(textspace - 0.2, chei, filter_names(nnames[i]), ha="right", va="center", size=16)

        for i in range(math.ceil(k / 2), k):
            chei = cline + minnotsignificant + (k - i - 1) * space_between_names
            line([(rankpos(ssums[i]), cline),
                  (rankpos(ssums[i]), chei),
                  (textspace + scalewidth + 0.1, chei)],
                 linewidth=linewidth)
            if labels:
                text(textspace + scalewidth - 0.3, chei - 0.075, format(ssums[i], '.4f'), ha="left", va="center",
                     size=10)
            text(textspace + scalewidth + 0.2, chei, filter_names(nnames[i]),
                 ha="left", va="center", size=16)

    def form_cliques(self, p_values, nnames):
        """
        This method forms the cliques
        """
        # first form the numpy matrix data
        m = len(nnames)
        g_data = np.zeros((m, m), dtype=np.int64)
        for p in p_values:
            if p[3] == False:
                i = np.where(nnames == p[0])[0][0]
                j = np.where(nnames == p[1])[0][0]
                min_i = min(i, j)
                max_j = max(i, j)
                g_data[min_i, max_j] = 1

        g = networkx.Graph(g_data)
        return networkx.find_cliques(g)

    def wilcoxon_holm(self, alpha=0.05, df_perf=None):
        """
        Applies the wilcoxon signed rank test between each pair of algorithm and then use Holm
        to reject the null's hypothesis
        """
        # print(pd.unique(df_perf['method_id']))
        # count the number of tested run_ids per method_id
        df_counts = pd.DataFrame({'count': df_perf.groupby(
            ['method_id']).size()}).reset_index()
        # get the maximum number of tested run_ids
        max_nb_run_ids = df_counts['count'].max()
        # get the list of method_ids who have been tested on nb_max_run_ids
        method_ids = list(df_counts.loc[df_counts['count'] == max_nb_run_ids]
                          ['method_id'])
        # test the null hypothesis using friedman before doing a post-hoc analysis
        friedman_p_value = friedmanchisquare(*(
            np.array(df_perf.loc[df_perf['method_id'] == c]['metric_val'])
            for c in method_ids))[1]
        if friedman_p_value >= alpha:
            # then the null hypothesis over the entire method_ids cannot be rejected
            # print('the null hypothesis over the entire method_ids cannot be rejected')
            exit()
        # get the number of method_ids
        m = len(method_ids)
        # init array that contains the p-values calculated by the Wilcoxon signed rank test
        p_values = []
        # loop through the algorithms to compare pairwise
        for i in range(m - 1):
            # get the name of method_id one
            method_id_1 = method_ids[i]
            # get the performance of method_id one
            perf_1 = np.array(df_perf.loc[df_perf['method_id'] == method_id_1]['metric_val']
                              , dtype=np.float64)
            for j in range(i + 1, m):
                # get the name of the second method_id
                method_id_2 = method_ids[j]
                # get the performance of method_id one
                perf_2 = np.array(df_perf.loc[df_perf['method_id'] == method_id_2]
                                  ['metric_val'], dtype=np.float64)
                # calculate the p_value
                p_value = wilcoxon(perf_1, perf_2, zero_method='pratt')[1]
                # appen to the list
                p_values.append((method_id_1, method_id_2, p_value, False))
        # get the number of hypothesis
        k = len(p_values)
        # sort the list in acsending manner of p-value
        p_values.sort(key=operator.itemgetter(2))

        # loop through the hypothesis
        for i in range(k):
            # correct alpha with holm
            new_alpha = float(alpha / (k - i))
            # test if significant after holm's correction of alpha
            if p_values[i][2] <= new_alpha:
                p_values[i] = (p_values[i][0], p_values[i][1], p_values[i][2], True)
            else:
                # stop
                break
        # compute the average ranks to be returned (useful for drawing the cd diagram)
        # sort the dataframe of performances
        sorted_df_perf = df_perf.loc[df_perf['method_id'].isin(method_ids)]. \
            sort_values(['method_id', 'run_id'])
        # get the rank data
        rank_data = np.array(sorted_df_perf['metric_val']).reshape(m, max_nb_run_ids)

        # create the data frame containg the accuracies
        df_ranks = pd.DataFrame(data=rank_data, index=np.sort(method_ids), columns=
        np.unique(sorted_df_perf['run_id']))

        # number of wins
        dfff = df_ranks.rank(ascending=False)
        # print(dfff[dfff == 1.0].sum(axis=1))

        # average the ranks
        average_ranks = df_ranks.rank(ascending=False).mean(axis=1).sort_values(ascending=False)
        # return the p-values and the average ranks
        return p_values, average_ranks, max_nb_run_ids


    def draw_cd_diagram(self, ax, df_perf=None, alpha=0.05, title=None, title_fontsize=22, labels=False, title_pad=None):
        """
        Draws the critical difference diagram given the list of pairwise method_ids that are
        significant or not
        """
        p_values, average_ranks, _ = self.wilcoxon_holm(df_perf=df_perf, alpha=alpha)

        # print(average_ranks)

        # for p in p_values:
        #     print(p)

        self.graph_ranks(ax, average_ranks.values, average_ranks.keys(),
                    reverse=True, textspace=1.5, labels=labels)

        if title:
            exec(gen_cmd_print_variables('ax.title'))
            ax.set_title(
                title,
                fontsize=title_fontsize,
                pad=title_pad,
                # fontdict=font,
                y=0.9, x=0.5
            )

    def plot_cd_diagram_from_metric_mat(self, ax, metric_mat, **draw_cdd_kwargs):

        all_run_ids = self.all_run_ids if self.all_run_ids is not None else [str(i) for i in range(metric_mat.shape[-1])]

        exec(gen_cmd_print_variables('len(all_run_ids), all_run_ids'))
        exec(gen_cmd_print_variables('len(self.all_method_ids), self.all_method_ids'))
        exec(gen_cmd_print_variables('metric_mat.shape'))

        assert metric_mat.shape == (len(self.all_method_ids), len(all_run_ids))
        n_methods, n_runs = metric_mat.shape

        df_perf = pd.DataFrame({
            'method_id': np.repeat(self.all_method_ids, n_runs),
            'run_id': np.tile(all_run_ids, n_methods),
            'metric_val': metric_mat.flatten()
        })
        exec(gen_cmd_print_variables('draw_cdd_kwargs'))
        self.draw_cd_diagram(ax, df_perf=df_perf, **draw_cdd_kwargs)

    def plot_cd_diagram_from_metric_matrix(
            self,
            metric_matrix, # (n_settings, n_methods, n_runs). 每个setting一个subplot, 用plot_line_chart_from_metric_mat具体画
            subplot_titles=None,

            fig_height=5,
            fig_width_per_subplot=6,
            n_subplot_rows = 1,
            tight_layout_h_pad=None,
            tight_layout_w_pad=None,
            tight_layout_rect=(0, 0, 1, 1),

            fig_save_fname=None,
            save_dpi=200,

            **subplot_kwargs,
    ):

        n_settings, n_methods, n_runs = metric_matrix.shape

        fig, axes = create_subplot_layout(
            n_settings, n_subplot_rows,
            fig_height, fig_width_per_subplot,
            save_dpi,
            tight_layout_h_pad=tight_layout_h_pad,
            tight_layout_w_pad=tight_layout_w_pad,
            tight_layout_rect=tight_layout_rect,
        )

        if subplot_titles is not None:
            assert 'title' not in subplot_kwargs
        else:
            subplot_titles = [subplot_kwargs.get('title', None)] * n_settings
        assert len(subplot_titles) == n_settings

        for ax, metric_mat, title in zip(axes, metric_matrix, subplot_titles):
            cur_subplot_kwargs = deepcopy(subplot_kwargs)
            cur_subplot_kwargs['title'] = title
            self.plot_cd_diagram_from_metric_mat(ax, metric_mat, **cur_subplot_kwargs)
        if fig_save_fname is not None:
            plt.savefig(fig_save_fname, bbox_inches='tight', dpi=save_dpi, pad_inches=0)
        return fig, axes


# if __name__ == '__main__':
#     all_method_ids = ['method_0', 'method_1', 'method_2', 'method_3']
#     all_setting_ids = ['setting_0', 'setting_1', 'setting_2']
#
#     metric_comparor = MetricComparer(all_method_ids)
#
#     metric_matrix = [
#         [
#             [0.1, 0.5, 0.3, 0.2, 0.8], [0.3, 0.2, 0.9, 0.1, 0.4], [0.7, 0.8, 0.1, 0.2, 0.3], [0.1, 0.3, 0.5, 0.7, 0.4]
#         ],
#         [
#             [0.9, 0.1, 0.2, 0.4, 0.5], [0.1, 0.7, 0.2, 0.4, 0.3], [0.5, 0.2, 0.1, 0.9, 0.5], [0.3, 0.1, 0.5, 0.7, 0.7]
#         ],
#         [
#             [0.2, 0.1, 0.9, 0.7, 0.6], [0.8, 0.7, 0.9, 0.5, 0.3], [0.7, 0.2, 0.1, 0.9, 0.2], [0.3, 0.1, 0.5, 0.7, 0.9]
#         ],
#     ]
#     metric_matrix = np.array(metric_matrix)
#
#     # metric_comparor.compare_from_metric_mat(metric_matrix[0], print_detailed_results=True)
#     metric_comparor.compare_from_metric_matrix(metric_matrix, all_setting_ids, print_detailed_results=True)


