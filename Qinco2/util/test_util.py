from util.index_util import *
from util.dtype_util import is_iterable, to_iterable
# from util.calc_util import union1d


def print_variables(s_variables, variables, pfx_msg=None, sfx_msg=None):
    """
    输出各个变量名的值
    :param s_variables: 要输出的各个变量名（可以是单个变量名，也可以是各个值的列表）
    :param variables: 要输入的各个变量值（可以是单个值，也可以是各个值的列表）。这个可以在外面通过eval得到
    :param pfx_msg, sfx_msg: 要输出的前缀和后缀信息
    """

    if not is_iterable(s_variables):
        s_vars, vars = [s_variables], [variables]
    else:
        s_vars, vars = s_variables, variables

    print_str = ', '.join(
        (s_var + ' = ' + str(vars[i]).replace('\n', '') for i, s_var in enumerate(s_vars))
    )
    if pfx_msg is not None:
        print_str = pfx_msg + ' ' + print_str
    if sfx_msg is not None:
        print_str += ' ' + sfx_msg
    print_str = 'print(' + repr(print_str) + ')'
    exec(print_str)


def gen_cmd_print_variables(str_all_vars, separator=', ', pfx_msg=None, sfx_msg=None):
    """
    用于在一行之内执行print_variables。用这个函数的时候，写成exec(gen_cmd_print_variables(...))
    :param str_all_vars: 一个字符串，字符串中列出所有要打印值的变量名，形如'var1, var2, var3'
    :param separator: s_var_list中用于分隔各个变量名的字符串
    :return to_exec：要执行的print_variables命令
    """

    var_names = str_all_vars.split(separator)
    for i in range(len(var_names)):
        var_names[i].replace(' ', '')

    if len(var_names) == 1:
        str_s_variables = '\'' + var_names[0] + '\''
        str_variables = 'eval(' + str_s_variables + ')'
    else:
        str_s_variables = '(' + ', '.join(('\'' + var_name + '\'' for var_name in var_names)) + ')'
        str_variables = 'list(map(eval, ' + str_s_variables + '))'
    to_exec = 'print_variables(' + ', '.join((str_s_variables, str_variables))
    if pfx_msg is not None:
        to_exec += ', pfx_msg=\'' + pfx_msg + '\''
    if sfx_msg is not None:
        to_exec += ', sfx_msg=\'' + sfx_msg + '\''
    to_exec += ')'
    return to_exec


# 输出：原始差别序列（绝对/相对），排序后差别序列（绝对/相对），排序后的顺序 -> 最大绝对差别、最大相对差别、差别最大的位置
def compare_two_arrays(a, b, axis=None, print_summary=True, pfx_msg='---', sfx_msg=None):
    """
    比较两个变量中差别最大的地方
    :param a, b: 要比较的两个变量
    :param axis: 用于np.max的axis参数，不能是数组
    :returns: abs_diff, rel_diff, abs_order, rel_order, sorted_abs_diff, sorted_rel_diff,
           max_abs_diff, max_rel_diff, max_abs_diff_pos, max_rel_diff_pos
    """

    if is_iterable(axis):
        print('Error! Non-axis scaler not yet supported!')
        exit(1)

    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    ndim = a.ndim
    if ndim > 2:
        print('Error! A maximum of 2 dimensions are supported currently!')
        exit(1)
    if a.shape != b.shape:
        print(f'Unable to compare due to shape differences between {a.shape} and {b.shape}')
        return # 不exit，只return

    abs_diff = np.abs(a - b)
    dividend = np.minimum(np.abs(a), np.abs(b)).flatten() # np.minimum返回element-wise的最小值
    dividend[np.where(dividend == 0)[0]] = -1  # 让除以0的地方肯定不是最大的地方
    dividend = dividend.reshape(dividend.shape)
    rel_diff = abs_diff.flatten() / dividend
    rel_diff[np.where(rel_diff < 0)[0]] = float('-inf')   # 无效的地方置为-inf
    # rel_diff_[union1d(np.where(rel_diff_ == float('nan'))[0],
    #                   np.where(rel_diff_ == float('inf'))[0])] = -float('inf')
    rel_diff = rel_diff.reshape(abs_diff.shape)

    abs_order, rel_order = np.argsort(-abs_diff, axis=axis), np.argsort(-rel_diff, axis=axis)
    sorted_abs_diff, sorted_rel_diff = -np.sort(-abs_diff, axis=axis), -np.sort(-rel_diff, axis=axis) # 这里多此一举地再用sort而不是直接用order，是为了不另外地做flatten（当axis is None的时候）

    if axis is None or ndim == 1:
        max_abs_diff, max_rel_diff = sorted_abs_diff[0], sorted_rel_diff[0]
        max_abs_diff_pos = ind_1D_to_multi(abs_order[0], a.shape)
        max_rel_diff_pos = ind_1D_to_multi(rel_order[0], a.shape)
    else:  # 2D, axis == 0 or axis == 1
        max_abs_diff_pos = abs_order[0] if axis == 0 else abs_order[:, 0]
        max_rel_diff_pos = rel_order[0] if axis == 0 else rel_order[:, 0]

        n_along_axis = abs_diff.shape[int(not axis)]
        max_abs_diff, max_rel_diff = np.empty(n_along_axis), np.empty(n_along_axis)
        for i in range(n_along_axis):
            max_abs_diff[i] = abs_diff[max_abs_diff_pos[i]][i] if axis == 0 else abs_diff[i][max_abs_diff_pos[i]]
            max_rel_diff[i] = rel_diff[max_rel_diff_pos[i]][i] if axis == 0 else rel_diff[i][max_rel_diff_pos[i]]

    # print_variables(('abs_diff', 'rel_diff'), list(map(eval, ('abs_diff', 'rel_diff'))))
    # exec(gen_cmd_print_variables(
    #     'sorted_abs_diff, sorted_rel_diff, abs_order, rel_order, '
    #     'max_abs_diff_pos, max_rel_diff_pos', pfx_msg=pfx_msg, sfx_msg=sfx_msg))

    if (to_iterable(max_abs_diff_pos) == to_iterable(max_rel_diff_pos)).all():
        max_diff_pos = max_abs_diff_pos
        s_vars = 'max_abs_diff, max_rel_diff, max_diff_pos'
    else:
        if ndim == 1:
            rel_diff_at_max_abs_diff_pos = rel_diff[max_abs_diff_pos]
            abs_diff_at_max_rel_diff_pos = abs_diff[max_rel_diff_pos]
        elif axis is None:
            rel_diff_at_max_abs_diff_pos = rel_diff.flatten()[abs_order[0]]
            abs_diff_at_max_rel_diff_pos = abs_diff.flatten()[rel_order[0]]
        else:  # 多维数组，沿着某一个axis。目前只支持二维！！！
            n_along_axis = abs_diff.shape[int(not axis)]
            rel_diff_at_max_abs_diff_pos, abs_diff_at_max_rel_diff_pos = np.empty(n_along_axis), np.empty(n_along_axis)

            for i in range(n_along_axis):  # ndim = 2 for now
                rel_diff_at_max_abs_diff_pos[i] = np.take(rel_diff, i, axis=int(not axis))[max_abs_diff_pos[i]]
                abs_diff_at_max_rel_diff_pos[i] = np.take(abs_diff, i, axis=int(not axis))[max_rel_diff_pos[i]]
        s_vars = 'max_abs_diff, max_rel_diff, max_abs_diff_pos, max_rel_diff_pos, ' \
                 'rel_diff_at_max_abs_diff_pos, abs_diff_at_max_rel_diff_pos'
    if print_summary:
        exec(gen_cmd_print_variables(s_vars, pfx_msg=pfx_msg, sfx_msg=sfx_msg))

    return abs_diff, rel_diff, abs_order, rel_order, sorted_abs_diff, sorted_rel_diff, \
           max_abs_diff, max_rel_diff, max_abs_diff_pos, max_rel_diff_pos


# a = [2, 3, 5]
# b = [3, 1, 0]
# compare_two_arrays(a, b, axis=None)
