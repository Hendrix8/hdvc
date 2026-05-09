# 注意：以下print_variables和gen_cmd_print_variables和test_util里完全一样！
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


def print_bash_for_loops(all_s_elm, all_s_array, init_n_tabs=0):
    '''

    用于快速生成可用于bash里大量嵌套循环的代码
    :param all_s_elm: 其中每个元素是每层循环中"each"部分对应的变量名
    :param all_s_array: 其中每个元素是每层循环中被循环的那个数组
    '''
    assert len(all_s_elm) == len(all_s_array)

    # 输出前半部分（"for" 部分）
    n_tabs = init_n_tabs
    for s_elm, s_array in zip(all_s_elm, all_s_array):

        # 输出制表符
        for _ in range(n_tabs):
            print("\t", end='')
        n_tabs += 1

        # 输出for语句
        print(f"for {s_elm} in \"${{{s_array}[@]}}\"; do")  # 打印大括号需要打两个大括号，而不是用"\"转义

    # 输出中间部分
    for _ in range(n_tabs):
        print("\t", end='')
    print("# Add command here.")

    # 输出"done"部分
    while n_tabs != init_n_tabs:
        n_tabs -= 1 # 先减
        for _ in range(n_tabs):
            print("\t", end='')
        print("done")

if __name__ == "__main__":
    all_s_elm = ('n_neighbors_pu', 'init_query_strategies', 'rand_seed', 'total_query_proportion', 'pu_func_name',
                 'sc_family', 'n_neighbors_init_query', 'patient_id', 'channel')
    all_s_array = ('all_n_neighbors_pu', 'all_init_query_strategies', 'all_rand_seeds', 'all_total_query_proportion', 'all_pu_func_names',
                 'all_sc_families', 'all_n_neighbors_init_query', 'all_patient_ids', 'all_channels')
    print_bash_for_loops(all_s_elm, all_s_array)