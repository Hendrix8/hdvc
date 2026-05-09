# -*- coding: UTF-8 -*-

# 检查过一遍，需检查第二遍
def format_str_vec(vec, sep='-'):
    return sep.join((str(v) for v in vec))


# 检查过一遍，需检查第二遍
# 以“起始值-步长-最终值”格式输出range那一类的量
def format_str_range(v_range, sep='-'):
    v0, vn, step = v_range[0], v_range[len(v_range) - 1], v_range[1] - v_range[0]
    return sep.join((str(v0), str(step), str(vn)))
