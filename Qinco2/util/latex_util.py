# -*- coding: UTF-8 -*-

TOPRULE = '\\toprule'
MIDRULE = '\\midrule'
END_TABULAR = '\\end{tabular}'

def makecell(contents_by_row, align='c'):
    """
    将列表内容转换为 LaTeX 的 \makecell 格式。

    :param contents_by_row: 包含各行文字的列表，例如 ['Sampling', 'Rate']
    :param align: 对齐方式，默认为 'c' (居中)，可选 'l' (左) 或 'r' (右)
    :return: 格式化后的 LaTeX 字符串
    """
    # 使用 \\\\ 来表示 LaTeX 中的换行符 \\
    rows = " \\\\ ".join(contents_by_row)

    # 构造最终的字符串
    # 使用 f-string 时，双反斜杠需要写成 \\\\ 来转义
    return f"\\makecell[{align}]{{{rows}}}"


def multicolumn(content, num_cols, align):
    """
    生成 LaTeX 的 \multicolumn 字符串。

    :param num_cols: 合并的列数 (int)
    :param align: 对齐与边框格式，例如 'c', '|c|', 'l'
    :param content: 单元格内容，可以是字符串，也可以是列表（自动调用 makecell）
    :return: 格式化后的 LaTeX 字符串
    """
    # 如果内容是一个列表，说明需要换行，自动调用之前的逻辑
    if isinstance(content, list):
        # 这里假设你已经定义了上一个回答中的 makecell 函数
        content = makecell(content)

    return f"\\multicolumn{{{num_cols}}}{{{align}}}{{{content}}}"


def multirow(content, num_rows, width='*', align='c'):
    """
    生成 LaTeX 的 \multirow 字符串。

    :param num_rows: 合并的行数 (int)
    :param content: 单元格内容，可以是字符串或列表（列表将自动转为 makecell）
    :param width: 宽度，默认为 '*' (自动宽度)
    :param align: 如果内容是列表，传递给 makecell 的对齐方式
    :return: 格式化后的 LaTeX 字符串
    """
    # 如果内容是列表，自动调用 makecell 处理多行内容
    if isinstance(content, list):
        # 嵌套之前定义的 makecell
        content = makecell(content, align=align)

    return f"\\multirow{{{num_rows}}}{{{width}}}{{{content}}}"


def empty_cells(n, trailing_ampersand=True):
    """
    生成连续的 LaTeX 空单元格。

    :param n: 需要生成的空单元格数量
    :param trailing_ampersand: 是否在最后一个单元格后面也加 &
    :return: 字符串，例如 ' &  & '
    """
    if n <= 0:
        return ""

    # 生成 n 个空位
    cells = [" " for _ in range(n)]

    # 用 " & " 连接
    result = " & ".join(cells)

    # 如果需要在末尾补一个 &
    if trailing_ampersand:
        result += " &"

    return result


def dense_row(contents_by_cell, end_row=True):
    """
    将列表内容连接成 LaTeX 的一行。

    :param contents_by_cell: 每一格内容的列表，例如 ['A', 'B', 'C']
    :param end_row: 是否在行尾添加 '\\\\'
    :return: 格式化后的 LaTeX 行字符串
    """
    # 将所有内容转换为字符串（防止传入数字等非字符串类型）
    # 使用 " & " 作为分隔符连接
    row_str = " & ".join(str(c) for c in contents_by_cell)

    if end_row:
        row_str += " \\\\"

    return row_str


def sparse_row(content_dict, total_columns, end_row=True):
    """
    在一行中只在特定位置填入内容，其余位置为空。

    :param total_columns: 这一行总共有多少列
    :param content_dict: 一个字典，Key 是列的索引（从 1 开始），Value 是内容
    :param end_row: 是否添加行末换行符 \\
    :return: 格式化后的 LaTeX 行字符串
    """

    # 注意：索引从1开始

    # 1. 初始化一个全是空字符串的列表
    row_cells = [""] * total_columns

    # 2. 根据字典，在特定位置填入内容
    for index, content in content_dict.items():
        if 0 <= index - 1 < total_columns:
            # 如果内容是列表，自动调用之前的 makecell (可选)
            if isinstance(content, list):
                content = makecell(content)
            row_cells[index - 1] = str(content)

    # 3. 调用之前定义的 print_row 函数输出
    return dense_row(row_cells, end_row=end_row)


def begin_tabular(num_columns, align='c', separator='|', left_border='@{}', right_border='@{}'):
    """
    生成 LaTeX tabular 环境的起始行。

    :param num_columns: 列数 (int)
    :param align: 每一列的对齐方式，默认为 'c'
    :param separator: 列之间的分隔符，默认为 '|'
    :param left_border: 表格最左侧的定义，默认为 '@{}' (去除首列间距)
    :param right_border: 表格最右侧的定义，默认为 '@{}' (去除末列间距)
    :return: 格式化后的 \begin{tabular}{...} 字符串
    """
    # 构造重复单元，例如 'c|'
    unit = f"{align}{separator}"

    # 重复 n-1 次，最后一次不加分隔符（除非你想要右侧也有竖线）
    # 比如 3 列 c|c|c
    pattern = unit * (num_columns - 1) + align

    # 拼接完整的格式字符串
    full_format = f"{left_border}{pattern}{right_border}"

    return f"\\begin{{tabular}}{{{full_format}}}"


def cline(start_col, end_col=None):
    """
    生成 LaTeX 的 \cline{i-j}
    :param start_col: 起始列索引（从 1 开始计数）
    :param end_col: 结束列索引
    """
    if end_col is None:
        end_col = start_col
    return f"\\cline{{{start_col}-{end_col}}}"

# 注意：这个相比cline，会在线交汇处有缺口
def cmidrule(start_col, end_col=None, trim='lr'):
    """
    生成 LaTeX 的 \cmidrule 字符串。

    :param start_col: 起始列索引（从 1 开始计数）
    :param end_col: 结束列索引，如果为 None，则默认与 start_col 相同
    :param trim: 缩进选项，通常为 'l', 'r' 或 'lr'。设为 None 则不缩进。
    :return: 格式化后的 \cmidrule 字符串
    """
    # 处理可选参数 end_col
    if end_col is None:
        end_col = start_col

    # 处理缩进选项 (lr)
    trim_str = f"({trim})" if trim else ""

    return f"\\cmidrule{trim_str}{{{start_col}-{end_col}}}"


def latex_style(content, command='textbf', options=None):
    """
    为内容添加 LaTeX 样式命令。

    :param content: 文本内容
    :param command: LaTeX 命令名，如 'textbf', 'textit', 'underline', 'url'
    :param options: 可选参数，放在 [] 中，例如 'textbf' 的特殊选项
    :return: 格式化后的字符串
    """
    opt_str = f"[{options}]" if options else ""
    return f"\\{command}{opt_str}{{{content}}}"