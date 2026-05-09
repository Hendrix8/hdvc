import sys

sys.path.append('/lustre/fswork/projects/rech/thj/uth68ud/PycharmProjects/Qinco2')

import matplotlib.pyplot as plt
import math

def create_subplot_layout(
        n_subplots, n_subplot_rows,
        fig_height, fig_width_per_subplot, dpi,
        tight_layout_h_pad=None, tight_layout_w_pad=None,
        tight_layout_rect=(0, 0, 1, 1)
):
    # 1. 计算列数与总宽度
    n_subplot_cols = math.ceil(n_subplots / n_subplot_rows)
    fig_width = fig_width_per_subplot * n_subplot_cols

    # 2. 创建画布
    fig, axes = plt.subplots(n_subplot_rows, n_subplot_cols, figsize=(fig_width, fig_height), dpi=dpi,
                             layout="constrained")

    # 兼容处理：确保 axes 是列表
    if n_subplot_rows == 1 and n_subplot_cols == 1:
        axes_flat = [axes]
    else:
        axes_flat = axes.flatten()

    # 3. 先调整整体布局 (这一步至关重要，必须在居中计算之前)
    # 它会决定子图在 rect 范围内的最终位置和大小
    fig.tight_layout(h_pad=tight_layout_h_pad, w_pad=tight_layout_w_pad, rect=tight_layout_rect)

    # 4. 隐藏多余的（没用上的）Axes
    for i in range(n_subplots, len(axes_flat)):
        axes_flat[i].axis('off')

    # 5. 居中处理逻辑
    last_row_actual_count = n_subplots % n_subplot_cols
    # 只有当最后一行不为空且没填满时才需要移动
    if 0 < last_row_actual_count < n_subplot_cols:
        # 此时 tight_layout 已经确立了标准子图的 position
        # 我们计算两个相邻子图左边缘的物理差值，作为“一格”的单位宽度
        if n_subplot_cols > 1:
            pos0 = axes_flat[0].get_position()
            pos1 = axes_flat[1].get_position()
            unit_width = pos1.x0 - pos0.x0
        else:
            unit_width = 0 # 只有一列时不需要移动

        # 计算偏移量：(空余格数 / 2) * 单位宽度
        offset = (n_subplot_cols - last_row_actual_count) * unit_width / 2

        # 找到最后一行所有有效的 axes 索引并应用偏移
        start_idx = n_subplots - last_row_actual_count
        for i in range(start_idx, n_subplots):
            pos = axes_flat[i].get_position()
            new_pos = [pos.x0 + offset, pos.y0, pos.width, pos.height]
            axes_flat[i].set_position(new_pos)

    return fig, axes_flat[:n_subplots]


def row_major_legend(ax, legend_ncol):
    """
    使图例以行优先（Row-major）顺序排列
    """

    handles, labels = ax.get_legend_handles_labels()

    # 计算重新排序后的索引
    # 原理：利用切片 [i::ncol] 提取每一列对应的元素并合并
    new_handles = sum([handles[i::legend_ncol] for i in range(legend_ncol)], [])
    new_labels = sum([labels[i::legend_ncol] for i in range(legend_ncol)], [])
    return new_handles, new_labels

if __name__ == '__main__':

    # --- 测试保存 ---
    # 5个子图，2行（第一行3个，第二行2个居中）
    fig, axes = create_subplot_layout(
        n_subplots=5,
        n_subplot_rows=2,
        fig_height=8,
        fig_width_per_subplot=4,
        tight_layout_h_pad=4.0,  # 增大行间距
        tight_layout_w_pad=4.0,  # 增大列间距
        tight_layout_rect=(0.05, 0.05, 0.95, 0.95) # 留出 5% 的边缘防止标签溢出
    )

    for i, ax in enumerate(axes):
        ax.plot([0, 1], [0, i], label=f'Line {i}')
        ax.set_title(f"Subplot {i+1}")
        ax.set_ylabel("Y Label")

    # 直接保存，不再使用 bbox_inches='tight'，实现所见即所得
    fig.savefig('final_layout_test.png', bbox_inches='tight', dpi=600)
    print("图片已保存为 final_layout_test.png")