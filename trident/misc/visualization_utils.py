from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import builtins
import sys
from typing import Sequence,List,Dict,Any
from trident.backend.common import if_none, get_plateform
from trident.backend.opencv_backend import array2image
from trident.misc.ipython_utils import is_in_ipython, is_in_colab
import math

if is_in_ipython():
    from IPython import display

if not is_in_colab:
    import matplotlib

    matplotlib.use('Qt4Agg' if not is_in_ipython() and not is_in_colab() else 'NbAgg')
else:
    import matplotlib

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib import colormaps
from matplotlib.colors import to_hex
import matplotlib.patches as patches
import matplotlib.font_manager
import PIL
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

fonts = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
default_font = None
if get_plateform() == 'linux':
    candidate = [f for f in fonts if 'Ubuntu-LI.ttf' in f or 'LiberationSans-Regular.ttf' in f]
    default_font = candidate[0] if len(candidate) > 0 else fonts[0]
elif get_plateform() == 'windows':
    candidate = [f for f in fonts if 'Microsoft Sans Serif' in matplotlib.font_manager.FontProperties(
        fname=f).get_name() or 'heiti' in matplotlib.font_manager.FontProperties(fname=f).get_name()]
    default_font = candidate[0] if len(candidate) > 0 else fonts[0]
# ImageFont.truetype(f, 20)
# if
# fontnames = [matplotlib.font_manager.FontProperties(fname=fname).get_name() for fname in fonts]
# default_font = None
#
# if sys.platform == 'win32':
#     if 'Microsoft Sans Serif' in fontnames:
#         default_font = matplotlib.rc('font', family='Microsoft Sans Serif')
#     else:
#         for name in fontnames:
#             if 'heiti' in name.lower():
#                 default_font = matplotlib.rc('font', family=name)
#                 break


import colorsys
import itertools
import numpy as np
from trident import context
from trident.context import split_path, make_dir_if_need
from trident.backend.common import get_time_suffix, unpack_singleton, get_plateform, PrintException

ctx = context._context()
_backend = ctx.get_backend()
working_directory = ctx.working_directory

if _backend == 'pytorch':
    import torch
    import torch.nn as nn
    from trident.backend.pytorch_backend import *
    from trident.backend.pytorch_ops import element_cosine_distance, argmin, sqrt, reduce_sum, expand_dims, int_shape

elif _backend == 'tensorflow':
    import tensorflow as tf
    from trident.backend.tensorflow_backend import *
    from trident.backend.tensorflow_ops import element_cosine_distance, argmin, sqrt, reduce_sum, expand_dims
from trident.data.image_common import *

__all__ = ['tile_rgb_images', 'loss_metric_curve', 'steps_histogram', 'generate_palette', 'plot_bbox',
           'plot_3d_histogram', 'plot_centerloss']


def generate_palette(num_classes: int, format: str = 'rgb'):
    """Generate palette used in visualization.

    Args:
        num_classes (int): numbers of colors need in palette?
        format (string):  'rgb' by pixel tuple or  'hex'  by hex formatted color.

    Returns:
        colors in rgb or hex format
    Examples:
        >>> generate_palette(10)
        [(128, 64, 64), (128, 102, 64), (115, 128, 64), (77, 128, 64), (64, 128, 89), (64, 128, 128), (64, 89, 128), (76, 64, 128), (115, 64, 128), (128, 64, 102)]
        >>> generate_palette(24,'hex')
        ['#804040', '#805040', '#806040', '#807040', '#808040', '#708040', '#608040', '#508040', '#408040', '#408050', '#408060', '#408070', '#408080', '#407080', '#406080', '#405080', '#404080', '#504080', '#604080', '#704080', '#804080', '#804070', '#804060', '#804050']

    """

    def hex_format(rgb_tuple):
        return '#{:02X}{:02X}{:02X}'.format(rgb_tuple[0], rgb_tuple[1], rgb_tuple[2])

    if num_classes is None:
        num_classes = 20
    hsv_tuples = [(x / num_classes, 1.0, 1.0) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (
    int(builtins.round(x[0] * 255.)), int(builtins.round(x[1] * 255.)), int(builtins.round(x[2] * 255.))), colors))
    if format == 'rgb':
        return colors
    elif format == 'hex':
        return [hex_format(color) for color in colors]


def plot_bbox(box, img, color=None, label=None, line_thickness=None, **kwargs):
    import cv2
    img_shape = (img.height, img.width, 3)
    try:

        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        # img = array2image(img)
        c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        draw = ImageDraw.Draw(img)
        draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline=color, fill=None, width=tl)
        fontcolor = (255, 255, 255)
        avg_color = np.array(list(color)).mean()
        min_color = np.array(list(color)).min()
        font = ImageFont.truetype(default_font, int(math.sqrt(img_shape[0] / 1000) * 16 + 2))
        # font=None#ImageFont.truetype(fonts[fontnames.index('Hiragino Sans GB')],int(math.sqrt(img_shape[0] / 1000) * 10 + 1))
        if avg_color >= 120 or min_color <= 32:
            fontcolor = (0, 0, 0)
        if label:
            # font = ImageFont.truetype(fonts[fontnames.index('Microsoft Sans Serif')], int(math.sqrt(img_shape[0] / 1000) * 10 + 1))
            tf = max(tl - 1, 1)  # font thickness

            t_box= draw.textbbox(c1,str(label), font=font)
            t_size=[t_box[2]-t_box[0],t_box[3]-t_box[1]]
            offset = font.getbbox(str(label))
            corner2 = c1[0], c1[1] - t_size[1] - 2 * offset[1] - 3
            corner3 = c1[0] + 2 * offset[0] + t_size[0] + 3, c1[1]

            draw.rectangle((corner2, corner3), fill=color, width=2)
            draw.text((c1[0] + 1, c1[1] - t_size[1] - offset[1] - 2), ' {0}'.format(label), fill=fontcolor, font=font)


    except Exception as e:
        print('image_size', img_shape, box)
        print(e)
        PrintException()

    # rgb_image = image2array(img)
    return img



# def tile_rgb_images(*imgs, row=3, save_path=None, imshow=False, legend=None, **kwargs):
#     if save_path is not None:
#         make_dir_if_need(save_path)
#
#     distinct_row = list(set([len(ims) for ims in imgs]))
#     if len(distinct_row) > 1:
#         raise ValueError(f'imgs should have same length, but got {distinct_row}')
#     else:
#         distinct_row = unpack_singleton(distinct_row)
#         if 1 <= row < distinct_row:
#             distinct_row = row
#
#     suffix = get_time_suffix()
#     fig = plt.figure(figsize=(len(imgs) * 3, distinct_row * 3))
#     plt.ion()
#
#     if len(imgs) == 1 and distinct_row == 1:
#         img = array2image(imgs[0][0])
#         if save_path is not None:
#             filename = save_path.format(suffix)
#             img.save(filename)
#
#         if imshow:
#             plt.imshow(img)
#             plt.axis("off")
#             plt.show()
#         return fig
#
#     for m in range(distinct_row * len(imgs)):
#         ax = plt.subplot(distinct_row, len(imgs), m + 1)
#
#         if m < len(imgs) and legend is not None and len(legend) == len(imgs):
#             ax.set_title(legend[m])
#
#         img = imgs[int(m % len(imgs))][int(m // len(imgs))]
#
#         if img.ndim == 2:
#             img = np.stack([img] * 3, axis=-1)
#         elif img.ndim == 3 and img.shape[-1] == 1:
#             img = np.concatenate([img] * 3, axis=-1)
#
#         ax.imshow(array2image(img))
#         ax.axis("off")
#
#     plt.tight_layout()
#
#     if save_path is not None:
#         filename = save_path.format(suffix)
#         plt.savefig(filename, bbox_inches='tight')
#         print(f"Image saved to: {filename}")
#         if ctx.enable_mlflow:
#             ctx.mlflow_logger.add_image(filename)
#
#     if imshow:
#         plt.show()
#
#     return fig

# def tile_rgb_images(*imgs, row=3, save_path=None, imshow=False, legend=None, **kwargs):
#     if save_path is not None:
#         make_dir_if_need(save_path)
#     distinct_row = list(set([len(ims) for ims in imgs]))
#     if len(distinct_row) > 1:
#         raise ValueError('imgs should have same length, but got {0}'.format(distinct_row))
#     else:
#         distinct_row = unpack_singleton(distinct_row)
#         if 1 <= row < distinct_row:
#             distinct_row = row
#     suffix = get_time_suffix()
#     fig = plt.figure()
#     plt.ion()
#     if len(imgs) == 1 and distinct_row == 1:
#         img = array2image(imgs[0][0])
#         if save_path is not None:
#             filename = save_path.format(suffix)
#             img.save(filename)
#
#         if imshow:
#
#             if is_in_ipython():
#                 plt.axis("off")
#                 plt.imshow(img)
#                 plt.ioff()
#                 display.display(plt.gcf())
#             else:
#                 plt.axis("off")
#                 plt.imshow(img, interpolation="nearest", animated=True)
#                 plt.ioff()
#
#         return fig
#     else:
#
#         # figure, ax = plt.subplots(2, 2)
#         # fig.set_size_inches(len(imgs) * 2, row * 2)
#
#         # plt.ion()  # is not None:
#
#         for m in range(distinct_row * len(imgs)):
#
#             plt.subplot(distinct_row, len(imgs), m + 1)
#             if m < len(imgs) and legend is not None and len(legend) == len(imgs):
#                 plt.gca().set_title(legend[m])
#
#             img = (imgs[int(m % len(imgs))][int(m // len(imgs))])
#             if len(img.shape) == 2:
#                 img = np.stack([img, img, img], axis=-1)
#             if len(img.shape) == 3 and img.shape[-1] == 1:
#                 img = np.concatenate([img, img, img], axis=-1)
#             plt.imshow(array2image(img), interpolation="nearest", animated=True)
#             plt.axis("off")
#         plt.tight_layout()
#
#         if save_path is not None:
#             filename = save_path.format(suffix)
#             plt.savefig(filename, bbox_inches='tight')
#             if ctx.enable_mlflow:
#                 ctx.mlflow_logger.add_image(filename)
#         if imshow:
#             if is_in_ipython():
#                 plt.ioff()
#                 display.display(plt.gcf())
#             else:
#                 plt.ioff()
#                 # plt.draw()
#                 # plt.show(block=False)
#     return fig

def tile_rgb_images(
    *imgs: Sequence[np.ndarray],
    row: int = 3,
    save_path: str | os.PathLike | None = None,
    imshow: bool = False,
    legend: List[str] | None = None,
    figsize_scale: float = 3.0,
):
    """
    以網格方式拼貼影像。

    參數
    ----
    *imgs
        多組影像序列，形狀需一致，例如：tile_rgb_images(batch1, batch2, ...)
    row
        最多顯示列數（<= 單組影像數量）；預設 3。
    save_path
        儲存路徑範本，例如 ``"./output/tile_{}.png"``，會以 ``get_time_suffix()``
        取代 ``{}``。若為 ``None`` 則不存檔。
    imshow
        是否於函式內 ``plt.show()``；若於 Jupyter 建議設 `True`。
    legend
        每一欄的標題清單，長度須等於 ``len(imgs)``。
    figsize_scale
        單格影像對應的 figsize 比例，預設 3。
    """
    # ----------- 輸入檢查 ----------- #
    if not imgs:
        raise ValueError("必須至少傳入一組影像批次，例如 tile_rgb_images(batch1)")

    distinct_row = list({len(ims) for ims in imgs})
    if len(distinct_row) > 1:
        raise ValueError(f"所有批次須同長度，但收到 {distinct_row}")

    n_row = min(row, unpack_singleton(distinct_row))
    n_col = len(imgs)

    if legend is not None and len(legend) != n_col:
        raise ValueError(f"legend 長度應為 {n_col}，但收到 {len(legend)}")

    # ----------- 建立 Figure ----------- #
    plt.close('all')  # 避免在 Jupyter 產生多餘的空白圖框
    plt.ioff()  # 關閉互動模式，避免多餘輸出
    fig = plt.figure(figsize=(n_col * figsize_scale, n_row * figsize_scale))

    # ----------- 繪圖 ----------- #
    for idx in range(n_row * n_col):
        ax = plt.subplot(n_row, n_col, idx + 1)

        col_id = idx % n_col
        row_id = idx // n_col

        if legend is not None and row_id == 0:
            ax.set_title(legend[col_id])

        img = imgs[col_id][row_id]

        # 灰階或單通道轉三通道
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=-1)
        elif img.ndim == 3 and img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)

        ax.imshow(array2image(img))
        ax.axis("off")

    fig.tight_layout()

    # ----------- 儲存 ----------- #
    if save_path is not None:
        if "{0}" not in str(save_path) and  "{}" not in str(save_path):
            # 若使用者未提供占位符，則直接在檔名結尾加時間
            save_path = f"{save_path}_{get_time_suffix()}"
        else:
            save_path = str(save_path).format(get_time_suffix())

        make_dir_if_need(save_path)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[INFO] 影像已儲存：{save_path}")

    # ----------- 顯示 / 關閉 ----------- #
    if imshow:
        plt.show()
        plt.close(fig)
    else:
        plt.close(fig)

    return None



def loss_metric_curve(losses, metrics, metrics_names, legend=None, calculate_base='epoch', max_iteration=None,
                      save_path=None, imshow=False, **kwargs):
    default_colors = [to_hex(i) for i in colormaps.get_cmap('tab20').colors]
    colors = []
    line_type = ['-', '--', '-.', ':']
    fig = plt.gcf()
    fig.set_size_inches(18, 8)
    plt.clf()
    plt.ion()  # is not None:

    loss_ax1 = fig.add_subplot(2, 2, 1)
    losses = unpack_singleton(losses)
    only_single_model = True
    if losses.__class__.__name__ == 'HistoryBase':
        steps, values = losses.get_series('total_losses')
        loss_ax1.plot(steps, values, label='total_losses')

    elif isinstance(losses, list):
        only_single_model = False
        for n in range(len(losses)):
            item = losses[n]
            if item.__class__.__name__ == 'HistoryBase':
                steps, values = item.get_series('total_losses')

                p = loss_ax1.plot(steps, values, label='total_losses' + str(n))
                colors.append(p[-1].get_color())

    loss_ax1.set_title('model loss', fontsize=14, fontweight='bold')
    loss_ax1.set_ylabel('loss')
    loss_ax1.set_xlabel(calculate_base)
    loss_ax1.legend(loc="upper right")
    plt.legend(loc=2)

    if max_iteration is not None:
        loss_ax1.set_xlim(0, max_iteration)

    metric_ax1 = fig.add_subplot(2, 2, 2)
    if len(metrics) == 0:
        pass
    else:

        metric_ax2 = metric_ax1.twinx()
        first_axis_range = None
        second_axis_range = None
        first_axis_keys = []
        second_axis_keys = []
        first_axis_limit = []
        second_axis_limit = []
        metrics = unpack_singleton(metrics)
        if metrics.__class__.__name__ == 'HistoryBase':
            metrics_need_plot = metrics_names[0] if isinstance(metrics_names, dict) else metrics_names
            if 'epoch' in metrics_need_plot:
                metrics_need_plot.remove('epoch')
            for n in range(len(metrics)):
                k, v = list(metrics.items())[n]
                if k in metrics_need_plot:
                    legend_label = k
                    steps, values = metrics.get_series(k)
                    line_color = default_colors[n]

                    values_np = np.array(values)
                    if first_axis_range is None:
                        first_axis_range = (values_np.min(), values_np.mean(), values_np.max())
                        first_axis_keys.append(k)
                        first_axis_limit = [first_axis_range[0], first_axis_range[2]]
                        metric_ax1.plot(steps, values, color=line_color, label=legend_label)
                    else:
                        if second_axis_range is None and (
                                values_np.mean() < first_axis_range[1] * 0.1 or values_np.mean() > first_axis_range[
                            1] * 10):
                            second_axis_range = (values_np.min(), values_np.mean(), values_np.max())
                            metric_ax2.plot(steps, values, color=line_color, label=legend_label)
                            second_axis_limit = [second_axis_range[0], second_axis_range[2]]
                            second_axis_keys.append(k)
                        elif second_axis_range is not None:
                            compare_array = np.array([list(first_axis_range), list(second_axis_range)])
                            this_array = np.array([[values_np.min(), values_np.mean(), values_np.max()]])
                            distance = expand_dims(sqrt(reduce_sum((compare_array - this_array) ** 2, axis=-1)), 0)
                            result = argmin(distance, axis=-1)[0]
                            if result == 0:
                                metric_ax1.plot(steps, values, color=line_color, label=legend_label)
                                first_axis_keys.append(k)
                                first_axis_limit = [min(first_axis_limit[0], values_np.min()),
                                                    max(first_axis_limit[1], values_np.max())]
                            else:
                                metric_ax2.plot(steps, values, color=line_color, label=legend_label)
                                second_axis_keys.append(k)
                                second_axis_limit = [min(second_axis_limit[0], values_np.min()),
                                                     max(second_axis_limit[1], values_np.max())]
                        else:
                            metric_ax1.plot(steps, values, color=line_color, label=legend_label)
                            first_axis_limit = [min(first_axis_limit[0], values_np.min()),
                                                max(first_axis_limit[1], values_np.max())]
                            first_axis_keys.append(k)

            metric_ax1.legend(loc="lower right")
            if not any([n != n for n in first_axis_limit]):
                if len(first_axis_limit) >= 2 and first_axis_limit[0] != first_axis_limit[1]:
                    metric_ax1.set_ylim(first_axis_limit[0], first_axis_limit[1])
                if len(second_axis_keys) > 0:
                    metric_ax2.legend()
                    metric_ax2.set_ylim(second_axis_limit[0], second_axis_limit[1])
            # plt.legend(loc='upper left')

        elif isinstance(metrics, list):
            legend_list = []
            for i in range(len(metrics)):
                item = metrics[i]
                line_color = default_colors[i]
                if len(colors) > 1:
                    line_color = colors[i]
                if item.__class__.__name__ == 'HistoryBase':
                    for j in range(len(item.items())):
                        metrics_need_plot = metrics_names[i]
                        k = list(item.keys())[j]
                        if k in metrics_need_plot:
                            legend_label = k + str(i)
                            if legend is not None and len(legend) == len(metrics):
                                legend_label = legend[i] + ' ' + k

                            steps, values = item.get_series(k)

                            values_np = np.array(values)
                            if first_axis_range is None:
                                first_axis_range = (values_np.min(), values_np.mean(), values_np.max())
                                first_axis_keys.append(k)
                                first_axis_limit = [first_axis_range[0], first_axis_range[2]]
                                metric_ax1.plot(steps, values, color=line_color, linestyle=line_type[j % 4],
                                                linewidth=int((j // 4) % 4) + 1, label=legend_label)
                            else:
                                if second_axis_range is None and (
                                        values_np.mean() < first_axis_range[1] * 0.1 or values_np.mean() >
                                        first_axis_range[1] * 10):
                                    second_axis_range = (values_np.min(), values_np.mean(), values_np.max())
                                    second_axis_keys.append(k)
                                    second_axis_limit = [second_axis_range[0], second_axis_range[2]]
                                    metric_ax2.plot(steps, values, color=line_color, linestyle=line_type[j % 4],
                                                    linewidth=int((j // 4) % 4) + 1, label=legend_label)
                                elif k in first_axis_keys:
                                    first_axis_limit = [min(first_axis_limit[0], values_np.min()),
                                                        max(first_axis_limit[1], values_np.max())]
                                    metric_ax1.plot(steps, values, color=line_color, linestyle=line_type[j % 4],
                                                    linewidth=int((j // 4) % 4) + 1, label=legend_label)
                                elif k in second_axis_keys:
                                    second_axis_limit = [min(second_axis_limit[0], values_np.min()),
                                                         max(second_axis_limit[1], values_np.max())]
                                    metric_ax2.plot(steps, values, color=line_color, linestyle=line_type[j % 4],
                                                    linewidth=int((j // 4) % 4) + 1, label=legend_label)
                                elif second_axis_range is not None:
                                    _, first_values = item.get_series(first_axis_keys[0])
                                    first_values = np.array(first_values)
                                    _, second_values = item.get_series(second_axis_keys[0])
                                    second_values = np.array(second_values)
                                    compare_array = np.array(
                                        [[first_values.min(), first_values.mean(), first_values.max()],
                                         [second_values.min(), second_values.mean(), second_values.max()]])
                                    this_array = np.array([[values_np.min(), values_np.mean(), values_np.max()]])
                                    distance = expand_dims(sqrt(reduce_sum((compare_array - this_array) ** 2, axis=-1)),
                                                           0)
                                    result = argmin(distance, axis=-1)[0]
                                    if result == 0:
                                        first_axis_keys.append(k)
                                        first_axis_limit = [min(first_axis_limit[0], values_np.min()),
                                                            max(first_axis_limit[1], values_np.max())]
                                        metric_ax1.plot(steps, values, color=line_color, linestyle=line_type[j % 4],
                                                        linewidth=int((j // 4) % 4) + 1, label=legend_label)
                                    else:
                                        second_axis_keys.append(k)
                                        second_axis_limit = [min(second_axis_limit[0], values_np.min()),
                                                             max(second_axis_limit[1], values_np.max())]
                                        metric_ax2.plot(steps, values, color=line_color, linestyle=line_type[j % 4],
                                                        linewidth=int((j // 4) % 4) + 1, label=legend_label)
                                else:
                                    first_axis_keys.append(k)
                                    first_axis_limit = [min(first_axis_limit[0], values_np.min()),
                                                        max(first_axis_limit[1], values_np.max())]
                                    metric_ax1.plot(steps, values, color=line_color, linestyle=line_type[j % 4],
                                                    linewidth=int((j // 4) % 4) + 1, label=legend_label)
                            if len(values) > 0 and legend is not None:
                                legend_list.append(['{0} {1}'.format(k, legend[i])])
                            elif len(values) > 0:
                                legend_list.append(['{0} {1}'.format(k, i)])

            metric_ax1.legend(loc="lower right")
            metric_ax1.set_ylim(first_axis_limit[0], first_axis_limit[1])
            if len(second_axis_keys) > 0:
                metric_ax2.legend()
                metric_ax2.set_ylim(second_axis_limit[0], second_axis_limit[1])
            else:
                metric_ax2.remove()
            # plt.legend(legend_list,loc='upper left')

        metric_ax1.set_title('model metrics', fontsize=14, fontweight='bold')

        metric_ax1.set_xlabel(calculate_base)

        if max_iteration is not None:
            metric_ax1.set_xlim(0, max_iteration)

    if save_path is not None:
        suffix = get_time_suffix()
        filename = save_path.format(suffix)

        plt.savefig(filename, bbox_inches='tight')
        if ctx.enable_mlflow:
            ctx.mlflow_logger.add_image(filename)
    plt.tight_layout()
    if imshow:
        if is_in_ipython():
            plt.ioff()
            display.display(plt.gcf())
        else:
            plt.ioff()
            # plt.draw()
            # plt.show(block=False)

    return fig


def polygon_under_graph(xlist, ylist):
    """
    Construct the vertex list which defines the polygon filling the space under
    the (xlist, ylist) line graph.  Assumes the xs are in ascending order.
    """
    return [(xlist[0], 0.), *zip(xlist, ylist), (xlist[-1], 0.)]


def make_histogram(values, bins, max_bins=None):
    """Convert values into a histogram proto using logic from histogram.cc."""
    if values.size == 0:
        raise ValueError('The input has no element.')
    values = values.reshape(-1)
    counts, limits = np.histogram(values, bins=bins)
    num_bins = len(counts)
    if max_bins is not None and num_bins > max_bins:
        subsampling = num_bins // max_bins
        subsampling_remainder = num_bins % subsampling
        if subsampling_remainder != 0:
            counts = np.pad(counts, pad_width=[[0, subsampling - subsampling_remainder]],
                            mode="constant", constant_values=0)
        counts = counts.reshape(-1, subsampling).sum(axis=-1)
        new_limits = np.empty((counts.size + 1,), limits.dtype)
        new_limits[:-1] = limits[:-1:subsampling]
        new_limits[-1] = limits[-1]
        limits = new_limits

    # Find the first and the last bin defining the support of the histogram:
    cum_counts = np.cumsum(np.greater(counts, 0, dtype=np.int32))
    start, end = np.searchsorted(cum_counts, [0, cum_counts[-1] - 1], side="right")
    start = int(start)
    end = int(end) + 1
    del cum_counts

    # TensorBoard only includes the right bin limits. To still have the leftmost limit
    # included, we include an empty bin left.
    # If start == 0, we need to add an empty one left, otherwise we can just include the bin left to the
    # first nonzero-count bin:
    counts = counts[start - 1:end] if start > 0 else np.concatenate([[0], counts[:end]])
    limits = limits[start:end + 1]

    if counts.size == 0 or limits.size == 0:
        raise ValueError('The histogram is empty, please file a bug report.')

    sum_sq = values.dot(values)
    return limits.tolist(), counts.tolist()


default_bins = []
default_bins.extend(np.arange(-0.02, 0.02, 0.002).tolist())
default_bins.extend(np.arange(-0.005, 0.005, 0.0005).tolist())
default_bins.extend(np.arange(-0.0002, 0.0002, 0.00002).tolist())
default_bins = sorted(list(set(default_bins)))


def plot_3d_histogram(ax, grads, sample_collected=None, bins=None, inteval=1, title='', **kwargs):
    global default_bins
    from mpl_toolkits.mplot3d import Axes3D
    if bins is None:
        bins = default_bins

    collected_samples = []
    if sample_collected is not None and len(sample_collected) > 0:
        sample_collected = np.array(sample_collected)
        sample = np.arange(len(sample_collected))
        collected_samples = sample[sample_collected == 1]

    # ax = fig.gca(projection='3d')
    # Make verts a list, verts[i] will be a list of (x,y) pairs defining polygon i
    verts = []
    # The ith polygon will appear on the plane y = zs[i]
    zs = np.arange(len(grads))
    if len(collected_samples) == len(grads):
        zs = collected_samples

    new_zs = []
    max_frequency = 0

    for i in range(len(grads)):
        if i % inteval == 0:
            a, b = np.histogram(grads[i].reshape([-1]), bins)
            ys = a
            xs = b[:-1]
            new_zs.append(zs[i])
            max_frequency = max(np.max(a), max_frequency)
            verts.append(polygon_under_graph(xs, ys))

    poly = PolyCollection(verts, facecolors=['r', 'g', 'b', 'y'], alpha=.4)
    ax.add_collection3d(poly, zs=new_zs, zdir='y')
    override = {'fontsize': 'small', 'verticalalignment': 'top', 'horizontalalignment': 'center'}
    ax.set_xlabel('gradients', override)
    ax.set_ylabel('steps', override)
    ax.set_zlabel('frequency', override)
    ax.set_xlim(min(bins), max(bins))
    ax.set_ylim(0, int(max(new_zs)))
    ax.set_zlim(0, int(max_frequency * 1.1))
    plt.title(title + ' Gradients Histogram')
    return plt.figure()


def steps_histogram(grads, weights=None, sample_collected=None, bins=None, size=(18, 8), inteval=1, title='',
                    save_path=None,
                    imshow=False, enable_tensorboard=False, **kwargs):
    global default_bins
    from mpl_toolkits.mplot3d import Axes3D
    if bins is None:
        bins = default_bins

    collected_samples = []
    if sample_collected is not None and len(sample_collected) > 0:
        sample_collected = np.array(sample_collected)
        sample = np.arange(len(sample_collected))
        collected_samples = sample[sample_collected == 1]

    plt.ion()
    fig = plt.figure(figsize=size)
    fig.patch.set_facecolor('white')

    if grads is not None:
        ax = fig.add_subplot(1, 2, 1,
                             projection='3d') if grads is not None and weights is not None else fig.add_subplot(1, 1, 1,
                                                                                                                projection='3d')

        # ax = fig.gca(projection='3d')
        # Make verts a list, verts[i] will be a list of (x,y) pairs defining polygon i
        verts = []
        # The ith polygon will appear on the plane y = zs[i]
        zs = np.arange(len(grads))
        if len(collected_samples) == len(grads):
            zs = collected_samples

        new_zs = []
        max_frequency = 0

        for i in range(len(grads)):
            if (i + 1) % inteval == 0:
                a, b = np.histogram(grads[i].reshape([-1]), bins)
                ys = a
                xs = b[:-1]
                new_zs.append(zs[i])
                max_frequency = max(np.max(a), max_frequency)
                verts.append(polygon_under_graph(xs, ys))

        poly = PolyCollection(verts, facecolors=['r', 'g', 'b', 'y'], alpha=.4)
        ax.add_collection3d(poly, zs=new_zs, zdir='y')
        override = {'fontsize': 'small', 'verticalalignment': 'top', 'horizontalalignment': 'center'}
        ax.set_xlabel('gradients', override)
        ax.set_ylabel('steps', override)
        ax.set_zlabel('frequency', override)
        ax.set_xlim(min(bins), max(bins))
        ax.set_ylim(0, int(max(new_zs)))
        ax.set_zlim(0, int(max_frequency * 1.1))
        plt.title(title + ' Gradients Histogram')

    if weights is not None:
        ax = fig.add_subplot(1, 2, 2, projection='3d') if grads is not None else fig.add_subplot(1, 1, 1,
                                                                                                 projection='3d')

        bins = [b * 10 for b in bins]

        # Make verts a list, verts[i] will be a list of (x,y) pairs defining polygon i
        verts = []
        # The ith polygon will appear on the plane y = zs[i]
        zs = np.arange(len(weights))
        if len(collected_samples) == len(weights):
            zs = collected_samples

        new_zs = []
        max_frequency = 0
        for i in range(len(weights)):
            if (i + 1) % inteval == 0:
                a, b = np.histogram(weights[i].reshape([-1]), bins)
                ys = a
                xs = b[:-1] + 0.001
                new_zs.append(zs[i])
                max_frequency = max(np.max(a), max_frequency)
                verts.append(polygon_under_graph(xs, ys))

        poly = PolyCollection(verts, facecolors=['r', 'g', 'b', 'y'], alpha=.4)
        ax.add_collection3d(poly, zs=new_zs, zdir='y')
        override = {'fontsize': 'small', 'verticalalignment': 'top', 'horizontalalignment': 'center'}
        ax.set_xlabel('weights', override)
        ax.set_ylabel('steps', override)
        ax.set_zlabel('frequency', override)

        ax.set_xlim(min(bins), max(bins))
        ax.set_ylim(0, int(max(new_zs)))
        ax.set_zlim(0, int(max_frequency * 1.1))
        plt.title('Weights Histogram')

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    if imshow == True:
        if is_in_ipython() or is_in_colab():
            display.display(plt.gcf())
            # plt.close(fig)
        else:
            plt.ioff()
            plt.show(block=False)
    return fig


def plot_centerloss(plt, feat, labels, num_class=10, title='', enable_tensorboard=False, **kwargs):
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff', '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    fig = plt.figure()
    for i in range(num_class):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
    plt.legend(range(num_class), loc='upper right')
    plt.xlim(xmin=feat[:, 0].min(), xmax=feat[:, 0].max())
    plt.ylim(ymin=feat[:, 1].min(), ymax=feat[:, 1].max())
    plt.title(title + ' center loss')
    return fig


def plot_confusion_matrix(cm, class_names, figsize=(16, 8), normalize=False, title="Confusion matrix", fname=None,
                          noshow=False, enable_tensorboard=False, **kwargs):
    """Render the confusion matrix and return matplotlib's figure with it.
    Normalization can be applied by setting `normalize=True`.
    """
    cmap = cm.Oranges

    if normalize:
        cm = cm.astype(np.float32) / cm.sum(axis=1)[:, np.newaxis]

    fig = plt.figure(figsize=figsize)
    plt.title(title)
    plt.imshow(cm, interpolation="nearest", cmap=cmap)

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    # f.tick_params(direction='inout')
    # f.set_xticklabels(varLabels, rotation=45, ha='right')
    # f.set_yticklabels(varLabels, rotation=45, va='top')

    plt.yticks(tick_marks, class_names)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    if fname is not None:
        plt.savefig(fname=fname)

    # if not noshow:
    #     plt.show()

    return fig
