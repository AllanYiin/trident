from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys

from trident.backend.opencv_backend import array2image

from trident.misc.ipython_utils import is_in_ipython, is_in_colab
import math

if is_in_ipython():
    from IPython import display


if not is_in_colab:
    import matplotlib

    matplotlib.use('TkAgg' if not is_in_ipython() and not is_in_colab() else 'NbAgg')
else:
    import matplotlib

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import matplotlib.patches as patches
import matplotlib.font_manager

fonts = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
fontnames = [matplotlib.font_manager.FontProperties(fname=fname).get_name() for fname in fonts]
default_font=None


if sys.platform == 'win32':
    if 'Microsoft Sans Serif' in fontnames:
        default_font=matplotlib.rc('font', family='Microsoft Sans Serif' )
    else:
        for name in fontnames:
            if 'heiti' in name.lower():
                default_font=matplotlib.rc('font', family=name)
                break

import PIL
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import colorsys
import itertools
import numpy as np
from trident.backend.common import get_time_suffix, make_dir_if_need, unpack_singleton
from trident.data.image_common import *

__all__ = ['tile_rgb_images', 'loss_metric_curve', 'steps_histogram', 'generate_palette', 'plot_bbox', 'plot_3d_histogram', 'plot_centerloss']


def generate_palette(num_classes):
    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    return colors


def plot_bbox(x, img, color=None, label=None, line_thickness=None,**kwargs):
    img_shape = (img.height, img.width, 3)
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    # img = array2image(img)
    draw = ImageDraw.Draw(img)
    draw.rectangle(((x[0], x[1]), (x[2], x[3])), outline=color, fill=None, width=tl)
    fontcolor = (255, 255, 255)
    avg_color = np.array(list(color)).mean()
    if avg_color > 150:
        fontcolor = (0, 0, 0)
    if label and sys.platform == 'win32':
        font = ImageFont.truetype(fonts[fontnames.index('Microsoft Sans Serif')], int(math.sqrt(img_shape[0] / 1000) * 10 + 1))
        tf = max(tl - 1, 1)  # font thickness
        size = draw.textsize(label, font=font)
        offset = font.getoffset(label)
        draw.rectangle(((x[0], x[1] - size[1] - 2 * (offset[1] + 1)), (x[0] + 2 * (size[0] + offset[0] + 1), x[1])), fill=color,
                       width=2)
        draw.text((x[0] + 2, x[1] - size[1] - offset[1] - 1), u'{0}'.format(label), fill=fontcolor, font=font)

    # rgb_image = image2array(img)
    return img


def tile_rgb_images(*imgs, row=3, save_path=None, imshow=False,**kwargs):
    make_dir_if_need(save_path)
    distinct_row=list(set([ len(ims) for ims in imgs]))
    if len(distinct_row)>1:
        raise ValueError('imgs should have same length, but got {0}'.format(distinct_row))
    else:
        distinct_row=unpack_singleton(distinct_row)
    if 1<=row<distinct_row:
        distinct_row=row
    suffix = get_time_suffix()
    if len(imgs) == 1 and distinct_row == 1:
        img = array2image(imgs[0][0])
        filename = save_path.format(suffix)
        img.save(filename)
        plt.imshow(img)
        if imshow:
            if is_in_ipython():
                plt.axis("off")
                plt.ioff()
                display.display(plt.gcf())
            else:
                plt.axis("off")
                plt.imshow(img, interpolation="nearest", animated=True)
                plt.ioff()
                plt.gcf().show()
        return plt
    else:
        fig = plt.gcf()
        #fig.set_size_inches(len(imgs) * 2, row * 2)
        plt.clf()
        plt.ion()  # is not None:

        for m in range(distinct_row * len(imgs)):
            plt.subplot(distinct_row, len(imgs), m + 1)
            img = array2image((imgs[int(m % len(imgs))][int(m // len(imgs))]))
            plt.imshow(img, interpolation="nearest", animated=True)
            plt.axis("off")
        filename = save_path.format(suffix)
        plt.savefig(filename, bbox_inches='tight')
        if imshow == True:
            #plSize = fig.get_size_inches()
            #fig.set_size_inches((int(round(plSize[0] * 0.75, 0)), int(round(plSize[1] * 0.75, 0))))
            if is_in_ipython():
                plt.ioff()
                display.display(plt.gcf())
            else:
                plt.ioff()
                plt.show(block=False)
        return fig


def loss_metric_curve(losses, metrics,  legend=None, calculate_base='epoch', max_iteration=None,
                      save_path=None, imshow=False, **kwargs):
    colors=[]
    line_type=['-','--','-.',':']
    fig = plt.gcf()
    fig.set_size_inches(18, 8)
    plt.clf()
    plt.ion()  # is not None:

    plt.subplot(2, 2, 1)
    if losses.__class__.__name__=='HistoryBase':
        steps,values=losses.get_series('total_losses')
        plt.plot(steps,values)
        plt.legend(['loss'], loc='upper left')
    elif isinstance(losses, list):
        for item in losses:
            if  item.__class__.__name__=='HistoryBase':
                steps, values = item.get_series('total_losses')
                p=plt.plot(steps,values)
                colors.append(p[-1].get_color())
        if legend is not None:
            plt.legend(['{0}'.format(lg) for lg in legend], loc='upper right')
        else:
            plt.legend(['{0}'.format(i) for i in range(len(losses))], loc='upper right')

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel(calculate_base)

    if max_iteration is not None:
        plt.xlim(0, max_iteration)

    plt.subplot(2, 2, 2)
    if  metrics.__class__.__name__=='HistoryBase':
        for k, v in metrics.items():

            steps, values = metrics.get_series(k)
            plt.plot(steps, values)
        plt.legend(list(metrics.keys()), loc='upper left')
    elif isinstance(metrics, list):
        legend_list = []
        for i in range(len(metrics)):
            item = metrics[i]
            line_color=colors[i]
            if  item.__class__.__name__=='HistoryBase':
                for j in range(len(item.items())):
                    k=list(item.keys())[j]
                    steps, values = item.get_series(k)
                    plt.plot(steps, values,color=line_color,linestyle =line_type[j%4],linewidth=int((j//4)%4)+1)
                    if len(values) > 0 and legend is not None:
                        legend_list.append(['{0} {1}'.format(k, legend[i])])
                    elif len(values) > 0:
                        legend_list.append(['{0} {1}'.format(k, i)])
        plt.legend(legend_list, loc='upper left')

    plt.title('model metrics')
    plt.ylabel('metrics')
    plt.xlabel(calculate_base)

    if max_iteration is not None:
        plt.xlim(0, max_iteration)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    if imshow:
        if is_in_ipython():
            plt.ioff()
            display.display(plt.gcf())
        else:
            plt.ioff()
            plt.draw()
            plt.show(block=False)

    return fig


def polygon_under_graph(xlist, ylist):
    """
    Construct the vertex list which defines the polygon filling the space under
    the (xlist, ylist) line graph.  Assumes the xs are in ascending order.
    """
    return [(xlist[0], 0.), *zip(xlist, ylist), (xlist[-1], 0.)]


default_bins = []
default_bins.extend(np.arange(-0.02, 0.02, 0.002).tolist())
default_bins.extend(np.arange(-0.005, 0.005, 0.0005).tolist())
default_bins.extend(np.arange(-0.0002, 0.0002, 0.00002).tolist())
default_bins = sorted(list(set(default_bins)))


def plot_3d_histogram(ax, grads, sample_collected=None, bins=None, inteval=1, title='',**kwargs):
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


def steps_histogram(grads, weights=None, sample_collected=None, bins=None, size=(18, 8), inteval=1, title='', save_path=None,
                    imshow=False, enable_tensorboard=False,**kwargs):
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
        ax = fig.add_subplot(1, 2, 1, projection='3d') if grads is not None and weights is not None else fig.add_subplot(1, 1, 1, projection='3d')

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
        ax = fig.add_subplot(1, 2, 2, projection='3d') if grads is not None else fig.add_subplot(1, 1, 1, projection='3d')

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
            if i % inteval == 0:
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
            plt.close(fig)
        else:
            plt.ioff()
            plt.show(block=False)
    return fig


def plot_centerloss(plt, feat, labels, num_class=10, title='', enable_tensorboard=False,**kwargs):
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff', '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    fig= plt.figure()
    for i in range(num_class):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
    plt.legend(range(num_class), loc='upper right')
    plt.xlim(xmin=feat[:, 0].min(), xmax=feat[:, 0].max())
    plt.ylim(ymin=feat[:, 1].min(), ymax=feat[:, 1].max())
    plt.title(title + ' center loss')
    return fig


def plot_confusion_matrix(cm, class_names, figsize=(16, 16), normalize=False, title="Confusion matrix", fname=None,
                          noshow=False , enable_tensorboard=False,**kwargs):
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

    if not noshow:
        plt.show()

    return fig



