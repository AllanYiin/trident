from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import threading
import os
import subprocess
import sys
import glob

import warnings

import tqdm
from  sys import stderr
import requests
from tqdm import tqdm
import gzip
import tarfile
import zipfile
import shutil
import hashlib
import glob
import six
import re
import numpy as np
from .utils import *
from ..backend.common import  get_session,get_trident_dir
from ..backend.image_common import *
from .datasets_common import *
try:
    from urllib.request import urlretrieve
except ImportError:
    from six.moves.urllib.request import urlretrieve



_session = get_session()
_trident_dir = os.path.join(get_trident_dir(), 'datasets')
_backend=_session.backend
if 'TRIDENT_BACKEND' in os.environ:
    _backend = os.environ['TRIDENT_BACKEND']

if not os.path.exists(_trident_dir):
    try:
        os.makedirs(_trident_dir)
    except OSError:
        # Except permission denied and potential race conditions
        # in multi-threaded environments.
        pass




def to_onehot(arr):
    if isinstance(arr, list):
        arr = np.array(arr)
    elif not isinstance(arr, np.ndarray):
        raise ValueError('You should input a list of integer or ndarray.')
    items = np.unique(arr)
    items = np.argsort(items)
    if np.min(items) < 0:
        raise ValueError('Negative value cannot convert to onhot.')
    elif np.sum(np.abs(np.round(arr) - arr)) > 0:
        raise ValueError('Only integer value can convert to onhot.')
    else:
        max_value = int(np.max(items))

        output_shape = list(arr.shape)
        output_shape.append(max_value + 1)
        output = np.zeros(output_shape, dtype=floatx())
        arr = arr.astype(np.uint8)
        for i in range(max_value):
            onehot = np.zeros(max_value + 1, dtype=floatx())
            onehot[i] = 1
            output[arr == i] = onehot
        return output




#
# def download_image(image, temproot, imageroot, flog=None):
#     # Check existing file.
#     try:
#         temppath, imagepath = (os.path.join(root, image['path']) for root in (temproot, imageroot))
#
#     except Exception as e:
#         sys.stderr('Unexpected exception before attempting download of image {0}.'.format(e))
#
#
#     # GET and save to temp location.
#     try:
#         r = requests.get(image['url'])
#         if r.status_code == 200:
#             ensure_parent_dir(temppath)
#             with open(temppath, 'wb') as fout:
#                 for chunk in r.iter_content(1024): fout.write(chunk)
#             logmsg('Saved  {}.'.format(temppath), flog=flog)
#         else:
#             logmsg('Status code {} when requesting {}.'.format(r.status_code, image['url']))
#             return DownloadResult.DOWNLOAD_FAILED
#     except Exception as e:
#         stderr('Unexpected exception when downloading image {!r}.'.format(image), e, flog=flog)
#         return DownloadResult.DOWNLOAD_FAILED
#     # Check contents.
#     try:
#         if check_image(image, temppath):
#             stderr('Image contents look good.)
#         else:
#             stderr('Image contents are wrong.')
#             return DownloadResult.MD5_FAILED
#     except Exception as e:
#         stderr('Unexpected exception when checking file contents for image {!r}.'.format(image), e)
#         return DownloadResult.MYSTERY_FAILED
#     # Move image to final location.
#     try:
#         ensure_parent_dir(imagepath)
#         os.rename(temppath, imagepath)
#     except Exception as e:
#         stderr('Unexpected exception when moving file from {} to {} for image {!r}.'.format(temppath, imagepath, image), e, flog=flog)
#         return DownloadResult.MYSTERY_FAILED
#     return DownloadResult.NEW_OK



def load_mnist(dataset_name='mnist', kind='train', is_flatten=None, is_onehot=None):
    dataset_name = dataset_name.strip().lower().replace('minist', 'mnist')

    if dataset_name.lower() not in ['mnist', 'fashion-mnist']:
        raise ValueError('Only mnist or fashion-mnist are valid  dataset_name.')
    kind = kind.strip().lower().replace('ing', '')
    if _backend in ['tensorflow', 'cntk'] and is_onehot is None:
        is_onehot = True

    base = 'http://yann.lecun.com/exdb/mnist/'
    if dataset_name == 'fashion-mnist':
        base = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'

    dirname = os.path.join(_trident_dir, dataset_name)
    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError:
            # Except permission denied and potential race conditions
            # in multi-threaded environments.
            pass

    """Load MNIST data from `path`"""
    if dataset_name == 'mnist' and kind == 'test':
        kind = 't10k'
    labels_file = '{0}-labels-idx1-ubyte.gz'.format(kind)
    images_file = '{0}-images-idx3-ubyte.gz'.format(kind)
    # if dataset_name == 'emnist' :
    #     labels_file='emnist-balanced-'+labels_file
    #     images_file = 'emnist-balanced-' + images_file

    download_file(base + labels_file, dirname, labels_file, dataset_name + '_labels_{0}'.format(kind))
    download_file(base + images_file, dirname, images_file, dataset_name + '_images_{0}'.format(kind))
    labels_path = os.path.join(dirname, labels_file)
    images_path = os.path.join(dirname, images_file)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
        if _backend == 'pytorch':
            labels = np.squeeze(labels).astype(np.int64)
        if is_onehot == True:
            if _backend == 'pytorch':
                warnings.warn('Pytorch not prefer onehot label, are you still want onehot label?',
                              category='data loading', stacklevel=1, source='load_mnist')
            labels = to_onehot(labels)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16)
        images=np.reshape(images,(len(labels), 784)).astype(dtype=_session.floatx)
        if is_flatten == False:
            images = np.reshape(images, (-1, 1,28, 28))

    dataset = DataProvider(dataset_name,data=images, labels=labels, scenario='train')
    dataset.is_flatten=is_flatten
    dataset.current_scenario=kind
    dataset.binding_class_names([0,1,2,3,4,5,6,7,8,9] if dataset_name=='mnist' else ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot'],'en-US')

    return dataset


def load_cifar(dataset_name='cifar10', kind='train', is_flatten=None, is_onehot=None):
    dataset_name = dataset_name.strip().lower().replace(' ', '')

    if dataset_name.lower() not in ['cifar10', 'cifar100']:
        raise ValueError('Only cifar10 or cifar100 are valid  dataset_name.')
    kind = kind.strip().lower().replace('ing', '')
    if _backend in ['tensorflow', 'cntk'] and is_onehot is None:
        is_onehot = True

    baseURL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    if dataset_name == 'cifar100':
        baseURL = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'

    dirname = os.path.join(_trident_dir, dataset_name.strip())
    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError:
            # Except permission denied and potential race conditions
            # in multi-threaded environments.
            pass

    """Load CIFAR data from `path`"""

    download_file(baseURL, dirname, baseURL.split('/')[-1].strip(), dataset_name)
    file_path = os.path.join(dirname, baseURL.split('/')[-1].strip())
    if '.tar' in file_path:
        extract_archive(file_path, dirname, archive_format='tar')
    extract_path= os.path.join(dirname, baseURL.split('/')[-1].strip().split('.')[0])
    filelist = [f for f in os.listdir(extract_path) if os.path.isfile(os.path.join(extract_path, f))]

    data, labels = open_pickle(os.path.join(extract_path,kind), 'data','fine_labels')
    data = data.reshape(data.shape[0], 3, 32, 32).astype(_session.floatx)
    if _backend == 'tensorflow':
        data=data.transpose([0,2,3,1])

    if _backend == 'pytorch':
        labels=np.squeeze(labels).astype(np.int64)
    else:
        if is_onehot == None:
            is_onehot = True
    if is_onehot==True:
        if _backend=='pytorch':
            warnings.warn('Pytorch not prefer onehot label, are you still want onehot label?', category='dataloading', stacklevel=1,source='load_cifar')
        labels=to_onehot(labels)


    dataset = DataProvider(dataset_name,data=data, labels=labels, scenario='train')
    dataset.binding_class_names(['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']if dataset_name=='cifar10' else [],'en-US')

    return dataset

def load_birdsnap(dataset_name='birdsnap', kind='train', is_flatten=None, is_onehot=None):
    dataset_name = dataset_name.strip().lower().replace(' ', '')

    if dataset_name.lower() not in ['birdsnap']:
        raise ValueError('Only _birdsnap are valid  dataset_name.')

    if _backend in ['tensorflow', 'cntk'] and is_onehot is None:
        is_onehot = True

    baseURL = 'http://thomasberg.org/datasets/birdsnap/1.1/birdsnap.tgz'
    dirname = os.path.join(_trident_dir, dataset_name.strip())
    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError:
            # Except permission denied and potential race conditions
            # in multi-threaded environments.
            pass

    """Load BirdSnap data from `path`"""
    download_file(baseURL, dirname, baseURL.split('/')[-1].strip(), dataset_name)
    file_path = os.path.join(dirname, baseURL.split('/')[-1].strip())
    if '.tar' in file_path:
        extract_archive(file_path, dirname, archive_format='tar')
    else:
        extract_archive(file_path, dirname, archive_format='auto')
    extract_path= os.path.join(dirname, baseURL.split('/')[-1].strip().split('.')[0])
    pid = subprocess.Popen([sys.executable, os.path.join(extract_path,"get_birdsnap.py")], stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)  # call subprocess

    filelist = [f for f in os.listdir(extract_path) if os.path.isfile(os.path.join(extract_path, f))]
    #
    #
    # images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784).astype(
    #     dtype=floatx())
    # if is_flatten == False:
    #     images = np.reshape(images, (-1, 28, 28))
    #
    #
    # labels = np.frombuffer(os.path.join(extract_path,'species.txt'), dtype=np.uint8, offset=8).astype(dtype=floatx())
    # if _backend == 'pytorch':
    #     labels = np.squeeze(labels).astype(np.int64)
    # if is_onehot == True:
    #     if _backend == 'pytorch':
    #         warnings.warn('Pytorch not prefer onehot label, are you still want onehot label?',
    #                       category='data loading', stacklevel=1, source='load_mnist')
    #     labels = to_onehot(labels)
    images=[]
    labels=[]
    return (images, labels)


def load_text(filname, delimiter=',', skiprows=0, label_index=None, is_onehot=None, shuffle=True):
    if _backend in ['tensorflow', 'cntk'] and is_onehot is None:
        is_onehot = True
    arr = np.genfromtxt(filname, delimiter=delimiter, skip_header=skiprows, dtype=floatx(), filling_values=0,
                        autostrip=True)
    data, labels = None, None
    if label_index is None:
        data = arr
    else:
        if label_index == 0:
            data, labels = arr[:, 1:], arr[:, 0:1]
        elif label_index == -1 or label_index == len(arr) - 1:
            data, labels = arr[:, :-1], arr[:, -1:]
        else:
            rdata, labels = np.concatenate([arr[:, :label_index], arr[:, label_index + 1:]], axis=0), arr[:,
                                                                                                      label_index:label_index + 1]
    labels = np.squeeze(labels)
    if _backend == 'pytorch':
        labels = np.squeeze(labels).astype(np.int64)
    if is_onehot == True:
        if _backend == 'pytorch':
            warnings.warn('Pytorch not prefer onehot label, are you still want onehot label?', category='data loading',
                          stacklevel=1, source='load_text')
        labels = to_onehot(labels)
    idxes = np.arange(len(data))
    dataset = DataProvider(filname.split('/')[-1].strip().split('.')[0],data=data, labels=labels, scenario='train')

    return dataset


# def _load_images_from_folder(basefolder, extensions= ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'), is_onehot=None, shuffle=True):
#     extensions='|'.join(list(extensions)).strip().replace('.','')
#     classes =sorted([d.name for d in os.scandir(basefolder) if d.is_dir()])
#     for cls in classes:
#         for root, _, files in sorted(os.walk(os.path.join(basefolder,cls))) :
#             for fname in sorted(files):
#                 if re.match(r'([\w]+\.(?:' + extensions + '))', f):
#                     path = os.path.join(root, fname)
#                         item = (path, class_to_idx[target])
#
#                         images.append(item)




def load_stanford_cars(dataset_name='cars', kind='train', is_flatten=None, is_onehot=None):
    dataset_name = dataset_name.strip().lower()

    if dataset_name.lower() not in ['car','cars']:
        raise ValueError('Only Cars is valid  dataset_name.')
    kind = kind.strip().lower().replace('ing', '')
    if _backend in ['tensorflow', 'cntk'] and is_onehot is None:
        is_onehot = True

    train_url='http://imagenet.stanford.edu/internal/car196/cars_train.tgz'
    test_url = 'http://imagenet.stanford.edu/internal/car196/cars_test.tgz'
    label_url='https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz'
    dirname = os.path.join(_trident_dir, dataset_name)
    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError:
            # Except permission denied and potential race conditions
            # in multi-threaded environments.
            pass



    download_file(train_url, dirname, train_url.split('/')[-1], dataset_name + '_images_{0}'.format('train'))
    train_imgs_path = os.path.join(dirname, train_url.split('/')[-1])

    download_file(test_url, dirname, test_url.split('/')[-1], dataset_name + '_images_{0}'.format('test'))
    test_imgs_path = os.path.join(dirname, test_url.split('/')[-1])

    download_file(label_url, dirname, label_url.split('/')[-1], dataset_name + '_labels_{0}'.format(kind))
    labels_path = os.path.join(dirname, label_url.split('/')[-1])


    extract_archive(os.path.join(dirname, train_url.split('/')[-1].strip()), dirname, archive_format='tar')
    extract_archive(os.path.join(dirname, test_url.split('/')[-1].strip()), dirname, archive_format='tar')
    extract_archive(os.path.join(dirname, label_url.split('/')[-1].strip()), dirname, archive_format='tar')

    extract_path = os.path.join(dirname, label_url.split('/')[-1].strip().split('.')[0].replace('car_devkit','devkit'))
    cars_meta=read_mat(os.path.join(extract_path,'cars_meta.mat'))['class_names'][0]  #size 196

    cars_annos = read_mat(os.path.join(extract_path, 'cars_train_annos.mat'))['annotations'][0]
    if kind=='test':
        cars_annos = read_mat(os.path.join(extract_path, 'cars_test_annos.mat'))['annotations'][0]

    images_path=[]
    labels=[]
    for item in cars_annos:
        bbox_x1,bbox_x2,bbox_y1,bbox_y2,classid,fname=item
        images_path.append(fname)
        labels.append(np.array([bbox_x1,bbox_y1,bbox_x2,bbox_y2,classid]))

    dataset = DataProvider(dataset_name,data=data, labels=labels, scenario='train')
    dataset.binding_class_names(cars_meta,'en-US')

    return dataset

