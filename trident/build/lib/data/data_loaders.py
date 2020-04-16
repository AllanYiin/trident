from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from backend.common import image_data_format,floatx,epsilon

import os
import glob
import warnings
import numpy as np
import tqdm
from tqdm import tqdm
import gzip
import six
from six.moves.urllib.error import HTTPError
from six.moves.urllib.error import URLError
from six.moves.urllib.request import urlopen

try:
    from urllib.request import urlretrieve
except ImportError:
    from six.moves.urllib.request import urlretrieve
from backend.image_common import *


_session =get_session()
_trident_dir=os.path.join(get_trident_dir(),'datasets')
_backend=_session.backend

if not os.path.exists(_trident_dir):
    try:
        os.makedirs(_trident_dir)
    except OSError:
        # Except permission denied and potential race conditions
        # in multi-threaded environments.
        pass


def to_onehot(arr):
    if isinstance(arr,list):
        arr=np.array(arr)
    elif not isinstance(arr,np.ndarray):
        raise ValueError('You should input a list of integer or ndarray.')
    items=np.unique(arr)
    items=np.argsort(items)
    if np.min(items)<0:
        raise ValueError('Negative value cannot convert to onhot.')
    elif np.sum(np.abs(np.round(arr)-arr))>0:
        raise ValueError('Only integer value can convert to onhot.')
    else:
        max_value=int(np.max(items))

        output_shape=list(arr.shape)
        output_shape.append(max_value+1)
        output=np.zeros(output_shape,dtype=floatx())
        arr=arr.astype(np.uint8)
        for i in range(max_value):
            onehot=np.zeros(max_value+1,dtype=floatx())
            onehot[i]=1
            output[arr==i]=onehot
        return output















def load_mnist( dataset_name='mnist',kind='train',is_flatten=None,is_onehot=None):
    dataset_name=dataset_name.strip().lower().replace('minist','mnist')

    if dataset_name.lower()  not in ['mnist','fashion-mnist']:
        raise ValueError('Only mnist or fashion-mnist are valid  dataset_name.')
    kind = kind.strip().lower().replace('ing', '')
    if _backend in ['tensorflow','cntk'] and is_onehot is None:
        is_onehot=True

    base = 'http://yann.lecun.com/exdb/mnist/'
    if dataset_name == 'fashion-mnist':
        base = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'

    dirname = os.path.join(_trident_dir,dataset_name)
    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError:
            # Except permission denied and potential race conditions
            # in multi-threaded environments.
            pass

    """Load MNIST data from `path`"""
    if dataset_name=='mnist' and kind=='test':
        kind='t10k'
    labels_file ='{0}-labels-idx1-ubyte.gz'.format(kind)
    images_file= '{0}-images-idx3-ubyte.gz'.format(kind)
    # if dataset_name == 'emnist' :
    #     labels_file='emnist-balanced-'+labels_file
    #     images_file = 'emnist-balanced-' + images_file

    download_file(base+labels_file, dirname, labels_file,dataset_name+'_labels_{0}'.format(kind))
    download_file(base+images_file, dirname, images_file,dataset_name+'_images_{0}'.format(kind))
    labels_path=os.path.join(dirname,labels_file)
    images_path = os.path.join(dirname, images_file)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,offset=8).astype(dtype=floatx())
        if _backend == 'pytorch':
            labels=np.squeeze(labels).astype(np.int64)
        if is_onehot==True:
            if _backend=='pytorch':
                warnings.warn('Pytorch not prefer onehot label, are you still want onehot label?', category='data loading', stacklevel=1,source='load_mnist')
            labels=to_onehot(labels)


    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784).astype(dtype=floatx())
        if is_flatten==False:
            images=np.reshape(images,(-1,28,28))
    return (images, labels)


class TqdmProgress(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize

def download_file(src,dirname,filename,desc=''):
    if os.path.exists(os.path.join(dirname,filename)):
        print('archive file is already existing, donnot need download again.')
    else:
        error_msg = 'URL fetch failure on {}: {} -- {}'
        try:
            try:
                with TqdmProgress(unit='B', unit_scale=True, miniters=1,desc=desc) as t:  # all optional kwargs
                    urlretrieve(src, filename=os.path.join(dirname,filename), reporthook=t.update_to, data=None)
            except HTTPError as e:
                raise Exception(error_msg.format(src,os.path.join(dirname,filename), e.code, e.__str__()))
            except URLError as e:
                raise Exception(error_msg.format(src,os.path.join(dirname,filename), e.errno, e.reason))
            except (Exception, KeyboardInterrupt) as e:
                if os.path.exists(os.path.join(dirname, filename)):
                    os.remove(os.path.join(dirname, filename))
        except:
            raise


def load_text(filname,delimiter=',',skiprows=0,label_index=None,is_onehot=None,shuffle=True):
    if _backend in ['tensorflow','cntk'] and is_onehot is None:
        is_onehot=True
    arr=np.genfromtxt(filname, delimiter=delimiter, skip_header=skiprows,dtype=floatx(),filling_values=0,autostrip=True)
    data,labels=None,None
    if label_index is None:
        data= arr
    else:
        if label_index==0:
            data,labels= arr[:,1:],arr[:,0:1]
        elif label_index==-1 or label_index==len(arr)-1:
            data,labels= arr[:,:-1],arr[:,-1:]
        else:
            rdata,labels=np.concatenate([arr[:,:label_index],arr[:,label_index+1:]],axis=0),arr[:,label_index:label_index+1]
    labels=np.squeeze(labels)
    if _backend == 'pytorch':
        labels = np.squeeze(labels).astype(np.int64)
    if is_onehot == True:
        if _backend == 'pytorch':
            warnings.warn('Pytorch not prefer onehot label, are you still want onehot label?', category='data loading', stacklevel=1, source='load_text')
        labels = to_onehot(labels)
    idxes = np.arange(len(data))
    if shuffle:

        np,random.shuffle(idxes)
        data=data[idxes]
    if labels is None:
        return (data,)
    else:
        if shuffle:
            labels = labels[idxes]
        return (data,labels)





