"""data_loader: The ready-to-use data provider"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import io
import re
import struct
from xml.etree import ElementTree
import numpy as np
import copy
import cv2
from trident.backend.opencv_backend import image2array, array2image
from trident.data.image_common import list_images
from trident.data.data_provider import *
from trident.data.dataset import *
from trident.data.mask_common import *
from trident.data.utils import *
from trident.backend.common import floatx,OrderedDict
from trident.backend.tensorspec import *
from trident.misc.ipython_utils import *

try:
    from urllib.request import urlretrieve
except ImportError:
    from six.moves.urllib.request import urlretrieve

_session = get_session()
_trident_dir = os.path.join(_session.trident_dir, 'datasets')
_backend = get_backend()

if 'TRIDENT_BACKEND' in os.environ:
    _backend = os.environ['TRIDENT_BACKEND']


if _backend=='pytorch':
    from trident.backend.pytorch_backend import *
    from  trident.backend.pytorch_ops import *

elif _backend == 'tensorflow':
    from trident.backend.tensorflow_backend import *
    from trident.backend.tensorflow_ops import *


if not os.path.exists(_trident_dir):
    try:
        os.makedirs(_trident_dir)
    except OSError:
        # Except permission denied and potential race conditions
        # in multi-threaded environments.
        pass





def load_mnist(dataset_name='mnist', **kwargs):
    """data loader for mnist data

    Args:
        dataset_name (string): if 'minist'  will return the traditional mnist data/ label,
            if 'fashion-mnist' will return the fashion-mnist' data


    Returns:
        a tuple of data and label

    """
    dataset_name = dataset_name.strip().lower().replace('minist', 'mnist')

    if dataset_name.lower() not in ['mnist', 'fashion-mnist']:
        raise ValueError('Only mnist or fashion-mnist are valid  dataset_name.')

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
    trainData = None
    testData = None
    for kind in ['train', 'test']:
        labels_file = '{0}-labels-idx1-ubyte.gz'.format(
            't10k' if dataset_name in ('mnist', 'fashion-mnist') and kind == 'test' else kind)
        images_file = '{0}-images-idx3-ubyte.gz'.format(
            't10k' if dataset_name in ('mnist', 'fashion-mnist') and kind == 'test' else kind)
        # if dataset_name == 'emnist' :
        #     labels_file='emnist-balanced-'+labels_file
        #     images_file = 'emnist-balanced-' + images_file

        is_data_download = download_file(base + labels_file, dirname, labels_file,
                                         dataset_name + '_labels_{0}'.format(kind))
        is_label_download = download_file(base + images_file, dirname, images_file,
                                          dataset_name + '_images_{0}'.format(kind))
        if is_data_download and is_label_download:
            labels_path = os.path.join(dirname, labels_file)
            images_path = os.path.join(dirname, images_file)
            labeldata = None
            imagedata = None
            with gzip.open(labels_path, 'rb') as lbpath:
                labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
                labels = np.squeeze(labels).astype(np.int64)
                labeldata = LabelDataset(labels.tolist(),object_type=ObjectType.classification_label)

            with gzip.open(images_path, 'rb') as imgpath:
                images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16)
                images = np.reshape(images, (len(labels), 784)).astype(dtype=_session.floatx)
                images = np.reshape(images, (-1, 28, 28))
                imagedata = ImageDataset(images, object_type=ObjectType.gray)
            if kind == 'train':
                trainData = Iterator(data=imagedata, label=labeldata)
            else:
                testData = Iterator(data=imagedata, label=labeldata)

            dataset = DataProvider(dataset_name, traindata=trainData, testdata=testData)
            dataset.binding_class_names(
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] if dataset_name == 'mnist' else ['T-shirt/top', 'Trouser', 'Pullover',
                                                                                'Dress', 'Coat', 'Sandal', 'Shirt',
                                                                                'Sneaker', 'Bag', 'Ankle boot'],
                'en-US')

            return dataset
        return None


def load_cifar(dataset_name='cifar10'):
    """data loader for mnist data

    Args:
        dataset_name (string): if 'cifar10'  will return the traditional cifar10 data/ label,
            if 'cifar100' will return the cifar100 data


    Returns:
        a tuple of data and label

    """
    dataset_name = dataset_name.strip().lower().replace(' ', '')

    if dataset_name.lower() not in ['cifar10', 'cifar100']:
        raise ValueError('Only cifar10 or cifar100 are valid  dataset_name.')
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
    _,filename,ext=split_path(baseURL)
    download_file(baseURL, dirname, filename+ext, dataset_name)
    file_path = os.path.join(dirname, filename+ext)


    if '.tar' in ext:
        extract_archive(file_path, dirname, archive_format='auto')
    filelist = glob.glob(dirname + '/*/*.*')
    extract_path ,_,_= split_path(filelist[0])
    filelist = [f for f in os.listdir(extract_path) if os.path.isfile(os.path.join(extract_path, f))]
    data=[]
    label=[]
    test_data=[]
    test_label=[]
    for file_path in filelist:
        if 'data_batch' in file_path:
            with open(os.path.join(extract_path,file_path), 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                data.append(entry['data'])
                label.append(entry['labels'])
        elif 'test_batch' in file_path:
            with open(os.path.join(extract_path,file_path), 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                test_data.append(entry['data'])
                test_label.append(entry['labels'])
    data = np.concatenate(data)
    data = data.reshape((data.shape[0], 3, 32, 32))
    data = data.transpose(0, 2, 3, 1).astype(np.float32)

    test_data = np.concatenate(test_data)
    test_data = test_data.reshape((test_data.shape[0], 3, 32, 32))
    test_data = test_data.transpose(0, 2, 3, 1).astype(np.float32)

    # Prepare labels
    label = np.concatenate(label)
    test_label = np.concatenate(test_label)

    trainData = Iterator(data=ImageDataset(data,object_type=ObjectType.rgb), label=LabelDataset(label,object_type=ObjectType.classification_label))
    testData = Iterator(data=ImageDataset(test_data,object_type=ObjectType.rgb), label=LabelDataset(test_label,object_type=ObjectType.classification_label))
    dataset = DataProvider(dataset_name, traindata=trainData, testdata=testData)
    dataset.binding_class_names(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship',
                                 'truck'] if dataset_name == 'cifar10' else [], 'en-US')
    return dataset

def load_kaggle(dataset_name='dogs-vs-cats',is_onehot=False):
    dataset_name = dataset_name.strip().lower().replace(' ', '')

    if dataset_name.lower() not in ['dogs-vs-cats']:
        raise ValueError('Only dogscats are valid  dataset_name.')

    if _backend in ['tensorflow', 'cntk'] and is_onehot is None:
        is_onehot = True

    baseURL = 'https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765'
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
    extract_archive(file_path, dirname, archive_format='zip')

    extract_path = os.path.join(dirname, baseURL.split('/')[-1].strip().split('.')[0])

    images = []
    labels = []
    return (images, labels)



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
    extract_path = os.path.join(dirname, baseURL.split('/')[-1].strip().split('.')[0])
    pid = subprocess.Popen([sys.executable, os.path.join(extract_path, "get_birdsnap.py")], stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, stdin=subprocess.PIPE)  # call subprocess

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
    images = []
    labels = []
    return (images, labels)


def load_text(filname=None, data=None, label=None,unit='char',mode='next_word',section_delimiter='\n\n',sequence_start_at='random',is_onehot=False,encoding='utf-8-sig',sequence_length=64, return_corpus=False,**kwargs):
    valid_sequence_start_at=['random','slide','follow_up','section_start']
    valid_mode=['next_word','skip_gram', 'cbow','onehot','1to1_seq2seq']
    if mode not in valid_mode:
        raise  ValueError('{0} is not valid mode '.format(mode))

    original_corpus = None
    corpus = None
    output_corpus=None
    if filname is not None:
        with io.open(filname, encoding=encoding) as f:
            original_corpus = f.read().lower()
            if unit == 'char':
                corpus = list(original_corpus)
            elif unit == 'word':
                corpus = original_corpus.split(' \t')
    if data is not None:
        if isinstance(data,str):
            if unit == 'char':
                corpus = list(data)
            elif unit == 'word':
                corpus = data.split(' \t')
        elif hasattr(data,"__iter__"):
            corpus= '\n\n'.join(data)
            if unit == 'char':
                corpus = list(corpus)
            elif unit == 'word':
                corpus = corpus.replace('\n\n',' \n \n ').split(' \t')
    if label is not None:
        if isinstance(label,str):
            if unit == 'char':
                output_corpus = list(label)
            elif unit == 'word':
                output_corpus = label.split(' \t')
        elif hasattr(label,"__iter__"):
            output_corpus = '\n\n'.join(label)
            if unit == 'char':
                output_corpus = list(output_corpus)
            elif unit == 'word':
                output_corpus = output_corpus.replace('\n\n',' \n \n ').split(' \t')
    dataprovider=None
    if mode=='next_word':
        corpus1=copy.deepcopy(corpus)
        corpus2= copy.deepcopy(corpus)
        data_seq=TextSequenceDataset(corpus1,sequence_length=sequence_length,is_onehot=is_onehot,symbol='input',sequence_offset=0,sequence_start_at=sequence_start_at,object_type=ObjectType.corpus)
        labels_seq =TextSequenceDataset(corpus2,sequence_length=sequence_length,is_onehot=is_onehot,symbol='label',sequence_offset=1,sequence_start_at=sequence_start_at,object_type=ObjectType.corpus)
        traindata=Iterator(data=data_seq,label=labels_seq)
        dataprovider = TextSequenceDataProvider(filname.split('/')[-1].strip().split('.')[0],traindata=traindata)
    elif mode=='skip_gram':
        corpus1 = copy.deepcopy(corpus)
        corpus2 = copy.deepcopy(corpus)
        data_seq = TextSequenceDataset(corpus1, sequence_length=sequence_length, is_onehot=is_onehot, symbol='input', sequence_offset=0, sequence_start_at=sequence_start_at)
        labels_seq = TextSequenceDataset(corpus2, sequence_length=sequence_length, is_onehot=is_onehot, symbol='label', sequence_offset=[-1,1], sequence_start_at=sequence_start_at)
        traindata = Iterator(data=data_seq, label=labels_seq)
        dataprovider = TextSequenceDataProvider(filname.split('/')[-1].strip().split('.')[0], traindata=traindata)
    elif mode == '1to1_seq2seq':
        if len(corpus)==len(output_corpus):
            data_seq = TextSequenceDataset(corpus, sequence_length=sequence_length, is_onehot=is_onehot, symbol='input', sequence_offset=0, sequence_start_at=sequence_start_at,object_type=ObjectType.corpus)
            labels_seq = TextSequenceDataset(output_corpus, sequence_length=sequence_length, is_onehot=is_onehot, symbol='label', sequence_offset=0, sequence_start_at=sequence_start_at,object_type=ObjectType.sequence_label)
            traindata = Iterator(data=data_seq, label=labels_seq)
            dataprovider = TextSequenceDataProvider('1to1_seq2seq', traindata=traindata)
        else:
            raise  ValueError('data ({0}) and label({1}) should have the same length in 1to1_seq2seq mide.'.format(len(corpus),len(output_corpus)))
    if return_corpus:
        return dataprovider,original_corpus
    else:
        return dataprovider


def load_folder_images(dataset_name='', base_folder=None, classes=None, shuffle=True, folder_as_label=True,
                       object_type=ObjectType.rgb):
    base_folder = sanitize_path(base_folder)
    if base_folder is not None and os.path.exists(base_folder):
        print(base_folder)
        if folder_as_label == True:
            class_names = []
            if classes is not None and isinstance(classes, list) and len(classes) > 0:
                class_names.extend(classes)
            else:
                for subdir in sorted(os.listdir(base_folder)):
                    if os.path.isdir(os.path.join(base_folder, subdir)):
                        class_names.append(subdir)
            if len(class_names) == 0:
                raise ValueError('No subfolder in base folder.')
            class_names = list(sorted(set(class_names)))
            print(class_names)
            labels = []
            imgs = []
            for i in range(len(class_names)):
                class_name = class_names[i]
                class_imgs = glob.glob(base_folder + '/{0}/*.*g'.format(class_name))
                if len(class_imgs)==0:
                    class_imgs = glob.glob(base_folder + '/{0}/*/*.*g'.format(class_name))
                print(base_folder + '/{0}/*.*g'.format(class_name))
                print(len(class_imgs))
                labels.extend([i] * len(class_imgs))
                imgs.extend(class_imgs)

            imagedata = ImageDataset(imgs, object_type=ObjectType.rgb, get_image_mode=GetImageMode.processed)
            print('extract {0} images...'.format(len(imagedata)))
            labelsdata = LabelDataset(labels,object_type=ObjectType.classification_label)
            labelsdata.binding_class_names(class_names)

            traindata = Iterator(data=imagedata, label=labelsdata)
            dataset = DataProvider(dataset_name, traindata=traindata)
            dataset.binding_class_names(class_names)

        else:
            imgs = list_images(base_folder)
            imagedata = ImageDataset(imgs, object_type=ObjectType.rgb, get_image_mode=GetImageMode.processed)
            traindata = Iterator(data=imagedata)
            dataset = DataProvider(dataset_name, traindata=traindata)
        return dataset
    else:
        raise ValueError('')


def load_stanford_cars(dataset_name='cars', kind='train', is_flatten=None, is_onehot=None):
    dataset_name = dataset_name.strip().lower()

    if dataset_name.lower() not in ['car', 'cars']:
        raise ValueError('Only Cars is valid  dataset_name.')
    kind = kind.strip().lower().replace('ing', '')
    if _backend in ['tensorflow', 'cntk'] and is_onehot is None:
        is_onehot = True

    train_url = 'http://imagenet.stanford.edu/internal/car196/cars_train.tgz'
    test_url = 'http://imagenet.stanford.edu/internal/car196/cars_test.tgz'
    label_url = 'https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz'
    dirname = os.path.join(_trident_dir, dataset_name)
    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError:
            # Except permission denied and potential race conditions
            # in multi-threaded environments.
            pass

    download_file(train_url, dirname, train_url.split('/')[-1], dataset_name + '_images_{0}'.format('train'))
    tar_file_path = os.path.join(dirname, train_url.split('/')[-1])

    download_file(test_url, dirname, test_url.split('/')[-1], dataset_name + '_images_{0}'.format('test'))
    test_imgs_path = os.path.join(dirname, test_url.split('/')[-1])

    download_file(label_url, dirname, label_url.split('/')[-1], dataset_name + '_labels_{0}'.format(kind))
    labels_path = os.path.join(dirname, label_url.split('/')[-1])

    extract_archive(os.path.join(dirname, train_url.split('/')[-1].strip()), dirname, archive_format='tar')
    extract_archive(os.path.join(dirname, test_url.split('/')[-1].strip()), dirname, archive_format='tar')
    extract_archive(os.path.join(dirname, label_url.split('/')[-1].strip()), dirname, archive_format='tar')

    extract_path = os.path.join(dirname, label_url.split('/')[-1].strip().split('.')[0].replace('car_devkit', 'devkit'))
    cars_meta = read_mat(os.path.join(extract_path, 'cars_meta.mat'))['class_names'][0]  # size 196

    cars_annos = read_mat(os.path.join(extract_path, 'cars_train_annos.mat'))['annotations'][0]
    if kind == 'test':
        cars_annos = read_mat(os.path.join(extract_path, 'cars_test_annos.mat'))['annotations'][0]

    images_path = []
    labels = []
    for item in cars_annos:
        bbox_x1, bbox_x2, bbox_y1, bbox_y2, classid, fname = item
        images_path.append(fname)
        labels.append(np.array([bbox_x1, bbox_y1, bbox_x2, bbox_y2, classid]))

    dataset = DataProvider(dataset_name, data=images_path, labels=labels, scenario='train')
    dataset.binding_class_names(cars_meta, 'en-US')

    return dataset


def load_lfw(format='aligned_face', is_paired=False):
    """

    Args:
        format (str):
        is_paired (bool): if True, will return  anchor-positive-negative pair, or return image-classification label.

    Returns:

    Exsamples:
        >>> dataloader=load_lfw(only_face_area=True, only_aligned_face=True)
        >>> len(dataloader)
        10

    """
    dataset_name = 'lfw'
    dirname = os.path.join(_trident_dir, dataset_name)
    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError:
            # Except permission denied and potential race conditions
            # in multi-threaded environments.
            pass

    tar_file_path = os.path.join(dirname, 'lfw.tgz')
    if format=='aligned_face':
        tar_file_path = os.path.join(dirname, 'lfw-crop.tar')
        download_file_from_google_drive('1sVbU8NHC7kzDkqByTtPiib9_EY23eeW7', dirname, 'lfw-crop.tar')
    elif format=='raw':
        download_file('http://vis-www.cs.umass.edu/lfw/lfw.tgz', dirname, 'lfw.tgz')

    download_file('http://vis-www.cs.umass.edu/lfw/pairsDevTrain.txt', dirname, 'pairsDevTrain.txt')
    download_file('http://vis-www.cs.umass.edu/lfw/pairsDevTest.txt', dirname, 'pairsDevTest.txt')

    # if only_face_area:
    #     if only_aligned_face:

    extract_archive(tar_file_path, dirname, archive_format='tar')

    # if _backend=='tensorflow':
    #     import trident.models.tensorflow_mtcnn as mtcnn
    # else:
    #     import trident.models.pytorch_mtcnn as mtcnn
    #
    # detector = mtcnn.Mtcnn(pretrained=True,verbose=False)
    # detector.minsize = 70
    #
    # detector.detection_threshould = [0.8, 0.8, 0.9]
    # detector.nms_threshould = [0.5, 0.5, 0.3]
    # faces = glob.glob(os.path.join(dirname, 'lfw-deepfunneled' if only_aligned_face else 'lfw') + '/*/*.*g')
    # faces_crop=glob.glob(os.path.join(dirname, 'lfw-crop')+ '/*/*.*g')
    # if len(faces_crop)==0 or len(faces_crop)!=len(faces):
    #     for  m in range(len(faces)):
    #         img_path=faces[m]
    #         new_path = img_path.replace(os.path.join(dirname, 'lfw-deepfunneled' if only_aligned_face else 'lfw'), os.path.join(dirname, 'lfw-crop'))
    #
    #         if os.path.exists(new_path) and image2array(new_path) is not None:
    #             pass
    #         else:
    #             make_dir_if_need(new_path)
    #             img = read_image(img_path)
    #             detector.detection_threshould = [0.8, 0.8, 0.9]
    #             results = detector.infer_single_image(img_path)
    #             if len(results)==0 or results is None:
    #                 detector.detection_threshould = [0.5, 0.7, 0.8]
    #                 results = detector.infer_single_image(img_path)
    #
    #             results = detector.rerec(to_tensor(results), img.shape)
    #             results[:, 0] = clip(results[:, 0], 0, img.shape[1])
    #             results[:, 1] = clip(results[:, 1], 0, img.shape[0])
    #             results[:, 2] = clip(results[:, 2], 0, img.shape[1])
    #             results[:, 3] = clip(results[:, 3], 0, img.shape[0])
    #             area=detector.area_of(results[:,:2],results[:,2:4])
    #             area_mask=area>500
    #             results=to_numpy(results[area_mask,:])
    #
    #             if len(results)>1:
    #                 area=area[area_mask]
    #                 idx=argmax(area).item()
    #                 results=results[idx:idx+1]
    #             if results is not None and len(results)>0:
    #                 for k in range(len(results)):
    #                     result =np.round(results[k][:4]).astype(np.uint8)
    #                     x1, y1, x2, y2 = result[0],result[1],result[2],result[3]
    #                     crop_img =np.clip( img.copy()[y1:y2,x1:x2, :],0,255).astype(np.uint8)
    #
    #                     array2image(crop_img).save(new_path)
    #             else:
    #
    #                 print(img_path+' get {0} results!'.format(results))
    #
    #         sys.stdout.write('\r {0}/{1} image processed!'.format(m,len(faces)))
    #         sys.stdout.flush()
    #
    #
    data_provider = load_folder_images(dataset_name,os.path.join(dirname,'lfw-crop' if format=='aligned_face' else 'lfw-deepfunneled'))

    if is_paired:
        storage=OrderedDict()
        data=list(data_provider.traindata.data)
        label=list(data_provider.traindata.label)
        class_names=data_provider._class_names
        for i in range(len(data)):
            class_label=class_names[label[i]]
            if class_label not in storage:
                storage[class_label] = []
            storage[class_label].append(data[i])

        metric=MetricIterator(storage.item_list)
        data_provider = DataProvider(dataset_name, data=metric, scenario='train')


    # extract_archive(tar_file_path, dirname, archive_format='tar')
    # dataset = load_folder_images(dataset_name, dirname)
    return data_provider


def load_examples_data(dataset_name):
    """data loader for AllanYiin deep learning course exsample data

    Args:
        dataset_name (string):
            'pokemon': pokemon images for autoencoder
            hanzi': Chinese hanzi hand writing recognition data
            'animals': Challenging animals recognition data
            'nsfw': porn detection data
            'simpsons': simpson images for gan
            'horse2zebra' : horse2zebra cyclegan training data
            'people': Supervisely human segmentation data
            autodrive':Streetview segmentation data
            'superresolution': Collections of high resolution images for superresolution model
            'anpr': Automatic number-plate recognition data
            'beauty':beauty detection data

    Returns:
        a tuple of data and label

    References
        If you want know more how to use these data to build model, please go to my github repository:
        https://github.com/AllanYiin/DeepBelief_Course5_Examples

    """
    dataset_name = dataset_name.strip().lower()
    if dataset_name.lower() not in ['pokemon', 'hanzi', 'animals', 'nsfw', 'simpsons', 'horse2zebra', 'people',
                                    'autodrive', 'superresolution', 'anpr', 'beauty','antisproofing','facelandmarks','dogs-vs-cats']:
        raise ValueError('Not a  valid  dataset_name.')
    dataset_name = 'examples_' + dataset_name
    dirname = os.path.join(_trident_dir, dataset_name)
    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError:
            # Except permission denied and potential race conditions
            # in multi-threaded environments.
            pass
    is_internet_ok = is_connected()
    if dataset_name == 'examples_pokemon':
        is_download=download_file_from_google_drive('1U-xc54fX9j9BcidvRa0ow6qjssMlSF2A', dirname, 'pokemon.tar')
        tar_file_path = os.path.join(dirname, 'pokemon.tar')
        extract_archive(tar_file_path, dirname, archive_format='tar')
        extract_path = os.path.join(dirname, 'pokemon')
        dataset = load_folder_images(dataset_name, extract_path, folder_as_label=False)
        print('get pokemon images :{0}'.format(len(dataset)))
        return dataset


    elif dataset_name == 'examples_hanzi':
        download_file_from_google_drive('13UEzSG0az113gpRPKPyKrIE2HDaA2P4H', dirname, 'hanzi.tar')
        tar_file_path = os.path.join(dirname, 'hanzi.tar')
        extract_path = os.path.join(dirname, 'hanzi')
        extract_archive(tar_file_path, dirname, archive_format='tar')
        dataset = load_folder_images(dataset_name, os.path.join(dirname, 'train'), folder_as_label=True,
                                     object_type=ObjectType.gray)

        dataset_test = load_folder_images(dataset_name, os.path.join(dirname, 'test'), folder_as_label=True,
                                          object_type=ObjectType.gray)

        dataset.testdata = dataset_test.traindata
        dataset.class_names['zh-cn'] = dataset.class_names['en-us']
        return dataset

    elif dataset_name == 'examples_animals':
        download_file_from_google_drive('19Cjq8OO6qd9k9TMZxlPjDpejDOdiHJoW', dirname, 'animals.tar')
        tar_file_path = os.path.join(dirname, 'animals.tar')
        extract_archive(tar_file_path, dirname, archive_format='tar')
        dataset = load_folder_images(dataset_name, dirname, folder_as_label=True)
        return dataset
    elif dataset_name == 'examples_nsfw':
        download_file_from_google_drive('1EXpV2QUrSFJ7zJn8NqtqFl1k6HvXsUzp', dirname, 'nsfw.tar')
        tar_file_path = os.path.join(dirname, 'nsfw.tar')
        extract_path = os.path.join(dirname, 'nsfw')
        extract_archive(tar_file_path, dirname, archive_format='tar')
        folders = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']
        data=unpickle(os.path.join(dirname,'porn_detection_data.pkl'))

        trainData = []
        testData = []
        trainLabel = []
        testLabel = []
        for n  in range(5):
            folder=folders[n]
            trainData.extend(data[folder]['train'])
            trainLabel.extend([n]*len(data[folder]['train']))
            testData.extend(data[folder]['test'])
            testLabel.extend([n] * len(data[folder]['test']))

        trainarray = ImageDataset(trainData,object_type=ObjectType.rgb)
        trainlabel = LabelDataset(trainLabel,object_type=ObjectType.classification_label)
        train_iter = Iterator(data=trainarray, label=trainlabel)

        testarray = ImageDataset(testData,object_type=ObjectType.rgb)
        testlabel = LabelDataset(testLabel,object_type=ObjectType.classification_label)
        test_iter = Iterator(data=testarray, label=testlabel)
        print('training images: {0}  test images:{1}'.format(len(trainarray), len(testarray)))

        dataset = DataProvider(dataset_name, traindata=train_iter, testdata=test_iter)
        dataset.binding_class_names(['drawing', 'hentai', 'neutral', 'porn', 'sexy'], 'en-us')
        dataset.binding_class_names(['繪畫', '色情漫畫', '中性', '色情', '性感'], 'zh-tw')
        dataset.binding_class_names(['绘画', '色情漫画', '中性', '色情', '性感'], 'zh-cn')
        dataset.scenario = 'train'
        return dataset
    elif dataset_name == 'examples_simpsons':
        download_file_from_google_drive('1hGNFbfBv3EZ4nx4Qod6PtSYzO8H4QIxC', dirname, 'simpsons.tar')
        tar_file_path = os.path.join(dirname, 'simpsons.tar')
        extract_path = os.path.join(dirname, 'simpsons')
        extract_archive(tar_file_path, extract_path, archive_format='tar')
        dataset = load_folder_images(dataset_name, extract_path, folder_as_label=False)
        dataset.traindata.label = RandomNoiseDataset(shape=(100), random_mode='normal')
        print('get simpsons images :{0}'.format(len(dataset)))
        return dataset
    elif dataset_name == 'examples_horse2zebra':
        download_file_from_google_drive('1pqj-T90Vh4wVNBV09kYZWgVPsZUA2f7U', dirname, 'horse2zebra.tar')
        tar_file_path = os.path.join(dirname, 'horse2zebra.tar')
        extract_path = os.path.join(dirname, 'horse2zebra')
        extract_archive(tar_file_path, dirname, archive_format='tar')
        trainA = ImageDataset(list_images(os.path.join(dirname, 'trainA')), object_type=ObjectType.rgb,
                              get_image_mode=GetImageMode.processed)
        trainB = ImageDataset(list_images(os.path.join(dirname, 'trainB')), object_type=ObjectType.rgb,
                              get_image_mode=GetImageMode.processed)
        testA = ImageDataset(list_images(os.path.join(dirname, 'testA')), object_type=ObjectType.rgb,
                             get_image_mode=GetImageMode.processed)
        testB = ImageDataset(list_images(os.path.join(dirname, 'testB')), object_type=ObjectType.rgb,
                             get_image_mode=GetImageMode.processed)
        train_iter = Iterator(data=trainA, unpair=trainB)
        test_iter = Iterator(data=testA, unpair=testB)
        dataset = DataProvider(dataset_name, traindata=train_iter, testdata=test_iter)
        print('get horse2zebra images :{0}'.format(len(dataset)))
        return dataset
    elif dataset_name == 'examples_people':
        download_file_from_google_drive('1H7mJJfWpmXpRxurMZQqY4N_UXWLbQ2pT', dirname, 'people.tar')
        tar_file_path = os.path.join(dirname, 'people.tar')
        extract_archive(tar_file_path, dirname, archive_format='tar')
        imgs = glob.glob(os.path.join(dirname, 'imgs', '*.*g'))
        masks = glob.glob(os.path.join(dirname, 'masks', '*.png'))
        # make_dir_if_need(os.path.join(dirname, 'trimap'))
        # for i in range(len(masks)):
        #     mask=mask2array(masks[i])
        #     trimap=mask2trimap(mask)
        #     save_mask(trimap,masks[i].replace('masks','trimap'))
        # print('trimap',len(masks))

        imgdata = ImageDataset(images=imgs, object_type=ObjectType.rgb)
        mskdata = MaskDataset(masks=masks, object_type=ObjectType.binary_mask)
        dataset = DataProvider(dataset_name=dataset_name, traindata=Iterator(data=imgdata, label=mskdata))
        print('get people images :{0}'.format(len(dataset)))
        return dataset
    elif dataset_name == 'examples_autodrive':
        download_file_from_google_drive('1JqPPeHqhWLqnI6bD8nuHcVx-Y56oIZMK', dirname, 'autodrive.tar')
        tar_file_path = os.path.join(dirname, 'autodrive.tar')
        extract_path = os.path.join(dirname, 'autodrive')
        extract_archive(tar_file_path, dirname, archive_format='tar')
        imgs = glob.glob(os.path.join(dirname, 'images', '*.*g'))
        masks = glob.glob(os.path.join(dirname, 'masks', '*.png'))

        imgdata = ImageDataset(images=imgs, object_type=ObjectType.rgb)
        mskdata = MaskDataset(masks=masks, object_type=ObjectType.color_mask)

        def parse_code(l):
            if len(l.strip().split("\t")) == 2:
                a, b = l.replace('\t\t', '\t').strip().split("\t")
                return tuple(int(i) for i in b.split(' ')), a

        label_codes, label_names = zip(
            *[parse_code(l) for l in open(os.path.join(dirname, "label_colors.txt")).readlines()])
        for i in range(len(label_codes)):
            mskdata.palette[label_names[i]] = label_codes[i]

        dataset = DataProvider(dataset_name=dataset_name, traindata=Iterator(data=imgdata, label=mskdata))
        print('get autodrive images :{0}'.format(len(dataset)))
        return dataset
    elif dataset_name == 'examples_superresolution':
        download_file_from_google_drive('1v1uoymrWI_MLSiGvSGW7tWJYSnzzXpEQ', dirname, 'superresolution.tar')
        tar_file_path = os.path.join(dirname, 'superresolution.tar')
        extract_archive(tar_file_path, dirname, archive_format='tar')
        imgs = glob.glob(os.path.join(dirname, '*.*g'))
        imgs.extend(glob.glob(os.path.join(dirname, '*.bmp')))
        print('get super resolution images :{0}'.format(len(imgs)))

        imgdata = ImageDataset(images=imgs * 2, object_type=ObjectType.rgb, symbol='lr')
        labeldata = ImageDataset(images=imgs * 2, object_type=ObjectType.rgb, symbol='hr')
        dataset = DataProvider(dataset_name=dataset_name, traindata=Iterator(data=imgdata, label=labeldata))
        return dataset
    elif dataset_name == 'examples_beauty':
        download_file_from_google_drive('1aJhxN9IqsxuayhRTm-gmxk6PiLe5wm9X', dirname, 'beauty.tar')
        tar_file_path = os.path.join(dirname, 'beauty.tar')
        extract_archive(tar_file_path, dirname, archive_format='tar')
        # 讀取圖片數據
        images_dict = {}
        with open(os.path.join(dirname, 'images_dict.pkl'), 'rb') as fp:
            images_dict = pickle.load(fp)

        f = open(os.path.join(dirname, 'All_Ratings.txt'), encoding='utf-8-sig').readlines()
        imgs = []
        landmarks = []
        ratings = []
        for row in f:
            data = row.strip().split('\t')
            if 'images\\' + data[0] in images_dict:
                img = images_dict['images\\' + data[0]][0]
                img = img.transpose([2, 0, 1])[::-1].transpose([1, 2, 0])
                imgs.append(img)
                landmark = images_dict['images\\' + data[0]][1].astype(np.float32)
                landmarks.append(landmark)
                rating = (float(data[1])) / 5.00
                ratings.append(rating)
        print('{0} faces loaded...'.format(len(imgs)))
        imgdata = ImageDataset(images=imgs, object_type=ObjectType.rgb, symbol='faces')
        landmarkdata = LandmarkDataset(landmarks=landmarks, object_type=ObjectType.landmarks, symbol='target_landmarks')
        labeldata = LabelDataset(data=ratings,object_type=ObjectType.classification_label, symbol='target_beauty')
        data_provider = DataProvider(dataset_name=dataset_name, traindata=Iterator(data=imgdata, label=Dataset.zip(landmarkdata,labeldata)))
        return data_provider

    elif dataset_name == 'examples_facelandmarks':
        download_file_from_google_drive('1GtswQBAHPa_bXaB4tW2uOOQ8Lxfz2L5B', dirname, 'ibug_300W.tar')
        tar_file_path = os.path.join(dirname, 'ibug_300W.tar')
        extract_archive(tar_file_path, dirname, archive_format='tar')
        root_dir=os.path.join(dirname, 'ibug_300W_large_face_landmark_dataset')
        image_paths = {}
        landmarks = {}
        crops = {}

        for mode in ['train','test']:
            make_dir_if_need(os.path.join(dirname, 'crops',mode))
            tree = ElementTree.parse(os.path.join(root_dir, 'labels_ibug_300W_{0}.xml'.format(mode)))
            root = tree.getroot()
            image_paths[mode]=[]
            landmarks[mode] = []
            crops[mode] = []

            offset=5
            for j in tqdm(range(len(root[2]))):
                try:
                    filename=root[2][j]
                    landmark = []
                    for num in range(68):
                        x_coordinate = int(filename[0][num].attrib['x'])
                        y_coordinate = int(filename[0][num].attrib['y'])
                        landmark.append([x_coordinate, y_coordinate])
                    landmark=np.asarray(landmark)

                    crop = filename[0].attrib
                    for k in crop.keys():
                        crop[k] = int(crop[k]) if isinstance(crop[k], str) else crop[k]
                    for k in crop.keys():
                        if k=='top' and int(landmark[:,1].min())<int(crop[k]):
                            crop[k] = int( landmark[:,1].min())
                            crop[ 'height']+=crop[k]-int(landmark[:,1].min())
                        elif k=='left' and int(landmark[:,0].min())<int(crop[k]):
                            crop[k] = int( landmark[:,0].min())
                            crop['width']+= crop[k] - int(landmark[:, 0].min())
                        elif k == 'width' and int(landmark[:, 0].max()-landmark[:, 0].min()) > int(crop[k]):
                            crop[k] = int(landmark[:, 0].max()-landmark[:, 0].min())
                        elif k == 'height' and int(landmark[:, 1].max()-landmark[:, 1].min()) > int(crop[k]):
                            crop[k] = int(landmark[:, 1].max()-landmark[:, 1].min())

                    crop['left']-=offset
                    crop['top'] -= offset
                    crop['width'] += 2*offset
                    crop['height'] += 2*offset


                    landmark[:,0]-=crop['left']
                    landmark[:, 1] -= crop['top']


                    if not os.path.exists(os.path.join(dirname, 'crops', mode, '{0}.png'.format(j))):
                        im=image2array(os.path.join(root_dir, filename.attrib['file']))
                        if im.ndim==2:
                            im=cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)
                        im=im[crop['top']:min(crop['top']+crop['height'],im.shape[0]),crop['left']:min(crop['left']+crop['width'],im.shape[1]),:]

                        if max(im.shape[:2])/max(min(im.shape[:2]),0)<=5:

                            array2image(im).save(os.path.join(dirname, 'crops',mode,'{0}.png'.format(j)))
                            image_paths[mode].append(os.path.join(dirname, 'crops', mode, '{0}.png'.format(j)))
                            crops[mode].append(crop)
                            landmarks[mode].append(landmark)
                        del im
                    else:
                        #im = image2array(os.path.join(dirname, 'crops',mode,'{0}.png'.format(j)))
                        image_paths[mode].append(os.path.join(dirname, 'crops',mode,'{0}.png'.format(j)))
                        crops[mode].append(crop)
                        landmarks[mode].append(landmark)

                    if j%100==0:
                        gc.collect()
                except Exception as e:
                   pass

        print('ibug 300w train dataset: images: {0} landmarks:{1} \n'.format(len(image_paths['train']),len(landmarks['train'])))
        print('ibug 300w test dataset: images: {0} landmarks:{1} \n'.format(len(image_paths['test']), len(landmarks['test'])))
        imdata=ImageDataset(images=image_paths['train'],symbol='faces',object_type=ObjectType.rgb)
        landmarkdata = LandmarkDataset(landmarks=landmarks['train'], symbol='landmarks',object_type=ObjectType.landmarks)
        imtestdata = ImageDataset(images=image_paths['test'], symbol='faces',object_type=ObjectType.rgb)
        landmarktestdata = LandmarkDataset(landmarks=landmarks['test'], symbol='landmarks',object_type=ObjectType.landmarks)
        data_provider=DataProvider(traindata=Iterator(data=imdata,label=landmarkdata),testdata=Iterator(data=imtestdata,label=landmarktestdata))
        return data_provider

    elif dataset_name == 'examples_antisproofing':
        download_file_from_google_drive('1e7Zjn2MHNCvA5gXdJUECzY8NjK4KVpa7', dirname, 'antisproofing.tar')
        tar_file_path = os.path.join(dirname, 'antisproofing.tar')
        make_dir_if_need(os.path.join(dirname, 'antisproofing'))
        extract_archive(tar_file_path, dirname, archive_format='tar')
        data_provider = load_folder_images(dataset_name,os.path.join(dirname, 'antisproofing'))
        return data_provider
    elif dataset_name == 'examples_anpr':
        download_file_from_google_drive('1uGBd8tXlP0TZAXNgrR6H0jl5MXj7VPbN', dirname, 'anpr.tar')
        tar_file_path = os.path.join(dirname, 'anpr.tar')
        extract_archive(tar_file_path, dirname, archive_format='tar')
        imgs = glob.glob(os.path.join(dirname, '*.*g'))
        # CCPD (Chinese City Parking Dataset, ECCV) and PDRC (license Plate Detection and Recognition Challenge)
        # https://github.com/detectRecog/CCPD
        provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
                     "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
        alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                     'W', 'X', 'Y', 'Z', 'O']
        ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
               'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

        def lp2char(lp):
            cols = lp.split('_')
            charstring = ''
            for i in range(len(cols)):
                if i == 0:
                    charstring += provinces[int(cols[i])]
                elif i == 1:
                    charstring += alphabets[int(cols[i])]
                else:
                    charstring += ads[int(cols[i])]
            return charstring

        width = 720
        height = 1160
        for im_path in imgs:
            lbl = im_path.split('/')[-1].rsplit('.', 1)[0].split('-')[-3]
            charstring = lp2char(lbl)
            iname = im_path.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('-')
            [leftUp, rightDown] = [[int(eel) for eel in el.split('&')] for el in iname[2].split('_')]
            box = [leftUp[0], leftUp[1], rightDown[0], rightDown[1]]
            ori_w, ori_h = [float(int(el)) for el in [width, height]]
            new_labels = [(leftUp[0] + rightDown[0]) / (2 * ori_w), (leftUp[1] + rightDown[1]) / (2 * ori_h),
                          (rightDown[0] - leftUp[0]) / ori_w, (rightDown[1] - leftUp[1]) / ori_h]
            download_file_from_google_drive('1e7Zjn2MHNCvA5gXdJUECzY8NjK4KVpa7', dirname, 'antisproofing.tar')
            tar_file_path = os.path.join(dirname, 'antisproofing.tar')
            make_dir_if_need(os.path.join(dirname, 'antisproofing'))
            extract_archive(tar_file_path, dirname, archive_format='tar')
            data_provider = load_folder_images(dataset_name, os.path.join(dirname, 'antisproofing'))
            return data_provider



    elif dataset_name == 'examples_dogs-vs-cats':
        download_file_from_google_drive('10czW0On7eIXkPP-MuQ-IRxMWdTizWjNC', dirname, 'dogs-vs-cats.tar')
        tar_file_path = os.path.join(dirname, 'dogs-vs-cats.tar')
        extract_archive(tar_file_path, dirname, archive_format='tar')
        data_provider = load_folder_images(dataset_name, dirname)
        return data_provider


    else:
        return None