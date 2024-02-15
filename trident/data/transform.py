import numbers
import random
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Sequence, Tuple, Dict, Union, Optional, Callable, Any
import collections

import matplotlib.pyplot as plt
import  numpy as np
import cv2


from trident.backend.common import OrderedDict
from trident.backend.common import *
from trident.backend.tensorspec import TensorSpec, object_type_inference, ObjectType

if get_backend() == 'pytorch':
    from trident.backend.pytorch_ops import *
elif get_backend() == 'tensorflow':
    from trident.backend.tensorflow_ops import *


__all__ = ['Transform', 'VisionTransform', 'TextTransform','Compose','OneOf']

class Transform(ABC):
    """
    Rewrite apply method in subclass.
    """
    def __init__(self, name=None):
        self.name=name
        self.is_spatial=False

    def apply_batch(self, inputs: Sequence[Tuple],spec:Optional[TensorSpec]=None):
        if spec  is None and self.is_spatial==True:
            spec=TensorSpec(shape=tensor_to_shape(inputs[0]), object_type=object_type_inference(inputs[0]))
        return tuple(self.apply(input,spec) for input in inputs)

    @abstractmethod
    def apply(self, input: Tuple,spec:TensorSpec):
        pass

    def  __call__(self, inputs: Union[Dict[TensorSpec,np.ndarray],np.ndarray],**kwargs):
        pass





    def __repr__(self):
        return self.__class__.__name__




class VisionTransform(Transform):
    r"""
    Base class of all transforms used in computer vision.
    Calling logic: apply_batch() -> apply() -> _apply_image() and other _apply_*()
    method. If you want to implement a self-defined transform method for image,
    rewrite _apply_image method in subclass.
    :param order: input type order. Input is a tuple containing different structures,
        order is used to specify the order of structures. For example, if your input
        is (image, boxes) type, then the ``order`` should be ("image", "boxes").
        Current available strings and data type are describe below:
        * "image": input image, with shape of `(H, W, C)`.
        * "coords": coordinates, with shape of `(N, 2)`.
        * "boxes": bounding boxes, with shape of `(N, 4)`, "xyxy" format,
          the 1st "xy" represents top left point of a box,
          the 2nd "xy" represents right bottom point.
        * "mask": map used for segmentation, with shape of `(H, W, 1)`.
        * "keypoints": keypoints with shape of `(N, K, 3)`, N for number of instances,
          and K for number of keypoints in one instance. The first two dimensions
          of last axis is coordinate of keypoints and the the 3rd dimension is
          the label of keypoints.
        * "polygons": a sequence containing numpy arrays, its length is the number of instances.
          Each numpy array represents polygon coordinate of one instance.
        * "category": categories for some data type. For example, "image_category"
          means category of the input image and "boxes_category" means categories of
          bounding boxes.
        * "info": information for images such as image shapes and image path.
        You can also customize your data types only if you implement the corresponding
        _apply_*() methods, otherwise ``NotImplementedError`` will be raised.
    """

    def __init__(self, keep_prob=0,name=None):
        super().__init__(name=name)
        self.keep_prob=keep_prob
        self._shape_info = None


    def apply_batch(self, inputs: Sequence[Tuple],spec:Optional[TensorSpec]=None):
        r"""Apply transform on batch input data."""
        if not isinstance(inputs,OrderedDict) :
            if self.is_spatial==True:
                self._shape_info =None
            if spec is None :
                spec = TensorSpec(shape=tensor_to_shape(inputs,need_exclude_batch_axis=True,is_singleton=True), object_type=ObjectType.rgb)
            return self.apply(inputs, spec)
        else:
            results=OrderedDict()
            sampledata= list(inputs.values())[0]
            spec=inputs.key_list[0]
            if (isinstance(sampledata, Iterable) and not isinstance(sampledata, np.ndarray)) or (is_tensor_like(sampledata) and spec.ndim == sampledata.ndim):
                for i in range(len(sampledata)):
                    self._shape_info = None
                    for spec, data in inputs.items():
                        if spec not in results:
                            results[spec] = []
                        results[spec].append(self.apply(data[i], spec))
            else:
                self._shape_info = None
                for spec, data in inputs.items():
                    results[spec]=self.apply(data, spec)
            return results



    def apply(self, input: Tuple,spec:TensorSpec):
        r"""Apply transform on single input data."""
        if spec is None:
            return self._apply_image(input,None)
        apply_func = self._get_apply(spec.object_type.value)
        if apply_func is None:
            return input
        else:

            img_data=apply_func(input,spec)
            if apply_func.__qualname__ == '_apply_image':
                img_data=self.check_pixel_range(img_data)
            return img_data


    def _get_apply(self, key):
        if ('image' in key or  'rgb' in key or  'gray' in key) and key!='image_path' :
            return getattr(self, "_apply_{}".format('image'), None)
        elif 'bbox' in key:
            return getattr(self, "_apply_{}".format('boxes'), None)
        elif 'mask' in key:
            return getattr(self, "_apply_{}".format('mask'), None)
        elif 'depth' in key:
            return getattr(self, "_apply_{}".format('mask'), None)
        elif 'densepose' in key:
            return getattr(self, "_apply_{}".format('image'), None)
        elif 'keypoint' in key or  'landmark' in key:
            return getattr(self, "_apply_{}".format('keypoints'), None)
        elif 'polygon' in key :
            return getattr(self, "_apply_{}".format('polygons'), None)
        elif 'label' in key:
            return getattr(self, "_apply_{}".format('labels'), None)
        return None

    def check_pixel_range(self,image):
        max_value=image.copy().max()
        min_value=image.copy().min()
        if max_value>255 or min_value<0:
            raise ValueError('{0} over bundary max:{1} :{2}'.format(self.__class__.__name__,max_value,min_value))
        elif max_value-min_value<1:
            raise  ValueError('{0} almost monotone max:{1} :{2}'.format(self.__class__.__name__,max_value,min_value))
        elif np.greater(image.copy(),127.5).astype(np.float32).mean()>0.95:
            raise  ValueError('{0} almost white max:{1} :{2}'.format(self.__class__.__name__,max_value,min_value))
        elif np.less(image.copy(),127.5).astype(np.float32).mean()>0.95:
            raise  ValueError('{0} almost black max:{1} :{2}'.format(self.__class__.__name__,max_value,min_value))
        return image


    def _apply_image(self, image,spec:TensorSpec):
        return image

    def _apply_coords(self, coords,spec:TensorSpec):
        return coords

    def _apply_boxes(self, boxes,spec:TensorSpec):
        if isinstance( self.output_size,numbers.Number):
            self.output_size=( self.output_size, self.output_size)
        eh, ew = self.output_size
        if ndim(boxes)==0:
            return boxes
        else:
            if ndim(boxes) == 1:
                boxes=np.expand_dims(boxes,0)
            B=boxes.shape[0]
            location= boxes[:, :4]
            class_info = boxes[:, 4:5] if boxes.shape[-1]>4 else None
            keypoints = boxes[:, 5:] if boxes.shape[-1]>5 else None
            idxs = np.array([(0, 1), (2, 1), (0, 3), (2, 3)]).flatten()
            location = np.asarray(location).reshape(-1, 4)[:, idxs].reshape(-1, 2)
            location = self._apply_coords(location,spec).reshape((-1, 4,2))
            x_ = location[:,:,0]
            y_ = location[:,:,1]
            xmin = np.min(x_, 1).reshape(-1, 1)
            ymin = np.min(y_, 1).reshape(-1, 1)
            xmax = np.max(x_, 1).reshape(-1, 1)
            ymax = np.max(y_, 1).reshape(-1, 1)


            if keypoints is not None:
                coords_keypoints = np.asarray(keypoints).reshape(-1, 2)
                keypoints = self._apply_keypoints(coords_keypoints, spec).reshape((-1, keypoints.shape[-1]))


            trans_boxes = np.concatenate((xmin, ymin,xmax,ymax), axis=1)
            trans_boxes[:, 0::2] =trans_boxes[:, 0::2]# , 0, ew)
            trans_boxes[:, 1::2] = trans_boxes[:, 1::2]#,0, eh)
            if class_info is not None  and class_info.shape[-1]>0 and keypoints is not None and len(keypoints)>0:
                trans_boxes = np.concatenate((trans_boxes, class_info,keypoints), axis=1)
            elif class_info is not None  and class_info.shape[-1]>0:
                trans_boxes = np.concatenate((trans_boxes, class_info), axis=1)
            return trans_boxes

    def _apply_mask(self, mask,spec:TensorSpec):
        return mask

    def _apply_keypoints(self, keypoints,spec:TensorSpec):
        coords, visibility = keypoints[..., :2], keypoints[..., 2:]
        #trans_coords = [self._apply_coords(p,spec) for p in coords]
        trans_coords = self._apply_coords(coords, spec)
        return np.concatenate((trans_coords, visibility), axis=-1)

    def _apply_polygons(self, polygons,spec:TensorSpec):
        return [[self._apply_coords(p,spec) for p in instance] for instance in polygons]

    def _apply_labels(self, labels,spec:TensorSpec):
        return  labels

    def  __call__(self, inputs: Union[Dict[TensorSpec,np.ndarray],np.ndarray],**kwargs):
        spec=kwargs.get('spec')
        return self.apply_batch(inputs,spec)




class TextTransform(Transform):
    r"""
    Base class of all transforms used in computer vision.
    Calling logic: apply_batch() -> apply() -> _apply_image() and other _apply_*()
    method. If you want to implement a self-defined transform method for image,
    rewrite _apply_image method in subclass.
    :param order: input type order. Input is a tuple containing different structures,
        order is used to specify the order of structures. For example, if your input
        is (image, boxes) type, then the ``order`` should be ("image", "boxes").
        Current available strings and data type are describe below:
        * "image": input image, with shape of `(H, W, C)`.

        * "mask": map used for segmentation, with shape of `(H, W, 1)`.

        * "category": categories for some data type. For example, "image_category"
          means category of the input image and "boxes_category" means categories of
          bounding boxes.
        * "info": information for images such as image shapes and image path.
        You can also customize your data types only if you implement the corresponding
        _apply_*() methods, otherwise ``NotImplementedError`` will be raised.
    """

    def __init__(self, name=None):
        super().__init__(name=name)
        self._text_info = None

    def _precalculate(self, textdata, **kwargs):
        pass



    def apply_batch(self, inputs: Sequence[Tuple],spec:Optional[TensorSpec]=None):
        r"""Apply transform on batch input data."""
        if not isinstance(inputs,OrderedDict) :
            self._text_info = None
            if spec is None and self.is_spatial==True:
                spec = TensorSpec(shape=tensor_to_shape(inputs,need_exclude_batch_axis=True,is_singleton=True), object_type=ObjectType.corpus)
            self._precalculate(inputs)
            return self.apply(inputs, spec)
        else:
            results=OrderedDict()
            self._text_info = None
            is_precalculate=False
            for k,v in inputs.items():
                if k.object_type is None:
                    k.object_type=object_type_inference(v)
                if isinstance(k,TensorSpec) and k.object_type==ObjectType.corpus:
                    self._precalculate(v)
                    is_precalculate=True
            if not is_precalculate:
                self._precalculate(inputs.value_list[0])
            for spec, data in inputs.items():
                results[spec]=self.apply(data, spec)
            return results



    def apply(self, input: Tuple,spec:TensorSpec):
        r"""Apply transform on single input data."""
        if spec is None:
            return self._apply_corpus(input,None)
        apply_func = self._get_apply(spec.object_type.value)
        if apply_func is None:
            return input
        else:
            return apply_func(input,spec)

    def _get_apply(self, key):
        if key is None or 'corpus' in key  :
            return getattr(self, "_apply_{}".format('corpus'), None)
        elif 'sequence_label' in key:
            return getattr(self, "_apply_{}".format('sequence_label'), None)
        elif 'sequence_mask' in key:
            return getattr(self, "_apply_{}".format('sequence_mask'), None)
        return None

    def _apply_corpus(self, corpus,spec:TensorSpec):
        return corpus

    def _apply_sequence(self, sequence,spec:TensorSpec):
        return sequence

    def _apply_sequence_labels(self, labels,spec:TensorSpec):
        return  labels

    def _apply_sequence_mask(self, mask,spec:TensorSpec):
        return  mask

    def  __call__(self, inputs: Union[Dict[TensorSpec,np.ndarray],np.ndarray],**kwargs):
        spec=kwargs.get('spec')
        return self.apply_batch(inputs,spec)


class Compose(Transform):
    r"""Composes several transforms together.
    Args:
        transforms: list of :class:`VisionTransform` to compose.
        batch_compose: whether use shuffle_indices for batch data or not.
            If True, use original input sequence.
            Otherwise, the shuffle_indices will be used for transforms.
        shuffle_indices: indices used for random shuffle, start at 1.
            For example, if shuffle_indices is [(1, 3), (2, 4)], then the 1st and 3rd transform
            will be random shuffled, the 2nd and 4th transform will also be shuffled.
        order: the same with :class:`VisionTransform`

    Examples:
        .. testcode::

           from megengine.data.transform import RandomHorizontalFlip, RandomVerticalFlip, CenterCrop, ToMode, Compose

           transform_func = Compose([
           RandomHorizontalFlip(),
           RandomVerticalFlip(),
           CenterCrop(100),
           ToMode("CHW"),
           ],
           shuffle_indices=[(1, 2, 3)]
           )
    """

    def __init__(
            self, transforms=[], batch_compose=False, shuffle_indices=None
    ):
        super().__init__()
        self.transforms = transforms
        #self._set_order()

        if batch_compose and shuffle_indices is not None:
            raise ValueError(
                "Do not support shuffle when apply transforms along the whole batch"
            )
        self.batch_compose = batch_compose

        if shuffle_indices is not None:
            shuffle_indices = [tuple(x - 1 for x in idx) for idx in shuffle_indices]
        self.shuffle_indices = shuffle_indices

    # def _set_order(self):
    #     for t in self.transforms:
    #         t.order = self.order
    #         if isinstance(t, Compose):
    #             t._set_order()

    def apply_batch(self,inputs: Sequence[Tuple],spec:Optional[TensorSpec]=None):
        if not isinstance(inputs, OrderedDict):
            if self.is_spatial == True:
                self._shape_info = None
            if spec is None:
                spec = TensorSpec(shape=tensor_to_shape(inputs, need_exclude_batch_axis=True, is_singleton=True), object_type=ObjectType.rgb)
            return self.apply(inputs, spec)
        else:
            results = OrderedDict()
            sampledata = list(inputs.values())[0]
            spec = inputs.key_list[0]
            if (isinstance(sampledata, Iterable) and not isinstance(sampledata, np.ndarray)) or (is_tensor_like(sampledata) and spec.ndim == sampledata.ndim):
                for i in range(len(sampledata)):
                    self._shape_info = None
                    for spec, data in inputs.items():
                        if spec not in results:
                            results[spec] = []
                        results[spec].append(self.apply(data[i], spec))
            else:
                if hasattr(self,'_shape_info'):
                    self._shape_info = None
                if hasattr(self, '_text_info'):
                    self._text_info = None
                for spec, data in inputs.items():
                    results[spec] = self.apply(data, spec)
            return results

    def apply(self, input: Tuple,spec:TensorSpec):
        for t in self._shuffle():
            input = t.apply(input,spec)
        return input

    def _shuffle(self):
        if self.shuffle_indices is not None:
            source_idx = list(range(len(self.transforms)))
            for idx in self.shuffle_indices:
                shuffled = np.random.permutation(idx).tolist()
                for src, dst in zip(idx, shuffled):
                    source_idx[src] = dst
            return [self.transforms[i] for i in source_idx]
        else:
            return self.transforms
    def  __call__(self, inputs: Union[Dict[TensorSpec,np.ndarray],np.ndarray],**kwargs):
        spec=kwargs.get('spec')
        return self.apply_batch(inputs,spec)


class OneOf(Transform):
    r"""Composes several transforms together.
    Args:
        transforms: list of :class:`VisionTransform` to compose.
        batch_compose: whether use shuffle_indices for batch data or not.
            If True, use original input sequence.
            Otherwise, the shuffle_indices will be used for transforms.
        shuffle_indices: indices used for random shuffle, start at 1.
            For example, if shuffle_indices is [(1, 3), (2, 4)], then the 1st and 3rd transform
            will be random shuffled, the 2nd and 4th transform will also be shuffled.
        order: the same with :class:`VisionTransform`

    Examples:
        .. testcode::

           from megengine.data.transform import RandomHorizontalFlip, RandomVerticalFlip, CenterCrop, ToMode, Compose

           transform_func = Compose([
           RandomHorizontalFlip(),
           RandomVerticalFlip(),
           CenterCrop(100),
           ToMode("CHW"),
           ],
           shuffle_indices=[(1, 2, 3)]
           )
    """

    def __init__(
            self, transforms=[], batch_compose=False
    ):
        super().__init__()
        self.transforms = transforms
        self.shuffle_indices=list(range(len(self.transforms)))
        self.random_index=random.choice( self.shuffle_indices)

        self.batch_compose = batch_compose


    def apply_batch(self,inputs: Sequence[Tuple],spec:Optional[TensorSpec]=None):
        if not isinstance(inputs, OrderedDict):
            if self.is_spatial == True:
                self._shape_info = None
            if spec is None:
                spec = TensorSpec(shape=tensor_to_shape(inputs, need_exclude_batch_axis=True, is_singleton=True), object_type=ObjectType.rgb)
            return self.apply(inputs, spec)
        else:
            results = OrderedDict()
            sampledata = list(inputs.values())[0]
            spec = inputs.key_list[0]
            if (isinstance(sampledata, Iterable) and not isinstance(sampledata, np.ndarray)) or (is_tensor_like(sampledata) and spec.ndim == sampledata.ndim):
                for i in range(len(sampledata)):
                    self.random_index=random.choice(range(len(self.transforms)))

                    if hasattr(self.transforms[self.random_index], '_shape_info'):
                        self.transforms[self.random_index]._shape_info = None
                    if hasattr(self.transforms[self.random_index], '_text_info'):
                        self.transforms[self.random_index]._text_info = None
                    for spec, data in inputs.items():
                        if spec not in results:
                            results[spec] = []
                        results[spec].append(self.apply(data[i], spec))
            else:
                self.random_index = random.choice(range(len(self.transforms)))
                if hasattr(self.transforms[self.random_index], '_shape_info'):
                    self.transforms[self.random_index]._shape_info = None
                if hasattr(self.transforms[self.random_index], '_text_info'):
                    self.transforms[self.random_index]._text_info = None
                for spec, data in inputs.items():
                    results[spec] = self.apply(data, spec)
            return results

    def apply(self, input: Tuple,spec:TensorSpec):
        t=self.transforms[self.random_index]
        input = t.apply(input,spec)
        return input

    def  __call__(self, inputs: Union[Dict[TensorSpec,np.ndarray],np.ndarray],**kwargs):
        spec=kwargs.get('spec')
        return self.apply_batch(inputs,spec)





