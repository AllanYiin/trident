import numbers
from abc import ABC, abstractmethod
from collections import Iterable
from typing import Sequence, Tuple, Dict, Union, Optional, Callable, Any
import collections
import  numpy as np
import cv2


from trident.backend.common import OrderedDict
from trident.backend.common import *
from trident.backend.tensorspec import TensorSpec, object_type_inference, ObjectType

if get_backend() == 'pytorch':
    from trident.backend.pytorch_ops import *
elif get_backend() == 'tensorflow':
    from trident.backend.tensorflow_ops import *


__all__ = ['Transform', 'VisionTransform', 'TextTransform']

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

    def __init__(self, name=None):
        super().__init__(name=name)
        self.output_size=None
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
        if 'image' in key or  'rgb' in key or  'gray' in key :
            return getattr(self, "_apply_{}".format('image'), None)
        elif 'bbox' in key:
            return getattr(self, "_apply_{}".format('boxes'), None)
        elif 'mask' in key:
            return getattr(self, "_apply_{}".format('mask'), None)

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
        raise NotImplementedError

    def _apply_coords(self, coords,spec:TensorSpec):
        raise NotImplementedError

    def _apply_boxes(self, boxes,spec:TensorSpec):
        if isinstance( self.output_size,numbers.Number):
            self.output_size=( self.output_size, self.output_size)
        eh, ew = self.output_size
        if ndim(boxes)==0:
            return boxes
        else:
            if ndim(boxes) == 1:
                boxes=np.expand_dims(boxes,0)
            class_info = boxes[:, 4:]
            boxes= boxes[:, :4]
            idxs = np.array([(0, 1), (2, 1), (0, 3), (2, 3)]).flatten()
            coords = np.asarray(boxes).reshape(-1, 4)[:, idxs].reshape(-1, 2)
            coords = self._apply_coords(coords,spec).reshape((-1, 4, 2))
            minxy = coords.min(axis=1)
            maxxy = coords.max(axis=1)
            trans_boxes = np.concatenate((minxy, maxxy), axis=1)
            trans_boxes[:, 0::2] =clip(trans_boxes[:, 0::2] , 0, ew)
            trans_boxes[:, 1::2] = clip(trans_boxes[:, 1::2],0, eh)
            if class_info.shape[-1]>0:
                trans_boxes = np.concatenate((trans_boxes, class_info), axis=1)
            return trans_boxes

    def _apply_mask(self, mask,spec:TensorSpec):
        raise NotImplementedError

    def _apply_keypoints(self, keypoints,spec:TensorSpec):
        coords, visibility = keypoints[..., :2], keypoints[..., 2:]
        #trans_coords = [self._apply_coords(p,spec) for p in coords]
        trans_coords = self._apply_coords(coords, spec)
        return np.concatenate((trans_coords, visibility), axis=-1)

    def _apply_polygons(self, polygons,spec:TensorSpec):
        return [[self._apply_coords(p,spec) for p in instance] for instance in polygons]

    def _apply_labels(self, labels,spec:TensorSpec):
        raise NotImplementedError

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
        raise NotImplementedError

    def _apply_sequence(self, sequence,spec:TensorSpec):
        raise NotImplementedError

    def _apply_sequence_labels(self, labels,spec:TensorSpec):
        raise NotImplementedError

    def _apply_sequence_mask(self, mask,spec:TensorSpec):
        raise NotImplementedError

    def  __call__(self, inputs: Union[Dict[TensorSpec,np.ndarray],np.ndarray],**kwargs):
        spec=kwargs.get('spec')
        return self.apply_batch(inputs,spec)





