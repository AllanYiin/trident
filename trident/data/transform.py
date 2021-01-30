from abc import ABC, abstractmethod
from collections import Iterable
from typing import Sequence, Tuple, Dict, Union, Optional
import collections
import  numpy as np
import cv2
from trident.data.image_common import object_type_inference

from trident.backend.pytorch_ops import tensor_to_shape

from trident.backend.common import OrderedDict
from trident.backend.common import *
from trident.backend.tensorspec import TensorSpec
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

    def apply_batch(self, inputs: Sequence[Tuple],spec:Optional[TensorSpec]=None):
        if spec  is None:
            spec=TensorSpec(shape=tensor_to_shape(inputs[0]), object_type=object_type_inference(inputs[0]))
        return tuple(self.apply(input,spec) for input in inputs)

    @abstractmethod
    def apply(self, input: Tuple,spec:TensorSpec):
        pass

    def  __call__(self, inputs: Union[Dict[TensorSpec,np.ndarray],np.ndarray]):
        if isinstance(inputs,Dict):
            results=OrderedDict()
            for spec, data in inputs:
                results[spec]=self.apply_batch(data,spec)





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


    def apply_batch(self, inputs: Sequence[Tuple],spec:Optional[TensorSpec]=None):
        r"""Apply transform on batch input data."""
        if not isinstance(inputs,OrderedDict) :
            if spec is None:
                spec = TensorSpec(shape=tensor_to_shape(inputs), object_type=object_type_inference(inputs[0]))
            return self.apply(inputs, spec)
        else:
            results=OrderedDict()
            for spec, data in inputs.items():
                if isinstance(data,Iterable):
                    results[spec] = [self.apply(d, spec) for d in data]
                else:
                    results[spec]=self.apply(data, spec)
            return results



    def apply(self, input: Tuple,spec:TensorSpec):
        r"""Apply transform on single input data."""

        apply_func = self._get_apply(spec.object_type.value)
        if apply_func is None:
            return input
        else:
            return apply_func(input,spec)


    def _get_apply(self, key):
        if 'image' in key or  'rgb' in key :
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



    def _apply_image(self, image,spec:TensorSpec):
        raise NotImplementedError

    def _apply_coords(self, coords,spec:TensorSpec):
        raise NotImplementedError

    def _apply_boxes(self, boxes,spec:TensorSpec):
        eh, ew = self.output_size
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

    def  __call__(self, inputs: Union[Dict[TensorSpec,np.ndarray],np.ndarray]):
        if isinstance(inputs,Dict):
            results=self.apply_batch(inputs)
            return results



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

    def __init__(self, order=None):
        super().__init__()
        if order is None:
            order = ("image",)
        elif not isinstance(order, collections.abc.Sequence):
            raise ValueError(
                "order should be a sequence, but got order={}".format(order)
            )
        for k in order:
            if k in ("batch",):
                raise ValueError("{} is invalid data type".format(k))
            elif k.endswith("category") or k.endswith("info"):
                # when the key is *category or info, we should do nothing
                # if the corresponding apply methods are not implemented.
                continue
            elif self._get_apply(k) is None:
                raise NotImplementedError("{} is unsupported data type".format(k))
        self.order = order

    def apply_batch(self, inputs: Sequence[Tuple],spec:Optional[TensorSpec]=None):
        r"""Apply transform on batch input data."""
        if spec is None:
            spec = TensorSpec(shape=tensor_to_shape(inputs[0]), object_type=object_type_inference(inputs[0]))
        return [self.apply(input, spec) for input in inputs]


    def apply(self, input: Tuple,spec:TensorSpec):
        r"""Apply transform on single input data."""
        if not isinstance(input, tuple):
            input = (input,)

        output = []
        for i in range(min(len(input), len(self.order))):
            apply_func = self._get_apply(self.order[i])
            if apply_func is None:
                output.append(input[i])
            else:
                output.append(apply_func(input[i]))
        if len(input) > len(self.order):
            output.extend(input[len(self.order):])

        if len(output) == 1:
            output = output[0]
        else:
            output = tuple(output)
        return output

    def _get_apply(self, key):
        return getattr(self, "_apply_{}".format(key), None)

    def _get_corpus(self, input: Tuple):
        if not isinstance(input, tuple):
            input = (input,)
        return input[self.order.index("image")]

    def _apply_corpus(self, image):
        raise NotImplementedError

    def _apply_sequence(self, coords):
        raise NotImplementedError

    def _apply_sequence_labels(self, labels):
        raise NotImplementedError

    def _apply_sequence_mask(self, mask):
        raise NotImplementedError


    def _apply_labels(self, labels):
        raise NotImplementedError




