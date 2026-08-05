from __future__ import absolute_import, division, print_function

import hashlib

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - optional at import time
    cv2 = None

from .errors import DataPipelineError
from .schema import infer_field_kind


class TransformContext(object):
    """Per-sample state. Transform instances remain immutable and worker-safe."""

    def __init__(self, seed=0, epoch=0, rank=0, worker_id=0, sample_id=None):
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.rank = int(rank)
        self.worker_id = int(worker_id)
        self.sample_id = sample_id
        self.records = []
        self._random_calls = {}

    def random_state(self, transform_name):
        # Worker id is deliberately excluded: changing worker count must not
        # change augmentation for the same sample/epoch/rank. The occurrence
        # avoids identical draws when a Compose contains the same class twice.
        occurrence = self._random_calls.get(transform_name, 0)
        self._random_calls[transform_name] = occurrence + 1
        value = "{0}:{1}:{2}:{3}:{4}:{5}".format(
            self.seed, self.epoch, self.rank, self.sample_id, transform_name,
            occurrence,
        ).encode("utf-8")
        digest = hashlib.sha256(value).digest()
        derived_seed = int.from_bytes(digest[:4], byteorder="little")
        return np.random.RandomState(derived_seed)


class TransformRecord(object):
    """Serializable parameters sampled by one transform invocation."""

    def __init__(self, name, params, version=1):
        self.name = name
        self.params = dict(params or {})
        self.version = int(version)

    def to_dict(self):
        def serializable(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, dict):
                return {key: serializable(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [serializable(item) for item in value]
            return value
        return {"name": self.name, "version": self.version,
                "params": serializable(self.params)}

    @classmethod
    def from_dict(cls, value):
        return cls(value["name"], value.get("params"), value.get("version", 1))

    def __repr__(self):
        return "TransformRecord({0!r}, {1!r})".format(self.name, self.params)


class SampleTransform(object):
    def __call__(self, sample, context=None, schema=None):
        raise NotImplementedError


class Compose(SampleTransform):
    def __init__(self, transforms):
        self.transforms = tuple(transforms or ())

    def __call__(self, sample, context=None, schema=None):
        for transform in self.transforms:
            sample = transform(sample, context=context, schema=schema)
        return sample


def _field_kind(name, schema):
    spec = schema.get(name) if schema is not None else None
    return spec.kind if spec is not None else infer_field_kind(name)


def _field_spec(name, schema):
    return schema.get(name) if schema is not None else None


def _image_shape(image, spec=None):
    if image is None or not hasattr(image, "shape"):
        raise ValueError("geometry transform requires an array-like image")
    layout = getattr(spec, "layout", None)
    if image.ndim == 2:
        return image.shape[0], image.shape[1]
    if layout in ("CHW", "NCHW"):
        return image.shape[-2], image.shape[-1]
    return image.shape[0], image.shape[1]


def _to_hwc(array, spec=None):
    layout = getattr(spec, "layout", None)
    if array.ndim == 3 and layout == "CHW":
        return np.transpose(array, (1, 2, 0)), True
    return array, False


def _restore_layout(array, transposed):
    if transposed and array.ndim == 3:
        return np.transpose(array, (2, 0, 1))
    return array


def _resize_array(array, output_size, interpolation, spec=None):
    if cv2 is None:
        raise ImportError("OpenCV is required by pipeline geometry transforms")
    source, transposed = _to_hwc(np.asarray(array), spec)
    height, width = output_size
    resized = cv2.resize(source, (int(width), int(height)), interpolation=interpolation)
    if source.ndim == 3 and resized.ndim == 2:
        resized = np.expand_dims(resized, -1)
    return _restore_layout(resized, transposed)


def _flip_array(array, spec=None):
    layout = getattr(spec, "layout", None)
    axis = -1 if layout == "CHW" and array.ndim == 3 else 1
    return np.ascontiguousarray(np.flip(array, axis=axis))


def _crop_and_pad(array, params, fill_value=0, spec=None):
    source, transposed = _to_hwc(np.asarray(array), spec)
    x = params["x"]
    y = params["y"]
    crop_h = params["crop_h"]
    crop_w = params["crop_w"]
    out_h = params["height"]
    out_w = params["width"]
    cropped = source[y:y + crop_h, x:x + crop_w]
    output_shape = (out_h, out_w) + tuple(source.shape[2:])
    output = np.empty(output_shape, dtype=source.dtype)
    output[...] = fill_value
    output[:cropped.shape[0], :cropped.shape[1]] = cropped
    return _restore_layout(output, transposed)


class GeometryTransform(SampleTransform):
    """Samples geometry once, then applies it to every compatible field."""

    def sample_params(self, image, image_spec, rng):
        raise NotImplementedError

    def apply_kind(self, value, kind, params, spec):
        method = getattr(self, "apply_{0}".format(kind), None)
        return method(value, params, spec) if callable(method) else value

    def _apply(self, sample, params, schema=None):
        output = sample.copy() if hasattr(sample, "copy") else dict(sample)
        for name, value in sample.items():
            if value is None:
                continue
            kind = _field_kind(name, schema)
            output[name] = self.apply_kind(value, kind, params,
                                           _field_spec(name, schema))
        return output

    def replay(self, sample, record, schema=None):
        """Apply a previously sampled record without consuming random state."""
        if isinstance(record, dict):
            record = TransformRecord.from_dict(record)
        if record.name != self.__class__.__name__:
            raise ValueError("record belongs to {0}, not {1}".format(
                record.name, self.__class__.__name__))
        return self._apply(sample, record.params, schema=schema)

    def __call__(self, sample, context=None, schema=None):
        if not isinstance(sample, dict):
            raise TypeError("geometry transforms require mapping samples")
        context = context or TransformContext()
        image_name = None
        for name in sample:
            if _field_kind(name, schema) == "image":
                image_name = name
                break
        if image_name is None:
            return sample
        rng = context.random_state(self.__class__.__name__)
        try:
            params = self.sample_params(sample[image_name],
                                        _field_spec(image_name, schema), rng)
            output = self._apply(sample, params, schema=schema)
            context.records.append(TransformRecord(self.__class__.__name__, params))
            return output
        except Exception as error:
            raise DataPipelineError(
                "geometry transform failed", stage=self.__class__.__name__,
                sample_id=context.sample_id, cause=error)


class GeometryCompose(Compose):
    def replay(self, sample, records, schema=None):
        if len(records) != len(self.transforms):
            raise ValueError("record count must match transform count")
        for transform, record in zip(self.transforms, records):
            replay = getattr(transform, "replay", None)
            if not callable(replay):
                raise TypeError("all composed transforms must support replay")
            sample = replay(sample, record, schema=schema)
        return sample


class Resize(GeometryTransform):
    def __init__(self, output_size, image_interpolation=None):
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = tuple(output_size)
        self.image_interpolation = (cv2.INTER_LINEAR if cv2 is not None and
                                    image_interpolation is None else image_interpolation)

    def sample_params(self, image, image_spec, rng):
        old_h, old_w = _image_shape(image, image_spec)
        new_h, new_w = self.output_size
        return dict(old_height=old_h, old_width=old_w,
                    height=int(new_h), width=int(new_w),
                    scale_y=float(new_h) / old_h,
                    scale_x=float(new_w) / old_w)

    def apply_image(self, value, params, spec):
        interpolation = self.image_interpolation
        if interpolation is None:
            interpolation = cv2.INTER_LINEAR
        return _resize_array(value, (params["height"], params["width"]),
                             interpolation, spec)

    def apply_mask(self, value, params, spec):
        return _resize_array(value, (params["height"], params["width"]),
                             cv2.INTER_NEAREST, spec)

    def apply_depth(self, value, params, spec):
        return _resize_array(value, (params["height"], params["width"]),
                             cv2.INTER_LINEAR, spec)

    def apply_densepose(self, value, params, spec):
        return self.apply_mask(value, params, spec)

    def apply_optical_flow(self, value, params, spec):
        flow = _resize_array(value, (params["height"], params["width"]),
                             cv2.INTER_LINEAR, spec)
        flow = np.array(flow, copy=True)
        flow[..., 0] *= params["scale_x"]
        flow[..., 1] *= params["scale_y"]
        return flow

    def apply_bbox(self, value, params, spec):
        boxes = np.array(value, copy=True)
        if boxes.size == 0:
            return boxes
        boxes[..., 0] *= params["scale_x"]
        boxes[..., 2] *= params["scale_x"]
        boxes[..., 1] *= params["scale_y"]
        boxes[..., 3] *= params["scale_y"]
        return boxes

    def _scale_points(self, value, params):
        points = np.array(value, copy=True)
        if points.size:
            points[..., 0] *= params["scale_x"]
            points[..., 1] *= params["scale_y"]
        return points

    def apply_keypoints(self, value, params, spec):
        return self._scale_points(value, params)

    def apply_landmarks(self, value, params, spec):
        return self._scale_points(value, params)

    def apply_polygon(self, value, params, spec):
        return [self._scale_points(polygon, params) for polygon in value]


class RandomHorizontalFlip(GeometryTransform):
    def __init__(self, probability=0.5):
        if not 0 <= probability <= 1:
            raise ValueError("probability must be in [0, 1]")
        self.probability = float(probability)

    def sample_params(self, image, image_spec, rng):
        height, width = _image_shape(image, image_spec)
        return dict(apply=bool(rng.uniform() < self.probability),
                    height=height, width=width)

    def apply_image(self, value, params, spec):
        return _flip_array(value, spec) if params["apply"] else value

    apply_mask = apply_image
    apply_depth = apply_image
    apply_densepose = apply_image

    def apply_optical_flow(self, value, params, spec):
        flow = self.apply_image(value, params, spec)
        if params["apply"]:
            flow = np.array(flow, copy=True)
            flow[..., 0] *= -1
        return flow

    def apply_bbox(self, value, params, spec):
        if not params["apply"]:
            return value
        boxes = np.array(value, copy=True)
        if boxes.size:
            old_x1 = boxes[..., 0].copy()
            old_x2 = boxes[..., 2].copy()
            boxes[..., 0] = params["width"] - old_x2
            boxes[..., 2] = params["width"] - old_x1
        return boxes

    def _flip_points(self, value, params):
        if not params["apply"]:
            return value
        points = np.array(value, copy=True)
        if points.size:
            points[..., 0] = params["width"] - points[..., 0]
        return points

    def apply_keypoints(self, value, params, spec):
        return self._flip_points(value, params)

    def apply_landmarks(self, value, params, spec):
        return self._flip_points(value, params)

    def apply_polygon(self, value, params, spec):
        return [self._flip_points(polygon, params) for polygon in value]


class RandomCrop(GeometryTransform):
    def __init__(self, output_size, image_fill=0, mask_fill=0):
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = tuple(output_size)
        self.image_fill = image_fill
        self.mask_fill = mask_fill

    def sample_params(self, image, image_spec, rng):
        source_h, source_w = _image_shape(image, image_spec)
        target_h, target_w = self.output_size
        crop_h = min(source_h, target_h)
        crop_w = min(source_w, target_w)
        max_y = source_h - crop_h
        max_x = source_w - crop_w
        y = int(rng.randint(0, max_y + 1)) if max_y else 0
        x = int(rng.randint(0, max_x + 1)) if max_x else 0
        return dict(x=x, y=y, crop_h=crop_h, crop_w=crop_w,
                    height=int(target_h), width=int(target_w))

    def apply_image(self, value, params, spec):
        return _crop_and_pad(value, params, self.image_fill, spec)

    def apply_mask(self, value, params, spec):
        return _crop_and_pad(value, params, self.mask_fill, spec)

    def apply_depth(self, value, params, spec):
        return _crop_and_pad(value, params, 0, spec)

    def apply_densepose(self, value, params, spec):
        return self.apply_mask(value, params, spec)

    def apply_optical_flow(self, value, params, spec):
        return _crop_and_pad(value, params, 0, spec)

    def apply_bbox(self, value, params, spec):
        boxes = np.array(value, copy=True)
        if boxes.size == 0:
            return boxes
        boxes[..., (0, 2)] -= params["x"]
        boxes[..., (1, 3)] -= params["y"]
        boxes[..., (0, 2)] = np.clip(boxes[..., (0, 2)], 0, params["width"])
        boxes[..., (1, 3)] = np.clip(boxes[..., (1, 3)], 0, params["height"])
        # Keep cardinality here so linked instance fields remain aligned.
        # SanitizeTargets owns the explicit drop/clip/error policy.
        return boxes

    def _crop_points(self, value, params):
        points = np.array(value, copy=True)
        if not points.size:
            return points
        points[..., 0] -= params["x"]
        points[..., 1] -= params["y"]
        visible = ((points[..., 0] >= 0) & (points[..., 0] < params["width"]) &
                   (points[..., 1] >= 0) & (points[..., 1] < params["height"]))
        if points.shape[-1] > 2:
            points[..., 2] = points[..., 2] * visible
        return points

    def apply_keypoints(self, value, params, spec):
        return self._crop_points(value, params)

    def apply_landmarks(self, value, params, spec):
        return self._crop_points(value, params)

    def apply_polygon(self, value, params, spec):
        return [self._crop_points(polygon, params) for polygon in value]


class GroupTransform(object):
    """Explicit contract for transforms consuming multiple samples."""

    def __call__(self, samples, contexts=None, schema=None):
        raise NotImplementedError


class SamplePool(object):
    """Bounded sample pool for Mosaic/CopyPaste-style transforms."""

    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._items = []

    def add(self, sample):
        self._items.append(sample)
        if len(self._items) > self.capacity:
            del self._items[0]

    def sample(self, count, rng):
        if count > len(self._items):
            raise ValueError("sample pool does not contain enough samples")
        indices = rng.choice(len(self._items), size=count, replace=False)
        return [self._items[int(index)] for index in indices]

    def __len__(self):
        return len(self._items)



def _homogeneous_points(value, matrix):
    points = np.asarray(value)
    if points.size == 0:
        return np.array(points, copy=True)
    result = np.array(points, copy=True, dtype=np.result_type(points.dtype, np.float32))
    xy = result[..., :2].reshape(-1, 2)
    homogeneous = np.concatenate([xy, np.ones((len(xy), 1))], axis=1)
    warped = np.dot(homogeneous, np.asarray(matrix, dtype=np.float64).T)
    denominator = np.where(np.abs(warped[:, 2:3]) < 1e-12, 1.0, warped[:, 2:3])
    result[..., :2] = (warped[:, :2] / denominator).reshape(result[..., :2].shape)
    return result


def _warp_array(value, params, interpolation, spec=None, fill_value=0):
    if cv2 is None:
        raise ImportError("OpenCV is required by pipeline geometry transforms")
    source, transposed = _to_hwc(np.asarray(value), spec)
    warped = cv2.warpPerspective(
        source, np.asarray(params["matrix"], dtype=np.float64),
        (int(params["width"]), int(params["height"])),
        flags=interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=fill_value)
    if source.ndim == 3 and warped.ndim == 2:
        warped = np.expand_dims(warped, -1)
    return _restore_layout(warped, transposed)


class MatrixGeometryTransform(GeometryTransform):
    """Base for transforms represented by a replayable 3x3 homography."""

    def __init__(self, image_fill=0, mask_fill=0):
        self.image_fill = image_fill
        self.mask_fill = mask_fill

    def apply_image(self, value, params, spec):
        return _warp_array(value, params, cv2.INTER_LINEAR, spec, self.image_fill)

    def apply_mask(self, value, params, spec):
        return _warp_array(value, params, cv2.INTER_NEAREST, spec, self.mask_fill)

    def apply_depth(self, value, params, spec):
        return _warp_array(value, params, cv2.INTER_LINEAR, spec, 0)

    def apply_densepose(self, value, params, spec):
        return self.apply_mask(value, params, spec)

    def apply_bbox(self, value, params, spec):
        boxes = np.asarray(value)
        if boxes.size == 0:
            return np.array(boxes, copy=True)
        coordinate_format = getattr(spec, "coordinate_format", None)
        if coordinate_format not in (None, "xyxy"):
            raise ValueError("matrix geometry currently requires xyxy bounding boxes")
        boxes = np.asarray(boxes, dtype=np.result_type(boxes.dtype, np.float32))
        corners = np.stack((
            boxes[..., (0, 1)], boxes[..., (2, 1)],
            boxes[..., (2, 3)], boxes[..., (0, 3)]), axis=-2)
        corners = _homogeneous_points(corners, params["matrix"])
        output = np.array(boxes, copy=True)
        output[..., 0] = corners[..., 0].min(axis=-1)
        output[..., 1] = corners[..., 1].min(axis=-1)
        output[..., 2] = corners[..., 0].max(axis=-1)
        output[..., 3] = corners[..., 1].max(axis=-1)
        output[..., (0, 2)] = np.clip(output[..., (0, 2)], 0, params["width"])
        output[..., (1, 3)] = np.clip(output[..., (1, 3)], 0, params["height"])
        return output

    def apply_keypoints(self, value, params, spec):
        return _homogeneous_points(value, params["matrix"])

    apply_landmarks = apply_keypoints

    def apply_polygon(self, value, params, spec):
        return [_homogeneous_points(polygon, params["matrix"]) for polygon in value]

    def apply_optical_flow(self, value, params, spec):
        flow = self.apply_image(value, params, spec)
        linear = np.asarray(params["matrix"], dtype=np.float64)[:2, :2]
        vectors = np.asarray(flow)[..., :2]
        flow[..., :2] = np.dot(vectors, linear.T)
        return flow


class RandomAffine(MatrixGeometryTransform):
    """Random rotation/scale/translation/shear sampled once for all fields."""

    def __init__(self, degrees=0, translate=None, scale=None, shear=0,
                 image_fill=0, mask_fill=0):
        super(RandomAffine, self).__init__(image_fill=image_fill, mask_fill=mask_fill)
        self.degrees = (-float(degrees), float(degrees)) if np.isscalar(degrees) else tuple(degrees)
        self.translate = tuple(translate or (0.0, 0.0))
        self.scale = tuple(scale or (1.0, 1.0))
        self.shear = (-float(shear), float(shear)) if np.isscalar(shear) else tuple(shear)
        if len(self.translate) != 2 or any(value < 0 or value > 1 for value in self.translate):
            raise ValueError("translate must contain fractions in [0, 1]")

    def sample_params(self, image, image_spec, rng):
        height, width = _image_shape(image, image_spec)
        angle = float(rng.uniform(*self.degrees))
        scale = float(rng.uniform(*self.scale))
        shear = float(rng.uniform(*self.shear))
        tx = float(rng.uniform(-self.translate[0], self.translate[0]) * width)
        ty = float(rng.uniform(-self.translate[1], self.translate[1]) * height)
        center = (width * 0.5, height * 0.5)
        affine = cv2.getRotationMatrix2D(center, angle, scale)
        matrix = np.vstack([affine, (0.0, 0.0, 1.0)])
        shear_matrix = np.array([[1.0, np.tan(np.deg2rad(shear)), tx],
                                 [0.0, 1.0, ty], [0.0, 0.0, 1.0]])
        matrix = np.dot(shear_matrix, matrix)
        return {"height": height, "width": width, "angle": angle,
                "scale": scale, "shear": shear, "translate": [tx, ty],
                "matrix": matrix.tolist(),
                "inverse_matrix": np.linalg.inv(matrix).tolist()}


class RandomPerspective(MatrixGeometryTransform):
    """Random corner displacement with a serializable inverse homography."""

    def __init__(self, distortion=0.1, probability=0.5, image_fill=0, mask_fill=0):
        super(RandomPerspective, self).__init__(image_fill=image_fill, mask_fill=mask_fill)
        if not 0 <= distortion <= 0.5 or not 0 <= probability <= 1:
            raise ValueError("distortion must be in [0, .5] and probability in [0, 1]")
        self.distortion = float(distortion)
        self.probability = float(probability)

    def sample_params(self, image, image_spec, rng):
        height, width = _image_shape(image, image_spec)
        source = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        if rng.uniform() < self.probability:
            jitter = np.column_stack((rng.uniform(-1, 1, 4) * width,
                                      rng.uniform(-1, 1, 4) * height))
            destination = source + jitter.astype(np.float32) * self.distortion
            applied = True
        else:
            destination = source.copy()
            applied = False
        matrix = cv2.getPerspectiveTransform(source, destination)
        return {"height": height, "width": width, "apply": applied,
                "source": source.tolist(), "destination": destination.tolist(),
                "matrix": matrix.tolist(),
                "inverse_matrix": np.linalg.inv(matrix).tolist()}


class SanitizeTargets(SampleTransform):
    """Clip/drop invalid boxes and keep explicitly linked fields aligned."""

    def __init__(self, policy="drop", min_area=0.0):
        if policy not in ("drop", "clip", "keep", "error"):
            raise ValueError("policy must be drop, clip, keep, or error")
        self.policy = policy
        self.min_area = float(min_area)

    def __call__(self, sample, context=None, schema=None):
        output = sample.copy() if hasattr(sample, "copy") else dict(sample)
        if schema is None or self.policy == "keep":
            return output
        bounds = None
        for image_field in schema:
            if image_field.kind == "image" and image_field.name in output:
                bounds = _image_shape(output[image_field.name], image_field)
                break
        for field in schema:
            if field.kind != "bbox" or field.name not in output:
                continue
            boxes = np.array(output[field.name], copy=True)
            if boxes.size == 0:
                continue
            in_bounds = np.ones(boxes.shape[:-1], dtype=bool)
            if bounds is not None:
                height, width = bounds
                in_bounds = ((boxes[..., 0] >= 0) & (boxes[..., 1] >= 0) &
                             (boxes[..., 2] <= width) & (boxes[..., 3] <= height))
                if self.policy in ("clip", "drop"):
                    boxes[..., (0, 2)] = np.clip(boxes[..., (0, 2)], 0, width)
                    boxes[..., (1, 3)] = np.clip(boxes[..., (1, 3)], 0, height)
            valid = (((boxes[..., 2] - boxes[..., 0]) *
                      (boxes[..., 3] - boxes[..., 1])) > self.min_area)
            if self.policy == "error" and not np.all(valid & in_bounds):
                raise DataPipelineError("invalid bounding box", stage="SanitizeTargets",
                                        field=field.name,
                                        sample_id=getattr(context, "sample_id", None))
            if self.policy == "clip":
                output[field.name] = boxes
            elif self.policy == "drop":
                output[field.name] = boxes[valid]
                for linked in schema:
                    if linked.metadata.get("linked_to") == field.name and linked.name in output:
                        value = output[linked.name]
                        if len(value) != len(valid):
                            raise ValueError("linked field length does not match boxes")
                        output[linked.name] = np.asarray(value)[valid]
        return output


class MixUp(GroupTransform):
    def __init__(self, alpha=0.2, image_field="image", target_field="label"):
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.alpha = float(alpha)
        self.image_field = image_field
        self.target_field = target_field

    def __call__(self, samples, contexts=None, schema=None):
        if len(samples) != 2:
            raise ValueError("MixUp requires exactly two samples")
        context = contexts[0] if contexts else TransformContext()
        ratio = float(context.random_state(self.__class__.__name__).beta(self.alpha, self.alpha))
        output = samples[0].copy()
        output[self.image_field] = (np.asarray(samples[0][self.image_field], dtype=np.float32) * ratio +
                                    np.asarray(samples[1][self.image_field], dtype=np.float32) * (1.0 - ratio))
        if self.target_field in samples[0] and self.target_field in samples[1]:
            output[self.target_field] = (samples[0][self.target_field],
                                         samples[1][self.target_field], ratio)
        output["_mixup"] = {"ratio": ratio}
        return output


class CutMix(GroupTransform):
    def __init__(self, alpha=1.0, image_field="image", target_field="label"):
        self.alpha = float(alpha)
        self.image_field = image_field
        self.target_field = target_field

    def __call__(self, samples, contexts=None, schema=None):
        if len(samples) != 2:
            raise ValueError("CutMix requires exactly two samples")
        context = contexts[0] if contexts else TransformContext()
        rng = context.random_state(self.__class__.__name__)
        first = np.array(samples[0][self.image_field], copy=True)
        second = np.asarray(samples[1][self.image_field])
        if first.shape != second.shape or first.ndim < 2:
            raise ValueError("CutMix images must have equal spatial shapes")
        height, width = first.shape[:2]
        ratio = float(rng.beta(self.alpha, self.alpha)) if self.alpha > 0 else 1.0
        cut_ratio = np.sqrt(1.0 - ratio)
        cut_w, cut_h = int(width * cut_ratio), int(height * cut_ratio)
        center_x, center_y = int(rng.randint(width)), int(rng.randint(height))
        x1, x2 = max(0, center_x - cut_w // 2), min(width, center_x + cut_w // 2)
        y1, y2 = max(0, center_y - cut_h // 2), min(height, center_y + cut_h // 2)
        first[y1:y2, x1:x2] = second[y1:y2, x1:x2]
        effective = 1.0 - float((x2 - x1) * (y2 - y1)) / (height * width)
        output = samples[0].copy()
        output[self.image_field] = first
        if self.target_field in samples[0] and self.target_field in samples[1]:
            output[self.target_field] = (samples[0][self.target_field],
                                         samples[1][self.target_field], effective)
        output["_cutmix"] = {"box": [x1, y1, x2, y2], "ratio": effective}
        return output


class Mosaic(GroupTransform):
    """Arrange four equally-shaped HWC images in a 2x2 canvas."""

    def __init__(self, image_field="image"):
        self.image_field = image_field

    def __call__(self, samples, contexts=None, schema=None):
        if len(samples) != 4:
            raise ValueError("Mosaic requires exactly four samples")
        images = [np.asarray(sample[self.image_field]) for sample in samples]
        shape = images[0].shape
        if any(image.shape != shape for image in images) or len(shape) < 2:
            raise ValueError("Mosaic images must have equal spatial shapes")
        output = samples[0].copy()
        output[self.image_field] = np.concatenate(
            [np.concatenate(images[:2], axis=1),
             np.concatenate(images[2:], axis=1)], axis=0)
        output["_mosaic"] = {"sample_count": 4}
        return output


class CopyPaste(GroupTransform):
    """Paste pixels selected by the second sample's boolean mask."""

    def __init__(self, image_field="image", mask_field="mask"):
        self.image_field = image_field
        self.mask_field = mask_field

    def __call__(self, samples, contexts=None, schema=None):
        if len(samples) != 2:
            raise ValueError("CopyPaste requires exactly two samples")
        image = np.array(samples[0][self.image_field], copy=True)
        source = np.asarray(samples[1][self.image_field])
        mask = np.asarray(samples[1][self.mask_field]).astype(bool)
        if image.shape != source.shape or image.shape[:2] != mask.shape[:2]:
            raise ValueError("CopyPaste image and mask shapes must align")
        selector = mask if image.ndim == 2 else mask[..., None]
        output = samples[0].copy()
        output[self.image_field] = np.where(selector, source, image)
        output["_copypaste"] = {"pixels": int(mask.sum())}
        return output