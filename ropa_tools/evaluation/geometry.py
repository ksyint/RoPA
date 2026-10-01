"""Coordinate transforms between decoded frames, processor crops and patch grids."""
from dataclasses import dataclass, replace
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FrameGeometry:
    image_height: int
    image_width: int
    resize_shorter: int
    crop_height: int
    crop_width: int
    grid_height: int
    grid_width: int

    def __post_init__(self):
        if min(self.image_height, self.image_width, self.resize_shorter,
               self.crop_height, self.crop_width, self.grid_height, self.grid_width) < 1:
            raise ValueError('All image and token-grid dimensions must be positive.')
        if self.resized_height < self.crop_height or self.resized_width < self.crop_width:
            raise ValueError('The resized image must cover the centered crop.')

    @property
    def resized_height(self):
        return int(self.image_height * self.resize_shorter / min(self.image_height, self.image_width))

    @property
    def resized_width(self):
        return int(self.image_width * self.resize_shorter / min(self.image_height, self.image_width))

    @property
    def offset(self):
        return ((self.resized_width - self.crop_width) // 2,
                (self.resized_height - self.crop_height) // 2)

    def to_crop(self, points):
        if points.shape[-1] != 2:
            raise ValueError('Point coordinates use a final x,y dimension.')
        output = points.float().clone()
        output[..., 0] = (points[..., 0] + .5) * self.resized_width / self.image_width - .5 - self.offset[0]
        output[..., 1] = (points[..., 1] + .5) * self.resized_height / self.image_height - .5 - self.offset[1]
        visible = ((output[..., 0] >= -.5) & (output[..., 0] < self.crop_width - .5)
                   & (output[..., 1] >= -.5) & (output[..., 1] < self.crop_height - .5))
        return output, visible

    def to_grid(self, points):
        crop, visible = self.to_crop(points)
        crop[..., 0] = (crop[..., 0] + .5) * self.grid_width / self.crop_width - .5
        crop[..., 1] = (crop[..., 1] + .5) * self.grid_height / self.crop_height - .5
        return crop, visible

    def to_image(self, grid_points):
        result = grid_points.float().clone()
        result[..., 0] = (result[..., 0] + .5) * self.crop_width / self.grid_width - .5
        result[..., 1] = (result[..., 1] + .5) * self.crop_height / self.grid_height - .5
        result[..., 0] = (result[..., 0] + self.offset[0] + .5) * self.image_width / self.resized_width - .5
        result[..., 1] = (result[..., 1] + self.offset[1] + .5) * self.image_height / self.resized_height - .5
        return result

    def normalize_256(self, image_points):
        result = image_points.float().clone()
        result[..., 0] *= 256 / self.image_width
        result[..., 1] *= 256 / self.image_height
        return result

    @classmethod
    def from_record(cls, record):
        crop = record.get('crop_size', 384)
        return cls(
            image_height=int(record['image_height']),
            image_width=int(record['image_width']),
            resize_shorter=int(record.get('resize_shorter', crop)),
            crop_height=int(record.get('crop_height', crop)),
            crop_width=int(record.get('crop_width', crop)),
            grid_height=int(record['grid_height']),
            grid_width=int(record['grid_width']),
        )


def resample_grid(features, source_shape, target_shape):
    if features.ndim != 3 or features.shape[1] != math.prod(source_shape):
        raise ValueError('Dense features need T,H*W,D matching the source grid.')
    if min(*source_shape, *target_shape) < 1:
        raise ValueError('Source and destination grids must be positive.')
    if tuple(source_shape) == tuple(target_shape):
        return features
    maps = features.reshape(len(features), *source_shape, -1).permute(0, 3, 1, 2)
    resized = F.interpolate(maps.float(), size=target_shape, mode='bilinear', align_corners=False)
    return resized.permute(0, 2, 3, 1).flatten(1, 2)


def query_frame_indices(timestamps, queries):
    if timestamps.ndim != 1 or len(timestamps) == 0:
        raise ValueError('Feature timestamps must be a nonempty vector.')
    if not torch.isfinite(timestamps).all() or not (timestamps[1:] > timestamps[:-1]).all():
        raise ValueError('Feature timestamps must be finite and strictly increasing.')
    if not torch.isfinite(queries).all():
        raise ValueError('Query timestamps must be finite.')
    if (queries < timestamps[0]).any() or (queries > timestamps[-1]).any():
        raise ValueError('A query timestamp lies outside the feature interval.')
    location = torch.searchsorted(timestamps, queries)
    upper = location.clamp_max(len(timestamps) - 1)
    lower = (location - 1).clamp_min(0)
    return torch.where((timestamps[upper] - queries).abs() < (timestamps[lower] - queries).abs(), upper, lower)


def interpolate_tracks_at_times(points, visible, source_times, target_times):
    if points.ndim != 3 or points.shape[-1] != 2 or visible.shape != points.shape[:2]:
        raise ValueError('Track interpolation needs Q,T,2 points and Q,T visibility.')
    if source_times.ndim != 1 or len(source_times) != points.shape[1]:
        raise ValueError('Track timestamps must match the original frame axis.')
    if target_times.ndim != 1 or not len(target_times):
        raise ValueError('Target timestamps must be a nonempty vector.')
    if not torch.isfinite(source_times).all() or not torch.isfinite(target_times).all():
        raise ValueError('Annotation and feature timestamps must be finite.')
    if not (source_times[1:] > source_times[:-1]).all():
        raise ValueError('Original frame timestamps must be strictly increasing.')
    if target_times.min() < source_times[0] or target_times.max() > source_times[-1]:
        raise ValueError('Target timestamps must remain within the annotation interval.')
    upper = torch.searchsorted(source_times, target_times).clamp_max(len(source_times) - 1)
    lower = (upper - 1).clamp_min(0)
    exact = source_times[upper] == target_times
    lower = torch.where(exact, upper, lower)
    span = source_times[upper] - source_times[lower]
    fraction = torch.where(span > 0, (target_times - source_times[lower]) / span.clamp_min(1e-12), 0.)
    first, second = points[:, lower], points[:, upper]
    interpolated = first + fraction[None, :, None] * (second - first)
    eligible = visible[:, lower] & visible[:, upper]
    interpolated = torch.where(eligible[..., None], interpolated, torch.zeros_like(interpolated))
    return interpolated, eligible


def feature_timestamps_from_record(record, device):
    values = record.get('tubelet_timestamps')
    if values is None:
        raise ValueError('Feature bank entry does not include presentation-time metadata.')
    timestamps = torch.tensor(values, dtype=torch.float64, device=device)
    if timestamps.shape != (record['shape'][0],):
        raise ValueError('Feature timestamps must match the saved feature frame count.')
    if not torch.isfinite(timestamps).all() or (timestamps[1:] < timestamps[:-1]).any():
        raise ValueError('Feature timestamps must be finite and nondecreasing.')
    return timestamps


def align_temporal_annotations(record, queries, targets, visible, source_timestamps):
    timestamps = feature_timestamps_from_record(record, targets.device)
    if not (timestamps[1:] > timestamps[:-1]).all():
        raise ValueError('Temporal annotation interpolation needs distinct feature timestamps.')
    source_timestamps = torch.as_tensor(source_timestamps, device=targets.device, dtype=torch.float64)
    points, eligible = interpolate_tracks_at_times(targets, visible, source_timestamps, timestamps)
    if not torch.equal(queries[:, 0], queries[:, 0].round()):
        raise ValueError('Query source-frame coordinates must be integral.')
    original_frames = queries[:, 0].long()
    if original_frames.min() < 0 or original_frames.max() >= len(source_timestamps):
        raise ValueError('A query frame lies outside the original timestamp list.')
    query_times = source_timestamps[original_frames]
    mapped = query_frame_indices(timestamps, query_times)
    identifiers = torch.arange(len(queries), device=queries.device)
    if not eligible[identifiers, mapped].all():
        raise ValueError('A query is occluded at its selected feature timestamp.')
    updated = queries.clone()
    updated[:, 0] = mapped
    updated[:, 1] = points[identifiers, mapped, 1]
    updated[:, 2] = points[identifiers, mapped, 0]
    return updated, points, eligible


def tracking_resolution(geometry, features, stride=None):
    if stride is None:
        return geometry, features
    if stride < 1 or geometry.crop_height % stride or geometry.crop_width % stride:
        raise ValueError('Tracking stride must divide both processor-crop dimensions.')
    height, width = geometry.crop_height // stride, geometry.crop_width // stride
    values = resample_grid(features, (geometry.grid_height, geometry.grid_width), (height, width))
    return replace(geometry, grid_height=height, grid_width=width), values


def interpolate_body_geometry(arrays, source_times, target_times, device):
    source_times = torch.as_tensor(source_times, dtype=torch.float64, device=device)
    upper = torch.searchsorted(source_times, target_times).clamp_max(len(source_times) - 1)
    lower = (upper - 1).clamp_min(0)
    lower = torch.where(source_times[upper] == target_times, upper, lower)
    span = source_times[upper] - source_times[lower]
    fraction = torch.where(span > 0, (target_times - source_times[lower]) / span.clamp_min(1e-12), 0.)
    result = dict(arrays)
    for name in ('boxes', 'normalizers'):
        if name not in arrays:
            continue
        values = torch.as_tensor(arrays[name], dtype=torch.float32, device=device)
        per_query = values.ndim == (3 if name == 'boxes' else 2)
        axis = 1 if per_query else 0
        if values.shape[axis] != len(source_times):
            raise ValueError(f'{name} must share the original annotation timestamp axis.')
        first, second = values.index_select(axis, lower), values.index_select(axis, upper)
        shape = [1] * values.ndim
        shape[axis] = len(target_times)
        output = first + fraction.reshape(shape).float() * (second - first)
        if not torch.isfinite(output).all():
            raise ValueError(f'Interpolated {name} contains nonfinite values.')
        result[name] = output
    return result
