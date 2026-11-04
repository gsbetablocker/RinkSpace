"""Read-only source images and saved support evidence on the native clock."""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import json
import numpy as np


def native_pts(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError('Source PTS must be an integer')
    return int(value)


class SourceVideo:
    """Decode the exact source timestamp; never substitute a nearby frame/FPS."""
    def __init__(self, path, time_base):
        import av
        self.path = Path(path)
        self.time_base = Fraction(*time_base)
        self.container = av.open(str(self.path))
        self.stream = self.container.streams.video[0]
        self._frames = None
        self._last_pts = None
        self._image = None

    def read(self, pts):
        pts = native_pts(pts)
        if pts == self._last_pts:
            return self._image
        target = pts * self.time_base
        # Sequential playback reuses the decoder, while scrubbing seeks to an
        # earlier keyframe in the stream's own integer time base.
        if self._frames is None or self._last_pts is None or not 0 < (pts-self._last_pts)*self.time_base < 1:
            self.container.seek(int(target / self.stream.time_base), stream=self.stream, backward=True)
            self._frames = self.container.decode(self.stream)
        for frame in self._frames:
            if frame.pts is None:
                continue
            actual = int(frame.pts) * frame.time_base
            if actual < target:
                continue
            if actual != target:
                self._frames = None
                raise ValueError(f'No exact source frame for PTS {pts}; next source time is {actual}')
            self._image = frame.to_ndarray(format='rgb24')
            self._last_pts = pts
            return self._image
        self._frames = None
        raise ValueError(f'No exact source frame for PTS {pts}')

    def close(self):
        self.container.close()


class SupportObservations:
    """Keep source roles/foot associations; do not propagate across exposures."""
    def __init__(self, path, time_base, *, shot=None):
        self.document = json.loads(Path(path).read_text(encoding='utf8'))
        if tuple(self.document['time_base']) != tuple(time_base):
            raise ValueError('Support observation time base differs from player source')
        if shot is not None and self.document.get('shot_id') != shot:
            raise ValueError('Support observation shot differs from player source')
        self.by_pts = {}
        for row in self.document['observations']:
            pts = native_pts(row['pts'])
            for key in ('source_id','shot_id'):
                if row.get(key,self.document.get(key)) != self.document.get(key):
                    raise ValueError('Support row '+key+' differs from source document')
            if tuple(row.get('time_base', time_base)) != tuple(time_base):
                raise ValueError('Support row time base differs from player source')
            if row['role'] not in ('fit', 'review') or row.get('foot') not in (None, 0, 1):
                raise ValueError('Invalid source region role/foot')
            region_outline(row)  # Validate source pixels once, without fitting.
            self.by_pts.setdefault(pts, []).append(row)

    def at(self, pts):
        return self.by_pts.get(native_pts(pts), [])


def region_outline(row, samples=64):
    center = np.asarray(row['centre_uv'], float)
    shape = np.asarray(row['region_shape_px2'], float)
    if center.shape != (2,) or shape.shape != (2, 2) or not np.isfinite(center).all() or not np.isfinite(shape).all() or not np.allclose(shape, shape.T, rtol=0, atol=1e-12):
        raise ValueError('Invalid source region ellipse')
    root = np.linalg.cholesky(shape)
    theta = np.linspace(0, 2*np.pi, samples+1)
    return center + np.column_stack([np.cos(theta), np.sin(theta)]) @ root.T


def source_value(state, key, frame, foot=None):
    """Missing saved fields remain unknown, independently of geometry labels."""
    if key not in state:
        return 'unknown'
    value = np.asarray(state[key])[frame]
    if foot is not None and np.ndim(value):
        value = value[foot]
    return value.item() if np.ndim(value) == 0 else value


def observation_used(state, row, frame):
    if row['role'] == 'review':
        return False
    if row.get('foot') not in (0, 1):
        return 'unknown'
    return source_value(state, 'support_image_in_optimization', frame, row['foot'])


def load_world_regions(path):
    """Optional already saved nominal-plane ranges, in display world metres."""
    document = json.loads(Path(path).read_text(encoding='utf8'))
    rows = document['rows']
    for row in rows:
        native_pts(row['pts'])
        points = np.asarray(row['world_polyline_m'], float)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3 or not np.isfinite(points).all():
            raise ValueError('Invalid conditional world region')
        if row['role'] not in ('fit', 'review') or row.get('foot') not in (None, 0, 1):
            raise ValueError('Invalid conditional world region role/foot')
    return rows
