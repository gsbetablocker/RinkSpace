"""Display current source coordinates on their actual exposure, without projection.

The producer supplies the already qualified active input. This module never
reads historical UV, fills missing frames, or projects a physical model.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


SCHEMA = 'rinkspace.current-active-source-points.v1'
SHARED_SCHEMA = 'rinkspace.current-active-source-points-shared-UV.v1'
_FROZEN_REFERENCES = {}


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def draw_current_source_pixel_markers(image, uv):
    """Shared marker renderer; source selection belongs to the caller."""
    for value in np.asarray(uv, np.float64).reshape(-1, 2):
        centre = tuple(np.rint(value * 16).astype(int))
        cv2.circle(image, centre, 2 * 16, (0, 255, 255), 1, cv2.LINE_AA, shift=4)
    return image


def _authenticate_shared(reference):
    path = Path(reference['path']).resolve()
    stat = path.stat()
    identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    cached = _FROZEN_REFERENCES.get(str(path))
    if cached != (identity, reference['sha256']):
        if file_sha(path) != reference['sha256']:
            raise ValueError('Frozen active source UV changed')
        _FROZEN_REFERENCES[str(path)] = (identity, reference['sha256'])
    return path


def freeze_current_source_point_inputs(path, point_groups, *, actual_pts,
                                       input_manifest, family_receipts):
    """Write the assembled live FIT pixels once per run, without any q."""
    path = Path(path).resolve()
    if path.exists():
        raise ValueError('A run must not overwrite its frozen active source pixels')
    pts = np.asarray(actual_pts, np.int64)
    if not np.array_equal(pts, np.unique(pts)) or not len(pts):
        raise ValueError('Exact sorted programme actual exposures required')
    if not family_receipts or any(r.get('complete_active_input') is not True for r in family_receipts):
        raise ValueError('All installed active source families need receipts')
    groups = {int(p): [] for p in pts}
    for stamp, uv in point_groups:
        values = np.asarray(uv, np.float64)
        if (int(stamp) not in groups or values.ndim != 2 or values.shape[1] != 2
                or not np.isfinite(values).all() or np.any(values < 0)
                or np.any(values >= [1920, 1080])):
            raise ValueError('Exact active native UV/exposure required')
        groups[int(stamp)].append(values)
    offsets, chunks = [0], []
    for stamp in pts:
        incoming = np.concatenate(groups[int(stamp)]) if groups[int(stamp)] else np.empty((0, 2))
        # Repeated current operators can use one identical source pixel. Its
        # marker is displayed once; no source point is density/error filtered.
        values = np.unique(incoming, axis=0)
        chunks.append(values)
        offsets.append(offsets[-1]+len(values))
    metadata = dict(schema=SHARED_SCHEMA, complete_current_active_input=True,
        input_manifest=dict(path=str(Path(input_manifest).resolve()), sha256=file_sha(input_manifest)),
        family_receipts=family_receipts, historical_or_retired_points_included=False,
        model_projected_points_included=False, nonpixel_placeholder_constraints_included=False,
        extra_point_labels=False, identical_current_pixel_coordinates_displayed_once=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, actual_pts=pts, offsets=np.asarray(offsets, np.int64),
                        uv=np.concatenate(chunks), metadata_json=json.dumps(metadata, ensure_ascii=False))
    reference = dict(path=str(path), sha256=file_sha(path), points=offsets[-1], actual_exposures=len(pts))
    _authenticate_shared(reference)
    return reference


def save_current_source_point_binding(path, camera_path, frozen_reference, *, q_sha256,
                                      input_manifest):
    """Bind a saved state to one immutable run pixel archive; never rescan UV."""
    shared = _authenticate_shared(frozen_reference)
    with np.load(shared, allow_pickle=False) as values:
        exposure_pts = values['actual_pts']
        m = json.loads(str(values['metadata_json'].item()))
    manifest = dict(path=str(Path(input_manifest).resolve()), sha256=file_sha(input_manifest))
    if m.get('schema') != SHARED_SCHEMA or m['input_manifest'] != manifest:
        raise ValueError('Frozen source input manifest differs from this run')
    camera_path, path = Path(camera_path).resolve(), Path(path).resolve()
    with np.load(camera_path, allow_pickle=False) as camera:
        native = camera['pts'].astype(np.int64)
        actual = camera['exposure_representative_pts'].astype(np.int64)
        cm = json.loads(str(camera['metadata_json'].item()))
    if not np.array_equal(exposure_pts, np.unique(actual)):
        raise ValueError('Frozen pixels and saved camera exposure inventories differ')
    metadata = dict(m, schema=SCHEMA, q_sha256=q_sha256,
        camera_sha256=file_sha(camera_path), source_video_sha256=cm['source_sha256'],
        time_base=cm['source_time_base'], native_pts_sha256=hashlib.sha256(native.tobytes()).hexdigest(),
        exposure_representative_pts_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
        shared_active_UV_frozen_once_per_run=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, actual_pts=exposure_pts,
        points_reference_json=json.dumps(frozen_reference, ensure_ascii=False),
        metadata_json=json.dumps(metadata, ensure_ascii=False))
    return dict(path=str(path), sha256=file_sha(path), points=frozen_reference['points'],
                actual_exposures=len(exposure_pts), q_sha256=q_sha256,
                frozen_active_UV_reference=dict(frozen_reference))


def save_current_source_points(path, camera_path, point_groups, *, q_sha256,
                               input_manifest, family_receipts):
    """Save exact active UV groups; callers must enumerate every source family.

    A group is (actual exposure PTS, N-by-2 current UV). Nonpixel inequalities,
    removed points, HELD rows and historical pre-correction UV are not groups.
    Empty exposures are explicit, never filled from neighbours.
    """
    camera_path, path = Path(camera_path).resolve(), Path(path).resolve()
    with np.load(camera_path, allow_pickle=False) as camera:
        native = camera['pts'].astype(np.int64)
        actual = camera['exposure_representative_pts'].astype(np.int64)
        camera_metadata = json.loads(str(camera['metadata_json'].item()))
    exposures = np.unique(actual)
    groups = {int(p): [] for p in exposures}
    for pts, uv in point_groups:
        if int(pts) not in groups:
            raise ValueError('Current programme points need an exact camera exposure')
        values = np.asarray(uv, dtype=np.float64)
        if (values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all()
                or np.any(values < 0) or np.any(values >= [1920, 1080])):
            raise ValueError('Current native pixel coordinates required')
        groups[int(pts)].append(values)
    if not family_receipts or any(r.get('complete_active_input') is not True for r in family_receipts):
        raise ValueError('Every selected current source family requires a complete input receipt')
    offsets, chunks = [0], []
    for pts in exposures:
        uv = np.concatenate(groups[int(pts)]) if groups[int(pts)] else np.empty((0, 2))
        chunks.append(uv)
        offsets.append(offsets[-1] + len(uv))
    metadata = dict(schema=SCHEMA, q_sha256=q_sha256,
        camera_sha256=file_sha(camera_path), source_video_sha256=camera_metadata['source_sha256'],
        time_base=camera_metadata['source_time_base'],
        native_pts_sha256=hashlib.sha256(native.tobytes()).hexdigest(),
        exposure_representative_pts_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
        input_manifest=dict(path=str(Path(input_manifest).resolve()), sha256=file_sha(input_manifest)),
        family_receipts=family_receipts, complete_current_active_input=True,
        historical_or_retired_points_included=False, model_projected_points_included=False,
        nonpixel_placeholder_constraints_included=False, extra_point_labels=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, actual_pts=exposures, offsets=np.asarray(offsets, np.int64),
                        uv=np.concatenate(chunks), metadata_json=json.dumps(metadata, ensure_ascii=False))
    return dict(path=str(path), sha256=file_sha(path), points=offsets[-1],
                actual_exposures=len(exposures), q_sha256=q_sha256)


class CurrentSourcePoints:
    def __init__(self, path, camera_path, *, q_sha256=None):
        self.path = Path(path).resolve()
        with np.load(self.path, allow_pickle=False) as values:
            self.pts = values['actual_pts'].copy()
            self.metadata = json.loads(str(values['metadata_json'].item()))
            if 'points_reference_json' in values:
                reference = json.loads(str(values['points_reference_json'].item()))
                # A consumer authenticates the complete immutable pixel file;
                # production state saves only use its already sealed identity.
                if file_sha(reference['path']) != reference['sha256']:
                    raise ValueError('Frozen current source UV reference changed')
                with np.load(reference['path'], allow_pickle=False) as shared:
                    shared_metadata = json.loads(str(shared['metadata_json'].item()))
                    if (shared_metadata.get('schema') != SHARED_SCHEMA
                            or not np.array_equal(self.pts, shared['actual_pts'])
                            or any(shared_metadata[k] != self.metadata[k]
                                   for k in ('input_manifest', 'family_receipts', 'complete_current_active_input',
                                             'historical_or_retired_points_included', 'model_projected_points_included',
                                             'nonpixel_placeholder_constraints_included', 'extra_point_labels'))):
                        raise ValueError('Bound state and frozen source input differ')
                    self.offsets = shared['offsets'].copy()
                    self.uv = shared['uv'].copy()
            else:
                self.offsets = values['offsets'].copy()
                self.uv = values['uv'].copy()
        m = self.metadata
        with np.load(camera_path, allow_pickle=False) as camera:
            native = camera['pts'].astype(np.int64)
            actual = camera['exposure_representative_pts'].astype(np.int64)
            cm = json.loads(str(camera['metadata_json'].item()))
        if (m.get('schema') != SCHEMA or m.get('complete_current_active_input') is not True
                or m.get('historical_or_retired_points_included') is not False
                or m.get('model_projected_points_included') is not False
                or m.get('nonpixel_placeholder_constraints_included') is not False
                or m.get('extra_point_labels') is not False
                or m.get('camera_sha256') != file_sha(camera_path)
                or m.get('source_video_sha256') != cm['source_sha256']
                or m.get('time_base') != cm['source_time_base']
                or m.get('native_pts_sha256') != hashlib.sha256(native.tobytes()).hexdigest()
                or m.get('exposure_representative_pts_sha256') != hashlib.sha256(actual.tobytes()).hexdigest()
                or (q_sha256 is not None and m.get('q_sha256') != q_sha256)):
            raise ValueError('Current source points must match the exact saved camera, model and source clock')
        if (not np.array_equal(self.pts, np.unique(actual)) or self.offsets.dtype.kind not in 'iu'
                or self.offsets.shape != (len(self.pts) + 1,) or self.offsets[0] != 0
                or np.any(np.diff(self.offsets) < 0) or self.uv.shape != (self.offsets[-1], 2)
                or not np.isfinite(self.uv).all() or np.any(self.uv < 0)
                or np.any(self.uv >= [1920, 1080])):
            raise ValueError('All current exposure point arrays must have an exact native pixel layout')
        self.lookup = {int(p): i for i, p in enumerate(self.pts)}
        self.reference = dict(path=str(self.path), sha256=file_sha(self.path),
                              points=len(self.uv), actual_exposures=len(self.pts),
                              current_active_points_only=True, extra_point_labels=False)

    def points(self, actual_pts):
        i = self.lookup[int(actual_pts)]
        return self.uv[self.offsets[i]:self.offsets[i+1]]

    def draw(self, image, actual_pts):
        # Thin fixed markers preserve the source texture at the point centre.
        # Every active point is drawn; no confidence, fit-error or density filter.
        return draw_current_source_pixel_markers(image, self.points(actual_pts))
