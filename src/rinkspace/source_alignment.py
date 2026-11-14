"""Native-PTS image correspondence candidates, never authoritative clock mappings.

Fingerprints use explicit original-pixel crops, RGB-to-gray conversion and area
resize. A matched anchor locates a search window only; each query image is
compared independently. Repeated exposures and textureless images stay visible
in the scores, ties and texture statistics. No frame or PTS is synthesized.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import math
import os
import re
import uuid

import numpy as np


@dataclass(frozen=True)
class FingerprintSequence:
    """Actual decoded PTS/time bases and uint8 gray images in decoding order."""

    pts: np.ndarray
    time_base_num: np.ndarray
    time_base_den: np.ndarray
    gray: np.ndarray
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        count = len(self.pts)
        for name in ('pts', 'time_base_num', 'time_base_den'):
            value = np.asarray(getattr(self, name))
            if value.ndim != 1 or len(value) != count or value.dtype.kind not in 'iu':
                raise ValueError(name+' must be a one-dimensional integer array')
            if value.dtype.kind == 'u' and np.any(value > np.iinfo(np.int64).max):
                raise ValueError(name+' exceeds signed 64-bit storage')
            value = np.asarray(value, dtype=np.int64)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if np.any(self.time_base_num <= 0) or np.any(self.time_base_den <= 0):
            raise ValueError('Frame time bases must be positive')
        if count > 1 and np.any(self.pts[1:] <= self.pts[:-1]):
            raise ValueError('Actual decoded PTS must be strictly increasing; do not reorder or repair them')
        gray = np.asarray(self.gray)
        if gray.dtype != np.uint8 or gray.ndim != 3 or gray.shape[0] != count or min(gray.shape[1:]) <= 0:
            raise ValueError('gray must be uint8 [frames, height, width]')
        gray.setflags(write=False)
        object.__setattr__(self, 'gray', gray)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name+' must be an integer')
    value = int(value)
    if not np.iinfo(np.int64).min <= value <= np.iinfo(np.int64).max:
        raise ValueError(name+' exceeds signed 64-bit storage')
    return value


def _sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _stat(path):
    stat = Path(path).stat()
    return dict(size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def _load_cache(npz_path, json_path, binding):
    metadata = json.loads(json_path.read_text(encoding='utf-8'))
    if metadata.get('binding') != binding or _sha256(npz_path) != metadata.get('npz_sha256'):
        raise ValueError('Fingerprint cache binding or array file changed')
    with np.load(npz_path, allow_pickle=False) as saved:
        if set(saved.files) != {'pts', 'time_base_num', 'time_base_den', 'gray'}:
            raise ValueError('Unexpected fingerprint cache arrays')
        result = FingerprintSequence(**{key: saved[key] for key in saved.files}, metadata=metadata)
    if len(result.pts) != metadata.get('decoded_frames'):
        raise ValueError('Fingerprint cache frame count changed')
    return result


def decode_fingerprints(source_path, *, source_sha256, first_pts,
                        stop_pts_exclusive, crop_xyxy, size_wh, cache_dir,
                        data_root=None, time_base=None, native_pts_step=None,
                        video_stream_index=0):
    """Decode an explicit native-PTS interval and cache NPZ/JSON under Data.

    crop_xyxy is an original-pixel half-open rectangle; size_wh is the output
    width/height. time_base, when supplied, is (numerator, denominator).
    native_pts_step is optional gap diagnostics, never a rule for creating PTS.
    A cold cache verifies the supplied source SHA. A warm cache reuses that
    verification only while the resolved source path, size and mtime_ns match.
    Decode uses actual frame.pts/frame.time_base and fails on missing clocks.
    """
    import av
    import cv2
    from .path_policy import PROJECT_ROOT
    from .storage import configured_data_root

    source = Path(source_path).resolve(strict=True)
    if not isinstance(source_sha256, str) or not re.fullmatch(r'[0-9a-f]{64}', source_sha256):
        raise ValueError('source_sha256 must be a lowercase SHA256')
    first_pts = _integer(first_pts, 'first_pts')
    stop_pts_exclusive = _integer(stop_pts_exclusive, 'stop_pts_exclusive')
    if stop_pts_exclusive <= first_pts:
        raise ValueError('Require a nonempty half-open native PTS interval')
    crop = tuple(_integer(v, 'crop_xyxy') for v in crop_xyxy)
    size = tuple(_integer(v, 'size_wh') for v in size_wh)
    if len(crop) != 4 or min(crop[:2]) < 0 or crop[2] <= crop[0] or crop[3] <= crop[1]:
        raise ValueError('crop_xyxy must be an explicit nonempty original-pixel rectangle')
    if len(size) != 2 or min(size) <= 0:
        raise ValueError('size_wh must contain positive width and height')
    video_stream_index = _integer(video_stream_index, 'video_stream_index')
    if video_stream_index < 0:
        raise ValueError('video_stream_index must be nonnegative')
    if native_pts_step is not None:
        native_pts_step = _integer(native_pts_step, 'native_pts_step')
        if native_pts_step <= 0:
            raise ValueError('native_pts_step must be positive')
    expected_tb = None
    if time_base is not None:
        if len(time_base) != 2:
            raise ValueError('time_base must be a numerator/denominator pair')
        num, den = (_integer(v, 'time_base') for v in time_base)
        if num <= 0 or den <= 0:
            raise ValueError('time_base must be positive')
        expected_tb = Fraction(num, den)
    data = configured_data_root(PROJECT_ROOT, data_root=data_root)
    cache = Path(cache_dir).resolve()
    if not cache.is_relative_to(data) or cache.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError('Fingerprint caches must stay under the configured external Data root')
    source_stat = _stat(source)
    binding = dict(schema='rinkspace.source_fingerprints.v1',
        source=dict(path=str(source), sha256=source_sha256, stat=source_stat),
        first_pts=first_pts, stop_pts_exclusive=stop_pts_exclusive,
        expected_time_base=([expected_tb.numerator, expected_tb.denominator] if expected_tb else None),
        native_pts_step=native_pts_step, video_stream_index=video_stream_index,
        crop_xyxy=list(crop), size_wh=list(size),
        method='av_rgb24_cv2_RGB2GRAY_INTER_AREA.v1', av_version=av.__version__, cv2_version=cv2.__version__)
    key = hashlib.sha256(_canonical(binding)).hexdigest()
    npz_path, json_path = cache/(key+'.npz'), cache/(key+'.json')
    if npz_path.exists() or json_path.exists():
        if not (npz_path.is_file() and json_path.is_file()):
            raise ValueError('Incomplete fingerprint cache; preserve it and use a new cache directory')
        result = _load_cache(npz_path, json_path, binding)
        if _stat(source) != source_stat:
            raise ValueError('Source changed while reading the fingerprint cache')
        return result
    if _sha256(source) != source_sha256 or _stat(source) != source_stat:
        raise ValueError('Source SHA256 differs or its stat changed during verification')
    pts, numerators, denominators, gray = [], [], [], []
    frame_size = None
    with av.open(str(source)) as container:
        stream = container.streams.video[video_stream_index]
        if stream.time_base is None:
            raise ValueError('Source video stream has no native time base')
        stream_tb = Fraction(stream.time_base)
        if expected_tb is not None and stream_tb != expected_tb:
            raise ValueError('Source stream time base differs from the requested native clock')
        container.seek(first_pts, stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                raise ValueError('Decoded frame has no actual PTS/time_base; no clock was reconstructed')
            actual_pts, actual_tb = _integer(frame.pts, 'decoded PTS'), Fraction(frame.time_base)
            if actual_tb != stream_tb:
                raise ValueError('Decoded time base changed; raw PTS interval cannot be reused')
            if actual_pts < first_pts:
                continue
            if actual_pts >= stop_pts_exclusive:
                break
            if pts and actual_pts <= pts[-1]:
                raise ValueError('Non-increasing actual decoded PTS; source was not reordered')
            current_size = [frame.width, frame.height]
            if frame_size is not None and current_size != frame_size:
                raise ValueError('Original frame dimensions changed inside the requested interval')
            frame_size = current_size
            x0, y0, x1, y1 = crop
            if x1 > frame.width or y1 > frame.height:
                raise ValueError('Original-pixel crop exceeds the actual source frame')
            rgb = frame.to_ndarray(format='rgb24')
            image = cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
            gray.append(cv2.resize(image, size, interpolation=cv2.INTER_AREA))
            pts.append(actual_pts)
            numerators.append(actual_tb.numerator)
            denominators.append(actual_tb.denominator)
    if not pts:
        raise ValueError('No actual decoded exposure in the requested native interval')
    if _stat(source) != source_stat:
        raise ValueError('Source changed during fingerprint decoding')
    arrays = dict(pts=np.asarray(pts, dtype=np.int64),
        time_base_num=np.asarray(numerators, dtype=np.int64),
        time_base_den=np.asarray(denominators, dtype=np.int64), gray=np.stack(gray))
    deviations = []
    if native_pts_step is not None:
        deviations = [dict(before_pts=a, after_pts=b, delta_pts=b-a)
                      for a, b in zip(pts, pts[1:]) if b-a != native_pts_step]
    metadata = dict(binding=binding, decoded_frames=len(pts), frame_size_wh=frame_size,
        first_actual_pts=pts[0], last_actual_pts=pts[-1],
        requested_first_pts_observed=pts[0] == first_pts,
        requested_last_step_observed=(pts[-1] == stop_pts_exclusive-native_pts_step
                                     if native_pts_step is not None else None),
        adjacent_pts_step_deviations=deviations, synthetic_frames=0,
        decoded_exposure_cadence_checked=False,
        cadence_note='Actual PTS are retained; repeated-image/exposure cadence is not classified.',
        npz_path=str(npz_path), json_path=str(json_path))
    cache.mkdir(parents=True, exist_ok=True)
    temporary = cache/(key+'.'+uuid.uuid4().hex+'.pending')
    try:
        with temporary.open('xb') as writer:
            np.savez_compressed(writer, **arrays)
        metadata['npz_sha256'] = _sha256(temporary)
        os.replace(temporary, npz_path)
        with temporary.open('x', encoding='utf-8', newline='\n') as writer:
            json.dump(metadata, writer, ensure_ascii=False, indent=2, allow_nan=False)
            writer.write('\n')
        os.replace(temporary, json_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return FingerprintSequence(**arrays, metadata=metadata)


def mae_candidates(fingerprint, target, *, first_pts=None,
                   stop_pts_exclusive=None, top_k=2):
    """Rank actual target exposures by gray MAE; positive margins prove no identity.

    Ties count across the full selected target interval, even when top_k is
    smaller. Gray standard deviations expose low texture without imposing an
    arbitrary confidence threshold or rejecting duplicate images.
    """
    query = np.asarray(fingerprint)
    if query.dtype != np.uint8 or query.shape != target.gray.shape[1:]:
        raise ValueError('Query must be a uint8 gray fingerprint of the same size')
    top_k = _integer(top_k, 'top_k')
    if top_k < 2:
        raise ValueError('top_k must retain at least the best and second candidate')
    selected = np.ones(len(target.pts), dtype=bool)
    if first_pts is not None:
        selected &= target.pts >= _integer(first_pts, 'first_pts')
    if stop_pts_exclusive is not None:
        selected &= target.pts < _integer(stop_pts_exclusive, 'stop_pts_exclusive')
    indices = np.flatnonzero(selected)
    scores = np.empty(len(indices), dtype=np.float64)
    signed_query = query.astype(np.int16)
    for start in range(0, len(indices), 256):
        chunk = indices[start:start+256]
        scores[start:start+len(chunk)] = np.abs(
            target.gray[chunk].astype(np.int16)-signed_query).mean(axis=(1, 2))
    order = np.argsort(scores, kind='stable')[:top_k]
    candidates = [dict(index=int(indices[i]), pts=int(target.pts[indices[i]]),
        time_base_num=int(target.time_base_num[indices[i]]),
        time_base_den=int(target.time_base_den[indices[i]]), mae=float(scores[i]),
        gray_stddev=float(target.gray[indices[i]].std())) for i in order]
    best = candidates[0] if candidates else None
    second = candidates[1] if len(candidates) > 1 else None
    return dict(candidates=candidates, best=best, second=second,
        candidate_count=len(indices), tied_best_count=int(np.sum(scores == best['mae'])) if best else 0,
        margin=second['mae']-best['mae'] if second else None,
        query_gray_stddev=float(query.std()), query_gray_range=int(query.max())-int(query.min()),
        correspondence_status='candidate_only', unique_match_claimed=False)


def _anchor(sequence, pts):
    pts = _integer(pts, 'anchor PTS')
    indices = np.flatnonzero(sequence.pts == pts)
    if len(indices) != 1:
        raise ValueError('Matched anchor must be an actual PTS in its fingerprint sequence')
    index = int(indices[0])
    return index, Fraction(int(sequence.time_base_num[index]), int(sequence.time_base_den[index]))


def local_alignment_candidates(reference, target, *, anchor_reference_pts,
                               anchor_target_pts, search_radius_pts,
                               first_reference_pts=None,
                               stop_reference_pts_exclusive=None):
    """Independently match each reference image near an anchor-predicted clock.

    search_radius_pts is an inclusive +/- radius in target native PTS units.
    Integer/Fraction arithmetic preserves even large PTS. Anchor clock motion
    selects candidate windows only; images can select different offsets, ties
    or the same target exposure. No sequence propagation or bijection is used.
    """
    if reference.gray.shape[1:] != target.gray.shape[1:]:
        raise ValueError('Both sequences must use the same fingerprint dimensions')
    anchor_index, reference_tb = _anchor(reference, anchor_reference_pts)
    _, target_tb = _anchor(target, anchor_target_pts)
    radius = _integer(search_radius_pts, 'search_radius_pts')
    if radius < 0:
        raise ValueError('search_radius_pts must be nonnegative')
    selected = np.ones(len(reference.pts), dtype=bool)
    if first_reference_pts is not None:
        selected &= reference.pts >= _integer(first_reference_pts, 'first_reference_pts')
    if stop_reference_pts_exclusive is not None:
        selected &= reference.pts < _integer(stop_reference_pts_exclusive, 'stop_reference_pts_exclusive')
    # A raw target-PTS window requires a consistent native time base.
    if any(Fraction(int(n), int(d)) != target_tb
           for n, d in zip(target.time_base_num, target.time_base_den)):
        raise ValueError('Target frame time bases differ inside the local search sequence')
    anchor_seconds = Fraction(int(reference.pts[anchor_index]))*reference_tb
    rows = []
    for index in np.flatnonzero(selected):
        pts = int(reference.pts[index])
        tb = Fraction(int(reference.time_base_num[index]), int(reference.time_base_den[index]))
        center = Fraction(int(anchor_target_pts))+(Fraction(pts)*tb-anchor_seconds)/target_tb
        lower, upper = math.ceil(center-radius), math.floor(center+radius)
        ranking = mae_candidates(reference.gray[index], target,
            first_pts=lower, stop_pts_exclusive=upper+1)
        rows.append(dict(reference_pts=pts, reference_time_base_num=tb.numerator,
            reference_time_base_den=tb.denominator,
            search_center_target_pts=dict(numerator=center.numerator, denominator=center.denominator),
            search_target_pts_inclusive=[lower, upper], **ranking))
    return dict(schema='rinkspace.local_source_alignment_candidates.v1',
        anchor=dict(reference_pts=int(anchor_reference_pts), target_pts=int(anchor_target_pts),
                    reference_time_base=[reference_tb.numerator, reference_tb.denominator],
                    target_time_base=[target_tb.numerator, target_tb.denominator]),
        search_radius_target_pts=radius, rows=rows,
        reference_fingerprints=reference.metadata.get('json_path'),
        target_fingerprints=target.metadata.get('json_path'),
        method='independent_gray_MAE_per_actual_reference_exposure',
        fixed_offset_is_authoritative=False, unique_correspondence_claimed=False,
        complete_bijection_claimed=False)
