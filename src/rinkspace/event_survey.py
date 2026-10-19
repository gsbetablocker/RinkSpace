"""Resumable, continuous event-video survey with explicit decoded PTS coverage.

Ranking scores are discovery aids, never calibrated ice masks or camera labels.
Every decoded frame is checked for a cut; each second receives richer features.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from fractions import Fraction

import av
import cv2
import numpy as np


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    pending.replace(path)


def file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def coverage_intervals(pts: np.ndarray, time_base: Fraction, expected_step: float) -> list[list[float]]:
    """Split actual decoded coverage at timestamp gaps; do not fill absent frames."""
    if not len(pts):
        return []
    seconds = pts.astype(np.float64) * float(time_base)
    if np.any(np.diff(seconds) <= 0):
        raise ValueError('Decoded PTS must strictly increase')
    breaks = np.flatnonzero(np.diff(seconds) > expected_step * 1.6) + 1
    return [[float(part[0]), float(part[-1] + expected_step)] for part in np.split(seconds, breaks)]


def cut_distance(previous: np.ndarray | None, current: np.ndarray) -> float:
    if previous is None:
        return 0.0
    # Low-resolution luma difference remains sensitive to a one-frame cut.
    return float(np.mean(cv2.absdiff(previous, current)) / 255)


def covers_interval(intervals: list[list[float]], start: float, end: float) -> bool:
    """A final timestamp alone does not establish uninterrupted decode coverage."""
    return bool(len(intervals) == 1 and intervals[0][0] <= start + 1e-6
                and intervals[0][1] >= end - 1e-6)


def visual_features(rgb: np.ndarray) -> dict:
    rgb = cv2.resize(rgb, (256, 144), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    chroma = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
    bright = (gray > 160) & (chroma < 42)
    edge = cv2.Canny(gray, 40, 110)
    lower = bright[48:]
    neutral_fraction = float(bright.mean())
    ice_candidate_fraction = float(lower.mean())
    edge_fraction = float((edge[48:] > 0).mean())
    # Board/ice boundary needs both bright ground and non-ground context.
    mixed_boundary = float(np.mean(np.any(bright[1:] != bright[:-1], axis=0)))
    dark_foreground = float(((gray[68:] < 95) & (~bright[68:])).mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    score = (1.8 * ice_candidate_fraction + .25 * mixed_boundary +
             .12 * min(sharpness / 80, 1) - .35 * dark_foreground)
    # Unverified scene-family descriptor; physical camera identity requires geometry.
    top = cv2.resize(rgb[:60], (16, 6), interpolation=cv2.INTER_AREA)
    return {'neutral_fraction': round(neutral_fraction, 5),
            'ice_candidate_fraction': round(ice_candidate_fraction, 5),
            'lower_edge_fraction': round(edge_fraction, 5),
            'boundary_candidate_score': round(mixed_boundary, 5),
            'dark_foreground_fraction': round(dark_foreground, 5),
            'sharpness': round(sharpness, 3), 'discovery_score': round(score, 5),
            'scene_descriptor_rgb': top.reshape(-1).tolist()}


def survey_chunk(source: dict, media_path: Path, output: Path, start: float, end: float,
                 source_hash: str, block_seconds: float = 30) -> dict:
    """Decode every frame in [start,end), recording all PTS and selected views."""
    chunk_dir = output / source['source_id'] / 'chunks' / f'{int(start):07d}'
    manifest_path = chunk_dir / 'manifest.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding='utf-8'))
        if (old.get('complete') and old.get('source_sha256') == source_hash
                and old.get('requested_interval_seconds') == [start, end]
                and covers_interval(old.get('decoded_intervals_seconds', []), start, end)):
            return old
    chunk_dir.mkdir(parents=True, exist_ok=True)
    pts, scores, cuts, seconds, candidates = [], [], [], [], []
    previous = None
    next_feature = start
    block_end = start + block_seconds
    best = None
    shot_id = f'{source["source_id"]}:{int(start)}:0'
    last_candidate_time = -math.inf
    last_candidate_shot = None

    def save_best(reason: str) -> None:
        nonlocal best, last_candidate_time, last_candidate_shot
        if best is None:
            return
        row, rgb = best
        best = None
        if row['shot_id'] == last_candidate_shot and row['source_time_seconds'] - last_candidate_time < .8:
            return
        if row['features']['ice_candidate_fraction'] < .3:
            return
        path = chunk_dir / f'pts_{row["source_pts"]:012d}.jpg'
        cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 94])
        row.update({'image_path': str(path.as_posix()), 'selection_reason': reason,
                    'camera_group_id': None, 'camera_group_status': 'unclassified',
                    'geometry_status': 'discovery_candidate_not_registered',
                    'geometry_eligibility': 'surveyed_withheld_for_geometry' if source['source_id'] == 'Rd66CneD9io' else 'requires_same_venue_visual_review',
                    'holdout_reason': 'Entire women FS broadcast reserved until CETIN EJV9lyZ7D28 duplicate/replay spans excluded' if source['source_id'] == 'Rd66CneD9io' else None,
                    'visual_review_status': 'algorithmically_screened_not_human_reviewed'})
        candidates.append(row)
        last_candidate_time = row['source_time_seconds']
        last_candidate_shot = row['shot_id']

    error = None
    with av.open(str(media_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        stream.codec_context.thread_count = 2
        rate = float(stream.average_rate)
        time_base = stream.time_base
        container.seek(int(start / time_base), stream=stream, backward=True)
        try:
            for frame in container.decode(stream):
                if frame.pts is None:
                    raise ValueError('Decoded frame has no PTS')
                time = float(frame.pts * frame.time_base)
                if time < start:
                    continue
                if time >= end:
                    break
                if frame.format.name in {'yuv420p', 'yuvj420p', 'nv12'}:
                    plane = frame.planes[0]
                    luma = np.frombuffer(plane, dtype=np.uint8).reshape(frame.height, plane.line_size)[:, :frame.width]
                    tiny = cv2.resize(luma, (80, 46), interpolation=cv2.INTER_AREA)
                    if frame.format.name != 'yuvj420p':
                        tiny = np.clip((tiny.astype(np.float32) - 16) * (255 / 219), 0, 255).astype(np.uint8)
                else:
                    tiny = frame.reformat(width=80, height=46, format='gray').to_ndarray()
                distance = cut_distance(previous, tiny)
                previous = tiny
                pts.append(frame.pts)
                scores.append(distance)
                if distance > .105:
                    save_best('best_one_second_view_before_cut')
                    shot_id = f'{source["source_id"]}:{frame.pts}'
                    cuts.append({'source_pts': frame.pts, 'source_time_seconds': time,
                                 'score': distance, 'status': 'cut_candidate_may_include_flash'})
                    next_feature = time
                if time >= next_feature:
                    rgb = frame.to_ndarray(format='rgb24')
                    features = visual_features(rgb)
                    row = {'source_id': source['source_id'], 'session_id': source['session_id'],
                           'background_epoch': source['background_epoch'],
                           'source_path': str(media_path.as_posix()), 'source_sha256': source_hash,
                           'source_pts': frame.pts, 'time_base': str(frame.time_base),
                           'source_time_seconds': time, 'original_time_offset_seconds': 0,
                           'image_size': [frame.width, frame.height], 'shot_id': shot_id,
                           'features': features}
                    seconds.append(row)
                    if best is None or features['discovery_score'] > best[0]['features']['discovery_score']:
                        best = (dict(row), rgb)
                    next_feature = time + 1
                if time >= block_end:
                    save_best('best_one_second_view_in_30_second_interval')
                    block_end = time + block_seconds
            save_best('last_view_in_chunk')
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
    pts_array = np.asarray(pts, dtype=np.int64)
    np.savez_compressed(chunk_dir / 'decoded_frame_pts_and_cut_scores.npz', source_pts=pts_array,
                        cut_score=np.asarray(scores, dtype=np.float32))
    # Descriptors are retained only for selected frames to keep the per-second index small.
    for row in seconds:
        row['features'] = {k: v for k, v in row['features'].items() if k != 'scene_descriptor_rgb'}
    dump(chunk_dir / 'one_second_features.json', seconds)
    dump(chunk_dir / 'cut_candidates.json', cuts)
    dump(chunk_dir / 'candidates.json', candidates)
    intervals = coverage_intervals(pts_array, time_base, 1 / rate)
    manifest = {'schema_version': 'rinkspace.continuous-event-survey-chunk.v1',
                'source_id': source['source_id'], 'source_sha256': source_hash,
                'source_path': str(media_path.as_posix()), 'source_time_base': str(time_base),
                'requested_interval_seconds': [start, end], 'decoded_intervals_seconds': intervals,
                'decoded_frame_count': len(pts), 'one_second_feature_count': len(seconds),
                'cut_candidate_count': len(cuts), 'candidate_count': len(candidates),
                'complete': error is None and covers_interval(intervals, start, end),
                'error': error, 'candidate_index': str((chunk_dir / 'candidates.json').as_posix()),
                'method': 'Every source frame decoded and tested at 80x46; 256x144 features at <=1 second intervals and after cut candidates; native-resolution proxy candidates every cut/30 seconds.',
                'limitations': ['Algorithmic screening is not exhaustive human viewing.',
                                'Cut candidates can include flashes and fast motion; subtle cuts can be missed.',
                                'Candidate extraction ranks one-second samples and detected cuts; sub-second views without a detected cut can be absent.',
                                'Bright neutral pixels are an ice-candidate ranking feature, not a validated ice mask.',
                                'Native proxy resolution is insufficient for high-precision ring measurement.']}
    dump(manifest_path, manifest)
    return manifest


def summarize_source(source: dict, output: Path, duration: float, source_hash: str) -> dict:
    source_dir = output / source['source_id']
    chunks = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((source_dir / 'chunks').glob('*/manifest.json'))]
    chunks = [row for row in chunks if row['source_sha256'] == source_hash]
    candidates = []
    for chunk in chunks:
        candidates.extend(json.loads(Path(chunk['candidate_index']).read_text(encoding='utf-8')))
    dump(source_dir / 'candidates.json', candidates)
    intervals = sorted(interval for chunk in chunks for interval in chunk['decoded_intervals_seconds'])
    merged = []
    for interval in intervals:
        if merged and interval[0] <= merged[-1][1] + .002:
            merged[-1][1] = max(merged[-1][1], interval[1])
        else:
            merged.append(list(interval))
    cursor, gaps = 0., []
    for start, end in merged:
        if start > cursor + .002:
            gaps.append([cursor, start])
        cursor = end
    if cursor < duration - .002:
        gaps.append([cursor, duration])
    summary = {'schema_version': 'rinkspace.continuous-event-survey.v1',
               'source_id': source['source_id'], 'title': source['title'],
               'source_sha256': source_hash, 'duration_seconds': duration,
               'decoded_intervals_seconds': merged, 'unprocessed_intervals_seconds': gaps,
               'decoded_frame_count': sum(row['decoded_frame_count'] for row in chunks),
               'one_second_feature_count': sum(row['one_second_feature_count'] for row in chunks),
               'cut_candidate_count': sum(row['cut_candidate_count'] for row in chunks),
               'candidate_count': len(candidates), 'continuous_decode_complete': not gaps,
               'all_useful_views_found': False,
               'geometry_eligibility': 'surveyed_withheld_for_geometry' if source['source_id'] == 'Rd66CneD9io' else 'candidate_level_venue_and_geometry_review_required',
               'candidate_index': str((source_dir / 'candidates.json').as_posix()),
               'human_review_scope': 'Only explicitly reviewed candidate images; per-source candidate_review.json if present.',
               'chunks': chunks}
    dump(source_dir / 'summary.json', summary)
    return summary
