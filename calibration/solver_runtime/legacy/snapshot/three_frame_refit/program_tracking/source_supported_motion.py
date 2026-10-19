"""Read an auditable source-support sidecar without changing original LK UV.

Adds measured board-interior points to the conditional camera diagnostic.
It does not promote them to measured 3D positions or precise contact edges.
Original and added points from one native pair stay in one noise group.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from track_sequence import _distributed_indices


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_supported_motion(data, sidecar_path: Path, base_manifest_path: Path, expected_sha256: str):
    sidecar_path = Path(sidecar_path)
    if not expected_sha256 or digest(sidecar_path) != expected_sha256:
        raise ValueError('Source support sidecar differs from the explicitly frozen SHA256')
    report = json.loads(sidecar_path.read_text(encoding='utf8'))
    if (report.get('schema') != 'rinkspace-source-board-motion-support-v1'
            or report['source_sha256'] != data['manifest']['source_sha256']
            or report['candidate_scene_sha256'] != data['diag']['model_sha256']
            or report['base_strict_manifest_sha256'] != digest(base_manifest_path)):
        raise ValueError('Source board support is not for this frozen source/model/packet')
    raw_fallback = report.get('source_point_selection') == 'all_raw_good_hint0_for_parent_zero_pairs'
    parent_zero_pairs = None
    if raw_fallback:
        parent = data.get('supported_motion_metadata')
        if (parent is None or report.get('parent_source_support_sha256') != parent['sha256']
                or digest(parent['path']) != parent['sha256']):
            raise ValueError('Raw-point fallback requires its already verified parent sidecar')
        parent_report = json.loads(Path(parent['path']).read_text(encoding='utf8'))
        parent_zero_pairs = {(int(row['source_file_id']), int(row['source_pair_index']))
                             for row in parent_report['pairs'] if row['support_count'] == 0}
    detector = report['detector_module']
    if digest(detector['path']) != detector['sha256']:
        raise ValueError('Source support detector changed after evidence was recorded')
    from source_board_support import classify_points, _source_mask
    from render_result import PersonMasks, DEFAULT_MASKS, DEFAULT_PEOPLE
    if Path(detector['path']).resolve() != Path(__file__).with_name('source_board_support.py').resolve():
        raise ValueError('Unsupported source detector implementation')
    for edge in report['source_edge_cache'].values():
        if digest(edge['path']) != edge['sha256'] or edge['source_sha256'] != report['source_sha256']:
            raise ValueError('Source edge cache provenance changed')
    timeline_path = Path(report['source_shots_path'])
    if not timeline_path.is_absolute():
        timeline_path = Path(__file__).resolve().parents[2] / timeline_path
    if digest(timeline_path) != report['source_shots_sha256']:
        raise ValueError('Native source timeline changed')
    timeline = json.loads(timeline_path.read_text(encoding='utf8'))
    if (timeline['source_sha256_from_timeline'] != report['source_sha256']
            or Path(timeline['source_path']).resolve() != Path(report['source_path']).resolve()):
        raise ValueError('Source support decoded a different video/timeline')
    frame_rows = {int(row['pts']): row for row in report['frames']}
    if len(frame_rows) != len(report['frames']):
        raise ValueError('Duplicate source frame evidence')
    correction = report.get('source_support_correction_review')
    if correction is not None:
        module = report['correction_module']
        if (Path(module['path']).resolve() != Path(__file__).with_name(
                'source_board_support_correction.py').resolve()
                or digest(module['path']) != module['sha256']
                or digest(correction['path']) != correction['sha256']):
            raise ValueError('Source correction code or reviewed evidence changed')
        review = json.loads(Path(correction['path']).read_text(encoding='utf8'))
        if (review.get('schema') != 'rinkspace-source-board-band-review-v1'
                or review['parent_source_support_sha256'] != report['parent_source_support_sha256']
                or [row['pts'] for row in review['cases']] !=
                   [row['pts'] for row in correction['cases']]):
            raise ValueError('Source correction review differs from frozen parent/cases')
        for case, recorded in zip(review['cases'], correction['cases']):
            if (case['pts'] not in frame_rows
                    or Path(case['source_png_path']).resolve() !=
                       Path(recorded['source_png_path']).resolve()
                    or case['source_png_sha256'] != recorded['source_png_sha256']
                    or digest(case['source_png_path']) != case['source_png_sha256']):
                raise ValueError('Reviewed exact-source PNG changed')
    checked_frames = set()
    people = PersonMasks(DEFAULT_MASKS, DEFAULT_PEOPLE)
    evidence_packets = {}

    @lru_cache(maxsize=4)
    def frame_support(stamp):
        frame = frame_rows[stamp]
        path = str(Path(frame['source_support_evidence_path']).resolve())
        if path not in evidence_packets:
            if digest(path) != frame['source_support_evidence_sha256']:
                raise ValueError('Source support column evidence changed')
            with np.load(path, allow_pickle=False) as z:
                evidence_packets[path] = {k: z[k].copy() for k in z.files}
        packet = evidence_packets[path]
        index = int(frame['evidence_index'])
        offsets = packet['accepted_offsets']
        if (not 0 <= index < len(packet['pts']) or int(packet['pts'][index]) != stamp
                or len(offsets) != len(packet['pts'])+1 or offsets[0] != 0
                or offsets[-1] != len(packet['accepted_columns'])
                or np.any(np.diff(offsets) < 0)):
            raise ValueError('Source support column PTS/offsets are inconsistent')
        lo, hi = map(int, offsets[index:index+2])
        columns = packet['accepted_columns'][lo:hi]
        if (columns.shape != (frame['accepted_source_columns'], 5)
                or not np.isfinite(columns).all()):
            raise ValueError('Source support column payload is inconsistent')
        mask, status = people.get(stamp, require_sam2=True)
        if (status != frame['sam2_status']
                or hashlib.sha256(mask.tobytes()).hexdigest() != frame['sam2_mask_sha256']):
            raise ValueError('Source support person mask changed')
        checked_frames.add(stamp)
        return {'accepted': columns}, _source_mask(mask)
    source_rows = {int(row['id']): row for row in report['source_files']}
    if len(source_rows) != len(report['source_files']):
        raise ValueError('Duplicate source cache identity')
    loaded, result, identities = {}, [], set()
    for row in report['pairs']:
        file_id, pair_index = int(row['source_file_id']), int(row['source_pair_index'])
        identity = file_id, pair_index
        if identity in identities:
            raise ValueError('Duplicate source pair in support sidecar')
        identities.add(identity)
        if raw_fallback and identity not in parent_zero_pairs:
            raise ValueError('Raw-point fallback is restricted to parent pairs without source support')
        spec = source_rows[file_id]
        original_spec = data['manifest']['source_files'][file_id]
        for field in ('id', 'kind', 'shot_index', 'sha256', 'max_matches'):
            if spec[field] != original_spec[field]:
                raise ValueError(f'Source motion {field} differs from frozen packet')
        if (spec['kind'] != 'native'
                or Path(spec['path']).resolve() != Path(original_spec['path']).resolve()
                or int(row['shot_index']) != int(spec['shot_index'])
                or int(row['shot_index']) not in data['shot_indices']):
            raise ValueError('Source support must stay inside selected native camera shots')
        if file_id not in loaded:
            if digest(spec['path']) != spec['sha256']:
                raise ValueError('Original motion cache changed')
            with np.load(spec['path'], allow_pickle=False) as z:
                loaded[file_id] = {k: z[k].copy() for k in z.files}
        motion = loaded[file_id]
        if not 0 <= pair_index < len(motion['pair_pts']):
            raise ValueError('Source pair index out of range')
        a, b = map(int, motion['pair_pts'][pair_index])
        if ([a, b] != row['pts'] or b-a != 256
                or a not in data['pts'] or b not in data['pts']):
            raise ValueError('Source support PTS changed or crosses a camera cut')
        lo, hi = map(int, motion['offsets'][pair_index:pair_index+2])
        indices = np.asarray(row['source_point_indices'], dtype=np.int64)
        global_indices = np.asarray(row['source_global_point_indices'], dtype=np.int64)
        if (indices.ndim != 1 or len(set(indices.tolist())) != len(indices)
                or np.any(indices < 0) or np.any(indices >= hi-lo)
                or not np.array_equal(indices+lo, global_indices)
                or len(indices) != row['support_count']):
            raise ValueError('Source point identity/count is inconsistent')
        if not len(indices):
            continue
        xy0, xy1 = motion['xy0'][lo:hi], motion['xy1'][lo:hi]
        fb = motion['fb_error_px'][lo:hi]
        good = (motion['homography_inlier'][lo:hi].astype(bool)
                & (fb < 2.5) & np.isfinite(xy0).all(axis=1) & np.isfinite(xy1).all(axis=1))
        pool = np.flatnonzero(good)
        chosen = (pool if raw_fallback else
                  pool[_distributed_indices(xy0[pool], fb[pool], int(spec['max_matches']))])
        if (not set(indices.tolist()) <= set(chosen.tolist())
                or np.any(motion['region_hint'][global_indices] != 0)):
            raise ValueError('Added points must be permitted original unknown-region LK rows')
        for stamp, original_uv in ((a, xy0[indices]), (b, xy1[indices])):
            support, exclude = frame_support(stamp)
            if not np.all(classify_points(original_uv, support, exclude)):
                raise ValueError('Added original UV lacks recorded source board support or is masked')
        result.append(dict(source_file_id=file_id, source_pair_index=pair_index,
                           shot_index=int(row['shot_index']), pts=(a,b),
                           source_point_indices=indices.copy(),
                           xy0=xy0[indices].astype(np.float64),
                           xy1=xy1[indices].astype(np.float64)))
    metadata = dict(path=str(sidecar_path.resolve()), sha256=digest(sidecar_path),
                    pair_count=len(result), point_count=sum(len(r['xy0']) for r in result),
                    frames_checked=len(checked_frames), source_uv_changed=False,
                    support_columns_and_original_uv_rechecked=True,
                    raw_bgr_hash_policy='Recorded at exact source decode; not redundantly decoded by loader',
                    minimum_pair_points=1,
                    raw_point_fallback=raw_fallback,
                    parent_source_support_sha256=report.get('parent_source_support_sha256'),
                    correction_review_sha256=correction['sha256'] if correction else None,
                    correction_source_pngs_checked=len(correction['cases']) if correction else 0,
                    limitation='Source-supported board identity; geometry remains conditional. '
                               'Sparse pairs do not independently determine a full camera pose.')
    return result, metadata


def grouped_native_pairs(data, shot_index):
    """Merge per-pair rows before quality/Huber/correlated-noise construction."""
    packet = data['packet']
    grouped = {}
    for i in np.flatnonzero(packet['shot_index'] == shot_index):
        lo, hi = map(int, packet['offsets'][i:i+2])
        file_id = int(packet['source_file_id'][i])
        pair_index = int(packet['source_pair_index'][i])
        grouped[file_id, pair_index] = dict(
            source_file_id=file_id, source_pair_index=pair_index, packet_pair_index=int(i),
            pts=tuple(map(int, packet['pair_pts'][i])),
            source_point_indices=packet['source_point_index'][lo:hi].copy(),
            xy0=packet['xy0'][lo:hi].copy(), xy1=packet['xy1'][lo:hi].copy())
    for row in data.get('supported_motion', []):
        if row['shot_index'] != shot_index:
            continue
        key = row['source_file_id'], row['source_pair_index']
        if key not in grouped:
            grouped[key] = {**row, 'packet_pair_index': None}
            continue
        current = grouped[key]
        if (current['pts'] != row['pts']
                or set(current['source_point_indices']) & set(row['source_point_indices'])):
            raise ValueError('Source-supported points duplicate/relabel frozen observations')
        for field in ('source_point_indices', 'xy0', 'xy1'):
            current[field] = np.concatenate((current[field], row[field]))
    for group in grouped.values():
        # Distinct cache indices must not count the same two image pixels twice.
        pairs = np.column_stack((group['xy0'], group['xy1']))
        if len(np.unique(pairs, axis=0)) != len(pairs):
            raise ValueError('Repeated original UV pair inside one native noise group')
    return [grouped[key] for key in sorted(grouped)]
