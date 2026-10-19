"""Opt-in physical15 motion factors for explicitly reviewed original LK rows.

The source-only hint-0 sidecar is a candidate inventory, never an admission
list. Admitted rows need an explicit agent-reviewed source class range or
individual review, exact raw LK indices, and an explicit physical surface
identity. A reviewed class range may expand original candidate indices without
requiring a person to inspect each LK pixel separately.
No pose-guided source edge is interpreted as a board rim or a world height.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2
import numpy as np

from continuous_brown_adapter import adapt_factor
from continuous_brown_board_pair import ContinuousBrownBoardPair
from continuous_source_motion_requalification import STATUS, sha256
from correlated_motion import wrap_correlated_pair
from observation_weighting import ObservationQuality
from projection import from_state, project_central_valid
from reference_board_pair import BoardIntersectionError
from robust_observations import pointwise_huber
from surface_pair import SurfacePairConstraint
from track_sequence import SurfacePlanes, _first_hit_planes


REVIEW_SCHEMA = 'rinkspace.main-hint0-reviewed-motion.v1'
REVIEW_AUTHORITIES = {
    'agent_original_frame_class_review',
    'human_original_frame_identity_review',
}
IDENTITIES = {
    'printed_board_face': STATUS['candidate_ad_print'],
    'painted_ice': STATUS['candidate_painted_ice'],
}
STRAIGHT_SIDES = ('upper_long_side', 'lower_long_side',
                  'left_short_side', 'right_short_side')
CONNECTED_BOARD_SIDES = {
    # The original same-event scene identifies these two consecutive parts.
    # The reviewed far-board ad1/ad2 source identity spans their connection.
    'upper_long_side_with_upper_right_rounding':
        ('upper_right', 'upper_long_side'),
}
BOARD_SIDES = STRAIGHT_SIDES + tuple(CONNECTED_BOARD_SIDES)


def _load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding='utf8'))


def _rows_digest(rows: list[dict]) -> str:
    """Seal original identities and numeric UV between review load and attach."""
    payload = [{
        key: (np.asarray(row[key], float).tolist() if key in ('xy0', 'xy1')
              else row[key])
        for key in ('shot_index', 'pair_index', 'source_pts', 'target_pts',
                    'original_local_index', 'original_global_index',
                    'identity', 'physical_side', 'xy0', 'xy1',
                    'source_frame_raw_bgr_sha256', 'preview_manifest_sha256',
                    'reviewer', 'review_note', 'review_mode',
                    'review_group_index')
    } for row in rows]
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def _review_preview(row: dict, source_pts: int, cache: dict, *,
                    sidecar_sha256: str, source_sha256: str) -> dict:
    path = Path(row['preview_manifest_path']).resolve()
    expected_hash = row['preview_manifest_sha256']
    if path not in cache:
        if not path.is_file() or sha256(path) != expected_hash:
            raise ValueError('Reviewed full-frame preview manifest hash changed')
        report = _load_json(path)
        if report.get('schema') not in (
                'rinkspace.main-tail-hint0-fullres-review.v1',
                'rinkspace.main-hint0-source-preview.v1'):
            raise ValueError('Review needs an original-frame source-only preview')
        if (report.get('sidecar_manifest_sha256',
                       report.get('source_sidecar_manifest_sha256'))
                != sidecar_sha256):
            raise ValueError('Source preview belongs to another candidate sidecar')
        if (report.get('source_video_sha256', source_sha256)
                != source_sha256):
            raise ValueError('Source preview belongs to another video')
        preview_rows = report['rows']
        by_pts = {int(item['source_pts']): item for item in preview_rows}
        if len(by_pts) != len(preview_rows):
            raise ValueError('Duplicate preview source PTS')
        cache[path] = report, by_pts, expected_hash
    report, by_pts, cached_hash = cache[path]
    if cached_hash != expected_hash:
        raise ValueError('Review mixed source-preview manifests at one path')
    if source_pts not in by_pts:
        raise ValueError('Reviewed PTS missing from its source preview')
    item = by_pts[source_pts]
    overlay = (item.get('fullres_overlay') or item.get('overlay'))
    overlay_hash = (item.get('fullres_overlay_sha256')
                    or item.get('overlay_sha256'))
    if (overlay is None or overlay_hash != row['preview_overlay_sha256']
            or not Path(overlay).is_file() or sha256(overlay) != overlay_hash
            or item['source_frame_raw_bgr_sha256']
            != row['source_frame_raw_bgr_sha256']):
        raise ValueError('Reviewed original-frame image provenance changed')
    return item


def _straight_side_panels(scene: dict, name: str) -> np.ndarray:
    """Derive a reviewed named physical straight side from existing scene XYZ."""
    if name not in STRAIGHT_SIDES:
        raise ValueError('Only explicitly reviewed straight physical sides are supported')
    bottom = np.asarray(scene['board_surface']['bottom_edge_xyz'], float)
    xy0 = bottom[:, :2]
    xy1 = np.roll(xy0, -1, axis=0)
    axis, extreme = ((1, np.max) if name == 'upper_long_side' else
                     (1, np.min) if name == 'lower_long_side' else
                     (0, np.min) if name == 'left_short_side' else
                     (0, np.max))
    value = extreme(xy0[:, axis])
    mask = ((np.abs(xy0[:, axis]-value) <= 1e-7)
            & (np.abs(xy1[:, axis]-value) <= 1e-7)
            & (np.linalg.norm(xy1-xy0, axis=1) > 1e-6))
    panels = np.flatnonzero(mask).astype(np.int32)
    if len(panels) < 2:
        raise ValueError(f'Physical straight side {name} has no verified panel arc')
    # A side is one contiguous scene interval, possibly crossing index zero.
    gaps = np.count_nonzero(~mask & np.roll(mask, 1))
    if gaps != 1:
        raise ValueError(f'Physical straight side {name} is not one connected arc')
    return panels


def _reviewed_board_panels(scene: dict, name: str) -> np.ndarray:
    """Bind reviewed board identity to scene parts, never a fitted hit panel."""
    if name in STRAIGHT_SIDES:
        return _straight_side_panels(scene, name)
    part_names = CONNECTED_BOARD_SIDES.get(name)
    if part_names is None:
        raise ValueError('Unknown reviewed physical board side')
    bottom = np.asarray(scene['board_surface']['bottom_edge_xyz'], float)
    ice = np.asarray(scene['ice_boundary']['points_xyz'], float)
    if (bottom.ndim != 2 or bottom.shape[1] != 3
            or ice.shape != bottom.shape
            or not np.allclose(bottom, ice, rtol=0., atol=1e-8)):
        raise ValueError('Board and named ice-boundary parts disagree')
    parts = scene['ice_boundary']['parts_counterclockwise']
    by_id = {part['id']: part for part in parts}
    if len(by_id) != len(parts) or any(part not in by_id for part in part_names):
        raise ValueError('Reviewed rounded board part is absent or duplicated')
    selected = []
    for expected, part_name in zip(('corner', 'straight'), part_names):
        part = by_id[part_name]
        start, count = int(part['first_vertex']), int(part['vertex_count'])
        if (part['kind'] != expected or count < 2
                or start < 0 or start+count > len(bottom)):
            raise ValueError('Reviewed rounded board scene part is malformed')
        selected.append(np.arange(start, start+count, dtype=np.int32))
    if int(selected[0][-1])+1 != int(selected[1][0]):
        raise ValueError('Reviewed corner and straight board are not connected')
    if not np.array_equal(selected[1], _straight_side_panels(scene, 'upper_long_side')):
        raise ValueError('Reviewed named upper straight differs from physical scene')
    panels = np.concatenate(selected)
    successor = np.roll(bottom[:, :2], -1, axis=0)
    if np.any(np.linalg.norm(successor[panels]-bottom[panels, :2], axis=1) <= 1e-6):
        raise ValueError('Reviewed rounded board contains a degenerate panel')
    return panels


def load_reviewed_main_motion(data: dict, sidecar_manifest_path: Path,
                              review_manifest_path: Path, *,
                              expected_sidecar_sha256: str,
                              expected_review_sha256: str) -> tuple[list[dict], dict]:
    """Load only separately reviewed raw LK rows; absence of review is fatal.

    The review manifest may name precise reviewed_rows or reviewed_groups.
    A group selects a reviewed original pair-index range, one source-only
    candidate class, and a verified named physical board extent for print.
    Its source-image evidence supports reviewing the class; the sidecar then
    expands exact original raw indices. No programmatic rank alone admits a
    factor, and no review is generated automatically by this function.
    """
    sidecar_path, review_path = map(Path, (sidecar_manifest_path,
                                          review_manifest_path))
    if (not expected_sidecar_sha256 or not expected_review_sha256
            or not sidecar_path.is_file() or not review_path.is_file()
            or sha256(sidecar_path) != expected_sidecar_sha256
            or sha256(review_path) != expected_review_sha256):
        raise ValueError('Explicit frozen source and reviewer-record hashes required')
    sidecar, review = _load_json(sidecar_path), _load_json(review_path)
    source_sha = data['diag']['source_sha256']
    scene_sha = data['diag']['model_sha256']
    if (sidecar.get('schema') != 'rinkspace.main-hint0-source-review-pending.v1'
            or sidecar['source_sha256'] != source_sha
            or review.get('schema') != REVIEW_SCHEMA
            or review.get('review_authority') not in REVIEW_AUTHORITIES
            or review.get('sidecar_manifest_sha256') != expected_sidecar_sha256
            or review.get('source_sha256') != source_sha
            or review.get('scene_sha256') != scene_sha):
        raise ValueError('Review does not bind this original source, scene and sidecar')
    reviewed = review.get('reviewed_rows', [])
    groups = review.get('reviewed_groups', [])
    if (not isinstance(reviewed, list) or not isinstance(groups, list)
            or not reviewed and not groups):
        raise ValueError('Explicit source-class groups or exact reviewed rows required')
    source_shots = _load_json(Path(__file__).with_name('source_shots.json'))
    if source_shots['source_sha256_from_timeline'] != source_sha:
        raise ValueError('Original shot timeline source hash changed')
    motion_cache, sidecar_cache, preview_cache = {}, {}, {}
    reviewed = list(reviewed)
    for group_index, group in enumerate(groups):
        shot = int(group['shot_index'])
        identity = group['identity']
        side = group.get('physical_side')
        first, end = (int(group['pair_index_start']),
                      int(group['pair_index_end_exclusive']))
        if (group.get('selection') != 'source_sidecar_candidate_class'
                or identity not in IDENTITIES
                or (identity == 'printed_board_face' and side not in BOARD_SIDES)
                or (identity == 'painted_ice' and side is not None)
                or not str(group.get('reviewer', '')).strip()
                or not str(group.get('review_note', '')).strip()
                or shot not in data['shot_indices']
                or source_shots['shots'][shot]['camera_id'] != 'main_long_side'):
            raise ValueError('Source-class group lacks explicit reviewed identity/side')
        report = sidecar['shots'].get(str(shot))
        if report is None:
            raise ValueError('Source-class group shot missing from frozen sidecar')
        path = Path(report['motion_path'])
        packet = Path(report['sidecar'])
        if sha256(path) != report['motion_sha256'] or sha256(packet) != report['sidecar_sha256']:
            raise ValueError('Original LK or sidecar changed during group expansion')
        with np.load(path, allow_pickle=False) as z:
            pair_pts = z['pair_pts'].copy()
            offsets = z['offsets'].copy()
        with np.load(packet, allow_pickle=False) as z:
            if (str(z['motion_sha256']) != report['motion_sha256']
                    or not np.array_equal(z['original_pair_pts'], pair_pts)
                    or not np.array_equal(z['original_offsets'], offsets)):
                raise ValueError('Group sidecar original index identity changed')
            status = z['status_code_by_original_raw_index'].copy()
        if not 0 <= first < end <= len(pair_pts):
            raise ValueError('Reviewed source-class range exceeds one actual shot')
        evidence = group.get('evidence')
        if not isinstance(evidence, list) or not evidence:
            raise ValueError('Source-class group needs original-frame review evidence')
        evidence_by_pts = {}
        for item in evidence:
            stamp = int(item['source_pts'])
            if stamp in evidence_by_pts or stamp not in set(map(int, pair_pts[first:end, 0])):
                raise ValueError('Review evidence PTS missing or duplicated in group')
            preview = _review_preview(
                item, stamp, preview_cache,
                sidecar_sha256=expected_sidecar_sha256,
                source_sha256=source_sha)
            if int(preview['target_pts']) != int(pair_pts[
                    int(np.searchsorted(pair_pts[:, 0], stamp)), 1]):
                raise ValueError('Review image differs from original source pair')
            evidence_by_pts[stamp] = item
        if not evidence_by_pts:
            raise ValueError('No source-original evidence for reviewed class')
        selected_count = 0
        for pair_index in range(first, end):
            a, b = map(int, pair_pts[pair_index])
            lo, hi = map(int, offsets[pair_index:pair_index + 2])
            indices = np.flatnonzero(status[lo:hi] == IDENTITIES[identity])
            selected_count += len(indices)
            for local in indices:
                sample = evidence_by_pts.get(a)
                reviewed.append({
                    'shot_index': shot, 'pair_index': pair_index,
                    'source_pts': a, 'target_pts': b,
                    'original_local_index': int(local),
                    'identity': identity, 'physical_side': side,
                    'reviewer': group['reviewer'], 'review_note': group['review_note'],
                    'review_mode': 'agent_reviewed_source_class_range',
                    'review_group_index': group_index,
                    # Exact source pixels in unsampled frames are bound by
                    # source-video SHA, native PTS and original raw motion SHA.
                    'source_frame_raw_bgr_sha256': (
                        sample['source_frame_raw_bgr_sha256'] if sample else None),
                    'preview_manifest_sha256': (
                        sample['preview_manifest_sha256'] if sample else None),
                    '_group_evidence': True,
                })
        if selected_count == 0:
            raise ValueError('Reviewed source class selected no original candidates')
    output, unique = [], set()
    for record in reviewed:
        shot = int(record['shot_index'])
        a, b = int(record['source_pts']), int(record['target_pts'])
        pair_index = int(record['pair_index'])
        local = int(record['original_local_index'])
        identity = record['identity']
        side = record.get('physical_side')
        if (identity not in IDENTITIES
                or (identity == 'printed_board_face' and side not in BOARD_SIDES)
                or (identity == 'painted_ice' and side is not None)
                or not str(record.get('reviewer', '')).strip()
                or not str(record.get('review_note', '')).strip()
                or shot not in data['shot_indices']
                or source_shots['shots'][shot]['camera_id'] != 'main_long_side'
                or a not in data['pts'] or b not in data['pts']
                or a not in data['captured']['templates']
                or b not in data['captured']['templates']):
            raise ValueError('Review row lacks a permitted explicit surface identity')
        identity_key = shot, pair_index, local
        if identity_key in unique:
            raise ValueError('One raw LK row cannot be duplicated or cross-labelled')
        unique.add(identity_key)
        shot_report = sidecar['shots'].get(str(shot))
        if shot_report is None:
            raise ValueError('Reviewed shot absent from candidate source sidecar')
        if shot not in motion_cache:
            motion_path = Path(shot_report['motion_path'])
            sidecar_file = Path(shot_report['sidecar'])
            if (sha256(motion_path) != shot_report['motion_sha256']
                    or sha256(sidecar_file) != shot_report['sidecar_sha256']):
                raise ValueError('Original LK cache or candidate packet hash changed')
            with np.load(motion_path, allow_pickle=False) as z:
                motion_cache[shot] = {key: z[key].copy() for key in (
                    'pair_pts', 'offsets', 'xy0', 'xy1', 'region_hint',
                    'homography_inlier', 'fb_error_px')}
            with np.load(sidecar_file, allow_pickle=False) as z:
                if (str(z['motion_sha256']) != shot_report['motion_sha256']
                        or not np.array_equal(z['original_pair_pts'],
                                              motion_cache[shot]['pair_pts'])
                        or not np.array_equal(z['original_offsets'],
                                              motion_cache[shot]['offsets'])):
                    raise ValueError('Candidate sidecar lost original LK indexing')
                sidecar_cache[shot] = z['status_code_by_original_raw_index'].copy()
        motion = motion_cache[shot]
        if (not 0 <= pair_index < len(motion['pair_pts'])
                or tuple(map(int, motion['pair_pts'][pair_index])) != (a, b)
                or b-a != 256):
            raise ValueError('Reviewed pair differs from original adjacent exposure')
        lo, hi = map(int, motion['offsets'][pair_index:pair_index + 2])
        global_index = lo + local
        if (not 0 <= local < hi-lo
                or int(sidecar_cache[shot][global_index]) != IDENTITIES[identity]
                or int(motion['region_hint'][global_index]) != 0
                or not motion['homography_inlier'][global_index]
                or not np.isfinite(motion['fb_error_px'][global_index])
                or motion['fb_error_px'][global_index] >= 2.5):
            raise ValueError('Human review did not select a valid original candidate')
        if not record.get('_group_evidence'):
            preview = _review_preview(
                record, a, preview_cache,
                sidecar_sha256=expected_sidecar_sha256,
                source_sha256=source_sha)
            if int(preview['target_pts']) != b:
                raise ValueError('Reviewed preview belongs to a different pair')
            # The sparse seven-frame preview lacks a per-row list. Its frozen
            # sidecar above remains the exact row-index authority.
            candidate_field = ('candidate_ad_print_local_indices'
                               if identity == 'printed_board_face'
                               else 'candidate_painted_ice_local_indices')
            if (candidate_field in preview
                    and local not in preview[candidate_field]):
                raise ValueError('Reviewed row absent from full-resolution preview')
        output.append({
            'shot_index': shot, 'pair_index': pair_index,
            'source_pts': a, 'target_pts': b,
            'original_local_index': local,
            'original_global_index': global_index,
            'identity': identity, 'physical_side': side,
            'xy0': motion['xy0'][global_index].astype(np.float64).copy(),
            'xy1': motion['xy1'][global_index].astype(np.float64).copy(),
            'source_frame_raw_bgr_sha256': record['source_frame_raw_bgr_sha256'],
            'preview_manifest_sha256': record['preview_manifest_sha256'],
            'reviewer': record['reviewer'], 'review_note': record['review_note'],
            'review_mode': record.get('review_mode', 'explicit_original_row_review'),
            'review_group_index': record.get('review_group_index'),
        })
    output.sort(key=lambda row: (row['shot_index'], row['pair_index'],
                                 row['original_local_index']))
    return output, {
        'schema': REVIEW_SCHEMA, 'review_manifest': str(review_path.resolve()),
        'review_manifest_sha256': expected_review_sha256,
        'candidate_manifest': str(sidecar_path.resolve()),
        'candidate_manifest_sha256': expected_sidecar_sha256,
        'source_sha256': source_sha, 'scene_sha256': scene_sha,
        'reviewed_original_row_count': len(output),
        'reviewed_group_count': len(groups),
        'review_authority': review['review_authority'],
        'validated_rows_sha256': _rows_digest(output),
        'ad_print_rows': sum(r['identity'] == 'printed_board_face' for r in output),
        'ice_paint_rows': sum(r['identity'] == 'painted_ice' for r in output),
        'unreviewed_candidate_factor_count': 0,
    }


class ReviewedIceSurfacePair(SurfacePairConstraint):
    """Dynamic C/R/K/Brown plane hit with current-state ice-first-hit guard."""

    @classmethod
    def create(cls, xy0: np.ndarray, xy1: np.ndarray, scene: dict,
               template: np.ndarray) -> 'ReviewedIceSurfacePair':
        n = len(xy0)
        result = cls(np.asarray(xy0, float), np.asarray(xy1, float),
                     np.zeros((n, 3)), np.tile([0., 0., 1.], (n, 1)),
                     np.ones(n, np.uint8), np.asarray(template, float).copy(),
                     False)
        result._reviewed_planes = SurfacePlanes.from_scene(scene)
        return result

    def _require_ice_first_hit(self, state: np.ndarray,
                               pixels: np.ndarray, stage: str) -> None:
        camera = from_state(state, self.camera_template)
        try:
            _, _, valid = _first_hit_planes(
                self._reviewed_planes, camera, pixels,
                np.ones(len(pixels), np.uint8))
        except (cv2.error, ValueError, FloatingPointError) as error:
            raise BoardIntersectionError('reviewed_ice_first_hit_' + stage,
                                         np.arange(len(pixels))) from error
        if not np.all(valid):
            raise BoardIntersectionError('reviewed_ice_first_hit_' + stage,
                                         np.flatnonzero(~valid))

    def projected(self, states: list[np.ndarray], jacobian: bool = False):
        self._require_ice_first_hit(states[0], self.xy0, 'source')
        result = super().projected(states, jacobian)
        self._require_ice_first_hit(states[1], result[0], 'target')
        return result

    def jacobians(self, states: list[np.ndarray]) -> list[np.ndarray]:
        # SurfacePairConstraint's analytic path bypasses projected(), so
        # validate the same two physical ice-first-hit branches explicitly.
        self.projected(states)
        return super().jacobians(states)


class ReviewedCentralBoardPair(ContinuousBrownBoardPair):
    """Existing global board first-hit plus current source/target Brown domain.

    NativeBoardPair checks panel and occlusion, but its project_fast target
    branch alone can produce a noncentral Brown ghost. This opt-in subclass
    retains the exact raw UV and first-hit factor while rejecting that branch.
    """

    def _project_pixels(self, states, source_pixels, with_jacobian=False):
        projected = super()._project_pixels(states, source_pixels,
                                            with_jacobian=with_jacobian)
        source, target = states
        hits = self.source_hits(source, source_pixels)
        self._require(hits)
        source_camera = from_state(source, self.camera0_template)
        target_camera = from_state(target, self.camera1_template)
        _, _, source_central = project_central_valid(source_camera, hits.xyz)
        _, _, target_central = project_central_valid(target_camera, hits.xyz)
        if not np.all(source_central):
            raise BoardIntersectionError('reviewed_board_source_noncentral',
                                         np.flatnonzero(~source_central))
        if not np.all(target_central):
            raise BoardIntersectionError('reviewed_board_target_noncentral',
                                         np.flatnonzero(~target_central))
        return projected

    def initial_validity(self, states):
        valid, reasons = super().initial_validity(states)
        reasons = list(reasons)
        if not np.any(valid):
            return valid, tuple(reasons)
        hits = self.source_hits(states[0])
        selected = np.flatnonzero(valid)
        c0 = from_state(states[0], self.camera0_template)
        c1 = from_state(states[1], self.camera1_template)
        _, _, s = project_central_valid(c0, hits.xyz[selected])
        _, _, t = project_central_valid(c1, hits.xyz[selected])
        for local, row in enumerate(selected):
            if not s[local]:
                reasons[int(row)] = 'reviewed_board_source_noncentral'
            elif not t[local]:
                reasons[int(row)] = 'reviewed_board_target_noncentral'
        valid = np.asarray([reason == 'ok' for reason in reasons], bool)
        return valid, tuple(reasons)


def build_reviewed_motion_pairs(data: dict, rows: list[dict]) -> list[tuple]:
    """Make factors only from a validated loader result, grouped by raw pair."""
    if not rows:
        return []
    grouped: dict[tuple, list] = defaultdict(list)
    seen = set()
    for row in rows:
        required = ('shot_index', 'pair_index', 'source_pts', 'target_pts',
                    'original_local_index', 'original_global_index',
                    'identity', 'physical_side', 'xy0', 'xy1',
                    'source_frame_raw_bgr_sha256', 'preview_manifest_sha256',
                    'reviewer', 'review_note')
        if any(key not in row for key in required):
            raise ValueError('Only fully validated reviewed-row records are allowed')
        unique = row['shot_index'], row['pair_index'], row['original_local_index']
        if unique in seen or not row['reviewer'] or not row['review_note']:
            raise ValueError('Duplicate or unreviewed row')
        seen.add(unique)
        key = (row['source_pts'], row['target_pts'], row['identity'],
               row['physical_side'])
        grouped[key].append(row)
    scene = data['scene']
    templates = data['captured']['templates']
    states = data['states']
    result = []
    for (a, b, identity, side), group in sorted(grouped.items(),
                                                 key=lambda item: str(item[0])):
        group.sort(key=lambda item: item['original_local_index'])
        uv0 = np.stack([r['xy0'] for r in group])
        uv1 = np.stack([r['xy1'] for r in group])
        template0 = from_state(states[a], templates[a])
        template1 = from_state(states[b], templates[b])
        if identity == 'printed_board_face':
            panels = _reviewed_board_panels(scene, side)
            pair = ReviewedCentralBoardPair(
                uv0, uv1, scene, template0, template1,
                allowed_panels=panels)
        else:
            pair = ReviewedIceSurfacePair.create(uv0, uv1, scene, template0)
        result.append((int(a), int(b), pair, {
            'identity': identity, 'physical_side': side,
            'shot_index': group[0]['shot_index'],
            'original_pair_index': group[0]['pair_index'],
            'original_local_indices': [int(r['original_local_index'])
                                       for r in group],
            'original_global_indices': [int(r['original_global_index'])
                                        for r in group],
        }))
    return result


def attach_reviewed_main_motion_factors(graph, data: dict, model,
                                        initial: dict, rows: list[dict], *,
                                        reviewed_metadata: dict) -> dict:
    """Attach reviewed original UV motion to the *same* free physical15 model.

    This deliberately does not admit any sidecar candidate by itself. It
    preserves the existing source-quality -> Huber -> correlated-noise order.
    No source pixel is dropped on a trial domain failure.
    """
    if (reviewed_metadata.get('schema') != REVIEW_SCHEMA
            or reviewed_metadata.get('reviewed_original_row_count') != len(rows)
            or reviewed_metadata.get('validated_rows_sha256') != _rows_digest(rows)
            or not reviewed_metadata.get('review_manifest_sha256')):
        raise ValueError('Explicit validated source-class review metadata required')
    if (not Path(reviewed_metadata['review_manifest']).is_file()
            or sha256(Path(reviewed_metadata['review_manifest']))
            != reviewed_metadata['review_manifest_sha256']):
        raise ValueError('Reviewed source identity file changed before attachment')
    pairs = build_reviewed_motion_pairs(data, rows)
    quality: ObservationQuality = data['captured']['source_quality']
    active = set(map(int, initial))
    sigma = float(data['config'].get('native_pair_sigma_px', 2.))
    similarity = data['config'].get('native_pair_similarity_noise_px')
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('Existing source motion sigma must be positive')
    installed = []
    for a, b, pair, identity in pairs:
        if (a not in model.frames or b not in model.frames
                or len(model.physical_state(a, initial)) != 15
                or len(model.physical_state(b, initial)) != 15):
            raise ValueError('Reviewed motion needs two dynamic physical15 cameras')
        raw, jac = ((pair._raw_residual, pair._raw_jacobians)
                    if isinstance(pair, ContinuousBrownBoardPair)
                    else (pair.residual, pair.jacobians))
        kind = ('reviewed_print_motion' if identity['identity']
                == 'printed_board_face' else 'reviewed_ice_motion')
        raw, jac = quality.wrap(raw, jac, quality.pair_scales(
            a, b, pair.xy0, pair.xy1, kind))
        raw, jac = pointwise_huber(raw, jac, group_size=2, delta=2*sigma)
        factor_sigma = sigma
        if similarity is not None:
            raw, jac, _ = wrap_correlated_pair(
                raw, jac, pair.xy1, sigma_point_px=sigma,
                sigma_modes_px=similarity)
            factor_sigma = 1.
        adapted = adapt_factor(
            (a, b), raw, jac,
            {'residual_dim': 2*len(pair.xy0), 'sigma': factor_sigma},
            active, data['states'], model)
        if adapted is None:
            raise ValueError('Reviewed motion factor has no active camera dependency')
        keys, residual, derivative, options = adapted
        source_keys = (int(a), int(b))
        def value(q, callback=residual, family=kind, pair_keys=source_keys):
            try:
                return callback(q)
            except BoardIntersectionError as error:
                error.evaluation_stage = 'residual'
                error.factor_family = family
                error.source_keys = pair_keys
                raise
        def jacobian(q, callback=derivative, family=kind, pair_keys=source_keys):
            try:
                return callback(q)
            except BoardIntersectionError as error:
                error.evaluation_stage = 'jacobian'
                error.factor_family = family
                error.source_keys = pair_keys
                raise
        graph.add_observation(keys, value, jac_fn=jacobian, **options)
        installed.append({**identity, 'source_pts': a, 'target_pts': b,
                          'pixel_count': len(pair.xy0), 'factor_keys': list(keys)})
    return {**reviewed_metadata,
            'attached_factor_count': len(installed),
            'attached_original_row_count': sum(r['pixel_count'] for r in installed),
            'dynamic_physical_camera_dim': 15,
            'source_quality_policy': 'same_existing_quality_huber_correlated_order',
            'geometry_limit': ('Printed-face factors use an explicit reviewed '
                               'scene board extent; painted-ice factors '
                               'require current-state first ice hit at both '
                               'exposures. Neither uses fixed-centre rays.'),
            'factors': installed}
