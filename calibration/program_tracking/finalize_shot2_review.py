"""Assemble selected official-shot observations for editable human review."""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2
import numpy as np
from calibration.program_tracking.extract_selected_frame_review import make_binding
from calibration.program_tracking.validate_opening_frame_batch import assemble
from calibration.program_tracking.framewise_model import evaluate_rows
from calibration.program_tracking.extract_shot2_ice import contact_domain_input_sha256
from calibration.program_tracking.render_opening_frame_batch import page_html, timestamp
from calibration.program_tracking.render_single_frame_source_review import native_overlay, source_segments, display_segments, LAYERS

DATA = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp/results/shot2_review_20261014')
OUT = ROOT / 'events/ankara_2026/programmes/okada_mei_sp/source_datasets/shot2_review_20261014'
MODEL = ROOT / 'events/ankara_2026/models/shot2_8a28_20261014'
SOURCE_MEDIA = DATA
EDITOR_URL = 'http://127.0.0.1:48174/'
PUBLISH_ENTRY = True
EXPECTED_SHOT = None

from rinkspace.storage import AssetStore

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def reference(path):
    return dict(path=str(path), sha256=sha(path))


def verify_candidate_source(candidate, record, kind):
    """Reject stale reads before current source metadata is copied over them."""
    source = candidate['source']
    current = record['source']
    for key in ('native_pts', 'time_base', 'programme_frame_index_1based',
                'source_video_sha256', 'png_sha256', 'decoded_BGR_sha256', 'mask_sha256'):
        if source.get(key) != current.get(key):
            raise ValueError(f'Stale {kind} candidate source: {record["index"]}/{key}')
    if sha(record['image_path']) != source['png_sha256'] or sha(record['mask_path']) != source['mask_sha256']:
        raise ValueError(f'Current {kind} source PNG or mask changed: {record["index"]}')
    ref = candidate.get('source_binding', source.get('source_binding'))
    if ref:
        expected = record.get('source_binding')
        if ref != expected or sha(ref['path']) != ref['sha256']:
            raise ValueError(f'Stale {kind} exact source binding: {record["index"]}')


def verify_ordered_native_contour(segment, frame_index, feature_id):
    """A clipped source pixel contour cannot bridge an absent material span."""
    if (segment.get('source_native_trace_topology') != 'ordered_unit_pixel_contour'
        and segment.get('measurement_origin') != 'current_native_whole_connected_material_boundary'):
        return
    trace = np.asarray(segment.get('source_native_readout_uv', []), float)
    if trace.ndim != 2 or trace.shape[1] != 2 or len(trace) < 2 or not np.isfinite(trace).all():
        raise ValueError(f'Invalid ordered native contour: {frame_index}/{feature_id}')
    if not np.allclose(trace, np.rint(trace), atol=1e-6, rtol=0):
        raise ValueError(f'Ordered native contour is not an actual pixel path: {frame_index}/{feature_id}')
    neighbors = np.diff(trace, axis=0)
    if segment.get('closed'):
        neighbors = np.vstack((neighbors, trace[0] - trace[-1]))
    if (np.linalg.norm(neighbors, axis=1) > np.sqrt(2) + 1e-6).any():
        raise ValueError(f'Disconnected native contour was joined: {frame_index}/{feature_id}')

def write(path, value):
    path = Path(path).resolve()
    if path.is_relative_to(ROOT):
        if len(str(path).encode('utf-16-le')) // 2 >= 260 or any(len(p.encode('utf-16-le')) // 2 > 255 for p in path.parts):
            raise ValueError('OneDrive path too long: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def source_board_material_band(features):
    """Use current measured board edges only as an exclusion guard."""
    axis = np.arange(1920)
    bounds = []
    for role in ('top', 'contact'):
        values = np.full(1920, np.nan)
        for feature in features:
            if not feature.get('target_id', '').startswith('board.' + role):
                continue
            for segment in feature['segments']:
                uv = np.asarray(segment.get('source_native_readout_uv', segment['uv']), float)
                if len(uv) < 2:
                    continue
                uv = uv[np.argsort(uv[:, 0])]
                xx, inverse = np.unique(uv[:, 0], return_inverse=True)
                yy = np.array([np.median(uv[inverse == j, 1]) for j in range(len(xx))])
                if len(xx) < 2:
                    continue
                covered = (axis >= xx[0]) & (axis <= xx[-1])
                current = np.interp(axis[covered], xx, yy)
                reducer = np.fmin if role == 'top' else np.fmax
                values[covered] = reducer(values[covered], current)
        known = np.flatnonzero(np.isfinite(values))
        if len(known) < 50:
            raise ValueError('Current measured board band is insufficient for a material ownership guard')
        bounds.append(np.interp(axis, known, values[known]))
    return bounds


def single_station_review_contour(feature, segment):
    """Tiny unbound material keeps its complete path and one measured station."""
    trace = np.asarray(segment.get('source_native_readout_uv', []), float)
    return bool(segment.get('draw_current_native_material_readout')
        and segment.get('closed')
        and feature.get('review_layer') == 'unbound_materials'
        and not feature.get('target_id')
        and feature.get('allowed_for_fit') is False
        and not segment.get('point_bindings')
        and not segment.get('corner_vertex_indices')
        and not segment.get('protected_vertex_indices')
        and trace.ndim == 2 and trace.shape[1] == 2 and len(trace) >= 3
        and np.linalg.norm(np.ptp(trace, axis=0)) <= 8)


def compact_redundant_points(features, ceiling=500):
    """Keep every feature, segment endpoint and typed corner under the ceiling."""
    removed = []
    def count():
        return sum(len(s['uv']) for f in features for s in f['segments'])
    while count() > ceiling:
        candidates = []
        for fi, feature in enumerate(features):
            for si, segment in enumerate(feature['segments']):
                uv = np.asarray(segment['uv'], float)
                material_trace=bool(segment.get('draw_current_native_material_readout'))
                source_only_closed=bool(material_trace and segment.get('closed'))
                minimum=1 if single_station_review_contour(feature, segment) else 2 if source_only_closed else 4 if segment.get('closed') else 2
                if len(uv) <= minimum:
                    continue
                protected = set(segment.get('protected_vertex_indices', []))
                protected.update(segment.get('corner_vertex_indices', {}).values())
                protected.update(j for j, binding in enumerate(segment.get('point_bindings', [])) if binding.get('kind') == 'corner')
                # Material contours keep their complete measured closed path
                # for display. Two separated actual boundary stations remain;
                # semantic corners are protected independently below.
                for j in range(1, len(uv) if source_only_closed else len(uv)-1):
                    if j in protected:
                        continue
                    following=uv[(j+1)%len(uv)]
                    edge = following - uv[j-1]
                    span = float(np.linalg.norm(edge))
                    if span < 1e-6:
                        error = float(np.linalg.norm(uv[j] - uv[j-1]))
                    else:
                        along = np.clip(float((uv[j]-uv[j-1]) @ edge) / span**2, 0, 1)
                        error = float(np.linalg.norm(uv[j] - uv[j-1] - along * edge))
                    geometry=feature.get('geometric_interpretation',{})
                    if geometry.get('kind')=='projected_physical_circle_ellipse' and segment.get('representative_points_share_circle_family_estimate'):
                        # These are correlated stations on an unchanged same-source ellipse.
                        # Retaining endpoints and the saved ellipse preserves the complete supported arc.
                        error=0.0
                    if material_trace:
                        error=0.0
                    max_span = 260 if str(feature.get('target_id','')).startswith('board.') else 80 if '.print.' in feature['id'] else 180
                    if segment.get('representative_points_share_circle_family_estimate') or segment.get('representative_points_share_line_estimate'):
                        max_span=360
                    if material_trace:
                        # Full measured source topology remains the display path.
                        # Representative spacing does not simplify that path.
                        max_span=float('inf')
                    if error <= .65 and span <= max_span:
                        candidates.append((error, span, fi, si, j))
        if not candidates:
            raise ValueError('Cannot meet500 without losing supported shape/corners')
        error, span, fi, si, j = min(candidates)
        feature = features[fi]
        segment = feature['segments'][si]
        removed.append(dict(feature_id=feature['id'], segment_index=si, uv=segment['uv'][j],
            reason=('redundant_representative_on_preserved_complete_native_material_contour'
                    if segment.get('draw_current_native_material_readout')
                    else 'redundant_nearly_collinear_representative_same_feature'),
            image_shape_departure_px=error, retained_neighbour_spacing_px=span,
            raw_material_readouts_retained=True))
        for key in ('uv', 'point_bindings', 'transition_iqr_px', 'localization_sigma_px', 'sigma_px', 'source_evidence', 'point_ids'):
            if key in segment:
                if len(segment[key]) != segment['vertex_count']:
                    raise ValueError('Per-vertex source array mismatch before compaction: ' + key)
                segment[key].pop(j)
        if 'protected_vertex_indices' in segment:
            segment['protected_vertex_indices'] = [v - (v > j) for v in segment['protected_vertex_indices'] if v != j]
        if 'corner_vertex_indices' in segment:
            segment['corner_vertex_indices'] = {key:v - (v > j) for key, v in segment['corner_vertex_indices'].items()}
        segment['vertex_count'] -= 1
    return removed

def run():
    if (OUT / 'user_acceptance.json').exists() or any(read(p).get('manual_editor') for p in (OUT / 'frames').glob('*.json')):
        raise ValueError('User-reviewed or manually edited packets must not be overwritten by automatic assembly')
    manifest = read(DATA / 'frame_manifest.json')
    shot = manifest['shot_scope']['shot_index']
    if type(shot) is not int or shot < 1 or (EXPECTED_SHOT is not None and shot != EXPECTED_SHOT):
        raise ValueError('Selected manifest does not match the requested shot')
    label = {1:'第一镜头',2:'第二镜头',3:'第三镜头'}.get(shot,f'第{shot}镜头')
    marker_name = f'SHOT{shot}_SELECTED_REVIEW_20261014'
    restart_link = f'START_SHOT{shot}_REVIEW.cmd'
    entry_label = label + ('返修版' if '_r2_' in OUT.name else '')
    selected_pts = [r['source']['native_pts'] for r in manifest['frames']]
    intervals = np.diff(selected_pts) / 12800
    if len(intervals) and ((intervals < .5 - 1e-9).any() or (intervals > 1 + 1e-9).any()):
        raise ValueError('Every selected-frame interval must remain between 0.5 and 1 second')
    registry = read(MODEL / 'feature_registry.json')
    scene = read(MODEL / 'scene.json')
    templates = read(MODEL / 'ad2_material_template.json')
    model = read(MODEL / 'model.json')
    if 'inherited_first_shot_acceptance_context' not in model:
        model['inherited_first_shot_acceptance_context'] = {key:deepcopy(model[key]) for key in
            ('source_point_count', 'accepted_source_frame_count', 'accepted_source_point_count', 'accepted_batch_datasets') if key in model}
    model.update(model_id=registry['model_id'], derived_model_role=f'selected_shot{shot}_source_review',
        geometry_and_targets_unchanged=False, inherited_base_ice_board_ad_geometry_unchanged=True,
        new_print_initializers_accuracy_accepted=False, user_accepted=False,
        accepted_source_frame_count=0, accepted_source_point_count=0, accepted_batch_datasets=[],
        candidate_source_frame_count=len(manifest['frames']),
        candidate_batch_datasets=[dict(path=str(OUT / 'batch_manifest.json'), frame_count=len(manifest['frames']),
            point_count=0, status='awaiting_user_selected_frame_review')])
    if 'programme_binding' in model:
        model['inherited_first_shot_programme_binding'] = model.pop('programme_binding')
    if 'camera_seed' in model and model['camera_seed'].get('path') != str(DATA / 'camera_seeds.npz'):
        model['inherited_first_shot_camera_seed'] = model['camera_seed']
    model['camera_seed'] = reference(DATA / 'camera_seeds.npz')
    write(MODEL / 'model.json', model)
    cameras = np.load(DATA / 'camera_seeds.npz', allow_pickle=False)
    bycamera = dict(zip(cameras['programme_indices'].tolist(), cameras['camera25']))
    rows = []
    for ordinal, record in enumerate(manifest['frames'], 1):
        index = int(record['index'])
        board = read(DATA / f'board_candidates/{index:06d}.json')
        ice = read(DATA / f'ice_candidates/{index:06d}.json')
        if board.get('printed_material_reread_pending'):
            raise ValueError(f'Board printing is still pending for this source frame: {index}')
        verify_candidate_source(board, record, 'board')
        verify_candidate_source(ice, record, 'ice')
        domain = ice['source'].get('ice_visible_domain', {})
        if domain.get('board_contact_input_sha256') and domain['board_contact_input_sha256'] != contact_domain_input_sha256(board):
            raise ValueError(f'Ice read used different current source board contact: {index}')
        features = deepcopy(board['features']) + deepcopy(ice['features'])
        features = [f for f in features if any(s['uv'] for s in f.get('segments', []))]
        unbound = deepcopy(board.get('unbound_static_materials', [])) + deepcopy(ice.get('unbound_static_materials', []))
        for feature in features + unbound:
            # Review-only materials retain the same complete measured path as
            # bound printing; compact their representatives consistently too.
            for segment in feature['segments']:
                # Preserve the current measured source-only ice stations before
                # reducing representatives, just as board material readers do.
                if (feature in unbound and not segment.get('source_native_readout_uv')
                        and segment.get('individual_native_pixel_readouts')
                        and segment.get('pixels_measured_in_this_frame')
                        and segment.get('source_evidence_measured_in_this_frame')):
                    segment['source_native_readout_uv'] = deepcopy(segment['uv'])
                    segment['source_native_trace_topology'] = 'ordered_measured_material_stations'
                    segment['source_native_station_evidence'] = deepcopy(segment.get('source_evidence', []))
                    segment['source_native_station_sigma_px'] = deepcopy(segment.get('sigma_px', []))
                raw_trace=segment.get('source_native_readout_uv')
                if raw_trace and segment.get('pixels_measured_in_this_frame') and segment.get('source_evidence_measured_in_this_frame'):
                    segment.update(draw_current_native_material_readout=True,
                        native_material_readout_display_only_not_additional_observations=True,
                        representative_points_share_material_contour_estimate=True)
                    uv=np.asarray(segment['uv'],float)
                    if segment.get('closed') and len(uv)>3 and not single_station_review_contour(feature, segment):
                        farthest=int(np.argmax(np.linalg.norm(uv-uv[0],axis=1)))
                        segment['protected_vertex_indices']=sorted(set(segment.get('protected_vertex_indices',[]))|{farthest})
        compaction = compact_redundant_points(features+unbound, ceiling=500)
        unbound_count = sum(len(s['uv']) for f in unbound for s in f['segments'])
        for feature in features:
            feature['allowed_for_fit'] = False
            feature['point_count'] = sum(len(s['uv']) for s in feature['segments'])
            feature['review_layer'] = feature.get('review_layer', 'ice_features' if feature['id'].startswith('ice.') else 'ad2_print')
            for si, segment in enumerate(feature['segments']):
                segment['vertex_count'] = len(segment['uv'])
                segment['point_ids'] = [f'shot{shot}/{index}/{feature["id"]}/{si}/{j}' for j in range(len(segment['uv']))]
        for feature in unbound:
            feature['point_count']=sum(len(s['uv']) for s in feature['segments'])
            for si,segment in enumerate(feature['segments']):
                segment['point_ids']=[f'unbound/{index}/{feature["id"]}/{si}/{j}' for j in range(len(segment['uv']))]
        count = sum(f['point_count'] for f in features)
        if count + unbound_count > 500:
            raise ValueError(f'Point ceiling exceeded: {index}, {count}; retain features and compact redundancy before publishing')
        for role in ('top', 'contact'):
            if not any(f['target_id'] == 'board.' + role or f['target_id'].startswith('board.' + role + '.') for f in features):
                raise ValueError(f'Visible mandatory board {role} missing: {index}')
        source = deepcopy(record['source'])
        source.update(mask_path=record['mask_path'], mask_sha256=sha(record['mask_path']),
                      navigation_overlay_path=str(DATA / f'navigation/{index:06d}.png'))
        if record.get('source_binding'):
            ref = record['source_binding']
            if sha(ref['path']) != ref['sha256']:
                raise ValueError(f'Current exact source binding changed: {index}')
            source['source_binding'] = deepcopy(ref)
        raw = cv2.imread(record['image_path'])
        if hashlib.sha256(raw.tobytes()).hexdigest() != source['decoded_BGR_sha256']:
            raise ValueError('Native RGB changed')
        mask = cv2.imread(record['mask_path'], 0)
        board_top, board_bottom = source_board_material_band(board['features'])
        for feature in features + unbound:
            for segment in feature['segments']:
                verify_ordered_native_contour(segment, index, feature['id'])
                uv = np.asarray(segment['uv'], float)
                if segment.get('draw_current_native_material_readout'):
                    # The complete actual readout is display-only, but it must
                    # still be current material rather than a removed person.
                    uv = np.vstack((uv, np.asarray(segment['source_native_readout_uv'], float)))
                if not np.isfinite(uv).all() or (uv < [0, 0]).any() or (uv > [1919, 1079]).any():
                    raise ValueError(f'Invalid source UV: {index}/{feature["id"]}')
                xy = np.rint(uv).astype(int)
                if mask[xy[:, 1], xy[:, 0]].any():
                    raise ValueError(f'Point overlaps source person exclusion: {index}/{feature["id"]}')
                if str(feature.get('target_id', '')).startswith('Ad'):
                    if ((uv[:, 1] < board_top[xy[:, 0]] - 2) |
                        (uv[:, 1] > board_bottom[xy[:, 0]] + 2)).any():
                        raise ValueError(f'Advertisement material point outside current measured board body: {index}/{feature["id"]}')
                if feature.get('physical_surface') == 'ice' and (uv[:, 1] < board_bottom[xy[:, 0]] - 2).any():
                    raise ValueError(f'Ice material point above current measured yellow body bottom: {index}/{feature["id"]}')
        packet = dict(schema='rinkspace.single-frame-source-review.v1', source=source,
            source_pts=source['native_pts'], programme_frame_index_1based=index,
            batch_sequence_index_1based=ordinal, frame_count=1, features=features, point_count=count,
            source_point_count=count+unbound_count,
            status=f'selected_shot{shot}_native_candidate_awaiting_user_review', user_accepted=False,
            user_hard_point_ceiling=500, observation_coordinates_from_projection=False,
            optimization_performed=False, source_clock='official50_native_PTS',
            sampling_role='selected_clear_frame_at_approximately_0.5_to_1_second_intervals',
            unresolved_source_regions=ice.get('unresolved_source_regions', []),
            per_feature_coverage=ice.get('per_feature_coverage', []),
            point_ceiling_compaction=compaction,
            excluded_features_by_user=board.get('excluded_features_by_user', []),
            unbound_static_materials=unbound,
            source_correlated_groups=deepcopy(board.get('source_correlated_groups',[]))+deepcopy(ice.get('source_correlated_groups',[])),
            unresolved_board_regions=deepcopy(board.get('unresolved_source_regions',[]))+[
                dict(deepcopy(limit), feature_id='unbound.board.white_cap_material',
                     note_zh='帽带白色材料：当前源中可独立辨认的边界不足；该区域未补点。')
                for limit in board.get('current_white_cap_material_read_limits',[])],
            board_material_ownership_guard=dict(source='current_measured_blue_cap_top_and_yellow_body_bottom',
                used_only_for_exclusion_not_observations=True,advertisement_identity_requires_separate_source_review=True),
            candidate_provenance=dict(board=reference(DATA / f'board_candidates/{index:06d}.json'),
                ice=reference(DATA / f'ice_candidates/{index:06d}.json')))
        packet_path = OUT / f'frames/{index:06d}.json'
        write(packet_path, packet)
        binding = make_binding(packet, packet_path, registry)
        binding_path = OUT / f'bindings/{index:06d}.json'
        write(binding_path, binding)
        solver_rows, targets, corners = assemble(packet, binding, registry, scene, templates)
        residual, jacobian = evaluate_rows(bycamera[index], solver_rows, jacobian=True)
        if not np.isfinite(residual).all() or not np.isfinite(jacobian).all():
            raise ValueError(f'Nonfinite model entry: {index}')
        counts = Counter()
        for observation in binding['observations']:
            counts[observation['target_id']] += len(observation['source_vertex_indices'])
        if sum(counts.values()) != count:
            raise ValueError('Binding inventory differs')
        rows.append(dict(index=index, source_pts=source['native_pts'], source_time=source['source_time'],
            point_count=count, source_point_count=count+unbound_count, unbound_material_point_count=unbound_count,
            packet=reference(packet_path), model_binding=reference(binding_path),
            target_point_counts=dict(counts), user_accepted=False,
            finite_model_residual_entry_verified=True, accuracy_or_user_acceptance_claimed=False))
    numbers = [r['source_point_count'] for r in rows]
    bound_count = sum(r['point_count'] for r in rows)
    model['candidate_batch_datasets'][0]['point_count'] = bound_count
    model['candidate_source_point_count'] = bound_count
    model['source_point_count'] = bound_count
    write(MODEL / 'model.json', model)
    batch = dict(schema='rinkspace.selected-source-frame-review-batch.v1',
        event_id=manifest['event_id'], programme_id=manifest['programme_id'], shot_index=shot, frame_count=len(rows),
        programme_frame_range=[rows[0]['index'], rows[-1]['index']],
        selected_programme_indices=[r['index'] for r in rows],
        full_shot_scope=manifest.get('full_physical_shot_scope', manifest.get('shot_scope')),
        processing_scope=manifest.get('shot_scope'),
        explicit_user_excluded_tail=manifest.get('explicit_user_excluded_tail'),
        source_manifest=reference(DATA / 'frame_manifest.json'), selection=reference(DATA / 'selection.json'),
        model=reference(MODEL / 'model.json'), frames=rows, user_accepted=False,
        status='selected_clear_frames_awaiting_user_review',
        requested_selection_interval_seconds=[.5, 1.0],
        actual_selected_intervals_seconds=intervals.tolist(),
        density=dict(total=sum(numbers), minimum=min(numbers), maximum=max(numbers), mean=float(np.mean(numbers)), hard_ceiling=500),
        all_points_have_unique_typed_model_targets=all(r['unbound_material_point_count']==0 for r in rows),
        all_bound_points_have_unique_typed_model_targets=True,
        unbound_materials_review_only_excluded_from_fit=True, finite_model_residual_entry_verified=True,
        actual_source_visual_review_is_separate=True, optimization_performed=False,
        all_source_frames_have_observations=len(rows)==manifest['shot_scope'].get('native_frame_count'),
        unselected_frames_have_no_observation_packets=True,
        editor_url=EDITOR_URL)
    write(OUT / 'batch_manifest.json', batch)
    entries = []
    tiles = []
    (DATA / 'overlays').mkdir(parents=True, exist_ok=True)
    for row in rows:
        index = row['index']
        packet = read(OUT / f'frames/{index:06d}.json')
        raw = cv2.imread(packet['source']['image_path'])
        segments = list(source_segments(packet))
        overlay_path = DATA / f'overlays/{index:06d}.png'
        overlay = native_overlay(raw, segments, packet=packet)
        cv2.imwrite(str(overlay_path), overlay, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        entries.append(dict(index=index, source=Path(packet['source']['image_path']).as_uri(),
            navigation=(DATA / f'navigation/{index:06d}.png').as_uri(), overlay=overlay_path.as_uri(),
            source_time=timestamp(packet['source']), pts=packet['source']['native_pts'], time_base='1/12800',
            point_count=packet['source_point_count'], counts={key:sum(len(uv) for layer, _, uv, _ in segments if layer == key) for key in LAYERS},
            status='等待用户逐张审核',
            notes=[r.get('note_zh',r.get('reason','材料边界暂不能可靠区分')) for r in packet.get('unresolved_source_regions',[])+packet.get('unresolved_board_regions',[])],
            inferred_segments=[], segments=display_segments(packet)))
        tile = cv2.resize(overlay, (480, 270))
        cv2.putText(tile, f'{index} {timestamp(packet["source"])}', (4, 19), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
        tiles.append(tile)
    sheet = np.zeros((270 * ((len(tiles) + 3) // 4), 1920, 3), np.uint8)
    for i, tile in enumerate(tiles):
        y, x = divmod(i, 4)
        sheet[y*270:(y+1)*270, x*480:(x+1)*480] = tile
    cv2.imwrite(str(DATA / 'points_contact_sheet.png'), sheet)
    page = AssetStore(ROOT).review_page(OUT)
    page.parent.mkdir(parents=True, exist_ok=True)
    text = page_html(entries, OUT / 'batch_manifest.json')
    text = text.replace('帧取点审阅', f'帧 · {label} {len(rows)} 张选中帧审阅')
    text = text.replace('<p class="meta" id="meta">', f'<p>仅列出每 0.5–1 秒挑选的尽量清晰帧；快速摇镜段仍有残留拖影，逐帧注明读取限制。全镜头其余帧没有读点数据。<a href="{EDITOR_URL}">进入可加点、删点审阅页</a></p><p class="meta" id="meta">')
    page.write_text(text, encoding='utf-8')
    entry = AssetStore(ROOT).project_entry()
    current_entry = entry.read_text(encoding='utf-8')
    block = (f'<!-- {marker_name}_BEGIN --><section class="surface" id="shot{shot}-selected-review-20261014">'
        f'<h3>冈田芽衣：{entry_label} {len(rows)} 张选帧取点，等待人工审核</h3>'
        f'<p>官方50fps原生PTS：{rows[0]["source_time"]}–{rows[-1]["source_time"]}。相邻选帧间隔'
        f'{intervals.min():.2f}–{intervals.max():.2f}秒，每个时间段均选尽量清晰的帧；快速摇镜帧保留残留运动模糊说明。</p>'
        f'<p>冰面直线、靶圈、其它已确认场地标识，围挡上下沿，广告边界、文字与图案。'
        f'每帧{min(numbers)}–{max(numbers)}点，总计{sum(numbers)}点；上限500，尚未人工验收、未重新拟合。</p>'
        f'<p><a href="{EDITOR_URL}">打开{label}可加点、删点审阅页</a> · '
        f'<a href="{restart_link}">重启{label}审阅后端</a> · '
        f'<a href="{AssetStore(ROOT).review_page(OUT).as_uri()}">静态审阅页</a></p>'
        f'</section><!-- {marker_name}_END -->\n')
    marker = rf'<!-- {marker_name}_BEGIN -->.*?<!-- {marker_name}_END -->\s*'
    if re.search(marker, current_entry, flags=re.S):
        current_entry = re.sub(marker, lambda _: block, current_entry, count=1, flags=re.S)
    else:
        current_entry = current_entry.replace('<!-- CURRENT_TASK_BEGIN -->', '<!-- CURRENT_TASK_BEGIN -->\n' + block, 1)
    if PUBLISH_ENTRY:entry.write_text(current_entry, encoding='utf-8')
    write(DATA / 'assembly_validation.json', dict(frame_count=len(rows), point_count=sum(numbers),
        all_source_hashes_and_typed_targets_verified=True, model_entry_finite=True,
        accuracy_or_user_acceptance_claimed=False, optimization_performed=False, review_html=str(page)))
    print(json.dumps(dict(frames=len(rows), density=batch['density'], review=str(page)), ensure_ascii=False), flush=True)

def main(argv=None):
    global DATA, OUT, MODEL, SOURCE_MEDIA, EDITOR_URL, PUBLISH_ENTRY, EXPECTED_SHOT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=DATA)
    parser.add_argument('--dataset',type=Path,default=OUT)
    parser.add_argument('--model',type=Path,default=MODEL)
    parser.add_argument('--source-media',type=Path)
    parser.add_argument('--editor-url',default=EDITOR_URL)
    parser.add_argument('--no-publish',action='store_true')
    parser.add_argument('--shot-index',type=int)
    args=parser.parse_args(argv)
    DATA=args.data.resolve();OUT=args.dataset.resolve();MODEL=args.model.resolve()
    SOURCE_MEDIA=(args.source_media or DATA).resolve();EDITOR_URL=args.editor_url
    PUBLISH_ENTRY=not args.no_publish
    EXPECTED_SHOT=args.shot_index
    run()

if __name__ == '__main__':
    main()
