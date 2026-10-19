"""Append authenticated clearer source readings to existing factor packets.

No source UV is generated here. Clear-anchor packets retain their optical
supports and identity authority; previously used retention points are FIT,
and separate unused material controls stay HELD. Existing parent rows remain
exactly the same JSON objects and keep their original roles and weights.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def append_clear_anchors(parent_paths, clear_paths, output_paths):
    parents = {kind: json.loads(Path(path).read_text(encoding='utf8'))
               for kind, path in parent_paths.items()}
    packets = deepcopy(parents)
    common = {(p['source_video_sha256'], p['time_base']) for p in parents.values()}
    if len(common) != 1:
        raise ValueError('Parent packets must belong to one authenticated video and clock')
    source, clock = next(iter(common))
    admitted = {kind: [] for kind in parents}
    witnesses = []
    for path in clear_paths:
        clear = json.loads(Path(path).read_text(encoding='utf8'))
        if (clear['source_video_sha256'], clear['time_base']) != (source, clock):
            raise ValueError('Clear source cannot cross video or source clock')
        witnesses.append(dict(path=str(Path(path).resolve()), sha256=sha(path)))
        records = [(deepcopy(r), 'fit') for r in clear['rows']]
        for r in clear.get('evaluation_rows', []):
            # These two controls already constrained the accepted initializer
            # experiments. Their later use cannot be represented as unseen.
            if r['curve']['kind'] == 'board_circle':
                matching = next(row for row, _ in records if row['curve'] == r['curve'])
                matching['prior_retention_point_indices'] = list(
                    range(len(matching['uv']), len(matching['uv']) + len(r['uv'])))
                for field in ('uv', 'sigma_px', 'normal', 'normal_interval_px', 'measurements'):
                    if field in matching:
                        matching[field].extend(deepcopy(r[field]))
                matching['retention_source_record_id'] = r['id']
            else:
                role = 'fit' if r['curve']['kind'] == 'board_top' else 'holdout'
                row = deepcopy(r)
                if role == 'fit': row['previous_initializer_retention_now_FIT'] = True
                records.append((row, role))
        for row, role in records:
            if (row.get('source_RGB_actual_measurement') is not True
                    or row.get('model_projection_used_to_generate_or_select_uv') is not False):
                raise ValueError('Actual source RGB, independently read from model projection, required')
            if sha(row['source_raw_frame']) != row['source_file_sha256']:
                raise ValueError('Native source frame bytes changed')
            kind = row['curve']['kind']
            target = {'house_ring': 'ice', 'current_ice_segment': 'ice',
                      'board_circle': 'circle', 'board_top': 'board',
                      'advertisement_boundary': 'board'}.get(kind)
            if target is None: raise ValueError('Unsupported qualified material identity')
            row['split'] = role
            row['source_anchor_admission'] = 'existing_user_complete_fit_task_clearer_source_reinspection'
            if target == 'ice':
                row['feature_id'] = (row['curve'].get('feature_id') or
                    f"{row['curve']['side']}{row['curve']['row']}")
                row['source_optical_normal_support_authorized'] = True
            elif target == 'board':
                row['point_correspondence'] = False
                row['curve_observation_kind'] = 'normal_interval'
            ids = {r['id'] for r in packets[target]['rows']}
            if row['id'] in ids: raise ValueError('Clear source must not duplicate a parent observation')
            packets[target]['rows'].append(row)
            admitted[target].append(dict(id=row['id'], split=role, uv_count=len(row['uv'])))
    report = dict(source_video_sha256=source, time_base=clock, source_packets=witnesses,
                  appended=admitted, independent_unused_frames_added_to_FIT=False,
                  original_parent_rows_and_roles_unchanged=True, accuracy_accepted=False)
    for kind, packet in packets.items():
        parent_count = len(parents[kind]['rows'])
        if packet['rows'][:parent_count] != parents[kind]['rows']:
            raise AssertionError('Parent target changed during source admission')
        packet['clear_source_anchor_append'] = dict(report,
            parent_path=str(Path(parent_paths[kind]).resolve()),
            parent_sha256=sha(parent_paths[kind]), original_parent_row_count=parent_count)
        packet['current_target_version'] = 'parent_preserved_plus_actual_clear_source_anchors_20261007'
        packet['observation_count'] = len(packet['rows'])
        packet['fit_observation_count'] = sum(r['split'] == 'fit' for r in packet['rows'])
        packet['holdout_observation_count'] = sum(r['split'] == 'holdout' for r in packet['rows'])
        packet['current_fit_scalar_count'] = 2 * sum(len(r['uv']) for r in packet['rows'] if r['split'] == 'fit')
        packet['current_primary_held_scalar_count'] = 2 * sum(len(r['uv']) for r in packet['rows'] if r['split'] == 'holdout')
        packet['accuracy_accepted'] = False
        path = Path(output_paths[kind]); path.parent.mkdir(parents=True, exist_ok=True)
        if kind == 'circle':
            reference = packet['geometry_contract']['same_print_source_evidence']
            if sha(reference['path']) != reference['sha256']:
                raise ValueError('Original same-print identity evidence changed')
            evidence = json.loads(Path(reference['path']).read_text(encoding='utf8'))
            evidence['clear_native_source_identity_append'] = dict(
                parent_identity_evidence=deepcopy(reference),
                identity_sources=witnesses,
                original_same_size_user_authority_unchanged=True,
                new_size_or_position_constraint_added=False)
            for row in packet['rows'][parent_count:]:
                evidence['source_frames'].append(dict(
                    advertisement_number=row['curve']['advertisement_id'],
                    pts=row['pts'], actual_pts=row['actual_pts'],
                    source_raw_frame=row['source_raw_frame'],
                    source_file_sha256=row['source_file_sha256'],
                    physical_identity_authority=row['physical_identity_authority'],
                    source_RGB_actual_measurement=True,
                    model_projection_used_to_generate_or_select_uv=False))
            identity_path = path.parent / 'same_print_source_identity_with_actual_clear_anchors.json'
            identity_path.write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
            packet['geometry_contract']['same_print_source_evidence'] = dict(
                path=str(identity_path.resolve()), sha256=sha(identity_path))
        path.write_text(json.dumps(packet, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    return report
