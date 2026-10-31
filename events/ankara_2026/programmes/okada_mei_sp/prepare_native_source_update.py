"""Update authenticated source groups/rows without resetting a native joint state.

Groups are (shot, original PTS, native path). Unprovided, unwithdrawn rows stay byte-identical.
Explicit source-review corrections may withdraw individual authenticated old rows.
Only genuinely new curve pixels receive native nuisance-station initialization;
all cameras, scene values, controls and existing source-pixel stations are kept.
"""
from pathlib import Path
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import re

import numpy as np

import prepare_native_multicamera as prepared
from prepare_native_extension import dependency_files, lossless_cast, source_coverage, validate_points
from prepare_spatial_refinement import _same_bytes, _write_prepared
from rinkspace.core.native import NativeCore
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
from rinkspace.domain.parameters import SceneParameterModel, ParameterState
from rinkspace.path_policy import guard_project_write
from rinkspace.session.gpu_geometry import NativeSceneGPU, initialize_curve_stations

HERE = Path(__file__).resolve().parent
CORRECTIONS_SCHEMA = 'rinkspace.source_row_corrections.v1'


def _relative(path, folder):
    path = Path(path)
    return (path if path.is_absolute() else folder/path).resolve(strict=True)


def _source_hash(manifest, mapping, shot_id):
    values = [mapping.get('source_video_sha256'), manifest.get('source_video_sha256'),
              manifest.get('source_inputs', {}).get(shot_id, {}).get('source_video_sha256')]
    values = {v for v in values if v is not None}
    if not values:
        values = {prepared.read(HERE/'rebuild_scope.json')['source']['sha256']}
    if len(values) != 1:
        raise ValueError('Current source video identity is inconsistent')
    value = values.pop()
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('Current source video SHA256 is required')
    return value


def _integer(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError('Invalid source provenance '+label)
    return value


def _sha256(value, label):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('Source correction needs a SHA256 for '+label)
    return value


def _exact_fields(value, fields, label):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError('Source correction '+label+' fields must be exactly: '+', '.join(sorted(fields)))


def authenticate_source_corrections(path, manifest, mapping, original):
    """Bind explicit withdrawals to exact parent rows and hashed review evidence.

    The BGR digest is a source-review declaration, not authenticated from the
    seven-field parent point array. Evidence content still needs source review;
    fitted residuals are never accepted as the stated reason for withdrawal.
    """
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    document = json.loads(raw.decode('utf-8-sig'))
    _exact_fields(document, ('schema', 'parent_input_sha256', 'source_video_sha256', 'rows'), 'document')
    if document['schema'] != CORRECTIONS_SCHEMA:
        raise ValueError('Unsupported source correction schema')
    if _sha256(document['parent_input_sha256'], 'parent input') != manifest['point_input_sha256']:
        raise ValueError('Source correction parent point input SHA256 differs')
    source_hash = _sha256(document['source_video_sha256'], 'source video')
    rows = document['rows']
    if not isinstance(rows, list) or not rows:
        raise ValueError('Source corrections need nonempty explicit rows')
    protected = {str(path): hashlib.sha256(raw).hexdigest()}
    fields = ('observation_id', 'shot_id', 'pts', 'path_id', 'uv', 'source_BGR_sha256',
              'reason', 'source_review_disposition', 'source_evidence')
    requested_ids = set()
    for row in rows:
        _exact_fields(row, fields, 'row')
        observation_id = row['observation_id']
        if not isinstance(observation_id, str) or not observation_id:
            raise ValueError('Source correction observation_id is required')
        requested_ids.add(observation_id)
    # Scan the large U96 parent array once, retaining indices only for requested
    # IDs. Keep every matching index so ambiguous parent identities still fail.
    requested_matches = {observation_id: [] for observation_id in requested_ids}
    for index, observation_id in enumerate(original['observation_ids']):
        matches = requested_matches.get(observation_id)
        if matches is not None:
            matches.append(index)
    seen, authenticated = set(), []
    for row in rows:
        candidates = requested_matches[row['observation_id']]
        if len(candidates) != 1:
            raise ValueError('Source correction must identify exactly one current observation_id')
        index = int(candidates[0])
        if index in seen:
            raise ValueError('Duplicate source correction for the same parent row')
        seen.add(index)
        owners = [shot for shot in manifest['shots'] if shot['row_begin'] <= index < shot['row_stop']]
        if len(owners) != 1 or row['shot_id'] != owners[0]['id']:
            raise ValueError('Source correction shot identity differs')
        if source_hash != _source_hash(manifest, mapping, row['shot_id']):
            raise ValueError('Source correction source video SHA256 differs')
        uv = row['uv']
        if (not isinstance(uv, list) or len(uv) != 2
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) for v in uv)):
            raise ValueError('Source correction needs an exact finite UV pair')
        if (_integer(row['pts'], 'correction pts') != int(original['pts'][index])
                or row['path_id'] != mapping['path_ids'][int(original['path_indices'][index])]
                or not np.array_equal(uv, original['uv'][index])):
            raise ValueError('Source correction PTS/path/UV differs from the exact parent row')
        _sha256(row['source_BGR_sha256'], 'declared source BGR')
        if row['source_review_disposition'] not in (
                'confirmed_not_target_material_boundary', 'source_material_support_occluded',
                'source_entity_identity_unresolved'):
            raise ValueError('Source correction requires source review of a rejected boundary or occluded material support')
        reason = row['reason']
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError('Source correction source-review reason is required')
        if re.search(r'residual|reprojection|(?:fit|projection)[ _-]*(?:error|loss)|残差|重投影|拟合(?:误差|损失)|投影误差', reason, re.I):
            raise ValueError('Source correction reason must use source evidence, not fitted residuals')
        evidence = row['source_evidence']
        if not isinstance(evidence, list) or not evidence:
            raise ValueError('Source correction needs nonempty source evidence files')
        reviewed, evidence_seen = [], set()
        for item in evidence:
            _exact_fields(item, ('path', 'sha256'), 'source evidence')
            if not isinstance(item['path'], str) or not item['path'].strip():
                raise ValueError('Source correction evidence path is required')
            evidence_path = _relative(item['path'], path.parent)
            if not evidence_path.is_file():
                raise ValueError('Source correction evidence must be a file')
            if evidence_path == path or str(evidence_path) in evidence_seen:
                raise ValueError('Source correction needs distinct external source evidence files')
            evidence_seen.add(str(evidence_path))
            expected = _sha256(item['sha256'], 'source evidence file')
            actual = protected.get(str(evidence_path))
            if actual is None:
                actual = prepared.digest(evidence_path)
            if actual != expected:
                raise ValueError('Source correction evidence file SHA256 differs: '+str(evidence_path))
            protected[str(evidence_path)] = actual
            reviewed.append(dict(path=str(evidence_path), sha256=expected))
        authenticated.append(dict(row, parent_row_index=index, source_evidence=reviewed))
    receipt = dict(schema=CORRECTIONS_SCHEMA, path=str(path), sha256=protected[str(path)],
        parent_input_sha256=document['parent_input_sha256'], source_video_sha256=source_hash,
        source_BGR_sha256_basis='source_review_declaration_not_parent_point_array_authentication',
        evidence_file_hashes_authenticated=True, source_evidence_content_automatically_verified=False,
        rows=authenticated)
    return dict(parent_rows=np.array([row['parent_row_index'] for row in authenticated], np.int64),
        protected=protected, raw=raw, receipt=receipt)


def authenticate_packet(folder, manifest, old_mapping, old_points):
    """Authenticate only the new packet and its explicitly referenced source rows."""
    folder = Path(folder).resolve(strict=True)
    if folder.is_file():
        if folder.name != 'point_observations.npz':
            raise ValueError('Point packets must be directories or point_observations.npz')
        folder = folder.parent
    point_path, mapping_path = folder/'point_observations.npz', folder/'mapping.json'
    mapping = prepared.read(mapping_path)
    points = prepared.npz(point_path)
    if set(points) != prepared.FIELDS:
        raise ValueError('Only the current seven unweighted point fields are permitted')
    if not len(points['pts']):
        raise ValueError('Empty point packets cannot replace or delete current source groups')
    validate_points(points, mapping)
    if mapping.get('independent_source_capture') is not True or mapping.get('fit_eligible') is False:
        raise ValueError('Point packet is not eligible independent source capture')
    if mapping.get('source_UV_synthesized') is True or mapping.get('source_measurements_and_pts_generated') is True:
        raise ValueError('Synthetic source measurements are not eligible')
    if mapping.get('pixel_domain') != 'native_distorted' or mapping.get('image_size') != [1920, 1080]:
        raise ValueError('Point packet native pixel domain differs')
    point_hash = prepared.digest(point_path)
    if mapping.get('point_input_sha256') != point_hash:
        raise ValueError('Point packet input digest differs')
    owners = [s for s in manifest['shots'] if
              np.all((points['pts'] >= s['first_pts']) & (points['pts'] < s['stop_pts_exclusive']))]
    if len(owners) != 1:
        raise ValueError('A point packet must belong to exactly one current shot')
    shot = owners[0]
    if mapping.get('shot_id', shot['id']) != shot['id']:
        raise ValueError('Point packet shot identity differs')
    step = int(manifest.get('source_clock', {}).get('native_pts_step', 256))
    if step <= 0 or np.any((points['pts']-shot['first_pts']) % step):
        raise ValueError('Packet contains a non-native source PTS')
    expected_source = _source_hash(manifest, old_mapping, shot['id'])
    if mapping.get('source_video_sha256') != expected_source:
        raise ValueError('Point packet source video SHA256 differs')
    try:
        capture = _relative(mapping['source_capture_jsonl'], folder)
        row_manifest = _relative(mapping['source_row_manifest'], folder)
        capture_hash = mapping['source_capture_sha256']
        row_hash = mapping['source_row_manifest_sha256']
    except KeyError as error:
        raise ValueError('Point packet needs current source-capture and source-row provenance') from error
    protected = {str(point_path): point_hash, str(mapping_path): prepared.digest(mapping_path)}
    for path, expected in [(capture, capture_hash), (row_manifest, row_hash)]:
        if prepared.digest(path) != expected:
            raise ValueError('Current source qualification dependency changed: '+str(path))
        protected[str(path)] = expected
    by_line, seen, source_locations = defaultdict(list), set(), set()
    with row_manifest.open(encoding='utf-8-sig') as stream:
        for line in stream:
            location = json.loads(line)
            i = _integer(location.get('packet_row'), 'packet_row')
            if i >= len(points['pts']) or i in seen:
                raise ValueError('Packet source-row coverage is not one-to-one')
            seen.add(i)
            path_id = mapping['path_ids'][int(points['path_indices'][i])]
            if (location.get('observation_id') != str(points['observation_ids'][i])
                    or location.get('pts') != int(points['pts'][i])
                    or location.get('source_row') != int(points['source_rows'][i])
                    or location.get('path_id') != path_id
                    or not np.array_equal(location.get('uv'), points['uv'][i])):
                raise ValueError('Point array differs from its current source-row location')
            if (_relative(location.get('source_jsonl', ''), folder) != capture
                    or location.get('source_jsonl_sha256') != capture_hash):
                raise ValueError('Point row refers to a different source capture')
            number = _integer(location.get('source_line'), 'source_line', 1)
            row_index = _integer(location.get('row_index'), 'row_index')
            if (number, row_index) in source_locations:
                raise ValueError('The same actual source row is repeated in a point packet')
            source_locations.add((number, row_index))
            by_line[number].append((i, location))
    if len(seen) != len(points['pts']):
        raise ValueError('Current source provenance does not cover every point')
    checked = 0
    with capture.open('rb') as stream:
        for number, raw in enumerate(stream, 1):
            if number not in by_line:
                continue
            frame = json.loads(raw)
            if frame.get('fit_eligible') is not True:
                raise ValueError('Current source frame is not explicitly fit_eligible')
            bgr_hash = frame.get('source_BGR_sha256')
            if not isinstance(bgr_hash, str) or re.fullmatch('[0-9a-f]{64}', bgr_hash) is None:
                raise ValueError('Qualified frame needs its original source BGR digest')
            source_rows = frame.get('rows')
            if not isinstance(source_rows, list):
                raise ValueError('Qualified frame needs actual measurement rows')
            actual_line_hash = hashlib.sha256(raw).hexdigest()
            for i, location in by_line[number]:
                row_index = location['row_index']
                if (location.get('source_line_sha256') != actual_line_hash
                        or location.get('source_BGR_sha256') != bgr_hash
                        or frame.get('pts') != int(points['pts'][i]) or row_index >= len(source_rows)):
                    raise ValueError('Actual source frame/row identity differs')
                row = source_rows[row_index]
                if (row.get('fit_eligible') is False or row.get('model_UV_synthesized') is True
                        or row.get('path_id') != location['path_id']
                        or not np.array_equal(row.get('uv'), points['uv'][i])
                        or not row.get('measurement_operator')):
                    raise ValueError('Source measurement is ineligible or differs from packet UV/path')
                kind = int(points['kind'][i])
                if row.get('kind', kind) != kind:
                    raise ValueError('Source observation kind differs')
                if kind == 0 and row.get('station') != float(points['stations'][i]):
                    raise ValueError('A fixed point needs its actual identified source station')
                checked += 1
            if checked == len(points['pts']):
                break
    if checked != len(points['pts']):
        raise ValueError('Actual source records do not cover every supplied point')
    points = {key: lossless_cast(value, old_points[key].dtype, key) for key, value in points.items()}
    declaration = mapping.get('source_replacement')
    explicit = isinstance(declaration, dict) and declaration.get('policy') == 'explicit_parent_observation_ids'
    parent_ids = (authenticate_parent_observation_ids(mapping, points, manifest, old_mapping, old_points, shot['id'])
                  if explicit else None)
    spans = {} if explicit else authenticate_replacement_spans(mapping, points)
    result = dict(folder=folder, shot_id=shot['id'], points=points, mapping=mapping,
        replacement_spans=spans,
        source_video_sha256=expected_source, protected=protected,
        source_capture=str(capture), source_capture_sha256=capture_hash,
        authenticated_rows=checked, fit_eligible=True)
    if explicit:
        result['replacement_parent_observation_ids'] = parent_ids
    return result


def _pixel_key(points, i):
    key = (int(points['pts'][i]), int(points['path_indices'][i]), int(points['kind'][i]),
           float(points['uv'][i, 0]), float(points['uv'][i, 1]))
    return key+(float(points['stations'][i]),) if int(points['kind'][i]) == 0 else key


def authenticate_parent_observation_ids(mapping, points, manifest, old_mapping, old_points, shot_id):
    """Authenticate a complete per-incoming-group list of explicitly retired IDs.

    Empty ID lists append observations to that group. No unlisted parent point,
    including another visible branch of the same native entity, is retired.
    The caller supplies the current parent arrays already bound to the manifest.
    """
    declaration = mapping.get('source_replacement')
    _exact_fields(declaration, ('policy', 'parent_point_input_sha256', 'groups'), 'explicit parent source replacement')
    if declaration['policy'] != 'explicit_parent_observation_ids' or not isinstance(declaration['groups'], list):
        raise ValueError('Unsupported explicit parent source replacement policy')
    parent_hash = _sha256(declaration['parent_point_input_sha256'], 'parent point input')
    if parent_hash != manifest.get('point_input_sha256'):
        raise ValueError('Explicit replacement parent point input SHA256 differs')
    incoming = defaultdict(set)
    for pts, path, kind in zip(points['pts'], points['path_indices'], points['kind']):
        incoming[int(pts), mapping['path_ids'][int(path)]].add(int(kind))
    if not incoming:
        raise ValueError('Explicit parent replacement needs nonempty incoming groups')
    result, requested = {}, set()
    for group in declaration['groups']:
        _exact_fields(group, ('pts', 'path_id', 'parent_observation_ids'), 'explicit parent source group')
        if not isinstance(group['path_id'], str) or not group['path_id']:
            raise ValueError('Explicit parent source group needs a native path_id')
        key = _integer(group['pts'], 'explicit parent group pts'), group['path_id']
        ids = group['parent_observation_ids']
        if key not in incoming or key in result or not isinstance(ids, list):
            raise ValueError('Explicit parent group must uniquely identify nonempty incoming observations')
        for observation_id in ids:
            if not isinstance(observation_id, str) or not observation_id:
                raise ValueError('Explicit replacement parent observation_id must be a nonempty string')
            if observation_id in requested:
                raise ValueError('Duplicate explicit replacement parent observation_id')
            requested.add(observation_id)
        result[key] = list(ids)
    if set(result) != set(incoming):
        raise ValueError('Explicit parent policy must cover every incoming source group')
    # Retain matches only for declared IDs, as in source-row corrections. A
    # duplicate identity in the current parent remains ambiguous and is refused.
    matches = {observation_id: [] for observation_id in requested}
    if requested:
        for index, observation_id in enumerate(old_points['observation_ids']):
            found = matches.get(observation_id)
            if found is not None:
                found.append(index)
    for key, ids in result.items():
        for observation_id in ids:
            indices = matches[observation_id]
            if len(indices) != 1:
                raise ValueError('Explicit replacement must identify exactly one current parent observation_id')
            index = indices[0]
            owners = [shot for shot in manifest['shots'] if shot['row_begin'] <= index < shot['row_stop']]
            if len(owners) != 1 or owners[0]['id'] != shot_id:
                raise ValueError('Explicit replacement parent observation belongs to another shot')
            old_key = int(old_points['pts'][index]), old_mapping['path_ids'][int(old_points['path_indices'][index])]
            if old_key != key:
                raise ValueError('Explicit replacement parent observation PTS/path differs from incoming group')
            if int(old_points['kind'][index]) not in incoming[key]:
                raise ValueError('Explicit replacement parent observation kind differs from incoming group')
    return result


def authenticate_replacement_spans(mapping, points):
    """Partial straight-line updates may retire only actually remeasured spans.

    Endpoints must be original incoming observations, never projected or
    extended coordinates. Empty space between separate branches stays intact.
    Other groups retain the existing complete-group replacement contract.
    """
    declaration = mapping.get('source_replacement')
    if declaration is None:
        return {}
    _exact_fields(declaration, ('policy', 'groups'), 'source replacement')
    if declaration['policy'] != 'partial_line_spans' or not isinstance(declaration['groups'], list):
        raise ValueError('Unsupported partial source replacement policy')
    groups = defaultdict(set)
    for pt, path, uv in zip(points['pts'], points['path_indices'], points['uv']):
        groups[int(pt), mapping['path_ids'][int(path)]].add(tuple(map(float, uv)))
    result = {}
    for group in declaration['groups']:
        _exact_fields(group, ('pts', 'path_id', 'spans'), 'partial source group')
        key = _integer(group['pts'], 'partial group pts'), group['path_id']
        if key not in groups or key in result or not isinstance(group['spans'], list) or not group['spans']:
            raise ValueError('Partial source group must uniquely identify nonempty incoming observations')
        spans = []
        for span in group['spans']:
            uv = np.asarray(span, dtype=float)
            if uv.shape != (2, 2) or not np.isfinite(uv).all():
                raise ValueError('Partial source span needs two actual finite UV endpoints')
            if any(tuple(point) not in groups[key] for point in uv):
                raise ValueError('Partial source span endpoint is not an actual incoming observation')
            spans.append(uv)
        if not all(in_remeasured_line_spans(uv, spans) for uv in groups[key]):
            raise ValueError('Partial source spans do not cover every incoming source observation')
        result[key] = spans
    return result


def in_remeasured_line_spans(uv, spans):
    """Use source tangential extent, without inventing an output pixel or weight."""
    uv = np.asarray(uv, float)
    for start, end in spans:
        direction = end-start
        length2 = float(direction@direction)
        if length2 == 0:
            if np.array_equal(uv, start):
                return True
        else:
            along = float((uv-start)@direction)
            # Floating point endpoint roundoff only; no spatial extrapolation.
            epsilon = 16*np.finfo(float).eps*max(1., length2)
            if -epsilon <= along <= length2+epsilon:
                return True
    return False


def replace_source_groups(original, state, manifest, mapping, packets, corrections=None):
    """Merge groups in shot blocks, retaining original rows and numeric origins."""
    path_ids = list(mapping['path_ids']); lookup = {path:i for i,path in enumerate(path_ids)}
    incoming, group_owner, group_counts, partial_spans = defaultdict(list), {}, Counter(), {}
    explicit_parent_ids, explicit_parent_sets = {}, {}
    for packet_index, packet in enumerate(packets):
        z = {key:value.copy() for key,value in packet['points'].items()}
        for path in packet['mapping']['path_ids']:
            if path not in lookup:
                lookup[path] = len(path_ids); path_ids.append(path)
        z['path_indices'] = np.array([lookup[packet['mapping']['path_ids'][i]] for i in z['path_indices']],
                                    dtype=original['path_indices'].dtype)
        groups = set(zip(map(int, z['pts']), map(int, z['path_indices'])))
        declaration = packet['mapping'].get('source_replacement')
        explicit = isinstance(declaration, dict) and declaration.get('policy') == 'explicit_parent_observation_ids'
        declared_ids = packet.get('replacement_parent_observation_ids')
        if explicit:
            expected = {(pts, path_ids[path]) for pts, path in groups}
            if declared_ids is None or set(declared_ids) != expected:
                raise ValueError('Explicit parent replacement groups must be authenticated before merge')
            if packet['mapping']['source_replacement']['parent_point_input_sha256'] != manifest['point_input_sha256']:
                raise ValueError('Explicit replacement parent point input SHA256 differs at merge')
        for pts, path in groups:
            key = packet['shot_id'], pts, path
            if key in group_owner:
                raise ValueError('Multiple packets claim the same replacement group')
            group_owner[key] = packet_index
            if explicit:
                explicit_parent_ids[key] = tuple(declared_ids[pts, path_ids[path]])
                explicit_parent_sets[key] = set(explicit_parent_ids[key])
            declared = packet.get('replacement_spans', {}).get((pts, path_ids[path]))
            if declared is not None:
                partial_spans[key] = declared
        group_counts.update((packet['shot_id'], int(pt), int(path)) for pt,path in zip(z['pts'],z['path_indices']))
        incoming[packet['shot_id']].append((packet_index, z))
    chunks, origins, station_origins, packet_indices, packet_rows, numerics, shots = [], [], [], [], [], [], []
    withdrawn = set(map(int, corrections['parent_rows'])) if corrections is not None else set()
    removed, replaced, changes = [], [], []
    offset = 0
    for shot in manifest['shots']:
        a, b = shot['row_begin'], shot['row_stop']
        old_indices = np.arange(a, b, dtype=np.int64)
        supplied = incoming.get(shot['id'], [])
        groups = {key[1:] for key in group_owner if key[0] == shot['id']}
        def replaces(i):
            key = shot['id'], int(original['pts'][i]), int(original['path_indices'][i])
            if key not in group_owner:
                return False
            if key in explicit_parent_ids:
                return str(original['observation_ids'][i]) in explicit_parent_sets[key]
            spans = partial_spans.get(key)
            return spans is None or in_remeasured_line_spans(original['uv'][i], spans)
        packet_retired = np.fromiter((replaces(i) for i in old_indices), dtype=bool, count=len(old_indices))
        retired = packet_retired | np.array([int(i) in withdrawn for i in old_indices], dtype=bool)
        kept = old_indices[~retired]; lost = old_indices[retired]; removed.extend(map(int,lost))
        packet_lost = old_indices[packet_retired]; replaced.extend(map(int,packet_lost))
        old_groups = defaultdict(list)
        for i in packet_lost:
            old_groups[int(original['pts'][i]), int(original['path_indices'][i])].append(int(i))
        cache = defaultdict(list)
        for i in packet_lost:
            if int(i) not in withdrawn:
                cache[_pixel_key(original, i)].append(int(i))
        chunks.append({key:value[kept] for key,value in original.items()})
        origins.append(kept); station_origins.append(kept)
        packet_indices.append(np.full(len(kept), -1, np.int32)); packet_rows.append(np.full(len(kept), -1, np.int64))
        numerics.append(state['stations'][kept].copy())
        count = len(kept)
        for packet_index, z in supplied:
            numeric = z['stations'].astype(state['stations'].dtype, copy=True)
            numeric_parent = np.full(len(z['pts']), -1, np.int64)
            for i in range(len(z['pts'])):
                candidates = cache.get(_pixel_key(z, i), [])
                if candidates:
                    values = state['stations'][candidates]
                    if all(value.tobytes() == values[0].tobytes() for value in values):
                        numeric[i] = values[0]; numeric_parent[i] = candidates[0]
            chunks.append(z); numerics.append(numeric)
            origins.append(np.full(len(z['pts']), -1, np.int64)); station_origins.append(numeric_parent)
            packet_indices.append(np.full(len(z['pts']), packet_index, np.int32)); packet_rows.append(np.arange(len(z['pts']), dtype=np.int64))
            count += len(z['pts'])
        info = copy.deepcopy(shot); info.update(row_begin=offset, row_stop=offset+count)
        shots.append(info); offset += count
        for pts, path in sorted(groups):
            key = shot['id'], pts, path
            changes.append(dict(shot_id=shot['id'], pts=pts, native_path_id=path_ids[path],
                parent_row_indices=old_groups.get((pts,path), []), old_rows=len(old_groups.get((pts,path), [])),
                incoming_rows=group_counts[key], packet_index=group_owner[key],
                partial_line_spans=([span.tolist() for span in partial_spans[key]] if key in partial_spans else None),
                action=('replace_remeasured_line_spans' if key in partial_spans else
                        ('replace_measured_group' if (pts,path) in old_groups else 'add_measured_group'))))
            if key in explicit_parent_ids:
                changes[-1].update(source_replacement_policy='explicit_parent_observation_ids',
                    parent_point_input_sha256=manifest['point_input_sha256'],
                    parent_observation_ids=list(explicit_parent_ids[key]),
                    action=('replace_explicit_parent_observation_ids' if explicit_parent_ids[key]
                            else 'append_explicit_parent_observation_ids'))
    new = {key:np.concatenate([z[key] for z in chunks]) for key in prepared.FIELDS}
    if len(np.unique(new['observation_ids'])) != len(new['pts']):
        raise ValueError('Updated observation IDs collide across source groups')
    numeric, parent, numeric_parent = map(np.concatenate, (numerics, origins, station_origins))
    retained = np.flatnonzero(parent >= 0)
    for start in range(0, len(retained), 32768):
        rows = retained[start:start+32768]; old = parent[rows]
        for key in prepared.FIELDS:
            _same_bytes(new[key][rows], original[key][old], 'unprovided source group '+key)
        _same_bytes(numeric[rows], state['stations'][old], 'unprovided source group stations')
    result_mapping = copy.deepcopy(mapping); result_mapping['path_ids'] = path_ids
    return dict(points=new, numeric=numeric, shots=shots, mapping=result_mapping,
        parent_rows=parent, numeric_parent_rows=numeric_parent,
        packet_indices=np.concatenate(packet_indices), packet_rows=np.concatenate(packet_rows),
        removed=np.asarray(removed,np.int64), replaced=np.asarray(replaced,np.int64),
        withdrawn=np.array(sorted(withdrawn),np.int64), changes=changes)


def bind_current_native_library(manifest, state, override=None):
    """Select only the actual DLL identity already recorded by this resume."""
    resume_hash = str(state.get('native_library_provenance_sha256', ''))
    if re.fullmatch('[0-9a-f]{64}', resume_hash) is None:
        raise ValueError('Resume needs its actual native_library_provenance_sha256')
    declared_hash = manifest['native_library_sha256']
    if override is None and declared_hash != resume_hash:
        raise ValueError('Prepared manifest DLL and resume provenance differ; provide --native-library matching the resume')
    library = Path(override if override is not None else manifest['native_library']).resolve(strict=True)
    actual_hash = prepared.digest(library)
    if actual_hash != resume_hash:
        raise ValueError('Actual native DLL SHA256 differs from resume.native_library_provenance_sha256')
    effective = copy.deepcopy(manifest)
    effective.update(native_library=str(library), native_library_sha256=actual_hash)
    revision = dict(explicit_override=override is not None,
        parent_manifest_library_path=manifest['native_library'], parent_manifest_library_sha256=declared_hash,
        resume_native_library_provenance_sha256=resume_hash,
        effective_library_path=str(library), effective_library_sha256=actual_hash,
        manifest_library_revision_changed=declared_hash != actual_hash,
        numeric_state_provenance_rewritten=False, numeric_state_parameters_changed=False)
    return effective, library, revision


def prepare(args):
    root, resume, out = [Path(p).resolve() for p in (args.run_root,args.resume,args.output)]
    out = guard_project_write(out)
    if out.exists() or root.is_relative_to(out):
        raise FileExistsError('Source update output must be new and separate')
    if out.is_relative_to(prepared.ROOT.resolve()):
        raise ValueError('Prepared point arrays belong outside OneDrive')
    packet_paths = list(getattr(args, 'point_packets', None) or [])
    correction_path = getattr(args, 'source_corrections', None)
    if not packet_paths and correction_path is None:
        raise ValueError('Provide point packets or explicit source corrections')
    manifest, mapping = prepared.read(root/'joint_manifest.json'), prepared.read(root/'mapping.json')
    protected = {str(p):prepared.digest(p) for p in
        (root/'joint_manifest.json',root/'mapping.json',root/'point_observations.npz',resume)}
    if protected[str(root/'point_observations.npz')] != manifest['point_input_sha256']:
        raise ValueError('Current PreparedProblem input digest differs')
    state = prepared.npz(resume)
    manifest, library_path, library_revision = bind_current_native_library(
        manifest, state, getattr(args, 'native_library', None))
    scene_path = Path(manifest['scene_path']).resolve(strict=True)
    if prepared.digest(scene_path) != manifest['scene_file_sha256']:
        raise ValueError('Current scene dependency differs')
    scene = prepared.read(scene_path); protected.update(dependency_files(manifest,scene_path,library_path,scene))
    model = SceneParameterModel(scene,scene_path,optimize_scene=True)
    original = prepared.npz(root/'point_observations.npz')
    validate_points(original,mapping)
    prepared.validate_snapshot(state,model.initial_state.layout,mapping,manifest,original)
    packets = [authenticate_packet(p,manifest,mapping,original) for p in packet_paths]
    corrections = (authenticate_source_corrections(correction_path,manifest,mapping,original)
                   if correction_path is not None else None)
    merged = replace_source_groups(original,state,manifest,mapping,packets,corrections)
    for packet in packets: protected.update(packet['protected'])
    if corrections is not None: protected.update(corrections['protected'])
    new, numeric, shots = merged['points'], merged['numeric'], merged['shots']
    coverage = {}; step = int(manifest.get('source_clock',{}).get('native_pts_step',256))
    for shot in shots:
        window = dict(first_pts=shot['first_pts'],last_pts=shot['last_pts'],stop_pts_exclusive=shot['stop_pts_exclusive'],
            native_frame_count=(shot['last_pts']-shot['first_pts'])//step+1)
        coverage[shot['id']] = source_coverage(new['pts'][shot['row_begin']:shot['row_stop']],window,step,args.allow_sparse_bootstrap)
        shot['source_coverage'] = coverage[shot['id']]
    timeline = ShotMotionTimeline(shots,manifest['control_dt'],manifest['first_control_time'])
    timeline.assert_support_separation(new['pts'],manifest['max_readout'])
    touched = (merged['numeric_parent_rows'] < 0) & (new['kind'] == 1)
    initialization = {}
    if packets:
        native = NativeSceneGPU(scene_path,NativeCore(library_path),
            parameter_state=ParameterState(model.initial_state.layout,state['scene_values']))
        if native.scene_revision != manifest['scene_revision']:
            raise ValueError('Current numeric native scene revision differs')
        used = {path for p in packets for path in p['mapping']['path_ids']}
        if used-native.paths.keys():
            raise ValueError('New source path absent from the current numeric native scene')
        for shot in shots:
            rows = np.arange(shot['row_begin'],shot['row_stop']); rows = rows[touched[rows]]
            if not len(rows):continue
            camera = dict(rig=state['rig'][shot['camera_index']].tolist(),reference_pts=shot['reference_pts'],
                time_base_num=shot['time_base_num'],time_base_den=shot['time_base_den'],
                first_control_time=shot['first_control_time'],control_dt=manifest['control_dt'],image_height=1080)
            seeded,note = initialize_curve_stations(native,camera=camera,
                controls=state['controls'][shot['control_begin']:shot['control_stop']],
                path_ids=[merged['mapping']['path_ids'][i] for i in new['path_indices'][rows]],
                stations=numeric[rows],pts=new['pts'][rows],observed=new['uv'][rows],kinds=new['kind'][rows],
                half_seconds=manifest['max_readout']/2)
            if note['unresolved']:
                failure=out.with_name(out.name+'_station_failure.json')
                prepared.write(failure,dict(shot=shot['id'],initialization=note,
                    input_root=str(root),resume=str(resume),source_rows_unchanged=True))
                raise ValueError('New source station initialization unresolved: '+str(failure))
            numeric[rows] = seeded; initialization[shot['id']] = dict(note,initialized_rows=len(rows))
    inherited = np.flatnonzero(merged['numeric_parent_rows'] >= 0)
    _same_bytes(numeric[inherited],state['stations'][merged['numeric_parent_rows'][inherited]],'resume numeric stations')
    _same_bytes(numeric[new['kind']==0],new['stations'][new['kind']==0],'fixed source stations')
    provenance = dict(schema='rinkspace.current_source_group_update.v1',parent_root=str(root),parent_resume=str(resume),
        parent_input_sha256=manifest['point_input_sha256'],parent_resume_sha256=protected[str(resume)],
        native_library_revision=library_revision,
        packet_inputs=[dict(path=str(p['folder']),shot_id=p['shot_id'],authenticated_rows=p['authenticated_rows'],
            source_capture=p['source_capture'],source_capture_sha256=p['source_capture_sha256'],fit_eligible=True)for p in packets],
        changed_groups=len(merged['changes']),replaced_groups=sum(c['old_rows']>0 for c in merged['changes']),
        appended_groups=sum(c['old_rows']==0 for c in merged['changes']),retired_rows=len(merged['removed']),
        packet_replaced_rows=len(merged['replaced']),withdrawn_rows=len(merged['withdrawn']),
        withdrawn_rows_also_replaced_by_packets=len(np.intersect1d(merged['withdrawn'],merged['replaced'])),
        source_corrections=(dict(path=corrections['receipt']['path'],sha256=corrections['receipt']['sha256'],
            archived_document='source_corrections.json',receipt='source_corrections_receipt.json')
            if corrections is not None else None),
        retained_parent_rows=int(np.sum(merged['parent_rows']>=0)),new_source_rows=int(np.sum(merged['parent_rows']<0)),
        reused_resume_stations_for_same_source_pixels=int(np.sum((merged['parent_rows']<0)&(merged['numeric_parent_rows']>=0))),
        native_initialized_new_kind1_rows=int(touched.sum()),station_initialization=initialization,
        allow_sparse_bootstrap=bool(args.allow_sparse_bootstrap),source_coverage=coverage,
        partial_source_coverage=any(c['partial_source_coverage']for c in coverage.values()),
        seven_unweighted_fields=True,source_UV_and_PTS_unchanged=True,numeric_globals_byte_preserved=True,
        empty_packet_deletion_allowed=False,fit_executed=False)
    for item, packet in zip(provenance['packet_inputs'], packets):
        declaration = packet['mapping'].get('source_replacement')
        if declaration is not None and declaration['policy'] == 'explicit_parent_observation_ids':
            item['source_replacement'] = copy.deepcopy(declaration)
    updated,out_mapping = _write_prepared(out,new,numeric,shots,state,manifest,merged['mapping'],model,provenance)
    if 'spatial_refinement_preparation' in manifest:
        updated['spatial_refinement_preparation'] = manifest['spatial_refinement_preparation']
    else:updated.pop('spatial_refinement_preparation',None)
    updated['source_group_update'] = provenance
    updated['partial_source_coverage'] = provenance['partial_source_coverage']
    prepared.write(out/'joint_manifest.json',updated)
    saved = prepared.npz(out/'initial_pixels.npz')
    prepared.validate_snapshot(saved,model.initial_state.layout,out_mapping,updated,new)
    for name in state.keys()-{'stations','source_rows','joint_numeric_dependencies_json','point_input_sha256'}:
        _same_bytes(saved[name],state[name],'unchanged current numeric block '+name)
    np.savez(out/'source_origins.npz',parent_row_indices=merged['parent_rows'],
        numeric_station_parent_rows=merged['numeric_parent_rows'],packet_indices=merged['packet_indices'],packet_rows=merged['packet_rows'])
    retired = merged['removed']
    np.savez(out/'replaced_source_observations.npz',**{key:value[retired]for key,value in original.items()},
        parent_row_indices=retired,numeric_stations=state['stations'][retired],
        explicitly_withdrawn=np.isin(retired,merged['withdrawn']),replaced_by_packet=np.isin(retired,merged['replaced']))
    if corrections is not None:
        (out/'source_corrections.json').write_bytes(corrections['raw'])
        prepared.write(out/'source_corrections_receipt.json',corrections['receipt'])
    with(out/'replacement_groups.jsonl').open('w',encoding='utf8')as stream:
        for change in merged['changes']:stream.write(json.dumps(change,ensure_ascii=False)+'\n')
    for path,expected in protected.items():
        if prepared.digest(path)!=expected:raise RuntimeError('Current input/source dependency changed: '+path)
    receipt = dict(provenance,output=str(out),new_input_sha256=updated['point_input_sha256'],
        protected_current_files_sha256=protected,snapshot_bind_and_validate_passed=True)
    prepared.write(out/'source_update_receipt.json',receipt)
    print(json.dumps(dict(output=str(out),changed_groups=provenance['changed_groups'],withdrawn_rows=provenance['withdrawn_rows'],rows=len(new['pts']),
        partial_source_coverage=provenance['partial_source_coverage'],fit_executed=False)),flush=True)
    return receipt


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-root','resume','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--point-packets',type=Path,nargs='+',default=[])
    parser.add_argument('--source-corrections',type=Path,
        help='Explicit source-review withdrawals bound to exact parent rows and hashed source evidence')
    parser.add_argument('--allow-sparse-bootstrap',action='store_true')
    parser.add_argument('--native-library',type=Path,
        help='Actual DLL already bound to resume; required when prepared manifest has a stale library revision')
    prepare(parser.parse_args())
