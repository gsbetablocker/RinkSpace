"""Commit independently replayed native material readouts as a new source generation.

The previous packet, source identities and failed readings remain immutable.
This transaction never turns a fit residual or projected model into source data.
"""
from collections import defaultdict
from pathlib import Path
import gzip
import hashlib
import json
import numpy as np
import av
import cv2
from .compact_identity_withhold_v2 import _source_identities
from .dense_joint_proofs import _array, _checked, _ref, _write
from .native_print_track_readout import batch_interfaces, clear_profile
from .source_worker import _channels

REVIEW = 'rinkfit.native_source_measurement_revision_review.v1'
QUALIFICATION = 'rinkfit.native_source_measurement_revision_qualification.v1'


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def validate_review(*, parent_packet, review):
    """Replay the declared native profiles without writing a packet or evidence."""
    pinned = {}
    def pin(ref, base):
        actual = Path(ref['file'])
        actual = (actual if actual.is_absolute() else Path(base)/actual).resolve()
        key = (str(actual).casefold(), ref['sha256'])
        if key not in pinned:
            pinned[key] = _checked(ref, base)
        return pinned[key]
    packet_path = Path(parent_packet).resolve()
    review_path = Path(review).resolve()
    parent = _json(packet_path)
    doc = _json(review_path)
    if (doc.get('schema') != REVIEW or
            doc.get('source_model_projection_used') is not False or
            doc.get('source_images_changed') is not False or
            doc.get('measured_uv_changed') is not True or
            pin(doc['parent_packet'], review_path.parent) != packet_path or
            not doc.get('evidence') or not doc.get('rows')):
        raise ValueError('Exact source-only native revision review required')
    cfg_path = pin(doc['source_config'], review_path.parent)
    cfg = _json(cfg_path)
    if cfg['source_video_sha256'] != 'e8b53966d56e72f3842d92c3ecbe6e9cc7e4670b7d0c9bd12f58fd2cff5ad57d':
        raise ValueError('Official native programme source changed')
    coverage_ref = doc.get('source_coverage')
    if coverage_ref is None:
        raise ValueError('The exact native coverage must have an explicit hash binding')
    coverage_path = pin(coverage_ref, review_path.parent)
    if coverage_path != Path(cfg['source_coverage']).resolve():
        raise ValueError('Coverage differs from the declared source reader')
    dependencies = [doc['parent_packet'], doc['source_config'], coverage_ref]
    dependencies.extend(doc['evidence'])
    for item in doc['rows']:
        if (not item.get('physical_identity_finding') or not item.get('physical_identity_evidence') or
                not item.get('material_label')):
            raise ValueError('A current material identity finding and its source evidence are required')
        dependencies.extend(item['physical_identity_evidence'])
    for ref in dependencies:
        pin(ref, review_path.parent)
    arrays = {k: _array(parent['arrays'][k], packet_path.parent) for k in
              ('uv', 'native_pts', 'admission', 'fit_role', 'kind', 'material_index', 'path',
               'query_uv', 'profile_normal', 'alpha', 'alpha_bounds')}
    indices = [int(item['raw_row']) for item in doc['rows']]
    if len(indices) != len(set(indices)) or any(i < 0 or i >= parent['row_count'] for i in indices):
        raise ValueError('Distinct existing source rows are required')
    identities, _ = _source_identities(packet_path, indices)
    by_pts = defaultdict(list)
    for item in doc['rows']:
        i = int(item['raw_row'])
        if (identities[i] != item['source_row_id'] or
                arrays['uv'][i].tolist() != item['old_uv'] or
                int(arrays['native_pts'][i]) != item['native_pts'] or
                arrays['admission'][i] != 1 or arrays['fit_role'][i] != 1 or
                arrays['kind'][i] != 0 or arrays['alpha'][i] != 0 or
                np.any(arrays['alpha_bounds'][i] != 0) or
                item.get('measurement_operator') != 'material_interface'):
            raise ValueError('Only exact active native sliding material profiles may be revised')
        if int(item.get('old_material_index', arrays['material_index'][i])) != int(arrays['material_index'][i]):
            raise ValueError('Original physical material differs')
        if int(item.get('old_path', arrays['path'][i])) != int(arrays['path'][i]):
            raise ValueError('Original finite path differs')
        item = dict(item, old_material_index=int(arrays['material_index'][i]),
                    old_path=int(arrays['path'][i]))
        item.setdefault('new_material_index', item['old_material_index'])
        item.setdefault('new_path', item['old_path'])
        if not isinstance(item['new_path'], int) or item['new_path'] < 0:
            raise ValueError('Invalid finite material path')
        by_pts[int(item['native_pts'])].append(item)
    coverage = {}
    with gzip.open(coverage_path, 'rt', encoding='utf-8') as stream:
        for frame in map(json.loads, stream):
            if int(frame['native_pts']) in by_pts:
                coverage[int(frame['native_pts'])] = frame
    if set(coverage) != set(by_pts):
        raise ValueError('Every revised exposure needs its original native RGB and mask binding')
    from calibration.program_tracking.extract_first_shot_ice import current_broadcast_exclusion
    import rinkfit_joint.source_worker as channel_reader
    import rinkfit_joint.native_print_track_readout as profile_reader
    import calibration.program_tracking.extract_first_shot_ice as mask_reader
    dependencies.extend(_ref(module.__file__) for module in (channel_reader, profile_reader, mask_reader))
    dependencies.append(_ref(__file__))
    video_ref = dict(file=str(Path(cfg['source_video']).resolve()), sha256=cfg['source_video_sha256'])
    pin(video_ref, cfg_path.parent)
    dependencies.append(video_ref)
    validated = []
    frames = []
    cv2.setNumThreads(1)
    with av.open(cfg['source_video']) as container:
        stream = container.streams.video[0]
        if str(stream.time_base) != '1/12800':
            raise ValueError('Original native PTS clock changed')
        for pts, items in sorted(by_pts.items()):
            container.seek(pts, stream=stream, backward=True)
            image = None
            for frame in container.decode(stream):
                if frame.pts == pts:
                    image = frame.to_ndarray(format='bgr24')
                    break
                if frame.pts > pts:
                    break
            if image is None or image.shape != (1080, 1920, 3):
                raise ValueError('Missing exact native exposure ' + str(pts))
            digest = hashlib.sha256(image.tobytes()).hexdigest()
            source = coverage[pts]
            if digest != source['source_BGR_sha256']:
                raise ValueError('The actual decoded native RGB differs')
            masks = [source['person_mask']]
            if source.get('unknown_mask'):
                masks.append(source['unknown_mask'])
            mask = np.zeros(image.shape[:2], np.uint8)
            for ref in masks:
                mask_path = pin(ref, coverage_path.parent)
                pixels = cv2.imread(str(mask_path), 0)
                if pixels is None or pixels.shape != mask.shape:
                    raise ValueError('Missing exact native source exclusion mask')
                mask = cv2.bitwise_or(mask, pixels)
                dependencies.append(ref)
            graphics, _ = current_broadcast_exclusion(image)
            mask[graphics] = 255
            navigation = np.asarray([item['navigation_uv'] for item in items], float)
            normals = np.asarray([item['normal'] for item in items], float)
            if navigation.shape != (len(items), 2) or normals.shape != navigation.shape:
                raise ValueError('Invalid actual finite source profile query')
            fresh = batch_interfaces(navigation, normals, _channels(image[:, :, ::-1]))
            supported = fresh['valid'] & clear_profile(fresh['uv'], fresh['normal'], mask)
            for k, item in enumerate(items):
                expected_uv = np.asarray(item['current_uv'], float)
                expected_profile = np.asarray(item['profile'], float)
                if (item['native_BGR_sha256'] != digest or not supported[k] or
                        expected_uv.shape != (2,) or expected_profile.shape != (5,) or
                        not np.isfinite(np.r_[expected_uv, expected_profile]).all() or
                        not np.allclose(fresh['uv'][k], expected_uv, rtol=0, atol=1e-6) or
                        not np.allclose(fresh['profile'][k], expected_profile, rtol=0, atol=1e-6) or
                        not np.allclose(fresh['normal'][k], normals[k], rtol=0, atol=1e-12)):
                    raise ValueError('Actual RGB profile or current visibility did not replay: ' + str(item['raw_row']))
                validated.append(dict(item, current_uv=fresh['uv'][k].tolist(),
                    normal=fresh['normal'][k].tolist(), profile=fresh['profile'][k].tolist(),
                    actual_RGB_profile_replay_max_abs_px=float(np.max(np.abs(fresh['uv'][k] - expected_uv))),
                    original_source_row_retained=True))
            frames.append(dict(native_pts=pts, source_BGR_sha256=digest,
                person_mask=source['person_mask'], unknown_mask=source.get('unknown_mask'),
                reviewed_rows=len(items)))
    if not any(item['old_uv'] != item['current_uv'] for item in validated):
        raise ValueError('A source measurement revision must contain an actual UV change')
    unique = {}
    for ref in dependencies:
        actual = pin(ref, review_path.parent)
        unique[(str(actual).casefold(), ref['sha256'])] = dict(file=str(actual), sha256=ref['sha256'])
    return dict(parent=parent, review=doc, rows=sorted(validated, key=lambda e:e['raw_row']),
                frames=frames, actual_dependencies=list(unique.values()),
                actual_native_RGB_replayed=True, source_model_projection_used=False)


def install(*, parent_packet, parent_qualification, parent_weights, review, output):
    """Install one reviewed source generation; the native guard verifies the delta."""
    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError('Use a new immutable source generation')
    checked = validate_review(parent_packet=parent_packet, review=review)
    parent = checked.pop('parent')
    checked.pop('review')
    q = _json(parent_qualification)
    w = _json(parent_weights)
    parent_ref = _ref(parent_packet)
    if q['source_packet'] != parent_ref or w['source_packet'] != parent_ref:
        raise ValueError('Qualification and engineering budgets must bind this exact source generation')
    if w['dense_source_qualification'] != _ref(parent_qualification):
        raise ValueError('Engineering budgets bind another qualification')
    arrays = dict(parent['arrays'])
    for key, ref in arrays.items():
        if q['membership_array_sha256'][key] != ref['sha256']:
            raise ValueError('Qualification array membership differs')
    old = {key:_array(arrays[key], Path(parent_packet).parent) for key in
           ('uv', 'query_uv', 'profile_normal', 'row', 'material_index', 'path')}
    changed = {}
    for item in checked['rows']:
        i = item['raw_row']
        for key, value in [('uv', item['current_uv']), ('query_uv', item['current_uv']),
                ('profile_normal', item['normal']), ('row', item['current_uv'][1]),
                ('material_index', item['new_material_index']), ('path', item['new_path'])]:
            if not np.array_equal(old[key][i], value):
                if key not in changed:
                    changed[key] = np.array(old[key], copy=True)
                changed[key][i] = value
    if not changed:
        raise ValueError('No actual source measurement or binding changed')
    out.mkdir(parents=True, exist_ok=False)
    for key, values in changed.items():
        file = out / (key + '.bin')
        values.tofile(file)
        arrays[key] = dict(_ref(file), dtype=values.dtype.str, shape=list(values.shape),
                           layout='C', byte_order='little')
    generation = dict(schema='rinkfit.native_source_measurement_revision.v1',
        parent_packet=parent_ref, parent_qualification=_ref(parent_qualification),
        review=_ref(review), issuer=_ref(__file__), rows=checked['rows'], frames=checked['frames'],
        actual_dependencies=checked['actual_dependencies'], changed_arrays=sorted(changed),
        revised_rows=[e['raw_row'] for e in checked['rows']], raw_rows=parent['row_count'],
        source_rows_deleted=0, original_source_UV_preserved_in_parent=True,
        actual_native_RGB_replayed=True, source_model_projection_used=False,
        source_noise_W_or_sigma_issued=False, source_images_changed=False,
        measured_uv_changed=True)
    revision = _write(out/'revision.json', generation)
    packet = _write(out/'packet.json', dict(schema=parent['schema'], dense_parent_packet=parent_ref,
        row_count=parent['row_count'], arrays=arrays, active_rows=parent['active_rows'],
        admitted_rows=parent['admitted_rows'], native_source_measurement_revision=revision,
        dense_parent_append=dict(source_manifests=[revision, _ref(review), _ref(__file__)])))
    qualification = dict(schema=QUALIFICATION, source_packet=packet, parent_packet=parent_ref,
        parent_qualification=_ref(parent_qualification), issuer=_ref(__file__), revision=revision,
        raw_rows=parent['row_count'], revised_rows=len(checked['rows']),
        membership_array_sha256={key:ref['sha256'] for key,ref in arrays.items()},
        original_source_UV_preserved_in_parent=True, source_rows_deleted=0,
        source_noise_W_or_sigma_issued=False, source_images_changed=False,
        measured_uv_changed=True, quality_accepted=False)
    qr = _write(out/'source_qualification.json', qualification)
    w.update(source_packet=packet, dense_source_qualification=qr,
             source_measurement_revision=revision)
    wr = _write(out/'engineering_weights.json', w)
    return dict(packet=packet, qualification=qr, weights=wr, revision=revision,
                revised_rows=len(checked['rows']), changed_arrays=sorted(changed),
                source_rows_deleted=0, quality_accepted=False)
