"""Recheck independent native-interface sources with their own RGB and masks."""
from pathlib import Path
from collections import Counter, defaultdict
import gzip
import json
import hashlib
import numpy as np
from .compact_identity_withhold_v2 import _source_identities, _array
from .io_utils import file_sha256, object_sha256
from .person_context_gate_v1 import build_person_context
from .source_worker import _channels, _remeasure
from .source_worker_v2 import support_pixel_audit


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _ref(path):
    return dict(file=str(Path(path).resolve()), sha256=file_sha256(path))


def _checked(ref):
    if file_sha256(ref['file']) != ref['sha256']:
        raise ValueError('Independent native source evidence changed')
    return Path(ref['file'])


def generate(*, parent_packet, evidence, original_cost_report, output):
    """Recheck all still-eligible rows of one independent F-ring generation.

    Reconstruct its actual +/-3 Gaussian interface profile at the ORIGINAL
    readout query, and its documented +/-12 navigation search. No quartiles,
    measurement positions, person labels, or visible spans are invented.
    """
    import av
    import cv2
    cv2.setNumThreads(1)
    output = Path(output)
    if output.exists():
        raise FileExistsError('Immutable native review output exists')
    ed = _json(evidence)
    rows = _json(_checked(ed['rows']))
    frames = _json(_checked(ed['frames']))
    _checked(ed['reader'])
    _checked(ed['source_profile_reader'])
    _checked(ed['raw_candidate_readings'])
    for ref in ed['review_images']:
        _checked(ref)
    if ed['qualification_uses_model_residual'] is not False:
        raise ValueError('Source-only native readout required')
    doc = _json(parent_packet)
    node = doc
    while node.get('native_ad_source_binding', {}).get('source_dataset_id') != ed['source_dataset_id']:
        node = _json(node['dense_parent_packet']['file'])
    registry = node['native_ad_source_binding']
    indices = np.arange(registry['row_start'], registry['row_start']+registry['count'])
    arrays = {k: _array(doc['arrays'][k]) for k in ('admission', 'fit_role', 'uv', 'native_pts', 'material_index')}
    indices = indices[(arrays['admission'][indices] == 1) & (arrays['fit_role'][indices] == 1)]
    ids, _ = _source_identities(parent_packet, indices)
    raw_by_id = {identity: i for i, identity in ids.items()}
    by_pts = defaultdict(list)
    for row in rows:
        if row['source_row_id'] not in raw_by_id:
            continue
        i = raw_by_id[row['source_row_id']]
        if row['source_uv'] != arrays['uv'][i].tolist() or int(row['native_pts']) != int(arrays['native_pts'][i]) or row['material_index'] != int(arrays['material_index'][i]):
            raise ValueError('Independent source ID/UV/PTS/material differs from current parent')
        if row['physical_surface'] != 'ice' or row['source_model_residual_used'] is not False or row['source_uv'] != row['current_native_profile']['current_uv']:
            raise ValueError('Original actual ice-interface record required')
        by_pts[int(row['native_pts'])].append((i, row))
    if sum(map(len, by_pts.values())) != len(indices):
        raise ValueError('Some eligible independent native rows lack original readings')
    sources = {}
    frame_documents = {}
    frame_map = {int(f['native_pts']): f for f in frames}
    for pt in by_pts:
        frame = frame_map[pt]
        frame_documents[pt] = _json(_checked(frame['source']))
        source = frame_documents[pt]['source']
        _checked(frame['image']); _checked(frame['mask'])
        if source['native_pts'] != pt or source['actual_pts'] != pt or source['time_base'] != '1/12800' or source['mask_sha256'] != frame['mask']['sha256'] or source['png_sha256'] != frame['image']['sha256']:
            raise ValueError('Own original native image/mask binding differs')
        sources[pt] = source
    source_files = {(s['source'], s['source_video_sha256']) for s in sources.values()}
    if len(source_files) != 1:
        raise ValueError('Independent generation mixes original sources')
    video_file, video_sha = source_files.pop()
    if file_sha256(video_file) != video_sha:
        raise ValueError('Independent original video changed')
    output.mkdir(parents=True)
    withdrawals = []
    counts = Counter()
    profiles = []
    max_replay_uv_error = 0.
    max_profile_value_error = 0.
    context_path = output/'contexts.jsonl.gz'
    audit_path = output/'rows.jsonl.gz'
    remaining = set(by_pts)
    with av.open(video_file) as video, gzip.open(context_path, 'wt', encoding='utf-8') as contexts, gzip.open(audit_path, 'wt', encoding='utf-8') as audits:
        stream = video.streams.video[0]
        if str(stream.time_base) != '1/12800':
            raise ValueError('Independent source native clock differs')
        video.seek(min(remaining), stream=stream, backward=True)
        context_line = 0
        for decoded in video.decode(stream):
            pt = int(decoded.pts)
            if pt not in remaining:
                if pt > max(remaining):
                    raise ValueError('Missing exact independent source frame')
                continue
            source = sources[pt]
            frame = frame_map[pt]
            bgr = decoded.to_ndarray(format='bgr24')
            rgb = bgr[:, :, ::-1].copy()
            if hashlib.sha256(bgr.tobytes()).hexdigest() != source['decoded_BGR_sha256'] or hashlib.sha256(rgb.tobytes()).hexdigest() != source['decoded_RGB_sha256']:
                raise ValueError('Own source RGB differs from official video')
            image = cv2.imread(frame['image']['file'])
            if image is None or not np.array_equal(image, bgr):
                raise ValueError('Own original PNG is not the decoded exposure')
            own_record = dict(shot_id='s06', native_pts=pt, source_video_sha256=video_sha,
                source_RGB_sha256=source['decoded_RGB_sha256'], mask_file=frame['mask']['file'],
                mask_sha256=frame['mask']['sha256'], graphics_mask_included=False)
            context = build_person_context(own_record, rgb=rgb)
            contexts.write(json.dumps(context[2], ensure_ascii=False, separators=(',', ':'))+'\n')
            channels = _channels(rgb)[:1]
            for i, row in by_pts[pt]:
                feature = frame_documents[pt]['features'][row['feature_index']]
                segment = feature['segments'][row['segment_index']]
                if (feature['id'] != row['material_label'] or feature['physical_surface'] != 'ice' or feature['measurement_operator'] != 'material_interface' or feature['material_transition'] != 'dark_to_light' or segment['uv'][row['source_point_index']] != row['source_uv']):
                    raise ValueError('Own finite static ice material / point annotation differs')
                original = row['current_native_profile']
                n = np.asarray(original['normal'], float)
                n /= np.linalg.norm(n)
                uv = np.asarray(row['source_uv'], float)
                query = uv-float(original['profile']['offset_px'])*n
                sites = query+np.arange(-3., 3.0001, .25)[:, None]*n
                navigation_sites = np.asarray(row['source_navigation_uv'])+np.arange(-12., 12.0001, .25)[:, None]*n
                person = support_pixel_audit(np.vstack([sites, navigation_sites]), context[0], gaussian_radius=3)
                unknown = support_pixel_audit(np.vstack([sites, navigation_sites]), context[1], gaussian_radius=3)
                replay = _remeasure(query, n, channels, 'material_interface')
                if replay['current_uv'] is None:
                    uv_error = None
                    material_reason = 'actual_native_interface_replay_unresolved'
                else:
                    uv_error = float(np.max(np.abs(np.asarray(replay['current_uv'])-uv)))
                    max_replay_uv_error = max(max_replay_uv_error, uv_error)
                    err = max(abs(float(replay['profile'][k])-float(original['profile'][k])) for k in ('peak_gradient', 'contrast', 'offset_px'))
                    max_profile_value_error = max(max_profile_value_error, err)
                    material_reason = ('native_original_profile_replay_differs' if uv_error > 1e-6 or err > 1e-5 else
                        'actual_native_interface_polarity_differs' if replay['profile']['polarity'] != 1 else None)
                decision = ('unknown_native_image_support' if person['outside_native_image'] else
                    'unknown_existing_person_mask_overlap' if person['person_mask_pixels'] else
                    'unknown_person_component_context' if unknown['person_mask_pixels'] else
                    'unknown_material_interface_support' if material_reason else
                    'clear_of_own_person_context_and_replayed_material_gate')
                counts[decision] += 1
                record = dict(raw_row=int(i), source_row_id=row['source_row_id'], native_pts=pt,
                    source_uv=row['source_uv'], original_source_record=row, original_source_record_sha256=object_sha256(row),
                    source_frame=frame['source'], original_native_image=frame['image'], original_person_mask=frame['mask'],
                    decision=decision, source_context_line_0based=context_line,
                    original_interface_query_uv=query.tolist(), exact_original_profile_replay=replay,
                    replay_UV_error_px=uv_error, original_reader_support=person, unknown_context_support=unknown,
                    material_evidence_unresolved_reason=material_reason, source_UV_changed=False,
                    source_RGB_redecoded_from_official_video=True, dense56_prior_qualification_reused=False,
                    fit_admission_issued=False)
                audits.write(json.dumps(record, ensure_ascii=False, separators=(',', ':'))+'\n')
                if decision.startswith('unknown_'):
                    withdrawals.append(dict(raw_row=int(i), source_row_id=row['source_row_id'], uv=row['source_uv'], native_pts=pt,
                        qualification='unknown', classification=decision, finding='Own original RGB/profile support is unresolved: '+decision,
                        native_images=[frame['image'], frame['mask']], source_frame=frame['source'],
                        original_source_record_sha256=object_sha256(row), source_context_line_0based=context_line))
                if replay.get('profile'):
                    profiles.append(float(replay['profile']['contrast']))
            context_line += 1
            remaining.remove(pt)
            if not remaining:
                break
    if remaining:
        raise ValueError('Official decode omitted independent source frames')
    context_ref = _ref(context_path)
    for row in withdrawals:
        row['source_context_ref'] = context_ref
    review = dict(schema='rinkfit.native_material_identity_review.v1', qualification_uses_model_residual=False,
        native_frame_extractor=_ref(__file__), parent_packet=_ref(parent_packet), original_source_evidence=_ref(evidence),
        original_cost_reference=_ref(original_cost_report), source_context=context_ref, source_audit=_ref(audit_path),
        rows=withdrawals, confirmed_nonmaterial_rows=0, unknown_rows=len(withdrawals),
        raw_rows_deleted=0, original_source_UV_ID_PTS_E_weights_changed=False, packet_installed=False)
    review_path = output/'review.json'
    review_path.write_text(json.dumps(review, ensure_ascii=False, separators=(',', ':'))+'\n', encoding='utf-8')
    result = dict(schema='rinkfit.independent_native_person_context_review.v1', parent_packet=review['parent_packet'],
        source_dataset_id=ed['source_dataset_id'], original_rows=len(rows), eligible_rows_reviewed=len(indices),
        native_exposures=len(by_pts), native_pts_range=[min(by_pts), max(by_pts)],
        source_seconds=[min(by_pts)/12800, max(by_pts)/12800], decisions=dict(counts),
        confirmed_nonmaterial_rows=0, unknown_rows=len(withdrawals), review=_ref(review_path),
        actual_interface_contrast_min=min(profiles) if profiles else None,
        actual_interface_contrast_max=max(profiles) if profiles else None,
        maximum_original_profile_replay_UV_error_px=max_replay_uv_error,
        maximum_original_profile_replay_value_error=max_profile_value_error,
        all_own_RGB_images_and_original_masks_verified=True, no_new_images_or_large_arrays=True,
        previous_dense56_qualification_assumed=False, original_records_retained=True,
        no_delta_to_install=not withdrawals, packet_installed=False, solver_started=False)
    (output/'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return result
