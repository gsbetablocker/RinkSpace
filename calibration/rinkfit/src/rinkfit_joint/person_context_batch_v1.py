"""Produce compact qualification withdrawals from actual same-exposure contexts.

All raw observations, original source streams, costs and failed readings remain
immutable. Hull pixels are UNKNOWN context, never new person segmentation.
"""
from pathlib import Path
from collections import Counter, defaultdict
import gzip
import json
import re
import hashlib
import numpy as np
from .compact_identity_withhold_v2 import _source_identities
from .io_utils import file_sha256, object_sha256
from .person_context_gate_v1 import build_person_context, review_row
from .source_record_store import SourceRecordStore


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _ref(path):
    return dict(file=str(Path(path).resolve()), sha256=file_sha256(path))


def generate(*, parent_packet, model, mask_index, source_binding, source_video,
             source_video_sha256, prior_cost_report, confirmed_review, output):
    """Review every eligible dense56 F1-F4 row in s05/s06; write no packet.

    Output review.json is accepted by compact_identity_withhold_v2.withhold.
    Existing mask PNGs plus decoded original RGB hashes support UNKNOWN context
    findings. Only prior independently reviewed RGB findings become confirmed
    nonmaterial classifications; no other hull pixel receives that label.
    """
    import av
    import cv2
    cv2.setNumThreads(1)
    output = Path(output)
    if output.exists():
        raise FileExistsError('Immutable context review output already exists')
    doc = _json(parent_packet)
    model_doc = _json(model)
    a = lambda key: np.memmap(doc['arrays'][key]['file'], mode='r',
        dtype=doc['arrays'][key]['dtype'], shape=tuple(doc['arrays'][key]['shape']),
        offset=doc['arrays'][key].get('offset_bytes', 0))
    arrays = {k: a(k) for k in ('admission', 'fit_role', 'material_index', 'shot_index', 'native_pts', 'uv')}
    path = Path(parent_packet)
    dense = None
    while dense is None:
        node = _json(path)
        dense = node.get('dense_source_binding')
        if dense is None:
            path = Path(node['dense_parent_packet']['file'])
    wanted_materials = {i: m['label'] for i, m in enumerate(model_doc['materials'])
                        if re.fullmatch(r'ice\.F[1-4]:(inner|middle|outer)', m['label'])}
    shot_ids = node['shot_ids']
    shot_indices = [shot_ids.index(s) for s in ('s05', 's06')]
    target = (arrays['admission'] == 1) & (arrays['fit_role'] == 1)
    target &= np.isin(arrays['material_index'], list(wanted_materials))
    target &= np.isin(arrays['shot_index'], shot_indices)
    all_family_rows = int(target.sum())
    outside = np.flatnonzero(target & ((np.arange(doc['row_count']) < dense['row_start']) |
        (np.arange(doc['row_count']) >= dense['row_start']+dense['count'])))
    target[outside] = False
    raw_rows = np.flatnonzero(target)
    identities, _ = _source_identities(parent_packet, raw_rows)
    by_id = {identities[int(i)]: int(i) for i in raw_rows}
    if len(by_id) != len(raw_rows):
        raise ValueError('Dense F source identities are ambiguous')
    binding = _json(source_binding)
    parts = [p for p in binding['source_parts'] if p['name'] in ('s05', 's06')]
    subjects = {}
    for part in parts:
        ref = part['reference']
        with SourceRecordStore(ref['store'], readonly=True) as store:
            store.db.execute('BEGIN')
            info = store.stream_info(ref['stream'])
            if info['kind'] != 'selected_rows' or info['semantic_sha256'] != ref['stream_semantic_sha256']:
                raise ValueError('Original selected-source stream changed')
            for row in store.iterate_records(ref['stream']):
                i = by_id.get(row['selected_row_id'])
                if i is None:
                    continue
                if (row['uv'] != arrays['uv'][i].tolist() or int(row['native_pts']) != int(arrays['native_pts'][i]) or
                    row['shot_id'] != shot_ids[int(arrays['shot_index'][i])] or
                    row['feature_id'] != wanted_materials[int(arrays['material_index'][i])]):
                    raise ValueError('Exact selected source / packet identity differs')
                if i in subjects:
                    raise ValueError('Repeated selected source')
                subjects[i] = dict(raw_row=i, source_row_id=row['selected_row_id'],
                    selected_record=row, selected_record_sha256=object_sha256(row), source_stream=ref)
            store.db.execute('COMMIT')
    print(json.dumps(dict(phase='selected_source_records_bound', rows=len(subjects)), ensure_ascii=False), flush=True)
    if set(subjects) != set(map(int, raw_rows)):
        raise ValueError('Some same-source F rows lack exact selected records')
    by_file = defaultdict(dict)
    for i, subject in subjects.items():
        ref = subject['selected_record']['source_pixel_or_annotation_ref']
        if set(ref)-{'file', 'jsonl_line_0based'} or 'jsonl_line_0based' not in ref:
            raise ValueError('Unsupported original canonical pointer')
        key = int(ref['jsonl_line_0based'])
        if key in by_file[ref['file']]:
            raise ValueError('Two selected rows alias one original canonical record')
        by_file[ref['file']][key] = i
    grouped = defaultdict(list)
    canonical_refs = []
    for name, lines in by_file.items():
        canonical_refs.append(_ref(name))
        pending = set(lines)
        with gzip.open(name, 'rt', encoding='utf-8') as stream:
            for number, line in enumerate(stream):
                if number not in lines:
                    continue
                i = lines[number]
                original = json.loads(line)
                selected = subjects[i]['selected_record']
                if original['uv'] != selected['uv'] or original['source_row_id'] != selected['source_point_id'] or original['native_pts'] != selected['native_pts'] or original['shot_id'] != selected['shot_id']:
                    raise ValueError('Original canonical record / selected reading differs')
                grouped[(original['shot_id'], int(original['native_pts']))].append((subjects[i], original))
                pending.remove(number)
                if not pending:
                    break
        if pending:
            raise ValueError('Missing original canonical record')
    print(json.dumps(dict(phase='canonical_records_bound', rows=len(subjects), native_exposures=len(grouped)), ensure_ascii=False), flush=True)
    frames = {}
    with gzip.open(mask_index, 'rt', encoding='utf-8') as stream:
        for frame in map(json.loads, stream):
            key = frame['shot_id'], int(frame['native_pts'])
            if key in grouped:
                if frame['source_video_sha256'] != source_video_sha256:
                    raise ValueError('Person mask belongs to another official source')
                frames[key] = frame
    if set(frames) != set(grouped):
        raise ValueError('Some F frames have no original person-mask binding')
    if file_sha256(source_video) != source_video_sha256:
        raise ValueError('Official original source video changed')
    findings = {}
    with gzip.open(confirmed_review, 'rt', encoding='utf-8') as stream:
        for item in map(json.loads, stream):
            finding = item.get('independent_source_finding')
            if finding:
                findings[finding['source_row_id']] = finding
    output.mkdir(parents=True)
    scope_counts = Counter()
    decisions = Counter()
    affected_frames = Counter()
    review_rows = []
    context_path = output/'contexts.jsonl.gz'
    preserved_path = output/'withdrawn_originals.jsonl.gz'
    native_pts = {key[1]: key for key in grouped}
    if len(native_pts) != len(grouped):
        raise ValueError('Native PTS belongs to more than one shot')
    remaining = set(native_pts)
    with av.open(str(source_video)) as video, gzip.open(context_path, 'wt', encoding='utf-8') as contexts, gzip.open(preserved_path, 'wt', encoding='utf-8') as preserved:
        stream = video.streams.video[0]
        if str(stream.time_base) != '1/12800':
            raise ValueError('Original native PTS clock required')
        video.seek(min(remaining), stream=stream, backward=True)
        context_line = 0
        for decoded in video.decode(stream):
            pt = int(decoded.pts)
            if pt not in remaining:
                if pt > max(remaining):
                    raise ValueError('Missing exact source exposure')
                continue
            key = native_pts[pt]
            context = build_person_context(frames[key], rgb=decoded.to_ndarray(format='rgb24'))
            contexts.write(json.dumps(context[2], ensure_ascii=False, separators=(',', ':'))+'\n')
            for subject, original in grouped[key]:
                scope_counts[key[0]] += 1
                result = review_row(original, context, source_finding=findings.get(original['source_row_id']))
                decisions[result['decision']] += 1
                if result['decision'] not in ('unknown_person_component_context', 'existing_person_mask_overlap', 'source_RGB_confirmed_nonmaterial_occlusion'):
                    continue
                category = 'confirmed_nonmaterial_occlusion' if result['independent_source_finding'] else 'unknown_person_context'
                evidence_images = [context[2]['person_mask_ref']]
                if result['independent_source_finding']:
                    evidence_images += [ref for ref in result['independent_source_finding']['evidence_refs'] if Path(ref['file']).suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp')]
                record = dict(raw_row=subject['raw_row'], source_row_id=subject['source_row_id'],
                    uv=original['uv'], native_pts=pt, shot_id=key[0], qualification='unknown',
                    classification=category, source_context_class=result['decision'],
                    finding=('Original source RGB independently confirms nonmaterial occlusion.' if category == 'confirmed_nonmaterial_occlusion' else
                        'Recorded material support intersects an existing person mask or its same-component convex UNKNOWN context. This does not classify all hull pixels as person and does not establish visible ice from zero mask values.'),
                    native_images=evidence_images, source_RGB_sha256=context[2]['source_RGB_sha256'],
                    source_video_sha256=source_video_sha256,
                    source_context_line_0based=context_line,
                    original_source_ref=subject['selected_record']['source_pixel_or_annotation_ref'],
                    original_source_point_id=original['source_row_id'],
                    original_selected_record_sha256=subject['selected_record_sha256'],
                    recorded_support=result['recorded_support'])
                review_rows.append(record)
                affected_frames[key] += 1
                preserved.write(json.dumps(dict(raw_row=subject['raw_row'], original_selected_record=subject['selected_record'],
                    original_selected_record_sha256=subject['selected_record_sha256'], source_stream=subject['source_stream'],
                    source_review=result), ensure_ascii=False, separators=(',', ':'))+'\n')
            context_line += 1
            if context_line % 100 == 0:
                print(json.dumps(dict(phase='actual_native_RGB_context_review', exposures=context_line, rows=sum(scope_counts.values()), withdrawals=len(review_rows)), ensure_ascii=False), flush=True)
            remaining.remove(pt)
            if not remaining:
                break
    if remaining:
        raise ValueError('Official source decode omitted requested exposures')
    context_ref = _ref(context_path)
    for row in review_rows:
        row['source_context_ref'] = context_ref
    review_rows.sort(key=lambda r: r['raw_row'])
    if len({r['raw_row'] for r in review_rows}) != len(review_rows) or not review_rows:
        raise ValueError('Empty or duplicate withdrawal review')
    review = dict(schema='rinkfit.native_material_identity_review.v1',
        qualification_uses_model_residual=False, native_frame_extractor=_ref(__file__),
        parent_packet=_ref(parent_packet), source_video=dict(file=str(Path(source_video).resolve()), sha256=source_video_sha256),
        source_mask_index=_ref(mask_index), source_binding=_ref(source_binding), canonical_sources=canonical_refs,
        current_person_context_implementation=_ref(Path(__file__).with_name('person_context_gate_v1.py')),
        original_cost_reference=_ref(prior_cost_report), original_raw_records=_ref(preserved_path),
        same_exposure_contexts=context_ref, rows=review_rows,
        source_rows_deleted=0, original_source_UV_ID_PTS_E_weights_changed=False,
        hull_is_unknown_not_person_segmentation=True, packet_installed=False)
    review_path = output/'review.json'
    review_path.write_text(json.dumps(review, ensure_ascii=False, separators=(',', ':'))+'\n', encoding='utf-8')
    manifest = dict(schema='rinkfit.person_context_qualification_batch.v1',
        parent_packet=review['parent_packet'], review=_ref(review_path),
        scope=dict(shots=['s05', 's06'], source='original_dense56_selected_source_parts',
            materials='ice.F1-F4 inner/middle/outer', active_F_rows_all_readers=all_family_rows,
            reviewed_same_source_rows=len(raw_rows), reviewed_rows_by_shot=dict(scope_counts),
            original_native_exposures=len(grouped), outside_same_source_rows=len(outside),
            outside_same_source_raw_rows=outside.tolist()),
        source_gate_decisions=dict(decisions), withheld_rows=len(review_rows),
        confirmed_nonmaterial_rows=sum(r['classification']=='confirmed_nonmaterial_occlusion' for r in review_rows),
        unknown_person_context_rows=sum(r['classification']=='unknown_person_context' for r in review_rows),
        affected_native_exposures=len(affected_frames), raw_rows_preserved=doc['row_count'],
        old_cost_report=review['original_cost_reference'], no_frame_images_or_large_arrays_written=True,
        original_source_records_retained=True, production_person_masks_unchanged=True,
        packet_installed=False, solver_started=False,
        installer='rinkfit_joint.compact_identity_withhold_v2.withhold')
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return manifest
