"""Revise source identity without expanding millions of inherited JSON rows."""
from pathlib import Path
import json
from collections import Counter
import numpy as np
from .dense_joint_proofs import _ref, _write, _checked

def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def _array(ref):
    return np.memmap(ref['file'], mode='r', dtype=ref['dtype'], shape=tuple(ref['shape']), offset=ref.get('offset_bytes', 0))

def source_identity(packet, row):
    path = Path(packet)
    while True:
        doc = _json(path)
        for name in ('native_ad_source_binding', 'dense_source_binding'):
            binding = doc.get(name, {})
            start = binding.get('row_start', 0)
            if binding and start <= row < start + binding['count']:
                local = row - start
                offsets = _array(binding['source_id_offsets'])
                with Path(binding['source_id_utf8']['file']).open('rb') as stream:
                    stream.seek(int(offsets[local]))
                    value = stream.read(int(offsets[local + 1] - offsets[local])).decode('utf8')
                return value, binding['source_dataset_id']
        if row < len(doc.get('source_row_ids', [])):
            return doc['source_row_ids'][row], doc['source_dataset_ids'][row]
        path = Path(doc['dense_parent_packet']['file'])

def withhold(*, parent_packet, parent_qualification, parent_weights, review, output):
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    doc = _json(parent_packet); q = _json(parent_qualification); w = _json(parent_weights); v = _json(review)
    if v['schema'] != 'rinkfit.native_material_identity_review.v1' or v['qualification_uses_model_residual']:
        raise ValueError('Original RGB identity review required')
    _checked(v['native_frame_extractor'], Path(review).parent)
    arrays = dict(doc['arrays']); uv = _array(arrays['uv']); pts = _array(arrays['native_pts']); idx = []
    for item in v['rows']:
        i = int(item['raw_row']); identity, _ = source_identity(parent_packet, i)
        if identity != item['source_row_id'] or uv[i].tolist() != item['uv'] or int(pts[i]) != item['native_pts']:
            raise ValueError('Source identity/UV/PTS mismatch')
        if item['qualification'] != 'unknown' or not item['finding'] or not item['native_images']:
            raise ValueError('Missing native source finding')
        for image in item['native_images']: _checked(image, Path(review).parent)
        idx.append(i)
    if not idx or len(idx) != len(set(idx)): raise ValueError('Empty/duplicate revision')
    idx.sort()
    for key in ('admission', 'fit_role'):
        values = np.array(_array(arrays[key]), copy=True)
        if not np.all(values[idx] == 1): raise ValueError('Reviewed rows were not eligible')
        values[idx] = 0; file = out / (key + '.u8'); values.tofile(file)
        arrays[key] = dict(_ref(file), dtype=values.dtype.str, shape=list(values.shape), layout='C', byte_order='little')
    revision = dict(schema='rinkfit.native_material_identity_revision.v1', parent_packet=_ref(parent_packet), parent_qualification=_ref(parent_qualification), review=_ref(review), withheld_rows=idx, raw_rows=doc['row_count'], source_UV_ID_PTS_and_operators_unchanged=True, source_rows_deleted=0, issuer=_ref(__file__))
    rr = _write(out / 'revision.json', revision)
    packet = _write(out / 'packet.json', dict(schema=doc['schema'], dense_parent_packet=_ref(parent_packet), row_count=doc['row_count'], arrays=arrays, active_rows=doc['active_rows'] - len(idx), admitted_rows=doc['admitted_rows'] - len(idx), native_material_identity_revision=rr, dense_parent_append=dict(source_manifests=[rr, _ref(review), _ref(__file__)])))
    registry = doc['native_ad_source_binding']; newheld = sum(i >= registry['row_start'] for i in idx)
    q.update(schema='rinkfit.native_material_identity_revision_qualification.v1', source_packet=packet, parent_packet=_ref(parent_packet), parent_qualification=_ref(parent_qualification), revision=rr, issuer=_ref(__file__), registry_kind='native_ad_source_binding', membership_array_sha256={key: value['sha256'] for key, value in arrays.items()}, all_raw_pixels_retained=True, quality_accepted=False)
    q['active_new_rows'] -= newheld; q['unknown_new_rows'] = registry['count'] - q['active_new_rows']
    qr = _write(out / 'source_qualification.json', q)
    w.update(source_packet=packet, dense_source_qualification=qr, active_rows=w['active_rows'] - len(idx), new_active_rows=w['new_active_rows'] - newheld)
    for key in ('weights.f64', 'active.u8', 'information_class.u8'):
        values = np.array(_array(w['files'][key]), copy=True); values[idx] = 0; file = out / key; values.tofile(file)
        w['files'][key] = dict(_ref(file), dtype=values.dtype.str, shape=list(values.shape), layout='C', byte_order='little')
        if key == 'weights.f64':
            tail = int(w['parent_rows']); w.update(retained_parent_budget=float(values[:tail].sum()), new_dense_budget=float(values[tail:].sum()), actual_total_engineering_budget=float(values.sum()))
    families = _array(registry['source_family_index'])
    removed = Counter(registry['source_family_ids'][families[i - registry['row_start']]] for i in idx if i >= registry['row_start'])
    budgets = []
    for item in w.get('component_budgets', []):
        count = item['rows'] - removed[item['id']]
        if count < 0: raise ValueError('Source component count mismatch')
        if count: budgets.append(dict(item, rows=count, engineering_budget=count * item['per_row_weight']))
    w['component_budgets'] = budgets
    wr = _write(out / 'engineering_weights.json', w)
    return dict(packet=packet, qualification=qr, weights=wr, withheld=len(idx), raw_rows=doc['row_count'])
