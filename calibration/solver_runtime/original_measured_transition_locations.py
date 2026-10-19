"""Restore saved source estimators without changing original observations.

The archived blur interval remains source evidence.  A qualified measured
position is evaluated by the existing typed raw operator at its original sigma.
No projection, residual, or model-dependent selection enters qualification.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

SCHEMA = 'rinkspace.original-saved-transition-location.v1'
CONTRACT_KEY = 'original_measured_transition_locations'
MODE = 'measured_transition_location'
LOCATOR_KEYS = ('block', 'collection_role', 'group_actual_pts', 'group_order', 'row_order')
SOURCE_FIELDS = ('id', 'pts', 'split', 'origin', 'uv', 'sigma', 'normal', 'normal_interval',
    'point_correspondence', 'curve_observation_kind', 'scanline_branch', 'scanline_fixed_axis', 'extent_side')


def plain(value):
    return json.loads(json.dumps(value, default=lambda x: x.tolist() if isinstance(x, np.ndarray)
        else x.item() if isinstance(x, np.generic) else str(x), allow_nan=False))


def canonical(value):
    return json.dumps(plain(value), sort_keys=True, separators=(',', ':'), ensure_ascii=False,
        allow_nan=False).encode('utf8')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def reference(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def checked(ref):
    path = Path(ref['path']).resolve()
    if sha(path) != ref['sha256']:
        raise ValueError('Saved source location authority bytes changed')
    return path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _at(value, path):
    if not isinstance(path, list) or not path or any(type(k) not in (int, str) for k in path):
        raise ValueError('Explicit saved source JSON location path required')
    for key in path:
        value = value[key]
    return value


def _same_saved_sample_value(before, after):
    """Exact JSON/NumPy numeric representation equivalence, never coercion."""
    if isinstance(before, np.ndarray) or isinstance(after, np.ndarray):
        try:
            a, b = np.asarray(before), np.asarray(after)
        except (TypeError, ValueError):
            return False
        if a.shape != b.shape or a.dtype.kind not in 'biuf' or b.dtype.kind not in 'biuf':
            return False
        before, after = a.tolist(), b.tolist()
    if isinstance(before, np.generic):
        before = before.item()
    if isinstance(after, np.generic):
        after = after.item()
    if isinstance(before, bool) or isinstance(after, bool):
        return type(before) is bool and type(after) is bool and before == after
    if type(before) in (int, float) or type(after) in (int, float):
        return (type(before) in (int, float) and type(after) in (int, float)
            and np.isfinite(before) and np.isfinite(after) and before == after)
    if isinstance(before, dict) or isinstance(after, dict):
        return (isinstance(before, dict) and isinstance(after, dict)
            and set(before) == set(after)
            and all(_same_saved_sample_value(before[k], after[k]) for k in before))
    if isinstance(before, (list, tuple)) or isinstance(after, (list, tuple)):
        return (isinstance(before, (list, tuple)) and isinstance(after, (list, tuple))
            and len(before) == len(after)
            and all(_same_saved_sample_value(a, b) for a, b in zip(before, after)))
    return type(before) is type(after) and canonical(before) == canonical(after)


def _same_runtime_field(key, before, after):
    # The original absolute archive converted the native Ad2/Ad3 sample UVs
    # to NumPy floats; the unchanged source JSON keeps fixed-y values as ints.
    # Only these sample containers need numeric representation equivalence.
    # Every leaf, key, list length and numeric-array shape remains checked.
    return (_same_saved_sample_value(before, after) if key == 'samples'
            else canonical(before) == canonical(after))


def runtime_source_differences(row, original):
    """Compact exact diagnostics; never expand WORLD or source profiles."""
    differences = []
    for key, before in original.items():
        if key in row and _same_runtime_field(key, before, row[key]):
            continue
        difference = dict(key=key, runtime_present=key in row,
            original_canonical_sha256=hashlib.sha256(canonical(before)).hexdigest())
        if key in row:
            after = row[key]
            difference['runtime_canonical_sha256'] = hashlib.sha256(canonical(after)).hexdigest()
            try:
                a, b = np.asarray(before), np.asarray(after)
                if a.dtype.kind in 'biuf' and b.dtype.kind in 'biuf':
                    difference.update(original_shape=list(a.shape), runtime_shape=list(b.shape))
                    if a.shape == b.shape:
                        difference['maximum_numeric_difference'] = float(np.max(np.abs(a.astype(float)-b.astype(float)))) if a.size else 0.
                elif isinstance(before, (str, int, float, bool, type(None))) and isinstance(after, (str, int, float, bool, type(None))):
                    difference.update(original_value=before, runtime_value=after)
            except (TypeError, ValueError):
                pass
        differences.append(difference)
    return differences


def validate_packet(path):
    path = Path(path).resolve(); packet = read(path)
    if packet.get('schema') != SCHEMA:
        raise ValueError('Explicit original saved-position qualification required')
    original = pickle.loads(checked(packet['original_absolute_packet']).read_bytes())['programme_absolute_rows']
    records = packet.get('records', [])
    indices = [r.get('original_packet_row_index') for r in records]
    if (not indices or any(type(i) is not int or not 0 <= i < len(original) for i in indices)
            or indices != sorted(set(indices))):
        raise ValueError('Ordered distinct original FIT source rows required')
    documents = {}
    for record, index in zip(records, indices):
        item = original[index]; row = item['source_row']; uv = np.asarray(row['uv'], float)
        if (record.get('locator') != plain({k: item[k] for k in LOCATOR_KEYS})
                or item['collection_role'] != 'FIT' or row.get('split') not in ('fit', 'development')
                or record.get('source_row_id') != row['id']
                or record.get('original_source_row_sha256') != hashlib.sha256(canonical(row)).hexdigest()
                or record.get('point_indices') != list(range(len(uv)))
                or not np.array_equal(record.get('measured_uv'), uv)
                or 'normal_interval' not in row or not record.get('estimator_kind')):
            raise ValueError('Saved location must exactly preserve the full original FIT observation')
        for ref in record.get('producer_references', []):
            checked(ref)
        source_ref = record['source_reference']; source_path = checked(source_ref)
        key = str(source_path)
        if key not in documents:
            documents[key] = read(source_path)
        source = documents[key]
        bindings = record.get('measurement_bindings', [])
        if [b.get('point_index') for b in bindings] != list(range(len(uv))):
            raise ValueError('Every original point needs its saved measured-location authority')
        for binding, observed in zip(bindings, uv):
            axis = binding.get('measured_axis')
            if type(axis) is not int or axis not in (0, 1):
                raise ValueError('Original measured source axis required')
            location = _at(source, binding['location_path'])
            if not np.isscalar(location) or not np.isfinite(location) or float(location) != observed[axis]:
                raise ValueError('Current source UV is not its original saved measured position')
            if 'observed_uv_path' in binding:
                if not np.array_equal(_at(source, binding['observed_uv_path']), observed):
                    raise ValueError('Saved measured source UV differs from original authority')
            elif 'fixed_coordinate_path' in binding:
                fixed = _at(source, binding['fixed_coordinate_path'])
                if not np.isscalar(fixed) or float(fixed) != observed[1-axis]:
                    raise ValueError('Original source scanline coordinate differs')
            else:
                raise ValueError('Saved source UV or fixed scanline coordinate required')
    contract = dict(schema=SCHEMA + '.installed', packet_reference=reference(path),
        original_absolute_packet=packet['original_absolute_packet'], observation=MODE,
        records=[dict(original_packet_row_index=i, locator=r['locator'], source_row_id=r['source_row_id'],
            original_source_row_sha256=r['original_source_row_sha256'], points=len(r['point_indices']),
            estimator_kind=r['estimator_kind']) for i, r in zip(indices, records)],
        original_UV_sigma_normal_interval_world_identity_and_split_unchanged=True,
        original_typed_raw_operator_and_scalar_layout_unchanged=True,
        source_blur_interval_preserved_as_evidence_not_location_dead_zone=True,
        source_only_saved_estimator_no_model_selection=True,
        original_HELD_not_fitted=True, fresh_normal_and_J_required=True, accuracy_accepted=False)
    return packet, original, contract


def install_original_measured_transition_locations(state, path):
    if getattr(state, 'source_actors', None) is not None:
        raise ValueError('Install measured source positions before starting source actors')
    from .original_measured_location_mode import install_measured_location_mode
    packet, original, contract = validate_packet(path)
    blocks = {b.name: b for b in state.blocks}; plans = []
    for record in packet['records']:
        index = record['original_packet_row_index']; item = original[index]; old = item['source_row']
        block = blocks[item['block']]; group, pos = int(item['group_order']), int(item['row_order'])
        pts, rows = block.groups[group]; row = rows[pos]
        if int(pts) != int(item['group_actual_pts']):
            raise ValueError('Original source exposure/group differs')
        if row.get('active_source_pixel_correction') is not None:
            raise ValueError('Historical saved-location proof cannot override a later pixel correction')
        # The original raw operator also reads WORLD, physical bindings,
        # branch guards and companion fields.  Equal pixels alone cannot
        # authenticate another physical primitive at this same locator.
        differences = runtime_source_differences(row, old)
        if differences:
            raise ValueError('Runtime source differs from the original saved measured observation: '
                + json.dumps(dict(original_packet_row_index=index, source_row_id=old['id'],
                    locator=record['locator'], differing_fields=differences), ensure_ascii=False))
        revised = deepcopy(row)
        result = install_measured_location_mode(revised, authority_ref=contract['packet_reference'],
            source_row_index=index, qualified_receipt_record=record)
        if result is not None:
            revised = result
        plans.append((block, group, pos, revised))
    # Validate all rows before replacing any active source input.
    for block, group, pos, revised in plans:
        pts, rows = block.groups[group]; copied = list(rows); copied[pos] = revised
        groups = list(block.groups); groups[group] = (pts, copied); block.groups = groups
        block.bundle.abs_groups = groups
    setattr(state, CONTRACT_KEY, contract)
    return contract
