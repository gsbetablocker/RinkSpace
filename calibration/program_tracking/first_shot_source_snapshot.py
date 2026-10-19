"""Safe complete source snapshots and explicit restoration of the public bundle.

This is a source metadata transport, never an r/J cache.  Historical source
constructors and their producers stay unchanged.  A restored bundle starts with
empty value/derivative/coverage caches and unopened execution pools.
"""
from collections import OrderedDict
from copy import deepcopy
from dataclasses import fields
from pathlib import Path
import hashlib
import json

import numpy as np

from .first_shot_safe_frame_export import _encode, _decode
from .first_shot_source_transition import reference, read

SCHEMA = 'rinkspace.complete-authenticated-native-source-snapshot.v1'
PROOF_FIELDS = ('source_revision_proof', 'source_noise_revision_proof',
                'source_native_line_revision_proof')


def _sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _assert_same(a, b, path='payload'):
    """Check metadata recursively and numerical arrays by bytes, not tolerance."""
    from .source_evidence_whitening import SourceWhiteningOperator
    if isinstance(a, SourceWhiteningOperator):
        if not isinstance(b, SourceWhiteningOperator):
            raise ValueError('Snapshot whitening type differs: ' + path)
        return _assert_same(a.to_record(), b.to_record(), path + '.operator')
    if isinstance(a, np.ndarray):
        if (not isinstance(b, np.ndarray) or a.dtype != b.dtype or a.shape != b.shape
                or a.tobytes() != b.tobytes()):
            raise ValueError('Snapshot numerical bytes differ: ' + path)
        return
    if isinstance(a, dict):
        if not isinstance(b, dict) or tuple(a) != tuple(b):
            raise ValueError('Snapshot metadata keys/order differ: ' + path)
        for key in a:
            _assert_same(a[key], b[key], path + '.' + key)
        return
    if isinstance(a, (tuple, list)):
        if type(a) is not type(b) or len(a) != len(b):
            raise ValueError('Snapshot sequence type/length differs: ' + path)
        for i, (left, right) in enumerate(zip(a, b, strict=True)):
            _assert_same(left, right, path + '[' + str(i) + ']')
        return
    if isinstance(a, np.generic):
        a = a.item()
    if a != b:
        raise ValueError('Snapshot scalar metadata differs: ' + path)


def coverage_to_record(inventory):
    if inventory is None:
        return None
    return dict(sections=[{f.name: deepcopy(getattr(section, f.name))
                           for f in fields(section)} for section in inventory.sections],
                unsupported=deepcopy(inventory.unsupported),
                requested_extent_columns=inventory.requested_extent_columns,
                initializer_strategy=inventory.initializer_strategy)


def coverage_from_record(record):
    if record is None:
        return None
    from .finite_source_line_coverage import NativeLineSections, NativeLineCoverageInventory
    sections = []
    for item in record['sections']:
        section = NativeLineSections(**deepcopy(item))
        # __post_init__ validates the original public contract, and normalizes
        # normals.  Preserve the authenticated already-normalized source bytes;
        # applying that normalization twice can change the last floating bit.
        for name in ('query_uv', 'profile_normal', 'measured_uv'):
            value = np.asarray(item[name], np.float64).copy()
            value.setflags(write=False)
            object.__setattr__(section, name, value)
        sections.append(section)
    inventory = NativeLineCoverageInventory(tuple(sections), tuple(record['unsupported']),
        record['requested_extent_columns'], record['initializer_strategy'])
    _assert_same(record, coverage_to_record(inventory), 'finite_coverage')
    return inventory


def compile_authenticated_coverage(state, frames, *, group_transition_reference):
    """Compile public finite metadata with the exact admitted v22 support rule."""
    from dataclasses import replace
    from . import finite_source_line_coverage as coverage
    from .first_shot_group_rebuild import read_transition, derive
    _, manifest, item, _, _ = read_transition(group_transition_reference['path'])
    if reference(group_transition_reference['path']) != group_transition_reference:
        raise ValueError('Saved source-support transition changed')
    original = coverage._compile_one
    rebuilt_line = None
    def compile_one(frame, row, feature, native_ref):
        nonlocal rebuilt_line
        if (frame['native_pts'] != item['native_pts']
                or 'ice.' + row['target']['curve_id'] != item['feature_id']):
            return original(frame, row, feature, native_ref)
        if native_ref != item['native_readout_packet']:
            raise ValueError('Restored finite metadata has another native readout packet')
        if rebuilt_line is None:
            _, rebuilt_line, *_ = derive(item)
        feature = deepcopy(feature)
        feature['source_line_fit'] = rebuilt_line
        value = original(frame, row, feature, native_ref)
        evidence = dict(value.evidence, reviewed_support_group_revision=manifest['group_revision'],
            historical_native_readout_count=len(read(item['support_audit'])['all_readouts']),
            current_verified_native_support_count=len(rebuilt_line['direct_readouts']),
            excluded_native_query_indices=read(item['support_audit'])['quarantined_query_indices'])
        return replace(value, evidence=evidence)
    coverage._compile_one = compile_one
    try:
        return coverage.read_coverage_revision(state.binding['finite_source_line_coverage_revision'], state, frames)
    finally:
        coverage._compile_one = original


def _frame_contract(frames):
    from .first_shot_fit import FrozenWhitening, _target_key
    addresses = set(); hashes = []; formal = points = 0; pts = []
    for ordinal, frame in enumerate(frames, 1):
        if int(frame['index']) != ordinal:
            raise ValueError('Complete snapshot requires the original ordered programme frame sequence')
        pts.append(int(frame['native_pts'])); count = 0; h = hashlib.sha256()
        for row in frame['rows']:
            if row['_target_key'] != _target_key(row['target']):
                raise ValueError('Complete snapshot has a stale target key')
            uv, sigma = np.asarray(row['uv']), np.asarray(row['sigma'])
            if uv.shape != (len(row['source_point_ids']), 2) or sigma.shape != (len(uv),):
                raise ValueError('Complete source UV/ID/sigma shape differs')
            for pid in row['source_point_ids']:
                address = int(frame['native_pts']), str(pid)
                if address in addresses:
                    raise ValueError('Complete snapshot repeats a native source address')
                addresses.add(address)
            h.update(json.dumps(row['source_point_ids'], separators=(',', ':')).encode())
            h.update(uv.tobytes()); h.update(sigma.tobytes()); count += len(uv)
        for group in frame['whitening_groups']:
            h.update(group['operator'].matrix.tobytes())
        if count != frame['source_point_count']:
            raise ValueError('Complete snapshot source count differs from actual rows')
        w = FrozenWhitening(frame); formal += w.count; points += count; hashes.append(h.hexdigest())
    return dict(frame_count=len(frames), source_points=points, formal_rows=formal,
        native_pts_sha256=_sha(np.asarray(pts, np.int64)), frame_inventory_sha256=hashes)


def export_source_snapshot(frames, *, binding_reference, policy, proofs,
        producer_references, output_directory, coverage_inventory, coverage_proof,
        parent_authentication_reference, source_transition_proof):
    """Persist all actual source metadata and verify its lossless round trip."""
    output = Path(output_directory); output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ('source.json', 'source_arrays.npz', 'roundtrip.json')):
        raise ValueError('Complete source snapshot files already exist; never overwrite them')
    proofs = deepcopy(proofs)
    alias = 'source_localization_noise_revision_proof'
    if alias in proofs:
        if 'source_noise_revision_proof' in proofs:
            _assert_same(proofs[alias], proofs['source_noise_revision_proof'], 'noise_proof_alias')
        else:
            proofs['source_noise_revision_proof'] = deepcopy(proofs[alias])
    if set(PROOF_FIELDS) - set(proofs):
        raise ValueError('All public original revision proofs must be explicitly supplied')
    binding = read(binding_reference)
    if (policy['source_binding'] != binding_reference or len(frames) != binding['frame_count']
            or policy['source_clock'] != binding['source_clock']):
        raise ValueError('Complete snapshot binding/policy/clock differs')
    refs = list(producer_references)
    for ref in refs + [parent_authentication_reference, binding_reference]:
        if reference(ref['path']) != ref:
            raise ValueError('Complete snapshot source authentication changed')
    payload = dict(frames=frames, binding_reference=binding_reference, policy=policy,
        proofs=proofs, producer_references=refs,
        finite_coverage=coverage_to_record(coverage_inventory), finite_coverage_proof=coverage_proof,
        parent_authentication_reference=parent_authentication_reference,
        source_transition_proof=source_transition_proof)
    arrays = {}; encoded = _encode(payload, arrays)
    np.savez_compressed(output / 'source_arrays.npz', **arrays)
    contract = _frame_contract(frames)
    receipt = dict(schema=SCHEMA, arrays=reference(output / 'source_arrays.npz'), payload=encoded,
        contract=contract, producer=reference(__file__), pickle_deserialization=False,
        complete_source_metadata_not_residual_or_J_cache=True,
        all_rows_native_queries_branch_semantics_complete_W_sigma_and_inactive_evidence_preserved=True)
    path = output / 'source.json'
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    restored, _ = load_source_snapshot(reference(path))
    _assert_same(payload, restored)
    coverage_from_record(restored['finite_coverage'])
    roundtrip = dict(schema='rinkspace.complete-native-source-snapshot-roundtrip.v1', passed=True,
        snapshot=reference(path), arrays=receipt['arrays'], contract=contract,
        complete_payload_every_array_dtype_shape_and_bytes_equal=True,
        complete_W_and_coverage_public_types_restored=True, residual_or_J_evaluated=False)
    (output / 'roundtrip.json').write_text(json.dumps(roundtrip, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return dict(snapshot=reference(path), roundtrip=reference(output / 'roundtrip.json'))


def load_source_snapshot(snapshot_reference):
    receipt = read(snapshot_reference)
    if (receipt.get('schema') != SCHEMA or receipt.get('pickle_deserialization') is not False
            or receipt.get('complete_source_metadata_not_residual_or_J_cache') is not True):
        raise ValueError('Explicit complete safe source snapshot required')
    ref = receipt['arrays']
    if reference(ref['path']) != ref:
        raise ValueError('Complete source snapshot arrays changed')
    with np.load(ref['path'], allow_pickle=False) as saved:
        payload = _decode(receipt['payload'], {key: saved[key].copy() for key in saved.files})
    if _frame_contract(payload['frames']) != receipt['contract']:
        raise ValueError('Complete source snapshot row/clock/operator contract changed')
    for ref in payload['producer_references'] + [payload['binding_reference'], payload['parent_authentication_reference']]:
        if reference(ref['path']) != ref:
            raise ValueError('Complete source snapshot producer/parent authentication changed')
    return payload, receipt


def restore_source_bundle(snapshot_reference, *, geometry_workers, source_workers):
    """Initialize every public FirstShotSourceBundle field, without old r/J."""
    from .first_shot_fit import FirstShotSourceBundle, FrozenWhitening
    from .first_shot_physical_state import FirstShotPhysicalState
    payload, receipt = load_source_snapshot(snapshot_reference)
    bundle = FirstShotSourceBundle.__new__(FirstShotSourceBundle)
    bundle.state = FirstShotPhysicalState.from_binding(payload['binding_reference']['path'])
    bundle.policy = deepcopy(payload['policy']); bundle.frames = payload['frames']
    bundle.descriptors = OrderedDict()
    for frame in bundle.frames:
        for row in frame['rows']:
            bundle.descriptors.setdefault(row['_target_key'], deepcopy(row['target']))
    bundle.keys = tuple(bundle.descriptors); bundle.target_descriptors = tuple(bundle.descriptors.values())
    for name in PROOF_FIELDS:
        setattr(bundle, name, deepcopy(payload['proofs'][name]))
    bundle.initial_target_proof = bundle.state.validate_initial_targets(bundle.target_descriptors)
    bundle.whitening = [FrozenWhitening(frame) for frame in bundle.frames]
    bundle.scales = np.concatenate([w.scales for w in bundle.whitening])
    bundle.quadratic = np.zeros(len(bundle.scales), bool)
    bundle.geometry_workers = int(geometry_workers); bundle.source_workers = int(source_workers)
    bundle._geometry_pool = bundle._source_pool = None
    bundle._world_cache = OrderedDict(); bundle._finite_coverage_cache = OrderedDict()
    bundle.timings = []; bundle.worker_receipts = {}; bundle.ledgers = []
    for frame, w, digest in zip(bundle.frames, bundle.whitening, receipt['contract']['frame_inventory_sha256'], strict=True):
        bundle.ledgers.append(dict(family='accepted_official_static_source', programme_index=frame['index'],
            native_pts=frame['native_pts'], time_base='1/12800', source_points=frame['source_point_count'],
            formal_rows=w.count, source_inventory_sha256=digest, source_packet=frame['source_packet'],
            association_policy=deepcopy(bundle.policy), complete_source_snapshot=deepcopy(snapshot_reference)))
    bundle.finite_coverage_inventory = coverage_from_record(payload['finite_coverage'])
    bundle.finite_coverage_revision_proof = deepcopy(payload['finite_coverage_proof'])
    if bundle.state.binding.get('finite_source_line_coverage_revision') is not None and bundle.finite_coverage_inventory is None:
        raise ValueError('Restored complete source is missing its authenticated finite coverage inventory')
    bundle.safe_source_snapshot_receipt = dict(snapshot=snapshot_reference, producer=reference(__file__),
        source_producers=deepcopy(payload['producer_references']),
        original_revision_proofs=deepcopy(payload['proofs']), parent_authentication=payload['parent_authentication_reference'],
        source_transition=deepcopy(payload['source_transition_proof']), complete_public_bundle_initialized=True,
        old_residual_J_or_value_caches_loaded=False, all_execution_pools_initially_closed=True)
    return bundle
