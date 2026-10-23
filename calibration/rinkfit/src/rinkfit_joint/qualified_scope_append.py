"""Rebind existing exact-ID board evidence after a qualified nonboard append.

This grants no authority to the new sources: their actual production appender
must validate its source gate.  All inherited raw values and active membership
are checked byte for byte before the existing certificate is rebound.
"""
from pathlib import Path
import copy
import hashlib
import json
import numpy as np
from .io_utils import file_sha256, write_json
from .qualified_scope import EvidenceResolver, _ref, CERTIFICATE_SCHEMA
from .board_source_qualification_guard import (assert_board_material_source_qualification,
                                               FAMILY_PREFIXES, BOUND_ARRAYS)


def _same_prefix(old, new, n, resolver, oldbase, newbase):
    if old['dtype'] != new['dtype'] or old['shape'][1:] != new['shape'][1:] or old['shape'][0] != n:
        raise ValueError('Source append changed inherited scalar layout')
    left, right = resolver(old, oldbase), resolver(new, newbase)
    size = n * np.dtype(old['dtype']).itemsize * int(np.prod(old['shape'][1:] or [1]))
    with left.open('rb') as a, right.open('rb') as b:
        a.seek(int(old.get('offset_bytes',0)))
        b.seek(int(new.get('offset_bytes',0)))
        while size:
            k = min(size, 1024*1024)
            x, y = a.read(k), b.read(k)
            if len(x) != k or x != y:
                raise ValueError('Source append changed inherited raw prefix')
            size -= k


def rebind_qualified_board_certificate(parent_packet, appended_packet, folder,
                                       qualified_append_bundle_ref, *, parent_document=None, native_parent_qualification_ref=None):
    """Production appender finalizer for an append with unchanged old members."""
    parent_path = Path(parent_packet).resolve()
    folder = Path(folder).resolve()
    parent = parent_document if parent_document is not None else json.loads(parent_path.read_text(encoding='utf8'))
    result = appended_packet
    n, m = int(parent['row_count']), int(result['row_count'])
    if m <= n or parent['shot_ids'] != result['shot_ids']:
        raise ValueError('Not an immutable append to the same complete shot scope')
    resolver = EvidenceResolver()
    bundle_path = resolver(qualified_append_bundle_ref)
    bundle = json.loads(bundle_path.read_text(encoding='utf8'))
    allowed_bundles = ('rinkfit.native_tracklet_qualified_source_bundle.v1', 'rinkfit.native_ordinary_ice_midline_qualified_bundle.v1')
    if bundle.get('schema') not in allowed_bundles or not bundle.get('source_ledger_preserved'):
        raise ValueError('Nonboard append lacks actual source-qualified bundle')
    for ref in bundle.get('typed_entity_qualification', []) + bundle['closed_chunks']:
        resolver(ref, bundle_path.parent)
    resolver(bundle['material_map'], bundle_path.parent)
    verified_dependencies=set()
    inherited_native_qualification = None
    if native_parent_qualification_ref is None:
        assert_board_material_source_qualification(parent, parent_path, verified_dependency_cache=verified_dependencies)
    else:
        # A genuine previous native consumer already checked this exact packet.
        # Rebinding unchanged source members does not re-run its 63668 physical
        # source audits; the new packet still requires a native current check.
        passed = json.loads(resolver(native_parent_qualification_ref).read_text(encoding='utf8'))
        probe_pass = (passed.get('native_source_qualification_guard_passed') is True
                      and passed.get('loaded_packet_buffers', {}).get('raw_rows') == n
                      and bool(passed.get('source_guard_membership_fingerprint')))
        # joint_fit --qualification-check performs the same production guard
        # and full packet parse, then emits this compact native receipt.
        expected_rank = sum(int(g['rank']) for g in parent['groups']) + 2*len(parent.get('independent_rows', []))
        expected_active = sum(len(g['rows']) for g in parent['groups']) + len(parent.get('independent_rows', []))
        qualification_pass = (passed.get('passed') is True
                              and passed.get('raw_rows') == n
                              and passed.get('active_rows') == expected_active
                              and passed.get('statistical_rank') == expected_rank
                              and passed.get('shots') == len(parent['shot_ids'])
                              and passed.get('rigs', 0) > 0)
        if (not (probe_pass or qualification_pass)
            or passed.get('source_packet_sha256') != file_sha256(parent_path)):
            raise ValueError('Inherited native qualification does not cover this exact immutable parent')
        inherited_native_qualification = native_parent_qualification_ref
    for key in ('source_row_ids', 'source_dataset_ids'):
        if result[key][:n] != parent[key] or len(result[key]) != m:
            raise ValueError('Append changed inherited raw source ID order')
    if any(str(s).startswith(FAMILY_PREFIXES) for s in result['source_row_ids'][n:]):
        raise ValueError('New recalled board sources need their own exact-ID qualification')
    if set(parent['arrays']) != set(result['arrays']):
        raise ValueError('Append changed source scalar field set')
    for name, old in parent['arrays'].items():
        _same_prefix(old, result['arrays'][name], n, resolver, parent_path.parent, folder)
    oldref = parent['board_material_source_qualification']
    cert = json.loads(resolver(oldref, parent_path.parent).read_text(encoding='utf8'))
    for directory in (parent_path.parent, resolver(oldref, parent_path.parent).parent):
        for name in ('SOURCE_IDENTITY_CERTIFICATE_RECALLED.json', 'SOURCE_IDENTITY_EVIDENCE_RECALL.json'):
            if (directory/name).exists():raise ValueError('Inherited source qualification has been recalled')
    if cert.get('schema') != CERTIFICATE_SCHEMA:
        raise ValueError('Nonboard rebind requires already verified incremental qualification')
    index = folder/'source_chunk_index.json'
    resolver(_ref(index))
    parent_generation = json.loads(resolver(parent['qualified_scope_generation'], parent_path.parent).read_text(encoding='utf8'))
    binding = dict(schema='rinkfit.qualified_source_scope_nonboard_append.v1',
        membership_decisions=parent_generation['membership_decisions'],
        parent_packet=_ref(parent_path), parent_qualification=oldref,
        actual_append_source_bundle=qualified_append_bundle_ref,
        actual_append_index=_ref(index), original_raw_rows=n, appended_raw_rows=m-n,
        raw_rows=m, complete_shot_scope_unchanged=True,
        all_inherited_arrays_including_admission_byte_identical=True,
        inherited_qualified_board_members_unchanged=True,
        new_nonboard_source_qualification_issued_by_this_rebind=False,
        whole_same_exposure_statistics_for_new_rows='statistical_unavailable',
        new_statistical_precision_invented=False, native_auxiliary_rebuild_required=True,
        inherited_native_qualification_ref=inherited_native_qualification,
        native_current_generation_check_required=True,
        original_physical_qualification_rescanned=inherited_native_qualification is None,
        compiler_module=_ref(Path(__file__)))
    write_json(folder/'qualified_generation.json', binding)
    cert = copy.deepcopy(cert)
    cert.update(source_scope_ref=_ref(folder/'qualified_generation.json'),
        membership_array_sha256={k:result['arrays'][k]['sha256'] for k in BOUND_ARRAYS},
        source_row_ids_sha256=hashlib.sha256(json.dumps(result['source_row_ids'],ensure_ascii=False,separators=(',',':')).encode()).hexdigest(),
        source_dataset_ids_sha256=hashlib.sha256(json.dumps(result['source_dataset_ids'],ensure_ascii=False,separators=(',',':')).encode()).hexdigest(),
        inherited_qualification_ref=oldref, nonboard_append_rebind=True)
    from .measurement_operator import measurement_operator_binding, assert_append_operator_prefix
    assert_append_operator_prefix(parent, result, n)
    cert['measurement_operator_binding'] = measurement_operator_binding(result)
    write_json(folder/'qualified_certificate.json',cert)
    result['board_material_source_qualification']=_ref(folder/'qualified_certificate.json')
    result['qualified_scope_generation']=_ref(folder/'qualified_generation.json')
    result['source_placeholder_station_half_is_not_solver_ready']=True
    result['all_new_source_nuisance_requires_fresh_physical_capture']=True
    if inherited_native_qualification is None:
        assert_board_material_source_qualification(result, folder/'packet.json', verified_dependency_cache=verified_dependencies)
    return result
