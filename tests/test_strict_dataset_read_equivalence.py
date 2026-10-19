"""Independent old/full reviewer equivalence for the compact runtime reader."""
from copy import deepcopy
import json

import numpy as np
import pytest

from calibration.program_tracking.compact_source_json import pack_source_json, unpack_source_json
from calibration.program_tracking.strict_absolute_source import StrictAbsoluteSourcePack
from calibration.program_tracking.strict_model_scope import file_sha
from calibration.program_tracking.test_strict_source_disjoint_merge import _fixture_pack
from calibration.solver_runtime.strict_dataset_read_runtime import StrictDatasetValidatedPack


def _readers(reference):
    return [reader(reference['path'], expected_sha256=reference['sha256'])
        for reader in (StrictAbsoluteSourcePack, StrictDatasetValidatedPack)]


def _mutate(reference, path, edit):
    with np.load(reference['path'], allow_pickle=False) as saved:
        values = {name:saved[name].copy() for name in saved.files}
    headers = unpack_source_json(values['rows_json'])
    proofs = unpack_source_json(values['source_proof_templates_json'])
    metadata = json.loads(str(values['metadata_json'].item()))
    edit(values, headers, proofs, metadata)
    values['rows_json'] = pack_source_json(headers)
    values['source_proof_templates_json'] = pack_source_json(proofs)
    values['metadata_json'] = json.dumps(metadata)
    np.savez_compressed(path, **values)
    return dict(path=str(path), sha256=file_sha(path))


@pytest.mark.parametrize('board,role,empty', [
    (False,'fit',False), (False,'held',False), (True,'held',False), (False,'fit',True)])
def test_full_public_reviewer_and_new_reader_preserve_literal_arrays_and_evidence(tmp_path, board, role, empty):
    reference = _fixture_pack(tmp_path/'source', board=board, split=role, empty=empty)
    before, after = _readers(reference)
    assert after.metadata == before.metadata
    assert after.rows == before.rows
    assert after.exclusions == before.exclusions
    assert after.proof_templates == before.proof_templates
    assert after.frame_bindings == before.frame_bindings
    assert after.values.keys() == before.values.keys()
    for field in before.values:
        expected, actual = before.values[field], after.values[field]
        assert actual.dtype == expected.dtype, field
        assert actual.shape == expected.shape, field
        assert actual.tobytes() == expected.tobytes(), field
    assert after.validation_receipt['every_original_UV_and_proof_index_checked'] is True
    assert after.validation_receipt['legacy_pixel_matching_map_constructed'] is False


def _bad(edit, reference, tmp_path):
    changed = _mutate(reference, tmp_path/'tampered.npz', edit)
    for reader in (StrictAbsoluteSourcePack, StrictDatasetValidatedPack):
        with pytest.raises((ValueError, IndexError, KeyError)):
            reader(changed['path'], expected_sha256=changed['sha256'])


@pytest.mark.parametrize('change', ['scope','actual','native','BGR','exclude','material',
    'nonstatic','pixel_unconfirmed','reviewer','review_note','source_evidence_hash',
    'identity_projection_claim'])
def test_each_template_evidence_gate_rejects_with_same_public_reviewer(tmp_path, change):
    reference = _fixture_pack(tmp_path/'source')
    def edit(values, headers, proofs, metadata):
        proof = proofs[0]
        if change == 'scope': proof['scope']='stand'
        elif change == 'actual': proof['actual_pts']=200
        elif change == 'native': proof['native_pts']+=1
        elif change == 'BGR': proof['decoded_BGR_sha256']='c'*64
        elif change == 'exclude': proof['decision']='exclude'
        elif change == 'material': proof['physical_material_id']='another_confirmed_material'
        elif change == 'nonstatic': proof['static_material_confirmed']=False
        elif change == 'pixel_unconfirmed': proof['pixel_location_confirmed']=False
        elif change == 'reviewer': proof['reviewer']=''
        elif change == 'review_note': proof['review_note']=''
        elif change == 'source_evidence_hash': proof['source_image_evidence']['sha256']='f'*64
        elif change == 'identity_projection_claim': proof['model_projection_or_fit_residual_used_for_identity']=True
    _bad(edit, reference, tmp_path)


@pytest.mark.parametrize('change', ['negative_uv','off_image_uv','nan_uv','duplicate_bytes',
    'negative_proof_code','absent_proof_code','paint_not_midpoint','false_sigma_status','false_pack_ready'])
def test_every_original_point_and_support_gate_remains_active(tmp_path, change):
    reference = _fixture_pack(tmp_path/'source')
    def edit(values, headers, proofs, metadata):
        if change == 'negative_uv': values['uv'][0,0]=-1.
        elif change == 'off_image_uv': values['uv'][0,0]=1920.
        elif change == 'nan_uv': values['uv'][0,0]=np.nan
        elif change == 'duplicate_bytes':
            values['uv'][1]=values['uv'][0]
            values['source_sides_uv'][1]=values['source_sides_uv'][0]
        elif change == 'negative_proof_code': values['source_proof_template_index'][0]=-1
        elif change == 'absent_proof_code': values['source_proof_template_index'][0]=len(proofs)
        elif change == 'paint_not_midpoint': values['uv'][0,0]+=.000001
        elif change == 'false_sigma_status': headers[0]['working_sigma_status']='explicit_source_working_sigma'
        elif change == 'false_pack_ready': metadata['source_uncertainty_ready_for_fit']=True
    _bad(edit, reference, tmp_path)


def test_exact_source_pixel_bytes_keep_signed_zero_distinct(tmp_path):
    reference = _fixture_pack(tmp_path/'source', board=True, split='held')
    def edit(values, headers, proofs, metadata):
        values['uv'][0]=[0.,410.]
        values['uv'][1]=[-0.,410.]
    changed = _mutate(reference, tmp_path/'signed_zero.npz', edit)
    before, after = _readers(changed)
    assert before.values['uv'][0].tobytes() != before.values['uv'][1].tobytes()
    assert after.values['uv'].tobytes() == before.values['uv'].tobytes()
    assert len(after.values['uv'])==3


def test_source_index_hash_binding_is_still_enforced_before_payload_read(tmp_path):
    reference = _fixture_pack(tmp_path/'source')
    for reader in (StrictAbsoluteSourcePack, StrictDatasetValidatedPack):
        with pytest.raises(ValueError):
            reader(reference['path'], expected_sha256='f'*64)


def _mixed_manifest(tmp_path):
    from calibration.program_tracking.strict_source_disjoint_merge import merge_disjoint_strict_absolute_source_packs
    from calibration.program_tracking.strict_source_fit_dataset import SCHEMA, reference, partition_support
    fit = _fixture_pack(tmp_path/'fit')
    held = _fixture_pack(tmp_path/'held', pts=200, split='held', board=True)
    empty = _fixture_pack(tmp_path/'empty', pts=300, empty=True)
    merged = merge_disjoint_strict_absolute_source_packs([fit,held,empty],tmp_path/'merged.npz')
    pack = StrictAbsoluteSourcePack(merged['path'],expected_sha256=merged['sha256'])
    indices=tmp_path/'indices.npz'
    selected=partition_support(pack.rows,pack.values['row_offsets'],pack.values['sigma_px'])
    np.savez_compressed(indices,row_offsets=pack.values['row_offsets'],
        promoted_held_rows=np.array([],np.int64),pending_named_point_indices=np.array([],np.int64),**selected)
    path=tmp_path/'manifest.json'
    path.write_text(json.dumps(dict(schema=SCHEMA,source=pack.metadata['source'],
        expected_actual_pts=[100,200,300],fit_points=2,held_points=3,
        cuts=[dict(source_pack=reference(merged['path']),support_index=reference(indices))])),encoding='utf8')
    return reference(path),reference(merged['path'])


def _exact_equal(left,right):
    if isinstance(left,np.ndarray):
        assert isinstance(right,np.ndarray)
        assert left.dtype==right.dtype and left.shape==right.shape and left.tobytes()==right.tobytes()
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left:_exact_equal(left[key],right[key])
    elif isinstance(left,list):
        assert len(left)==len(right)
        for a,b in zip(left,right):_exact_equal(a,b)
    else:assert left==right


def test_single_pack_pass_compact_FIT_HELD_source_rows_match_original_two_role_consumer(tmp_path,monkeypatch):
    from calibration.program_tracking.strict_source_fit_dataset_stream import iter_dataset_rows
    from calibration.solver_runtime.strict_source_dataset_adapter import compact_dataset_rows
    from calibration.solver_runtime import strict_dataset_read_runtime as runtime
    manifest,source=_mixed_manifest(tmp_path)
    expected=[compact_dataset_rows(iter_dataset_rows(manifest,split=role),split=role)
        for role in ('fit','held')]
    calls=[];actual_reader=runtime.StrictDatasetValidatedPack
    def observed(*args,**kwargs):
        calls.append(args[0]);return actual_reader(*args,**kwargs)
    monkeypatch.setattr(runtime,'StrictDatasetValidatedPack',observed)
    fit,held,receipt=runtime.load_strict_dataset_rows_once(manifest)
    _exact_equal(expected[0],fit);_exact_equal(expected[1],held)
    assert len(calls)==1 and receipt['full_source_pack_passes']==1
    assert sum(len(row['uv']) for row in fit)==2
    assert sum(len(row['uv']) for row in held)==3
    assert np.isnan(held[0]['sigma']).all()
    assert fit[0]['source_locators'][0]['local_point_indices']==[0,2]


def test_thin_factor_rows_cannot_hide_future_explicit_loss_metadata(tmp_path):
    from calibration.program_tracking.strict_source_fit_dataset import reference
    from calibration.solver_runtime.strict_dataset_read_runtime import load_strict_dataset_rows_once
    manifest,source=_mixed_manifest(tmp_path)
    def edit(values,headers,proofs,metadata):headers[0]['robust_loss']='future_explicit_loss'
    changed=_mutate(source,tmp_path/'loss_declared.npz',edit)
    packet=json.loads(open(manifest['path'],encoding='utf8').read())
    packet['cuts'][0]['source_pack']=changed
    path=tmp_path/'loss_manifest.json';path.write_text(json.dumps(packet),encoding='utf8')
    with pytest.raises(ValueError,match='Explicit dataset loss policy'):
        load_strict_dataset_rows_once(reference(path))
