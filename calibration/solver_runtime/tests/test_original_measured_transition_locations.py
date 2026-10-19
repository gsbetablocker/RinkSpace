from copy import deepcopy
import hashlib
import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from calibration.solver_runtime import original_measured_transition_locations as source


def fixture_packet(tmp_path):
    rows = []
    for index in range(2):
        row = dict(id='Ad7:left', pts=123+index, split='fit', origin='saved_rgb',
            uv=np.array([[4., 8.], [5., 9.]]), sigma=np.array([3., 4.]),
            normal=np.array([[1., 0.], [1., 0.]]),
            normal_interval=np.array([[-2., 2.], [-3., 3.]]),
            world=np.array([[0., 0., 0.], [0., 1., 0.]]))
        rows.append(dict(block='main1', collection_role='FIT', group_actual_pts=123+index,
            group_order=index, row_order=0, source_row=row))
    original = tmp_path/'original.pkl'; original.write_bytes(pickle.dumps({'programme_absolute_rows': rows}))
    receipt = tmp_path/'source.json'
    receipt.write_text(json.dumps({'samples': [{'uv': [4., 8.], 'q': [2., 4., 6.]},
        {'uv': [5., 9.], 'q': [2., 5., 8.]}]}), encoding='utf8')
    producer = tmp_path/'producer.py'; producer.write_text('# original source estimator', encoding='utf8')
    records = []
    for i, item in enumerate(rows):
        records.append(dict(original_packet_row_index=i,
            locator={k: item[k] for k in source.LOCATOR_KEYS}, source_row_id=item['source_row']['id'],
            original_source_row_sha256=hashlib.sha256(source.canonical(item['source_row'])).hexdigest(),
            point_indices=[0, 1], measured_uv=item['source_row']['uv'].tolist(),
            source_reference=source.reference(receipt), producer_references=[source.reference(producer)],
            estimator_kind='positive_gradient_q50', measurement_bindings=[dict(point_index=p,
                measured_axis=0, location_path=['samples', p, 'q', 1],
                observed_uv_path=['samples', p, 'uv']) for p in range(2)]))
    packet = dict(schema=source.SCHEMA, original_absolute_packet=source.reference(original), records=records)
    path=tmp_path/'qualified.json'; path.write_text(json.dumps(packet), encoding='utf8')
    return path, packet, rows


def save(path, value):
    path.write_text(json.dumps(value), encoding='utf8')


def test_original_q50_has_independent_saved_authority(tmp_path):
    path, _, _ = fixture_packet(tmp_path)
    _, _, contract = source.validate_packet(path)
    assert sum(r['points'] for r in contract['records']) == 4
    assert contract['original_UV_sigma_normal_interval_world_identity_and_split_unchanged']


@pytest.mark.parametrize('mutation', ['q50', 'partial_points', 'source_bytes', 'row_sigma', 'fixed_uv', 'held'])
def test_rejects_unmeasured_or_changed_source(tmp_path, mutation):
    path, packet, rows = fixture_packet(tmp_path)
    if mutation in ('q50', 'fixed_uv'):
        ref = packet['records'][0]['source_reference']; receipt = source.read(ref['path'])
        receipt['samples'][0]['q' if mutation == 'q50' else 'uv'][1] += 1.
        save(__import__('pathlib').Path(ref['path']), receipt)
        ref.update(source.reference(ref['path']))
        packet['records'][1]['source_reference'] = deepcopy(ref)
    elif mutation == 'partial_points':
        packet['records'][0]['measurement_bindings'].pop()
    elif mutation == 'source_bytes':
        __import__('pathlib').Path(packet['records'][0]['source_reference']['path']).write_text('{}')
    else:
        if mutation == 'held': rows[0]['collection_role'] = 'HELD'
        else: rows[0]['source_row']['sigma'][0] = 100.
        original = __import__('pathlib').Path(packet['original_absolute_packet']['path'])
        original.write_bytes(pickle.dumps({'programme_absolute_rows': rows}))
        packet['original_absolute_packet'] = source.reference(original)
    save(path, packet)
    with pytest.raises(ValueError): source.validate_packet(path)


def test_failed_later_row_does_not_partially_install(tmp_path):
    path, _, original = fixture_packet(tmp_path)
    groups=[(r['group_actual_pts'], [deepcopy(r['source_row'])]) for r in original]
    groups[1][1][0]['uv'][0, 0] += 1.
    block=SimpleNamespace(name='main1', groups=groups, bundle=SimpleNamespace(abs_groups=groups))
    before=pickle.dumps(groups)
    with pytest.raises(ValueError, match='Runtime source differs'):
        source.install_original_measured_transition_locations(SimpleNamespace(blocks=[block]), path)
    assert pickle.dumps(block.groups) == before


def test_install_preserves_all_observed_fields_and_source_layout(tmp_path):
    path, _, original = fixture_packet(tmp_path)
    groups=[(r['group_actual_pts'], [deepcopy(r['source_row'])]) for r in original]
    block=SimpleNamespace(name='main1', groups=groups, bundle=SimpleNamespace(abs_groups=groups))
    state=SimpleNamespace(blocks=[block])
    contract=source.install_original_measured_transition_locations(state, path)
    assert contract is getattr(state, source.CONTRACT_KEY)
    from calibration.solver_runtime.original_measured_location_mode import uses_original_normal_support
    for (_, rows), item in zip(block.groups, original):
        assert not uses_original_normal_support(rows[0])
        for key, value in item['source_row'].items():
            assert source.canonical(rows[0][key]) == source.canonical(value)


def truth_case(tmp_path):
    from calibration.solver_runtime import truth_original_location_revision as proof
    from calibration.solver_runtime import truth_source_location_revision as common
    path, _, _ = fixture_packet(tmp_path)
    manifest = dict(full_parameters=2, source_rows=8, source_contract={'motion': {'unchanged': True}},
        source_jacobian_reuse={'enabled': False})
    mp=tmp_path/'manifest.json'; save(mp, manifest)
    q=np.array([1., 2.]); arrays=tmp_path/'state.npz'; np.savez(arrays, q=q)
    authority=tmp_path/'authority.json'; save(authority, {'physical_support': [-1., 1.]})
    state=dict(complete_state_reference=source.reference(arrays),
        source_input_manifest_reference=source.reference(mp),
        complete_q_sha256=hashlib.sha256(q.tobytes()).hexdigest(), source_target=common._target(manifest),
        families={'native': dict(authority=source.reference(authority), group_ids=['keep'],
            limits=[.01], uncertainties=[.002])})
    truth=tmp_path/'truth.json'; save(truth, state)
    prepared=proof.prepare_original_location_revision(truth, {}, path, tmp_path/'derived')
    _, _, contract=source.validate_packet(path)
    manifest[source.CONTRACT_KEY]=contract; manifest['source_contract'][source.CONTRACT_KEY]=contract
    factors={'native': SimpleNamespace(contract={'authority': source.reference(authority)}, records=[{'row': {'id': 'keep'}}])}
    return common, prepared['proof_reference']['path'], truth, q, manifest, factors


def test_truth_caps_keep_exact_previous_values(tmp_path):
    common, proof, truth, q, manifest, factors=truth_case(tmp_path)
    before=source.read(truth)
    result=common.authenticate_location_revision(proof, truth, q, manifest, factors)
    assert result['packet']['families'] == before['families']
    assert source.read(truth) == before
    assert result['migration']['actual_current_source_target'] != before['source_target']


@pytest.mark.parametrize('changed', ['q', 'other_loss', 'counts', 'J_reuse', 'authority', 'integration', 'contract'])
def test_truth_revision_rejects_unrelated_changes(tmp_path, changed):
    common, proof, truth, q, manifest, factors=truth_case(tmp_path)
    if changed == 'q': q[0] += 1
    elif changed == 'other_loss': manifest['source_contract']['motion']['unchanged']=False
    elif changed == 'counts': manifest['source_rows'] += 2
    elif changed == 'J_reuse': manifest['source_jacobian_reuse']['enabled']=True
    elif changed == 'authority': factors['native'].contract['authority']={'path': 'different', 'sha256': 'different'}
    elif changed == 'integration': manifest['original_board_curve_integration']={'weight': 2}
    else: manifest[source.CONTRACT_KEY]=dict(manifest[source.CONTRACT_KEY], observation='support')
    with pytest.raises(ValueError): common.authenticate_location_revision(proof, truth, q, manifest, factors)


def test_composes_exact_native_scanline_revision_without_resetting_caps(tmp_path):
    import importlib.util
    from pathlib import Path
    from calibration.solver_runtime import truth_original_location_revision as proof
    from calibration.solver_runtime import truth_source_location_revision as common
    fixture_file=Path(__file__).resolve().parents[3]/'tests/test_truth_source_scanline_revision.py'
    spec=importlib.util.spec_from_file_location('scanline_revision_test_fixture', fixture_file)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    native_dir=tmp_path/'native'; native_dir.mkdir()
    native=module.case.__wrapped__(native_dir)
    original_dir=tmp_path/'original'; original_dir.mkdir()
    path, _, _=fixture_packet(original_dir)
    prepared=proof.prepare_original_location_revision(native.truth, native.prepared['source_options'],
        path, tmp_path/'composed', base_operator_revision_proof=native.prepared['proof_reference']['path'])
    manifest=deepcopy(native.manifest); _, _, contract=source.validate_packet(path)
    manifest[source.CONTRACT_KEY]=contract; manifest['source_contract'][source.CONTRACT_KEY]=contract
    result=common.authenticate_location_revision(prepared['proof_reference']['path'], native.truth,
        native.q, manifest, native.factors)
    for family, before in native.packet['families'].items():
        for field in ('group_ids', 'limits', 'uncertainties'):
            assert result['packet']['families'][family][field] == before[field]
    manifest['source_contract']['board']['unexpected_weight']=2
    with pytest.raises(ValueError):
        common.authenticate_location_revision(prepared['proof_reference']['path'], native.truth,
            native.q, manifest, native.factors)
