from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from calibration.solver_runtime import truth_source_location_revision as api


def save(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return api.reference(path)


def digest(value):
    return hashlib.sha256(api.canonical(value)).hexdigest()


@pytest.fixture(params=[(api.RECEIPT_SCHEMA, 'measured_transition_midpoint'),
                        (api.LOCATION_RECEIPT_SCHEMA, 'measured_transition_location')])
def case(tmp_path, request):
    receipt_schema, mode = request.param
    receipt0 = save(tmp_path / 'existing_measurement.json', {'existing': True})
    rows = [dict(id=f'board{i}', split='holdout' if i == 1 else 'fit', actual_pts=10+i,
                 uv=[[2., 3.]], sigma_px=[4.], normal=[[1., 0.]],
                 normal_interval_px=[[-20., 20.]],
                 curve={'kind': 'advertisement_boundary', 'edge': 'left_vertical'},
                 source_raw_frame='unchanged.png', source_file_sha256='same_image') for i in range(3)]
    rows[2].update(boundary_location_observation='measured_transition_midpoint',
                   boundary_location_measurement=receipt0)
    old_path = tmp_path / 'old_board.json'
    old_ref = save(old_path, {'schema': 'board-packet', 'source_video_sha256': 'same_video', 'rows': rows})
    receipt_path = tmp_path / 'receipt.json'
    receipt = dict(schema=receipt_schema, rows=[dict(original_row_index=0, id='board0', split='fit',
        source_row_reference=old_ref, source_row_canonical_json_sha256=digest(rows[0]),
        measured_boundary_row=rows[0])])
    receipt_ref = save(receipt_path, receipt)
    new = api._read(old_path)
    new['rows'][0].update(boundary_location_observation=mode,
                          boundary_location_measurement=receipt_ref)
    new[api.REVISION_KEY] = dict(parent_packet_reference=old_ref, receipt_reference=receipt_ref,
        changed_row_indices=[0], changed_row_ids=['board0'],
        all_original_source_rows_except_two_location_fields_unchanged=True)
    new_path = tmp_path / 'new_board.json'; save(new_path, new)
    options = {api.BOARD: str(old_path)}
    for key, index in ((api.NATIVE, 2), (api.CONFIRMED, 0)):
        row = rows[index]
        path = tmp_path / ('native.json' if key == api.NATIVE else 'confirmed.json')
        save(path, {'records': [dict(id=row['id'], family='native_board_edge', packet_reference=old_ref,
            original_row_index=index, source_row_canonical_json_sha256=digest(row), uv=row['uv'],
            sigma_px=row['sigma_px'], support=[[-20., 20.]])]})
        options[key] = str(path)
    arrays = tmp_path / 'guard.npz'; np.savez(arrays, uv=np.array([[2., 3.]]))
    docs = {
        api.GUARD: dict(primary_authority_reference=api.reference(options[api.NATIVE]),
            guard_arrays_reference=api.reference(arrays), records=[dict(id='board2',
                authority_source_row_canonical_json_sha256=digest(rows[2]), vertex_count=1)]),
        api.CAP: dict(primary_authority_reference=api.reference(options[api.NATIVE]),
            full_trace_reference={}, records=[dict(id='board2', tangent_bounds=[-1., 1.])]),
        api.ENVELOPE: dict(authority_references={'native': api.reference(options[api.NATIVE])},
            records=[dict(id='board2', normal_expansion_px=2.)]),
        api.INTEGRATION: dict(packets={'board': old_ref, 'ice': receipt0}, records=[dict(
            id='curve0', family='native_board', source_row_reference=dict(old_ref, pointer='/rows/0'),
            packet_reference=old_ref, packet_row_index=0, original_row_canonical_json_sha256=digest(rows[0]),
            point_integration_weights=[1.], stored_sigma_px=[4.], working_sigma_px=[4.], uv=rows[0]['uv'])]),
    }
    for key, doc in docs.items():
        path = tmp_path / (key.lstrip('-') + '.json'); save(path, doc); options[key] = str(path)
    wide_ref = save(tmp_path / 'wide_authority.json', {'wide': True})
    board_contract = dict(schema='rinkspace.additional-real-native-board-edge-factor.v4',
        packet_path=old_ref['path'], packet_sha256=old_ref['sha256'], fit_ids=['board0', 'board2'],
        scalar_rows=4, measured_boundary_location_objective=dict(observation='measured_transition_midpoint',
            ids=['board2'], measurement_receipts={receipt0['path']: receipt0['sha256']},
            finite_curve_position_residual_uses_original_sigma=True,
            optical_transition_footprint_is_not_a_location_dead_zone=True,
            stored_UV_normal_interval_sigma_and_human_guards_unchanged=True))
    native_contract = dict(authority=api.reference(options[api.NATIVE]), groups=1,
        full_guard_reference=api.reference(options[api.GUARD]),
        stroke_caps=api.reference(options[api.CAP]), envelope=api.reference(options[api.ENVELOPE]))
    confirmed_contract = dict(authority=api.reference(options[api.CONFIRMED]), groups=1)
    integration = dict(policy_reference=api.reference(options[api.INTEGRATION]),
        source_packets={'native_board': old_ref}, groups=[{'budget': 1.}])
    manifest = dict(full_parameters=3, source_rows=8,
        additional_real_native_board_edges=board_contract,
        highest_direct_human_hard_truth={'native': native_contract, 'wide': {'authority': wide_ref}},
        highest_user_confirmed_native_sources=confirmed_contract, native_base_curve_integration=integration,
        source_contract={'board': board_contract, 'native': native_contract,
                         'confirmed': confirmed_contract, 'integration': integration,
                         'motion': {'rows': 4, 'sigma': [3., 3.]}}, source_jacobian_reuse={'enabled': False})
    manifest_path = tmp_path / 'manifest.json'; save(manifest_path, manifest)
    q = np.array([1., 2., 3.]); state_path = tmp_path / 'state.npz'; np.savez(state_path, q=q)
    packet = dict(schema='rinkspace.joint-source-truth-restoration-state.v1',
        complete_state_reference=api.reference(state_path), source_input_manifest_reference=api.reference(manifest_path),
        complete_q_sha256=hashlib.sha256(q.tobytes()).hexdigest(), source_target=api._target(manifest),
        families={k: dict(authority=ref, group_ids=[name], limits=[0. if k=='native' else .03],
                         uncertainties=[.004]) for k, ref, name in (
            ('native', native_contract['authority'], 'board2'),
            ('confirmed', confirmed_contract['authority'], 'board0'), ('wide', wide_ref, 'wide0'))})
    truth = tmp_path / 'truth.json'; save(truth, packet)
    prepared = api.prepare_location_revision(options, new_path, tmp_path / 'rebind',
                                             changed_row_indices=[0], truth_state_path=truth)
    derived = prepared['source_options']
    new_manifest = deepcopy(manifest)
    # Independent expected runtime metadata assembled from the unchanged actor
    # counts/weights and the new immutable file bindings.
    nc = dict(native_contract, authority=api.reference(derived[api.NATIVE]),
        full_guard_reference=api.reference(derived[api.GUARD]),
        stroke_caps=api.reference(derived[api.CAP]), envelope=api.reference(derived[api.ENVELOPE]))
    cc = dict(confirmed_contract, authority=api.reference(derived[api.CONFIRMED]))
    ic = dict(integration, policy_reference=api.reference(derived[api.INTEGRATION]),
              source_packets={'native_board': api.reference(new_path)})
    bc = deepcopy(board_contract); bc.update(packet_path=str(new_path.resolve()), packet_sha256=api.reference(new_path)['sha256'])
    bc['measured_boundary_location_objective'].update(observation=mode, ids=['board0', 'board2'],
        measurement_receipts={receipt_ref['path']: receipt_ref['sha256'], receipt0['path']: receipt0['sha256']})
    new_manifest.update(additional_real_native_board_edges=bc, highest_direct_human_hard_truth={'native':nc,'wide':{'authority':wide_ref}},
        highest_user_confirmed_native_sources=cc, native_base_curve_integration=ic,
        source_contract={'board':bc,'native':nc,'confirmed':cc,'integration':ic,'motion':manifest['source_contract']['motion']})
    factors = {k: SimpleNamespace(contract={'authority': rec['authority']},
        records=[{'record' if k=='wide' else 'row': {'id': rec['group_ids'][0]}}])
        for k, rec in packet['families'].items()}
    factors['native'].contract['authority'] = nc['authority']; factors['confirmed'].contract['authority'] = cc['authority']
    return SimpleNamespace(old_path=old_path, new_path=new_path, truth=truth, options=options,
        prepared=prepared, manifest=new_manifest, factors=factors, q=q, packet=packet, tmp=tmp_path)


def authenticate(c):
    return api.authenticate_location_revision(c.prepared['proof_reference']['path'], c.truth,
                                             c.q, c.manifest, c.factors)


def test_current_location_migration_preserves_caps_and_support(case):
    before = case.truth.read_bytes()
    result = authenticate(case)
    assert case.truth.read_bytes() == before
    assert result['migration']['fresh_normal_and_J_required'] is True
    assert result['migration']['actual_current_source_target'] != case.packet['source_target']
    for family, record in case.packet['families'].items():
        current = result['packet']['families'][family]
        for key in ('group_ids', 'limits', 'uncertainties'):
            assert current[key] == record[key]
    old_policy = api._read(case.options[api.INTEGRATION])
    new_policy = api._read(case.prepared['source_options'][api.INTEGRATION])
    for key in ('uv', 'stored_sigma_px', 'working_sigma_px', 'point_integration_weights'):
        assert old_policy['records'][0][key] == new_policy['records'][0][key]
    old_guard = api._read(case.options[api.GUARD])
    new_guard = api._read(case.prepared['source_options'][api.GUARD])
    assert old_guard['guard_arrays_reference'] == new_guard['guard_arrays_reference']


def test_actual_production_continuation_loader(case):
    from calibration.solver_runtime import install
    install()
    from run_current_named66_full_programme import load_truth_restoration_state
    result = load_truth_restoration_state(case.truth, case.q, case.manifest, case.factors,
        source_location_revision_proof=case.prepared['proof_reference']['path'])
    assert result['authenticated_source_location_revision']['fresh_normal_and_J_required']
    assert result['authenticated_source_location_revision']['protected_group_count'] == 3
    assert result['complete_q_sha256'] == case.packet['complete_q_sha256']
    for family in case.packet['families']:
        for key in ('group_ids', 'limits', 'uncertainties'):
            assert result['families'][family][key] == case.packet['families'][family][key]
    assert api._read(case.new_path)['rows'][1] == api._read(case.old_path)['rows'][1]


@pytest.mark.parametrize('has_actual_pts', [False, True])
def test_location_receipt_uses_original_clock_alias(case, has_actual_pts):
    old = api._read(case.old_path)
    original = old['rows'][0]
    if has_actual_pts:
        original['pts'] = 99  # explicit actual_pts retains precedence
    else:
        original['pts'] = original.pop('actual_pts')
    old_ref = save(case.old_path, old)
    new = api._read(case.new_path)
    modes = {k: new['rows'][0][k] for k in api.MODE_FIELDS}
    new['rows'][0] = dict(original, **modes)
    receipt_path = Path(modes[api.MODE_FIELDS[1]]['path'])
    receipt = api._read(receipt_path)
    receipt['rows'][0].update(source_row_reference=old_ref,
        source_row_canonical_json_sha256=digest(original))
    receipt_ref = save(receipt_path, receipt)
    new['rows'][0][api.MODE_FIELDS[1]] = receipt_ref
    new[api.REVISION_KEY].update(parent_packet_reference=old_ref, receipt_reference=receipt_ref)
    save(case.new_path, new)
    api.validate_location_revision(case.old_path, case.new_path, [0])
    receipt['rows'][0]['measured_boundary_row']['actual_pts'] += 1
    receipt_ref = save(receipt_path, receipt)
    new['rows'][0][api.MODE_FIELDS[1]] = receipt_ref
    new[api.REVISION_KEY]['receipt_reference'] = receipt_ref
    save(case.new_path, new)
    with pytest.raises(ValueError, match='original measured source row'):
        api.validate_location_revision(case.old_path, case.new_path, [0])


@pytest.mark.parametrize('field,value', [('uv', [[2.1,3.]]), ('sigma_px',[3.]),
    ('normal_interval_px',[[-40.,40.]]), ('split','holdout'), ('actual_pts',11)])
def test_location_revision_rejects_other_source_changes(case, field, value):
    new = api._read(case.new_path); new['rows'][0][field] = value; save(case.new_path, new)
    with pytest.raises(ValueError): api.validate_location_revision(case.old_path, case.new_path, [0])


def test_location_revision_rejects_holdout_promotion(case):
    new = api._read(case.new_path); new['rows'][1].update(boundary_location_observation='measured_transition_midpoint',
        boundary_location_measurement=new[api.REVISION_KEY]['receipt_reference']); save(case.new_path, new)
    with pytest.raises(ValueError): api.validate_location_revision(case.old_path, case.new_path, [0,1])


def test_location_revision_rejects_other_target_change(case):
    case.manifest['source_contract']['motion']['sigma'] = [30., 30.]
    with pytest.raises(ValueError, match='outside'): authenticate(case)


@pytest.mark.parametrize('change', ['q','group','reuse','authority','policy'])
def test_location_revision_rejects_wrong_continuation(case, change):
    if change == 'q': case.q[0] += .1
    elif change == 'group': case.factors['confirmed'].records[0]['row']['id'] = 'another_group'
    elif change == 'reuse': case.manifest['source_jacobian_reuse']['enabled'] = True
    elif change == 'authority': case.factors['native'].contract['authority'] = case.packet['families']['native']['authority']
    else:
        p=case.prepared['source_options'][api.INTEGRATION]; value=api._read(p)
        value['records'][0]['point_integration_weights']=[.5]; save(p, value)
    with pytest.raises(ValueError): authenticate(case)


@pytest.mark.parametrize('indices', [[0.], [True], [0,0], [1], []])
def test_exact_reviewed_row_inventory(case, indices):
    with pytest.raises(ValueError): api.validate_location_revision(case.old_path, case.new_path, indices)
