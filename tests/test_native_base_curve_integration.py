from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from calibration.solver_runtime import install
install()
from calibration.solver_runtime.curve_source_grouping import (
    NATIVE_BASE_SCHEMA, wrap_native_base_curve_integration)
from calibration.solver_runtime.source_point_qualification import file_sha256, original_uv_sha256
from calibration.solver_runtime.source_loss import weighted_source_cost, source_irls_weights
from rinkspace.source_sampling import source_support_integration_weights
from z190_streaming_source_normal import SourceRowBlock, accumulate_source_normal, replay_source_cost


def setup(tmp_path):
    def write(name, data):
        path = tmp_path/name
        path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf8')
        return dict(path=str(path), sha256=file_sha256(path))
    policies, factors, blocks, refs = [], {}, [], {}
    for family, attribute, stream in (
        ('native_ice', 'native_ice_factor', 'additional_real_native_ice_curves'),
        ('native_board', 'native_board_factor', 'additional_real_native_board_edges')):
        uv = np.array([[0., 0.], [10., 0.], [20., 0.]])
        curve = dict(kind='authenticated_line', feature_id=family)
        row = dict(id=family, split='fit', uv=uv.tolist(), sigma_px=6., curve=curve)
        raw = write(family+'.json', dict(rows=[row]))
        trace = write(family+'_trace.json', dict(user_stroke_raw_native_px=uv.tolist(), physical_curve=curve))
        runtime = dict(row, uv=uv, sigma=np.full(3, 6.), actual_pts=1)
        scales = np.full(6, 6.)
        factors[attribute] = SimpleNamespace(packet_path=raw['path'], rows=[runtime], row_count=6,
            scales=scales, contract=dict(packet_sha256=raw['sha256']))
        refs['ice' if family == 'native_ice' else 'board'] = raw
        mass = source_support_integration_weights(uv, [[0, 1, 2]])
        canonical = json.dumps(row, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf8')
        policies.append(dict(id=family, family=family, packet_row_index=0, source_row_id=family,
            source_row_reference=dict(raw, format='json', pointer='/rows/0'),
            original_row_canonical_json_sha256=hashlib.sha256(canonical).hexdigest(),
            original_uv_float64_sha256=original_uv_sha256(row), uv=uv.tolist(),
            native_active_base_scalar_pairs=np.arange(6).reshape(-1, 2).tolist(), stored_sigma_px=6., working_sigma_px=6.,
            prior_sampling_multiplier=1., physical_identity_basis='registered_direct_user_complete_stroke',
            physical_identity_authority=trace, authenticated_registered_trace_indices=[0, 1, 2], curve=curve,
            actual_pts=1, split='fit', source_video_sha256='a'*64, physical_curve_id=family, operator_id='native vector',
            source_integration_paths=[[0, 1, 2]], point_integration_mass=mass.tolist(),
            point_integration_weights=mass.tolist(), sampling_kind=['measured']*3))
        r=np.array([-.4, 4., -7., 10., 3., .2])
        J=csr_matrix(np.arange(12.).reshape(6, 2)/10.)
        blocks.append(SourceRowBlock(r, J, np.zeros(6, bool), scales,
            dict(family=stream, original_start=0, original_stop=6)))
    centre = SourceRowBlock(np.array([3., -9.]), csr_matrix([[1., 0.], [0., 1.]]),
        np.ones(2, bool), np.full(2, 6*np.sqrt(3)), dict(family='direct_manual_centres', original_start=12, original_stop=14))
    blocks.append(centre)
    class Source(SimpleNamespace):
        def blocks(self,q,jacobian=True): yield from blocks
    base = Source(**factors, contract=dict(original=True), count=2, updater=None,
        scales=np.r_[factors['native_ice_factor'].scales, factors['native_board_factor'].scales, centre.scales],
        quadratic=np.r_[np.zeros(12, bool), np.ones(2, bool)])
    policy=dict(schema=NATIVE_BASE_SCHEMA, packets=refs, apply_to_base_native_blocks_only=True,
        do_not_apply_to_mean_normalized_direct_manual_centre_blocks=True, records=policies)
    path=tmp_path/'policy.json';path.write_text(json.dumps(policy),encoding='utf8')
    return base, path, blocks, policy


def test_actual_source_block_robust_cost_gradient_H_and_unchanged_centre(tmp_path):
    base,path,blocks,_=setup(tmp_path)
    wrapper=wrap_native_base_curve_integration(base,path)
    revised=list(wrapper.blocks(np.zeros(2)))
    assert revised[-1] is blocks[-1]
    assert wrapper.scales is base.scales and wrapper.quadratic is base.quadratic
    for old,new in zip(blocks[:2], revised[:2]):
        assert new.r is old.r and new.J is old.J and new.scales is old.scales
        a=np.repeat([.25,.5,.25],2)
        assert np.array_equal(new.loss_weights.integration_weights,a)
        normal=accumulate_source_normal(lambda:iter([new]),2,expected_row_count=6,frozen_profile_token='same',retain_native_replay=True,source_state=np.zeros(2))
        robust=source_irls_weights(old.r,old.quadratic)
        assert normal.cost==weighted_source_cost(old.r,old.quadratic,a)
        assert np.allclose(normal.gradient,old.J.T@(a*robust*old.r))
        assert np.allclose(normal.normal.toarray(),old.J.toarray().T@((a*robust)[:,None]*old.J.toarray()))
        with pytest.raises(ValueError):
            replay_source_cost(lambda:iter([old]),normal.contract,frozen_profile_token='same')


def test_default_none_exact_and_duplicate_budget_application_rejected(tmp_path):
    base,path,blocks,_=setup(tmp_path)
    assert wrap_native_base_curve_integration(base,None) is base
    wrapper=wrap_native_base_curve_integration(base,path)
    first=wrapper.apply_block(blocks[0])
    with pytest.raises(ValueError,match='applied twice'):wrapper.apply_block(first)


def test_declared_packet_sigma_budget_separation_preserves_real_quadratic_normal(tmp_path):
    base,path,blocks,policy=setup(tmp_path)
    # This row's already admitted budget is encoded in stored sigma. It is
    # independent of the existing policy record and must not get normalized
    # twice. The actual NativeIce path is quadratic.
    raw=tmp_path/'native_ice.json';packet=json.loads(raw.read_text(encoding='utf8'))
    row=packet['rows'][0];weights=source_support_integration_weights(row['uv'],[[0,1,2]])
    stored=6./np.sqrt(weights)
    row.update(sigma_px=stored.tolist(),original_working_sigma_px=[6.]*3,
        source_integration_weights=weights.tolist(),source_visible_paths=[[0,1,2]])
    raw.write_text(json.dumps(packet),encoding='utf8');reference=dict(path=str(raw),sha256=file_sha256(raw))
    policy['packets']['ice']=reference;policy['records']=policy['records'][1:]
    factor=base.native_ice_factor;factor.contract['packet_sha256']=reference['sha256']
    factor.rows[0].update(sigma_px=stored.tolist(),sigma=stored.copy());factor.scales[:]=np.repeat(stored,2)
    blocks[0].quadratic[:]=True
    path.write_text(json.dumps(policy),encoding='utf8')
    old=wrap_native_base_curve_integration(base,path).apply_block(blocks[0])
    policy['separate_declared_packet_sampling_normalization']=True
    path.write_text(json.dumps(policy),encoding='utf8')
    owner=wrap_native_base_curve_integration(base,path);new=owner.apply_block(blocks[0])
    assert new.r is old.r and new.J is old.J and new.scales is old.scales
    a,b=[accumulate_source_normal(lambda:iter((v,)),2,expected_row_count=6,frozen_profile_token='same')for v in (old,new)]
    assert np.isclose(a.cost,b.cost,rtol=2e-15)
    assert np.allclose(a.gradient,b.gradient,rtol=2e-15,atol=1e-15)
    assert np.allclose(a.normal.toarray(),b.normal.toarray(),rtol=2e-15,atol=1e-15)
    assert len(owner.native_base_curve_integration['declared_packet_sampling_normalizations'])==1
    # If this explicit decomposition is used with a robust source, its
    # threshold is working sigma6, not the historic point-density multiplier.
    mask=np.zeros(6,bool);working=blocks[0].r*np.repeat(stored/6.,2)
    expected=float(np.repeat(weights,2)@np.where(abs(working)<=2.,.5*working**2,2.*abs(working)-2.))
    assert weighted_source_cost(new.r,mask,new.loss_weights)==expected
    assert weighted_source_cost(old.r,mask,old.loss_weights)!=expected
    row['source_integration_weights'][0]+=.1;raw.write_text(json.dumps(packet),encoding='utf8')
    policy['packets']['ice']['sha256']=file_sha256(raw);factor.contract['packet_sha256']=file_sha256(raw)
    path.write_text(json.dumps(policy),encoding='utf8')
    with pytest.raises(ValueError,match='support changed'):wrap_native_base_curve_integration(base,path)


@pytest.mark.parametrize('mutation', ['uv','slots','sigma','held','trace'])
def test_native_binding_fail_closed(tmp_path,mutation):
    base,path,_,policy=setup(tmp_path)
    if mutation=='uv':policy['records'][0]['uv'][0][0]=1.
    elif mutation=='slots':policy['records'][0]['native_active_base_scalar_pairs'][0]=[2,3]
    elif mutation=='sigma':policy['records'][0]['stored_sigma_px']=7.
    elif mutation=='held':policy['records'][0]['source_row_reference']['pointer']='/rows/9'
    else:policy['records'][0]['authenticated_registered_trace_indices'][0]=1
    path.write_text(json.dumps(policy),encoding='utf8')
    with pytest.raises((ValueError, IndexError)):wrap_native_base_curve_integration(base,path)


def rgb_boundary_setup(tmp_path, mutation=None):
    base, path, blocks, policy = setup(tmp_path)
    record = policy['records'][1]
    packet_path = tmp_path/'native_board.json'
    packet = json.loads(packet_path.read_text(encoding='utf8'))
    row = packet['rows'][0]
    raw = tmp_path/'original_RGB.bytes'
    raw.write_bytes(b'original native pixels: identity fixture')
    row.update(normal=[[0., 1.]]*3, normal_interval_px=[[-8., 8.]]*3,
               source_raw_frame=str(raw), source_file_sha256=file_sha256(raw))
    measured = dict(deepcopy(row), actual_pts=1,
        source_integration_paths=record['source_integration_paths'],
        point_integration_mass=record['point_integration_mass'])
    if mutation == 'held': measured['split'] = 'holdout'
    elif mutation == 'exposure': measured['actual_pts'] = 2
    elif mutation == 'sigma': measured['sigma_px'] = 9.
    elif mutation == 'normal': measured['normal'][0] = [1., 0.]
    receipt = tmp_path/'measured_boundary.json'
    receipt.write_text(json.dumps(dict(
        schema='rinkspace.actual-RGB-measured-boundary-location-receipt.v1',
        source_video_sha256=record['source_video_sha256'],
        records=[dict(measured_boundary_row=measured)])), encoding='utf8')
    reference = dict(path=str(receipt), sha256=file_sha256(receipt))
    row.update(boundary_location_observation='measured_transition_midpoint',
               boundary_location_measurement=reference)
    packet_path.write_text(json.dumps(packet), encoding='utf8')
    packet_ref = dict(path=str(packet_path), sha256=file_sha256(packet_path))
    policy['packets']['board'] = packet_ref
    record.update(source_row_reference=dict(packet_ref, format='json', pointer='/rows/0'),
        original_row_canonical_json_sha256=hashlib.sha256(json.dumps(row, sort_keys=True,
            separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf8')).hexdigest(),
        physical_identity_basis='authenticated_original_RGB_boundary_locations',
        physical_identity_authority=reference, authenticated_rgb_measurement_index=0)
    base.native_board_factor.contract['packet_sha256'] = packet_ref['sha256']
    base.native_board_factor.rows[0].update(row, uv=np.asarray(row['uv']), actual_pts=1)
    path.write_text(json.dumps(policy), encoding='utf8')
    if mutation == 'pixels': raw.write_bytes(b'changed actual pixels')
    return base, path, blocks


def test_original_RGB_boundary_budget_changes_loss_only_and_preserves_human_centres(tmp_path):
    base, path, original = rgb_boundary_setup(tmp_path)
    revised = list(wrap_native_base_curve_integration(base, path).blocks(np.zeros(2)))
    assert revised[-1] is original[-1]
    board = revised[1]
    assert board.r is original[1].r and board.J is original[1].J
    assert board.scales is original[1].scales
    assert np.array_equal(board.loss_weights.integration_weights, [.25, .25, .5, .5, .25, .25])
    expected = sum(w * weighted_source_cost(original[1].r[i:i+2],
        original[1].quadratic[i:i+2], None) for i, w in zip((0, 2, 4), (.25, .5, .25)))
    assert weighted_source_cost(board.r, board.quadratic, board.loss_weights.integration_weights) == expected


@pytest.mark.parametrize('mutation', ['held', 'exposure', 'sigma', 'normal', 'pixels'])
def test_original_RGB_boundary_budget_rejects_changed_measurement(tmp_path, mutation):
    base, path, _ = rgb_boundary_setup(tmp_path, mutation)
    with pytest.raises(ValueError):
        wrap_native_base_curve_integration(base, path)


@pytest.mark.parametrize('mutation',[None,'inner','role','operator','support','raw_pixel'])
def test_original_RGB_house_outer_material_budget_exact_typed_role(tmp_path,mutation):
    base,path,blocks,policy=setup(tmp_path);record=policy['records'][0]
    packet_path=tmp_path/'native_ice.json';packet=json.loads(packet_path.read_text(encoding='utf8'));row=packet['rows'][0]
    raw=tmp_path/'actual_native_RGB.bytes';raw.write_bytes(b'actual original native pixels')
    curve=dict(kind='house_ring',side='F',row=2,ring='outer')
    row.update(curve=curve,normal=[[0.,1.]]*3,normal_interval_px=None,
        source_raw_frame=str(raw),source_file_sha256=file_sha256(raw),
        measurement_role='filled_outer_material_boundary',measurement_operator='integrated_actual_signed_luminance_transition_q50')
    measured=dict(deepcopy(row),actual_pts=1,source_integration_paths=record['source_integration_paths'],
        point_integration_mass=record['point_integration_mass'])
    receipt=tmp_path/'measured_F2.json';receipt.write_text(json.dumps(dict(
        schema='rinkspace.actual-RGB-measured-boundary-location-receipt.v1',source_video_sha256='a'*64,
        records=[dict(measured_boundary_row=measured)])),encoding='utf8')
    reference=dict(path=str(receipt),sha256=file_sha256(receipt));row['actual_RGB_measurement_receipt']=reference
    if mutation=='inner':row['curve']['ring']='inner'
    elif mutation=='role':row['measurement_role']='painted_line_center'
    elif mutation=='operator':row['measurement_operator']='model_projection'
    elif mutation=='support':row['normal_interval_px']=[[-5.,5.]]*3
    elif mutation=='raw_pixel':raw.write_bytes(b'changed original pixels')
    packet_path.write_text(json.dumps(packet),encoding='utf8');packetref=dict(path=str(packet_path),sha256=file_sha256(packet_path))
    policy['packets']['ice']=packetref
    record.update(curve=row['curve'],source_row_reference=dict(packetref,format='json',pointer='/rows/0'),
        original_row_canonical_json_sha256=hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest(),
        physical_identity_basis='authenticated_original_RGB_boundary_locations',physical_identity_authority=reference,authenticated_rgb_measurement_index=0)
    factor=base.native_ice_factor;factor.contract['packet_sha256']=packetref['sha256'];factor.rows[0].update(row,uv=np.asarray(row['uv']),actual_pts=1)
    path.write_text(json.dumps(policy),encoding='utf8')
    if mutation is None:
        owner=wrap_native_base_curve_integration(base,path);actual=owner.apply_block(blocks[0])
        assert actual.r is blocks[0].r and actual.J is blocks[0].J and actual.scales is blocks[0].scales
        assert np.array_equal(actual.loss_weights.integration_weights,[.25,.25,.5,.5,.25,.25])
    else:
        with pytest.raises(ValueError):wrap_native_base_curve_integration(base,path)
