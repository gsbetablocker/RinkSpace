from copy import deepcopy
import json

import numpy as np
import pytest

from calibration.program_tracking.strict_absolute_source import (
    StrictAbsolutePackWriter, StrictAbsoluteSourcePack,
    normalize_selected_ice_rows, normalize_selected_board_rows,
)
from calibration.program_tracking.strict_model_scope import file_sha
from calibration.program_tracking.strict_model_scope import ExactSourceScopeReview, REVIEW_SCHEMA, write_motion_partition
from calibration.solver_runtime.strict_model_scope_inputs import StrictProgrammeSourceInputs


def fixture_inputs(tmp_path):
    original=tmp_path/'original_source.bin'
    original.write_bytes(b'Explicit test fixture, not programme evidence')
    seed=tmp_path/'source_identity_seed.json'
    seed.write_text('{"fixture_only":true}',encoding='utf8')
    binding=dict(source_sha256=file_sha(original),source_path=str(original),
        time_base='1/15360',native_pts=110,actual_pts=100,decoded_native_BGR_sha256='b'*64)
    temporal=dict(kind='confirmed_original_source_identity_anchor',physical_material_id='C.paint',
        shot_id='main1:0',current_actual_pts=100,same_physical_entity_confirmed=True,
        current_frame_RGB_measured=True,model_projection_or_fit_residual_used_for_identity=False,
        source_identity_authority=dict(path=str(seed),sha256=file_sha(seed)),source_identity_confirmed=True)
    row=dict(curve_id='C:segment0',physical_material_id='C.paint',actual_pts=100,shot_id='main1:0',
        original_candidate_indices=[0],admission_status='admitted_source_material_center_temporal_identity',
        static_material_confirmed=True,actual_source_RGB_measured=True,pixel_location_confirmed=True,
        model_projection_or_fit_residual_used_for_identity=False,temporal_evidence=temporal,sigma_px=2.)
    sides=np.array([[[871.33167,311.26328],[878.46466,312.4438]]],np.float32)
    measurement=dict(source_binding=deepcopy(binding),
        candidate_uv=sides.astype(float).mean(1).astype(np.float32),candidate_sides=sides,
        candidate_query=np.array([0]),query_curve=np.array([0]),candidate_flags=np.array([8],np.uint8))
    inventory=[dict(curve_id='C:segment0',feature_id='C',world_curve_role='segment0',operator='paint_center')]
    return binding,row,measurement,inventory


def test_working_board_support_requires_actual_adjacent_pixel_normal():
    from calibration.program_tracking.strict_absolute_source import _measured_board_readout_support
    row=dict(uv=[[3.,10.],[7.,10.],[11.,10.]],normal=[[0.,1.]]*3,
        normal_from_actual_source_neighbour_pixels=True,source_neighbour_adjacency_confirmed=True,
        source_readout_uncertainty=dict(actual_source_RGB_measured=True,
            paint_width_used_as_center_tolerance=False,source_native_pixel_quantization_step_px=1.,
            source_interpolation_grid_px=1./32.,actual_readout_blur_interval_px=[[-.3,.4]]*3))
    navigation=deepcopy(row);navigation['normal']=[[2**-.5,2**-.5]]*3
    _measured_board_readout_support(navigation)
    assert 'sigma_px' not in navigation
    missing=deepcopy(row);missing['source_readout_uncertainty'].pop('source_native_pixel_quantization_step_px')
    _measured_board_readout_support(missing)
    assert 'sigma_px' not in missing
    _measured_board_readout_support(row)
    assert np.allclose(row['sigma_px'],.4+.5+1./64.)
    assert row['source_readout_uncertainty']['is_calibrated_uncertainty'] is False


def make_pack(tmp_path, *, held=False):
    binding,row,measurement,inventory=fixture_inputs(tmp_path)
    normalized=normalize_selected_ice_rows([row],measurement,source_binding=binding,curve_inventory=inventory)
    review=normalized['review']
    writer=StrictAbsolutePackWriter(tmp_path/'strict_source.npz',source=review.source,
        expected_actual_pts=[100,200],legacy_pixel_families=['native_ice_factor','wide_source','ad_factor'])
    writer.add_row(normalized['rows'][0],review)
    if held:
        second=deepcopy(normalized['rows'][0]);second['split']='held';second['family']='held_only'
        writer.add_row(second,review)
    writer.mark_exposure_processed(100)
    receipt=writer.finish()
    return StrictAbsoluteSourcePack(receipt['path'],expected_sha256=receipt['sha256']),normalized


def test_source_midpoint_is_recomputed_from_packed_actual_sides(tmp_path):
    binding,row,measurement,inventory=fixture_inputs(tmp_path)
    normalized=normalize_selected_ice_rows([row],measurement,source_binding=binding,curve_inventory=inventory)
    center=normalized['rows'][0]['uv']
    assert not np.array_equal(center,measurement['candidate_uv'].astype(float))
    assert np.array_equal(center,measurement['candidate_sides'].astype(float).mean(1))
    assert np.array_equal(normalized['rows'][0]['raw_candidate_uv'],measurement['candidate_uv'])
    writer=StrictAbsolutePackWriter(tmp_path/'pack.npz',source=normalized['review'].source,expected_actual_pts=[100])
    assert writer.add_row(normalized['rows'][0],normalized['review'])==1
    writer.mark_exposure_processed(100)
    receipt=writer.finish()
    pack=StrictAbsoluteSourcePack(receipt['path'],expected_sha256=receipt['sha256'])
    assert np.array_equal(pack.fit_rows[0]['uv'],center)
    assert len(pack.proof_templates)==1
    assert pack.metadata['source_measurement_processing_is_not_manual_visual_completion'] is True


def test_layer_boundary_writer_retains_the_same_measured_q50_as_final_source_gate(tmp_path):
    binding,row,measurement,inventory=fixture_inputs(tmp_path)
    row.update(curve_id='R2:outer',physical_material_id='R2.outer',
        admission_status='admitted_source_physical_boundary_temporal_identity')
    row['temporal_evidence']['physical_material_id']='R2.outer'
    row['source_readout_uncertainty']=dict(kind='actual_source_readout_blur_and_quantization',
        actual_source_RGB_measured=True,paint_width_used_as_center_tolerance=False,
        actual_readout_blur_interval_px=[[-.2,.3]],normal_interval_px=[[-.7,.8]])
    row['sigma_px']=[.8]
    inventory=[dict(curve_id='R2:outer',feature_id='R2',world_curve_role='outer',operator='material_interface')]
    measurement['candidate_uv']=np.array([[100.,30.]],np.float32)
    measurement['candidate_sides']=np.repeat(measurement['candidate_uv'][:,None],2,axis=1)
    quantiles=np.repeat(measurement['candidate_sides'][:,:,None].astype(float),3,axis=2)
    quantiles[:,:,0,1]-=.2
    quantiles[:,:,1,1]+=.031
    quantiles[:,:,2,1]+=.3
    measurement['candidate_side_quantiles_uv']=quantiles
    # One isolated measured boundary has no neighbouring-pixel normal. That
    # missing uncertainty must not reset the exact observed boundary pixel.
    normalized=normalize_selected_ice_rows([row],measurement,source_binding=binding,curve_inventory=inventory)
    actual=normalized['rows'][0]
    assert np.array_equal(actual['uv'],quantiles[:,:,1].mean(1))
    assert np.array_equal(actual['raw_candidate_uv'],measurement['candidate_uv'])
    assert 'source_readout_uncertainty' not in actual and 'sigma_px' not in actual
    writer=StrictAbsolutePackWriter(tmp_path/'boundary.npz',source=normalized['review'].source,expected_actual_pts=[100])
    assert writer.add_row(actual,normalized['review'])==1
    writer.mark_exposure_processed(100)
    receipt=writer.finish()
    pack=StrictAbsoluteSourcePack(receipt['path'],expected_sha256=receipt['sha256'])
    assert np.array_equal(pack.values['uv'],quantiles[:,:,1].mean(1))
    assert np.array_equal(pack.values['source_sides_uv'],quantiles[:,:,1])
    assert np.isnan(pack.values['sigma_px']).all()
    assert pack.metadata['source_uncertainty_ready_for_fit'] is False


def test_every_legacy_family_unknown_and_held_physically_excluded(tmp_path):
    pack,normalized=make_pack(tmp_path,held=True)
    row=normalized['rows'][0]
    uv=np.vstack([row['uv'],[400.,500.],[401.,501.]])
    original=uv.copy()
    source=dict(uv=uv,sigma_px=np.array([2.,3.,4.]),split='fit')
    active,retired=pack.compact_legacy_absolute_inputs([
        dict(family='source_remeasured_ice',row=source,actual_pts=100,model_target=row['model_target']),
        dict(family='wide_source',row=source,actual_pts=100,model_target=None),
        dict(family='unlisted_unknown_family',row=source,actual_pts=100,model_target=None),
        dict(family='held_only',row=source,actual_pts=100,model_target=row['model_target']),
        dict(family='source_remeasured_ice',row=dict(source,split='held'),actual_pts=100,model_target=row['model_target']),
    ])
    assert len(active)==1
    assert active[0]['row']['uv'].shape==(1,2)
    assert active[0]['row']['sigma_px'].tolist()==[2.]
    assert active[0]['row']['strict_source_original_point_indices']==[0]
    assert [x['original_point_indices'].tolist() for x in retired]==[[1,2],[0,1,2],[0,1,2],[0,1,2],[0,1,2]]
    assert np.array_equal(source['uv'],original)
    assert len(pack.fit_rows)==1 and len(pack.held_rows)==1
    assert pack.metadata['all_exposures_processed'] is False


def test_pending_reflection_and_missing_continuity_are_not_training_or_fit_truth(tmp_path):
    binding,row,measurement,inventory=fixture_inputs(tmp_path)
    candidates=[]
    for changes in (dict(admission_status='pending_exact_source_review'),
            dict(static_material_confirmed=False),dict(temporal_evidence={})):
        candidate=deepcopy(row);candidate.update(changes);candidates.append(candidate)
    output=normalize_selected_ice_rows(candidates,measurement,source_binding=binding,curve_inventory=inventory)
    assert output['rows']==[] and len(output['rejected_rows'])==3
    assert output['review'].records=={}


def test_wrong_source_candidate_and_ordinary_wide_edge_never_enter(tmp_path):
    binding,row,measurement,inventory=fixture_inputs(tmp_path)
    bad=deepcopy(measurement);bad['candidate_flags'][0]=10
    with pytest.raises(ValueError,match='excluded source candidate'):
        normalize_selected_ice_rows([row],bad,source_binding=binding,curve_inventory=inventory)
    normalized=normalize_selected_ice_rows([row],measurement,source_binding=binding,curve_inventory=inventory)
    edge=deepcopy(normalized['rows'][0]);edge['observation_semantics']='paint_layer_boundary'
    writer=StrictAbsolutePackWriter(tmp_path/'pack.npz',source=normalized['review'].source,expected_actual_pts=[100])
    assert writer.add_row(edge,normalized['review'])==0
    assert writer.exclusions[0]['reason']=='ordinary_wide_paint_requires_centerline'


def test_known_physical_board_boundary_keeps_exact_source_pixels(tmp_path):
    binding,row,measurement,inventory=fixture_inputs(tmp_path)
    row.update(scope='board',feature_id='board.upper',physical_material_id='far_board.upper',
        model_target=dict(kind='current_scene_board_edge',edge='upper'),observation_semantics='board_physical_boundary',
        admission_status='admitted_source_physical_boundary_temporal_identity',uv=np.array([[311.,410.]]))
    row['temporal_evidence']['physical_material_id']=row['physical_material_id']
    mb=dict(video_sha256=binding['source_sha256'],time_base=binding['time_base'],native_pts=110,
        actual_pts=100,decoded_BGR_sha256='b'*64)
    normalized=normalize_selected_board_rows([row],dict(source=mb),source_binding=binding)
    assert len(normalized['rows'])==1
    assert normalized['rows'][0]['uv'].tolist()==[[311.,410.]]
    mismatch=dict(mb,native_pts=111)
    with pytest.raises(ValueError,match='native source binding differ'):
        normalize_selected_board_rows([row],dict(source=mismatch),source_binding=binding)


def test_discrete_material_corner_keeps_role_and_2d_readout_without_contour_or_sigma(tmp_path):
    from calibration.program_tracking.strict_absolute_source import board_material_point_target_inventory
    binding,row,measurement,inventory=fixture_inputs(tmp_path)
    role='upper_left_printed_outline_corner'
    point=np.array([[311.,410.]])
    border=[dict(uv=[309.,410.+k],blur=[-.4,.5]) for k in range(4)]
    readout=dict(uv=point[0],actual_2d_readout_bounds_xy_px=[[310.,409.],[312.,411.]],
        actual_source_RGB_measured=True,source_guides_used_as_output=False,
        fitted_scene_or_camera_used_for_readout=False,
        measurement_method='actual_original_RGB_two_visible_printed_border_intersection',
        visible_border_readings=[border,border])
    row.update(physical_material_id=f'Ad12.material_point.{role}',advertisement_number=12,
        curve=dict(kind='shared_board_pattern_material_points',advertisement_id=12),
        admission_status='admitted_source_advertisement_temporal_identity',
        observation_semantics='board_print_feature',uv=point,normal=[],
        material_roles=[role],point_correspondence=True,anchor_point_indices=[0],
        source_neighbour_adjacency_confirmed=False,source_material_point_readout=readout,
        actual_2d_readout_bounds_xy_px=np.asarray(readout['actual_2d_readout_bounds_xy_px'])[None])
    row['temporal_evidence']['physical_material_id']=row['physical_material_id']
    targets=board_material_point_target_inventory(dict(shared_board_patterns=dict(
        shield_fixed_material_points_xyz_scene_units={'12':{role:[1.,2.,3.]}})))
    measured=dict(source=binding,advertisement_rows=[row])
    normalized=normalize_selected_board_rows([row],measured,source_binding=binding,
        board_material_point_targets=targets)
    selected=normalized['rows'][0]
    assert selected['material_roles']==[role]
    assert selected['model_target']==dict(row['curve'],material_roles=[role])
    assert np.isnan(selected['sigma_px']) and np.asarray(selected['normal']).size==0
    writer=StrictAbsolutePackWriter(tmp_path/'point.npz',source=normalized['review'].source,expected_actual_pts=[100])
    assert writer.add_row(selected,normalized['review'])==1
    writer.mark_exposure_processed(100);receipt=writer.finish()
    pack=StrictAbsoluteSourcePack(receipt['path'],expected_sha256=receipt['sha256'])
    assert pack.metadata['source_uncertainty_ready_for_fit'] is False
    with pytest.raises(ValueError,match='absent from this exact frozen model'):
        normalize_selected_board_rows([row],measured,source_binding=binding)
    joined=deepcopy(row);joined['normal']=[[0.,1.]]
    with pytest.raises(ValueError,match='own actual two-border readout'):
        normalize_selected_board_rows([joined],dict(source=binding,advertisement_rows=[joined]),
            source_binding=binding,board_material_point_targets=targets)


def test_omitted_old_family_receipt_prevents_partial_gate(tmp_path):
    pack,normalized=make_pack(tmp_path)
    assert pack.require_complete_legacy_family_gate([dict(family='native_ice_factor',pixel_input_count=1)])
    with pytest.raises(ValueError,match='omitted an old absolute pixel family'):
        pack.require_complete_legacy_family_gate([dict(family='shared_board_patterns',pixel_input_count=1)])


def test_loader_revalidates_center_and_does_not_trust_flags(tmp_path):
    pack,normalized=make_pack(tmp_path)
    values={key:value.copy() for key,value in pack.values.items()}
    values['uv'][0]=values['source_sides_uv'][0,0]
    values.update(metadata_json=json.dumps(pack.metadata),rows_json=json.dumps(pack.rows),
        exclusions_json=json.dumps(pack.exclusions),source_proof_templates_json=json.dumps(pack.proof_templates))
    tampered=tmp_path/'incorrect_center.npz';np.savez(tampered,**values)
    with pytest.raises(ValueError,match='centerline differs'):
        StrictAbsoluteSourcePack(tampered,expected_sha256=file_sha(tampered))


def test_public_runtime_consumer_replaces_all_old_absolute_arrays(tmp_path):
    pack,normalized=make_pack(tmp_path)
    source=pack.metadata['source']
    review_path=tmp_path/'pending_motion_review.json'
    review_path.write_text(json.dumps(dict(schema=REVIEW_SCHEMA,source=source,reviewed_endpoints=[])),encoding='utf8')
    review=ExactSourceScopeReview.load(review_path)
    original=tmp_path/'historical_motion.npz'
    np.savez(original,a=[[20.,30.]],b=[[21.,31.]])
    motion=write_motion_partition(tmp_path/'strict_motion.npz',block='main1',a=np.array([[20.,30.]]),
        b=np.array([[21.,31.]]),ia=np.array([0]),ib=np.array([1]),stamps=np.array([100,200]),
        sigma=np.ones((1,2)),active_to_original=np.array([41]),review=review,
        original_references=[dict(path=str(original),sha256=file_sha(original))])
    binding_path=tmp_path/'strict_motion_inputs.json'
    binding_path.write_text(json.dumps(dict(schema='rinkspace.strict-model-scope-original-motion-inputs.v1',
        unknown_or_unreviewed_default='excluded',source=source,blocks=[motion])),encoding='utf8')
    inventory=tmp_path/'original_all_family_inventory.npz'
    np.savez(inventory,metadata_json=json.dumps(dict(family_receipts=[
        dict(family='programme:main1:motion',pixel_input_count=2),
        dict(family='native_ice_factor',pixel_input_count=3),
        dict(family='wide_source',outside_programme_pixel_count=3),
        dict(family='ad_factor',pixel_input_count=3)])))
    consumer=StrictProgrammeSourceInputs(
        motion_inputs=dict(path=str(binding_path),sha256=file_sha(binding_path)),
        absolute_inputs=dict(path=str(tmp_path/'strict_source.npz'),sha256=file_sha(tmp_path/'strict_source.npz')),
        current_pixel_inventory=dict(path=str(inventory),sha256=file_sha(inventory)))
    entries=[dict(family=family,row=dict(uv=np.array([[401.,510.],[400.,520.],[410.,530.]])),
        actual_pts=100,model_target=None,split='fit') for family in ('native_ice_factor','wide_source','ad_factor')]
    result=consumer.consume_absolute_inputs(entries)
    assert len(result['fit_rows'])==1 and result['fit_rows'][0]['uv'].shape==(1,2)
    assert result['unchecked_legacy_rows_returned'] is False
    assert len(result['retired_legacy_points'])==3
    assert result['validated_legacy_provenance']==[]
    assert consumer.motion_blocks['main1']['a'].shape==(0,2)
