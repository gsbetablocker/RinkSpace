"""Native replacements are atomic, identity-gated and preserve covariance coordinates."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from calibration.program_tracking.first_shot_native_line_revision import (
    SCHEMA, revise_native_lines, certify_native_group,
)
from calibration.program_tracking.first_shot_noise_revision import source_members_digest
from calibration.program_tracking.official_static_surface_dataset import reference
from calibration.program_tracking.source_evidence_whitening import latent_whitening_operator
from calibration.program_tracking.first_shot_fit import FrozenWhitening
from calibration.program_tracking.finite_source_line_coverage import _compile_one, FiniteLineCoverageError


@pytest.fixture
def native_source(tmp_path):
    def save(name,value):
        p=tmp_path/name;p.write_text(json.dumps(value),encoding='utf8');return reference(p)
    pts=300000;fid='ice.lane1:edge0';gid=f'{pts}/{fid}/source_line'
    target=dict(kind='ice_curve',curve_id='lane1:edge0',residual='point_to_named_finite_ice_curve')
    chart=np.array([.6,.8]);uv=np.array([[10.,20.],[20.,20.],[30.,20.],[40.,20.]])
    direct=[]
    for i,p in enumerate(uv):
        ev=dict(query_uv=(p-[0.,1.]).tolist(),profile_normal=[0.,1.],query_index=i,
            actual_source_sides_uv=[(p-[0.,2.]).tolist(),(p+[0.,2.]).tolist()],
            actual_source_support_outside_person_and_broadcast_masks=True,
            measurement_origin='current_native_signed_RGB_material_profiles')
        direct.append(dict(uv=p.tolist(),sigma_px=2.,source_evidence=ev))
    line=dict(current_frame_only=True,direct_readouts=direct,unit_normal=chart.tolist(),
        point_origin=[0.,0.],unit_tangent=[-.8,.6],normal_offset_coefficients=[0.,0.])
    feature=dict(id=fid,source_line_fit=line,measurement_operator='paint_center')
    source=dict(native_pts=pts,actual_pts=pts,time_base='1/12800',source_video_sha256='video',decoded_BGR_sha256='pixels')
    native=save('native.json',dict(source=source,features=[feature]))
    packet=save('packet.json',dict(source=source,ice_source_readout_packet=native))
    row=dict(uv=uv[[0,-1]].copy(),sigma=np.ones(2),source_point_ids=['rep0','rep1'],
        source_feature_id=fid,original_vertex_indices=[0,1],target=target,point_correspondence=False,
        _target_key='ice1',_continuous_world_line=True,_line_end_extent_geometry_columns=(94,))
    board=dict(uv=np.array([[100.,200.]]),sigma=np.array([3.]),source_point_ids=['unchanged-board'],
        source_feature_id='board',original_vertex_indices=[0],target=dict(kind='board_edge'),point_correspondence=False)
    basis=np.zeros((2,4));basis[0,:2]=chart;basis[1,2:]=chart
    old=latent_whitening_operator(row['source_point_ids'],np.eye(2),explicit_noise_factor=np.eye(2),physical_basis=basis,group_id=gid,covariance_origin='fixture')
    L=np.c_[np.eye(4)*.7,np.array([[.2,0.],[.1,.2],[0.,.1],[.2,.2]])]
    factor_path=tmp_path/'factor.npz';np.savez(factor_path,localization_noise_factor=L)
    group=dict(operator=old,point_indices=[0,1],localization_noise_revision=dict(localization_noise_factor=L.tolist()))
    frame=dict(index=2,native_pts=pts,rows=[row,board],source_point_count=3,source_packet=packet,whitening_groups=[group])
    ids=[f'{pts}/{fid}/query/{i}'for i in range(4)]
    digest=hashlib.sha256(json.dumps(direct,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    evidence=[dict(query_id=ids[i],source_point_id=ids[i],original_native_uv=p.tolist(),
        original_query_uv=direct[i]['source_evidence']['query_uv'],profile_normal=[0.,1.],noise_coordinate_normal=chart.tolist())for i,p in enumerate(uv)]
    item=dict(frame=2,native_pts=pts,feature_id=fid,group_id=gid,old_operator=old.to_record(),old_point_indices=[0,1],
        current_noise_source_members_sha256=source_members_digest(frame,[0,1]),original_source_packet=packet,native_readout_packet=native,
        native_source_point_ids=ids,identity_scope_evidence_query_IDs=ids,native_count=4,native=evidence,
        old_native_support_canonical_sha256=digest,fixed_chart=dict(unit_normal=chart.tolist()),material_target=target,
        old_representatives=[dict(source_point_id=x)for x in row['source_point_ids']],
        native_material_identity_independently_RGB_reviewed_for_all_existing_query_IDs=True,
        native_noise_factor=dict(reference=reference(factor_path),array_key='localization_noise_factor',shape=list(L.shape),matrix_float64_bytes_sha256=hashlib.sha256(L.tobytes()).hexdigest()))
    review=dict(native_pts=pts,group_id=gid,native_support_sha256=digest,native_queries=[dict(query_id=ids[i],original_native_uv=p.tolist(),
        actual_source_sides_uv=direct[i]['source_evidence']['actual_source_sides_uv'],
        independent_RGB_material_identity_confirmed=True,actual_source_support_outside_person_and_broadcast_masks=True)for i,p in enumerate(uv)])
    binding=dict(source=dict(sha256='video'),source_clock=dict(time_base='1/12800'))
    base=save('binding.json',binding)
    def revision(item_override=None,review_override=None):
        identity=save('identity.json',dict(schema='rinkspace.independent-native-static-material-identity-scope.v1',base_binding=base,
            whole_original_group_replacement_only=True,world_camera_model_or_fit_residual_used_for_identity=False,
            boolean_motion_or_current_identity_flags_used_as_admission=False,groups=[review if review_override is None else review_override]))
        recipe=save('recipe.json',dict(base_binding=base,identity_evidence=identity,groups=[item if item_override is None else item_override]))
        return save('revision.json',dict(schema=SCHEMA,base_binding=base,recipe=recipe,identity_review=identity))
    return dict(frame=frame,item=item,review=review,revision=revision,binding=binding,L=L,feature=feature,native=native,row=row,chart=chart)


def test_atomic_native_queries_replace_summary_and_preserve_unaffected_rows(native_source):
    s=native_source;frame=s['frame'];old=FrozenWhitening(frame);other=dict(frame,index=3,native_pts=300256)
    result,proof=revise_native_lines([frame,other],s['revision'](),s['binding'])
    new=result[0];w=FrozenWhitening(new)
    assert result[1] is other and new['rows'][0] is frame['rows'][1]
    assert new['source_point_count']==5 and w.count==6 and old.count==4
    assert not {'rep0','rep1'} & {p for row in new['rows']for p in row['source_point_ids']}
    group=new['whitening_groups'][0];row=new['rows'][1]
    np.testing.assert_array_equal(row['uv'],[r['original_native_uv']for r in s['item']['native']])
    # L is in chart-normal coordinates, deliberately different from query normals.
    Q=np.zeros((4,8))
    for i in range(4):Q[i,2*i:2*i+2]=s['chart']
    A=group['operator'].matrix@Q.T@s['L']
    np.testing.assert_allclose(A@A.T,np.eye(4),atol=1e-12)
    assert proof['frames'][0]['formal_blocks'][0]['old_formal_local_slice']==[0,2]
    assert proof['frames'][0]['formal_blocks'][0]['new_formal_local_slice']==[0,4]
    assert proof['frames'][1]['old_global_formal_start']==4
    assert proof['frames'][1]['new_global_formal_start']==6
    assert proof['prediction_domain']['predicted_coordinates_or_parameters_clamped'] is False


@pytest.mark.parametrize('failure',['identity','UV','query','partial','L','oldW'])
def test_native_scope_cannot_silently_admit_or_reweight(native_source,failure):
    s=native_source;item=deepcopy(s['item']);review=deepcopy(s['review'])
    if failure=='identity':review['native_queries'][1]['independent_RGB_material_identity_confirmed']=False
    elif failure=='UV':item['native'][1]['original_native_uv'][0]+=.1
    elif failure=='query':item['native_source_point_ids'][0]='invented-query'
    elif failure=='partial':item['old_point_indices']=[0]
    elif failure=='L':s['frame']['whitening_groups'][0]['localization_noise_revision']['localization_noise_factor'][0][0]+=.1
    elif failure=='oldW':item['old_operator']['whitening_matrix'][0][0]+=.1
    with pytest.raises(ValueError):revise_native_lines([s['frame']],s['revision'](item,review),s['binding'])


def test_coverage_authenticates_native_L_and_never_restores_rank_two(native_source):
    s=native_source;frames,_=revise_native_lines([s['frame']],s['revision'](),s['binding']);frame=frames[0]
    row=frame['rows'][1];group=frame['whitening_groups'][0]
    section=_compile_one(frame,row,s['feature'],s['native'])
    assert len(section.query_ids)==4 and section.evidence['compressed_estimator_restored'] is False
    assert group['operator'].rank==4
    group['native_line_replacement']['localization_noise_factor'][0][0]+=.1
    with pytest.raises(FiniteLineCoverageError):_compile_one(frame,row,s['feature'],s['native'])


def test_real_lane_origin_is_admitted_only_with_authenticated_native_factor(native_source):
    s=native_source
    frames,_=revise_native_lines([s['frame']],s['revision'](),s['binding'])
    feature=deepcopy(s['feature'])
    for r in feature['source_line_fit']['direct_readouts']:
        r['source_evidence']['measurement_origin']='current_native_source_identified_lane_long_edge_two_luminance_flanks_physical_transverse_section'
    # Origin alone never upgrades a historical line to a native identity scope.
    with pytest.raises(FiniteLineCoverageError):_compile_one(s['frame'],s['row'],feature,s['native'])
    # The canonical support must still match: changing only its origin is rejected.
    with pytest.raises(FiniteLineCoverageError):_compile_one(frames[0],frames[0]['rows'][1],feature,s['native'])
