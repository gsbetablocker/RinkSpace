"""Whole-group derived means certify exactly the same final native support as W."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from calibration.program_tracking.first_shot_source_evidence import (
    _influence, rebuild_final_native_line_estimator,
)
from calibration.program_tracking.first_shot_revision import (
    SOURCE_SCHEMA, build_correlated_estimator_replacement, revise_frames,
)
from calibration.program_tracking.source_evidence_whitening import latent_whitening_operator
from calibration.program_tracking.first_shot_fit import FrozenWhitening
from calibration.program_tracking.finite_source_line_coverage import compile_native_line_sections


def save(path, value):
    path.write_text(json.dumps(value), encoding='utf8')
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.fixture
def source(tmp_path):
    pts=256256;fid='ice.lane1:edge0';target=dict(kind='ice_curve',curve_id='lane1:edge0')
    origin=np.array([50.,100.]);tangent=np.array([1.,0.]);normal=np.array([0.,1.])
    stations=np.array([0.,20.,60.,100.]);offsets=.08*stations+3.+[.1,-.2,.2,-.1]
    direct=[dict(uv=(origin+[t,d]).tolist(),sigma_px=s,
        source_evidence=dict(query_uv=(origin+[t,0.]).tolist(),profile_normal=normal.tolist(),
            query_index=i,measurement_origin='current_native_source_identified_thin_paint_two_local_luminance_flanks',
            actual_source_support_outside_person_and_broadcast_masks=True,
            signed_interface_iqr_contrast=[2.+i,5.-.2*i]))
        for i,(t,d,s)in enumerate(zip(stations,offsets,[.5,1.,.6,.8],strict=True))]
    line=dict(point_origin=origin.tolist(),unit_tangent=tangent.tolist(),unit_normal=normal.tolist(),
        normal_offset_coefficients=[0.,0.],current_frame_only=True,direct_readouts=direct)
    clock=dict(native_pts=pts,actual_pts=pts,time_base='1/12800',source_video_sha256='video',decoded_BGR_sha256='RGB')
    native=save(tmp_path/'native.json',dict(source=clock,features=[dict(id=fid,source_line_fit=line)]))
    packet=save(tmp_path/'packet.json',dict(source=clock,ice_source_readout_packet=native))
    def row(uv,ids,vertices):
        return dict(target=target.copy(),_target_key='line',uv=np.asarray(uv,float),sigma=np.arange(len(ids))+.8,
            source_point_ids=ids,original_vertex_indices=vertices,point_correspondence=False,
            source_feature_id=fid,source_segment_index=0,native_pts=pts,actual_pts=pts,time_base='1/12800',
            source_video_sha256='video',source_packet=packet,_continuous_world_line=True,
            _line_end_extent_geometry_columns=[97])
    rows=[row([[60.,100.],[90.,100.]],['p0','p1'],[0,1]),
        dict(target=dict(kind='ad_corner',advertisement_id=4,role='left_top'),_target_key='corner',
             uv=np.array([[150.,160.]]),sigma=np.array([1.3]),source_point_ids=['ordinary'],
             original_vertex_indices=[0],point_correspondence=True,source_feature_id='Ad4.left_top',source_segment_index=0),
        row([[140.,100.]],['p2'],[2])]
    ids=('p0','p1','p2');rep_station=np.array([10.,40.,90.]);design=np.c_[stations,np.ones(4)]
    basis=np.zeros((3,6));basis[:,1::2]=np.eye(3)
    noise=_influence(design,np.ones(4))*np.array([r['sigma_px']for r in direct])
    old=latent_whitening_operator(ids,np.c_[rep_station,np.ones(3)],explicit_noise_factor=noise,
        physical_basis=basis,group_id=f'{pts}/{fid}/source_line',covariance_origin='original native line convention')
    group=dict(point_indices=[0,1,3],operator=old,source_evidence=dict(source_readout_packet=native))
    frame=dict(index=1,native_pts=pts,rows=rows,source_point_count=4,source_packet=packet,whitening_groups=[group])
    binding=dict(source=dict(sha256='video',width=1920,height=1080),source_clock=dict(time_base='1/12800'),
        complete_fit_input=dict(path='dataset'),model=dict(path='model'))
    revision=dict(schema=SOURCE_SCHEMA,source_video_sha256='video',source_time_base='1/12800',
        base_binding=save(tmp_path/'base.json',binding),original_source_UV_IDs_and_approvals_modified=False)
    item=build_correlated_estimator_replacement(frame,group,binding,
        reason='Recorded final native support has a different linear estimator mean',evidence_references=[native])
    revision['correlated_estimator_replacements']=[item]
    def run():return revise_frames([frame],save(tmp_path/'revision.json',revision),binding)
    return SimpleNamespace(frame=frame,group=group,binding=binding,revision=revision,item=item,run=run,
        line=line,native=native,packet=packet,tmp_path=tmp_path)


def test_whole_group_is_atomic_and_original_facts_and_formal_rank_are_preserved(source):
    original=np.concatenate([r['uv']for r in source.frame['rows']]).copy()
    sigma=np.concatenate([r['sigma']for r in source.frame['rows']]).copy();W=source.group['operator'].matrix.copy()
    frames,_,proof=source.run();frame=frames[0];uv=np.concatenate([r['uv']for r in frame['rows']])
    assert frame['source_point_count']==4 and proof['added_native_readouts']==0
    assert proof['correlated_estimator_replacements']==1 and proof['derived_correlated_representatives']==3
    assert proof['new_independent_residual_rows_from_estimator_replacements']==0
    assert proof['original_source_files_modified']is False
    assert FrozenWhitening(frame).count==FrozenWhitening(source.frame).count==4
    assert [r['source_point_ids'][0]for r in frame['rows']]==['p0','p1','ordinary','p2']
    assert not np.array_equal(uv[[0,1,3]],original[[0,1,3]])
    np.testing.assert_array_equal(uv[2],original[2]);np.testing.assert_array_equal(np.concatenate([r['sigma']for r in frame['rows']]),sigma)
    for index in [0,1,3]:
        row=frame['rows'][index]
        np.testing.assert_array_equal(row['original_source_uv'][0],original[index])
        assert row['derived_correlated_source_representative'] and row['new_independent_native_readout']is False
        assert 'added_native_readout'not in row
    np.testing.assert_array_equal(np.concatenate([r['uv']for r in source.frame['rows']]),original)
    np.testing.assert_array_equal(source.group['operator'].matrix,W)
    assert not np.array_equal(frame['whitening_groups'][0]['operator'].matrix,W)


@pytest.mark.parametrize('field',['UV','ID','sigma','target','W','indices'])
def test_original_group_authentication_cannot_be_bypassed(source,field):
    item=source.item
    if field=='UV':item['original_members'][0]['original_UV'][1]+=.01
    elif field=='ID':item['original_members'][0]['source_point_id']='invented'
    elif field=='sigma':item['original_members'][0]['original_sigma_px']+=.01
    elif field=='target':item['original_members'][0]['original_target']['curve_id']='lane2:edge0'
    elif field=='W':item['original_operator']['whitening_matrix'][0][0]+=.01
    else:item['original_point_indices']=item['original_point_indices'][::-1]
    with pytest.raises(ValueError):source.run()


@pytest.mark.parametrize('field',['partial_mean','mean','D','normal','newW','independent'])
def test_partial_or_mismatched_estimator_outputs_are_rejected(source,field):
    item=source.item
    if field=='partial_mean':item['derived_representatives'].pop()
    elif field=='mean':item['derived_representatives'][0]['UV'][1]+=.01
    elif field=='D':item['estimator_contract']['D_sha256']='other support D'
    elif field=='normal':item['estimator_contract']['new_physical_normal'][0]+=.01
    elif field=='newW':item['new_operator']['whitening_matrix'][0][0]+=.01
    else:item['independent_native_readouts_added']=1
    with pytest.raises(ValueError):source.run()


def test_partial_quarantine_or_competing_operator_cannot_reuse_derived_group(source):
    m=source.item['original_members'][0]
    removed=dict(native_pts=source.frame['native_pts'],source_point_id=m['source_point_id'],
        original_UV=m['original_UV'],original_target=m['original_target'],authority='agent_rgb_review',
        reason='Explicit native identity quarantine',evidence_references=[source.native])
    source.revision['quarantined_correspondences']=[removed]
    with pytest.raises(ValueError,match='Partial identity/quarantine'):source.run()
    source.revision.pop('quarantined_correspondences')
    source.revision['correlated_operator_replacements']=[dict(source.item)]
    with pytest.raises(ValueError,match='Conflicting operator-only'):source.run()


def test_changed_final_support_cannot_keep_old_mean_and_W_certificate(source):
    native=json.loads(open(source.native['path'],encoding='utf8').read())
    native['features'][0]['source_line_fit']['direct_readouts'][0]['uv'][1]+=.2
    new_native=save(source.tmp_path/'changed_native.json',native)
    packet=json.loads(open(source.packet['path'],encoding='utf8').read());packet['ice_source_readout_packet']=new_native
    new_packet=save(source.tmp_path/'changed_packet.json',packet)
    source.frame['source_packet']=new_packet;source.item['source_packet']=new_packet
    source.item['native_readout_packet']=new_native;source.item['evidence_references']=[new_native]
    with pytest.raises(ValueError,match='same final native support/D'):source.run()


def test_coverage_keeps_original_native_queries_and_clock_with_exact_replacement(source):
    old=compile_native_line_sections([source.frame],extent_columns=[97])
    frames,_,_=source.run();new=compile_native_line_sections(frames,extent_columns=[97])
    assert len(old)==len(new)==1 and not new.unsupported
    before=old.sections[0];after=new.sections[0]
    assert before.native_pts==after.native_pts==source.frame['native_pts'] and before.query_ids==after.query_ids
    for key in ['query_uv','profile_normal','measured_uv']:
        assert getattr(before,key).tobytes()==getattr(after,key).tobytes()
    assert after.evidence['derived_correlated_estimator_replacement']
    assert after.evidence['actual_queries_are_existing_full_W_estimator_native_support']


@pytest.mark.parametrize('feature_id',['ice.lane1:edge0','ice.C:segment0'])
def test_exact_mean_native_noise_differential_and_actual_source_kernel(feature_id,source):
    from calibration.solver_runtime import install
    install()
    from calibration.program_tracking.first_shot_source_kernel import evaluate_projected_targets
    line=deepcopy(source.line);ids=('p0','p1','p2');uv=np.array([[60.,100.],[90.,100.],[140.,100.]])
    base=rebuild_final_native_line_estimator(line,feature_id,ids,uv,group_id='native/line')
    rows=line['direct_readouts'];p=np.array([r['uv']for r in rows]);sigma=np.array([r['sigma_px']for r in rows])
    origin=np.array(line['point_origin']);v=np.array(line['unit_tangent']);n=np.array(line['unit_normal'])
    design=np.c_[(p-origin)@v,np.ones(len(p))]
    weights=np.array([min(r['source_evidence']['signed_interface_iqr_contrast'])**2 for r in rows])if feature_id=='ice.C:segment0'else np.ones(len(p))
    # Independent least-squares oracle; never call the production influence here.
    coefficient=np.linalg.lstsq(np.sqrt(weights)[:,None]*design,np.sqrt(weights)*((p-origin)@n),rcond=None)[0]
    output_design=np.c_[(uv-origin)@v,np.ones(len(uv))]
    mean=origin+output_design[:,0,None]*v+(output_design@coefficient)[:,None]*n
    np.testing.assert_allclose(base['uv'],mean,rtol=0,atol=4e-14)
    assert base['metadata']['physical_sensitivity_normal_projection_factor']<1.
    direction=np.array([.3,-.8,.5,.2]);h=1e-5;means=[]
    for sign in [1.,-1.]:
        noisy=deepcopy(line)
        for r,point,s,z in zip(noisy['direct_readouts'],p,sigma,direction,strict=True):r['uv']=(point+sign*h*s*z*n).tolist()
        means.append(rebuild_final_native_line_estimator(noisy,feature_id,ids,uv,group_id='native/line')['uv'])
    raw_dmean=((means[0]-means[1])/(2*h)).ravel()
    expected=base['physical_sensitivity']@base['noise_factor']@direction
    np.testing.assert_allclose(base['physical_basis']@raw_dmean,expected,rtol=2e-8,atol=3e-9)
    camera=np.zeros(25);camera[6:8]=np.log(1000.);camera[8:10]=[960.,540.]
    station=np.array([-20.,120.]);screen=origin+station[:,None]*v+(np.c_[station,np.ones(2)]@coefficient)[:,None]*n
    world=np.c_[(screen-[960.,540.])/100.,np.full(2,10.)]
    residual=[]
    for m in means:
        row=dict(_target_key='line',uv=m,point_correspondence=False,_continuous_world_line=True)
        r=evaluate_projected_targets(camera,[row],{'line':world},jacobian=False)[0]
        residual.append(base['operator'].apply(r))
    np.testing.assert_allclose((residual[0]-residual[1])/(2*h),-base['operator'].matrix@raw_dmean,rtol=2e-7,atol=2e-8)
