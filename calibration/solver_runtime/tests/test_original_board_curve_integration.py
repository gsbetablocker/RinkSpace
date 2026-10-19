from pathlib import Path
from types import SimpleNamespace
import copy,json,pickle,sys
import numpy as np
import pytest
from scipy.sparse import csr_matrix
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from calibration.solver_runtime import install
install()
from calibration.solver_runtime.source_loss import SourceLossWeights,weighted_source_cost,source_irls_weights
from calibration.solver_runtime.original_board_curve_integration import (
    SCHEMA,compose_disjoint_loss_weights,dense_trace_anchor_masses,declared_working_sigma,
    wrap_original_board_curve_integration,file_reference,verified_original_observation_subset,
    reconcile_isolated_anchor_with_declared_trace,AuthenticatedCorrectedRGBAnchors,
    CORRECTED_RGB_SCHEMA,CORRECTED_RGB_BASIS,OriginalDetectorSupport)
from z190_streaming_source_normal import SourceRowBlock

def test_disjoint_existing_Huber_and_working_normalization_are_preserved():
    old_mask=np.array([True,True,False,False,False,False]);new_mask=np.array([False,False,True,True,False,False])
    old=SourceLossWeights([.2,.8,1.,1.,1.,1.],[2.,2.,1.,1.,1.,1.])
    new=SourceLossWeights([1.,1.,.1,.9,1.,1.],[1.,1.,3.,3.,1.,1.])
    joined=compose_disjoint_loss_weights(old,new,existing_slots=old_mask,additional_slots=new_mask,rows=6)
    r=np.array([.3,5.,.2,4.,-.5,7.]);quad=np.array([False,False,False,False,True,False])
    expected=(weighted_source_cost(r[:2],quad[:2],SourceLossWeights([.2,.8],[2.,2.]))
        +weighted_source_cost(r[2:4],quad[2:4],SourceLossWeights([.1,.9],[3.,3.]))
        +weighted_source_cost(r[4:],quad[4:],None))
    assert weighted_source_cost(r,quad,joined)==expected
    direction=np.array([1.,.2,-.4,.3,.1,-.1]);h=1e-6
    fd=(weighted_source_cost(r+h*direction,quad,joined)-weighted_source_cost(r-h*direction,quad,joined))/(2*h)
    assert abs(fd-np.dot(source_irls_weights(r,quad,joined)*r,direction))<1e-8
    with pytest.raises(ValueError,match='overlap'):
        compose_disjoint_loss_weights(old,new,existing_slots=old_mask,additional_slots=old_mask,rows=6)
    with pytest.raises(ValueError,match='outside'):
        compose_disjoint_loss_weights(old,new,existing_slots=np.zeros(6,bool),additional_slots=new_mask,rows=6)

def test_dense_trace_mass_preserves_source_gaps_and_original_anchors():
    trace=np.array([[0.,0.],[10.,1.],[20.,0.],[100.,0.],[110.,2.],[120.,0.]])
    policy=dict(method='declared_regular_source_x_grid',step_native_px=10.)
    selected=trace[[0,2,3,5]].copy();mass,proof=dense_trace_anchor_masses(trace,selected,gap_policy=policy)
    assert proof['visible_source_fragments']==2
    assert abs(mass.sum()-(2*np.sqrt(101)+2*np.sqrt(104)))<1e-12
    assert np.all(mass>0.) and np.array_equal(selected,trace[[0,2,3,5]])
    assert max(mass)<21. # The absent 80px source interval receives no mass.
    with pytest.raises(ValueError,match='exactly once'):
        dense_trace_anchor_masses(trace,np.array([[5.,.5]]),gap_policy=policy)

def fixture(tmp_path):
    rows=[]
    for name,n,sigma in [('far_top',3,4.),('far_contact',4,7.),('board_top',2,[2.,5.])]:
        rows.append(dict(id=name,pts=20,split='fit',origin='independent_source_material',uv=np.c_[np.arange(n)*10.,np.full(n,5.)],
            sigma=np.broadcast_to(sigma,(n,)).copy(),world=np.array([[0.,0.,0.],[1.,1.,0.]]),point_correspondence=False))
    rows[2]['uv'][:,0]+=50.
    near=dict(id='reviewed_lower_board_top',pts=20,split='fit',origin='confirmed_source',uv=np.array([[20.,30.],[20.,40.]]),
        sigma=np.full(2,4.*np.sqrt(2)),world=np.array([[0.,0.,1.],[1.,0.,1.]]),
        geometry=dict(side='lower_long_side',corner_transition_excluded=True,source_scene_vertex_indices=[336,360]))
    rows.append(near)
    rows.append(dict(id='ad1:left_top',pts=20,split='fit',origin='actual_material_corner',uv=np.array([[12.,13.]]),
        sigma=np.array([3.]),world=np.array([[0.,0.,1.]]),point_correspondence=True,physical_ids=['ad1:left_top']))
    groups=[(20,rows)];entries=[dict(block='main1',group_order=0,row_order=i,group_actual_pts=20,collection_role='FIT',source_row=copy.deepcopy(row)) for i,row in enumerate(rows)]
    packet=tmp_path/'original.pkl';packet.write_bytes(pickle.dumps(dict(programme_absolute_rows=entries)))
    policy=tmp_path/'policy.json';policy.write_text(json.dumps(dict(schema=SCHEMA,curve_budget=1.,enable_point_compression=False,
        source_video_sha256='0'*64,source_station_gap_policy=dict(method='inferred_median_station_spacing',gap_factor=1.75,source_gap_visibility_certified=False),
        original_packet=file_reference(packet))))
    sigma=np.r_[[1.,1.],np.concatenate([np.repeat(r['sigma'],2) for r in rows]),[1.]]
    count=len(sigma);r=np.linspace(-3.,6.,count);J=csr_matrix(np.arange(count*12,dtype=float).reshape(count,12)/100.)
    old_mask=np.zeros(count,bool);near_slots=np.arange(2+2*(3+4+2),2+2*(3+4+2+2));old_mask[near_slots]=True
    weights=np.ones(count);norm=np.ones(count);weights[near_slots]=np.repeat([.3,.7],2);norm[near_slots]=np.sqrt(2)
    old_weights=SourceLossWeights(weights,norm)
    block=SourceRowBlock(r,J,np.zeros(count,bool),sigma,dict(family='programme:main1',original_start=0,original_stop=count),old_weights)
    source=SimpleNamespace(count=12,scales=sigma,quadratic=block.quadratic,contract={},blocks=lambda q,jacobian=True:iter((block,)))
    actor=SimpleNamespace(name='main1',groups=groups,quadratic=block.quadratic,
        bundle=SimpleNamespace(motion_rows=2,source_residual_rows=count-1))
    state=SimpleNamespace(blocks=[actor],original_absolute_input_compaction_loss_weights={'programme:main1':old_weights})
    return source,state,policy,block,old_mask

def test_real_source_block_prefix_identity_corners_and_existing_budget(tmp_path):
    base,state,policy,old,mask=fixture(tmp_path)
    owner=wrap_original_board_curve_integration(base,state,policy);new=next(owner.blocks(np.zeros(12)))
    assert new.r is old.r and new.J is old.J and new.scales is old.scales and new.quadratic is old.quadratic
    assert owner.scales is base.scales and owner.count==base.count
    assert np.array_equal(new.loss_weights.integration_weights[mask],old.loss_weights.integration_weights[mask])
    assert np.array_equal(new.loss_weights.working_normalization_multiplier[mask],old.loss_weights.working_normalization_multiplier[mask])
    # Both original top fragments share one physical curve/exposure budget.
    top=np.r_[np.arange(2,8),np.arange(16,20)]
    assert np.isclose(new.loss_weights.integration_weights[top].sum()/2,1.)
    assert np.all(new.loss_weights.integration_weights[-3:]==1.) # Material corner and roll.
    with pytest.raises(ValueError,match='already applied'):owner.apply_block(new)
    assert wrap_original_board_curve_integration(base,state,None) is base

def test_only_explicit_sampling_sigma_removed():
    row=dict(curve_observation_kind='confirmed_lower_long_side_top_horizontal_support',uv=[[0,0],[1,1]],
        sigma_modelling_decision='Original EDGE_SIGMA_PX=4 engineering outside-support sigma times sqrt(unique exposure scan rows); not measured covariance')
    working,prior,_=declared_working_sigma(row,np.full(2,4*np.sqrt(2)))
    assert np.allclose(working,4.) and np.allclose(prior,np.sqrt(2))
    arbitrary=dict(row);arbitrary.pop('sigma_modelling_decision')
    with pytest.raises(ValueError,match='authority'):declared_working_sigma(arbitrary,np.full(2,4*np.sqrt(2)))
    working,prior,_=declared_working_sigma({},np.array([3.,4.,7.]))
    assert np.array_equal(working,[3.,4.,7.]) and np.array_equal(prior,[1.,1.,1.])

def test_original_combined_cap_count_policy_requires_its_source_lineage():
    row=dict(curve_observation_kind='confirmed_lower_long_side_top_horizontal_support',
        uv=[[0.,0.],[1.,1.],[2.,2.]],original_lower_source_uv=np.array([[1.,1.],[2.,2.]]),
        original_lower_source_sigma=np.full(2,4.*np.sqrt(2)),upper_source_qualification={'path':'original_upper_source','sha256':'0'*64},
        sigma_modelling_decision='one correlated physical cap edge per actual exposure; original engineering sigma4 times sqrt(combined upper+lower row count); original lower sigma separately retained')
    working,prior,basis=declared_working_sigma(row,np.full(3,4.*np.sqrt(3)))
    assert np.allclose(working,4.) and np.allclose(prior,np.sqrt(3)) and 'combined' in basis
    changed=copy.deepcopy(row);changed['original_lower_source_uv'][0,0]+=1.
    with pytest.raises(ValueError,match='lineage'):declared_working_sigma(changed,np.full(3,4.*np.sqrt(3)))
    changed=copy.deepcopy(row);changed.pop('upper_source_qualification')
    with pytest.raises(ValueError,match='authority'):declared_working_sigma(changed,np.full(3,4.*np.sqrt(3)))

def test_numerical_singleton_does_not_duplicate_existing_trace_support():
    first=dict(id='trace',source_video_sha256='a'*64,actual_pts=100,physical_curve_id='board_top',operator_id='finite_curve',
        uv=np.array([[10.,20.]]),working_sigma_px=np.array([4.]),point_integration_mass=np.array([84.]),
        source_support_basis='original_dense_source_detector_trace')
    second=copy.deepcopy(first);second.update(id='isolated',working_sigma_px=np.array([3.54]),
        point_integration_mass=np.array([1.]),source_support_basis='isolated_original_measured_anchors')
    original_sigma=second['working_sigma_px'].copy();proof=reconcile_isolated_anchor_with_declared_trace([first,second])
    assert first['point_integration_mass'][0]==second['point_integration_mass'][0]==84.
    assert len(proof)==1 and np.array_equal(second['working_sigma_px'],original_sigma)
    third=copy.deepcopy(first);third.update(id='conflicting_trace',point_integration_mass=np.array([83.]))
    with pytest.raises(ValueError,match='conflict'):reconcile_isolated_anchor_with_declared_trace([first,third])

def test_exact_active_point_lineage_including_legitimate_new_qualification(tmp_path):
    from calibration.solver_runtime.original_absolute_input_compaction import slice_observed_points
    base,state,policy,old,mask=fixture(tmp_path)
    original=copy.deepcopy(state.blocks[0].groups[0][1][0])
    selected=slice_observed_points(original,[0,2])
    keep,sigma=verified_original_observation_subset(selected,original)
    assert np.array_equal(keep,[0,2]) and np.array_equal(sigma,[4.,4.,4.])
    broken=copy.deepcopy(selected);broken['uv'][0,1]+=.5
    with pytest.raises(ValueError,match='lineage'):verified_original_observation_subset(broken,original)
    missing=copy.deepcopy(selected);missing.pop('active_original_absolute_input_compaction')
    with pytest.raises(ValueError,match='certified active subset'):verified_original_observation_subset(missing,original)

def test_fresh_family_replays_disjoint_curve_policy_in_original_wrapper_order(tmp_path,monkeypatch):
    import structured_programme_source_candidate.activated_small_source_blocks as refresh
    base,state,policy,old,_=fixture(tmp_path)
    owner=wrap_original_board_curve_integration(base,state,policy)
    class Programme:
        actors=None
        def iter_source_blocks(self,q,jacobian):return iter((old,))
    programme=Programme();programme.state=state;programme.scales=old.scales;programme.quadratic=old.quadratic
    class Bias:
        def __init__(self):
            self.cap={'iter_source_blocks':programme.iter_source_blocks};self.programme={'main1':state.blocks[0]}
            self.validated_programme=set();self.count=12
        def unpack(self,value):return value,None
        def blocks(self,value,jacobian):yield from self.cap['iter_source_blocks'](value,jacobian)
    bias=Bias()
    def reapply_existing(raw):
        return SourceRowBlock(raw.r,raw.J,raw.quadratic,raw.scales,raw.row_ledger,old.loss_weights)
    compaction=SimpleNamespace(absolute_input_compaction={},apply_block=reapply_existing)
    monkeypatch.setattr(refresh,'ExplicitBiasSource',Bias)
    monkeypatch.setattr(refresh,'CurrentProgrammeSource',Programme)
    monkeypatch.setattr(refresh,'SOURCE_SHA','0'*64)
    monkeypatch.setattr(refresh,'ordered_block_evaluations',lambda owners,value,jacobian,threads:iter(((owners[0],(old.r,old.J)),)))
    result=refresh._fresh_complete_programme_family([owner,compaction,bias,programme],np.zeros(12),old.row_ledger)
    ordinary=next(owner.blocks(np.zeros(12)))
    assert result.r is old.r and result.J is old.J
    assert np.array_equal(result.loss_weights.integration_weights,ordinary.loss_weights.integration_weights)
    assert np.array_equal(result.loss_weights.working_normalization_multiplier,ordinary.loss_weights.working_normalization_multiplier)
    assert 'original_board_curve_integration' in result.row_ledger
    second=wrap_original_board_curve_integration(base,state,policy)
    assert owner.contract==second.contract
    assert 'preparation_seconds' not in owner.contract['original_board_curve_integration']


def corrected_rgb_fixture(tmp_path):
    import cv2,hashlib
    from structured_programme_source_candidate import original_ice_pixel_correction as pixels
    base,state,policy,block,mask=fixture(tmp_path)
    policy_data=json.loads(policy.read_text());policy_data['source_video_sha256']=pixels.SOURCE_SHA
    policy.write_text(json.dumps(policy_data),encoding='utf8')
    parent=policy_data['original_packet'];entries=pickle.loads(Path(parent['path']).read_bytes())['programme_absolute_rows']
    old=entries[0]['source_row'];new_uv=old['uv'].copy();new_uv[:,1]-=2.
    raw=tmp_path/'actual_native.png';image=np.zeros((1080,1920,3),np.uint8);image[:3,:,0]=127
    assert cv2.imwrite(str(raw),image)
    raw_ref=dict(file_reference(raw),pts=20,native_bgr24_sha256=hashlib.sha256(image.tobytes()).hexdigest())
    locator={k:entries[0][k] for k in ('block','collection_role','group_actual_pts','group_order','row_order')}
    from calibration.solver_runtime.source_point_qualification import original_uv_sha256
    measurement=dict(original_packet_row_index=0,locator=locator,original_full_uv_float64_sha256=original_uv_sha256(old),
        full_corrected_uv=new_uv.tolist(),source_support_basis=CORRECTED_RGB_BASIS,
        source_visible_paths=[[0],[1],[2]],source_raw_frame=raw_ref,
        raw_source_pixels_read=True,observed_UV_generated_from_model=False)
    receipt=tmp_path/'independent_RGB_receipt.json';receipt.write_text(json.dumps(dict(schema=CORRECTED_RGB_SCHEMA,
        source_video_sha256=pixels.SOURCE_SHA,time_base='1/15360',original_absolute_packet=parent,records=[measurement])),encoding='utf8')
    record=dict(original_packet_row_index=0,point_indices=[0,1,2],corrected_uv=new_uv.tolist(),
        corrected_normal_interval_px=None,correction_reason='synthetic wrong-edge binding test; no real vision accuracy claim',
        physical_edge_role='vertical_board_blue_upper_boundary',measurement_role='board_top_boundary',
        physical_identity_changed=False,original_geometry_identity='original_named_board_top',original_course_z_policy='top',
        original_pts=20,original_id=old['id'],original_split=old['split'],original_collection_role='FIT',
        original_actual_pts=20,original_block='main1',original_group_order=0,original_row_order=0,
        original_uv=old['uv'].tolist(),original_sigma=old['sigma'].tolist(),original_full_uv_float64_sha256=original_uv_sha256(old),
        source_only_reading=True,source_boundary_measurement=dict(file_reference(receipt),record_index=0))
    correction=tmp_path/'pixel_correction.json';correction.write_text(json.dumps(dict(schema=pixels.PACKET_SCHEMA,
        source_video_sha256=pixels.SOURCE_SHA,time_base='1/15360',model_projection_used_for_selection=False,
        original_absolute_packet=parent,source_evidence=[dict(file_reference(raw),pts=20)],records=[record])),encoding='utf8')
    material=SimpleNamespace(old_world=old['world'].copy(),contract=dict(z_policy='top'))
    state.geometry=SimpleNamespace(_binding=lambda row:('original',dict(kind='course',material=material,
        contract=dict(physical_identity='original_named_board_top'))))
    state.original=SimpleNamespace(aliases={20:20})
    pixels.install_original_ice_pixel_correction(state,correction)
    return base,state,policy,block,mask,correction,receipt,entries,raw


def reinstall_corrected_receipt(state,correction,receipt,entries):
    """Use the actual installer again after an explicitly changed test receipt."""
    from structured_programme_source_candidate.original_ice_pixel_correction import install_original_ice_pixel_correction
    packet=json.loads(correction.read_text());packet['records'][0]['source_boundary_measurement']=dict(file_reference(receipt),record_index=0)
    correction.write_text(json.dumps(packet),encoding='utf8')
    state.blocks[0].groups=[(20,[copy.deepcopy(e['source_row']) for e in entries])]
    install_original_ice_pixel_correction(state,correction)


def test_installed_corrected_rgb_singletons_keep_original_sigma_and_source_block(tmp_path,monkeypatch):
    base,state,policy,old,mask,correction,receipt,entries,_=corrected_rgb_fixture(tmp_path)
    original_masses=OriginalDetectorSupport.masses
    def forbid_bad_dense_reuse(self,original,*args,**kwargs):
        if original['id']=='far_top':raise AssertionError('An old wrong-edge trace cannot describe corrected RGB support')
        return original_masses(self,original,*args,**kwargs)
    monkeypatch.setattr(OriginalDetectorSupport,'masses',forbid_bad_dense_reuse)
    owner=wrap_original_board_curve_integration(base,state,policy);new=next(owner.blocks(np.zeros(12)))
    assert new.r is old.r and new.J is old.J and new.scales is old.scales
    assert np.array_equal(new.loss_weights.integration_weights[mask],old.loss_weights.integration_weights[mask])
    top=np.r_[np.arange(2,8),np.arange(16,20)]
    np.testing.assert_allclose(new.loss_weights.integration_weights[top],.2) # Three corrected + two unchanged anchors share one budget.
    proof=owner.original_board_curve_integration
    row=next(r for r in proof['source_rows'] if r['id']=='far_top')
    assert row['support_basis']==CORRECTED_RGB_BASIS and row['source_support_mass']==3.
    assert not row['corrected_RGB_source_proof']['dense_original_detector_support_reused']
    assert not row['corrected_RGB_source_proof']['continuous_support_claim']
    assert proof['authenticated_source_correction_packet_reference']==file_reference(correction)
    assert proof['authenticated_corrected_RGB_measurement_references']==[file_reference(receipt)]
    np.testing.assert_array_equal(state.blocks[0].groups[0][1][0]['sigma'],entries[0]['source_row']['sigma'])


def test_corrected_full_row_and_actual_compaction_parent_hash_are_both_authenticated(tmp_path):
    from calibration.solver_runtime.original_absolute_input_compaction import slice_observed_points
    base,state,policy,_,_,_,_,entries,_=corrected_rgb_fixture(tmp_path)
    corrected=state.blocks[0].groups[0][1][0];active=slice_observed_points(corrected,[0,2])
    data=json.loads(policy.read_text());verifier=AuthenticatedCorrectedRGBAnchors(state,data['original_packet'],data['source_video_sha256'])
    keep,stored,mass,proof=verifier.row(active,entries[0],0)
    np.testing.assert_array_equal(keep,[0,2]);np.testing.assert_array_equal(mass,[1.,1.])
    np.testing.assert_array_equal(stored,entries[0]['source_row']['sigma'])
    with pytest.raises(ValueError,match='lineage'):
        verified_original_observation_subset(active,entries[0]['source_row'])
    bad=copy.deepcopy(active);bad['active_original_absolute_input_compaction']['original_uv_float64_sha256']='0'*64
    with pytest.raises(ValueError,match='lineage'):verifier.row(bad,entries[0],0)
    bad=copy.deepcopy(active);bad['normal']=np.ones((2,2))
    with pytest.raises(ValueError,match='normal changed'):verifier.row(bad,entries[0],0)


def test_row_correction_metadata_without_the_installed_contract_cannot_admit_changed_uv(tmp_path):
    base,state,policy,_,_,_,_,_,_=corrected_rgb_fixture(tmp_path)
    del state.original_ice_pixel_correction
    with pytest.raises(ValueError,match='installed immutable source correction contract'):
        wrap_original_board_curve_integration(base,state,policy)


@pytest.mark.parametrize('defect', ['dense_path','wrong_locator','old_UV','model_UV','wrong_BGR','inactive_unknown'])
def test_corrected_rgb_support_provenance_and_visibility_are_fail_closed(tmp_path,defect):
    base,state,policy,_,_,correction,receipt,entries,_=corrected_rgb_fixture(tmp_path)
    data=json.loads(receipt.read_text());row=data['records'][0]
    if defect=='dense_path':row['source_visible_paths']=[[0,1,2]]
    elif defect=='wrong_locator':row['locator']['row_order']=1
    elif defect=='old_UV':row['full_corrected_uv']=entries[0]['source_row']['uv'].tolist()
    elif defect=='model_UV':row['observed_UV_generated_from_model']=True
    elif defect=='wrong_BGR':row['source_raw_frame']['native_bgr24_sha256']='0'*64
    else:row.update(measured_mask=[True,False,True],source_visible_paths=[[0],[2]])
    receipt.write_text(json.dumps(data),encoding='utf8');reinstall_corrected_receipt(state,correction,receipt,entries)
    with pytest.raises(ValueError):wrap_original_board_curve_integration(base,state,policy)


def test_unknown_original_anchor_adds_no_support_after_actual_input_compaction(tmp_path):
    from calibration.solver_runtime.original_absolute_input_compaction import slice_observed_points
    _,state,policy,_,_,correction,receipt,entries,_=corrected_rgb_fixture(tmp_path)
    data=json.loads(receipt.read_text());data['records'][0].update(measured_mask=[True,False,True],source_visible_paths=[[0],[2]])
    receipt.write_text(json.dumps(data),encoding='utf8');reinstall_corrected_receipt(state,correction,receipt,entries)
    active=slice_observed_points(state.blocks[0].groups[0][1][0],[0,2]);p=json.loads(policy.read_text())
    verifier=AuthenticatedCorrectedRGBAnchors(state,p['original_packet'],p['source_video_sha256'])
    _,_,mass,proof=verifier.row(active,entries[0],0)
    np.testing.assert_array_equal(mass,[1.,1.]);assert proof['excluded_anchor_cells_not_redistributed']


def test_corrected_singletons_share_duplicate_trace_budget_without_changing_their_measurement_role():
    first=dict(id='trace',source_video_sha256='a'*64,actual_pts=100,physical_curve_id='board_top',operator_id='finite_curve',
        uv=np.array([[10.,20.]]),working_sigma_px=np.array([4.]),point_integration_mass=np.array([84.]),
        source_support_basis='original_dense_source_detector_trace')
    corrected=copy.deepcopy(first);corrected.update(id='actual_RGB',point_integration_mass=np.array([1.]),
        source_support_basis=CORRECTED_RGB_BASIS)
    proof=reconcile_isolated_anchor_with_declared_trace([first,corrected])
    assert len(proof)==1 and corrected['point_integration_mass'][0]==84.
    assert corrected['source_support_basis']==CORRECTED_RGB_BASIS
