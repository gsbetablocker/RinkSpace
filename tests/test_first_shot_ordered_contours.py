import itertools
import numpy as np
import pytest
from calibration.program_tracking.first_shot_ordered_contours import ordered_nearest, source_tangents, _ordered_path


def test_narrow_hole_opposite_side_is_not_a_sliding_correspondence():
    path = np.array([[0.,0.],[20.,0.],[20.,60.],[0.,60.],[0.,0.]])
    uv = np.array([[10.,10.],[10.,30.],[10.,50.]])
    p = np.repeat(path[None], len(uv), axis=0)
    # A tiny horizontal translation crosses the medial axis of the narrow hole.
    old, old_feet, _ = ordered_nearest(uv, p-.001, np.ones((3,4),bool), world_closed=True)
    new, new_feet, _ = ordered_nearest(uv, p+.001, np.ones((3,4),bool), world_closed=True)
    assert np.array_equal(old, [1,1,1]) and np.array_equal(new,[1,1,1])
    assert np.max(np.linalg.norm(new_feet-old_feet,axis=1)) < .003
    assert np.max(np.linalg.norm(uv-np.repeat(path[3:4],3,axis=0),axis=1)) > 0


def test_native_row_cameras_differ_without_changing_world_order():
    path=np.array([[0.,0.],[20.,0.],[20.,60.],[0.,60.],[0.,0.]])
    uv=np.array([[21.,10.],[22.,30.],[23.,50.]])
    p=np.repeat(path[None],3,axis=0);p[:,:,0]+=np.array([1.,2.,3.])[:,None]
    chosen,feet,_=ordered_nearest(uv,p,np.ones((3,4),bool),world_closed=True)
    assert np.array_equal(chosen,[1,1,1]);assert np.array_equal(feet,uv)


def test_closed_seam_crossed_once():
    path=np.array([[0.,0.],[20.,0.],[20.,60.],[0.,60.],[0.,0.]])
    uv=np.array([[0.,30.],[0.,5.],[5.,0.],[15.,0.]])
    chosen,feet,_=ordered_nearest(uv,np.repeat(path[None],4,axis=0),np.ones((4,4),bool),world_closed=True)
    assert np.array_equal(chosen,[3,3,0,0]);assert np.array_equal(feet,uv)


def test_order_dp_matches_exhaustive_for_nonuniform_foot_stations():
    rng=np.random.default_rng(948)
    for _ in range(30):
        stations=np.arange(5)[None]+rng.random((4,5));cost=rng.random((4,5));valid=rng.random((4,5))>.15
        choice,value=_ordered_path(stations,cost,valid)
        options=[(sum(cost[i,j]for i,j in enumerate(ix)),ix)for ix in itertools.product(range(5),repeat=4)
            if all(valid[i,j]for i,j in enumerate(ix))and all(stations[i,ix[i]]<=stations[i+1,ix[i+1]]for i in range(3))]
        exact=min(options,default=(np.inf,None))
        assert value==pytest.approx(exact[0])
        assert (choice is None)==(exact[1] is None)


def test_source_segments_do_not_join_occlusion_gaps_or_invent_reversal():
    assert np.array_equal(source_tangents([[0,0],[0,10]]),[[0,1],[0,1]])
    with pytest.raises(ValueError,match='duplicate'):source_tangents([[0,0],[0,0]])
    with pytest.raises(ValueError,match='reversal'):source_tangents([[0,0],[0,10],[0,0]])
    with pytest.raises(ValueError,match='at least two'):source_tangents([[0,0]])


def test_selected_finite_edge_preserves_native_camera_and_geometry_derivatives(monkeypatch):
    from calibration.solver_runtime import install
    install()
    from calibration.program_tracking import first_shot_source_kernel as kernel
    from calibration.program_tracking import first_shot_ordered_contours as ordered
    monkeypatch.setitem(ordered._ORIGINAL,'evaluate_projected_targets',kernel.evaluate_projected_targets)
    camera=np.zeros(25);camera[6:8]=np.log([900.,870.]);camera[8:11]=[960.,540.,1.2]
    camera[11:15]=[.002,-.0001,.0002,-.0003]
    world=np.array([[-1.,-.5,6.],[1.,-.5,6.],[1.,.5,6.],[-1.,.5,6.],[-1.,-.5,6.]])
    uv=kernel._interfaces()[0](camera,np.array([[1.,0.,6.]]))[0]+[-10.,2.]
    row=dict(_target_key='hole',target=dict(kind='ad_print'),uv=uv,point_correspondence=False,_ordered_curve_edge_index=1)
    g=np.zeros(world.shape+(2,));g[:,:,0]=[1.,0.,0.];g[:,2,1]=1.
    r,j,geo=ordered.evaluate_projected_targets(camera,[row],{'hole':world},{'hole':g})
    rv=ordered.evaluate_projected_targets(camera,[row],{'hole':world},jacobian=False)[0]
    np.testing.assert_array_equal(r,rv)
    scale=np.r_[np.full(3,100.),np.ones(5),np.full(3,1000.),np.ones(4)];h=1e-6
    for col in range(15):
        plus=camera.copy();minus=camera.copy();plus[col]+=h*scale[col];minus[col]-=h*scale[col]
        rp=ordered.evaluate_projected_targets(plus,[row],{'hole':world},jacobian=False)[0]
        rm=ordered.evaluate_projected_targets(minus,[row],{'hole':world},jacobian=False)[0]
        np.testing.assert_allclose((rp-rm)/(2*h),j[:,col],rtol=2e-6,atol=2e-5)
    for col in range(2):
        rp=ordered.evaluate_projected_targets(camera,[row],{'hole':world+h*g[:,:,col]},jacobian=False)[0]
        rm=ordered.evaluate_projected_targets(camera,[row],{'hole':world-h*g[:,:,col]},jacobian=False)[0]
        np.testing.assert_allclose((rp-rm)/(2*h),geo[:,col],rtol=2e-6,atol=2e-5)


def test_open_world_order_cannot_be_replaced_by_independent_nearest():
    path=np.array([[0.,0.],[10.,0.],[10.,10.],[0.,10.],[0.,20.],[10.,20.]])
    uv=np.array([[1.,.1],[1.,19.9],[9.,20.1]])
    chosen,feet,receipt=ordered_nearest(uv,np.repeat(path[None],3,axis=0),np.ones((3,5),bool),world_closed=False)
    assert np.all(np.diff(chosen)>=0)
    assert np.isfinite(feet).all()


def test_qualified_arc_cannot_fall_back_when_candidate_loses_material_side_path(monkeypatch):
    from types import SimpleNamespace
    from calibration.program_tracking import first_shot_ordered_contours as ordered
    from calibration.program_tracking import first_shot_source_kernel as kernel
    uv=np.array([[0.,0.],[0.,1.]])
    target=dict(kind='ad_print',advertisement_id=5,contour_key='hole',contour_role='printed_internal_hole_boundary')
    rows=[dict(target=target,_target_key='k',uv=uv[i:i+1],point_correspondence=False,source_feature_id='f',source_segment_index=0,_source_material_side=dict(material='white_hole_inside_dark_print',gradient=[1.,0.],component_bound=.5))for i in range(2)]
    frame=dict(rows=rows)
    monkeypatch.setattr(ordered,'_qualified_segment',lambda rows,**kwargs:(uv,False))
    class Domain(ValueError):
        def __init__(self,key,indices):super().__init__(key)
    monkeypatch.setattr(kernel,'_interfaces',lambda:(None,None,None,Domain))
    monkeypatch.setattr(kernel,'_project_target',lambda *a:SimpleNamespace(uv=np.array([[0.,1.],[0.,0.]]),segment_valid=np.array([True])))
    trajectory=SimpleNamespace(camera15_mapping=lambda *a,**kw:(np.zeros(25),None,None))
    with pytest.raises(Domain,match='print_arc_no_original_material_side_path'):
        ordered.prepare_frame(frame,{'k':np.array([[0.,0.,1.],[0.,1.,1.]])},trajectory,allowed_targets=[(5,'hole')])


def test_unqualified_source_semantics_are_static_and_explicit(monkeypatch):
    from types import SimpleNamespace
    from calibration.program_tracking import first_shot_ordered_contours as ordered
    from calibration.program_tracking import first_shot_source_kernel as kernel
    target=dict(kind='ad_print',advertisement_id=5,contour_key='hole');uv=np.array([[0.,0.],[0.,1.]])
    rows=[dict(target=target,_target_key='k',uv=uv[i:i+1],point_correspondence=False,source_feature_id='f',source_segment_index=0)for i in range(2)]
    frame=dict(rows=rows)
    def unsupported(rows,**kwargs):raise ValueError('source order not qualified')
    monkeypatch.setattr(ordered,'_qualified_segment',unsupported)
    monkeypatch.setattr(kernel,'_interfaces',lambda:(None,None,None,None))
    result,records=ordered.prepare_frame(frame,{},SimpleNamespace(),allowed_targets=[(5,'hole')])
    assert records[0]['status']=='unqualified_source_arc'
    assert all(a is b for a,b in zip(result['rows'],rows))
    assert '_ordered_curve_edge_index'not in rows[0]


def test_original_operator_context_restored_after_diagnostic_failure():
    from calibration.program_tracking import first_shot_ordered_contours as ordered
    assert ordered._ENABLED.get()
    with pytest.raises(RuntimeError):
        with ordered.original_semantics():
            assert not ordered._ENABLED.get()
            raise RuntimeError('diagnostic')
    assert ordered._ENABLED.get()


def test_alias_needs_exact_reviewed_point_lineage_and_keeps_visible_subruns(tmp_path):
    import json,hashlib
    from calibration.program_tracking.first_shot_ordered_contours import _qualified_segment
    def put(name,value):
        p=tmp_path/name;p.write_text(json.dumps(value),encoding='utf8');return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    old=dict(kind='ad_print',advertisement_id=5,contour_key='old');target=dict(kind='ad_print',advertisement_id=5,contour_key='hole')
    uv=np.array([[0.,0.],[0.,1.],[0.,2.],[0.,3.]])
    packet=put('packet.json',dict(features=[dict(id='Ad5.print.old',world_correspondence_status='named_same_physical_print_component',current_source_only_readout=True,model_or_template_generated_observation=False,segments=[dict(uv=uv.tolist(),closed=True,measurement_origin='current_native_material_normal_half_transition',pixels_measured_in_this_frame=True)])]))
    evidence=put('rgb_review.json',dict(original_rgb_review=True));changes=[];rows=[]
    for i in (1,2):
        pid='Ad5.print.old/segment/0/vertex/'+str(i);item=dict(native_pts=256256,source_point_id=pid,original_UV=uv[i].tolist(),original_target=old,target=target,authority='agent_rgb_review',identity_confirmed_by_user=False,reason='reviewed same physical hole',evidence_references=[evidence]);changes.append(item)
        rows.append(dict(target=target,uv=uv[i:i+1],source_feature_id='Ad5.print.old',source_packet=packet,source_segment_index=0,original_vertex_indices=[i],source_point_ids=[pid],source_identity_revision=item,original_model_target=old))
    ancestor=put('ancestor.json',dict(identity_corrections=changes));revision=put('revision.json',dict(identity_corrections=[],parent_source_revision_ref=ancestor));frame=dict(native_pts=256256,source_revision_reference=revision)
    actual,closed=_qualified_segment(rows,frame=frame)
    np.testing.assert_array_equal(actual,uv[1:3]);assert not closed
    with pytest.raises(ValueError,match='lineage'):_qualified_segment(rows)
    bad=[dict(rows[0]),dict(rows[1],uv=uv[2:3]+.01)]
    with pytest.raises(ValueError,match='original pixel'):_qualified_segment(bad,frame=frame)
    # A later estimator revision can replace the frame root while retaining
    # the corrected row; provenance must come from the actual source binding.
    from calibration.program_tracking.first_shot_ordered_contours import bind_alias_lineage
    stale=put('stale.json',dict(identity_corrections=[]))
    binding=put('binding.json',dict(source=dict(sha256='video'),source_revision=stale,
        post_native_ordinary_revision=revision))
    current=dict(frame,source_revision_reference=stale,rows=rows)
    report=bind_alias_lineage([current],binding,allowed_targets=[(5,'hole')])
    assert report['points']==2
    actual,_=_qualified_segment(rows,frame=current)
    np.testing.assert_array_equal(actual,uv[1:3])
    assert current['source_revision_reference']==stale
    modified=dict(rows[0],source_identity_revision=dict(rows[0]['source_identity_revision'],reason='invented'))
    with pytest.raises(ValueError,match='latest binding-declared'):
        bind_alias_lineage([dict(current,rows=[modified])],binding,allowed_targets=[(5,'hole')])
    (tmp_path/'rgb_review.json').write_text('{}',encoding='utf8')
    with pytest.raises(ValueError,match='evidence changed'):_qualified_segment(rows,frame=frame)


def test_explicit_ordered_successor_only_configures_named_verified_arm_and_restores_loader(tmp_path):
    import importlib.util,json,hashlib
    from calibration.program_tracking.first_shot_ordered_successor import reviewed_ordered_successor
    def ref(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    old=tmp_path/'old.py';old.write_text('original ordered policy',encoding='utf8')
    new=tmp_path/'new.py';new.write_text('reviewed alias successor',encoding='utf8')
    arm=tmp_path/'arm.py';arm.write_text('POLICY_SHA='+repr(ref(old)['sha256'])+'\n',encoding='utf8')
    replay=tmp_path/'replay.json';replay.write_text(json.dumps(dict(state_sha='actual-q',source_pixels_changed=False,counts=dict(associated=1,unqualified_single_source_vertex=0),records=[dict(status='associated',source_points=2)])),encoding='utf8')
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(schema='rinkspace.reviewed-ordered-arc-producer-successor.v1',actual_state_sha256='actual-q',new_target_requires_fresh_complete_J=True,historical_arm=ref(arm),historical_policy=ref(old),successor_policy=ref(new),actual_same_state_replay=ref(replay),contiguous_arcs=1,qualified_points=2)),encoding='utf8')
    original=importlib.util.spec_from_file_location
    with reviewed_ordered_successor(ref(manifest),physical_q_sha256='actual-q')as certificates:
        spec=importlib.util.spec_from_file_location('reviewed_source_and_export_arm',arm);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        assert module.POLICY_SHA==ref(new)['sha256']
        assert len(certificates)==1
        other=importlib.util.spec_from_file_location('unrelated',arm);unrelated=importlib.util.module_from_spec(other);other.loader.exec_module(unrelated)
        assert unrelated.POLICY_SHA==ref(old)['sha256']
    assert importlib.util.spec_from_file_location is original
    with pytest.raises(ValueError,match='same-state'):
        with reviewed_ordered_successor(ref(manifest),physical_q_sha256='other-q'):pass
    with pytest.raises(RuntimeError):
        with reviewed_ordered_successor(ref(manifest),physical_q_sha256='actual-q'):raise RuntimeError('installation failure')
    assert importlib.util.spec_from_file_location is original


def test_material_side_does_not_invent_direction_from_source_order_or_model_winding():
    from calibration.program_tracking.first_shot_arc_polarity import material_side_nearest
    path=np.array([[0.,0.],[20.,0.],[20.,60.],[0.,60.],[0.,0.]])
    uv=np.array([[10.,10.],[10.,30.],[10.,50.]])
    gradient=np.tile([-40.,0.],(3,1))  # Original RGB: white is to the left.
    projected=np.repeat(path[None],3,axis=0);valid=np.ones((3,4),bool);bound=np.full(3,.5)
    edges,feet,_=material_side_nearest(uv,projected,valid,gradient,bound)
    np.testing.assert_array_equal(edges,[1,1,1])
    expected=np.c_[np.full(3,20.),uv[:,1]]
    np.testing.assert_array_equal(feet,expected)
    # A reversed or shuffled original point list is not a different boundary.
    order=np.array([2,0,1])
    _,reordered,_=material_side_nearest(uv[order],projected[order],valid[order],gradient[order],bound[order])
    np.testing.assert_array_equal(reordered,expected[order])
    _,reversed_world,_=material_side_nearest(uv,projected[:,::-1],valid,gradient,bound)
    np.testing.assert_array_equal(reversed_world,expected)
    # Medial-axis proximity never permits switching to the black-facing side.
    _,shifted,_=material_side_nearest(uv,projected+.001,valid,gradient,bound)
    assert np.max(np.linalg.norm(shifted-feet,axis=1))<.002


def test_material_side_requires_real_gradient_and_closed_nonzero_interior():
    from calibration.program_tracking.first_shot_arc_polarity import material_side_nearest,source_light_gradients
    image=np.zeros((30,40,3),np.uint8);image[5:25,10:30]=240
    gradient,bound=source_light_gradients(image,np.array([[29.5,15.]]))
    assert gradient[0,0]<-50 and abs(gradient[0,1])<1e-12 and bound==.5
    path=np.array([[[0.,0.],[20.,0.],[20.,20.],[0.,20.],[0.,0.]]]);uv=np.array([[10.,10.]])
    with pytest.raises(ValueError,match='original RGB'):
        material_side_nearest(uv,path,np.ones((1,4),bool),np.zeros((1,2)),np.array([.5]))
    with pytest.raises(ValueError,match='closed physical'):
        material_side_nearest(uv,path[:,:-1],np.ones((1,3),bool),np.array([[-40.,0.]]),np.array([.5]))
