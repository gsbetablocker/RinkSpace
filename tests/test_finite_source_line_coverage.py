import numpy as np
import pytest

from calibration.solver_runtime import install
install()

from calibration.program_tracking.finite_source_line_coverage import (
    NativeLineSections, NativeLineCoverageInventory, FiniteLineCoverageError,
    finite_line_query_intersections, initialize_native_line_coverage, check_native_line_coverage)


def camera():
    return np.r_[np.zeros(6),np.log([100.,100.]),[960.,540.,0.],np.zeros(14)]


def sections(query=(990.,530.),normal=(0.,1.)):
    return NativeLineSections(1,256256,'line','ice.real_material',
        np.asarray([query]),np.asarray([normal]),np.asarray([[990.,540.]]),
        ('actual_native_query_1',),{'actual_material_endpoint_measured':False})


def test_finite_material_must_reach_real_query_even_with_zero_normal_error():
    cam=camera();src=sections();world=np.array([[-5.,0.,10.],[0.,0.,10.]])
    before_cam=cam.copy();before_world=world.copy();before_query=src.query_uv.copy()
    result=finite_line_query_intersections(cam,world,src)
    assert not result['passed']
    assert result['queries'][0]['candidates'][0]['fraction']==pytest.approx(1.6)
    assert result['queries'][0]['candidates'][0]['query_section_error_px']<1e-10
    # No UV, geometry or camera edit is performed to make the domain pass.
    assert np.array_equal(cam,before_cam) and np.array_equal(world,before_world)
    assert np.array_equal(src.query_uv,before_query)


def test_extent_is_a_lower_bound_and_extension_does_not_measure_an_endpoint():
    src=sections();cam=camera()
    for endpoint in [3.,5.,15.]:
        r=finite_line_query_intersections(cam,[[-5.,0.,10.],[endpoint,0.,10.]],src)
        assert r['passed'] and r['finite_endpoint_observed'] is False
        assert r['lower_bound_semantics']


def test_support_can_lie_exactly_at_a_finite_material_endpoint():
    r=finite_line_query_intersections(camera(),[[-5.,0.,10.],[3.,0.,10.]],sections())
    assert r['passed']
    assert r['queries'][0]['candidates'][0]['fraction']==pytest.approx(1.)


def test_an_intersection_behind_the_camera_is_not_material_support():
    r=finite_line_query_intersections(camera(),[[-5.,0.,-10.],[5.,0.,-10.]],sections())
    assert not r['passed'] and r['missing_queries']==['actual_native_query_1']


def test_brown4_uses_actual_curved_projection_and_all_real_root_candidates():
    cam=camera();cam[11]=.2
    world=np.array([[-5.,0.,10.],[5.,0.,10.]])
    r=finite_line_query_intersections(cam,world,sections())
    assert r['passed']
    actual=r['queries'][0]['candidates'][0]
    assert actual['UV'][0]==pytest.approx(990.,abs=1e-7)
    assert abs(actual['station_scene']-8.)>.01


class MovingPhysicalCamera:
    readout_milliseconds=12.
    def camera25_at_rows(self,rows,*,geometry_navigation=False):
        rows=np.asarray(rows,float)
        assert np.all((rows>=0)&(rows<=1079)) and not geometry_navigation
        c=np.tile(camera(),(len(rows),1))
        c[:,1]=.03*(rows-540.)
        return c


def test_row_exposure_uses_intersection_y_not_query_center_or_source_seed_row():
    src=sections((995.,540.),(.2,.98));world=np.array([[-5.,2.,10.],[5.,2.,10.]])
    r=finite_line_query_intersections(camera(),world,src,row_context=MovingPhysicalCamera(),readout_milliseconds=12.)
    assert r['passed']
    point=r['queries'][0]['candidates'][0]
    y=540.+20./1.3
    assert point['UV'][1]==pytest.approx(y,abs=1e-6)
    assert point['UV'][0]==pytest.approx(995.+.2/.98*(y-540.),abs=1e-6)
    assert abs(point['UV'][1]-540.)>10.
    assert point['sensor_row_error_px']<=1e-6
    assert point['query_section_error_px']<=1e-6


def test_nonzero_row_timing_cannot_silently_use_a_central_camera():
    with pytest.raises(ValueError,match='actual row-camera context'):
        finite_line_query_intersections(camera(),[[-5.,0.,10.],[5.,0.,10.]],sections(),readout_milliseconds=12.)


def test_central_seed_outside_sensor_can_have_a_real_rolling_row_root_inside():
    world=np.array([[-5.,54.,10.],[5.,54.,10.]])
    src=sections((990.,1000.))
    assert not finite_line_query_intersections(camera(),world,src)['passed']
    result=finite_line_query_intersections(camera(),world,src,
        row_context=MovingPhysicalCamera(),readout_milliseconds=12.)
    assert result['passed']
    actual=result['queries'][0]['candidates'][0]
    assert actual['UV'][1]==pytest.approx(540.+540./1.3,abs=1e-6)
    assert actual['sensor_row_error_px']<=1e-6
    # Initializing at a legal row must not clamp an actually external root.
    world[:,1]=100.
    assert not finite_line_query_intersections(camera(),world,src,
        row_context=MovingPhysicalCamera(),readout_milliseconds=12.)['passed']


def test_actual_native_queries_cannot_be_invented_outside_the_sensor():
    with pytest.raises(ValueError,match='inside the sensor'):
        sections((-20.,540.))


def test_high_order_brown_cannot_silently_enter_a_brown4_guard():
    cam=camera();cam[15]=.001
    with pytest.raises(ValueError,match='Brown4'):
        finite_line_query_intersections(cam,[[-5.,0.,10.],[5.,0.,10.]],sections())


class PhysicalState:
    row_readout_index=None
    def __init__(self):self.calls=[]
    def camera25(self,q,index):return camera()
    def guard(self,q):self.calls.append(np.asarray(q).copy())


def movable_sections():
    src=sections()
    return NativeLineSections(src.index,src.native_pts,src.target_key,src.source_feature_id,
        src.query_uv,src.profile_normal,src.measured_uv,src.query_ids,src.evidence,end_columns=(0,))


def test_minimum_endpoint_restart_preserves_all_other_parameter_bytes():
    state=PhysicalState();q=np.array([0.,.4,-.2],np.float64);before=q.copy()
    src=movable_sections();inventory=NativeLineCoverageInventory((src,),(),(0,))
    build=lambda value:{'line':np.array([[-5.,0.,10.],[value[0],0.,10.]])}
    candidate,proof=initialize_native_line_coverage(state,q,inventory,build)
    assert candidate[0]==pytest.approx(3.,abs=1e-9)
    assert np.array_equal(candidate[1:],before[1:])and np.array_equal(q,before)
    assert proof['unknown_terminal_values_are_only_source_supported_lower_bounds']
    assert proof['parameter_values_not_clipped']
    assert check_native_line_coverage(state,candidate,inventory,build(candidate))['passed']
    again,unchanged=initialize_native_line_coverage(state,candidate,inventory,build)
    assert again is candidate and unchanged['all_parameter_values_byte_exact']


def test_initializer_refuses_to_move_an_undeclared_start():
    state=PhysicalState();q=np.array([0.,.4,-.2],np.float64)
    src=sections((850.,530.))
    src=NativeLineSections(src.index,src.native_pts,src.target_key,src.source_feature_id,
        src.query_uv,src.profile_normal,src.measured_uv,src.query_ids,src.evidence,end_columns=(0,))
    with pytest.raises(FiniteLineCoverageError,match='fixed or ambiguous endpoint'):
        initialize_native_line_coverage(state,q,(src,),lambda value:{'line':np.array([[-5.,0.,10.],[value[0],0.,10.]])})
    assert np.array_equal(q,[0.,.4,-.2])


def test_incomplete_source_inventory_cannot_pass_as_empty_or_partial_coverage():
    inventory=NativeLineCoverageInventory((),({'reason':'original source SHA changed'},),(0,))
    with pytest.raises(FiniteLineCoverageError,match='inventory is incomplete'):
        check_native_line_coverage(PhysicalState(),np.zeros(3),inventory,{})


def test_minimum_extension_cannot_override_existing_material_footprint():
    class BoundedState(PhysicalState):
        def guard(self,q):
            if q[0]>2.:raise ValueError('actual material leaves ice footprint')
    q=np.array([0.,.4,-.2]);before=q.copy()
    with pytest.raises(ValueError,match='ice footprint'):
        initialize_native_line_coverage(BoundedState(),q,(movable_sections(),),
            lambda value:{'line':np.array([[-5.,0.,10.],[value[0],0.,10.]])})
    assert np.array_equal(q,before)


def test_existing_initializer_restores_only_an_end_that_needed_real_coverage():
    state=PhysicalState();state.initial_q=np.array([4.,100.,200.])
    q=np.array([0.,.4,-.2]);before=q.copy()
    inventory=NativeLineCoverageInventory((movable_sections(),),(),(0,),
        initializer_strategy='source_required_and_existing_bound_initial_values')
    build=lambda value:{'line':np.array([[-5.,0.,10.],[value[0],0.,10.]])}
    candidate,proof=initialize_native_line_coverage(state,q,inventory,build)
    assert candidate[0]==4.
    assert np.array_equal(candidate[1:],q[1:]) and np.array_equal(q,before)
    assert proof['endpoint_initializer_selections'][0]['selection']=='existing_parameter_initializer'
    assert proof['endpoint_initializer_selections'][0]['existing_initial_value_is_hard_lower_bound']is False
    # Once material is already covered, this initializer does not enforce its
    # old value as a continuing lower bound or reset unrelated coordinates.
    covered=np.array([3.2,.4,-.2]);again,p=initialize_native_line_coverage(state,covered,inventory,build)
    assert again is covered and p['all_parameter_values_byte_exact']


def _native_coverage_with_noise(tmp_path):
    import hashlib
    from calibration.program_tracking.first_shot_source_evidence import _influence
    from calibration.program_tracking.first_shot_noise_revision import make_noise_replacement
    from calibration.program_tracking.source_evidence_whitening import (
        latent_whitening_operator, SourceWhiteningOperator)
    fid='ice.real_material'; pts=256256
    native_x=np.array([970.,980.,1000.,1010.]); representative_x=np.array([975.,990.,1005.])
    D=_influence(np.c_[native_x,np.ones(4)],np.ones(4))
    S=np.c_[representative_x,np.ones(3)]
    B=np.zeros((3,6)); B[:,1::2]=np.eye(3)
    support=np.diag([10.,12.,14.,16.]); ids=('source_a','source_b','source_c')
    old=latent_whitening_operator(ids,S,explicit_noise_factor=D@support,
        physical_basis=B,group_id=f'{pts}/{fid}/source_line',covariance_origin='historical profile support')
    row=dict(target=dict(kind='ice_curve',curve_id='real_material'),_target_key='line',
             uv=np.c_[representative_x,np.full(3,540.)],sigma=np.full(3,12.),
             source_point_ids=list(ids),source_feature_id=fid,original_vertex_indices=[0,1,2])
    group=dict(operator=old,point_indices=[0,1,2])
    frame=dict(index=1,native_pts=pts,source_packet={'path':'source','sha256':'fixture'},
               rows=[row],whitening_groups=[group])
    actual=[dict(uv=[x,540.],sigma_px=s,source_evidence=dict(
        measurement_origin='current_native_source_identified_thin_paint_two_local_luminance_flanks',
        actual_source_support_outside_person_and_broadcast_masks=True,
        query_uv=[x,530.],profile_normal=[0.,1.],query_index=i))
        for i,(x,s) in enumerate(zip(native_x,np.diag(support)))]
    feature=dict(id=fid,source_line_fit=dict(current_frame_only=True,direct_readouts=actual,
        point_origin=[0.,540.],unit_tangent=[1.,0.],unit_normal=[0.,1.],normal_offset_coefficients=[0.,0.]))
    evidence=tmp_path/'noise_evidence.json'; evidence.write_text('{}',encoding='utf-8')
    item=make_noise_replacement(frame,group,physical_basis=B,estimator_output_sensitivity=S,
        coefficient_influence=D,original_support_factor=support,
        localization_noise_factor=np.c_[.4*np.eye(4),.3*np.ones(4)],
        evidence_references=[dict(path=str(evidence),sha256=hashlib.sha256(evidence.read_bytes()).hexdigest())],
        noise_method='fixture native midpoint noise with shared component')
    revised=dict(group,operator=SourceWhiteningOperator.from_record(item['new_operator']),
        original_support_operator=old.to_record(),localization_noise_revision=item)
    return frame,row,feature,group,revised


def test_native_finite_coverage_accepts_new_noise_without_changing_real_sections(tmp_path):
    from calibration.program_tracking.finite_source_line_coverage import _compile_one
    frame,row,feature,old,new=_native_coverage_with_noise(tmp_path)
    native_ref={'path':'original_native','sha256':'fixture'}
    before=_compile_one(frame,row,feature,native_ref)
    frame['whitening_groups']=[new]
    after=_compile_one(frame,row,feature,native_ref)
    assert np.array_equal(before.query_uv,after.query_uv)
    assert np.array_equal(before.measured_uv,after.measured_uv)
    assert before.query_ids==after.query_ids
    assert after.evidence['localization_noise_replacement_authenticated']
    assert after.evidence['current_full_W_sha256']!=before.evidence['current_full_W_sha256']
    # A new localization covariance cannot extend a too-short world line.
    short=np.array([[-5.,0.,10.],[0.,0.,10.]])
    assert not finite_line_query_intersections(camera(),short,before)['passed']
    assert not finite_line_query_intersections(camera(),short,after)['passed']


@pytest.mark.parametrize('mutation',['source_UV','reverted_W','changed_D'])
def test_finite_coverage_noise_certificate_cannot_hide_changed_measurement(tmp_path,mutation):
    from calibration.program_tracking.finite_source_line_coverage import _compile_one
    frame,row,feature,old,new=_native_coverage_with_noise(tmp_path)
    frame['whitening_groups']=[new]
    if mutation=='source_UV': row['uv'][0,1]+=.01
    if mutation=='reverted_W': new['operator']=old['operator']
    if mutation=='changed_D': new['localization_noise_revision']['coefficient_influence'][0][0]+=.01
    with pytest.raises(ValueError,match='source membership|localization covariance|original source covariance'):
        _compile_one(frame,row,feature,{'path':'native','sha256':'fixture'})
