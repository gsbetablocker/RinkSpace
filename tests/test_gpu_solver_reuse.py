"""Exact direction storage and generation-scoped GPU reuse regressions."""
import os
from unittest.mock import patch
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
from rinkspace.core.gpu import ResidentResidualProblem

gpu=pytest.mark.skipif(not os.environ.get('RINKSPACE_G4_LIBRARY'),reason='explicit new CUDA/cuDSS DLL required')

def test_sparse_shot_chart_preserves_columns_gauges_and_exposure_halos():
    shots=[]
    for i,camera in enumerate((0,1,0)):
        ref=i*10000
        shots.append(dict(id=f's{i}',camera_index=camera,control_begin=32*i,control_stop=32*(i+1),
            row_begin=2*i,row_stop=2*i+2,reference_pts=ref,first_pts=ref,last_pts=ref+3840,
            stop_pts_exclusive=ref+3841,time_base_num=1,time_base_den=12800,
            first_control_time=-.2,knot_stride=4,motion_reference_time=0.))
    timeline=ShotMotionTimeline(shots,.02)
    dense,labels=timeline.directions(continue_endpoint_cubics=True)
    sparse,sparse_labels=timeline.directions(continue_endpoint_cubics=True,sparse=True)
    assert labels==sparse_labels and sparse.has_canonical_format
    np.testing.assert_array_equal(sparse.toarray(),dense.reshape(3*timeline.count,-1))
    np.testing.assert_array_equal((.001*sparse).toarray(),(.001*dense).reshape(3*timeline.count,-1))
    assert sparse.nnz==np.count_nonzero(dense)

@gpu
@pytest.mark.parametrize('pixels',[False,True])
def test_dense_oracle_and_csr_preserve_mixed_motion_rig_scene_jacobian_and_trial(pixels):
    import test_gpu_joint_scene as fixture
    f=fixture.JointSceneGPU();f.setUp()
    scene=np.zeros((63,2));scene[43,0]=.0002;scene[4,1]=.002
    motion=np.zeros((8,3,2));motion[:,0,0]=.001;motion[:,1,1]=.001
    motion[4,2,0]=1e-290 # Exact nonzeros are retained regardless of magnitude.
    rig=np.zeros((27,2));rig[6,0]=.0002;rig[15,1]=.0001
    with f.make() as target:observed=target.evaluate('cpu')['values'][:,3:5]+[.02,-.01]
    results=[]
    for oracle in (True,False):
        with patch.dict(os.environ,RINKSPACE_DENSE_MOTION_ORACLE='1' if oracle else '0'),f.make(directions=scene,observed=observed) as p:
            p.configure_camera_parameters(directions=rig,max_readout=.08)
            directions=motion if oracle else csr_matrix(motion.reshape(24,2))
            plan=f.plan(2);plan['directions']=directions
            if pixels:p.configure_pixel_linearization(directions=directions,local_ids=plan['local_ids'],local_scales=plan['local_scales'])
            else:p.configure_linearization(**plan)
            p.linearize('cpu');cpu=p.probe_rows(np.arange(30),'cpu')
            p.linearize('cuda');jac=p.probe_rows(np.arange(30));blocks=p.export_blocks()
            np.testing.assert_allclose(jac,cpu,atol=3e-10,rtol=1e-9)
            p.configure_solver(lower=[],upper=[],device_budget_bytes=p.creation['free_after_alloc_bytes'],host_budget_bytes=2**30,damping_metric='jacobi_image')
            trial=p.lm_step();state=p.export_state();system=p.export_step_system()
            results.append((jac,blocks,trial,state,system))
    a,b=results
    np.testing.assert_allclose(a[0],b[0],atol=1e-13,rtol=1e-13)
    for key in a[1]:np.testing.assert_allclose(a[1][key],b[1][key],atol=1e-13,rtol=1e-13)
    assert a[2]['summary']['accepted']==b[2]['summary']['accepted']
    for key in ('old_cost','trial_cost','predicted_decrease','actual_decrease','gain_ratio','lambda_used'):
        assert a[2]['summary'][key]==pytest.approx(b[2]['summary'][key],abs=1e-12,rel=1e-12)
    for key in a[3]:np.testing.assert_allclose(a[3][key],b[3][key],atol=1e-10,rtol=1e-10)
    for key in a[4]:np.testing.assert_allclose(a[4][key],b[4][key],atol=1e-12,rtol=1e-12)

@gpu
def test_rejected_retry_reuses_damping_metric_at_same_linearization():
    import test_g3_gpu as fixture
    f=fixture.G3GPU();f.setUp()
    with ResidentResidualProblem(f.core,**f.kw) as base:uv=base.evaluate('cpu')['values'][:,3:5]
    f.kw['observed']=50.+1e5*(uv-50.)
    directions=np.zeros((8,3,1));directions[:,2,0]=.001
    plan={**f.plan,'directions':directions,'local_ids':np.full(4,-1,np.int32),'local_scales':np.zeros(0)}
    with f.run_problem(plan) as p:
        p.configure_solver(lower=[],upper=[],device_budget_bytes=p.creation['free_after_alloc_bytes'],host_budget_bytes=2**30,damping_metric='jacobi_image',initial_lambda=1e-20)
        assert not p.lm_step()['summary']['accepted']
        first=p.performance_metrics()
        assert not p.lm_step()['summary']['accepted']
        second=p.performance_metrics()
        assert first['linearization_generation']==second['linearization_generation']
        assert second['damping_metric_rebuilds']==1 and second['damping_metric_cache_hits']>=1

@gpu
def test_accepted_error_cache_and_refine_match_full_current_projection():
    import test_g3_gpu as fixture
    f=fixture.G3GPU();f.setUp()
    with ResidentResidualProblem(f.core,**f.kw) as p:
        p.configure_pixel_linearization(directions=f.directions,local_ids=f.plan['local_ids'],local_scales=f.plan['local_scales'])
        p.configure_solver(lower=np.zeros(2),upper=np.ones(2),device_budget_bytes=p.creation['free_after_alloc_bytes'],host_budget_bytes=2**30,profile_trial_stations=True)
        assert p.lm_step()['summary']['accepted']
        def check():
            errors=p.evaluate_errors();full=p.evaluate('cuda')
            np.testing.assert_array_equal(full['status'],0)
            np.testing.assert_allclose(errors,np.hypot(full['values'][:,9],full['values'][:,10]),rtol=1e-12,atol=1e-11)
        check();p.refine_curve_stations();check()

@gpu
def test_face_storage_grows_from_actual_count_and_serial_parallel_order_matches():
    from test_camera_envelope import CameraEnvelopeTests
    import test_g3_gpu as fixture
    camera=CameraEnvelopeTests();camera.setUp();f=fixture.G3GPU();f.setUp()
    rig=camera.rig.copy();rig[37]=.02;rig[20]=.1+4*(128*np.finfo(float).eps)
    kw={**f.kw,'rig':rig,'controls':np.tile([0.,.15,np.log(100.)],(8,1)),'codes':np.zeros(4,np.int32)}
    rd=np.zeros((27,2));rd[11,0]=1.;rd[15,1]=1.
    with ResidentResidualProblem(f.core,**kw) as p:
        p.configure_geometry([dict(kind=4,vertices=[[0.,3.,0.]],parameters=[])])
        p.configure_camera_parameters(directions=rd,max_readout=.08)
        p.configure_camera_envelope(forward_bounds=[.1,.3],pupil_bounds=[-.1,.1],max_pan_tilt=camera.limits[4],
            max_tilt_deviation=camera.limits[5],elevation_bounds=camera.limits[6:8],roll_bounds=camera.limits[8:10])
        p.configure_pixel_linearization(directions=np.zeros((8,3,2)),local_ids=np.full(4,-1),local_scales=[]);f.attach(p)
        assert p.performance_metrics()['stationarity_face_storage_bytes']==0
        serial=p.stationarity_diagnostics(include_constraints=True)
        perf=p.performance_metrics()
        assert perf['stationarity_face_storage_bytes']==perf['stationarity_active_faces']*perf['stationarity_face_sizeof']
        with patch.dict(os.environ,RINKSPACE_STATIONARITY_FACE_BACKEND='serial'):
            other=p.stationarity_diagnostics(include_constraints=True)
        np.testing.assert_array_equal(serial['constraint_records'],other['constraint_records'])

@gpu
def test_sparse_raw_pixels_reject_implicit_query_constraints():
    from test_gpu_profile_queries import ProfileQueriesGPU
    f=ProfileQueriesGPU();f.setUp()
    with f.make() as p:
        with pytest.raises(RuntimeError,match='cannot inherit source query'):
            p.configure_pixel_linearization(directions=csr_matrix(f.plan['directions'].reshape(24,3)),
                local_ids=np.full(30,-1,np.int32),local_scales=[])

@gpu
def test_geometry_samples_refresh_after_accepted_scene_change():
    import test_gpu_joint_scene as fixture
    f=fixture.JointSceneGPU();f.setUp()
    directions=np.zeros((63,1));directions[43,0]=.001
    with f.make(state=f.state+directions[:,0]*.5) as target:
        observed=target.evaluate('cuda')['values'][:,3:5]
    answers=[]
    for cached in ('0','1'):
        with patch.dict(os.environ,RINKSPACE_STATION_GEOMETRY_CACHE=cached),f.make(directions=directions,observed=observed) as p:
            p.configure_pixel_linearization(directions=np.zeros((8,3,1)),local_ids=np.arange(30,dtype=np.int32),local_scales=np.full(30,.001))
            p.configure_solver(lower=np.zeros(30),upper=np.r_[np.ones(10),np.full(5,3.),np.ones(15)],
                device_budget_bytes=p.creation['free_after_alloc_bytes'],host_budget_bytes=2**30,profile_trial_stations=True)
            p.refine_curve_stations()
            assert p.lm_step()['summary']['accepted']
            assert p.export_state()['scene_values'][43]!=f.state[43]
            p.refine_curve_stations();p.linearize('cuda')
            answers.append((p.export_state(),p.evaluate_errors(),p.probe_rows(np.arange(30)),p.performance_metrics()))
    for key in answers[0][0]:np.testing.assert_allclose(answers[0][0][key],answers[1][0][key],atol=3e-11,rtol=3e-12)
    np.testing.assert_allclose(answers[0][1],answers[1][1],atol=1e-10,rtol=1e-11)
    np.testing.assert_allclose(answers[0][2],answers[1][2],atol=3e-11,rtol=3e-12)
    assert 0<answers[1][3]['station_unique_geometry_samples']<answers[1][3]['station_projection_samples']
