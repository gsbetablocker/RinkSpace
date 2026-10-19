from types import SimpleNamespace
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from calibration.program_tracking.first_shot_finite_ad_coverage import (
    projected_station_margin, AdSupportSection, evaluate_ad_support, FiniteAdCoveragePolicy)


def test_finite_support_rejects_missing_station_without_inventing_corner_cost():
    p=np.array([[0.,4.],[10.,4.]])
    m,i,sign,edge=projected_station_margin(p,[True],[18.,0.],[1.,0.])
    assert m==-8. and i==1 and sign==1.
    # Normal misalignment remains a separate original localization factor.
    assert projected_station_margin(p,[True],[5.,100.],[1.,0.])[0]>0
    assert projected_station_margin(p,[True],[10.0003,0.],[1.,0.],.0005)[0]>0
    with pytest.raises(ValueError,match='no central'):projected_station_margin(p,[False],[5.,0.],[1.,0.])


def test_coverage_is_union_of_finite_valid_segments_not_global_convex_hull():
    p=np.array([[0.,0.],[2.,0.],[8.,0.],[10.,0.]])
    assert projected_station_margin(p,[True,False,True],[5.,0.],[1.,0.])[0]==-3.


def test_active_endpoint_camera_and_world_derivatives_match_actual_projection(monkeypatch):
    from calibration.solver_runtime import install
    install()
    from calibration.program_tracking import first_shot_row_camera as rows
    cam=np.zeros(25);cam[6:8]=np.log([900.,880.]);cam[8:10]=[960.,540.];cam[11:15]=[.003,-.0002,.0001,-.0003]
    class Trajectory:
        def __init__(self,q):self.q=q
        def camera15_mapping(self,y,*,jacobian):
            c=cam.copy();c[6]+=self.q[1];J=np.zeros((15,1));J[6,0]=1.
            return c,np.array([1]),J if jacobian else None
    monkeypatch.setattr(rows,'prepare_frame_row_camera',lambda state,q,index:Trajectory(q))
    state=SimpleNamespace(world_parameter_indices=np.array([0]))
    section=AdSupportSection(1,256256,'ad','Ad4.top_edge',0,np.array([[1300.,500.]]),np.array([1.,0.]),0.,('native-support',),(0,),{})
    def world(q):return np.array([[-1.,0.,6.],[1.+q[0],0.,6.]])
    D=np.zeros((2,3,1));D[1,0,0]=1.;q=np.array([.1,.01])
    actual=evaluate_ad_support(state,q,[section],{'ad':world(q)},{'ad':D})
    assert actual['margins'][0]<0
    for column in range(2):
        d=np.eye(2)[column]*1e-6
        finite=(evaluate_ad_support(state,q+d,[section],{'ad':world(q+d)})['margins']-evaluate_ad_support(state,q-d,[section],{'ad':world(q-d)})['margins'])/2e-6
        np.testing.assert_allclose(actual['J'][:,column].toarray().ravel(),finite,rtol=2e-6,atol=1e-5)


def test_restoration_preserves_covered_materials_and_reverts_to_true_source_cost():
    from calibration.solver_runtime import install
    install()
    owner=SimpleNamespace(source_producers=[],recovery_references=[],_references=[],
        solver_kwargs=lambda:dict(checkpoint_identity={}),checkpoint_identity=lambda:{})
    policy=FiniteAdCoveragePolicy.__new__(FiniteAdCoveragePolicy);policy.owner=owner;policy.sections=(1,2);policy.unsupported=();policy.context={}
    values={0.:np.array([8.,0.]),1.:np.array([4.,0.]),2.:np.array([4.,.1]),3.:np.zeros(2)}
    policy.evaluate=lambda q,jacobian=False:dict(group_violations=values[float(q[0])])
    policy.install(__import__('calibration.program_tracking.first_shot_source_transition',fromlist=['reference']).reference(__file__))
    policy.context=dict(q_sha256=__import__('hashlib').sha256(np.array([0.]).tobytes()).hexdigest(),contraction=dict(common_fraction=.5))
    accept=owner.solver_kwargs()['source_trial_acceptance']
    options=dict(before_cost=10.,after_cost=11.,default_source_cost_accepts=False,trial_context=dict(alpha=1.,physical_and_complete_source_guard_passed=True))
    assert accept(np.array([0.]),np.array([1.]),**options)['accept']
    assert not accept(np.array([0.]),np.array([2.]),**options)['accept']
    ordinary=accept(np.array([3.]),np.array([3.]),**options)
    assert ordinary['mode']=='source_objective'and not ordinary['accept']


def test_recovery_prediction_uses_actual_step_and_keeps_QP_certificate_scope():
    from calibration.program_tracking.first_shot_full_physical import recovery_step_receipt
    H=csr_matrix(np.array([[4.,1.],[1.,3.]]));g=np.array([-3.,2.]);q=np.zeros(2)
    proposed=np.array([.5,-.3]);recovered=np.array([.5,-.1])
    r=recovery_step_receipt(q,proposed,recovered,H,g)
    assert r['changed_columns']==[1]
    assert r['proposed_values']==[-.3]and r['recovered_values']==[-.1]
    assert r['proposed_source_prediction']!=r['recovered_source_prediction']
    assert r['original_QP_KKT_does_not_certify_a_changed_recovered_step']


def test_policy_uses_actual_programme_world_and_full_world_camera_columns(monkeypatch):
    from collections import OrderedDict
    from calibration.solver_runtime import install
    install()
    from calibration.program_tracking import first_shot_row_camera as rows
    cam=np.zeros(25);cam[6:8]=np.log([900.,880.]);cam[8:10]=[960.,540.]
    class Trajectory:
        def __init__(self,q):self.q=q
        def camera15_mapping(self,y,*,jacobian):
            c=cam.copy();c[6]+=self.q[6];J=np.zeros((15,1));J[6,0]=1.
            return c,np.array([6]),J if jacobian else None
    monkeypatch.setattr(rows,'prepare_frame_row_camera',lambda state,q,index:Trajectory(q))
    section=AdSupportSection(1,256256,'ad','Ad4.top_edge',0,np.array([[1300.,500.]]),np.array([1.,0.]),0.,('native-support',),(0,),{})
    def actual_world(q,jacobian):
        P=np.array([[-1.,0.,6.],[1.+q[0]+q[5],0.,6.]])
        D=np.zeros((2,3,6));D[1,0,[0,5]]=1.
        return {'ad':P},({'ad':D}if jacobian else None)
    def wrong_unshifted_world(*a):raise AssertionError('Unshifted constructor world must not replace FIT world')
    state=SimpleNamespace(world_parameter_indices=np.arange(6),scene=lambda q:None)
    programme=SimpleNamespace(state=state,world=actual_world,bundle=SimpleNamespace(world=wrong_unshifted_world,_world_cache={}))
    owner=SimpleNamespace(programme=programme,programme_count=7,effective_q_and_chain=lambda q:(q,np.eye(7)),fixed=np.array([],int),guard=lambda q:True)
    policy=FiniteAdCoveragePolicy.__new__(FiniteAdCoveragePolicy);policy.owner=owner;policy.sections=(section,);policy.cache=OrderedDict()
    q=np.array([.1,0,0,0,0,.2,.01]);actual=policy.evaluate(q,True)
    assert actual['J'].shape==(1,7)
    for column in(0,5,6):
        d=np.eye(7)[column]*1e-6
        fd=(policy.evaluate(q+d)['margins']-policy.evaluate(q-d)['margins'])/2e-6
        np.testing.assert_allclose(actual['J'][:,column].toarray().ravel(),fd,rtol=1e-6,atol=1e-5)
    report=policy.directional_check(q,np.array([.2,0,0,0,0,.3,-.1]))
    assert report['passed']and report['full_camera_and_world_direction']
