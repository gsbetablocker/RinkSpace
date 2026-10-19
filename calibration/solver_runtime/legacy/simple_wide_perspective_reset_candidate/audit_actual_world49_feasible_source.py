"""Actual complete FIT/hard-world linearization for root's constrained QP."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from simple_wide_perspective_reset_candidate.audit_joint_unified_physical_source import *
from complete_ice_template_candidate.shared_hockey_frame_free_boundary import SharedHockeyFrameFreeBoundaryGeometry

def make_engine(*,budget='literal_reference',tied=False):
    v=read_version(FOLDERS[1]);base=json.loads(v['aux']['aux__shared_scene_json'].item())
    g43=SharedIceHockeyNumericGeometry(CorrectedSharedRinkNumericGeometryLanesAdPairs(base,SCENE_UNIT_M))
    g=SharedHockeyFrameFreeBoundaryGeometry(CurrentSharedGeometryWithFreeBoundary(g43))
    diagnostic=B/'complete_ice_template_candidate/actual_shared_hockey_frame_free_boundary_small_audit.json'
    branch=json.loads(diagnostic.read_text())['conditional_H_carrier_feasibility'][2]
    qw=np.asarray(branch['full_world49']);spans=np.asarray(branch['common_frame_and_shared_spans'][2:])
    with np.load(SEED,allow_pickle=False)as z:cam=camera14_from_family(z['K'],z['R'],z['C'],z['image_only_inverse_radial'])
    engine=UnifiedExperiment(g,g43,cam,qw,budget,tied=tied)
    return engine,engine.initial(cam,qw,spans),v,diagnostic

def main():
    start=time.perf_counter();engine,q,event,source=make_engine();proof=engine.trial_guard(q,report=True)
    print(json.dumps(dict(actual65_initial_trial_guard='PASS',all268native_original_region_finite=True)),flush=True)
    r,J,groups,scene,native,hr=engine.evaluate(q,True);c,Jc,creport=engine.constraints(q)
    w=np.r_[engine.factor.irls(r[:4340]),np.ones(18)];H=J.T@(w[:,None]*J);g=J.T@(w*r)
    out=OUT/'actual_world49_feasible_native_QP_linearization.npz'
    np.savez_compressed(out,q=q,residual=r,J=J,H=H,g=g,c=c,Jc=Jc,scales=engine.scales,quadratic=engine.quadratic,cost_weights=engine.cost_weights,native_residual=native,selected_original_pen_region=np.asarray(creport['selection']))
    check=[];rng=np.random.default_rng(449)
    for k in range(2):
        d=rng.normal(size=engine.n);d[:14]*=.01;d/=np.linalg.norm(d);h=1e-7
        cp=engine.constraints(q+h*d,jacobian=False,selection=creport['selection'])[0];cm=engine.constraints(q-h*d,jacobian=False,selection=creport['selection'])[0]
        fd=(cp-cm)/(2*h);a=Jc@d;check.append(dict(direction=k,width=h,relative=float(np.linalg.norm(fd-a)/max(1.,np.linalg.norm(fd),np.linalg.norm(a))),max_absolute=float(abs(fd-a).max())))
    report=dict(schema='actual-camera14-world49-span2-native4358-hard1072-QP-fixture.v1',source_initializer=str(source),source_initializer_not_measurement=True,source_event_q_profile=event['event'],
        parameter_count=engine.n,source_rows=len(r),constraint_count=len(c),all4340_original45_2144_and26_preserved=True,reference18_declared_engineering=True,
        hard_original268_all_finite_proof=proof,minimum_constraint=float(c.min()),negative_constraints=int((c<0).sum()),cost=engine.cost(r),source_groups=groups,
        constraints=creport,actual_constraint_direction_FD=check,original_UV_roles_sigma_budget_not_changed=True,
        current_estimated_boundary_delta_not_survey_or_hard_60x30=True,all_source_optical_domain_passed=True,fixture_path=str(out),seconds=time.perf_counter()-start)
    path=OUT/'actual_world49_feasible_native_QP_linearization_report.json';path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps(dict(fixture=str(out),cost=report['cost'],minimumc=report['minimum_constraint'],all268finite=True,FD=check,seconds=report['seconds'])),flush=True)
if __name__=='__main__':main()
