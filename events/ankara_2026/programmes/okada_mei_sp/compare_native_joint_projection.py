"""Read-only projection comparison of two explicit full multicamera states.

Both states use the same source and path definitions. The shared probe uses
baseline stations in both models, so a reoptimized nuisance station cannot hide
an overlay change. Candidate saved-station residuals are evaluated separately.
"""
from pathlib import Path
import argparse,json,hashlib,time
import numpy as np
import prepare_native_multicamera as prepared
from rinkspace.core.native import NativeCore
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
from rinkspace.domain.parameters import SceneParameterModel,ParameterState
from rinkspace.session.gpu_geometry import NativeSceneGPU
from rinkspace.path_policy import guard_project_write
from export_native_six_review import _verify_native_library_binding


def digest(path):
    with Path(path).open('rb') as h:return hashlib.file_digest(h,'sha256').hexdigest()


def summary(x):
    x=np.asarray(x,float)
    if not len(x):return None
    return dict(median=float(np.median(x)),p95=float(np.quantile(x,.95)),p99=float(np.quantile(x,.99)),maximum=float(x.max()),rms=float(np.sqrt(np.mean(x*x))))


def main(a):
    started=time.monotonic();out=guard_project_write(a.output)
    if out.exists() or out.with_suffix('.npz').exists():raise FileExistsError(out)
    ma=prepared.read(a.baseline_run/'joint_manifest.json');mb=prepared.read(a.candidate_run/'joint_manifest.json')
    pa=prepared.read(a.baseline_run/'mapping.json');pb=prepared.read(a.candidate_run/'mapping.json')
    for key in ('shots','cameras','control_dt','first_control_time','max_readout','point_input_sha256','scene_path','scene_file_sha256','native_library','native_library_sha256'):
        left,right=ma[key],mb[key]
        if key=='shots':
            left=[{k:v for k,v in s.items() if k!='local_motion_refinement'} for s in left]
            right=[{k:v for k,v in s.items() if k!='local_motion_refinement'} for s in right]
        if left!=right:raise ValueError('Comparison changes a projection dependency: '+key)
    if pa['path_ids']!=pb['path_ids']:raise ValueError('Material paths differ')
    protected=[a.baseline_state,a.candidate_state,a.baseline_run/'point_observations.npz',a.candidate_run/'point_observations.npz',Path(ma['native_library']),Path(ma['scene_path'])]
    identities={str(p):digest(p) for p in protected}
    if identities[str(a.baseline_run/'point_observations.npz')]!=ma['point_input_sha256'] or identities[str(a.candidate_run/'point_observations.npz')]!=ma['point_input_sha256']:raise ValueError('Source input changed')
    z=prepared.npz(a.baseline_run/'point_observations.npz');va=prepared.npz(a.baseline_state);vb=prepared.npz(a.candidate_state)
    model=SceneParameterModel(prepared.read(ma['scene_path']),Path(ma['scene_path']),optimize_scene=True)
    for m,v,p in ((ma,va,pa),(mb,vb,pb)):
        prepared.validate_snapshot(v,model.initial_state.layout,p,m,z);_verify_native_library_binding(m,v)
    core=NativeCore(ma['native_library']);timeline=ShotMotionTimeline(ma['shots'],ma['control_dt'],ma['first_control_time'])
    times=timeline.row_times(z['pts']);camera_ids=np.empty(len(times),np.int32)
    for shot in ma['shots']:camera_ids[shot['row_begin']:shot['row_stop']]=shot['camera_index']
    paths=np.asarray(pa['path_ids'])[z['path_indices']].tolist()
    camera={**prepared.read(ma['cameras'][0]['camera_definition']),'first_control_time':timeline.first,'control_dt':timeline.dt}
    def evaluate(m,v,stations,label):
        native=NativeSceneGPU(m['scene_path'],core,parameter_state=ParameterState(model.initial_state.layout,v['scene_values']))
        with native.resident(camera=camera,controls=v['controls'],path_ids=paths,stations=stations,pts=z['pts'],observed=z['uv'],motion_times=times,rigs=v['rig'],camera_ids=camera_ids) as resident:
            resident.configure_projection_time_support(half_seconds=m['max_readout']/2)
            result=resident.evaluate('cuda')
        if np.any(result['status']):raise ValueError('Invalid full projection: '+label)
        print(json.dumps(dict(event='evaluated',label=label,rows=len(times),elapsed_seconds=time.monotonic()-started)),flush=True)
        return result['values']
    before=evaluate(ma,va,va['stations'],'baseline')
    common=evaluate(mb,vb,va['stations'],'candidate_at_baseline_material_stations')
    after=evaluate(mb,vb,vb['stations'],'candidate_at_saved_fit_stations')
    delta=common[:,3:5]-before[:,3:5];norm=np.linalg.norm(delta,axis=1)
    before_error=np.linalg.norm(before[:,9:11],axis=1);after_error=np.linalg.norm(after[:,9:11],axis=1)
    metres=prepared.read(prepared.ROOT/'events/ankara_2026/model_constraints/ankara_2026_metric_scale_assumed_v1.json')['converted_span_xy_m'][0]
    geometry=np.linalg.norm(common[:,:3]-before[:,:3],axis=1)*metres
    shots={}
    for shot in ma['shots']:
        s=slice(shot['row_begin'],shot['row_stop'])
        shots[shot['id']]=dict(rows=shot['row_stop']-shot['row_begin'],same_material_projection_change_px=summary(norm[s]),before_error_px=summary(before_error[s]),after_error_px=summary(after_error[s]))
    pts,inverse,counts=np.unique(z['pts'],return_inverse=True,return_counts=True)
    peaks=np.zeros(len(pts));np.maximum.at(peaks,inverse,norm)
    rms=np.sqrt(np.bincount(inverse,weights=norm*norm)/counts)
    top=np.argsort(peaks)[-20:][::-1]
    worst=np.argsort(norm)[-10:][::-1]
    indices_a=prepared.read(a.activity) if a.activity else None
    result=dict(schema='rinkspace.native_joint_projection_comparison.v1',identities=identities,
        baseline_state=str(a.baseline_state),candidate_state=str(a.candidate_state),rows=len(times),source_exposures=len(pts),
        source_uv_modified=False,optimizer_started=False,stations_refined=False,
        lever_y_bitwise_unchanged=bool(np.array_equal(va['rig'][:,19],vb['rig'][:,19])),
        lever_y_conditional_cm=(vb['rig'][:,19]*metres*100).tolist(),pupil2_before=va['rig'][:,23].tolist(),pupil2_after=vb['rig'][:,23].tolist(),
        pupil2_still_active=bool(indices_a and 'pupil2' in indices_a['camera_model']['active_camera_names']),
        motion_columns=None if indices_a is None else indices_a['motion_columns'],scene_columns=None if indices_a is None else indices_a['scene_columns'],
        unique_camera_columns=None if indices_a is None else indices_a['unique_camera_columns'],
        same_material_projection_change_px=summary(norm),conditional_geometry_displacement_m=summary(geometry),
        baseline_cost=float(.5*np.dot(before_error,before_error)),candidate_cost=float(.5*np.dot(after_error,after_error)),
        before_error_px=summary(before_error),after_error_px=summary(after_error),shots=shots,
        changed_numeric_entries={k:int(np.count_nonzero(va[k]!=vb[k])) for k in ('rig','controls','scene_values','stations')},
        top_changed_frames=[dict(pts=int(pts[i]),source_seconds=float(pts[i]/12800),maximum_projection_change_px=float(peaks[i]),rms_projection_change_px=float(rms[i])) for i in top],
        worst_material_samples=[dict(pts=int(z['pts'][i]),path_id=paths[i],delta_uv=delta[i].tolist(),before_uv=before[i,3:5].tolist(),after_uv=common[i,3:5].tolist()) for i in worst],
        method='Full production native scene and implicit row exposure. Identical baseline source material stations are used in both model probes; candidate saved-station residuals are evaluated separately.',
        limitations=['Projection-change support is the current active source material locations, not every unobserved model point.',
            'Metre displacements use the current conditional scale, not a surveyed metric calibration.',
            'This warm-start fixed-at-current-value comparison does not isolate whether lever_y freedom helped the earlier optimizer reach this branch.'])
    for path,sha in identities.items():
        if digest(path)!=sha:raise ValueError('Comparison input changed: '+path)
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out.with_suffix('.npz'),source_pts=pts,frame_maximum_change_px=peaks,frame_rms_change_px=rms,common_material_delta_uv=delta)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+chr(10),encoding='utf-8')
    print(str(out),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline-run','candidate-run','baseline-state','candidate-state','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--activity',type=Path)
    main(p.parse_args())
