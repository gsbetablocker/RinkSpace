"""Resume the saved continuous spin with source-supported root positioning.

Only existing case/reference/observations and frozen hands are consumed. The
support placement is an initialization from the same fit pixels and nominal
ice, not extra independent 3D evidence or a fixed rotation centre.
"""
from pathlib import Path
import argparse,json,time,copy,os
import numpy as np
from scipy.interpolate import PchipInterpolator
from rinkspace.storage import AssetStore
from .skate_physics import fk_state
from .support_root_placement import runner,smoothstep
from .skate_template import profile_height
from .action_clip_input import camera25_from_snapshot
from .camera_bridge import rays
from .rebuild_prepare import load,digest
from .fullbody_round import write_json
from .skate_run_policy import RunPolicy


def support_position_seed(problem,x,track_path,camera_info):
    """Translate the attached body from current blade pose and admitted pixels.

    Use the blade midpoint's actual height above the current lowest runner,
    rather than falsely equating the midpoint with the geometric minimum.
    No source review sample enters this initialization. The final solver still
    optimizes all existing root/body columns and finite soft ice factors.
    """
    track=load(track_path);use=np.asarray(track['optimization_use'],bool)
    if use.sum()<4:raise ValueError('Not enough existing fit support samples')
    pts=np.asarray(track['pts'])[use];uv=np.asarray(track['uv'])[use]
    ref=problem.reference;t=ref.times;source_pts=ref.p['pts']
    lower,upper=int(pts.min()),int(pts.max())
    source_seconds=source_pts/12800.
    curve=PchipInterpolator(pts.astype(float),uv,axis=0,extrapolate=False)
    pixels=curve(np.clip(source_pts,lower,upper))
    delta=problem.body.state.increments(x[17:],t)
    state=fk_state(ref,ref.at(t),delta)
    pars=ref.binding.packet()['parameters']
    xmid=.5*(-pars.runner_rear_extension_m+pars.sole_length_m-pars.runner_front_inset_m)
    q=np.array([xmid,0.,float(profile_height(xmid,pars))])
    middle=state['O'][:,1]+np.einsum('nij,j->ni',state['R'][:,1],q)
    lowest,_=runner(ref,state)
    height=middle[:,2]-lowest[:,2]
    frame=np.arange(len(source_pts));cam=problem.camera
    motion=cam.core.evaluate_motion_camera(cam.p['rig'],cam.p['controls'],
        float(cam.p['first_control_time']),float(cam.p['control_dt']),
        cam.p['motion_times'][frame]+float(cam.p['rig'][37])*(pixels[:,1]/1079.-.5))
    snapshots=camera25_from_snapshot(motion);target=[]
    for pixel,camera,z in zip(pixels,snapshots,height):
        direction,valid=rays(pixel[None],camera)
        C=camera[:3]*camera_info['metres_per_scene_unit']/ref.binding.ratio
        distance=(z-C[2])/direction[0,2]
        if not valid[0] or distance<=0:raise ValueError('No forward native support ray')
        target.append(C+distance*direction[0])
    target=np.asarray(target)
    a=max(82.,lower/12800.-.4);b=lower/12800.;c=upper/12800.;d=min(94.8,c+.4)
    weight=smoothstep((source_seconds-a)/(b-a))*smoothstep((d-source_seconds)/(d-c))
    shift=weight[:,None]*(target-middle)
    B=problem.body.state.basis(t)
    controls=np.linalg.lstsq(B,shift,rcond=1e-11)[0]
    result=x.copy();body=result[17:].reshape(problem.body.state.m,problem.body.state.d)
    for axis in range(3):
        column=int(np.flatnonzero(problem.body.state.active==axis)[0]);body[:,column]+=controls[:,axis]
    fit=B@controls
    mask=weight>.999
    return result,dict(method='current attached blade midpoint -> same-source native Brown/RS ray at midpoint-minus-runner height; root controls only',fit_anchor_count=int(use.sum()),source_fit_pts=pts.tolist(),support_foot=1,source_window_seconds=[b,c],nominal_ice_only=True,extra_residuals_added=False,fixed_world_centre=False,root_shift_median_m=np.median(np.linalg.norm(fit[mask]*ref.binding.ratio,axis=1)).item(),root_shift_max_m=np.max(np.linalg.norm(fit[mask]*ref.binding.ratio,axis=1)).item(),control_fit_rms_m=float(np.sqrt(np.mean((fit[mask]-shift[mask])**2))*ref.binding.ratio),root_control_basis_unchanged=True,articulation_and_hands_unchanged=True)


def run(project,base,previous,checkpoint,output,budget=600.,steps=8,*,parameter=None):
    from scripts import profile_contact_solver as adapter
    from .body_execution import BodyExecution,BodyExecutionConfig
    from .skate_integration import restore_checkpoint_layers
    from .exmr_integration import PROGRAMME,save_packet,CASE_FIELD
    project,base,previous,checkpoint,output=map(lambda p:Path(p).resolve(),(project,base,previous,checkpoint,output))
    from rinkspace.path_policy import guard_project_write
    parameter=Path(parameter).resolve() if parameter else project/PROGRAMME/'spin_support_foot_1104.npz'
    guard_project_write(parameter,project_root=project)
    if parameter.exists():raise FileExistsError('Preserve existing parameter; supply --parameter with a new path')
    if output.exists():raise FileExistsError('Preserve previous continuation output')
    output.mkdir(parents=True)
    print(json.dumps(dict(main_log=str(output/'progress.jsonl'),checkpoint=str(checkpoint),budget_seconds=budget,max_accepted=steps)),flush=True)
    args=adapter.parser().parse_args(['--kind','current_spin','--project',str(project),'--base-root',str(base),
        '--case-root',str(previous/'common/exmr_hand_nofoot'),'--state',str(checkpoint),'--initial',str(previous/'common/exmr_hand_nofoot/initial.npz'),
        '--evidence',str(project/PROGRAMME/'action_evidence_1103.json'),'--clip','spin_s05','--support',str(previous/'support_and_free_shoe_regions.json'),
        '--grasp-relations',str(previous/'grasp_relations.json'),'--grasp-geometry',str(previous/'palm_grip_geometry.json'),
        '--native-library',str(base.parent/'body_perf_1103/native_cpu/rinkspace_core.dll'),
        '--native-threads','4','--blas-threads','1','--no-optimize-shared','--backend','cpu','--workers','12',
        '--local-derivatives','analytic','--compatibility-backend','cuda_resident','--operation','solve',
        '--quality-baseline',str(base/'exmr_spin_compare_1104/native_protocol/common/gvhmr/profile.json'),
        '--output',str(output)])
    args.run_policy=previous/'common/exmr_hand_nofoot/fit/run_policy.json';args.jacobian_seconds_hint=15.
    # Bind the selected implementation before constructing fixed identities.
    # The generic camera DLL can be rebuilt by another task.
    os.environ['RINKSPACE_BODY_NATIVE_LIBRARY']=str(args.native_library)
    execution=BodyExecution(BodyExecutionConfig(workers=12,blas_threads=1,local_derivatives='analytic',compatibility_backend='cuda_resident'))
    started=time.perf_counter()
    try:
        with (output/'progress.jsonl').open('w',encoding='utf8',buffering=1) as log,execution.limits():
            setup=adapter.construct_problem(args,execution,log);problem,x=setup.problem,setup.x
            np.savez_compressed(output/'parent_checkpoint.npz',**load(checkpoint))
            located,placement=support_position_seed(problem,x,previous/'source_support_track.npz',setup.camera_info)
            write_json(output/'support_position_seed.json',placement)
            problem.binding.x=located[:17].copy();np.savez_compressed(output/'support_position_seed.npz',**problem.checkpoint_fields(located))
            # One actual seed evaluation supplies the production solver's J;
            # do not profile a matrix and then clear it before taking a step.
            tick=time.perf_counter();value=problem.evaluate(located,True);jsecs=time.perf_counter()-tick
            problem.jacobian_seconds_hint=jsecs
            write_json(output/'execution.json',dict(policy=execution.record(),jacobian_seconds=jsecs,preparation_seconds=time.perf_counter()-started,source_state=str(checkpoint),source_state_sha256=digest(checkpoint),placement=placement))
            remaining=budget-(time.perf_counter()-started)
            if remaining<2*jsecs:raise RuntimeError('Budget cannot fund actual progress')
            result,report=problem.solve(located,RunPolicy(wall_seconds=remaining,max_accepted=steps,max_outer=1))
            saved=load(problem.output/'candidate.npz')
            # The original reference is immutable and is archived byte-for-byte.
            initial=project/PROGRAMME/'spin_support_foot_initial_1104.npz'
            for path in (initial,parameter):
                if len(str(path).encode('utf-16-le'))//2>=260:raise ValueError('OneDrive path too long')
            source_initial=previous/'common/exmr_hand_nofoot/initial.npz'
            if initial.exists():assert initial.read_bytes()==source_initial.read_bytes()
            else:initial.write_bytes(source_initial.read_bytes())
            if parameter.exists():raise FileExistsError('Preserve existing exported candidate')
            saved.update(initial_reference=np.array(initial.relative_to(project).as_posix()),initial_reference_sha256=np.array(digest(initial)),
                original_gvhmr_pose_anchor=np.array(False),support_position_initialization_json=np.array(json.dumps(placement,sort_keys=True)))
            np.savez_compressed(parameter,**saved)
            # Export from that exact selected state, including actual hands.
            restore_checkpoint_layers(problem.reference,saved,include_primary=True)
            case=output/'common/exmr_hand_nofoot';case.mkdir(parents=True,exist_ok=True)
            final=save_packet(problem,setup.camera_info,setup.objective['supports'],setup.objective['support'],case,status='joint_fit',report=report)
            summary=dict(parameter=str(parameter),result=str(case/'fitted/result.npz'),accepted=problem.accepted,retained_step=report['selection']['retained_step'],
                budget_seconds=budget,wall_seconds=time.perf_counter()-started,source_context_seconds=[76,95],accuracy_accepted=False,placement=placement,stop=report['stop'])
            write_json(output/'run_summary.json',summary);print(json.dumps(summary),flush=True)
            return summary
    finally:execution.close()


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--base-root',type=Path,required=True);p.add_argument('--previous-output',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--parameter',type=Path);p.add_argument('--budget',type=float,default=600.);p.add_argument('--max-steps',type=int,default=8);a=p.parse_args(argv)
    return run(a.project,a.base_root,a.previous_output,a.checkpoint,a.output,a.budget,a.max_steps,parameter=a.parameter)

if __name__=='__main__':main()
