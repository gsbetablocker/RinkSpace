"""Native GPU raw-pixel joint fitting with independent per-shot trajectories."""
from pathlib import Path
import argparse,ctypes,json,time,threading,traceback,collections
import numpy as np
import prepare_native_joint_shots as prepared
import fit_native_s01_joint as single
from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import SceneParameterModel,ParameterState
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
from rinkspace.domain.camera_experiment import (CAMERA_NAMES, load_camera_experiment, build_camera_directions,
    validate_experiment_resume, guard_experiment_run_directory)
from rinkspace.session.gpu_geometry import NativeSceneGPU,scene_coordinate_charts

def safe(x):
    if isinstance(x,dict):return {k:safe(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [safe(v) for v in x]
    if isinstance(x,np.ndarray):return safe(x.tolist())
    if isinstance(x,(float,np.floating)) and not np.isfinite(x):return str(x)
    return x.item() if isinstance(x,np.generic) else x

def solve(args):
    root=prepared.RUN
    manifest=prepared.read(root/'joint_manifest.json')
    if getattr(args,'diagnostic_library',None):
        if not (args.diagnose_step or args.diagnose_stations):raise ValueError('A library comparison must be an explicit frozen-state diagnostic')
        manifest={**manifest,'native_library':str(args.diagnostic_library),'native_library_sha256':prepared.digest(args.diagnostic_library)}
    experiment_path=getattr(args,'camera_experiment',None)
    if experiment_path:
        if args.phase!='joint' or args.output is None:
            raise ValueError('A camera model experiment requires joint phase and an explicit independent output directory')
        guard_experiment_run_directory(args.output,experiment_path)
    out=args.output or Path(manifest.get('joint_output',root/'joint'));out.mkdir(parents=True,exist_ok=True)

    z=prepared.npz(root/'point_observations.npz');mapping=prepared.read(root/'mapping.json')
    if prepared.digest(root/'point_observations.npz')!=manifest['point_input_sha256']:raise ValueError('Imported input changed')
    if prepared.digest(manifest['scene_path'])!=manifest['scene_file_sha256']:raise ValueError('Native scene changed after import')
    snapshot=prepared.npz(args.resume or root/'initial_pixels.npz')
    if str(snapshot['point_input_sha256'])!=manifest['point_input_sha256'] or not np.array_equal(snapshot['source_rows'],z['source_rows']):raise ValueError('Resume belongs to another imported input')
    log=(out/'solve.jsonl').open('a',encoding='utf-8',buffering=1);concise=(out/'fit_progress.log').open('a',encoding='utf-8',buffering=1)
    start=time.monotonic();last_cost=None;last_errors=None
    def emit(event,**data):log.write(json.dumps(safe(dict(event=event,elapsed_s=time.monotonic()-start,**data)),ensure_ascii=False,allow_nan=False)+'\n')
    print('MAIN LOG: '+str(out/'solve.jsonl'),flush=True);print('CONCISE LOG: '+str(out/'fit_progress.log'),flush=True)
    concise.write('\nstep elapsed_s cost decrease joint_drop curve_drop max_px >20px status S01_cost S04_cost\n')
    timeline=ShotMotionTimeline(manifest['shots'],manifest['control_dt'],manifest['first_control_time'])
    times=timeline.row_times(z['pts']);timeline.assert_support_separation(z['pts'],manifest['max_readout'])
    if prepared.digest(manifest['native_library'])!=manifest['native_library_sha256']:raise ValueError('Selected reviewed solver library changed')
    core=NativeCore(manifest['native_library']);model=SceneParameterModel(prepared.read(manifest['scene_path']),Path(manifest['scene_path']),optimize_scene=True)
    prepared.validate_joint_snapshot(snapshot,model.initial_state.layout,mapping,manifest,z)
    state=ParameterState(model.initial_state.layout,snapshot['scene_values']);rig=snapshot['rig'];controls=snapshot['controls']
    experiment=None;camera_indices=tuple(range(31 if len(rig)==43 else 27))
    if experiment_path:
        rig,camera_indices,experiment=load_camera_experiment(experiment_path,rig)
    validate_experiment_resume(str(snapshot.get('camera_experiment_sha256','')),experiment,
        allow_change=getattr(args,'allow_experiment_change',False))
    frozen_scene=snapshot['scene_values'].copy() if experiment and experiment['scene_mode']=='frozen' else None
    native=NativeSceneGPU(manifest['scene_path'],core,parameter_state=state)
    camera={**prepared.read(single.prep.CAMERA),'rig':rig.tolist(),'first_control_time':timeline.first,'control_dt':timeline.dt}
    paths=[mapping['path_ids'][i] for i in z['path_indices']]
    all_motion,labels=timeline.directions()
    if args.phase=='s04_initialize':
        keep=[i for i,v in enumerate(labels) if v[0]=='s04'];all_motion=all_motion[:,:,keep];labels=[labels[i] for i in keep]
    nm=len(labels);camera_size=31 if len(rig)==43 else 27
    dependencies={b['id'] for b in state.blocks.values() if b['kind'] in ('board_shape','shared_geometry')}
    for p in set(paths):
        if '::' in p:dependencies.update(model.dependencies(p))
    active=state.active_columns(dependencies) if args.phase=='joint' and frozen_scene is None else np.zeros(0,int)
    if args.phase!='joint':camera_indices=()
    camcols=len(camera_indices);q=nm+camcols+len(active)
    directions=np.zeros((len(controls),3,q));directions[:,:,:nm]=.001*all_motion;del all_motion
    rd=np.zeros((camera_size,q));sd=np.zeros((len(state.values),q));charts=scene_coordinate_charts(state)
    rs=np.array([.001]*6+[.001]*3+[.00025]*3+[.00025]*3+[1.,1.,.001,1.,1.,.03,.03,.1,.1,.003,.003,.001])
    if camera_size==31:rs=np.r_[rs,.05,.1,.003,.003]
    rd,_=build_camera_directions(rig,camera_indices,rs,motion_columns=nm,scene_columns=len(active))
    for j,k in enumerate(active):
        b=next(b for b in state.blocks.values() if b['available'] and b['offset']<=k<b['offset']+b['size']);kind=b['kind']
        step=.01 if charts[k]>0 or kind=='print_size' else (.0002 if kind=='print_placement' and k-b['offset']<2 else (.001 if kind=='print_placement' else (.002 if kind.startswith('print_') else .001)))
        sd[k,nm+camcols+j]=step
    scale=prepared.read(single.prep.ROOT/'events/ankara_2026/model_constraints/ankara_2026_metric_scale_assumed_v1.json')['converted_span_xy_m'][0]
    lower,upper,envelope,range_path,range_fact=single.mount_configuration(scale,rig)
    # Numerical solve rows for the initial trajectory fit are only S04. S01's
    # complete state is retained byte for byte until the full joint phase.
    rows=np.arange(len(paths))
    if args.phase=='s04_initialize':
        shot=next(s for s in timeline.shots if s['id']=='s04');rows=rows[shot['row_begin']:shot['row_stop']]
        if args.initialization_surfaces=='board-advertisement':
            rows=rows[np.array([native.paths[paths[i]]['layer']!='ice' for i in rows])]
    elif args.initialization_surfaces!='all':raise ValueError('A subset is only allowed for explicit trajectory initialization')
    rp=[paths[i] for i in rows];local=np.full(len(rows),-1,np.int32);curve=np.flatnonzero(z['kind'][rows]==1);local[curve]=np.arange(len(curve))
    periodic=np.array([native.paths[rp[i]]['kind']==7 for i in curve]);lo=np.where(periodic,-np.inf,0.);hi=np.where(periodic,np.inf,1.)
    activity=dict(phase=args.phase,rows=len(rows),total_input_rows=len(paths),shared_camera_columns=camcols,
        camera_experiment=experiment,active_camera_chart=list(camera_indices),active_camera_names=[CAMERA_NAMES[i] for i in camera_indices],motion_columns=nm,motion_labels=labels,scene_columns=len(active),active_scene_blocks=sorted(dependencies) if len(active) else [],
        source_weight_fields=False,robust_loss=False,source_error_allowance=False,frame_windows=timeline.frame_windows(),initialization_surfaces=args.initialization_surfaces,
        continuous_across_edits=False,conditional_metres_per_L=scale,numeric_ranges_approved_by_user=False,
        physical_acceptance=False,small_constraint_dual_backend=manifest.get('small_constraint_dual_backend','gpu'),range_source=str(range_path),manifest=str(root/'joint_manifest.json'),
        point_input_sha256=manifest['point_input_sha256'],accepted_curve_search='full_bound_path',native_library=manifest['native_library'],
        native_library_sha256=manifest['native_library_sha256'],numeric_seed=str(args.resume or root/'initial_pixels.npz'))
    prepared.write(out/'activity.json',safe(activity));emit('compile_joint_problem',**activity)
    def diagnostics(evaluation,step,status,*,joint_drop=0.,curve_drop=0.):
        nonlocal last_cost,last_errors
        if isinstance(evaluation,dict):
            invalid=evaluation['status']!=0;errors=np.linalg.norm(evaluation['values'][:,9:11],axis=1);hot_uv=None
        else:
            errors=np.asarray(evaluation);invalid=~np.isfinite(errors)
            selected=np.flatnonzero((errors>20.)&~invalid)
            view=resident.evaluate_rows(selected.astype(np.int32)) if len(selected) else None
            if view is not None and view['status'].any():raise RuntimeError('A hotspot became invalid')
            hot_uv={} if view is None else dict(zip(view['rows'],view['values'][:,3:5]))
        good=~invalid & np.isfinite(errors);per={};hot=[]
        for s in timeline.shots:
            selection=(rows>=s['row_begin'])&(rows<s['row_stop']);valid=selection&good;es=errors[valid]
            per[s['id']]=dict(rows=int(selection.sum()),invalid=int(np.count_nonzero(selection&~good)),
                cost=float(.5*np.dot(es,es)),maximum=float(es.max()) if len(es) else None,above_20px=int(np.count_nonzero(es>20.)),
                median=float(np.median(es)) if len(es) else None,p95=float(np.percentile(es,95)) if len(es) else None)
        for j in np.flatnonzero(good&(errors>20.)):
            i=int(rows[j]);hot.append(dict(row=i,observation_id=str(z['observation_ids'][i]),source_row=int(z['source_rows'][i]),pts=int(z['pts'][i]),
                path_id=paths[i],observed_uv=z['uv'][i].tolist(),fitted_uv=(evaluation['values'][j,3:5] if hot_uv is None else hot_uv[j]).tolist(),error_px=float(errors[j])))
        hot.sort(key=lambda r:r['error_px'],reverse=True)
        cost=float(.5*np.dot(errors[good],errors[good]));result=dict(step=step,status=status,cost=cost,invalid=int((~good).sum()),
            joint_drop=float(joint_drop),curve_drop=float(curve_drop),maximum=float(errors[good].max()) if good.any() else None,above_20px=int(np.count_nonzero(good&(errors>20.))),shots=per)
        prepared.write(out/'current_quality.pending.json',safe(result));(out/'current_quality.pending.json').replace(out/'current_quality.json')
        prepared.write(out/'current_error_hotspots.pending.json',dict(**safe(result),observations=hot));(out/'current_error_hotspots.pending.json').replace(out/'current_error_hotspots.json')
        if good.all():
            delta=0. if last_cost is None else last_cost-cost
            concise.write(f"{step:5d} {time.monotonic()-start:10.2f} {cost:18.6f} {delta:13.6f} {joint_drop:13.6f} {curve_drop:13.6f} {errors.max():10.3f} {(errors>20).sum():7d} {status} {per['s01']['cost']:.6f} {per['s04']['cost']:.6f}\n")
            last_cost=cost;last_errors=errors
        else:concise.write(f"{step:5d} {time.monotonic()-start:.2f} invalid_rows={int((~good).sum())} {status}\n")
        emit('quality',**result);return result
    with native.resident(camera=camera,controls=controls,path_ids=rp,stations=snapshot['stations'][rows],pts=z['pts'][rows],observed=z['uv'][rows],
            scene_directions=sd,motion_times=times[rows]) as resident:
        resident.configure_camera_parameters(directions=rd,lower=lower,upper=upper,max_readout=manifest['max_readout'])
        evaluated=resident.evaluate('cuda');quality=diagnostics(evaluated,0,'seed')
        if args.evaluate_only:
            np.savez(out/'evaluation_pixels.npz',**evaluated,rows=rows);return
        if quality['invalid']:raise ValueError('Seed has invalid projections; initialize explicitly before solving, never drop these observations')
        resident.configure_mount_constraints(side='negative_y')
        resident.configure_camera_envelope(**envelope,frame_time_intervals=timeline.frame_windows())
        emit('linearization_compiled',metrics=resident.configure_pixel_linearization(directions=directions,local_ids=local,local_scales=np.full(len(curve),.001)))
        class MemoryStatus(ctypes.Structure):
            _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended')]
        memory=MemoryStatus();memory.length=ctypes.sizeof(memory)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):raise OSError('Cannot read available host RAM')
        emit('solver_attached',metrics=resident.configure_solver(lower=lo,upper=hi,device_budget_bytes=resident.creation['free_after_alloc_bytes'],
            host_budget_bytes=memory.avail_phys,damping_metric='jacobi_image',initial_lambda=args.initial_lambda))
        if getattr(args,'diagnose_stations',False):
            # Explicit frozen-state experiment. Only private output snapshots
            # change; shared camera, geometry and trajectories stay identical.
            records=[]
            selected_mode=getattr(args,'diagnose_station_mode',None)
            stages=[('before',None),(selected_mode,selected_mode)] if selected_mode else [('before',None),('local','local'),('global','global'),('local_after_global','local')]
            for name,mode in stages:
                refinement=None if mode is None else resident.refine_curve_stations(mode=mode)
                diagnostic=resident.stationarity_diagnostics(include_constraints=True)
                arrays={k:diagnostic.pop(k) for k in list(diagnostic) if isinstance(diagnostic[k],np.ndarray)}
                np.savez(out/f'{name}_constraints.npz',**arrays)
                v=resident.export_state();v.update(source_rows=z['source_rows'],point_input_sha256=np.array(manifest['point_input_sha256']))
                prepared.bind_joint_snapshot(v,state.layout,mapping,manifest)
                np.savez(out/f'{name}_pixels.npz',**v)
                errors=resident.evaluate_errors();np.save(out/f'{name}_errors.npy',errors)
                record=dict(stage=name,refinement=refinement,stationarity=diagnostic,quality=diagnostics(errors,0,name,curve_drop=0. if refinement is None else refinement['pixel_cost_decrease']))
                records.append(record);emit('station_search_experiment',**record)
                print(json.dumps(safe(record)),flush=True)
            prepared.write(out/'station_search_experiment.json',safe(records));log.close();concise.close();return
        if args.diagnose_step:
            diagnostic=resident.stationarity_diagnostics(include_constraints=True)
            arrays={k:diagnostic.pop(k) for k in list(diagnostic) if isinstance(diagnostic[k],np.ndarray)}
            np.savez(out/'before_constraints.npz',**arrays)
            prepared.write(out/'before_stationarity.json',safe(diagnostic))
            np.savez(out/'before_evaluation.npz',**resident.evaluate('cuda'))
        else:
            initial_refinement=resident.refine_curve_stations(mode='global')
            emit('initial_station_refinement',metrics=initial_refinement)
        def checkpoint():
            v=resident.export_state();full_stations=snapshot['stations'].copy();full_stations[rows]=v['stations'];v['stations']=full_stations
            v.update(source_rows=z['source_rows'],point_input_sha256=np.array(manifest['point_input_sha256']))
            prepared.bind_joint_snapshot(v,state.layout,mapping,manifest)
            if experiment:
                v['camera_experiment_sha256']=np.array(experiment['profile_sha256'])
                v['camera_experiment_json']=np.array(json.dumps(experiment['profile'],sort_keys=True,separators=(',',':')))
                if frozen_scene is not None and not np.array_equal(v['scene_values'],frozen_scene):
                    raise RuntimeError('Frozen scene changed during camera-only fitting')
                fixed=experiment['profile']['fixed_rig_values']
                from rinkspace.domain.camera_experiment import apply_camera_experiment
                expected,_,_=apply_camera_experiment(v['rig'],experiment['profile'])
                if not np.array_equal(expected,v['rig']):raise RuntimeError('An explicitly fixed optical coordinate changed')
            np.savez(out/'accepted_pixels_pending.npz',**v);(out/'accepted_pixels_pending.npz').replace(out/'accepted_pixels_latest.npz')
            return v
        def physical(v):
            poses=core.evaluate_motion_camera(v['rig'],v['controls'],timeline.first,timeline.dt,np.unique(times))
            motion={}
            for shot in timeline.shots:
                shot_camera={**camera,'reference_pts':shot['reference_pts'],'first_control_time':shot['first_control_time']}
                motion[shot['id']]=single.continuous_motion_diagnostic(core,v['controls'][shot['control_begin']:shot['control_stop']],shot_camera,v['rig'],z['pts'][shot['row_begin']:shot['row_stop']])
            return dict(continuous_motion_by_shot=motion,pivot_height_conditional_m=float(v['rig'][17]*scale),optical_height_conditional_m=[float(x) for x in np.percentile(poses['centre'][:,2]*scale,[0,100])],
                lever_camera_xyz_cm=(v['rig'][18:21]*scale*100).tolist(),fx_fy_ratio=float(poses['fx'][0]/poses['fy'][0]) if 'fx' in poses else None,
                principal_point=v['rig'][26:28].tolist(),log_focal_axis_ratio=float(v['rig'][28]),focal_axis_ratio=float(np.exp(v['rig'][28])),axis_spacing_cm=float(v['rig'][21]*scale*100),physical_acceptance=False)
        v=checkpoint();diagnostics(resident.evaluate('cuda'),0,'initial',curve_drop=0. if args.diagnose_step else initial_refinement['pixel_cost_decrease']);emit('physical_candidate',values=physical(v))
        def progress(record):
            summary=record['summary'];step=int(summary['trial_count'])
            if args.diagnose_step:
                np.savez(out/'step_system.npz',**resident.export_step_system(sparse=True),changes=resident.probe_step_changes())
                np.savez(out/'trial_pixels.npz',**resident.export_state(trial=True))
                prepared.write(out/'step_diagnostic.json',safe(record));emit('diagnostic_step',record=record);return False
            if summary['accepted']:
                # Search the whole bound curve after every accepted joint update.
                # Native invalidates its linearization on station changes; the
                # live controller then rebuilds and checks this exact objective.
                record['station_refinement']=resident.refine_curve_stations(mode='global')
                record['station_refinement_mode']='global'
                e=resident.evaluate_errors()
                joint_drop=summary['actual_decrease'];curve_drop=record['station_refinement']['pixel_cost_decrease']
                if not np.isclose(last_cost,summary['old_cost'],rtol=1e-10,atol=1e-6):
                    raise RuntimeError('Accepted joint step began from an unreported objective')
                ql=diagnostics(e,step,'accepted',joint_drop=joint_drop,curve_drop=curve_drop)
                record['combined_actual_decrease']=joint_drop+curve_drop
                if ql['invalid']:raise RuntimeError('Accepted state lost source observations')
                expected=summary['trial_cost']-record['station_refinement']['pixel_cost_decrease']
                if not np.isclose(ql['cost'],expected,rtol=1e-10,atol=1e-6):raise RuntimeError('Raw pixel objective disagrees with solver')
                v=checkpoint();record['physical']=physical(v)
            else:
                concise.write(f"{step:5d} {time.monotonic()-start:10.2f} {last_cost:18.6f} {0.:13.6f} {0.:13.6f} {0.:13.6f} {last_errors.max():10.3f} {(last_errors>20).sum():7d} rejected\n")
            emit('lm_step',record=record)
            return not (out/'stop_requested').exists()
        done=threading.Event()
        def watch():
            while not done.wait(.2):
                if (out/'stop_requested').exists():resident.request_stop();return
        watcher=threading.Thread(target=watch,daemon=True);watcher.start()
        try:
            passes=[]
            while True:
                result=resident.solve(progress=progress,numerical_progress=lambda r:emit('factorization_retry',record=r));passes.append(dict(result))
                if args.diagnose_step or result['stop_reason']=='stopped_by_caller':break
                full=resident.refine_curve_stations(mode='global');diagnostic=resident.stationarity_diagnostics()
                emit('final_full_curve_search',metrics=full,stationarity=diagnostic)
                diagnostics(resident.evaluate_errors(),sum(int(p['trials']) for p in passes),'curve_verify',curve_drop=full['pixel_cost_decrease'])
                checkpoint()
                if full['failed_rows']:raise RuntimeError('Final full station search failed')
                resolution=max(64*np.finfo(float).eps,result['tolerances']['relative_cost'])*max(1.,abs(diagnostic['objective']))
                if full['pixel_cost_decrease']>resolution and diagnostic['constraint_projected_gradient_inf']>1e-7:
                    emit('continue_after_full_curve_search',same_problem_and_live_optimizer=True);continue
                if diagnostic['constraint_projected_gradient_inf']>1e-7:result['numerically_converged']=False
                result['final_verified_stationarity']=diagnostic;result['final_global_station_verification']=full
                # The final full curve search can move stations even when its
                # cost gain is below the declared progress tolerance. Report
                # that exported state, not the preceding solve-pass scratch.
                result['final_cost']=diagnostic['objective']
                result['projected_gradient_inf']=result['constraint_projected_gradient_inf']=diagnostic['constraint_projected_gradient_inf']
                result['stationarity_diagnostics']=diagnostic
                break
            result['solve_passes']=passes
            result['total_trials']=sum(int(p['trials']) for p in passes)
            result['total_accepted']=sum(int(p['accepted']) for p in passes)
            result['total_rejected']=sum(int(p['rejected']) for p in passes)
            v=checkpoint();e=resident.evaluate('cuda');np.savez(out/'evaluation_pixels.npz',values=e['values'],status=e['status'],rows=rows)
            diagnostics(e,result['total_trials'],result['stop_reason']);prepared.write(out/'result.json',safe(result));emit('returned',result=result)
        except BaseException as error:
            checkpoint();emit('failure',error=str(error),traceback=traceback.format_exc());raise
        finally:done.set();watcher.join();log.close();concise.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['joint','s04_initialize'],default='joint');p.add_argument('--resume',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--evaluate-only',action='store_true');p.add_argument('--initial-lambda',type=float,default=0.)
    p.add_argument('--initialization-surfaces',choices=['all','board-advertisement'],default='all')
    p.add_argument('--diagnose-step',action='store_true',help='Export one native trial in this explicit output directory, then stop')
    p.add_argument('--diagnose-stations',action='store_true',help='Compare local and full native curve searches at a frozen camera/scene state, then stop')
    p.add_argument('--diagnose-station-mode',choices=['local','global'],help='With --diagnose-stations, compare one search from the untouched resume state')
    p.add_argument('--camera-experiment',type=Path,help='Explicit named camera mask and frozen/joint scene hypothesis')
    p.add_argument('--allow-experiment-change',action='store_true',help='Declare a different model using an experimental numerical seed in a new output directory')
    p.add_argument('--diagnostic-library',type=Path,help='Compare an explicitly built library without replacing the selected production library')
    args=p.parse_args()
    if args.diagnose_station_mode and not args.diagnose_stations:p.error('--diagnose-station-mode requires --diagnose-stations')
    solve(args)
