"""Reproducible S01 cached-data short-window round; no inference or repackaging."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,time
from datetime import datetime,timezone
from rinkspace.path_policy import guard_project_write
import numpy as np
from .rebuild_prepare import load,digest
from .fullbody_state import FrozenReference,WindowState,camera_binding,matrices
from .fullbody_observations import build_ledger
from .fullbody_fit import FullBodyProblem,verify_problem
from .rebuild_exposure_body import moving_projection
from .footwear_geometry import extra_rotations


WINDOWS=((24.50,27.00),(32.20,33.80),(35.20,36.80),(37.40,39.00),(40.00,41.50))


def write_json(path,value):
    path=guard_project_write(path);text=json.dumps(value,ensure_ascii=False,indent=2)
    if path.is_file() and path.read_text(encoding='utf8')==text:return
    pending=guard_project_write(path.with_name(path.name+'.pending'))
    pending.write_text(text,encoding='utf8');pending.replace(path)



class NeedReviewError(RuntimeError):
    """Existing work is preserved until its actual inputs can be identified."""


def array_digest(named_arrays):
    h=hashlib.sha256()
    for name,array in named_arrays:
        a=np.ascontiguousarray(array)
        header=json.dumps(dict(name=name,dtype=a.dtype.str,shape=a.shape),sort_keys=True,separators=(',',':')).encode('utf8')
        h.update(len(header).to_bytes(8,'little'));h.update(header);h.update(a.tobytes(order='C'))
    return h.hexdigest()


def selected_observations_sha256(observations,seconds):
    """Hash only the window's actual selection and adopted source measurements.

    Unused candidate UV/scores and unrelated windows do not invalidate a fit.
    Numeric dtype normalization makes equal mathematical inputs compare equally.
    """
    obs=load(observations) if isinstance(observations,(str,Path)) else observations
    np.testing.assert_array_equal(obs['time_base'],[1,12800])
    pts=np.asarray(obs['pts'],dtype='<i8');a,b=map(float,seconds)
    frames=(pts/12800>=a-1e-9)&(pts/12800<=b+1e-9)
    use=np.asarray(obs['use'][frames],bool)
    return array_digest([('pts',pts[frames]),('use',use.astype('u1')),
        ('uv',np.asarray(obs['uv'][frames][use],dtype='<f8')),
        ('sigma_px',np.asarray(obs['sigma_px'][frames][use],dtype='<f8')),
        ('source_id',np.asarray(obs['source_id'][frames][use],dtype='<i8')),
        ('semantic_index',np.asarray(obs['semantic_index'][frames][use],dtype='<i8')),
        ('lower_body_inferred',np.asarray(obs['lower_body_inferred'][frames],dtype='u1'))])


def camera_identity(info):
    keys=('state_sha256','native_library_sha256','shot','metres_per_scene_unit',
          'solver_metres_per_scene_unit','length_export_ratio','source_time_base')
    return {key:info[key] for key in keys}


def window_input_signature(reference,camera,info,output,mode,seconds):
    """Public input signature used by new fits and explicit old-window review."""
    from .fullbody_fit import footwear_rows
    obs=load(Path(output)/'observations.npz')
    np.testing.assert_array_equal(obs['pts'],reference.p['pts'])
    ws=WindowState(reference,seconds,mode)
    frames=np.flatnonzero((obs['pts']/12800>=seconds[0]-1e-9)&(obs['pts']/12800<=seconds[1]+1e-9))
    _,equipment_sources=footwear_rows(reference.root,frames,reference.p['pts'])
    camera_arrays=[(key,np.asarray(camera.p[key],dtype='<f8')) for key in
                   ('rig','controls','first_control_time','control_dt','motion_times')]
    return dict(schema='rinkspace.fullbody-window-input.v1',mode=mode,window_seconds=list(map(float,seconds)),
        source_pts=obs['pts'][frames].tolist(),selected_observations_sha256=selected_observations_sha256(obs,seconds),
        reference_hashes=reference.hashes,camera=camera_identity(info),camera_numeric_sha256=array_digest(camera_arrays),
        equipment_sources=equipment_sources,active_state_columns=ws.active.tolist(),knots=ws.knots.tolist(),
        taper_seconds=ws.taper,objective=dict(prior_scales=dict(root_displacement_m=.35,angular_displacement_deg=25,
            root_velocity_m_s=3,angular_velocity_deg_s=400,root_acceleration_m_s2=15,angular_acceleration_deg_s2=800),
            robust_body='2D isotropic pseudo-Huber, c=2 after engineering sigma',ftol=1e-7,xtol=1e-7,gtol=1e-4))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def check_saved_camera(output,info,required=False):
    path=Path(output)/'camera_binding.json'
    if not path.is_file():
        if required:raise NeedReviewError('need review: missing prepared camera binding: '+str(path))
        return
    if camera_identity(read_json(path))!=camera_identity(info):
        has_results=any(Path(output).glob('*/w*/result.npz')) or any(Path(output).glob('*/result.npz'))
        if required or has_results:
            raise NeedReviewError('need review: saved camera/scale/native identity differs; existing results preserved. Use a separate output directory: '+str(output))


def check_prepared(reference,info,output):
    output=Path(output);check_saved_camera(output,info,required=True)
    ledger=read_json(output/'ledger.json')
    if ledger.get('observation_array_sha256')!=digest(output/'observations.npz'):
        raise NeedReviewError('need review: observations.npz does not match the current ledger; prepare the selected decisions first')
    obs=load(output/'observations.npz')
    np.testing.assert_array_equal(obs['pts'],reference.p['pts'])
    np.testing.assert_array_equal(obs['time_base'],[1,12800])


def validate_window(reference,camera,info,output,mode,index,record_recovery=True):
    """Return completion status; refuse stale/anonymous/partial work, never replace it."""
    output=Path(output);dest=output/mode/f'w{index+1}'
    result_path=dest/'result.npz';config_path=dest/'config.json'
    if not result_path.is_file():
        if any((dest/name).exists() for name in ('config.json','last_accepted.npz','accepted.pending.npz','result.json')):
            raise NeedReviewError('need review: partial window retained; cross-process optimizer resume is unavailable: '+str(dest))
        return False
    if not config_path.is_file() or not (dest/'result.json').is_file():
        raise NeedReviewError('need review: completed window is missing its config/result report: '+str(dest))
    signature=window_input_signature(reference,camera,info,output,mode,WINDOWS[index]);config=read_json(config_path)
    expected={key:signature[key] for key in ('mode','window_seconds','source_pts','active_state_columns','knots','taper_seconds','reference_hashes','equipment_sources')}
    expected.update(signature['objective'])
    mismatched=[key for key,value in expected.items() if config.get(key)!=value]
    if mismatched:
        raise NeedReviewError('need review: stale window config '+str(dest)+'; mismatched '+', '.join(mismatched))
    packet=load(result_path)
    np.testing.assert_array_equal(packet['pts'],signature['source_pts'])
    np.testing.assert_allclose(packet['knots'],signature['knots'],rtol=0,atol=0)
    np.testing.assert_array_equal(packet['active'],signature['active_state_columns'])
    config_sha=digest(config_path);result_sha=digest(result_path)
    signed_path=dest/'input_signature.json';review_path=dest/'reuse_validation.json'
    if signed_path.is_file() or review_path.is_file():
        record=read_json(signed_path if signed_path.is_file() else review_path)
        schema='rinkspace.fullbody-window-completion.v1' if signed_path.is_file() else 'rinkspace.fullbody-window-reuse.v1'
        if record.get('schema')!=schema or record.get('signature')!=signature or record.get('config_sha256')!=config_sha or record.get('result_sha256')!=result_sha:
            raise NeedReviewError('need review: window signature or result/config SHA mismatch: '+str(dest))
        if not signed_path.is_file():
            verification=record.get('verification',{})
            if verification.get('selected_observations_equal') is not True or not verification.get('method'):
                raise NeedReviewError('need review: missing explicit old/new selected-observation equivalence proof: '+str(review_path))
        return True
    # A complete current observation-file hash is also direct input evidence.
    # Old windows without this evidence require an explicit equivalence review.
    if config.get('observation_array_sha256')==digest(output/'observations.npz'):
        if record_recovery:
            write_json(signed_path,dict(schema='rinkspace.fullbody-window-completion.v1',signature=signature,
                config_sha256=config_sha,result_sha256=result_sha,recovered_from='exact current observation_array_sha256 in saved solver config'))
        return True
    raise NeedReviewError('need review: old window has no selected-input signature or provably identical observation hash; preserve it and supply reuse_validation.json after exact-array comparison: '+str(dest))


def save_npz_if_changed(path,**arrays):
    path=guard_project_write(path)
    if path.is_file():
        old=load(path)
        if set(old)==set(arrays) and all(np.array_equal(old[key],value) for key,value in arrays.items()):return
    pending=guard_project_write(path.with_name(path.stem+'.pending.npz'))
    np.savez_compressed(pending,**arrays);pending.replace(path)


def log_event(output_directory,event,**values):
    path=guard_project_write(Path(output_directory)/'run_progress.jsonl')
    with path.open('a',encoding='utf8',buffering=1) as stream:
        stream.write(json.dumps(dict(event=event,utc=datetime.now(timezone.utc).isoformat(),**values),ensure_ascii=False)+'\n')


def preflight_windows(reference,camera,info,output,modes,window=None):
    for mode in modes:
        for i in range(len(WINDOWS)):
            if window is None or window==i+1:validate_window(reference,camera,info,output,mode,i)


def fit_windows(reference,camera,info,output,mode,window=None):
    for i,seconds in enumerate(WINDOWS):
        if window is not None and window!=i+1:continue
        dest=output/mode/f'w{i+1}'
        if validate_window(reference,camera,info,output,mode,i):
            print('Preserving verified completed window '+str(dest),flush=True)
            log_event(output,'window_reused',mode=mode,window=seconds,path=str(dest));continue
        signature=window_input_signature(reference,camera,info,output,mode,seconds)
        with (output/'fit_progress.jsonl').open('a',encoding='utf8',buffering=1) as log:
            log.write(json.dumps(dict(event='window_start',mode=mode,window=seconds,camera=info['state_sha256'],output=str(dest)))+'\n')
            log_event(output,'window_start',mode=mode,window=seconds,path=str(dest),fit_log=str(output/'fit_progress.jsonl'))
            problem=FullBodyProblem(reference,camera,output/'observations.npz',seconds,mode,log)
            _,report=problem.solve(dest)
            write_json(dest/'input_signature.json',dict(schema='rinkspace.fullbody-window-completion.v1',signature=signature,
                config_sha256=digest(dest/'config.json'),result_sha256=digest(dest/'result.npz')))
            log_event(output,'window_complete',mode=mode,window=seconds,report=report)
            print(json.dumps(report),flush=True)


def verify_round(reference,camera,output):
    result=verify_reference(reference,output)
    log_event(output,'verify_reference_complete',result=result)
    problem=FullBodyProblem(reference,camera,output/'observations.npz',WINDOWS[0],'full')
    derivative=verify_problem(problem,output/'derivative_checks.json')
    log_event(output,'verify_problem_complete',mode='full',window=WINDOWS[0],result=derivative)
    print(json.dumps(dict(reference=result,derivative=derivative)),flush=True)


def run_subprocess(command,project,output,label):
    env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join((str(project/'src'),str(project)))
    env.setdefault('OPENBLAS_NUM_THREADS','2');env.setdefault('OMP_NUM_THREADS','2')
    env['PYTHONUNBUFFERED']='1';env['PYTHONUTF8']='1'
    log_event(output,'subprocess_start',label=label,command=list(map(str,command)))
    options=dict(cwd=project,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf8',errors='replace')
    if os.name=='nt':options['creationflags']=subprocess.CREATE_NO_WINDOW
    with subprocess.Popen(list(map(str,command)),**options) as child:
        for line in child.stdout:
            print(line,end='',flush=True);log_event(output,'subprocess_output',label=label,text=line.rstrip('\r\n'))
        status=child.wait()
    log_event(output,'subprocess_complete',label=label,returncode=status)
    if status:raise subprocess.CalledProcessError(status,command)


def prepare_review_and_cache(args,output):
    from rinkspace.storage import AssetStore
    project=Path(args.project).resolve();assets=AssetStore(project)
    if not args.skip_review:
        python=Path(args.review_python) if args.review_python else assets.resolve(project/'.venv-body/Scripts/python.exe',must_exist=True)
        run_subprocess([python,'-B','-X','utf8','-m','calibration.program_tracking.body_tracking.fullbody_review',
            '--root',args.root,'--output',output],project,output,'development_report')
    if not args.skip_cache:
        config_path=Path(args.viewer_config or project/'events/ankara_2026/programmes/okada_mei_sp/viewer3d_fullbody_1102.json').resolve()
        config=read_json(config_path);root=Path(args.root).resolve()
        if (assets.data/config['source_run']).resolve()!=root or (assets.data/config.get('scene_root',config['source_run'])).resolve()!=output:
            raise NeedReviewError('need review: viewer config source/scene does not refer to this run; provide --viewer-config: '+str(config_path))
        metric_path=(project/config.get('metric_binding','')).resolve()
        camera_path=(assets.data/config['camera_snapshot']).resolve() if 'camera_snapshot' in config else output/'camera_snapshot.npz'
        if metric_path!=output/'metric_binding.json' or camera_path!=output/'camera_snapshot.npz':
            raise NeedReviewError('need review: viewer metric/camera snapshot does not bind this output: '+str(config_path))
        variants=dict(ref1102='reference',obs1102='legacy',full1102='full')
        by_id={v['id']:v for v in config['variants']}
        for variant,mode in variants.items():
            if variant not in by_id or by_id[variant].get('kind')!='fullbody' or (root/by_id[variant]['result']).resolve()!=output/mode/'result.npz':
                raise NeedReviewError('need review: viewer variant does not bind the exported result: '+variant)
        python=Path(args.cache_python) if args.cache_python else assets.data/'runtime_tools/results3d/Scripts/python.exe'
        if not python.is_file():raise FileNotFoundError(python)
        for variant in variants:
            run_subprocess([python,'-B','-X','utf8','-m','rinkspace.viewer3d.prepare','--project',project,
                '--data-root',assets.data,'--config',config_path,'--variant',variant],project,output,'viewer_cache_'+variant)


def prepare(reference,project,output,camera,info,state):
    output=guard_project_write(output);output.mkdir(parents=True,exist_ok=True)
    decisions=json.loads((output/'source_review/decisions.json').read_text(encoding='utf8'))
    ledger=build_ledger(reference.root,output,decisions)
    write_json(output/'camera_binding.json',info)
    write_json(output/'metric_binding.json',dict(schema='rinkspace.native-body-uniform-metric-binding.v1',
        metres_per_scene_unit=info['metres_per_scene_unit'],snapshot=info['state_path'],
        source_span_xy_scene_units=(2*state['board'][:2]).tolist(),
        policy='equal_weight_squared_dimension_error',uniform_length_ratio=info['length_export_ratio'],
        source_unit=info['solver_metres_per_scene_unit'],unit_change_is_fit_improvement=False))
    save_npz_if_changed(output/'camera_snapshot.npz',**state)
    # Exact current native scene, reused by the existing 3D player.
    from rinkspace.domain.parameters import SceneParameterModel,ParameterState
    from rinkspace.session.gpu_geometry import NativeSceneGPU
    from rinkspace.review.scene import sample_native_review_scene
    manifest=json.loads((Path(info['prepared'])/'joint_manifest.json').read_text(encoding='utf8'))
    scene_path=Path(manifest['scene_path']);raw=json.loads(scene_path.read_text(encoding='utf8'))
    model=SceneParameterModel(raw,scene_path,optimize_scene=True)
    parameters=ParameterState(model.initial_state.layout,state['scene_values'])
    native=NativeSceneGPU(scene_path,camera.core,parameter_state=parameters)
    mapping=json.loads((Path(info['prepared'])/'mapping.json').read_text(encoding='utf8'))
    scene=sample_native_review_scene(raw,camera.core,scene_path,parameter_state=parameters,
        observed_path_ids=set(mapping['path_ids']),native=native)
    save_npz_if_changed(output/'review_scene.npz',xyz=scene['xyz'])
    write_json(output/'review_scene.json',dict(cuts=scene['cuts'],inventory=scene['inventory'],
        unlocalized=scene['missing_entities'],camera_state_sha256=info['state_sha256']))
    return ledger


def verify_reference(reference,output):
    frames=np.array([146,207,695,932]);t=reference.times[frames];r=reference.at(t);s=reference.world(r)
    old=reference.body.forward(reference.p['pose_axis_angle'][frames],reference.b9(t),
        additional_rotations=extra_rotations(reference.legs(t)),with_frames=True)
    new=reference.body.forward_rotations(r['local_R'],with_frames=True)
    errors=[float(np.max(abs(a-b))) for a,b in zip(old,new)]
    zero=float(np.max(abs(s['coco']-reference.joint['COCO_world_mid'][frames])))
    root=float(np.max(abs(s['P']-reference.joint['P_world_mid'][frames])))
    if max([*errors,zero,root])>1e-10:raise ValueError('Frozen reference does not replay J1')
    delta=np.zeros((len(t),69));delta[:,6+(16-1)*3+1]=.1;delta[:,6+(18-1)*3]=.07
    upper=reference.world(r,delta)
    pelvis_error=float(np.max(abs(upper['native'][:,0]-s['native'][:,0])))
    gear_error=float(np.max(abs(upper['gear']-s['gear'])))
    wrist_change=float(np.linalg.norm(upper['native'][:,20]-s['native'][:,20],axis=1).max())
    if max(pelvis_error,gear_error)>1e-10 or wrist_change<.005:raise ValueError('Upper-limb isolation failed')
    delta[:]=0;delta[:,:3]=[.03,-.02,.01];delta[:,3:6]=[.02,-.03,.01]
    moved=reference.world(r,delta);world_rotation=moved['Q']@np.swapaxes(s['Q'],-1,-2)
    expected=moved['P'][:,None]+np.einsum('nij,nkj->nki',world_rotation,s['world']-s['P'][:,None])
    rigid=float(np.max(abs(moved['world']-expected)))
    if rigid>1e-10:raise ValueError('Root does not act on all geometry together')
    result=dict(nonzero_matrix_entry_max_abs=errors,zero_reference_coco_max_abs=zero,zero_root_max_abs=root,
        upper_change_wrist_m=wrist_change,upper_gear_max_abs=gear_error,upper_pelvis_max_abs=pelvis_error,
        root_rigid_all_geometry_max_abs=rigid,verified_pts=reference.p['pts'][frames].tolist(),accuracy_claim=False)
    write_json(output/'state_checks.json',result);return result


def export_mode(reference,camera,info,output,mode):
    n=len(reference.times);ratio=info['length_export_ratio'];pts=reference.p['pts']
    states=[];configs=[];reports=[]
    if mode!='reference':
        for i,seconds in enumerate(WINDOWS):
            validate_window(reference,camera,info,output,mode,i)
            path=output/mode/f'w{i+1}'/'result.npz'
            if not path.is_file():raise ValueError(f'Window not optimized: {path}')
            z=load(path);ws=WindowState(reference,seconds,mode)
            np.testing.assert_allclose(ws.knots,z['knots']);np.testing.assert_array_equal(ws.active,z['active'])
            states.append((ws,z['x']))
            configs.append(json.loads((path.parent/'config.json').read_text(encoding='utf8')))
            reports.append(json.loads((path.parent/'result.json').read_text(encoding='utf8')))
    ledger=read_json(output/'ledger.json')
    corrected_pts=sorted({int(c['pts']) for c in ledger.get('changes',[]) if str(c.get('adoption','')).startswith('selected')})
    names=('P','Q','local_R','native','coco','gear');allfields={k:[] for k in names}
    body_uv=[];gear_uv=[];row_world=[];row_time=[];root_uv=[];inside=[]
    for start in range(0,n,64):
        frames=np.arange(start,min(start+64,n));t=reference.times[frames]
        times=(t[:,None]+np.linspace(-.5,.5,5)*float(camera.p['rig'][37])).ravel()
        delta=np.zeros((len(times),69))
        for ws,x in states:delta+=ws.increments(x,times)
        s=reference.world(reference.at(times),delta)
        for key in names:allfields[key].append(s[key].reshape(len(frames),5,*s[key].shape[1:])[:,2])
        W=s['world'].reshape(len(frames),5,113,3)
        prediction=moving_projection(camera,W,frames)
        body_uv.append(prediction['uv'][:,96:]);gear_uv.append(prediction['uv'][:,:96].reshape(-1,2,48,2))
        row_world.append(prediction['world']);inside.append(prediction['inside_exposure_rows'])
        row_time.append(pts[frames,None]/12800+float(camera.p['rig'][37])*(np.clip(prediction['uv'][...,1],0,1079)/1079-.5))
        root_uv.append(moving_projection(camera,s['P'].reshape(len(frames),5,1,3),frames)['uv'][:,0])
    arrays={k:np.concatenate(v) for k,v in allfields.items()};P=arrays['P']*ratio;G=P.copy();G[:,2]=0
    window_mask=np.zeros(n,bool)
    for a,b in WINDOWS:window_mask|=(pts/12800>=a-1e-9)&(pts/12800<=b+1e-9)
    if mode=='reference':window_mask[:]=False
    packet=dict(state_schema_version=np.array('rinkspace.unique-body.v1'),pts=pts,source_pts=pts,time_base=np.array([1,12800]),
        window_mask=window_mask,P_world_mid=P,Q_world=arrays['Q'],Q_world_mid=arrays['Q'],local_R_mid=arrays['local_R'],
        native22_world_mid=arrays['native']*ratio,COCO_world_mid=arrays['coco']*ratio,
        boot_world_mid=arrays['gear'][:,:,:14]*ratio,blade_world_mid=arrays['gear'][:,:,14:]*ratio,
        gear_world_mid=arrays['gear']*ratio,G_world_mid=G,COCO_uv=np.concatenate(body_uv),equipment_uv=np.concatenate(gear_uv),
        P_uv_dynamic=np.concatenate(root_uv),world_at_row=np.concatenate(row_world)*ratio,source_row_time=np.concatenate(row_time),
        inside_exposure_rows=np.concatenate(inside),metres_per_scene_unit=np.array(info['metres_per_scene_unit']),
        solver_metres_per_scene_unit=np.array(info['solver_metres_per_scene_unit']),body_length_ratio=np.array(ratio),
        coordinate_convention=np.array('world Z up, metres; P anatomical pelvis, G its vertical projection'),
        shared_geometry=reference.geometry,shared_geometry_unit=np.array('frozen legacy parameterization, geometry output uniformly reexpressed'),
        source_lower_body_inferred=reference.p['lower_body_inferred'],contact_state=np.full((n,2),'unknown'),
        reference_id=np.array('J1_s2'),reference_hashes_json=np.array(json.dumps(reference.hashes,sort_keys=True)),
        camera_binding_json=np.array(json.dumps(info,sort_keys=True)),observation_ledger_id=np.array(digest(output/'ledger.json')),
        solver_config_json=np.array(json.dumps(configs,sort_keys=True)),
        active_blocks_json=np.array(json.dumps([c['active_state_columns'] for c in configs])),
        stopping_reason_json=np.array(json.dumps([r['stopping_reason'] for r in reports])),
        costs_json=np.array(json.dumps([r['final_parts'] for r in reports],sort_keys=True)),
        association_corrected_pts=np.asarray(corrected_pts,dtype=np.int64),user_accepted=np.array(False))
    dest=guard_project_write(output/mode);dest.mkdir(exist_ok=True);save_npz_if_changed(dest/'result.npz',**packet)
    # Reload exact written representation and compare the authoritative arrays.
    again=load(dest/'result.npz')
    for key in ('P_world_mid','Q_world','local_R_mid','COCO_world_mid','gear_world_mid','source_pts'):
        np.testing.assert_array_equal(again[key],packet[key])
    write_json(dest/'export.json',dict(schema='rinkspace.fullbody-export.v1',mode=mode,frames=n,
        optimized_frames=int(window_mask.sum()),outside_window='unchanged J1_s2 reference, not new fit',
        result_sha256=digest(dest/'result.npz'),length_ratio=ratio,source_time_base=[1,12800],
        continuous_state=dict(reference_hashes=reference.hashes,window_files=[str(output/mode/f'w{i+1}'/'result.npz') for i in range(len(states))]),
        common_midpoint_and_row_time_separate=True,reload_equal=True))
    return packet


def main(argv=None,log_announced=False):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['run','prepare','verify','fit','export'])
    p.add_argument('--project',default=str(Path(__file__).resolve().parents[3]));p.add_argument('--root',required=True)
    p.add_argument('--output',required=True);p.add_argument('--mode',choices=['legacy','full','reference'],default='full')
    p.add_argument('--window',type=int,choices=range(1,len(WINDOWS)+1))
    p.add_argument('--skip-cache',action='store_true',help='run: omit the three existing viewer cache preparations')
    p.add_argument('--skip-review',action='store_true',help='run: omit metrics and static report generation; numerical verification still runs')
    p.add_argument('--review-python',help='existing Python containing matplotlib/PyAV; defaults to registered body_cuda')
    p.add_argument('--cache-python',help='existing results3d Python')
    p.add_argument('--viewer-config',help='existing viewer config binding this root/output and ref1102/obs1102/full1102')
    args=p.parse_args(argv)
    if args.action=='run' and args.window is not None:p.error('run reproduces both modes and all five windows; --window is for fit only')
    if args.action=='fit' and args.mode=='reference':p.error('reference is exported without optimization; fit requires legacy or full')
    output=Path(args.output).resolve();args.root=str(Path(args.root).resolve());args.project=str(Path(args.project).resolve())
    # Check final names before any OneDrive mkdir or temporary-file creation.
    for relative in ('run_progress.jsonl','fit_progress.jsonl','camera_pose.pending.npz','review_scene.pending.npz',
                     'legacy/w5/input_signature.json.pending','full/w5/reuse_validation.json.pending'):
        guard_project_write(output/relative)
    output.mkdir(parents=True,exist_ok=True)
    if not log_announced:print(json.dumps(dict(main_log=str(output/'run_progress.jsonl'),fit_log=str(output/'fit_progress.jsonl'))),flush=True)
    log_event(output,'start',action=args.action,root=args.root,output=str(output),project=args.project)
    try:
        ref=FrozenReference(args.root);camera,info,state,poses=camera_binding(args.project,ref)
        check_saved_camera(output,info,required=args.action not in ('run','prepare'))
        if args.action in ('prepare','run'):
            prepare(ref,Path(args.project),output,camera,info,state)
            log_event(output,'prepare_complete',camera=info['state_sha256'],ledger_sha256=digest(output/'ledger.json'))
        check_prepared(ref,info,output)
        save_npz_if_changed(output/'camera_pose.npz',pts=ref.p['pts'],centre=poses['centre'],
            rotation=poses['rotation'],motion_times=camera.p['motion_times'],metres_per_scene_unit=np.array(info['metres_per_scene_unit']))
        if args.action=='prepare':
            print(json.dumps(verify_reference(ref,output)),flush=True)
        elif args.action=='verify':verify_round(ref,camera,output)
        elif args.action=='fit':
            preflight_windows(ref,camera,info,output,[args.mode],args.window)
            fit_windows(ref,camera,info,output,args.mode,args.window)
        elif args.action=='export':
            export_mode(ref,camera,info,output,args.mode);print('Exported '+args.mode,flush=True)
        else:
            preflight_windows(ref,camera,info,output,['legacy','full'])
            verify_round(ref,camera,output)
            for mode in ('legacy','full'):fit_windows(ref,camera,info,output,mode)
            for mode in ('reference','legacy','full'):
                export_mode(ref,camera,info,output,mode)
                log_event(output,'export_complete',mode=mode,path=str(output/mode/'result.npz'))
            prepare_review_and_cache(args,output)
            from .fullbody_delivery import deliver
            delivery=deliver(Path(args.project),output)
            print(json.dumps({key:delivery[key] for key in ('binding','compact_parameters')},ensure_ascii=False),flush=True)
            log_event(output,'delivery_complete',binding=delivery['binding'])
        log_event(output,'complete',action=args.action)
    except BaseException as error:
        log_event(output,'failed',action=args.action,error=repr(error));raise


if __name__=='__main__':main()
