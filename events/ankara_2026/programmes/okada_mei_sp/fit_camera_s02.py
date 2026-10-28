"""S02 camera candidate using the existing native GPU solver and frozen peer geometry."""
from pathlib import Path
import argparse, collections, ctypes, gzip, hashlib, io, json, sys, threading, time, traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'src'))
from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import SceneParameterModel, ParameterState
from rinkspace.domain.motion_chart import UniformMotionChart
from rinkspace.session.gpu_geometry import NativeSceneGPU, complete_camera_rig
from rinkspace.vision_native import file_ref
from fit_native_s01_joint import continuous_motion_diagnostic
import points_s02 as source
OUT=source.OUT/'camera_fit'
PEER=source.DATA/'runs/native_joint_s01_s04_20261027'
read=source.read

def safe(x):
    if isinstance(x,dict): return {k:safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [safe(v) for v in x]
    if isinstance(x,np.ndarray): return safe(x.tolist())
    if isinstance(x,(float,np.floating)) and not np.isfinite(x): return str(x)
    return x.item() if isinstance(x,np.generic) else x

def write(p,x):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.pending')
    tmp.write_text(json.dumps(safe(x),ensure_ascii=False,indent=2)+'\n',encoding='utf8');tmp.replace(p)

def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/'manifest.json').exists(): return
    peer=read(PEER/'joint_manifest.json');scene_path=Path(peer['scene_path'])
    model=SceneParameterModel(read(scene_path),scene_path,optimize_scene=True)
    parent=Path(peer['joint_output'])/'accepted_pixels_latest.npz'
    blob=parent.read_bytes();sha=hashlib.sha256(blob).hexdigest()
    if sha!=file_ref(parent)['sha256']: raise ValueError('Peer snapshot changed while reading')
    state=dict(np.load(io.BytesIO(blob),allow_pickle=False))
    deps=json.loads(str(state['joint_numeric_dependencies_json']))
    if str(state['point_input_sha256'])!=peer['point_input_sha256']: raise ValueError('Peer state/input mismatch')
    if deps['layout_revision']!=model.initial_state.layout['layout_revision']: raise ValueError('Peer geometry layout mismatch')
    if deps['scene_revision']!=model.initial_state.layout['scene_content_revision']: raise ValueError('Peer geometry revision mismatch')
    if file_ref(scene_path)['sha256']!=peer['scene_file_sha256']: raise ValueError('Scene file changed')
    latest=ParameterState(model.initial_state.layout,state['scene_values'])
    native_scene=latest.apply_scene(read(scene_path))
    if not np.array_equal(native_scene['board'],state['board']) or not np.array_equal(native_scene['shared_layout'],state['layout']):
        raise ValueError('Peer exported geometry does not equal its shared state')
    peer_z=dict(np.load(PEER/'point_observations.npz',allow_pickle=False));peer_map=read(PEER/'mapping.json')
    if file_ref(PEER/'point_observations.npz')['sha256']!=peer['point_input_sha256']:raise ValueError('Peer input changed')
    counts=collections.Counter(peer_map['path_ids'][i] for i in peer_z['path_indices'])
    supported=set()
    for path in counts:
        if '::' in path:supported.update(model.dependencies(path))
    z=dict(np.load(source.OUT/'points/point_observations.npz',allow_pickle=False));mapping=read(source.OUT/'points/mapping.json')
    selected=[];coverage=[]
    for i,path in enumerate(mapping['path_ids']):
        rows=np.flatnonzero(z['path_indices']==i)
        if '::' in path:
            missing=sorted(set(model.dependencies(path))-supported)
            eligible=not missing
            basis='all_print_blocks_have_peer_observation_dependencies' if eligible else 'independent_print_blocks_without_peer_observations'
        else:
            missing=[];eligible=counts[path]>0
            basis='same_native_path_has_peer_observations' if eligible else 'no_direct_peer_path_observations_shared_geometry_may_be_partial'
        if eligible:selected.extend(rows.tolist())
        coverage.append(dict(path_id=path,rows=len(rows),peer_direct_rows=counts[path],camera_initialization=eligible,basis=basis,unobserved_blocks=missing))
    selected=np.sort(selected).astype(np.int64)
    np.save(OUT/'camera_rows.npy',selected)
    np.savez(OUT/'geometry_seed.npz',scene_values=state['scene_values'],board=state['board'],layout=state['layout'])
    camera=read(source.CAMERA);write(OUT/'camera_seed.json',camera)
    np.save(OUT/'controls_seed.npy',np.load(source.CAMERA.parent/camera['controls_ref']))
    write(OUT/'geometry_coverage.json',dict(paths=coverage,scope='camera_initialization_only',all_original_observations_preserved=True,
        peer_observations_are_constraints_not_visual_acceptance=True,missing_blocks_are_not_fitted_anchors=True,
        rows=len(selected),all_rows=len(z['pts']),frames=len(np.unique(z['pts'][selected]))))
    refs=[file_ref(scene_path),file_ref(peer['native_library']),file_ref(source.OUT/'points/point_observations.npz'),file_ref(source.CAMERA),file_ref(source.CAMERA.parent/camera['controls_ref'])]
    refs.extend(file_ref(p) for p in [OUT/'geometry_seed.npz',OUT/'camera_rows.npy',OUT/'camera_seed.json',OUT/'controls_seed.npy'])
    # Scene dependencies are protected, while the peer's live output is never written or locked.
    for ref,_ in model.assets:refs.append(file_ref(scene_path.parent/ref['path']))
    write(OUT/'manifest.json',dict(scene_path=str(scene_path),native_library=peer['native_library'],native_library_sha256=peer['native_library_sha256'],
        geometry_origin=dict(path=str(parent),sha256=sha),peer_input_sha256=peer['point_input_sha256'],
        layout_revision=deps['layout_revision'],point_input_sha256=mapping['point_input_sha256'],
        source_video=mapping['source_video'],source_pts=[source.FIRST,source.LAST],source_time_base='1/12800',
        geometry_frozen_during_camera_solve=True,protected=refs))
    print(json.dumps(dict(event='prepared',camera_rows=len(selected),all_rows=len(z['pts']),frames=len(np.unique(z['pts'][selected])),peer_state_sha256=sha)),flush=True)

def load():
    m=read(OUT/'manifest.json')
    for r in m['protected']:
        if file_ref(r['path'])!=r: raise ValueError('Selected camera problem dependency changed: '+r['path'])
    z=dict(np.load(source.OUT/'points/point_observations.npz',allow_pickle=False));mapping=read(source.OUT/'points/mapping.json')
    g=dict(np.load(OUT/'geometry_seed.npz',allow_pickle=False));scene_path=Path(m['scene_path'])
    model=SceneParameterModel(read(scene_path),scene_path,optimize_scene=True)
    state=ParameterState(model.initial_state.layout,g['scene_values'])
    core=NativeCore(m['native_library']);native=NativeSceneGPU(scene_path,core,parameter_state=state)
    return m,z,mapping,g,core,native

def solve(args):
    prepare();m,z,mapping,g,core,native=load()
    folder=OUT/getattr(args,'output_name',args.phase);folder.mkdir(exist_ok=True)
    if (folder/'result.json').exists() and args.resume is None:raise ValueError('Existing result; select an explicit saved camera state to continue')
    if (folder/'result.json').exists():
        old_result=read(folder/'result.json')
        with (folder/'previous_results.jsonl').open('a',encoding='utf8') as archive:archive.write(json.dumps(old_result,ensure_ascii=False)+'\n')
        if args.resume:
            (folder/('resume_source_'+file_ref(args.resume)['sha256'][:12]+'.npz')).write_bytes(args.resume.read_bytes())
    rows=np.load(OUT/'camera_rows.npy');paths=[mapping['path_ids'][i] for i in z['path_indices']]
    camera=read(OUT/'camera_seed.json');controls=np.load(OUT/'controls_seed.npy');rig=complete_camera_rig(camera)
    stations=np.load(source.OUT/'points/initialized_points.npz')['stations']
    if args.resume:
        old=dict(np.load(args.resume,allow_pickle=False))
        if str(old['point_input_sha256'])!=m['point_input_sha256'] or not np.array_equal(old['scene_values'],g['scene_values']):raise ValueError('Resume input/geometry mismatch')
        rig=old['rig'];controls=old['controls'];stations=old['stations']
    seed_change=None
    if args.resume is None and abs(rig[15]) <= g['board'][0]-g['board'][2] and -g['board'][1] < rig[16] < 0:
        old_pivot=rig[15:18].copy();rig[16]=-g['board'][1]-1e-6
        seed_change=dict(reason='legacy_S02_pivot_inside_latest_board; initialize_on_existing_negative_y_outside_face',before=old_pivot.tolist(),after=rig[15:18].tolist(),added_physical_bound=False)
    camera={**camera,'rig':rig.tolist()}
    pose_indices=[] if args.phase=='motion' else [0,1,2,6,7,8]
    if args.phase=='image_intrinsics':pose_indices += [15,16,17]
    if args.phase=='navigation':
        pose_indices=getattr(args,'camera_parameter_indices',None)
        if pose_indices is None:pose_indices=list(range(31 if len(rig)==43 else 27))
        if len(pose_indices)!=len(set(pose_indices)) or any(k<0 or k>=(31 if len(rig)==43 else 27) for k in pose_indices):
            raise ValueError('Invalid auxiliary camera parameter chart')
    motion,labels=UniformMotionChart.create(len(controls),camera['first_control_time'],camera['control_dt']).directions(reference_angle_gauge=bool(pose_indices))
    nm=motion.shape[2];q=nm+len(pose_indices);directions=np.zeros((len(controls),3,q));directions[:,:,:nm]=.001*motion
    rd=np.zeros((31 if len(rig)==43 else 27,q))
    camera_scales=np.array([.001]*6+[.001]*3+[.00025]*3+[.00025]*3+[1.,1.,.001,1.,1.,.03,.03,.1,.1,.003,.003,.001]+([.05,.1,.003,.003] if len(rig)==43 else []))
    for i,k in enumerate(pose_indices):rd[k,nm+i]=camera_scales[k]
    local=np.full(len(rows),-1,np.int32);curves=np.flatnonzero(z['kind'][rows]==1);local[curves]=np.arange(len(curves))
    periodic=np.array([native.paths[paths[rows[i]]]['kind']==7 for i in curves]);lower=np.where(periodic,-np.inf,0.);upper=np.where(periodic,np.inf,1.)
    times=(z['pts']-camera['reference_pts'])/12800
    support=min(times.min()-camera['first_control_time']-camera['control_dt'],camera['first_control_time']+(len(controls)-2)*camera['control_dt']-times.max())
    max_readout=float(2*support-1e-9)
    start=time.monotonic();log=(folder/'solve.jsonl').open('a',encoding='utf8',buffering=1)
    def emit(event,**v):
        item=dict(event=event,elapsed_seconds=time.monotonic()-start,**v);log.write(json.dumps(safe(item),ensure_ascii=False)+'\n')
        if event=='lm_step':
            r=v['record'];summary=r['summary'];brief=dict(event=event,elapsed_seconds=item['elapsed_seconds'],trial=summary['trial_count'],accepted=summary['accepted'],quality=r.get('quality'),cost=summary['trial_cost'])
        elif event=='returned':brief=dict(event=event,elapsed_seconds=item['elapsed_seconds'],stop_reason=v['result']['stop_reason'],quality=v['result']['quality'])
        else:brief=item
        print(json.dumps(safe(brief),ensure_ascii=False),flush=True)
    print('MAIN LOG: '+str(folder/'solve.jsonl'),flush=True)
    if seed_change:emit('camera_seed_adaptation',**seed_change)
    emit('setup',rows=len(rows),all_rows=len(z['pts']),frames=len(np.unique(z['pts'][rows])),motion_columns=nm,camera_chart=pose_indices,scene_columns=0,
        geometry_source=m['geometry_origin'],unconstrained_geometry_used_as_camera_anchor=False,point_input=m['point_input_sha256'],
        auxiliary_projection_not_physical_camera_calibration=args.phase=='navigation',outside_mount_constraint=args.phase!='navigation')
    def quality(e):
        invalid=e['status']!=0;err=np.linalg.norm(e['values'][:,9:11],axis=1);good=~invalid
        if not good.all():raise ValueError('Invalid camera projection rows: '+str(np.count_nonzero(invalid)))
        return dict(cost=float(.5*np.dot(err,err)),median=float(np.median(err)),p95=float(np.percentile(err,95)),maximum=float(err.max()),invalid=0)
    with native.resident(camera=camera,controls=controls,path_ids=[paths[i] for i in rows],stations=stations[rows],pts=z['pts'][rows],observed=z['uv'][rows]) as resident:
        resident.configure_camera_parameters(directions=rd,max_readout=max_readout)
        before=resident.evaluate('cuda');emit('before',**quality(before));np.savez(folder/'before.npz',**before,rows=rows)
        if args.phase!='navigation':resident.configure_mount_constraints(side='outside')
        emit('linearization',metrics=resident.configure_pixel_linearization(directions=directions,local_ids=local,local_scales=np.full(len(curves),.001)))
        class MemoryStatus(ctypes.Structure):
            _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended')]
        memory=MemoryStatus();memory.length=ctypes.sizeof(memory);ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory))
        emit('solver',metrics=resident.configure_solver(lower=lower,upper=upper,device_budget_bytes=resident.creation['free_after_alloc_bytes'],host_budget_bytes=memory.avail_phys,damping_metric='jacobi_image'))
        def save():
            v=resident.export_state()
            if not np.array_equal(v['board'],g['board']) or not np.array_equal(v['layout'],g['layout']):raise ValueError('Frozen geometry changed')
            full=stations.copy();full[rows]=v['stations'];v.update(stations=full,scene_values=g['scene_values'],source_rows=z['source_rows'],point_input_sha256=np.array(m['point_input_sha256']))
            np.savez(folder/'accepted_camera_pending.npz',**v);(folder/'accepted_camera_pending.npz').replace(folder/'accepted_camera_latest.npz')
            return v
        emit('station_initialization',metrics=resident.refine_curve_stations());save()
        def progress(record):
            if record['summary']['accepted']:
                record['station_refinement']=resident.refine_curve_stations(mode='local');save()
                errors=resident.evaluate_errors();record['quality']=dict(cost=float(.5*np.dot(errors,errors)),median=float(np.median(errors)),p95=float(np.percentile(errors,95)),maximum=float(errors.max()))
                write(folder/'current_quality.json',record['quality'])
            emit('lm_step',record=record)
            return not (folder/'stop_requested').exists()
        done=threading.Event()
        def watch():
            while not done.wait(.2):
                if (folder/'stop_requested').exists():resident.request_stop();return
        watcher=threading.Thread(target=watch,daemon=True);watcher.start()
        try:
            passes=[]
            while True:
                result=resident.solve(progress=progress,numerical_progress=lambda r:emit('factorization_retry',record=r));passes.append(dict(result))
                if result['stop_reason']=='stopped_by_caller':break
                full=resident.refine_curve_stations(mode='global');diagnostic=resident.stationarity_diagnostics();save()
                emit('final_full_curve_search',metrics=full,stationarity=diagnostic)
                if full['failed_rows']:raise RuntimeError('Final station search failed')
                resolution=max(64*np.finfo(float).eps,result['tolerances']['relative_cost'])*max(1.,abs(diagnostic['objective']))
                if full['pixel_cost_decrease']>resolution and diagnostic['constraint_projected_gradient_inf']>1e-7:
                    emit('continue_after_full_curve_search',same_live_optimizer=True);continue
                result['final_station_search']=full;result['final_verified_stationarity']=diagnostic
                result['final_cost']=diagnostic['objective']
                if diagnostic['constraint_projected_gradient_inf']>1e-7:result['numerically_converged']=False
                break
            result['passes']=passes
            v=save();after=resident.evaluate('cuda')
            np.savez(folder/'after.npz',**after,rows=rows);result['quality']=quality(after)
            result['geometry_unchanged']=True;result['all_original_points_unchanged']=file_ref(source.OUT/'points/point_observations.npz')['sha256']==m['point_input_sha256']
            result['full_original_point_count']=len(z['pts']);result['camera_initialization_rows']=len(rows)
            result['motion']=continuous_motion_diagnostic(core,v['controls'],camera,v['rig'],z['pts'])
            result['human_accepted']=False;write(folder/'result.json',result);emit('returned',result=result)
        except BaseException as e:
            save();emit('failure',error=str(e),traceback=traceback.format_exc());raise
        finally:done.set();watcher.join();log.close()
    load()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['prepare','motion','pose','image_intrinsics','navigation'],default='pose');p.add_argument('--resume',type=Path)
    a=p.parse_args()
    if a.phase=='prepare':prepare()
    else:solve(a)
