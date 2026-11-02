"""Project the existing quadratic pupil onto its complete physical zoom interval.

Keep source UV/PTS, scene, motion and stations unchanged. The following fit is
joint; this only supplies a nearby feasible seed for the new lens envelope.
"""
from pathlib import Path
from itertools import combinations
from copy import deepcopy
import argparse,json,os,shutil
import numpy as np
import prepare_native_multicamera as prepared
from initialize_camera_repair import evaluate,extent
from audit_camera_physical_state import zoom_extrema
from fit_native_s01_joint import mount_configuration
from rinkspace.core.native import NativeCore
from rinkspace.path_policy import guard_project_write


def project_quadratic(original,H,zlo,zhi,forward,pupil,reference):
    # Three-variable strictly convex projection. Supporting planes are added
    # at exact quadratic extrema until the entire connected focal span fits.
    A=[];b=[];padding=1e-8
    def interval(row,lo,hi):
        A.extend([row,[-x for x in row]]);b.extend([hi-padding,-lo-padding])
    interval([1.,0.,0.],*reference)
    for z in (zlo,zhi):
        interval([1.,z,z*z],*forward)
        interval([0.,z,z*z],*pupil)
    H=H/np.trace(H);inverse=np.linalg.inv(H)
    last=None;passes=0
    while True:
        a=np.asarray(A);bounds=np.asarray(b);choices=[]
        if np.max(a@original-bounds)<=1e-12:choices.append(original.copy())
        for size in (1,2,3):
            for indices in combinations(range(len(a)),size):
                selected=a[list(indices)];matrix=selected@inverse@selected.T
                if np.linalg.matrix_rank(matrix,tol=1e-12*np.linalg.norm(matrix,2))<size:continue
                dual=np.linalg.solve(matrix,selected@original-bounds[list(indices)])
                if np.min(dual)<-1e-12:continue
                x=original-inverse@selected.T@dual
                if np.max(a@x-bounds)<=1e-12:choices.append(x)
        if not choices:raise ValueError('No feasible complete quadratic pupil seed')
        chosen=min(choices,key=lambda x:float((x-original)@H@(x-original)))
        extrema=[zlo,zhi]
        if chosen[2]!=0:
            vertex=-chosen[1]/(2*chosen[2])
            if zlo<vertex<zhi:extrema.append(vertex)
        additions=[];minimum=np.inf
        for z in extrema:
            for row,limits in (([1.,z,z*z],forward),([0.,z,z*z],pupil)):
                value=np.dot(row,chosen);minimum=min(minimum,value-limits[0],limits[1]-value)
                if value<limits[0]+padding-1e-12 or value>limits[1]-padding+1e-12:
                    additions.append((row,*limits))
        passes+=1
        if not additions:return chosen,dict(cutting_plane_passes=passes,minimum_physical_margin_m=float(minimum),padding_m=padding,linear_faces=len(A))
        if last is not None and np.array_equal(last,chosen):raise RuntimeError('Quadratic envelope projection stalled')
        last=chosen.copy()
        for row,lo,hi in additions:interval(row,lo,hi)


def main(a):
    out=guard_project_write(a.output.resolve())
    if out.exists():raise FileExistsError(out)
    m=prepared.read(a.prepared/'joint_manifest.json');mapping=prepared.read(a.prepared/'mapping.json')
    points=prepared.npz(a.prepared/'point_observations.npz');state=prepared.npz(a.state)
    for key,sha in [('scene_path','scene_file_sha256'),('native_library','native_library_sha256')]:
        if prepared.digest(m[key])!=m[sha]:raise ValueError('Current dependency changed: '+key)
    if prepared.digest(a.prepared/'point_observations.npz')!=m['point_input_sha256']:raise ValueError('Current points changed')
    core=NativeCore(a.native_library)
    if not hasattr(core._lib,'rs_camera_joined_zoom_domain_version'):raise ValueError('New joined-domain library required')
    old,timeline,model=evaluate(a.prepared,m,state,points,mapping,core)
    prepared.validate_snapshot(state,model.initial_state.layout,mapping,m,points)
    if np.any(old['status']):raise ValueError('Original native projection invalid')
    result={k:v.copy() for k,v in state.items()};notes=[]
    metres=prepared.read(prepared.ROOT/'events/ankara_2026/model_constraints/ankara_2026_metric_scale_assumed_v1.json')['converted_span_xy_m'][0]
    times=timeline.row_times(points['pts'])
    for i,rig in enumerate(state['rig']):
        lo,hi,envelope,_,_=mount_configuration(metres,rig,a.working_envelope)
        ranges=[]
        for shot,window in zip(m['shots'],timeline.frame_windows(),strict=True):
            if shot['camera_index']!=i:continue
            expanded=window+[-abs(rig[37])/2,abs(rig[37])/2]
            begin=max(1,int(np.ceil((expanded[0]-timeline.first)/timeline.dt-1)))
            end=min(len(state['controls'])-3,int(np.floor((expanded[1]-timeline.first)/timeline.dt)))
            span=[timeline.first+begin*timeline.dt,timeline.first+(end+1)*timeline.dt]
            ranges.append(np.asarray(zoom_extrema(core,state['controls'],timeline.first,timeline.dt,span))-np.log(rig[24]))
        zlo=min(v[0] for v in ranges);zhi=max(v[1] for v in ranges)
        rows=np.concatenate([np.arange(s['row_begin'],s['row_stop']) for s in m['shots'] if s['camera_index']==i])
        H=np.zeros((3,3));chunks=[]
        for start in range(0,len(rows),32768):
            row=rows[start:start+32768];t=times[row]+rig[37]*(old['values'][row,6]/1079-.5)
            camera=core.evaluate_motion_camera(rig,state['controls'],timeline.first,timeline.dt,t)
            if np.any(camera['status']):raise ValueError('Original row exposure invalid')
            z=camera['zoom_log_ratio'];X=np.c_[np.ones(len(z)),z,z*z];H+=X.T@X
            chunks.append((z,camera['rotation']))
        original=rig[[20,22,23]]*metres
        chosen,qp=project_quadratic(original,H,zlo,zhi,np.asarray(envelope['forward_bounds'])*metres,np.asarray(envelope['pupil_bounds'])*metres,(lo[20]*metres,hi[20]*metres))
        new=result['rig'][i];new[[20,22,23]]=chosen/metres
        # Exact fixed scalar storage must equal the declared box, rather than
        # carrying an old accepted-step rounding error across a new equality.
        fixed=np.isfinite(lo)&(lo==hi);new[fixed]=lo[fixed]
        shift=np.zeros(3)
        for z,rotation in chunks:
            delta=(original[0]-chosen[0])+(original[1]-chosen[1])*z+(original[2]-chosen[2])*z*z
            shift+=np.sum(rotation[:,2,:]*(delta/metres)[:,None],axis=0)
        shift/=len(rows);new[15:18]+=shift
        if np.any(new<lo-3e-14) or np.any(new>hi+3e-14):raise ValueError('Seed outside scalar mount limits')
        windows=timeline.frame_windows()[[j for j,s in enumerate(m['shots']) if s['camera_index']==i]]
        limits=[*envelope['forward_bounds'],*envelope['pupil_bounds'],envelope['max_pan_tilt'],envelope['max_tilt_deviation'],*envelope['elevation_bounds'],*envelope['roll_bounds'],*([np.inf]*9),*((windows-timeline.first)/timeline.dt).ravel()]
        cert=core.evaluate_camera_envelope(new,result['controls'],timeline.first,timeline.dt,limits,joined_zoom_domain=True)
        if cert['margin'] < -128*np.finfo(float).eps:raise ValueError('New continuous certificate failed: '+str(cert))
        notes.append(dict(camera_index=i,old_reference_linear_quadratic_m=original.tolist(),new_reference_linear_quadratic_m=chosen.tolist(),pivot_shift_m=(shift*metres).tolist(),complete_joined_zoom_log_ratio=[float(zlo),float(zhi)],continuous_certificate=cert,projection=qp))
    candidate,_,_=evaluate(a.prepared,m,result,points,mapping,core)
    if np.any(candidate['status']):raise ValueError('New projection invalid')
    change=np.linalg.norm(candidate['values'][:,3:5]-old['values'][:,3:5],axis=1)
    quality={s['id']:dict(projection_change=extent(change[s['row_begin']:s['row_stop']]),source_error=extent(np.linalg.norm(candidate['values'][s['row_begin']:s['row_stop'],9:11],axis=1))) for s in m['shots']}
    target=deepcopy(m);target['native_library']=str(a.native_library.resolve());target['native_library_sha256']=prepared.digest(a.native_library)
    target['camera_working_envelope']=dict(path=str(a.working_envelope.resolve()),sha256=prepared.digest(a.working_envelope))
    if a.coupling is not None:
        from rinkspace.domain.camera_coupling import read_camera_coupling,build_multicamera_directions
        from rinkspace.domain.camera_experiment import load_camera_experiment
        coupling=read_camera_coupling(a.coupling)
        selections=[load_camera_experiment(Path(m['camera_parameterization']['profile_path']),r)[1] for r in result['rig']]
        _,camera_layout=build_multicamera_directions(result['rig'],selections,np.ones(31),coupling)
        target['camera_parameterization'].update(coupling_path=str(a.coupling.resolve()),coupling_sha256=prepared.digest(a.coupling))
        target['camera_parameterization']['unique_camera_columns']=camera_layout['unique_camera_columns']
    target['current_resume_snapshot']=str(out/'initial_pixels.npz')
    target['joined_pupil_initialization']=dict(previous_prepared=str(a.prepared.resolve()),previous_state=str(a.state.resolve()),previous_state_sha256=prepared.digest(a.state),source_uv_pts_unchanged=True,scene_controls_stations_unchanged=True,fit_started=False)
    prepared.camera_working_envelope_path(target)
    prepared.bind_snapshot(result,model.initial_state.layout,mapping,target)
    prepared.validate_snapshot(result,model.initial_state.layout,mapping,target,points)
    for key in ('controls','scene_values','stations','source_rows'):
        if state[key].tobytes()!=result[key].tobytes():raise RuntimeError('Initializer changed '+key)
    out.mkdir(parents=True)
    try:os.link(a.prepared/'point_observations.npz',out/'point_observations.npz')
    except OSError:shutil.copyfile(a.prepared/'point_observations.npz',out/'point_observations.npz')
    prepared.write(out/'mapping.json',mapping);prepared.write(out/'joint_manifest.json',target)
    np.savez(out/'initial_pixels.npz',**result)
    receipt=dict(schema='rinkspace.joined_pupil_initialization.v1',**target['joined_pupil_initialization'],notes=notes,shots=quality,point_input_sha256=target['point_input_sha256'],native_library_sha256=target['native_library_sha256'])
    prepared.write(out/'initialization.json',receipt)
    print(json.dumps(dict(output=str(out),notes=notes,shots=quality)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','state','output','native-library','working-envelope'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--coupling',type=Path)
    main(p.parse_args())
