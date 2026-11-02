"""Transfer the current source view into a rigid compact-camera mounting model."""
from pathlib import Path
from copy import deepcopy
import argparse,json,os,shutil,numpy as np
import prepare_native_multicamera as p
from initialize_camera_repair import evaluate,extent
from transfer_camera_projection import transfer_projection
from rinkspace.core.native import NativeCore
from rinkspace.domain.camera_experiment import load_camera_experiment
from rinkspace.domain.camera_coupling import read_camera_coupling,build_multicamera_directions
from rinkspace.path_policy import guard_project_write


def main(a):
    out=guard_project_write(a.output.resolve())
    if out.exists():raise FileExistsError(out)
    m=p.read(a.prepared/'joint_manifest.json');mapping=p.read(a.prepared/'mapping.json');z=p.npz(a.prepared/'point_observations.npz');v=p.npz(a.state)
    if p.digest(a.prepared/'point_observations.npz')!=m['point_input_sha256']:raise ValueError('Current input changed')
    core=NativeCore(m['native_library']);old,timeline,model=evaluate(a.prepared,m,v,z,mapping,core)
    p.validate_snapshot(v,model.initial_state.layout,mapping,m,z)
    if np.any(old['status']):raise ValueError('Current projection invalid')
    candidate={k:x.copy() for k,x in v.items()};times=timeline.row_times(z['pts']);notes=[]
    for i,rig in enumerate(v['rig']):
        fixed,indices,_=load_camera_experiment(a.profile,rig);candidate['rig'][i]=fixed
        rows=np.concatenate([np.arange(s['row_begin'],s['row_stop']) for s in m['shots'] if s['camera_index']==i]);shift=np.zeros(3)
        for begin in range(0,len(rows),32768):
            rr=rows[begin:begin+32768];tt=times[rr]+rig[37]*(old['values'][rr,6]/1079-.5)
            previous=core.evaluate_motion_camera(rig,v['controls'],timeline.first,timeline.dt,tt);current=core.evaluate_motion_camera(fixed,v['controls'],timeline.first,timeline.dt,tt)
            if np.any(previous['status']) or np.any(current['status']):raise ValueError('Invalid camera while transferring centre')
            shift+=np.sum(previous['centre']-current['centre'],axis=0)
        shift/=len(rows);candidate['rig'][i,15:18]+=shift
        notes.append(dict(camera_index=i,old_forward_reference=float(rig[20]),old_pupil=rig[22:24].tolist(),new_forward_reference=float(fixed[20]),new_pupil=fixed[22:24].tolist(),pivot_shift=shift.tolist()))
    current,_,_=evaluate(a.prepared,m,candidate,z,mapping,core)
    if np.any(current['status']):raise ValueError('Rigid-centre seed needs a closer view transfer')
    candidate,current,transfer=transfer_projection(m,z,mapping,model,core,timeline,old,candidate,current,a.working_envelope,channels=('pan','tilt','log_f'),backend='cuda')
    for key in ('scene_values','stations','source_rows'):
        if v[key].tobytes()!=candidate[key].tobytes():raise RuntimeError('Initializer changed '+key)
    target=deepcopy(m);target['camera_working_envelope']=dict(path=str(a.working_envelope.resolve()),sha256=p.digest(a.working_envelope));target['camera_parameterization'].update(profile_path=str(a.profile.resolve()),profile_sha256=p.digest(a.profile))
    selections=[load_camera_experiment(a.profile,rig)[1] for rig in candidate['rig']];coupling=read_camera_coupling(target['camera_parameterization']['coupling_path']);_,layout=build_multicamera_directions(candidate['rig'],selections,np.ones(31),coupling);target['camera_parameterization']['unique_camera_columns']=layout['unique_camera_columns'];target['current_resume_snapshot']=str(out/'initial_pixels.npz')
    p.bind_snapshot(candidate,model.initial_state.layout,mapping,target);p.validate_snapshot(candidate,model.initial_state.layout,mapping,target,z)
    quality={}
    for shot in m['shots']:
        sl=slice(shot['row_begin'],shot['row_stop']);quality[shot['id']]=dict(projection_change=extent(np.linalg.norm(current['values'][sl,3:5]-old['values'][sl,3:5],axis=1)),source_error=extent(np.linalg.norm(current['values'][sl,9:11],axis=1)))
    out.mkdir(parents=True)
    try:os.link(a.prepared/'point_observations.npz',out/'point_observations.npz')
    except OSError:shutil.copyfile(a.prepared/'point_observations.npz',out/'point_observations.npz')
    p.write(out/'mapping.json',mapping);p.write(out/'joint_manifest.json',target);np.savez_compressed(out/'initial_pixels.npz',**candidate)
    receipt=dict(schema='rinkspace.rigid_projection_centre_initialization.v1',parent_prepared=str(a.prepared),parent_state=str(a.state),parent_state_sha256=p.digest(a.state),point_input_sha256=target['point_input_sha256'],source_UV_PTS_unchanged=True,scene_stations_unchanged=True,camera_columns=layout['unique_camera_columns'],projection_transfer=transfer,notes=notes,shots=quality,fit_started=False)
    p.write(out/'initialization.json',receipt);print(json.dumps(dict(output=str(out),camera_columns=layout['unique_camera_columns'],shots=quality)),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','state','output','profile','working-envelope'):parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
