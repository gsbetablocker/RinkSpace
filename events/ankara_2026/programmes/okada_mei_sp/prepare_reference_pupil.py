"""Remove the weak independent forward-reference coordinate without moving the current view."""
from pathlib import Path
from copy import deepcopy
import argparse,json,os,shutil,numpy as np
import prepare_native_multicamera as p
from initialize_camera_repair import evaluate,extent
from rinkspace.core.native import NativeCore
from rinkspace.domain.camera_coupling import change_zoom_reference,read_camera_coupling,build_multicamera_directions
from rinkspace.domain.camera_experiment import load_camera_experiment,CAMERA_NAMES
from audit_camera_physical_state import zoom_extrema
from rinkspace.path_policy import guard_project_write


def main(a):
    out=guard_project_write(a.output.resolve())
    if out.exists():raise FileExistsError(out)
    m=p.read(a.prepared/'joint_manifest.json');mapping=p.read(a.prepared/'mapping.json');z=p.npz(a.prepared/'point_observations.npz');v=p.npz(a.state)
    if p.digest(a.prepared/'point_observations.npz')!=m['point_input_sha256']:raise ValueError('Current input changed')
    core=NativeCore(m['native_library']);old,timeline,model=evaluate(a.prepared,m,v,z,mapping,core);p.validate_snapshot(v,model.initial_state.layout,mapping,m,z)
    if np.any(old['status']):raise ValueError('Parent projection invalid')
    metres=p.read(p.ROOT/'events/ankara_2026/model_constraints/ankara_2026_metric_scale_assumed_v1.json')['converted_span_xy_m'][0];anchor=a.reference_m/metres
    candidate={k:x.copy() for k,x in v.items()};notes=[]
    for i,rig in enumerate(v['rig']):
        ranges=[zoom_extrema(core,v['controls'],timeline.first,timeline.dt,window+[-abs(rig[37])/2,abs(rig[37])/2]) for shot,window in zip(m['shots'],timeline.frame_windows()) if shot['camera_index']==i]
        loglo=min(x[0] for x in ranges);loghi=max(x[1] for x in ranges)
        roots=np.roots([rig[23],rig[22],rig[20]-anchor]);valid=[float(np.real(r)) for r in roots if abs(np.imag(r))<1e-10 and loglo-1e-10<=np.real(r)+np.log(rig[24])<=loghi+1e-10]
        if not valid:raise ValueError('No existing focal reference reaches the chosen forward anchor for camera '+str(i))
        delta=min(valid,key=lambda x:abs(x+np.log(rig[24])-.5*(loglo+loghi)));fref=float(rig[24]*np.exp(delta));new=change_zoom_reference(rig,fref);new[20]=anchor;candidate['rig'][i]=new
        notes.append(dict(camera_index=i,previous_focal_reference_pixels=float(rig[24]),new_focal_reference_pixels=fref,reference_forward_m=a.reference_m,joined_focal_span_pixels=np.exp([loglo,loghi]).tolist()))
    changed,_,_=evaluate(a.prepared,m,candidate,z,mapping,core)
    if np.any(changed['status']):raise ValueError('Reference transfer changed optical validity')
    delta=extent(np.linalg.norm(changed['values'][:,3:5]-old['values'][:,3:5],axis=1))
    if delta['maximum']>1e-6:raise ValueError('Reference migration did not preserve the existing projection: '+str(delta))
    target=deepcopy(m);target['camera_working_envelope']=dict(path=str(a.working_envelope.resolve()),sha256=p.digest(a.working_envelope));target['camera_parameterization'].update(profile_path=str(a.profile.resolve()),profile_sha256=p.digest(a.profile));names=target['camera_parameterization'].get('active_names_per_rig')
    if names is None:names=[[CAMERA_NAMES[k] for k in load_camera_experiment(a.profile,rig)[1]] for rig in candidate['rig']]
    names=[[name for name in row if name!='lever_z'] for row in names];target['camera_parameterization']['active_names_per_rig']=names;target['camera_parameterization']['reference_pupil_anchor']=notes
    selections=[]
    for i,rig in enumerate(candidate['rig']):
        checked,indices,_=load_camera_experiment(a.profile,rig)
        if not np.array_equal(checked,rig):raise ValueError('Reference profile does not preserve seed')
        selections.append(tuple(k for k in indices if CAMERA_NAMES[k] in names[i]))
    _,layout=build_multicamera_directions(candidate['rig'],selections,np.ones(31),read_camera_coupling(target['camera_parameterization']['coupling_path']));target['camera_parameterization']['unique_camera_columns']=layout['unique_camera_columns'];target['current_resume_snapshot']=str(out/'initial_pixels.npz');p.bind_snapshot(candidate,model.initial_state.layout,mapping,target);p.validate_snapshot(candidate,model.initial_state.layout,mapping,target,z)
    for key in ('scene_values','controls','stations','source_rows'):
        if candidate[key].tobytes()!=v[key].tobytes():raise ValueError('Reference migration changed '+key)
    out.mkdir(parents=True)
    try:os.link(a.prepared/'point_observations.npz',out/'point_observations.npz')
    except OSError:shutil.copyfile(a.prepared/'point_observations.npz',out/'point_observations.npz')
    p.write(out/'mapping.json',mapping);p.write(out/'joint_manifest.json',target);np.savez_compressed(out/'initial_pixels.npz',**candidate)
    receipt=dict(schema='rinkspace.reference_pupil_initialization.v1',parent_prepared=str(a.prepared),parent_state=str(a.state),parent_state_sha256=p.digest(a.state),point_input_sha256=target['point_input_sha256'],reference_m=a.reference_m,notes=notes,source_UV_PTS_unchanged=True,scene_controls_stations_byte_unchanged=True,projection_change_px=delta,camera_columns=layout['unique_camera_columns'],fit_started=False)
    p.write(out/'reference_pupil_receipt.json',receipt);print(json.dumps(receipt),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','state','output','profile','working-envelope'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--reference-m',type=float,default=.05);main(parser.parse_args())
