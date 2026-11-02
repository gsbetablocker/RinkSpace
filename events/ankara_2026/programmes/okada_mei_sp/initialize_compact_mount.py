"""Initialize a centred mount near the existing view, then transfer its view direction.

A centred/intersecting compact head and one-period row-time model are explicit
engineering choices. Source pixels and scene geometry do not supply new values.
"""
from pathlib import Path
from copy import deepcopy
import argparse,os,shutil,json
import numpy as np
import prepare_native_multicamera as p
from initialize_camera_repair import evaluate,extent
from fit_native_s01_joint import mount_configuration
from rinkspace.core.native import NativeCore
from rinkspace.domain.camera_experiment import load_camera_experiment
from rinkspace.domain.camera_coupling import read_camera_coupling,build_multicamera_directions
from rinkspace.path_policy import guard_project_write


def main(a):
    out=guard_project_write(a.output.resolve())
    if out.exists():raise FileExistsError(out)
    m=p.read(a.prepared/'joint_manifest.json');mapping=p.read(a.prepared/'mapping.json');z=p.npz(a.prepared/'point_observations.npz');v=p.npz(a.state)
    if p.digest(a.prepared/'point_observations.npz')!=m['point_input_sha256']:raise ValueError('Current input changed')
    if p.digest(m['native_library'])!=m['native_library_sha256']:raise ValueError('Native library changed')
    core=NativeCore(m['native_library']);old,timeline,model=evaluate(a.prepared,m,v,z,mapping,core)
    p.validate_snapshot(v,model.initial_state.layout,mapping,m,z)
    if np.any(old['status']):raise ValueError('Previous projection invalid')
    candidate={k:x.copy() for k,x in v.items()};notes=[];times=timeline.row_times(z['pts'])
    metres=p.read(p.ROOT/'events/ankara_2026/model_constraints/ankara_2026_metric_scale_assumed_v1.json')['converted_span_xy_m'][0]
    for i,rig in enumerate(v['rig']):
        lower,upper,envelope,_,_=mount_configuration(metres,rig,a.working_envelope)
        new=candidate['rig'][i];new[18]=0.;new[19]=lower[19];new[21]=0.
        rows=np.concatenate([np.arange(s['row_begin'],s['row_stop']) for s in m['shots'] if s['camera_index']==i])
        shift=np.zeros(3)
        for begin in range(0,len(rows),32768):
            rr=rows[begin:begin+32768];tt=times[rr]+rig[37]*(old['values'][rr,6]/1079-.5)
            before=core.evaluate_motion_camera(rig,v['controls'],timeline.first,timeline.dt,tt);after=core.evaluate_motion_camera(new,v['controls'],timeline.first,timeline.dt,tt)
            if np.any(before['status']) or np.any(after['status']):raise ValueError('Invalid row camera')
            shift+=np.sum(before['centre']-after['centre'],axis=0)
        shift/=len(rows);new[15:18]+=shift
        notes.append(dict(camera_index=i,old_lateral_lever_m=float(rig[18]*metres),new_lateral_lever_m=0.,old_axis_spacing_m=float(rig[21]*metres),new_axis_spacing_m=0.,pivot_shift_m=(shift*metres).tolist(),old_readout_seconds=float(rig[37]),new_readout_seconds=float(np.clip(rig[37],lower[37],upper[37]))))
    centred,_,_=evaluate(a.prepared,m,candidate,z,mapping,core)
    if np.any(centred['status']):raise ValueError('Centred mount needs a closer projection seed; source rows retained')
    for i,rig in enumerate(candidate['rig']):
        lower,upper,*_=mount_configuration(metres,rig,a.working_envelope);rig[37]=np.clip(rig[37],lower[37],upper[37])
    final,_,_=evaluate(a.prepared,m,candidate,z,mapping,core)
    if np.any(final['status']):raise ValueError('One-period row model needs a closer projection seed; source rows retained')
    row_only=final
    from transfer_camera_projection import transfer_projection
    candidate,final,transfer=transfer_projection(m,z,mapping,model,core,timeline,old,candidate,final,a.working_envelope)
    certificates=[]
    for i,rig in enumerate(candidate['rig']):
        lower,upper,envelope,_,_=mount_configuration(metres,rig,a.working_envelope)
        if np.any(rig<lower) or np.any(rig>upper):raise ValueError('Outside scalar working bounds')
        windows=timeline.frame_windows()[[j for j,s in enumerate(m['shots']) if s['camera_index']==i]]
        limits=[*envelope['forward_bounds'],*envelope['pupil_bounds'],envelope['max_pan_tilt'],envelope['max_tilt_deviation'],*envelope['elevation_bounds'],*envelope['roll_bounds'],*([np.inf]*9),*((windows-timeline.first)/timeline.dt).ravel()]
        cert=core.evaluate_camera_envelope(rig,candidate['controls'],timeline.first,timeline.dt,limits,joined_zoom_domain=True)
        if cert['margin'] < -128*np.finfo(float).eps:raise ValueError('Continuous physical envelope failed: '+str(cert))
        certificates.append(cert)
    target=deepcopy(m);target['camera_working_envelope']=dict(path=str(a.working_envelope.resolve()),sha256=p.digest(a.working_envelope))
    target['camera_parameterization'].update(profile_path=str(a.profile.resolve()),profile_sha256=p.digest(a.profile))
    selections=[]
    for rig in candidate['rig']:
        copied,indices,_=load_camera_experiment(a.profile,rig)
        if not np.array_equal(copied,rig):raise ValueError('Profile unexpectedly changes numeric seed')
        selections.append(indices)
    coupling=read_camera_coupling(target['camera_parameterization']['coupling_path']);_,layout=build_multicamera_directions(candidate['rig'],selections,np.ones(31),coupling)
    target['camera_parameterization']['unique_camera_columns']=layout['unique_camera_columns'];target['current_resume_snapshot']=str(out/'initial_pixels.npz')
    target['compact_mount_initialization']=dict(parent_prepared=str(a.prepared),parent_state=str(a.state),parent_state_sha256=p.digest(a.state),source_UV_PTS_unchanged=True,scene_stations_unchanged=True,pan_tilt_controls_transferred_from_previous_view=True,engineering_nominal_not_measured=True,fit_started=False)
    p.bind_snapshot(candidate,model.initial_state.layout,mapping,target);p.validate_snapshot(candidate,model.initial_state.layout,mapping,target,z)
    for key in ('scene_values','stations','source_rows'):
        if v[key].tobytes()!=candidate[key].tobytes():raise RuntimeError('Initializer changed '+key)
    quality={}
    for shot in m['shots']:
        sl=slice(shot['row_begin'],shot['row_stop']);quality[shot['id']]=dict(centred_mount_projection_change=extent(np.linalg.norm(centred['values'][sl,3:5]-old['values'][sl,3:5],axis=1)),additional_readout_projection_change=extent(np.linalg.norm(row_only['values'][sl,3:5]-centred['values'][sl,3:5],axis=1)),total_projection_change=extent(np.linalg.norm(final['values'][sl,3:5]-old['values'][sl,3:5],axis=1)))
    out.mkdir(parents=True)
    try:os.link(a.prepared/'point_observations.npz',out/'point_observations.npz')
    except OSError:shutil.copyfile(a.prepared/'point_observations.npz',out/'point_observations.npz')
    p.write(out/'mapping.json',mapping);p.write(out/'joint_manifest.json',target);np.savez(out/'initial_pixels.npz',**candidate)
    receipt=dict(schema='rinkspace.compact_mount_initialization.v1',**target['compact_mount_initialization'],camera_columns=layout['unique_camera_columns'],notes=notes,certificates=certificates,projection_transfer=transfer,shots=quality,point_input_sha256=target['point_input_sha256'])
    p.write(out/'initialization.json',receipt);print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','state','output','working-envelope','profile'):parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
