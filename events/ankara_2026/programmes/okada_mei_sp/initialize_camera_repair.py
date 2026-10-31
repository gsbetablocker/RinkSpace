"""Projection-preserving seeds for the authorized camera-model simplification.

Sources and scene geometry are copied unchanged into each initial state.
The subsequent production fit remains joint over all current scene columns.
"""
from pathlib import Path
import argparse,copy,json,os,time,shutil
import numpy as np
os.environ.setdefault('CUDA_PATH',r'D:/Program Files/CUDA')
import prepare_native_multicamera as prepared
from fit_native_s01_joint import mount_configuration
from rinkspace.core.native import NativeCore
from rinkspace.domain.camera_coupling import change_zoom_reference
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
from rinkspace.domain.parameters import SceneParameterModel,ParameterState
from rinkspace.session.gpu_geometry import NativeSceneGPU
from rinkspace.path_policy import guard_project_write


def evaluate(root,m,v,z,mapping,core):
    model=SceneParameterModel(prepared.read(m['scene_path']),Path(m['scene_path']),optimize_scene=True)
    state=ParameterState(model.initial_state.layout,v['scene_values'])
    native=NativeSceneGPU(m['scene_path'],core,parameter_state=state)
    timeline=ShotMotionTimeline(m['shots'],m['control_dt'],m['first_control_time'])
    camera={**prepared.read(m['cameras'][0]['camera_definition']),'first_control_time':timeline.first,'control_dt':timeline.dt}
    ids=np.empty(len(z['pts']),np.int32)
    for s in m['shots']:ids[s['row_begin']:s['row_stop']]=s['camera_index']
    with native.resident(camera=camera,rigs=v['rig'],camera_ids=ids,controls=v['controls'],
            path_ids=np.asarray(mapping['path_ids'])[z['path_indices']].tolist(),stations=v['stations'],
            pts=z['pts'],observed=z['uv'],motion_times=timeline.row_times(z['pts'])) as resident:
        resident.configure_projection_time_support(half_seconds=m['max_readout']/2)
        for i,physical in enumerate(m['cameras']):
            resident.configure_mount_constraints(side=physical.get('mount_side','negative_y'),camera_index=i)
        e=resident.evaluate('cuda')
    return e,timeline,model


def extent(values):
    a=np.asarray(values,float)
    return dict(rms=float(np.sqrt(np.mean(a*a))),median=float(np.median(a)),p95=float(np.quantile(a,.95)),maximum=float(a.max()))


def share_radial(core,m,z,v,old,timeline,magnification):
    result={k:a.copy() for k,a in v.items()};rigs=result['rig']
    mags=[float(magnification),1.]
    fref=v['rig'][0,24]/mags[0]
    # One common physical-zoom hinge, chosen from the existing secondary
    # basis. Its coefficients remain fitted, including the wide corrections.
    boundary=float(v['rig'][1,38])
    for i,d in enumerate(mags):
        rigs[i]=change_zoom_reference(rigs[i],fref*d)
        rigs[i,38]=boundary+np.log(d)
    H=np.zeros((14,14));b=np.zeros(14)
    for s in m['shots']:
        i=s['camera_index']
        if i not in (0,1):continue
        for begin in range(s['row_begin'],s['row_stop'],32768):
            end=min(begin+32768,s['row_stop']);e=old['values'][begin:end]
            t=(z['pts'][begin:end]-s['reference_pts'])/12800+timeline.offset(s)
            t+=v['rig'][i,37]*(e[:,6]/1079-.5)
            c=core.evaluate_motion_camera(v['rig'][i],v['controls'],timeline.first,timeline.dt,t)
            if np.any(c['status']):raise ValueError('Old optical row invalid')
            q=np.einsum('nij,nj->ni',c['rotation'],e[:,:3]-c['centre'])
            xy=q[:,:2]/q[:,2,None];x,y=xy.T;r2=x*x+y*y
            f=np.c_[c['fx'],c['fy']];ideal=xy*f+np.c_[c['cx'],c['cy']]
            logf=c['log_f'];zoom=logf-np.log(rigs[i,24]);wide=np.maximum(rigs[i,38]-logf,0.)**2
            a=np.zeros((len(x),2,14));r1=xy*r2[:,None]*f;r4=r1*r2[:,None]
            a[:,:,0]=r1;a[:,:,1]=r1*zoom[:,None];a[:,:,2]=r4;a[:,:,3]=r4*zoom[:,None]
            a[:,:,4]=r1*wide[:,None];a[:,:,5]=r4*wide[:,None]
            tangent1=np.c_[2*x*y,r2+2*y*y]*f;tangent2=np.c_[r2+2*x*x,2*x*y]*f
            j=6+4*i;a[:,:,j]=tangent1;a[:,:,j+1]=tangent2
            a[:,:,j+2]=tangent1*wide[:,None];a[:,:,j+3]=tangent2*wide[:,None]
            A=a.reshape(-1,14);target=(e[:,3:5]-ideal).ravel();H+=A.T@A;b+=A.T@target
    scale=np.sqrt(np.diag(H))
    if np.any(scale<=0):raise ValueError('Unsupported new optical coordinate')
    coefficients=np.linalg.lstsq(H/scale[:,None]/scale[None,:],b/scale,rcond=None)[0]/scale
    for i in (0,1):
        rigs[i,[31,32,33,34,39,40]]=coefficients[:6]
        rigs[i,[35,36,41,42]]=coefficients[6+4*i:10+4*i]
    coupling=dict(schema='rinkspace.camera_coupling.v1',id='shared_radial_crop'+str(int(magnification)),
        hardware_calibrated=False,basis='Fixed delivery magnifications are explicit known-mode hypotheses, not measured event settings. Shared native radial functions are initialized against the delivered projection.',
        groups=[dict(id='main_secondary_radial',camera_indices=[0,1],
            shared_names=['k10','k11','k20','k21','wide_k1','wide_k2'],
            delivery_magnifications=mags,reference_focal_base_pixels=float(fref),wide_boundary_base_log_f=boundary)])
    return result,coupling


def linear_pupil(core,m,z,v,old,timeline):
    result={k:a.copy() for k,a in v.items()};notes=[]
    for i,rig in enumerate(v['rig']):
        rows=np.concatenate([np.arange(s['row_begin'],s['row_stop']) for s in m['shots'] if s['camera_index']==i])
        t=timeline.row_times(z['pts'])[rows]+rig[37]*(old['values'][rows,6]/1079-.5)
        c=core.evaluate_motion_camera(rig,v['controls'],timeline.first,timeline.dt,t)
        zoom=c['zoom_log_ratio'];forward=rig[20]+rig[22]*zoom+rig[23]*zoom*zoom
        X=np.c_[np.ones(len(zoom)),zoom]
        unconstrained=np.linalg.lstsq(X,forward,rcond=None)[0]
        scale=prepared.read(prepared.ROOT/'events/ankara_2026/model_constraints/ankara_2026_metric_scale_assumed_v1.json')['converted_span_xy_m'][0]
        lo,hi,envelope,_,_=mount_configuration(scale,rig)
        from audit_camera_physical_state import zoom_extrema
        ranges=[]
        for shot,window in zip(m['shots'],timeline.frame_windows(),strict=True):
            if shot['camera_index']!=i:continue
            expanded=window+[-abs(rig[37])/2,abs(rig[37])/2]
            # Match the production certificate's complete cells, not only
            # the exposure midpoints or observed zoom samples.
            first_cell=max(1,int(np.ceil((expanded[0]-timeline.first)/timeline.dt-1)))
            last_cell=min(len(v['controls'])-3,int(np.floor((expanded[1]-timeline.first)/timeline.dt)))
            span=[timeline.first+first_cell*timeline.dt,timeline.first+(last_cell+1)*timeline.dt]
            ranges.append(np.asarray(zoom_extrema(core,v['controls'],timeline.first,timeline.dt,span))-np.log(rig[24]))
        zlo=min(x[0] for x in ranges);zhi=max(x[1] for x in ranges)
        A=[];limits=[];padding=512*np.finfo(float).eps
        def interval(row,lower,upper):
            A.extend([row,[-x for x in row]]);limits.extend([upper-padding,-lower-padding])
        interval([1.,0.],lo[20],hi[20])
        for zoom_end in (zlo,zhi):
            interval([1.,zoom_end],*envelope['forward_bounds'])
            interval([0.,zoom_end],*envelope['pupil_bounds'])
        A=np.asarray(A);limits=np.asarray(limits);H=X.T@X;inverse=np.linalg.inv(H)
        possibilities=[unconstrained]
        for normal,bound in zip(A,limits,strict=True):
            possibilities.append(unconstrained-inverse@normal*((normal@unconstrained-bound)/(normal@inverse@normal)))
        for j in range(len(A)):
            for k in range(j):
                matrix=A[[j,k]]
                if abs(np.linalg.det(matrix))>1e-14:possibilities.append(np.linalg.solve(matrix,limits[[j,k]]))
        feasible=[x for x in possibilities if np.max(A@x-limits)<=16*np.finfo(float).eps]
        if not feasible:raise ValueError('No feasible linear pupil projection')
        chosen=min(feasible,key=lambda x:float((x-unconstrained)@H@(x-unconstrained)))
        constant,slope=chosen
        new=result['rig'][i];new[20]=constant;new[22]=slope;new[23]=0.
        # A single fixed pivot shift preserves the source-weighted reference
        # optical centre. No per-frame centre is introduced.
        now=core.evaluate_motion_camera(new,v['controls'],timeline.first,timeline.dt,t)
        shift=np.mean(c['centre']-now['centre'],axis=0);new[15:18]+=shift
        notes.append(dict(camera_index=i,new_forward_reference=float(constant),new_pupil_slope=float(slope),
            old_pupil_quadratic=float(rig[23]),pivot_shift=shift.tolist(),
            unconstrained_forward_reference_and_slope=unconstrained.tolist(),constrained_forward_reference_and_slope=chosen.tolist(),
            native_certificate_zoom_range=[float(zlo),float(zhi)],maximum_linear_constraint_violation=float(np.max(A@chosen-limits))))
    return result,notes


def main(args):
    out=guard_project_write(args.output);out.mkdir(parents=True,exist_ok=False)
    root=args.run_root;m=prepared.read(root/'joint_manifest.json');mapping=prepared.read(root/'mapping.json')
    z=prepared.npz(root/'point_observations.npz');v=prepared.npz(args.state)
    for key,sha in [('scene_path','scene_file_sha256'),('native_library','native_library_sha256')]:
        if prepared.digest(m[key])!=m[sha]:raise ValueError('Current dependency changed')
    if prepared.digest(root/'point_observations.npz')!=m['point_input_sha256']:raise ValueError('Point input changed')
    core=NativeCore(m['native_library']);old,timeline,model=evaluate(root,m,v,z,mapping,core)
    prepared.validate_snapshot(v,model.initial_state.layout,mapping,m,z)
    if np.any(old['status']):raise ValueError('Original seed invalid')
    old_hash=prepared.digest(args.state);coupling=None;notes=None
    if args.stage=='shared':candidate,coupling=share_radial(core,m,z,v,old,timeline,args.magnification)
    elif args.stage=='mount':candidate={k:a.copy() for k,a in v.items()}
    else:candidate,notes=linear_pupil(core,m,z,v,old,timeline)
    profile=prepared.read(args.profile)
    descriptor=dict(profile_path=str(args.profile.resolve()),profile_sha256=prepared.digest(args.profile),stage=args.stage)
    if coupling is not None:
        if args.coupling_output is None:raise ValueError('Shared seed requires a coupling output path')
        path=guard_project_write(args.coupling_output)
        if path.exists():
            if prepared.read(path)!=coupling:raise ValueError('Existing coupling differs')
        else:prepared.write(path,coupling)
        descriptor.update(coupling_path=str(path.resolve()),coupling_sha256=prepared.digest(path))
    elif 'camera_parameterization' in m:
        for k in ('coupling_path','coupling_sha256'):descriptor[k]=m['camera_parameterization'][k]
    m=copy.deepcopy(m);m['camera_parameterization']=descriptor
    for name in ('point_observations.npz','mapping.json'):shutil.copyfile(root/name,out/name)
    prepared.write(out/'joint_manifest.json',m)
    prepared.bind_snapshot(candidate,model.initial_state.layout,mapping,m)
    prepared.validate_snapshot(candidate,model.initial_state.layout,mapping,m,z)
    for k in ('controls','scene_values','stations'):
        if not np.array_equal(candidate[k],v[k]):raise RuntimeError('Initializer unexpectedly changed '+k)
    evaluation,_,_=evaluate(out,m,candidate,z,mapping,core)
    good=evaluation['status']==0;d=np.linalg.norm(evaluation['values'][:,3:5]-old['values'][:,3:5],axis=1)
    quality={}
    for s in m['shots']:
        sl=slice(s['row_begin'],s['row_stop']);ok=good[sl]
        quality[s['id']]=dict(invalid=int(np.count_nonzero(~ok)),
            projection_change=extent(d[sl][ok]) if np.any(ok) else None,
            source_error=extent(np.linalg.norm(evaluation['values'][sl,9:11][ok],axis=1)) if np.any(ok) else None)
    scale=prepared.read(prepared.ROOT/'events/ankara_2026/model_constraints/ankara_2026_metric_scale_assumed_v1.json')['converted_span_xy_m'][0]
    envelopes=[]
    for i,rig in enumerate(candidate['rig']):
        lo,hi,envelope,_,_=mount_configuration(scale,rig)
        box=np.flatnonzero((rig<lo-3e-14)|(rig>hi+3e-14)).tolist()
        windows=timeline.frame_windows()[[j for j,s in enumerate(m['shots']) if s['camera_index']==i]]
        limits=[*envelope['forward_bounds'],*envelope['pupil_bounds'],envelope['max_pan_tilt'],envelope['max_tilt_deviation'],
            *envelope['elevation_bounds'],*envelope['roll_bounds'],*([np.inf]*9),*((windows-timeline.first)/timeline.dt).ravel()]
        cert=core.evaluate_camera_envelope(rig,candidate['controls'],timeline.first,timeline.dt,limits)
        envelopes.append(dict(camera_index=i,outside_scalar_storage_indices=box,continuous=cert))
    prepared.write(out/'initialization.json',dict(schema='rinkspace.camera_model_initialization.v1',stage=args.stage,
        previous_state=str(args.state),previous_state_sha256=old_hash,point_input_sha256=m['point_input_sha256'],
        controls_scene_stations_unchanged=True,source_uv_unchanged=True,fit_started=False,
        candidate_invalid_rows=int(np.count_nonzero(~good)),shots=quality,physical_envelopes=envelopes,
        pupil_notes=notes,camera_parameterization=descriptor))
    np.savez(out/'initial_pixels.npz',**candidate)
    np.savez_compressed(out/'initial_projection.npz',values=evaluation['values'],status=evaluation['status'])
    print(json.dumps(dict(output=str(out),invalid=int(np.count_nonzero(~good)),shots=quality,envelopes=envelopes),allow_nan=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run-root','state','output','profile'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--stage',choices=['shared','mount','pupil'],required=True)
    p.add_argument('--magnification',type=int,choices=[1,2],default=2)
    p.add_argument('--coupling-output',type=Path)
    main(p.parse_args())
