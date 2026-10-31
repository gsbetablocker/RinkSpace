"""Read-only physical review of an explicitly frozen native multicamera state.

Uses the production native scene, row-time root, camera and spline derivatives.
Temporary zero-readout lens probes never replace the delivered camera state.
No solver, station refinement, source admission or parameter write is performed.
"""
from pathlib import Path
import argparse, hashlib, json, time
import numpy as np
from scipy.optimize import minimize_scalar
import prepare_native_multicamera as prepared
from fit_native_s01_joint import mount_configuration
from rinkspace.core.native import NativeCore
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
from rinkspace.domain.parameters import SceneParameterModel, ParameterState
from rinkspace.session.gpu_geometry import NativeSceneGPU
from rinkspace.path_policy import guard_project_write


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def summary(a):
    a=np.asarray(a,float)
    if not a.size or not np.isfinite(a).all():
        raise ValueError('A reported diagnostic contains no finite support')
    return dict(min=float(a.min()),median=float(np.median(a)),p95=float(np.quantile(a,.95)),
                p99=float(np.quantile(a,.99)),max=float(a.max()),rms=float(np.sqrt(np.mean(a*a))))


def zoom_extrema(core,controls,first,dt,span):
    """All cubic zoom extrema on the exact requested closed time interval."""
    candidates=[float(span[0]),float(span[1])]
    lo=max(1,int(np.floor((span[0]-first)/dt)))
    hi=min(len(controls)-3,int(np.floor((span[1]-first)/dt)))
    for cell in range(lo,hi+1):
        t=first+cell*dt
        a=max(0.,(span[0]-t)/dt); b=min(1.,(span[1]-t)/dt)
        if a>b:continue
        c=controls[cell-1:cell+3,2]
        polynomial=[(-c[0]+3*c[1]-3*c[2]+c[3])/2,c[0]-2*c[1]+c[2],(c[2]-c[0])/2]
        polynomial=np.trim_zeros(polynomial,'f')
        roots=np.roots(polynomial) if len(polynomial)>1 else []
        candidates.extend([t+a*dt,t+b*dt])
        candidates.extend(t+float(u.real)*dt for u in roots if abs(u.imag)<1e-12 and a<float(u.real)<b)
    times=np.unique(candidates)
    k=core.evaluate_motion_kinematics(controls,first,dt,times)
    if np.any(k['status']):raise ValueError('Invalid zoom extremum')
    values=k['position'][:,2]
    return [float(values.min()),float(values.max())]


def pupil_range(rig,log_range,scale):
    z=np.asarray(log_range)-np.log(rig[24]); points=list(z)
    if rig[23]!=0:
        vertex=-rig[22]/(2*rig[23])
        if z[0]<=vertex<=z[1]:points.append(vertex)
    p=rig[22]*np.asarray(points)+rig[23]*np.asarray(points)**2
    return dict(zoom_log_ratio=z.tolist(),pupil_shift_m=[float(p.min()*scale),float(p.max()*scale)],
                complete_forward_m=[float((rig[20]+p.min())*scale),float((rig[20]+p.max())*scale)])


def lens_map(core,rig,focal,grid):
    temporary=rig.copy();temporary[37]=0.
    controls=np.tile([0.,0.,np.log(focal)],(4,1))
    c=core.evaluate_motion_camera(temporary,controls,0.,1.,[1.])
    ray=np.c_[(grid[:,0]-c['cx'][0])/c['fx'][0],(grid[:,1]-c['cy'][0])/c['fy'][0],np.ones(len(grid))]
    world=ray@c['rotation'][0]+c['centre'][0]
    out=core.project_motion(temporary,controls,0.,1.,1080,1.,world)
    if np.any(out['status']):raise ValueError('Invalid common-focal lens probe')
    return out['uv']


def main(args):
    started=time.monotonic(); output=guard_project_write(args.output)
    if output.exists() or output.with_suffix('.npz').exists():raise FileExistsError(output)
    m=read(args.run_root/'joint_manifest.json');mapping=read(args.run_root/'mapping.json')
    identities={str(p):digest(p) for p in [args.state,args.run_root/'point_observations.npz',
        args.run_root/'joint_manifest.json',args.run_root/'mapping.json',Path(m['scene_path']),Path(m['native_library'])]}
    if identities[str(args.run_root/'point_observations.npz')]!=m['point_input_sha256']:
        raise ValueError('Point input identity differs')
    for key,sha in [('scene_path','scene_file_sha256'),('native_library','native_library_sha256')]:
        if identities[m[key]]!=m[sha]:raise ValueError('Current dependency differs: '+key)
    z=prepared.npz(args.run_root/'point_observations.npz');v=prepared.npz(args.state)
    model=SceneParameterModel(read(m['scene_path']),Path(m['scene_path']),optimize_scene=True)
    prepared.validate_snapshot(v,model.initial_state.layout,mapping,m,z)
    state=ParameterState(model.initial_state.layout,v['scene_values'])
    core=NativeCore(m['native_library']);timeline=ShotMotionTimeline(m['shots'],m['control_dt'],m['first_control_time'])
    scale_file=prepared.ROOT/'events/ankara_2026/model_constraints/ankara_2026_metric_scale_assumed_v1.json'
    metres=read(scale_file)['converted_span_xy_m'][0]
    native=NativeSceneGPU(m['scene_path'],core,parameter_state=state)
    paths=np.asarray(mapping['path_ids'])[z['path_indices']]
    times=timeline.row_times(z['pts']);camera_ids=np.empty(len(times),np.int32)
    for s in m['shots']:camera_ids[s['row_begin']:s['row_stop']]=s['camera_index']
    camera={**read(m['cameras'][0]['camera_definition']),'first_control_time':timeline.first,'control_dt':timeline.dt}
    with native.resident(camera=camera,controls=v['controls'],path_ids=paths.tolist(),stations=v['stations'],
            pts=z['pts'],observed=z['uv'],motion_times=times,rigs=v['rig'],camera_ids=camera_ids) as resident:
        resident.configure_projection_time_support(half_seconds=m['max_readout']/2)
        evaluation=resident.evaluate('cuda')
    if np.any(evaluation['status']):raise ValueError('Current full source projection is invalid')
    ev=evaluation['values']; errors=np.linalg.norm(ev[:,9:11],axis=1)
    report=dict(schema='rinkspace.native_camera_physical_state.v1',state=str(args.state),identities=identities,
        rows=len(times),source_exposures=len(np.unique(z['pts'])),metric_scale=metres,
        metric_status='conditional assumed metres, not surveyed',scale_source=str(scale_file),
        state_modified=False,optimizer_started=False,station_refinement=False,source_uv_modified=False,
        whole_state_error_pixels=summary(errors),rigs=[],shots={},common_focal_lens_maps=[],
        limitations=['No manufacturer intrinsic/readout/pupil calibration for the actual event capture chain.',
        'Same camera family does not establish equal crop, extender, focus or optical stabilization settings.',
        'Mechanical bounds are assistant working estimates, not independently measured information.',
        'Observed row strips join the earliest and latest actual landmark row exposures in each frame; gaps within those strips are not new observations.'])
    arrays={}
    for s in m['shots']:
        sid=s['id'];sl=slice(s['row_begin'],s['row_stop']); rig=v['rig'][s['camera_index']]
        pts=z['pts'][sl]; tt=times[sl];e=ev[sl];source_origin=s['reference_pts']/12800-timeline.offset(s)
        row_times=tt+rig[37]*(e[:,6]/1079-.5)
        frames,inverse=np.unique(pts,return_inverse=True)
        ftime=(frames-s['reference_pts'])/12800+timeline.offset(s)
        first_rows=np.full(len(frames),np.inf);last_rows=np.full(len(frames),-np.inf)
        np.minimum.at(first_rows,inverse,row_times);np.maximum.at(last_rows,inverse,row_times)
        pose=core.evaluate_motion_camera(rig,v['controls'],timeline.first,timeline.dt,ftime)
        angle=np.degrees(np.arcsin(np.clip(pose['rotation'][:,2,2],-1,1)))
        roll=np.degrees(np.arctan2(pose['rotation'][:,0,2],-pose['rotation'][:,1,2]))
        temporary=rig.copy();temporary[37]=0.
        pin=core.project_motion(temporary,v['controls'],timeline.first,timeline.dt,1080,tt,e[:,:3])
        if np.any(pin['status']):raise ValueError('Invalid temporary no-readout projection')
        rs_displacement=np.linalg.norm(e[:,3:5]-pin['uv'],axis=1)
        # Native instantaneous lens values at the solved source-row times.
        optical=core.evaluate_motion_camera(rig,v['controls'],timeline.first,timeline.dt,row_times)
        if np.any(optical['status']):raise ValueError('Invalid actual row time')
        ray=np.einsum('nij,nj->ni',optical['rotation'],e[:,:3]-optical['centre'])
        xy=ray[:,:2]/ray[:,2,None];r2=np.sum(xy*xy,axis=1);scale_xy=np.c_[optical['fx'],optical['fy']]
        ideal=xy*scale_xy+np.c_[optical['cx'],optical['cy']]
        radial1=xy*(optical['k1']*r2)[:,None]*scale_xy
        radial2=xy*(optical['k2']*r2*r2)[:,None]*scale_xy
        x,y=xy.T;p1=optical['p1'];p2=optical['p2']
        tangent=np.c_[2*p1*x*y+p2*(r2+2*x*x),p1*(r2+2*y*y)+2*p2*x*y]*scale_xy
        reconstructed=ideal+radial1+radial2+tangent
        discrepancy=float(np.max(abs(reconstructed-e[:,3:5])))
        if discrepancy>2e-5:raise ValueError('Brown component diagnostic differs from native forward')
        net=np.linalg.norm(e[:,3:5]-ideal,axis=1)
        components=np.linalg.norm(radial1,axis=1)+np.linalg.norm(radial2,axis=1)+np.linalg.norm(tangent,axis=1)
        peak_values=np.zeros((3,3));peak_times=np.zeros((3,3));peak_frames=np.zeros((3,3),np.int64)
        for j,(a,b) in enumerate(zip(first_rows,last_rows)):
            ext=core.motion_derivative_extrema(v['controls'],timeline.first,timeline.dt,[a,b])
            larger=np.abs(ext['value'])>np.abs(peak_values)
            peak_values[larger]=ext['value'][larger];peak_times[larger]=ext['time'][larger]+source_origin
            peak_frames[larger]=frames[j]
        rows_motion={}
        for c,name in enumerate(('pan','tilt','log_f')):
            factor=180/np.pi if c<2 else 1.
            rows_motion[name]={name2:dict(signed_value=float(peak_values[d,c]*factor),
                source_seconds=float(peak_times[d,c]),frame_pts=int(peak_frames[d,c]))
                for d,name2 in enumerate(('speed','acceleration','jerk'))}
        span=[ftime[0]-abs(rig[37])/2,ftime[-1]+abs(rig[37])/2]
        zoom_range=zoom_extrema(core,v['controls'],timeline.first,timeline.dt,span)
        wide=optical['log_f']<rig[38]
        report['shots'][sid]=dict(camera_index=s['camera_index'],frames=len(frames),rows=len(pts),
            source_interval=[float(frames[0]/12800),float(frames[-1]/12800)],
            observed_row_time_interval=[float(first_rows.min()+source_origin),float(last_rows.max()+source_origin)],
            observed_row_strips_continuous_derivative_peaks=rows_motion,
            errors_pixels=summary(errors[sl]),solved_row_range=summary(e[:,6]),
            full_row_root_max_equality_px=float(np.max(abs(e[:,7]))),root_contraction_bound=summary(e[:,8]),
            actual_landmark_rs_displacement_pixels=summary(rs_displacement),
            actual_landmark_distortion_pixels=summary(net),
            radial_k1_pixels=summary(np.linalg.norm(radial1,axis=1)),radial_k2_pixels=summary(np.linalg.norm(radial2,axis=1)),
            tangential_pixels=summary(np.linalg.norm(tangent,axis=1)),
            component_sum_over_net_rms=float(np.sqrt(np.mean(components**2))/max(np.sqrt(np.mean(net**2)),1e-30)),
            native_brown_reconstruction_max_component_px=discrepancy,
            elevation_degrees=[float(angle.min()),float(angle.max())],roll_degrees=[float(roll.min()),float(roll.max())],
            optical_height_conditional_m=[float(pose['centre'][:,2].min()*metres),float(pose['centre'][:,2].max()*metres)],
            log_f_continuous_sensor_time_range=zoom_range,
            pupil=pupil_range(rig,zoom_range,metres),wide_extension_rows=int(np.count_nonzero(wide)),
            wide_extension_frames=len(np.unique(pts[wide])),wide_extension_source_interval=None if not np.any(wide) else
                [float(pts[wide].min()/12800),float(pts[wide].max()/12800)])
        arrays[sid+'_source_seconds']=frames/12800
        arrays[sid+'_row_min_seconds']=first_rows+source_origin;arrays[sid+'_row_max_seconds']=last_rows+source_origin
        print(json.dumps(dict(event='shot_checked',shot=sid,rows=len(pts),brown_check_px=discrepancy)),flush=True)
    perimeter=core.board_perimeter(v['board']);stations=np.linspace(0,perimeter,4097)[:-1]
    boundary=np.asarray(core.board_points(v['board'],stations,np.zeros(len(stations))))[:,:2]
    for i,rig in enumerate(v['rig']):
        cam=m['cameras'][i];lo,hi,envelope,config,_=mount_configuration(metres,rig)
        hits=[]
        for k,name in [(17,'pivot_z'),(18,'lever_x'),(19,'lever_y'),(20,'lever_z'),(21,'axis_spacing')]:
            distance=min(abs(rig[k]-lo[k]),abs(rig[k]-hi[k]))*metres
            if distance<1e-8:hits.append(dict(parameter=name,value_m=float(rig[k]*metres),
                bound='lower' if abs(rig[k]-lo[k])<abs(rig[k]-hi[k]) else 'upper'))
        windows=timeline.frame_windows()[[j for j,s in enumerate(m['shots']) if s['camera_index']==i]]
        limits=[*envelope['forward_bounds'],*envelope['pupil_bounds'],envelope['max_pan_tilt'],
                envelope['max_tilt_deviation'],*envelope['elevation_bounds'],*envelope['roll_bounds'],
                *([np.inf]*9),*((windows-timeline.first)/timeline.dt).ravel()]
        certificate=core.evaluate_camera_envelope(rig,v['controls'],timeline.first,timeline.dt,limits)
        R=rig[:9].reshape(3,3);j=int(np.argmin(np.linalg.norm(boundary-rig[15:17],axis=1)));step=perimeter/len(stations)
        near=minimize_scalar(lambda s:np.sum((np.asarray(core.board_point(v['board'],s%perimeter,0.))[:2]-rig[15:17])**2),
            bounds=[stations[j]-2*step,stations[j]+2*step],method='bounded',options={'xatol':1e-13})
        shot_ranges=[report['shots'][s['id']]['log_f_continuous_sensor_time_range'] for s in m['shots'] if s['camera_index']==i]
        observed_pupils=[pupil_range(rig,r,metres) for r in shot_ranges]
        joined=[min(r[0] for r in shot_ranges),max(r[1] for r in shot_ranges)]
        full_zoom_prediction=pupil_range(rig,joined,metres)
        report['rigs'].append(dict(index=i,id=cam['id'],shots=[s['id'] for s in m['shots'] if s['camera_index']==i],
            fx_over_fy=float(np.exp(rig[28])),principal_point=rig[26:28].tolist(),principal_zoom_slope=rig[29:31].tolist(),
            pivot_conditional_m=(rig[15:18]*metres).tolist(),lever_camera_conditional_m=(rig[18:21]*metres).tolist(),
            axis_spacing_conditional_m=float(rig[21]*metres),readout_ms=float(rig[37]*1000),
            scalar_working_bound_hits=hits,working_envelope_native_certificate=certificate,working_envelope_config=str(config),
            nearest_model_board_horizontal_distance_m=float(np.sqrt(near.fun)*metres),
            R0_orthogonality_error=float(np.max(abs(R@R.T-np.eye(3)))),R0_determinant=float(np.linalg.det(R)),
            pan_axis_vertical_angle_degrees=float(np.degrees(np.arccos(np.clip(rig[11],-1,1)))),
            tilt_axis_camera_right_angle_degrees=float(np.degrees(np.arccos(np.clip(rig[12],-1,1)))),
            pan_norm_error=float(abs(np.linalg.norm(rig[9:12])-1)),tilt_norm_error=float(abs(np.linalg.norm(rig[12:15])-1)),
            tilt_dot_R0_pan=float(np.dot(rig[12:15],R@rig[9:12])),
            full_zoom_span_ratio=float(np.exp(joined[1]-joined[0])),observed_focal_intervals_pixels=np.exp(shot_ranges).tolist(),
            pupil_shift_observed_range_m=[min(x['pupil_shift_m'][0] for x in observed_pupils),max(x['pupil_shift_m'][1] for x in observed_pupils)],
            complete_forward_observed_range_m=[min(x['complete_forward_m'][0] for x in observed_pupils),max(x['complete_forward_m'][1] for x in observed_pupils)],
            prediction_across_unobserved_zoom_gaps=full_zoom_prediction))
    gx,gy=np.meshgrid(np.linspace(0,1919,9),np.linspace(0,1079,7));grid=np.c_[gx.ravel(),gy.ravel()]
    arrays['lens_grid']=grid
    for focal in (5000.,7500.,10000.):
        maps=[lens_map(core,r,focal,grid) for r in v['rig']]
        pairs=[]
        for i in range(len(maps)):
            arrays[f'lens_{int(focal)}_rig{i}']=maps[i]
            for j in range(i):
                delta=maps[i]-maps[j];norm=np.linalg.norm(delta,axis=1)
                design=np.c_[np.ones(len(grid)),(grid-[959.5,539.5])/[959.5,539.5]]
                affine=np.linalg.lstsq(design,delta,rcond=None)[0]
                remaining=np.linalg.norm(delta-design@affine,axis=1)
                pairs.append(dict(rigs=[m['cameras'][j]['id'],m['cameras'][i]['id']],difference_pixels=summary(norm),
                    centre_translation_component_pixels=affine[0].tolist(),non_affine_difference_pixels=summary(remaining)))
        report['common_focal_lens_maps'].append(dict(focal_pixels=focal,grid_shape=[9,7],pairs=pairs))
    for path,sha in identities.items():
        if digest(path)!=sha:raise RuntimeError('Diagnostic input changed during read-only review')
    report['elapsed_seconds']=time.monotonic()-started
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output.with_suffix('.npz'),**arrays)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+chr(10),encoding='utf-8')
    print(str(output),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run-root','state','output'):parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
