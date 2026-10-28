"""Full S01 native GPU joint fit of unweighted original pixel observations."""
from pathlib import Path
import importlib.util,json,time,argparse,traceback,os,ctypes,threading
import numpy as np
spec=importlib.util.spec_from_file_location("s01prep",Path(__file__).with_name("prepare_native_s01.py"));prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import SceneParameterModel,ParameterState
from rinkspace.domain.motion_chart import UniformMotionChart
from rinkspace.session.gpu_geometry import NativeSceneGPU,scene_coordinate_charts,complete_camera_rig


def mount_configuration(metres,rig):
    path=prep.ROOT/'events/ankara_2026/model_constraints/camera_hardware/z190_mount_working_ranges_20261026.json'
    fact=prep.read(path);w=fact['working_ranges'];offset=w['complete_tilt_axis_to_projection_centre_camera_m']
    lower=np.full(len(rig),-np.inf);upper=np.full(len(rig),np.inf)
    lower[17]=w['single_pan_reference_height_above_ice_m']['lower']/metres;upper[17]=w['single_pan_reference_height_above_ice_m']['upper']/metres
    lower[18:21]=np.asarray(offset['lower'])/metres;upper[18:21]=np.asarray(offset['upper'])/metres
    lower[21]=w['axis_spacing_m']['lower']/metres;upper[21]=w['axis_spacing_m']['upper']/metres
    envelope=dict(forward_bounds=[offset['lower'][2]/metres,offset['upper'][2]/metres],
        pupil_bounds=[w['zoom_pupil_shift_from_reference_m']['lower']/metres,w['zoom_pupil_shift_from_reference_m']['upper']/metres],
        max_pan_tilt=np.radians(w['pan_axis_angle_from_vertical_degrees']['upper']),
        max_tilt_deviation=np.radians(w['tilt_axis_angle_from_camera_right_degrees']['upper']),
        elevation_bounds=np.radians([w['first_shot_optical_axis_elevation_degrees'][k] for k in ('lower','upper')]).tolist(),
        roll_bounds=np.radians([w['first_shot_body_roll_degrees'][k] for k in ('lower','upper')]).tolist())
    # Numeric motion limits require an explicit sourced/labelled working
    # configuration. Their absence is recorded, never silently filled in.
    motion=fact.get('continuous_motion_working_limits')
    if motion is not None:
        if motion.get('units')!='radians_seconds_log_f' or not motion.get('basis'):
            raise ValueError('Motion limits require explicit units and their evidence/engineering basis')
        for key in ('max_speed','max_acceleration','max_jerk'):
            if key in motion:envelope[key]=[np.inf if value is None else value for value in motion[key]]
    return lower,upper,envelope,path,fact


def continuous_motion_diagnostic(core,controls,camera,rig,pts):
    clock=camera['time_base_num']/camera['time_base_den'];origin=camera['reference_pts']*clock
    half=abs(rig[37])/2;span=[(int(np.min(pts))-camera['reference_pts'])*clock-half,
        (int(np.max(pts))-camera['reference_pts'])*clock+half]
    extrema=core.motion_derivative_extrema(controls,camera['first_control_time'],camera['control_dt'],span)
    channels={}
    for c,name in enumerate(('pan','tilt','log_f')):
        scale=180/np.pi if c<2 else 1.
        channels[name]={key:dict(signed_value=float(extrema['value'][d,c]*scale),
            source_seconds=float(extrema['time'][d,c]+origin),cell=int(extrema['cell'][d,c]))
            for d,key in enumerate(('speed','acceleration','jerk'))}
    return dict(full_row_source_interval_seconds=[float(x+origin) for x in span],
        exact_continuous_extrema=True,angular_units='degrees per second power',zoom_units='log(f) per second power',channels=channels)


def prepare_start(seed):
    """Initialize from an explicit native candidate; source data supplies no parameters."""
    camera=prep.read(prep.CAMERA)
    state=SceneParameterModel(prep.read(prep.SCENE),prep.SCENE,optimize_scene=True).initial_state
    z,mapping,_=prep.load_pixel_input(prep.RUN);input_sha=prep.digest(prep.RUN/'point_observations.npz')
    if seed is None:
        snapshot=dict(rig=complete_camera_rig(camera),controls=np.load(prep.CAMERA.parent/camera['controls_ref']),scene_values=state.values)
    else:
        snapshot=dict(np.load(seed,allow_pickle=False))
        camera=prep.validate_snapshot_identity(snapshot,state.layout,camera,mapping,points=z,point_input_sha256=input_sha)
    core=NativeCore(prep.LIBRARY)
    rig=complete_camera_rig({**camera,'rig':snapshot['rig'].tolist()});old=rig.copy();controls=snapshot['controls'].copy()
    # This final halo coefficient has zero basis at every exposure midpoint.
    # Give the post-shot readout halo a finite extrapolated seed, then fit it
    # with every other control; this is not a trajectory penalty or smoothing.
    halo_before=controls[-1].copy();controls[-1]=2*controls[-2]-controls[-3]
    snapshot['controls']=controls
    scale=prep.read(prep.ROOT/'events/ankara_2026/model_constraints/ankara_2026_metric_scale_assumed_v1.json')['converted_span_xy_m'][0]
    lower,upper,envelope,path,fact=mount_configuration(scale,rig)
    reference=core.evaluate_motion_camera(rig,controls,camera['first_control_time'],camera['control_dt'],[0.])
    rig[18:21]=np.asarray(fact['working_ranges']['complete_tilt_axis_to_projection_centre_camera_m']['initial_reference'])/scale
    rig[21]=np.clip(rig[21],lower[21],upper[21]);rig[22:24]=0.
    updated=core.evaluate_motion_camera(rig,controls,camera['first_control_time'],camera['control_dt'],[0.])
    rig[15:18]+=reference['centre'][0]-updated['centre'][0]
    rig[17]=np.clip(rig[17],lower[17],upper[17])
    snapshot.update(rig=rig,stations=z['stations'],source_rows=z['source_rows'],point_input_sha256=np.array(input_sha))
    prep.bind_snapshot_identity(snapshot,state.layout,camera,mapping,native_library=prep.LIBRARY)
    np.savez(prep.RUN/'pixel_initial.npz',**snapshot)
    prep.write(prep.RUN/'pixel_initial.json',dict(native_seed=str(seed) if seed else 'selected current native files',native_seed_sha256=prep.digest(seed) if seed else None,
        initial_rig_before=old.tolist(),initial_rig_after=rig.tolist(),range_source=str(path),numeric_ranges_approved_by_user=False,
        final_halo_initial_before=halo_before.tolist(),final_halo_initial_after=controls[-1].tolist(),final_halo_remains_fitted=True,
        reason='Start complete native model inside explicit compact-tripod working geometry; all installation and pupil parameters remain fitted',
        source_observations_provided_parameters=False))
    print(str(prep.RUN/'pixel_initial.npz'),flush=True)


def run(stage='pixels',resume=None,diagnose_step=False,initial_lambda=None,validate_only=False,motion_knot_stride=None,camera_observability=False,stationarity_diagnostics=False):
    if stage!='pixels':raise ValueError('S01 currently uses only unweighted pixel observations')
    if camera_observability and not validate_only:raise ValueError('Camera information audit is read-only and requires --validate-only')
    logfile=prep.RUN/'solve.jsonl';log=logfile.open('a',encoding='utf-8',buffering=1);start=time.monotonic()
    concise=None if validate_only else (prep.RUN/'fit_progress.log').open('a',encoding='utf-8',buffering=1)
    last_cost=None;last_pixel_tail=None
    if concise is not None:
        concise.write('\nstep   elapsed_s                cost          decrease      max_px    >20px  status\n')
    def cost_progress(step,cost,status,errors=None):
        nonlocal last_cost,last_pixel_tail
        if concise is None:return
        if errors is not None:
            last_pixel_tail=(float(np.max(errors)),int(np.count_nonzero(errors>20.)))
        if last_pixel_tail is None:raise RuntimeError('Current accepted pixel diagnostics are missing')
        decrease=0. if last_cost is None else last_cost-cost
        concise.write(f'{int(step):5d}  {time.monotonic()-start:10.2f}  {cost:18.8f}  {decrease:16.8f}  {last_pixel_tail[0]:10.3f}  {last_pixel_tail[1]:7d}  {status}\n')
        last_cost=cost
    def save_pixel_hotspots(evaluated,errors,step):
        rows=np.flatnonzero(errors>20.)
        rows=rows[np.argsort(errors[rows])[::-1]]
        selected=None
        if evaluated is None:
            selected=resident.evaluate_rows(np.unique(np.r_[rows,np.argmax(errors)]).astype(np.int32))
            if np.any(selected['status']):raise RuntimeError('A hotspot row became invalid')
            selected={int(row):uv for row,uv in zip(selected['rows'],selected['values'][:,3:5])}
        def point(i):
            return dict(observation_id=str(z['observation_ids'][i]),source_row=int(z['source_rows'][i]),
                pts=int(z['pts'][i]),source_seconds=float(z['pts'][i]*camera['time_base_num']/camera['time_base_den']),
                path_id=paths[i],observed_uv=z['uv'][i].tolist(),fitted_uv=(selected[i] if selected is not None else evaluated['values'][i,3:5]).tolist(),error_px=float(errors[i]))
        report=dict(schema='rinkspace.current_pixel_hotspots.v1',point_input_sha256=input_sha,
            state='current_accepted_after_station_refinement',step=int(step),cost=float(.5*np.dot(errors,errors)),
            rows=len(errors),maximum=point(int(np.argmax(errors))),
            above_20px=int(len(rows)),threshold_comparison='Euclidean native-pixel residual strictly greater than 20 px',
            observations=[point(int(i)) for i in rows])
        temporary=prep.RUN/'current_error_hotspots.pending.json'
        prep.write(temporary,report);temporary.replace(prep.RUN/'current_error_hotspots.json')
    def json_safe(value):
        if isinstance(value,dict):return {k:json_safe(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [json_safe(v) for v in value]
        if isinstance(value,(float,np.floating)) and not np.isfinite(value):return str(value)
        return value.item() if isinstance(value,np.generic) else value
    def emit(event,**record):
        data=dict(event=event,stage=stage,elapsed_seconds=time.monotonic()-start,**record);line=json.dumps(json_safe(data),ensure_ascii=False,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x));log.write(line+'\n');print(line,flush=True)
    print('MAIN LOG: '+str(logfile),flush=True)
    core=NativeCore(prep.LIBRARY);source_scene=prep.read(prep.SCENE);model=SceneParameterModel(source_scene,prep.SCENE,optimize_scene=True)
    state=model.initial_state;camera=prep.read(prep.CAMERA);controls=np.load(prep.CAMERA.parent/camera['controls_ref']);rig=complete_camera_rig(camera)
    z,mapping,stations=prep.load_pixel_input(prep.RUN)
    input_sha=prep.digest(prep.RUN/'point_observations.npz')
    scale_path=prep.ROOT/'events/ankara_2026/model_constraints/ankara_2026_metric_scale_assumed_v1.json'
    scale=prep.read(scale_path)
    metres=scale['converted_span_xy_m'][0] # conditional uniform scene length; never assert measured metres
    lower,upper,envelope,mount_path,mount=mount_configuration(metres,rig)
    previous_stride=1
    if resume:
        snapshot=dict(np.load(resume,allow_pickle=False))
        camera=prep.validate_snapshot_identity(snapshot,state.layout,camera,mapping,points=z,point_input_sha256=input_sha)
        state=ParameterState(state.layout,snapshot['scene_values']);rig=complete_camera_rig({**camera,'rig':snapshot['rig'].tolist()});controls=snapshot['controls'];stations=snapshot['stations']
        previous_stride=int(snapshot.get('motion_knot_stride',1))
        emit('resume_accepted_numeric_state',path=str(resume),optimizer='new LM damping initialized explicitly; no claim of bitwise optimizer continuation')
    stride=previous_stride if motion_knot_stride is None else motion_knot_stride
    reference_time=(int(z['pts'].min())-camera['reference_pts'])*camera['time_base_num']/camera['time_base_den']
    if resume:reference_time=float(snapshot.get('motion_reference_time',reference_time))
    motion_chart=UniformMotionChart.create(len(controls),camera['first_control_time'],camera['control_dt'],stride,reference_time)
    if stride!=previous_stride:
        original=controls;controls=motion_chart.initialize(controls,bound_zoom=True)
        emit('explicit_motion_space_change',old_stride=previous_stride,new_stride=stride,
            control_interval_seconds=stride*camera['control_dt'],native_storage_grid_unchanged=True,
            initialization_control_change_max=np.max(np.abs(controls-original),axis=0).tolist(),
            initializer='approximate old trajectory once; initial zoom coefficients stay within old valid sampled zoom range, then all motion coefficients remain fitted',
            observation_objective_unchanged=True,postprocess_smoothing=False)
    if np.any(rig<lower)|np.any(rig>upper):raise ValueError('Native initial state outside declared physical working ranges')
    camera={**camera,'rig':rig.tolist()};native=NativeSceneGPU(prep.SCENE,core,parameter_state=state)
    if mapping.get('scene_revision')!=native.scene_revision:
        raise ValueError('Prepared input belongs to a different native scene revision')
    if mapping.get('camera_sha256')!=prep.digest(prep.CAMERA) or mapping.get('control_sha256')!=prep.digest(prep.CAMERA.parent/camera['controls_ref']):
        raise ValueError('Prepared input belongs to a different camera definition')
    receipt_path=prep.RUN/'point_input.json'
    if receipt_path.is_file():
        receipt=prep.read(receipt_path)
        if receipt.get('schema') in ('rinkspace.current_point_input_ad_replacement.v1','rinkspace.current_point_input_ad_increment.v1','rinkspace.current_point_input_time_selection.v1','rinkspace.live_native_ad_increment.v1','rinkspace.live_native_source_identity_repair.v1') and receipt.get('sha256')!=input_sha:
            raise ValueError('Prepared replacement input changed after replacement')
    paths=[mapping['path_ids'][i] for i in z['path_indices']]
    dependencies={b['id'] for b in state.blocks.values() if b['kind'] in ('board_shape','shared_geometry')}
    for path in set(paths):
        if '::' in path:dependencies.update(model.dependencies(path))
    active=state.active_columns(dependencies);nc=len(controls)
    # Reference pan/tilt are coordinate gauges. R0 and both physical axes remain fitted.
    motion_directions,motion=motion_chart.directions()
    nm=len(motion);camera_size=31 if len(rig)==43 else 27;q=nm+camera_size+len(active);directions=np.zeros((nc,3,q));rd=np.zeros((camera_size,q));sd=np.zeros((len(state.values),q));charts=scene_coordinate_charts(state)
    directions[:,:,:nm]=.001*motion_directions
    del motion_directions
    rs=np.array([.001]*6+[.001]*3+[.00025]*3+[.00025]*3+[1.,1.,.001,1.,1.,.03,.03,.1,.1,.003,.003,.001])
    if camera_size==31:rs=np.r_[rs,.05,.1,.003,.003]
    rd[:,nm:nm+camera_size]=np.diag(rs)
    for j,k in enumerate(active):
        block=next(b for b in state.blocks.values() if b['available'] and b['offset']<=k<b['offset']+b['size'])
        kind=block['kind'];step=.001
        if charts[k]>0:step=.01
        elif kind=='print_size':step=.01
        elif kind=='print_placement':step=.0002 if k-block['offset']<2 else .001
        elif kind.startswith('print_'):step=.002
        sd[k,nm+camera_size+j]=step
    local=np.full(len(paths),-1,np.int32);localmap={};lo=[];hi=[];scales=[]
    for row,path in enumerate(paths):
        if z['kind'][row]!=1:continue
        key=row;periodic=native.paths[path]['kind']==7
        if key not in localmap:
            localmap[key]=len(localmap);lo.append(-np.inf if periodic else 0.);hi.append(np.inf if periodic else 1.);scales.append(.001)
        local[row]=localmap[key]
    bounds=dict(lower=np.asarray(lo),upper=np.asarray(hi));max_readout=2*min((int(z['pts'].min())-camera['reference_pts'])/12800-camera['first_control_time']-camera['control_dt'],camera['first_control_time']+(nc-2)*camera['control_dt']-((int(z['pts'].max())-camera['reference_pts'])/12800))
    max_readout=float(np.nextafter(max_readout,0.))
    activity=dict(schema='rinkspace.s01_joint_activity.v1',camera=str(prep.CAMERA),camera_sha256=prep.digest(prep.CAMERA),
        scene_revision=native.scene_revision,active_camera_chart=list(range(camera_size)),active_scene_indices=active.tolist(),
        active_scene_blocks=sorted(dependencies),inactive_reason='unobserved print blocks and explicit dependent/gauge coordinates omitted; temporal space is explicitly recorded in motion_basis',
        motion_coordinates=motion,reference_angle_gauge='pan/tilt at recorded motion reference time held at existing value; reference orientation is free',
        motion_reference_time=reference_time,
        motion_basis=dict(family='uniform_cubic_exact_native_refinement',knot_stride=stride,
            native_storage_dt=camera['control_dt'],fitted_control_dt=stride*camera['control_dt'],
            fitted_first=motion_chart.coarse_first,fitted_control_count=motion_chart.refinement.shape[1],
            postprocess_smoothing=False,selection='explicit run choice; accuracy requires source-motion validation'),
        source_rows=len(paths),source_frames=len(np.unique(z['pts'])),phase=stage,point_input_sha256=input_sha,
        numerical_path_count=len(set(paths)),native_display_path_count=len(native.paths),
        domain_path_scope='paths referenced by actual observations; full shared parameter definitions retained',
        loss=dict(family='unweighted_squared_pixel_distance',formula='0.5 * sum_i((u_pred-u_obs)^2+(v_pred-v_obs)^2)',
            source_weights=False,covariance=False,robust_loss=False,error_allowance=False,source_query_constraints=False),
        damping_metric='jacobi_image',motion_prior=dict(enabled=False),
        projection_domain=dict(pixel_plane='unclipped; original predicted-row equation',
            time_half_seconds=max_readout/2,temporal_support='same interval as compiled motion dependencies',
            certificate='positive depth, native optical domain, unique implicit row root',
            observed_row_substitution=False),
        wide_optics=dict(enabled=len(rig)==43,formula='h=max(selected_knot-log_f,0)^2; Brown[k1,k2,p1,p2] += four fitted coefficients*h',selected_basis_knot=float(rig[38]) if len(rig)==43 else None,coefficients_fitted=True),
        physical_constraints=dict(pivot='fixed across shot, negative-y board exterior, positive height',nonzero_lever=True,
            working_range_source=str(mount_path),conditional_metres_per_L=metres,
            numeric_ranges_approved_by_user=False,complete_tilt_axis_to_pupil_envelope_enforced=True,
            continuous_motion_limits={key:envelope.get(key) for key in ('max_speed','max_acceleration','max_jerk')},
            verified_metre_scale=False,readout_support_seconds=max_readout,readout_limit_reason='allocated full spline halo, not manufacturer sensor rate'),
        original_source_and_native_parameter_authorities_separate=True)
    prep.write(prep.RUN/f'activity_{stage}.json',activity)
    emit('compile_joint_problem',rows=len(paths),frames=len(np.unique(z['pts'])),retained_columns=q,motion_columns=nm,camera_columns=camera_size,scene_columns=len(active),station_variables=len(localmap))
    emit('initial_continuous_motion',diagnostic=continuous_motion_diagnostic(core,controls,camera,rig,z['pts']),
        declared_limits={key:envelope.get(key) for key in ('max_speed','max_acceleration','max_jerk')})
    with native.resident(camera=camera,controls=controls,path_ids=paths,stations=stations,pts=z['pts'],observed=z['uv'],scene_directions=sd) as resident:
        resident.configure_camera_parameters(directions=rd,lower=lower,upper=upper,max_readout=max_readout)
        resident.configure_mount_constraints(side='negative_y')
        resident.configure_camera_envelope(**envelope,frame_time_bounds=[
            (int(z['pts'].min())-camera['reference_pts'])/12800,
            (int(z['pts'].max())-camera['reference_pts'])/12800])
        emit('resident_created',metrics=resident.creation)
        plan=resident.configure_pixel_linearization(directions=directions,local_ids=local,local_scales=np.asarray(scales))
        emit('linearization_compiled',metrics=plan)
        def attach_solver():
            class MemoryStatus(ctypes.Structure):
                _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended')]
            memory=MemoryStatus();memory.length=ctypes.sizeof(memory)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):raise OSError('Cannot query available host memory')
            config=resident.configure_solver(**bounds,device_budget_bytes=resident.creation['free_after_alloc_bytes'],host_budget_bytes=memory.avail_phys,damping_metric='jacobi_image',initial_lambda=initial_lambda if initial_lambda is not None else (100. if diagnose_step else 0.))
            emit('solver_attached',metrics=config)
        def save_stationarity(label):
            if not hasattr(resident._lib,'rs_g4_stationarity_diagnostics'):
                report=dict(available=False,full_nonlinear_kkt_checked=False,reason='explicit native library predates current-state constraint diagnostics')
                prep.write(prep.RUN/f'stationarity_{label}.json',report);emit('stationarity_diagnostics_unavailable',label=label,metrics=report)
                if stationarity_diagnostics:raise RuntimeError(report['reason'])
                return report
            before=resident.export_state()
            diagnostic=resident.stationarity_diagnostics(include_constraints=True)
            arrays={key:diagnostic.pop(key) for key in ('constraint_records','normalized_constraint_jacobian','raw_retained_gradient','projected_retained_descent')}
            rows=np.flatnonzero(local>=0);values=before['stations'][rows]
            lower_active=np.isfinite(bounds['lower'])&(values<=bounds['lower']+1e-12)
            upper_active=np.isfinite(bounds['upper'])&(values>=bounds['upper']-1e-12)
            arrays.update(station_source_rows=z['source_rows'][rows],station_scale=np.asarray(scales),
                station_lower=bounds['lower'],station_upper=bounds['upper'],station_lower_active=lower_active,station_upper_active=upper_active)
            after=resident.export_state()
            for key,value in before.items():
                if not np.array_equal(value,after[key]):raise RuntimeError('Stationarity diagnostics changed accepted state: '+key)
            arrays.update(camera_chart_directions=rd,scene_chart_directions=sd)
            np.savez(prep.RUN/f'stationarity_{label}.npz',**arrays)
            diagnostic.update(parameters_bitwise_unchanged=True,point_input_sha256=input_sha,
                active_station_lower_count=int(np.count_nonzero(lower_active)),active_station_upper_count=int(np.count_nonzero(upper_active)),
                retained_chart=dict(motion_columns=nm,camera_columns=camera_size,scene_indices=active.tolist(),motion_scale=.001,
                    directions='camera_chart_directions and scene_chart_directions in the attached NPZ'),
                constraint_family_names={'0':'rig_bounds','1':'scene_bounds','2':'shared_geometry_relation','3':'continuous_camera_envelope','4':'print_domain','5':'mount_geometry'},
                arrays=str(prep.RUN/f'stationarity_{label}.npz'),full_shot_image_accuracy_claimed=False)
            prep.write(prep.RUN/f'stationarity_{label}.json',json_safe(diagnostic));emit('stationarity_diagnostics',label=label,metrics=diagnostic)
            return diagnostic
        if validate_only:
            # Validation consumes the delivered stations exactly. It must not allocate
            # an optimizer or refine station values before checking the input.
            before=resident.evaluate('cuda');linear=resident.linearize('cuda');evaluated=resident.evaluate('cuda')
            checked=resident.export_model();checked['scene_values']=resident.export_scene_parameters()
            if not np.array_equal(before['status'],evaluated['status']) or not np.array_equal(before['values'],evaluated['values']):raise ValueError('Read-only linearization changed native evaluation')
            if np.any(evaluated['status']) or not np.isfinite(evaluated['values']).all():raise ValueError('Current native cannot evaluate every retained point')
            raw_cost=float(.5*np.sum(evaluated['values'][:,9:11]**2))
            if not np.isclose(raw_cost,linear['summary']['huber_cost'],rtol=1e-11,atol=1e-7):raise ValueError('GPU objective is not the complete unweighted pixel sum')
            if plan.get('objective')!='unweighted_squared_pixel_distance' or linear['summary']['downweighted_groups']!=0:raise ValueError('Unexpected point weighting')
            for key,want in (('rig',rig),('scene_values',state.values)):
                if not np.array_equal(checked[key],want):raise ValueError('Input verification altered global native parameter: '+key)
            if linear['summary']['failed_rows']!=0:raise ValueError('Native linearization has invalid rows')
            report=dict(passed=True,input_sha256=input_sha,rows=len(paths),frames=len(np.unique(z['pts'])),
                invalid_rows=0,objective='unweighted_squared_pixel_distance',raw_pixel_cost=raw_cost,
                camera_columns=camera_size,motion_columns=nm,scene_columns=len(active),retained_columns=q,
                station_variables=len(localmap),native_library=str(prep.LIBRARY),native_library_sha256=prep.digest(prep.LIBRARY),
                full_joint_linearization=linear,optimizer_steps=0,global_native_parameters_changed=False,
                trajectory_space_changed_before_validation=stride!=previous_stride,
                source_native_authority_files_changed=False,
                station_refinement_in_temporary_handle_only=False,point_input_changed=False,full_shot_accuracy_claimed=False,
                delivered_stations_evaluated_without_refinement=True,optimizer_workspace_allocation_tested=False,
                evaluation_bitwise_unchanged_after_linearization=True,linearization_plan=plan)
            if stationarity_diagnostics:
                attach_solver();report['stationarity']=save_stationarity('input');report['optimizer_workspace_allocation_tested']=True
            prep.write(prep.RUN/'fit_input_validation.json',report);emit('input_validation_passed',report=report)
            if camera_observability:
                from audit_native_s01_camera_observability import audit,CAMERA_NAMES
                names=[f'motion/{coordinate}' for coordinate in motion]+list(CAMERA_NAMES[:camera_size])
                for k in active:
                    block=next(b for b in state.blocks.values() if b['available'] and b['offset']<=k<b['offset']+b['size'])
                    names.append(f"{block['id']}/{k-block['offset']}")
                audit(resident,{**z,'numeric_stations':stations},evaluated,nm,camera_size,names,
                      prep.RUN/'camera_identifiability/current_information',prep.write,emit,
                      context=dict(point_input_sha256=input_sha,scene_revision=native.scene_revision,
                          state=str(resume),state_sha256=prep.digest(resume) if resume else None,
                          native_library_sha256=prep.digest(prep.LIBRARY),camera_chart_steps=rs.tolist()))
            return False
        attach_solver()
        save_stationarity('input')
        emit('initial_curve_correspondence_refinement',metrics=resident.refine_curve_stations())
        if not validate_only:
            initial=resident.evaluate('cuda')
            if np.any(initial['status']) or not np.isfinite(initial['values']).all():raise ValueError('Initial state has invalid retained pixel observations')
            initial_errors=np.linalg.norm(initial['values'][:,9:11],axis=1)
            cost_progress(0,float(.5*np.sum(initial_errors**2)),'initial',initial_errors)
            save_pixel_hotspots(initial,initial_errors,0)
        def checkpoint():
            snapshot=resident.export_state();snapshot.update(source_rows=z['source_rows'],point_input_sha256=np.array(input_sha),motion_knot_stride=np.array(stride))
            snapshot.update(motion_grid_first_control_time=np.array(camera['first_control_time']),motion_grid_control_dt=np.array(camera['control_dt']),
                motion_reference_time=np.array(reference_time),fit_first_pts=np.array(z['pts'].min()),fit_stop_pts_exclusive=np.array(z['pts'].max()+256))
            prep.bind_snapshot_identity(snapshot,state.layout,camera,mapping,native_library=prep.LIBRARY)
            tmp=prep.RUN/'accepted_pixels_pending.npz';np.savez(tmp,**snapshot);tmp.replace(prep.RUN/'accepted_pixels_latest.npz');return snapshot
        checkpoint();last_report=time.monotonic();transition=False;previous_valid=None;last_rejected_signature=None
        def progress(record):
            nonlocal last_report,transition,previous_valid,last_rejected_signature
            summary=record['summary'];snapshot=None
            if diagnose_step:
                np.savez(prep.RUN/'step_direction_audit.npz',changes=resident.probe_step_changes(),source_rows=z['source_rows'],**resident.export_step_system(sparse=True))
                np.savez(prep.RUN/'step_direction_trial.npz',**resident.export_state(trial=True))
                prep.write(prep.RUN/'step_direction_audit.json',json_safe(record))
                emit('diagnostic_step',record=record);return False
            if not summary['accepted'] and summary['reason_code']==3:
                signature=(summary['first_failed_row'],summary['trial_domain_failures'])
                if signature!=last_rejected_signature:
                    np.savez(prep.RUN/'latest_rejected_trial.npz',**resident.export_state(trial=True))
                    prep.write(prep.RUN/'latest_rejected_trial.json',json_safe(record));last_rejected_signature=signature
            if summary['accepted']:
                record['curve_correspondence_refinement']=resident.refine_curve_stations(mode='local' if hasattr(resident._lib,'rs_g3_refine_stations_ex') else 'global')
                record['pixel_cost_after_correspondence_refinement']=summary['trial_cost']-record['curve_correspondence_refinement']['pixel_cost_decrease']
                errors=resident.evaluate_errors()
                if not np.isfinite(errors).all():raise RuntimeError('An accepted state lost a retained pixel observation')
                pixel_cost=float(.5*np.dot(errors,errors))
                if not np.isclose(pixel_cost,record['pixel_cost_after_correspondence_refinement'],rtol=1e-10,atol=1e-7):raise RuntimeError('Reported objective differs from the complete raw pixel residuals')
                record['raw_pixel_diagnostics']=dict(rows=len(errors),invalid_rows=0,cost=pixel_cost,rms=float(np.sqrt(np.mean(errors*errors))),
                    median=float(np.median(errors)),p95=float(np.percentile(errors,95)),p99=float(np.percentile(errors,99)),maximum=float(errors.max()),
                    above_20px=int(np.count_nonzero(errors>20.)))
                save_pixel_hotspots(None,errors,summary['trial_count'])
                snapshot=checkpoint()
                record['continuous_motion']=continuous_motion_diagnostic(core,snapshot['controls'],camera,snapshot['rig'],z['pts'])
                poses=core.evaluate_motion_camera(snapshot['rig'],snapshot['controls'],camera['first_control_time'],camera['control_dt'],(np.unique(z['pts'])-camera['reference_pts'])/12800)
                zoom=poses['zoom_log_ratio'];pupil_cm=(snapshot['rig'][22]*zoom+snapshot['rig'][23]*zoom*zoom)*metres*100
                record['physical_candidate_diagnostic']=dict(lever_camera_xyz_cm=(snapshot['rig'][18:21]*metres*100).tolist(),axis_spacing_cm=float(snapshot['rig'][21]*metres*100),pupil_shift_min_max_cm=[float(pupil_cm.min()),float(pupil_cm.max())],pivot_to_optical_min_max_cm=[float(x) for x in np.percentile(np.linalg.norm(poses['centre']-snapshot['rig'][15:18],axis=1)*metres*100,[0,100])],verified_metric_scale=False,physical_acceptance=False,log_focal_axis_ratio=float(snapshot['rig'][28]),focal_axis_ratio=float(np.exp(snapshot['rig'][28])),distortion_radius_native=float(snapshot['rig'][25]))
                ext=camera.get('inactive_extension_proofs',[])
                if ext and len(snapshot['rig'])==38 and float(np.min(snapshot['controls'][:,2]))<ext[0]['activation_log_f_native']:
                    record['requires_native_wide_optics_extension']=True;emit('accepted_step_requires_model_domain_extension',record=record);return False
            current_cost=record['raw_pixel_diagnostics']['cost'] if summary['accepted'] else (last_cost if last_cost is not None else float(summary['old_cost']))
            cost_progress(summary['trial_count'],current_cost,'accepted' if summary['accepted'] else 'rejected',errors if summary['accepted'] else None)
            emit('lm_step',record=record);last_report=time.monotonic()
            if (prep.RUN/'stop_requested').exists():emit('safe_restart_requested');return False
            return True
        def numeric(record):emit('numerical_factorization_retry',record=record)
        watcher_done=threading.Event()
        def watch_stop():
            while not watcher_done.wait(.2):
                if (prep.RUN/'stop_requested').exists():resident.request_stop();return
        watcher=threading.Thread(target=watch_stop,daemon=True);watcher.start()
        try:
            passes=[]
            while True:
                result=resident.solve(progress=progress,numerical_progress=numeric);passes.append(result)
                if diagnose_step or result['stop_reason']=='stopped_by_caller':break
                full_search=resident.refine_curve_stations(mode='global')
                diagnostic=(resident.stationarity_diagnostics() if hasattr(resident._lib,'rs_g4_stationarity_diagnostics') else
                    dict(objective=result['final_cost']-full_search['pixel_cost_decrease'],constraint_projected_gradient_inf=result['projected_gradient_inf'],full_nonlinear_kkt_checked=False))
                emit('final_full_curve_search',metrics=full_search,stationarity=diagnostic)
                checkpoint()
                if full_search['failed_rows']:raise RuntimeError('Final global station verification failed')
                resolution=64*np.finfo(float).eps*max(1.,abs(diagnostic['objective']))
                if full_search['pixel_cost_decrease']>resolution and diagnostic['constraint_projected_gradient_inf']>1e-7:
                    emit('continue_after_full_curve_search',same_problem_and_live_optimizer=True);continue
                if diagnostic['constraint_projected_gradient_inf']>1e-7:
                    result['numerically_converged']=False
                result['final_global_station_verification']=full_search
                break
            result['solve_passes']=[dict(item) for item in passes[:-1]]
            result['all_passes_totals']={key:sum(item[key] for item in passes) for key in ('trials','accepted','rejected','wall_ms','explicit_application_d2h_bytes','diagnostic_d2h_bytes_during_solve','instrumented_d2h_bytes_during_solve')}
            final_diagnostic=save_stationarity('final')
            result['final_verified_stationarity']=final_diagnostic
            snapshot=checkpoint();evaluation=resident.evaluate('cuda')
            np.savez(prep.RUN/f'evaluation_{stage}.npz',values=evaluation['values'],status=evaluation['status'])
            emit('native_solve_returned',result=result,transition_to_exact=transition)
            final_errors=np.linalg.norm(evaluation['values'][:,9:11],axis=1)
            if np.any(evaluation['status']) or not np.isfinite(final_errors).all():raise RuntimeError('Final accepted state has invalid pixel observations')
            result['native_final_cost_before_station_verification']=result['final_cost']
            result['final_cost']=float(.5*np.dot(final_errors,final_errors))
            if 'constraint_projected_gradient_inf' in final_diagnostic:
                result['projected_gradient_inf']=final_diagnostic['constraint_projected_gradient_inf']
                result['constraint_projected_gradient_inf']=final_diagnostic['constraint_projected_gradient_inf']
            result['driver_wall_ms']=1000*(time.monotonic()-start)
            cost_progress(result['all_passes_totals']['trials'],float(.5*np.dot(final_errors,final_errors)),result['stop_reason'],final_errors)
            save_pixel_hotspots(evaluation,final_errors,result['all_passes_totals']['trials'])
            if concise is not None:concise.close()
            prep.write(prep.RUN/f'result_{stage}.json',json_safe(dict(result=result,transition_to_exact=transition,completed_full_fit=False)))
        except BaseException as e:
            try:checkpoint()
            except BaseException:pass
            if concise is not None:
                concise.write('solver_failure: '+str(e)+'\n');concise.close()
            emit('solver_failure',error=str(e),traceback=traceback.format_exc());raise
        finally:
            watcher_done.set();watcher.join()
    return transition

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['pixels'],default='pixels');parser.add_argument('--resume',type=Path)
    parser.add_argument('--stationarity-diagnostics',action='store_true',help='Export no-update original gradients and active constraint normals in validation mode');parser.add_argument('--camera-observability',action='store_true',help='Read-only data-only camera/nuisance information audit; requires --validate-only')
    parser.add_argument('--native-library',type=Path,help='Explicit built native library for verified solver comparisons; does not import parameters or observations')
    parser.add_argument('--prepare-start',type=Path,nargs='?',const=Path('current'));parser.add_argument('--diagnose-step',action='store_true');parser.add_argument('--initial-lambda',type=float);parser.add_argument('--run-dir',type=Path,help='Read an already prepared current input; never replay source preparation');parser.add_argument('--validate-only',action='store_true',help='Compile and linearize all current points without taking an optimizer step or replacing a state');parser.add_argument('--motion-knot-stride',type=int,help='Explicit fitted cubic interval as a multiple of the native storage dt; inherited from resume state when omitted');parser.add_argument('--scene-path',type=Path,help='Explicit native scene definition; never read model parameters from point data');args=parser.parse_args()
    if args.scene_path is not None:prep.SCENE=args.scene_path.resolve(strict=True)
    if args.run_dir is not None:prep.RUN=args.run_dir.resolve()
    if args.native_library is not None:prep.LIBRARY=args.native_library.resolve(strict=True)
    prepare_start(None if args.prepare_start==Path('current') else args.prepare_start) if args.prepare_start else run(args.stage,args.resume,args.diagnose_step,args.initial_lambda,args.validate_only,args.motion_knot_stride,args.camera_observability,args.stationarity_diagnostics)
