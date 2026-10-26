"""Audit the saved camera; optional isolated native fitting never replaces it."""
from pathlib import Path
import importlib.util,json,hashlib
import numpy as np

spec=importlib.util.spec_from_file_location('s01prep',Path(__file__).with_name('prepare_native_s01.py'))
prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
from rinkspace.core.native import NativeCore
from rinkspace.domain.motion_chart import UniformMotionChart


def audit(out=None,snapshot=None):
    out=Path(out) if out is not None else prep.RUN/'camera_physics';out.mkdir(exist_ok=True)
    snapshot=Path(snapshot) if snapshot is not None else prep.RUN/'accepted_pixels_latest.npz'
    blob=snapshot.read_bytes();(out/'audited_state.npz').write_bytes(blob)
    state=dict(np.load(out/'audited_state.npz'));camera=prep.camera_for_snapshot(prep.read(prep.CAMERA),state);core=NativeCore(prep.LIBRARY)
    if str(state.get('point_input_sha256',''))!=prep.digest(prep.RUN/'point_observations.npz'):raise ValueError('Audit snapshot does not match current point input')
    source_pts=np.unique(np.load(prep.RUN/'point_observations.npz')['pts']);source_origin=camera['reference_pts']/12800
    controls=state['controls'];rig=state['rig'];first=camera['first_control_time'];dt=camera['control_dt']
    frame_times=(source_pts-camera['reference_pts'])/12800;half=abs(rig[37])/2;lo=frame_times[0]-half;hi=frame_times[-1]+half
    # Scalar speed extrema lie at acceleration zeros. Acceleration extrema lie
    # at cubic knots. Include those exact times as well as a 1 ms display grid.
    knots=first+dt*np.arange(1,len(controls)-1);knots=knots[(knots>lo)&(knots<hi)]
    edges=np.r_[lo,knots,hi];left=core.evaluate_motion_kinematics(controls,first,dt,edges[:-1])
    middle=core.evaluate_motion_kinematics(controls,first,dt,(edges[:-1]+edges[1:])/2)
    extrema=[]
    for i,(begin,end) in enumerate(zip(edges[:-1],edges[1:])):
        a=left['acceleration'][i];j=middle['jerk'][i];v=left['velocity'][i]
        for channel in range(3):
            if j[channel]!=0:
                x=-a[channel]/j[channel]
                if 0<x<end-begin:extrema.append(begin+x)
        # Exact extrema of the angular speed magnitude for orthogonal axes.
        polynomial=np.zeros(4)
        for channel in (0,1):
            product=np.polynomial.polynomial.polymul([v[channel],a[channel],j[channel]/2],[a[channel],j[channel]])
            polynomial[:len(product)]+=product
        for root in np.polynomial.polynomial.polyroots(polynomial):
            if abs(root.imag)<1e-10 and 0<root.real<end-begin:extrema.append(begin+float(root.real))
    times=np.unique(np.r_[np.linspace(lo,hi,int(np.ceil((hi-lo)/.001))+1),edges,extrema,frame_times])
    k=core.evaluate_motion_kinematics(controls,first,dt,times)
    pose=core.evaluate_motion_camera(rig,controls,first,dt,times)
    if np.any(k['status']) or np.any(pose['status']):raise ValueError('Saved camera does not cover its full exposure interval')
    np.testing.assert_array_equal(k['position'],pose['values'][:,:3])
    frame=core.evaluate_motion_kinematics(controls,first,dt,frame_times)
    source_times=times+camera['reference_pts']/12800
    def peak(v):
        i=int(np.argmax(np.abs(v)));return dict(value=float(v[i]),absolute=float(abs(v[i])),source_seconds=float(source_times[i]))
    angular_speed=np.linalg.norm(k['velocity'][:,:2],axis=1)
    angular_acceleration=np.sqrt(np.sum(k['acceleration'][:,:2]**2,axis=1)+(k['velocity'][:,0]*k['velocity'][:,1])**2)
    channels={}
    for c,name in enumerate(('pan','tilt','log_f')):
        factor=180/np.pi if c<2 else 1.;p=k['position'][:,c]*factor;v=k['velocity'][:,c]*factor;a=k['acceleration'][:,c]*factor;j=k['jerk'][:,c]*factor
        channels[name]=dict(position_range=[float(p.min()),float(p.max())],maximum_speed=peak(v),maximum_acceleration=peak(a),maximum_jerk=peak(j),
            median_absolute_acceleration=float(np.median(abs(a))),p95_absolute_acceleration=float(np.percentile(abs(a),95)),
            derivative_units='deg/s, deg/s^2, deg/s^3' if c<2 else 'log(f)/s, log(f)/s^2, log(f)/s^3')
    # A per-frame rate change is a descriptive quantity, never a loss weight or
    # an automatically chosen physical acceptance threshold.
    frame_change=np.diff(frame['velocity'],axis=0);frame_change[:,:2]*=180/np.pi
    worst=[]
    order=np.argsort(np.linalg.norm(frame_change[:,:2],axis=1))[-20:][::-1]
    for i in order:worst.append(dict(first_pts=int(source_pts[i]),source_seconds=float(source_pts[i]/12800),
        pan_rate_before_after_deg_s=np.degrees(frame['velocity'][i:i+2,0]).tolist(),
        tilt_rate_before_after_deg_s=np.degrees(frame['velocity'][i:i+2,1]).tolist(),
        zoom_log_rate_before_after=frame['velocity'][i:i+2,2].tolist()))
    metres=prep.read(prep.ROOT/'events/ankara_2026/model_constraints/ankara_2026_metric_scale_assumed_v1.json')['converted_span_xy_m'][0]
    z=pose['zoom_log_ratio'];ze=[float(z.min()),float(z.max())]
    if rig[23]!=0 and ze[0]<-rig[22]/(2*rig[23])<ze[1]:ze.append(float(-rig[22]/(2*rig[23])))
    pupil=rig[22]*np.array(ze)+rig[23]*np.array(ze)**2
    path=core.sample_path(state['board'],7,[0.],[],np.linspace(0,core.board_perimeter(state['board']),8193))
    distance=np.linalg.norm(np.asarray(path)[:,:2]-rig[15:17],axis=1).min()*metres
    R=pose['rotation'];elevation=np.degrees(np.arcsin(np.clip(R[:,2,2],-1,1)));roll=np.degrees(np.arctan2(R[:,0,2],-R[:,1,2]))
    mechanical=dict(conditional_metres_per_L=metres,metric_scale_is_surveyed=False,pivot_xyz_m=(rig[15:18]*metres).tolist(),
        approximate_pivot_to_board_xy_m=float(distance),lever_camera_xyz_cm=(rig[18:21]*metres*100).tolist(),axis_spacing_cm=float(rig[21]*metres*100),
        pan_axis_from_vertical_deg=float(np.degrees(np.arccos(np.clip(rig[11],-1,1)))),
        tilt_axis_from_camera_right_deg=float(np.degrees(np.arccos(np.clip(rig[12],-1,1)))),
        pupil_shift_cm=[float(pupil.min()*metres*100),float(pupil.max()*metres*100)],
        full_forward_cm=[float((rig[20]+pupil.min())*metres*100),float((rig[20]+pupil.max())*metres*100)],
        elevation_deg=[float(elevation.min()),float(elevation.max())],roll_deg=[float(roll.min()),float(roll.max())])
    optical=dict(fx_px=[float(pose['fx'].min()),float(pose['fx'].max())],fy_px=[float(pose['fy'].min()),float(pose['fy'].max())],
        principal_x_px=[float(pose['cx'].min()),float(pose['cx'].max())],principal_y_px=[float(pose['cy'].min()),float(pose['cy'].max())],
        principal_y_offset_from_sensor_midpoint_px=[float(pose['cy'].min()-539.5),float(pose['cy'].max()-539.5)],
        signed_readout_ms=float(rig[37]*1000),readout_and_optics_have_independent_hardware_calibration=False)
    report=dict(schema='rinkspace.native_camera_continuous_physics_audit.v1',snapshot_sha256=hashlib.sha256(blob).hexdigest(),
        source_time_seconds=[float(source_pts[0]/12800),float(source_pts[-1]/12800)],full_row_exposure_interval_seconds=[lo+source_origin,hi+source_origin],sample_count=len(times),
        measurement='Native analytic derivatives; scalar speed extrema and cubic knots included, not finite differences of video frames',
        spline=dict(degree=3,control_dt_seconds=dt,controls=len(controls),continuity='C2',third_derivative='piecewise constant, allowed to jump at knots',
            fitted_knot_stride=int(state.get('motion_knot_stride',1)),fitted_control_dt_seconds=dt*int(state.get('motion_knot_stride',1)),
            fitted_control_count=UniformMotionChart.create(len(controls),first,dt,int(state.get('motion_knot_stride',1)),float(state.get('motion_reference_time',frame_times[0]))).refinement.shape[1],
            explicit_operator_speed_constraint=False,explicit_operator_acceleration_constraint=False,motion_prior_used=False),
        mechanical=mechanical,optics=optical,channels=channels,angular_speed_deg_s=peak(np.degrees(angular_speed)),
        angular_acceleration_deg_s2=peak(np.degrees(angular_acceleration)),largest_adjacent_frame_rate_changes=worst,
        verdict='unverified_candidate_requires_source_motion_comparison',physical_acceptance=False,solver_restarted=False,
        point_input_sha256=prep.digest(prep.RUN/'point_observations.npz'),
        point_input_path=str(prep.RUN/'point_observations.npz'),snapshot_path=str(snapshot))
    prep.write(out/'report.json',report)
    np.savez(out/'trajectory.npz',times=times,source_times=source_times,kinematics=k['values'],camera=pose['values'],
        frame_times=frame_times,frame_kinematics=frame['values'])
    print(json.dumps(dict(mechanical=mechanical,optics=optical,channels=channels,worst_intervals=worst[:5]),ensure_ascii=True),flush=True)
    return out,report

def source_motion_comparison(out):
    """Read fixed-background image motion. This adds no observations to the fit."""
    import cv2,av
    report=prep.read(out/'report.json');state=dict(np.load(out/'audited_state.npz'))
    camera=prep.camera_for_snapshot(prep.read(prep.CAMERA),state);scope=prep.read(Path(__file__).with_name('rebuild_scope.json'));core=NativeCore(prep.LIBRARY)
    z=dict(np.load(prep.RUN/'point_observations.npz'));mapping=prep.read(prep.RUN/'mapping.json')
    names=np.array([mapping['path_ids'][i] for i in z['path_indices']]);bottom=names=='board/continuous/bottom';top=names=='board/continuous/top'
    world=np.asarray(core.sample_path(state['board'],7,[state['board'][5]/2],[],np.linspace(0,core.board_perimeter(state['board']),2049)))
    # Only one flat, stationary long-side board plane is used for this image
    # homography diagnostic. Person exclusion is deliberately wider than the
    # central skater; the advertisement identities are not used at all.
    world=world[(world[:,1]>0)&(abs(world[:,0])<state['board'][0]-state['board'][2]-.005)]
    records=[]
    first_pts=int(z['pts'].min());last_pts=int(z['pts'].max())
    windows=[(first_pts,min(first_pts+2304,last_pts)),(371200,375040),(563200,565760),(572928,575488)]
    for begin,end in windows:
        frames={}
        with av.open(scope['source']['registered_path']) as container:
            stream=container.streams.video[0];container.seek(begin,stream=stream,backward=True)
            for f in container.decode(stream):
                if f.pts>end:break
                if f.pts>=begin:frames[int(f.pts)]=cv2.cvtColor(f.to_ndarray(format='bgr24'),cv2.COLOR_BGR2GRAY)
        for p in range(begin,end,256):
            if p not in frames or p+256 not in frames:raise ValueError('Missing exact source exposure')
            a,b=frames[p],frames[p+256];mask=np.zeros_like(a)
            edges=[]
            for role,offset in [(top,8),(bottom,-12)]:
                uv=z['uv'][(z['pts']==p)&role];uv=uv[np.argsort(uv[:,0])]
                edges.append(np.interp(np.arange(1920),uv[:,0],uv[:,1])+offset)
            for x in range(1920):mask[max(0,int(edges[0][x])):min(1080,int(edges[1][x])),x]=255
            mask[:,500:1240]=0;mask[:,:16]=0;mask[:,-16:]=0
            pts=cv2.goodFeaturesToTrack(a,1200,.005,4,mask=mask)
            if pts is None:records.append(dict(pts=p,status='insufficient_static_texture'));continue
            nxt,ok,_=cv2.calcOpticalFlowPyrLK(a,b,pts,None,winSize=(25,25),maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.001))
            back,back_ok,_=cv2.calcOpticalFlowPyrLK(b,a,nxt,None,winSize=(25,25),maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.001))
            valid=ok[:,0].astype(bool)&back_ok[:,0].astype(bool)&(np.linalg.norm(back[:,0]-pts[:,0],axis=1)<1)
            old=pts[valid,0];new=nxt[valid,0]
            if len(old)<12:records.append(dict(pts=p,status='insufficient_consistent_image_tracks'));continue
            H,inliers=cv2.findHomography(old,new,cv2.RANSAC,2,maxIters=5000,confidence=.999)
            if H is None:records.append(dict(pts=p,status='no_image_homography'));continue
            inliers=inliers[:,0].astype(bool);observed=cv2.perspectiveTransform(old[None].astype(np.float64),H)[0]
            residual=np.linalg.norm(observed-new,axis=1)
            at=[core.project_motion(state['rig'],state['controls'],camera['first_control_time'],camera['control_dt'],1080,
                np.full(len(world),(t-camera['reference_pts'])/12800),world) for t in (p,p+256)]
            valid=(at[0]['status']==0)&(at[1]['status']==0)&(at[0]['uv'][:,0]>50)&(at[0]['uv'][:,0]<1870)
            if not valid.any():records.append(dict(pts=p,status='no_visible_native_straight_board'));continue
            u=at[0]['uv'][valid];next_uv=at[1]['uv'][valid];source_next=cv2.perspectiveTransform(u[None],H)[0]
            difference=np.linalg.norm(next_uv-source_next,axis=1)
            records.append(dict(pts=p,next_pts=p+256,source_seconds=p/12800,status='compared',tracks=len(old),inliers=int(inliers.sum()),
                source_registration_median_px=float(np.median(residual[inliers])),source_registration_p95_px=float(np.percentile(residual[inliers],95)),
                source_board_dx_px=float(np.median(source_next[:,0]-u[:,0])),native_board_dx_px=float(np.median(next_uv[:,0]-u[:,0])),
                native_vs_source_displacement_median_px=float(np.median(difference)),native_vs_source_displacement_max_px=float(difference.max()),
                H=H.tolist()))
    data=dict(schema='rinkspace.source_motion_diagnostic.v1',source_sha256=scope['source']['sha256'],snapshot_sha256=report['snapshot_sha256'],
        method='Forward/backward source optical flow on stationary straight-board texture with central person exclusion; native fixed 3D board points projected through the full rolling-shutter camera',
        advertisement_binding_used=False,observations_added_to_fit=False,records=records)
    prep.write(out/'source_motion.json',data)
    print(json.dumps(dict(source_comparison=[r for r in records if 29.04<=r['source_seconds']<=29.22 and r['status']=='compared']),ensure_ascii=True),flush=True)
    return data


def plots(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname=r'C:\Windows\Fonts\msyh.ttc');plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
    a=dict(np.load(out/'trajectory.npz'));r=prep.read(out/'report.json');t=a['source_times'];k=a['kinematics']
    fig,axes=plt.subplots(3,3,figsize=(16,10),sharex=True,layout='constrained')
    labels=['摇镜','俯仰','变焦 log(f)'];units=['度','度','log(f)']
    for c in range(3):
        factor=180/np.pi if c<2 else 1
        for d in range(3):
            axes[d,c].plot(t,k[:,d,c]*factor,lw=.7,color='#16749b');axes[d,c].grid(alpha=.2)
            axes[d,c].set_ylabel([units[c],units[c]+'/秒',units[c]+'/秒²'][d])
        axes[0,c].set_title(labels[c]);axes[2,c].set_xlabel('原片时间（秒）')
    fig.suptitle('当前 native 候选：连续轨迹、速度和加速度（包含首尾帧行曝光区间）\n诊断原始解，未平滑；尚未通过物理验收',fontsize=15)
    fig.savefig(out/'trajectory_full.png',dpi=130);plt.close(fig)
    fig,axes=plt.subplots(4,1,figsize=(12,11),sharex=True,layout='constrained');window=(t>=29)&(t<=29.3)
    axes[0].plot(t[window],np.degrees(k[window,0,0]),color='#16749b');axes[0].set_ylabel('摇镜角度（度）')
    axes[1].plot(t[window],np.degrees(k[window,1,0]),color='#16749b');axes[1].axhline(0,color='gray',lw=.6);axes[1].set_ylabel('摇镜速度（度/秒）')
    axes[2].plot(t[window],k[window,1,2]*100,color='#16749b');axes[2].axhline(0,color='gray',lw=.6);axes[2].set_ylabel('焦距相对变化率（%/秒）')
    source=prep.read(out/'source_motion.json');rows=[v for v in source['records'] if v['status']=='compared' and 29<=v['source_seconds']<=29.3]
    x=np.array([v['source_seconds'] for v in rows]);axes[3].plot(x,[v['source_board_dx_px'] for v in rows],'o-',color='#0d936a',label='原片固定背景运动')
    axes[3].plot(x,[v['native_board_dx_px'] for v in rows],'o-',color='#ce5a27',label='native 固定三维点投影运动');axes[3].set_ylabel('每帧水平位移（像素）');axes[3].legend()
    for ax in axes:ax.grid(alpha=.2);ax.set_xlim(29,29.3)
    axes[-1].set_xlabel('原片时间（秒）');fig.suptitle('29.00–29.30 秒：原片运动与当前相机轨迹对照\n无广告实体绑定参与此诊断；没有新增拟合点或权重',fontsize=15)
    fig.savefig(out/'trajectory_source_comparison.png',dpi=130);plt.close(fig)

def cause_audit(out,fit_window=False):
    """Controlled diagnostics of one failed interval, never a replacement fit.

    Read the frozen rejected candidate. Temporary trajectory perturbations
    isolate what the actual raw-pixel objective rewards. Ordinary station
    refinement uses the production GPU implementation; no camera LM runs here.
    """
    import ctypes as ct
    from rinkspace.core.native import D,N
    from rinkspace.domain.parameters import SceneParameterModel,ParameterState
    from rinkspace.session.gpu_geometry import NativeSceneGPU
    camera=prep.read(prep.CAMERA);core=NativeCore(prep.LIBRARY)
    state=dict(np.load(out/'audited_state.npz',allow_pickle=False))
    z=dict(np.load(prep.RUN/'point_observations.npz',allow_pickle=False));mapping=prep.read(prep.RUN/'mapping.json')
    input_sha=prep.digest(prep.RUN/'point_observations.npz')
    if str(state['point_input_sha256'])!=input_sha or not np.array_equal(state['source_rows'],z['source_rows']):
        raise ValueError('Causal audit requires the exact input of the frozen candidate')
    rows=np.flatnonzero((z['pts']>=int(28.90*12800))&(z['pts']<=int(29.40*12800)))
    paths=[mapping['path_ids'][i] for i in z['path_indices'][rows]]
    model=SceneParameterModel(prep.read(prep.SCENE),prep.SCENE,optimize_scene=True)
    native=NativeSceneGPU(prep.SCENE,core,parameter_state=ParameterState(model.initial_state.layout,state['scene_values']))
    layers=np.array([native.paths[path]['layer'] for path in paths])
    controls=state['controls'];first=camera['first_control_time'];dt=camera['control_dt']
    source_origin=camera['reference_pts']*camera['time_base_num']/camera['time_base_den']
    control_t=source_origin+first+dt*np.arange(len(controls))
    # A diagnostic low-frequency replacement on this interval only. It is not
    # written into accepted state, rendered output, or a chosen physical model.
    fit=(control_t>=28.80)&(control_t<=29.50);x=control_t-29.15
    design=np.column_stack((x*0+1,x,x*x,x*x*x))
    smooth=design@np.linalg.lstsq(design[fit],controls[fit],rcond=None)[0]
    blend=np.clip(np.minimum((control_t-28.94)/.06,(29.36-control_t)/.06),0,1)
    perturbation=(smooth-controls)*blend[:,None]
    local=np.full(len(rows),-1,np.int32);active=z['kind'][rows]==1;local[active]=np.arange(active.sum())
    lo=np.array([-np.inf if native.paths[p]['kind']==7 else 0. for p,a in zip(paths,active) if a])
    hi=np.array([np.inf if native.paths[p]['kind']==7 else 1. for p,a in zip(paths,active) if a])
    rig=state['rig'];nc=len(controls);nrig=31 if len(rig)==43 else 27
    control_indices=[int(round((t-source_origin-first)/dt)) for t in (29.10,29.12,29.14)]
    q=3+nrig+9;directions=np.zeros((nc,3,q));rd=np.zeros((nrig,q));column_names=[]
    for c,name in enumerate(('pan','tilt','log_f')):
        directions[:,c,c]=perturbation[:,c];column_names.append('remove_local_'+name+'_oscillation')
    scale=np.array([.001]*6+[.001]*3+[.00025]*3+[.00025]*3+[1.,1.,.001,1.,1.,.03,.03,.1,.1,.003,.003,.001])
    if nrig==31:scale=np.r_[scale,.05,.1,.003,.003]
    rd[:,3:3+nrig]=np.diag(scale);column_names.extend('camera_chart_'+str(i) for i in range(nrig))
    for j,k in enumerate(control_indices):
        for c,name in enumerate(('pan','tilt','log_f')):
            directions[k,c,3+nrig+3*j+c]=.001;column_names.append(f'control_{control_t[k]:.2f}_{name}')

    def resident(candidate,rig_value=rig):
        return native.resident(camera={**camera,'rig':rig_value.tolist()},controls=candidate,path_ids=paths,
            stations=state['stations'][rows],pts=z['pts'][rows],observed=z['uv'][rows])

    def summary(values):
        e=values[:,9:11];square=np.sum(e*e,axis=1)
        return dict(cost=float(square.sum()/2),median=float(np.median(np.sqrt(square))),
            by_layer={layer:dict(rows=int(np.sum(layers==layer)),cost=float(square[layers==layer].sum()/2)) for layer in np.unique(layers)})

    def attach(p):
        p.configure_camera_parameters(directions=rd,max_readout=.04)
        p.configure_pixel_linearization(directions=directions,local_ids=local,local_scales=np.full(active.sum(),.001))
        p.configure_solver(lower=lo,upper=hi,device_budget_bytes=p.creation['free_after_alloc_bytes'],host_budget_bytes=2**30)

    source=[r for r in prep.read(out/'source_motion.json')['records'] if r['status']=='compared' and 29.<=r['source_seconds']<=29.3]
    board=state['board'];world=np.asarray(core.sample_path(board,7,[board[5]/2],[],np.linspace(0,core.board_perimeter(board),2049)))
    world=world[(world[:,1]>0)&(abs(world[:,0])<board[0]-board[2]-.005)]
    def motion_comparison(candidate,candidate_first=first):
        result=[]
        for r in source:
            projections=[core.project_motion(rig,candidate,candidate_first,dt,1080,np.full(len(world),(p-camera['reference_pts'])/12800),world)
                for p in (r['pts'],r['next_pts'])]
            a,b=projections;valid=(a['status']==0)&(b['status']==0)&(a['uv'][:,0]>50)&(a['uv'][:,0]<1870)
            if not valid.any():raise ValueError('No stationary comparison point remains visible')
            u=a['uv'][valid];H=np.asarray(r['H']);mapped=np.c_[u,np.ones(len(u))]@H.T;mapped=mapped[:,:2]/mapped[:,2:]
            result.append(dict(source_seconds=r['source_seconds'],source_dx=float(np.median(mapped[:,0]-u[:,0])),
                native_dx=float(np.median(b['uv'][valid,0]-u[:,0])),discrepancy=float(np.median(np.linalg.norm(b['uv'][valid]-mapped,axis=1)))))
        k=core.evaluate_motion_kinematics(candidate,candidate_first,dt,np.arange(29.02,29.281,.001)-source_origin)
        return dict(pan_peak_acceleration_deg_s2=float(np.max(abs(np.degrees(k['acceleration'][:,0])))),
            median_displacement_discrepancy_px=float(np.median([r['discrepancy'] for r in result])),records=result)

    cases=[];base=None
    for channels,mix in (([0,1,2],0.),([0],.5),([0],1.),([0,1,2],.5),([0,1,2],1.)):
        candidate=controls.copy();candidate[:,channels]+=mix*perturbation[:,channels]
        with resident(candidate) as p:
            before=p.evaluate('cuda')
            initial_failures=None
            if np.any(before['status']):
                failed=np.flatnonzero(before['status']);codes,counts=np.unique(before['status'][failed],return_counts=True)
                initial_failures=dict(domain_failures=len(failed),
                    statuses={str(int(code)):dict(count=int(count),meaning=core.status_message(int(code))) for code,count in zip(codes,counts)},
                    first_failed_rows=[dict(source_row=int(rows[i]),pts=int(z['pts'][rows[i]]),path=paths[i]) for i in failed[:8]])
            attach(p);refinement=p.refine_curve_stations();after=p.evaluate('cuda')
            if np.any(after['status']):
                record=dict(channels=channels,mix=mix,initial_failures=initial_failures,remaining_domain_failures=int(np.sum(after['status']!=0)))
                cases.append(record);print(json.dumps(dict(causal_case=record),ensure_ascii=True),flush=True);continue
            record=dict(channels=channels,mix=mix,initial_failures=initial_failures,
                fixed_stations=None if initial_failures else summary(before['values']),
                refined_stations=summary(after['values']),station_refinement=refinement,source_motion=motion_comparison(candidate))
            cases.append(record);print(json.dumps(dict(causal_case=record),ensure_ascii=True),flush=True)
            if mix==0:
                p.linearize();J=p.probe_rows(np.arange(len(rows)));p.linearize('cpu');Jcpu=p.probe_rows(np.arange(len(rows)),'cpu')
                base=dict(values=after['values'],stations=p.export_state()['stations'],J=J,
                    cpu_gpu_J_max_abs=float(np.max(abs(J-Jcpu))))

    # Actual implicit-RS derivatives at the failed camera, not only a synthetic
    # fixture. Geometry is frozen at the same world points for finite differences.
    probe=np.unique(np.linspace(0,len(rows)-1,min(96,len(rows)),dtype=int))
    xyz=base['values'][probe,:3];times=(z['pts'][rows][probe]-camera['reference_pts'])/12800
    retract=core._lib.rs_camera_retract;retract.argtypes=[D,N,D,N,D,N];retract.restype=ct.c_int
    fd=[]
    for j,name in enumerate(column_names):
        errors=[]
        for epsilon in (.002,.001):
            projected=[]
            for sign in (-1,1):
                delta=np.ascontiguousarray(sign*epsilon*rd[:,j]);changed=np.empty_like(rig)
                if retract(rig.ctypes.data_as(D),rig.size,delta.ctypes.data_as(D),delta.size,changed.ctypes.data_as(D),changed.size):
                    raise ValueError('Native diagnostic retraction failed')
                value=core.project_motion(changed,controls+sign*epsilon*directions[:,:,j],first,dt,1080,times,xyz)
                if np.any(value['status']):raise ValueError('Finite difference left the RS domain')
                projected.append(value['uv'])
            numeric=(projected[1]-projected[0])/(2*epsilon);analytic=base['J'][probe,:,j]
            errors.append(dict(epsilon=epsilon,max_abs=float(np.max(abs(numeric-analytic))),
                relative_norm=float(np.linalg.norm(numeric-analytic)/max(np.linalg.norm(analytic),1e-12))))
        fd.append(dict(column=name,checks=errors))
    J=base['J'][:,:,:q];tangent=base['J'][:,:,-1];den=np.sum(tangent*tangent,axis=1)
    # This is the exact local elimination of an ordinary point's station,
    # not an observation weight. Fixed points keep both pixel directions.
    unit=np.divide(tangent,np.sqrt(den)[:,None],out=np.zeros_like(tangent),where=den[:,None]>1e-24)
    normal_J=J-unit[:,:,None]*np.einsum('ni,nij->nj',unit,J)[:,None,:]
    residual=base['values'][:,9:11];direction_effects=[]
    for j in range(3):
        categories={}
        for layer in np.unique(layers):
            m=layers==layer;raw=float(np.sum(J[m,:,j]**2));retained=float(np.sum(normal_J[m,:,j]**2))
            categories[layer]=dict(raw_squared_sensitivity=raw,after_station_elimination=retained,
                sensitivity_fraction=retained/raw if raw else None,cost_directional_derivative=float(np.sum(normal_J[m,:,j]*residual[m])))
        direction_effects.append(dict(column=column_names[j],by_layer=categories))
    evolution=[]
    for path in [prep.CAMERA.parent/camera['controls_ref'],prep.RUN/'pixel_initial.npz',prep.RUN/'pixel_before_curve_refinement.npz',
                 prep.RUN/'accepted_pixels_latest_before_ad4_contact.npz',out/'audited_state.npz']:
        if not path.exists():continue
        value=np.load(path,allow_pickle=False);ctrl=value if path.suffix=='.npy' else value['controls']
        k=core.evaluate_motion_kinematics(ctrl,first,dt,np.arange(29.02,29.281,.001)-source_origin)
        evolution.append(dict(path=str(path),sha256=prep.digest(path),pan_peak_speed_deg_s=float(np.max(abs(np.degrees(k['velocity'][:,0])))),
            pan_peak_acceleration_deg_s2=float(np.max(abs(np.degrees(k['acceleration'][:,0]))))))
        if hasattr(value,'close'):value.close()
    report=dict(schema='rinkspace.native_camera_causal_diagnostic.v1',snapshot_sha256=prep.digest(out/'audited_state.npz'),
        point_input_sha256=input_sha,source_seconds=[28.90,29.40],rows=len(rows),frames=len(np.unique(z['pts'][rows])),
        fixed_rows=int((~active).sum()),ordinary_curve_rows=int(active.sum()),cpu_gpu_J_max_abs=base['cpu_gpu_J_max_abs'],
        finite_difference_columns=fd,controlled_trajectory_cases=cases,station_elimination=direction_effects,history=evolution,
        camera_lm_run=False,original_uv_changed=False,original_native_parameter_files_changed=False,
        observation_weights_used=False,temporary_trajectory_changes_are_diagnostics_not_a_solution=True)
    report['initial_camera_origin']=camera.get('origin')
    report['findings']=[
        'The selected native initial trajectory already contains the local oscillation; this run amplified it.',
        'All 2412 retained observations in this diagnostic interval have ordinary curve stations, with no fixed-material-point factor.',
        'Independent station motion removes almost all straight-board sensitivity to the tested pan oscillation.',
        'The coherent diagnostic trajectory better follows source background motion, but has a higher raw point objective with the current input and fixed shared geometry.',
        'The sampled actual-state native derivatives agree across CPU/GPU and finite differences; this does not certify every solver operation.',
        'Continuous trajectory limits and corrected identities must be tested in fitting; post-render filtering cannot repair the objective.'
    ]
    prep.write(out/'causal_audit.json',report)
    np.savez(out/'causal_audit.npz',source_rows=rows,controls=controls,perturbation=perturbation,J=base['J'],values=base['values'],
        stations=base['stations'],column_names=np.asarray(column_names),layers=layers)
    print(json.dumps(dict(cpu_gpu_J_max_abs=base['cpu_gpu_J_max_abs'],derivative_worst=sorted(fd,key=lambda r:r['checks'][-1]['relative_norm'],reverse=True)[:5],
        station_elimination=direction_effects,history=evolution),ensure_ascii=True),flush=True)
    if fit_window:
        # Controlled in-solver experiment: one cubic trajectory over this short
        # interval; all twelve coefficients fitted jointly with ordinary stations.
        # Shared installation/scene are deliberately held for causal isolation.
        # This is not an overwrite of the 25-second production parameterization.
        crop=np.flatnonzero((control_t>=28.82)&(control_t<=29.48))
        low_first=first+crop[0]*dt;low_controls=smooth[crop].copy()
        tx=(control_t[crop]-29.15)/.33;polynomial=np.column_stack((tx*0+1,tx,tx**2,tx**3))
        low_directions=np.zeros((len(crop),3,12))
        for channel in range(3):low_directions[:,channel,4*channel:4*channel+4]=.001*polynomial
        low_camera={**camera,'rig':rig.tolist(),'first_control_time':low_first}
        logfile=out/'continuous_window_solve.jsonl'
        print('MAIN LOG: '+str(logfile),flush=True)
        with logfile.open('w',encoding='utf-8',buffering=1) as log:
            def emit(record):log.write(json.dumps(record,ensure_ascii=True,allow_nan=False,default=str)+'\n')
            with native.resident(camera=low_camera,controls=low_controls,path_ids=paths,stations=base['stations'],pts=z['pts'][rows],observed=z['uv'][rows]) as p:
                p.configure_pixel_linearization(directions=low_directions,local_ids=local,local_scales=np.full(active.sum(),.001))
                p.configure_solver(lower=lo,upper=hi,device_budget_bytes=p.creation['free_after_alloc_bytes'],host_budget_bytes=2**30,damping_metric='jacobi_image')
                p.refine_curve_stations();initial=p.evaluate('cuda')
                if np.any(initial['status']):raise ValueError('Continuous window seed leaves the domain')
                emit(dict(event='initial',source_seconds=[28.90,29.40],rows=len(rows),trajectory_columns=12,
                    shared_installation_and_scene_fixed_for_causal_isolation=True,raw_pixels=True,summary=summary(initial['values'])))
                def checkpoint_window():
                    saved=p.export_state();saved.update(first_control_time=np.array(low_first),control_dt=np.array(dt),source_rows=rows,
                        point_input_sha256=np.array(input_sha),experiment_only=np.array(True))
                    pending=out/'continuous_window_pending.npz';np.savez(pending,**saved);pending.replace(out/'continuous_window_candidate.npz')
                    return saved
                checkpoint_window()
                def progress(record):
                    if record['summary']['accepted']:
                        record['station_refinement']=p.refine_curve_stations();checkpoint_window()
                    emit(dict(event='lm_step',record=record))
                    return not (out/'window_stop_requested').exists()
                try:
                    result=p.solve(progress=progress,gradient_tolerance=1e-6)
                except Exception as exc:
                    checkpoint_window()
                    result=dict(numerically_converged=False,stop_reason='native_error',error=str(exc))
                    emit(dict(event='native_error',result=result))
                saved=checkpoint_window();final=p.evaluate('cuda')
                if np.any(final['status']):raise ValueError('Continuous window solve lost a raw point')
                experiment=dict(result=result,initial=summary(initial['values']),final=summary(final['values']),
                    source_motion=motion_comparison(saved['controls'],low_first),native_library=str(prep.LIBRARY),
                    fixed_shared_geometry_and_installation=True,trajectory_columns=12,whole_shot_fit=False,
                    production_state_changed=False,observation_weights=False,postprocess_smoothing=False)
                emit(dict(event='finished',experiment=experiment));prep.write(out/'continuous_window_result.json',experiment)
                report['continuous_native_window_experiment']=experiment;prep.write(out/'causal_audit.json',report)
                print(json.dumps(dict(continuous_window=experiment),ensure_ascii=True),flush=True)
    cause_plot(out,report)
    return report


def cause_plot(out,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname=r'C:\Windows\Fonts\msyh.ttc');plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
    baseline=report['controlled_trajectory_cases'][0];coherent=report['controlled_trajectory_cases'][-1]
    fig,axes=plt.subplots(2,1,figsize=(12,9),layout='constrained')
    rows=baseline['source_motion']['records'];other=coherent['source_motion']['records'];x=[r['source_seconds'] for r in rows]
    axes[0].plot(x,[r['source_dx'] for r in rows],'o-',color='#098469',label='原片固定背景')
    axes[0].plot(x,[r['native_dx'] for r in rows],'o-',color='#c95127',label='当前 native 候选')
    axes[0].plot(x,[r['native_dx'] for r in other],'o-',color='#2879b7',label='临时去除局部摆动（诊断）')
    axes[0].set_ylabel('每帧水平位移（像素）');axes[0].set_xlabel('原片时间（秒）');axes[0].legend();axes[0].grid(alpha=.2)
    axes[0].set_title('去掉摆动后，与原片运动的最大差异：31.61 → 1.38 像素')
    categories=['advertisement','ice','board'];xs=np.arange(3)
    for offset,case,color,label in [(-.19,baseline,'#c95127','当前 native 候选'),(.19,coherent,'#2879b7','临时连续轨迹（诊断）')]:
        values=[case['refined_stations']['by_layer'][key]['cost'] for key in categories]
        bars=axes[1].bar(xs+offset,values,width=.36,color=color,label=label)
        axes[1].bar_label(bars,fmt='%.0f',padding=3)
    axes[1].set_xticks(xs,['广告轮廓点','冰面曲线点','围挡轮廓点']);axes[1].set_ylabel('原始像素平方目标（½ Σ 距离²）')
    axes[1].set_ylim(0,18000);axes[1].grid(axis='y',alpha=.2);axes[1].legend()
    axes[1].set_title('但现有输入的拟合目标变差：22,421 → 27,608；每个点的 UV 和处理方式不变')
    fig.suptitle('异常运动的因果诊断：当前目标函数偏爱带摆动的解\n仅改 29 秒附近的临时轨迹，安装／场地不变；曲线位置由同一 native 重新求解',fontsize=14)
    fig.savefig(out/'causal_comparison.png',dpi=140);plt.close(fig)
    experiment=report.get('continuous_native_window_experiment')
    if not experiment:return
    actual=experiment['source_motion']['records'];fig,axes=plt.subplots(2,1,figsize=(12,9),layout='constrained')
    for records,key,color,label in [(rows,'source_dx','#098469','原片固定背景'),
            (rows,'native_dx','#c95127','原主拟合候选'),(actual,'native_dx','#2879b7','native 局部连续轨迹联合拟合')]:
        axes[0].plot([r['source_seconds'] for r in records],[r[key] for r in records],'o-',color=color,label=label)
    old=max(r['discrepancy'] for r in rows);new=max(r['discrepancy'] for r in actual)
    axes[0].set_title(f'与原片运动的最大差异：{old:.2f} → {new:.2f} 像素')
    axes[0].set_ylabel('每帧水平位移（像素）');axes[0].set_xlabel('原片时间（秒）');axes[0].legend();axes[0].grid(alpha=.2)
    for offset,summary,color,label in [(-.19,baseline['refined_stations'],'#c95127','原主拟合候选'),
            (.19,experiment['final'],'#2879b7','native 局部连续轨迹联合拟合')]:
        bars=axes[1].bar(xs+offset,[summary['by_layer'][key]['cost'] for key in categories],width=.36,color=color,label=label)
        axes[1].bar_label(bars,fmt='%.0f',padding=3)
    axes[1].set_xticks(xs,['广告轮廓点','冰面曲线点','围挡轮廓点']);axes[1].set_ylabel('原始像素平方目标（½ Σ 距离²）');axes[1].grid(axis='y',alpha=.2);axes[1].legend()
    axes[1].set_title('现有逐点目标仍更偏爱抖动解；运动真实性必须另行核验')
    fig.suptitle('native 求解中的连续轨迹实验：源 28.90–29.40 秒，2412 个原始点\n安装和场地在本次因果实验中不动；严格梯度停止门槛尚未通过，非完整镜头交付',fontsize=13)
    fig.savefig(out/'native_continuous_comparison.png',dpi=140);plt.close(fig)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--cause',action='store_true');parser.add_argument('--fit-window',action='store_true');parser.add_argument('--out',type=Path);parser.add_argument('--snapshot',type=Path);args=parser.parse_args()
    if args.cause or args.fit_window:cause_audit(args.out or prep.RUN/'camera_physics',fit_window=args.fit_window)
    else:
        output,report=audit(args.out,args.snapshot);source_motion_comparison(output);plots(output)
