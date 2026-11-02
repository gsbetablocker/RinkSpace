"""Transfer the current view into existing pan/tilt control coordinates.

The old native projection is an initializer target, never a source observation.
There is no per-frame centre or roll, no source UV rewrite and no scene update.
"""
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix,bmat,diags
from scipy.sparse.linalg import lsqr
import prepare_native_multicamera as p
from initialize_camera_repair import evaluate,extent
from rinkspace.domain.motion_chart import UniformMotionChart,cubic_samples
from rinkspace.domain.multi_shot_motion import continue_exposure_halos
from rinkspace.domain.parameters import ParameterState
from rinkspace.session.gpu_geometry import NativeSceneGPU
from fit_native_s01_joint import mount_configuration


def transfer_projection(m,z,mapping,model,core,timeline,previous,candidate,current,working_envelope,*,channels=('pan','tilt'),backend='cpu'):
    names=('pan','tilt','log_f');axes=[names.index(name) for name in channels]
    if not axes or len(set(axes))!=len(axes) or backend not in ('cpu','cuda'):raise ValueError('Invalid view-transfer channels/backend')
    dimension=len(axes)
    target=previous['values'][:,3:5];state={k:v.copy() for k,v in candidate.items()}
    camera={**p.read(m['cameras'][0]['camera_definition']),'first_control_time':timeline.first,'control_dt':timeline.dt}
    camera_ids=np.empty(len(z['pts']),np.int32)
    for shot in m['shots']:camera_ids[shot['row_begin']:shot['row_stop']]=shot['camera_index']
    paths=np.asarray(mapping['path_ids'])[z['path_indices']].tolist();frames=np.unique(z['pts']);frame_ids=np.searchsorted(frames,z['pts'])
    metres=p.read(p.ROOT/'events/ankara_2026/model_constraints/ankara_2026_metric_scale_assumed_v1.json')['converted_span_xy_m'][0]
    native=NativeSceneGPU(m['scene_path'],core,parameter_state=ParameterState(model.initial_state.layout,state['scene_values']))
    bases=[]
    for shot in m['shots']:
        count=shot['control_stop']-shot['control_begin'];chart=UniformMotionChart.create(count,shot['first_control_time'],timeline.dt,shot['knot_stride'],shot['motion_reference_time'])
        raw=np.zeros((count,3,chart.refinement.shape[1]));raw[:,0,:]=chart.refinement
        interval=timeline.local_exposure_time(shot,np.asarray([shot['first_pts'],shot['last_pts']]))
        basis=continue_exposure_halos(raw,shot['first_control_time'],timeline.dt,interval)[:,0,:]
        channel_bases=[basis for _ in axes]
        if shot.get('local_motion_refinement'):
            block,labels=chart.directions(reference_angle_gauge=False)
            block=continue_exposure_halos(block,shot['first_control_time'],timeline.dt,interval)
            keep=np.any(block!=0.,axis=(0,1));block=block[:,:,keep];labels=[v for v,k in zip(labels,keep) if k]
            block,labels=timeline._local_refinement_directions(shot,block,labels,reference_angle_gauge=False,closed=True)
            channel_bases=[block[:,axis,[j for j,label in enumerate(labels) if label[1]==axis]].copy() for axis in axes]
        selected=np.flatnonzero((frames>=shot['first_pts'])&(frames<=shot['last_pts']))
        times=timeline.local_exposure_time(shot,frames[selected])
        indices,weights=cubic_samples(count,shot['first_control_time'],timeline.dt,times)
        samples=[csr_matrix(np.einsum('ij,ijk->ik',weights,B[indices])) for B in channel_bases]
        bases.append((shot,channel_bases,selected,samples))
    def certificates(controls):
        for i,rig in enumerate(state['rig']):
            _,_,e,_,_=mount_configuration(metres,rig,working_envelope)
            windows=timeline.frame_windows()[[j for j,s in enumerate(m['shots']) if s['camera_index']==i]]
            limits=[*e['forward_bounds'],*e['pupil_bounds'],e['max_pan_tilt'],e['max_tilt_deviation'],*e['elevation_bounds'],*e['roll_bounds'],*([np.inf]*9),*((windows-timeline.first)/timeline.dt).ravel()]
            cert=core.evaluate_camera_envelope(rig,controls,timeline.first,timeline.dt,limits,joined_zoom_domain=True)
            if cert['margin'] < -128*np.finfo(float).eps:return False
        return True
    history=[];error=current['values'][:,3:5]-target;cost=float(np.sum(error*error));initial=extent(np.linalg.norm(error,axis=1))
    while True:
        # Constant local-angle directions give the native implicit rolling-row
        # derivative at every row. They are sampled per exposure below; the
        # actual update remains the existing independent-shot cubic controls.
        directions=np.zeros((len(state['controls']),3,dimension))
        for j,axis in enumerate(axes):directions[:,axis,j]=.001
        with native.resident(camera=camera,rigs=state['rig'],camera_ids=camera_ids,controls=state['controls'],path_ids=paths,stations=state['stations'],pts=z['pts'],observed=z['uv'],motion_times=timeline.row_times(z['pts'])) as resident:
            resident.configure_projection_time_support(half_seconds=m['max_readout']/2)
            resident.configure_camera_parameters(directions=np.zeros((31*len(state['rig']),dimension)),max_readout=m['max_readout'])
            resident.configure_pixel_linearization(directions=directions,local_ids=np.full(len(z['pts']),-1,np.int32),local_scales=np.zeros(0))
            resident.linearize(backend)
            jac=resident.probe_rows(np.arange(len(z['pts']),dtype=np.int32),backend=backend)[:,:,:dimension].copy()
        H=np.zeros((len(frames),dimension,dimension));b=np.zeros((len(frames),dimension))
        for j in range(dimension):
            np.add.at(b[:,j],frame_ids,np.einsum('ij,ij->i',jac[:,:,j],-error))
            for k in range(dimension):np.add.at(H[:,j,k],frame_ids,np.einsum('ij,ij->i',jac[:,:,j],jac[:,:,k]))
        del jac
        solution=np.einsum('nij,nj->ni',np.linalg.pinv(H,rcond=128*np.finfo(float).eps,hermitian=True),b)*.001
        eigenvalues,eigenvectors=np.linalg.eigh(H);whiten=np.sqrt(np.maximum(eigenvalues,0.))[:,:,None]*eigenvectors.transpose(0,2,1)
        delta=np.zeros_like(state['controls'])
        for shot,channel_bases,selected,samples in bases:
            C=whiten[selected];rhs=np.einsum('nij,nj->ni',C,solution[selected])
            A=bmat([[diags(C[:,j,k])@samples[k] for k in range(dimension)] for j in range(dimension)],format='csr')
            fitted=lsqr(A,rhs.T.ravel(),atol=1e-12,btol=1e-12)[0]
            sl=slice(shot['control_begin'],shot['control_stop']);start=0
            for axis,B in zip(axes,channel_bases):
                stop=start+B.shape[1];delta[sl,axis]=B@fitted[start:stop];start=stop
        if np.max(abs(delta))<=128*np.finfo(float).eps*max(1.,np.max(abs(state['controls']))):break
        alpha=1.;accepted=False;old_cost=cost
        while True:
            proposal={k:v.copy() for k,v in state.items()};proposal['controls']+=alpha*delta
            if np.array_equal(proposal['controls'],state['controls']):break
            if certificates(proposal['controls']):
                projected,_,_=evaluate(Path('.'),m,proposal,z,mapping,core)
                if not np.any(projected['status']):
                    trial_error=projected['values'][:,3:5]-target;trial_cost=float(np.sum(trial_error*trial_error))
                    if trial_cost<cost:
                        state,current,error,cost=proposal,projected,trial_error,trial_cost;accepted=True;break
            alpha*=.5
        if not accepted:break
        note=dict(iteration=len(history)+1,projection_change=extent(np.linalg.norm(error,axis=1)),step_alpha=alpha,cost=cost)
        history.append(note);print(__import__('json').dumps(dict(projection_transfer=note)),flush=True)
        if old_cost-cost<=1e-6*max(1.,old_cost):break
    for key in ('rig','scene_values','stations','source_rows'):
        if state[key].tobytes()!=candidate[key].tobytes():raise RuntimeError('Look-direction transfer changed '+key)
    return state,current,dict(target='previous_native_projection_at_same_material_stations',source_UV_PTS_never_modified=True,initializer_targets_not_production_observations=True,only_existing_pan_tilt_controls_changed=(2 not in axes),zoom_controls_unchanged=(2 not in axes),changed_motion_channels=list(channels),backend=backend,initial_projection_change=initial,final_projection_change=extent(np.linalg.norm(error,axis=1)),history=history)
