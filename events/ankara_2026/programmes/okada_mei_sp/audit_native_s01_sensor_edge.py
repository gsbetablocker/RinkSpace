"""Read-only diagnosis of sensor-edge clipping; never changes a fit objective."""
from pathlib import Path
import io,json
import numpy as np
from scipy.optimize import minimize_scalar
import prepare_native_s01 as prep
from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import SceneParameterModel,ParameterState
from rinkspace.session.gpu_geometry import NativeSceneGPU


def arrays(path):
    with np.load(io.BytesIO(path.read_bytes()),allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def audit(directory,source_row):
    state_path=directory/'accepted_pixels_latest.npz';sha=prep.digest(state_path)
    state=arrays(state_path);points=arrays(directory/'point_observations.npz')
    evaluation=arrays(directory/'evaluation_pixels.npz');mapping=prep.read(directory/'mapping.json')
    assert str(state['point_input_sha256'])==prep.digest(directory/'point_observations.npz')
    row=int(np.flatnonzero(points['source_rows']==source_row)[0]);uv=points['uv'][row]
    identity=mapping['path_ids'][points['path_indices'][row]]
    core=NativeCore(prep.LIBRARY);model=SceneParameterModel(prep.read(prep.SCENE),prep.SCENE,optimize_scene=True)
    native=NativeSceneGPU(prep.SCENE,core,parameter_state=ParameterState(model.initial_state.layout,state['scene_values']))
    path=native.paths[identity];assert path['kind']==0
    ends=np.asarray(core.sample_path(native.scene['board'],path['kind'],path['parameters'],path['vertices'],[0.,1.]))
    camera=prep.camera_for_snapshot(prep.read(prep.CAMERA),state)
    frame_time=(points['pts'][row]-camera['reference_pts'])/12800.;height=camera['image_height']
    def project_at_row(stations,y):
        time=frame_time+state['rig'][37]*(y/(height-1)-.5)
        pose=core.evaluate_motion_camera(state['rig'],state['controls'],camera['first_control_time'],camera['control_dt'],[time])
        if np.any(pose['status']):raise ValueError('Diagnostic row exceeds native motion support')
        static=np.r_[pose['rotation'][0].ravel(),pose['centre'][0],
            [pose[k][0] for k in ['fx','fy','cx','cy']],0.,[pose[k][0] for k in ['k1','k2','p1','p2']],0.]
        xyz=ends[0]+np.asarray(stations)[:,None]*(ends[1]-ends[0])
        predicted,status=core.project(static,xyz)
        return np.asarray(predicted),np.asarray(status),time
    ts=np.linspace(0,1,4097);pred,status,_=project_at_row(ts,uv[1])
    distance=np.linalg.norm(pred-uv,axis=1);distance[status!=0]=np.inf
    best=int(distance.argmin());assert np.isfinite(distance[best])
    def free_row_projection(station):
        y=float(uv[1])
        for _ in range(100):
            q,status,time=project_at_row([station],y)
            if status[0]:raise ValueError('Invalid native diagnostic ray')
            if abs(q[0,1]-y)<1e-10:return q[0],time,float(q[0,1]-y)
            y=float(q[0,1])
        raise ValueError('Diagnostic predicted-row equation did not converge')
    def objective(station):
        q,_,_=free_row_projection(station)
        return float(np.sum((q-uv)**2))
    result=minimize_scalar(objective,bounds=(ts[max(0,best-2)],ts[min(len(ts)-1,best+2)]),method='bounded',options={'xatol':1e-14})
    if not result.success:raise ValueError(result.message)
    q,time,equality=free_row_projection(result.x);xyz=ends[0]+result.x*(ends[1]-ends[0])
    strict=core.project_motion(state['rig'],state['controls'],camera['first_control_time'],camera['control_dt'],height,[frame_time],[xyz])
    assert prep.digest(state_path)==sha
    return dict(run=str(directory),state_sha256=sha,input_sha256=prep.digest(directory/'point_observations.npz'),
        source_row=source_row,observation_id=str(points['observation_ids'][row]),pts=int(points['pts'][row]),path=identity,
        observed_uv=uv.tolist(),current_station=float(state['stations'][row]),current_fitted_uv=evaluation['values'][row,3:5].tolist(),
        current_error_px=float(np.linalg.norm(evaluation['values'][row,9:11])),
        diagnostic_station=float(result.x),diagnostic_uv=q.tolist(),diagnostic_error_px=float(np.sqrt(result.fun)),
        predicted_row_equality_px=equality,native_strict_status_at_diagnostic_point=int(strict['status'][0]),
        native_certified_projection=(dict(uv=strict['uv'][0].tolist(),
            root_equality_px=float(strict['root_equality_px'][0]),
            contraction_bound=float(strict['contraction_bound'][0])) if strict['status'][0]==0 else None),
        diagnostic_row_excess_below_image_px=float(q[1]-(height-1)),
        exposure_offset_ms=float((time-frame_time)*1000),
        method='Native instantaneous pose and native static projection, solving the same predicted-row equation without imposing 0<=row<=height-1; local finite-line nearest-point diagnostic only.',
        production_objective_changed=False,native_parameters_changed=False,source_points_changed=False)


if __name__=='__main__':
    alternate=prep.RUN/'camera_identifiability/artwork_central/conditional_step'
    records=[audit(d,179945) for d in [prep.RUN,alternate]]
    out=prep.RUN/'ice_residual_audit/sensor_edge_diagnostic.json'
    prep.write(out,dict(records=records,global_or_fit_acceptance_claimed=False))
    for r in records:print(json.dumps(r,ensure_ascii=True),flush=True)
