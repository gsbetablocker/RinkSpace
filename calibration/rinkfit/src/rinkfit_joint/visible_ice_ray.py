"""Necessary visible-ice source-ray inequalities; no fixed world observation.

Uses actual source floating v for RS time, complete production camera geometry,
and Brown including the source-supported wide extension. The ice intersection
is a derived witness recomputed for every physical trial, never source XYZ.
"""
from __future__ import annotations
import numpy as np
from .joint_model import graph_from_joint, as_event_view
from .camera_graph import CameraGraph
from .wide_optics_candidate import _camera_geometry, _inverse_brown
from rinkfit_start.native_scene import NativeScene


def visible_ice_sensor_ray(model, shot_id, native_pts, source_uv):
    uv = np.asarray(source_uv, float)
    out = dict(inverse_valid=False, arithmetic_valid=False, finite_ice=False,
               visibility_evaluated=False, feasible=False, reason='not_evaluated',
               intersection_is_derived_trial_witness_not_source_world_point=True)
    if uv.shape != (2,) or not np.isfinite(uv).all() or not 0 <= uv[0] <= 1919 or not 0 <= uv[1] <= 1079:
        out['reason'] = 'source_UV_outside_native_sensor'; return out
    graph = CameraGraph(graph_from_joint(model)); camera = graph.native_camera(shot_id)
    scene = NativeScene(as_event_view(model, shot_id))
    local = (int(native_pts)-model['shots'][shot_id]['time_origin_pts'])/12800.
    time = local+camera.tau*(uv[1]/1079.-.5)
    R, C, logf, z = _camera_geometry(camera, np.array([time])); R=R[0]; C=C[0]; z=float(z[0]); logf=float(logf[0])
    o=camera.optics; hinge=max(camera.wide_boundary-logf,0.); basis=hinge*hinge
    coeff=np.array([np.polynomial.polynomial.polyval(z,o['k1']),np.polynomial.polynomial.polyval(z,o['k2']),o['p1'],o['p2']])+camera.wide_coefficients*basis
    def polymin(c1,c2,r2):
        v=min(1.,1+c1*r2+c2*r2*r2)
        if c2>0:
            t=-c1/(2*c2)
            if 0<t<r2: v=min(v,1+c1*t+c2*t*t)
        return v
    radius=float(o['radius_max']); k1,k2,p1,p2=coeff
    lower=min(polymin(k1,k2,radius**2),polymin(3*k1,5*k2,radius**2))-6*np.hypot(p1,p2)*radius
    out.update(source_row_time=float(time),C=C.tolist(),Brown_disk_lower=float(lower),wide_basis=float(basis),Brown_coefficients=coeff.tolist())
    if not lower>0: out['reason']='uncertified_whole_Brown_disk'; return out
    fxy=np.exp(logf)*np.exp(np.array([o.get('eta',0.),-o.get('eta',0.)])/2)
    pp=np.array(o['principal_point'])+z*np.array(o.get('principal_linear',[0.,0.]))
    try: xy=_inverse_brown(((uv-pp)/fxy)[None],coeff[None])[0]
    except (ValueError,FloatingPointError,np.linalg.LinAlgError) as e: out['reason']='inverse_not_converged:'+str(e); return out
    if np.linalg.norm(xy)>radius: out['reason']='inverse_root_outside_retained_Brown_disk'; return out
    out['inverse_valid']=True; direction=R.T@np.r_[xy,1.]
    out.update(inverse_xy=xy.tolist(),direction=direction.tolist(),inverse_ray_g=float(xy@xy-radius**2),camera_above_ice_g=float(-C[2]))
    if abs(direction[2])<=1e-14: out['reason']='ray_parallel_to_ice_or_singular'; return out
    depth=-C[2]/direction[2]; X=C+depth*direction; X[2]=0.; rg=float(scene.containment(X[None,:2])[0])
    out.update(depth=float(depth),forward_depth_g=float(-depth),ice_intersection=X.tolist(),rink_g=rg,finite_ice=bool(rg<=0))
    replay=camera.at(X,np.array(time)); err=float(np.linalg.norm(replay['uv']-uv))
    out['forward_error_px']=err
    if not bool(replay['valid']):out['reason']='inverse_ray_forward_replay_invalid';return out
    if rg>0:out['reason']='ray_ice_intersection_outside_finite_rink_visibility_not_attempted';return out
    # The same convex finite-board root and entry-height test as joint_visibility;
    # use independent NumPy bisection, never a current source material station.
    occlusion=-1.; entry=None
    if float(scene.containment(C[None,:2])[0])>1e-10:
        a,b=0.,1.
        for _ in range(48):
            t=(a+b)/2; Q=C+t*(X-C)
            if float(scene.containment(Q[None,:2])[0])<=0:b=t
            else:a=t
        t=(a+b)/2
        if t<1-1e-7:
            Q=C+t*(X-C);occlusion=float((scene.height-Q[2])/.02);entry=Q.tolist()
    out.update(occlusion_g=occlusion,near_board_entry=entry,visibility_evaluated=True,arithmetic_valid=True)
    out['feasible']=bool(C[2]>0 and depth>0 and occlusion<=1e-8)
    out['reason']='source_ice_ray_physically_visible' if out['feasible'] else 'source_ice_ray_physical_inequality_violation'
    return out
