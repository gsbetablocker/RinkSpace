"""Finite possible support for explicitly unresolved Lane display proxies.

This profiles an existing finite endpoint interval for source support. It is
not a measured endpoint, definite-paint export, infinite line or new source.
The measured/estimated painted right end is left unchanged. Callers must use
the returned world and its same derivative for source and coverage together.
No runtime hook or source/guard replacement is installed by this module.
"""
import numpy as np


def bounded_lane_support(scene,target,world,world_jac=None,*,goal_world_jac=None):
    """Use the explicit interval's current goal-line carrier as possible cap.

    Geometry derivatives, when requested, include the moving interval cap.
    goal_world_jac has shape (2,3,n), in the same geometry coordinates as world_jac.
    Missing qualification never silently releases a finite endpoint.
    """
    curve=target.get('curve_id','');fid,_,edge=curve.partition(':')
    if target.get('kind')!='ice_curve' or fid not in {f'lane{i}'for i in range(1,5)} or edge not in ('edge0','edge2'):
        raise ValueError('Only explicit Lane long-edge source targets are qualified')
    features={f['id']:f for f in scene['ice_features']}
    feature=features[fid];role='upper'if edge=='edge0'else'lower'
    if (feature.get('polygon_left_vertices_role')!='unmeasured_display_proxy_not_physical_endpoints'
            or feature.get('left_endpoint_status')!='unresolved'
            or feature.get('left_endpoint_coordinate_roles',{}).get(role)!='unmeasured_display_proxy'):
        raise ValueError('Explicit unresolved display-proxy authority required')
    intervals=feature.get('left_endpoint_x_intervals_scene_units',{})
    interval=np.asarray(intervals.get(role),float)
    if interval.shape!=(2,)or not np.isfinite(interval).all()or interval[0]>interval[1]:
        raise ValueError('Original finite endpoint interval required')
    authority=feature.get('left_endpoint_x_interval_bound_sources',{}).get('lower_x',{})
    if authority.get('feature_id')!='goal_line_left' or authority.get('geometry')!='segments_xyz[*][*][0]':
        raise ValueError('Explicit goal-line lower interval authority required')
    goal=np.asarray(features['goal_line_left']['segments_xyz'],float)
    points=np.asarray(world,float)
    if goal.shape!=(1,2,3)or points.ndim!=2 or points.shape[1]!=3 or len(points)<2:
        raise ValueError('One finite goal carrier and ordered straight Lane support required')
    if not np.isfinite(goal).all()or not np.isfinite(points).all()or np.any(abs(np.r_[goal.ravel()[2::3],points[:,2]])>1e-10):
        raise ValueError('Finite coplanar ice geometry required')
    goal=goal[0];a,b=points[[0,-1]];v=b-a;w=goal[1]-goal[0]
    vv=float(v@v)
    if vv<=0:raise ValueError('Nonzero finite Lane support required')
    t=(points-a)@v/vv
    if np.max(abs(points-a-t[:,None]*v))>1e-9*max(1.,np.max(abs(points))):
        raise ValueError('Exact straight Lane material required')
    M=np.column_stack((v[:2],-w[:2]));rhs=goal[0,:2]-a[:2]
    if abs(np.linalg.det(M))<=128*np.finfo(float).eps*np.linalg.norm(v)*np.linalg.norm(w):
        raise ValueError('Finite current interval cap is not transverse to Lane')
    parameter=np.linalg.solve(M,rhs);cap=a+parameter[0]*v
    old_proxy=0 if edge=='edge0'else-1
    if (edge=='edge0'and parameter[0]>1e-10)or(edge=='edge2'and parameter[0]<1-1e-10):
        raise ValueError('Current possible endpoint cap would shorten the guaranteed core')
    aa=cap if edge=='edge0'else a;bb=b if edge=='edge0'else cap
    result=aa+t[:,None]*(bb-aa)
    result[-1 if edge=='edge0'else 0]=b if edge=='edge0'else a
    derivative=None
    if world_jac is not None:
        J=np.asarray(world_jac,float);G=np.asarray(goal_world_jac,float)
        if (J.ndim!=3 or J.shape[:2]!=points.shape or G.shape!=(2,3,J.shape[2])
                or not np.isfinite(J).all()or not np.isfinite(G).all()):
            raise ValueError('Same complete world and moving goal-cap geometry Jacobians required')
        da,db=J[[0,-1]];dv=db-da;dw=G[1]-G[0]
        r=G[0,:2]-da[:2]-parameter[0]*dv[:2]+parameter[1]*dw[:2]
        ds=np.linalg.solve(M,r)[0]
        dcap=da+parameter[0]*dv+v[:,None]*ds
        dt=(np.einsum('nik,i->nk',J-da[None],v)+np.einsum('ni,ik->nk',points-a,dv))/vv
        dt-=t[:,None]*(2*v@dv/vv)
        dAA=dcap if edge=='edge0'else da;dBB=db if edge=='edge0'else dcap
        derivative=dAA[None]+t[:,None,None]*(dBB-dAA)[None]+(bb-aa)[None,:,None]*dt[:,None,:]
        derivative[-1 if edge=='edge0'else 0]=db if edge=='edge0'else da
    report=dict(schema='rinkspace.qualified-finite-possible-Lane-support.v1',curve_id=curve,
        proxy_side='start'if edge=='edge0'else'end',original_display_proxy=points[old_proxy].tolist(),
        current_possible_cap=cap.tolist(),original_proxy_parameter_of_cap=float(parameter[0]),
        original_interval_lower_bound_authority=authority,original_finite_interval=interval.tolist(),
        true_other_endpoint_preserved_byte_exact=True,unknown_endpoint_profiled_not_measured=True,
        definite_paint_scene_and_user_endpoint_intervals_not_modified=True,
        support_finite=True,extended_to_rink_boundary=False,added_parameter_or_source=False,
        source_and_coverage_must_use_same_returned_support=True)
    return result,derivative,report
