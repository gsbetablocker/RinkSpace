"""Conditional whole-rink geometry fitted jointly to source-backed observations.

The model has a complete extent even where its parameters are assumptions.  A
parameter prior is never an image measurement. Visible arc points, line normals
and identified intersections have different residuals; ellipse centres and the
unseen continuation of a clipped ellipse are not point observations.
"""
from __future__ import annotations

import copy
from collections import defaultdict
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.optimize import least_squares

from .geometry import homography as checked_homography, project


INVALID_PROJECTION_RESIDUAL = 1e6


def value(item, parameters):
    if isinstance(item,dict):
        if 'sum' in item:
            return sum(value(x,parameters) for x in item['sum'])
        if 'difference' in item:
            a,b=item['difference'];return value(a,parameters)-value(b,parameters)
        if 'negative' in item:
            return -value(item['negative'],parameters)
        raise ValueError('Unsupported explicitly tied geometry parameter')
    return float(parameters[item] if isinstance(item, str) else item)


def feature_points(feature, parameters, count=96):
    """Return a physical feature in conditional, non-metric model coordinates."""
    def v(x):
        return value(x, parameters)
    kind = feature['kind']
    if kind in {'line', 'polyline', 'point', 'direction'}:
        return np.array([[v(x), v(y)] for x, y in feature['points']], float)
    if kind == 'axis_line':
        c = v(feature['coordinate'])
        a, b = map(v, feature['extent'])
        return np.array([[c, a], [c, b]] if feature['axis'] == 'x' else [[a, c], [b, c]])
    if kind in {'circle', 'arc'}:
        cx, cy = map(v, feature['centre'])
        radius = v(feature['radius'])
        if radius <= 0:
            raise ValueError('Circle radius must be positive')
        start, end = feature.get('angle_degrees', [0, 360])
        t = np.linspace(np.deg2rad(start), np.deg2rad(end), count + 1)
        return np.c_[cx + radius * np.cos(t), cy + radius * np.sin(t)]
    if kind == 'rounded_boundary':
        x0, y0, x1, y1 = map(v, feature['bounds'])
        r = v(feature['radius'])
        if not 0 < r < min(x1-x0, y1-y0)/2:
            raise ValueError('Invalid rounded boundary hypothesis')
        parts = []
        for cx, cy, start in [(x1-r,y1-r,0),(x0+r,y1-r,90),
                              (x0+r,y0+r,180),(x1-r,y0+r,270)]:
            t = np.deg2rad(np.linspace(start, start+90, max(12, count//4)))
            parts.append(np.c_[cx+r*np.cos(t),cy+r*np.sin(t)])
        points = np.concatenate(parts)
        return np.vstack([points, points[0]])
    raise ValueError(f'Unsupported physical feature kind: {kind}')


def point_polyline_distances(points, polyline, valid=None):
    """Distance only from actually observed points to the projected model curve."""
    points, polyline = np.asarray(points, float), np.asarray(polyline, float)
    a, b = polyline[:-1], polyline[1:]
    if valid is not None:
        valid=np.asarray(valid,bool)
        # Preserve adjacency: filtering vertices first would bridge missing,
        # behind-camera or folded intervals with an invented line segment.
        adjacent=valid[:-1]&valid[1:]
        a,b=a[adjacent],b[adjacent]
    if not len(a):
        return np.full(len(points),INVALID_PROJECTION_RESIDUAL)
    ab = b-a
    ap = points[:, None, :] - a[None, :, :]
    t = np.clip(np.sum(ap*ab, axis=2) / np.maximum(np.sum(ab*ab, axis=1), 1e-12), 0, 1)
    delta = ap-t[:, :, None]*ab
    return np.sqrt(np.min(np.sum(delta*delta, axis=2), axis=1))


def distorted_project(homography, points, distortion=None):
    xy=project(homography,points)
    if distortion and distortion.get('k1',0):
        centre=np.asarray(distortion['centre_image'],float)
        normal=(xy-centre)/float(distortion['scale_image'])
        xy=centre+(xy-centre)*(1+float(distortion['k1'])*np.sum(normal**2,axis=1))[:,None]
    return xy


def _radial_parameters(distortion):
    if not distortion or not distortion.get('k1',0):
        return 0.,np.zeros(2),1.
    k=float(distortion['k1']);centre=np.asarray(distortion['centre_image'],float)
    scale=float(distortion['scale_image'])
    if centre.shape!=(2,) or not np.isfinite([k,scale,*centre]).all() or scale<=0:
        raise ValueError('Invalid radial distortion parameters')
    return k,centre,scale


def undistort_observed_points(points, distortion=None):
    """Invert only the central monotonic radial branch; folded roots are invalid.

    A negative k has a finite maximum distorted radius on that branch. Pixels
    beyond it are not assigned a different polynomial root to rescue a fit.
    """
    points=np.asarray(points,float)
    valid=np.isfinite(points).all(axis=1)
    k,centre,scale=_radial_parameters(distortion)
    if not k:
        return points.copy(),valid
    direction=(points-centre)/scale
    rd=np.linalg.norm(direction,axis=1)
    if k<0:
        turn=np.sqrt(-1/(3*k))
        valid&=rd < (2/3)*turn
        upper=np.full(len(points),turn)
    else:
        upper=np.maximum(rd,1.)
    lower=np.zeros(len(points))
    target=np.where(valid,rd,0.)
    for _ in range(40):
        middle=(lower+upper)/2
        below=middle*(1+k*middle*middle)<target
        lower=np.where(below,middle,lower);upper=np.where(below,upper,middle)
    radius=(lower+upper)/2
    factor=np.divide(radius,rd,out=np.ones_like(rd),where=rd>0)
    xy=centre+direction*factor[:,None]*scale
    xy[~valid]=np.nan
    return xy,valid


def infer_visible_branch_sign(homography, points_image, distortion=None):
    """Choose the matrix-relative branch containing the actual source pixels.

    This is not calibrated 3-D cheirality. Its sign reverses if H is multiplied
    by -1, and an image-space horizon may leave some observations unsupported.
    """
    h=checked_homography(homography)
    xy,valid=undistort_observed_points(points_image,distortion)
    denominators=np.c_[xy,np.ones(len(xy))] @ np.linalg.inv(h)[2]
    valid&=np.isfinite(denominators)&(np.abs(denominators)>1e-10)
    if not np.any(valid):
        return None
    median=float(np.median(denominators[valid]))
    return float(np.sign(median)) if abs(median)>1e-10 else None


def valid_projected_points(homography, points, distortion=None, *, visible_branch_sign):
    """Return raw-image samples and a domain mask without reconnecting gaps."""
    h=checked_homography(homography)
    if visible_branch_sign not in (-1,1):
        raise ValueError('visible_branch_sign must be +1 or -1 relative to H')
    points=np.asarray(points,float)
    q=np.c_[points,np.ones(len(points))] @ h.T
    valid=np.isfinite(q).all(axis=1)&(visible_branch_sign*q[:,2]>1e-10)
    xy=np.full((len(points),2),np.nan)
    xy[valid]=q[valid,:2]/q[valid,2:]
    k,centre,scale=_radial_parameters(distortion)
    if k:
        radius_squared=np.sum(((xy-centre)/scale)**2,axis=1)
        factor=1+k*radius_squared
        derivative=1+3*k*radius_squared
        valid&=np.isfinite(factor)&(factor>0)&(derivative>0)
        xy[valid]=centre+(xy[valid]-centre)*factor[valid,None]
    valid&=np.isfinite(xy).all(axis=1)
    xy[~valid]=np.nan
    return xy,valid


def observation_residual(observation, feature, parameters, homography, distortion=None,
                         *, visible_branch_sign=None, return_domain=False):
    world=feature_points(feature, parameters)
    kind = observation.get('type', 'curve')
    if kind not in {'point','line'} and distortion is not None and len(world)==2:
        world=np.linspace(world[0],world[1],64)
    observed = np.asarray(observation['points_image'], float)
    h=checked_homography(homography)
    branch=visible_branch_sign
    if branch is None:
        branch=infer_visible_branch_sign(h,observed,distortion)
    if branch is not None and branch not in (-1,1):
        raise ValueError('visible_branch_sign must be +1 or -1 relative to H')
    undistorted,observed_valid=undistort_observed_points(observed,distortion)
    inverse_w=np.c_[undistorted,np.ones(len(observed))] @ np.linalg.inv(h)[2]
    observed_valid&=np.isfinite(inverse_w)&((branch or 0)*inverse_w>1e-10)
    if branch is None:
        predicted=np.full((len(world),2),np.nan);valid=np.zeros(len(world),bool)
    else:
        predicted,valid=valid_projected_points(h,world,distortion,visible_branch_sign=branch)
    domain={'visible_branch_sign':branch,'valid_model_vertex_count':int(valid.sum()),
            'model_vertex_count':len(valid),'valid_adjacent_segment_count':int(np.sum(valid[:-1]&valid[1:])),
            'invalid_observation_count':int(np.sum(~observed_valid)),
            'radial_domain':'positive_scale_and_positive_radial_derivative',
            'invalid_projection_residual_px':INVALID_PROJECTION_RESIDUAL}
    if kind == 'point':
        if predicted.shape != observed.shape:
            raise ValueError('Identified point observations require matching points')
        residual=predicted-observed
        residual[~(valid&observed_valid)]=INVALID_PROJECTION_RESIDUAL
        residual=residual.ravel()
    elif kind == 'line':
        if len(world)!=2:
            raise ValueError('Line-normal observations require a straight line')
        if distortion and distortion.get('k1',0):
            # A line annotation constrains its supporting line, not the model
            # segment's endpoints. Previously enabling even k1=0 changed this
            # contract to a finite polyline, pulling the scene to fake endpoints.
            # Find the nearest point on the radially distorted supporting line
            # in raw pixels. The undistorted foot seeds a short 1-D refinement.
            q=np.c_[world,np.ones(2)]@h.T
            line=np.cross(q[0],q[1]);norm=float(np.linalg.norm(line[:2]))
            if norm <= 1e-12:
                residual=np.full(len(observed),INVALID_PROJECTION_RESIDUAL)
            else:
                normal=line[:2]/norm
                offset=(np.nan_to_num(undistorted)@normal+line[2]/norm)
                feet=np.nan_to_num(undistorted)-offset[:,None]*normal
                tangent=np.array([normal[1],-normal[0]])
                k,centre,scale=_radial_parameters(distortion)

                def raw_and_tangent(xy):
                    delta=xy-centre
                    radius2=np.sum(delta**2,axis=1)/scale**2
                    factor=1+k*radius2
                    raw=centre+delta*factor[:,None]
                    derivative=(factor[:,None]*tangent +
                        (2*k/scale**2)*(delta@tangent)[:,None]*delta)
                    central=(factor>0)&(1+3*k*radius2>0)
                    return raw,derivative,central

                for _ in range(6):
                    raw,derivative,central=raw_and_tangent(feet)
                    step=np.sum((raw-observed)*derivative,axis=1)/np.maximum(
                        np.sum(derivative**2,axis=1),1e-12)
                    feet-=np.where(central&observed_valid,step,0.)[:,None]*tangent
                raw,derivative,central=raw_and_tangent(feet)
                raw_normal=np.c_[-derivative[:,1],derivative[:,0]]
                raw_normal/=np.maximum(np.linalg.norm(raw_normal,axis=1)[:,None],1e-12)
                residual=np.sum((observed-raw)*raw_normal,axis=1)
                foot_inverse_w=np.c_[feet,np.ones(len(feet))]@np.linalg.inv(h)[2]
                closest_valid=(central & np.isfinite(raw).all(axis=1) &
                               ((branch or 0)*foot_inverse_w>1e-10))
                residual[~closest_valid]=INVALID_PROJECTION_RESIDUAL
                domain['invalid_closest_line_point_count']=int(np.sum(~closest_valid))
            domain['line_semantics']='infinite_supporting_line_redistorted_to_raw_pixels'
        else:
            # Keep the existing infinite supporting-line normal semantics. A
            # model endpoint at infinity does not destroy the supporting line.
            q=np.c_[world,np.ones(2)]@h.T
            if np.min(np.abs(q[:,2]))>1e-10:
                a,b=project(homography,world)
                normal=np.array([a[1]-b[1],b[0]-a[0]])
                norm=float(np.linalg.norm(normal))
                residual=((observed-a)@(normal/norm) if norm>1e-12 else
                          np.full(len(observed),INVALID_PROJECTION_RESIDUAL))
            else:
                line=np.cross(q[0],q[1]);norm=float(np.linalg.norm(line[:2]))
                residual=(np.c_[observed,np.ones(len(observed))]@line/norm if norm>1e-12
                          else np.full(len(observed),INVALID_PROJECTION_RESIDUAL))
            domain['line_semantics']='infinite_supporting_line_in_valid_observed_branch'
        residual[~observed_valid]=INVALID_PROJECTION_RESIDUAL
    elif kind == 'curve':
        residual=point_polyline_distances(observed,predicted,valid)
        residual[~observed_valid]=INVALID_PROJECTION_RESIDUAL
    else:
        raise ValueError('Observation type must be point, line or visible curve')
    domain['penalized_residual_count']=int(np.sum(np.abs(residual)>=INVALID_PROJECTION_RESIDUAL))
    return (residual,domain) if return_domain else residual


def _pack_h(h):
    # Optimisation uses model coordinates /100 and image pixels /1000.
    h = np.diag([.001,.001,1.]) @ np.asarray(h,float) @ np.diag([100.,100.,1.])
    return (h/h[2,2]).ravel()[:8]


def _unpack_h(x):
    h = np.r_[x,1.].reshape(3,3)
    return np.diag([1000.,1000.,1.]) @ h @ np.diag([.01,.01,1.])


def initial_h(view):
    if 'H_image_from_model_initial' in view:
        h = np.asarray(view['H_image_from_model_initial'],float)
    else:
        seed = view['initial_correspondences']
        world = np.asarray(seed['points_model'],float)
        image = np.asarray(seed['points_image'],float)
        if len(world)<4 or len(world)!=len(image):
            raise ValueError('A view needs four identified seed points or a sourced initial H')
        h, _ = cv2.findHomography(world, image, 0)
    if h is None or h.shape!=(3,3) or not np.isfinite(h).all() or abs(np.linalg.det(h))<1e-12:
        raise ValueError(f"Invalid view initialisation: {view['id']}")
    return h/h[2,2]


def full_extent_projection_diagnostic(homography, bounds):
    x0,y0,x1,y1 = map(float,bounds)
    corners=np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
    h=np.asarray(homography,float)
    w=np.c_[corners,np.ones(4)] @ h[2]
    crosses=bool(w.min()<=1e-9 and w.max()>=-1e-9)
    determinant=float(np.linalg.det(h))
    return {'horizon_crosses_full_extent':crosses,'homogeneous_denominator_range':w.tolist(),
            'matrix_nonsingular':bool(abs(determinant)>1e-12),
            'orientation_sign':None if crosses else float(np.sign(determinant/w.mean()**3)),
            'whole_projection_valid':bool(not crosses and abs(determinant)>1e-12),
            'scope':'numerical projection of full conditional extent; not geometric accuracy'}


def fit_whole_rink(spec, *, max_nfev=600):
    """Joint physical-parameter / multi-view fit, with explicit prior accounting.

    'check' observations are excluded from optimization. Their errors are
    diagnostics unless the author supplied a genuinely independent protocol.
    Fixing a pose or parameter is explicit in the input, never inferred from
    solver success. Geometry with prior-only parameters remains hypothetical.
    """
    parameters = {k:float(p['initial']) for k,p in spec['parameters'].items()}
    variable_names = [k for k,p in spec['parameters'].items() if not p.get('fixed',False)]
    features = {p['id']:p for p in spec['features']}
    views = {v['id']:v for v in spec['views']}
    poses = {k:initial_h(v) for k,v in views.items()}
    variable_views = [k for k,v in views.items() if not v.get('pose_fixed',False)]
    all_obs = spec['observations']
    fit_obs = [o for o in all_obs if o.get('role','fit')=='fit']
    if not fit_obs:
        raise ValueError('No actual image observations')
    for ob in all_obs:
        if ob['view_id'] not in views or ob['feature_id'] not in features:
            raise ValueError('Observation refers to absent view or physical feature')
        if ob.get('sigma_image_px',0)<=0:
            raise ValueError('Observation uncertainty must be positive and explicit')
        if not np.isfinite(np.asarray(ob['points_image'],float)).all():
            raise ValueError('Nonfinite observation')
    for view in views.values():
        if view.get('visible_branch_sign') is not None and view['visible_branch_sign'] not in (-1,1):
            raise ValueError('A view visible_branch_sign must be +1 or -1')
    branch_pixels={}
    branch_sources={}
    for key,view in views.items():
        support=[o for o in fit_obs if o['view_id']==key]
        branch_sources[key]='explicit_view_sign' if view.get('visible_branch_sign') is not None else 'fit_observation_inverse_h'
        if not support:
            # A check-only fixed pose has no role in optimization. It may still
            # receive its own explicitly labeled projection diagnostics.
            support=[o for o in all_obs if o['view_id']==key]
            if branch_sources[key]!='explicit_view_sign':branch_sources[key]='check_only_view_inverse_h'
        branch_pixels[key]=np.concatenate([np.asarray(o['points_image'],float) for o in support]) if support else np.empty((0,2))
    x0 = np.r_[[parameters[k] for k in variable_names],
               np.concatenate([_pack_h(poses[k]) for k in variable_views]) if variable_views else []]
    lower = [spec['parameters'][k]['bounds'][0] for k in variable_names] + [-np.inf]*(8*len(variable_views))
    upper = [spec['parameters'][k]['bounds'][1] for k in variable_names] + [np.inf]*(8*len(variable_views))
    nparams = len(variable_names)
    def unpack(x):
        ps = {**parameters, **dict(zip(variable_names, x[:nparams]))}
        hs = dict(poses)
        for n,k in enumerate(variable_views):
            hs[k] = _unpack_h(x[nparams+8*n:nparams+8*(n+1)])
        return ps,hs
    def distortion(k,ps):
        d=copy.deepcopy(views[k].get('radial_distortion'))
        if d is not None:
            d['k1']=value(d.get('k1',0),ps)
        return d
    def view_branches(ps,hs):
        return {k:(v['visible_branch_sign'] if v.get('visible_branch_sign') is not None else
                   infer_visible_branch_sign(hs[k],branch_pixels[k],distortion(k,ps))) for k,v in views.items()}
    def view_observation_residual(o,ps,hs,branches,*,return_domain=False):
        key=o['view_id'];branch=branches[key]
        if branch is not None:
            return observation_residual(o,features[o['feature_id']],ps,hs[key],distortion(key,ps),
                                        visible_branch_sign=branch,return_domain=return_domain)
        count=np.asarray(o['points_image']).size if o.get('type')=='point' else len(o['points_image'])
        raw=np.full(count,INVALID_PROJECTION_RESIDUAL)
        domain={'visible_branch_sign':None,'reason':'no_valid_observed_projection_branch',
                'penalized_residual_count':count,'invalid_projection_residual_px':INVALID_PROJECTION_RESIDUAL}
        return (raw,domain) if return_domain else raw
    def image_residuals(ps, hs):
        result=[]
        branches=view_branches(ps,hs)
        for o in fit_obs:
            raw=view_observation_residual(o,ps,hs,branches)
            # Adjacent samples on a marked curve are correlated. Dense tracing
            # must not multiply the physical group's evidence arbitrarily.
            effective=o.get('effective_independent_samples',min(4,len(raw)))
            result.append(raw/o['sigma_image_px'] * np.sqrt(effective/max(len(raw),1)))
        return result
    residual_size=sum(np.asarray(o['points_image']).size if o.get('type')=='point'
                      else len(o['points_image']) for o in fit_obs)
    residual_size+=sum(bool(spec['parameters'][k].get('prior_sigma')) for k in variable_names)
    def residual(x):
        ps,hs=unpack(x)
        try:
            data=image_residuals(ps,hs)
            priors=[(ps[k]-parameters[k])/spec['parameters'][k]['prior_sigma']
                    for k in variable_names if spec['parameters'][k].get('prior_sigma')]
            out=np.r_[np.concatenate(data),priors]
            if not np.isfinite(out).all():
                raise ValueError('Nonfinite trial')
            return out
        except (ValueError,np.linalg.LinAlgError,FloatingPointError):
            return np.full(residual_size,1e6)
    initial_jac=None
    initial_rank=None
    if len(x0) and not variable_names:
        baseline=residual(x0)
        initial_jac=np.empty((len(baseline),len(x0)))
        for j in range(len(x0)):
            dx=1e-6*max(1.,abs(x0[j]));xx=x0.copy();xx[j]+=dx
            initial_jac[:,j]=(residual(xx)-baseline)/dx
        initial_rank=int(np.linalg.matrix_rank(initial_jac,tol=max(np.linalg.norm(initial_jac,2)*1e-6,1e-8)))
    if initial_rank is not None and initial_rank<len(x0):
        fit=SimpleNamespace(x=x0,success=False,message='Camera is underconstrained by actual geometry; seed retained, not localized',
                            nfev=0,jac=initial_jac)
    elif len(x0):
        fit = least_squares(residual,x0,bounds=(lower,upper),loss='soft_l1',f_scale=1.,
                            x_scale='jac',max_nfev=max_nfev,ftol=1e-8,xtol=1e-8,gtol=1e-8)
    else:
        fit=SimpleNamespace(x=x0,success=True,message='All parameters fixed; diagnostic projection only',
                            nfev=0,jac=np.empty((len(residual(x0)),0)))
    ps,hs=unpack(fit.x)
    solved_bounds=[value(x,ps) for x in spec['structure_bounds']]
    diagnostics=[]
    feature_views=defaultdict(set)
    final_branches=view_branches(ps,hs)
    for ob in all_obs:
        raw,domain=view_observation_residual(ob,ps,hs,final_branches,return_domain=True)
        err=np.linalg.norm(raw.reshape(-1,2),axis=1) if ob.get('type')=='point' else np.abs(raw)
        diagnostics.append({**copy.deepcopy(ob),'median_image_px':float(np.median(err)),
            'p95_image_px':float(np.percentile(err,95)),'max_image_px':float(np.max(err)),
            'residual_count':len(err),'projection_domain':domain,'independent_validation':False})
        if ob.get('role','fit')=='fit':
            feature_views[ob['feature_id']].add(ob['view_id'])
    singular=np.linalg.svd(fit.jac,compute_uv=False)
    # Priors can hide weak data directions. Diagnose the image-only Jacobian.
    base=np.concatenate(image_residuals(ps,hs))
    data_jac=np.empty((len(base),len(fit.x)))
    for k in range(len(fit.x)):
        dx=1e-5*max(abs(fit.x[k]),1.)
        perturbed=fit.x.copy();perturbed[k]+=dx
        pp,hh=unpack(perturbed)
        data_jac[:,k]=(np.concatenate(image_residuals(pp,hh))-base)/dx
    data_singular=np.linalg.svd(data_jac,compute_uv=False)
    data_rank=int(np.linalg.matrix_rank(data_jac)) if data_jac.size else 0
    image_underconstrained=data_rank<len(fit.x)
    primitives=[]
    for feature in spec['features']:
        primitives.append({k:copy.deepcopy(v) for k,v in feature.items() if k not in {'points','centre','radius','bounds','coordinate','extent'}})
        primitives[-1].update(points_model=feature_points(feature,ps).tolist(),
            source_view_ids=sorted(feature_views[feature['id']]),
            parameter_definition=copy.deepcopy(feature))
    output=copy.deepcopy(spec)
    output.update(schema_version='rinkspace.whole-rink.v1',status='conditional_candidate',
        geometry_level='euclidean_conditional',unit='normalized_rink_width_100',
        length_m=None,width_m=None,corner_radius_m=None,
        parameter_estimates={k:{**v,'value':float(ps[k]),'bounds_are':'declared_hypothesis_range_not_confidence_interval',
              'at_bound':bool(not v.get('fixed') and min(ps[k]-v['bounds'][0],v['bounds'][1]-ps[k])<1e-4)}
              for k,v in spec['parameters'].items()},
        view_poses=[{**copy.deepcopy(v),'H_image_from_model':hs[k].tolist(),
                     'visible_branch_sign':final_branches[k],
                     'visible_branch_source':branch_sources[k],
                     'H_model_from_image':np.linalg.inv(hs[k]).tolist() if not distortion(k,ps) or not distortion(k,ps).get('k1') else None,
                     'H_model_from_undistorted_image':np.linalg.inv(hs[k]).tolist(),
                     'radial_distortion':distortion(k,ps),
                     'full_extent_projection':full_extent_projection_diagnostic(hs[k],solved_bounds)}
                    for k,v in views.items()],
        structure={'primitives':primitives,'bounds':solved_bounds,
                   'label':spec.get('label','整场条件草模；非实测尺寸')},
        fitting_diagnostics={'solver_converged':bool(fit.success),'message':fit.message,
            'iterations':int(fit.nfev),'fit_observation_count':len(fit_obs),
            'check_observation_count':len(all_obs)-len(fit_obs),'unknown_count':len(fit.x),
            'jacobian_rank_with_priors':int(np.linalg.matrix_rank(fit.jac)) if fit.jac.size else 0,
            'image_only_jacobian_rank':data_rank,
            'image_underconstrained':image_underconstrained,
            'condition_with_priors':float(singular[0]/max(singular[-1],1e-14)) if len(singular) else None,
            'image_only_condition':float(data_singular[0]/max(data_singular[-1],1e-14))
                if len(data_singular) and not image_underconstrained else None,
            'fit_projection_domain_valid':all(not d['projection_domain']['penalized_residual_count']
                                             for d in diagnostics if d.get('role','fit')=='fit'),
            'observation_diagnostics':diagnostics,'independent_accuracy_validated':False,
            'warning':'Convergence and small annotation residuals are not physical validation.'})
    boundary=next((p for p in primitives if p['kind']=='rounded_boundary'),None)
    if boundary is not None:
        output['structure']['boundary']=boundary
        output['structure']['primitives']=[p for p in primitives if p is not boundary]
    return output
