"""Independent rational source query and branch-local numerical root replay.

The alpha coordinate is the frozen source scalar, not Euclidean image distance.
A converged nonsingular numerical root does not by itself certify the complete
relevant root interval; that obligation has its own explicit status.
"""
from pathlib import Path
import json
import numpy as np
from .data import sha256_file

VERSION='native_rational_query_finite_three_variable_fallback_v2'


def load_query_packet(path, read_array, semantic_hash, source_rows):
    path=Path(path);m=json.loads(path.read_text(encoding='utf-8'))
    if m['semantic_manifest']['sha256']!=semantic_hash:raise ValueError('Query/semantic identity mismatch')
    arrays={k:read_array(path.parent,s) for k,s in m['arrays'].items()}
    if not np.array_equal(arrays['source_rows'],source_rows):raise ValueError('Query source profile row identity mismatch')
    if m['rational_coefficients']!='ascending powers; pixel=(numerator_u, numerator_v)/denominator':raise ValueError('Unknown rational source query chart')
    cert=m['certificates']
    if sha256_file(cert['path'])!=cert['sha256']:raise ValueError('Frozen query certificate changed')
    return m,arrays


def rational_query(curves, alpha):
    a=np.asarray(alpha,float);coeff=np.stack((curves['numerator_u'],curves['numerator_v'],curves['denominator']),axis=1)
    value=((coeff[:,:,3]*a[:,None]+coeff[:,:,2])*a[:,None]+coeff[:,:,1])*a[:,None]+coeff[:,:,0]
    derivative=(3*coeff[:,:,3]*a[:,None]+2*coeff[:,:,2])*a[:,None]+coeff[:,:,1]
    with np.errstate(divide='ignore',invalid='ignore'):
        uv=value[:,:2]/value[:,2,None]
        tangent=(derivative[:,:2]*value[:,2,None]-value[:,:2]*derivative[:,2,None])/value[:,2,None]**2
    valid=np.isfinite(uv).all(axis=1)&np.isfinite(tangent).all(axis=1)&(abs(value[:,2])>1e-14)
    return uv,tangent,valid


def recover_query_roots(scene,camera,material,branch,station,frame_time,curves,alpha_seed=None):
    s=np.asarray(station,float).copy();bounds=curves['bounds']
    if alpha_seed is None:
        alpha=np.clip(np.zeros(len(s)),bounds[:,0],bounds[:,1])
    else:alpha=np.clip(np.asarray(alpha_seed,float),bounds[:,0],bounds[:,1])
    original_s=s.copy();original_alpha=alpha.copy()
    def project(ss):
        world,finite,_=scene.world_rows(material,branch,ss)
        output=camera.project_rs(world,frame_time);output['valid'] &= finite
        return output
    valid=np.ones(len(s),bool);det=np.full(len(s),np.nan);conditioning=np.full(len(s),np.inf)
    for iteration in range(20):
        out=project(s);uv,da,qgood=rational_query(curves,alpha);e=out['uv']-uv
        sp=np.minimum(1.,s+1e-5);sm=np.maximum(0.,s-1e-5)
        plus=project(sp);minus=project(sm);ds=(plus['uv']-minus['uv'])/(sp-sm)[:,None]
        # J=[projection'(s), -query'(alpha)], solved independently of AL witnesses.
        det=-ds[:,0]*da[:,1]+ds[:,1]*da[:,0]
        product=np.linalg.norm(ds,axis=1)*np.linalg.norm(da,axis=1)
        conditioning=np.divide(abs(det),product,out=np.zeros_like(det),where=product>0)
        good=out['valid']&plus['valid']&minus['valid']&qgood&(conditioning>1e-7)&(abs(det)>1e-10)
        valid &= good
        active=(np.linalg.norm(e,axis=1)>1e-8)&valid
        if not np.any(active):break
        step_s=np.divide(-da[:,1]*e[:,0]+da[:,0]*e[:,1],det,out=np.zeros_like(s),where=good)
        step_a=np.divide(-ds[:,1]*e[:,0]+ds[:,0]*e[:,1],det,out=np.zeros_like(s),where=good)
        # Bounded simultaneous step; never turn an out-of-support query into a root.
        factor=np.minimum(1.,np.divide(.1,abs(step_s),out=np.ones_like(s),where=abs(step_s)>0))
        candidate_s=s-factor*step_s;candidate_a=alpha-factor*step_a
        inside=(candidate_s>=0)&(candidate_s<=1)&(candidate_a>=bounds[:,0])&(candidate_a<=bounds[:,1])
        valid &= inside|~active
        s=np.where(active&inside,candidate_s,s);alpha=np.where(active&inside,candidate_a,alpha)
    out=project(s);uv,da,qgood=rational_query(curves,alpha);e=out['uv']-uv
    sp=np.minimum(1.,s+1e-5);sm=np.maximum(0.,s-1e-5);plus=project(sp);minus=project(sm)
    ds=(plus['uv']-minus['uv'])/(sp-sm)[:,None];det=-ds[:,0]*da[:,1]+ds[:,1]*da[:,0]
    product=np.linalg.norm(ds,axis=1)*np.linalg.norm(da,axis=1)
    conditioning=np.divide(abs(det),product,out=np.zeros_like(det),where=product>0)
    valid &= plus['valid']&minus['valid']&(conditioning>1e-7)&(abs(det)>1e-10)
    out.update(stations=s,alpha=alpha,query_uv=uv,query_equality_px=np.linalg.norm(e,axis=1),
        local_jacobian_determinant=det,local_transversality_sine=conditioning,
        valid=out['valid']&valid&qgood&(np.linalg.norm(e,axis=1)<=1e-7),
        relevant_root_interval_certified=np.zeros(len(s),bool),
        root_certificate_limitation='branch-local numerical nonsingularity only; independent relevant-interval enclosure pending')
    return _recover_failed_finite_roots(scene,camera,material,branch,frame_time,curves,
        original_s,original_alpha,out)


def _recover_failed_finite_roots(scene,camera,material,branch,frame_time,curves,
                                 original_s,original_alpha,out):
    """Recover failed rows without requiring the initial fixed-S RS root.

    The numerical search has the original finite station, query scalar and
    native image-row bounds. Its three equations use camera forward projection
    at the unknown row time. Only a true whole-readout RS root and a strict
    analytic interval local-root certificate can promote a failed row. Existing
    successful rows are never recomputed or rewritten. Failure is unresolved,
    not a claim of nonexistence on the complete finite query interval.
    """
    n=len(original_s);failed=np.flatnonzero(~out['valid'])
    attempted=np.zeros(n,bool);succeeded=np.zeros(n,bool)
    reason=np.full(n,'not_needed',dtype='U160')
    out.update(finite_three_variable_fallback_attempted=attempted,
        finite_three_variable_fallback_succeeded=succeeded,
        finite_three_variable_fallback_reason=reason)
    if not len(failed):return out
    if not callable(getattr(camera,'at',None)) or not hasattr(camera,'tau'):
        reason[failed]='native_row_time_projection_unavailable';return out
    from scipy.optimize import least_squares
    from .native_query_certificate import certify_profile_roots
    material=np.asarray(material);branch=np.asarray(branch)
    frame_time=np.broadcast_to(np.asarray(frame_time,float),(n,))
    for i in failed:
        attempted[i]=True
        c={k:np.asarray(curves[k])[i:i+1] for k in
            ['numerator_u','numerator_v','denominator','bounds']}
        low=np.array([0.,c['bounds'][0,0],0.])
        high=np.array([1.,c['bounds'][0,1],1079.])
        if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(low>=high):
            reason[i]='invalid_original_finite_query_bounds';continue
        m=material[i:i+1];b=branch[i:i+1];t=frame_time[i:i+1]
        a0=float(np.clip(original_alpha[i],low[1],high[1]))
        q0,_,qgood=rational_query(c,np.array([a0]))
        if not qgood[0]:reason[i]='invalid_original_query_seed';continue
        # Clipping supplies an initial guess only; final physical projection
        # and the proof retain their exact finite bounds without clamping.
        x0=np.array([np.clip(original_s[i],0.,1.),a0,np.clip(q0[0,1],0.,1079.)])
        def equations(x):
            world,finite,_=scene.world_rows(m,b,np.array([x[0]]))
            projection=camera.at(world,t+camera.tau*(x[2]/1079.-.5))
            q,_,legal=rational_query(c,np.array([x[1]]))
            uv=projection['uv'][0]
            if not finite[0] or not legal[0] or not np.isfinite(uv).all():
                return np.full(3,1e12)
            return np.array([uv[0]-q[0,0],uv[1]-q[0,1],x[2]-uv[1]])
        reason[i]='finite_three_variable_residual_unresolved'
        try:
            fit=least_squares(equations,x0,bounds=(low,high),jac='3-point',
                x_scale=np.array([1.,max(1.,high[1]-low[1]),1079.]),
                ftol=1e-12,xtol=1e-12,gtol=1e-12,max_nfev=160)
            x=fit.x
            if (not np.isfinite(x).all() or np.any(x<low) or np.any(x>high)
                    or np.max(abs(equations(x)))>1e-7):continue
            world,finite,_=scene.world_rows(m,b,np.array([x[0]]))
            candidate=camera.project_rs(world,t)
            q,da,qgood=rational_query(c,np.array([x[1]]))
            error=float(np.linalg.norm(candidate['uv'][0]-q[0]))
            if not (finite[0] and candidate['valid'][0] and qgood[0] and error<=1e-7):
                reason[i]='finite_three_variable_true_RS_or_query_check_failed';continue
            certificate=certify_profile_roots(scene,camera,m,b,np.array([x[0]]),t,c,
                np.array([x[1]]),candidate['row'],numerical_valid=np.array([True]))
            if not certificate['certified'][0]:
                reason[i]='finite_three_variable_'+str(certificate['reason'][0]);continue
            J=(certificate['jacobian_lower'][0]+certificate['jacobian_upper'][0])/2
            vs=J[1,0]/J[2,2]
            ds=np.array([J[0,0]+J[0,2]*vs,vs])
            determinant=-ds[0]*da[0,1]+ds[1]*da[0,0]
            sine=abs(determinant)/(np.linalg.norm(ds)*np.linalg.norm(da[0]))
            if not (np.isfinite(sine) and sine>1e-7 and abs(determinant)>1e-10):
                reason[i]='finite_three_variable_local_transversality_unresolved';continue
            for key,value in candidate.items():
                if key=='certificate':
                    for name,array in value.items():
                        if isinstance(array,np.ndarray) and name in out[key]:out[key][name][i]=array[0]
                elif key in out and isinstance(value,np.ndarray):out[key][i]=value[0]
            out['stations'][i]=x[0];out['alpha'][i]=x[1];out['query_uv'][i]=q[0]
            out['query_equality_px'][i]=error
            out['local_jacobian_determinant'][i]=determinant
            out['local_transversality_sine'][i]=sine
            out['valid'][i]=True;out['relevant_root_interval_certified'][i]=True
            succeeded[i]=True;reason[i]='certified_finite_three_variable_local_root'
        except (ValueError,FloatingPointError,RuntimeError,np.linalg.LinAlgError):
            reason[i]='finite_three_variable_numerical_or_certificate_failure'
    out['root_certificate_limitation']='Local finite-branch certificates only; global query-interval uniqueness or nonexistence is not asserted'
    return out
