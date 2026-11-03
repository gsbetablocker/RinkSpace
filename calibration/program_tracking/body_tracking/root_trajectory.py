"""Conditional root initializers and regularization of the actual trajectory.

An ankle keypoint remains an anatomical image proxy. Its height relative to
an attached runner is a model estimate used only to initialize a candidate;
these samples are never exported as blade pixels or contact observations.
"""
import numpy as np
from scipy.interpolate import BSpline
from .camera_bridge import rays, native_to_rectified, rectified_intrinsics


def proxy_root_samples(camera25, uv, qualified, relative_coco, runner_z,
                       contact_candidate, known_foot, metres_per_scene_unit):
    n=len(camera25); result=np.full((n,3),np.nan); selected=np.full(n,-1,int)
    for i in np.flatnonzero(contact_candidate):
        foot=int(known_foot[i])
        feet=[foot] if foot in (0,1) else [0,1]
        feet=[f for f in feet if qualified[i,15+f]]
        if not feet:continue
        # An unknown side remains unknown evidence. This tentative choice is
        # recorded separately and cannot create a support objective row.
        foot=min(feet,key=lambda f:runner_z[i,f])
        proxy=relative_coco[i,15+foot]
        height=proxy[2]-runner_z[i,foot]
        ray,good=rays(uv[i,15+foot][None],camera25[i])
        if not good[0] or abs(ray[0,2])<1e-8:continue
        centre=camera25[i,:3]*metres_per_scene_unit
        distance=(height-centre[2])/ray[0,2]
        if distance<=0:continue
        result[i]=centre+distance*ray[0]-proxy
        selected[i]=foot
    return result,selected


def smooth_root_samples(times, samples, *, spacing=.2, sigma_m=.15, acceleration_mps2=15.):
    times=np.asarray(times,float);valid=np.isfinite(samples).all(axis=1)
    if valid.sum()<4:raise ValueError('At least four actual conditional root samples required')
    a,b=times[[0,-1]];knots=np.r_[[a]*4,np.arange(a+spacing,b-1e-8,spacing),[b]*4]
    B=BSpline.design_matrix(times,knots,3).toarray()
    acceleration=BSpline(knots,np.eye(B.shape[1]),3)(times,2)/acceleration_mps2
    weights=np.ones(valid.sum());target=samples[valid]
    for _ in range(5):
        scale=np.sqrt(weights)/sigma_m
        A=np.vstack((B[valid]*scale[:,None],acceleration))
        controls=np.linalg.lstsq(A,np.vstack((target*scale[:,None],np.zeros((len(times),3)))),rcond=None)[0]
        distance=np.linalg.norm(B[valid]@controls-target,axis=1)/sigma_m
        weights=1/np.maximum(1.,distance)
    return BSpline(knots,controls,3)(times)


def image_height_given_xy(camera25, uv, qualified, relative_coco, root_xy, metres_per_scene_unit):
    """Body-ray least squares in the remaining vertical translation only."""
    import cv2
    result=np.full(len(camera25),np.nan)
    for i,camera in enumerate(camera25):
        ids=np.flatnonzero(qualified[i])
        if len(ids)<4:continue
        rect,good=native_to_rectified(uv[i,ids],camera);ids=ids[good];rect=rect[good]
        if len(ids)<4:continue
        K=rectified_intrinsics(camera);ray=(rect-K[:2,2])/np.diag(K)[:2]
        R=cv2.Rodrigues(camera[3:6])[0]
        base=np.r_[root_xy[i],0.]-camera[:3]*metres_per_scene_unit
        local=(relative_coco[i,ids]+base)@R.T
        column=R[:,2]
        A=(column[:2]-ray*column[2]).ravel()
        rhs=(ray*local[:,2,None]-local[:,:2]).ravel()
        denominator=A@A
        if denominator>1e-12:result[i]=A@rhs/denominator
    return result
