"""One soft outside-or-on-board condition per fixed camera installation.

The current rink outline is a working estimate. A signed metre distance is
not a surveyed camera position. Its engineering sigma expresses tolerance;
no camera is translated independently at individual exposures.
"""
import cv2
import numpy as np


def fixed_pivot_board_side_support(polygon_xy_m,pivot_xyz_m,engineering_sigma_m):
    polygon=np.asarray(polygon_xy_m,float);pivot=np.asarray(pivot_xyz_m,float)
    sigma=float(engineering_sigma_m)
    if polygon.ndim!=2 or polygon.shape[1]!=2 or pivot.shape!=(3,)or sigma<=0 or not np.isfinite(sigma):
        raise ValueError('Actual finite metric outline, fixed XYZ and explicit positive engineering tolerance required')
    if not np.isfinite(polygon).all()or not np.isfinite(pivot).all():raise ValueError('Finite board support required')
    start=polygon;delta=np.roll(polygon,-1,axis=0)-start
    length2=np.einsum('ij,ij->i',delta,delta);valid=length2>0
    start,delta,length2=start[valid],delta[valid],length2[valid]
    t=np.clip(np.einsum('ij,ij->i',pivot[:2]-start,delta)/length2,0.,1.)
    closest=start+t[:,None]*delta;differences=pivot[:2]-closest
    distance2=np.einsum('ij,ij->i',differences,differences);index=int(np.argmin(distance2))
    distance=float(np.sqrt(distance2[index]))
    side=cv2.pointPolygonTest(np.ascontiguousarray(polygon,dtype=np.float32),tuple(map(float,pivot[:2])),False)
    signed=distance if side>0 else (-distance if side<0 else 0.)
    residual=np.array([max(signed,0.)/sigma]);jacobian=np.zeros((1,3))
    if signed>0 and distance>0:jacobian[0,:2]=differences[index]/(distance*sigma)
    return residual,jacobian,dict(signed_inside_distance_m=signed,
        positive_means_inside_current_estimated_outline=True,engineering_sigma_m=sigma,
        engineering_sigma_is_not_a_user_measurement=True,hard_inside_rejection=False,
        on_the_boundary_has_zero_error=True,fixed_XYZ_not_per_frame=True)
