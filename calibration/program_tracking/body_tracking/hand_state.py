"""Fixed hand curves on the original source clock; no extra optimization DOFs."""
import numpy as np
from scipy.spatial.transform import Rotation,RotationSpline
SCHEMA='rinkspace.hand-local-R.v1'
MEAN='flat_hand_mean=True; actual local rotations; no additive mean'
META=('hand_pose_schema_version','hand_mean_convention','hand_pose_origin','hand_prior_sha256')

def validate_hand(value,n):
    a=np.asarray(value,dtype=float)
    if a.shape!=(n,2,15,3,3) or not np.isfinite(a).all():raise ValueError('Expected finite hand rotations (N,2,15,3,3)')
    if not np.allclose(a.swapaxes(-1,-2)@a,np.eye(3),atol=2e-6,rtol=0) or np.any(np.linalg.det(a)<.99999):raise ValueError('Hand rotations must be SO(3)')
    return a

def hand_curves(times,value):
    a=validate_hand(value,len(times)).reshape(len(times),30,3,3)
    return [RotationSpline(times,Rotation.from_matrix(a[:,j])) for j in range(30)]

def evaluate_hands(curves,t):
    return np.stack([c(t).as_matrix() for c in curves],axis=1).reshape(len(t),2,15,3,3)

def prior_hand_fields(prior,target_times,origin_pts,prior_label,prior_sha):
    present=('left_hand_pose' in prior,'right_hand_pose' in prior)
    if not all(present):
        if any(present) or 'exmr' in prior_label.lower():raise ValueError('Explicit ExMR prior requires both 45D hand channels')
        return {}
    pts=np.asarray(prior['pts']);n=len(pts)
    raw=np.stack([np.asarray(prior[k]) for k in ('left_hand_pose','right_hand_pose')],axis=1)
    if raw.shape!=(n,2,45) or not np.isfinite(raw).all():raise ValueError('Expected actual left/right 45D axis angles')
    R=Rotation.from_rotvec(raw.reshape(-1,3)).as_matrix().reshape(n,2,15,3,3)
    curves=hand_curves((pts-origin_pts)/12800.,R)
    return dict(hand_local_R=evaluate_hands(curves,target_times),hand_prior_pts=pts.copy(),hand_prior_local_R=R,
        hand_pose_schema_version=np.array(SCHEMA),hand_mean_convention=np.array(MEAN),
        hand_pose_origin=np.array(prior_label),hand_prior_sha256=np.array(prior_sha))
