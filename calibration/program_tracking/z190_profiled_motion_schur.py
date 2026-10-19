"""Native-source GN elimination of each positive-depth nuisance.

The exact profile residual and its derivatives remain in the tested physical
kernel.  A two-component profiled residual's orientation also changes with
camera parameters. Squaring that full derivative adds residual-dependent
orientation stiffness. Joint camera/depth Gauss-Newton instead eliminates
one raw depth column. Interior unknown rows therefore carry one local
constraint; finite planes and explicit positive-depth endpoints stay intact.
"""
from __future__ import annotations
import numpy as np
from z190_moving_center_motion import moving_center_motion_value_jacobians


def schur_profile_blocks(result,sigma):
    residual,ja,jb,js,state=result
    n=len(residual)//2;sig=np.asarray(sigma,float).reshape(n,2)
    weights=np.asarray(state['frozen_component_weights'],float)/sig**2
    tangent=np.asarray(state['profile_nuisance_tangent_native'],float)
    interior=np.asarray(state['unknown_interior_rows'],bool)
    denominator=np.sum(weights*tangent*tangent,axis=1)
    active=interior&(denominator>1e-22)&~np.asarray(state['profile_nonpositive_hessian_rows'],bool)
    blocks=[];delta=[];res=residual.reshape(n,2)
    for exact,key in zip((ja,jb,js),('raw_unprofiled_Ja','raw_unprofiled_Jb','raw_unprofiled_Jshared')):
        block=exact.reshape(n,2,4).copy();raw=np.asarray(state[key],float)
        factor=np.einsum('ni,nip->np',weights*tangent,raw)/np.maximum(denominator,1e-30)[:,None]
        projected=raw-tangent[:,:,None]*factor[:,None,:]
        block[active]=projected[active]
        difference=np.einsum('ni,nip->np',weights*res,block-exact.reshape(n,2,4))
        delta.append(float(np.max(np.abs(difference),initial=0.)))
        blocks.append(block.reshape(-1,4))
    checks=dict(rows=n,unknown_interior_schur_rows=int(active.sum()),finite_rows=int(np.sum(state['finite_rows'])),
        endpoint_or_unidentified_rows=int(np.sum(~interior&~state['finite_rows'])),
        maximum_native_weighted_gradient_delta=max(delta),
        maximum_tangent_projection_native=float(max(np.max(np.abs(np.einsum('ni,nip->np',weights[active]*tangent[active],x.reshape(n,2,4)[active])),initial=0.)for x in blocks)))
    return(residual,*blocks),checks


class SchurMotionKernel:
    def __init__(self,*,minimum_source_depth_scene_units=0.,schur=True):
        self.latest_checks=[]
        self.minimum_source_depth_scene_units=float(minimum_source_depth_scene_units)
        self.schur=bool(schur)
    def __call__(self,**k):
        planes=k.pop('planes');jac=k.pop('jacobian');sigma=k['sigma']
        result=moving_center_motion_value_jacobians(k.pop('a'),k.pop('b'),k.pop('fa'),k.pop('fb'),k.pop('ra'),k.pop('rb'),
            k.pop('da'),k.pop('db'),k.pop('ca'),k.pop('cb'),k.pop('dca_shared'),k.pop('dcb_shared'),k.pop('dca_view'),k.pop('dcb_view'),
            principal_px=k.pop('pp'),k1=k.pop('k1'),sigma=k.pop('sigma'),profile_weight=k.pop('profile_weight'),
            finite_normals=planes[:,:3],finite_d=-planes[:,3],finite_mask=np.linalg.norm(planes[:,:3],axis=1)>0.,jac=jac,return_state=jac,
            minimum_source_depth_scene_units=self.minimum_source_depth_scene_units)
        if not jac:return result
        if not self.schur:return result[:4]
        blocks,checks=schur_profile_blocks(result,sigma);self.latest_checks.append(checks)
        return blocks
