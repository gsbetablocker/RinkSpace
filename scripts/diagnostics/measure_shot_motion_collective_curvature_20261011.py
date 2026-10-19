"""Contract the original shot-motion J in nine fixed collective directions.

One original native Schur J lives in RAM only; no solve, fitted state, image,
Jacobian cache, or normal-matrix library is produced.
"""
import json
import time
import cv2
import numpy as np

import joint_absolute_source_context_20261011 as ctx
import audit_low6_private_RF4_causal_response_20261011 as local
import audit_shot5_common_rotation_gauge_20261011 as shot
from calibration.program_tracking.z190_physical_camera import ViewState
from calibration.solver_runtime.source_loss import source_irls_weights

a=ctx.a


def main():
    started=time.perf_counter();c=ctx.load_context();q=c.q.copy()
    names=c.arrays['aux__block_names'].tolist();ordinal=names.index('low')
    with np.load(c.path,allow_pickle=False)as z:
        for key in ('profile__weights_'+str(ordinal),'aux__current_depth_and_material_domain_contract_json'):
            c.arrays[key]=z[key].copy()
    motion=local.reconstruct_motion(c);pa,pb=motion['pa'],motion['pb']
    ia_inside=(pa>=shot.LO)&(pa<shot.HI);ib_inside=(pb>=shot.LO)&(pb<shot.HI)
    if np.any(ia_inside^ib_inside):raise ValueError('Cross-shot admitted edges need both endpoint columns')
    motion['selected']=np.flatnonzero(ia_inside|ib_inside)
    ids=motion['selected'];src=motion['source'];stamps=motion['stamps'];ia,ib=src['ia'][ids],src['ib'][ids]
    if len(ids)!=264138:raise ValueError('Original shot5 motion identity changed')
    baseline,checks,cache=shot.motion_values(c,motion,q)
    cameras=cache['cameras'];rotations=cache['rotations'];nviews=len(stamps)
    direction=np.zeros((nviews,4,9));rotation_derivatives=np.empty((nviews,3,3,3));centres=np.empty((nviews,3,4))
    inshot=(stamps>=shot.LO)&(stamps<shot.HI)
    times=stamps[inshot].astype(float)
    time_basis=2*(times-times.min())/(times.max()-times.min())-1
    focal=cameras[inshot,6];zoom=focal-focal.mean();zoom/=np.max(abs(zoom))
    basis=np.column_stack((np.ones(len(times)),time_basis,zoom));axes=np.eye(3)
    skew=lambda x:np.array([[0.,-x[2],x[1]],[x[2],0.,-x[0]],[-x[1],x[0],0.]])
    offset=0;mapping_error=0.
    for v,at in enumerate(stamps):
        b=c.state.bypts[int(at)];start=b.start+4*b.bundle.index[int(at)];view=q[start:start+4]
        cal=b.bundle.calibration;_,_,view_j=cal.camera25_and_jacobians(ViewState(tuple(view[:3]),float(view[3]-np.log(1920.))))
        centres[v]=view_j[:3];R,j=cv2.Rodrigues(view[:3]);rotation_derivatives[v]=j.reshape(3,3,3)
        if not inshot[v]:continue
        chart=rotation_derivatives[v].reshape(3,9).T
        for axis,e in enumerate(axes):
            wanted=-R@skew(e);tangent=np.linalg.lstsq(chart,wanted.ravel(),rcond=None)[0]
            mapping_error=max(mapping_error,float(np.max(abs(chart@tangent-wanted.ravel()))))
            for k in range(3):direction[v,:3,3*k+axis]=tangent*basis[offset,k]
        offset+=1
    if offset!=1381 or mapping_error>1e-13:raise ValueError('Collective rotation chart derivative failed')
    sig=src['motion_sigma'].reshape(-1,2)[ids]
    (raw,Ja,Jb,Jshared),jchecks=motion['old'].native_motion(
        a=src['a'][ids],b=src['b'][ids],fa=np.exp(cameras[ia,6]),fb=np.exp(cameras[ib,6]),
        ra=rotations[ia],rb=rotations[ib],da=rotation_derivatives[ia],db=rotation_derivatives[ib],
        ca=cameras[ia,:3],cb=cameras[ib,:3],dca_shared=np.zeros((3,4)),dcb_shared=np.zeros((3,4)),
        dca_view=centres[ia],dcb_view=centres[ib],pp=cameras[ia[0],8:10],k1=cameras[ia[0],11],planes=cache['planes'],
        jacobian=True,sigma=sig,profile_weight=motion['profile'][ids],minimum_source_depth_scene_units=motion['floor'],schur=True)
    raw=np.asarray(raw).reshape(-1,2)
    if not np.array_equal(raw,baseline):raise ValueError('Original r differs between public value/J calls')
    # Match the production outer retirement transformation and then its
    # Huber2 IRLS exactly. The frozen depth-profile weights are already an
    # input to native_motion; they are not multiplied a second time here.
    retirement=c.manifest['static_motion_source_role_retirement']
    scalar=np.asarray(retirement['global_scalar_slots'],int)-retirement['programme_block']['original_start']
    point=scalar//2;positions=np.searchsorted(ids,point);inside=positions<len(ids)
    inside[inside]&=ids[positions[inside]]==point[inside]
    active=np.ones((len(ids),2),bool);active[positions[inside],scalar[inside]%2]=False
    r=np.where(active,raw/sig,0.)
    weight=source_irls_weights(r.ravel(),np.zeros(r.size,bool),None).reshape(-1,2)/sig**2
    weight[~active]=0.
    AA,BB=Ja.reshape(-1,2,4),Jb.reshape(-1,2,4)
    response=np.einsum('nsi,nik->nsk',AA,direction[ia])+np.einsum('nsi,nik->nsk',BB,direction[ib])
    diag=np.zeros((nviews,4));same=ia==ib
    np.add.at(diag,ia[~same],np.einsum('ns,nsi,nsi->ni',weight[~same],AA[~same],AA[~same]))
    np.add.at(diag,ib[~same],np.einsum('ns,nsi,nsi->ni',weight[~same],BB[~same],BB[~same]))
    together=AA[same]+BB[same]
    np.add.at(diag,ia[same],np.einsum('ns,nsi,nsi->ni',weight[same],together,together))
    gram=np.einsum('ns,nsi,nsj->ij',weight,response,response)
    masks=dict(unknown_depth=~src['finite_mask'][ids],finite_ice=motion['ice'][ids],finite_board=motion['board'][ids])
    branches={name:np.einsum('ns,nsi,nsj->ij',weight[mask],response[mask],response[mask])for name,mask in masks.items()}
    diagonal_penalty=np.einsum('vi,vik,vik->k',diag,direction,direction)
    floor_penalty=np.einsum('vi,vik,vik->k',np.maximum(diag,1.),direction,direction)
    iteration=a.read(a.RUN/'solver/iterations.json')[0];lam=float(iteration['damping'])
    rows=[]
    for k in range(9):
        actual=float(gram[k,k]);penalty=float(diagonal_penalty[k]);row=dict(
            mode=('common_world_rotation','time_linear_world_rotation','current_logf_centered_world_rotation')[k//3],
            world_axis=('x','y','z')[k%3],motion_GN_directional_curvature=actual,
            motion_diagonal_directional_penalty=penalty,diagonal_over_data=penalty/actual if actual>0 else None,
            last_actual_lambda=lam,last_lambda_times_motion_diagonal=float(lam*floor_penalty[k]),
            last_lambda_motion_lower_bound_over_motion_data=float(lam*floor_penalty[k]/actual)if actual>0 else None,
            data_curvature_by_branch={name:float(matrix[k,k])for name,matrix in branches.items()})
        rows.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
    assert np.array_equal(c.q,q)
    del Ja,Jb,Jshared,AA,BB,response
    print(json.dumps(dict(stage='completed_collective_motion_curvature',rows=rows,
        points=len(ids),private_RF4_views=int(inshot.sum()),retired_source_scalars=int((~active).sum()),
        same_actual_endpoint_points=int(same.sum()),physical_rotation_derivative_mapping_error=mapping_error,
        native_value_equals_original_J_value=True,original_frozen_profile_and_sigma=True,source_IRLS_once=True,
        curvature_is_original_Schur_Gauss_Newton_not_exact_nonlinear_Hessian=True,
        motion_only_not_complete_source_normal=True,diagonal_is_lower_bound_for_complete_nonnegative_source_normal=True,
        numerical_damping_comparison_uses_last_original_lambda_not_a_new_QP=True,
        optimized=False,new_q_or_images_or_J_or_normal_files_written=False,seconds=time.perf_counter()-started),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
