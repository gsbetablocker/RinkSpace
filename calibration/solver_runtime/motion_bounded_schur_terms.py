"""Original motion-chunk moments for the eliminated lambda lower boundary.

The source kernel, frozen profile metric, original sigma and chunk order are
unchanged. These optional numerical moments do not add rows or a depth prior.
Only converged, unknown, nominal interior Schur rows have a lower-bound model;
finite planes and existing endpoint/unidentified rows keep their original J.
"""
from __future__ import annotations

import numpy as np


def _wrapper_chain(source):
    seen=set();current=source
    while current is not None:
        if id(current) in seen:raise ValueError('Cyclic complete source wrapper chain')
        seen.add(id(current));yield current
        fields=vars(current)
        current=fields.get('base',fields.get('parent'))


def _family_loss_and_retirement(source,family,start,rows):
    """Read the actual installed payload, never infer a sampling policy."""
    payload=None;retired=[]
    for owner in _wrapper_chain(source):
        # Outermost composed payload is exactly what blocks() exposes.
        values=vars(owner).get('_payloads',{})
        if payload is None and family in values:payload=values[family]
        for name in ('original_role_retirement','static_motion_role_retirement'):
            role=vars(owner).get(name)
            if role is None:continue
            ids=np.asarray(getattr(role,'active_global_rows',getattr(role,'global_rows',())),int)
            retired.extend(ids[(ids>=start)&(ids<start+rows)].tolist())
    return payload,np.unique(np.asarray(retired,int))-start


def prepare_programme_bounded_terms(source,q,*,profile_token=None):
    """Prepare only current programme-motion moments in the complete chart.

    No absolute factors, source images, complete source J or checkpoint cache
    are evaluated. Original 65536-point kernel chunks, finite roles, frozen
    profiles, installed loss payloads and source-role retirements are reused.
    Each chunk is returned as sparse global m/v rows plus scalar moments.
    """
    import hashlib
    from scipy.sparse import csr_matrix
    from z190_continuous_source_bundle import SCENE_UNIT_M
    from secondary_optics_candidate.secondary_optics_graph import PHYSICAL_COLUMN_SCALES
    from calibration.solver_runtime.source_loss import source_irls_weights,loss_weight_arrays,SourceLossWeights

    q=np.asarray(q,float)
    if (q.shape!=(int(source.count),)or not np.isfinite(q).all()or
            profile_token is not None and (not isinstance(profile_token,str)or not profile_token)):
        raise ValueError('Exact complete source state and frozen profile token required')
    layout=getattr(source,'parameter_layout',None)
    evaluation=q if layout is None else layout.evaluation_q(q)
    state=source.state
    core=evaluation[:state.structured_programme_coordinate_chart.core_count]
    chunks=[];cursor=0;total_points=0;active_points=0
    for owner in state.blocks:
        bundle=owner.bundle;family='programme:'+owner.name
        nrows=len(owner.quadratic);n=len(bundle.a)
        if bundle.motion_rows!=2*n or np.asarray(source.quadratic[cursor:cursor+nrows]).shape!=(nrows,):
            raise ValueError('Complete original programme motion prefix changed')
        original_points=np.asarray(getattr(bundle,'original_motion_compaction_active_to_old',np.arange(n)))
        if (original_points.shape!=(n,)or not np.issubdtype(original_points.dtype,np.integer)or
                np.any(original_points<0)or len(np.unique(original_points))!=n):
            raise ValueError('Exact one-to-one original motion compaction point identity required')
        payload,retired=_family_loss_and_retirement(source,family,cursor,nrows)
        if payload is None:integration=normalization=None
        else:integration,normalization=loss_weight_arrays(payload,nrows)
        # This is the existing complete owner chart, including axial/logf
        # optical-centre effects in its public cache derivatives.
        local=owner.local(core);cal=bundle.current_calibration(local)
        vv,rr,dr,cc,ds,dv=bundle.cache(local,True)
        focal=np.exp(vv[:,3]);pp=np.array([1920.*cal.principal_x_fraction[0],1080.*cal.principal_y_fraction[0]])
        frozen=bundle.profile_weights
        if frozen is not None and np.shape(frozen)!=(n,2):
            raise ValueError('Complete original frozen motion profile required')
        for lo in range(0,n,65536):
            hi=min(lo+65536,n);ia,ib=bundle.ia[lo:hi],bundle.ib[lo:hi]
            inputs=dict(a=bundle.a[lo:hi],b=bundle.b[lo:hi],fa=focal[ia],fb=focal[ib],
                ra=rr[ia],rb=rr[ib],da=dr[ia],db=dr[ib],ca=cc[ia],cb=cc[ib],
                dca_shared=ds[ia],dcb_shared=ds[ib],dca_view=dv[ia],dcb_view=dv[ib],
                pp=pp,k1=cal.k1[0],planes=bundle.finite_planes[lo:hi],jacobian=True,
                sigma=bundle.motion_sigma[2*lo:2*hi].reshape(-1,2),
                profile_weight=None if frozen is None else frozen[lo:hi])
            terms=native_chunk_terms(inputs,
                minimum_source_depth_scene_units=bundle.motion_kernel.minimum_source_depth_scene_units,
                optics_column_scales=PHYSICAL_COLUMN_SCALES,scene_unit_m=SCENE_UNIT_M)
            size=hi-lo;r=terms['r'];j=terms['tangent'];J=terms['projected_J'];m=terms['projection_factors']
            # Original source-role zeros remain structural zeros here as in
            # the actual stream. They cannot acquire a nuisance correction.
            inactive=retired[(retired>=2*lo)&(retired<2*hi)]-2*lo
            if len(inactive):
                r.reshape(-1)[inactive]=0.;j.reshape(-1)[inactive]=0.
                J.reshape(-1,18)[inactive]=0.
            small_loss=None if payload is None else SourceLossWeights(
                integration[2*lo:2*hi],None if normalization is None else normalization[2*lo:2*hi])
            outer=source_irls_weights(r.reshape(-1),np.asarray(source.quadratic[cursor+2*lo:cursor+2*hi]),small_loss).reshape(size,2)
            gn=np.sum(outer*j*j,axis=1);g=np.sum(outer*j*r,axis=1)
            v=np.einsum('ni,nip->np',outer*j,J)
            active=terms['nominal_interior']&(gn>0.)
            selected=np.flatnonzero(active)
            if not len(selected):continue
            # Original shared4 has a structural padding slot; global pivot
            # metres map by the original fixed-XYZ scatter factor once.
            columns=np.c_[np.broadcast_to(owner.xyz_slots,(size,3)),
                owner.start+4*ia[:,None]+np.arange(4),
                owner.start+4*ib[:,None]+np.arange(4),
                np.broadcast_to(owner.optics_chart.indices,(size,3)),
                np.broadcast_to(owner.optics_chart.mount_indices,(size,3))]
            keep=np.r_[np.arange(3),np.arange(4,18)]
            multiplier=np.r_[np.full(3,1./(100.*SCENE_UNIT_M)),np.ones(14)]
            values=m[:,keep]*multiplier;vvv=v[:,keep]*multiplier
            if layout is not None:columns=layout.evaluation_to_canonical[columns]
            indices=columns[selected].reshape(-1);pointer=np.arange(len(selected)+1,dtype=np.int64)*17
            M=csr_matrix((values[selected].reshape(-1),indices,pointer),shape=(len(selected),len(q)))
            V=csr_matrix((vvv[selected].reshape(-1),indices,pointer.copy()),shape=M.shape)
            M.sum_duplicates();M.sort_indices();V.sum_duplicates();V.sort_indices()
            chunks.append(dict(projection_factors=M,cross_moments=V,lambda0=terms['lambda0'][selected],
                inner_gradient=terms['inner_gradient'][selected],inner_GN=terms['inner_GN'][selected],
                outer_GN=gn[selected],outer_gradient=g[selected],
                source_identity=dict(family=family,active_point_chunk=(lo,hi),
                    active_scalar_prefix=cursor,active_bundle_point_indices=(lo+selected),
                    original_precompaction_global_bundle_indices=original_points[lo+selected],
                    frozen_profile_token=profile_token,
                    actual_frozen_profile_sha256=None if frozen is None else hashlib.sha256(np.ascontiguousarray(frozen[lo:hi]).tobytes()).hexdigest(),
                    structural_retired_scalars_in_chunk=len(inactive),
                    upper_depth_bound_model_unchanged=True)))
            active_points+=len(selected)
        total_points+=n;cursor+=nrows
    if not chunks:raise ValueError('Current complete motion has no identified interior nuisance rows')
    return tuple(chunks)


def native_chunk_terms(native_inputs, *, minimum_source_depth_scene_units,
                       optics_column_scales, scene_unit_m):
    """Return normalized local moments, with all actual motion camera columns.

    Columns are shared4, source-view4, target-view4, optics3, mount3. The caller
    applies its existing global scatter and the padding/shared unit conversion.
    ``projection_factors`` are dimensionless d(lambda)/d(column) factors;
    ``tangent`` and ``projected_J`` are divided by original sigma exactly once.
    The inner metric is retained separately; an outer IRLS weight is never
    multiplied into the original nuisance-profile metric here.
    """
    from z190_motion_native import native_motion
    from secondary_optics_candidate.z190_motion_optics_extension import (
        _line_geometry, _raw_optics, _weights,
    )

    k=dict(native_inputs)
    if k.get('jacobian') is not True:
        raise ValueError('Original native Jacobian inputs required for bounded moments')
    k.pop('return_trace',None);k.pop('return_nuisance',None)
    if any(k.get(name) is None for name in (
            'da','db','dca_shared','dcb_shared','dca_view','dcb_view')):
        raise ValueError('Every original native camera derivative is required')
    scales=np.asarray(optics_column_scales,float)
    unit=float(scene_unit_m)
    if (scales.shape!=(3,)or not np.isfinite(scales).all()or
            not np.isfinite(unit)or unit<=0):
        raise ValueError('Actual three optics column scales and positive scene unit required')
    baseline,checks,trace,state=native_motion(**k,
        minimum_source_depth_scene_units=minimum_source_depth_scene_units,
        schur=True,return_nuisance=True)
    r,ja,jb,js=baseline;n=len(trace)
    sigma=np.asarray(k['sigma'],float)
    if sigma.ndim==1:sigma=sigma.reshape(-1,1)
    sigma=np.broadcast_to(sigma,(n,2))
    active=(trace[:,9]!=0)&(trace[:,12]==0)&(trace[:,3]>1e-22)
    # The new native moments use the very same raw columns and divisor that
    # produced Ja/Jb/Js. No inversion of a projected normal is attempted.
    factors=state['fixed_lambda_projection_factors']
    camera_factors=np.concatenate((factors[:,2],factors[:,0],factors[:,1]),axis=1)
    camera_j=np.concatenate((js.reshape(n,2,4),ja.reshape(n,2,4),jb.reshape(n,2,4)),axis=2)
    geometry=_line_geometry(k,trace[:,0])
    tangent=state['profile_nuisance_tangent_native']
    weights=_weights(k,n)
    raw_optics=_raw_optics(k,trace[:,0],geometry)*scales[None,None,:]
    optic_factors=np.einsum('ni,nip->np',weights*tangent,raw_optics)/np.maximum(trace[:,3],1e-30)[:,None]
    optic_j=raw_optics-tangent[:,:,None]*optic_factors[:,None,:]
    # This is the original shared camera-local lever derivative in metres,
    # including its actual rotation-dependent optical-centre response.
    mount=dict(k)
    ds_a=np.zeros((n,3,4));ds_b=np.zeros((n,3,4))
    ds_a[:,:,:3]=np.transpose(np.broadcast_to(k['ra'],(n,3,3)),(0,2,1))/unit
    ds_b[:,:,:3]=np.transpose(np.broadcast_to(k['rb'],(n,3,3)),(0,2,1))/unit
    mount.update(da=np.zeros((3,3,3)),db=np.zeros((3,3,3)),
        dca_shared=ds_a,dcb_shared=ds_b,dca_view=np.zeros((3,4)),dcb_view=np.zeros((3,4)))
    mounted,_,mount_trace,mount_state=native_motion(**mount,
        minimum_source_depth_scene_units=minimum_source_depth_scene_units,
        schur=True,return_nuisance=True)
    if not np.array_equal(mounted[0],r)or not np.array_equal(mount_trace[:,:13],trace[:,:13]):
        raise ValueError('Original mount replay changed the frozen chunk pixels or depth branch')
    mount_j=mounted[3].reshape(n,2,4)[:,:,:3]
    mount_factors=mount_state['fixed_lambda_projection_factors'][:,2,:3]
    projection_factors=np.concatenate((camera_factors,optic_factors,mount_factors),axis=1)
    projected=np.concatenate((camera_j,optic_j,mount_j),axis=2)/sigma[:,:,None]
    projection_factors[~active]=0.;projected[~active]=0.
    if not np.isfinite(projected).all()or not np.isfinite(projection_factors).all():
        raise ValueError('Finite original eliminated-depth local moments required')
    return dict(
        r=r.reshape(n,2)/sigma,
        tangent=tangent/sigma,
        projected_J=projected,
        projection_factors=projection_factors,
        lambda0=trace[:,0].copy(),
        inner_gradient=trace[:,2].copy(),inner_GN=trace[:,3].copy(),
        nominal_interior=active,
        native_trace=trace,
        local_column_order=('shared4','source_view4','target_view4','optics3','mount3'),
        original_frozen_profile_weights=None if k.get('profile_weight')is None else np.asarray(k['profile_weight']).copy(),
        checks=checks,
        upper_depth_bound_model_unchanged=True,
        finite_endpoint_and_unidentified_rows_unchanged=True,
    )
