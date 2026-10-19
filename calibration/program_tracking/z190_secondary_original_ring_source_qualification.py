"""Opt-in requalification of the 31 original secondary R2/R3 boundary reads.

The original development UV, engineering sigma, identities and origins stay
immutable. Twenty-nine source-qualified outer transitions replace the old
nearest-curve objective; two unknown tips remain historical report records.
No import prepares source motion or starts a solve.
"""
from __future__ import annotations
from copy import copy
from pathlib import Path
from types import SimpleNamespace
import hashlib,json
import numpy as np
from projection import project_central_valid
from reference_board_pair import BoardIntersectionError
from z190_secondary_connected_cap_source import connected_cap_absolute_value_jac15
from z190_source_camera_diagnostic import brown4_world_jac

SOURCE_SHA256='d28ad433a3194546f227a4da15ab1daee645248d4f458bcb0977c80d7d5f533e'
PTS=229528320
STRICT_KIND='original_outer_source_qualification_strict_scanline'
OPERATOR_REVISION='same_source_branch_finite_extent_v2'


def _source_branch_extent_indices(projected,branch):
    """Endpoints where the same upper/lower scanline envelope meets its extent.

    A vertical extremal segment has two different branch endpoints. Resolve
    that tie using the declared source branch, never the nearest source Y.
    The finite sampled curve is piecewise differentiable; an extremum/segment
    switch remains a genuine nonsmooth point rather than a fictitious C1 fit.
    """
    if branch not in ('min_y','max_y'):
        raise ValueError('Expected the immutable source upper/lower branch')
    select=np.argmin if branch=='min_y'else np.argmax
    endpoints=[]
    for extent in (np.min(projected[:,0]),np.max(projected[:,0])):
        indices=np.flatnonzero(projected[:,0]==extent)
        endpoints.append(int(indices[select(projected[indices,1])]))
    return endpoints


def qualified_original_ring_absolute_value_jac15(camera,rows,jacobian=True):
    """Keep the typed scanline inside; measure same-branch XY error outside.

    UV/support/physical branch stay immutable. A wrong predicted extent is a
    finite lateral error, not a failed camera projection. Outside the extent,
    the branch ends at its own physical endpoint; both X and Y residuals and
    their camera derivatives are retained. Inside values/J are unchanged.
    """
    values=[];blocks=[]
    for row in rows:
        if row.get('original_outer_source_qualification_operator')!=STRICT_KIND:
            result=connected_cap_absolute_value_jac15(camera,[row],jacobian)
            if jacobian:value,matrix=result;values.append(value);blocks.append(matrix)
            else:values.append(result)
            continue
        projected,depth,valid=project_central_valid(camera,np.asarray(row['world'],float))
        good=valid&(depth>0.)&np.isfinite(projected).all(axis=1)
        if not np.all(good):
            raise BoardIntersectionError('qualified_outer_curve_not_central_forward',np.flatnonzero(~good))
        x=np.asarray(row['uv'])[:,0];lo,hi=np.min(projected[:,0]),np.max(projected[:,0])
        outside=(x<lo)|(x>hi)
        # Legacy intervals required zero to be inside support. Requalification
        # can establish that the immutable legacy read is on the other side
        # of the actual outer transition. Keep that original UV and use the
        # same signed clip formula, without expanding the new source interval.
        rawrow=dict(row);rawrow.pop('normal_interval')
        result=connected_cap_absolute_value_jac15(camera,[rawrow],jacobian)
        if jacobian:value,matrix=result
        else:value=result
        normals=np.asarray(row['normal'],float);bounds=np.asarray(row['normal_interval'],float)
        sigma=np.asarray(row['sigma'],float)
        if (normals.shape!=(len(row['uv']),2)or not np.allclose(normals,[0.,1.],rtol=0.,atol=0.)
                or bounds.shape!=normals.shape or not np.isfinite(bounds).all()
                or np.any(bounds[:,0]>=bounds[:,1])):
            raise ValueError('Qualified one-sided native normal support is invalid')
        if np.any(outside):
            endpoints=_source_branch_extent_indices(projected,row['scanline_branch'])
            selected=np.where(x[outside]<lo,endpoints[0],endpoints[1])
            value=value.copy()
            value.reshape(-1,2)[outside]=(projected[selected]-np.asarray(row['uv'])[outside])/sigma[outside,None]
            if jacobian:
                _,depth,j11,valid=project_central_valid(camera,np.asarray(row['world'],float),with_jacobian=True)
                good=valid&(depth>0.)&np.isfinite(j11.reshape(-1,2,11)).all(axis=(1,2))
                if not np.all(good):
                    raise BoardIntersectionError('qualified_outer_curve_not_central_forward',np.flatnonzero(~good))
                allj=np.c_[j11,brown4_world_jac(camera,np.asarray(row['world'],float)).reshape(-1,4)].reshape(-1,2,15)
                matrix=matrix.copy()
                matrix.reshape(-1,2,15)[outside]=allj[selected]/sigma[outside,None,None]
        limits=bounds/sigma[:,None];raw=value.reshape(-1,2).copy()
        signed=np.einsum('ni,ni->n',raw,normals)
        raw-=normals*np.clip(signed,limits[:,0],limits[:,1])[:,None]
        values.append(raw.ravel())
        if jacobian:
            inside=(signed>limits[:,0])&(signed<limits[:,1]);mapped=matrix.reshape(len(raw),2,-1).copy()
            selected=normals[inside];component=np.einsum('ni,nij->nj',selected,mapped[inside])
            mapped[inside]-=selected[:,:,None]*component[:,None,:]
            blocks.append(mapped.reshape(matrix.shape))
    value=np.concatenate(values)if values else np.empty(0)
    return(value,np.vstack(blocks)if blocks else np.empty((0,15)))if jacobian else value


def load_qualification(path):
    path=Path(path);payload=path.read_bytes();digest=hashlib.sha256(payload).hexdigest()
    if digest!=SOURCE_SHA256:raise ValueError('Original ring qualification packet SHA changed')
    packet=json.loads(payload)
    if (packet['schema']!='rinkspace.secondary-original-R2-R3-source-outer-qualification.v3'
            or packet['source_actual_pts']!=PTS or packet['source_native_pts']!=PTS
            or not packet['not_blind'] or len(packet['records'])!=2):
        raise ValueError('Unexpected original development source qualification')
    return packet,dict(path=str(path.resolve()),sha256=digest)


def qualify_absolute_groups(groups,aliases,path):
    """Pure row transaction; return new groups, unknown reports and ledger."""
    packet,authority=load_qualification(path)
    if int(aliases[PTS])!=PTS:raise ValueError('Original source canonical exposure changed')
    evidence={r['physical_curve_id']:r for r in packet['records']}
    if set(evidence)!={'R2_outer','R3_outer'}:raise ValueError('Physical circle source identities changed')
    result=[];unknown=[];changed=[];removed=[];removed_sigma=[];cursor=0;matched=set()
    for actual,rows in groups:
        updated=[]
        for row in rows:
            count=len(row['uv']);identity=row['id']
            if int(actual)!=PTS or identity not in evidence:
                updated.append(row);cursor+=2*count;continue
            if identity in matched:raise ValueError('Original physical source group duplicated')
            matched.add(identity);rec=evidence[identity];original=rec['original_source_record']['row']
            for key in ('id','pts','split','origin','point_correspondence'):
                if row.get(key)!=original.get(key):raise ValueError('Original source metadata changed: '+key)
            for key in ('uv','sigma','world'):
                if not np.array_equal(np.asarray(row[key]),np.asarray(original[key])):
                    raise ValueError('Original source numeric field changed: '+key)
            if row.get('source_role')!=original.get('source_role') or row['split']!='fit':
                raise ValueError('Original development source role changed')
            qualified=rec['source_qualified_rows'];unqualified=rec['source_unknown_rows']
            indices=[int(r['original_point_index'])for r in qualified]
            unknown_indices=[int(r['original_point_index'])for r in unqualified]
            if sorted(indices+unknown_indices)!=list(range(count))or len(set(indices+unknown_indices))!=count:
                raise ValueError('Original point inventory is not an exact partition')
            if len(unqualified)!=1 or len(qualified)!=(16 if identity=='R2_outer'else 13):
                raise ValueError('Qualified/unknown physical source counts changed')
            for measurement in qualified+unqualified:
                index=int(measurement['original_point_index'])
                if (not np.array_equal(measurement['source_uv'],np.asarray(row['uv'])[index])
                        or float(measurement['sigma_px'])!=float(np.asarray(row['sigma'])[index])
                        or measurement['split']!=row['split'] or measurement['id']!=identity
                        or int(measurement['source_native_pts'])!=PTS
                        or int(measurement['source_actual_pts'])!=PTS):
                    raise ValueError('Qualification changed an original point/role/sigma/PTS')
            for branch in ('upper','lower'):
                measurements=[r for r in qualified if r['source_branch']==branch]
                selected=[int(r['original_point_index'])for r in measurements]
                if not selected:raise ValueError('Expected both physical outer branches')
                if (any(r['source_normal_uv']!=[0.,1.] or not r['source_visible']
                        or r['model_projection_used_to_generate_or_select_uv_or_support']
                        or r['qualification']!='SOURCE_QUALIFIED_OUTER_TRANSITION'for r in measurements)):
                    raise ValueError('Unqualified source boundary cannot enter objective')
                typed=dict(row,uv=np.asarray(row['uv'])[selected].copy(),sigma=np.asarray(row['sigma'])[selected].copy(),
                    normal=np.asarray([r['source_normal_uv']for r in measurements],float),
                    normal_interval=np.asarray([r['normal_interval_px']for r in measurements],float),
                    physical_curve_id=identity,canonical_physical_geometry_id=identity,
                    physical_curve_id_authority=authority,source_branch_operator_authority=authority,
                    curve_observation_kind='axis_scanline_branch',scanline_fixed_axis='x',
                    scanline_branch='min_y'if branch=='upper'else'max_y',must_preserve_upper_branch=branch=='upper',
                    original_outer_source_qualification_operator=STRICT_KIND,
                    original_outer_source_operator_revision=OPERATOR_REVISION,
                    original_source_point_indices=selected,original_source_qualification=authority,
                    original_full_source_uv=np.asarray(row['uv']).copy(),original_full_source_sigma=np.asarray(row['sigma']).copy(),
                    original_development_qualification_update_not_blind=True,
                    source_qualified_measurements=measurements,solver_admitted=True,
                    actual_admission='original_development_source_qualification_updated',
                    sigma_modelling_decision='Original per-point engineering sigma preserved exactly; no new correlation multiplier',
                    point_correspondence=False,source_world_station_assigned=False,
                    projected_endpoint_fallback_allowed=False,
                    finite_same_source_branch_extent_extension_allowed=True,
                    different_physical_branch_or_nearest_arc_fallback_allowed=False)
                if 'source_uv'in row:typed['source_uv']=np.asarray(row['source_uv'])[selected].copy()
                if not np.isfinite(typed['normal_interval']).all()or np.any(typed['normal_interval'][:,0]>typed['normal_interval'][:,1]):
                    raise ValueError('Source boundary finite support is invalid')
                updated.append(typed)
            for measurement,index in zip(unqualified,unknown_indices):
                if measurement['normal_interval_px']is not None or measurement['source_normal_uv']is not None:
                    raise ValueError('Unknown source must not invent a finite support')
                historical=dict(row,uv=np.asarray(row['uv'])[[index]].copy(),sigma=np.asarray(row['sigma'])[[index]].copy(),
                    original_source_point_indices=[index],original_source_qualification=authority,
                    original_source_unknown_measurement=measurement,solver_admitted=False,
                    actual_admission='unknown_original_development_anchor_reportonly',
                    review_report_role='historical_original_UV_comparison_only_not_qualified_boundary_error',
                    source_support_unknown=True,not_blind=True)
                if 'source_uv'in row:historical['source_uv']=np.asarray(row['source_uv'])[[index]].copy()
                unknown.append(historical);removed.extend([cursor+2*index,cursor+2*index+1])
                removed_sigma.extend([float(np.asarray(row['sigma'])[index])]*2)
            changed.append(dict(id=identity,pts=PTS,original_origin=row['origin'],original_points=count,
                qualified_points=len(qualified),unknown_original_point_indices=unknown_indices,
                original_uv_sigma_split_world_kept=True,old_nearest_curve_objective_replaced_not_duplicated=True,
                physical_branch_support_not_point_correspondence=True))
            cursor+=2*count
        result.append((int(actual),updated))
    if matched!=set(evidence)or len(unknown)!=2 or len(removed)!=4:
        raise ValueError('The two original groups were not found exactly once')
    new_count=sum(2*len(r['uv'])for _,rows in result for r in rows)
    if new_count!=cursor-4:raise ValueError('Qualification changed unrelated source rows')
    ledger=dict(schema='rinkspace.original-secondary-ring-source-qualification-admission.v1',source=authority,
        operator_revision=OPERATOR_REVISION,objective_change_requires_fresh_jacobian_and_profile=True,
        outside_predicted_extent_retains_xy_error=True,in_domain_value_and_jacobian_unchanged=True,
        original_development_source_requalification=True,not_blind=True,
        original_absolute_scalar_rows=cursor,new_absolute_scalar_rows=new_count,
        removed_original_absolute_scalar_indices=removed,removed_original_scalar_sigma=removed_sigma,
        qualified_original_points=29,unknown_original_report_points=2,groups=changed,
        all_other_absolute_rows_motion_materials_aliases_and_roles_unchanged=True,
        source_qualified_world_accuracy_accepted=False,solve_started=False)
    return result,unknown,ledger


def qualify_before_source_layout_rebuild(prepared,path):
    """Shallow source transaction; caller must immediately rebuild cap bundle.

    Original large motion/profile arrays are shared. The driver already makes
    one cap bundle, which supplies the new source CSR; do not allocate a second
    complete motion layout for this two-group qualification.
    """
    old=prepared.bundle;groups,unknown,ledger=qualify_absolute_groups(old.abs_groups,prepared.aliases,path)
    bundle=copy(old);bundle.abs_groups=groups
    bundle.native_residual_sigma=np.r_[old.motion_sigma,np.concatenate([np.repeat(r['sigma'],2)for _,rows in groups for r in rows])]
    bundle.source_residual_rows=len(bundle.native_residual_sigma)
    data=vars(prepared).copy();data.update(bundle=bundle,original_ring_source_qualification=ledger)
    return SimpleNamespace(**data),unknown,ledger


def restore_prequalification_contract(quadratic,scales,absolute_start,ledger):
    """Reinsert exactly the four unknown legacy scalar rows for seed checking."""
    quadratic=np.asarray(quadratic,bool);scales=np.asarray(scales,float)
    if quadratic.shape!=scales.shape:raise ValueError('Normalized source contract shape mismatch')
    removed=np.asarray(ledger['removed_original_absolute_scalar_indices'],int)+int(absolute_start)
    size=len(scales)+4
    if len(set(removed.tolist()))!=4 or np.any(removed<0)or np.any(removed>=size):
        raise ValueError('Explicit source qualification row transaction is invalid')
    keep=np.ones(size,bool);keep[removed]=False
    previous_q=np.empty(size,bool);previous_s=np.empty(size,float)
    previous_q[keep]=quadratic;previous_s[keep]=scales
    previous_q[removed]=False;previous_s[removed]=ledger['removed_original_scalar_sigma']
    return previous_q,previous_s
