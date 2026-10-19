"""Authenticate a small wide normal and check real optical branch seams.

The actual source value, UV, sigma and finite support are never changed.
At a certified optical-domain or finite nearest/clamp seam, a stable real
one-sided derivative can certify the existing piecewise Jacobian. Line search
still evaluates the original source with its current masks at every candidate.
"""
import json,hashlib
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from acceleration import nearest_segments
from wide_anchor_candidate.full_brown_projection import project_central_valid_full_brown
from .paired_programme_chart import CORE_COUNT

def install_same_state_wide_replay(factor,q,camera_indices,path,manifest,*,core_getter=None):
    core_count=int(getattr(factor,'physical_parameter_count',CORE_COUNT))
    physical_core=(lambda value:np.asarray(value)[:core_count]) if core_getter is None else core_getter
    path=Path(path);old_manifest_path=path.parent/'input_manifest.json'
    old=json.loads(old_manifest_path.read_text(encoding='utf8'))
    # JSON source contracts have tuple-valued chart descriptions in memory.
    # Compare their serialized representation, as stored in the authenticated
    # manifest, rather than mistaking tuple versus list for a source change.
    manifest=json.loads(json.dumps(manifest,default=lambda x:x.tolist()if isinstance(x,np.ndarray)else x.item()if isinstance(x,np.generic)else str(x)))
    with np.load(path,allow_pickle=False)as z:
        if not np.array_equal(q,z['q']):raise ValueError('Wide replay q differs from complete current state')
        if not np.array_equal(camera_indices,z['wide_private_indices']):raise ValueError('Wide replay private chart differs')
        if str(z['source_manifest_sha256'])!=hashlib.sha256(old_manifest_path.read_bytes()).hexdigest():raise ValueError('Wide replay source manifest hash changed')
        for key in ('current_geometry','source_contract','latest9a8c_reference18_preserved','user_trace_sha256','additional_real_native_ice_curves'):
            if old[key]!=manifest[key]:raise ValueError('Wide replay source identity differs: '+key)
        wr=z['residual'].copy();wc=csr_matrix((z['core_J_data'].copy(),z['core_J_indices'].copy(),z['core_J_indptr'].copy()),shape=tuple(z['core_J_shape']))
        wp=z['private_J'].copy()
    if wc.shape!=(5808,core_count)or wp.shape!=(5808,25):raise ValueError('Wide replay dimensions differ')
    actual=factor.evaluate(physical_core(q),q[camera_indices],False)[0]
    difference=float(abs(actual-wr).max())
    if difference>8e-11:raise ValueError('Actual current wide values differ from authenticated replay')
    key=factor._inputs(physical_core(q),q[camera_indices])[-1];factor._cache[key]=(wr,wc,wp)
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),same_complete_q_exact=True,
        source_contracts_exact=True,actual_current_value_max_absolute_difference=difference,
        original_source_value_support_UV_sigma_roles_unchanged=True,programme_normal_replayed=False)

def _branch(camera,world,observed):
    xyz=np.asarray(world,float);uv,depth,central=project_central_valid_full_brown(camera,xyz)
    a=uv[:-1];d=uv[1:]-a;l2=np.sum(d*d,axis=1)
    mid=project_central_valid_full_brown(camera,.5*(xyz[:-1]+xyz[1:]))[-1]
    valid=(central[:-1]&central[1:]&mid&np.isfinite(a).all(axis=1)&np.isfinite(uv[1:]).all(axis=1)
        &(depth[:-1]>0)&(depth[1:]>0)&(l2>1e-4))
    index,raw,clipped,nearest=nearest_segments(observed,a,d,l2,valid)
    category=np.where(raw<=0,-1,np.where(raw>=1,1,0))
    return dict(valid=valid,central=central,midpoint=mid,index=index,category=category,raw_t=raw,nearest=nearest)

def check_actual_wide_directional_branches(factor,q,camera_indices,direction,step,base,observed,predicted,*,core_getter=None):
    core_count=int(getattr(factor,'physical_parameter_count',CORE_COUNT))
    physical_core=(lambda value:np.asarray(value)[:core_count]) if core_getter is None else core_getter
    checked=observed.copy();bad=abs(observed-predicted)>(3e-3+1e-3*abs(predicted))
    report=dict(source_value_changed=False,source_rows_removed=0,original_masks_recomputed_at_each_candidate=True,
        derivative_role='Local piecewise derivative; proved optical-domain or finite nearest/clamp seams may use the same stable real branch on one side',seams=[])
    if not bad.any():report['checked_observed']=checked;return report
    states={};residuals={};p=q+step*direction;m=q-step*direction
    for label,value in [('base',q),('plus',p),('minus',m)]:
        world,camera,*_=factor._inputs(physical_core(value),value[camera_indices]);curves,top,ads=factor._curve_worlds(world)
        states[label]=(camera,curves,top)
        residuals[label]=base if label=='base'else factor.evaluate(physical_core(value),value[camera_indices],False)[0]
    families=[];cursor=0
    for i,(row,uv,sigma)in enumerate(zip(factor.observations,factor.uvs,factor.sigmas)):
        count=2*len(uv);families.append((row.get('id',str(row.get('target'))),cursor,cursor+count,i,uv));cursor+=count
    cursor=4342
    for i,(row,uv)in enumerate(zip(factor.trace_packet['contours'],factor.trace_uv)):
        count=2*len(uv);families.append((row['id'],cursor,cursor+count,None,uv));cursor+=count
    for name,start,stop,index,uv in families:
        if not bad[start:stop].any():continue
        branches={label:_branch(camera,curves[index]if index is not None else top,uv)
            for label,(camera,curves,top)in states.items()}
        for k in np.flatnonzero(bad[start:stop].reshape(-1,2).any(axis=1)):
            scalar=slice(start+2*k,start+2*k+2);b=branches['base'];segment=int(b['index'][k]);found=None
            for side,other in [('minus','plus'),('plus','minus')]:
                stable=branches[side];crossed=branches[other]
                same_branch=(int(stable['index'][k])==segment and stable['category'][k]==b['category'][k]
                    and stable['valid'][segment])
                optical_seam=(not crossed['valid'][segment] and np.any(b['central']!=crossed['central']))
                # Match the existing NativeIce qualification: every original
                # optical/finite validity mask is identical, exactly one real
                # side retains the nominal nearest segment and clamp category,
                # and the actual selected segments all remain valid.
                nearest_seam=(all(np.array_equal(branches[label][field],b[field])
                    for label in ('plus','minus') for field in ('valid','central','midpoint'))
                    and (int(crossed['index'][k])!=segment or crossed['category'][k]!=b['category'][k])
                    and all(branches[label]['valid'][segment]
                        and branches[label]['valid'][int(branches[label]['index'][k])]
                        for label in ('base','plus','minus')))
                if same_branch and (optical_seam or nearest_seam):
                    slope=((base-residuals[side])if side=='minus'else(residuals[side]-base))/step
                    error=float(abs(slope[scalar]-predicted[scalar]).max())
                    tolerance=float(np.max(3e-3+1e-3*abs(slope[scalar])))
                    if np.isfinite(slope[scalar]).all() and error<=tolerance:
                        found=(side,slope,error,tolerance,crossed,'finite_nearest_or_clamp'if nearest_seam else'optical_domain');break
            if found is None:continue
            side,slope,error,tolerance,crossed,seam_kind=found;checked[scalar]=slope[scalar]
            report['seams'].append(dict(family=name,source_UV_index=int(k),scalar_rows=[start+2*int(k),start+2*int(k)+1],
                seam_kind=seam_kind,base_segment=segment,stable_side=side,base_raw_t=float(b['raw_t'][k]),
                base_clamp_category=int(b['category'][k]),crossed_clamp_category=int(crossed['category'][k]),
                crossed_segment=int(crossed['index'][k]),central_knots_changed=np.flatnonzero(b['central']!=crossed['central']).tolist(),
                one_sided_max_absolute_error=error,one_sided_tolerance=tolerance,
                original_two_sided_max_absolute_error=float(abs(observed[scalar]-predicted[scalar]).max())))
    report['checked_observed']=checked;return report
