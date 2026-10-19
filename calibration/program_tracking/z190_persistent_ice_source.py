"""Append explicit source-qualified static ice tracks to an intact bundle.

This family is separate from the original motion cache. Each track uses one
observed anchor ray intersected with z=0, shared by all of its observed target
exposures. It has no per-pair free depth. Pixels, sigmas and source authority
come from the frozen ledger, never from camera projections or fit residuals.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
from z190_continuous_source_bundle import SOURCE_SHA
from z190_continuous_source_bundle import SCENE_UNIT_M,radial_inverse
from z190_main_zoom_k1_source_information import read,ref
from z190_moving_pivot_source_bundle import MovingPivotSourceBundle


def _motion_components(bundle):
    parent=np.arange(len(bundle.stamps))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return int(i)
    for a,b in zip(bundle.ia,bundle.ib):
        a,b=find(int(a)),find(int(b))
        if a!=b:parent[b]=a
    return np.array([find(i)for i in range(len(parent))],int)


def _uv(value):
    uv=np.asarray(value,np.float64)
    if uv.shape!=(2,)or not np.isfinite(uv).all()or not(0<=uv[0]<1920 and 0<=uv[1]<1080):
        raise ValueError('Persistent source needs a measured native pixel inside the image')
    return uv


def _check_image(point,cache):
    path=Path(point['raw_png']);expected=point['native_bgr_sha256']
    key=(str(path),expected)
    if key not in cache:
        if point.get('raw_png_sha256')and hashlib.sha256(path.read_bytes()).hexdigest()!=point['raw_png_sha256']:
            raise ValueError('Persistent native PNG source authority changed')
        pixels=cv2.imread(str(path))
        if pixels is None or pixels.shape!=(1080,1920,3)or hashlib.sha256(pixels.tobytes()).hexdigest()!=expected:
            raise ValueError('Persistent source raw native image changed')
        cache.add(key)


def append_persistent_ice_source(bundle,path,aliases,*,held_native_pts=(230011136,230011392)):
    """Return (new_bundle, admission_manifest) without mutating the original.

    The packet's explicit new_rows are a distinct source family. Original
    rows keep their exact order and indices; all priors and original profile
    weights survive. Noncanonical native observations remain in the ledger
    but do not receive duplicate weight for the same physical exposure.
    """
    packet=read(path);authority=ref(Path(path));old_count=len(bundle.a)
    if packet.get('schema')!='rinkspace.persistent-ice-z0-source.v1':
        raise ValueError('Persistent ice packet schema is not explicitly supported')
    if packet.get('source_sha256')!=SOURCE_SHA or packet.get('frozen_before_fit')is not True:
        raise ValueError('Frozen verified source authority is required')
    if packet.get('model_projection_used_to_generate_or_select_uv')is not False:
        raise ValueError('Persistent source pixels must be selected without model projection')
    tracks={r['track_id']:r for r in packet['tracks']}
    if len(tracks)!=len(packet['tracks']):raise ValueError('Persistent material track identities duplicated')
    held={int(aliases[int(p)])for p in held_native_pts};components=_motion_components(bundle)
    image_cache=set();candidates={};excluded=[]
    for row in packet['new_rows']:
        if row.get('split')!='fit':
            excluded.append(dict(row=row,reason='source role is not fit'));continue
        track_id=row['track_id'];track=tracks[track_id]
        if row.get('original_index')is not None:raise ValueError('New persistent family must not claim an original motion index')
        if row.get('surface_id')!='ice_z0' or row.get('same_static_material_confirmed')is not True or row.get('both_end_manually_qualified')is not True:
            raise ValueError('Actual both-end static ice material qualification is required')
        anchor=track['anchor'];target=row['target']
        if row['anchor']!=anchor or not np.array_equal(row['xy0'],anchor['uv'])or not np.array_equal(row['xy1'],target['uv']):
            raise ValueError('Persistent source UV differs from frozen shared anchor/target')
        if row.get('source_visible')is not True or row.get('target_visible')is not True:
            raise ValueError('A persistent fit factor cannot use a hidden source endpoint')
        _check_image(anchor,image_cache);_check_image(target,image_cache)
        na,nb=int(anchor['source_native_pts']),int(target['source_native_pts'])
        pa,pb=int(aliases[na]),int(aliases[nb])
        if pa not in bundle.index or pb not in bundle.index:raise ValueError('Persistent source outside complete installation scope')
        if pa in held or pb in held:raise ValueError('Exact user exposure aliases must remain new-source held')
        if components[bundle.index[pa]]!=components[bundle.index[pb]]:raise ValueError('Persistent source crosses an original camera cut')
        a,b=_uv(anchor['uv']),_uv(target['uv'])
        sigma=np.broadcast_to(np.asarray(row['sigma_native_px'],float),(2,)).copy()
        if not np.isfinite(sigma).all()or np.any(sigma<=0):raise ValueError('Explicit positive engineering source sigma is required')
        if not row.get('sigma_semantics')or row.get('model_projection_used_to_generate_or_select_uv')is not False or row.get('frozen_before_fit')is not True:
            raise ValueError('Persistent sigma must have a declared engineering interpretation')
        if pa==pb:
            excluded.append(dict(row=row,reason='same actual exposure as anchor'));continue
        item=dict(row=row,track_id=track_id,a=a,b=b,ia=bundle.index[pa],ib=bundle.index[pb],pa=pa,pb=pb,na=na,nb=nb,sigma=sigma)
        key=(track_id,pb)
        if key in candidates:
            previous=candidates[key]
            # The confirmed physical representative wins, without averaging.
            if nb==pb and previous['nb']!=pb:
                excluded.append(dict(row=previous['row'],reason='duplicate native alias; physical representative source read retained'));candidates[key]=item
            elif previous['nb']==pb or nb!=pb:
                excluded.append(dict(row=row,reason='duplicate native alias; one physical exposure retained'))
            else:raise ValueError('Duplicated canonical source observation')
        else:candidates[key]=item
    if not candidates:raise ValueError('Persistent packet has no qualified new physical source factors')
    added=list(candidates.values());added.sort(key=lambda r:(r['track_id'],r['pb']))
    a=np.array([r['a']for r in added]);b=np.array([r['b']for r in added]);ia=np.array([r['ia']for r in added],int);ib=np.array([r['ib']for r in added],int)
    sigma=np.array([r['sigma']for r in added]).ravel();absolute=bundle.native_residual_sigma[bundle.motion_rows:]
    base=SimpleNamespace(stamps=bundle.stamps,index=bundle.index,a=np.r_[bundle.a,a],b=np.r_[bundle.b,b],ia=np.r_[bundle.ia,ia],ib=np.r_[bundle.ib,ib],motion_sigma=np.r_[bundle.motion_sigma,sigma],abs_groups=bundle.abs_groups,native_residual_sigma=np.r_[bundle.motion_sigma,sigma,absolute])
    planes=np.r_[bundle.finite_planes,np.tile([0.,0.,1.,0.],(len(added),1))]
    records=list(bundle.finite_records)
    ledger=[]
    for i,r in enumerate(added):
        record=dict(global_bundle_index=old_count+i,source_family='persistent_ice_z0_source',surface_kind='ice_plane',plane=[0,0,1,0],source_qualification=authority,original_index=None,track_id=r['track_id'],source_native_pts=r['na'],target_native_pts=r['nb'],source_actual_pts=r['pa'],target_actual_pts=r['pb'],source_uv=r['a'].tolist(),target_uv=r['b'].tolist(),sigma_engineering_px=r['sigma'].tolist(),record=r['row'])
        records.append(record);ledger.append(record)
    new=MovingPivotSourceBundle(base,bundle.calibration,bundle.motion_kernel,reference_pts=bundle.reference_pts,initial_reference_optical_height_scene=bundle.initial_reference_optical_height_scene,finite_planes=planes,finite_records=records,scene=bundle.scene,absolute_evaluator=bundle.absolute_evaluator)
    new.robust_phase=bundle.robust_phase
    new.profile_weights=None if bundle.profile_weights is None else np.r_[bundle.profile_weights,np.ones((len(added),2))]
    for name in('a','b','ia','ib','finite_planes'):
        if not np.array_equal(getattr(new,name)[:old_count],getattr(bundle,name)):raise ValueError('Persistent admission changed original source prefix:'+name)
    if not np.array_equal(new.motion_sigma[:bundle.motion_rows],bundle.motion_sigma):raise ValueError('Original native source sigma changed')
    if new.abs_groups is not bundle.abs_groups or not np.array_equal(new.stamps,bundle.stamps):raise ValueError('Persistent admission changed absolute source/exposure inventory')
    manifest=dict(packet=authority,original_motion_rows_exact_prefix=old_count,added_persistent_finite_rows=len(added),added_residual_rows=2*len(added),track_ids=sorted(set(r['track_id']for r in added)),original_UV_endpoint_order_sigma_finite_planes_exact_prefix=True,original_profile_weights_exact_prefix=True,absolute_source_groups_world_UV_roles_sigma_unchanged=True,actual_exposure_inventory_unchanged=True,all_added_ice_z0_shared_anchor_per_track=True,no_per_pair_unknown_depth_added=True,excluded_exact_user_actual_pts=sorted(held),excluded_native_alias_rows=excluded,added_source_ledger=ledger,source_derived_qualification_not_camera_classification=True,accuracy_accepted=False)
    return new,manifest


def persistent_ice_native_values(bundle,q,rows,aliases,*,jacobian=False):
    """Replay only explicitly measured source rows in native pixels.

    Used for actual-source derivative fixtures and held reporting. This never
    admits a row or mutates the bundle/profile; the caller selects fit/held.
    """
    rows=list(rows);a=np.asarray([r['anchor']['uv']for r in rows],float);b=np.asarray([r['target']['uv']for r in rows],float)
    ia=np.asarray([bundle.index[int(aliases[int(r['anchor']['source_native_pts'])])]for r in rows],int)
    ib=np.asarray([bundle.index[int(aliases[int(r['target']['source_native_pts'])])]for r in rows],int)
    cache=bundle.cache(q,jacobian);vv,rr,dr,cc,ds,dv=cache
    f=np.exp(vv[:,3]);pp=np.array([1920*bundle.calibration.principal_x_fraction[0],1080*bundle.calibration.principal_y_fraction[0]])
    result=bundle.motion_kernel(fa=f[ia],fb=f[ib],ra=rr[ia],rb=rr[ib],da=dr[ia]if jacobian else None,db=dr[ib]if jacobian else None,ca=cc[ia],cb=cc[ib],dca_shared=ds[ia]if jacobian else None,dcb_shared=ds[ib]if jacobian else None,dca_view=dv[ia]if jacobian else None,dcb_view=dv[ib]if jacobian else None,a=a,b=b,pp=pp,k1=bundle.calibration.k1[0],planes=np.tile([0.,0.,1.,0.],(len(rows),1)),jacobian=jacobian,sigma=np.asarray([r['sigma_native_px']for r in rows]),profile_weight=np.ones((len(rows),2)))
    if not jacobian:return result[0]if isinstance(result,tuple)else result
    values,ja,jb,shared=result[:4];jac=np.zeros((len(values),len(q)))
    jac[:,:4]=shared
    for i,(left,right)in enumerate(zip(ia,ib)):
        jac[2*i:2*i+2,4+4*left:8+4*left]+=ja[2*i:2*i+2]
        jac[2*i:2*i+2,4+4*right:8+4*right]+=jb[2*i:2*i+2]
    return values,jac


def persistent_ice_source_report(bundle,q,path,aliases):
    """Native fit and held target replay, keeping shared-anchor correlation explicit."""
    packet=read(path);rows=packet['new_rows'];values=persistent_ice_native_values(bundle,q,rows,aliases).reshape(-1,2)
    return [dict(id=r['id'],track_id=r['track_id'],split=r['split'],source_native_pts=r['source_native_pts'],target_native_pts=r['target_native_pts'],native_error_xy_px=e.tolist(),native_error_norm_px=float(np.linalg.norm(e)),sigma_engineering_px=r['sigma_native_px'],repeated_anchor_uv_correlation=True,targets_not_independent_anchor_measurements=True)for r,e in zip(rows,values)]


def persistent_ice_world_report(bundle,q,path,aliases):
    """Report the common anchor and observed target-ray scatter separately.

    Source UV is measured; world XYZ and metric displacement are conditional
    on this camera candidate. Sharing one anchor does not make the actual
    target backprojections accurate, which this report keeps visible.
    """
    packet=read(path);vv,rr,_,cc,_,_=bundle.cache(q,False)
    pp=np.array([1920*bundle.calibration.principal_x_fraction[0],1080*bundle.calibration.principal_y_fraction[0]])
    polygon=np.asarray(bundle.scene['ice_boundary']['points_xyz'],float)[:,:2].astype(np.float32)
    def world(point):
        p=int(aliases[int(point['source_native_pts'])]);i=bundle.index[p]
        xy,_=radial_inverse((np.asarray(point['uv'],float)[None]-pp)/np.exp(vv[i,3]),bundle.calibration.k1[0])
        ray=rr[i].T@np.r_[xy[0],1.]
        if abs(ray[2])<1e-10:raise ValueError('Persistent material ray tangent to ice')
        depth=-cc[i,2]/ray[2]
        if depth<=0:raise ValueError('Persistent source ray has nonpositive physical ice depth')
        xyz=cc[i]+depth*ray
        clearance=cv2.pointPolygonTest(polygon,tuple(xyz[:2]),True)*SCENE_UNIT_M
        return xyz,clearance
    result=[]
    for track in packet['tracks']:
        rows=[r for r in packet['new_rows']if r['track_id']==track['track_id']]
        if not rows:continue
        anchor,clearance=world(track['anchor']);targets=[]
        for row in rows:
            observed,target_clearance=world(row['target']);delta=(observed-anchor)*SCENE_UNIT_M
            targets.append(dict(target_native_pts=row['target_native_pts'],split=row['split'],observed_target_ray_ice_xyz_m=(observed*SCENE_UNIT_M).tolist(),conditional_world_displacement_from_common_anchor_m=delta.tolist(),conditional_world_displacement_norm_m=float(np.linalg.norm(delta)),ice_boundary_signed_clearance_m=float(target_clearance),inside_actual_ice_footprint=target_clearance>=0))
        result.append(dict(track_id=track['track_id'],anchor_native_pts=track['anchor']['source_native_pts'],common_anchor_ice_xyz_scene=anchor.tolist(),common_anchor_ice_xyz_m=(anchor*SCENE_UNIT_M).tolist(),ice_boundary_signed_clearance_m=float(clearance),inside_actual_ice_footprint=clearance>=0,all_targets_share_this_anchor_in_factor=True,no_independent_pair_depth=True,observed_target_backprojections_not_replaced_by_anchor=targets,maximum_conditional_fit_displacement_m=max(r['conditional_world_displacement_norm_m']for r in targets if r['split']=='fit'),maximum_conditional_held_displacement_m=max(r['conditional_world_displacement_norm_m']for r in targets if r['split']=='held'),world_values_are_camera_conditional_not_source_measurements=True))
    return result
