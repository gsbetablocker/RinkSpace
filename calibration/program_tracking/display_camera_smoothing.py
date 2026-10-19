"""Cosmetic, pixel-bounded camera smoothing for programme display only.

The saved physical camera, world and trajectory are never changed. One
display camera projects every layer; repeated native aliases share it exactly.
"""
from __future__ import annotations

import hashlib
import cv2
import numpy as np

try:
    from .projection import project_fast
except ImportError:
    from projection import project_fast


def _sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def display_model_probes(model_paths, advertisement_paths):
    """Endpoints, midpoint and arc quarters of every actual displayed path.

Only literal duplicate probe points are removed. Independent printed markers
remain their original material points; no connecting outline is invented.
"""
    points=[]
    def path_points(xyz):
        xyz=np.asarray(xyz,float)
        if len(xyz)<2:return xyz
        distance=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xyz,axis=0),axis=1))]
        if distance[-1]==0:return xyz[:1]
        keep=np.r_[True,np.diff(distance)>0]
        samples=distance[-1]*np.array([0.,.25,.5,.75,1.])
        return np.column_stack([np.interp(samples,distance[keep],xyz[keep,k])for k in range(3)])
    points.extend(path_points(xyz)for _,xyz in model_paths)
    points.extend(np.asarray(p['xyz'],float)if p.get('primitive')=='points'else path_points(p['xyz'])
                  for p in advertisement_paths)
    return np.unique(np.concatenate(points),axis=0)


def smooth_display_cameras(pts, camera25, representatives, cut_ranges, world_probes,
                           *, max_shift_px=1., image_size_wh=(1920, 1080),
                           native_step_pts=256):
    """Return a separate display array and stats, using centered actual 3-taps.

    Fixed representative probes cover every supplied model path. Their viewport
visibility bounds the common camera change, never the displayed inventory.
Cut endpoints, missing native neighbours and invalid cameras pass through.
"""
    pts=np.asarray(pts,np.int64);reps=np.asarray(representatives,np.int64)
    raw=np.ascontiguousarray(camera25);probes=np.asarray(world_probes,float)
    cut_ranges=tuple(cut_ranges)
    if (pts.ndim!=1 or reps.shape!=pts.shape or raw.shape!=(len(pts),25)
            or np.any(np.diff(pts)<=0) or np.any(np.diff(reps)<0)
            or not np.issubdtype(raw.dtype,np.floating)
            or probes.ndim!=2 or probes.shape[1:]!=(3,) or not len(probes)
            or not np.isfinite(probes).all() or not np.isfinite(max_shift_px)
            or max_shift_px<=0 or native_step_pts<=0):
        raise ValueError('Ordered cameras, finite complete model probes and positive pixel cap required')
    cuts=np.full(len(pts),-1,np.int64)
    for k,(start,end) in enumerate(cut_ranges):
        if end<=start:raise ValueError('Source cuts need an exclusive end after their start')
        inside=(pts>=start)&(pts<end)
        if np.any(cuts[inside]>=0):raise ValueError('Source cuts overlap')
        cuts[inside]=k
    actual,first,inverse=np.unique(reps,return_index=True,return_inverse=True)
    canonical=raw[first]
    if not np.array_equal(raw,canonical[inverse],equal_nan=True):
        raise ValueError('Native aliases must already share the same original camera')
    groups=[np.arange(a,b)for a,b in zip(first,np.r_[first[1:],len(pts)])]
    group_cut=np.array([cuts[g[0]]if np.all(cuts[g]==cuts[g[0]])else -1 for g in groups])
    result=canonical.copy();width,height=image_size_wh
    counts=dict(filtered_actual_exposures=0,clipped_blends=0,skipped_cut_endpoints=0,
                skipped_missing_neighbours=0,skipped_invalid_cameras=0,skipped_no_visible_probes=0)
    max_change=0.
    for cut in range(len(cut_ranges)):
        ids=np.flatnonzero(group_cut==cut)
        counts['skipped_cut_endpoints']+=min(2,len(ids))
        for pos in range(1,len(ids)-1):
            before,i,after=ids[pos-1:pos+2]
            if (pts[groups[i][0]]-pts[groups[before][-1]]!=native_step_pts
                    or pts[groups[after][0]]-pts[groups[i][-1]]!=native_step_pts):
                counts['skipped_missing_neighbours']+=1;continue
            previous,center,following=canonical[[before,i,after]]
            if not np.isfinite([previous,center,following]).all():
                counts['skipped_invalid_cameras']+=1;continue
            rotation=cv2.Rodrigues(center[3:6])[0]
            tangent=.25*(cv2.Rodrigues(cv2.Rodrigues(previous[3:6])[0]@rotation.T)[0].ravel()
                         +cv2.Rodrigues(cv2.Rodrigues(following[3:6])[0]@rotation.T)[0].ravel())
            delta=.25*(previous+following-2*center)
            if not np.any(np.r_[delta[:3],tangent,delta[6:8]]):continue
            # Pixel bounds need only the same full Brown forward projection.
            # The renderer keeps its original optical-domain/occlusion checks.
            uv0,depth0=project_fast(center,probes)
            valid0=np.isfinite(uv0).all(axis=1)&(depth0>.05)
            visible0=valid0&(uv0[:,0]>=0)&(uv0[:,0]<width)&(uv0[:,1]>=0)&(uv0[:,1]<height)
            if not visible0.any():
                counts['skipped_no_visible_probes']+=1;continue
            blend=1.;accepted=None
            while blend>np.finfo(float).eps:
                candidate=center.copy()
                candidate[:3]+=blend*delta[:3]
                candidate[3:6]=cv2.Rodrigues(cv2.Rodrigues(blend*tangent)[0]@rotation)[0].ravel()
                candidate[6:8]+=blend*delta[6:8]
                uv,depth=project_fast(candidate,probes)
                valid=np.isfinite(uv).all(axis=1)&(depth>.05)
                visible=valid&(uv[:,0]>=0)&(uv[:,0]<width)&(uv[:,1]>=0)&(uv[:,1]<height)
                if np.any(visible0&~valid) or np.any(visible&~valid0):shift=np.inf
                else:
                    selected=(visible0|visible)&valid0&valid
                    shift=float(np.max(np.linalg.norm(uv[selected]-uv0[selected],axis=1)))
                if shift<=max_shift_px:
                    accepted=candidate;break
                blend*=min(.5,max_shift_px/shift)if np.isfinite(shift)else .5
            if accepted is not None:
                result[i]=accepted
                counts['filtered_actual_exposures']+=1
                counts['clipped_blends']+=int(blend<1.)
                max_change=max(max_change,shift)
    display=result[inverse]
    stats=dict(schema='rinkspace.cosmetic-display-camera-smoothing.v1',enabled=True,
               method='distinct_actual_centered_3tap_1_4_1_2_1_4_SO3_C_logf',
               max_shift_native_px=float(max_shift_px),maximum_measured_probe_shift_native_px=max_change,
               display_shift_is_probe_estimate_not_every_raster_vertex_bound=True,
               world_probe_count=len(probes),native_presentations=len(pts),distinct_actual_exposures=len(actual),
               cut_count=len(cut_ranges),native_step_pts=int(native_step_pts),**counts,
               skipped_unassigned_or_cross_cut_actual_exposures=int(np.count_nonzero(group_cut<0)),
               aliases_share_one_display_camera_exactly=True,cut_endpoints_and_gaps_unmodified=True,
               principal_skew_and_Brown_coefficients_unmodified=True,
               pixel_cap_uses_complete_Brown_forward_projection=True,
               renderer_original_optical_domain_and_occlusion_checks_unmodified=True,
               raw_camera_sha256=_sha(raw),display_camera_sha256=_sha(display),
               raw_camera_world_and_trajectory_artifacts_unmodified=True,
               display_camera_is_cosmetic_not_new_solver_state=True,accuracy_accepted=False)
    return display,stats
