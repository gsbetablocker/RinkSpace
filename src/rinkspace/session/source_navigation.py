"""Current-native search proposals. Nothing here produces measured pixels."""
from __future__ import annotations
import hashlib
import numpy as np
from ..domain.scene_assets import scene_content_identity


def _sample_stations(geometry, cut):
    if cut['layer'] == 'advertisement':
        return np.asarray(geometry.samples['print_sampling_stations'][cut['instance_id']][cut['template_path_id']], float)
    row=geometry.current_paths[cut['id']]
    lo,hi=cut.get('sampling_interval', cut.get('support') or (0.,1.))
    count=max(257, 4*len(row.get('vertices',[]))) if row.get('vertices') else 257
    values=np.linspace(lo,hi,count)
    if row.get('vertices'):
        knots=np.linspace(0.,1.,len(row['vertices']))
        values=np.unique(np.r_[values,knots[(knots>=lo)&(knots<=hi)]])
    if len(values)!=cut['end']-cut['begin']:
        # Existing sampler's actual base count is the current scene contract.
        if not row.get('vertices'):values=np.linspace(lo,hi,cut['end']-cut['begin'])
        else:raise ValueError('Native representation station grid differs')
    return values


def project_navigation(native_geometry, camera_state, native_pts, uncertainty):
    if type(native_pts) is not int:raise ValueError('Navigation needs native integer PTS')
    camera=camera_state['camera'];controls=np.asarray(camera_state['controls'],float)
    radius=float(uncertainty['search_radius_px'])
    if not np.isfinite(radius) or radius<=0:raise ValueError('Search radius must be positive')
    scene_revision=scene_content_identity(native_geometry.scene,native_geometry.scene_path)
    if camera_state.get('scene_revision',scene_revision)!=scene_revision:raise ValueError('Navigation scene revision differs')
    points=native_geometry.samples['xyz']
    t=(native_pts-int(camera['reference_pts']))*camera['time_base_num']/camera['time_base_den']
    projected=native_geometry.core.project_motion(camera['rig'],controls,camera['first_control_time'],
        camera['control_dt'],camera['image_height'],t,points)
    revision=hashlib.sha256(controls.astype('<f8').tobytes()).hexdigest()
    proposals=[]
    signals={row['native_path_id']:{'blue':'blue_excess','red':'red_excess','dark':'dark_band'}[row['material_signal']]
        for row in native_geometry.ice_inventory()}
    for cut in native_geometry.samples['cuts']:
        begin,end=cut['begin'],cut['end'];uv=projected['uv'][begin:end]
        good=(projected['status'][begin:end]==0)&np.isfinite(uv).all(1)
        good&=(uv[:,0]>=-radius)&(uv[:,0]<1920+radius)&(uv[:,1]>=-radius)&(uv[:,1]<1080+radius)
        idx=np.flatnonzero(good)
        if len(idx)<2:continue
        # Drawing samples can be dense. Retain actual native sample coordinates
        # with roughly six image pixels between source profile queries.
        keep=[int(idx[0])]
        for j in idx[1:]:
            if j!=keep[-1]+1 and not good[j-1] or np.linalg.norm(uv[j]-uv[keep[-1]])>=6.:keep.append(int(j))
        if keep[-1]!=idx[-1]:keep.append(int(idx[-1]))
        idx=np.asarray(keep);q=uv[idx]
        tangent=uv[np.minimum(idx+1,len(uv)-1)]-uv[np.maximum(idx-1,0)]
        normal=np.column_stack((-tangent[:,1],tangent[:,0]));length=np.linalg.norm(normal,axis=1)
        valid=length>1e-10;idx,q,normal,length=idx[valid],q[valid],normal[valid],length[valid]
        if len(q)<2:continue
        normal/=length[:,None]
        gaps=np.r_[True,[not np.all(good[a:b+1]) for a,b in zip(idx[:-1],idx[1:])]]
        row=native_geometry.current_paths[cut['id']]
        proposal=dict(id=cut['id'],candidate_kind='native_projection',points_uv=q,
            normals=normal,segment_ids=np.cumsum(gaps)-1,stations=_sample_stations(native_geometry,cut)[idx],
            measurement=row.get('measurement','contour'),layer=cut['layer'],label=row['label'],
            origin='model_projection_search_hint',identity_confirmed=False,search_radius_px=radius,
            model_search_projection=dict(model_revision=native_geometry.refs[0]['sha256'],
                scene_content_revision=scene_revision,camera_revision=revision,native_pts=native_pts,
                readout_mode=camera['readout_mode']),source_evidence='mathematical_only',
            hint_is_source_measurement=False)
        if cut['id'] in signals:proposal['signal']=signals[cut['id']]
        proposals.append(proposal)
    return proposals


def initial_stations(proposals, path_id, actual_uv):
    """Initialize nuisance coordinates on an already resolved current path.

    This changes no measured UV and gives no identity authority to a projection.
    """
    proposal=next((p for p in proposals if p['id']==path_id),None)
    if proposal is None:raise ValueError('No valid current navigation support for '+path_id)
    q=np.asarray(proposal['points_uv']);s=np.asarray(proposal['stations']);seg=np.asarray(proposal['segment_ids'])
    a=q[:-1];d=q[1:]-a;norm=np.sum(d*d,axis=1)
    valid=(norm>1e-10)&(seg[:-1]==seg[1:])
    if not valid.any():raise ValueError('No continuous navigation segment for '+path_id)
    a,d,norm,left,right=a[valid],d[valid],norm[valid],s[:-1][valid],s[1:][valid]
    result=[]
    for uv in np.asarray(actual_uv):
        t=np.clip(np.sum((uv-a)*d,axis=1)/norm,0.,1.)
        best=int(np.argmin(np.sum((a+t[:,None]*d-uv)**2,axis=1)))
        result.append(float(left[best]+t[best]*(right[best]-left[best])))
    return np.asarray(result)
