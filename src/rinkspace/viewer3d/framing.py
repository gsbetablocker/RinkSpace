"""Shared observing-camera framing for the player and headless checks.

Saved model coordinates, source clocks and motion are never modified.
"""
import json
import numpy as np



def comparison_panel_sizes(width,source_visible,control_width=425.):
    """Reserve a useful 3D viewport when both source and controls are present."""
    width=float(width)
    if not np.isfinite(width) or width<=0:raise ValueError('Invalid window width')
    left=float(control_width)+28.;right=650. if source_visible else 0.
    available=max(1.,width-min(320.,.4*width))
    scale=min(1.,available/(left+right))
    return dict(control_width=max(1.,left*scale-28.),
        source_width=max(1.,right*scale-20.) if source_visible else 0.,
        left_inset=left*scale,right_inset=right*scale)


def comparison_points(sequence,frame,*,include_surface=False,support_foot=None,grasp_hand=None,free_foot=0):
    if grasp_hand is not None:
        if isinstance(grasp_hand,bool) or grasp_hand not in ('left','right',0,1):raise ValueError('Invalid grasp hand')
        if isinstance(free_foot,bool) or free_foot not in (0,1):raise ValueError('Invalid grasp free foot')
        side='left' if grasp_hand in ('left',0) else 'right';wrist=20 if side=='left' else 21
        state=sequence.state;native=np.asarray(state['joints'][frame])
        if native.shape!=(22,3):raise ValueError('Grasp framing requires native22 joints')
        # A wrist is only a framing landmark, never a substitute palm point.
        points=[native[wrist:wrist+1],state['gear'][frame,free_foot].reshape(-1,3)]
        geometry=json.loads(str(state['grasp_geometry_json'])) if 'grasp_geometry_json' in state else {}
        if geometry.get('grip_hand',side)==side and 'grasp_palm_world_mid' in state:
            points.append(state['grasp_palm_world_mid'][frame].reshape(-1,3))
        if geometry.get('grip_foot',free_foot)==free_foot and 'grasp_segment_world_mid' in state:
            points.append(state['grasp_segment_world_mid'][frame].reshape(-1,3))
        if include_surface:
            for key in ('boot_surface','skate_lower','skate_upper','skate_edge'):
                if key in state:points.append(state[key][frame,free_foot].reshape(-1,3))
            patch=geometry.get('palm_patches',{}).get(side,{})
            if 'vertex_ids' in patch:
                ids=np.asarray(patch['vertex_ids'])
                if ids.ndim!=1 or ids.dtype.kind not in 'iu' or np.any(ids<0) or np.any(ids>=sequence.vertices.shape[1]):
                    raise ValueError('Invalid grasp palm patch vertex IDs')
                points.append(sequence.vertices[frame,ids].reshape(-1,3))
        values=np.concatenate(points).astype(float,copy=False)
        if not len(values) or not np.isfinite(values).all():raise ValueError('Invalid comparison framing geometry')
        return values
    if support_foot is not None:
        if support_foot not in (0,1):raise ValueError('Invalid comparison support foot')
        points=[sequence.state['gear'][frame,support_foot].reshape(-1,3)]
        if include_surface and 'boot_surface' in sequence.state:points.append(sequence.state['boot_surface'][frame,support_foot].reshape(-1,3))
        values=np.concatenate(points).astype(float,copy=False)
        if not len(values) or not np.isfinite(values).all():raise ValueError('Invalid comparison framing geometry')
        return values
    points=[sequence.state[key][frame].reshape(-1,3) for key in ('joints','gear')]
    if include_surface:
        points.append(sequence.vertices[frame].reshape(-1,3))
        if 'boot_surface' in sequence.state:points.append(sequence.state['boot_surface'][frame].reshape(-1,3))
    points=np.concatenate(points).astype(float,copy=False)
    if not len(points) or not np.isfinite(points).all():raise ValueError('Invalid comparison framing geometry')
    return points


def fit_comparison_camera(player,sequences,frame,padding=1.12,*,left_inset=0.,right_inset=0.,support_foot=None,grasp_hand=None,free_foot=0):
    """Fit the observing camera only; retain its selected viewing direction."""
    camera=player.scene.camera
    scope=dict(support_foot=support_foot,grasp_hand=grasp_hand,free_foot=free_foot)
    joint_gear=np.concatenate([comparison_points(sequence,frame,**scope) for sequence in sequences.values()])
    lower,upper=joint_gear.min(axis=0),joint_gear.max(axis=0)
    center=(lower+upper)*.5
    points=np.concatenate([comparison_points(sequence,frame,include_surface=True,**scope) for sequence in sequences.values()])
    back=np.asarray(camera.position-camera.target,dtype=float)
    distance=float(np.linalg.norm(back))
    if not np.isfinite(distance) or distance<=0:raise ValueError('Invalid observing direction')
    back/=distance
    right=np.cross(camera.up,back);right_length=float(np.linalg.norm(right))
    if not np.isfinite(right_length) or right_length<=0:raise ValueError('Observing up and direction are parallel')
    right/=right_length;vertical=np.cross(back,right)
    width,height=player.window_size
    if left_inset<0 or right_inset<0 or left_inset+right_inset>=width or height<=0:raise ValueError('Invalid comparison view region')
    aspect=float(width-left_inset-right_inset)/height
    projected=(points-center)@np.stack([right,vertical,back],axis=1)
    minimum_half_height=.22 if grasp_hand is not None else .05
    half_height=max(float(np.max(abs(projected[:,1]))),float(np.max(abs(projected[:,0])))/aspect,minimum_half_height)*padding
    distance=max(distance,float(projected[:,2].max())+camera.near+2*half_height)
    if distance-float(projected[:,2].min())>=camera.far:raise ValueError('Comparison bounds exceed observing camera depth range')
    target=center-right*((left_inset-right_inset)/height*half_height)
    camera.target=target;camera.position=target+back*distance
    camera.is_ortho=True;camera.ortho_size=half_height
    camera.update_matrices(width,height)
    method='grasp-side wrist, saved true palm and free-shoe union; only observing camera changes' if grasp_hand is not None else 'joint/gear union AABB center; full body and dynamic boot surface included in view extents'
    return dict(method=method,
                camera_only=True,support_foot=support_foot,grasp_hand=grasp_hand,free_foot=free_foot if grasp_hand is not None else None,world_joint_gear_min=lower.tolist(),world_joint_gear_max=upper.tolist(),
                common_center=center.tolist(),view_direction=(-back).tolist(),padding=padding,
                viewport_size=[int(width),int(height)],left_inset_pixels=left_inset,right_inset_pixels=right_inset,
                observing_target=target.tolist(),ortho_half_height=half_height)


def comparison_frustum_bounds(player,sequence,frame,*,left_inset=0.,right_inset=0.,support_foot=None,grasp_hand=None,free_foot=0):
    points=comparison_points(sequence,frame,include_surface=True,support_foot=support_foot,grasp_hand=grasp_hand,free_foot=free_foot)
    homogeneous=np.column_stack([points,np.ones(len(points))])
    clip=homogeneous@player.scene.camera.get_view_projection_matrix().T
    ndc=clip[:,:3]/clip[:,3,None]
    inside=bool(np.isfinite(ndc).all() and np.all(abs(ndc)<=1.))
    clear_overlay=bool(inside and np.all(ndc[:,0]>=2*left_inset/player.window_size[0]-1)
                       and np.all(ndc[:,0]<=1-2*right_inset/player.window_size[0]))
    return dict(ndc_min=ndc.min(axis=0).tolist(),ndc_max=ndc.max(axis=0).tolist(),
                all_body_and_gear_inside=inside,all_body_and_gear_in_view_region=clear_overlay,points=len(points))
