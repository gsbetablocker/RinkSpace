"""Shared observing-camera framing for the player and headless checks.

Saved model coordinates, source clocks and motion are never modified.
"""
import numpy as np


def comparison_points(sequence,frame,*,include_surface=False):
    points=[sequence.state[key][frame].reshape(-1,3) for key in ('joints','gear')]
    if include_surface:
        points.append(sequence.vertices[frame].reshape(-1,3))
        if 'boot_surface' in sequence.state:points.append(sequence.state['boot_surface'][frame].reshape(-1,3))
    points=np.concatenate(points).astype(float,copy=False)
    if not len(points) or not np.isfinite(points).all():raise ValueError('Invalid comparison framing geometry')
    return points


def fit_comparison_camera(player,sequences,frame,padding=1.12,*,left_inset=0.):
    """Fit the observing camera only; retain its selected viewing direction."""
    camera=player.scene.camera
    joint_gear=np.concatenate([comparison_points(sequence,frame) for sequence in sequences.values()])
    lower,upper=joint_gear.min(axis=0),joint_gear.max(axis=0)
    center=(lower+upper)*.5
    points=np.concatenate([comparison_points(sequence,frame,include_surface=True) for sequence in sequences.values()])
    back=np.asarray(camera.position-camera.target,dtype=float)
    distance=float(np.linalg.norm(back))
    if not np.isfinite(distance) or distance<=0:raise ValueError('Invalid observing direction')
    back/=distance
    right=np.cross(camera.up,back);right_length=float(np.linalg.norm(right))
    if not np.isfinite(right_length) or right_length<=0:raise ValueError('Observing up and direction are parallel')
    right/=right_length;vertical=np.cross(back,right)
    width,height=player.window_size
    if not 0<=left_inset<width or height<=0:raise ValueError('Invalid comparison view region')
    aspect=float(width-left_inset)/height
    projected=(points-center)@np.stack([right,vertical,back],axis=1)
    half_height=max(float(np.max(abs(projected[:,1]))),float(np.max(abs(projected[:,0])))/aspect,.05)*padding
    distance=max(distance,float(projected[:,2].max())+camera.near+2*half_height)
    if distance-float(projected[:,2].min())>=camera.far:raise ValueError('Comparison bounds exceed observing camera depth range')
    target=center-right*(left_inset/height*half_height)
    camera.target=target;camera.position=target+back*distance
    camera.is_ortho=True;camera.ortho_size=half_height
    camera.update_matrices(width,height)
    return dict(method='joint/gear union AABB center; full body and dynamic boot surface included in view extents',
                camera_only=True,world_joint_gear_min=lower.tolist(),world_joint_gear_max=upper.tolist(),
                common_center=center.tolist(),view_direction=(-back).tolist(),padding=padding,
                viewport_size=[int(width),int(height)],left_inset_pixels=left_inset,
                observing_target=target.tolist(),ortho_half_height=half_height)


def comparison_frustum_bounds(player,sequence,frame,*,left_inset=0.):
    points=comparison_points(sequence,frame,include_surface=True)
    homogeneous=np.column_stack([points,np.ones(len(points))])
    clip=homogeneous@player.scene.camera.get_view_projection_matrix().T
    ndc=clip[:,:3]/clip[:,3,None]
    inside=bool(np.isfinite(ndc).all() and np.all(abs(ndc)<=1.))
    clear_overlay=bool(inside and np.all(ndc[:,0]>=2*left_inset/player.window_size[0]-1))
    return dict(ndc_min=ndc.min(axis=0).tolist(),ndc_max=ndc.max(axis=0).tolist(),
                all_body_and_gear_inside=inside,all_body_and_gear_in_view_region=clear_overlay,points=len(points))
