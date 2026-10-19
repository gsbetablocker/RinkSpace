"""Rebind a fixed source circle identity to a versioned scene, never its UV."""
from __future__ import annotations
import hashlib,re
import numpy as np


def rebind_fr_circle(row,original,candidate):
    name=row['id'].split('_')[0]
    if not re.fullmatch(r'[FR][1-4]',name):return row
    previous=next(x for x in original['ice_features']if x['id']==name)
    updated=next(x for x in candidate['ice_features']if x['id']==name)
    role=row['id'].split('_')[1]if '_'in row['id']else 'outer'
    if role not in ('outer','middle','inner','center'):role='outer'
    oldring=next((r for r in previous['rings']if r['role']==role),previous)
    newring=next((r for r in updated['rings']if r['role']==role),updated)
    before=np.asarray(row['world'],float);angle=np.arctan2(before[:,1]-oldring['center_xyz'][1],before[:,0]-oldring['center_xyz'][0])
    result=dict(row);result['world']=np.c_[newring['center_xyz'][0]+newring['radius']*np.cos(angle),
        newring['center_xyz'][1]+newring['radius']*np.sin(angle),np.zeros(len(before))]
    result['source_pixel_sha256_preserved']=hashlib.sha256(np.asarray(row['uv'],np.float64).tobytes()).hexdigest()
    result['original_world_sha256']=hashlib.sha256(before.tobytes()).hexdigest()
    result['world_scene_binding']=dict(feature=name,ring_role=role,source_uv_changed=False,
        physical_identity_status='retained source identity, not reclassified by projection',
        versioned_geometry='shared user hard FR relations and common F column candidate')
    return result


def _closed_path(points):
    xyz=np.asarray(points,float)
    return xyz.copy() if np.array_equal(xyz[0],xyz[-1]) else np.vstack([xyz,xyz[:1]])


def rebind_scene_geometry(row,original,candidate):
    """Bind retained physical source IDs to one full versioned scene.

    Source positions, normal blur intervals, weights and fit/holdout splits do
    not change. Unknown world identities raise instead of silently retaining
    a stale curve from the preceding field revision.
    """
    result=dict(row);name=str(row['id']);before=np.asarray(row['world'],float)
    previous={f['id']:f for f in original['ice_features']}
    updated={f['id']:f for f in candidate['ice_features']}
    identity=None
    if row.get('point_correspondence'):
        physical=row.get('physical_ids')
        if not physical:
            try:
                from .prepare_global_geometry_source_keys import ad_roles_from_original_pixels
            except ImportError:
                from prepare_global_geometry_source_keys import ad_roles_from_original_pixels
            physical=ad_roles_from_original_pixels(row,[])
        if not physical and re.fullmatch(r'ad\d+:(left|right)_(top|bottom)',name):physical=[name]
        if not physical or len(physical)!=len(row['uv']):
            raise ValueError(f"No retained measured point role for source {row['pts']}/{name}")
        try:
            from .ad_landmarks import AdGeometry
        except ImportError:
            from ad_landmarks import AdGeometry
        xyz=[]
        for physical_id in physical:
            match=re.fullmatch(r'ad(\d+):((left|right)_(top|bottom))',physical_id)
            if not match:raise ValueError(f'Unsupported source point identity {physical_id}')
            geometry=AdGeometry(candidate,int(match[1]))
            xyz.append(geometry.corner(geometry.initial,match[2]))
        result['world']=np.asarray(xyz,float);result['physical_ids']=list(physical)
        identity=dict(kind='actual_advertisement_corners',physical_ids=list(physical))
    elif name in ('far_top','board_top','board_top_near_visible'):
        result['world']=_closed_path(candidate['board_surface']['top_edge_xyz'])
        identity=dict(kind='finite_full_board_top_perimeter')
    elif name in ('far_contact','ice_board_contact'):
        result['world']=_closed_path(candidate['ice_boundary']['points_xyz'])
        identity=dict(kind='finite_full_ice_board_contact_perimeter')
    elif name in ('reviewed_lower_board_top','lower_long_side_top'):
        try:
            from .secondary_nearboard_top_source import lower_long_side_top
        except ImportError:
            from secondary_nearboard_top_source import lower_long_side_top
        result['world'],result['geometry']=lower_long_side_top(candidate)
        identity=dict(kind='user_identified_finite_lower_long_side_ice_facing_top_edge')
    else:
        circle=re.fullmatch(r'(.+)_(outer|middle|inner|center)',name)
        lane=re.fullmatch(r'(lane[1-4])_(?:edge)?([0-3])',name)
        if circle and circle[1]in updated and updated[circle[1]]['kind']=='circle':
            feature,role=circle[1],circle[2]
            oldfeature,newfeature=previous[feature],updated[feature]
            oldring=next((r for r in oldfeature.get('rings',[])if r['role']==role),oldfeature)
            newring=next((r for r in newfeature.get('rings',[])if r['role']==role),newfeature)
            angle=np.arctan2(before[:,1]-oldring['center_xyz'][1],before[:,0]-oldring['center_xyz'][0])
            centre=np.asarray(newring['center_xyz'],float)
            result['world']=centre+newring['radius']*np.c_[np.cos(angle),np.sin(angle),np.zeros(len(angle))]
            identity=dict(kind='retained_circle_and_ring',feature=feature,ring_role=role)
        elif lane:
            feature,edge=lane[1],int(lane[2]);f=updated[feature]
            if edge in f.get('open_edges',[]):raise ValueError(f'Unmeasured Lane proxy edge {name} cannot bind as paint')
            polygon=np.asarray(f['polygon_xyz'],float);old=np.asarray(previous[feature]['polygon_xyz'],float)
            direction=old[(edge+1)%len(old)]-old[edge]
            t=((before-old[edge])@direction)/float(direction@direction)
            result['world']=polygon[edge]+t[:,None]*(polygon[(edge+1)%len(polygon)]-polygon[edge])
            identity=dict(kind='finite_known_lane_edge',feature=feature,edge=edge)
        elif name in updated and updated[name]['kind']=='polyline':
            segments=updated[name]['segments_xyz']
            if len(segments)!=1:raise ValueError(f'Multiple finite segments require an explicit segment ID for {name}')
            path=np.asarray(segments[0],float);old=np.asarray(previous[name]['segments_xyz'][0],float)
            direction=old[-1]-old[0];t=((before-old[0])@direction)/float(direction@direction)
            result['world']=path[0]+t[:,None]*(path[-1]-path[0])
            identity=dict(kind='finite_model_polyline',feature=name)
        else:
            raise ValueError(f"No modeled physical curve identity for source {row['pts']}/{name}")
    result['source_uv_float64_sha256_preserved']=hashlib.sha256(np.asarray(row['uv'],np.float64).tobytes()).hexdigest()
    result['original_world_sha256']=hashlib.sha256(before.tobytes()).hexdigest()
    result['world_scene_binding']=dict(identity,source_uv_changed=False,
        physical_identity_status='retained original source identity; never selected from candidate projection',
        versioned_geometry='full supplied candidate scene, not a mixture of field versions')
    return result
