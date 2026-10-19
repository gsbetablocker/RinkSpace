"""Rebuild the portable input packet from the user's read-only review bundle.

Only allow_pickle=False NumPy arrays and JSON are read. No production owner,
historical launcher, authentication bypass, source relabeling or fit is run.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
from collections import defaultdict
import numpy as np

FIELDS = ("source_point_ids", "source_target_labels", "native_pts", "frame_indices", "source_uv")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def plain(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, np.generic): return obj.item()
    if isinstance(obj, dict): return {str(k):plain(v) for k,v in obj.items()}
    if isinstance(obj, (tuple,list)): return [plain(v) for v in obj]
    return obj


def decode(obj, arrays):
    if isinstance(obj, dict):
        if set(obj) == {"_safe_npz_array"}: return arrays[obj["_safe_npz_array"]].copy()
        if set(obj) == {"_safe_tuple"}: return tuple(decode(v,arrays) for v in obj["_safe_tuple"])
        return {k:decode(v,arrays) for k,v in obj.items()}
    if isinstance(obj,list): return [decode(v,arrays) for v in obj]
    return obj


def descriptor_labels(desc, labels):
    kind=desc['kind']
    if kind == 'board_edge':
        prefix='board.'+desc['role']+'.'
        policy=desc.get('association_policy',{})
        if policy.get('original_part_is_context_not_a_hard_correspondence'):
            return [s for s in labels if s.startswith(prefix)]
        return [prefix+desc['part']]
    if kind in ('ad_edge','ad_corner'):
        return [f"Ad{desc['advertisement_id']}.{desc['role']}"]
    if kind in ('ad_print','ad_print_curve'):
        return [f"Ad{desc['advertisement_id']}.print.{desc['contour_key']}"]
    if kind == 'ice_curve': return ['ice.'+desc['curve_id']]
    raise ValueError(f'Unimplemented source descriptor {kind}; do not guess aliases')


def plane_for_group(points, bounds):
    """Only the four whole rink straight faces, never one plane per logo."""
    x0,y0,x1,y1=bounds
    for axis,coordinate,name in [(0,x0,'board_x_min'),(0,x1,'board_x_max'),
                                 (1,y0,'board_y_min'),(1,y1,'board_y_max')]:
        if np.max(np.abs(points[:,axis]-coordinate)) < 1e-9:
            return name
    return None


def explicit_contour_topology(scene, descriptor, vertex_count, source_scene_sha256, templates=None):
    """Recover recorded connectivity, never infer a closing edge by distance.

    Older geometry exports stripped ``closed`` from advertisements while their
    source-conditioned copy retained it.  These are two representations of the
    same named material; conflicting explicit flags are a hard export error.
    """
    if descriptor.get('kind') not in ('ad_print', 'ad_print_curve'):
        return None
    number=int(descriptor['advertisement_id']);key=descriptor['contour_key'];evidence=[]
    if descriptor.get('physical_primitive')=='circle_on_continuous_unwrapped_board':
        return dict(path_offsets=[0,int(vertex_count)],path_closed=[True],
            topology_provenance=dict(method='explicit_source_closed_circle_primitive',source_scene_sha256=source_scene_sha256,
                source_descriptor=descriptor,endpoint_distance_used=False))
    if descriptor.get('template') and isinstance(templates,dict):
        has_ad=str(number) in templates.get('advertisements',{})
        template=templates['advertisements'][str(number)] if has_ad else templates
        contour=template.get('contours',{}).get(key)
        if contour is not None and isinstance(contour.get('closed'),bool):
            escaped=key.replace('~','~0').replace('/','~1')
            prefix=f'/payload/templates/advertisements/{number}' if has_ad else '/payload/templates'
            evidence.append((bool(contour['closed']),prefix+'/contours/'+escaped+'/closed'))
    if descriptor.get('geometry_source') != 'scene_existing_external_contour':
        if not evidence:return None
        return dict(path_offsets=[0,int(vertex_count)],path_closed=[evidence[0][0]],
            topology_provenance=dict(method='explicit_named_template_closed_flag',source_scene_sha256=source_scene_sha256,
                json_pointers=[pointer for _,pointer in evidence],endpoint_distance_used=False))
    for ai,ad in enumerate(scene.get('advertisements', [])):
        if int(ad['user_number'])!=number:continue
        for ci,contour in enumerate(ad.get('external_contours', [])):
            if contour.get('id')==key and isinstance(contour.get('closed'),bool):
                evidence.append((bool(contour['closed']),f'/payload/scene/advertisements/{ai}/external_contours/{ci}/closed'))
    for ai,ad in enumerate(scene.get('source_conditioned_board_prints',{}).get('patterns', [])):
        if int(ad['advertisement_id'])!=number:continue
        for ci,contour in enumerate(ad.get('external_contours', [])):
            if contour.get('id')==key and isinstance(contour.get('closed'),bool):
                evidence.append((bool(contour['closed']),f'/payload/scene/source_conditioned_board_prints/patterns/{ai}/external_contours/{ci}/closed'))
    if not evidence:return None
    if len({value for value,_ in evidence})!=1:
        raise ValueError('Conflicting explicit contour topology for '+key)
    return dict(path_offsets=[0,int(vertex_count)],path_closed=[evidence[0][0]],
        topology_provenance=dict(method='explicit_named_material_closed_flag',source_scene_sha256=source_scene_sha256,
            json_pointers=[pointer for _,pointer in evidence],endpoint_distance_used=False))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('review_root',type=Path)
    ap.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]/'data')
    ap.add_argument('--camera-audit-module',type=Path,default=Path(__file__).resolve().parent/'frozen_camera_math.py',
                    help='Audited frozen_camera_math.py, used only to export a labeled legacy seed')
    args=ap.parse_args(); out=args.out.resolve(); out.mkdir(parents=True,exist_ok=True)
    run=args.review_root.resolve()/'data/audit_1017/motion_joint_body2'
    with np.load(run/'immutable_programme_source_inventory.npz',allow_pickle=False) as z:
        obs={k:z[k].copy() for k in FIELDS}
    hashes={k:{'dtype_str':v.dtype.str,'shape':list(v.shape),
               'sha256_c_contiguous_array_bytes':sha(np.ascontiguousarray(v).tobytes())}
            for k,v in obs.items()}
    np.savez_compressed(out/'observations.npz',**obs)
    with np.load(run/'source_sigma_and_formal_offsets.npz',allow_pickle=False) as z:
        np.savez_compressed(out/'source_sigma.npz',**{k:z[k] for k in z.files})
    source=json.loads((run/'actual_geometry_final/actual_scene.json').read_text())
    with np.load(run/'actual_geometry_final/actual_scene.npz',allow_pickle=False) as z:
        packet=decode(source['payload'],z)
    scene=packet['scene']; bounds=np.array(scene['parameters']['bounds'],float)
    length=float(bounds[2]-bounds[0]); scaled_bounds=bounds/length
    labels=sorted(set(obs['source_target_labels'].tolist()))
    all_world={k:np.asarray(v,float)/length for k,v in packet['world_by_target'].items()}
    # An ad belongs to a straight face only if ALL its exported material vertices
    # belong to the same actual rink face. A single line never defines a plane.
    ad_points=defaultdict(list)
    for key,world in all_world.items():
        desc=json.loads(key)
        if 'advertisement_id' in desc: ad_points[desc['advertisement_id']].append(world)
    ad_planes={i:plane_for_group(np.concatenate(parts),scaled_bounds) for i,parts in ad_points.items()}
    height=float(scene['parameters']['board_height'])/length
    x0,y0,x1,y1=scaled_bounds
    planes={
      'ice':{'origin':[0.,0.,0.],'axis_u':[1.,0.,0.],'axis_v':[0.,1.,0.], 'membership':'source ice materials and source contact boundary'},
      'board_top':{'origin':[0.,0.,height],'axis_u':[1.,0.,0.],'axis_v':[0.,1.,0.], 'membership':'shared-height model candidate; not an independently measured height'},
      'board_x_min':{'origin':[float(x0),0.,0.],'axis_u':[0.,1.,0.],'axis_v':[0.,0.,1.]},
      'board_x_max':{'origin':[float(x1),0.,0.],'axis_u':[0.,1.,0.],'axis_v':[0.,0.,1.]},
      'board_y_min':{'origin':[0.,float(y0),0.],'axis_u':[1.,0.,0.],'axis_v':[0.,0.,1.]},
      'board_y_max':{'origin':[0.,float(y1),0.],'axis_u':[1.,0.,0.],'axis_v':[0.,0.,1.]}}
    features={f['id']:f for f in scene['ice_features']}
    records={s:{'label':s,'variants':[],'source_identity_trusted':True} for s in labels}
    world_arrays={}; variants={}
    for index,(key,world) in enumerate(all_world.items()):
        desc=json.loads(key); name=f'w{index:03d}'; world_arrays[name]=world
        kind=desc['kind']; targets=descriptor_labels(desc,labels)
        targets=[s for s in targets if s in records]
        if not targets: raise ValueError(f'Exported descriptor maps to no source label: {desc}')
        plane=None
        if kind=='ice_curve': plane='ice'
        elif kind=='board_edge': plane='ice' if desc['role']=='contact' else 'board_top'
        elif 'advertisement_id' in desc: plane=ad_planes[desc['advertisement_id']]
        primitive={'kind':'point' if len(world)==1 else 'polyline', 'world_array':name}
        topology=desc.get('curve_topology')
        if topology:
            primitive['path_offsets']=topology['offsets']; primitive['path_closed']=topology['closed']
            primitive['topology_provenance']={'method':'explicit_source_descriptor_curve_topology','descriptor':desc}
            if topology['offsets'][-1] != len(world): raise ValueError('Explicit topology size mismatch')
        else:
            recovered=explicit_contour_topology(scene,desc,len(world),sha((run/'actual_geometry_final/actual_scene.json').read_bytes()),packet.get('templates'))
            if recovered:primitive.update(recovered)
        if kind=='ice_curve':
            feature_id,role=desc['curve_id'].split(':',1)
            f=features.get(feature_id,{})
            rings=[v for v in f.get('rings',[]) if v['role']==role]
            if f.get('kind')=='circle' and (rings or role=='outer'):
                ring=rings[0] if rings else f
                primitive={'kind':'circle','center_xyz':(np.array(ring['center_xyz'])/length).tolist(),
                           'radius':float(ring['radius'])/length,'angle_interval':[0.,float(2*np.pi)]}
            elif len(world)>=2 and np.linalg.matrix_rank(world-world[0],tol=1e-10)==1:
                primitive={'kind':'segment','a_xyz':world[0].tolist(),'b_xyz':world[-1].tolist()}
        item={'variant_id':name,'descriptor':desc,'plane_candidate':plane,
              'geometry':primitive,'world_array':name,
              'geometry_role':'legacy_static_seed_only_not_world_truth',
              'observation_semantic':'physical_point' if kind=='ad_corner' else 'finite_curve',
              'connectivity_provenance':'ordered evaluated model geometry, never observed point-array order'}
        variants[name]=item
        for target in targets: records[target]['variants'].append(name)
    assert all(v['variants'] for v in records.values())
    np.savez_compressed(out/'world_seeds.npz',**world_arrays)
    sys.path.insert(0,str(args.camera_audit_module.resolve().parent))
    from frozen_camera_math import FrozenCamera
    camera=FrozenCamera(run)
    from scipy.spatial.transform import Rotation
    mid=camera.camera25(np.arange(1,1249),539.5)
    np.savez_compressed(out/'camera_seeds.npz', frame_indices=np.arange(1,1249,dtype=np.int32),
        native_pts=np.arange(256256,575489,256,dtype=np.int64),
        R_world_to_camera=Rotation.from_rotvec(mid[:,3:6]).as_matrix(),
        center_world=mid[:,:3]/length, focal_px=np.exp(mid[:,6:8]),
        principal_xy=mid[:,8:10], brown_k1_k2_p1_p2=mid[:,11:15])
    shared={'pivot':(camera.P/length).tolist(),'lever':(camera.lever/length).tolist(),
        'pan_axis_world':camera.n.tolist(),'tilt_axis_camera':camera.b.tolist(),
        'R0_world_to_camera':camera.R0.tolist(),'axis_spacing':float(camera.axis/length),
        'readout_seconds':camera.readout_ms/1000.,'legacy_time_basis':'linear inter-frame; not the new cubic trajectory',
        'seed_role':'historical candidate, not accuracy accepted or a unique initialization'}
    dump(out/'seeds.json',{'schema':'rinkfit.seed_catalog.v1','canonical_q_sha256':source['canonical_q_sha256'],
        'source_scene_sha256':sha((run/'actual_geometry_final/actual_scene.json').read_bytes()),
        'world_gauge':{'rink_length':1.,'legacy_units_per_new_unit':length,'verified_metre_scale':False},
        'planes':planes,'legacy_head':shared,'scene_parameters_legacy':plain(scene['parameters']),
        'scene_hard_relations':plain({k:v for k,v in scene.items() if 'hard_' in k or 'shared_' in k}),
        'materials':list(records.values()),'variants':variants})
    media=json.loads((run/'media_manifest.json').read_text())
    frame_fields=('programme_frame_index_1based','native_frame_index_1based','native_pts','actual_pts',
                  'time_base','source_time','source','source_video_sha256')
    dump(out/'media.json',{'schema':'rinkfit.source_media.v1','video_included':False,
        'frames':[{k:f[k] for k in frame_fields if k in f} for f in media['frames']],
        'source_frame_map':media['source_frame_map']})
    dump(out/'input_manifest.json',{'schema':'rinkfit.input_manifest.v1','shot_id':'ankara_2026/okada_mei_sp/shot1',
        'observations':182017,'frames':1248,'material_labels':171,'frame_min':1,'frame_max':1248,
        'pts_origin':256256,'pts_step':256,'time_base':[1,12800],'image_wh':[1920,1080],
        'pixel_centers':'integer','primary_key':['shot_id','native_pts','source_point_id'],
        'review_status':'all UV and original material identities accepted by user',
        'source_file':str(run/'immutable_programme_source_inventory.npz'),'arrays':hashes,
        'repacked_without_array_value_or_dtype_changes':True,
        'statistics':{'scalar_sigma_file':'source_sigma.npz','full_grouped_W_included':False,
          'scalar_sigma_must_not_replace_full_grouped_W':'resolve original full packet through existing project adapter before S5'},
        'semantics':{'physical_corner_kind_from_source_descriptor':True,
          'ordinary_curve_station_is_unknown':True,'verified_run_order_included':False,
          'true_profile_metadata_included':False,
          'missing_metadata_does_not_revoke_reviewed_observations':True}})
    print(json.dumps({'output':str(out),'observations':len(obs['source_uv']),'labels':len(records),
        'seed_variants':len(variants),'multi_variant_labels':[k for k,v in records.items() if len(v['variants'])>1],
        'files':{p.name:p.stat().st_size for p in out.iterdir() if p.is_file()}}))

if __name__=='__main__': main()
