"""Source-bound two-boot observation preparation; no camera/scene updates."""
from pathlib import Path
import argparse,json,hashlib,time
import cv2
import numpy as np
from .rebuild_prepare import load,digest
from rinkspace.review.render import source_frames




def _edge_midpoint_occluder(origin,target,triangles,excluded,span):
    """First two-sided triangle hit strictly before an edge midpoint, or None."""
    direction=target-origin;length=float(np.linalg.norm(direction))
    if length<=span*1e-12:raise ValueError('Camera centre lies on a candidate shell edge')
    for fi,triangle in enumerate(triangles):
        if fi in excluded:continue
        a,b,c=triangle;e1=b-a;e2=c-a;cross=np.cross(direction,e2);det=float(e1@cross)
        tolerance=1e-12*length*np.linalg.norm(e1)*np.linalg.norm(e2)
        if abs(det)<=tolerance:continue
        offset=origin-a;u=float(offset@cross)/det
        q=np.cross(offset,e1);v=float(direction@q)/det;t=float(e2@q)/det
        if u>=-1e-10 and v>=-1e-10 and u+v<=1+1e-10 and 1e-10<t<1-1e-9:return fi
    return None


def visible_shell_edges(vertices_world,faces,camera_centre,region_ids=None):
    """Approximate visible edges of the explicit low-resolution boot surface.

    Faces must use consistent outward directed winding. Shared front/back face
    transitions supply silhouette candidates; every open edge is a candidate,
    including the two-sided thin collar rim. A camera-to-edge-midpoint ray
    rejects candidates hidden by another face. This is a low-resolution mesh
    visibility approximation, not exact pixel segmentation or full-edge
    visibility. It considers only self-occlusion by this shell, not the body or
    the other boot. No convex hull, surface closing, or vertices are created.

    Returns a dictionary with local-shell edge index pairs, adjacent face-region
    IDs, edge types and rejected midpoint-occlusion evidence. region_ids, when
    supplied, is one scalar label per face; the function does not infer anatomy
    or material regions. The caller can pass result['edges'] to pixel_factors.
    """
    vertices=np.asarray(vertices_world,float);faces=np.asarray(faces);camera=np.asarray(camera_centre,float)
    if vertices.shape!=(14,3) or camera.shape!=(3,) or not np.isfinite(vertices).all() or not np.isfinite(camera).all():
        raise ValueError('Expected 14 finite world shell vertices and one finite camera centre')
    if faces.ndim!=2 or faces.shape[1]!=3 or not len(faces) or faces.dtype.kind not in 'iu' or np.any(faces<0) or np.any(faces>=14):
        raise ValueError('Expected explicit integer triangle indices into the 14 shell vertices')
    if np.any(faces[:,0]==faces[:,1]) or np.any(faces[:,1]==faces[:,2]) or np.any(faces[:,2]==faces[:,0]):
        raise ValueError('Shell faces must have three distinct vertices')
    span=float(np.max(np.linalg.norm(vertices-vertices.mean(axis=0),axis=1)))
    if span<=0:raise ValueError('Degenerate shell extent')
    triangles=vertices[faces];centres=triangles.mean(axis=1)
    normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]);areas=np.linalg.norm(normals,axis=1)
    if np.any(areas<=1e-12*span**2):raise ValueError('Degenerate explicit shell triangle')
    toward=camera-centres;distance=np.linalg.norm(toward,axis=1)
    if np.any(distance<=span*1e-12):raise ValueError('Camera centre lies on a shell face')
    facing=np.einsum('ij,ij->i',normals,toward)/(areas*distance);front=facing>1e-12
    if region_ids is None:regions=['unspecified_outer_shell']*len(faces)
    else:
        regions=np.asarray(region_ids)
        if regions.shape!=(len(faces),) or regions.dtype.kind not in 'iufUS':
            raise ValueError('region_ids must contain one scalar label per explicit face')
        if regions.dtype.kind=='f' and not np.isfinite(regions).all():raise ValueError('Nonfinite face-region label')
        regions=[value.item() if isinstance(value,np.generic) else value for value in regions]
    adjacency={}
    for fi,face in enumerate(faces):
        for j in range(3):
            a,b=int(face[j]),int(face[(j+1)%3]);key=tuple(sorted((a,b)))
            adjacency.setdefault(key,[]).append((fi,a,b))
    edges=[];edge_regions=[];types=[];adjacent_faces=[];occluded=[];occluders=[]
    for edge,rows in sorted(adjacency.items()):
        if len(rows)>2:raise ValueError('Nonmanifold explicit shell edge')
        if len(rows)==2 and rows[0][1:]==rows[1][1:]:raise ValueError('Inconsistent directed shell face winding')
        connected=[row[0] for row in rows]
        if len(rows)==1:kind='open_boundary'
        elif front[connected[0]]!=front[connected[1]]:kind='silhouette'
        else:continue
        hidden_by=_edge_midpoint_occluder(camera,vertices[list(edge)].mean(axis=0),triangles,set(connected),span)
        if hidden_by is not None:
            occluded.append(edge);occluders.append(hidden_by);continue
        labels=[]
        for fi in connected:
            if regions[fi] not in labels:labels.append(regions[fi])
        edges.append(edge);edge_regions.append(labels);types.append(kind);adjacent_faces.append(connected)
    return dict(schema='rinkspace.visible-shell-edges.v1',edges=np.asarray(edges,dtype=np.int32).reshape(-1,2),
        region_ids=edge_regions,edge_types=types,adjacent_faces=adjacent_faces,face_front_facing=front,
        face_facing_cosine=facing,occluded_edges=np.asarray(occluded,dtype=np.int32).reshape(-1,2),
        midpoint_occluder_faces=occluders,
        method='outward directed face orientation, silhouette/open edges, two-sided midpoint ray self-occlusion',
        approximation='low-resolution explicit mesh visibility; midpoint only, not exact segmentation or complete edge visibility; body/other-boot occlusion not tested')


def _audit_json_value(value):
    """JSON null records missing numeric evidence; original arrays stay untouched."""
    if isinstance(value,np.ndarray):return _audit_json_value(value.tolist())
    if isinstance(value,np.generic):return _audit_json_value(value.item())
    if isinstance(value,dict):return {str(key):_audit_json_value(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [_audit_json_value(item) for item in value]
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def audit_observation_semantics(root,frames=None,pts=None,output=None):
    """Read the existing admitted rows and audit physical meaning without edits.

    Detector ankle proxies, outer-boot contours/heel/toe edges, rocker-boundary
    candidates and missing toe-pick evidence remain separate observation types.
    This does not requalify observations, identify contact, or infer shoe-internal
    anatomy. No source decoding, neural inference or fitting is performed.
    """
    from .fullbody_fit import footwear_rows
    root=Path(root);raw=load(root/'input.npz');clock=raw['pts'] if pts is None else np.asarray(pts)
    np.testing.assert_array_equal(clock,raw['pts'])
    frames=np.arange(len(clock)) if frames is None else np.asarray(frames,dtype=int)
    if frames.ndim!=1 or np.any(frames<0) or np.any(frames>=len(clock)) or len(np.unique(frames))!=len(frames):
        raise ValueError('Audit frames must be unique valid source-frame indices')
    gear,sources=footwear_rows(root,frames,clock);review=json.loads(Path(sources['review_path']).read_text(encoding='utf8'))
    records=[]
    for fi in frames:
        for foot,joint in enumerate((15,16)):
            records.append(dict(kind='raw_ankle_detector_proxy',pts=int(clock[fi]),foot=foot,
                source_id='COCO17',source_joint_index=joint,raw_uv=raw['source_uv'][fi,joint].tolist(),
                raw_score=float(raw['source_score'][fi,joint]),
                raw_uv_nonfinite={str(axis):str(float(value)) for axis,value in enumerate(raw['source_uv'][fi,joint]) if not np.isfinite(value)},
                raw_score_nonfinite=str(float(raw['source_score'][fi,joint])) if not np.isfinite(raw['source_score'][fi,joint]) else None,
                raw_qualified=bool(raw['source_qualified'][fi,joint]),
                lower_body_inferred=bool(raw['lower_body_inferred'][fi]),
                interpretation='detector proxy under clothing/boot occlusion, not an exact bony ankle centre',
                admitted_as_exact_bony_ankle=False))
    for row in gear.visible_rows:
        records.append(dict(kind='outer_boot_shell_boundary',pts=int(row['pts']),foot=int(row['foot']),
            raw_uv=np.asarray(row['observed']).tolist(),sigma_px=float(row['sigma_px']),
            source_only=True,normalization_source_samples=48,original_role=row.get('role'),
            physical_identity_status=row.get('physical_identity_status','existing track hypothesis; not independently confirmed'),
            interpretation='visible external boot shell boundary; not bare foot, inner cavity, or a specific tongue/collar annotation',
            original_geometry_semantic_id=row.get('geometry_semantic_id'),human_confirmed=bool(row.get('human_confirmed',False))))
    for row in gear.landmark_rows:
        for name,indices in (('heel',[0,7]),('toe',[1,8])):
            records.append(dict(kind='outer_boot_'+name+'_edge',pts=int(row['pts']),foot=int(row['foot']),
                raw_uv=list(row[name]),sigma_px=float(row['sigma_px']),finite_shell_vertex_pair=indices,
                original_role=row.get('role'),physical_identity_status=row.get('physical_identity_status','existing track hypothesis; not independently confirmed'),
                interpretation='external lower boot '+name+' edge; not bare-foot '+name+', bony ankle, or measured ice contact',
                human_confirmed=bool(row.get('human_confirmed',False))))
    for row in gear.blade_rows:
        records.append(dict(kind='blade_rocker_boundary_candidates',pts=int(row['pts']),foot=int(row['foot']),
            raw_uv=np.asarray(row['uv']).tolist(),sigma_px=float(row['sigma_px']),partial=bool(row['partial']),
            candidate_edge_ids=['blade_left','blade_right'],local_vertex_indices=[list(range(14,31)),list(range(31,48))],
            candidate_selection='one finite side candidate per complete source curve, never independent per-point switching',
            source_side_confirmed=bool(row.get('source_side_confirmed',False)),
            physical_identity_status=row.get('physical_identity_status','existing track hypothesis; not independently confirmed'),
            interpretation='visible lower blade-boundary segment against approximate rocker geometry; does not identify toe pick or contact point',
            original_role=row.get('role')))
    report=dict(schema='rinkspace.footwear-observation-semantics.v1',source_time_base=[1,12800],
        coordinate_space='native source image pixels; original values preserved',evaluation_role='development',
        nonfinite_source_encoding='JSON null, with raw_uv_nonfinite/raw_score_nonfinite retaining NaN/Inf status; no replacement observation inferred',
        sources=dict(input_path=str(root/'input.npz'),input_sha256=digest(root/'input.npz'),
                     equipment=sources,video=review.get('source',{})),source_pts=clock[frames].tolist(),records=records,
        counts=dict(raw_ankle_detector_proxy_points=2*len(frames),
            nonfinite_raw_ankle_proxy_points=int(np.any(~np.isfinite(raw['source_uv'][frames][:,[15,16]]),axis=-1).sum()),
            outer_boot_contour_records=len(gear.visible_rows),
            outer_boot_contour_source_points=sum(len(row['observed']) for row in gear.visible_rows),
            outer_boot_heel_toe_points=2*len(gear.landmark_rows),blade_curve_records=len(gear.blade_rows),
            blade_original_curve_vertices=sum(len(row['uv']) for row in gear.blade_rows),toe_pick_source_points=0,
            independently_annotated_tongue_collar_points=0),
        missing=dict(toe_pick=dict(status='unobserved in these existing admitted source rows',source_points=0,
            blade_front_endpoint_is_toe_pick=False,model_candidate_is_source_observation=False),
            tongue_collar=dict(status='no independently annotated region-specific contour in these admitted rows',source_points=0),
            shoe_internal_fit=dict(status='not observed; cavity, heel pocket, footbed and skeletal ankle relations remain model assumptions')),
        unchanged_source_values=True,body_model_projection_used_as_measurement=False,
        limitations=['Raw detector ankle is a proxy, not measured internal ankle anatomy.',
            'Outer boot heel/toe and rocker edges are not bare-foot landmarks or confirmed ice-contact points.',
            'No toe-pick observations are fabricated from blade endpoints or model candidates.',
            'Region-specific upper deformation cannot be certified by an unlabelled whole-boot contour.'])
    report=_audit_json_value(report)
    if output is not None:
        from rinkspace.path_policy import guard_project_write
        destination=guard_project_write(output)
        source_assets={root/'input.npz',Path(sources['review_path']),Path(sources['admission_path']),Path(sources['review_path']).parent/'config.json'}
        if destination.resolve() in {p.resolve() for p in source_assets}:
            raise ValueError('Semantic audit output cannot overwrite original source observations or admission/config records')
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    return report


def prepare(root,project,review_path,dataset=None):
    root,project=Path(root),Path(project);cache=root/'footwear_v2';cache.mkdir(exist_ok=True);out=Path(dataset) if dataset else cache;out.mkdir(exist_ok=True);(cache/'source_crops').mkdir(exist_ok=True)
    p=load(root/'input.npz');base=load(root/'pose/result.npz');meta=json.loads((root/'input.json').read_text(encoding='utf8'))
    review_path=Path(review_path);review=json.loads(review_path.read_text(encoding='utf8'));scope=json.loads((review_path.parent/review['source_scope']).read_text(encoding='utf8'));source={**scope['source'],**scope['source_clock']}
    # Unseen detailed boot observations in contiguous time windows. These
    # pixels do not calibrate shape, pose choices, or residual scales.
    holdouts=review['check_windows_s'];guard=review['check_guard_s']
    requested=np.asarray(review['fit_source_seconds'])
    selected=set(int(p['pts'][np.argmin(abs(p['pts']/12800-t))]) for t in requested)
    for lo,hi in holdouts:selected.update(p['pts'][(p['pts']/12800>=lo)&(p['pts']/12800<=hi)].tolist())
    rows=[]
    for pts,bgr in source_frames(source['registered_path'],source,p['pts']):
        if pts not in selected:continue
        i=int(np.searchsorted(p['pts'],pts));t=pts/12800
        hold=any(lo<=t<=hi for lo,hi in holdouts);protected=any(lo-guard<=t<=hi+guard for lo,hi in holdouts)
        role='check' if hold else ('protected' if protected else 'fit')
        qualified=p['source_qualified'][i];xy=p['source_uv'][i];bh=p['source_bbox'][i,3]-p['source_bbox'][i,1]
        foot_ids=np.array([13,14,15,16]);good=qualified[foot_ids]
        if p['lower_body_inferred'][i] or not good.all():
            rows.append(dict(pts=pts,role=role,status='unknown_lower_body',source_seconds=t));continue
        lo=np.maximum(np.floor(xy[foot_ids].min(0)-[bh*.22,bh*.08]).astype(int),[0,0]);hi=np.minimum(np.ceil(xy[foot_ids].max(0)+[bh*.22,bh*.24]).astype(int),[1920,1080])
        crop=bgr[lo[1]:hi[1],lo[0]:hi[0]].copy();file=cache/'source_crops'/f'{pts}.png'
        if not file.exists():
            if not cv2.imwrite(str(file),crop):raise OSError(file)
        ankle=xy[[15,16]];knee=xy[[13,14]];centres=ankle+(ankle-knee)*.08
        boxes=np.c_[centres-[bh*.13,bh*.085],centres+[bh*.13,bh*.16]]
        boxes[:,:2]=np.maximum(boxes[:,:2],lo);boxes[:,2:]=np.minimum(boxes[:,2:],hi)
        rows.append(dict(pts=pts,source_seconds=t,role=role,status='source_crop_available',crop=str(file),crop_xyxy=[*lo.tolist(),*hi.tolist()],
            crop_sha256=digest(file),decoded_native_crop_rgb_sha256=hashlib.sha256(crop[:,:,::-1].tobytes()).hexdigest(),
            ankle_source_uv=ankle.tolist(),knee_source_uv=knee.tolist(),search_centres_uv=centres.tolist(),search_boxes_uv=boxes.tolist(),
            source_left_right='saved COCO physical identity candidate; crossing/occlusion must be checked',body_model_projection_used_as_measurement=False))
    report=dict(schema='rinkspace.two-boot-observation-crops.v1',source=source,shot='s01',base_body_sha256=digest(root/'pose/result.npz'),
        input_sha256=digest(root/'input.npz'),camera_snapshot_sha256=digest(root/'camera_snapshot.npz'),
        detailed_boot_check_windows_s=holdouts,check_guard_s=guard,source_time_base='1/12800',rows=rows,
        inference='none at preparation; boxes navigate only; unchanged decoded RGB cropped without resizing',
        camera_and_scene_frozen=True,base_pelvis_frozen=True)
    (out/'source_index.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    # Source-only atlas omits check windows until calibration decisions freeze.
    usable=[r for r in rows if r['status']=='source_crop_available' and r['role']=='fit']
    for page,start in enumerate(range(0,len(usable),12)):
        tiles=[]
        for r in usable[start:start+12]:
            crop=cv2.imread(r['crop']);box=np.full((340,480,3),24,np.uint8);s=min(470/crop.shape[1],295/crop.shape[0]);im=cv2.resize(crop,None,fx=s,fy=s,interpolation=cv2.INTER_CUBIC)
            box[40:40+im.shape[0],5:5+im.shape[1]]=im;cv2.putText(box,f'{r["source_seconds"]:.3f}s / {r["pts"]}',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1,cv2.LINE_AA)
            for foot,xy in enumerate(r['ankle_source_uv']):
                at=np.rint((np.asarray(xy)-r['crop_xyxy'][:2])*s+[5,40]).astype(int);cv2.circle(box,tuple(at),5,(0,210,255) if foot==0 else (255,180,80),1,cv2.LINE_AA);cv2.putText(box,'LR'[foot],tuple(at+[8,-5]),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1,cv2.LINE_AA)
            tiles.append(box)
        while len(tiles)%3:tiles.append(np.full_like(tiles[0],24))
        cv2.imwrite(str(out/f'source_atlas_{page}.jpg'),np.vstack([np.hstack(tiles[k:k+3]) for k in range(0,len(tiles),3)]),[cv2.IMWRITE_JPEG_QUALITY,94])
    print(json.dumps(dict(rows=len(rows),fit=sum(r['role']=='fit' for r in rows),check=sum(r['role']=='check' for r in rows),protected=sum(r['role']=='protected' for r in rows),atlases=(len(usable)+11)//12)),flush=True)

def candidate_atlas(root):
    out=Path(root)/'footwear_v2';index=json.loads((out/'source_index.json').read_text(encoding='utf8'))
    rows=[r for r in index['rows'] if r['role']=='fit' and (out/'mask_candidates'/str(r['pts'])).with_name(str(r['pts'])+'_qa.jpg').exists()]
    for page,start in enumerate(range(0,len(rows),12)):
        tiles=[]
        for row in rows[start:start+12]:
            image=cv2.imread(str(out/'mask_candidates'/(str(row['pts'])+'_qa.jpg')));box=np.full((300,420,3),24,np.uint8)
            scale=min(410/image.shape[1],260/image.shape[0]);image=cv2.resize(image,None,fx=scale,fy=scale)
            box[35:35+image.shape[0],5:5+image.shape[1]]=image
            cv2.putText(box,str(row['source_seconds'])+'s / '+str(row['pts']),(8,23),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA);tiles.append(box)
        while len(tiles)%3:tiles.append(np.full_like(tiles[0],24))
        cv2.imwrite(str(out/('mask_atlas_'+str(page)+'.jpg')),np.vstack([np.hstack(tiles[k:k+3]) for k in range(0,len(tiles),3)]))
    print(len(rows),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--project',type=Path,required=True);ap.add_argument('--review',type=Path,required=True);ap.add_argument('--dataset',type=Path);a=ap.parse_args();prepare(a.root,a.project,a.review,a.dataset)
