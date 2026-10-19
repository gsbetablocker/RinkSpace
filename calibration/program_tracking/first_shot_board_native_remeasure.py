"""Source-only board-group remeasurement using the existing native reader and joint D/W.

No camera, adapter or fit state is loaded. Original accepted members are search
navigation and remain immutable archives; measurements come only from RGB.
"""
from copy import deepcopy
from pathlib import Path
import hashlib,json
import cv2
import numpy as np
from .extract_first_shot_board_source import profile_read,path_stations,segment_records
from .first_shot_source_evidence import prepare_frame_evidence

def reference(path):
    p=Path(path)
    return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def _same_source_intersection(horizontal,vertical):
    # Same cv2 DIST_HUBER side intersections as extract_frame/refined_ad_mapping.
    h=cv2.fitLine(np.asarray(horizontal,np.float32),cv2.DIST_HUBER,0,.001,.001).ravel().astype(float)
    v=cv2.fitLine(np.asarray(vertical,np.float32),cv2.DIST_HUBER,0,.001,.001).ravel().astype(float)
    t=np.linalg.solve(np.column_stack((v[:2],-h[:2])),h[2:]-v[2:])[0]
    return v[2:]+t*v[:2]

def remeasure(packet,members,registry,*,advertisement_id):
    """Remeasure one complete named ad group, never admitting the result here.

    members contain the actual current target, original ID and UV. Contiguous
    original visible runs initialize each side; no hidden material is supplied.
    """
    prefix='Ad'+str(advertisement_id)+'.';source=packet['source']
    raw=cv2.imread(source['image_path']);mask=cv2.imread(source['mask_path'],0)
    if raw is None or mask is None:raise ValueError('Original RGB/mask required')
    if hashlib.sha256(raw.tobytes()).hexdigest()!=source['decoded_BGR_sha256']:
        raise ValueError('Decoded source RGB changed')
    mask=cv2.dilate((mask>0).astype(np.uint8),np.ones((5,5),np.uint8))>0
    mask[:120,:570]=True;mask[945:,1660:]=True
    current={m['source_point_id']:m for m in members};corners={};paths={r:[] for r in ['top_edge','bottom_edge','left_vertical','right_vertical']}
    for m in members:
        t=registry[m['current_target_label']]
        if t['kind']=='ad_corner':corners[t['role']]=np.asarray(m['UV'],float)
        elif t['kind']!='ad_edge':raise ValueError('Only one whole ad boundary/corner group allowed')
        if t['advertisement_id']!=advertisement_id:raise ValueError('Mixed physical advertisements')
    for feature in packet['features']:
        if not feature['id'].startswith(prefix):continue
        for si,segment in enumerate(feature['segments']):
            for role in paths:
                run=[]
                def finish():
                    if len(run)>=2:paths[role].append(np.asarray(run,float))
                    run.clear()
                for j,p in enumerate(segment['uv']):
                    pid=f'{feature["id"]}/segment/{si}/vertex/{j}';m=current.get(pid)
                    accepted=False
                    if m:
                        t=registry[m['current_target_label']]
                        accepted=(t['kind']=='ad_edge'and t['role']==role)
                        if t['kind']=='ad_corner':
                            side,level=t['role'].split('_');accepted=role in [level+'_edge',side+'_vertical']
                    if accepted:run.append(m['UV'])
                    else:finish()
                finish()
    features=[];queries=[]
    for role,runs in paths.items():
        if 'vertical'in role:
            side=role.split('_')[0];points=[p for run in runs for p in run]
            points.extend(corners[k]for k in [side+'_top',side+'_bottom']if k in corners)
            if len(points)>=2:
                p=np.unique(np.asarray(points),axis=0);p=p[np.argsort(p[:,1])];runs=[p]
        if not runs:continue
        fid=prefix+role;segments=[]
        for ri,nav in enumerate(runs):
            path=path_stations(nav,12)
            if len(path)<2 or np.linalg.norm(path[-1]-path[0])<8:continue
            direction={'top_edge':[0,1],'bottom_edge':[0,-1],'left_vertical':[1,0],'right_vertical':[-1,0]}[role]
            weights=[-1.,.5,.5]if advertisement_id==6 else [1.,-.5,-.5]if advertisement_id in [3,5,7]else [.5,-1.,.5]
            rows,_=profile_read(raw,mask,path,weights,24,maximum_blur=18.,minimum_amplitude=10.,gradient_direction=direction,signal_mode='chroma')
            tangent=np.roll(path,-1,axis=0)-np.roll(path,1,axis=0);tangent[0]=path[1]-path[0];tangent[-1]=path[-1]-path[-2]
            normal=np.column_stack((-tangent[:,1],tangent[:,0]))/np.maximum(np.linalg.norm(tangent,axis=1)[:,None],.01)
            for qi,(nav_uv,row,n)in enumerate(zip(path,rows,normal)):
                q=dict(feature_id=fid,source_visible_navigation_run=ri,native_query_index=qi,native_pts=source['native_pts'],navigation_uv=nav_uv.tolist(),profile_normal=n.tolist(),profile_radius_px=24,actually_measured=row is not None,independent_RGB_review_complete=False)
                if row is not None:q.update(uv=row[0].tolist(),transition_iqr_px=row[1],native_chroma_contrast=row[2],original_native_profile_sigma_support_px=row[1]/1.349)
                queries.append(q)
            segments.extend(segment_records(rows,spacing=200,straight='vertical'in role,smooth_curve='vertical'not in role,target_id=fid))
        if segments:features.append(dict(id=fid,target_id=fid,physical_surface='board',segments=segments,allowed_for_fit=False,current_source_only_candidate=True))
    by={f['id']:f for f in features};receipts=[]
    for role,old in corners.items():
        side,level=role.split('_');h=by.get(prefix+level+'_edge');v=by.get(prefix+side+'_vertical')
        if h is None or v is None:
            receipts.append(dict(role=role,old_UV=old.tolist(),admitted_to_candidate=False,reason='Actual adjacent source side not measured'));continue
        p=_same_source_intersection(np.vstack([s['uv']for s in h['segments']]),np.vstack([s['uv']for s in v['segments']]))
        x,y=np.rint(p).astype(int);ok=4<=x<1916 and 4<=y<1076 and not mask[y-3:y+4,x-3:x+4].any()
        distance,s,j=min([(float(np.linalg.norm(np.asarray(s['uv'][j])-p)),s,j)for s in h['segments']for j in [0,len(s['uv'])-1]],key=lambda z:z[0]);ok=bool(ok and distance<=24)
        if ok:
            s['uv'][j]=p.tolist();s['point_bindings'][j]=dict(kind='corner',physical_target=prefix+role,required_residual='point_to_corresponding_model_corner_2d',measurement_origin='current_source_measured_adjacent_edge_intersection');s.setdefault('corner_vertex_indices',{})[role]=j
        receipts.append(dict(role=role,UV=p.tolist(),old_UV=old.tolist(),admitted_to_candidate=ok,nearest_actual_endpoint_distance=distance))
    for f in features:
        used=np.vstack([s.get('source_native_readout_uv',s['uv'])for s in f['segments']])
        for q in queries:
            if q['feature_id']==f['id']and q['actually_measured']:
                q['consumed_by_public_segment_estimator']=bool(np.min(np.linalg.norm(used-q['uv'],axis=1))<.001)
    return retain_visible_means(dict(source=deepcopy(source),features=features,queries=queries,corner_receipts=receipts,production_installed=False,independent_RGB_review_complete=False))

def retain_visible_means(native):
    """Same final source-mean cleanup as extract_frame, with explicit receipts.

    Raw native queries remain archived. This only rejects automatic derived
    means outside the image/current mask; covariance is rebuilt afterward.
    """
    native=deepcopy(native);mask=cv2.imread(native['source']['mask_path'],0)
    mask=cv2.dilate((mask>0).astype(np.uint8),np.ones((5,5),np.uint8))>0
    mask[:120,:570]=True;mask[945:,1660:]=True
    rejected=[];features=[]
    for f in native['features']:
        cleaned=[]
        for si,s in enumerate(f['segments']):
            p=np.asarray(s['uv']);inside=(p[:,0]>=0)&(p[:,0]<=1919)&(p[:,1]>=0)&(p[:,1]<=1079)
            xy=np.rint(np.clip(p,[0,0],[1919,1079])).astype(int);masked=mask[xy[:,1],xy[:,0]];good=inside&~masked;keep=np.flatnonzero(good).tolist()
            final=good if len(keep)>=2 else np.zeros(len(p),bool)
            for j in np.flatnonzero(~final):
                rejected.append(dict(source_feature_id=f['id'],candidate_segment_index=si,candidate_vertex_index=int(j),UV=p[j].tolist(),reason='outside_actual_native_image'if not inside[j]else 'current_source_person_or_broadcast_mask'if masked[j]else 'remaining_visible_run_has_fewer_than_two_representatives'))
            if len(keep)<2:continue
            if not good.all():
                for key in ['uv','point_bindings','transition_iqr_px','localization_sigma_px']:
                    if key in s:s[key]=[s[key][j]for j in keep]
                if 'corner_vertex_indices'in s:s['corner_vertex_indices']={k:keep.index(v)for k,v in s['corner_vertex_indices'].items()if v in keep}
                s.update(vertex_count=len(keep),closed=False)
            cleaned.append(s)
        f['segments']=cleaned
        if cleaned:features.append(f)
    native['features']=features;native['final_mean_visibility_rejections']=rejected
    present={pb['physical_target']for f in features for seg in f['segments']for pb in seg['point_bindings']if pb['kind']=='corner'}
    for corner in native['corner_receipts']:corner['survived_final_source_visibility_guard']=any(target.endswith('.'+corner['role'])for target in present)
    for q in native['queries']:
        if not q['actually_measured']:continue
        f=next((f for f in features if f['id']==q['feature_id']),None)
        if f:
            used=np.vstack([seg.get('source_native_readout_uv',seg['uv'])for seg in f['segments']]);q['consumed_by_public_segment_estimator']=bool(np.min(np.linalg.norm(used-q['uv'],axis=1))<.001)
        else:q['consumed_by_public_segment_estimator']=False
    return native

def compile_joint(native,candidate_packet_ref,registry):
    """Call the same 201+ public joint source mean/D/covariance/W preparation."""
    rows=[]
    for feature in native['features']:
        actual=[q for q in native['queries']if q['feature_id']==feature['id']and q['actually_measured']]
        for si,segment in enumerate(feature['segments']):
            for j,p in enumerate(segment['uv']):
                nearest=min(actual,key=lambda q:np.linalg.norm(np.asarray(q['uv'])-p))
                # A measured profile support initializes the required field; the public
                # complete joint propagation replaces it with the final D*L support.
                sigma=nearest['original_native_profile_sigma_support_px']
                target=registry[segment['point_bindings'][j]['physical_target']]
                rows.append(dict(model_target=target,source_point_ids=[f'{feature["id"]}/candidate_segment/{si}/vertex/{j}'],sigma=np.asarray([sigma]),uv=np.asarray([p]),source_feature_id=feature['id'],source_segment=segment,source_feature=feature))
    if not rows:raise ValueError('No current material candidate')
    frame=dict(packet=candidate_packet_ref,native_pts=native['source']['native_pts'],source_image=reference(native['source']['image_path']),mask=reference(native['source']['mask_path']))
    import calibration.program_tracking.first_shot_source_evidence as module
    original=module.latent_whitening_operator;capture=[]
    def record(ids,sensitivity,*args,**kwargs):
        op=original(ids,sensitivity,*args,**kwargs);capture.append((np.asarray(sensitivity),np.asarray(kwargs['explicit_noise_factor']),np.asarray(kwargs['physical_basis']),np.asarray(op.matrix)));return op
    module.latent_whitening_operator=record
    try:evidence=prepare_frame_evidence(rows,frame)
    finally:module.latent_whitening_operator=original
    if len(capture)!=1:raise ValueError('Exactly one shared advertisement family required')
    D,L,Q,W=capture[0];F=Q.T@D@L;C=F@F.T;error=float(np.max(np.abs(W@C@W.T-np.eye(len(W)))))
    if not np.isfinite(error)or error>1e-7:raise ValueError('Actual joint W covariance QA failed')
    qa=dict(whitened_covariance_max_abs_error=error,D_shape=list(D.shape),L_shape=list(L.shape),Q_shape=list(Q.shape),W_shape=list(W.shape),all_finite=all(np.isfinite(a).all()for a in [D,L,Q,W]),public_joint_call_count=1,conditional_D_is_same_existing_201_recipe=True)
    return rows,evidence,dict(D=D,L=L,Q=Q,W=W,C=C),qa
