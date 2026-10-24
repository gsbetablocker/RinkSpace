"""Lock a source-confirmed seed to its physical finite path, before profiling.

Path identity comes from original-image review. The camera may initialize a
station only inside that path; a source segment number is never a path proof.
This adaptor is independent of archived packet producers and changes no pixels.
"""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import hashlib,json
import av,cv2,numpy as np
from .compact_identity_revision import _array
from .compact_identity_withhold_v2 import _source_identities
from .dense_joint_proofs import _ref
from .io_utils import write_json
from .source_seed_correspondence_v1 import correlation
from .native_print_track_readout import batch_interfaces,clear_profile
from .source_worker import _channels


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _check_ref(ref,cache):
    path=Path(ref['file'])
    if str(path) not in cache:
        cache[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    if cache[str(path)]!=ref['sha256']:
        raise ValueError('Changed original source evidence: '+str(path))
    return path


def load_seed_paths(*,source_config,path_review):
    """Read an explicit original-image seed/entity/path authority, never infer it."""
    cfg=_read(source_config);review=_read(path_review);cache={};result={}
    if (review.get('schema')!='rinkfit.source_seed_physical_path_review.v1'
            or review.get('source_model_projection_used') is not False
            or review.get('source_original_images_reviewed') is not True
            or review.get('source_segment_integer_is_path_authority') is not False
            or review.get('source_video_sha256')!=cfg['source_video_sha256']):
        raise ValueError('Explicit original-image physical path review required')
    for entry in review['rows']:
        si=int(entry['seed_index']);seed=cfg['seeds'][si];anchor=cfg['source_anchors'][seed['source_index']]
        if (si in result or entry.get('physical_entity_path_source_confirmed') is not True
                or entry['label']!=seed['label'] or int(entry['native_pts'])!=int(seed['native_pts'])
                or entry['source_uv']!=seed['uv'] or not entry.get('physical_source_basis')):
            raise ValueError('Physical path authority does not match immutable original seed')
        if entry['source_annotation']!=anchor['source']:
            raise ValueError('Seed authority changed its original annotation')
        source=_read(_check_ref(entry['source_annotation'],cache));feature=source['features'][seed['feature_index']]
        segment=feature['segments'][seed['source_segment']]
        if (feature['target_id']!=seed['label']
                or segment[seed['source_field']][seed['source_point_index']]!=seed['uv']):
            raise ValueError('Seed does not identify its literal original pixel')
        if str(Path(entry['source_image']['file']))!=str(Path(anchor['image'])):
            raise ValueError('Seed authority changed its original image')
        _check_ref(entry['source_image'],cache)
        atlas=_read(_check_ref(entry['canonical_atlas'],cache));matches=[p for p in atlas['paths']
            if p['label']==entry['label'] and int(p['path'])==int(entry['path'])]
        if len(matches)!=1 or atlas.get('source_model_projection_used') is not False:
            raise ValueError('Reviewed path is not a literal original-image finite contour')
        for ref in entry.get('source_review_evidence',[]):_check_ref(ref,cache)
        result[si]=dict(entry,authority=_ref(path_review))
    return result


def constrain_rows(*,rows,seed_indices,source_config,path_review):
    """Prepare known rows for the existing profiler with allowed_paths=[known].

    Unknown identities raise before any initializer can run. Caller retains its
    original rows; this function neither removes a row nor profiles a station.
    """
    if len(rows)!=len(seed_indices):raise ValueError('Seed/row layout differs')
    known=load_seed_paths(source_config=source_config,path_review=path_review)
    unknown=sorted({int(i)for i in seed_indices if int(i)not in known})
    if unknown:raise ValueError('Original-image path authority missing for seeds '+str(unknown))
    result=deepcopy(rows)
    for row,si in zip(result,seed_indices):
        entry=known[int(si)]
        if row['material_label']!=entry['label']:raise ValueError('Known seed material changed')
        path=int(entry['path'])
        if path not in row['allowed_paths']:raise ValueError('Known source path absent from material')
        row['allowed_paths']=[path];row['path_index']=path
    return result


def _source_registration(anchor,current,allowed,heldout):
    ai={int(s):i for i,s in enumerate(anchor['seed_index'])}
    ci={int(s):i for i,s in enumerate(current['seed_index'])}
    others=sorted((set(ai)&set(ci)&allowed)-{heldout})
    if len(others)<8:return None
    src=np.asarray([anchor['uv'][ai[s]]for s in others],float)
    dst=np.asarray([current['uv'][ci[s]]for s in others],float)
    fits=[]
    affine,inside=cv2.estimateAffine2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=3.,maxIters=5000,confidence=.999)
    if affine is not None:fits.append(('affine',np.r_[affine,[[0,0,1]]],inside))
    homography,inside=cv2.findHomography(src,dst,cv2.RANSAC,3.,maxIters=5000,confidence=.999)
    if homography is not None:fits.append(('homography',homography,inside))
    result=[]
    for kind,H,inside in fits:
        good=inside.ravel().astype(bool)
        if int(good.sum())<8:continue
        points=src[good].astype(np.float32);hull=cv2.convexHull(points).reshape(-1,2)
        den=np.c_[hull,np.ones(len(hull))]@H[2]
        if (np.ptp(points,axis=0)[0]<40 or np.ptp(points,axis=0)[1]<10
                or cv2.contourArea(hull)<150 or min(den)*max(den)<=0
                or np.linalg.det(H)/den.mean()**3<=0):continue
        err=np.linalg.norm(cv2.perspectiveTransform(src[good][None],H)[0]-dst[good],axis=1)
        if np.quantile(err,.95)>3.:continue
        result.append(dict(kind=kind,H=H,inliers=int(good.sum()),registration_p95_px=float(np.quantile(err,.95)),
            source_convex_hull=hull,other_source_seed_indices=np.asarray(others)[good].tolist()))
    return result


def review_existing(*,packet,model,source_config,source_evidence,row_start,path_review,old_cost_report,output):
    """Audit actual eligible rows; emit same-material path corrections only.

    Per-current evidence uses other immutable source tracks, held-out pixel
    error, inlier footprint, original RGB patch correlation, signed profile
    replay, original mask and unchanged limits. It never uses a fitted camera.
    """
    out=Path(output)
    if out.exists():raise FileExistsError('Immutable path review already exists')
    out.mkdir(parents=True)
    known=load_seed_paths(source_config=source_config,path_review=path_review)
    cfg=_read(source_config);pk=_read(packet);m=_read(model);e=_read(source_evidence)
    arrays={k:_array(pk['arrays'][k])for k in ['uv','native_pts','material_index','path','admission']}
    columns={k:_array(e['columns'][k])for k in ['seed_index','uv','normal','profile','flow_fb']}
    local=np.flatnonzero(np.isin(columns['seed_index'],list(known)));rows=local+int(row_start)
    ids,_=_source_identities(Path(packet),rows.tolist());frames={int(f['native_pts']):f for f in _read(e['frames']['file'])}
    candidate_frames={int(f['native_pts']):f for f in _read(Path(cfg['output'])/'frame_index.json')}
    anchors={};allowed={};cache={};by_pts={};audits=[];accepted=[];unresolved=[];correct=0
    for si,proof in known.items():
        ap=int(proof['native_pts']);fr=candidate_frames[ap];_check_ref(dict(file=fr['file'],sha256=fr['sha256']),cache)
        data=np.load(fr['file']);anchors[si]={k:data[k]for k in data.files}
        image=cv2.imread(proof['source_image']['file']);expected=fr['source_BGR_sha256']
        if hashlib.sha256(image.tobytes()).hexdigest()!=expected:raise ValueError('Original anchor pixels differ')
        anchors[si]['gray']=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        allowed[si]={j for j,s in enumerate(cfg['seeds'])if s['native_pts']==ap and s['label']==proof['label']}
    for li,i0 in zip(local,rows):
        i=int(i0);si=int(columns['seed_index'][li]);pt=int(arrays['native_pts'][i]);sid=ids[i]
        if sid!=e['source_dataset_id']+'/s'+sid.split('/s',1)[1].split('/')[0]+'/'+str(pt)+'/seed'+str(si):
            raise ValueError('Immutable source seed/row identity differs')
        if not np.array_equal(arrays['uv'][i],columns['uv'][li]):raise ValueError('Original source UV changed')
        if arrays['admission'][i]!=1:continue
        by_pts.setdefault(pt,[]).append((int(li),i,si))
    from calibration.program_tracking.extract_first_shot_ice import current_broadcast_exclusion
    seen=set()
    with av.open(cfg['source_video'])as con:
        stream=con.streams.video[0];stream.thread_count=1
        if str(stream.time_base)!='1/12800':raise ValueError('Native source clock differs')
        if by_pts:con.seek(min(by_pts),stream=stream,backward=True)
        for frame in con.decode(stream):
            pt=int(frame.pts)
            if not by_pts or pt>max(by_pts):break
            if pt not in by_pts:continue
            seen.add(pt);image=frame.to_ndarray(format='bgr24');fr=candidate_frames[pt]
            if hashlib.sha256(image.tobytes()).hexdigest()!=fr['source_BGR_sha256']:raise ValueError('Current original RGB changed')
            _check_ref(dict(file=fr['file'],sha256=fr['sha256']),cache);current=np.load(fr['file'])
            mask=np.zeros(image.shape[:2],np.uint8)
            for ref in [fr['person_mask'],fr.get('unknown_mask')]:
                if ref:mask=cv2.bitwise_or(mask,cv2.imread(str(_check_ref(ref,cache)),0))
            graphics,_=current_broadcast_exclusion(image);mask[graphics]=255
            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);channels=_channels(image[:,:,::-1])
            for li,i,si in by_pts[pt]:
                uv=arrays['uv'][i];proof=known[si];anchor=anchors[si];ai=int(np.flatnonzero(anchor['seed_index']==si)[0]);aq=anchor['uv'][ai]
                profile=columns['profile'][li];normal=columns['normal'][li];normal=normal/np.linalg.norm(normal)
                measured=batch_interfaces((uv-profile[2]*normal)[None],normal[None],channels)
                z=dict(raw_row=i,source_row_id=ids[i],seed_index=si,native_pts=pt,uv=uv.tolist(),old_path=int(arrays['path'][i]),
                    confirmed_source_path=int(proof['path']),original_profile=profile.tolist(),source_flow_FB_px=float(columns['flow_fb'][li]),
                    source_BGR_sha256=fr['source_BGR_sha256'],source_original_record_ref=dict(file=fr['file'],sha256=fr['sha256']),
                    source_pixels_changed=False,source_model_projection_used=False)
                reason=None
                if m['materials'][int(arrays['material_index'][i])]['label']!=proof['label']:reason='original_named_material_changed'
                elif (not measured['valid'][0]or not np.allclose(measured['uv'][0],uv,atol=1e-9,rtol=0)
                        or not np.allclose(measured['profile'][0],profile,atol=1e-7,rtol=0)):reason='current_signed_RGB_replay_differs'
                elif profile[3]!=anchor['profile'][ai,3]or profile[4]!=anchor['profile'][ai,4]:reason='original_signed_interface_changed'
                elif columns['flow_fb'][li]>.8:reason='source_tracking_FB_unconfirmed'
                elif not clear_profile(uv[None],normal[None],mask)[0]:reason='original_current_mask_excluded'
                if reason is None:
                    supports=[]
                    for reg in _source_registration(anchor,current,allowed[si],si)or[]:
                        H=reg['H'];mapped=cv2.perspectiveTransform(aq[None,None],H)[0,0];inverse=cv2.perspectiveTransform(uv[None,None],np.linalg.inv(H))[0,0]
                        hull=reg['source_convex_hull'];dist=cv2.pointPolygonTest(hull,tuple(map(float,aq)),True)
                        currentdist=cv2.pointPolygonTest(hull,tuple(map(float,inverse)),True)
                        err=float(np.linalg.norm(mapped-uv));corr=correlation(anchor['gray'],gray,H,aq)
                        if min(dist,currentdist)<0 or err>3. or not np.isfinite(corr)or corr<.65:continue
                        supports.append(dict(kind=reg['kind'],H=H.tolist(),inliers=reg['inliers'],registration_p95_px=reg['registration_p95_px'],
                            other_source_seed_indices=reg['other_source_seed_indices'],source_convex_hull=hull.tolist(),
                            original_seed_inside_hull_px=dist,current_foot_inside_hull_px=currentdist,heldout_source_pixel_error_px=err,local_source_pixel_correlation=corr))
                    if not supports:reason='heldout_source_registration_or_RGB_unconfirmed'
                    else:z['source_registration']=max(supports,key=lambda q:q['local_source_pixel_correlation'])
                if reason:
                    z.update(reason=reason,original_row_retained=True);unresolved.append(z)
                else:
                    z.update(status='source_confirmed_same_seed_same_physical_path',source_signed_RGB_replay_exact=True)
                    if z['old_path']==int(proof['path']):correct+=1
                    else:
                        old=int(arrays['material_index'][i]);accepted.append(dict(z,old_label=proof['label'],new_label=proof['label'],old_material_index=old,
                            new_material_index=old,new_path=int(proof['path']),new_station=.5,source_station_is_numeric_initializer=True,
                            identity_basis='reviewed_original_seed_physical_path_and_heldout_current_source_RGB_interface'))
                audits.append(z)
    if seen!=set(by_pts):raise ValueError('Current original image census incomplete')
    refs=[_ref(p)for p in [__file__,packet,source_config,source_evidence,path_review,old_cost_report]]
    doc=dict(schema='rinkfit.native_print_identity_rebind_review.v1',source_model_projection_used=False,source_pixels_changed=False,
        source_video_sha256=cfg['source_video_sha256'],native_time_base='1/12800',parent_packet=_ref(packet),evidence=refs,
        rows=accepted,unresolved=unresolved,already_correct_rows=correct,actual_eligible_source_rows=len(audits),source_seeds=len(known),
        original_cost_report=_ref(old_cost_report),original_E_W_and_source_UV_ID_PTS_preserved=True,original_rows_deleted=0,
        unconfirmed_source_rows_retained=True,whole_panel_or_3D_projection_used=False,
        qualification='Original-image reviewed seed/path authority plus held-out other-source-seed 2D registration, inlier footprint, .65 local RGB correlation, 3px held-out/p95 and exact original signed profile replay. No fitted camera or source-segment-to-path shortcut.')
    write_json(out/'review.json',doc);write_json(out/'source_rows.json',audits)
    return dict(review=_ref(out/'review.json'),eligible_rows=len(audits),rebound_rows=len(accepted),already_correct_rows=correct,
        unresolved_rows=len(unresolved),unresolved_reasons=dict(Counter(z['reason']for z in unresolved)))
