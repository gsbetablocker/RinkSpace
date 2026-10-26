"""Source-only entity verification for the S01 native advertisement replacement."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ad_replacement_s01 as base
import collections, functools, hashlib, json, time
import cv2
import numpy as np
from scipy.spatial import cKDTree

VERSION='source_entity_rgb_masks_v2'
OUT=base.OUT
read,write,sha=base.read,base.write,base.sha


def prior_evidence(data,atlases):
    """Read old decisions as explicit hypotheses, never as new admissions."""
    aa=base.assets();hints={};withdraw={};retired={};registrations=collections.defaultdict(list)
    sources=[]
    def record_source(p):
        sources.append(dict(path=str(p),sha256=sha(p)))
        return read(p)
    repair=base.SOURCE/'full_original/data_current/ad23_repair'
    for n,name,aid in [(2,'ad2_s01',1),(3,'ad3_source',70)]:
        p=repair/f'{name}_registrations_with_tracks.json'
        for frame in record_source(p):
            for r in frame['candidates']:
                if r.get('H') is not None:
                    registrations[(n,int(frame['native_pts']))].append(dict(atlas_id=aid,H=r['H'],p95=r.get('registration_p95_px',0),inliers=r.get('inliers',0),method='historical_source_image_hypothesis_rechecked',source=str(p),source_kind=r.get('kind')))
        p=repair/f'ad{n}_unique_shape_verified.json'
        for r in record_source(p)['confirmed']:
            row=int(r['row']);label=r['new_label']
            if n==3 and label.startswith('hd_ad3/'):
                target='Ad3.print.official_svg.'+label.removeprefix('hd_ad3/')
                if target in aa[n]['paths']:hints[row]=target
                else:retired[row]=dict(reason='latest_identified_DAIKICHI_source_entity_omitted_by_current_native',entity=label,source=str(p))
            else:
                ids=aa[n]['aliases'].get(label,[])
                if len(ids)==1:hints[row]=ids[0]
    withdraw[28179]=dict(reason='explicit_actual_source_neck_shoulder_occlusion_review',source=str(repair/'ad2_unique_shape_verified.json'))
    p=base.SOURCE/'ads_original/ad6_candidate/point_withdrawal_delta.json'
    for r in record_source(p)['observations']:
        row=int(r['raw_row'])
        if str(data['ids'][row])!=r['observation_id'] or not np.array_equal(data['uv'][row],r['immutable_uv']):raise ValueError('Explicit Ad6 source withdrawal identity mismatch')
        withdraw[row]=dict(reason='verified_white_stroke_interior_not_material_edge',source=str(p),evidence=r['image_evidence'])
    # Latest post-export Ad4/5 physical-entity corrections.
    for filename in ['ad45_delta.json','image_correspondence12_increment.json']:
        p=base.SOURCE/'full_original/ad45_identity'/filename
        d=record_source(p)
        for r in d.get('records',[])+d.get('keep',[]):
            row=int(r['row']);n=int(data['ad'][row]);label=r.get('entity','')
            ids=aa[n]['aliases'].get(label,[])
            if len(ids)==1:hints[row]=ids[0]
            elif 'central' in label or 'skater' in label:
                retired[row]=dict(reason='identified_central_skater_graphic_omitted_by_current_native',entity=label,source=str(p))
        for r in d.get('pending_records',[]):
            withdraw[int(r['row'])]=dict(reason='explicit_source_motion_ghost_or_white_background_no_unique_edge',source=str(p))
    groups={g['name']:g for g in read(base.SOURCE/'full_original/ad45_identity/source_groups12.json')}
    # Image charts from the latest independently stored source registration pass.
    for filename in ['image_correspondence11.json','image_correspondence12_increment.json']:
        p=base.SOURCE/'full_original/ad45_identity'/filename
        if not p.exists():continue
        d=record_source(p)
        for r in d.get('source_registrations',[]):
            reg=r.get('registration');g=groups.get(r['group'])
            if not reg or not g:continue
            candidates=[a for a in atlases if Path(a['image'])==Path(g['source_image']) and a['ad']==g['ad']]
            if not candidates:continue
            a=candidates[0];allowed=[]
            for pp in g['paths']:
                ids=aa[g['ad']]['aliases'].get(pp['label'],[])
                if len(ids)==1:allowed.append(ids[0])
            if not allowed:continue
            registrations[(g['ad'],int(r['native_pts']))].append(dict(atlas_id=a['id'],H=reg['H'],p95=reg['registration_p95_px'],inliers=reg['inliers'],method='historical_source_image_hypothesis_rechecked',source=str(p),allowed_path_ids=sorted(set(allowed))))
    lookup={s:i for i,s in enumerate(data['ids'])}
    for n in (7,8,9,10):
        a=aa[n];store=a['asset']['observation_store'];p=(a['path'].parent/store['source_correspondence']['path']).resolve();d=record_source(p)
        if n==7:
            for r in d['rows']:
                sid=f"raw:12:s01_original:{r['native_pts']}:{r['source_point_id']}"
                if sid in lookup and len(r['candidate_feature_ids'])==1:hints[lookup[sid]]=r['candidate_feature_ids'][0]
        else:
            bank=np.load((a['path'].parent/store['arrays']['path']).resolve(),allow_pickle=False)
            for sid,target in zip(bank['stable_id'],bank['current_feature_id']):
                row=lookup.get(str(sid))
                if row is None:continue
                if target:hints[row]=str(target)
                elif n==9:retired[row]=dict(reason='source_ornament_script_or_fine_detail_omitted_by_current_native',source=str(p))
            if n==8:
                for r in d['remain_excluded']:
                    withdraw[int(r['full_row'])]=dict(reason='false_W_Y_material_bridge',source=str(p))
    # Only explicitly reviewed exposures are vetoed; no shot-wide propagation.
    p=base.SOURCE/'ads_original/clarity_ad7_11/clarity_facts.json'
    for ad in record_source(p)['advertisements']:
        n=int(ad['ad'][2:])
        for r in ad['review']:
            if r['qualification']=='blurred':
                for row in np.flatnonzero((data['ad']==n)&(data['pts']==r['native_pts'])):
                    withdraw[int(row)]=dict(reason='explicit_source_review_motion_blur_fine_interface_unresolved',source=str(p),source_pts=r['native_pts'])
    p=base.SOURCE/'ads_original/ad11_candidate/identity_facts.json'
    d=record_source(p)
    for r in d['visible_evidence']:
        if 'strongly motion blurred' in r['clarity']:
            for row in np.flatnonzero((data['ad']==11)&np.isin(data['pts'],r['pts'])):
                withdraw[int(row)]=dict(reason='explicit_source_review_motion_blur_thin_ring_unresolved',source=str(p))
    visibility=read(OUT/'visibility.json')
    for row in visibility['inside_person_mask_rows']:
        withdraw[row]=dict(reason='inside_current_native_SAM2_person_exclusion',source=str(OUT/'visibility.json'))
    write(OUT/'source_evidence_index.json',dict(sources=sources,hints={str(k):v for k,v in hints.items()},withdrawals={str(k):v for k,v in withdraw.items()},retired={str(k):v for k,v in retired.items()},historical_hypotheses_are_not_admissions=True))
    return hints,withdraw,retired,registrations


def image_candidates(ref,kp,des,matcher):
    if ref['descriptors'] is None or des is None or len(des)<5:return []
    pairs=matcher.knnMatch(ref['descriptors'],des,k=2)
    pairs=[m for pair in pairs if len(pair)==2 for m,n in [pair] if m.distance<.77*n.distance]
    # Duplicate SIFT orientations must not count as independent geometric support.
    unique={tuple(np.round(np.r_[ref['keypoints'][m.queryIdx],kp[m.trainIdx]],2)):m for m in pairs}
    pairs=list(unique.values())
    if len(pairs)<5:return []
    src=ref['keypoints'][[m.queryIdx for m in pairs]];dst=kp[[m.trainIdx for m in pairs]]
    results=[]
    for typ in ['affine','homography']:
        if typ=='affine':
            A,mask=cv2.estimateAffine2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=2.,maxIters=3000,confidence=.999,refineIters=15)
            H=np.vstack([A,[0.,0.,1.]]) if A is not None else None
        else:
            if len(pairs)<7:continue
            H,mask=cv2.findHomography(src,dst,cv2.RANSAC,2.,maxIters=3000,confidence=.999)
        if H is None or mask is None:continue
        ok=mask.ravel().astype(bool)
        if ok.sum()<5 or ok.mean()<.2:continue
        err=np.linalg.norm(cv2.perspectiveTransform(src[ok][None],H)[0]-dst[ok],axis=1)
        p95=float(np.percentile(err,95))
        if p95>2 or cv2.contourArea(cv2.convexHull(dst[ok].astype('float32')))<30:continue
        if np.linalg.det(H[:2,:2])<=0:continue
        results.append(dict(H=H.tolist(),p95=p95,inliers=int(ok.sum()),matches=len(pairs),method='independent_actual_source_SIFT_'+typ,source_inlier_hull=cv2.convexHull(src[ok].astype('float32'))[:,0,:].tolist()))
    return results


def patch_ncc(ref,current,H,uv,radius=5,shift=None):
    off=np.mgrid[-radius:radius+1,-radius:radius+1].reshape(2,-1).T[:,::-1]
    target=uv[:,None,:]+off[None]
    query=target if shift is None else target-shift[:,None,:]
    source=cv2.perspectiveTransform(query.reshape(1,-1,2),np.linalg.inv(H))[0].reshape(query.shape)
    a=base.samples(ref,source);b=base.samples(current,target)
    a-=a.mean(1,keepdims=True);b-=b.mean(1,keepdims=True)
    den=np.sqrt((a*a).sum(1)*(b*b).sum(1))
    return np.divide((a*b).sum(1),den,out=np.full(len(uv),-1.),where=den>1)


def local_evidence(ref,reg,gray,uv):
    H=np.asarray(reg['H'],float)
    proj=cv2.perspectiveTransform(ref['points'][None],H)[0]
    finite=np.isfinite(proj).all(1)&(np.abs(proj).max(1)<10000)
    lab=ref['labels'];allowed=reg.get('allowed_path_ids')
    if allowed:
        finite&=np.isin(lab,[j for j,p in enumerate(ref['metadata']['paths']) if p['path_id'] in allowed])
    inds=np.flatnonzero(finite)
    if len(inds)<4:return [],{}
    dist,idx=cKDTree(proj[finite]).query(uv,k=min(48,len(inds)))
    if dist.ndim==1:return [],{}
    nearest=inds[idx[:,0]];labels=np.array([p['path_id'] for p in ref['metadata']['paths']]);bestlabel=labels[lab[nearest]]
    others=np.min(np.where(labels[lab[inds[idx]]]!=bestlabel[:,None],dist,np.inf),axis=1)
    rr=np.flatnonzero((dist[:,0]<=6)&((others-dist[:,0])>=.6))
    if not len(rr):return [],{}
    xy=uv[rr];ni=nearest[rr];foot=proj[ni]
    # Local image flow refines only the source-to-source chart. Observation UV
    # and the native geometry are not changed.
    lo=np.maximum(np.floor(np.vstack([xy,foot]).min(0)-22).astype(int),[0,0])
    hi=np.minimum(np.ceil(np.vstack([xy,foot]).max(0)+23).astype(int),[gray.shape[1],gray.shape[0]])
    if np.any(hi-lo<20):return [],{}
    shift=np.eye(3);shift[:2,2]=-lo
    warped=cv2.warpPerspective(ref['gray'],shift@H,tuple(hi-lo),flags=cv2.INTER_LINEAR)
    actual=gray[lo[1]:hi[1],lo[0]:hi[0]]
    f0=(foot-lo).astype('float32').reshape(-1,1,2)
    f1,status,_=cv2.calcOpticalFlowPyrLK(warped,actual,f0,None,winSize=(21,21),maxLevel=2,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    if f1 is None:return [],{}
    f2,status2,_=cv2.calcOpticalFlowPyrLK(actual,warped,f1,None,winSize=(21,21),maxLevel=2,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    if f2 is None:return [],{}
    flow=(f1-f0)[:,0];fb=np.linalg.norm((f2-f0)[:,0],axis=1)
    validflow=status.ravel().astype(bool)&status2.ravel().astype(bool)&(fb<=.75)&(np.linalg.norm(flow,axis=1)<=5)
    flow[~validflow]=0
    refined=foot+flow;edge_distance=np.linalg.norm(xy-refined,axis=1)
    small=patch_ncc(ref['gray'],gray,H,xy,5,flow)
    context=patch_ncc(ref['gray'],gray,H,xy,15,flow)
    # Local normal from the actual source contour, never from current native geometry.
    left=ni.copy();right=ni.copy()
    for j in np.unique(lab[ni]):
        rrj=np.flatnonzero(lab[ni]==j);members=np.flatnonzero(lab==j);start=int(members[0]);stop=int(members[-1])
        closed=np.linalg.norm(ref['points'][start]-ref['points'][stop])<1e-6
        if closed:
            size=max(1,stop-start);left[rrj]=start+(ni[rrj]-start-2)%size;right[rrj]=start+(ni[rrj]-start+2)%size
        else:
            left[rrj]=np.maximum(ni[rrj]-2,start);right[rrj]=np.minimum(ni[rrj]+2,stop)
    tangent=proj[right]-proj[left];length=np.linalg.norm(tangent,axis=1)
    normal=np.c_[-tangent[:,1],tangent[:,0]]/np.maximum(length[:,None],1e-12)
    tt=np.linspace(-6,6,49)
    profile=base.samples(gray,xy[:,None,:]+tt[None,:,None]*normal[:,None,:])
    refxy=cv2.perspectiveTransform((xy-flow)[:,None,:],np.linalg.inv(H))[:,0]
    source_query=xy[:,None,:]-flow[:,None,:]+np.array([-2.,2.])[None,:,None]*normal[:,None,:]
    source_query=cv2.perspectiveTransform(source_query.reshape(1,-1,2),np.linalg.inv(H))[0].reshape(-1,2,2)
    ref_values=base.samples(ref['gray'],source_query);sign=np.sign(ref_values[:,1]-ref_values[:,0])
    deriv=np.diff(profile,axis=1)*sign[:,None]
    local=deriv[:,12:36];peak=12+np.argmax(local,axis=1)
    peak_offset=-6+(peak+.5)*.25
    # Half-height transition width measures whether a specific edge remains
    # locatable; the frame-wide camera residual is never consulted.
    widths=[];contrast=[]
    for k,j in enumerate(peak):
        pk=deriv[k,j];l=j;r=j
        while l>0 and deriv[k,l-1]>=pk*.5:l-=1
        while r+1<deriv.shape[1] and deriv[k,r+1]>=pk*.5:r+=1
        widths.append((r-l+1)*.25);contrast.append(float(profile[k,min(j+7,48)]-profile[k,max(j-6,0)])*sign[k])
    widths=np.asarray(widths);contrast=np.asarray(contrast)
    results=[];fail={}
    for k,i in enumerate(rr):
        reason=None
        if small[k]<.70 or context[k]<.65:reason='current_glyph_context_not_confirmed'
        elif abs(peak_offset[k])>1.75 or edge_distance[k]>2.5 or contrast[k]<8:reason='original_pixel_not_on_verified_material_interface'
        elif widths[k]>3.0:reason='motion_spread_material_boundary_not_resolved'
        elif others[i]-dist[i,0]<max(1.,min(reg.get('p95',1),2.)):reason='neighboring_entity_boundary_ambiguous'
        if reason:
            fail[int(i)]=reason;continue
        results.append(dict(local_row=int(i),path_id=str(bestlabel[i]),atlas_id=int(ref['metadata']['id']),registration_method=reg['method'],
            local_correlation=float(small[k]),glyph_context_correlation=float(context[k]),source_reference_distance=float(dist[i,0]),
            refined_source_boundary_distance=float(edge_distance[k]),signed_material_contrast=float(contrast[k]),
            current_edge_peak_offset=float(peak_offset[k]),edge_transition_fwhm=float(widths[k]),
            local_flow_forward_backward=float(fb[k]) if validflow[k] else None,local_flow=flow[k].tolist(),
            source_curve_fraction=float(ref['fractions'][ni[k]]),native_geometry_used_for_identity=False))
    return results,fail


@functools.lru_cache(maxsize=16)
def anchor_keypoints(pt,ad,image,rows_tuple,mask_path):
    gray=base.gray_read(image);uv=np.asarray(rows_tuple).reshape(-1,2)
    mask=np.zeros_like(gray);lo=np.maximum(np.floor(uv.min(0)-40).astype(int),0);hi=np.minimum(np.ceil(uv.max(0)+41).astype(int),[gray.shape[1],gray.shape[0]])
    if np.all(hi>lo):mask[lo[1]:hi[1],lo[0]:hi[0]]=255
    mask[cv2.imread(mask_path,0)>0]=0
    kp,des=cv2.SIFT_create(nfeatures=3000,contrastThreshold=.012).detectAndCompute(gray,mask)
    return dict(gray=gray,keypoints=np.asarray([k.pt for k in kp]),descriptors=des)


def verify(ads=None):
    cv2.setNumThreads(2);data=base.load();atlasdoc=read(OUT/'atlases.json');ats=atlasdoc['atlases']
    atlas_sha=sha(OUT/'atlases.json')
    hints,withdraw,retired,historical=prior_evidence(data,ats)
    visibility={r['pts']:r for r in read(OUT/'visibility.json')['frames']}
    sift=cv2.SIFT_create(nfeatures=3500,contrastThreshold=.012);matcher=cv2.BFMatcher(cv2.NORM_L2)
    refs={a['id']:base.prepare_reference(a,sift) for a in ats if ads is None or a['ad'] in ads}
    byad=collections.defaultdict(list)
    for a in ats:
        if a['id'] in refs:byad[a['ad']].append(a['id'])
    existing={};anchors=collections.defaultdict(list)
    for p in (OUT/'image_checks').glob('*.json'):
        n=int(p.stem.split('_')[0][2:])
        if n not in byad:continue
        d=read(p);pt=d['pts'];existing[(n,pt)]=d
        if d['registrations'] and len(d['confirmed'])>=3:anchors[n].append(pt)
    for n in anchors:anchors[n]=sorted(anchors[n])
    rows=np.flatnonzero(np.isin(data['ad'],list(byad)))
    points_by_frame={pt:rows[data['pts'][rows]==pt] for pt in sorted(set(map(int,data['pts'][rows])))}
    start=time.monotonic();total=collections.Counter()
    for frame_number,(pt,frame_rows) in enumerate(points_by_frame.items()):
        vis=visibility[pt];bgr=cv2.imread(vis['image'])
        if bgr is None:raise ValueError('Actual source frame missing')
        if vis.get('source_BGR_sha256') and hashlib.sha256(bgr.tobytes()).hexdigest()!=vis['source_BGR_sha256']:raise ValueError('Source frame pixels do not match PTS authority')
        gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY);human=cv2.imread(vis['mask_path'],0)
        for n in sorted(set(map(int,data['ad'][frame_rows]))):
            sub=frame_rows[data['ad'][frame_rows]==n];dest=OUT/'verified_entities'/f'ad{n}_{pt}.json'
            if dest.exists():
                prior=read(dest)
                if prior['version']!=VERSION or prior['atlas_sha256']!=atlas_sha:raise ValueError('Verification cache revision mismatch')
                total.update(prior['counts']);continue
            excluded={int(r):(withdraw.get(int(r)) or retired.get(int(r))) for r in sub if int(r) in withdraw or int(r) in retired}
            active=np.asarray([r for r in sub if int(r) not in excluded],int)
            if not len(active):
                votes={};regs=[];fail={};conflicts=[];confirmed=[]
            else:
                uv=data['uv'][active];mask=np.zeros_like(gray);lo=np.maximum(np.floor(uv.min(0)-40).astype(int),0);hi=np.minimum(np.ceil(uv.max(0)+41).astype(int),[1920,1080])
                if np.all(hi>lo):mask[lo[1]:hi[1],lo[0]:hi[0]]=255
                mask[human>0]=0
                kp,des=sift.detectAndCompute(gray,mask);kp=np.asarray([k.pt for k in kp])
                regs=[]
                for r in existing.get((n,pt),{}).get('registrations',[]):regs.append(dict(r,method='independent_first_pass_source_image_registration'))
                regs.extend(historical.get((n,pt),[]))
                for aid in byad[n]:
                    for r in image_candidates(refs[aid],kp,des,matcher):regs.append(dict(r,atlas_id=aid))
                # A fresh source-to-source registration to a nearby exposure
                # recovers cropped words; there is no temporal interpolation.
                nearby=sorted(anchors[n],key=lambda t:abs(t-pt))
                for apt in [x for x in nearby if 0<abs(x-pt)<=5120][:2]:
                    anc=existing[(n,apt)];ar=np.asarray(anc['rows'],int);v=visibility[apt]
                    ref=anchor_keypoints(apt,n,v['image'],tuple(data['uv'][ar].ravel()),v['mask_path'])
                    local=image_candidates(ref,kp,des,matcher)
                    for r in local[:1]:
                        for old in anc['registrations'][:3]:
                            H=np.asarray(r['H'])@np.asarray(old['H'])
                            regs.append(dict(atlas_id=old['atlas_id'],H=H.tolist(),p95=max(r['p95'],old['p95']),inliers=min(r['inliers'],old['inliers']),method='independent_adjacent_actual_exposure_registration',anchor_pts=apt))
                votes=collections.defaultdict(list);fail={}
                for ri,r in enumerate(regs):
                    ref=refs.get(r['atlas_id'])
                    if ref is None:continue
                    try:accepted,failed=local_evidence(ref,r,gray,uv)
                    except (cv2.error,np.linalg.LinAlgError):continue
                    for j,reason in failed.items():fail[int(active[j])]=reason
                    for e in accepted:
                        row=int(active[e.pop('local_row')]);e['registration_index']=ri
                        votes[row].append(e)
                confirmed=[];conflicts=[]
                for row,v in votes.items():
                    paths={x['path_id'] for x in v}
                    if len(paths)>1:
                        expected=hints.get(row);preferred=[x for x in v if x['path_id']==expected]
                        # Resolve only with both an independently named entity
                        # record and stronger current whole-glyph evidence.
                        other=max((x['glyph_context_correlation'] for x in v if x['path_id']!=expected),default=0)
                        if preferred and max(x['glyph_context_correlation'] for x in preferred)>=other+.10:v=preferred
                        else:
                            conflicts.append(dict(row=row,choices=sorted(paths)));continue
                    best=max(v,key=lambda x:x['glyph_context_correlation']+x['local_correlation'])
                    confirmed.append(dict(row=row,stable_id=str(data['ids'][row]),historical_entity_hint=hints.get(row),**best))
            passed={r['row'] for r in confirmed}
            conflict_rows={r['row'] for r in conflicts}
            decisions=[]
            for row in sub:
                row=int(row)
                if row in passed:continue
                ev=excluded.get(row)
                if ev is None:ev=dict(reason='conflicting_actual_entity_matches' if row in conflict_rows else fail.get(row,'no_verified_source_image_entity_registration'))
                decisions.append(dict(row=row,stable_id=str(data['ids'][row]),**ev))
            counts=collections.Counter(r['reason'] for r in decisions);counts['confirmed']=len(confirmed);counts['checked']=len(sub);total.update(counts)
            write(dest,dict(schema='rinkspace.ad_source_entity_verified.v2',version=VERSION,atlas_sha256=atlas_sha,ad=n,pts=pt,source_frame=vis,confirmed=confirmed,excluded=decisions,conflicts=conflicts,registrations=regs,counts=dict(counts),native_model_used_for_image_identity=False,original_UV_PTS_unchanged=True))
        if frame_number%40==0:base.emit('entity_verification_progress',frames=frame_number+1,total_frames=len(points_by_frame),counts=dict(total),seconds=round(time.monotonic()-start,1))
    base.emit('entity_verification_complete',counts=dict(total),seconds=round(time.monotonic()-start,1))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--ads',type=int,nargs='+');a=p.parse_args();verify(a.ads)
