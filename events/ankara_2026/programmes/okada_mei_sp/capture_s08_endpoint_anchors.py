"""Read the known ANKARA word in S08's first frames using source-image navigation.

The navigation starts at its existing reviewed source registration and follows
actual source texture backwards. No native projection navigates these readings.
"""
from pathlib import Path
import argparse,json,hashlib
import av,cv2,numpy as np
import prepare_native_multicamera as p
from capture_ad4_source_repair import word_reference
from capture_independent_ad4_glyph_gap import read_public,save_profiles,safe,write
from rinkspace.core.native import NativeCore
from rinkspace.path_policy import guard_project_write



def read_transported_edges(pattern,image,mask,H,anchor_H,archived,binding):
    from rinkspace.vision_source_shapes import read_source_material_transition_profiles
    from rinkspace.session.print_patterns import project,sample
    inverse=np.linalg.inv(anchor_H);queries=[];normals=[];reference_queries=[];labels=[]
    relative=H@inverse
    for row in archived:
        r=row['original_measurement_record'];q=np.asarray(r['search_uv']);raw=project(inverse,q[None])[0];reference_queries.append(raw)
        qnew=project(H,raw[None])[0];direction=np.asarray(r['normal']);step=project(relative,np.array([q,q+[-direction[1],direction[0]]]))
        tangent=step[1]-step[0];normal=np.array([tangent[1],-tangent[0]]);normal/=np.linalg.norm(normal)
        if np.dot(normal,direction)<0:normal*=-1
        queries.append(qnew);normals.append(normal);labels.append(r['path_id'].split('::',1)[1])
    queries=np.asarray(queries);normals=np.asarray(normals);radius=np.maximum(4.,49*abs(normals[:,0])/2+3)
    public=read_source_material_transition_profiles(np.ascontiguousarray(image[:,:,::-1]),binding=binding,search_points_uv=queries,normals=normals,search_intervals_px=np.c_[-radius,radius],transition='dark_to_light',tangent_half_width_px=1,smoothing_sigma_px=.6,engineering_support_margin_px=.25,keep_profiles=True)
    captured=dict(public_readings=public['readings'],search_uv=queries,normals=normals,intervals=np.c_[-radius,radius],query_path_labels=labels)
    projected=project(H,pattern.points);lo=np.maximum(np.floor(projected.min(0)-5).astype(int),[0,0]);hi=np.minimum(np.ceil(projected.max(0)+6).astype(int),[1920,1080]);signal=pattern.ink_signal(image).astype(np.float32);hidden=cv2.dilate(mask,np.ones((7,7),np.uint8))
    values=signal[lo[1]:hi[1],lo[0]:hi[0]][hidden[lo[1]:hi[1],lo[0]:hi[0]]==0].astype(np.uint8)
    threshold,_=cv2.threshold(values.reshape(-1,1),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    accepted=[];reasons={}
    for i,(r,q,n,old,raw) in enumerate(zip(public['readings'],queries,normals,archived,reference_queries,strict=True)):
        offsets=np.asarray(r['source_q10_q50_q90_normal_offsets_px']);reason=None
        if any(r['source_flags'].values()) or not np.isfinite(offsets).all():reason='public_source_profile_unresolved'
        else:
            positions=q+np.linspace(offsets[0]-1,offsets[2]+1,33)[:,None]*n
            if np.any(positions<0) or np.any(positions>[1919,1079]) or np.any(sample(hidden,positions)>0):reason='source_profile_occluded_or_outside'
            else:
                sides=sample(signal,q+np.array([offsets[0]-1,offsets[2]+1])[:,None]*n)
                if not sides[0]<threshold<sides[1]:reason='not_source_blue_to_white_ink_boundary'
        if reason:reasons[reason]=reasons.get(reason,0)+1;continue
        accepted.append(dict(path_id=old['original_measurement_record']['path_id'],candidate=i,uv=np.asarray(r['actual_source_uv']).tolist(),search_uv=q.tolist(),normal=n.tolist(),actual_normal_quantiles=offsets.tolist(),actual_current_material_sides=sides.tolist(),current_ink_class_boundary=float(threshold),source_profile_flags=r['source_flags'],current_source_RGB_measured=True,model_UV_used_as_measurements=False,source_query_reference_uv=raw.tolist(),transported_from_reviewed_observation_id=old['observation_id']))
    return accepted,dict(queries=len(queries),accepted=len(accepted),reasons=reasons,qualification='pending individual source-identity review; no whole-word registration claim'),captured


def main(a):
    out=guard_project_write(a.output.resolve())
    if out.exists():raise FileExistsError(out)
    m=p.read(a.prepared/'joint_manifest.json');state=p.npz(a.state)
    core=NativeCore(m['native_library']);pattern,_,_,atlas=word_reference(m,state,core)
    if p.digest(atlas['image'])!=atlas['image_sha256']:raise ValueError('Reference source changed')
    sm=p.read(a.readouts/'source_measurements.json');review=p.read(a.readouts/'source_review.json')
    record=next(r for r in sm if r['pts']==1743360 and r['original_measurement_record'].get('source_atlas_index')==6)
    locator=record['original_measurement_record']['source_readout_locator'];registration_path=Path(locator['path'])
    if p.digest(registration_path)!=locator['sha256']:raise ValueError('Original source registration changed')
    original=json.loads(registration_path.read_text('utf-8').splitlines()[locator['line']-1]);registered=next(x for x in original['patterns'] if x['atlas_index']==6)['registration']
    previous_H=np.asarray(registered['H']);anchor=1743360;requested=[1742848,1743104];frames={}
    with av.open(str(a.source)) as container:
        stream=container.streams.video[0]
        if str(stream.time_base)!='1/12800':raise ValueError('Wrong source clock')
        container.seek(requested[0],stream=stream,backward=True)
        for frame in container.decode(stream):
            if frame.pts>anchor:break
            if frame.pts>=requested[0]:frames[int(frame.pts)]=frame.to_ndarray(format='bgr24')
    if sorted(frames)!=requested+[anchor]:raise ValueError('Missing original source exposure')
    for frame in p.read(a.readouts/'readouts.json')['frames']:
        pt=frame['pts']
        if pt in frames and hashlib.sha256(frames[pt].tobytes()).hexdigest()!=frame['decoded_BGR_sha256']:raise ValueError('Original source pixels differ')
    out.mkdir(parents=True);(out/'profiles').mkdir();write(out/'word_reference.json',atlas)
    previous=anchor;H_by_pts={anchor:previous_H};navigation=[]
    for pt in reversed(requested):
        refgray=cv2.cvtColor(frames[previous],cv2.COLOR_BGR2GRAY);gray=cv2.cvtColor(frames[pt],cv2.COLOR_BGR2GRAY)
        projected=cv2.perspectiveTransform(pattern.points[None].astype(float),previous_H)[0];lo=np.maximum(np.floor(projected.min(0)-12).astype(int),[0,0]);hi=np.minimum(np.ceil(projected.max(0)+13).astype(int),[1920,1080]);mask=np.zeros((1080,1920),np.uint8);mask[lo[1]:hi[1],lo[0]:hi[0]]=255
        features=cv2.goodFeaturesToTrack(refgray,0,.005,5,mask=mask)
        if features is None:raise ValueError('No source word texture')
        now,ok,_=cv2.calcOpticalFlowPyrLK(refgray,gray,features,None,winSize=(31,31),maxLevel=5,criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,60,.001));back,bok,_=cv2.calcOpticalFlowPyrLK(gray,refgray,now,None,winSize=(31,31),maxLevel=5,criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,60,.001))
        good=ok[:,0].astype(bool)&bok[:,0].astype(bool)&(np.linalg.norm(back[:,0]-features[:,0],axis=1)<1.)
        affine,inliers=cv2.estimateAffine2D(features[good,0],now[good,0],method=cv2.RANSAC,ransacReprojThreshold=1.,confidence=.999,refineIters=10)
        if affine is None:raise ValueError('Source word texture has no consistent warp')
        relative=np.eye(3);relative[:2]=affine;H=relative@previous_H
        H_by_pts[pt]=H;navigation.append(dict(from_pts=previous,to_pts=pt,consistent_tracks=int(good.sum()),affine_inliers=int(inliers.sum()),source_affine=relative.tolist(),model_UV_used=False));previous,previous_H=pt,H
    supported={r['observation_id'] for r in review['rows'] if r['status']=='identity_compatible' and r['pts']==anchor}
    archived=[r for r in sm if r['pts']==anchor and r['observation_id'] in supported and '.ANKARA.' in r['original_measurement_record']['path_id']]
    audit=[];all_rows=[];counts=[]
    for pt in requested:
        image=frames[pt];mask=cv2.imread(str(a.masks/'s08'/(str(pt)+'.png')),cv2.IMREAD_GRAYSCALE)
        if mask is None:raise ValueError('Missing actual same-exposure foreground mask')
        registration=None if a.edge_transport else pattern.register(image,mask,H_by_pts[pt]);binding=dict(source_sha256=a.source_sha256,source_pts=pt,time_base='1/12800',shot_id='s08',event_id='ankara_2026',rig_id='2',pixel_domain='native_distorted',width=1920,height=1080,decoded_native_rgb_sha256=hashlib.sha256(np.ascontiguousarray(image[:,:,::-1]).tobytes()).hexdigest())
        if a.edge_transport:
            registration=dict(status='source_edge_transport_pending_review',H=H_by_pts[pt].tolist(),source_anchor_pts=anchor,method='backward actual-source texture transport of individually reviewed ink edges',whole_word_registration_claimed=False)
            rows,summary,captured=read_transported_edges(pattern,image,mask,H_by_pts[pt],np.asarray(registered['H']),archived,binding)
        else:rows,summary,captured=read_public(pattern,image,mask,registration,binding)
        profile=out/'profiles'/(str(pt)+'.npz');save_profiles(profile,captured);registration['profile_path']=str(profile)
        raw=out/(str(pt)+'_original.png');cv2.imwrite(str(raw),image);overlay=image.copy()
        for i,row in enumerate(rows,1):
            row.update(pts=pt,kind=1,fit_eligible=False,model_UV_synthesized=False,evidence_image=str(raw),short_index=i,source_BGR_sha256=hashlib.sha256(image.tobytes()).hexdigest(),measurement_operator='known_ANKARA_source_texture_navigation_original_RGB_ink_q50',source_profile_proof=dict(path=str(profile),sha256=p.digest(profile)),source_UV_synthesized=False)
            uv=tuple(np.rint(row['uv']).astype(int));cv2.circle(overlay,uv,3,(0,0,255),1,cv2.LINE_AA);cv2.putText(overlay,str(i),(uv[0]+4,uv[1]-5),cv2.FONT_HERSHEY_SIMPLEX,.42,(0,255,255),1,cv2.LINE_AA)
        cv2.imwrite(str(out/(str(pt)+'_readouts.png')),overlay);all_rows.extend(rows);counts.append(dict(pts=pt,rows=len(rows),status=registration['status'],readout=summary));audit.append(dict(pts=pt,source_BGR_sha256=hashlib.sha256(image.tobytes()).hexdigest(),word_registration=registration))
    # Existing anchor UV stays exact. These arrays locate its archived query
    # on the original photograph; they do not create a new pixel measurement.
    anchor_profile=out/'profiles'/(str(anchor)+'_archived.npz')
    np.savez_compressed(anchor_profile,query_path_labels=np.array([r['original_measurement_record']['path_id'].split('::',1)[1] for r in archived]),actual_source_uv=np.array([r['original_measurement_record']['uv'] for r in archived]),search_uv=np.array([r['original_measurement_record']['search_uv'] for r in archived]))
    anchor_registration={**registered,'profile_path':str(anchor_profile),'profile_origin':'existing exact original RGB measurements; not reread or fitted'};audit.append(dict(pts=anchor,source_BGR_sha256=hashlib.sha256(frames[anchor].tobytes()).hexdigest(),word_registration=anchor_registration))
    with (out/'source_audit.jsonl').open('x',encoding='utf-8') as f:
        for r in audit:f.write(json.dumps(safe(r),ensure_ascii=False,allow_nan=False)+chr(10))
    write(out/'readouts.json',dict(schema='rinkspace.endpoint_ANKARA_source_candidates.v1',source=str(a.source),source_sha256=a.source_sha256,rows=all_rows,counts=counts,source_texture_navigation=navigation,no_native_projection_used=True,fit_eligible=False,review_state='awaiting_source_identity_review',user_visual_acceptance=False,pixel_accuracy_accepted=False))
    write(out/'receipt.json',dict(reference_registration_path=str(registration_path),reference_registration_sha256=p.digest(registration_path),source_UV_synthesized=False,original_source_rows_modified=False,counts=counts,source_texture_navigation=navigation))
    print(json.dumps(dict(output=str(out),counts=[dict(pts=x['pts'],rows=x['rows'],status=x['status']) for x in counts])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','state','readouts','source','masks','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--source-sha256',required=True)
    parser.add_argument('--edge-transport',action='store_true',help='Read only individually reviewed anchor edges via actual source texture; no whole-word registration claim')
    main(parser.parse_args())
