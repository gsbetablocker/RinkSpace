import os
for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
exec(open(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2/dense56_v1/shoe_fix/audit_static.py',encoding='utf8').read().split('mode=sys.argv')[0])
import av
mode='full_v3';aa=rz(O/mode/'index.stream.jsonl.gz');shot=sys.argv[1]if len(sys.argv)>1 else'all';aa=[q for q in aa if shot=='all'or q['shot_id']==shot];a={q['native_pts']:q for q in aa};source=json.load(open(R/'dense56_v1/parts/s05/video_config.json',encoding='utf8'))['source'];out=O/mode/f'body_source_motion_{shot}.jsonl.gz';start=time.perf_counter();count=0;p0=None;g0=None;m0=None;b0=None;rows=[]
with av.open(source['file'])as c,gzip.open(out,'wt',encoding='utf8')as w:
 st=c.streams.video[0];c.seek(min(a),stream=st,backward=True)
 for fr in c.decode(st):
  p=int(fr.pts)
  if p>max(a):break
  if p not in a:continue
  r=a[p];bgr=fr.to_ndarray(format='bgr24');assert hashlib.sha256(bgr.tobytes()).hexdigest()==r['source_BGR_sha256'];g=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY);m=cv2.imread(r['mask_file'],0)>0;box=r['bbox_search_only'];roi=np.zeros_like(m);roi[int(box[1]):int(box[3]),int(box[0]):int(box[2])]=True;work=m&roi;record=dict(native_pts=p,shot_id=r['shot_id'],source_BGR_sha256=r['source_BGR_sha256'],source_RGB_sha256=r['source_RGB_sha256'],mask_ref=dict(file=r['mask_file'],sha256=r['mask_sha256']),person_motion_QA_only_never_static_scene_observations=True)
  if p0 is not None and p-p0==256 and r['shot_id']==a[p0]['shot_id']:
   q=cv2.goodFeaturesToTrack(g0,maxCorners=0,qualityLevel=.02,minDistance=6,mask=m0.astype('u1'),blockSize=5)
   if q is not None:
    z,s,e=cv2.calcOpticalFlowPyrLK(g0,g,q,None,winSize=(21,21),maxLevel=4,criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,30,.01));bk,bs,be=cv2.calcOpticalFlowPyrLK(g,g0,z,None,winSize=(21,21),maxLevel=4,criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,30,.01));fb=np.linalg.norm(bk[:,0]-q[:,0],axis=1);uv=z[:,0];xx=np.rint(uv[:,0]).astype(int);yy=np.rint(uv[:,1]).astype(int);inside=(xx>=0)&(xx<1920)&(yy>=0)&(yy<1080);xx=np.clip(xx,0,1919);yy=np.clip(yy,0,1079);ok=(s[:,0]>0)&(bs[:,0]>0)&inside&(fb<.75)&m[yy,xx];record.update(previous_native_pts=p0,previous_source_BGR_sha256=a[p0]['source_BGR_sha256'],previous_mask_ref=dict(file=a[p0]['mask_file'],sha256=a[p0]['mask_sha256']),actual_source_uv=q[ok,0].tolist(),current_uv=uv[ok].tolist(),FB_error_px=fb[ok].tolist(),candidate_actual_texture_corners=len(q),current_person_supported_track_count=int(ok.sum()),rejected_or_occluded_texture_tracks=int((~ok).sum()),no_cross_cut=True)
   else:record.update(previous_native_pts=p0,status='no_texture_in_current_person',current_person_supported_track_count=0)
  else:record.update(status='shot_start_no_cross_cut_flow',current_person_supported_track_count=0)
  w.write(json.dumps(record,separators=(',',':'))+chr(10));rows.append(record);p0=p;g0=g;m0=work;b0=box;count+=1
  if count%100==0:print(count,round(time.perf_counter()-start,2),flush=True)
report=dict(frames=count,complete=count==({'s05':1376,'s06':486}.get(shot,1862)),source_video_sha256=source['sha256'],all_native_BGR_verified=True,source_body_motion_is_only_mask_QA_never_scene_static_observation=True,cross_cut_tracks=0,person_supported_actual_LK_FB_tracks=sum(x['current_person_supported_track_count']for x in rows),frames_with_supported_source_person_motion=sum(x['current_person_supported_track_count']>0 for x in rows),zero_track_frames=[x['native_pts']for x in rows if not x['current_person_supported_track_count']],index=dict(file=str(out),sha256=sh(out)),elapsed_seconds=time.perf_counter()-start)
(O/mode/f'body_source_motion_{shot}_receipt.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report),flush=True)
