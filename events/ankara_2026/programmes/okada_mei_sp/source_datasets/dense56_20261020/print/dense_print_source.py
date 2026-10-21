import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import av,cv2,numpy as np,json,gzip,hashlib,time,collections,sys
from pathlib import Path
from scipy.spatial import cKDTree
P=Path(r'D:/OneDrive/codex_projs/RinkSpace');D=Path(r'D:/CodexData/RinkSpace_Data');R=D/'runs/okada_rec_1019/recover_v2';O=R/'dense_print_v1';BO=R/'dense_board_v1';J=D/'runs/okada_joint9_1018a'
cv2.setNumThreads(2)
SHOTS=[('s02',575744,720640,'shot2_r2_20261014'),('s03',720896,745728,'shot3_20261014'),('s04',745984,948736,'shot4_20261014')]
VIDEO=D/'events/ankara_2026/programmes/okada_mei_sp/Mei OKADA (JPN) ｜ Junior Women Short Program ｜ Ankara 2026.mp4'
SHA='e8b53966d56e72f3842d92c3ecbe6e9cc7e4670b7d0c9bd12f58fd2cff5ad57d'
def load(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,j):Path(p).write_text(json.dumps(j,ensure_ascii=False,indent=2),encoding='utf8')
def gzrows(p,rows):
 with gzip.open(p,'wt',encoding='utf8',compresslevel=5) as f:
  for x in rows:f.write(json.dumps(x,separators=(',',':'),ensure_ascii=False)+'\n')
def sha(b):return hashlib.sha256(b).hexdigest()
def frame_json(pts):
 q=BO/'frames'/f'{pts}.json'
 while not q.exists():time.sleep(.5)
 return load(q)
def fullmask(pts,rgb,led):
 while not (R/'dense_masks_v1'/'masks'/f'{pts}.png').exists():time.sleep(.5)
 # Same-exposure dense SAM2 supplied by parent; never infer body from rink model.
 for base in (R/'dense_source_masks_v1',R/'dense_masks_v1'):
  for name in (f'{pts}.png',f'frames/{pts}.png',f'masks/{pts}.png'):
   q=base/name
   if q.exists():
    im=cv2.imread(str(q),0)
    if im is not None and im.shape==rgb.shape[:2]:return im>0,dict(kind='same_exposure_existing_SAM2_mask',file=str(q),sha256=sha(q.read_bytes()))
 h,w=rgb.shape[:2];mask=np.zeros((h,w),np.uint8);mask[:170,:620]=1;mask[900:,1600:]=1
 refs=[]
 for row in led.get(pts,{}).get('current_person_boxes',[]):
  b=row['source_binding'];assert b['source_pts']==pts and b['decoded_native_rgb_sha256']==sha(rgb.tobytes())
  a=np.maximum([0,0],np.floor(np.array(row['box_xyxy'])[:2]-5).astype(int));z=np.minimum([w,h],np.ceil(np.array(row['box_xyxy'])[2:]+5).astype(int));mask[a[1]:z[1],a[0]:z[0]]=1;refs.append(row)
 return mask>0,dict(kind='same_exposure_source_bound_body_boxes_conservative',ledger='dense_remaining_v1/frame_ledger.jsonl.gz or dense_s03_v1/frame_ledger.jsonl.gz',box_count=len(refs),mask_sha256=sha(mask.tobytes()),visible_board_inside_person_box_not_claimed_complete=True)
def board_region(board,h,w):
 roles=board['roles']; ys=[]
 for role in ('board.top','board.contact'):
  arr=np.asarray(roles[role]['trace_uv'],float).reshape(-1,2)
  if len(arr)<2:return np.zeros((h,w),np.uint8),None
  arr=arr[np.argsort(arr[:,0])]; xx=np.arange(w); y=np.interp(xx,arr[:,0],arr[:,1],left=np.nan,right=np.nan);ys.append(y)
 mask=np.zeros((h,w),np.uint8)
 for x in range(w):
  a,b=ys[0][x],ys[1][x]
  if np.isfinite(a+b) and b>a+6:mask[max(0,int(np.ceil(a+4))):min(h,int(np.floor(b-4))),x]=255
 return mask,ys

def source_edges(rgb,mask):
 gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY); sm=cv2.GaussianBlur(rgb,(3,3),.6);edges=cv2.Canny(gray,22,55,L2gradient=True)
 for ch in (0,2):edges|=cv2.Canny(sm[:,:,ch],28,65,L2gradient=True)
 edges[mask==0]=0
 contours,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
 pts=[];cids=[];local=0
 for c in contours:
  c=c[:,0,:]
  if len(c)<10:continue
  local+=1
  # Preserve actual source contour polyline; no smooth curve is used as a pixel observation.
  c=c[::4]
  pts.extend(c);cids.extend([local]*len(c))
 if not pts:return gray,np.zeros((0,2)),np.zeros((0,2)),np.zeros(0),np.zeros(0),np.zeros(0,int)
 pts=np.asarray(pts,np.float32);cids=np.asarray(cids)
 # Eliminate retraced Canny duplicates before subpixel read.
 cells=np.floor(pts/2).astype(int);_,idx=np.unique(cells,axis=0,return_index=True);idx=np.sort(idx);pts=pts[idx];cids=cids[idx]
 gx=np.stack([cv2.Sobel(sm[:,:,c],cv2.CV_32F,1,0,ksize=3)for c in range(3)],2);gy=np.stack([cv2.Sobel(sm[:,:,c],cv2.CV_32F,0,1,ksize=3)for c in range(3)],2)
 x=pts[:,0].astype(int);y=pts[:,1].astype(int);mag=gx[y,x]**2+gy[y,x]**2;channel=mag.argmax(1);ii=np.arange(len(pts));n=np.c_[gx[y,x,channel],gy[y,x,channel]];n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-5)
 t=np.arange(-3,3.01,.25,dtype=np.float32);query=pts[:,None,:]+n[:,None,:]*t[None,:,None];profile=cv2.remap(rgb.astype(np.float32),query[:,:,0],query[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
 # Half-transition is measured from current RGB plateaus, matching S01 ink-boundary convention.
 left=np.median(profile[:,0:5],axis=1);right=np.median(profile[:,-5:],axis=1);ch=np.argmax(abs(right-left),axis=1);sign=np.sign(right[ii,ch]-left[ii,ch]);contrast=abs(right[ii,ch]-left[ii,ch]);v=profile[ii,:,ch];mid=(right[ii,ch]+left[ii,ch])/2;center=v-mid[:,None];cross=(center[:,:-1]*center[:,1:]<=0)&(np.diff(v,axis=1)*sign[:,None]>0)
 weights=np.where(cross,abs((t[:-1]+t[1:])/2)[None,:],1e9);ix=weights.argmin(1);a=v[ii,ix];b=v[ii,ix+1];offset=t[ix]+(mid-a)/np.where(abs(b-a)>1e-5,b-a,1)*.25
 good=(contrast>=12)&(weights[ii,ix]<1.8)&(np.abs(offset)<1.8)&np.isfinite(offset)
 uv=pts+n*offset[:,None];xx=np.rint(uv[:,0]).astype(int);yy=np.rint(uv[:,1]).astype(int);good&=(xx>=4)&(xx<rgb.shape[1]-4)&(yy>=4)&(yy<rgb.shape[0]-4)
 good&=mask[np.clip(yy,0,rgb.shape[0]-1),np.clip(xx,0,rgb.shape[1]-1)]>0
 return gray,uv[good],n[good],contrast[good],offset[good],cids[good]

def track(prevgray,gray,prevuv,uv):
 out={}
 if prevgray is None or len(prevuv)==0 or len(uv)==0:return out
 a=np.asarray(prevuv,np.float32).reshape(-1,1,2);b,s,e=cv2.calcOpticalFlowPyrLK(prevgray,gray,a,None,winSize=(17,17),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,20,.01),minEigThreshold=1e-5)
 c,t,eb=cv2.calcOpticalFlowPyrLK(gray,prevgray,b,None,winSize=(17,17),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,20,.01),minEigThreshold=1e-5)
 fb=np.linalg.norm(a[:,0]-c[:,0],axis=1);tree=cKDTree(uv);dist,ix=tree.query(b[:,0],distance_upper_bound=2.0);used=set()
 for i in np.argsort(dist):
  j=int(ix[i])
  if s[i,0] and t[i,0] and fb[i]<.75 and dist[i]<2 and j not in used:out[j]=(int(i),float(fb[i]),float(dist[i]));used.add(j)
 return out

def seeds():
 ans={}
 for sid,first,last,folder in SHOTS:
  for path in sorted((P/f'events/ankara_2026/programmes/okada_mei_sp/source_datasets/{folder}/frames').glob('*.json')):
   f=load(path);pts=int(f.get('source_pts',f['source']['native_pts']));rows=[]
   for feature in f.get('features',[]):
    if not feature.get('id','').startswith('Ad'):continue
    for si,s in enumerate(feature.get('segments',[])):
     xy=s.get('source_native_readout_uv')
     origin='original_dense_native_readout'
     if xy is None and s.get('pixels_measured_in_this_frame'):xy=s.get('uv',[]);origin=s.get('measurement_origin','original_pixel_readout')
     if xy:
      for i,p in enumerate(xy):rows.append(dict(uv=p,feature_id=feature['id'],source_file=str(path),source_pointer=f'/features/{f["features"].index(feature)}/segments/{si}/'+('source_native_readout_uv'if s.get('source_native_readout_uv')is not None else'uv')+f'/{i}',origin=origin))
   ans[pts]=rows
 return ans

def run():
 O.mkdir(exist_ok=True);(O/'frames').mkdir(exist_ok=True);seed=seeds();ledger={}
 for sub in ('dense_remaining_v1','dense_s03_v1'):
  with gzip.open(J/sub/'frame_ledger.jsonl.gz','rt',encoding='utf8')as f:
   for line in f:
    v=json.loads(line);ledger[int(v['native_pts'])]=v
 source=hashlib.sha256()
 with open(VIDEO,'rb')as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):source.update(b)
 assert source.hexdigest()==SHA
 write(O/'recipe.json',dict(schema='rinkspace.dense_native_board_ink_half_transition.v1',source_sha256=SHA,source_time_base='1/12800',all_native_frames=1458,source_rule='S01 current_native_material_normal_half_transition; glyph boundaries are ink material edges',source_seed_frames=len(seed),source_seed_rows=sum(map(len,seed.values())),camera_model_used=False,contour_samples_each_reread_from_actual_current_RGB=True,FB_threshold_px=.75,FB_is_image_continuity_not_world_correspondence=True,source_sigma_unknown=True,no_solver=True))
 counter=0;summary=[];prevgray=None;prevuv=[];previds=[];prevfeatures=[];prevsid=None
 with av.open(str(VIDEO))as container:
  stream=container.streams.video[0];assert str(stream.time_base)=='1/12800';container.seek(575744,stream=stream,backward=True)
  for frame in container.decode(stream):
   pts=int(frame.pts)
   if pts<575744:continue
   if pts>948736:break
   sid=next((sid for sid,a,b,_ in SHOTS if a<=pts<=b),None)
   if sid is None:continue
   rgb=frame.to_ndarray(format='rgb24');rgbsha=sha(rgb.tobytes());rawbgrsha=sha(rgb[:,:,::-1].copy().tobytes());old=ledger.get(pts)
   if old and old.get('decoded_native_BGR_sha256'):assert old['decoded_native_BGR_sha256']==rawbgrsha
   board=frame_json(pts);mask,mreceipt=fullmask(pts,rgb,ledger);region,ys=board_region(board,*rgb.shape[:2]);region[mask]=0
   gray,uv,norm,contrast,offset,cids=source_edges(rgb,region)
   eigen=cv2.cornerEigenValsAndVecs(gray,5,3); qxy=np.rint(uv).astype(int); ev=eigen[qxy[:,1],qxy[:,0],:2]if len(uv)else np.zeros((0,2)); twod=np.min(ev,axis=1)/np.maximum(np.max(ev,axis=1),1e-12)if len(uv)else np.zeros(0)
   if sid!=prevsid:prevgray=None;prevuv=[];previds=[];prevfeatures=[]
   flow=track(prevgray,gray,prevuv,uv);ids=[];features=[];rows=[];seedrows=seed.get(pts,[]);stree=cKDTree(np.array([r['uv']for r in seedrows],float))if seedrows else None;sdist,six=stree.query(uv,distance_upper_bound=3)if stree is not None and len(uv)else(np.zeros(len(uv))+1e9,np.zeros(len(uv),int))
   for i,p in enumerate(uv):
    prior=flow.get(i);counter+=1;fid=previds[prior[0]]if prior else f'{sid}.board_ink.{counter}';localfeature=f'{sid}.board_print.component.{pts}.{int(cids[i])}'
    semantic=None;semanticsource=None
    if stree is not None and sdist[i]<3:semantic=seedrows[int(six[i])]['feature_id'];semanticsource=seedrows[int(six[i])]
    elif prior:semantic=prevfeatures[prior[0]]
    if semantic:localfeature=semantic
    ids.append(fid);features.append(semantic)
    row=dict(shot_id=sid,native_pts=pts,feature_id=localfeature,segment_id=f'{pts}.contour.{int(cids[i])}',point_id=fid,uv=[round(float(p[0]),5),round(float(p[1]),5)],source_kind='board_ink_material_half_transition',physical_surface='board',board_instance=semantic.split('.')[0]if semantic else None,crosswalk_status='source_frame_label_propagated_by_current_FB'if semantic else'board_material_confirmed_component_unbound',world_material_crosswalk='unknown_not_used',pixel_support=dict(normal=[round(float(z),5)for z in norm[i]],contrast_native_DN=round(float(contrast[i]),3),half_transition_offset_px=round(float(offset[i]),5),local_structure_eigenvalue_ratio=round(float(twod[i]),5),two_dimensional_texture_support=bool(twod[i]>.1)),mask_evidence='frame_coverage_record',FB_evidence=dict(previous_native_pts=pts-256,error_px=prior[1],current_edge_relocalization_px=prior[2],world_point_correspondence=False)if prior else dict(status='new_current_native_pixel_measurement_no_temporal_match'))
    if semanticsource:row['seed_source_pointer']=semanticsource['source_pointer'];row['seed_source_file']=semanticsource['source_file']
    rows.append(row)
   # Keep the exact original UV record separately; it is not replaced with its detected neighbor.
   for j,s in enumerate(seedrows):
    x,y=map(float,s['uv']);xi,yi=int(round(x)),int(round(y));visible=0<=xi<1920 and 0<=yi<1080 and region[yi,xi]>0
    if visible:
     rows.append(dict(shot_id=sid,native_pts=pts,feature_id=s['feature_id'],segment_id=s['feature_id']+'.original_readout',point_id=f'{sid}.{pts}.original.{j}',uv=[x,y],source_kind='preserved_original_native_advertisement_readout',physical_surface='board',board_instance=s['feature_id'].split('.')[0],crosswalk_status='original_source_frame_local_identity',world_material_crosswalk='unknown_not_used',pixel_support=dict(origin=s['origin'],original_UV_unchanged=True),seed_source_file=s['source_file'],seed_source_pointer=s['source_pointer'],mask_evidence='frame_coverage_record',FB_evidence=dict(status='exact_original_current_exposure_readout')))
   fo=O/'frames'/f'{pts}.jsonl.gz';gzrows(fo,rows)
   cov=dict(schema='rinkspace.dense_native_board_ink_frame.v1',shot_id=sid,native_pts=pts,time_base='1/12800',decoded_native_RGB_sha256=rgbsha,decoded_native_BGR_sha256=rawbgrsha,points=len(rows),new_current_half_transition_points=len(uv),current_FB_supported_points=len(flow),original_advertisement_readouts_preserved=sum(r['source_kind'].startswith('preserved')for r in rows),measured_connected_segments=len(set(map(int,cids))),board_region_pixels=int(np.count_nonzero(region)),mask=mreceipt,board_trace_file=str(BO/'frames'/f'{pts}.json'),source_rows_file=str(fo),source_rows_sha256=sha(fo.read_bytes()),all_visible_fine_print_complete=False,unknown_crosswalk_is_not_unknown_surface=True)
   write(O/'frames'/f'{pts}.coverage.json',cov);summary.append(cov);write(O/'progress.json',dict(frames=len(summary),last_native_pts=pts,last_shot_id=sid,total_rows=sum(x['points']for x in summary),last_frame=cov))
   prevgray=gray;prevuv=uv;previds=ids;prevfeatures=features;prevsid=sid
   if len(summary)%20==0:print(json.dumps(dict(frames=len(summary),pts=pts,points=len(rows),FB=len(flow)),ensure_ascii=True),flush=True)
   if pts in (575744,598272,650240,719616,720896,733440,744704,745984,800000,878848,948736):
    show=rgb[:,:,::-1].copy()
    for r in rows:
     x,y=np.rint(r['uv']).astype(int);cv2.circle(show,(int(x),int(y)),1,(0,255,255),-1,cv2.LINE_AA)
    cv2.imwrite(str(O/f'{sid}_{pts}_points.jpg'),show,[cv2.IMWRITE_JPEG_QUALITY,93])
 assert len(summary)==1458,len(summary)
 gzrows(O/'coverage.jsonl.gz',summary)
 with gzip.open(O/'rows.jsonl.gz','wt',encoding='utf8',compresslevel=5)as dst:
  for c in summary:
   with gzip.open(c['source_rows_file'],'rt',encoding='utf8')as f:
    for line in f:dst.write(line)
 per={sid:dict(frames=sum(c['shot_id']==sid for c in summary),points=sum(c['points']for c in summary if c['shot_id']==sid),minimum_per_frame=min(c['points']for c in summary if c['shot_id']==sid),FB_supported_points=sum(c['current_FB_supported_points']for c in summary if c['shot_id']==sid))for sid,*_ in SHOTS}
 write(O/'manifest.json',dict(schema='rinkspace.dense_native_board_ink_generation.v1',frames=len(summary),per_shot=per,source_sha256=SHA,rows=str(O/'rows.jsonl.gz'),rows_sha256=sha((O/'rows.jsonl.gz').read_bytes()),coverage=str(O/'coverage.jsonl.gz'),no_world_model_or_solver=True,source_sigma_unknown=True,all_points_are_observed_board_ink_candidates_not_all_are_named_world_anchors=True))
 print(json.dumps(per),flush=True)
if __name__=='__main__':run()
