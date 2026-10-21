from pathlib import Path
import sys,json,gzip,hashlib,math,time,av,cv2,numpy as np
O=Path(sys.argv[1]);N=O/'near_board_v1';T=N/'final_v3';T.mkdir(exist_ok=False);(T/'frames').mkdir();(T/'masks').mkdir();(T/'QA').mkdir();sys.path.insert(0,str(O));from source_graphics import current_graphics
cv2.setNumThreads(2)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,j):Path(p).write_text(json.dumps(j,ensure_ascii=False,separators=(',',':')),encoding='utf8')
cfg=json.load(open(O/'consumer_s06.json',encoding='utf8'));bm=json.load(open(cfg['board_manifest']['file'],encoding='utf8'));am=json.load(open(cfg['advert_manifest']['file'],encoding='utf8'))
mi={r['native_pts']:r for r in map(json.loads,gzip.open(cfg['mask_index']['file'],'rt',encoding='utf8')) if r['shot_id']=='s06'};bc={r['native_pts']:r for r in map(json.loads,gzip.open(bm['files']['qualified_coverage.jsonl.gz']['file'],'rt',encoding='utf8')) if r['shot_id']=='s06'}
det=json.load(open(N/'detect_probe.json',encoding='utf8'));kn=[x for x in det if x['detected']];kp=np.array([x['native_pts']for x in kn]);kb=np.array([x['best']['m']*1079+x['best']['b']for x in kn]);km=np.array([x['best']['m']for x in kn]);km=np.where(km>.58,np.median(km[km<.58]),km)
def nav(p):
 if p<kp[0]:bottom=kb[0]+(p-kp[0])*(kb[1]-kb[0])/(kp[1]-kp[0]);m=km[0]
 elif p>kp[-1]:bottom=kb[-1]+(p-kp[-1])*(kb[-1]-kb[-2])/(kp[-1]-kp[-2]);m=km[-1]
 else:bottom=float(np.interp(p,kp,kb));m=float(np.interp(p,kp,km))
 return m,bottom-m*1079,bottom
# Navigation is used only to search raw image profiles. Every returned UV is this exposure's measured half material transition.
def profiles(b,m,bb,guard):
 g=cv2.GaussianBlur(cv2.cvtColor(b,cv2.COLOR_BGR2GRAY).astype(np.float32),(0,0),.8);n=np.array([1.,-m]);n/=np.linalg.norm(n);ds=np.arange(-70.,70.001,.5);pts=[];unknown=[]
 for y in range(270,1079,6):
  x=m*y+bb
  if x < -45 or x>680:continue
  xy=np.array([x,y])+ds[:,None]*n;xx=np.broadcast_to(xy[:,0][None,:],(5,len(ds))).astype(np.float32);yy=(xy[:,1][None,:]+np.arange(-2,3)[:,None]).astype(np.float32)
  v=cv2.remap(g,xx,yy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE).mean(0);gradient=np.gradient(v);pk=np.flatnonzero((gradient[1:-1]>=gradient[:-2])&(gradient[1:-1]>=gradient[2:]))+1;opts=[]
  for j in pk:
   if abs(ds[j])>52 or j<56 or j+56>=len(ds)or gradient[j]<.25:continue
   uv0=np.array([x,y])+ds[j]*n
   left_extent=min(25.,(uv0[0]-1)/n[0],(1076-uv0[1])/max(1e-6,-n[1]));
   if left_extent<5:continue
   plate=np.array([uv0-left_extent*n,uv0+25*n])
   if np.any(plate[:,0]<0)or np.any(plate[:,0]>1919)or np.any(plate[:,1]<3)or np.any(plate[:,1]>1076):continue
   if uv0[0]<4 or uv0[0]>650 or uv0[1]<270 or uv0[1]>1076:continue
   lc=int(left_extent*2);lo=float(np.median(v[j-lc:j-max(5,lc//2)]));hi=float(np.median(v[j+22:j+50]));contrast=hi-lo
   if not(6<contrast and 50<lo<155 and hi>135):continue
   if lo>=145 and contrast<25:continue # weak near-white ice gradient does not prove gray cap material
   cs=[]
   for q in(.1,.5,.9):
    target=lo+q*contrast;cross=[k for k in range(j-50,j+50)if v[k]<=target<v[k+1]]
    if not cross:break
    k=min(cross,key=lambda k:abs(k-j));cs.append(float(ds[k]+(target-v[k])/(v[k+1]-v[k])*.5))
   if len(cs)!=3 or not(cs[0]<=cs[1]<=cs[2])or cs[2]-cs[0]>40:continue
   uv=np.array([x,y])+cs[1]*n;a=np.array([x,y])+cs[0]*n;z=np.array([x,y])+cs[2]*n
   # Complete source transition support must lie in the actual source frame, and outside persons/UI/unknown.
   sam=np.array([a+(z-a)*tt for tt in np.linspace(0,1,max(3,int(np.linalg.norm(z-a))*2+1))]);iv=np.round(sam).astype(int)
   if np.any(iv[:,0]<0)or np.any(iv[:,0]>=1920)or np.any(iv[:,1]<0)or np.any(iv[:,1]>=1080):continue
   if np.any(guard[iv[:,1],iv[:,0]]):continue
   plate_support=np.stack([uv0+tt*n for tt in np.arange(-left_extent,25.01,.5)]);pi=np.round(plate_support).astype(int)
   if np.any(guard[pi[:,1],pi[:,0]]):continue
   quality=abs(cs[1])/14 + max(0,14-contrast)/8
   opts.append((quality,dict(uv=uv.tolist(),normal=n.tolist(),interval=[a.tolist(),z.tolist()],contrast=contrast,gray_plateaus=[lo,hi],normal_transition_width=float(cs[2]-cs[0]),direct=contrast>=12 and cs[2]-cs[0]<=24 and left_extent>=12,material_plateau_support_endpoints_uv=plate.tolist(),left_plateau_extent=left_extent)))
  if opts:pts.append(min(opts,key=lambda x:x[0])[1])
  else:unknown.append(dict(navigation_y=y,navigation_x=x,reason='no_complete_current_gray_to_lighter_material_transition_or_occluded'))
 return pts,unknown
c=av.open(cfg['source']['file']);s=c.streams.video[0];c.seek(1301248,stream=s,backward=True);index=[];thumbs=[];states={};started=time.perf_counter();pcounts=[]
for f in c.decode(s):
 p=f.pts
 if p<1301248:continue
 if p>1425408:break
 b=f.to_ndarray(format='bgr24');r=mi[p];bsha=hashlib.sha256(b.tobytes()).hexdigest();assert bsha==r['source_BGR_sha256'];rgbsha=hashlib.sha256(cv2.cvtColor(b,cv2.COLOR_BGR2RGB).tobytes()).hexdigest();assert rgbsha==r['source_RGB_sha256']
 person=cv2.imread(r['mask_file'],0)>0;assert sha(r['mask_file'])==r['mask_sha256'];guard=cv2.dilate(person.astype('uint8'),np.ones((17,17),np.uint8))>0;uref=r.get('source_unknown_exclusion_mask_ref');
 if uref:
  assert sha(uref['file'])==uref['sha256'];um=cv2.imread(uref['file'],0)>0;guard|=cv2.dilate(um.astype('uint8'),np.ones((17,17),np.uint8))>0
 gm,grec=current_graphics(b);guard|=gm;m,bb,bottom=nav(p);pts,unknown=profiles(b,m,bb,guard)if bottom>-45 else([],[])
 # Refit navigation only from actual current source half-transitions; recompute all UV from raw pixels once more.
 if len(pts)>=8:
  v=np.array([q['uv']for q in pts]);fit=np.polyfit(v[:,1],v[:,0],1);res=v[:,0]-np.polyval(fit,v[:,1]);keep=np.abs(res-np.median(res))<max(4,2*np.median(np.abs(res-np.median(res))));
  if keep.sum()>=6:
   fit=np.polyfit(v[keep,1],v[keep,0],1)
   if .2<fit[0]<.85:pts,unknown=profiles(b,float(fit[0]),float(fit[1]),guard)
 # Reject isolated or source-wrong points that do not follow this actual long edge. No rejected point is moved.
 if len(pts)>=4:
  v=np.array([q['uv']for q in pts]);fit=np.polyfit(v[:,1],v[:,0],1);res=v[:,0]-np.polyval(fit,v[:,1]);mad=float(np.median(np.abs(res-np.median(res))));keep=np.abs(res-np.median(res))<max(5,3*mad);pts=[q for q,k in zip(pts,keep)if k]
 state='visible'if pts else 'absent_no_source_qualified_near_boundary';near=np.zeros((1080,1920),np.uint8);bound=None
 if pts:
  v=np.array([q['uv']for q in pts]);order=np.argsort(v[:,1]);v=v[order];xp=v[:,0];yp=v[:,1];fit=np.polyfit(yp,xp,1)if len(v)>=2 else np.array([m,bb]);ys=np.arange(1080);xs=np.interp(ys,yp,xp);xs[ys<yp[0]]=np.polyval(fit,ys[ys<yp[0]]);xs[ys>yp[-1]]=np.polyval(fit,ys[ys>yp[-1]]);width=max(q['normal_transition_width']for q in pts)/2;xs+=max(2,min(15,width));xs=np.clip(xs,0,1919);near=(np.arange(1920)[None,:]<=xs[:,None]).astype('uint8');near[xs<1,:]=0;bound=np.stack([xs,ys],axis=1)[::6].tolist()
 elif bottom>-10:
  # Only exclusion/navigation at mixed image borders; no fake measured row. Actual left-bottom pixel gray component recorded.
  corner=cv2.cvtColor(b[-12:,:12],cv2.COLOR_BGR2GRAY);ice_reference=float(np.median(cv2.cvtColor(b[-12:,30:60],cv2.COLOR_BGR2GRAY)));dark=int((corner<ice_reference-6).sum());
  if dark>0:
   state='border_mixed_unknown_no_complete_material_profile';xs=np.clip(m*np.arange(1080)+bb+6,0,1919);near=(np.arange(1920)[None,:]<=xs[:,None]).astype('uint8');near[xs<1,:]=0
 anref=bc[p];an=json.load(open(anref['anchored_file'],encoding='utf8'));assert sha(anref['anchored_file'])==anref['anchored_file_sha256'];assert an['decoded_native_BGR_sha256']==bsha
 foot=np.array(an['roles']['board.contact']['full_native_y'],dtype=float);good=np.isfinite(foot);foot=np.interp(np.arange(1920),np.flatnonzero(good),foot[good])
 ice=(np.arange(1080)[:,None]>=np.ceil(foot)[None,:])&(~guard)&(~near.astype(bool))
 ad=json.load(open(Path(am['consumer_frame_directory'])/f'{p}.json',encoding='utf8'));bodyref=ad['body_material_ref'];body=json.load(open(bodyref['file'],encoding='utf8'));assert sha(bodyref['file'])==bodyref['sha256'];bmr=body['body_material_mask_ref'];assert sha(bmr['file'])==bmr['sha256'];assert body['decoded_native_RGB_sha256']==rgbsha;bodymask=(cv2.imread(bmr['file'],0)>0)&(~guard)&(~near.astype(bool))
 refs={}
 for kind,mask in [('near_outside_exclusion',near>0),('ice_visible',ice),('board_body_visible',bodymask)]:
  q=T/'masks'/f'{p}_{kind}.png';cv2.imwrite(str(q),mask.astype('uint8')*255);refs[kind+'_mask_ref']=dict(file=str(q),sha256=sha(q))
 new=[]
 for j,q in enumerate(pts):
  new.append(dict(source_row_id=f'near56/s06/{p}/top/{j}',shot_id='s06',native_pts=p,time_base='1/12800',feature_id='board.top',role='board.top',branch_id='near_lower_long_side',segment_id=f's06/{p}/board.top/near_lower_long_side',uv=q['uv'],definition='ice_facing_boundary_of_gray_board_top_surface_same_zero_thickness_board_upper_boundary_as_far_blue_upper_edge',source_kind='actual_current_native_RGB_gray_cap_half_material_transition'if q['direct']else'weak_actual_current_native_RGB_gray_cap_half_material_transition',visibility='visible',occlusion=False,source_UV_is_direct_measurement=True,pixels_measured_in_this_frame=True,source_evidence_measured_in_this_frame=True,current_native_pixel_profile_queried=True,weak_source=not q['direct'],model_or_world_projection_used=False,samples_independent=False,decoded_native_BGR_sha256=bsha,pixel_support=dict(normal=q['normal'],position_interval_endpoints_uv=q['interval'],gray_plateaus=q['gray_plateaus'],material_contrast=q['contrast'],interval_is_not_sigma=True,current_pixel_profile_reread=True,normal_only=True,material_plateau_support_endpoints_uv=q['material_plateau_support_endpoints_uv'],left_plateau_extent_native_px=q['left_plateau_extent'])))
 fj=dict(schema='rinkspace.current_source_near_board_domains.v1',shot_id='s06',native_pts=p,time_base='1/12800',source_BGR_sha256=bsha,source_RGB_sha256=rgbsha,state=state,near_top_branches=[dict(branch_id='near_lower_long_side',feature_id='board.top',polyline_uv=[q['uv']for q in pts],point_count=len(pts),definition='same_zero_thickness_board_upper_boundary_as_far_blue_upper_edge')]if pts else[],new_source_rows=new,source_unknown_scan_rows=unknown,near_boundary_exclusion_polyline=bound,near_exclusion_interpolation_is_negative_visibility_only=True,near_top_surface_and_outside_equipment_are_not_advertisement_or_ice=True,near_vertical_advertisement_face_visible=False,source_navigation=dict(method='current_source_Hough_and_adjacent_current_source_material_profiles_navigation_only',m=m,b=bb,bottom=bottom,used_as_positive_observation=False),old_far_board_source_ref=dict(file=anref['anchored_file'],sha256=anref['anchored_file_sha256']),old_board_body_material_ref=bodyref,old_board_body_mask_ref=bmr,current_person_mask_ref=dict(file=r['mask_file'],sha256=r['mask_sha256']),current_source_unknown_exclusion_mask_ref=uref,source_graphics_receipt=grec,person_and_unknown_guard_native_px=8,source_UV_moved=False,physical_world_or_station_issued=False,**refs)
 fp=T/'frames'/f'{p}.json';write(fp,fj);index.append(dict(native_pts=p,shot_id='s06',source_BGR_sha256=bsha,source_RGB_sha256=rgbsha,state=state,near_top_points=len(new),frame_file=str(fp),frame_sha256=sha(fp),**refs));pcounts.append(len(new));states[state]=states.get(state,0)+1
 if pts or state.startswith('border'):
  im=b.copy();tint=np.zeros_like(im);tint[:,:,2]=150;im[near>0]=cv2.addWeighted(im,.7,tint,.3,0)[near>0]
  for q in pts:cv2.circle(im,tuple(np.round(q['uv']).astype(int)),3,(0,255,0)if q['direct']else(0,120,0),-1)
  native_overlay=im.copy();im=cv2.resize(im,(640,360));cv2.putText(im,str(p)+' '+state,(5,345),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,255,255),1);zoom=cv2.resize(native_overlay[250:1080,:700],(560,664));zoom=cv2.copyMakeBorder(zoom,0,30,0,0,cv2.BORDER_CONSTANT,value=(0,0,0));cv2.putText(zoom,str(p)+' '+state,(4,686),cv2.FONT_HERSHEY_SIMPLEX,.43,(0,255,255),1);thumbs.append((p,zoom))
  if p in [1334016,1334272,1339648,1342976,1346048,1351424,1351680]:cv2.imwrite(str(T/'QA'/f'{p}.jpg'),b,[cv2.IMWRITE_JPEG_QUALITY,96]);cv2.imwrite(str(T/'QA'/f'{p}_overlay.jpg'),native_overlay,[cv2.IMWRITE_JPEG_QUALITY,96])
 if len(index)%100==0:print(json.dumps({'frames':len(index),'states':states,'rows':sum(pcounts),'elapsed':time.perf_counter()-started}),flush=True)
with gzip.open(T/'index.jsonl.gz','wt',encoding='utf8',compresslevel=6)as f:
 for r in index:f.write(json.dumps(r,separators=(',',':'))+'\n')
for page in range((len(thumbs)+5)//6):
 canvas=np.zeros((2082,1120,3),np.uint8)
 for j,(p,im)in enumerate(thumbs[page*6:page*6+6]):canvas[(j//2)*694:(j//2+1)*694,(j%2)*560:(j%2+1)*560]=im
 cv2.imwrite(str(T/'QA'/f'page_{page:02d}.jpg'),canvas,[cv2.IMWRITE_JPEG_QUALITY,92])
write(T/'QA'/'index.json',dict(frames=[p for p,im in thumbs],pages=(len(thumbs)+5)//6))
write(T/'manifest.json',dict(schema='rinkspace.current_source_near_board_domains.v1',complete=True,accepted_for_final_source_qualification=False,frames=len(index),new_source_rows=sum(pcounts),states=states,index=dict(file=str(T/'index.jsonl.gz'),sha256=sha(T/'index.jsonl.gz')),source=cfg['source'],source_reader=dict(file=str(Path(__file__)),sha256=sha(__file__)),source_navigation_positive_observation=False,all486_current_source_decoded=True,source_UV_moved=False,physical_model_accepted=False))
print(json.dumps({'complete':True,'frames':len(index),'states':states,'new_source_rows':sum(pcounts),'qa_pages':(len(thumbs)+5)//6}),flush=True)
