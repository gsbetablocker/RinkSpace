import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import av,cv2,numpy as np,json,gzip,hashlib,time,sys,copy
from pathlib import Path
from scipy.ndimage import median_filter,gaussian_filter1d
P=Path(r'D:/OneDrive/codex_projs/RinkSpace');D=Path(r'D:/CodexData/RinkSpace_Data');R=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2');J=D/'runs/okada_joint9_1018a';O=R/'dense56_board_v1/material_v5';M=R/'dense56_masks_v1'
cv2.setNumThreads(1)
sys.path.insert(0,str(R/'dense56_v1'))
from source_graphics import current_graphics
PTS=list(range(948992,1425408+1,256))
def readj(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,r):
 p=Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(r,ensure_ascii=False,separators=(',',':'),allow_nan=False),encoding='utf8');t.replace(p)
def sid(p):return 's05' if p<1301248 else 's06'
def caption_mask():
 m=np.zeros((1080,1920),bool);m[28:102,32:494]=True;m[106:218,80:394]=True;m[224:255,80:272]=True;m[978:1034,1674:1864]=True;return m
def mask_for(p,wait=False):
 mp=M/'masks'/(str(p)+'.png')
 while wait and not mp.exists():time.sleep(.15)
 if mp.exists():
  m=cv2.imread(str(mp),0)
  if m is not None:return m>0,str(mp),'actual_current_native_person_pixel_mask'
 for a in ANCHORS.get(sid(p),[]):
  if a['native_pts']==p:
   mp=Path(a['source'].get('mask_path',''))
   if mp.is_file():return cv2.imread(str(mp),0)>0,str(mp),'existing_native_keyframe_person_pixel_mask'
 return np.zeros((1080,1920),bool),None,'navigation_only_mask_pending'
def find_base(ch,role,g,force_guide=False,rgb=None,exclude=None,physical_top=None):
 h,w=ch.shape;xs=np.arange(w);sm=cv2.GaussianBlur(ch,(3,3),.7);gr=np.gradient(sm,axis=0);exclude=np.zeros((h,w),bool)if exclude is None else exclude;navigation_exclude=cv2.dilate(exclude.astype('u1'),np.ones((33,33),'u1'))>0
 if role=='board.top':
  core=cv2.dilate(sm,np.ones((7,1),'u1'),anchor=(0,0));following=np.roll(cv2.erode(sm,np.ones((29,1),'u1'),anchor=(0,0)),-4,axis=0);thin=(core>8)&(following<.5*core)&((core-following)>7);prior=np.roll(sm,5,axis=0);score=np.maximum(gr,0)*thin*(prior<.7*core);binary=(score>1)&~navigation_exclude
 else:
  material=(sm<-3)&(rgb[:,:,1]-rgb[:,:,2]>2)&(rgb[:,:,0]-rgb[:,:,1]>-8)&(rgb[:,:,0]-rgb[:,:,1]<60)&(rgb[:,:,0]-rgb[:,:,2]>4)&((rgb[:,:,1]-rgb[:,:,2])>.62*(rgb[:,:,0]-rgb[:,:,2]));material&=~navigation_exclude
  if physical_top is not None:material&=np.arange(h)[:,None]>np.asarray(physical_top)[None,:]+25
  # True yellow strip has local excess over the neighboring white board/ice; warm broad scene color is not the strip.
  above=np.roll(cv2.blur(sm,(1,12),anchor=(0,0)),18,axis=0);below=np.roll(cv2.blur(sm,(1,12),anchor=(0,0)),-18,axis=0);excess=-sm-np.maximum(-above,-below);material&=excess>2.
  longrun=cv2.morphologyEx(material.astype('u1'),cv2.MORPH_OPEN,np.ones((41,1),'u1'))>0;material&=~longrun
  support=cv2.dilate(material.astype('u1'),np.ones((10,1),'u1'),anchor=(0,9))>0;score=np.maximum(gr,0)*support;binary=material
 lines=cv2.HoughLinesP(binary.astype('u1')*255,1,np.pi/720,75,minLineLength=300,maxLineGap=65);opts=[]
 if lines is not None:
  for x0,y0,x1,y1 in lines[:,0]:
   if abs(x1-x0)<250 or abs(y1-y0)>.35*abs(x1-x0)or min(y0,y1)<180:continue
   nav=y0+(xs-x0)*(y1-y0)/(x1-x0);dist=float(np.median(abs(nav-np.asarray(g))));opts.append((abs(x1-x0)/(1+dist/150.)**2,[int(x0),int(y0),int(x1),int(y1)],nav))
 if not opts:raise ValueError('actual current whole-width material band unavailable '+role)
 _,line,nav=max(opts,key=lambda a:a[0]);radius=95 if role=='board.contact'else 65
 if role=='board.contact':
  yellow=cv2.morphologyEx(material.astype('u1'),cv2.MORPH_CLOSE,np.ones((3,3),'u1'));n,labels,stats,_=cv2.connectedComponentsWithStats(yellow,8);chosen=[]
  for k in range(1,n):
   x0,y0,ww,hh,area=stats[k]
   if ww<40 or area<20 or hh>max(18,.5*ww):continue
   yy0,xx0=np.where(labels[y0:y0+hh,x0:x0+ww]==k);xx0=xx0+x0;yy0=yy0+y0;near=abs(yy0-nav[xx0])<65
   if np.unique(xx0[near]).size>=8:chosen.append(k)
  # Build identity from ALL current thin strip pieces before deciding which local pieces belong.
  # A single 250px fragment must not force every weak column to its height across a masked gap.
  scaffold=[k for k in chosen if stats[k,2]>=60];main=np.isin(labels,scaffold);has=main.any(axis=0);head=np.argmax(main,axis=0);spine=np.interp(xs,xs[has],head[has])if has.sum()>1 else nav.copy();spine=median_filter(spine,257,mode='nearest');dominant=[];short=[]
  for k in chosen:
   x0,y0,ww,hh,area=stats[k];yy0,xx0=np.where(labels[y0:y0+hh,x0:x0+ww]==k);xx0=xx0+x0;yy0=yy0+y0
   if np.unique(xx0[abs(yy0-spine[xx0])<10]).size>=3:dominant.append(k)
  chosen=dominant+short;connected=np.isin(labels,chosen);support=cv2.dilate(connected.astype('u1'),np.ones((10,1),'u1'),anchor=(0,9))>0;score=np.maximum(gr,0)*support
 if role=='board.top':
  blue=binary;blue=cv2.morphologyEx(blue.astype('u1'),cv2.MORPH_CLOSE,np.ones((3,3),'u1'));n,labels,stats,_=cv2.connectedComponentsWithStats(blue,8);chosen=[]
  for k in range(1,n):
   x0,y0,ww,hh,area=stats[k]
   if ww<150 or area<15 or area/max(ww,1)>35:continue
   yy0,xx0=np.where(labels[y0:y0+hh,x0:x0+ww]==k);xx0=xx0+x0;yy0=yy0+y0;near=abs(yy0-nav[xx0])<13
   if np.unique(xx0[near]).size>=8:chosen.append(k)
  connected=np.isin(labels,chosen);connected_ahead=cv2.dilate(connected.astype('u1'),np.ones((7,1),'u1'),anchor=(0,0))>0;score*=connected_ahead
 band_present=connected.any(axis=0);top=np.argmax(connected,axis=0).astype(float);top[~band_present]=np.nan;goodband=np.flatnonzero(band_present);band_top=np.interp(xs,goodband,top[goodband])if len(goodband)>1 else nav.copy();band_top=median_filter(band_top,size=257,mode='nearest')if role=='board.top'else band_top;off=np.arange(-8,65 if role=='board.contact'else 20)[:,None];xx=xs[None,:];yy=np.clip(np.rint(band_top).astype(int)[None,:]+off,2,h-3);local=score[yy,xx];allowed=~navigation_exclude[yy,xx]&band_present[None,:];local=np.where(allowed,local,0.);peaks=(local>=score[np.clip(yy-1,0,h-1),xx])&(local>=score[np.clip(yy+1,0,h-1),xx])&(local>.35)&allowed
 if role=='board.top':idx=np.argmax(np.where(peaks,local,0.),axis=0);valid=peaks.any(axis=0)
 else:
  mx=np.max(local,axis=0);peaks&=local>=.65*mx[None,:];idx=np.argmax(peaks,axis=0);valid=peaks.any(axis=0)&(mx>.35)
 yp=yy[idx,xs].astype(float);expected=np.interp(xs,xs[valid],yp[valid])if valid.sum()>1 else nav.copy();expected=median_filter(expected,size=51,mode='nearest');missing=~valid
 # Weak visible columns are locally reread around this SAME FRAME's connected material support.
 for x in np.flatnonzero(missing):
  lo=max(2,int(round(expected[x]))-6);hi=min(h-3,int(round(expected[x]))+7);z=np.arange(lo,hi);v=np.maximum(gr[z,x],0)-.03*(z-expected[x])**2;yp[x]=z[int(np.argmax(v))]
 local_reread=[];middle=median_filter(yp,size=129,mode='nearest');bad=abs(yp-middle)>4
 for x0 in np.flatnonzero(bad):
  x=int(x0);old=float(yp[x]);lo=max(2,int(round(middle[x]))-5);hi=min(h-2,int(round(middle[x]))+6);z=np.arange(lo,hi);physical=score[z,x];eligible=(physical>.35)&~exclude[z,x]
  if eligible.any():v=np.where(eligible,physical-.03*(z-middle[x])**2,-1e6);yy1=int(z[np.argmax(v)]);valid[x]=True
  else:v=np.maximum(gr[z,x],0)-.1*(z-middle[x])**2;yy1=int(z[np.argmax(v)]);valid[x]=False
  yp[x]=float(yy1);local_reread.append(dict(x=x,original_candidate_uv=[x,old],current_native_requalified_uv=[x,float(yy1)],current_same_material_support=bool(valid[x]),navigation_only_from_current_connected_band=True,source_curve_not_emitted_as_UV=True))
 yi=np.clip(np.rint(yp).astype(int),1,h-2)
 for x in range(w):
  y=int(yi[x]);z=gr[y-1:y+2,x]
  if len(z)==3 and z[0]-2*z[1]+z[2]<-1e-5:yp[x]+=float(np.clip(.5*(z[0]-z[2])/(z[0]-2*z[1]+z[2]),-.5,.5))
 grad=gr[yi,xs];before=sm[np.clip(yi-5,0,h-1),xs];after=sm[np.clip(yi+5,0,h-1),xs];di=valid&(grad>1)&((after>8)&(before<.7*core[yi,xs])if role=='board.top'else(before<-8));ey=np.clip(np.rint(expected).astype(int),0,h-1);occluded=exclude[ey,xs]|exclude[yi,xs];meta=dict(selected_material_component_stats=[stats[k].tolist()for k in chosen],current_native_full_width_material_band=True,hough_line_navigation_only=line,source_shape_extrapolated_for_newly_visible_columns=False,current_material_columns=int(valid.sum()),weak_same_frame_connected_material_columns=int(missing.sum()),full_material_occlusion=occluded.tolist(),source_UV_from_curve_fit=False,navigational_mask_halo_px=16,final_exclusion_uses_actual_mask_not_navigational_halo=True,current_yellow_local_excess_and_thin_run_identity=role=='board.contact',current_cap_below_board_relation_required=physical_top is not None,rejected_local_material_candidates_and_actual_current_reread=local_reread)
 if role=='board.top':
  cy=np.clip(np.rint(yp).astype(int),2,h-42);dy=np.arange(2,41)[:,None];ly=cy[None,:]+dy;lx=xs[None,:];blue_before=sm[np.clip(ly-5,0,h-1),lx];white_after=sm[np.clip(ly+5,0,h-1),lx];body=(blue_before>8)&(abs(white_after)<12)&(white_after<.5*blue_before);neg=np.maximum(-gr[ly,lx],0);s=np.where(body,neg,0.);ix=np.argmax(s,axis=0);loy=ly[ix,xs].astype(float);li=loy.astype(int);ld=(s[ix,xs]>1)&~exclude[li,xs]
  for d in range(-5,6):ld&=~exclude[np.clip(li+d,0,h-1),xs]
  widths=[max(1.,6/max(float(neg[ix[x],x]),.5))if ld[x]else None for x in range(w)]
  meta.update(full_blue_cap_lower_y=loy.tolist(),full_blue_cap_lower_direct=ld.tolist(),full_blue_cap_lower_localization_radius=widths,cap_body_interface_only_for_source_domain=True,cap_body_interface_not_exported_as_fitting_rows=True,unknown_lower_interface_not_automatically_body=True)
 return yp,di,grad,before,after,expected,meta
ANCHORS={}
for s,d in [('s05','shot5_20261014'),('s06','shot6_20261014')]:
 aa=[]
 for f in (P/'events/ankara_2026/programmes/okada_mei_sp/source_datasets'/d/'frames').glob('*.json'):
  q=readj(f);pp=int(q['source'].get('native_pts',q['source'].get('actual_pts')));curves={};orig={};manual=[]
  for role,token in [('board.top','board_top'),('board.contact','board_contact')]:
   ar=[];nr=0
   for ft in q.get('features',[]):
    if not ft.get('id','').startswith(token):continue
    for k,seg in enumerate(ft.get('segments',[])):
     a=seg.get('source_native_readout_uv')or seg.get('uv')or[];ar.extend(a);nr+=len(seg.get('source_native_readout_uv')or[])
     if 'user' in str(seg.get('measurement_origin','')).lower() or 'manual' in str(seg.get('measurement_origin','')).lower():manual.append(dict(feature_id=ft['id'],segment_index=k,uv=seg.get('uv'),measurement_origin=seg.get('measurement_origin')))
   ar=np.array(ar,float).reshape(-1,2)
   if len(ar)<2:raise ValueError('source board role missing '+str(f)+' '+role)
   ar=ar[np.argsort(ar[:,0])];ux=np.unique(ar[:,0]);uy=np.array([np.median(ar[ar[:,0]==x,1])for x in ux]);curve=np.interp(np.arange(1920),ux,uy)
   # The navigation profile only; real output is independently reread in each current native image.
   outside=(np.arange(1920)<ux[0])|(np.arange(1920)>ux[-1]);curve[outside]=float(np.median(uy));curves[role]=curve
   orig[role]=dict(original_source_readouts=nr,total_navigation_points=len(ar),source_x_span=[float(ux[0]),float(ux[-1])],source_segments=[dict(feature_id=ft['id'],segment_index=k,measurement_origin=seg.get('measurement_origin'),original_uv_preserved=True,source_native_readout_count=len(seg.get('source_native_readout_uv')or[]),source_x_span=[float(min(v[0]for v in (seg.get('source_native_readout_uv')or seg.get('uv')))),float(max(v[0]for v in (seg.get('source_native_readout_uv')or seg.get('uv'))))])for ft in q.get('features',[])if ft.get('id','').startswith(token)for k,seg in enumerate(ft.get('segments',[]))if(seg.get('source_native_readout_uv')or seg.get('uv'))])
  aa.append(dict(native_pts=pp,file=str(f),sha256=sha(f),curves=curves,source_origin=orig,source=q['source'],manual_annotations=manual))
 ANCHORS[s]=sorted(aa,key=lambda a:a['native_pts'])
def motion_navigation(prev,im,p):
 pr,pi=prev;pg=cv2.cvtColor(pi,cv2.COLOR_BGR2GRAY);cg=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);a=np.array(pr['roles']['board.top']['full_native_y']);b=np.array(pr['roles']['board.contact']['full_native_y']);yy=np.arange(1080)[:,None];band=((yy>a[None,:]+10)&(yy<b[None,:]-10));pm=mask_for(pr['native_pts'],True)[0];pm=pm|current_graphics(pi)[0];band&=~pm;p0=cv2.goodFeaturesToTrack(pg,maxCorners=500,qualityLevel=.01,minDistance=12,mask=(band*255).astype('u1'));H=None
 rec=dict(previous_native_pts=pr['native_pts'],current_native_pts=p,source_kind='actual_static_board_body_2D_texture_LK_forward_backward_affine_navigation',cross_shot=False,source_navigation_only=True,fit_UV_transferred=False,measurement_UV_feedback_into_navigation=False,caption_pixels_excluded=True,full_anchor_shape_transformed_without_global_polynomial=True)
 if p0 is not None and len(p0)>=8:
  p1,st,_=cv2.calcOpticalFlowPyrLK(pg,cg,p0,None,winSize=(25,25),maxLevel=4,criteria=(3,30,.01));pb,bs,_=cv2.calcOpticalFlowPyrLK(cg,pg,p1,None,winSize=(25,25),maxLevel=4,criteria=(3,30,.01));fb=np.linalg.norm(pb-p0,axis=2).ravel();good=st.ravel().astype(bool)&bs.ravel().astype(bool)&(fb<1.5);q=p1.reshape(-1,2);qm=np.rint(q).astype(int);inside=(qm[:,0]>=0)&(qm[:,0]<1920)&(qm[:,1]>=0)&(qm[:,1]<1080);cm=mask_for(p,True)[0]|current_graphics(im)[0];good&=inside;good[inside]&=~cm[qm[inside,1],qm[inside,0]]
  if good.sum()>=8:
   A,ins=cv2.estimateAffinePartial2D(p0.reshape(-1,2)[good],p1.reshape(-1,2)[good],method=cv2.RANSAC,ransacReprojThreshold=2.5,maxIters=500,confidence=.99,refineIters=5)
   if A is not None and int(ins.sum())>=12 and int(ins.sum())>=.65*int(good.sum()) and .94<np.hypot(A[0,0],A[1,0])<1.06 and abs(A[1,2])<40:H=np.vstack([A,[0.,0.,1.]])
   rec.update(candidate_tracks=len(p0),forward_backward_valid=int(good.sum()),inliers=int(ins.sum())if ins is not None else 0,source_uv=p0.reshape(-1,2)[good].tolist(),current_uv=p1.reshape(-1,2)[good].tolist(),forward_backward_error_px=fb[good].tolist())
 out={};x=np.arange(1920);validout={}
 for role,rr in pr['roles'].items():
  y=np.array(rr['full_navigation_y']);valid=np.array(rr.get('full_navigation_valid_x',[True]*1920),bool);xy=np.c_[x,y]
  if H is not None and valid.sum()>=2:
   q=cv2.perspectiveTransform(xy[valid].reshape(-1,1,2).astype('f8'),H).reshape(-1,2);order=np.argsort(q[:,0]);qx=q[order,0];qy=q[order,1];nav=np.interp(x,qx,qy);v=(x>=qx[0])&(x<=qx[-1]);nav[~v]=float(np.median(qy));out[role]=nav;validout[role]=v
  else:out[role]=y.copy();validout[role]=valid.copy()
 rec['full_navigation_valid_x']={ro:v.tolist()for ro,v in validout.items()};rec['old_KF_end_slope_extrapolated_for_new_columns']=False
 rec['image_navigation_H']=H.tolist()if H is not None else None;rec['fallback']='same_shot_previous_actual_material_shape_then_current_local_pixel_search'if H is None else None
 return out,rec
def anchor_for(p):return min(ANCHORS[sid(p)],key=lambda a:abs(a['native_pts']-p))
def anchored_guide(p,role):
 a=anchor_for(p);return a['curves'][role].copy(),dict(native_pts=a['native_pts'],file=a['file'],sha256=a['sha256'],source_origin=a['source_origin'],source_curve_valid_x_span=a['source_origin'][role]['source_x_span'],source_curve_navigation_only=True,source_original_UV_modified=False,current_native_RGB_read_required=True,PTS_interpolation_used=False)
def read_frame(p,im,prev=None):
 rgb=im[:,:,::-1].astype('f4');ch=rgb[:,:,2]-(rgb[:,:,0]+rgb[:,:,1])/2;mask,mp,mk=mask_for(p,True);graphics,graphics_receipt=current_graphics(im);exclusion=mask|graphics;roles={};proofs={};motion=None;proposal=None
 exact=next((a for a in ANCHORS[sid(p)]if a['native_pts']==p),None)
 if prev is not None and prev[0]['shot_id']==sid(p):proposal,motion=motion_navigation(prev,im,p)
 for role in ['board.top','board.contact']:
  g,ap=anchored_guide(p,role);source_span=ap['source_curve_valid_x_span'];navvalid=(np.arange(1920)>=source_span[0])&(np.arange(1920)<=source_span[1])
  if not exact and proposal is not None:g=proposal[role];ap=copy.deepcopy(prev[0]['keyframe_navigation'][role]);navvalid=np.array(motion['full_navigation_valid_x'][role],bool)
  ap.update(exact_keyframe_reset=exact is not None,actual_adjacent_board_motion_consumed=not exact and proposal is not None,current_native_pts=p,measurement_UV_feedback_into_navigation=False,source_anchor_shape_propagated_independently=True)
  yp,di,grad,before,after,curve,material_meta=find_base(ch,role,g,force_guide=True,rgb=rgb,exclude=exclusion,physical_top=np.array(roles['board.top']['full_native_y'])if role=='board.contact'else None)
  # Out-of-image source guides cannot be clipped and turned into synthetic material observations.
  out=(curve<-1)|(curve>1080);yp[out]=curve[out];di[out]=False
  line=material_meta['hough_line_navigation_only'];x0,y0,x1,y1=line;current_band_line=y0+(np.arange(1920)-x0)*(y1-y0)/(x1-x0);g[~navvalid]=current_band_line[~navvalid];ap.pop('source_origin',None);ap['source_original_detail_in_anchor_recipe']=str(O/'source_anchor_recipe.json')
  roles[role]=dict(material_band=material_meta,full_native_x=list(range(1920)),full_navigation_valid_x=navvalid.tolist(),full_navigation_y=g.tolist(),full_native_y=yp.tolist(),full_direct=di.tolist(),full_gradient=grad.tolist(),full_chroma_before=before.tolist(),full_chroma_after=after.tolist(),navigation_native_pts=p,navigation_source_kind='existing_keyframe_reset_or_adjacent_static_board_texture_navigation_then_current_native_material_reread',current_native_material_reread=True,source_fitted_curve_not_output_UV=True)
  proofs[role]=ap
 if motion is not None:motion.pop('full_navigation_valid_x',None)
 hh=hashlib.sha256(im.tobytes()).hexdigest()
 if exact and hh!=exact['source']['decoded_BGR_sha256']:raise ValueError('native BGR identity differs from original keyframe '+str(p))
 return dict(schema='rinkfit.dense_native_board_true_material_boundaries_source_frame.v1',native_pts=p,shot_id=sid(p),rig_id='end'if sid(p)=='s05'else'secondary',time_base='1/12800',decoded_native_BGR_sha256=hh,roles=roles,rows=[],keyframe_navigation=proofs,temporal_source_check=motion,mask_path=mp,mask_kind=mk,graphics_receipt=graphics_receipt,source_video_sha256=readj(J/'programme_scope.json')['source']['sha256'],paired_blue_cap_and_white_board_body_semantics=True,yellow_contact_from_current_native_material_not_reflection=True,source_generation='s05_s06_full_current_material_component_source_v5')
def run_master(lo=None,hi=None,label=None):
 target=[p for p in PTS if(lo is None or p>=lo)and(hi is None or p<=hi)]
 (O/'anchored_frames').mkdir(exist_ok=True)
 if label is None or label=='s05a':write(O/'source_anchor_recipe.json',dict(schema='rinkfit.existing_keyframe_native_board_source_anchors.v1',shots={s:[{k:a[k]for k in ['native_pts','file','sha256','source_origin','manual_annotations']}for a in aa]for s,aa in ANCHORS.items()},anchor_count=sum(map(len,ANCHORS.values())),source_original_UV_preserved=True,current_RGB_reader_required=True,source_navigation_not_pixel_measurement=True,PTS_linear_interpolation_used=False,measurement_UV_feedback_into_navigation=False,navigation='complete original keyframe source shape propagated only by adjacent current actual board-body LK-FB affine navigation; full reset every original keyframe',graphics_exclusion=dict(file=str(R/'dense56_v1/source_graphics.py'),sha256=sha(R/'dense56_v1/source_graphics.py'),method='actual current panel presence plus shared current broadcast UI'),newly_visible_source='current complete native material components and profiles; source KF end slope is not extrapolated',local_top_selection='current connected blue cap upper exterior profile',local_contact_selection='current dominant connected yellow kickplate lower material profile; local branches are reread in current RGB, never curve UV',mask_navigation_halo_only_px=16,fitting_started=False,previous_final_reader=dict(file=str(R/'dense_board_v1/source_reader_anchored.py'),sha256=sha(R/'dense_board_v1/source_reader_anchored.py')),material_boundary_semantics=str(P/'events/ankara_2026/model_constraints/user_board_edge_semantics_20261010.json')))
 scope=readj(J/'programme_scope.json');t=time.perf_counter();done=[];counts={};prev=None
 with av.open(scope['source']['path'])as c:
  s=c.streams.video[0];c.seek(target[0],stream=s,backward=True)
  for f in c.decode(s):
   p=int(f.pts)
   if p<target[0]:continue
   if p>target[-1]:break
   im=f.to_ndarray(format='bgr24');dst=O/'anchored_frames'/(str(p)+'.json')
   if dst.exists():r=readj(dst)
   else:r=read_frame(p,im,prev);write(dst,r)
   prev=(r,im);done.append(p);counts[sid(p)]=counts.get(sid(p),0)+1
   if len(done)%25==0 or len(done)==1:
    q=dict(current_source_native_frames=len(done),last_pts=p,shots=counts,total=len(target),elapsed_seconds=time.perf_counter()-t);write(O/('anchored_progress'+('_'+label if label else '')+'.json'),q);print(json.dumps(q),flush=True)
 if done!=target:raise ValueError('missing native frame')
 write(O/('anchored_complete'+('_'+label if label else '')+'.json'),dict(complete=True,native_frames=len(done),shots=counts,reader_sha256=sha(__file__),source_video_sha256=scope['source']['sha256'],blue_cap_top_not_ad_top=True,yellow_kickplate_lower_not_reflection=True,source_original_UV_preserved=True,source_clock='1/12800',source_fps=50,quality_review_required_before_final_delivery=True))
if __name__=='__main__':
 if len(sys.argv)==4:run_master(int(sys.argv[1]),int(sys.argv[2]),sys.argv[3])
 else:run_master()
