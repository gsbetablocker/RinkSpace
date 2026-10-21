"""Source-only wide-context fixed-birth readout for one immutable physical Ad batch."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_key]='1'
import sys
sys.path[:0]=[r'D:/OneDrive/codex_projs/RinkSpace/calibration/rinkfit/src',r'D:/OneDrive/codex_projs/RinkSpace']
from pathlib import Path
from collections import defaultdict,Counter
import hashlib,json,gzip
import av,cv2,numpy as np
from rinkfit_joint.source_texture_tracklets import _source_mask
from rinkfit_joint.io_utils import read_jsonl,file_sha256
from rinkfit_joint.native_entity_tracklets import _ncc,NAVIGATION

def ref(path):
 p=Path(path).resolve();return dict(file=str(p),sha256=file_sha256(p))
def uvhash(uv):return hashlib.sha256(np.asarray(uv,dtype='<f8').tobytes()).hexdigest()
def patch(g,u,n):return cv2.getRectSubPix(g.astype(np.float32),(n,n),tuple(np.asarray(u,float)))
def support_bad(u,n,mask):
 r=n//2;x,y=np.asarray(u,float);h,w=mask.shape
 if x-r<0 or y-r<0 or x+r>=w or y+r>=h:return 'image_edge'
 a=np.floor([x-r,y-r]).astype(int);b=np.ceil([x+r,y+r]).astype(int)
 if mask[a[1]:b[1]+1,a[0]:b[0]+1].any():return 'actor_graphics_or_guard_support'
 return None

def register(template,current,old_uv,mask,n=97):
 r=n//2;old=np.asarray(old_uv,float)
 lo=np.maximum(np.floor(old-[r+160,r+40]).astype(int),[0,0]);hi=np.minimum(np.ceil(old+[r+160,r+40]+1).astype(int),[1920,1080]);b=current[lo[1]:hi[1],lo[0]:hi[0]].astype(np.float32)
 if min(b.shape)<n:return dict(status='search_too_small')
 z=cv2.matchTemplate(b,template,cv2.TM_CCOEFF_NORMED)
 # Actual mask support, including the entire wide patch, governs every candidate.
 good=cv2.erode((mask==0).astype(np.uint8),np.ones((n,n),np.uint8))
 mz=good[lo[1]+r:hi[1]-r,lo[0]+r:hi[0]-r];z[mz==0]=-2
 if z.max()<=-1:return dict(status='no_unmasked_wide_candidate',search_origin=lo.tolist(),search_shape=list(b.shape))
 _,peak,_,loc=cv2.minMaxLoc(z);x,y=loc;other=z.copy();other[max(0,y-31):y+32,max(0,x-31):x+32]=-2;ry,rx=np.unravel_index(other.argmax(),other.shape);runner=float(other[ry,rx]);w=np.array([[1,0,x],[0,1,y]],np.float32)
 out=dict(template_shape=[n,n],template_half=r,search_origin=lo.tolist(),search_shape=list(b.shape),translation_peak_local_top_left=[x,y],translation_peak_uv=(lo+[x+r,y+r]).tolist(),translation_peak_NCC=float(peak),other_branch_peak_NCC=runner,other_branch_peak_uv=(lo+[rx+r,ry+r]).tolist(),translation_branch_gap=float(peak-runner),branch_exclusion_halfwidth_xy=[31,31])
 try:cc,w=cv2.findTransformECC(template,b,w,cv2.MOTION_AFFINE,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,60,.0001),None,3)
 except cv2.error as e:return dict(out,status='affine_ECC_failed',error=str(e).split('\n')[0])
 uv=lo+w@np.array([r,r,1.]);corners=np.array([[0,0,1],[n-1,0,1],[n-1,n-1,1],[0,n-1,1]],float)@w.T+lo;warped=cv2.warpAffine(b,w,(n,n),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP);ncc=_ncc(template,warped);A=w[:,:2];sv=np.linalg.svd(A,compute_uv=False);bad=support_bad(uv,n,mask)
 status='registered'
 if not np.isfinite(w).all() or np.linalg.det(A)<=0:status='nonfinite_or_fold'
 elif np.any(corners.min(0)<[0,0]) or np.any(corners.max(0)>[1919,1079]):status='affine_support_leaves_image'
 elif bad:status='registered_'+bad
 elif np.any(abs(uv-(lo+[x+r,y+r]))>[31,5]):status='affine_changed_translation_branch'
 out.update(status=status,warp=w.astype(float).tolist(),registered_uv=uv.tolist(),registered_corners_native_uv=corners.tolist(),registered_NCC=float(ncc),affine_ECC=float(cc),affine_singular_values=sv.tolist(),affine_determinant=float(np.linalg.det(A)))
 return out

def run(config,original_rows,output):
 cfg=json.loads(Path(config).read_text(encoding='utf8'));rows=json.loads(Path(original_rows).read_text(encoding='utf8'));out=Path(output);out.mkdir(parents=True,exist_ok=True);src=cfg['source'];cv2.setNumThreads(1)
 if file_sha256(src['path'])!=src['sha256']:raise ValueError('Changed actual official source')
 bypts=defaultdict(list)
 for row in rows:bypts[int(row['native_pts'])].append(row)
 first,last=min(bypts),max(bypts);led={int(q['native_pts']):q for p in cfg['actor_ledgers'] for q in read_jsonl(p) if first<=int(q['native_pts'])<=last};roles=json.loads(Path(cfg['roles']).read_text(encoding='utf8'))['frame_roles'];birth={};records=[];stats=Counter();frame_receipt=[];images={};n=97;r=n//2
 with av.open(src['path']) as con:
  st=con.streams.video[0]
  if str(st.time_base)!='1/12800':raise ValueError('Wrong native time base')
  con.seek(first,stream=st,backward=True)
  for fr in con.decode(st):
   p=int(fr.pts)
   if p<first:continue
   if p>last:break
   if p not in bypts:continue
   if roles[str(p)]['role']!='development_candidate':raise ValueError('Protected exposure requested for readout')
   rgb=fr.to_ndarray(format='rgb24');g=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);mask,digest=_source_mask(rgb,led[p],src,p,'s04')
   for guv in cfg.get('guard_source_uv_by_pts',{}).get(str(p),[]):cv2.circle(mask,tuple(np.rint(guv).astype(int)),33,255,-1)
   for box in cfg.get('guard_support_bbox_by_pts',{}).get(str(p),[]):
    a,b,c,d=map(int,box);mask[max(0,b):min(1080,d+1),max(0,a):min(1920,c+1)]=255
   for reg in [s for s in cfg['regions'] if int(s['native_pts'])==p and s.get('native_person_mask')]:
    mr=reg['native_person_mask']
    if file_sha256(mr['file'])!=mr['sha256']:raise ValueError('Changed existing selected mask')
    mm=cv2.imread(mr['file'],cv2.IMREAD_GRAYSCALE);mask=np.maximum(mask,cv2.dilate((mm>0).astype(np.uint8)*255,np.ones((33,33),np.uint8)))
   frame_receipt.append(dict(native_pts=p,time_base=[1,12800],RGB_sha256=digest,BGR_sha256=hashlib.sha256(rgb[:,:,::-1].copy().tobytes()).hexdigest(),mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),row_count=len(bypts[p]),role=roles[str(p)]['role']))
   for row in bypts[p]:
    if digest!=row['current_rgb_sha256']:raise ValueError('Changed actual row RGB')
    tid=row['track_id'];uv=np.asarray(row['current_uv'],float);sid=row['source_row_id'];q=dict(source_row_id=sid,track_id=tid,surface_instance_id='Ad2',surface_class='board',native_pts=p,time_base=[1,12800],current_RGB_sha256=digest,source_video_sha256=src['sha256'],original_uv=uv.tolist(),original_uv_binary64_sha256=uvhash(uv),current_uv=uv.tolist(),source_UV_changed=False,model_or_world_projection_used=False)
    if tid not in birth:
     bad=support_bad(uv,n,mask);birth[tid]=dict(pts=p,uv=uv.copy(),RGB=digest,bad=bad,template=patch(g,uv,n) if not bad else None,gray=g.copy() if not bad else None,mask=mask.copy() if not bad else None)
     q.update(status='original_birth_reference' if not bad else 'unknown_birth_support',unknown_reason=bad,birth_native_pts=p,birth_uv=uv.tolist(),birth_RGB_sha256=digest,template_shape=[n,n],template_origin_native_uv=(uv-r).tolist())
    else:
     seed=birth[tid];q.update(birth_native_pts=seed['pts'],birth_uv=seed['uv'].tolist(),birth_RGB_sha256=seed['RGB'],template_shape=[n,n],template_origin_native_uv=(seed['uv']-r).tolist(),age_native_steps=(p-seed['pts'])//256)
     if seed['bad']:q.update(status='unknown_birth_support',unknown_reason=seed['bad'])
     else:
      f=register(seed['template'],g,uv,mask,n);q['forward']=f
      if f['status']!='registered':q.update(status='unknown_forward_registration',unknown_reason=f['status'])
      else:
       new=np.asarray(f['registered_uv']);now=patch(g,new,n);rev=register(now,seed['gray'],seed['uv'],seed['mask'],n);q['reverse']=rev;q['old_wide_context_NCC']=_ncc(seed['template'],patch(g,uv,n));q['new_minus_original_uv']=(new-uv).tolist();q['new_minus_original_px']=float(np.linalg.norm(new-uv));q['direct_current_uv']=new.tolist()
       if rev['status']!='registered':q.update(status='unknown_reverse_registration',unknown_reason=rev['status'])
       else:
        A=np.asarray(f['warp'])[:,:2];B=np.asarray(rev['warp'])[:,:2];bias=np.asarray(rev['registered_uv'])-seed['uv'];corn=np.array([[-r,-r],[-r,r],[r,-r],[r,r]],float)@(B@A-np.eye(2)).T+bias;ce=np.linalg.norm(corn,axis=1);q.update(reverse_birth_uv=rev['registered_uv'],centre_cycle_px=float(np.linalg.norm(bias)),affine_patch_cycle_corner_errors_px=ce.tolist(),affine_patch_cycle_max_px=float(ce.max()),forward_branch_identity_margin=float(f['translation_branch_gap']-(1-f['translation_peak_NCC'])),reverse_branch_identity_margin=float(rev['translation_branch_gap']-(1-rev['translation_peak_NCC'])))
        if min(f['registered_NCC'],rev['registered_NCC'])<NAVIGATION['patch_ncc_min']:q.update(status='unknown_context_appearance',unknown_reason='existing_NCC075_not_met')
        elif ce.max()>NAVIGATION['reciprocal_max_px']:q.update(status='unknown_wide_patch_cycle',unknown_reason='existing_1px_whole_patch_cycle_reference_not_met')
        elif min(q['forward_branch_identity_margin'],q['reverse_branch_identity_margin'])<=0:q.update(status='unknown_competing_context_branch',unknown_reason='alternative_peak_competes_at_or_within_actual_image_registration_mismatch')
        else:
         q.update(status='wide_context_direct_source_supported',unknown_reason=None)
         if q['new_minus_original_px']>NAVIGATION['reciprocal_max_px']:q.update(current_uv=new.tolist(),source_UV_changed=True)
    records.append(q);stats[q['status']]+=1
   if len(frame_receipt)%25==0:print(json.dumps(dict(pts=p,frames=len(frame_receipt),rows=len(records),counts=dict(stats))),flush=True)
 with (out/'readout_rows.jsonl').open('w',encoding='utf8') as f:
  for q in records:f.write(json.dumps(q,separators=(',',':'),allow_nan=False)+'\n')
 changed=[q for q in records if q['source_UV_changed']];unknown=[q for q in records if q['status'].startswith('unknown')]
 (out/'frame_receipt.json').write_text(json.dumps(frame_receipt,indent=2),encoding='utf8');(out/'summary.json').write_text(json.dumps(dict(schema='rinkfit.wide_context_source_readout_audit.v1',source=src,total_original_rows=len(rows),total_output_rows=len(records),tracks=len(birth),counts=dict(stats),changed_source_mean_rows=len(changed),unknown_original_means_retained=len(unknown),new_source_covariance_issued=False,model_or_world_projection_used=False,reader_scope='all S04 Ad2 existing physical source tracks, no model-residual subsetting',policy=dict(template_shape=[97,97],search_extra_halfwidth_xy=[160,40],existing_NAVIGATION=NAVIGATION,change_policy='only source-supported displacement over existing reciprocal 1px reference; supported <=1px original means retained',uniqueness='best-vs-distinct-context-branch NCC gap exceeds actual best translation image mismatch 1-NCC in both directions; engineering identity evidence, no calibrated source uncertainty'),outputs=dict(readout_rows=ref(out/'readout_rows.jsonl'),frame_receipt=ref(out/'frame_receipt.json')),changed_rows=[dict(source_row_id=q['source_row_id'],native_pts=q['native_pts'],new_minus_original_px=q['new_minus_original_px']) for q in changed],limitations=['97px fixed-birth affine context can fail with long zoom, occlusion or boundary support; all failures retain original means as unknown.','This is source-only identity/readout review. No model residual is used to select a branch or suppress a source row.','No covariance/metric/world accuracy is issued; original source row identities and UV bits remain in lineage.']),indent=2,allow_nan=False),encoding='utf8')
 print(json.dumps(dict(done=True,total=len(records),counts=dict(stats),changed=len(changed),unknown=len(unknown))),flush=True)
 return records
if __name__=='__main__':
 import sys
 run(*sys.argv[1:])
