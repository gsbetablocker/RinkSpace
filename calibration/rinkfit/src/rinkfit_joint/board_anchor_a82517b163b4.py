import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys
sys.path[:0]=[r'D:/OneDrive/codex_projs/RinkSpace/calibration/rinkfit/src',r'D:/OneDrive/codex_projs/RinkSpace']
from pathlib import Path
from collections import defaultdict,Counter
import av,cv2,numpy as np,json,hashlib
from rinkfit_joint.source_texture_tracklets import _source_mask
from rinkfit_joint.native_entity_tracklets import _ncc
from rinkfit_joint.io_utils import read_jsonl,file_sha256
cv2.setNumThreads(1)
WH=np.array([161,65]);H=(WH-1)/2

def sha(a):return hashlib.sha256(a.tobytes()).hexdigest()
def support(corners,mask):
 lo=np.floor(np.min(corners,axis=0)-1).astype(int);hi=np.ceil(np.max(corners,axis=0)+1).astype(int)
 edge=lo.min()<0 or np.any(hi>=[1920,1080]);num=-1 if edge else int(np.count_nonzero(mask[lo[1]:hi[1]+1,lo[0]:hi[0]+1]))
 return dict(bbox_plus1px_halo=[lo.tolist(),hi.tolist()],excluded_pixels=num,status='pass' if num==0 else 'image_edge' if num<0 else 'actor_graphics_guard_support')
def rectcorn(c):return np.array([[-H[0],-H[1]],[-H[0],H[1]],[H[0],-H[1]],[H[0],H[1]]])+c

def reg(t,g,guess,mask):
 lo=np.maximum(np.floor(guess-H-[160,40]).astype(int),[0,0]);hi=np.minimum(np.ceil(guess+H+[160,40]+1).astype(int),[1920,1080]);b=g[lo[1]:hi[1],lo[0]:hi[0]].astype(np.float32)
 if np.any(np.array(b.shape[::-1])<WH):return dict(status='search_too_small')
 z=cv2.matchTemplate(b,t,cv2.TM_CCOEFF_NORMED);good=cv2.erode((mask==0).astype(np.uint8),np.ones(tuple(WH[::-1]),np.uint8));mz=good[lo[1]+int(H[1]):hi[1]-int(H[1]),lo[0]+int(H[0]):hi[0]-int(H[0])];z[mz==0]=-2
 if z.max()<=-1:return dict(status='no_unmasked_candidate')
 _,peak,_,loc=cv2.minMaxLoc(z);x,y=loc;oth=z.copy();oth[max(0,y-31):y+32,max(0,x-31):x+32]=-2;ry,rx=np.unravel_index(oth.argmax(),oth.shape);runner=float(oth[ry,rx]);w=np.array([[1,0,x],[0,1,y]],np.float32)
 out=dict(translation_uv=(lo+[x,y]+H).tolist(),translation_NCC=float(peak),other_branch_uv=(lo+[rx,ry]+H).tolist(),other_branch_NCC=runner,branch_identity_margin=float(2*peak-runner-1),search_origin=lo.tolist(),search_shape=list(b.shape),template_shape=WH.tolist())
 try:cc,w=cv2.findTransformECC(t,b,w,cv2.MOTION_AFFINE,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,60,.0001),None,3)
 except cv2.error:return dict(out,status='affine_ECC_failed')
 A=w[:,:2];anchor=lo+w@np.r_[H,1];cor=rectcorn(np.zeros(2))@A.T+anchor;sp=support(cor,mask);warped=cv2.warpAffine(b,w,tuple(map(int,WH)),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
 status='registered'
 if not np.isfinite(w).all() or np.linalg.det(A)<=0:status='nonfinite_or_fold'
 elif sp['status']!='pass':status=sp['status']
 elif np.any(abs(anchor-np.asarray(out['translation_uv']))>[31,5]):status='affine_changed_translation_branch'
 out.update(status=status,anchor=anchor.tolist(),A=A.tolist(),warp=w.tolist(),corners=cor.tolist(),complete_support=sp,NCC=float(_ncc(t,warped)),ECC=float(cc));return out

def run(cfgpath,rowpath,outpath):
 cfg=json.loads(Path(cfgpath).read_text(encoding='utf8'));rows=json.loads(Path(rowpath).read_text(encoding='utf8'));out=Path(outpath);out.mkdir(exist_ok=True,parents=True)
 assert file_sha256(cfg['source']['path'])==cfg['source']['sha256'];by=defaultdict(list)
 for q in rows:by[q['native_pts']].append(q)
 led={q['native_pts']:q for p in cfg['actor_ledgers'] for q in read_jsonl(p) if q['native_pts'] in by};roles=json.loads(Path(cfg['roles']).read_text(encoding='utf8'))['frame_roles'];seeds={};rr=[];frames=[];counts=Counter();first=min(by);last=max(by)
 with av.open(cfg['source']['path']) as con:
  st=con.streams.video[0];assert str(st.time_base)=='1/12800';con.seek(first,stream=st,backward=True)
  for fr in con.decode(st):
   p=int(fr.pts)
   if p<first:continue
   if p>last:break
   if p not in by:continue
   assert roles[str(p)]['role']=='development_candidate';rgb=fr.to_ndarray(format='rgb24');g=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32);mask,dig=_source_mask(rgb,led[p],cfg['source'],p,cfg.get('shot_id','s04'))
   for u in cfg.get('guard_source_uv_by_pts',{}).get(str(p),[]):cv2.circle(mask,tuple(np.rint(u).astype(int)),33,255,-1)
   for box in cfg.get('guard_support_bbox_by_pts',{}).get(str(p),[]):
    a,b,c,d=map(int,box);mask[max(0,b):min(1080,d+1),max(0,a):min(1920,c+1)]=255
   for rg in cfg['regions']:
    if rg['native_pts']==p and rg.get('native_person_mask'):
     mr=rg['native_person_mask'];assert file_sha256(mr['file'])==mr['sha256'];mm=cv2.imread(mr['file'],0);mask=np.maximum(mask,cv2.dilate((mm>0).astype(np.uint8)*255,np.ones((33,33),np.uint8)))
   frames.append(dict(native_pts=p,time_base=[1,12800],RGB_sha256=dig,BGR_sha256=sha(rgb[:,:,::-1].copy()),mask_sha256=sha(mask),rows=len(by[p]),role=roles[str(p)]['role']))
   for row in by[p]:
    assert dig==row['current_rgb_sha256'];u=np.array(row['current_uv'],float);tid=row['track_id'];q=dict(source_row_id=row['source_row_id'],track_id=tid,surface_instance_id=row['surface_instance_id'],native_pts=p,time_base=[1,12800],current_RGB_sha256=dig,source_video_sha256=cfg['source']['sha256'],original_uv=u.tolist(),current_uv=u.tolist(),source_UV_changed=False,template_shape=WH.tolist(),model_or_world_projection_used=False)
    if tid not in seeds:
     # At image edge place the source point two pixels inside the asymmetric context; never extrapolate the point outside the template.
     a=u.copy()
     if u[0]<161:a[0]=u[0]+H[0]-2
     elif u[0]>1920-161:a[0]=u[0]-H[0]+2
     a[1]=np.clip(a[1],H[1]+2,1079-H[1]-2)
     sp=support(rectcorn(a),mask);bad=sp['status']!='pass';seeds[tid]=dict(p=p,u=u,a=a,offset=u-a,RGB=dig,mask=mask.copy() if not bad else None,g=g.copy() if not bad else None,t=cv2.getRectSubPix(g,tuple(map(int,WH)),tuple(a)) if not bad else None,bad=bad,support=sp)
     q.update(status='unknown_birth_support' if bad else 'original_birth_reference',unknown_reason=sp['status'] if bad else None,birth_native_pts=p,birth_uv=u.tolist(),birth_anchor=a.tolist(),birth_complete_support=sp)
    else:
     s=seeds[tid];q.update(birth_native_pts=s['p'],birth_uv=s['u'].tolist(),birth_anchor=s['a'].tolist(),birth_RGB_sha256=s['RGB'],birth_complete_support=s['support'])
     if s['bad']:q.update(status='unknown_birth_support',unknown_reason=s['support']['status'])
     else:
      f=reg(s['t'],g,u-s['offset'],mask);q['forward']=f
      if f['status']!='registered':q.update(status='unknown_forward_registration',unknown_reason=f['status'])
      else:
       a=np.array(f['anchor']);A=np.array(f['A']);new=a+A@s['offset'];q['direct_current_uv']=new.tolist();q['new_minus_original_px']=float(np.linalg.norm(new-u));q['new_minus_original_uv']=(new-u).tolist();q['old_context_NCC']=float(_ncc(s['t'],cv2.getRectSubPix(g,tuple(map(int,WH)),tuple(u-s['offset']))));rev=reg(cv2.getRectSubPix(g,tuple(map(int,WH)),tuple(a)),s['g'],s['a'],s['mask']);q['reverse']=rev
       if rev['status']!='registered':q.update(status='unknown_reverse_registration',unknown_reason=rev['status'])
       else:
        B=np.array(rev['A']);bias=np.array(rev['anchor'])-s['a'];cy=rectcorn(np.zeros(2))@(B@A-np.eye(2)).T+bias;ce=np.linalg.norm(cy,axis=1);pointcycle=B@A@s['offset']+bias-s['offset'];q.update(affine_patch_cycle_corner_errors_px=ce.tolist(),affine_patch_cycle_max_px=float(ce.max()),point_cycle_px=float(np.linalg.norm(pointcycle)))
        if min(f['NCC'],rev['NCC'])<.75:q.update(status='unknown_context_appearance',unknown_reason='existing_NCC075_not_met')
        elif ce.max()>1:q.update(status='unknown_wide_patch_cycle',unknown_reason='existing_1px_whole_patch_cycle_reference_not_met')
        elif min(f['branch_identity_margin'],rev['branch_identity_margin'])<=0:q.update(status='unknown_competing_context_branch',unknown_reason='alternative_peak_competes_at_or_within_actual_image_registration_mismatch')
        else:
         q.update(status='wide_context_direct_source_supported',unknown_reason=None)
         if np.linalg.norm(new-u)>1:q.update(current_uv=new.tolist(),source_UV_changed=True)
    rr.append(q);counts[q['status']]+=1
   if len(frames)%25==0:print(json.dumps(dict(pts=p,frames=len(frames),rows=len(rr),counts=dict(counts))),flush=True)
 with (out/'readout_rows.jsonl').open('w',encoding='utf8') as f:
  for q in rr:f.write(json.dumps(q,separators=(',',':'),allow_nan=False)+'\n')
 (out/'frame_receipt.json').write_text(json.dumps(frames,indent=2),encoding='utf8');(out/'summary.json').write_text(json.dumps(dict(rows=len(rr),tracks=len(seeds),counts=dict(counts),changed=sum(q['source_UV_changed'] for q in rr),source_video=cfg['source'],source_config=dict(file=cfgpath,sha256=file_sha256(cfgpath)),reader=dict(file=str(Path(__file__).resolve()),sha256=file_sha256(__file__)),source_rows_input=dict(file=rowpath,sha256=file_sha256(rowpath)),model_or_world_projection_used=False),indent=2),encoding='utf8');print(json.dumps(dict(done=True,counts=dict(counts))),flush=True)
if __name__=='__main__':run(*sys.argv[1:])
