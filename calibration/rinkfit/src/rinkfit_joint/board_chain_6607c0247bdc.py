import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys
sys.path[:0]=[r'D:/OneDrive/codex_projs/RinkSpace/calibration/rinkfit/src',r'D:/OneDrive/codex_projs/RinkSpace']
from pathlib import Path
from collections import defaultdict,Counter
import json,av,cv2,numpy as np,hashlib
from rinkfit_joint.board_wide_549fb411ea18 import register,patch
from rinkfit_joint.source_texture_tracklets import _source_mask
from rinkfit_joint.io_utils import read_jsonl,file_sha256
cv2.setNumThreads(1)
def full(f,mask,p,d,sid,rgb,bgr):
 co=np.array(f['registered_corners_native_uv']);lo=np.floor(co.min(0)-1).astype(int);hi=np.ceil(co.max(0)+1).astype(int);edge=lo.min()<0 or np.any(hi>=[1920,1080]);num=-1 if edge else int(np.count_nonzero(mask[lo[1]:hi[1]+1,lo[0]:hi[0]+1]));return dict(source_row_id=sid,native_pts=p,direction=d,actual_RGB_sha256=rgb,actual_BGR_sha256=bgr,mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),registered_corners_native_uv=co.tolist(),conservative_affine_support_bbox_plus_1px_halo=[lo.tolist(),hi.tolist()],excluded_pixels_in_bbox=num,status='pass_full_affine_context_support' if num==0 else 'unknown_actual_affine_context_support')
def run(cfgpath,rowpath,outpath):
 cfg=json.loads(Path(cfgpath).read_text(encoding='utf8'));rows=json.loads(Path(rowpath).read_text(encoding='utf8'));out=Path(outpath);out.mkdir(parents=True,exist_ok=True);assert file_sha256(cfg['source']['path'])==cfg['source']['sha256'];by=defaultdict(list)
 for q in rows:by[q['native_pts']].append(q)
 led={q['native_pts']:q for path in cfg['actor_ledgers'] for q in read_jsonl(path) if q['native_pts'] in by};roles=json.loads(Path(cfg['roles']).read_text(encoding='utf8'))['frame_roles'];seeds={};result=[];frames=[]
 with av.open(cfg['source']['path']) as con:
  st=con.streams.video[0];assert str(st.time_base)=='1/12800';con.seek(min(by),stream=st,backward=True)
  for fr in con.decode(st):
   p=int(fr.pts)
   if p>max(by):break
   if p not in by:continue
   assert roles[str(p)]['role']=='development_candidate'
   rgb=fr.to_ndarray(format='rgb24');bgrhash=hashlib.sha256(rgb[:,:,::-1].copy().tobytes()).hexdigest();g=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32);mask,dig=_source_mask(rgb,led[p],cfg['source'],p,cfg['shot_id'])
   for u in cfg.get('guard_source_uv_by_pts',{}).get(str(p),[]):cv2.circle(mask,tuple(np.rint(u).astype(int)),33,255,-1)
   for rg in cfg['regions']:
    if rg['native_pts']==p and rg.get('native_person_mask'):
     mr=rg['native_person_mask'];assert file_sha256(mr['file'])==mr['sha256'];mm=cv2.imread(mr['file'],0);mask=np.maximum(mask,cv2.dilate((mm>0).astype(np.uint8)*255,np.ones((33,33),np.uint8)))
   frames.append(dict(native_pts=p,time_base=[1,12800],RGB_sha256=dig,BGR_sha256=bgrhash,mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest()))
   for row in by[p]:
    assert row['current_rgb_sha256']==dig;tid=row['track_id'];u=np.array(row['current_uv'],float);sid=row['source_row_id'];q=dict(source_row_id=sid,track_id=tid,shot_id=cfg['shot_id'],surface_instance_id=row['surface_instance_id'],surface_class='board',native_pts=p,time_base=[1,12800],current_RGB_sha256=dig,current_BGR_sha256=bgrhash,source_video_sha256=cfg['source']['sha256'],original_uv=u.tolist(),original_uv_binary64_sha256=hashlib.sha256(np.asarray(u,dtype='<f8').tobytes()).hexdigest(),current_uv=u.tolist(),source_UV_changed=False,model_or_world_projection_used=False,template_shape=[97,97],reference_kind='previous_direct_source_verified_exposure_same_physical_track')
    if tid not in seeds:
     # Original source birth is the fixed root. Only actual direct readouts carry it forward.
     cor=np.array([[-48,-48],[-48,48],[48,-48],[48,48]],float)+u;lo=np.floor(cor.min(0)-1).astype(int);hi=np.ceil(cor.max(0)+1).astype(int);bad=lo.min()<0 or np.any(hi>=[1920,1080]) or mask[lo[1]:hi[1]+1,lo[0]:hi[0]+1].any();q.update(reference_kind='original_native_source_birth_reference',root_birth_row_id=sid,status='unknown_birth_support' if bad else 'original_birth_reference',unknown_reason='actual_birth_context_support' if bad else None,birth_native_pts=p,birth_uv=u.tolist(),birth_RGB_sha256=dig,root_birth_native_pts=p,root_birth_uv=u.tolist())
     if not bad:seeds[tid]=dict(p=p,u=u,gray=g.copy(),mask=mask.copy(),rgb=dig,bgr=bgrhash,sid=sid,root_p=p,root_u=u.tolist())
    else:
     s=seeds[tid];q.update(birth_native_pts=s['p'],birth_uv=s['u'].tolist(),birth_RGB_sha256=s['rgb'],reference_source_row_id=s['sid'],reference_native_pts=s['p'],reference_uv=s['u'].tolist(),reference_uv_binary64_sha256=hashlib.sha256(np.asarray(s['u'],dtype='<f8').tobytes()).hexdigest(),reference_uv_applied_within_source_chain_mean=True,reference_UV_application_scope='source chain direct readout; selected fit mean separately bound in generation patch',root_birth_row_id='static_board_texture/'+tid+'/'+str(s['root_p']),root_birth_native_pts=s['root_p'],root_birth_uv=s['root_u'])
     if p-s['p']!=256:
      q.update(status='unknown_reference_gap',unknown_reason='adjacent_native_exposure_reference_missing',reference_kind='no_legal_adjacent_reference_available',last_legal_reference_source_row_id=s['sid'],last_legal_reference_native_pts=s['p'])
      for k in ('reference_source_row_id','reference_native_pts','reference_uv','reference_uv_binary64_sha256','reference_uv_applied_within_source_chain_mean','reference_UV_application_scope'):q.pop(k,None)
     else:
      f=register(patch(s['gray'],s['u'],97),g,u,mask,97);q['forward']=f
      if f['status']!='registered':q.update(status='unknown_forward_registration',unknown_reason=f['status'])
      else:
       v=np.array(f['registered_uv']);rev=register(patch(g,v,97),s['gray'],s['u'],s['mask'],97);q['reverse']=rev;q.update(direct_current_uv=v.tolist(),new_minus_original_px=float(np.linalg.norm(v-u)),new_minus_original_uv=(v-u).tolist())
       if rev['status']!='registered':q.update(status='unknown_reverse_registration',unknown_reason=rev['status'])
       else:
        A=np.array(f['warp'])[:,:2];B=np.array(rev['warp'])[:,:2];bias=np.array(rev['registered_uv'])-s['u'];ce=np.linalg.norm(np.array([[-48,-48],[-48,48],[48,-48],[48,48]])@(B@A-np.eye(2)).T+bias,axis=1);q.update(affine_patch_cycle_corner_errors_px=ce.tolist(),affine_patch_cycle_max_px=float(ce.max()),centre_cycle_px=float(np.linalg.norm(bias)),forward_branch_identity_margin=f['translation_branch_gap']-(1-f['translation_peak_NCC']),reverse_branch_identity_margin=rev['translation_branch_gap']-(1-rev['translation_peak_NCC']))
        sup=[full(f,mask,p,'forward',sid,dig,bgrhash),full(rev,s['mask'],s['p'],'reverse',sid,s['rgb'],s['bgr'])];q['full_affine_context_support']=sup
        if min(f['registered_NCC'],rev['registered_NCC'])<.75:q.update(status='unknown_context_appearance',unknown_reason='existing_NCC075_not_met')
        elif ce.max()>1:q.update(status='unknown_wide_patch_cycle',unknown_reason='existing_1px_whole_patch_cycle_reference_not_met')
        elif min(q['forward_branch_identity_margin'],q['reverse_branch_identity_margin'])<=0:q.update(status='unknown_competing_context_branch',unknown_reason='alternative_peak_competes_at_or_within_actual_image_registration_mismatch')
        elif any(x['excluded_pixels_in_bbox']!=0 for x in sup):q.update(status='unknown_full_affine_context_support',unknown_reason='actual_registered_full_context_plus1px_sampler_halo_intersects_exclusion')
        else:
         q.update(status='wide_context_direct_source_supported',unknown_reason=None,current_uv=v.tolist(),source_UV_changed=bool(np.linalg.norm(v-u)>1));seeds[tid]=dict(s,p=p,u=v,gray=g.copy(),mask=mask.copy(),rgb=dig,bgr=bgrhash,sid=sid)
    result.append(q)
 with (out/'readout_rows.jsonl').open('w',encoding='utf8') as f:
  for q in result:f.write(json.dumps(q,separators=(',',':'),allow_nan=False)+'\n')
 (out/'frame_receipt.json').write_text(json.dumps(frames,indent=2),encoding='utf8');print(json.dumps(dict(rows=len(result),counts=dict(Counter(q['status'] for q in result)),reader_sha256=file_sha256(__file__))),flush=True)
if __name__=='__main__':run(*sys.argv[1:])
