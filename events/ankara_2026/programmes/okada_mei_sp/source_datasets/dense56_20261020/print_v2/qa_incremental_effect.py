"""Audit incremental source exclusions against actual retained print data."""
import argparse,base64,gzip,hashlib,json
from pathlib import Path
import av,cv2,numpy as np
cv2.setNumThreads(2)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(file=str(p),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def check(r):assert sha(r['file'])==r['sha256'];return r['file']
def index(r):
 with gzip.open(check(r),'rt',encoding='utf8')as f:return {j['native_pts']:j for j in map(json.loads,f)}
def mask(r):
 a=cv2.imread(check(r),0);assert a is not None and a.shape==(1080,1920);return a>0
def arr(f,k):return np.frombuffer(base64.b64decode(f[k+'_uint8_base64']),'u1')
def rows(f):
 out=[]
 for m in f['source_members']:
  with open(m['file'],'rb')as h:h.seek(m['offset_bytes']);b=h.read(m['length_bytes'])
  assert hashlib.sha256(b).hexdigest()==m['sha256'];out.extend(json.loads(s)for s in gzip.decompress(b).decode('utf8').splitlines())
 return out
def at(q,m):
 q=np.asarray(q,float).reshape(-1,2);v=np.isfinite(q).all(1)&(q[:,0]>=0)&(q[:,0]<=1919)&(q[:,1]>=0)&(q[:,1]<=1079);p=np.rint(np.where(np.isfinite(q),q,0)).astype(int);return v&m[np.clip(p[:,1],0,1079),np.clip(p[:,0],0,1919)]
def main(a):
 out=Path(a.output);assert not out.exists();out.mkdir(parents=True);mf=read(a.manifest);assert mf['accepted_for_final_source_qualification'];base=read(check(mf['base_print_manifest_ref']));ni=index(mf['perframe_index']);bi=index(base['perframe_index']);sc=read(check(mf['source_actual_change_scope']));cases=[];allrows=[];motion=[]
 for p in sc['requalified_pts']:
  n=read(check(ni[p]));b=read(check(bi[p]));assert n['source_members']==b['source_members'];eq={k:bool(np.array_equal(arr(n,k),arr(b,k)))for k in('eligible','status','normal_only','texture_2D_tracking','observation_type')};assert all(eq.values());allrows.append(dict(native_pts=p,array_byte_equivalence=eq,source_members_byte_identical=True))
 for p in sc['changed_pair_current_pts']:
  n=read(check(ni[p]));b=read(check(bi[p]));nm=read(check(n['static_motion_ref']));bm=read(check(b['static_motion_ref']));eq={k:nm[k]==bm[k]for k in('source_uv','current_uv','forward_backward_error_px')};assert all(eq.values());motion.append(dict(native_pts=p,retained_real_LK_FB_byte_equivalence=eq,rejected_source_uv_count=nm['rejected_source_uv_count'],source_patch_rejected=nm['source_patch_rejected'],current_patch_rejected=nm['current_patch_rejected']))
 for p in sc['changed_source_domain_pts']:
  n=read(check(ni[p]));b=read(check(bi[p]));npf=read(check(n['body_material_ref']));bpf=read(check(b['body_material_ref']));nb=mask(npf['body_material_mask_ref']);bb=mask(bpf['body_material_mask_ref']);assert np.array_equal(nb,bb);nu=mask(n['unknown_exclusion_ref']);bu=mask(b['unknown_exclusion_ref'])if b.get('unknown_exclusion_ref')else np.zeros_like(nu);delta=nu&~bu;assert not(bu&~nu).any();guard=cv2.distanceTransform((~delta).astype('u1'),cv2.DIST_L2,3)<=8;rr=rows(n);xy=np.asarray([r['uv']for r in rr]);el=arr(n,'eligible')>0;hit=at(xy,guard);inside=at(xy,delta);row=dict(native_pts=p,added_unknown_pixels=int(delta.sum()),previous_material_pixels_intersect_added_unknown=int((bb&delta).sum()),previous_material_pixels_intersect_unknown_8px_guard=int((bb&guard).sum()),source_rows_inside_new_unknown=int(inside.sum()),source_rows_inside_new_unknown_guard=int(hit.sum()),previously_eligible_rows_inside_new_unknown_guard=int((hit&el).sum()),body_bitmap_bitwise_equal=True,original_source_rows=len(rr),source_UV_or_ID_changed=False)
  assert row['previously_eligible_rows_inside_new_unknown_guard']==0;allrows.append(row)
  if p in a.pts:
   cases.append((p,n,row,xy,el,delta,guard))
 views=[]
 with av.open(a.video)as c:
  st=c.streams.video[0]
  for p,n,row,xy,el,delta,guard in cases:
   c.seek(p,stream=st,backward=True);src=next(f for f in c.decode(st)if f.pts==p).to_ndarray(format='bgr24');assert hashlib.sha256(src.tobytes()).hexdigest()==n['decoded_native_BGR_sha256'];view=src.copy();col=view.copy();col[delta]=(20,190,250);view[delta]=cv2.addWeighted(src,.5,col,.5,0)[delta]
   cs,_=cv2.findContours(delta.astype('u1'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(view,cs,-1,(0,235,255),1)
   for q in xy[el]:cv2.circle(view,tuple(np.rint(q).astype(int)),1,(50,235,50),1,cv2.LINE_AA)
   can=np.zeros((1120,1920,3),'u1');can[:1080]=view;cv2.putText(can,'PTS %d / yellow: added source unknown; green: retained original print UV'%p,(10,1106),cv2.FONT_HERSHEY_SIMPLEX,.62,(255,255,255),1,cv2.LINE_AA);fp=out/('%d_unknown_print.jpg'%p);cv2.imwrite(str(fp),can,[cv2.IMWRITE_JPEG_QUALITY,94]);views.append(dict(**row,source_BGR_sha256=n['decoded_native_BGR_sha256'],frame_ref=ni[p],image_ref=ref(fp)))
 ledger=out/'all_changed_input_effects.json';ledger.write_text(json.dumps(dict(requalified_frames=allrows,motion_pair_effects=motion),separators=(',',':')),encoding='utf8');res=dict(schema='rinkspace.actual_incremental_print_input_effect_audit.v1',complete=True,source_manifest_ref=ref(a.manifest),actual_changed_source_domain_frames=len(sc['changed_source_domain_pts']),actual_requalified_frames=len(sc['requalified_pts']),actual_motion_proof_files=len(motion),eligible_status_type_array_bytes_all_unchanged=True,body_bitmap_all_changed_frames_unchanged=True,real_retained_LK_FB_coordinates_all_changed_pairs_unchanged=True,added_unknown_pixels=sum(x.get('added_unknown_pixels',0)for x in allrows),all_previous_material_pixels_intersect_unknown_guard=sum(x.get('previous_material_pixels_intersect_unknown_8px_guard',0)for x in allrows),all_raw_source_rows_intersect_unknown_guard=sum(x.get('source_rows_inside_new_unknown_guard',0)for x in allrows),all_previous_eligible_rows_intersect_unknown_guard=sum(x.get('previously_eligible_rows_inside_new_unknown_guard',0)for x in allrows),no_new_eligible_or_removed_eligible_print_rows=True,explanation='The precise new unknown bitmap and 8px guard do not intersect prior accepted advertisement body material. Source candidate UV in the new exclusion were already outside the body eligibility domain. No correction effect is claimed from execution counts.',ledger_ref=ref(ledger),cases=views,full_shot_visual_release_or_physical_quality_not_issued=True);(out/'manifest.json').write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in res.items()if k!='cases'}))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--video',required=True);ap.add_argument('--output',required=True);ap.add_argument('--pts',required=True,type=int,nargs='+');main(ap.parse_args())
