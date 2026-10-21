"""Actual-source advertisement before/after evidence at previously reported failures."""
import argparse,base64,gzip,hashlib,json
from pathlib import Path
import av,cv2,numpy as np
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf8'))
def check(r):assert sha(r['file'])==r['sha256'];return r['file']
def rows(f):
 out=[]
 for m in f['source_members']:
  with open(m['file'],'rb')as h:h.seek(m['offset_bytes']);b=h.read(m['length_bytes'])
  assert hashlib.sha256(b).hexdigest()==m['sha256'];out.extend(json.loads(s)for s in gzip.decompress(b).decode('utf8').splitlines())
 return out
def flag(f,k):return np.frombuffer(base64.b64decode(f[k+'_uint8_base64']),'u1')
def main(manifest,video,out,pts):
 mf=load(manifest);assert mf['complete']is True;idx={}
 with gzip.open(check(mf['perframe_index']),'rt',encoding='utf8')as h:
  for line in h:j=json.loads(line);idx[j['native_pts']]=j
 wanted=sorted(set(pts)&set(idx));assert wanted
 target=Path(out);assert not target.exists();target.mkdir(parents=True);records=[]
 with av.open(video)as c:
  stream=c.streams.video[0]
  for p in wanted:
   c.seek(p,stream=stream,backward=True);source=next(f for f in c.decode(stream)if f.pts==p).to_ndarray(format='bgr24')
   new=load(check(idx[p]));oldref=new['parent_failed_v1_source_frame_ref'];old=load(check(oldref));rr=rows(new);assert new['source_members'][:len(old['source_members'])]==old['source_members'];assert hashlib.sha256(source.tobytes()).hexdigest()==new['decoded_native_BGR_sha256']
   before=flag(old,'eligible')>0;before=np.r_[before,np.zeros(len(rr)-len(before),bool)];after=flag(new,'eligible')>0;assert len(rr)==len(before)==len(after)
   left=source.copy();right=source.copy();lost=np.flatnonzero(before&~after);kept=np.flatnonzero(after);gained=np.flatnonzero(~before&after)
   for i in np.flatnonzero(before):cv2.circle(left,tuple(np.rint(rr[i]['uv']).astype(int)),1,(210,50,210),1,cv2.LINE_AA)
   for i in kept:cv2.circle(right,tuple(np.rint(rr[i]['uv']).astype(int)),1,(50,235,50),1,cv2.LINE_AA)
   for i in lost:cv2.circle(right,tuple(np.rint(rr[i]['uv']).astype(int)),2,(30,30,245),1,cv2.LINE_AA)
   for i in gained:cv2.circle(right,tuple(np.rint(rr[i]['uv']).astype(int)),2,(240,230,10),1,cv2.LINE_AA)
   # Captions outside the original picture, never hiding source ice or board.
   canv=np.zeros((1118,3840,3),'u1');canv[:1080,:1920]=left;canv[:1080,1920:]=right
   cv2.putText(canv,f'OLD qualification PTS {p} / {before.sum()} rows',(10,1107),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1,cv2.LINE_AA)
   cv2.putText(canv,f'NEW: green eligible {after.sum()} / red rejected {len(lost)} / cyan new-qualified {len(gained)}',(1930,1107),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1,cv2.LINE_AA)
   fp=target/f'{p}_old_new.jpg';cv2.imwrite(str(fp),canv,[cv2.IMWRITE_JPEG_QUALITY,94])
   records.append(dict(native_pts=p,shot_id=new['shot_id'],new_frame_ref=idx[p],old_frame_ref=oldref,source_BGR_sha256=new['decoded_native_BGR_sha256'],old_eligible=int(before.sum()),new_eligible=int(after.sum()),old_eligible_now_rejected=len(lost),old_excluded_now_eligible=len(gained),source_UV_and_raw_members_unchanged=True,image_ref=dict(file=str(fp),sha256=sha(fp))))
 (target/'manifest.json').write_text(json.dumps(dict(schema='rinkspace.actual_source_print_requalification_counterexample_views.v1',complete=True,source_manifest_ref=dict(file=str(manifest),sha256=sha(manifest)),cases=records,all_data_from_actual_native_source_and_current_generation=True,not_full_video_visual_release_acceptance=True),ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps(records))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--video',required=True);ap.add_argument('--output',required=True);ap.add_argument('--pts',nargs='+',required=True,type=int);a=ap.parse_args();main(a.manifest,a.video,a.output,a.pts)
