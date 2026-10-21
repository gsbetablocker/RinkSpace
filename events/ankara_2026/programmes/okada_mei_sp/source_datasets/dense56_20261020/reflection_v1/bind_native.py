import av,json,gzip,hashlib,time
from pathlib import Path
O=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2/dense56_v1');Q=O/'reflection_v1';cfg=json.load(open(O/'consumer_s05.json',encoding='utf8'));src=cfg['source'];h=hashlib.sha256()
with open(src['file'],'rb') as f:
 for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
assert h.hexdigest()==src['sha256']
c=av.open(src['file']);s=c.streams.video[0];s.thread_type='AUTO';c.seek(948992,stream=s,backward=True);n=0;seen=[]
with gzip.open(Q/'native_bindings.jsonl.gz','wt',encoding='utf8') as out:
 for f in c.decode(s):
  if f.pts<948992:continue
  if f.pts>1300992:break
  rgb=f.to_ndarray(format='rgb24');bgr=rgb[:,:,::-1].copy();rec=dict(shot_id='s05',native_pts=f.pts,native_time_base=str(f.time_base),native_frame_duration_pts=256,native_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),native_BGR_sha256=hashlib.sha256(bgr.tobytes()).hexdigest(),source_video_sha256=h.hexdigest(),pixel_format='RGB24 and channel-reversed BGR24',shape_hw=list(rgb.shape[:2]));out.write(json.dumps(rec)+'\n');n+=1;seen.append(f.pts)
assert seen==list(range(948992,1300992+1,256)),(n,seen[0],seen[-1]);print('immutable file verified',h.hexdigest(),'frames',n)
json.dump(dict(source=src,actual_source_fullfile_sha256=h.hexdigest(),actual_native_frames=n,first_pts=seen[0],last_pts=seen[-1]),open(Q/'native_binding_receipt.json','w',encoding='utf8'),indent=2)
