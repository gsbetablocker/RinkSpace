"""Official native-clock RGB access with role and exact BGR authentication."""
from pathlib import Path
import hashlib
import av
from .io_utils import read_json,read_jsonl,file_sha256
from .role_guard import FrameRoleGuard

def iter_training_native_rgb(run,requested_pts):
 """Yield (pts, RGB, receipt); training never opens reserved/guard pixels."""
 run=Path(run);wanted={int(p) for p in requested_pts}
 if not wanted:return
 scope=read_json(run/'programme_scope.json');guard=FrameRoleGuard(run/'roles/frame_roles.json');video=Path(scope['source']['path'])
 if any(p<0 or p%256 for p in wanted):raise ValueError('Exact original PTS lattice required')
 ledger={int(r['native_pts']):r for n in ('dense_s03_v1','dense_remaining_v1') for r in read_jsonl(run/n/'frame_ledger.jsonl.gz')}
 for pts in wanted:
  row=ledger[pts];guard.assert_training_pixels([pts],shot_id=row['shot_id'])
 if file_sha256(video)!=scope['source']['sha256']:raise ValueError('Official video changed')
 seen=set()
 with av.open(str(video)) as container:
  stream=container.streams.video[0]
  if str(stream.time_base)!='1/12800':raise ValueError('Official native time base changed')
  container.seek(min(wanted),stream=stream,backward=True)
  for frame in container.decode(stream):
   pts=int(frame.pts)
   if pts>max(wanted):break
   if pts not in wanted:continue
   bgr=frame.to_ndarray(format='bgr24');sha=hashlib.sha256(bgr.tobytes()).hexdigest()
   if bgr.shape!=(1080,1920,3) or sha!=ledger[pts]['decoded_native_BGR_sha256']:raise ValueError('Actual decoded exposure differs from original ledger')
   rgb=bgr[:,:,::-1].copy();seen.add(pts);yield pts,rgb,dict(native_pts=pts,time_base='1/12800',source_video_sha256=scope['source']['sha256'],decoded_native_BGR_sha256=sha,decoded_native_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),shot_id=ledger[pts]['shot_id'],native_RGB_resampled=False,training_role_verified=True)
 if seen!=wanted:raise ValueError('Requested actual native exposure missing')
