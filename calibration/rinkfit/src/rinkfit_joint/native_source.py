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


def iter_scope_native_rgb(scope_path, requested_pts, *, roles_path):
 """Decode current explicit scope, including S01, independently of an old ledger.

 The immutable frame-role authority is still checked before reading pixels.
 This path authenticates exact source PTS and RGB, without fixed-offset mapping.
 """
 scope_path=Path(scope_path).resolve();scope=read_json(scope_path)
 guard=FrameRoleGuard(roles_path);wanted={int(p) for p in requested_pts}
 if not wanted:return
 shots={p:s['id'] for s in scope['shots'] for p in wanted
        if int(s['first_pts'])<=p<=int(s['last_pts']) and (p-int(s['first_pts']))%256==0}
 if set(shots)!=wanted:raise ValueError('PTS outside explicit current programme shot scope')
 for p in wanted:guard.assert_training_pixels([p],shot_id=shots[p])
 source=scope['source'];video=Path(source['path'])
 if file_sha256(video)!=source['sha256']:raise ValueError('Official source SHA differs')
 if source['time_base']!='1/12800':raise ValueError('Expected official native clock')
 seen=set()
 with av.open(str(video)) as container:
  stream=container.streams.video[0]
  if str(stream.time_base)!=source['time_base']:raise ValueError('Native time base differs')
  container.seek(min(wanted),stream=stream,backward=True)
  for frame in container.decode(stream):
   pts=frame.pts
   if pts>max(wanted):break
   if pts not in wanted:continue
   if pts in seen:raise ValueError('Duplicate native exposure')
   bgr=frame.to_ndarray(format='bgr24');rgb=bgr[:,:,::-1].copy();seen.add(pts)
   if rgb.shape!=(source['height'],source['width'],3):raise ValueError('Native raster differs')
   yield pts,rgb,dict(native_pts=pts,time_base=source['time_base'],source_video_sha256=source['sha256'],
    decoded_native_BGR_sha256=hashlib.sha256(bgr.tobytes()).hexdigest(),
    decoded_native_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),shot_id=shots[pts],
    native_RGB_resampled=False,training_role_verified=True,scope_reference=str(scope_path),roles_reference=str(Path(roles_path).resolve()))
 if seen!=wanted:raise ValueError('Missing exact native exposures: '+str(sorted(wanted-seen)))
