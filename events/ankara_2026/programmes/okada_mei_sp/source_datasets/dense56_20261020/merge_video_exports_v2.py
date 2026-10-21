"""Bind unchanged complete shot video exports to the merged source generation."""
from pathlib import Path
import json,hashlib,sys
O=Path(sys.argv[1]);target=O/'videos_v2';dest=target/'manifest.json'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
if dest.exists():raise FileExistsError(dest)
gp=O/'generation_v2/manifest.json';g=read(gp);assert g['complete']and g['frames']==1862
cfg=read(O/'generation_v2/video_config.json');videos=[];children=[]
member=g['gzip_member_concatenation_without_UV_or_row_ID_change']['selected_frames.jsonl.gz']
for j,(sid,n,a,b)in enumerate([('s05',1376,948992,1300992),('s06',486,1301248,1425408)]):
 mp=target/sid/'manifest.json';m=read(mp);assert m['schema']=='rinkfit.native_source_points_only_video.v1'and m['frames']==n and m['by_shot']=={sid:n}
 assert m['source']==cfg['source']and m['only_original_pixels_and_source_points']and not m['original_image_rescaled']and not m['physical_model_used']and not m['point_UV_interpolated_by_renderer']
 ref=m['selected_frame_file'];assert sha(ref['file'])==ref['sha256']==member[j]['sha256']and Path(ref['file']).resolve()==Path(member[j]['file']).resolve()
 assert len(m['videos'])==1;v=m['videos'][0];assert v['shot_id']==sid and v['frames']==n and v['source_pts']==[a,b]and v['output_frame_times_verified']and sha(v['file'])==v['sha256']
 videos.append(v);lp=mp.parent/'frame_ledger.jsonl.gz';children.append(dict(file=str(mp),sha256=sha(mp),selected_frame_file=ref,frame_ledger_ref=dict(file=str(lp),sha256=sha(lp))))
assert sum(v['drawn_point_rows']for v in videos)==g['point_rows']
m=dict(schema='rinkspace.native_source_points_only_video_assembly.v2',complete=True,source=cfg['source'],source_generation=dict(file=str(gp),sha256=sha(gp)),frames=1862,by_shot={'s05':1376,'s06':486},drawn_source_point_rows=g['point_rows'],videos=videos,selected_frame_file=cfg['selected_frame_file'],child_video_exports=children,merged_selected_frames_preserve_child_gzip_members_bytewise=True,only_original_pixels_and_source_points=True,original_image_rescaled=False,physical_model_used=False,point_UV_interpolated_by_renderer=False,source_frames_missing_or_duplicated=0,output_frame_times_verified=True,source_quality_review_separate=True)
dest.write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(dict(complete=True,frames=1862,point_rows=g['point_rows'],videos=videos),ensure_ascii=False))
