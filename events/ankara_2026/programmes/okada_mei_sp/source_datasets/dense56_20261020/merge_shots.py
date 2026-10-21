"""Concatenate independent frozen shot generations without changing source UV."""
from pathlib import Path
from collections import Counter
import sys,json,gzip,hashlib,shutil
O=Path(sys.argv[1]);target=O/'generation_v1';parents=[O/'parts'/s/'manifest.json'for s in('s05','s06')]
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.load(open(p,encoding='utf8'))
def write(p,j):Path(p).write_text(json.dumps(j,ensure_ascii=False,indent=2),encoding='utf8')
if target.exists():raise FileExistsError(target)
ms=[read(p)for p in parents];pins=[]
for p,m,s,n in zip(parents,ms,['s05','s06'],[1376,486]):
 assert m['complete']and m['shot_ids']==[s]and m['frames']==n
 pins.append(dict(file=str(p),sha256=sha(p)))
 for name,ref in m['files'].items():assert sha(ref['file'])==ref['sha256']
expected=[(s,p)for s,a,b in [('s05',948992,1300992),('s06',1301248,1425408)]for p in range(a,b+1,256)];seen=[];counts=[]
for m in ms:
 with gzip.open(m['files']['selected_frames.jsonl.gz']['file'],'rt',encoding='utf8')as f:
  for line in f:
   r=json.loads(line);seen.append((r['shot_id'],r['native_pts']));counts.append(len(r['points']));assert len(r['points'])<=500
assert seen==expected and sum(counts)==sum(m['point_rows']for m in ms)
target.mkdir();files={};members={}
for name in ms[0]['files']:
 dst=target/name;pos=0;members[name]=[]
 with open(dst,'wb')as out:
  for m in ms:
   ref=m['files'][name];src=Path(ref['file']);n=src.stat().st_size
   with open(src,'rb')as f:shutil.copyfileobj(f,out,4*1024*1024)
   members[name].append(dict(file=str(src),sha256=ref['sha256'],combined_file_offset=pos,member_bytes=n));pos+=n
 files[name]=dict(file=str(dst),sha256=sha(dst))
def counters(key):
 q=Counter()
 for m in ms:q.update(m.get(key,{}))
 return dict(q)
def maps(key):
 q={}
 for m in ms:
  for k,v in m.get(key,{}).items():assert k not in q;q[k]=v
 return q
manifest=dict(schema='rinkspace.source_only_dense56_generation.v1',complete=True,frames=1862,point_rows=sum(counts),point_count_per_native_frame=dict(min=min(counts),max=max(counts)),by_part=counters('by_part'),by_shot=maps('by_shot'),features_by_shot=maps('features_by_shot'),source_kind_counts=counters('source_kind_counts'),display_classes=counters('display_classes'),manual_source_rows=sum(m['manual_source_rows']for m in ms),rejected_source_rows_by_actual_reason=counters('rejected_source_rows_by_actual_reason'),unknown_world_or_named_crosswalk_rows=counters('unknown_world_or_named_crosswalk_rows'),source_correction_rows_applied=sum(m['source_correction_rows_applied']for m in ms),parent_shot_generations=pins,files=files,gzip_member_concatenation_without_UV_or_row_ID_change=members,source_UV_moved=False,old_generations_unchanged=True,physical_solver_or_live_packet_changed=False,selected_rows_are_independent_measurements=False,all_visible_features_perfect_claimed=False)
for p in pins:assert sha(p['file'])==p['sha256']
write(target/'manifest.json',manifest);cfg=read(parents[0].parent/'video_config.json');cfg['selected_frame_file']=files['selected_frames.jsonl.gz'];cfg['shots']=[dict(id=s,first_pts=a,last_pts=b)for s,a,b in [('s05',948992,1300992),('s06',1301248,1425408)]];write(target/'video_config.json',cfg);print(json.dumps(dict(complete=True,frames=1862,rows=sum(counts),per_frame=manifest['point_count_per_native_frame'])))
