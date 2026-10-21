"""Merge already sealed per-shot print qualifications without rewriting any UV/member."""
import argparse,gzip,hashlib,json
from pathlib import Path
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf8'))
def ref(p):return dict(file=str(p),sha256=sha(p))
def main(out,parts):
 target=Path(out);assert not target.exists();parent=[];frames=[];shots={}
 for p in parts:
  m=load(p);assert m['complete']is True and m['accepted_for_source_consumer']is True
  assert m['source_video_sha256']=='e8b53966d56e72f3842d92c3ecbe6e9cc7e4670b7d0c9bd12f58fd2cff5ad57d'
  i=m['perframe_index'];assert sha(i['file'])==i['sha256']
  with gzip.open(i['file'],'rt',encoding='utf8')as f:rr=[json.loads(s)for s in f]
  assert len(rr)==m['frames'];frames.extend(rr);parent.append((m,ref(p)))
  for sid,v in m['per_shot'].items():assert sid not in shots;shots[sid]=v
 assert set(shots)=={'s05','s06'}and len(frames)==1862;frames.sort(key=lambda r:r['native_pts']);assert [r['native_pts']for r in frames]==list(range(948992,1425408+1,256))
 canonical={}
 for mm,rr in parent:
  for item in mm['canonical_member_files']:
   if item['file']in canonical:assert canonical[item['file']]['sha256']==item['sha256']
   else:canonical[item['file']]=item
 target.mkdir(parents=True);ip=target/'index.jsonl.gz'
 with gzip.open(ip,'wt',encoding='utf8')as f:
  for r in frames:f.write(json.dumps(r,separators=(',',':'))+'\n')
 m=dict(schema='rinkspace.merged_requalified_real_native_advertisement_generation.v2',complete=True,accepted_for_source_consumer=True,accepted_for_final_source_qualification=True,frames=1862,first_pts=948992,last_pts=1425408,native_step_pts=256,source_time_base='1/12800',source_video_sha256=parent[0][0]['source_video_sha256'],per_shot=shots,perframe_index=ref(ip),per_shot_qualification_manifests={sid:r for mm,r in parent for sid in mm['per_shot']},canonical_member_files=list(canonical.values()),source_UV_or_ID_moved=False,all_source_frame_refs_unchanged=True,source_members_immutable_byte_identical=True,new_two_exposure_21px_source_domains_actually_consumed=True,world_material_crosswalk_not_issued=True,independent_weights_not_issued=True,no_camera_model_or_solver=True,executed_small_source_dependencies=[ref(__file__)])
 for mm,r in parent:assert sha(r['file'])==r['sha256']
 (target/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(dict(file=str(target/'manifest.json'),sha256=sha(target/'manifest.json'),frames=1862)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('parts',nargs='+');a=p.parse_args();main(a.output,a.parts)
