"""Retained native five-profile samples for numerical source localization noise.

These arrays are measurements for the statistical compiler, not a declared
variance or an independent duplicate of the retained curve coordinates.
"""
from pathlib import Path
from collections import defaultdict
import argparse, json
import numpy as np
from PIL import Image
from .io_utils import read_json, read_jsonl, write_json, write_jsonl, file_sha256
from .source_worker import _channels,_sample,_points,_normal,_raw_support,_visible_indices

def extract_profiles(normalized,native,output,max_samples=16):
 normalized=Path(normalized);native=Path(native);out=Path(output)
 if out.exists():raise ValueError('A new immutable profile output is required')
 rows=list(read_jsonl(normalized/'observations.jsonl.gz'));groups=list(read_jsonl(normalized/'groups.jsonl.gz'));gi={g['group_id']:g for g in groups}
 by=defaultdict(list)
 for row in rows:by[row['provenance_group_id']].append(row)
 frames=read_json(native/'manifest.json')['frames'];fg=defaultdict(list)
 for group in groups:
  if group.get('group_type')=='segment_source_evidence':fg[(group['shot_id'],group['native_pts'])].append(group)
 out.mkdir(parents=True);profiles=[];index=[];uv=[];normals=[];channel=[];off=np.arange(-3,3.0001,.25);toff=np.arange(-2,3,dtype=float)
 for fr in frames:
  png=native/fr['native_png']
  if file_sha256(png)!=fr['native_png_sha256']:raise ValueError('Source native PNG changed')
  c=_channels(np.asarray(Image.open(png).convert('RGB')));families=defaultdict(list)
  for group in fg[(fr['shot_id'],fr['native_pts'])]:
   members=by[group['group_id']];feature=gi.get(group.get('feature_source_group_id'),{});raw,evidence,kind=_raw_support(group,feature,members[0].get('target_id'))
   ix=_visible_indices(raw,members,any(r.get('manual_authoritative') for r in members));fid=group.get('feature_source_group_id') or group['group_id']
   for i in ix:
    e=evidence[i] if len(evidence)==len(raw) else {};n=_normal(raw,int(i),e)
    if not np.isfinite(n).all():continue
    points=_points(e.get('actual_source_sides_uv'))
    if len(points)!=2 or np.linalg.norm(points[1]-points[0])<.25:points=raw[int(i):int(i)+1]
    for side,point in enumerate(points):families[fid].append({'group_id':group['group_id'],'raw_index':int(i),'side':side,'uv':point,'normal':n,'raw_kind':kind})
  for fid,items in sorted(families.items()):
   seen=set();unique=[]
   for item in items:
    key=tuple(item['uv'])
    if key in seen:continue
    seen.add(key);unique.append(item)
   selected=np.unique(np.linspace(0,len(unique)-1,min(max_samples,len(unique))).astype(int))
   for k in selected:
    item=unique[int(k)];q=item['uv'];n=item['normal'];t=np.array([-n[1],n[0]])
    coords=q+toff[:,None,None]*t+off[None,:,None]*n
    allc=_sample(c,coords.reshape(-1,2)).reshape(5,25,3);gradient=np.gradient(allc[2],.25,axis=0);ch=int(np.abs(gradient).max(axis=0).argmax());a=allc[:,:,ch]
    i=len(index);profiles.append(a.astype(np.float32));uv.append(q);normals.append(n);channel.append(ch)
    index.append({'profile_index':i,'source_family_id':fid,'source_group_id':item['group_id'],'raw_index':item['raw_index'],'side':item['side'],'native_pts':fr['native_pts'],'shot_id':fr['shot_id'],'native_png_sha256':fr['native_png_sha256'],'source_kind':item['raw_kind'],'family_actual_available_locations':len(unique),'selection_rule':'uniform original readout order over deduplicated actual support','pixels_from_model':False})
  print(json.dumps({'pts':fr['native_pts'],'profiles':len(index)}),flush=True)
 np.savez_compressed(out/'native_profiles.npz',intensity=np.stack(profiles),uv=np.asarray(uv,dtype=np.float64),normal=np.asarray(normals,dtype=np.float64),channel=np.asarray(channel,dtype=np.uint8),normal_offsets=off,tangent_offsets=toff)
 write_jsonl(out/'profile_index.jsonl.gz',index)
 result={'schema':'rinkfit.native_source_localization_profiles.v1','profile_count':len(index),'frame_count':len(frames),'family_count':len({x['source_family_id'] for x in index}),'maximum_locations_per_family':max_samples,'native_manifest_sha256':file_sha256(native/'manifest.json'),'arrays_sha256':file_sha256(out/'native_profiles.npz'),'index_sha256':file_sha256(out/'profile_index.jsonl.gz'),'sampling':'five actual parallel native profiles; gaussian channel sigma=.8px; correlations retained','noise_variance_claimed':False,'duplicate_statistical_observations':False}
 write_json(out/'manifest.json',result);return result

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--normalized',required=True);ap.add_argument('--native',required=True);ap.add_argument('--out',required=True);ap.add_argument('--max-samples',type=int,default=16);a=ap.parse_args();print(json.dumps(extract_profiles(a.normalized,a.native,a.out,a.max_samples)))
