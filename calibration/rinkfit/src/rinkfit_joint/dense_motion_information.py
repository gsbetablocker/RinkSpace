"""Source-support de-correlation of actual two-exposure static board matches.

The selection limits overlapping 21-pixel LK supports, not frame or sample counts.
It neither changes source UV nor creates world points, tracks, or fit factors.
"""
from pathlib import Path
import argparse,gzip,json
from collections import Counter,defaultdict
import numpy as np
from .io_utils import file_sha256,read_json,read_jsonl,write_json


def _ref(p):
 p=Path(p).resolve();return dict(file=str(p),sha256=file_sha256(p))


def _independent(source,current,fb,patch_size):
 # Axis-aligned supports of the actual LK reader must be disjoint in both images.
 order=sorted(range(len(fb)),key=lambda i:(fb[i],source[i,0],source[i,1],i))
 bins=[defaultdict(list),defaultdict(list)];accepted=[]
 for i in order:
  overlap=False
  for image,uv in enumerate((source,current)):
   cell=np.floor(uv[i]/patch_size).astype(int)
   for dx in (-1,0,1):
    for dy in (-1,0,1):
     for j in bins[image].get((int(cell[0]+dx),int(cell[1]+dy)),()):
      if np.all(np.abs(uv[i]-uv[j])<patch_size):overlap=True;break
     if overlap:break
    if overlap:break
   if overlap:break
  if overlap:continue
  accepted.append(i)
  for image,uv in enumerate((source,current)):
   key=tuple(np.floor(uv[i]/patch_size).astype(int));bins[image][key].append(i)
 return np.array(sorted(accepted),dtype='<u4')


def select_board_motion_information(pair_index,output):
 pair_index=Path(pair_index).resolve();output=Path(output).resolve()
 if output.exists():raise FileExistsError(output)
 for p in (output,output/'selected.u32',output/'pair_offsets.u64',output/'manifest.json'):
  if len(str(p).encode('utf-16-le'))//2>=260:raise ValueError('Windows output path too long')
 rows=list(read_jsonl(pair_index));selected=[];offsets=[0];counts=Counter();frames=[];seen=set()
 for pair_number,entry in enumerate(rows):
  p=Path(entry['file']);ref=_ref(p)
  if ref['sha256']!=entry['sha256']:raise ValueError('Changed source pair '+str(p))
  row=read_json(p);uv=np.asarray(row['source_uv'],float);cur=np.asarray(row['current_uv'],float);fb=np.asarray(row['forward_backward_error_px'],float)
  key=(row['shot_id'],row['previous_native_pts'],row['native_pts'])
  if key in seen or row['cross_cut'] or row['native_pts']-row['previous_native_pts']!=256:raise ValueError('Cut, duplicate, or non-native pair')
  seen.add(key)
  if row['shot_id'] not in ('s05','s06') or entry['shot_id']!=row['shot_id'] or entry['native_pts']!=row['native_pts'] or entry['previous_native_pts']!=row['previous_native_pts']:raise ValueError('Actual source pair clock differs')
  if row['source_world_projection_used']is not False or row['pixel_coordinates_not_model_prediction']is not True or row['full_native_21px_LK_patch_current_material_and_actor_and_graphics_exclusion']is not True or row['source_or_current_UV_not_moved']is not True:raise ValueError('Real static material and complete LK support evidence required')
  if row['current_source_RGB_sha256']!=entry['current_RGB_sha256'] or row['previous_source_RGB_sha256']!=entry['previous_RGB_sha256']:raise ValueError('RGB proof identity differs')
  if uv.shape!=(len(fb),2) or cur.shape!=uv.shape or not np.isfinite(uv).all() or not np.isfinite(cur).all() or not np.isfinite(fb).all() or (fb<0).any() or len(fb)!=entry['actual_2D_correspondence_pairs']:raise ValueError('Complete original UV/FB denominator differs')
  chosen=_independent(uv,cur,fb,21);selected.append(chosen);offsets.append(offsets[-1]+len(chosen));sid=row['shot_id'];counts[sid+'_raw']+=len(fb);counts[sid+'_selected']+=len(chosen)
  if len(fb) and not len(chosen):raise ValueError('Nonempty source exposure lost all information support')
  spread=np.linalg.eigvalsh(np.cov(uv[chosen].T,bias=True)) if len(chosen)>1 else np.array([0.,0.])
  frames.append(dict(pair_index=pair_number,shot_id=sid,previous_native_pts=row['previous_native_pts'],native_pts=row['native_pts'],raw_matches=len(fb),selected_matches=len(chosen),repeated_overlapping_support=len(fb)-len(chosen),spatial_covariance_eigenvalues_px2=spread.tolist()))
 output.mkdir(parents=True);np.concatenate(selected).tofile(output/'selected.u32');np.asarray(offsets,dtype='<u8').tofile(output/'pair_offsets.u64')
 with gzip.open(output/'pair_selection.jsonl.gz','wt',encoding='utf8')as f:
  for row in frames:f.write(json.dumps(row,separators=(',',':'))+'\n')
 result=dict(schema='rinkfit.actual_board_motion_support_information_selection.v1',pair_index=_ref(pair_index),reader=_ref(__file__),selected_indices=dict(**_ref(output/'selected.u32'),dtype='<u4',shape=[offsets[-1]]),pair_offsets=dict(**_ref(output/'pair_offsets.u64'),dtype='<u8',shape=[len(offsets)]),selection_ledger=_ref(output/'pair_selection.jsonl.gz'),raw_matches=sum(r['raw_matches']for r in frames),selected_matches=offsets[-1],pairs=len(frames),counts=dict(counts),minimum_selected_per_pair=min(r['selected_matches']for r in frames),maximum_selected_per_pair=max(r['selected_matches']for r in frames),empty_source_pairs=sum(not r['raw_matches']for r in frames),source_UV_copied_or_changed=False,world_points_created=False,long_track_identity_inferred=False,qualification_changed=False,fit_factors_installed=False,support_size_native_px=[21,21],selection_policy='ascending measured FB error then exact source coordinates; retain only patches disjoint in both actual images; no numerical sample cap',not_selected_reason='overlapping source support retained as correlated validation evidence',statistical_independence_or_noise_precision_claimed=False)
 write_json(output/'manifest.json',result);return result


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('pair_index');p.add_argument('output');a=p.parse_args();r=select_board_motion_information(a.pair_index,a.output);print(json.dumps({k:r[k]for k in ('raw_matches','selected_matches','pairs','counts','minimum_selected_per_pair','maximum_selected_per_pair')}))

if __name__=='__main__':main()
