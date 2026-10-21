"""Current native ring interfaces: exact reader lineage, finite branch and mask.
Source-only qualification; no W, world correspondence or solver-state restore.
"""
from pathlib import Path
from collections import Counter,defaultdict
import gzip,json,struct
import numpy as np
from .io_utils import read_json,read_jsonl,write_json,file_sha256

MAP={'R2:inner':288,'R2:middle':289,'R2:outer':290,'R3:inner':156,'R3:middle':157,'R3:outer':158}

def ref(p):return dict(file=str(Path(p).resolve()),sha256=file_sha256(p))
def checked(r):
 p=Path(r.get('file',r.get('path')))
 if file_sha256(p)!=r['sha256']:raise ValueError('changed actual source dependency '+str(p))
 return p

def _finite(raw,qi,ci,exclusion):
 q=np.asarray(raw['query_uv'][qi],float);n=np.asarray(raw['query_normal'][qi],float);v=np.asarray(raw['profile_valid'][qi],bool);offset=np.asarray(raw['profile_offsets'][qi],float)[v];signal=np.asarray(raw['profile_values'][qi],float)[v];smooth=np.asarray(raw['profile_smoothed'][qi],float)[v];mu=np.asarray(raw['candidate_uv'][ci],float)
 if int(raw['candidate_query'][ci])!=qi:raise ValueError('candidate-to-original-query identity')
 # Reconstruct the actual public signed-interface integration component.
 second=np.diff(smooth,n=2);noise=float(1.4826*np.median(np.abs(second-np.median(second))));gradient=np.diff(smooth);transition_sign=None;matches=[]
 for sign in(-1,1):
  active=sign*gradient>max(.25,noise);starts=np.flatnonzero(active & ~np.r_[False,active[:-1]]);ends=np.flatnonzero(active & ~np.r_[active[1:],False])+1;merged=[]
  for a,b in zip(starts,ends):
   if merged and a-merged[-1][1]<5:merged[-1][1]=int(b)
   else:merged.append([int(a),int(b)])
  for a,b in merged:
   lo=max(0,a-4);hi=min(len(offset)-1,b+4);mass=np.maximum(sign*gradient[lo:hi],0.);amount=float(mass.sum())
   if amount<=max(1.,3*noise):continue
   qs=np.interp([.25,.5,.75],np.r_[0.,np.cumsum(mass)]/amount,offset[lo:hi+1]);uvs=q[None]+n[None]*qs[:,None]
   if np.array_equal(uvs,np.asarray(raw['candidate_side_quantiles_uv'][ci,0])):matches.append((lo,hi,qs,uvs,sign))
 if len(matches)!=1:raise ValueError('selected source interface lacks unique actual finite integration branch')
 lo,hi,qs,uvs,sign=matches[0]
 if not np.array_equal(mu,uvs[1]):raise ValueError('actual selected mean differs from q50 reader output')
 original=offset[lo:hi+1];literal=q[None]+n[None]*original[:,None];alpha=original-qs[1]
 # Whole actual median-five / bilinear / Gaussian-five dependency footprint.
 guard_offset=offset[max(0,lo-2):min(len(offset),hi+3)];t=np.array([-n[1],n[0]]);xy=q[None,None]+n[None,None]*guard_offset[None,:,None]+t[None,None]*np.arange(-2,3)[:,None,None];xy=np.rint(xy*32.)/32.;pixels=np.floor(xy).astype(np.int64).reshape(-1,2);bad=[]
 for dx,dy in((0,0),(1,0),(0,1),(1,1)):
  px=pixels[:,0]+dx;py=pixels[:,1]+dy;inside=(px>=0)&(px<exclusion.shape[1])&(py>=0)&(py<exclusion.shape[0]);bad.extend(np.flatnonzero(~inside).tolist());bad.extend(np.flatnonzero(inside & exclusion[np.clip(py,0,exclusion.shape[0]-1),np.clip(px,0,exclusion.shape[1]-1)]).tolist())
 return dict(alpha=alpha.tolist(),native_uv=literal.tolist(),profile=signal[lo:hi+1].tolist(),profile_smoothed=smooth[lo:hi+1].tolist(),normal_uv=n.tolist(),query_origin_uv=q.tolist(),actual_selected_profile_offset=qs[1],source_mean_uv=mu.tolist(),material_transition='dark_to_light'if sign>0 else'light_to_dark',actual_source_quantiles_uv=uvs.tolist(),integration_indices=[lo,hi],source_q50_interval_membership=bool(original[0]<=qs[1]<=original[-1]),all_native_samples_exact_reader_formula=True,sampler='public material-reader native 1/32 bilinear median-five then Gaussian-five',sample_count=len(original),whole_actual_sampler_support_clear=not bad,masked_or_outside_dependency_sites=len(set(bad)),formal_noise_statistics_ready=False)

def _compile(generations,review_ref,model_ref,parent_packet_ref):
 import cv2
 from .r_inner_prefix_v1 import array
 review=read_json(checked(review_ref));approved={int(r['native_pts']):set(r['approved_targets'])for r in review['per_frame']};new=set(review['approved_new_exposures']);model=read_json(checked(model_ref));parentpath=checked(parent_packet_ref);parent=read_json(parentpath);seen=set();A={k:array(parent,k,parentpath.parent,seen)for k in('native_pts','admission','fit_role')};oldcounts=Counter(map(int,A['native_pts'][(A['admission']==1)&(A['fit_role']==1)]));rawrows=[];geometry=[];queries=[];counts=Counter();framecounts=Counter();query_unknown=0;allqueryrows=0;seen_ids=set();cache={};dependencies=[review_ref,model_ref,parent_packet_ref,ref(__file__)];exposures={}
 for gr in generations:
  gm=read_json(checked(gr));dependencies.append(gr)
  if gm.get('complete')is not True:raise ValueError('incomplete current source generation')
  for r in gm['reader_refs']:checked(r);dependencies.append(r)
  frames={}
  for fr in gm['closed_frames']:
   fm=read_json(checked(fr));frames[int(fm['native_pts'])]=fm;dependencies.append(fr)
  checked(gm['source_rows']);dependencies.append(gm['source_rows'])
  for r in read_jsonl(gm['source_rows']['file']):
   allqueryrows+=1;rid=r['source_row_id']
   if rid in seen_ids:raise ValueError('duplicate actual source query id')
   seen_ids.add(rid);pts=int(r['native_pts']);target=r['target_id'].removeprefix('ice.');fm=frames[pts];receipt=r['native_receipt'];exposures[pts]=receipt
   if receipt!=fm['native_receipt']or receipt['native_pts']!=pts or receipt['shot_id']!=r['shot_id']or receipt['time_base']!='1/12800':raise ValueError('source RGB/clock/shot identity')
   if receipt['source_video_sha256']!='e8b53966d56e72f3842d92c3ecbe6e9cc7e4670b7d0c9bd12f58fd2cff5ad57d':raise ValueError('wrong official source')
   if target not in MAP or model['materials'][MAP[target]]['label']!='ice.'+target:raise ValueError('wrong ring/layer material map')
   if 'source_uv'not in r:query_unknown+=1;counts['query_without_selected_native_interface']+=1;continue
   profile=checked(r['actual_profile']);dependencies.extend((r['actual_profile'],r['current_mask'],r['current_exclusion'],r['source_reference_packet']))
   if str(profile)not in cache:
    with np.load(profile,allow_pickle=False)as z:cache[str(profile)]={k:z[k].copy()for k in z.files}
   raw=cache[str(profile)];qi=int(r['actual_profile_query_index']);ci=int(r['actual_readout_support']['candidate_index']);mu=np.asarray(raw['candidate_uv'][ci]);normal=np.asarray(raw['query_normal'][qi]);sup=r['actual_readout_support']
   if mu.tolist()!=r['source_uv']or struct.pack('<2d',*mu).hex()!=r['source_UV_binary64_le']:raise ValueError('actual source mean binary64 changed')
   if normal.tolist()!=r['profile_normal']or sup['query_index']!=qi:raise ValueError('source normal/query identity changed')
   if int(raw['candidate_flags'][ci]) & 3:raise ValueError('rejected native support resurrected')
   maskpath=checked(r['current_exclusion']);ex=cv2.imread(str(maskpath),0)>0;q=_finite(raw,qi,ci,ex)
   if q['material_transition']!=r['material_transition']or not q['source_q50_interval_membership']:raise ValueError('wrong actual material direction/finite interval')
   role_ok=target in approved.get(pts,set());ok=role_ok and pts in new and q['whole_actual_sampler_support_clear'];reason='actual current ring/layer+finite signed-interface support qualified'if ok else'existing source exposure diagnostics only'if pts not in new else'full actual kernel touches current exclusion/outside'if not q['whole_actual_sampler_support_clear']else'current source ring/layer not visually qualified'
   z=dict(source_row_id=rid,source_family_id=f'ice_ring_native/{r["shot_id"]}/{pts}/{target}',shot_id=r['shot_id'],native_pts=pts,time_base='1/12800',target_id='ice.'+target,joint_material_id='ice.'+target+'|ice|material_interface|'+r['material_transition'],joint_material_index=MAP[target],source_uv=r['source_uv'],current_uv=r['source_uv'],normal=r['profile_normal'],physical_surface='ice',point_correspondence=False,kind='finite_material_curve',entity_station_fixed=False,source_geometry_eligible=ok,source_admitted=ok,solver_eligible=False,solver_eligible_for_statistical_objective=False,statistical_objective_eligible=False,numeric_statistics_ready=False,source_noise_W_supplied=False,source_pixel_identity_from_model=False,current_pixels_measured=True,source_acquisition_mode='same_current_native_signed_material_interface',current_native_receipt=receipt,current_mask=r['current_mask'],current_exclusion=r['current_exclusion'],actual_profile=r['actual_profile'],actual_profile_query_index=qi,actual_profile_candidate_index=ci,source_reference_packet=r['source_reference_packet'],source_visual_review=review_ref,reason=reason,finite_query=q,source_position_interval=r['source_position_interval'],source_native_sampling_uncertainty='source interface quartile blur width only; no calibrated sigma/W',independence_claimed=False)
   rawrows.append(z);counts[reason]+=1
   if ok:geometry.append(z);queries.append(dict(source_row_id=rid,**q));framecounts[pts]+=1
 for pts,n in framecounts.items():
  if n+oldcounts[pts]>500:raise ValueError('actual selected old+new frame limit exceeds500')
 for pts,receipt in exposures.items():
  if pts not in new:continue
  # Source role/clock must be explicit in the original authority; no schema-only promotion.
  from .role_guard import FrameRoleGuard
  for r in dependencies:
   if str(r.get('file',r.get('path'))).endswith('roles/frame_roles.json')or str(r.get('file',r.get('path'))).endswith('roles\\frame_roles.json'):
    FrameRoleGuard(checked(r)).assert_training_pixels([pts],shot_id=receipt['shot_id']);break
  else:raise ValueError('actual frame-role authority absent')
 return dict(raw=rawrows,geometry=geometry,queries=queries,counts=dict(counts),all_source_queries=allqueryrows,unknown_query_count=query_unknown,per_exposure=[dict(native_pts=pts,previous_selected_rows=oldcounts[pts],new_qualified_rows=n,combined_selected_rows=oldcounts[pts]+n)for pts,n in sorted(framecounts.items())],refs=dependencies)

def qualify(generations,review,model,parent_packet,output):
 out=Path(output)
 if out.exists():raise ValueError('new immutable source generation required')
 gens=[ref(p)for p in generations];rr=ref(review);mm=ref(model);pp=ref(parent_packet);x=_compile(gens,rr,mm,pp);out.mkdir(parents=True)
 files={}
 for key,name in(('raw','raw_ledger.jsonl.gz'),('geometry','geometry_rows.jsonl.gz'),('queries','finite_queries.jsonl.gz')):
  with gzip.open(out/name,'wt',encoding='utf8')as f:
   for r in x[key]:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
  files[name]=ref(out/name)
 manifest=dict(schema='rinkfit.actual_native_ring_source_qualification.v1',generations=gens,review=rr,model=mm,parent_source_packet=pp,consumer=ref(__file__),counts=x['counts'],all_source_query_rows=x['all_source_queries'],unknown_query_rows=x['unknown_query_count'],actual_native_mean_rows=len(x['raw']),geometry_qualified_new_rows=len(x['geometry']),finite_queries=len(x['queries']),per_exposure=x['per_exposure'],files=files,refs=x['refs'],unknown_queries_remain_in_complete_parent_generations=True,raw_navigation_UV_not_inserted_as_observations=True,geometry_is_not_numeric_W_or_world_correspondence=True,finite_source_branch_is_not_profile_statistics=True,current_solver_packet_modified=False)
 write_json(out/'manifest.json',manifest);return manifest

def assert_current(manifest):
 p=Path(manifest);m=read_json(p)
 if m['schema']!='rinkfit.actual_native_ring_source_qualification.v1'or m['consumer']!=ref(__file__):raise ValueError('wrong ring source consumer')
 for r in m['refs']:checked(r)
 x=_compile(m['generations'],m['review'],m['model'],m['parent_source_packet'])
 for key,name in(('raw','raw_ledger.jsonl.gz'),('geometry','geometry_rows.jsonl.gz'),('queries','finite_queries.jsonl.gz')):
  if list(read_jsonl(checked(m['files'][name])))!=x[key]:raise ValueError('actual ring source '+key+' identity/branch changed')
 if m['geometry_qualified_new_rows']!=len(x['geometry'])or m['counts']!=x['counts']or m['per_exposure']!=x['per_exposure']:raise ValueError('source-only eligibility/count changed')
 return dict(schema='rinkfit.actual_ring_source_consumer_receipt.v1',manifest=ref(p),geometry_qualified_rows=len(x['geometry']),finite_query_rows=len(x['queries']),actual_native_mean_rows=len(x['raw']),all_query_rows=x['all_source_queries'],unknown_queries_retained=x['unknown_query_count'],current_solver_packet_modified=False,W_or_world_truth_issued=False)
