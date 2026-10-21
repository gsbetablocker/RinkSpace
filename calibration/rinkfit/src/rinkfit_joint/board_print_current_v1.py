"""Actual frozen-profile/role consumer for a source-only board successor.

Does not reinterpret a capture checkpoint as formal explicit state. Existing
source arrays and W stay exact; E dependency components are only conservatively
joined through the actual same native RGB at the new batch boundary.
"""
from pathlib import Path
import copy,collections,hashlib,json
import numpy as np
from .io_utils import read_json,read_jsonl,write_json,file_sha256
from .packet_storage import write_packet
from .r_inner_prefix_v1 import array
from .board_print_append_v1 import checked,require,ref
SCHEMA='rinkfit.current_printed_board_source_packet_contract.v1'
def finalize(prototype,output):
 src=Path(prototype).resolve();out=Path(output).resolve();require(not out.exists(),'immutable dependency successor exists');p=read_json(src);parentpath=Path(p['source_only_printed_board_successor']['parent_packet']['file']);parent=read_json(parentpath);seen=set();qa=read_json(checked(p['source_only_printed_board_successor']['qualification'],seen));proof=list(read_jsonl(checked(qa['profile_proofs'],seen)));n=parent['row_count'];N=p['row_count'];A={k:array(p,k,src.parent,seen)for k in p['arrays']};active=(A['admission']==1)&(A['fit_role']==1);oldgroups=parent['engineering_pixel_dependency_groups']['groups'];d=np.arange(N,dtype=np.int64)
 def root(i):
  while d[i]!=i:d[i]=d[d[i]];i=int(d[i])
  return i
 def union(a,b):
  a=root(a);b=root(b)
  if a!=b:d[max(a,b)]=min(a,b)
 for g in oldgroups:
  rr=g['rows']
  for i in rr[1:]:union(rr[0],i)
 byframe=collections.defaultdict(list);actualrgb={};newby=collections.defaultdict(list)
 for j,q in enumerate(proof):
  i=n+j
  if not active[i]:continue
  key=(q['shot_id'],q['native_pts']);require(key not in actualrgb or actualrgb[key]==q['actual_RGB_sha256'],'New same exposure has conflicting original native RGB');actualrgb[key]=q['actual_RGB_sha256'];newby[key].append(i)
 for i in np.flatnonzero(active):
  key=(p['shot_ids'][int(A['shot_index'][i])],int(A['native_pts'][i]))
  if key in actualrgb:byframe[key].append(int(i))
 for rr in byframe.values():
  for i in rr[1:]:union(rr[0],i)
 affected={root(i)for rr in newby.values()for i in rr};components=collections.defaultdict(list)
 for i in np.flatnonzero(active):
  r=root(i)
  if r in affected:components[r].append(int(i))
 groups=[copy.deepcopy(g)for g in oldgroups if not g['rows']or root(g['rows'][0])not in affected];oldids=collections.defaultdict(list)
 for g in oldgroups:
  if g['rows']and root(g['rows'][0])in affected:oldids[root(g['rows'][0])].append(g['id'])
 for k,rr in sorted(components.items()):groups.append(dict(id='board_same_RGB_closure:'+hashlib.sha256(np.asarray(rr,dtype='<u8').tobytes()).hexdigest()[:24],rows=rr,inherited_dependency_groups=oldids[k],source_exposure_pixel_dependence_preserved=True,source_independence_or_noise_precision_claimed=False))
 out.mkdir();closure=dict(schema='rinkfit.actual_board_same_native_RGB_dependency_closure.v1',parent_packet=ref(parentpath),actual_source_qualification=p['source_only_printed_board_successor']['qualification'],rows=[dict(shot_id=k[0],native_pts=k[1],actual_RGB_sha256=actualrgb[k],old_joint_rows=[i for i in rr if i<n],new_joint_rows=newby[k],independent_pixel_noise_claimed=False)for k,rr in sorted(byframe.items())],old_dependency_groups_not_split=True,new_noise_W_or_precision_issued=False)
 write_json(out/'same_RGB_closure.json',closure);p['engineering_pixel_dependency_groups']['groups']=groups;p['engineering_pixel_dependency_groups']['board_current_same_RGB_closure']=ref(out/'same_RGB_closure.json');p['source_only_printed_board_successor']['source_dependency_closure']=ref(out/'same_RGB_closure.json');p['source_only_printed_board_successor']['source_consumer']=ref(__file__)
 for f in [Path(__file__),Path(__file__).with_name('geometry_source_append_v2.py')]:p['source_manifests'].append(ref(f))
 p['source_manifests'].append(ref(out/'same_RGB_closure.json'));write_packet(out/'packet.json',p);report=assert_current(out/'packet.json');write_json(out/'source_guard_positive.json',report);return report

def assert_current(packet_path):
 path=Path(packet_path).resolve();p=read_json(path);b=p['source_only_printed_board_successor'];seen=set();require('source_capture_complete_state_binding'not in p and b.get('parent_numeric_binding_not_transferred')is True,'Source successor cannot inherit stale numeric CP identity');require(b['schema']=='rinkfit.current_printed_board_source_append.v1'and b['prototype_only']and not b['accepted_physical_parent_bound'],'Capture parent must be bound by a separate genuine capture-state contract');parentpath=checked(b['parent_packet'],seen);old=read_json(parentpath);qa=read_json(checked(b['qualification'],seen));checked(qa['consumer'],seen)
 for r in qa['actual_dependencies']:checked(r,seen)
 for r in p['source_manifests']:checked(r,seen)
 raw=list(read_jsonl(checked(qa['raw_compact'],seen)));proof=list(read_jsonl(checked(qa['profile_proofs'],seen)));aliases=read_json(checked(p['new_actual_profile_aliases'],seen));aliasby={x['joint_row']:x for x in aliases};n=old['row_count'];N=p['row_count'];require(N==n+len(raw)==n+len(proof),'Board source exact raw denominator');require(p['shot_ids']==old['shot_ids']and p['material_binding']==old['material_binding'],'Board changed inherited physical material/clock identity');A={k:array(p,k,path.parent,seen)for k in p['arrays']}
 for k in old['arrays']:require(A[k][:n].tobytes()==array(old,k,parentpath.parent,seen).tobytes(),'Board source prefix changed '+k)
 for k in ('source_row_ids','source_dataset_ids','source_family_ids'):require(p[k][:n]==old[k]and len(p[k])==N,'Inherited exact source identity '+k)
 oldkeys={(old['source_dataset_ids'][i],int(A['shot_index'][i]),int(A['native_pts'][i]),old['source_row_ids'][i])for i in range(n)};newkeys=[(p['source_dataset_ids'][i],int(A['shot_index'][i]),int(A['native_pts'][i]),p['source_row_ids'][i])for i in range(n,N)];require(len(set(newkeys))==N-n and not oldkeys.intersection(newkeys),'New currentselected source identity duplicated');require(len(set(p['source_row_ids'][n:]))==N-n and not set(p['source_row_ids'][n:]).intersection(old['source_row_ids']),'New board source ID reused')
 for k in ('queries','groups'):require(p[k][:len(old[k])]==old[k],'Inherited full Query/W group prefix changed')
 for k in ('stroke_pairs','independent_rows','independent_statistic_kind','joint_statistics_rank','source_station_domains'):require(p.get(k)==old.get(k),'Inherited source mathematics changed '+k)
 qs={q['row_index']:q for q in p['queries'][len(old['queries']):]};admitted=[]
 for j,(r,q)in enumerate(zip(raw,proof)):
  i=n+j;require(r['source_row_id']==q['source_row_id']==p['source_row_ids'][i]and int(A['native_pts'][i])==q['native_pts']and p['shot_ids'][int(A['shot_index'][i])]==q['shot_id'],'Appended board source clock/identity differs');require(np.array_equal(A['uv'][i],q['source_uv'])and np.array_equal(A['query_uv'][i],q['source_uv'])and np.array_equal(A['profile_normal'][i],q['normal']),'Appended board actual original UV/normal changed');ok=q['qualified']and i not in aliasby;require(int(A['admission'][i])==int(ok)and int(A['fit_role'][i])==1,'Unknown/wrong-role source or duplicate admitted');require(int(A['kind'][i])==0 and int(A['material_index'][i])==q['joint_material_index'],'Appended board finite carrier changed')
  if i in aliasby:
   a=aliasby[i];c=int(a['canonical_joint_row']);require(n<=c<i and a['canonical_source_row_id']==p['source_row_ids'][c]and a['physical_qualification_withdrawn']is False,'Board alias is not a retained prior canonical actual readout');prev=proof[c-n];require(all(q[z]==prev[z]for z in('source_uv','seed_uv','normal','normal_offsets','channel','polarity','actual_intensity_binary64_sha256','actual_RGB_sha256','joint_material_index')),'Merged board aliases are different actual profile inputs')
  if not ok:require(i not in qs,'Inactive raw board row has a solver Query');continue
  admitted.append(i);off=np.asarray(q['normal_offsets']);seed=np.asarray(q['seed_uv']);normal=np.asarray(q['normal']);query=qs.get(i);require(query is not None and query['source_row_id']==r['source_row_id']and query['alpha']==off.tolist()and query['native_uv']==(seed[None,:]+off[:,None]*normal[None,:]).tolist()and query['source_profile_channel']==q['channel']and query['source_profile_polarity']==q['polarity'],'Original 49-point board Query changed');require(float(A['alpha'][i])==q['alpha']and np.array_equal(A['alpha_bounds'][i],[off[0],off[-1]]),'Actual board alpha/window not initialized from original optical readout');require(p['allowed_paths'][str(i)]==list(range(8)),'Board finite path branch set redirected')
 require(set(qs)==set(admitted),'Unexpected added board Query');grows=[]
 for g in p['groups'][len(old['groups']):]:require(g['W']==[]and g['rank']==0 and g['statistic_kind']=='constraint_only_unknown_noise'and not g['unknown_noise_source_has_statistical_precision'],'Invented new board noise W/rank');grows+=g['rows']
 require(sorted(grows)==admitted,'New unknown-noise source partition differs')
 dep=p['engineering_pixel_dependency_groups']['groups'];owner={}
 for gi,g in enumerate(dep):
  for i in g['rows']:require(i not in owner,'Overlapping E dependence partition');owner[i]=gi
 for g in old['engineering_pixel_dependency_groups']['groups']:require(len({owner[i]for i in g['rows']})<=1,'Old E source dependence group split')
 closure=read_json(checked(b['source_dependency_closure'],seen))
 for c in closure['rows']:require(len({owner[i]for i in c['old_joint_rows']+c['new_joint_rows']})==1,'Actual same RGB source split into independent E groups')
 active=(A['admission']==1)&(A['fit_role']==1);counts=collections.Counter((int(A['shot_index'][i]),int(A['native_pts'][i]))for i in np.flatnonzero(active));require(max(counts.values())<=500,'Actual native frame exceeds500 measurements')
 return dict(schema=SCHEMA,passed=True,packet=ref(path),parent_packet=ref(parentpath),actual_source_qualification=b['qualification'],old_raw_rows=n,rows=N,old_all18_arrays_W_and_Query_exact=True,new_raw_rows=len(raw),new_physical_source_qualified=int(sum(q['qualified']for q in proof)),exact_profile_aliases=len(aliases),new_admitted_rows=len(admitted),new_unknown_or_wrong_role_raw=len(raw)-sum(q['qualified']for q in proof),original_49_profile_Queries_actually_consumed=len(qs),new_W_rank=0,same_RGB_closure_frames=len(closure['rows']),inherited_E_groups_not_split=True,max_actual_fitted_points_per_native_frame=max(counts.values()),accepted_physical_parent_bound=False,solver_started=False)
