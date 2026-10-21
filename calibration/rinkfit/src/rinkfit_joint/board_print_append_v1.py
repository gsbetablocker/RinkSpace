"""Immutable current-role board source append, with real frozen profile Queries.

This module neither starts a solver nor upgrades source-noise or CP semantics.
A source packet prototype may be constructed before the accepted physical parent;
complete capture-state import is deliberately a separate versioned boundary.
"""
from pathlib import Path
import copy,gzip,json,struct,collections
import numpy as np
from .io_utils import read_json,read_jsonl,write_json,file_sha256,object_sha256
from .geometry_source_append_v2 import append_geometry_sources
from .packet_storage import write_packet
from .r_inner_prefix_v1 import array
SOURCE_SHA='e8b53966d56e72f3842d92c3ecbe6e9cc7e4670b7d0c9bd12f58fd2cff5ad57d'
SCHEMA='rinkfit.current_printed_board_source_append.v1'
KEEP=('source_row_id','source_family_id','shot_id','native_pts','source_uv','current_uv','normal','joint_material_id','joint_material_index','point_correspondence','source_geometry_eligible','solver_eligible_for_statistical_objective','statistical_objective_eligible','numeric_statistics_ready','solver_eligible')
def ref(p):return dict(file=str(Path(p).resolve()),sha256=file_sha256(p))
def require(ok,text):
 if not ok:raise ValueError(text)
def checked(v,seen):
 p=Path(v.get('file',v.get('path'))).resolve();key=(str(p),v['sha256'])
 if key not in seen:
  from .exact_dependency_aliases import resolve_dependency
  p=resolve_dependency(v);key=(str(p),v['sha256']);seen.add(key)
 return p
def jload(p):
 p=Path(p)
 if p.suffix=='.gz':
  with gzip.open(p,'rt',encoding='utf8')as f:return json.load(f)
 return read_json(p)
def compact(r):return {k:r[k]for k in KEEP if k in r}
def qualify_bundle(bundle_path,output):
 out=Path(output);require(not out.exists(),'immutable board qualification exists');bundle=read_json(bundle_path);seen=set();require(bundle['schema']=='rinkfit.board_geometry_source_bundle.v1'and bundle['source_geometry_only']and bundle['requirements']['source_noise_W_not_supplied'],'Unknown board source bundle semantics')
 for c in bundle['code_refs']:checked(c,seen)
 checked(bundle['current_exact_qualification']['recipe'],seen);out.mkdir(parents=True)
 rawfile=out/'raw_compact.jsonl.gz';geofile=out/'geometry_compact.jsonl.gz';prooffile=out/'profile_proofs.jsonl.gz';counts=collections.Counter();frame_counts=collections.defaultdict(collections.Counter);allids=set();max_replay=0.;stats=collections.Counter()
 with gzip.open(rawfile,'wt',encoding='utf8')as rawout,gzip.open(geofile,'wt',encoding='utf8')as geoout,gzip.open(prooffile,'wt',encoding='utf8')as proofout:
  for part in bundle['source_parts']:
   cm=read_json(checked(part['chunk'],seen));original={r['source_row_id']:r for r in read_jsonl(checked(cm['original_source_journal'],seen))};geometry={r['source_row_id']:r for r in read_jsonl(checked(part['geometry_rows_ref'],seen))};queries={r['source_row_id']:r for r in read_jsonl(checked(part['frozen_queries_ref'],seen))};frames={};members={}
   for mf in part['current_exact_qualifications']:
    m=read_json(checked(mf,seen));pts=int(m['native_pts']);require(m['raw_source_UV_Query_W_or_solver_packet_unchanged']and m['max_actual_RGB_cubic_UV_replay_error_px']==0.,'Board original actual RGB replay failed')
    for x in list(m['files'].values())+[m['source_frame_proof'],m['source_journal'],m['current_mask'],m['recipe']]:checked(x,seen)
    require(m['native_receipt']['source_video_sha256']==SOURCE_SHA and m['native_receipt']['native_pts']==pts and not m['native_receipt']['native_RGB_resampled'],'Wrong official board source clock/RGB')
    frames[pts]=m;members[pts]={x['source_row_id']:x for x in jload(m['files']['membership.json.gz']['file'])}
   used=set()
   for r in read_jsonl(checked(part['raw_ledger_ref'],seen)):
    rid=r['source_row_id'];pts=int(r['native_pts']);require(rid not in allids and rid in original and rid in queries and pts in frames,'Board row duplicated/missing exact current source');allids.add(rid);used.add(rid);old=original[rid];q=queries[rid];m=frames[pts];member=members[pts][rid];h=object_sha256(old);mu=np.asarray(r['current_uv'],float);normal=np.asarray(r['normal'],float);seed=np.asarray(old['navigation_seed_uv'],float)
    require(h==r['original_source_record_sha256']==q['exact_original_source_record_sha256']==member['source_record_sha256'],'Board original full source-record lineage changed')
    require(struct.pack('<2d',*mu).hex()==r['raw_UV_binary64_le']==member['current_UV_binary64_le']and r['source_uv']==r['current_uv']==old['current_uv']==q['source_uv'],'Original board binary64 UV changed')
    require(r['native_video_decode_receipt']==m['native_receipt']and r['current_mask']==m['current_mask'],'Board row actual exposure/mask changed')
    cert=r['measurement_contract_certificate'];ok=member['new_status']=='pass';require(r['source_geometry_eligible']==ok and (rid in geometry)==ok and r['source_admitted']==ok,'Current board role membership differs')
    if ok:require(geometry[rid]==r and cert['actual_RGB_material_identity_verified']and cert['operator_selected_correct_material_transition']and cert['operator_qualification_status']=='pass'and r['source_material_role_requalification']['status']=='pass','Board geometry bypassed actual material/operator qualification')
    require(not any(r[x]for x in ('solver_eligible','solver_eligible_for_statistical_objective','statistical_objective_eligible','numeric_statistics_ready')),'Unissued board statistical precision')
    profile=old['source_measurement_audit']['profile'];continuous=old['source_measurement_audit']['continuous_cubic_material_profile'];offset=np.asarray(profile['normal_offsets'],float);t=float(continuous['normal_offset_px']);require(offset.shape==(49,)and np.all(np.diff(offset)>0)and np.array_equal(offset,q['normal_offsets'])and np.array_equal(offset,continuous['actual_profile_normal_offsets'])and q['channel']==profile['channel']and q['polarity']==profile['polarity']and q['normal']==old['normal']and q['seed_uv']==old['navigation_seed_uv']and q['operator_version']==continuous['operator_version']=='fixed_outer_material_continuous_cubic_peak_v1','Frozen board 49-sample Query changed')
    require(normal.shape==(2,)and abs(np.linalg.norm(normal)-1)<1e-9 and offset[0]<=t<=offset[-1],'Invalid actual board finite Query domain')
    error=float(np.max(abs(seed+t*normal-mu)));require(error<=1e-9,'Board Query does not reproduce unchanged actual optical mean');max_replay=max(max_replay,error)
    iref=q['intensity_reference'];pf=checked(iref,seen)
    with np.load(pf)as data:values=np.asarray(data[iref['array']][int(iref['row_index'])],float)
    require(values.shape==(49,)and np.isfinite(values).all(),'Missing real board intensity profile')
    for x in cert['source_evidence']:
     if x.get('file')or Path(x.get('path','')).is_absolute():checked(x,seen)
    rec=dict(source_row_id=rid,source_family_id=r['source_family_id'],shot_id=r['shot_id'],native_pts=pts,qualified=ok,combined_source_status=member['new_status'],joint_material_index=r['joint_material_index'],source_uv=r['source_uv'],seed_uv=seed.tolist(),normal=normal.tolist(),alpha=t,normal_offsets=offset.tolist(),channel=q['channel'],polarity=q['polarity'],intensity_reference=iref,actual_intensity_binary64_sha256=__import__('hashlib').sha256(values.astype('<f8').tobytes()).hexdigest(),source_journal=r['source_journal'],original_source_record_sha256=h,current_exact_membership_ref=r['current_role_operator_membership_ref'],actual_RGB_sha256=m['native_receipt']['decoded_native_RGB_sha256'],source_noise_W_not_created=True)
    rawout.write(json.dumps(compact(r),separators=(',',':'))+'\n');proofout.write(json.dumps(rec,separators=(',',':'))+'\n')
    if ok:geoout.write(json.dumps(compact(r),separators=(',',':'))+'\n')
    counts[member['new_status']]+=1;frame_counts[(r['shot_id'],pts)]['raw']+=1;frame_counts[(r['shot_id'],pts)]['qualified']+=int(ok);stats['raw']+=1;stats['qualified']+=int(ok)
   require(used==set(queries)and set(geometry)<used,'Board raw/Query/geometry exact closed denominator differs')
 require(stats['raw']==bundle['counts']['raw_rows']and stats['qualified']==bundle['counts']['source_geometry_qualified_rows'],'Board source bundle count mismatch')
 report=dict(schema=SCHEMA,bundle=ref(bundle_path),consumer=ref(__file__),raw_rows=stats['raw'],qualified_rows=stats['qualified'],status_counts=dict(counts),frames=len(frame_counts),frame_counts=[dict(shot_id=k[0],native_pts=k[1],**v)for k,v in sorted(frame_counts.items())],actual_49_sample_profile_mean_replay_max_px=max_replay,actual_dependencies=[dict(file=k[0],sha256=k[1])for k in sorted(seen)],raw_compact=ref(rawfile),geometry_compact=ref(geofile),profile_proofs=ref(prooffile),unknown_and_wrong_role_retained_in_raw=True,source_noise_precision_issued=False,solver_started=False)
 write_json(out/'qualification.json',report);return report

def build_prototype(parent_packet,topology_model,qualification_file,output):
 """Actually append qualified geometry and frozen Queries; no accepted CP claim."""
 out=Path(output);require(not out.exists(),'immutable board packet prototype exists');parent=read_json(parent_packet);qa=read_json(qualification_file);seen=set();raw=list(read_jsonl(checked(qa['raw_compact'],seen)));geo=list(read_jsonl(checked(qa['geometry_compact'],seen)));proof={r['source_row_id']:r for r in read_jsonl(checked(qa['profile_proofs'],seen))};oldids=set(parent['source_row_ids']);require(not oldids.intersection(proof),'Board source IDs already in current selected packet')
 model=read_json(topology_model);out.mkdir(parents=True);mapping={'semantic_to_material':{}}
 for key,mi in [('board.top|board|board_upper_material_edge|top',348),('board.contact|board|ice_board_interface|contact',349)]:
  require(model['materials'][mi]['label']==key.split('|')[0]and len(model['materials'][mi]['paths'])==8,'Existing finite board carrier changed');mapping['semantic_to_material'][key]=dict(material_index=mi,geometry_ready=True)
 write_json(out/'material_map.json',mapping)
 append_geometry_sources(parent_packet,qa['geometry_compact']['file'],qa['raw_compact']['file'],out/'material_map.json',topology_model,out/'append',qualification_file,write_provisional_W=False)
 p=read_json(out/'append/packet.json');n=parent['row_count'];N=p['row_count'];A={k:np.array(array(p,k,out/'append',seen))for k in p['arrays']};active=(A['admission']==1)&(A['fit_role']==1);oldactive=(array(parent,'admission',Path(parent_packet).parent,seen)==1)&(array(parent,'fit_role',Path(parent_packet).parent,seen)==1);counts=collections.Counter((parent['shot_ids'][int(s)],int(t))for s,t in zip(array(parent,'shot_index',Path(parent_packet).parent,seen)[oldactive],array(parent,'native_pts',Path(parent_packet).parent,seen)[oldactive]));newgroups=collections.defaultdict(list);queries=list(parent['queries']);source_refs=[ref(qualification_file),qa['bundle']];alias=[]
 olduv=array(parent,'uv',Path(parent_packet).parent,seen);oldmi=array(parent,'material_index',Path(parent_packet).parent,seen);oldpts=array(parent,'native_pts',Path(parent_packet).parent,seen);oldshot=array(parent,'shot_index',Path(parent_packet).parent,seen);keyset={(int(oldshot[i]),int(oldpts[i]),int(oldmi[i]),olduv[i].astype('<f8').tobytes()):i for i in np.flatnonzero(oldactive)}
 for j,r in enumerate(raw):
  i=n+j;q=proof[r['source_row_id']]
  if not q['qualified']:continue
  key=(int(A['shot_index'][i]),int(A['native_pts'][i]),int(A['material_index'][i]),A['uv'][i].astype('<f8').tobytes());
  # Equal means from different real optical profiles remain correlated sources.
  if key in keyset and keyset[key]>=n:
   canonical=keyset[key];previous=proof[p['source_row_ids'][canonical]]
   if all(q[z]==previous[z]for z in ('seed_uv','normal','normal_offsets','channel','polarity','actual_intensity_binary64_sha256','actual_RGB_sha256','joint_material_index')):
    A['admission'][i]=0;alias.append(dict(joint_row=i,canonical_joint_row=canonical,source_row_id=r['source_row_id'],canonical_source_row_id=p['source_row_ids'][canonical],exact_same_frozen_actual_profile_input=True,physical_qualification_withdrawn=False));continue
  keyset[key]=i;counts[(r['shot_id'],int(r['native_pts']))]+=1;newgroups[(r['shot_id'],r['native_pts'],r['source_family_id'])].append(i)
  off=np.asarray(q['normal_offsets']);seed=np.asarray(q['seed_uv']);normal=np.asarray(q['normal']);A['alpha'][i]=q['alpha'];A['alpha_bounds'][i]=[off[0],off[-1]]
  queries.append(dict(row_index=i,source_row_id=r['source_row_id'],alpha=off.tolist(),native_uv=(seed[None,:]+off[:,None]*normal[None,:]).tolist(),normal=normal.tolist(),actual_acquisition_window_preserved=True,deterministic_query_is_not_tangential_noise_precision=True,frozen_profile_proof=qa['profile_proofs'],source_profile_channel=q['channel'],source_profile_polarity=q['polarity']))
 require(max(counts.values())<=500,'New source packet violates per-frame 500 actual fitted measurements')
 for k in A:
  require(A[k][:n].tobytes()==array(parent,k,Path(parent_packet).parent,seen).tobytes(),'Board prototype changed inherited source/aux prefix '+k)
  if k in('alpha','alpha_bounds','admission'):
   f=out/(k+'.bin');A[k].tofile(f);p['arrays'][k]={**p['arrays'][k],**ref(f)}
 p['queries']=queries;p['active_rows']=int(np.count_nonzero((A['admission']==1)&(A['fit_role']==1)));p['recapture_rows']=np.flatnonzero((A['admission']==1)&(A['fit_role']==1)&(np.arange(N)>=p['new_source_start'])).tolist();write_json(out/'actual_profile_aliases.json',alias);p['new_actual_profile_aliases']=ref(out/'actual_profile_aliases.json')
 for key,rows in sorted(newgroups.items()):p['groups'].append(dict(id='current_board_geometry:'+key[2],source_group_id=key[2],rows=rows,rank=0,W=[],statistic_kind='constraint_only_unknown_noise',unknown_noise_source_refs=source_refs,unknown_noise_source_has_statistical_precision=False,E_RS_Query_and_finite_material_constraints_retained=True))
 p['constraint_only_unknown_noise_rows']=sum(len(g['rows'])for g in p['groups']if g.get('statistic_kind')=='constraint_only_unknown_noise');p['constraint_only_unknown_noise_groups']=sum(g.get('statistic_kind')=='constraint_only_unknown_noise'for g in p['groups']);p['source_family_journal_refs']=parent.get('source_family_journal_refs',[])
 p['source_manifests']+=source_refs+[qa['profile_proofs'],ref(__file__)];p['source_only_printed_board_successor']=dict(schema=SCHEMA,parent_packet=ref(parent_packet),qualification=ref(qualification_file),prototype_only=True,accepted_physical_parent_bound=False,requires_v5_actual_accepted_complete_capture_primal=True,old_capture_state_not_formal_explicit_or_d78_exact=True,full_inherited_aux_rebind_pending=True,new_raw_rows=len(raw),new_active_geometry_rows=len(geo)-len(alias),new_frozen_49_sample_queries=len(geo)-len(alias),new_statistical_rank=0,source_noise_unknown=True,placeholder_source_sigma_not_a_1px_noise_claim=True,formal_Query_W_capture_mode_consumption_complete=False)
 p['source_only_printed_board_successor']['parent_numeric_binding_not_transferred']=True
 if 'source_capture_complete_state_binding'in p:p['source_only_printed_board_successor']['archived_parent_capture_binding']=p.pop('source_capture_complete_state_binding')
 p['queries_consumed_from_original_49_samples']=True;p['physical_noise_likelihood_claimed']=False;write_packet(out/'packet.json',p)
 report=dict(schema=SCHEMA,passed=True,packet=ref(out/'packet.json'),parent_packet=ref(parent_packet),qualification=ref(qualification_file),old_all18_source_and_aux_prefix_exact=True,old_W_groups_queries_and_station_domains_exact=True,new_raw_rows=len(raw),new_active_geometry_rows=len(geo)-len(alias),new_unknown_or_wrong_role_raw=len(raw)-len(geo),new_source_qualified_exact_profile_aliases=len(alias),source_qualified_rows=len(geo),new_Query_count=len(geo)-len(alias),new_49_profile_sample_count=49*(len(geo)-len(alias)),max_active_points_per_frame=max(counts.values()),active_rows=p['active_rows'],rows=N,new_W_rank=0,old_joint_statistics_rank_unchanged=p['joint_statistics_rank']==parent['joint_statistics_rank'],accepted_physical_parent_bound=False,solver_started=False,frame_counts=[dict(shot_id=k[0],native_pts=k[1],active_points=v)for k,v in sorted(counts.items())]);write_json(out/'report.json',report);return report
