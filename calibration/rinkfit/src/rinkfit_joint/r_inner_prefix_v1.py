"""Strict R inner-source append after an explicitly CP-bound V9 parent.
The historical R19 and V9 issuers are called unchanged. No source precision or
physical correspondence is manufactured by this versioned boundary adapter.
"""
from pathlib import Path
import copy,importlib.util,json,sys
import numpy as np
from .io_utils import read_json,write_json,file_sha256
from .accepted_aux_prefix import _digest,_prefix_hash
from .reference_source_append import append_constraint_source_chunks
from . import paint_operator_source_current as parent_guard

SCHEMA='rinkfit.actual_R_inner_source_after_bound_V9_prefix.v1'
CONTRACT='actual_33_R_inner_finite_source_after_CP_bound_V9_v1'
BINDING_SCHEMA='rinkfit.accepted_R_inner_source_append_state_packet_binding.v1'
STATE_SCHEMA='rinkfit.R_inner_append_inherited_engineering_state_contract.v1'
PREFIX='Ad4_R_inner_curvature_v1/'
SOURCE=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2/R_curvature_source_v1')
FROZEN={'strict_source_qualification.json':'903055c01267062bd75caf336b40e761ef6b5a92d0b162291ae500d1fee0f54d','generation/manifest.json':'ebae14abd504cd116c078933bb17841edaf0189da58f6faf4b00b88a2ffe8b0e','source_operator.py':'bfc8c0e8d565df7e95d8a8549dfe3f1fefaf9081c07c0700688bf40c886f7ae9','qualify.py':'b0dbff45e499ad34fe8c687cf673d6ad4483fb0693522549a5167f52fee05c2b'}

def ref(p):return dict(file=str(Path(p).resolve()),sha256=file_sha256(p))
def checked(r,base=None,seen=None):
 p=Path(r.get('file',r.get('path')));p=p if p.is_absolute()else Path(base or SOURCE)/p;p=p.resolve();k=(str(p),r['sha256'])
 if seen is None or k not in seen:
  if file_sha256(p)!=r['sha256']:raise ValueError('R source/state dependency changed '+str(p))
  if seen is not None:seen.add(k)
 return p

def array(d,k,base,seen=None):
 a=d['arrays'][k];return np.memmap(checked(a,base,seen),dtype=a['dtype'],mode='r',offset=a.get('offset_bytes',0),shape=tuple(a['shape']))
def source_module(name):
 p=checked(dict(file=str(SOURCE/(name+'.py')),sha256=FROZEN[name+'.py']));spec=importlib.util.spec_from_file_location('_rinkfit_frozen_R_inner_'+name,p);m=importlib.util.module_from_spec(spec)
 if str(SOURCE)not in sys.path:sys.path.insert(0,str(SOURCE))
 spec.loader.exec_module(m);return m

def source_proofs(*,replay=False):
 for n,h in FROZEN.items():checked(dict(file=str(SOURCE/n),sha256=h))
 q=read_json(SOURCE/'strict_source_qualification.json')
 if (q['schema']!='rinkfit.R_inner_curvature_strict_source_gate.v1'or q['raw_rows']!=40 or q['qualified_source_rows']!=33 or len(q['actual_source_proofs'])!=33 or len(q['pending_rows'])!=7 or q['old_R19_scope_changed']or q['noise_precision_issued']or not q['source_and_query_and_station_domain_frozen']or q['exact_frozen_RGB_replay_max_px']!=0):raise ValueError('Incomplete frozen R33 source qualification')
 if replay and source_module('qualify').qualify(negative_tests=True)!=q:raise ValueError('R33 exact current RGB replay differs from frozen qualification')
 for x in q['actual_source_proofs']:
  checked(x['source_packet'])
  if not x['source_row_id'].startswith(PREFIX)or x['native_pts']not in (2206464,2321408)or x['material_index']!=73 or x['path_index']!=0 or x['station_intervals']!=[[.25,.75]]or x['point_correspondence']or x['entity_station_fixed']or not x['source_qualified']or not x['unknown_noise']or x['query']['alpha']!=[-3,3]:raise ValueError('R33 source scope/unknown station changed')
 return q

def native_query(p,i):
 q=p['query'];uv=np.asarray(q['uv'],float);n=np.asarray(q['n'],float);a=q['alpha']
 return dict(row_index=int(i),source_factor_row_id=p['source_row_id'],alpha=[float(v)for v in a],native_uv=[(uv+float(x)*n).tolist()for x in a],normal=n.tolist())

def _norm_groups(groups,base,seen):
 out=[]
 for g in groups:
  z=copy.deepcopy(g)
  if isinstance(z['W'],dict):z['W']['file']=str(checked(z['W'],base,seen))
  out.append(z)
 return out

def assert_r_inner_prefix(packet,packet_path,*,parent_document=None,certificate=None,verified_dependency_cache=None,replay_source=False):
 path=Path(packet_path).resolve();seen=set()if verified_dependency_cache is None else verified_dependency_cache
 cp,cert=(checked(packet['board_material_source_qualification'],path.parent,seen),None)if certificate is None else certificate
 cert=read_json(cp)if cert is None else cert
 if cert.get('schema')!=SCHEMA or cert.get('parent_recursion_contract')!=CONTRACT or cert.get('issuer_module_ref')!=ref(__file__)or cert.get('recursive_parent_guard_module_ref')!=ref(parent_guard.__file__):raise ValueError('Different R33 prefix issuer or explicit V9 parent guard')
 for k in ('issuer_module_ref','recursive_parent_guard_module_ref','source_reader_module_ref','strict_source_qualification','actual_append_source_bundle','parent_model'):checked(cert[k],cp.parent,seen)
 if cert['source_reader_module_ref']!=ref(SOURCE/'source_operator.py')or cert['strict_source_qualification']!=ref(SOURCE/'strict_source_qualification.json'):raise ValueError('R source authority redirected')
 pp=checked(cert['parent_packet'],cp.parent,seen);old=parent_document if parent_document is not None else read_json(pp);parent_guard.assert_board_source_qualification_current(old,pp,verified_dependency_cache=seen)
 if old.get('accepted_explicit_state_binding',{}).get('schema')!='rinkfit.accepted_operator_transition_state_packet_binding.v1':raise ValueError('R append needs an explicit complete-state-bound V9 parent')
 from .joint_solver import _source_membership_fingerprint
 fp=_source_membership_fingerprint(packet)
 for dependency in cert['runtime_inherited_source_recall_dependencies']:
  checked(dependency,cp.parent,seen)
  if not any(x.get('sha256')==dependency['sha256']and checked(x,path.parent,seen)==checked(dependency,cp.parent,seen)and x.get('source_membership_fingerprint')==fp for x in packet.get('source_identity_watchlist',[])):raise ValueError('Inherited source recall dependency missing from current runtime guard')
 if cert['parent_packet']not in cert['runtime_inherited_source_recall_dependencies']or old['board_material_source_qualification']not in cert['runtime_inherited_source_recall_dependencies']:raise ValueError('R append dropped parent recall authority')
 q=source_proofs(replay=replay_source);proofs=q['actual_source_proofs'];N=int(old['row_count']);M=int(packet['row_count'])
 if (M!=N+33 or cert['original_raw_rows']!=N or cert['raw_rows']!=M or cert['appended_raw_rows']!=33 or packet['material_binding']!=old['material_binding']or packet['shot_ids']!=old['shot_ids']):raise ValueError('R source append raw scope/material/clock differs')
 for k in ('source_row_ids','source_dataset_ids','source_family_ids'):
  if len(packet[k])!=M or packet[k][:N]!=old[k]:raise ValueError('R source changed inherited '+k)
 for k in ('independent_rows','independent_statistic_kind','joint_statistics_rank','stroke_pairs'):
  if packet.get(k)!=old.get(k):raise ValueError('R source changed inherited statistic/pair scope '+k)
 if set(packet['arrays'])!=set(old['arrays']):raise ValueError('R source array ABI changed')
 A={}
 for k in old['arrays']:
  a=array(old,k,pp.parent,seen);b=array(packet,k,path.parent,seen)
  if b.shape!=(M,*a.shape[1:])or b.dtype!=a.dtype or b[:N].tobytes()!=a.tobytes():raise ValueError('R source changed inherited array '+k)
  A[k]=b
 if packet['queries'][:len(old['queries'])]!=old['queries']or len(packet['queries'])!=len(old['queries'])+33:raise ValueError('R source changed old Query scope')
 od=old['source_station_domains'];nd=packet['source_station_domains']
 if nd['rows'][:len(od['rows'])]!=od['rows']or len(nd['rows'])!=len(od['rows'])+33 or any(nd.get(k)!=v for k,v in od.items()if k not in ('rows','source_refs')):raise ValueError('R source changed old R19 finite station union')
 oldR=[x for x in od['rows']if int(A['material_index'][int(x['row_index'])])==73]
 if len(oldR)!=19 or cert['old_R19_source_domain_sha256']!=_digest(oldR):raise ValueError('Old R19 domain identity differs')
 groups=_norm_groups(packet['groups'],path.parent,seen);oldgroups=_norm_groups(old['groups'],pp.parent,seen)
 if groups[:len(oldgroups)]!=oldgroups:raise ValueError('R source changed inherited W/groups')
 members=[]
 for g in groups[len(oldgroups):]:
  if g['statistic_kind']!='constraint_only_unknown_noise'or g['rank']!=0 or g['W']!=[]or g.get('unknown_noise_source_has_statistical_precision',True):raise ValueError('Unissued R source statistical precision')
  members+=g['rows']
 if sorted(members)!=list(range(N,M)):raise ValueError('New R source constraints must partition each of 33 once')
 for j,p in enumerate(proofs):
  i=N+j
  if (packet['source_row_ids'][i]!=p['source_row_id']or int(A['native_pts'][i])!=p['native_pts']or not np.array_equal(A['uv'][i],p['current_uv'])or not np.array_equal(A['query_uv'][i],p['query']['uv'])or not np.array_equal(A['profile_normal'][i],p['normal'])or float(A['alpha'][i])!=p['measured_alpha']or not np.array_equal(A['alpha_bounds'][i],p['query']['alpha'])):raise ValueError('New R source UV/ID/PTS/normal/Query/alpha changed')
  if any(int(A[k][i])!=v for k,v in {'material_index':73,'path':0,'kind':0,'admission':1,'fit_role':1,'shot_index':8,'stroke_pair_index':-1,'stroke_source_side':-1}.items())or float(A['E'][i])!=2:raise ValueError('New R ordinary finite material/role differs')
  if packet['queries'][len(old['queries'])+j]!=native_query(p,i)or nd['rows'][len(od['rows'])+j]!=dict(row_index=i,path_index=0,allowed_station_intervals=[[.25,.75]],source_row_id=p['source_row_id']):raise ValueError('New R finite Query or unknown station union redirected')
 b=packet.get('accepted_explicit_state_binding',{});contract=read_json(checked(b['prefix_contract'],path.parent,seen))
 if (b.get('schema')!=BINDING_SCHEMA or contract.get('schema')!=STATE_SCHEMA or b['parent_packet']!=cert['parent_packet']or contract['parent_packet']!=cert['parent_packet']or contract['parent_accepted_binding']!=old['accepted_explicit_state_binding']or contract['current_model']!=cert['parent_model']or contract['new_source_qualification']!=packet['board_material_source_qualification']or b['preserved_raw_rows']!=old['accepted_explicit_state_binding']['preserved_raw_rows']or b['inherited_parent_raw_rows']!=N or b['fresh_appended_rows']!=M-b['preserved_raw_rows']or b['state_token']!=old['accepted_explicit_state_binding']['state_token']):raise ValueError('R append complete-state inheritance contract differs')
 return dict(passed=True,source_proofs=33,old_R19_scope_exact=True,parent_all_arrays_W_groups_and_queries_exact=True,preserved_complete_CP_raw_rows=b['preserved_raw_rows'],fresh_rows=b['fresh_appended_rows'],formal_acceptance_claimed=False)

def install(parent_packet,model_path,output,*,prepared_packet=None):
 pp=Path(parent_packet).resolve();mp=Path(model_path).resolve();out=Path(output).resolve()
 if out.exists():raise FileExistsError('New immutable R prefix directory required')
 q=source_proofs(replay=True);parent=read_json(pp);parent_guard.assert_board_source_qualification_current(parent,pp);model=read_json(mp);pb=parent.get('accepted_explicit_state_binding',{})
 if pb.get('schema')!='rinkfit.accepted_operator_transition_state_packet_binding.v1'or pb['qualified_physical_suffix']['current_model']!=ref(mp):raise ValueError('Parent is not bound to this exact V9 complete-state model')
 if model['materials'][73]['label']!='Ad4.print.word_ANKARA_R_arc1':raise ValueError('Current finite R material identity differs')
 def finalize(packet,folder,old):
  N=old['row_count'];proofs=q['actual_source_proofs'];M=packet['row_count']
  if M!=N+33:raise ValueError('R prepared source must append exactly 33, preserving 7 journal-only pending')
  loaded={k:np.fromfile(packet['arrays'][k]['file'],dtype=packet['arrays'][k]['dtype']).reshape(packet['arrays'][k]['shape'])for k in ('query_uv','profile_normal','alpha','alpha_bounds')}
  domains=copy.deepcopy(old['source_station_domains']);queries=copy.deepcopy(old['queries'])
  for j,p in enumerate(proofs):
   i=N+j;loaded['query_uv'][i]=p['query']['uv'];loaded['profile_normal'][i]=p['normal'];loaded['alpha'][i]=p['measured_alpha'];loaded['alpha_bounds'][i]=p['query']['alpha'];queries.append(native_query(p,i));domains['rows'].append(dict(row_index=i,path_index=0,allowed_station_intervals=[[.25,.75]],source_row_id=p['source_row_id']))
  domains['source_refs'].append(ref(SOURCE/'strict_source_qualification.json'))
  for k,a in loaded.items():
   if prepared_packet is not None:
    before=np.fromfile(packet['arrays'][k]['file'],dtype=packet['arrays'][k]['dtype']).reshape(packet['arrays'][k]['shape'])
    if not np.array_equal(a,before):raise ValueError('Prepared append has different source Query seed '+k)
   else:a.tofile(packet['arrays'][k]['file'])
   packet['arrays'][k]['sha256']=file_sha256(packet['arrays'][k]['file'])
  packet.update(queries=queries,source_station_domains=domains,recapture_rows=list(range(pb['preserved_raw_rows'],M)),all_new_source_nuisance_requires_fresh_physical_capture=True,source_placeholder_station_half_is_not_solver_ready=True)
  oldmat=array(old,'material_index',pp.parent);oldR=[x for x in old['source_station_domains']['rows']if int(oldmat[int(x['row_index'])])==73]
  cert=dict(schema=SCHEMA,parent_recursion_contract=CONTRACT,issuer_module_ref=ref(__file__),recursive_parent_guard_module_ref=ref(parent_guard.__file__),source_reader_module_ref=ref(SOURCE/'source_operator.py'),strict_source_qualification=ref(SOURCE/'strict_source_qualification.json'),source_qualification_issuer=ref(SOURCE/'qualify.py'),source_manifest=ref(SOURCE/'generation/manifest.json'),actual_append_source_bundle=ref(SOURCE/'prepared_bundle.json'),parent_packet=ref(pp),parent_model=ref(mp),original_raw_rows=N,raw_rows=M,appended_raw_rows=33,inherited_material_count=len(model['materials']),inherited_physical_block_count=len(model['blocks']),old_R19_source_domain_sha256=_digest(oldR),old_R19_scope_changed=False,source_station_unknown=True,noise_precision_invented=False)
  from .joint_solver import _source_membership_fingerprint
  inherited_refs=[ref(pp),old['board_material_source_qualification']]
  parent_fp=_source_membership_fingerprint(old)
  inherited_refs += [dict(file=str(checked(x,pp.parent)),sha256=x['sha256'])for x in old.get('source_identity_watchlist',[])if x.get('source_membership_fingerprint')==parent_fp]
  cert['runtime_inherited_source_recall_dependencies']=inherited_refs
  child_fp=_source_membership_fingerprint(packet)
  packet['source_identity_watchlist']=[dict(x,source_membership_fingerprint=child_fp,dependency_scope='inherited_V9_source_prefix_of_R_append')for x in inherited_refs]
  cp=folder/'qualified_certificate.json';write_json(cp,cert);packet['board_material_source_qualification']=ref(cp)
  state=dict(schema=STATE_SCHEMA,parent_packet=ref(pp),parent_accepted_binding=pb,current_model=ref(mp),new_source_qualification=ref(cp),inherited_parent_raw_rows=N,new_raw_rows=M,preserved_complete_CP_raw_rows=pb['preserved_raw_rows'],inherited_fresh_rows=N-pb['preserved_raw_rows'],new_R_fresh_rows=33,old_physical_values_exact=True,all_parent_four_auxiliary_values_exact=True,old_AL_duals_resumed=False,formal_acceptance_claimed=False)
  sf=folder/'inherited_state_contract.json';write_json(sf,state);packet['accepted_explicit_state_binding']=dict(schema=BINDING_SCHEMA,state_kind='engineering_candidate',parent_packet=ref(pp),prefix_contract=ref(sf),current_model=ref(mp),preserved_raw_rows=pb['preserved_raw_rows'],inherited_parent_raw_rows=N,fresh_appended_rows=M-pb['preserved_raw_rows'],state_token=pb['state_token'],new_appended_rows_must_be_initialized=True,formal_acceptance_claimed=False,native_replay_pass_claimed=False,old_AL_duals_resumed=False,source_UV_W_E_unchanged=True)
  assert_r_inner_prefix(packet,folder/'packet.json',parent_document=old,certificate=(cp,cert));return packet
 roles=SOURCE.parents[3]/'okada_joint9_1018a/roles/frame_roles.json'
 # Roles are the same frozen programme-owned exposure ledger used by V9.
 roles=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_joint9_1018a/roles/frame_roles.json')
 if prepared_packet is None:
  report=append_constraint_source_chunks(pp,[ref(SOURCE/'prepared_bundle.json')],mp,SOURCE/'generation/material_map.json',roles,out,finalize_packet_with_parent=finalize)
 else:
  source_path=Path(prepared_packet).resolve();packet=read_json(source_path)
  if packet['parent_all_source_constraint_packet']!=ref(pp)or packet['row_count']!=parent['row_count']+33:raise ValueError('Prepared source append belongs to a different parent/generation')
  out.mkdir(parents=True);packet=finalize(packet,out,parent)
  from .packet_storage import write_packet
  write_packet(out/'packet.json',packet);report=read_json(source_path.parent/'report.json');report.update(packet=ref(out/'packet.json'),array_generation_reused_without_copy=ref(source_path),source_arrays_rewritten=False)
  write_json(out/'report.json',report)
 write_json(out/'strict_prefix_check.json',dict(schema='rinkfit.R33_actual_bound_V9_append_check.v1',passed=True,report=report,Solve_started=False,quality_acceptance_claimed=False));return report
