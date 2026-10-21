"""Versioned complete enriched primal import; source issuers remain immutable."""
from pathlib import Path
import copy,json
import numpy as np
from .io_utils import file_sha256
from .r_inner_prefix_v1 import checked,array
from .accepted_aux_prefix import _digest,_prefix_hash,_runtime_query_digest,_runtime_group_digest,AUX,OBJECTIVE
from .enriched_checkpoint_v1 import verify_source_enriched_checkpoint
from .source_math_identity_v1 import require
SCHEMA='rinkfit.accepted_source_enriched_state_packet_binding.v1'
CONTRACT='rinkfit.source_enriched_complete_primal_prefix_contract.v1'
def consumer_refs():
 root=Path(__file__).resolve().parent;headers=root.parents[1]/'cpp/include/rinkfit'
 return [ref(root/n)for n in ('source_enriched_current_v1.py','source_math_identity_v1.py','enriched_checkpoint_v1.py','enriched_aux_prefix_v1.py')]+[ref(headers/n)for n in ('joint_source_enriched_guard_v1.hpp','joint_source_math_identity_v1.hpp','joint_source_enriched_guard_v2.hpp','joint_enriched_checkpoint_v1.hpp','joint_enriched_aux_prefix.hpp','joint_aux_prefix_current_v5.hpp')]
def ref(p):return dict(file=str(Path(p).resolve()),sha256=file_sha256(p))
def without_file(v):
 z=copy.deepcopy(v)
 if isinstance(z,dict):z.pop('file',None);z.pop('path',None)
 return z

def assert_source_prefix(old,current,oldbase,newbase,seen):
 n=old['row_count'];m=current['row_count'];require(m>=n,'enriched prefix loses raw rows')
 require(old['schema']==current['schema']and old['shot_ids']==current['shot_ids']and set(old['arrays'])==set(current['arrays']),'enriched prefix source array/clock ABI')
 for k in old['arrays']:
  if k in AUX:continue
  a=array(old,k,oldbase,seen);b=array(current,k,newbase,seen)
  require(a.dtype==b.dtype and a.shape[0]==n and b.shape[0]==m and a.shape[1:]==b.shape[1:]and _prefix_hash(a)==_prefix_hash(b[:n]),'all inherited source arrays '+k)
 for k in ('source_row_ids','source_dataset_ids','source_family_ids'):
  require(len(old[k])==n and len(current[k])==m and current[k][:n]==old[k],'inherited source identity '+k)
 for k in ('queries','stroke_pairs'):
  a=old.get(k,[]);b=current.get(k,[])
  if k=='queries':b=[v for v in b if v['row_index']<n]
  else:b=[v for v in b if all(i<n for i in v['rows'])]
  require(a==b,'inherited raw source ledger '+k)
 a=old.get('source_station_domains',{});b=current.get('source_station_domains',{});aa=copy.deepcopy(a);bb=copy.deepcopy(b)
 if 'rows'in bb:bb['rows']=[v for v in bb['rows']if v['row_index']<n]
 require(aa==bb,'inherited finite material source station domains')
 require(old.get('independent_rows',[])==[i for i in current.get('independent_rows',[])if i<n]and old.get('independent_statistic_kind')==current.get('independent_statistic_kind'),'inherited independent original W membership')
 a=old['groups'];b=current['groups'];require(len(b)>=len(a),'old W/groups missing')
 for x,y in zip(a,b):
  x=copy.deepcopy(x);y=copy.deepcopy(y)
  for z,base in ((x,oldbase),(y,newbase)):
   if isinstance(z['W'],dict):checked(z['W'],base,seen);z['W']=without_file(z['W'])
  require(x==y,'old exact W/group partition changed')
 partition=set()
 for g in b[len(a):]:
  require(g['statistic_kind']=='constraint_only_unknown_noise'and g['rank']==0 and g['W']==[],'new source rows acquired unqualified precision')
  for i in g['rows']:require(n<=i<m and i not in partition,'new constraint source group partition');partition.add(i)
 require(partition==set(range(n,m)),'new appended source group coverage')
 return n,m

def typed_source_queries(packet,bindings,base,seen):
 qs=copy.deepcopy(packet['queries']);domains={}
 for b in bindings:
  if b['kind']=='source_query_branches':
   reg=json.loads(checked(b['binding']['registry'],base,seen).read_text(encoding='utf8'));domains={d['row_index']:d['allowed_alpha']for d in reg['alpha_domains']}
 def point(q,a):
  if q.get('denominator'):
   z=(a-float(q.get('alpha_origin',0)))/float(q.get('alpha_scale',1))
   def poly(c):
    y=0.
    for x in reversed(c):y=y*z+float(x)
    return y
   d=poly(q['denominator']);require(abs(d)>=1e-14,'source rational Query denominator');return[poly(q['numerator_x'])/d,poly(q['numerator_y'])/d]
  knots=q['alpha'];k=next((i-1 for i,v in enumerate(knots)if v>a),len(knots)-2);u=(a-float(knots[k]))/(float(knots[k+1])-float(knots[k]));return[float(q['native_uv'][k][j])*(1.-u)+float(q['native_uv'][k+1][j])*u for j in range(2)]
 for q in qs:
  if q['row_index']in domains:
   lo,hi=domains[q['row_index']];knots=[float(lo)]+[float(v)for v in q['alpha']if lo<v<hi]+[float(hi)];uv=[point(q,a)for a in knots];q['alpha']=knots;q['native_uv']=uv
 return qs

def _metadata(path,kind,configuration_path,*,state_dir,source_packet):
 meta=json.loads(Path(path).read_text(encoding='utf8'));cfg=ref(configuration_path)
 config=json.loads(Path(configuration_path).read_text(encoding='utf8'));ordered=[]
 if 'visible_ice_source_constraints'in config:ordered.append(dict(kind='visible_ice_source_constraints',consumer_config=config['visible_ice_source_constraints']))
 if 'source_query_branch_registry'in config:ordered.append(dict(kind='source_query_branches',binding=config['source_query_branch_registry']))
 require(ordered==meta['source_math_bindings'],'actual config/source mathematics attachments differ')
 require(meta.get('objective_id')==OBJECTIVE,'actual engineering objective metadata')
 if kind=='engineering_candidate':
  require(meta.get('schema')=='rinkfit.joint_engineering_recovery_run.v1'and not any(meta.get(k,True)for k in ('source_recalled','formal_Phase_II_KKT_passed','common_worst_budget_consumed','original_W_likelihood_consumed'))and not meta.get('callback_error',''),'production engineering receipt flags')
 elif kind=='native_initialization_snapshot':
  require(meta.get('schema')=='rinkfit.recovery_initialization_preflight.v1'and meta.get('solver_started')is False,'genuine native initialization preflight receipt')
  state_dir=Path(state_dir).resolve();sm=json.loads((state_dir/'source_math.json').read_text(encoding='utf8'))
  require(sm['mathematical_source_id']==meta['mathematical_source_id']and sm['source_math_bindings']==meta['source_math_bindings']and Path(meta['initial_checkpoint']).resolve()==state_dir/'engineering_checkpoint.json','initial snapshot source mathematics/CP references')
  meta=dict(meta,candidate_state_token=sm['state_token'],model_sha256=file_sha256(state_dir/'model.json'),source_packet=str(Path(source_packet).resolve()))
 else:raise ValueError('Unknown genuine enriched state metadata kind')
 require(file_sha256(source_packet)==meta['source_packet_sha256'],'actual metadata raw source hash')
 return meta,cfg

def objective_config_identity(configuration_path,packet,base,seen):
 from .source_math_identity_v1 import compact,digest_text
 cfg=json.loads(Path(configuration_path).read_text(encoding='utf8'));active=(array(packet,'admission',base,seen)==1)&(array(packet,'fit_role',base,seen)==1)
 objective=dict(objective_id=OBJECTIVE,physical_constraint_penalty=float(cfg.get('physical_constraint_penalty',10.)),pixel_total_weight=float(cfg.get('pixel_total_weight',int(np.count_nonzero(active)))))
 for key in ('visible_ice_source_constraints','source_query_branch_registry'):
  if key in cfg:objective[key]=cfg[key]
 return digest_text(compact(objective))

def rebind_source_enriched_state(packet_path,state_dir,output_path,*,metadata_file,metadata_kind,configuration_path,accepted_source_packet=None):
 src=Path(packet_path).resolve();state=Path(state_dir).resolve();out=Path(output_path).resolve();seen=set();require(not out.exists(),'immutable enriched output exists')
 canonical=json.loads(src.read_text(encoding='utf8'))
 # Keep authority direct; an already numerical-wrapped canonical source is unwrapped.
 authority=canonical.get('source_math_source_authority');canonical_path=checked(authority['canonical_source_packet'],src.parent,seen)if authority else src
 current=json.loads(canonical_path.read_text(encoding='utf8'))
 original_meta=json.loads(Path(metadata_file).read_text(encoding='utf8'));rawpath=Path(accepted_source_packet).resolve()if accepted_source_packet is not None else(Path(original_meta['source_packet']).resolve()if 'source_packet'in original_meta else src)
 require(file_sha256(rawpath)==original_meta['source_packet_sha256'],'accepted actual raw source packet')
 meta,cfg=_metadata(metadata_file,metadata_kind,configuration_path,state_dir=state,source_packet=rawpath);require(file_sha256(rawpath)==meta['source_packet_sha256'],'accepted raw source packet changed');old=json.loads(rawpath.read_text(encoding='utf8'))
 require(objective_config_identity(configuration_path,old,rawpath.parent,seen)==meta['configuration_sha256'],'independently recomputed objective configuration digest')
 modelpath=state/'model.json';require(file_sha256(modelpath)==meta['model_sha256'],'actual complete primal model file changed');model=json.loads(modelpath.read_text(encoding='utf8'))
 manifestpath=state/'explicit_nuisance/manifest.json';manifest=json.loads(manifestpath.read_text(encoding='utf8'));n,m=assert_source_prefix(old,current,rawpath.parent,canonical_path.parent,seen)
 require(manifest['packet_sha256']==meta['source_packet_sha256']and manifest['raw_rows']==n,'explicit all-raw manifest generation')
 aux={};descriptors={}
 for k,(name,dtype)in AUX.items():
  p=state/'explicit_nuisance'/name;require(file_sha256(p)==manifest['files'][name]['sha256']and p.stat().st_size==n*np.dtype(dtype).itemsize,'complete explicit array '+k);aux[k]=np.memmap(p,mode='r',dtype=dtype,shape=(n,));require(np.isfinite(aux[k]).all(),'nonfinite complete explicit '+k);descriptors[k]=dict(ref(p),dtype=dtype,shape=[n])
 active=(array(old,'admission',rawpath.parent,seen)==1)&(array(old,'fit_role',rawpath.parent,seen)==1)
 kinds=array(old,'kind',rawpath.parent,seen);queries=[v['row_index']for v in old['queries']]
 proof=verify_source_enriched_checkpoint(state/'engineering_checkpoint.json',meta,model,old,aux,active,queries,[int(i)for i in np.flatnonzero(active)if kinds[i]!=1],base=rawpath.parent,verified_dependency_cache=seen)
 # Source projection roles are fixed, candidate values alone can have evolved.
 canonical_model_path=checked(authority['canonical_source_model'],src.parent,seen)if authority else checked(current['accepted_explicit_state_binding']['current_model'],canonical_path.parent,seen)
 cm=json.loads(canonical_model_path.read_text(encoding='utf8'));am=copy.deepcopy(model)
 for z in (cm,am):
  for b in z['blocks']:b.pop('values',None)
 require(cm==am,'enriched candidate changed canonical physical projection metadata')
 out.parent.mkdir(parents=True,exist_ok=True);out.mkdir();doc=copy.deepcopy(current);aux_contract={}
 for k,(name,dtype)in AUX.items():
  if m==n:descriptor=descriptors[k]
  else:
   p=out/(k+('.u32'if k=='path'else'.f64'));v=np.memmap(p,mode='w+',dtype=dtype,shape=(m,));v[:n]=aux[k];v[n:]=array(current,k,canonical_path.parent,seen)[n:];v.flush();del v;descriptor=dict(ref(p),dtype=dtype,shape=[m])
  doc['arrays'][k]=descriptor;aux_contract[k]=dict(candidate_file=ref(state/'explicit_nuisance'/name),current_descriptor=descriptor,prefix_sha256=_prefix_hash(aux[k]))
 runtime_group_hash,runtime_group_count=_runtime_group_digest(old,rawpath.parent,seen)
 contract=dict(schema=CONTRACT,consumer_module_refs=consumer_refs(),canonical_source_packet=ref(canonical_path),accepted_raw_source_packet=ref(rawpath),accepted_model=ref(modelpath),actual_metadata=ref(metadata_file),metadata_kind=metadata_kind,configuration=cfg,snapshot_source_math=ref(state/'source_math.json')if metadata_kind=='native_initialization_snapshot'else None,complete_engineering_checkpoint=ref(state/'engineering_checkpoint.json'),explicit_nuisance_manifest=ref(manifestpath),source_math_bindings=meta['source_math_bindings'],mathematical_source_id=meta['mathematical_source_id'],configuration_sha256=meta['configuration_sha256'],state_token=meta['candidate_state_token'],preserved_raw_rows=n,current_raw_rows=m,source_arrays={k:dict(accepted_descriptor=old['arrays'][k],current_descriptor=doc['arrays'][k],prefix_sha256=_prefix_hash(array(old,k,rawpath.parent,seen)))for k in old['arrays']if k not in AUX},auxiliaries=aux_contract,runtime_raw_query_prefix_sha256=_digest(old['queries']),runtime_query_prefix_sha256=_runtime_query_digest(typed_source_queries(old,meta['source_math_bindings'],rawpath.parent,seen)),runtime_old_group_count=runtime_group_count,runtime_group_prefix_sha256=runtime_group_hash,old_AL_duals_resumed=False,formal_acceptance_claimed=False)
 (out/'prefix_contract.json').write_text(json.dumps(contract,indent=2),encoding='utf8')
 doc['accepted_explicit_state_binding']=dict(schema=SCHEMA,state_kind=metadata_kind,prefix_contract=ref(out/'prefix_contract.json'),state_token=meta['candidate_state_token'],configuration_sha256=meta['configuration_sha256'],preserved_raw_rows=n,fresh_appended_rows=m-n,old_AL_duals_resumed=False,formal_acceptance_claimed=False)
 from .source_enriched_current_v1 import SCHEMA as AUTH,KEY
 import rinkfit_joint.source_enriched_current_v1 as dispatcher
 doc[KEY]=dict(schema=AUTH,issuer_module_ref=ref(dispatcher.__file__),canonical_source_packet=ref(canonical_path),canonical_source_model=ref(canonical_model_path),numerical_state_kind='source_enriched_complete_primal',source_math_bindings=meta['source_math_bindings'])
 branches=[b['binding']for b in meta['source_math_bindings']if b['kind']=='source_query_branches']
 if branches:doc['source_query_branch_registry']=branches[0]
 dest=out/'packet.json';dest.write_text(json.dumps(doc,indent=2),encoding='utf8')
 (out/'bind_receipt.json').write_text(json.dumps(dict(schema='rinkfit.source_enriched_complete_primal_rebind_receipt.v1',packet=ref(dest),model=ref(modelpath),checkpoint_verification=proof,preserved_raw_rows=n,fresh_rows=m-n,old_CP_modified=False,old_AL_duals_resumed=False,source_qualification_issued=False,formal_acceptance_claimed=False),indent=2),encoding='utf8')
 assert_enriched_binding(json.loads(dest.read_text(encoding='utf8')),dest,verified_dependency_cache=seen)
 return dict(packet=ref(dest),model=ref(modelpath),receipt=ref(out/'bind_receipt.json'))

def assert_enriched_binding(packet,path,*,verified_dependency_cache=None):
 path=Path(path);seen=set()if verified_dependency_cache is None else verified_dependency_cache;b=packet['accepted_explicit_state_binding'];require(b['schema']==SCHEMA and not b['old_AL_duals_resumed']and not b['formal_acceptance_claimed'],'enriched binding schema/claims');contractpath=checked(b['prefix_contract'],path.parent,seen);c=json.loads(contractpath.read_text(encoding='utf8'));require(c['schema']==CONTRACT and not c['old_AL_duals_resumed']and not c['formal_acceptance_claimed'],'enriched contract schema/claims')
 require(c['consumer_module_refs']==consumer_refs(),'enriched implementation/source consumer dependencies changed')
 for x in c['consumer_module_refs']:checked(x,contractpath.parent,seen)
 rawpath=checked(c['accepted_raw_source_packet'],contractpath.parent,seen);old=json.loads(rawpath.read_text(encoding='utf8'));n,m=assert_source_prefix(old,packet,rawpath.parent,path.parent,seen);require(n==c['preserved_raw_rows']==b['preserved_raw_rows']and m==c['current_raw_rows']and m-n==b['fresh_appended_rows'],'enriched row denominator')
 modelpath=checked(c['accepted_model'],contractpath.parent,seen)
 if c['metadata_kind']=='native_initialization_snapshot':checked(c['snapshot_source_math'],contractpath.parent,seen)
 configpath=checked(c['configuration'],contractpath.parent,seen)
 meta,cfg=_metadata(checked(c['actual_metadata'],contractpath.parent,seen),c['metadata_kind'],configpath,state_dir=modelpath.parent,source_packet=rawpath);
 require(objective_config_identity(configpath,old,rawpath.parent,seen)==meta['configuration_sha256'],'recomputed actual objective configuration')
 require(cfg==c['configuration']and meta['source_packet_sha256']==c['accepted_raw_source_packet']['sha256']and meta['candidate_state_token']==c['state_token']==b['state_token']and meta['configuration_sha256']==c['configuration_sha256']==b['configuration_sha256']and meta['source_math_bindings']==c['source_math_bindings']==packet['source_math_source_authority']['source_math_bindings']and meta['mathematical_source_id']==c['mathematical_source_id'],'actual metadata binding')
 modelpath=checked(c['accepted_model'],contractpath.parent,seen);require(file_sha256(modelpath)==meta['model_sha256'],'enriched full primal model receipt');model=json.loads(modelpath.read_text(encoding='utf8'));manifest=json.loads(checked(c['explicit_nuisance_manifest'],contractpath.parent,seen).read_text(encoding='utf8'));require(manifest['raw_rows']==n and manifest['packet_sha256']==meta['source_packet_sha256'],'enriched manifest generation');aux={}
 for k,(name,dtype)in AUX.items():
  a=c['auxiliaries'][k];p=checked(a['candidate_file'],contractpath.parent,seen);require(p.stat().st_size==n*np.dtype(dtype).itemsize and a['candidate_file']['sha256']==manifest['files'][name]['sha256']and packet['arrays'][k]==a['current_descriptor'],'enriched explicit auxiliary binding '+k);aux[k]=np.memmap(p,mode='r',dtype=dtype,shape=(n,));require(_prefix_hash(array(packet,k,path.parent,seen)[:n])==a['prefix_sha256']==_prefix_hash(aux[k]),'enriched all-four auxiliary prefix '+k)
 active=(array(old,'admission',rawpath.parent,seen)==1)&(array(old,'fit_role',rawpath.parent,seen)==1);kinds=array(old,'kind',rawpath.parent,seen)
 return verify_source_enriched_checkpoint(checked(c['complete_engineering_checkpoint'],contractpath.parent,seen),meta,model,old,aux,active,[v['row_index']for v in old['queries']],[int(i)for i in np.flatnonzero(active)if kinds[i]!=1],base=rawpath.parent,verified_dependency_cache=seen)
