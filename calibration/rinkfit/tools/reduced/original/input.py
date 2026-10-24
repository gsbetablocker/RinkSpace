"""Typed fresh prepared+adaptive-curvature+trust -> joint full-S Phase-I input.
Only original dataset/J/physical box plus frozen coupled A assembly; no Solve.
No legacy frame/model/station seed or historical fixed-S Vguard is inherited.
"""
from pathlib import Path
import argparse,sys,json,hashlib,importlib.util,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src'))
from scipy import sparse

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def verify(d,base=None):
 p=Path(d.get('path',d.get('file')));p=p if p.is_absolute()else Path(base)/p
 if sha(p)!=d['sha256']:raise ValueError('Changed immutable descriptor '+str(p))
 return p

def load(name,path):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m

def run(prepared,curvature,trust,output,operator_spec):
 OP=Path(operator_spec).resolve();op=read(OP)
 if op.get('schema')!='rinkfit.joint_full_lawful_S_frozen_operator_spec.v1':raise ValueError('Frozen generic operator spec')
 paths={k:verify(op[k])for k in ['strict_norm_source','full_norm_source','profile_domain_source','coupled_A_source','group_W_source','secondary_source','station_TR_limiter_source','print_det_guard_source','group_W_recipe']}
 strict=load('fixed_entity_norm_constraints',paths['strict_norm_source']);load('source_affine_norm_constraints',paths['full_norm_source']);load('profile_domain_constraints',paths['profile_domain_source']);core=load('_joint_fullS_coupled_A',paths['coupled_A_source']);array=strict.array
 D=Path(output).resolve();D.mkdir(exist_ok=False);P=Path(prepared).resolve();pr=read(P);CP=verify(pr['checkpoint']);cp=read(CP);C=Path(curvature).resolve();cur=read(C);rad=array(cur['arrays']['radius'],C.parent)
 for key in ['checkpoint','transport']:
  if cur[key]['sha256']!=pr[key]['sha256']:raise ValueError('Mixed prepared/curvature '+key)
  verify(cur[key])
 if cur['complete_auxiliary_sha256']!={k:v['sha256']for k,v in cp['row_state_files'].items()}:raise ValueError('Mixed full raw-N nuisance arrays')
 c=core.FiniteVisiblePrimary(verify(pr['linearization']),verify(pr['transport']),verify(pr['profile_domains']),float(trust),station_TR=rad)
 for key in ['model','packet','semantic']:
  if cur[key]['sha256']!=c.tr[key]['sha256']:raise ValueError('Mixed physical/source curvature '+key)
  verify(cur[key])
 sparse.save_npz(D/'canonical_original_A.npz',c.A);np.savez_compressed(D/'rhs_bounds.npz',rhs=c.rhs,lower=c.lower,upper=c.upper)
 np.savez_compressed(D/'coupled_reachability_proof.npz',all_rhs=c.all_reachrhs,kept_indices=c.reach_keep,dropped_indices=c.reach_removed,upper_enclosure=c.reach_upper_enclosure,sensor_source_rows=c.sensorrows,original_lower_S=c.Ls,original_upper_S=c.Us,physical_lower=c.lower[:-1],physical_upper=c.upper[:-1])
 spec={k:op[k]for k in ['group_W_recipe','group_W_source','full_norm_source','secondary_source','station_TR_limiter_source','print_det_guard_source','operator_bundle']};spec.update(schema='rinkfit.joint_full_lawful_S_PhaseI_input.v1',linearization=pr['linearization'],transport=pr['transport'],profile_domains=pr['profile_domains'],original_A=ref(D/'canonical_original_A.npz'),rhs_bounds=ref(D/'rhs_bounds.npz'),station_TR_manifest=ref(C),station_TR_array_key='radius',trust_scaled=float(trust),prepared=ref(P),source=ref(__file__),operator_spec=ref(OP),primary_epigraph_coordinate='squared_budget_b',coupled_A_scope='Current coupled finite-S reachability and original verified qbar/physical scene/exact profile domains; no historical fixed-S temporary Vguards',coupled_A_source=op['coupled_A_source'],profile_domain_source=op['profile_domain_source'])
 (D/'input.json').write_text(json.dumps(spec,indent=2),encoding='utf8');context=dict(schema='rinkfit.generic_current_full_S_joint_primary_coupled_A_context.v1',prepared=ref(P),checkpoint=pr['checkpoint'],linearization=pr['linearization'],transport=pr['transport'],profile_domains=pr['profile_domains'],curvature=ref(C),source=ref(__file__),operator_spec=ref(OP),actual_coupled_A_sources=[op[k]for k in ['coupled_A_source','profile_domain_source','full_norm_source','strict_norm_source']],canonical_original_A=ref(D/'canonical_original_A.npz'),rhs_bounds=ref(D/'rhs_bounds.npz'),coupled_reachability_proof=ref(D/'coupled_reachability_proof.npz'),original_preserved_counts=c.preserved_counts,shape=c.A.shape,A_nnz=c.A.nnz,coupled_released_S_count=len(c.released),reachability_rows_before_redundancy=c.all_reachA.shape[0],kept_reachability_rows=len(c.reach_keep),omitted_box_proven_redundant=len(c.reach_removed),input=ref(D/'input.json'),no_Solve=True,all_original_source_W_E_unchanged=True,no_legacy_numeric_template_dependency=True)
 (D/'context.json').write_text(json.dumps(context,indent=2),encoding='utf8');print(json.dumps(dict(input=ref(D/'input.json'),context=ref(D/'context.json'),shape=c.A.shape,kept_sensor=len(c.reach_keep),omitted=len(c.reach_removed))),flush=True);return context

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepared',required=True);p.add_argument('--station-trust-manifest',required=True);p.add_argument('--trust',required=True,type=float);p.add_argument('--output',required=True);p.add_argument('--operator-spec',required=True);a=p.parse_args();run(a.prepared,a.station_trust_manifest,a.trust,a.output,a.operator_spec)
