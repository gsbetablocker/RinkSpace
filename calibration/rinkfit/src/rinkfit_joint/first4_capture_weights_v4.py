"""Preserve the actual capture parent budgets and append a source-only channel.

These are engineering L2 budgets. Neither E nor this issuer is a physical
noise model. The original W/Query data are kept for the separate formal mode.
"""
from pathlib import Path
import argparse,collections,hashlib,json
import numpy as np
from .first4_capture_weights_v3 import read,ref,key_digest,load
from .io_utils import file_sha256
SCHEMA='rinkfit.explicit_engineering_source_weights.v4'

def checked(r):
 p=Path(r['file']);
 if not p.is_absolute() or file_sha256(p)!=r['sha256']:raise ValueError('changed weight/source authority '+str(p))
 return p

def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()

def issue(parent_packet,parent_pixel_ledger,parent_candidate,next_packet,output,*,authority_pixel_ledger,printed_qualification):
 pp,lp,cp,npk,out=map(Path,(parent_packet,parent_pixel_ledger,parent_candidate,next_packet,output));ap=Path(authority_pixel_ledger);sp=Path(printed_qualification)
 if out.exists():raise FileExistsError('immutable explicit source-weight generation exists')
 p,q,ledger,binding=read(pp),read(npk),read(lp),read(cp/'source_uv_capture_state_binding.json');a=load(p,pp.parent);b=load(q,npk.parent);n,N=p['row_count'],q['row_count']
 if binding['packet_sha256']!=file_sha256(pp) or binding['model_sha256']!=file_sha256(cp/'model.json'):raise ValueError('parent complete state/source differs')
 if binding['objective_id']!='rinkfit.sourceUV_trueRS_finite_capture.v4':raise ValueError('v4 complete physical parent required')
 if ledger['schema']!='rinkfit.engineering_pixel_contributions.v1' or ledger['active_rows']!=p['active_rows'] or ledger['original_W_likelihood_consumed']:raise ValueError('not actual engineering parent ledger')
 if p['shot_ids']!=q['shot_ids'] or p['shot_ids']!=['s01','s02','s03','s04'] or N<=n:raise ValueError('source scope/prefix changed')
 for k in ['source_row_ids','source_dataset_ids','source_family_ids']:
  if p[k]!=q[k][:n]:raise ValueError('inherited source identity changed '+k)
 source_fields=['native_pts','shot_index','kind','material_index','E','uv','profile_normal','admission','fit_role','stroke_pair_index','stroke_source_side','alpha_bounds','query_uv']
 for k in source_fields:
  if not exact(a[k],b[k][:n]):raise ValueError('inherited source UV/identity/E/role changed '+k)
 if p['queries']!=[x for x in q['queries']if int(x['row_index'])<n]:raise ValueError('inherited literal Queries changed')
 if p['groups']!=[x for x in q['groups']if all(int(i)<n for i in x['rows'])]:raise ValueError('inherited source W or full group changed')
 A=(a['admission']==1)&(a['fit_role']==1);B=(b['admission']==1)&(b['fit_role']==1)
 if not np.array_equal(A,B[:n]):raise ValueError('parent active partition changed')
 # This parent has actual explicit arrays; null legacy per_row_weight is never read.
 oldref=ledger.get('explicit_source_weights_v3',{}).get('binding')
 if not oldref:raise ValueError('actual parent explicit budget binding absent')
 oldmanifest=read(checked(oldref))
 if oldmanifest['schema']!='rinkfit.explicit_engineering_source_weights.v3' or oldmanifest['source_packet']!=ref(pp):raise ValueError('wrong explicit parent budget/source')
 ow=oldmanifest['files']['weights.f64'];oi=oldmanifest['files']['information_class.u8'];oa=oldmanifest['files']['active.u8']
 for r in [ow,oi,oa]:checked(r)
 parent_weights=np.fromfile(ow['file'],dtype='<f8');parent_classes=np.fromfile(oi['file'],dtype='u1');parent_active=np.fromfile(oa['file'],dtype='u1')
 if parent_weights.shape!=(n,) or parent_classes.shape!=(n,) or not np.array_equal(parent_active,A):raise ValueError('parent explicit binary denominator differs')
 reconstructed=np.zeros(n,dtype='<f8');assignment=np.full(n,0xffffffff,dtype='<u4');seen=set()
 for j,g in enumerate(ledger['groups']):
  rr=g['actual_row_indices'];ww=g['actual_per_row_weight']
  if g['per_row_weight'] is not None or g['uniform_group_formula_consumed'] or len(rr)!=g['rows'] or len(rr)!=len(ww):raise ValueError('parent ledger lacks actual source arrays')
  for i,w in zip(rr,ww):
   if not 0<=i<n or i in seen or not A[i] or not np.isfinite(w) or w<=0:raise ValueError('parent actual budget coverage/positive failure')
   if g['shot']!=p['shot_ids'][int(a['shot_index'][i])] or int(g['native_pts'])!=int(a['native_pts'][i]):raise ValueError('parent actual exposure differs')
   seen.add(i);reconstructed[i]=w;assignment[i]=j
 if seen!=set(np.flatnonzero(A).tolist()) or not exact(reconstructed,parent_weights):raise ValueError('parent ledger/explicit binary per-source bits differ')
 if not np.isfinite(parent_weights).all() or not np.array_equal(parent_weights>0,A):raise ValueError('parent actual budget partition differs')
 authority=read(ap)
 if authority['schema']!='rinkfit.engineering_pixel_contributions.v1' or authority.get('original_W_likelihood_consumed',True):raise ValueError('authority is not original engineering ledger')
 ag=[dict(group_index=j,id=g['id'],rows=int(g['rows']),per_row_weight=float(g['per_row_weight']))for j,g in enumerate(authority['groups'])if g['shot']=='s03'and g['family']=='board']
 channel_budget=sum(g['rows']*g['per_row_weight']for g in ag)
 if not np.isfinite(channel_budget) or channel_budget<=0:raise ValueError('original S03/board budget missing')
 # The native source proof supplies actual roles and source families. It was
 # already executed against actual RGB; this issuer does not invent source truth.
 proof=read(sp);parent_model=read(cp/'model.json')
 if proof['schema']!='rinkfit.current_native_printed_board_source_qualification.v1' or proof['old_raw_rows']!=n or proof['raw_rows']!=N:raise ValueError('not exact printed source suffix proof')
 canonical_path=checked(proof['source_packet']);canonical=read(canonical_path);c=load(canonical,canonical_path.parent)
 if canonical['row_count']!=N:raise ValueError('canonical source denominator changed')
 for k in ['source_row_ids','source_dataset_ids','source_family_ids']:
  if canonical[k]!=q[k]:raise ValueError('printed source registry differs '+k)
 for k in source_fields:
  if not exact(c[k],b[k]):raise ValueError('printed source changed beyond allowed auxiliary binding '+k)
 if canonical['queries']!=q['queries'] or canonical['groups']!=q['groups'] or canonical['engineering_pixel_dependency_groups']!=q['engineering_pixel_dependency_groups']:raise ValueError('printed original Query/W/dependency changed')
 qualified=checked(proof['qualified_rows_native']);records=[json.loads(s)for s in qualified.read_text(encoding='utf8').splitlines()if s];byrow={int(r['global_row']):r for r in records}
 new=np.flatnonzero(B&(np.arange(N)>=n));newset=set(new.tolist())
 if len(records)!=len(byrow) or set(byrow)!=newset or len(new)!=proof['count']:raise ValueError('printed per-ID proof/active partition differs')
 units={};depgroup=q['engineering_pixel_dependency_groups']['groups']
 for j,g in enumerate(depgroup):
  for raw in g['rows']:
   i=int(raw)
   if i in units:raise ValueError('duplicate source dependency membership')
   units[i]=(j,g['id'])
 weights=np.zeros(N,dtype='<f8');weights[:n]=parent_weights;classes=np.zeros(N,dtype='u1');classes[:n]=parent_classes
 newunit=np.full(N,0xffffffff,dtype='<u4');queries={int(x['row_index']):x for x in q['queries']};channel=collections.defaultdict(list);exposures=set();roles=collections.defaultdict(set);deps=collections.defaultdict(set)
 for i in new:
  r=byrow[int(i)];sid=q['shot_ids'][int(b['shot_index'][i])];pt=int(b['native_pts'][i]);role=r['physical_role'];fam=q['source_family_ids'][i];cq=queries.get(int(i))
  if sid!='s03' or (int(b['kind'][i])!=0 or parent_model['materials'][int(b['material_index'][i])]['paths'][int(b['path'][i])]['kind']!=3) or cq is None or len(cq['alpha'])!=49 or len(cq['native_uv'])!=49 or role not in ['board.top','board.contact'] or not np.isfinite(b['E'][i]) or b['E'][i]<=0:raise ValueError('new row is not actual S03 printed normal profile')
  if r['source_row_id']!=q['source_row_ids'][i] or r['source_dataset_id']!=q['source_dataset_ids'][i] or r['source_family_id']!=fam or r['shot_id']!=sid or r['native_pts']!=pt or r['material_index']!=int(b['material_index'][i]) or np.asarray(r['current_uv'],dtype='<f8').tobytes()!=b['uv'][i].tobytes():raise ValueError('printed per-ID actual role/source binding differs')
  if r['physical_role_review']['status']!='pass' or r['original_operator_review']['status']!='operator_pass':raise ValueError('unqualified actual role/operator')
  j,unit=units[int(i)];key=(sid,pt,role,fam,unit);channel[key].append(int(i));exposures.add((sid,pt));roles[sid,pt].add((role,fam));deps[sid,pt,role,fam].add(unit);classes[i]=4;newunit[i]=j
 descriptors=[]
 for key,rr in sorted(channel.items()):
  sid,pt,role,fam,unit=key;w=channel_budget/len(exposures)/len(roles[sid,pt])/len(deps[sid,pt,role,fam])/len(rr);weights[rr]=w
  descriptors.append(dict(shot=sid,native_pts=pt,physical_role=role,actual_source_family_id=fam,source_dependency_id=unit,source_dependency_index=int(newunit[rr[0]]),raw_rows=rr,rows=len(rr),per_row_weight=w,source_noise_independence_claimed=False))
 if not np.array_equal(weights>0,B) or not np.isfinite(weights).all() or not exact(weights[:n],parent_weights) or not np.array_equal(classes[:n],parent_classes):raise ValueError('inherited budget/class or complete active partition drift')
 if not np.isclose(weights[new].sum(),channel_budget,rtol=1e-13,atol=1e-10):raise ValueError('new engineering channel does not close')
 out.mkdir(parents=True);files={}
 for name,x in [('weights.f64',weights),('active.u8',B.astype('u1')),('information_class.u8',classes),('parent_group.u32',assignment),('new_dependency.u32',newunit)]:
  x.tofile(out/name);files[name]={**ref(out/name),'dtype':x.dtype.str,'shape':list(x.shape)}
 report=dict(schema=SCHEMA,parent_packet=ref(pp),source_packet=ref(npk),parent_pixel_ledger=ref(lp),parent_complete_binding=ref(cp/'source_uv_capture_state_binding.json'),parent_complete_checkpoint=ref(cp/'engineering_checkpoint.json'),parent_model=ref(cp/'model.json'),parent_state_token=binding['state_token'],parent_explicit_source_weights=oldref,issuer=ref(__file__),source_printed_qualification=ref(sp),canonical_printed_source=proof['source_packet'],qualified_rows_native=proof['qualified_rows_native'],budget_authority_pixel_ledger=ref(ap),budget_authority_groups=ag,new_profile_budget_authority='exact original 9683 S03/board engineering ledger sum; no new multiplier',source_key_registry_sha256=key_digest(q,b),parent_prefix_key_registry_sha256=key_digest(p,a),source_E_binary64_le_sha256=hashlib.sha256(np.asarray(b['E'],dtype='<f8').tobytes()).hexdigest(),source_UV_binary64_le_sha256=hashlib.sha256(np.asarray(b['uv'],dtype='<f8').tobytes()).hexdigest(),parent_rows=n,raw_rows=N,parent_active_rows=int(A.sum()),inherited_active_rows=int(A.sum()),active_rows=int(B.sum()),new_direct_normal_profile_rows=len(new),new_raw_rows=N-n,information_classes={**oldmanifest['information_classes'],'4':'new actual S03 printed direct normal profile; separate engineering source channel'},files=files,retained_parent_budget=float(weights[:n].sum()),new_profile_channel_budget=float(channel_budget),actual_total_engineering_budget=float(weights.sum()),new_profile_exposures=len(exposures),new_profile_units=descriptors,old_budget_bits_preserved=True,old_information_class_bits_preserved=True,source_noise_W_or_sigma_issued=False,statistical_precision_or_noise_independence_claimed=False,source_source_UV_Query_W_or_E_changed=False,old_2D_point_budget_diluted_by_new_normal_rows=False,parent_complete_auxiliary_identity_delegated_to='capture_primal_prefix_v4',physical_Solver_started=False,production_consumer='rinkfit/joint_capture_weights_v4.hpp::load_joint_capture_weights_v4',quality_accepted=False)
 (out/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf8');return report

def main():
 p=argparse.ArgumentParser()
 for s in ['parent-packet','parent-ledger','parent-candidate','packet','out','authority-ledger','printed-qualification']:p.add_argument('--'+s,required=True)
 a=p.parse_args();r=issue(a.parent_packet,a.parent_ledger,a.parent_candidate,a.packet,a.out,authority_pixel_ledger=a.authority_ledger,printed_qualification=a.printed_qualification)
 print(json.dumps({k:r[k]for k in ['parent_state_token','raw_rows','active_rows','new_direct_normal_profile_rows','retained_parent_budget','new_profile_channel_budget','actual_total_engineering_budget']},indent=2))
if __name__=='__main__':main()
