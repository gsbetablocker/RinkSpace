"""Fresh-state physical SOC restoration inside an ORIGINAL reference total step.
No objective/source change, nonlinear acceptance, optimum or KKT is inferred.
"""
from pathlib import Path
import argparse,copy,hashlib,json,sys,time,importlib.util
import numpy as np
from scipy import sparse
import clarabel

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def verify(r,base=None):
 p=Path(r.get('path',r.get('file')));p=p if p.is_absolute()else Path(base)/p
 if sha(p)!=r['sha256']:raise ValueError('Changed immutable reference '+str(p))
 return p

def run(input_path,output,context_only=False,stop_file=None):
 started=time.perf_counter();IP=Path(input_path).resolve();s=read(IP)
 if s['schema']!='rinkfit.fresh_candidate_total_step_SOC_restoration_input.v1':raise ValueError('Explicit fresh-candidate restoration scope required')
 REFP=verify(s['reference_prepared']);REF=read(REFP);CANDP=verify(s['candidate_prepared']);CAND=read(CANDP);PROP=verify(s['original_proposal']);prop=read(PROP)
 for d in [REF,CAND]:
  for k in ['checkpoint','linearization','transport','profile_domains']:verify(d[k])
 RC=verify(REF['checkpoint']);rc=read(RC);CC=verify(CAND['checkpoint']);cc=read(CC);TM=read(verify(CAND['transport']));RT=read(verify(REF['transport']))
 for k in ['packet','semantic']:
  if TM[k]['sha256']!=RT[k]['sha256']:raise ValueError('Restoration cannot change source objective '+k)
 if prop['transport']['sha256']!=REF['transport']['sha256']:raise ValueError('Original proposal belongs to another reference')
 B=float(s['working_b']);trust=float(s['original_total_trust_scaled'])
 if not(np.isfinite(B)and 0<B<min(float(rc['b']),float(prop['before_original_affine_zero_q'])))or not np.isfinite(trust)or trust<=0:raise ValueError('Working target must descend original reference beforeq under both evaluators')
 if trust!=float(rc['control']['trust']):raise ValueError('Do not enlarge or shrink original total physical trust')
 bundle=verify(s['primary_bundle']);bd=read(bundle)
 for x in bd.get('frozen_sources',[]):verify(x.get('frozen',x))
 ENTRY=verify(bd.get('entry',s['primary_entry'])) if 'entry'in bd else verify(s['primary_entry'])
 root=ENTRY.parent;sys.path.insert(0,str(root))
 from finite_visible_norm_core_i17_r3 import FiniteVisiblePrimary,array
 c=FiniteVisiblePrimary(verify(CAND['linearization']),verify(CAND['transport']),verify(CAND['profile_domains']),trust,station_TR=np.zeros(182017))
 oldmodel=read(verify(RT['model']));candidate_model=c.model
 if oldmodel['camera']!=candidate_model['camera']or oldmodel['materials']!=candidate_model['materials']:raise ValueError('Physical camera/material hypothesis changed')
 def block_template(b):return{k:v for k,v in b.items()if k!='values'}
 if list(map(block_template,oldmodel['blocks']))!=list(map(block_template,candidate_model['blocks'])):raise ValueError('Original blocks/bounds/scales/gauge changed')
 oldvalues=np.concatenate([b['values']for b in oldmodel['blocks']]);raw=dict(np.load(verify(prop['proposal'])))
 if not np.array_equal(raw['parent_values'],oldvalues)or not np.array_equal(oldvalues+raw['physical_delta'],c.value):raise ValueError('Fresh candidate is not the original full proposed physical state')
 if not np.array_equal(raw['scales'],c.scale)or not np.array_equal(raw['free_ambient_indices'],c.free):raise ValueError('Original physical coordinate layout changed')
 d0=(c.value[c.free]-oldvalues[c.free])/c.scale[c.free];lower0=np.concatenate([b['lower']for b in oldmodel['blocks']]);upper0=np.concatenate([b['upper']for b in oldmodel['blocks']]);total_lo=np.maximum(-trust,(lower0[c.free]-oldvalues[c.free])/c.scale[c.free]);total_hi=np.minimum(trust,(upper0[c.free]-oldvalues[c.free])/c.scale[c.free])
 lo=total_lo-d0;hi=total_hi-d0
 if np.any(lo>0)or np.any(hi<0):raise ValueError('Original full candidate is outside original total box/TR')
 D=Path(output).resolve();D.mkdir(exist_ok=False)
 # All J/value rows are FRESH at the same candidate physical+rawN auxiliary state.
 # Ordinary S is a temporary legal witness in this corrector, not a new hard fact.
 A=c.A[:,:-1].tocsr();bcol=np.asarray(c.A[:,-1].toarray()).ravel();rhs=c.rhs-bcol*B
 unit=trust;lin=(A*unit).tocsr();zl=lo/unit;zu=hi/unit
 mag=abs(A)@np.maximum(abs(lo),abs(hi));eps=64*np.finfo(float).eps*(1+abs(rhs)+mag);rs=np.exp2(np.maximum(0,np.ceil(np.log2((32e-10)/eps))))
 linear=sparse.vstack([lin.multiply(rs[:,None]),sparse.eye(c.n),-sparse.eye(c.n)],format='csc');linear_rhs=np.r_[rhs*rs,zu,-zl]
 guard_source=verify(s['print_guard_source']);sp=importlib.util.spec_from_file_location('private_SOC_restoration_print_guard',guard_source);gm=importlib.util.module_from_spec(sp);sys.modules[sp.name]=gm;sp.loader.exec_module(gm);guard=gm.from_full_norm_operator(c.full,c.free,c.scale);GA,gb,gd=guard.clarabel_soc6(np.zeros(c.n),np.full(c.n,unit),0)
 J=c.full.J.tocsr();K=(J*unit).tocsr();r0=c.full.residual;source_ids=c.rows;idx=np.full(182017,-1,int);idx[source_ids]=np.arange(len(source_ids));initial_q=np.einsum('ij,ij->i',r0,r0)
 members=set(map(int,source_ids[(initial_q>=.9*B)|(c.kind[source_ids]==1)]));history=[];final=None
 def audit(z):
  correction=unit*z;total=d0+correction;rr=r0+(J@correction).reshape(-1,2);q=np.einsum('ij,ij->i',rr,rr);bad=source_ids[q>B];v=A@correction-rhs;allow=64*np.finfo(float).eps*(1+abs(rhs)+abs(A)@abs(correction));sensor=c.row0[c.sensorrows]+c.Jv@correction;values=c.value.copy();values[c.free]+=c.scale[c.free]*correction
  return dict(all_active_source_rows=len(source_ids),maximum_affine_q=float(q.max()),worst_source_row=int(source_ids[q.argmax()]),violating_source_rows=bad.tolist(),all_source_affine_norms_within_b=not len(bad),fresh_original_A_beyond_IEEE=int(np.count_nonzero(v>allow)),fresh_original_scene_profile_and_pixel_rows=c.preserved_counts,all_fresh_sensor_domains=bool(np.all(sensor>=0)&np.all(sensor<=1079)),original_total_box_and_TR_strict=bool(np.all(total>=total_lo)&np.all(total<=total_hi)),actual_final_parameter_box_strict=bool(np.all(values>=lower0)&np.all(values<=upper0)),print_guard=guard.audit(correction),correction=correction,total=total,values=values)
 context=dict(schema='rinkfit.fresh_candidate_total_step_SOC_restoration_context.v1',input=ref(IP),source=ref(__file__),reference_checkpoint=ref(RC),candidate_checkpoint=ref(CC),candidate_prepared=ref(CANDP),working_b=B,original_total_trust=trust,original_proposal=ref(PROP),fresh_scene_profile_and_verified_qbar_rows=c.preserved_counts,all_active_rows=len(source_ids),physical_coordinates=c.n,original_W_E_and_source_model_unchanged=True,value_and_J_at_same_fresh_candidate=True,original_total_step_box_not_correction_box=True,temporary_fixed_original_S_witness_not_permanent=True,numerical_objective='Minimum squared original-scaled physical correction, algorithmic restoration selection only; no statistical or smoothing prior.',no_nonlinear_or_KKT_acceptance=True)
 np.savez_compressed(D/'original_total_box.npz',candidate_total_step=d0,lower=total_lo,upper=total_hi,correction_lower=lo,correction_upper=hi,original_reference_values=oldvalues,candidate_values=c.value,scales=c.scale,free_ambient_indices=c.free)
 sparse.save_npz(D/'fresh_linear_A.npz',A);np.savez_compressed(D/'fresh_linear_rhs.npz',rhs=rhs,linear_row_scale=rs);(D/'context.json').write_text(json.dumps(context,indent=2),encoding='utf8')
 if context_only:
  report=dict(context,context_checks_passed=True,proposed_for_actual_nonlinear_trial=False,no_Solve=True);(D/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(context=ref(D/'report.json'))));return
 generation=0
 while True:
  if stop_file and Path(stop_file).exists():break
  I=D/('generation_%03d'%generation);I.mkdir();ids=np.array(sorted(members),dtype=np.int64);ii=idx[ids];two=(2*ii[:,None]+np.arange(2)).ravel();core=sparse.vstack([sparse.csr_matrix((len(ids),c.n)),K[two]],format='csr');perm=np.column_stack([np.arange(len(ids)),len(ids)+2*np.arange(len(ids)),len(ids)+2*np.arange(len(ids))+1]).ravel();SOC=core[perm].tocsc();soc_rhs=np.column_stack([np.full(len(ids),np.sqrt(B)),r0[ii]]).ravel();AA=sparse.vstack([linear,GA,-SOC],format='csc');bb=np.r_[linear_rhs,gb,soc_rhs]
  cones=[clarabel.NonnegativeConeT(len(linear_rhs))]+[clarabel.SecondOrderConeT(k)for k in gd]+[clarabel.SecondOrderConeT(3)for _ in ids]
  settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_threads=1;settings.tol_feas=1e-12;settings.tol_gap_abs=1e-12;settings.tol_gap_rel=1e-12;settings.max_iter=100
  print('FRESH_SOC_RESTORATION',json.dumps(dict(generation=generation,source_cones=len(ids),original_total_trust=trust,working_b=B,elapsed=time.perf_counter()-started)),flush=True)
  solver=clarabel.DefaultSolver(sparse.eye(c.n,format='csc'),np.zeros(c.n),AA,bb,cones,settings);sol=solver.solve();status=str(sol.status)
  if sol.x is None or not np.isfinite(sol.x).all():history.append(dict(generation=generation,status=status,no_finite_primal=True));break
  z=np.asarray(sol.x);a=audit(z);np.savez_compressed(I/'raw_solver.npz',z=z,dual=np.asarray(sol.z),slack=np.asarray(sol.s));public={k:v for k,v in a.items()if not isinstance(v,np.ndarray)};new=set(a['violating_source_rows'])-members;item=dict(generation=generation,solver_status=status,iterations=int(sol.iterations),audit=public,new_source_rows=len(new),elapsed=time.perf_counter()-started);history.append(item);(I/'report.json').write_text(json.dumps(item,indent=2));print('FRESH_SOC_RESTORATION_AUDIT',json.dumps(item),flush=True)
  if new:members.update(new);generation+=1;continue
  passed=a['all_source_affine_norms_within_b']and a['fresh_original_A_beyond_IEEE']==0 and a['all_fresh_sensor_domains']and a['original_total_box_and_TR_strict']and a['actual_final_parameter_box_strict']and a['print_guard']['convex_print_guard_pass']and a['print_guard']['exact_original_print_g_pass']
  if not passed:break
  correction_phys=np.zeros_like(c.value);correction_phys[c.free]=c.scale[c.free]*a['correction'];total_phys=a['values']-oldvalues;np.savez_compressed(D/'proposal.npz',correction_scaled=a['correction'],correction_physical=correction_phys,total_step_scaled=a['total'],total_physical_delta=total_phys,original_reference_values=oldvalues,candidate_values=c.value,scales=c.scale,free_ambient_indices=c.free,working_b=B)
  mm=copy.deepcopy(candidate_model)
  for i,b in enumerate(mm['blocks']):b['values']=a['values'][c.offset[i]:c.offset[i+1]].tolist()
  mm['second_order_restoration_provenance']=dict(context=ref(D/'context.json'),proposal=ref(D/'proposal.npz'),not_accepted_candidate=True,original_total_trust_preserved=True)
  (D/'model.json').write_text(json.dumps(mm,indent=2),encoding='utf8');final=dict(proposal=ref(D/'proposal.npz'),model=ref(D/'model.json'),working_b=B,correction_scaled_inf=float(np.max(abs(a['correction']))),total_step_scaled_inf=float(np.max(abs(a['total']))),full_source_affine_audit=public);break
 report=dict(schema='rinkfit.fresh_candidate_total_step_SOC_restoration_proposal.v1',context=ref(D/'context.json'),input=ref(IP),source=ref(__file__),reference_checkpoint=ref(RC),candidate_checkpoint=ref(CC),fresh_transport=CAND['transport'],fresh_profile_domains=CAND['profile_domains'],fresh_linearization=CAND['linearization'],history=history,proposed_for_actual_nonlinear_trial=bool(final),final=final,working_b=B,source_W_E_unchanged=True,original_total_box_TR_not_relaxed=True,raw_solver_status_not_overridden=True,original_S_not_permanently_frozen=True,PhaseII_W_optimum=False,PHR_multiplier_update_permitted=False,true_KKT_or_final_fit=False,elapsed_seconds=time.perf_counter()-started)
 (D/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(dict(report=ref(D/'report.json'),proposed=bool(final),elapsed_seconds=report['elapsed_seconds'])));return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--check-context-only',action='store_true');p.add_argument('--stop-file');a=p.parse_args();run(a.input,a.output,a.check_context_only,a.stop_file)
