"""Conditional original-W secondary with existing physical and finite-S columns.
Data operator. It never certifies nonlinear feasibility, Phase-II optimum, or KKT.
The caller supplies the SAME-STATE GroupWStation, source norm operator, original
A/rhs (including original b column), and complete primary station recovery.
"""
from pathlib import Path
import hashlib,json,time
import numpy as np
from scipy import sparse
import clarabel

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def csr_equal(a,b):
 a=a.tocsr();b=b.tocsr();return a.shape==b.shape and np.array_equal(a.indptr,b.indptr)and np.array_equal(a.indices,b.indices)and np.array_equal(a.data,b.data)

def solve(chart,norms,engineering_E,primary_delta,working_b,original_A,original_rhs,trust,output,*,seed_membership=None,stop_file=None,assembly_only=False,primary_epigraph_coordinate="squared_budget_b",primary_epigraph_value=None,scene_guard=None):
 start=time.monotonic();O=Path(output).resolve();O.mkdir(exist_ok=False)
 n=len(chart.free);k=len(chart.selected);rows=norms.rows;N=len(chart.station_seed);E=np.asarray(engineering_E,float)
 d0=np.asarray(primary_delta,float);b=float(working_b);A=original_A.tocsr();rhs=np.asarray(original_rhs,float)
 if N!=182017 or len(rows)!=182016 or np.flatnonzero(~np.isin(np.arange(N),rows)).tolist()!=[25537]:raise ValueError('Original active source ledger')
 if d0.shape!=(n,)or E.shape!=(N,)or np.any(E[rows]<=0)or not np.isfinite(np.r_[d0,b,E[rows]]).all()or b<=0:raise ValueError('Finite original coordinates/E/b')
 if A.shape!=(len(rhs),n+1)or norms.J.shape!=(2*len(rows),n):raise ValueError('Original physical matrix dimensions')
 cp_ref=chart.authority['checkpoint'];cp_path=Path(cp_ref['path'])
 if sha(cp_path)!=cp_ref['sha256']:raise ValueError('Changed complete original checkpoint')
 parent_checkpoint=json.loads(cp_path.read_text(encoding='utf8'))
 if primary_epigraph_coordinate not in ['squared_budget_b','radius_sqrt_b']:raise ValueError('Explicit primary epigraph coordinate required')
 epigraph=b if primary_epigraph_coordinate=='squared_budget_b' else float(primary_epigraph_value) if primary_epigraph_value is not None else float('nan')
 if not np.isfinite(epigraph)or epigraph<0:raise ValueError('Exact canonical primary radius is required; it is not reconstructed from rounded b')
 if primary_epigraph_coordinate=='radius_sqrt_b'and epigraph*epigraph>b:raise ValueError('Canonical squared budget must enclose the declared original radius')
 before_original_native_b=float(parent_checkpoint['b']);before_affine_zero_b=float(norms.values(np.zeros(n))[0].max())
 if not assembly_only and not b<min(before_original_native_b,before_affine_zero_b):raise ValueError('Declared-inexact Phase-I working budget must decrease under both original same-state before evaluators')
 # Strong state authority is checked by the frozen group-W loader. Also bind
 # this norm object to the same original transport/model/source authority.
 for key in ['model','packet','semantic']:
  if norms.metadata[key]['sha256']!=chart.authority[key]['sha256']:raise ValueError('Mixed source/physical norm authority '+key)
 if norms.metadata['transport']['sha256']!=chart.authority['transport']['sha256']:raise ValueError('Mixed station/norm transport')
 lowp=np.maximum(chart.lower_physical,-float(trust));upp=np.minimum(chart.upper_physical,float(trust))
 if np.any(d0<lowp)or np.any(d0>upp):raise ValueError('Primary outside original physical box/TR')
 # Reconstruct the original normalized source residual with the FULL station
 # seed, including every unselected ordinary source; profiles/entities stay fixed.
 tp=Path(chart.authority['transport']['path']);td=json.loads(tp.read_text(encoding='utf8'))
 ad=td['arrays']['station_UV_right'];ap=Path(ad.get('path',ad.get('file')));ap=ap if ap.is_absolute()else tp.parent/ap
 if sha(ap)!=ad['sha256']:raise ValueError('Changed original station tangent')
 tangent=np.fromfile(ap,ad['dtype']).reshape(N,2)
 seed_uv=tangent*chart.station_seed[:,None];r0=norms.residual.copy();r0+=seed_uv[rows]/E[rows,None]
 raw_uv_rows=(2*rows[:,None]+np.arange(2)).ravel();Ts=chart.T[raw_uv_rows].tocsr();Ts=sparse.diags(np.repeat(1/E[rows],2))@Ts
 J=sparse.hstack([norms.J,Ts],format='csr');W=sparse.hstack([chart.Jphysical,chart.Jstation],format='csc')
 center=np.r_[d0,np.zeros(k)];ss=np.maximum(abs(chart.station_lower),abs(chart.station_upper))
 if k and(np.any(ss<=0)or not np.isfinite(ss).all()):raise ValueError('Finite nonzero original station-column scaling')
 scale=np.r_[np.full(n,float(trust)),ss];lo=np.r_[lowp,chart.station_lower];hi=np.r_[upp,chart.station_upper];zl=(lo-center)/scale;zu=(hi-center)/scale
 R=r0+(J@center).reshape(-1,2);K=(J@sparse.diags(scale)).tocsr();rw=chart.rW+W@center;KW=(W@sparse.diags(scale)).tocsc();primary_W=float(.5*rw@rw)
 # All old scene/profile/verified-source and reachability rows retain their
 # original coefficients, RHS, budget coordinate and physical bounds.
 baseA=sparse.hstack([A[:,:-1],sparse.csr_matrix((A.shape[0],k))],format='csr');br=rhs-np.asarray(A[:,-1].toarray()).ravel()*epigraph
 B=(baseA@sparse.diags(scale)).tocsr();bc=br-baseA@center
 sensor=sparse.hstack([chart.sensor_Jphysical,chart.sensor_Jstation],format='csr');sv=chart.sensor_row0+sensor@center;V=(sensor@sparse.diags(scale)).tocsr()
 allsensor=sparse.vstack([V,-V],format='csr');s_rhs=np.r_[1079-sv,sv]
 # Box-wide enclosing sums prove omitted sensor inequalities redundant.
 upper_est=allsensor.maximum(0)@zu+allsensor.minimum(0)@zl
 mag=abs(allsensor)@np.maximum(abs(zl),abs(zu));terms=np.diff(allsensor.indptr);g=(2*terms+8)*np.finfo(float).eps;g=g/(1-g)
 enclosure=np.nextafter(upper_est+g*mag/(1-g),np.inf);keep=np.flatnonzero(enclosure>s_rhs)
 B=sparse.vstack([B,allsensor[keep]],format='csc');bc=np.r_[bc,s_rhs[keep]]
 unit=sparse.eye(n+k,format='csc');B=sparse.vstack([B,unit,-unit],format='csc');bc=np.r_[bc,zu,-zl]
 guardA,guardrhs,guarddims=(scene_guard.clarabel_soc6(d0,scale[:n],k)if scene_guard is not None else(sparse.csc_matrix((0,n+k)),np.empty(0),[]))
 factor=max(1.,primary_W);P=sparse.triu((KW.T@KW)/factor,format='csc');q=np.asarray(KW.T@rw).ravel()/factor
 def audit(z):
  x=center+scale*z;delta=x[:n];eta=x[n:];rr=r0+(J@x).reshape(-1,2);qa=np.einsum('ij,ij->i',rr,rr);native=np.r_[delta,epigraph];v=A@native-rhs;eps=64*np.finfo(float).eps*(1+abs(rhs)+abs(A)@abs(native));ys=chart.sensor_row0+sensor@x
  wr=chart.rW+W@x;box=bool(np.all(x>=lo)and np.all(x<=hi));domain=bool(np.all(ys>=0)and np.all(ys<=1079));bad=rows[qa>b]
  return dict(print_determinant_guard=scene_guard.audit(delta)if scene_guard is not None else None,original_A_beyond_IEEE=int(np.count_nonzero(v>eps)),strict_original_box_and_finite_station_cell=box,all_active_sensor_domains=domain,all_source_affine_norms_within_b=bool(len(bad)==0),violating_source_rows=bad.tolist(),violating_source_count=len(bad),maximum_affine_q=float(qa.max()),worst_source_row=int(rows[int(np.argmax(qa))]),original_W_half_cost=float(.5*wr@wr),physical_delta_scaled=delta,station_eta=eta,normalized_step=z)
 anchor=audit(np.zeros(n+k));anchor_ok=anchor['original_A_beyond_IEEE']==0 and anchor['strict_original_box_and_finite_station_cell']and anchor['all_active_sensor_domains']and anchor['all_source_affine_norms_within_b']
 if not anchor_ok:
  (O/'anchor_failure.json').write_text(json.dumps({a:v for a,v in anchor.items()if not isinstance(v,np.ndarray)},indent=2));raise ValueError('Supplied primary recovery is not strictly feasible under the SAME original residual and full station seed')
 if assembly_only:
  report=dict(schema='rinkfit.finite_visible_original_W_coupled_secondary_assembly.v1',authority=chart.authority,no_Solve=True,no_physical_proposal=True,physical_columns=n,station_columns=k,original_A_rows=A.shape[0],all_source_rows=len(rows),original_W_rank=len(chart.rW),primary_anchor_audit={a:v for a,v in anchor.items()if not isinstance(v,np.ndarray)},objective_H_nnz=P.nnz,retained_sensor_rows=len(keep),sensor_rows_box_proven_redundant=len(s_rhs)-len(keep),unselected_station_seed_preserved=True,profile_and_entity_S_not_released=True,elapsed_seconds=time.monotonic()-start,source=ref(__file__))
  (O/'report.json').write_text(json.dumps(report,indent=2));return report
 membership=set(map(int,rows[norms.kind==1]));membership.update(map(int,[] if seed_membership is None else seed_membership));membership.update(map(int,rows[np.einsum('ij,ij->i',R,R)>=.9*b]))
 if not membership.issubset(set(map(int,rows))):raise ValueError('Invalid active norm membership')
 history=[];generation=0;final=None
 while True:
  if stop_file and Path(stop_file).exists():break
  I=O/('generation_%03d'%generation);I.mkdir();ids=np.array(sorted(membership),dtype=np.int64);idx=np.searchsorted(rows,ids);ri=np.repeat(3*np.arange(len(ids))+1,2)+np.tile(np.arange(2),len(ids));two=(2*idx[:,None]+np.arange(2)).ravel()
  soc=sparse.vstack([sparse.csr_matrix((len(ids),n+k)),K[two]],format='csr')
  # Interleave [sqrt(b), residual_u, residual_v] for each source cone.
  perm=np.column_stack([np.arange(len(ids)),len(ids)+2*np.arange(len(ids)),len(ids)+2*np.arange(len(ids))+1]).ravel();soc=soc[perm].tocsc();cb=np.column_stack([np.full(len(ids),np.sqrt(b)),R[idx]]).ravel()
  AA=sparse.vstack([B,guardA,-soc],format='csc');bb=np.r_[bc,guardrhs,cb];cones=[clarabel.NonnegativeConeT(len(bc))]+[clarabel.SecondOrderConeT(d)for d in guarddims]+[clarabel.SecondOrderConeT(3)for _ in ids]
  settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_threads=1;settings.max_iter=100;settings.tol_gap_abs=1e-10;settings.tol_gap_rel=1e-10;settings.tol_feas=1e-10
  print(json.dumps(dict(stage='finite_S_original_W_QP',generation=generation,physical_columns=n,station_columns=k,norm_cones=len(ids),elapsed_seconds=time.monotonic()-start)),flush=True)
  solver=clarabel.DefaultSolver(P,q,AA,bb,cones,settings);sol=solver.solve();raw_status=str(sol.status)
  if sol.x is None or len(sol.x)!=n+k or not np.isfinite(sol.x).all():
   history.append(dict(generation=generation,solver_status=raw_status,missing_or_nonfinite_primal=True));break
  z=np.asarray(sol.x);au=audit(z)
  np.savez_compressed(I/'raw_solver.npz',x=z,z=np.asarray(sol.z),s=np.asarray(sol.s));new=set(au['violating_source_rows'])-membership
  item=dict(generation=generation,solver_status=raw_status,solver_iterations=int(sol.iterations),norm_cones=len(ids),full_source_audit={a:v for a,v in au.items()if not isinstance(v,np.ndarray)},new_source_constraints=len(new),elapsed_seconds=time.monotonic()-start);history.append(item);(I/'report.json').write_text(json.dumps(item,indent=2));print(json.dumps(item),flush=True)
  if new:membership.update(new);generation+=1;continue
  # A numerical QP status is never promoted. Only strict original-unit primal
  # gates and an actual original-W reduction from the feasible primary anchor
  # can authorize this declared-inexact conditional proposal.
  good=au['original_A_beyond_IEEE']==0 and au['strict_original_box_and_finite_station_cell']and au['all_active_sensor_domains']and au['all_source_affine_norms_within_b']and au['original_W_half_cost']<primary_W
  if scene_guard is not None:good=good and au['print_determinant_guard']['convex_print_guard_pass']and au['print_determinant_guard']['exact_original_print_g_pass']
  if not good:break
  total=chart.station_seed.copy();total[chart.selected]+=au['station_eta'];delta=au['physical_delta_scaled'];physical=np.zeros(len(chart.parent_values));physical[chart.free]=chart.scale[chart.free]*delta
  np.savez_compressed(O/'proposal_step.npz',physical_delta=physical,step_scaled=delta,free_ambient_indices=chart.free,parent_values=chart.parent_values,scales=chart.scale,canonical_primal=np.r_[delta,b],station_delta_full=total,station_eta=au['station_eta'],station_source_rows=chart.selected)
  wv=(chart.rW-chart.original_rW)+chart.Jphysical@delta+chart.Jstation@au['station_eta'];linear=float(chart.original_rW@wv);quadratic=float(.5*wv@wv)
  final=dict(proposed=True,proposal=ref(O/'proposal_step.npz'),predicted_W_baseline=float(.5*chart.original_rW@chart.original_rW),predicted_W_linear_coefficient=linear,predicted_W_quadratic_coefficient=quadratic,working_target_b=b,primary_W_half_cost=primary_W,predicted_W_half_cost=au['original_W_half_cost'],all_source_affine_audit={a:v for a,v in au.items()if not isinstance(v,np.ndarray)},raw_solver_status=raw_status,numeric_solver_status_not_overridden=True,declared_inexact_conditional_W_secondary=True,full_source_actual_nonlinear_line_required=True,source_station_coordinates_are_original_not_new_DOF=True,PhaseII_joint_station_optimum=False,PHR_multiplier_update_permitted=False,final_fit_or_KKT=False);break
 report=dict(schema='rinkfit.finite_visible_original_W_coupled_secondary.v1',authority=chart.authority,proposed=bool(final),final=final,history=history,physical_columns=n,station_columns=k,raw_rows=N,active_rows=len(rows),original_W_rank=len(chart.rW),primary_working_b=b,primary_epigraph_coordinate=primary_epigraph_coordinate,fixed_primary_epigraph_value=epigraph,physical_proposal_authorized=bool(final),all_sensor_rows_retained_in_final_audit=len(chart.sensor_rows),sensor_rows_with_box_wide_redundancy_proof=len(s_rhs)-len(keep),retained_sensor_linear_rows=len(keep),same_original_A_rhs_and_bounds_retained=True,print_determinant_convex_guard_authority=scene_guard.authority if scene_guard is not None else None,added_print_guard_SOC_dimensions=guarddims,unselected_full_station_seed_preserved=True,only_explicit_original_S_columns_released=True,elapsed_seconds=time.monotonic()-start,source=ref(__file__))
 (O/'report.json').write_text(json.dumps(report,indent=2));return report
