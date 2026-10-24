"""Generic original full-S/all-source convex Phase-I primary.
min b is strictly monotone equivalent to min radius sqrt(b). Original affine
A(delta,b) is kept in b coordinates; no nonconvex substitution b=t*t in A.
All original ordinary S are latent variables, with original finite/cell/sensor
and declared numerical curvature trust. No new physical shape DOF or W change.
"""
from pathlib import Path
import argparse,sys,json,hashlib,time,numpy as np
from scipy import sparse
import clarabel

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def verify(d,base=None):
 p=Path(d.get('path',d.get('file')));p=p if p.is_absolute()else Path(base)/p
 if sha(p)!=d['sha256']:raise ValueError('Changed immutable input '+str(p))
 return p

def run(input_path,output,do_solve=False,QA_proposal=None):
 start=time.monotonic();IP=Path(input_path).resolve();spec=read(IP);O=Path(output).resolve();O.mkdir(exist_ok=False)
 if spec.get('schema')not in ['rinkfit.finite_visible_original_W_secondary_input.v1','rinkfit.joint_full_lawful_S_PhaseI_input.v1']:raise ValueError('Typed original full-S Phase-I input required')
 required=['linearization','transport','original_A','rhs_bounds','group_W_recipe','group_W_source','full_norm_source','station_TR_manifest','station_TR_limiter_source','print_det_guard_source']
 paths={k:verify(spec[k])for k in required};Q=paths['linearization'];T=paths['transport'];td=read(T);cpP=verify(td['checkpoint']);cp=read(cpP);N=int(td['raw_rows'])
 for key in ['group_W_source','full_norm_source','station_TR_limiter_source','print_det_guard_source']:sys.path.insert(0,str(paths[key].parent))
 from group_W_station import assemble,save
 from station_curvature_TR_chart import apply_station_curvature_TR
 from source_affine_norm_constraints import load_all_source_affine_norms
 from print_det_convex_guard import from_full_norm_operator
 requested=np.arange(N,dtype='<u8');rp=O/'all_requested_source_rows.u64';rp.write_bytes(requested.tobytes());sd=dict(schema='rinkfit.station_column_selection.v1',checkpoint=ref(cpP),transport=ref(T),source_rows=dict(file=rp.name,dtype='<u8',shape=[N],sha256=sha(rp)));SEL=O/'station_selection_manifest.json';SEL.write_text(json.dumps(sd,indent=2),encoding='utf8')
 seedP=verify(spec['station_seed'])if spec.get('station_seed')else None
 chart=assemble(Q,T,selection=SEL,station_seed=seedP,station_trust=spec.get('station_trust'));chart,trinfo=apply_station_curvature_TR(chart,paths['station_TR_manifest'],spec.get('station_TR_array_key','radius'));save(chart,O/'group_W_chart')
 n=len(chart.free);k=len(chart.selected);model=chart.model;offset=np.r_[0,np.cumsum([len(b['values'])for b in model['blocks']])];norms=load_all_source_affine_norms(Q,T,offset,chart.scale,chart.free);rows=norms.rows;m=len(rows)
 if N!=182017 or m!=182016 or np.flatnonzero(~np.isin(np.arange(N),rows)).tolist()!=[25537]:raise ValueError('Original all-source ledger')
 A=sparse.load_npz(paths['original_A']).tocsr();bd=np.load(paths['rhs_bounds']);rhs=bd['rhs'];lowb=float(bd['lower'][-1]);highb=float(bd['upper'][-1]);trust=float(spec['trust_scaled'])
 if A.shape!=(len(rhs),n+1)or lowb<0:raise ValueError('Original physical-plus-b affine master')
 PK=verify(chart.authority['packet']);pk=read(PK);ed=pk['arrays']['engineering_budget_px'];E=np.fromfile(verify(ed,PK.parent),ed['dtype']).reshape(ed['shape']);tand=td['arrays']['station_UV_right'];tan=np.fromfile(verify(tand,T.parent),tand['dtype']).reshape(N,2)
 allclo=np.fromfile(verify(td['arrays']['station_delta_lower_right'],T.parent),td['arrays']['station_delta_lower_right']['dtype']);allchi=np.fromfile(verify(td['arrays']['station_delta_upper_right'],T.parent),td['arrays']['station_delta_upper_right']['dtype']);curv=read(paths['station_TR_manifest']);rd=curv['arrays'][spec.get('station_TR_array_key','radius')];radius=np.fromfile(verify(rd,paths['station_TR_manifest'].parent),rd['dtype'])
 rawuv=(2*rows[:,None]+np.arange(2)).ravel();Ts=chart.T[rawuv].tocsr();Ts=sparse.diags(np.repeat(1/E[rows],2))@Ts;J=sparse.hstack([norms.J,Ts],format='csr');rseed=norms.residual+tan[rows]*chart.station_seed[rows,None]/E[rows,None]
 slo=chart.station_lower;shi=chart.station_upper;ss=np.maximum(abs(slo),abs(shi));lo=np.r_[np.maximum(chart.lower_physical,-trust),slo,lowb];hi=np.r_[np.minimum(chart.upper_physical,trust),shi,highb]
 qref=max(1.,float(cp['b']));units=np.r_[np.full(n,trust),ss,qref]
 if np.any(units<=0)or not np.isfinite(units).all():raise ValueError('Finite invertible original-coordinate units')
 # Preserve every original A row as affine in b, including all budget supports.
 original_expanded=sparse.hstack([A[:,:n],sparse.csr_matrix((len(rhs),k)),A[:,n:]],format='csr');Al=(original_expanded@sparse.diags(units)).tocsc();bl=rhs.copy();budget_col=np.asarray(A[:,-1].toarray()).ravel()
 sensor=sparse.hstack([chart.sensor_Jphysical,chart.sensor_Jstation,sparse.csr_matrix((len(chart.sensor_rows),1))],format='csr');V=(sensor@sparse.diags(units)).tocsc();sr=chart.sensor_row0
 # Same outward box-wide redundancy proof as the frozen W secondary; every
 # source still undergoes complete original-unit sensor audit after Solve.
 VS=sparse.vstack([V[:,:-1],-V[:,:-1]],format='csr');vrhs=np.r_[1079-sr,sr];zlo=lo[:-1]/units[:-1];zhi=hi[:-1]/units[:-1];upper_est=VS.maximum(0)@zhi+VS.minimum(0)@zlo;mag=abs(VS)@np.maximum(abs(zlo),abs(zhi));terms=np.diff(VS.indptr);gamma=(2*terms+8)*np.finfo(float).eps;gamma=gamma/(1-gamma);enclosure=np.nextafter(upper_est+gamma*mag/(1-gamma),np.inf);sensor_keep=np.flatnonzero(enclosure>vrhs);sensor_drop=np.flatnonzero(enclosure<=vrhs);sensorA=sparse.hstack([VS[sensor_keep],sparse.csr_matrix((len(sensor_keep),1))],format='csc');Al=sparse.vstack([Al,sensorA],format='csc');bl=np.r_[bl,vrhs[sensor_keep]]
 np.savez_compressed(O/'sensor_redundancy_proof.npz',kept_indices=sensor_keep,dropped_indices=sensor_drop,upper_enclosure=enclosure,original_rhs=vrhs,source_rows=chart.sensor_rows,lower_box=zlo,upper_box=zhi)
 lowz=lo/units;hiz=hi/units;eye=sparse.eye(n+k+1,format='csc');ui=np.flatnonzero(np.isfinite(hiz));Al=sparse.vstack([Al,-eye,eye[ui]],format='csc');bl=np.r_[bl,-lowz,hiz[ui]]
 guard=from_full_norm_operator(norms,chart.free,chart.scale);guardA,guardb,guarddims=guard.clarabel_soc6(np.zeros(n),units[:n],k+1)
 # SOC4(beta+1,2*rU/sqrt(qref),2*rV/sqrt(qref),beta-1).
 JC=(J@sparse.diags(units[:-1]/np.sqrt(qref))).tocsr();bi=sparse.csr_matrix((np.full(m,-1.),(np.arange(m),np.full(m,n+k))),shape=(m,n+k+1));jr=sparse.hstack([-2*JC,sparse.csr_matrix((2*m,1))],format='csr');stack=sparse.vstack([bi,jr,bi],format='csr');perm=np.column_stack([np.arange(m),m+2*np.arange(m),m+2*np.arange(m)+1,3*m+np.arange(m)]).ravel();soc=stack[perm].tocsc();cb=np.column_stack([np.ones(m),2*rseed/np.sqrt(qref),-np.ones(m)]).ravel()
 AA=sparse.vstack([Al,guardA,soc],format='csc');bb=np.r_[bl,guardb,cb];cones=[clarabel.NonnegativeConeT(len(bl))]+[clarabel.SecondOrderConeT(int(d))for d in guarddims]+[clarabel.SecondOrderConeT(4)for _ in rows];P=sparse.csc_matrix((n+k+1,n+k+1));c=np.zeros(n+k+1);c[-1]=1
 W=sparse.hstack([chart.Jphysical,chart.Jstation],format='csc')
 def audit_native(x):
  delta=x[:n];eta=x[n:n+k];b=float(x[-1]);rr=rseed+(J@x[:-1]).reshape(m,2);qa=np.einsum('ij,ij->i',rr,rr);av=A@np.r_[delta,b]-rhs;eps=64*np.finfo(float).eps*(1+abs(rhs)+abs(A)@abs(np.r_[delta,b]));ys=sr+sensor@x;total=chart.station_seed.copy();total[chart.selected]+=eta;wr=chart.rW+W@x[:-1];gd=guard.audit(delta)
  return dict(original_A_beyond_IEEE=int(np.count_nonzero(av>eps)),strict_original_box_and_finite_station_cell=bool(np.all(x>=lo)and np.all(x<=hi)),total_deltaS_finite_cell_and_curvature=bool(np.all(total[chart.selected]>=allclo[chart.selected])and np.all(total[chart.selected]<=allchi[chart.selected])and np.all(abs(total)<=radius)),all_active_sensor_domains=bool(np.all(ys>=0)and np.all(ys<=1079)),all_source_affine_norms_within_b=bool(np.all(qa<=b)),violating_source_count=int(np.count_nonzero(qa>b)),maximum_affine_q=float(qa.max()),worst_source_row=int(rows[np.argmax(qa)]),original_W_half_cost=float(.5*wr@wr),print_determinant_guard=gd)
 sparse.save_npz(O/'primary_A.npz',AA);np.savez_compressed(O/'primary_arrays.npz',rhs=bb,objective=c,units=units,lower_native=lo,upper_native=hi,cone_dimensions=np.array([int(d.dim)for d in cones]),budget_source_row_indices=np.flatnonzero(budget_col),budget_column=budget_col,selected_source_rows=chart.selected)
 qa=None
 if QA_proposal:
  zp=Path(QA_proposal).resolve();z=np.load(zp);assert np.array_equal(z['parent_values'],chart.parent_values)and np.array_equal(z['scales'],chart.scale)and np.array_equal(z['station_source_rows'],chart.selected);xn=np.r_[z['step_scaled'],z['station_eta'],z['canonical_primal'][-1]];reference_report=read(zp.parent/'typed_proposal_report.json');assert reference_report['proposal']['sha256']==sha(zp);rcP=verify(reference_report['group_W_chart']);rc=read(rcP);rd=rc['arrays']['rW'];expected_rW=np.fromfile(verify(rd,rcP.parent),rd['dtype']).reshape(rd['shape']);assert chart.rW.tobytes()==expected_rW.tobytes();qa=dict(proposal=ref(zp),reference_report=ref(zp.parent/'typed_proposal_report.json'),reference_chart=ref(rcP),all_original_native_constraints=audit_native(xn),same_seed_rW_IEEE=chart.rW.tobytes()==expected_rW.tobytes(),W_from_current_operators=audit_native(xn)['original_W_half_cost'],unchanged_unreleased_sources_keep_original_witness=True)
 assembly=dict(schema='rinkfit.joint_full_lawful_S_all_source_PhaseI_convex_b_assembly.v1',input=ref(IP),source=ref(__file__),authority=chart.authority,linearization=ref(Q),transport=ref(T),group_W_chart=ref(O/'group_W_chart/manifest.json'),station_TR_application=trinfo,selection=ref(SEL),original_A=ref(paths['original_A']),original_rhs_bounds=ref(paths['rhs_bounds']),original_A_rows=A.shape[0],original_A_nnz=A.nnz,budget_nonzero_rows=int(np.count_nonzero(budget_col)),budget_negative_rows=int(np.count_nonzero(budget_col<0)),budget_positive_rows=int(np.count_nonzero(budget_col>0)),original_A_kept_exact_before_invertible_units=True,b_coordinate='squared_budget_b',objective='min beta; b=qref*beta; strictly monotone-equivalent to min sqrt(b)',qref=qref,derived_radius_is_not_an_independent_solver_coordinate=True,nonconvex_t_substitution_not_used=True,physical_columns=n,station_columns=k,epigraph_columns=1,all_source_SOC4_count=m,print_SOC6_count=len(guarddims),original_W_rank=214112,all_sensor_rows=len(sr),retained_sensor_rows=len(sensor_keep),sensor_rows_box_proven_redundant=len(sensor_drop),sensor_redundancy_proof=ref(O/'sensor_redundancy_proof.npz'),rows=AA.shape[0],columns=n+k+1,A_nnz=AA.nnz,P_nnz=0,assembly_seconds=time.monotonic()-start,QA=qa,no_Solve=not do_solve,not_physical_or_KKT_certificate=True)
 (O/'assembly.json').write_text(json.dumps(assembly,indent=2),encoding='utf8');print('ASSEMBLY',json.dumps({kk:vv for kk,vv in assembly.items()if kk not in ['authority','QA']}),flush=True)
 if not do_solve:return assembly
 settings=clarabel.DefaultSettings();settings.verbose=True;settings.max_threads=1;settings.max_iter=100;settings.tol_gap_abs=1e-10;settings.tol_gap_rel=1e-10;settings.tol_feas=1e-10
 ts=time.monotonic();solver=clarabel.DefaultSolver(P,c,AA,bb,cones,settings);sol=solver.solve();seconds=time.monotonic()-ts
 if sol.x is None:raise ValueError('Primary returned no primal')
 raw=np.asarray(sol.x);xn=units*raw;rawb=float(xn[-1]);delta=xn[:n];eta=xn[n:n+k];rr=rseed+(J@xn[:-1]).reshape(m,2);qa=np.einsum('ij,ij->i',rr,rr);required=rhs-A[:,:n]@delta;neg=budget_col<0;lowerb=float(np.max(required[neg]/budget_col[neg]))if np.any(neg)else lowb;canonical=np.nextafter(max(rawb,float(qa.max()),lowerb,lowb),np.inf);xn[-1]=canonical;au=audit_native(xn);good=au['original_A_beyond_IEEE']==0 and au['strict_original_box_and_finite_station_cell']and au['total_deltaS_finite_cell_and_curvature']and au['all_active_sensor_domains']and au['all_source_affine_norms_within_b']and au['print_determinant_guard']['convex_print_guard_pass']and au['print_determinant_guard']['exact_original_print_g_pass'];beforeq=float(np.max(np.einsum('ij,ij->i',norms.residual,norms.residual)));ready=good and canonical<min(float(cp['b']),beforeq)
 np.savez_compressed(O/'raw_solver.npz',x=raw,z=np.asarray(sol.z),s=np.asarray(sol.s));total=chart.station_seed.copy();total[chart.selected]+=eta;physical=np.zeros(len(chart.parent_values));physical[chart.free]=chart.scale[chart.free]*delta;np.savez_compressed(O/'proposal.npz',physical_delta=physical,step_scaled=delta,free_ambient_indices=chart.free,parent_values=chart.parent_values,scales=chart.scale,canonical_primal=np.r_[delta,canonical],station_delta_full=total,station_eta=eta,station_source_rows=chart.selected)
 sp=O/'station_seed_delta.f64';sp.write_bytes(total.astype('<f8').tobytes());sm=dict(schema='rinkfit.station_seed_delta.v1',checkpoint=chart.authority['checkpoint'],transport=chart.authority['transport'],station_delta=dict(file=sp.name,dtype='<f8',shape=[N],sha256=sha(sp)),not_a_new_source_or_material_fact=True);SMP=O/'station_seed_manifest.json';SMP.write_text(json.dumps(sm,indent=2),encoding='utf8')
 out=dict(schema='rinkfit.joint_full_lawful_S_all_source_PhaseI_convex_b_primary.v1',ready_for_conditional_W=ready,assembly=ref(O/'assembly.json'),source=ref(__file__),authority=chart.authority,linearization=ref(Q),transport=ref(T),proposal=ref(O/'proposal.npz'),station_seed=ref(SMP),station_selection=ref(SEL),A=ref(paths['original_A']),rhs_bounds=ref(paths['rhs_bounds']),station_TR_manifest=spec['station_TR_manifest'],group_W_chart=ref(O/'group_W_chart/manifest.json'),operator_recipe_refs=dict(group_W_recipe=spec['group_W_recipe'],group_W_source=spec['group_W_source'],full_norm_source=spec['full_norm_source'],station_TR_limiter_source=spec['station_TR_limiter_source'],station_TR_manifest=spec['station_TR_manifest'],print_det_guard_source=spec['print_det_guard_source'],entry=ref(__file__)),primary_epigraph_coordinate='squared_budget_b',raw_solver_b=rawb,canonical_primary_b=canonical,canonical_b=canonical,derived_radius_sqrt_b=float(np.sqrt(canonical)),qref=qref,canonical_epigraph_upper_lift=canonical-rawb,before_complete_b=float(cp['b']),before_affine_qmax=beforeq,raw_solver_status=str(sol.status),solver_iterations=int(sol.iterations),solve_seconds=seconds,strict_original_primal_gate=good,all_source_affine_audit=au,declared_inexact_PhaseI_working_target=ready,primary_optimality_verified=False,numeric_solver_status_not_overridden=True,PHR_multiplier_update_permitted=False,final_KKT=False,elapsed_seconds=time.monotonic()-start)
 (O/'report.json').write_text(json.dumps(out,indent=2),encoding='utf8')
 if ready:
  nextspec=dict(spec,schema='rinkfit.finite_visible_original_W_secondary_input.v1',primary_report=ref(O/'report.json'),primary_proposal=ref(O/'proposal.npz'),station_seed=ref(SMP),station_selection=ref(SEL),working_b=canonical,primary_epigraph_coordinate='squared_budget_b',primary_epigraph_value=canonical);nextspec.pop('assembly_only',None);(O/'secondary_input.json').write_text(json.dumps(nextspec,indent=2),encoding='utf8')
 print('DONE',json.dumps({kk:vv for kk,vv in out.items()if kk not in ['authority','all_source_affine_audit']}),flush=True);return out

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--solve',action='store_true');p.add_argument('--QA-proposal');a=p.parse_args();run(a.input,a.output,a.solve,a.QA_proposal)
