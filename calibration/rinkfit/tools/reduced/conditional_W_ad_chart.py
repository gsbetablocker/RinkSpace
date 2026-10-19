"""Whole-source original-W secondary in any saved physical/time primary chart.

Reuses the already assembled original primary cones, bounds, E and full-W
operators. Fixes only its budget coordinate, and optimizes the same physical
and station chart. No raw ambient-column assumption and no new motion prior.
"""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
from scipy import sparse
import clarabel
from rinkfit_start.ad_chart import PrintOnlyAdChart
from rinkfit_start.camera_spline_chart import time_chart_from_model
from rinkfit_start.native_scene import NativeScene
from types import SimpleNamespace


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def verify(d,base=None):
    p=Path(d.get('path',d.get('file')));p=p if p.is_absolute()else Path(base)/p
    if sha(p)!=d['sha256']:raise ValueError('Changed immutable source '+str(p))
    return p

def ar(d,base):return np.fromfile(verify(d,base),d['dtype']).reshape(d['shape'])



def original_ice_endpoint_corrections(model):
    """Restore existing finite-line endpoints to the original convex rink.

    This exact corner-ray projection adds no model parameter. The caller must
    recheck every original affine norm, box, guard and complete scene.
    """
    scene=NativeScene(model);changes=[];seen=set()
    for material in model['materials']:
        for path in material['paths']:
            if path['kind']!=0:continue
            bi=int(path['geometry_block'])
            if bi in seen:continue
            seen.add(bi);block=model['blocks'][bi];xy=np.asarray(block['values'],float).reshape(2,2)
            for endpoint,g in enumerate(scene.containment(xy)):
                if g<=0.:continue
                if block.get('constant',False):raise ValueError('Cannot restore a fixed endpoint')
                fixed=set(block.get('fixed_indices',[]));point=xy[endpoint]
                origin=np.array([.5-scene.rx,scene.width/2-scene.ry]);radii=np.array([scene.rx,scene.ry]);v=np.maximum((abs(point)-origin)/radii,0.)
                factor=np.nextafter(float(np.sum(v**scene.p)**(-1/scene.p)),0.);target=point.copy();outer=v>0.
                target[outer]=np.sign(point[outer])*(origin[outer]+radii[outer]*v[outer]*factor)
                target[outer]=np.nextafter(target[outer],np.zeros(np.count_nonzero(outer)))
                for axis in np.flatnonzero(target!=point):
                    component=2*endpoint+int(axis)
                    if component in fixed:raise ValueError('Cannot restore a fixed endpoint component')
                    changes.append(dict(material=material['label'],block=bi,component=component,before=float(point[axis]),after=float(target[axis]),delta_L=float(target[axis]-point[axis]),original_containment=float(g)))
    return changes

def baseline_engineering_threshold(cp,authority,active_sources):
    """Reuse only the audited baseline engineering threshold in a new scope.

    This is not a full evaluation of baseline W in the new source scope. The
    current checkpoint must itself declare the immutable baseline reference.
    """
    descriptor=cp.get('baseline_reference')
    if descriptor is None:raise ValueError('Current target has no frozen baseline reference')
    base_path=verify(descriptor);baseline=read(base_path);metrics=baseline.get('metrics',{})
    sem_path=verify(authority['semantic']);sem=read(sem_path)
    mask_desc=sem['active_source_mask'];mask=ar(mask_desc,sem_path.parent)
    raw=int(cp['raw_rows']);maximum_row=metrics.get('maximum_source_row');budget=float(baseline['b'])
    if mask.shape!=(raw,)or mask.dtype.kind not in 'bu'or mask.dtype.itemsize!=1 or np.any((mask!=0)&(mask!=1)):raise ValueError('Baseline threshold requires the strict selected raw-N active mask')
    if int(mask.sum())!=active_sources or sem.get('active_observations')!=active_sources or cp.get('active_rows')!=active_sources:raise ValueError('Baseline threshold belongs to another active source scope')
    if baseline.get('raw_rows')!=raw or not metrics.get('valid')or not np.isfinite(budget)or budget<0 or float(metrics['max_q'])!=budget:raise ValueError('Frozen baseline engineering threshold is not its true complete maximum')
    if not isinstance(maximum_row,int)or maximum_row<0 or maximum_row>=raw or not mask[maximum_row]:raise ValueError('Baseline maximum source row was excluded; its threshold cannot be called unchanged')
    evidence=dict(baseline_checkpoint=ref(base_path),selected_semantic=ref(sem_path),selected_active_mask=mask_desc,raw_rows=raw,active_rows=active_sources,baseline_maximum_source_row=maximum_row,baseline_maximum_source_retained=True,engineering_budget=budget,threshold_only_not_new_scope_baseline_W=True,original_raw_E_bound_by_selected_scope_authority=True)
    return budget,evidence


def run(primary,output,budget_from_full_report=None,resume_primal=None,baseline_engineering_budget=False):
    start=time.perf_counter();D=Path(primary).resolve();O=Path(output).resolve();O.mkdir(exist_ok=False)
    executed=O/'secondary_source.py';executed.write_bytes(Path(__file__).read_bytes())
    assembly=read(D/'assembly.json');authority=assembly['authority'];CP=verify(authority['checkpoint']);cp=read(CP);model=read(verify(cp['model']))
    arrays=np.load(D/'primary_arrays.npz');proposal=np.load(D/'proposal.npz');units=arrays['units'];lo=arrays['lower_native'];hi=arrays['upper_native'];rhs=arrays['rhs'];dims=arrays['cone_dimensions'];n=int(assembly['physical_columns']);k=int(assembly['station_columns']);m=int(assembly['all_active_source_SOC4']);qref=float(units[-1])
    AA=sparse.load_npz(D/'primary_A.npz').tocsc();anchor_raw_native=proposal['native_primal'][:-1].copy();anchor_native=np.minimum(np.maximum(anchor_raw_native,lo[:-1]),hi[:-1]);anchor=anchor_native/units[:-1];anchor_clamp=float(np.max(abs(anchor_native-anchor_raw_native)))
    first=int(dims[0]);uvrows=(AA.shape[0]-4*m+4*np.arange(m)[:,None]+np.array([1,2])).ravel()
    K=(-np.sqrt(qref)/2)*AA[uvrows,:-1];r0=(np.sqrt(qref)/2*rhs[uvrows]).reshape(m,2)
    def source_q(z):
        r=r0+(K@z).reshape(m,2);return np.einsum('ij,ij->i',r,r)
    qa=source_q(anchor);bc=np.asarray(AA[:first,-1].toarray()).ravel();lin=np.asarray(AA[:first,:-1]@anchor).ravel();negative=bc<0
    required_beta=float(np.max((lin-rhs[:first])[negative]/(-bc[negative]),initial=0.))
    primary_budget=np.nextafter(max(float(qa.max()),float(proposal['native_primal'][-1]),qref*required_beta),np.inf)
    budget=primary_budget;budget_reference=None;baseline_budget_evidence=None
    if baseline_engineering_budget and budget_from_full_report is not None:raise ValueError('Select one explicitly identified W budget authority')
    if budget_from_full_report is not None:
        budget_path=Path(budget_from_full_report).resolve();budget_report=read(budget_path)
        if not budget_report.get('source_W_E_unchanged')or budget_report.get('active_rows')!=m or budget_report.get('semantic',{}).get('sha256')!=authority['semantic']['sha256']or not budget_report['best']['valid']:raise ValueError('Different-budget reference is not a complete unchanged-source actual evaluation')
        budget=float(budget_report['best']['max_q'])
        if not np.isfinite(budget)or budget<primary_budget:raise ValueError('Chosen existing full-state budget cannot contain this primary anchor')
        budget_reference=ref(budget_path)
    if baseline_engineering_budget:
        budget,baseline_budget_evidence=baseline_engineering_threshold(cp,authority,m)
        if budget<primary_budget:raise ValueError('Baseline engineering threshold cannot contain this primary anchor')
        budget_reference=baseline_budget_evidence['baseline_checkpoint']
    fixed_rhs=rhs-np.asarray(AA[:,-1].toarray()).ravel()*(budget/qref);A=AA[:,:-1].tocsc()
    # The SAME W and same native station columns; only their coordinate lift
    # differs from the legacy fine storage parameterization.
    JW=sparse.load_npz(D/'full_original_W_Jphysical.npz');JS=sparse.load_npz(D/'full_original_W_Jstation.npz');rW=np.load(D/'original_W_residual.npy')
    W=sparse.hstack([JW,JS],format='csc');KW=(W@sparse.diags(units[:-1])).tocsc()
    aw=rW+KW@anchor;factor=max(1.,float(.5*aw@aw));H=sparse.triu((KW.T@KW)/factor,format='csc');linear=np.asarray(KW.T@aw).ravel()/factor
    # Exact affine coordinate translation about the known primary witness.
    # It changes neither the original cones, original W nor fixed budget.
    centered_rhs=fixed_rhs-A@anchor
    cones=[clarabel.NonnegativeConeT(int(dims[0]))]+[clarabel.SecondOrderConeT(int(d))for d in dims[1:]]
    settings=clarabel.DefaultSettings();settings.verbose=True;settings.max_threads=1;settings.tol_gap_abs=1e-10;settings.tol_gap_rel=1e-10;settings.tol_feas=1e-10
    print('W_ASSEMBLY',json.dumps(dict(physical_columns=n,station_columns=k,all_source_norms=m,W_rank=len(rW),rows=A.shape[0],columns=A.shape[1],H_nnz=H.nnz,budget=budget)),flush=True)
    resume_reference=None
    if resume_primal is None:
        solve_start=time.perf_counter();solver=clarabel.DefaultSolver(H,linear,A,centered_rhs,cones,settings);solution=solver.solve();seconds=time.perf_counter()-solve_start
        if solution.x is None:raise ValueError('No finite conditional-W primal returned')
        solver_raw=anchor+np.asarray(solution.x)
    else:
        resume_path=Path(resume_primal).resolve();replay=np.load(resume_path)
        if ('primary_assembly_sha256'in replay and str(replay['primary_assembly_sha256'])!=sha(D/'assembly.json'))or float(replay['budget'])!=budget or not np.array_equal(replay['units'],units)or not np.array_equal(replay['primary_anchor'],anchor):raise ValueError('Finite primal replay differs from this original state, layout or budget')
        solver_raw=replay['solver_raw'].copy();seconds=0.;solution=SimpleNamespace(status='PreviouslySolvedFinitePrimalReplay',iterations=0);resume_reference=ref(resume_path)
    raw_native=units[:-1]*solver_raw;raw=np.clip(raw_native,lo[:-1],hi[:-1])/units[:-1];raw_numeric_box_repair_max=float(np.max(abs(units[:-1]*raw-raw_native)));nonnegative=first;guard_count=int(assembly['original_print_SOC6']);guard_first=first;guard_last=first+6*guard_count
    def audit(z):
        s=fixed_rhs-A@z;tol=64*np.finfo(float).eps*(1+abs(fixed_rhs[:first])+abs(A[:first])@abs(z));q=source_q(z);native=units[:-1]*z
        gs=s[guard_first:guard_last].reshape(guard_count,6);guard=bool(np.all(gs[:,0]>=np.linalg.norm(gs[:,1:],axis=1)))
        return dict(q=q,all_source_norms=bool(np.all(q<=budget)),linear_beyond_IEEE=int(np.count_nonzero(s[:first]<-tol)),
            strict_coordinate_and_station_box=bool(np.all(native>=lo[:-1])and np.all(native<=hi[:-1])),print_SOC_guards=guard,
            original_W_half_cost=float(.5*np.sum((rW+KW@z)**2)))
    aa=audit(anchor);rr=audit(raw)
    def good(a):return a['all_source_norms']and a['linear_beyond_IEEE']==0 and a['strict_coordinate_and_station_box']and a['print_SOC_guards']
    if not good(aa):raise ValueError('Same-budget primary anchor did not pass every original affine source/box/guard')
    fraction=1.
    if not good(rr):
        low=0.;high=1.
        while np.nextafter(low,np.inf)<high:
            mid=(low+high)/2
            if good(audit(anchor+mid*(raw-anchor))):low=mid
            else:high=mid
        fraction=low
    restored=anchor+fraction*(raw-anchor);final=audit(restored)
    if not good(final):raise ValueError('Strict conditional-W primal restoration failed')
    physical_map=sparse.load_npz(D/'full_physical_from_solver_scaled.npz');chart=PrintOnlyAdChart(model);physical0=chart.pack()
    time_info=assembly.get('time_subspace');time_chart=time_chart_from_model(model)if time_info is not None else None
    np.savez_compressed(O/'solver_before_domain.npz',solver_raw=solver_raw,affine_restored_solver=restored,primary_anchor=anchor,units=units,budget=budget,affine_restoration_fraction=fraction,primary_assembly_sha256=np.asarray(sha(D/'assembly.json')))
    def decode(z):
        nv=units[:-1]*z;dv=np.asarray(physical_map@nv[:n]).ravel();cm,dm=chart.unpack(physical0+dv,return_domain=True);member=None
        if time_chart is not None:_,member=time_chart.recover(cm)
        return cm,dm,member,nv,dv
    nonlinear_fraction=1.;domain_failure=None;endpoint_repair=None
    try:candidate,domain,membership,native,delta=decode(restored)
    except ValueError as exc:
        domain_failure=str(exc);edge=restored.copy();repaired=False
        try:
            unvalidated=chart.unpack(physical0+physical_map@(units[:-1]*restored)[:n],validate=False);changes=original_ice_endpoint_corrections(unvalidated);corrected=restored.copy()
            if not changes:raise ValueError('No independent ice endpoint accounts for this failure')
            for change in changes:
                physical_row=chart.column[int(chart.offset[change['block']]+change['component'])];mapping=physical_map.getrow(physical_row)
                if mapping.nnz!=1:raise ValueError('Endpoint correction is not one existing independent parameter')
                j=int(mapping.indices[0]);corrected[j]+=change['delta_L']/(mapping.data[0]*units[j])
            correction_audit=audit(corrected)
            if not good(correction_audit):raise ValueError('Endpoint correction violates an original affine norm, box or guard')
            candidate,domain,membership,native,delta=decode(corrected);restored=corrected;final=correction_audit;nonlinear_fraction=None;repaired=True;endpoint_repair=dict(passed=True,changes=changes,no_new_DOF=True,original_full_domain_and_all_affine_constraints_rechecked=True)
        except ValueError as repair_error:endpoint_repair=dict(passed=False,reason=str(repair_error))
        if not repaired:
            decode(anchor);low=0.;high=1.
            while np.nextafter(low,np.inf)<high:
                mid=(low+high)/2;trial=anchor+mid*(edge-anchor)
                try:
                    if not good(audit(trial)):raise ValueError('Original affine domain after floating point interpolation')
                    decode(trial);low=mid
                except ValueError:high=mid
            nonlinear_fraction=low;restored=anchor+low*(edge-anchor);final=audit(restored)
            if not good(final):raise ValueError('Whole-scene restoration lost the original affine constraints')
            candidate,domain,membership,native,delta=decode(restored)
    mp=O/'model.json';mp.write_text(json.dumps(candidate,ensure_ascii=False,indent=2),encoding='utf8')
    selected=np.load(D/'station_source_rows.npy').astype(int);station=ar(cp['row_state_files']['station'],CP.parent)+np.load(D/'station_seed_full.npy');station[selected]+=native[n:]
    sp=O/'station.f64';station.astype('<f8').tofile(sp)
    auxiliary={key:dict(path=str(verify(desc,CP.parent)),sha256=desc['sha256'],dtype=desc['dtype'],shape=desc['shape'])for key,desc in cp['row_state_files'].items()};auxiliary['station']=dict(path=str(sp),sha256=sha(sp),dtype='<f8',shape=[len(station)])
    manifest=dict(schema='rinkfit.physical_print_chart_trial_seed.v1',parent_checkpoint=ref(CP),model=ref(mp),row_state_files=auxiliary,fraction=1.,
        requires_fresh_true_RS_profile_and_all_station_evaluation=True,accepted_nonlinear_candidate=False,coarse_time_membership=membership)
    cm=O/'candidate_manifest.json';cm.write_text(json.dumps(manifest,indent=2),encoding='utf8')
    np.savez_compressed(O/'proposal.npz',solver_raw=raw,restored_solver=restored,native_coordinates=native,reduced_physical_delta=delta,source_q=final['q'],station_rows=selected)
    summary=lambda a:{key:value for key,value in a.items()if key!='q'}
    report=dict(schema='rinkfit.generic_physical_time_chart_original_W_secondary.v1',source=ref(executed),primary=ref(D/'report.json'),primary_assembly=ref(D/'assembly.json'),candidate_manifest=ref(cm),
        physical_columns=n,station_columns=k,all_active_sources=m,original_W_rank=len(rW),unchanged_primary_working_budget=budget if budget_reference is None else None,fixed_working_budget=budget,primary_reference_working_budget=primary_budget,budget_mode='baseline_engineering_threshold_budget'if baseline_engineering_budget else('same_primary_budget'if budget_reference is None else 'different_existing_full_state_budget'),budget_reference=budget_reference,baseline_budget_evidence=baseline_budget_evidence,changed_budget_is_not_same_budget_secondary=budget_reference is not None,raw_solver_status=str(solution.status),iterations=int(solution.iterations),solve_seconds=seconds,
        parent_W_half_cost=float(.5*rW@rW),primary_W_half_cost=aa['original_W_half_cost'],raw_W_half_cost=rr['original_W_half_cost'],raw_numeric_box_repair_max=raw_numeric_box_repair_max,raw_primal_audit=summary(rr),anchor_primal_audit=summary(aa),restored_W_half_cost=final['original_W_half_cost'],
        W_change_from_parent=final['original_W_half_cost']-float(.5*rW@rW),W_change_from_primary=final['original_W_half_cost']-aa['original_W_half_cost'],
        primary_anchor_numeric_box_repair_max=anchor_clamp,exact_solver_coordinate_translation='z=primary_anchor+y; original constraints and W unchanged',exact_budget_restoration_fraction=fraction,whole_scene_anchor_segment_restoration_fraction=nonlinear_fraction,full_affine_trial_scene_failure=domain_failure,existing_endpoint_domain_repair=endpoint_repair,resumed_finite_primal=resume_reference,maximum_restored_q=float(final['q'].max()),all_source_primal_audit=summary(final),coarse_time_membership=membership,exact_original_domain=domain,
        accepted_nonlinear_candidate=False,no_new_motion_prior=True,elapsed_seconds=time.perf_counter()-start)
    (O/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print('W_DONE',json.dumps({key:report[key]for key in ['physical_columns','station_columns','maximum_restored_q','fixed_working_budget','unchanged_primary_working_budget','restored_W_half_cost','W_change_from_parent','exact_budget_restoration_fraction','elapsed_seconds']}),flush=True);return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--primary',required=True);p.add_argument('--output',required=True);p.add_argument('--budget-from-full-report');p.add_argument('--resume-primal');p.add_argument('--baseline-engineering-budget',action='store_true');a=p.parse_args();run(a.primary,a.output,a.budget_from_full_report,a.resume_primal,a.baseline_engineering_budget)
