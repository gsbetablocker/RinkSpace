"""All current original truth/physical tangent rows, with an LP dual witness.

No programme motion graph, source J/H/g, fit, accepted state or source image
is constructed. The only new derivatives are the public small truth/domain
interfaces, on exactly the saved complete state and its original free chart.
"""
from pathlib import Path
import os, sys, time, json, hashlib, copy
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RINKSPACE_MOTION_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts/diagnostics')]
sys.dont_write_bytecode=True
from calibration.solver_runtime import install, configure_native_runtime
install();configure_native_runtime(stage='readonly_q8a15_all86_phase_I_witness',native_threads=1)
import audit_current_user_ice_truth as a
import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import linprog
from structured_programme_source_candidate.source_step_curvature_tightening import constraints_on_declared_free_chart
from calibration.solver_runtime.finite_rectangle_station_step_constraints import finite_rectangle_station_constraints,station_model_implementation_sources
from calibration.solver_runtime.startup_source_auth_reuse_runtime import install_startup_source_auth_reuse_runtime,SourceAuthenticationScope
from calibration.solver_runtime.constraint_sample_reuse_runtime import install_constraint_sample_reuse_runtime
from structured_programme_source_candidate import feasibility_priority_restoration as priority
from calibration.solver_runtime.separated_priority_LP_runtime import _full_problem_KKT
from calibration.solver_runtime.full_priority_clarabel_runtime import full_original_clarabel_linprog,_LATEST
from installation_mount_candidate.mount_graph import pivot_outside_linear_step_constraints,pivot_outside_domain
from structured_programme_source_candidate.fixed_pivot_positive_domain_fd import positive_pivot_height_columns

RUN=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003/okada_review86_q276_vertical_print_station_one_cycle_20261011')
BOOT=Path(str(RUN)+'.bootstrap')
Q='8a15a4ed0c82beb490993db9c58294af592096e8159d9652b5bd5c50217c71eb'
DIRECTORY=RUN/'accepted_states'/Q
DEST=BOOT/'q8a15_complete_truth_phase_I_readonly'
CONFIG=ROOT/'events/ankara_2026/programmes/okada_mei_sp/full_source_review86_q276_vertical_print_station_one_cycle_20261011.json'

def emit(**value):print(json.dumps(value,ensure_ascii=False),flush=True)
def ah(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=a.sha(p))
def save(p,v):a.save(p,v)

def circle_height_domain(circle,q,selected):
    # Literal public driver function, selected by AST without importing/building
    # the driver graph or any optimizer. Its source bytes are in the report.
    import ast
    path=ROOT/'calibration/solver_runtime/legacy/run_current_named66_full_programme.py'
    tree=ast.parse(path.read_text(encoding='utf8'))
    node=next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=='circle_height_step_constraints')
    namespace={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace[node.name](circle,q,selected)

def face_map(family,factor,response,limits,offset):
    if 'row_carrier_projection'in response:
        ids=np.asarray(response['row_group_index'],int)
        added=np.asarray(response['row_limit_expansion'],float)
        np.testing.assert_array_equal(added,limits[ids]);return ids+offset,added
    records=factor.full_stroke_records if family=='native'else factor.records
    sizes=[2*len(r['uv'])for r in records]if family!='confirmed'else[
        r.get('carrier_size',len(r['row']['uv'])*(2 if r['normals']is not None else 1))for r in records]
    group=np.repeat(np.arange(len(records))+offset,sizes);added=np.repeat(limits,sizes)
    lg=group.copy();la=added.copy()
    if family=='confirmed':
        cursor=0
        for r,size in zip(records,sizes):
            if not r.get('expand_lower_faces',r['normals']is not None):lg[cursor:cursor+size]=-1;la[cursor:cursor+size]=0.
            cursor+=size
    mask=np.asarray(response['retained_face_mask'],bool)
    assert mask.shape==(2*len(group),)
    return np.r_[lg,group][mask],np.r_[la,added][mask]

def signature(factors):
    result=[]
    for family,factor in factors.items():
        records=(factor.records+factor.full_stroke_records)if family=='native'else factor.records
        for i,r in enumerate(records):
            result.append(dict(family=family,record=i,arrays={k:dict(shape=list(np.shape(r[k])),sha256=ah(r[k]))
                for k in ('uv','normals','bounds','tangent_bounds','radius')if r.get(k)is not None}))
    return result

def row_description(row,physical,blocks,groups):
    if row<len(physical):return physical[row]
    for block in blocks:
        if not block['start']<=row<block['stop']:continue
        family=block['family'];local=row-block['start'];factor=block['factor'];response=block['response']
        group=int(block['local_group'][local])
        if group<0:
            count=len(response['nominal_carrier']);carrier=int(np.flatnonzero(response['retained_face_mask'])[local])%count
            sizes=[r.get('carrier_size',len(r['row']['uv'])*(2 if r['normals']is not None else 1))for r in factor.records]
            group=int(np.searchsorted(np.cumsum(sizes),carrier,side='right'))
        record=(factor.full_stroke_records if family=='native'else factor.records)[group]
        desc=dict(family=family,group_index=group,group_id=groups[block['group_offset']+group]['id'])
        if 'row_carrier_projection'in response:
            pr=response['row_carrier_projection'].getrow(local)
            desc['carrier_projection_indices']=pr.indices.tolist();desc['carrier_projection_coefficients']=pr.data.tolist()
            provenance=block['result']['report']['finite_rectangle_station_local_model']['face_provenance']
            first=len(response['row_group_index'])-len(provenance)
            if local>=first:
                prov=provenance[local-first];desc['finite_station_face']=prov
                vertex=prov['source_vertex'];desc.update(source_vertex=vertex,source_UV=record['uv'][vertex].tolist(),
                    original_normal=record['normals'][vertex].tolist(),normal_bounds=record['bounds'][vertex].tolist(),
                    tangent_bounds=record['tangent_bounds'][vertex].tolist())
            else:
                carrier=int(pr.indices[0]);sizes=[2*len(r['uv'])for r in factor.full_stroke_records]
                index=carrier-int(np.sum(sizes[:group]));vertex=index//2
                desc.update(source_vertex=vertex,axis='normal'if index%2==0 else'tangent',
                    side='lower'if pr.data[0]>0 else'upper',source_UV=record['uv'][vertex].tolist())
        else:
            mask=response['retained_face_mask'];count=len(response['nominal_carrier'])
            index=int(np.flatnonzero(mask)[local]);carrier=index%count
            sizes=[2*len(r['uv'])for r in factor.records]if family=='wide'else[
                r.get('carrier_size',len(r['row']['uv'])*(2 if r['normals']is not None else 1))for r in factor.records]
            if block['local_group'][local]<0:
                group=int(np.searchsorted(np.cumsum(sizes),carrier,side='right'));record=factor.records[group]
                desc.update(group_index=group,group_id=groups[block['group_offset']+group]['id'],protected_lower_face=True)
            relative=carrier-int(np.sum(sizes[:group]));two=family=='wide'or record['normals']is not None
            vertex=relative//2 if two else relative
            desc.update(source_vertex=vertex,axis=('normal'if relative%2==0 else'tangent')if two else'radial_or_source_operator',
                side='lower'if index<count else'upper',source_UV=(record['uv']if family=='wide'else record['row']['uv'])[vertex].tolist())
        return desc
    raise ValueError('Original face lacks provenance')

def main():
    global DEST
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=DEST)
    parser.add_argument('--no-save-domain-matrix',action='store_true')
    parser.add_argument('--source-primal-witness',action='store_true')
    args=parser.parse_args();DEST=args.output
    began=time.perf_counter();DEST.mkdir(exist_ok=True)
    reportpath=DEST/('complete_original_phase_I_primal_witness.json'if args.source_primal_witness else'complete_truth_phase_I_causal_witness.json')
    if reportpath.exists():raise FileExistsError('Preserved completed witness already exists')
    options=a.read(CONFIG)['source_options'];manifest=a.read(RUN/'input_manifest.json')
    caps=a.read(DIRECTORY/'truth_restoration_state.json');assert caps['complete_q_sha256']==Q
    install_startup_source_auth_reuse_runtime(report_directory=DEST/'source_auth')
    install_constraint_sample_reuse_runtime()
    with SourceAuthenticationScope(DEST,DEST/'source_auth'):
        state,q,identity,error=a.load_state(DIRECTORY,Path(options['--world-init']),manifest)
        assert error==0.;assert identity['q_sha256']==Q
        identity['_accepted_directory']=str(DIRECTORY)
        state.current_user_pivot_outside_constraint_contract=manifest['camera_pivot_exterior_constraints']
        board=a.NativeBoardEdgeFactor(state,options['--additional-board-source'],len(q))
        ice=a.NativeIceCurveFactor(state,options['--additional-ice-source'],len(q))
        human=a.bind_human(state,board,ice,options,len(q))
        wide=a.wide_factor(state,identity,options,len(q),options['--practical-human-location-envelope'])
        circle,material,placement=a.existing_LOW_source_owners(state,q,DIRECTORY,options,a.read(DIRECTORY/'scene_UNACCEPTED.json'))
        confirmed=a.UserConfirmedSourceTruth(state,options['--user-confirmed-source-truth-authority'],board,ice,
            parameter_count=len(q),circle_factor=circle,material_factor=material,placement_source=placement)
    factors=dict(native=human,wide=wide,confirmed=confirmed)
    base=a.values(human,wide,confirmed,q);before_sig=signature(factors);before_q=ah(q)
    with np.load(DIRECTORY/'complete_state_UNACCEPTED.npz',allow_pickle=False)as z:canonical=z['q'].copy()
    domain=a.read(RUN/'actual_initial_shared_step_domain.json')
    selected_c=np.asarray(domain['effective_shared_parameter_indices'],int)
    fixed_c=np.asarray(domain['algebraically_eliminated_declared_fixed_shared_indices'],int)
    layout=state.canonical_parameter_layout;selected=layout.canonical_to_evaluation[selected_c]
    # Include all originally declared fixed coordinates, including those which
    # were not members of requested shared. No unfixed dependency is omitted.
    direction_file=next((RUN/'actual_complete_source_directions').glob('*.npz'))
    with np.load(direction_file,allow_pickle=False)as z:fixed_c=z['fixed_parameter_indices'].copy()
    fixed=layout.canonical_to_evaluation[fixed_c]
    for family,factor in factors.items():assert np.isin(factor.active,np.r_[selected,fixed]).all(),family
    worldslots=np.asarray(state.geometry_parameter_indices,int)
    print_domain=state.geometry.current.print_vertical_domain_step_constraints(q[worldslots])
    pm=np.zeros((len(print_domain['lower']),len(selected)));positions={int(c):i for i,c in enumerate(selected)}
    for i,c in enumerate(worldslots):
        if c not in positions:
            assert not np.any(print_domain['matrix'][:,i]);continue
        pm[:,positions[int(c)]]=print_domain['matrix'][:,i]
    circle_domain=circle_height_domain(circle,q,selected)
    pivot_domain=pivot_outside_linear_step_constraints(state,q,selected)
    physical=[dict(family='physical_pivot_outside',**r)for r in pivot_domain['report']['installations']]
    physical+=[dict(family='physical_print',**r)for r in print_domain['report']['bounds']]
    physical+=[dict(family='physical_circle_height',**r)for r in circle_domain['report']['bounds']]
    matrices=[pivot_domain['matrix'],pm,circle_domain['matrix']]
    targets=[pivot_domain['lower'],print_domain['lower'],circle_domain['lower']]
    owners=[np.full(len(r),-1,int)for r in targets]
    groups=[];blocks=[];cursor=len(physical);offset=0
    emit(stage='same_saved_q_full86_small_owners_loaded',q_sha256=Q,free_columns=len(selected),seconds=time.perf_counter()-began)
    for family,factor in factors.items():
        t=time.perf_counter();limits=np.asarray(caps['families'][family]['limits'],float)
        vals=np.asarray(base['buffers'][family]);ids=caps['families'][family]['group_ids']
        assert len(ids)==len(vals)==len(limits)
        source_ids=[r['id']for r in base['primary'if family=='native'else'wide'if family=='wide'else'confirmed_details']]
        assert ids==source_ids;assert np.all(vals<=limits+1e-8)
        groups.extend(dict(family=family,index=i,id=identity,original_violation_px=float(v),original_cap_px=float(cap))
            for i,(identity,v,cap)in enumerate(zip(ids,vals,limits)))
        kw=dict(group_limits=limits,return_step_carrier_response=True)
        builder=constraints_on_declared_free_chart
        if family=='native':builder=finite_rectangle_station_constraints;kw.update(full_stroke=True,endpoint_inward_restoration=False)
        result=builder(factor,q,selected,fixed,**kw);response=result['step_carrier_response']
        owned,added=face_map(family,factor,response,limits,offset)
        matrices.append(result['matrix']);targets.append(result['lower']+added);owners.append(owned)
        local_group=owned-offset
        blocks.append(dict(family=family,factor=factor,result=result,response=response,start=cursor,
            stop=cursor+len(owned),group_offset=offset,local_group=local_group))
        cursor+=len(owned);offset+=len(limits)
        emit(stage='original_truth_family_tangent_built',family=family,groups=len(limits),rows=len(owned),seconds=time.perf_counter()-t)
    assert len(groups)==86;assert before_q==ah(q);assert before_sig==signature(factors)
    after=a.values(human,wide,confirmed,q)
    for k in base:assert a.exact(base[k],after[k]),k
    A=np.vstack(matrices);b=np.concatenate(targets);owned=np.concatenate(owners)
    del matrices
    v=np.array([g['original_violation_px']for g in groups]);coeff=np.zeros(len(b));is_owned=owned>=0
    coeff[is_owned]=np.where(v[owned[is_owned]]>1e-8,v[owned[is_owned]],0.)
    lo=np.full(len(selected),-np.inf);hi=np.full(len(selected),np.inf);box_metadata=[]
    for c,interval in domain['finite_working_intervals'].items():
        c=int(c);pos=int(np.flatnonzero(selected_c==c)[0]);lo[pos]=interval[0]-canonical[c];hi[pos]=interval[1]-canonical[c]
        entry=next((inst,r)for inst,r in manifest['camera_mount_working_constraints']['by_installation'].items()if c in r['mount_parameter_indices'])
        box_metadata.append(dict(column=c,selected_position=pos,installation=entry[0],axis='xyz'[entry[1]['mount_parameter_indices'].index(c)],
            original_interval=interval,current=float(canonical[c])))
    for c in positive_pivot_height_columns(state.blocks):
        if q[c]<=1e-6:lo[positions[int(c)]]=max(0.,lo[positions[int(c)]])
    original_arrays=dict(lower=lo,upper=hi,original_constraint_lower=b,row_group_index=owned,current_group_violations=v,
        selected_canonical_columns=selected_c,selected_evaluation_columns=selected,declared_fixed_canonical_columns=fixed_c,
        q=q,canonical_q=canonical)
    matrix=csr_matrix(A)
    matrix_identity=dict(shape=list(matrix.shape),nnz=matrix.nnz,
        data_sha256=ah(matrix.data),indices_sha256=ah(matrix.indices),indptr_sha256=ah(matrix.indptr),
        matrix_persisted=not args.no_save_domain_matrix)
    if not args.no_save_domain_matrix:
        original_arrays.update(A_data=matrix.data,A_indices=matrix.indices,A_indptr=matrix.indptr,A_shape=np.array(matrix.shape))
    input_path=DEST/'original_complete_truth_domain.npz'
    # Construct the same reversible column/row equilibration as the original
    # public phase-I contract. The source metric is explicitly None.
    il=np.flatnonzero(np.isfinite(lo));iu=np.flatnonzero(np.isfinite(hi));eye=np.eye(len(lo))
    rows=np.vstack((A,eye[il],-eye[iu]));rhs=np.r_[b,lo[il],-hi[iu]]
    expansion=np.r_[coeff,np.zeros(len(il)+len(iu))]
    cn=np.max(abs(rows),axis=0);D=np.divide(1.,cn,out=np.ones(len(cn)),where=cn>0.)
    B=rows*D[None,:];norm=np.sqrt(np.einsum('ij,ij->i',B,B)+expansion*expansion)
    active=norm!=0.;assert np.all(rhs[~active]<=0.)
    lp_A=-np.c_[B[active],expansion[active]]/norm[active,None];lp_b=-rhs[active]/norm[active]
    objective=np.r_[np.zeros(len(lo)),1.];bounds=[(None,None)]*len(lo)+[(0.,1.)]
    original_arrays.update(original_full_rhs=rhs,original_full_expansion=expansion,
        reversible_column_scale=D,reversible_row_norm=norm,exact_nonconstant_row_mask=active,
        original_row_family_block_ranges=np.array([[r['start'],r['stop']]for r in blocks]))
    if input_path.exists()and args.source_primal_witness:
        with np.load(input_path,allow_pickle=False)as old:
            assert set(old.files)==set(original_arrays)
            for key,value in original_arrays.items():np.testing.assert_array_equal(value,old[key],err_msg=key)
        old_identity=a.read(DEST/'complete_domain_assembly.json')['original_matrix_identity']
        assert old_identity==matrix_identity
    else:np.savez_compressed(input_path,**original_arrays)
    assembly=dict(schema='rinkspace.current-complete-all86-original-domain-assembly.v1',script=ref(__file__),config=ref(CONFIG),
        state=ref(DIRECTORY/'complete_state_UNACCEPTED.npz'),state_identity=ref(DIRECTORY/'state_identity.json'),
        camera=ref(identity['camera_path']),caps=ref(DIRECTORY/'truth_restoration_state.json'),
        source_manifest=ref(RUN/'input_manifest.json'),q_sha256=Q,evaluation_q_sha256=ah(q),
        all86_groups=groups,original_support_rows=len(b),physical_prefix_rows=len(physical),shared_columns=len(lo),
        original_nonconstant_rows=int(active.sum()),exact_constant_rows=int((~active).sum()),
        all_original_nonfixed_dependency_columns_present=True,source_arrays_before_after_bitexact=True,
        truth_before_after_current_q_bitexact=True,all_original_caps_unchanged=True,
        physical_domains=dict(pivot=pivot_domain['report'],print_vertical=print_domain['report'],circle=circle_domain['report'],mount_boxes=box_metadata),
        original_full_source_nonlinear_guard_not_reconstructed=True,
        exact_original_driver_shared_linear_domain_reconstructed=True,
        current_station_implementation=station_model_implementation_sources(),
        input_arrays=ref(input_path),assembly_seconds=time.perf_counter()-began,
        original_matrix_identity=matrix_identity,
        nominal_t1_delta0_original_unit_maximum_deficit=float((rhs-expansion).max()),
        nominal_t1_delta0_positive_face_count=int(np.count_nonzero(rhs-expansion>0.)),
        nominal_t1_delta0_positive_deficits_above_original_guard_comparison=int(np.count_nonzero(rhs-expansion>1e-8)),
        full_source_J_H_g_or_motion_graph_constructed=False,new_fit_or_accepted_q=False,source_images_written=0)
    assembly_path=DEST/('original_primal_gate_complete_domain_reproduction.json'if args.source_primal_witness else'complete_domain_assembly.json')
    save(assembly_path,assembly)
    emit(stage='all86_and_original_physical_phase_I_ready',rows=len(b),original_full_LP_rows=len(rhs),columns=len(objective),input_arrays=str(input_path))
    call=dict(A_ub=lp_A,b_ub=lp_b,bounds=bounds,method='highs-ipm',
        options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    if args.source_primal_witness:
        from unittest.mock import patch
        from scipy.optimize import OptimizeResult
        from calibration.solver_runtime.priority_feasibility_witness_runtime import install_priority_feasibility_witness_runtime
        installation=install_priority_feasibility_witness_runtime()
        captured=[];fallback=[];original_lp=priority.linprog;original_conic=priority._verified_same_original_conic_phaseI
        def capture_primary(c,**kw):
            np.testing.assert_array_equal(c,objective);np.testing.assert_array_equal(kw['A_ub'],lp_A);np.testing.assert_array_equal(kw['b_ub'],lp_b)
            result=original_lp(c,**kw);captured.append(result);return result
        def capture_conic(*a,**kw):
            result=original_conic(*a,**kw);fallback.append(result);return result
        with patch.object(priority,'linprog',capture_primary),patch.object(priority,'_verified_same_original_conic_phaseI',capture_conic):
            contracted=priority.contract_original_support_bounds(lo,hi,A,b,owned,v)
        assert len(captured)==1 and len(fallback)==1
        result=fallback[0];x=np.asarray(result.x);fraction=float(x[-1]);direction=D*x[:-1]
        returned=float(contracted['common_fraction']);assert returned==.5*(1.+fraction)
        np.testing.assert_array_equal(contracted['matrix'],A);np.testing.assert_array_equal(contracted['lower'],b-returned*coeff)
        dual=np.asarray(result.original_phaseI_normalized_dual);assert dual.shape==(len(lp_b)+2,)
        lam=np.zeros(len(rhs));lam[active]=dual[:-2]/norm[active]
        lower_dual=float(dual[-2]);upper_dual=float(dual[-1]);slack=rows@direction+expansion*fraction-rhs
        midpoint_slack=rows@direction+expansion*returned-rhs
        assert 0.<=fraction<1.-64*np.finfo(float).eps
        assert np.min(slack[active]/norm[active])>=-1e-9 and np.all(slack[~active]>=0.)
        assert np.min(midpoint_slack[active]/norm[active])>=-1e-9 and np.all(midpoint_slack[~active]>=0.)
        low=np.zeros(len(x));up=np.zeros(len(x));low[-1]=lower_dual;up[-1]=-upper_dual
        diagnostic_result=OptimizeResult(x=x,ineqlin=OptimizeResult(marginals=-dual[:-2]),lower=OptimizeResult(marginals=low),upper=OptimizeResult(marginals=up))
        dual_passed,dual_KKT=_full_problem_KKT(objective,lp_A,lp_b,bounds,diagnostic_result,np.arange(len(lp_b)),1e-9,1e-9)
        actual_primary=captured[0];primary_lambda=np.zeros(len(rhs));primary_lambda[active]=-np.asarray(actual_primary.ineqlin.marginals)/norm[active]
        candidate_path=DEST/'original_phase_I_actual_primal_and_return_mapping.npz'
        np.savez_compressed(candidate_path,primary_normalized_primal=np.asarray(actual_primary.x),primary_original_unit_dual=primary_lambda,
            conic_normalized_primal=x,conic_normalized_dual=dual,original_unit_direction=direction,original_unit_row_multipliers=lam,
            auxiliary_lower_dual=np.asarray(lower_dual),auxiliary_upper_dual=np.asarray(upper_dual),
            certified_remaining_fraction=np.asarray(fraction),actual_driver_midpoint_fraction=np.asarray(returned),
            all_original_unit_slack=slack,all_original_unit_midpoint_slack=midpoint_slack,actual_contracted_support_lower=contracted['lower'],
            selected_canonical_columns=selected_c,selected_evaluation_columns=selected,reversible_column_scale=D,reversible_row_norm=norm,
            exact_nonconstant_row_mask=active)
        report=dict(schema='rinkspace.original-all86-primal-hint-backend-false-success-repair.v1',
            assembly=ref(assembly_path),unchanged_original_domain_arrays=ref(input_path),actual_primal_dual_and_return_mapping=ref(candidate_path),
            original_matrix_identity=matrix_identity,all_original_input_arrays_and_complete_matrix_hashes_reproduced=True,
            all86_original_groups_and_physical_domain_retained=True,original_rows=len(rhs),original_shared_columns=len(lo),
            primary_backend_false_success_rejected=not contracted['report']['phase_I_primary_backend_complete_primal_check']['passed'],
            original_contract_report=contracted['report'],primal_witness_runtime_installation=installation,
            complete_original_primal_witness_certified=True,certified_common_remaining_fraction=fraction,
            actual_driver_midpoint_remaining_fraction=returned,
            original_unit_all_face_minimum_slack=float(slack.min()),original_normalized_all_face_minimum_slack=float(np.min(slack[active]/norm[active])),
            original_unit_midpoint_all_face_minimum_slack=float(midpoint_slack.min()),
            independent_original_LP_dual_KKT=dual_KKT,LP_dual_optimality_certified=dual_passed,
            bottleneck_dual_attribution_claimed=False,common_fraction_is_primal_hint_not_proven_minimum=True,
            original_source_QP_full_KKT_still_required=True,original_nonlinear_source_truth_cost_gates_still_required=True,
            new_fit_or_source_J_H_g_or_accepted_q=False,source_images_written=0,seconds=time.perf_counter()-began)
        save(reportpath,report)
        emit(stage='original_complete_primal_hint_and_mapping_saved',report=str(reportpath),sha256=a.sha(reportpath),
            common_primal_hint=fraction,driver_midpoint=returned,normalized_primal_min=float(np.min(slack[active]/norm[active])),
            dual_KKT_certified=dual_passed)
        return
    attempts=[];result=linprog(objective,**call)
    passed,KKT=_full_problem_KKT(objective,lp_A,lp_b,bounds,result,np.arange(len(lp_b)),1e-9,1e-9)if result.success else(False,dict(reason=str(result.message)))
    attempts.append(dict(backend='original full HiGHS IPM',success=bool(result.success),complete_KKT=KKT))
    emit(stage='original_phase_I_IPM_witness',passed=passed,summary=KKT)
    if not passed:
        # An actually failed certificate allows the existing same-original-LP
        # backend path. No row, bound, objective, sigma or threshold changes.
        result=full_original_clarabel_linprog(linprog,objective,**call)
        trace=_LATEST.get();passed,KKT=_full_problem_KKT(objective,lp_A,lp_b,bounds,result,np.arange(len(lp_b)),1e-9,1e-9)if result.success else(False,dict(reason=str(result.message)))
        attempts.append(dict(backend='existing complete Clarabel route after actual failed certificate',trace=trace,complete_KKT=KKT))
        emit(stage='same_original_phase_I_backend_witness',passed=passed,summary=KKT)
    if not passed:
        # Preserve an actual failed numerical candidate as evidence. Never
        # present its multipliers or objective as a certified minimum/ranking.
        candidate_ref=None
        if getattr(result,'x',None)is not None:
            candidate_x=np.asarray(result.x,float)
            if candidate_x.shape==objective.shape and np.isfinite(candidate_x).all():
                candidate=dict(normalized_primal=candidate_x,original_unit_direction=D*candidate_x[:-1])
                if hasattr(result,'ineqlin')and getattr(result.ineqlin,'marginals',None)is not None:
                    candidate_y=np.asarray(result.ineqlin.marginals,float)
                    candidate_lambda=np.zeros(len(rhs));candidate_lambda[active]=-candidate_y/norm[active]
                    candidate.update(original_unit_row_multipliers=candidate_lambda,
                        original_unit_slack=rows@candidate['original_unit_direction']+expansion*candidate_x[-1]-rhs,
                        normalized_inequality_multipliers=candidate_y,
                        normalized_lower_multipliers=np.asarray(result.lower.marginals),
                        normalized_upper_multipliers=np.asarray(result.upper.marginals))
                candidate_path=DEST/'uncertified_original_LP_candidate.npz'
                np.savez_compressed(candidate_path,**candidate);candidate_ref=ref(candidate_path)
        save(reportpath,dict(assembly=ref(DEST/'complete_domain_assembly.json'),attempts=attempts,
            uncertified_actual_candidate_arrays=candidate_ref,
            complete_primal_dual_KKT_certified=False,causal_dual_attribution_not_claimed=True,new_fit_or_full_source_J=False))
        raise ValueError('Original complete phase-I witness could not be certified; preserved inputs')
    x=np.asarray(result.x);direction=D*x[:-1];fraction=float(x[-1])
    lam=np.zeros(len(rhs));lam[active]=-np.asarray(result.ineqlin.marginals)/norm[active]
    lower_dual=float(result.lower.marginals[-1]);upper_dual=float(-result.upper.marginals[-1])
    slack=rows@direction+expansion*fraction-rhs
    original_stationarity=np.r_[np.zeros(len(lo)),1.]-np.c_[rows,expansion].T@lam
    original_stationarity[-1]-=lower_dual-upper_dual
    dual_objective=float(rhs@lam-upper_dual)
    original_kkt=dict(all_original_rows_primal_minimum_slack=float(slack.min()),minimum_original_unit_multiplier=float(lam.min()),
        maximum_original_unit_stationarity=float(abs(original_stationarity).max()),
        maximum_original_unit_complementarity=float(abs(lam*slack).max()),
        original_primal_objective=fraction,original_dual_objective=dual_objective,original_primal_dual_gap=fraction-dual_objective,
        exact_constant_faces_checked=True)
    positive_rows=np.flatnonzero(lam>0.)
    # Keep every positive multiplier, including tiny values. Rankings use their
    # exact epigraph mass lambda*v, not incomparable pixel/metre multipliers.
    attribution=[]
    for row in positive_rows:
        if row<len(b):desc=row_description(int(row),physical,blocks,groups)
        elif row<len(b)+len(il):
            pos=int(il[row-len(b)]);desc=dict(family='physical_coordinate_lower',column=int(selected_c[pos]),
                detail=next((r for r in box_metadata if r['selected_position']==pos),None))
        else:
            pos=int(iu[row-len(b)-len(il)]);desc=dict(family='physical_coordinate_upper',column=int(selected_c[pos]),
                detail=next((r for r in box_metadata if r['selected_position']==pos),None))
        desc.update(global_original_row=int(row),original_unit_dual=float(lam[row]),original_unit_slack=float(slack[row]),
            original_row_target=float(rhs[row]),epigraph_fraction_mass=float(lam[row]*expansion[row]),
            complementarity=float(lam[row]*slack[row]))
        attribution.append(desc)
    attribution.sort(key=lambda r:abs(r['epigraph_fraction_mass']),reverse=True)
    summaries=[]
    for index,g in enumerate(groups):
        idx=np.flatnonzero(owned==index)
        summaries.append(dict(g,original_faces=len(idx),positive_dual_faces=int(np.count_nonzero(lam[idx]>0.)),
            epigraph_fraction_mass=float(lam[idx]@coeff[idx]),dual_objective_contribution=float(lam[idx]@b[idx]),
            minimum_original_unit_slack=float(slack[idx].min())if len(idx)else None))
    witness_arrays=dict(LP_normalized_primal=x,original_unit_direction=direction,original_unit_dual=lam,
        auxiliary_lower_dual=np.asarray(lower_dual),auxiliary_upper_dual=np.asarray(upper_dual),
        common_fraction=np.asarray(fraction),all_original_unit_slack=slack,
        all_original_unit_stationarity=original_stationarity)
    witness_path=DEST/'original_complete_truth_phase_I_primal_dual_witness.npz'
    np.savez_compressed(witness_path,**witness_arrays)
    report=dict(schema='rinkspace.current-all86-original-phase-I-primal-dual-causal-audit.v1',
        assembly=ref(DEST/'complete_domain_assembly.json'),original_domain_arrays=ref(input_path),primal_dual_witness_arrays=ref(witness_path),
        original_matrix_identity=matrix_identity,
        complete_primal_dual_KKT_certified=True,backend_attempts=attempts,normalized_KKT=KKT,original_units_KKT=original_kkt,
        optimum_common_remaining_violation_fraction=fraction,maximum_common_linear_violation_reduction_fraction=1.-fraction,
        original_driver_midpoint_fraction=.5*(1.+fraction),group_dual_attribution=summaries,
        all_positive_original_unit_dual_rows=attribution,positive_dual_rows=len(attribution),
        original_epigraph_stationarity_mass=float(expansion@lam),auxiliary_lower_dual=lower_dual,auxiliary_upper_dual=upper_dual,
        source_q_UV_sigma_limits_supports_weights_and_original_chart_changed=False,
        new_fit_full_source_J_H_g_or_motion_graph=False,source_images_written=0,
        accepted_state_or_accuracy_claimed=False,global_nonlinear_physical_incompatibility_claimed=False,
        limits=['Only the exact current original driver shared linear tangent domain is certified.',
            'LP duals attribute the common violation contraction objective, not the original source-cost gradient or source QP duals.',
            'Protected support constraints permit finite raw-centre bias within their unchanged support. Their activity does not prove the model is globally incompatible.',
            'Nonlinear all-source guard and value/cost checks would still be needed for any accepted source step. This diagnostic accepts none.'],
        seconds=time.perf_counter()-began)
    save(reportpath,report)
    emit(stage='complete_all86_phase_I_causal_witness_saved',report=str(reportpath),sha256=a.sha(reportpath),
        fraction=fraction,KKT=original_kkt,top_groups=sorted(summaries,key=lambda r:r['epigraph_fraction_mass'],reverse=True)[:8])

if __name__=='__main__':main()
