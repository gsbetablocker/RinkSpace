"""Apply source-authenticated point maps to the existing owned source bundles.

The motion operator, per-point sigma, complete exposure chart and physical
material producer remain the same. Only admitted duplicate/self/retired input
records leave active storage; their original indices stay in the map ledger.
"""
from copy import deepcopy
import hashlib,json,time
from pathlib import Path
import numpy as np
from z190_moving_pivot_source_bundle import motion_source_csr_layout
from .static_motion_source_role_retirement import StaticMotionSourceRoleRetirement

# Input compaction changes 65536-point batch membership; inverse Brown
# deliberately retains its batch-wide 2e-13 Newton stop, so a retained row can
# receive one more converged Newton update. This tight numerical identity gate
# is separate from the much wider finite-difference derivative check. It is a
# conservative verification tolerance, not a certified operation error bound.
# Exact input/CSR identity remains mandatory below.
COMPACTION_NUMERICAL_ATOL=1e-10
COMPACTION_NUMERICAL_RTOL=1e-12

def _retained_csr_numerical_identity(actual,expected):
    """Compare each complete derivative row at its own reference scale.

    Use one consistent Euclidean norm for the complete gradient and its
    perturbation: both map a Euclidean parameter step to this scalar residual.
    Brown batch stopping can perturb a source ray by a few final-place bits;
    its propagated gradient change is compared to the whole original gradient,
    not to a selected component. The absolute/relative constants, every stored
    entry and the source operator stay unchanged. This numerical equivalence
    check is not a certified operation error bound.
    """
    if (actual.shape!=expected.shape or
        not np.array_equal(actual.indptr,expected.indptr) or
        not np.array_equal(actual.indices,expected.indices)):
        raise ValueError('Actual retained source CSR row/column identity changed')
    if not np.isfinite(actual.data).all()or not np.isfinite(expected.data).all():
        return False,float('inf')
    counts=np.diff(expected.indptr);nonempty=np.flatnonzero(counts)
    if not len(nonempty):return True,0.
    starts=expected.indptr[nonempty]
    row_scale=np.hypot.reduceat(expected.data,starts)
    row_difference=np.hypot.reduceat(actual.data-expected.data,starts)
    allowance=COMPACTION_NUMERICAL_ATOL+COMPACTION_NUMERICAL_RTOL*row_scale
    ratio=row_difference/allowance
    return bool(np.all(ratio<=1.)),float(ratio.max())

def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _reference(ref):
    path=Path(ref['path']).resolve()
    if _sha(path)!=ref['sha256']:raise ValueError('Compaction source/map identity changed')
    return path
def _prefix(bundle,ref,active_key,count):
    with np.load(_reference(ref),allow_pickle=False)as z:
        active=z[active_key]
        if len(active)!=count:raise ValueError('Compaction source admission count changed')
        fields={
            'a':('source_uv_a',bundle.a),'b':('source_uv_b',bundle.b),
            'ia':('source_endpoint_index_a'if 'source_endpoint_index_a'in z.files else 'endpoint_index_a',bundle.ia),
            'ib':('source_endpoint_index_b'if 'source_endpoint_index_b'in z.files else 'endpoint_index_b',bundle.ib),
            'sigma':('original_sigma_xy'if 'original_sigma_xy'in z.files else 'source_original_sigma_xy',bundle.motion_sigma.reshape(-1,2))}
        checks={k:np.array_equal(np.asarray(value)[:count],z[name][active])for k,(name,value)in fields.items()}
        if not all(checks.values()):raise ValueError('Actual compaction input prefix differs: '+str(checks))
        return checks
def _maps(low_path,main_path):
    low=json.loads(Path(low_path).read_text(encoding='utf8'));main=json.loads(Path(main_path).read_text(encoding='utf8'))
    if low.get('schema')!='rinkspace.source-clock-exact-duplicate-retirement-compaction-candidate.v1' or main.get('schema')!='rinkspace.source-exact-pair-UV-role-sigma-duplicate-compaction-candidate.v1':
        raise ValueError('Explicit source-clock/exact-duplicate compaction maps required')
    records=dict(low=dict(map=low['candidate_arrays'],archive=low['historical_source_ledger'],active_key='active_original_global_index',prefix=low['old_active_prefix_points'],count=low['current_manifest_low_shape']))
    for item in main['classification']:
        records[item['block']]=dict(map=item['map'],archive=item['original_admission_cache'],active_key='active_original_row_index',prefix=item['before_points'],count=item['before_points'])
    return low,main,records

def _qualified_finite_duplicate_aliases(state,block,removed,preserved,old_to_active):
    """Alias only original exact duplicates with both current material certificates.

    The old map stays unchanged. Both original source records remain in the role
    certificate; just one already-retained point contributes runtime rows.
    """
    bundle=block.bundle
    if not np.array_equal(bundle.finite_mask[removed],bundle.finite_mask[preserved]):
        raise ValueError('Exact duplicate copies must have the same physical source operator')
    selected=np.flatnonzero(bundle.finite_mask[removed])
    if not len(selected):return {}
    role=getattr(state,'original_motion_far_board_roles',None)
    if role is None or block.name=='low':
        raise ValueError('Finite duplicate alias requires both admitted original main source roles')
    rows={(r['source_role_block'],int(r['original_global_bundle_index'])):r for r in role.records}
    bound=role.current_binding(state.geometry,np.asarray(state.q)[state.geometry_parameter_indices])
    bindings={(r['source_role_block'],int(r['global_bundle_index'])):i for i,r in enumerate(bound['records'])}
    runtime={int(r['global_bundle_index']):r for r in bundle.finite_records}
    aliases={}
    for k in selected:
        drop,keep=int(removed[k]),int(preserved[k]);a=rows.get((block.name,drop));b=rows.get((block.name,keep))
        if a is None or b is None:
            raise ValueError('Both finite duplicate copies need authenticated source material records')
        if (int(a.get('current_runtime_duplicate_keeper_precompaction_bundle_index',-1))!=keep
                or int(a.get('current_runtime_equivalent_compact_bundle_index',-1))!=int(old_to_active[keep])
                or int(a.get('current_compact_bundle_index',-2))!=-1):
            raise ValueError('Finite source certificate must authenticate its exact old-map representative')
        for field in ('source_pts','target_pts','source_uv','target_uv','source_sigma_px'):
            if not np.array_equal(a[field],b[field]):
                raise ValueError('Finite duplicate source clock/UV/sigma certificate differs: '+field)
        for index in (drop,keep):
            r=runtime.get(index,{})
            if (r.get('source_role_binding_id')!=role.binding_id or r.get('source_role_block')!=block.name
                    or r.get('surface_kind')!='finite_board_face'
                    or r.get('physical_surface_identity')!='source_qualified_current_far_inner_board_course'):
                raise ValueError('Finite duplicate runtime operator lacks its current public material binding')
        ia,ib=bindings[(block.name,drop)],bindings[(block.name,keep)]
        for field in ('planes','plane_jacobian','extent_world','extent_world_jacobian'):
            if not np.array_equal(bound[field][ia],bound[field][ib]):
                raise ValueError('Finite duplicate current physical value/J differs: '+field)
        if bundle.profile_weights is not None and not np.array_equal(bundle.profile_weights[drop],bundle.profile_weights[keep]):
            raise ValueError('Finite duplicate profile weights differ')
        if not np.array_equal(bundle.finite_planes[drop],bundle.finite_planes[keep]):
            raise ValueError('Finite duplicate active planes differ')
        aliases[drop]=keep
    return aliases


def compact_original_motion_inputs(state,capold,template,low_path,main_path,*,retirement_packet,verify_value_J=True):
    if getattr(state,'source_actors',None)is not None:raise ValueError('Input compaction must precede owned actor construction')
    low,main,specs=_maps(low_path,main_path)
    retirement=StaticMotionSourceRoleRetirement(retirement_packet);retirement.qualify_state(state)
    old_q=state.q.copy();old_cursor=new_cursor=0;global_keep=[];reports=[];maps={};finite_aliases_by_block={}
    original_bad_slots=(1478186,1478187);scalar_remap={}
    for block in state.blocks:
        bundle=block.bundle;n=len(bundle.a);old_rows=len(block.quadratic);spec=specs.get(block.name);finite_aliases={}
        if spec is None:
            keep=np.arange(n);old_to_active=keep.copy();checks={}
        else:
            if n!=spec['count']:raise ValueError('Actual complete motion point count changed: '+block.name)
            checks=_prefix(bundle,spec['archive'],spec['active_key'],spec['prefix'])
            with np.load(_reference(spec['map']),allow_pickle=False)as z:maps[block.name]={k:z[k].copy()for k in z.files}
            keep=np.asarray(maps[block.name]['keep_old_point_indices'],int);old_to_active=np.asarray(maps[block.name]['old_point_to_compact_point'],int)
            if old_to_active.shape!=(n,)or np.any(keep<0)or np.any(keep>=n)or np.any(np.diff(keep)<=0)or not np.array_equal(old_to_active[keep],np.arange(len(keep))):raise ValueError('Exact ordered source point bijection required')
            removed=np.flatnonzero(old_to_active<0)
            if not np.array_equal(np.setdiff1d(np.arange(n),keep),removed):raise ValueError('Compaction map complement changed')
            if block.name=='low':
                if np.any(bundle.finite_mask[removed]):raise ValueError('A physical finite source cannot be retired or compacted as a low self row')
                self_rows=maps['low']['removed_same_actual_exposure_old_point_indices']
                if not np.array_equal(np.flatnonzero(bundle.ia==bundle.ib),self_rows):raise ValueError('Actual authenticated self-exposure input differs')
                if not np.isin(np.arange(spec['prefix'],n),keep).all():raise ValueError('All new append rows must remain')
            else:
                removed_pairs=maps[block.name]['removed_exact_duplicate_old_point_indices'];preserved=maps[block.name]['preserved_equal_source_point_for_each_removed']
                sigma=bundle.motion_sigma.reshape(-1,2)
                if not np.isin(preserved,keep).all():raise ValueError('Each exact duplicate needs its retained original representative')
                for array in (bundle.a,bundle.b,bundle.ia,bundle.ib,sigma):
                    if not np.array_equal(array[removed_pairs],array[preserved]):raise ValueError('Actual duplicate pair/UV/sigma bytes differ')
                if not np.array_equal(np.sort(removed_pairs),removed):raise ValueError('Every removed main row must be an original exact duplicate')
                finite_aliases=_qualified_finite_duplicate_aliases(state,block,removed_pairs,preserved,old_to_active)
        finite_aliases_by_block[block.name]=finite_aliases
        motion_keep=(2*keep[:,None]+np.arange(2)).ravel()
        scalar_keep=np.r_[motion_keep,np.arange(2*n,old_rows)]
        native_keep=np.r_[motion_keep,np.arange(2*n,bundle.source_residual_rows)]
        global_keep.append(old_cursor+scalar_keep)
        for old in original_bad_slots:
            if old_cursor<=old<old_cursor+old_rows:
                position=np.flatnonzero(scalar_keep==old-old_cursor)
                scalar_remap[old]=new_cursor+int(position[0])if len(position)==1 else -1
        report=dict(block=block.name,before_points=n,after_points=len(keep),removed_points=n-len(keep),old_scalar_start=old_cursor,
            new_scalar_start=new_cursor,old_scalar_rows=old_rows,new_scalar_rows=len(scalar_keep),exact_prefix_fields=checks,
            all_removed_actual_finite_mask_false=not bool(finite_aliases),source_sigma_subselected_without_N_normalization=True)
        if finite_aliases:
            report['qualified_finite_exact_duplicate_aliases']=dict(
                schema='rinkspace.source-qualified-finite-exact-duplicate-runtime-aliases.v1',
                source_role_binding_sha256=state.original_motion_far_board_roles.binding_id,
                pairs=[dict(removed_original_global_bundle_index=drop,
                    duplicate_runtime_representative_original_global_bundle_index=kept,
                    active_global_bundle_index=int(old_to_active[kept]))for drop,kept in sorted(finite_aliases.items())],
                before_finite_points=int(bundle.finite_mask.sum()),
                after_finite_points=int(bundle.finite_mask[keep].sum()),
                both_source_certificates_preserved=True,
                exact_original_pair_UV_sigma_and_current_plane_value_J_checked=True,
                original_compaction_maps_unchanged=True,
                runtime_records_and_geometry_scatter_unique=True)
        if block.name=='low':report.update(actual_self_point_count=len(maps['low']['removed_same_actual_exposure_old_point_indices']),actual_self_list_bitexact_candidate=True,all_new194_append_points_kept=True)
        if len(keep)!=n:
            local=block.local(state.q)
            if verify_value_J:
                started=time.perf_counter();before_r,before_J=bundle.evaluate(local,True)
                if block.name=='low':
                    self_scalar=(2*maps['low']['removed_same_actual_exposure_old_point_indices'][:,None]+np.arange(2)).ravel()
                    assembled=before_J[self_scalar].tocsr();assembled.has_canonical_format=False;assembled.sum_duplicates()
                    report.update(same_exposure_residual_is_not_assumed_zero=True,
                        same_exposure_actual_r_max=float(abs(before_r[self_scalar]).max()),
                        same_exposure_coalesced_local_J_max=float(abs(assembled.data).max())if assembled.nnz else 0.)
                    del assembled
            for key in ('a','b','ia','ib'):setattr(bundle,key,getattr(bundle,key)[keep].copy())
            bundle.motion_sigma=bundle.motion_sigma.reshape(-1,2)[keep].copy().ravel()
            bundle.native_residual_sigma=bundle.native_residual_sigma[native_keep].copy()
            bundle.motion_rows=2*len(keep);bundle.source_residual_rows=len(bundle.native_residual_sigma)
            bundle.finite_planes=bundle.finite_planes[keep].copy();bundle.finite_mask=bundle.finite_mask[keep].copy()
            if bundle.profile_weights is not None:
                weights=np.asarray(bundle.profile_weights)
                if weights.shape!=(n,2):raise ValueError('Frozen profile weights must map each original motion point')
                bundle.profile_weights=weights[keep].copy()
            finite_positions={};retained_finite=[]
            for position,record in enumerate(bundle.finite_records):
                old=int(record['global_bundle_index']);active=int(old_to_active[old])
                if active<0:
                    if old in finite_aliases:continue
                    raise ValueError('Finite material input may not be removed without its exact qualified representative')
                finite_positions[position]=len(retained_finite)
                record['original_precompaction_global_bundle_index']=old;record['global_bundle_index']=active
                retained_finite.append(record)
            bundle.finite_records=retained_finite
            block.quadratic=block.quadratic[scalar_keep].copy()
            bundle.indices,bundle.pointer,bundle.swap_rows=motion_source_csr_layout(bundle.ia,bundle.ib,bundle.index,bundle.abs_groups)
            if len(bundle.pointer)-1!=bundle.source_residual_rows:raise ValueError('Compacted CSR row count changed')
            if hasattr(block,'course_original_board_records'):
                retained_course=[];course_keep=[];course_positions=[]
                for i,record in enumerate(block.course_original_board_records):
                    old=int(record['global_bundle_index']);active=int(old_to_active[old])
                    if active<0:
                        if old in finite_aliases:continue
                        raise ValueError('Finite course geometry cannot lose its source row')
                    record['original_precompaction_global_bundle_index']=old;record['global_bundle_index']=active
                    retained_course.append(record);course_keep.append(i)
                    course_positions.append(finite_positions[int(block.course_board_record_positions[i])])
                block.course_original_board_records=retained_course
                block.course_original_board_planes=block.course_original_board_planes[course_keep].copy()
                block.course_board_record_positions=np.asarray(course_positions,int)
                block.course_board_indices=np.asarray([r['global_bundle_index']for r in retained_course],int)
                if len(np.unique(block.course_board_indices))!=len(block.course_board_indices):
                    raise ValueError('Finite course geometry scatter must remain unique')
                block._course_current_geometry_key=block._course_camera_cache_key=None
            if hasattr(block,'actor_output_capacity'):
                block.actor_output_capacity=dict(block.actor_output_capacity,rows=len(block.quadratic),motion_rows=bundle.motion_rows,
                    nnz=int(block.actor_output_capacity['nnz'])-36*(n-len(keep))-12*len(finite_aliases))
            if verify_value_J:
                after_r,after_J=bundle.evaluate(local,True)
                expected_r=before_r[scalar_keep];expected_J=before_J[scalar_keep]
                if (after_J.shape!=expected_J.shape or
                    not np.array_equal(after_J.indptr,expected_J.indptr) or
                    not np.array_equal(after_J.indices,expected_J.indices)):
                    raise ValueError('Actual retained source CSR row/column identity changed: '+block.name)
                dr=float(abs(after_r-expected_r).max());difference=after_J-expected_J;dj=float(abs(difference.data).max())if difference.nnz else 0.
                r_pass=np.allclose(after_r,expected_r,atol=COMPACTION_NUMERICAL_ATOL,rtol=COMPACTION_NUMERICAL_RTOL)
                J_pass,J_error_fraction=_retained_csr_numerical_identity(after_J,expected_J)
                if not r_pass or not J_pass:raise ValueError('Actual same-state retained source value/J changed: '+str((block.name,dr,dj)))
                report.update(same_state_retained_local_r_max_difference=dr,same_state_retained_local_J_max_difference=dj,
                    same_state_retained_r_J_bitexact=bool(np.array_equal(after_r,expected_r)and np.array_equal(after_J.data,expected_J.data)),
                    same_state_retained_CSR_row_column_identity_bitexact=True,
                    same_state_retained_r_J_passed_tight_compaction_numerical_tolerance=True,
                    compaction_numerical_tolerance=dict(atol=COMPACTION_NUMERICAL_ATOL,rtol=COMPACTION_NUMERICAL_RTOL,
                        J_relative_scale='each_reference_CSR_row_Euclidean_norm',
                        J_difference_measure='complete_CSR_row_Euclidean_norm',J_max_error_fraction=J_error_fraction,
                        inverse_Brown_batch_Newton_stop=2e-13,certified_operation_error_bound=False),
                    retained_r_max_scale=float(abs(expected_r).max()),retained_J_max_scale=float(abs(expected_J.data).max()),
                    batch_wide_inverse_Brown_Newton_stop_preserved=True,
                    actual_retained_r_J_check_seconds=time.perf_counter()-started)
                del before_r,before_J,after_r,after_J,expected_r,expected_J,difference
        bundle.original_motion_compaction_old_to_active=old_to_active
        bundle.original_motion_compaction_active_to_old=keep
        reports.append(report);old_cursor+=old_rows;new_cursor+=len(scalar_keep)
    role=getattr(state,'original_motion_far_board_roles',None)
    if role is not None:
        lookup={b.name:b for b in state.blocks}
        for record in role.records:
            block=lookup[record['source_role_block']];active=int(block.bundle.original_motion_compaction_old_to_active[int(record['original_global_bundle_index'])])
            if active<0:
                representative=finite_aliases_by_block[block.name].get(int(record['original_global_bundle_index']))
                if representative is None:raise ValueError('Every original qualified far-board role needs a preserved exact runtime representative')
                active=int(block.bundle.original_motion_compaction_old_to_active[representative])
                record['duplicate_runtime_representative_original_global_bundle_index']=representative
            record['active_global_bundle_index']=active
        role._current_binding_cache=None
    for key in ('quadratic','scales'):
        original=np.asarray(capold[key]);saved=np.asarray(template[key])
        if not np.array_equal(original[:old_cursor],saved[:old_cursor]):raise ValueError('Frozen original programme masks/scales differ before compaction')
        keep=np.r_[*global_keep,np.arange(old_cursor,len(original))]
        capold[key]=original[keep].copy()
        saved_keep=np.r_[*global_keep,np.arange(old_cursor,len(saved))]
        template[key]=saved[saved_keep].copy()
    lowblock=next(b for b in state.blocks if b.name=='low');lowreport=next(r for r in reports if r['block']=='low')
    ids=np.asarray([r['current_bundle_motion_row']for r in retirement.records],int)
    mapped=lowblock.bundle.original_motion_compaction_old_to_active[ids]
    if np.any(mapped!=-1):raise ValueError('All explicitly retired original overlay inputs must leave active computation')
    state.original_motion_compaction_retirement_prequalification=dict(
        schema='rinkspace.actual-original-retirement-qualified-before-compaction.v1',packet_path=str(retirement.packet_path),
        packet_sha256=retirement.contract['packet_sha256'],qualified_records=deepcopy(retirement.qualified),
        original_block_binding=deepcopy(retirement.block_binding),original_point_indices=ids,mapped_active_point_indices=mapped,
        compact_block_binding=dict(family='programme:low',original_start=lowreport['new_scalar_start'],
            original_stop=lowreport['new_scalar_start']+lowreport['new_scalar_rows'],motion_point_count=lowreport['after_points']),
        candidate_map_reference=deepcopy(low['candidate_arrays']),original_qualification_passed=True)
    state.original_motion_compaction_scalar_remap=scalar_remap
    if not np.array_equal(state.q,old_q):raise ValueError('Source input compaction changed complete parameter state')
    for block in state.blocks:block.local(state.q)
    contract=dict(schema='rinkspace.authenticated-original-motion-input-compaction.v1',
        low_candidate=dict(path=str(Path(low_path).resolve()),sha256=_sha(low_path)),
        main_candidate=dict(path=str(Path(main_path).resolve()),sha256=_sha(main_path)),blocks=reports,
        removed_motion_points=sum(r['removed_points']for r in reports),removed_motion_scalar_rows=2*sum(r['removed_points']for r in reports),
        source_sigma_weights_global_density_normalization_changed=False,all_camera_view_parameter_indices_and_exposure_aliases_preserved=True,
        all16364_qualified_far_board_roles_preserved=True,source_history_and_original_packets_preserved=True,
        original_retirement_scalar_remap={str(k):v for k,v in scalar_remap.items()},fresh_complete_source_J_required=True,accuracy_accepted=False)
    state.original_motion_input_compaction_contract=contract
    return contract
