"""Cheap source-only feasibility diagnosis using the existing joint solver.

No original motion graph or old full J is loaded. A resulting complete q is
an initializer, not a fit accepted against the retired/quarantined old target.
Dense human stroke vertices are guards, never independent soft FIT weights.
"""
from pathlib import Path
import argparse,hashlib,json,os,pickle,sys,time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(Path(__file__).resolve().parent)]
from calibration.solver_runtime import install
install()
import numpy as np
from prepared_cache_helpers import install_mappingproxy_pickle,install_geometry_unpickle
from verify_current_native_primitives_20261007 import state_camera_adapter
from structured_programme_source_candidate.native_board_edge_source import NativeBoardEdgeFactor
from structured_programme_source_candidate.native_ice_curve_source import NativeIceCurveFactor
from structured_programme_source_candidate.human_stroke_truth import HumanStrokeTruth,WideStrokeTruth,UserConfirmedSourceTruth,direct_manual_centre_factors,direct_manual_centre_block
from structured_programme_source_candidate.fixed_pivot_positive_domain_fd import positive_pivot_coordinate_derivative
from z190_streaming_source_normal import SourceRowBlock
from structured_programme_source_candidate.current_wide_source import CurrentWideSource
from structured_programme_source_candidate.native_camera_local import native_camera_local
from wide_anchor_candidate.full_brown_projection import project_central_valid_full_brown
from structured_programme_source_candidate.current_print_roundoff_recovery import recover_ad17_print_roundoff
from installation_mount_candidate.mount_graph import pivot_outside_domain,pivot_outside_linear_step_constraints
from z190_joint_source_solver import solve_joint_source


def sha(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def single_view_parameter_scope(state,world,material,actual_pts):
    """World/material may vary; only the observed view may move its camera."""
    block=state.bypts[int(actual_pts)]
    camera=np.r_[block.xyz_slots,block.optics_chart.indices,block.optics_chart.mount_indices,
                 block.start+4*block.bundle.index[int(actual_pts)]+np.arange(4)].astype(int)
    return np.unique(np.r_[world,material,camera]).astype(int)


def retract_selected_pivots(trial,outside,selected):
    """Preserve fixed cameras when a changed course invalidates their pivot."""
    value=trial.copy();retractions=[];selected=set(np.asarray(selected,int).tolist())
    for row in outside['installations']:
        if row['valid']:continue
        slots=np.asarray(row['pivot_parameter_indices'],int)
        if not all(int(col)in selected for col in slots[:2]):continue
        before=value[slots[:2]].copy();rounding=float(row['numerical_boundary_roundoff_working_m'])
        value[slots[:2]]=np.asarray(row['nearest_course_xy_working_m'])+2*rounding*np.asarray(row['outward_normal_xy'])
        retractions.append(dict(installation_id=row['installation_id'],previous_pivot_xy_working_m=before.tolist(),
            actual_allowed_pivot_xy_working_m=value[slots[:2]].tolist(),actual_outside_margin_before_working_m=row['outward_margin_working_m'],
            only_existing_numeric_boundary_roundoff_used=True,pivot_Z_and_camera_RF_unchanged=True))
    return value,retractions


def fixed_camera_pivot_step_constraints(state,q,selected):
    """Project original whole-installation inequalities with fixed deltas zero."""
    contract=getattr(state,'current_user_pivot_outside_constraint_contract',None)
    fixed_xy=[]if contract is None else[col for entry in contract['by_installation'].values()for col in entry['pivot_parameter_indices'][:2]]
    expanded=np.unique(np.r_[np.asarray(selected,int),fixed_xy]).astype(int)
    original=pivot_outside_linear_step_constraints(state,q,expanded)
    position={int(col):i for i,col in enumerate(expanded)}
    columns=np.asarray([position[int(col)]for col in selected],int)
    return dict(matrix=original['matrix'][:,columns],lower=original['lower'],report=dict(original['report'],
        shared_parameter_indices=np.asarray(selected,int).tolist(),fixed_pivot_columns=np.setdiff1d(expanded,selected).tolist(),
        fixed_pivot_increments_exactly_zero=True,all_installation_world_constraints_preserved=True))


def main():
    ap=argparse.ArgumentParser()
    for key in ('state','camera-identity','owner-runtime-report','board-packet','ice-packet','human-authority','full-stroke-guard','output'):
        ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--wide-human-authority',type=Path)
    ap.add_argument('--physical-course-scope-source',type=Path)
    ap.add_argument('--stroke-cap-support',type=Path)
    ap.add_argument('--write-practical-location-envelope',type=Path,help='Freeze actual initializer deviations for full-source continuation; do not refine drawing raster pixels')
    ap.add_argument('--practical-location-envelope',type=Path,help='Inspect an explicit engineering manual-location policy without source changes or raster refinement')
    ap.add_argument('--phase-I-initializer',type=Path)
    ap.add_argument('--local-physical-initializer',action='store_true',help='Use only the causal common course/paired print/right goal fields and existing low calibration/observed RF chart; preserve the other original layout fields')
    ap.add_argument('--confirmed-source-authority',type=Path)
    ap.add_argument('--single-frame-pts',type=int,help='All existing independent world/material fields and the true physical camera chart participate for this source frame')
    ap.add_argument('--manual-location-tolerance-px',type=float,default=0.,help='Explicit practical hand-drawing error in native pixels, in addition to saved raster footprint; never claimed as optical precision')
    ap.add_argument('--fit-single-frame-admitted-sources',action='store_true',help='Include actual same-frame FIT ice/advertisement observations alongside highest-priority human contours')
    ap.add_argument('--native-board-track-packet',type=Path,help='Existing source-conditioned actual printed-component anchor observations for this frame; not direct hand drawings')
    ap.add_argument('--fit-direct-human-centres',action='store_true',help='Secondary actual finite-curve centre residual on sparse hand-drawing rows, inside the retained practical human guard; correlated traces use their mean, not dense vertex counts')
    ap.add_argument('--inspect-only',action='store_true')
    ap.add_argument('--inspect-facet-coverage',action='store_true')
    args=ap.parse_args();started=time.perf_counter()
    if args.output.exists():raise ValueError('Preserve prior feasibility evidence; output must be new')
    arrays=dict(np.load(args.state,allow_pickle=False));parent_q=arrays['q'].copy()
    identity=json.loads(args.camera_identity.read_text(encoding='utf8'))
    parent_sha=hashlib.sha256(parent_q.tobytes()).hexdigest()
    if identity['q_sha256']!=parent_sha or sha(identity['camera_path'])!=identity['camera_sha256']:
        raise ValueError('Stopped complete q and immutable camera archive identity differ')
    cameras=dict(np.load(identity['camera_path'],allow_pickle=False))
    owner=json.loads(args.owner_runtime_report.read_text(encoding='utf8'))['material_owner']
    if sha(owner['path'])!=owner['sha256']:raise ValueError('Existing actual material owner identity changed')
    install_mappingproxy_pickle();install_geometry_unpickle();provider=pickle.loads(Path(owner['path']).read_bytes())
    state,camera_error=state_camera_adapter(arrays,cameras,provider)
    if args.physical_course_scope_source is not None:state.current_user_physical_course_scope_source=args.physical_course_scope_source
    q=state.evaluation_q(parent_q).copy();layout=state.canonical_parameter_layout
    if args.phase_I_initializer is not None:
        with np.load(args.phase_I_initializer,allow_pickle=False)as previous:
            if set(previous.files)!=set(arrays):raise ValueError('Phase-I continuation must preserve every complete parent array')
            for key,value in arrays.items():
                if key not in ('q','identity_json')and not np.array_equal(value,previous[key]):
                    raise ValueError('Phase-I changed the original auxiliary/profile: '+key)
            q=state.evaluation_q(previous['q']).copy()
    board=NativeBoardEdgeFactor(state,args.board_packet,len(q));ice=NativeIceCurveFactor(state,args.ice_packet,len(q))
    truth=HumanStrokeTruth(state,args.human_authority,board,ice);truth.bind_full_stroke_guard(args.full_stroke_guard)
    if args.stroke_cap_support is not None:truth.bind_stroke_cap_support(args.stroke_cap_support)
    wide_truth=None;wide_initial_camera_error=None;wide_initial_TOP_error=None
    if args.wide_human_authority is not None:
        scene_path=args.camera_identity.parent/'scene_UNACCEPTED.json'
        if sha(scene_path)!=identity['scene_sha256']:raise ValueError('Original actual scene export identity changed')
        scene=json.loads(scene_path.read_text(encoding='utf8'))
        exported=[r['camera25']for r in scene['frame_cameras']if r.get('source_pts')==56218880]
        if len(exported)!=1:raise ValueError('Unique existing private wide camera export required')
        authority=json.loads(args.wide_human_authority.read_text(encoding='utf8'))
        trace=json.loads(Path(authority['trace_packet_reference']['path']).read_text(encoding='utf8'))
        factor=CurrentWideSource.__new__(CurrentWideSource)
        factor.geometry=provider.current;factor.qualified_scene=provider.qualified_scene
        factor.geometry_parameter_indices=state.geometry_parameter_indices
        factor.observations=[];factor.ad_packet={'rows':[]};factor.template_camera25=np.asarray(exported[0],float)
        factor.trace_packet=trace;factor.trace_uv=[np.asarray(r['observed_centres_source_px'],float)for r in trace['contours']]
        camera_indices=layout.canonical_to_evaluation[np.asarray(arrays['aux__current_wide_camera_parameter_indices'],int)]
        wide_initial_camera_error=float(abs(factor.camera(state.evaluation_q(parent_q)[camera_indices])-exported[0]).max())
        top=factor._curve_worlds(state.evaluation_q(parent_q)[state.geometry_parameter_indices])[1]
        expected=np.asarray(scene['board_surface']['top_edge_xyz'],float)
        if not np.array_equal(expected[0],expected[-1]):expected=np.vstack((expected,expected[0]))
        if expected.shape!=top.shape:raise ValueError('Existing wide TOP producer differs from production scene export')
        wide_initial_TOP_error=float(abs(top-expected).max())
        if wide_initial_camera_error!=0. or wide_initial_TOP_error>5e-14:
            raise ValueError('Existing wide hard operator must reproduce saved camera/TOP values')
        wide_truth=WideStrokeTruth(factor,args.wide_human_authority,len(q),camera_indices)
    if args.manual_location_tolerance_px:
        tolerance=float(args.manual_location_tolerance_px)
        if not np.isfinite(tolerance)or tolerance<0.:raise ValueError('Finite nonnegative practical manual location tolerance required')
        for item in truth.records+truth.full_stroke_records+([]if wide_truth is None else wide_truth.records):
            if item.get('row',{}).get('practical_manual_location_uncertainty_native_px',0.)>0.:
                # This appended source already includes hand error, pixel-cell
                # footprint and actual registration. Do not count it twice.
                continue
            item['bounds']=item['bounds']+np.array([-tolerance,tolerance])
            if 'tangent_bounds'in item:item['tangent_bounds']=item['tangent_bounds']+np.array([-tolerance,tolerance])
        truth.contract['explicit_practical_hand_drawing_tolerance_px']=tolerance
        truth.contract['already_admitted_practical_manual_support_not_expanded']=True
        if wide_truth is not None:wide_truth.contract['explicit_practical_hand_drawing_tolerance_px']=tolerance
    practical_carrier_check=None
    if args.practical_location_envelope is not None:
        if args.stroke_cap_support is None:raise ValueError('Actual drawing cap source support required for engineering location policy')
        before=truth.queries(q)+truth.queries(q,full_stroke=True)+([]if wide_truth is None else wide_truth.queries(q))
        truth.bind_practical_location_envelope(args.practical_location_envelope)
        if wide_truth is not None:wide_truth.bind_practical_location_envelope(args.practical_location_envelope)
        after=truth.queries(q)+truth.queries(q,full_stroke=True)+([]if wide_truth is None else wide_truth.queries(q))
        carriers=('signed_normal_px','coverage_tangent_px','finite_scanline_intersection','finite_primitive_station',
            'nearest_finite_vector_px','nearest_finite_segment_index','nearest_finite_segment_station')
        if not all(a['id']==b['id']and all(np.array_equal(a['query'][k],b['query'][k])for k in carriers)for a,b in zip(before,after)):
            raise ValueError('Manual location policy changed a finite physical carrier or source association')
        practical_carrier_check=dict(actual_all_finite_carriers_and_associations_bitexact=True,
            policy_path=str(args.practical_location_envelope.resolve()),policy_sha256=sha(args.practical_location_envelope),
            source_geometry_camera_UV_sigma_and_physical_domain_unchanged=True,source_r_J_H_g_not_changed_by_guard=True,
            old_halfspaces_not_reused=True)
    mount=json.loads(str(arrays['aux__current_camera_mount_working_constraints_json'].item()))
    state.current_user_pivot_outside_constraint_contract=json.loads(str(arrays['aux__current_camera_pivot_exterior_constraints_json'].item()))
    world=state.geometry_parameter_indices
    pivots=np.asarray(arrays['aux__installation_xyz_slots'],int).reshape(-1,3)
    if args.single_frame_pts is not None:
        if args.local_physical_initializer:raise ValueError('Single-frame all-parameter initialization must not inherit subset modes')
        indices=[i for i,r in enumerate(truth.records)if r['row']['actual_pts']==args.single_frame_pts]
        if not indices:raise ValueError('No directly authenticated source observations for requested actual frame')
        truth.records=[truth.records[i]for i in indices];truth.full_stroke_records=[truth.full_stroke_records[i]for i in indices]
        block=state.bypts[args.single_frame_pts];truth.blocks={args.single_frame_pts:block}
        material_keys=('aux__current_independent_pattern_placement_parameter_indices','aux__current_board_circle_parameter_indices')
        material=np.concatenate([layout.canonical_to_evaluation[np.asarray(arrays[k],int)]for k in material_keys if k in arrays])if any(k in arrays for k in material_keys)else np.empty(0,int)
        actual_manifest_path=args.owner_runtime_report.parent/'input_manifest.json'
        actual_manifest=json.loads(actual_manifest_path.read_text(encoding='utf8'))
        circle_canonical=np.asarray(actual_manifest['independent_circle_material_parameter_indices'],int)
        if circle_canonical.shape!=(6,):raise ValueError('Actual existing six circle slots required for the complete physical chart')
        material=np.r_[material,layout.canonical_to_evaluation[circle_canonical]]
        pattern_start=int(arrays['aux__shared_pattern_start'])
        material=np.r_[material,layout.canonical_to_evaluation[pattern_start+np.arange(9)]]
        truth.active=single_view_parameter_scope(state,world,material,args.single_frame_pts)
        from structured_programme_source_candidate.fixed_pivot_positive_domain_fd import positive_pivot_height_columns
        truth.height_columns=positive_pivot_height_columns(truth.blocks.values())
        truth.row_count=2*sum(len(r['uv'])for r in truth.records);truth.quadratic=np.ones(truth.row_count,bool);truth.scales=np.ones(truth.row_count)
        truth.contract=dict(truth.contract,single_actual_source_pts=args.single_frame_pts,source_groups=len(truth.records),source_UV_queries=truth.row_count//2,
            all_existing_independent_world_parameters_participate=True,other_installation_hardware_and_other_exposure_RF4_fixed=True)
    selected=np.unique(np.r_[truth.active,pivots.ravel(),mount['shared_mount_parameter_indices'],
                            []if wide_truth is None else wide_truth.active]).astype(int)
    confirmed=None
    if args.confirmed_source_authority is not None and not args.local_physical_initializer:
        confirmed=UserConfirmedSourceTruth(state,args.confirmed_source_authority,board,ice,parameter_count=len(q))
        for item in confirmed.records:item['radius']=np.ceil(item['radius'])
        # Earlier explicitly confirmed source locations protect the restored
        # c134 result at its actual engineering precision. They are not a
        # demand to subpixel-polish the old hand/blurred source readings.
        _,_,_,prior_groups=confirmed._carriers(q)
        for item,group in zip(confirmed.records,prior_groups):
            extra=float(np.ceil(group['maximum_violation_px']))
            item['radius']=item['radius']+extra
            if item['bounds']is not None:item['bounds']=item['bounds']+np.array([-extra,extra])
        confirmed.contract['restored_c134_group_location_support']=prior_groups
        selected=np.unique(np.r_[selected,confirmed.active]).astype(int)
    if args.local_physical_initializer:
        if args.confirmed_source_authority is None:raise ValueError('The already-correct F1/Ad7 source authority must protect a local physical initializer')
        confirmed=UserConfirmedSourceTruth(state,args.confirmed_source_authority,board,ice,parameter_count=len(q))
        for item in confirmed.records:item['radius']=np.ceil(item['radius'])
        causal_world=world[np.r_[8,14,np.arange(44,50),55,56,57,59,60,61,70,71,72,73]]
        all_calibration=np.unique(np.r_[pivots.ravel(),arrays['aux__current_shared_mount_parameter_indices'],
                                      arrays['aux__current_all_installation_optics_parameter_indices']])
        low=state.bypts[230024192]
        low_calibration=np.r_[low.xyz_slots,low.optics_chart.mount_indices,low.optics_chart.indices]
        camera=np.setdiff1d(truth.active,np.r_[world,all_calibration])
        selected=np.unique(np.r_[causal_world,camera,low_calibration,
                                []if wide_truth is None else wide_truth.camera_indices]).astype(int)
    if args.single_frame_pts is not None:
        # Other-frame guards protect the shared scene with their cameras fixed.
        # Their dependencies must not become single-view optimization freedoms.
        selected=single_view_parameter_scope(state,world,material,args.single_frame_pts)
    fixed=np.setdiff1d(np.arange(len(q)),selected)
    scope_reference=q.copy()
    bounds={int(i):(float(low),float(high))for i,low,high in zip(mount['shared_mount_parameter_indices'],mount['delta_lower_m'],mount['delta_upper_m'])if int(i)in selected}
    frame_factors=[]
    if args.fit_direct_human_centres:
        if args.single_frame_pts is None:raise ValueError('Explicit actual frame required for hand-drawing centre correction')
        frame_factors.extend(direct_manual_centre_factors(state,truth,args.board_packet,args.ice_packet,len(q),actual_pts=args.single_frame_pts))
    if args.fit_single_frame_admitted_sources:
        if args.single_frame_pts is None:raise ValueError('Same-frame source fitting requires explicit actual source PTS')
        human_ids={r['row']['id']for r in truth.records}
        for factor in (board,ice):
            chosen=[];scalar=[];offset=0
            for row in factor.rows:
                stop=offset+2*len(row['uv'])
                if row['actual_pts']==args.single_frame_pts and row['id']not in human_ids:
                    chosen.append(row);scalar.extend(range(offset,stop))
                offset=stop
            if not chosen:continue
            factor.rows=chosen;factor.held=[];factor.blocks={args.single_frame_pts:state.bypts[args.single_frame_pts]}
            factor.scales=factor.scales[scalar];factor.quadratic=factor.quadratic[scalar];factor.row_count=len(scalar);factor._cache.clear()
            if factor is board:
                factor.active=truth.active.copy();factor.active_world=world.copy()
                factor.contract['fit_ids']=[r['id']for r in chosen];factor.contract['held_ids']=[]
            frame_factors.append(factor)
    if args.native_board_track_packet is not None:
        if args.single_frame_pts is None:raise ValueError('Actual printed anchor continuation requires an explicit source PTS')
        from structured_programme_source_candidate.native_board_track_source import NativeBoardTrackFactor
        tracks=NativeBoardTrackFactor(state,args.native_board_track_packet,len(q))
        selected_anchors=[row for row in tracks.anchor_rows if row['actual_pts']==args.single_frame_pts]
        if not selected_anchors:raise ValueError('No actual admitted printed-component anchors for this frame')
        tracks.rows=[];tracks.held=[];tracks.anchor_rows=selected_anchors;tracks.anchor_held=[]
        tracks.fit_evaluation_rows=selected_anchors
        tracks.scales=np.concatenate([r['sigma'].ravel()for r in selected_anchors])
        tracks.quadratic=np.ones(len(tracks.scales),bool);tracks.row_count=len(tracks.scales)
        tracks._cache.clear()
        tracks.contract.update(single_frame_actual_print_anchor_pts=args.single_frame_pts,
            actual_print_anchor_ids=[r['id']for r in selected_anchors],
            actual_print_anchors_are_direct_human_drawings=False,scalar_rows=tracks.row_count)
        frame_factors.append(tracks)
    def physical(value):
        if args.single_frame_pts is not None and not np.array_equal(value[fixed],scope_reference[fixed]):
            raise ValueError('Single-view initializer changed a fixed camera or parameter; joint source observations are required')
        provider.validate_parameters(value[world])
        if not np.isfinite(value).all()or np.any(value[pivots[:,2]]<=0.):raise ValueError('Existing positive pivot physical domain violated')
        for entry in mount['by_installation'].values():
            lever=np.asarray(entry['base_lever_m'])+value[np.asarray(entry['mount_parameter_indices'],int)]
            tolerance=mount['numeric_comparison_tolerance_working_m']
            if (np.linalg.norm(lever)<=1e-8 or np.any(lever<np.asarray(entry['absolute_lever_lower_m'])-tolerance)
                or np.any(lever>np.asarray(entry['absolute_lever_upper_m'])+tolerance)):
                raise ValueError('Existing user mount working range violated')
        if not pivot_outside_domain(state,value)['valid']:raise ValueError('Existing physical pivot outside current course violated')
        truth.queries(value)
        if frame_factors:
            # Highest-priority direct drawing remains in its original practical
            # support while actual same-frame material observations are fitted.
            actual_groups=truth.report(value)['full_stroke_guard']['rows']
            if any(row['maximum_violation_px']>allowed for row,allowed in zip(actual_groups,truth_initial_group_allowances)):
                raise ValueError('Same-frame material fit would move a confirmed human contour outside its initial engineering envelope')
            if wide_truth is not None and wide_truth.report(value)['maximum_violation_px']>wide_initial_group_allowance:
                raise ValueError('Actual print fit would move the original human corner TOP evidence outside its practical support')
        if confirmed is not None:
            try:confirmed.guard(value)
            except ValueError as error:
                if args.single_frame_pts is None:raise
                raise ValueError('Single-view shared-world trial violates retained other-frame source guard; joint source observations are required: '+str(error))from error
        return True
    def domains(value,columns):
        pivot=fixed_camera_pivot_step_constraints(state,value,columns)if args.single_frame_pts is not None else pivot_outside_linear_step_constraints(state,value,columns)
        domain=provider.current.print_vertical_domain_step_constraints(value[world])
        matrix=np.zeros((len(domain['matrix']),len(columns)));position={int(col):i for i,col in enumerate(columns)}
        for wi,col in enumerate(world):
            if int(col)in position:matrix[:,position[int(col)]]=np.asarray(domain['matrix'])[:,wi]
        confirmed_domain=None if confirmed is None else confirmed.constraints(value,columns)
        return dict(matrix=np.vstack((pivot['matrix'],matrix,np.empty((0,len(columns)))if confirmed_domain is None else confirmed_domain['matrix'])),lower=np.r_[pivot['lower'],domain['lower'],[]if confirmed_domain is None else confirmed_domain['lower']],
                    report=dict(phase_I_direct_human_truth=True,unchanged_pivot_outside_and_print_vertical_domains=True))
    def recover(parent,trial):
        value,report=recover_ad17_print_roundoff(provider.current,trial[world])
        fixed_trial=trial.copy();fixed_trial[world]=value
        outside=pivot_outside_domain(state,fixed_trial)
        fixed_trial,pivot_retractions=retract_selected_pivots(fixed_trial,outside,selected)
        if pivot_retractions:
            report=dict(print_roundoff_recovery=report,actual_pivot_outside_domain_retractions=pivot_retractions,
                nonlinear_source_cost_and_all_human_guards_still_required=True,new_user_position_or_tolerance_not_invented=True)
        return dict(q=fixed_trial,report=report)
    args.output.mkdir(parents=True)
    initial=truth.report(q)
    truth_initial_group_allowance=float(np.ceil(initial['full_stroke_guard']['maximum_violation_px']))
    truth_initial_group_allowances=np.asarray([np.ceil(row['maximum_violation_px'])for row in initial['full_stroke_guard']['rows']],float)
    wide_initial_group_allowance=0.
    if wide_truth is not None:
        initial['original_wide_human_truth']=wide_truth.report(q)
        wide_initial_group_allowance=float(np.ceil(initial['original_wide_human_truth']['maximum_violation_px']))
    (args.output/'initial_direct_human_guard.json').write_text(json.dumps(initial,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    if args.single_frame_pts is not None:
        (args.output/'actual_single_frame_all_parameter_scope.json').write_text(json.dumps(dict(
            source_pts=args.single_frame_pts,parent_q_sha256=parent_sha,selected_evaluation_indices=selected.tolist(),
            selected_canonical_indices=layout.canonical_indices(selected).tolist(),world_parameters=len(world),
            all_world_fields_selected=bool(np.isin(world,selected).all()),source_UV_queries=truth.row_count//2,
            full_stroke_guard_vertices=sum(len(r['uv'])for r in truth.full_stroke_records),
            other_installation_hardware_and_other_exposure_RF4_fixed=True,
            other_frame_source_guards_retained_with_fixed_cameras=True,
            original_source_UV_sigma_unchanged=True,old_complete_source_cost_not_an_initializer_acceptance_gate=True,
            other_visible_marks_require_same_frame_source_append=True,shared_axis_remains_exact=True,
            two_dimensional_warp_or_new_camera_DOF=False),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        (args.output/'actual_single_frame_admitted_source_view.json').write_text(json.dumps(dict(
            source_pts=args.single_frame_pts,independent_circle_canonical_indices=circle_canonical.tolist(),
            original_shared_pattern_canonical_indices=(pattern_start+np.arange(9)).tolist(),
            circle_layout_authority=dict(path=str(actual_manifest_path),sha256=sha(actual_manifest_path)),
            rows=[dict(id=r['id'],actual_pts=r['actual_pts'],uv_count=len(r['uv']),curve=r['curve'])for factor in frame_factors for r in factor.rows],
            actual_print_anchor_rows=[dict(id=r['id'],actual_pts=r['actual_pts'],advertisement_number=r['advertisement_number'],uv_count=len(r['uv']))for factor in frame_factors if hasattr(factor,'anchor_rows')for r in factor.anchor_rows],
            original_source_UV_sigma_and_split_preserved=True,other_frames_not_initializer_objectives=True),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    if args.local_physical_initializer:
        (args.output/'actual_local_physical_initializer_scope.json').write_text(json.dumps(dict(
            parent_q_sha256=parent_sha,selected_evaluation_indices=selected.tolist(),
            selected_canonical_indices=layout.canonical_indices(selected).tolist(),
            active_world_names=[provider.parameter_names[i]for i,col in enumerate(world)if col in selected],
            unchanged_other_world_names=[provider.parameter_names[i]for i,col in enumerate(world)if col not in selected],
            confirmed_original_UV_sigma_unchanged=True,confirmed_engineering_guard_radii_rounded_up_whole_pixels=True,
            old_complete_source_cost_not_an_initializer_acceptance_gate=True,
            shared_axis_remains_exact=True,two_dimensional_warp_or_new_camera_DOF=False),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    if args.write_practical_location_envelope is not None:
        if args.phase_I_initializer is None or wide_truth is None or args.stroke_cap_support is None:
            raise ValueError('Stopped complete canonical initializer and all actual manual operators required')
        policies=[]
        primary=truth.queries(q);full=truth.queries(q,full_stroke=True)
        for i,(a,b)in enumerate(zip(primary,full)):
            queries=[a['query'],b['query']];magnitude=max(float(abs(r['uv']).max())for r in (truth.records[i],truth.full_stroke_records[i]))
            rounding=64*np.finfo(float).eps*max(1.,magnitude)
            policies.append(dict(family='native',id=a['id'],full_vertices=len(truth.full_stroke_records[i]['uv']),
                actual_initial_normal_outside_raster_px=max(float(abs(x['normal_violation_px']).max())for x in queries),
                actual_initial_tangent_outside_source_footprint_px=max(float(abs(x['coverage_violation_px']).max())for x in queries),
                normal_expansion_px=max(float(abs(x['normal_violation_px']).max())for x in queries)+rounding,
                tangent_expansion_px=max(float(abs(x['coverage_violation_px']).max())for x in queries)+rounding,
                float64_operation_allowance_px=rounding))
        for record,row in zip(wide_truth.records,wide_truth.queries(q)):
            x=row['query'];rounding=64*np.finfo(float).eps*max(1.,float(abs(record['uv']).max()))
            policies.append(dict(family='wide',id=row['id'],full_vertices=len(record['uv']),
                actual_initial_normal_outside_raster_px=float(abs(x['normal_violation_px']).max()),
                actual_initial_tangent_outside_source_footprint_px=float(abs(x['coverage_violation_px']).max()),
                normal_expansion_px=float(abs(x['normal_violation_px']).max())+rounding,
                tangent_expansion_px=float(abs(x['coverage_violation_px']).max())+rounding,float64_operation_allowance_px=rounding))
        meta=dict(schema='rinkspace.fixed-human-visual-initializer-location-envelope.v1',
            user_rejected_subpixel_zero_band_refinement=True,location_envelope_is_measured_statistical_precision=False,
            fixed_for_subsequent_full_source_trials=True,source_UV_sigma_and_historical_packets_unchanged=True,
            purpose='Permit stopped visually improved initializer to continue; protect manual locations from old soft-cost drift. No physical precision inferred from drawing pixels.',
            no_further_phase_I_raster_polishing=True,complete_initializer_reference=dict(path=str(args.phase_I_initializer.resolve()),sha256=sha(args.phase_I_initializer)),
            canonical_q_sha256=hashlib.sha256(layout.canonical_q(q).tobytes()).hexdigest(),
            authority_references=dict(native=truth.contract['authority'],wide=wide_truth.contract['authority']),records=policies)
        if args.write_practical_location_envelope.exists():raise ValueError('Preserve prior location policy, write a new revision')
        args.write_practical_location_envelope.write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        truth.bind_practical_location_envelope(args.write_practical_location_envelope);wide_truth.bind_practical_location_envelope(args.write_practical_location_envelope)
        proof=dict(schema='rinkspace.actual-practical-manual-location-initializer-guard-check.v1',
            location_policy=dict(path=str(args.write_practical_location_envelope.resolve()),sha256=sha(args.write_practical_location_envelope)),
            canonical_q_sha256=meta['canonical_q_sha256'],native=truth.report(q),wide=wide_truth.report(q),
            no_motion_graph_or_J_loaded=True,full_source_started=False,no_pixel_precision_accuracy_claim=True)
        if not proof['native']['feasible']or not proof['wide']['feasible']:raise ValueError('Actual frozen initializer must satisfy its practical manual-location policy')
        (args.output/'actual_practical_manual_location_guard_check.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        print(json.dumps(dict(location_policy=proof['location_policy'],native_feasible=True,wide_feasible=True,elapsed_seconds=time.perf_counter()-started)));return
    if args.inspect_facet_coverage:
        course,cq,_=provider.current.print_course_binding(q[world]);perimeter=course.sample_perimeter(cq,with_jacobian=False)
        parts=perimeter[2];part_index=next(i for i,r in enumerate(parts)if r['id']=='lower_long_side')
        selected_parts=[parts[(part_index+k)%len(parts)]for k in (-1,0,1)]
        first=int(selected_parts[0]['first_vertex']);count=sum(int(r['vertex_count'])for r in selected_parts)
        ids=np.arange(first,first+count+1)%len(perimeter[0]);rows=[]
        from structured_programme_source_candidate.human_stroke_truth import finite_stroke_queries
        for full in truth.full_stroke_records[:2]:
            level=1 if full['row']['curve']['kind']=='board_top'else 0
            camera_block=truth.blocks[full['row']['actual_pts']]
            camera=camera_block.bundle.camera(native_camera_local(camera_block,q[:state.structured_programme_coordinate_chart.core_count]),full['row']['actual_pts'])
            allworld=np.asarray(perimeter[level]);uv,depth,central=project_central_valid_full_brown(camera,allworld)
            original=truth.queries(q,full_stroke=True)[level if level==0 else 1]['query']
            joined=finite_stroke_queries(camera,allworld[ids],full['uv'],full['normals'],full['bounds'])
            items=[]
            for index in range(5):
                a=uv;delta=np.roll(uv,-1,axis=0)-uv;length=np.einsum('ni,ni->n',delta,delta)
                station=np.clip(np.einsum('ni,ni->n',full['uv'][index]-a,delta)/np.maximum(length,1e-30),0.,1.)
                feet=a+station[:,None]*delta;distance=np.linalg.norm(feet-full['uv'][index],axis=1)
                valid=central&np.roll(central,-1)&(depth>0)&(np.roll(depth,-1)>0)
                nearest=int(np.argmin(np.where(valid,distance,np.inf)))
                primitive=next(r['id']for r in parts if int(r['first_vertex'])<=nearest<int(r['first_vertex'])+int(r['vertex_count']))
                items.append(dict(full_vertex=index,source_UV=full['uv'][index].tolist(),nearest_existing_perimeter_segment=nearest,
                    nearest_existing_primitive=primitive,nearest_existing_segment_UV=feet[nearest].tolist(),nearest_segment_station=float(station[nearest]),
                    nearest_existing_distance_px=float(distance[nearest]),original_straight_only_residual=original['residual'].reshape(-1,2)[index].tolist(),
                    connected_adjacent_original_perimeter_residual=joined['residual'].reshape(-1,2)[index].tolist()))
            rows.append(dict(id=full['row']['id'],source_raw_frame=full['record']['source_raw_frame'],physical_level=level,
                original_straight_only_vertices=25,adjacent_existing_course_parts=selected_parts,adjacent_original_vertex_indices=ids.tolist(),first_source_vertices=items))
        proof=dict(schema='rinkspace.actual-direct-human-lower-course-connected-coverage-diagnostic.v1',
            canonical_q_sha256=hashlib.sha256(layout.canonical_q(q).tobytes()).hexdigest(),rows=rows,
            no_source_scope_or_geometry_producer_changed=True,nearest_existing_segments_are_diagnostic_only=True)
        path=args.output/'actual_existing_lower_course_connected_coverage_diagnostic.json'
        path.write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(proof))
        return
    if args.inspect_only:
        canonical=layout.canonical_q(q)if layout is not None else q
        proof=dict(actual_operator_check_only=True,q_sha256=hashlib.sha256(canonical.tobytes()).hexdigest(),
            evaluation_q_sha256=hashlib.sha256(q.tobytes()).hexdigest(),
            original_wide_camera_difference=wide_initial_camera_error,original_wide_TOP_difference=wide_initial_TOP_error,
            original_wide_human_truth=None if wide_truth is None else wide_truth.report(q),
            no_motion_graph_or_J_loaded=True,fulljoint_started=False)
        proof['practical_manual_location_carrier_check']=practical_carrier_check
        if wide_truth is not None:
            diagnostics=[]
            for row in wide_truth.queries(q):
                query=row['query'];worst=int(np.argmax(np.linalg.norm(query['residual'].reshape(-1,2),axis=1)))
                diagnostics.append(dict(id=row['id'],worst_vertex=worst,
                    signed_normal_scanline_px=float(query['signed_normal_px'][worst]),
                    nearest_finite_projection_vector_px=query['nearest_finite_vector_px'][worst].tolist(),
                    nearest_finite_projection_signed_normal_px=float(query['nearest_finite_signed_normal_px'][worst]),
                    nearest_finite_projection_tangent_px=float(query['nearest_finite_tangent_px'][worst]),
                    nearest_finite_projection_segment_index=int(query['nearest_finite_segment_index'][worst]),
                    nearest_finite_projection_station=float(query['nearest_finite_segment_station'][worst])))
            proof['wide_normal_scanline_and_nearest_finite_branch_diagnostic']=diagnostics
        if args.physical_course_scope_source is not None:
            from rinkspace.vision_camera_initialization import frozen_scene_curve
            course,cq,_=provider.current.print_course_binding(q[world]);scene=course.scene(cq,rebuild_grid=False)
            scene.setdefault('ice_features',[]);checks=[]
            for row in board.rows:
                if row['curve'].get('source_structure')!='physical_course_facet':continue
                actual=board._world(q,row['curve']);exported=frozen_scene_curve(scene,board.effective_curve(row['curve']))
                if actual.shape!=exported.shape:raise ValueError('Authorized scope export differs in finite knot identity')
                difference=float(abs(actual-exported).max())
                if difference>5e-14:raise ValueError('Authorized same top/contact course and scene export differ')
                checks.append(dict(id=row['id'],effective_curve=board.effective_curve(row['curve']),
                    actual_world_shape=list(actual.shape),scene_export_max_difference=difference))
            proof['actual_scope_source_scene_and_fullguard_shared_producer_checks']=checks
            proof['native_complete_human_guard']=truth.report(q)
        path=args.output/'actual_existing_wide_TOP_hardguard_operator_check.json'
        path.write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(proof))
        return
    def reviewed(value,event):
        report=truth.report(value)
        if wide_truth is not None:report['original_wide_human_truth']=wide_truth.report(value)
        (args.output/'latest_direct_human_guard.json').write_text(json.dumps(dict(report,event=event),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        wide_feasible=wide_truth is None or report['original_wide_human_truth']['feasible']
        if report['feasible']and wide_feasible and not frame_factors:return dict(stop_branch=True,reason='All native direct human sparse queries, complete strokes and existing wide support envelopes are feasible; this is Phase-I initialization only')
        working=np.r_[truth.residual(value),[]if wide_truth is None else wide_truth.residual(value)]
        if frame_factors:return dict(primary_truth_report=report,same_frame_native_reviews=[factor.review(value)for factor in frame_factors])
        if np.all(working==0.):
            return dict(stop_branch=True,reason='Current explicit feasibility cuts reached their interior targets; add actual remaining complete-stroke violations, never soften hard support')
        return dict(primary_truth_report=report)
    def source_blocks(value,jacobian):
        native=truth.block(value,jacobian);yield native
        if wide_truth is not None:yield wide_truth.block(value,jacobian,original_start=len(native.r))
        offset=len(native.r)+(0 if wide_truth is None else wide_truth.row_count)
        for factor in frame_factors:
            block=factor.block(value,jacobian,original_start=offset)
            if hasattr(factor,'_manual_centre_mean_scale'):
                block=direct_manual_centre_block(factor,value,jacobian,original_start=offset)
            yield block;offset+=len(block.r)
        if confirmed is not None:
            carrier,lo,hi,_=confirmed._carriers(value);r=carrier-np.clip(carrier,lo,hi);J=None
            if jacobian:
                from scipy.sparse import csr_matrix
                compact=np.zeros((len(r),len(selected)))
                for i,col in enumerate(selected):
                    if int(col)in set(confirmed.active.tolist()):
                        compact[:,i]=positive_pivot_coordinate_derivative(
                            lambda x:(lambda a:a[0]-np.clip(a[0],a[1],a[2]))(confirmed._carriers(x)),
                            value,int(col),confirmed.steps.get(int(col),1e-6),confirmed.height_columns,base=r)
                rows,cols=np.nonzero(compact);J=csr_matrix((compact[rows,cols],(rows,selected[cols])),shape=(len(r),len(value)))
            yield SourceRowBlock(r,J,np.ones(len(r),bool),np.ones(len(r)),dict(
                family='user_confirmed_physical_initializer_protection',original_start=offset,
                original_stop=offset+len(r)))
    def evaluate(value,jacobian):
        blocks=list(source_blocks(value,jacobian));r=np.concatenate([b.r for b in blocks])
        if not jacobian:return r
        from scipy.sparse import vstack
        return r,vstack([b.J for b in blocks],format='csr')
    parent=dict(path=str(args.state.resolve()),sha256=sha(args.state),q_sha256=parent_sha)
    checkpoint_identity=dict(schema='rinkspace.direct-human-phase-I-initializer.v1',parent=parent,
        authority=truth.contract['authority'],full_stroke_guard=truth.contract['full_stroke_guard'],
        original_wide_authority=None if wide_truth is None else wide_truth.contract['authority'],
        old_source_full_graph_or_guard_not_loaded=True,primary_cost_is_feasibility_not_source_fit=True)
    checkpoint_arrays={k:v for k,v in arrays.items()if k.startswith(('aux__','profile_'))}
    native_cuts=[set()for _ in truth.records];wide_cuts=[]if wide_truth is None else[set()for _ in wide_truth.records]
    phase_stages=[];stage=0;working=q.copy();result=None
    def update_cuts(queries,existing):
        added=0
        for row,cuts in zip(queries,existing):
            vectors=row['query']['residual'].reshape(-1,2)
            indices={int(np.argmax(abs(vectors[:,axis])))for axis in range(2)}
            indices.add(int(np.argmax(np.linalg.norm(vectors,axis=1))))
            for i in indices:
                if np.any(vectors[i]!=0.)or not cuts:
                    if i not in cuts:cuts.add(i);added+=1
        return added
    while True:
        native_added=update_cuts(truth.queries(working,full_stroke=True),native_cuts)
        wide_added=0 if wide_truth is None else update_cuts(wide_truth.queries(working),wide_cuts)
        before=truth.report(working);wide_before=None if wide_truth is None else wide_truth.report(working)
        if before['feasible']and(wide_before is None or wide_before['feasible'])and not frame_factors:break
        if stage and native_added+wide_added==0:
            phase_stages[-1]['no_new_violating_vertex_available']=True
            phase_stages[-1]['original_hard_band_global_infeasibility_proven']=False
            break
        inventory=truth.set_phase_I_guard_vertices([sorted(x)for x in native_cuts],interior_target_px=0. if args.local_physical_initializer or args.single_frame_pts is not None else .01)
        wide_inventory=None if wide_truth is None else wide_truth.set_phase_I_guard_vertices([sorted(x)for x in wide_cuts],interior_target_px=0. if args.local_physical_initializer else .01)
        stage+=1;stage_path=args.output/('feasibility_cut_stage_'+str(stage).zfill(3))
        stage_identity=dict(checkpoint_identity,phase_I_guard_cuts=inventory,wide_phase_I_guard_cuts=wide_inventory,
            temporary_initializer_interior_margin_px=.01,original_hard_support_not_changed=True)
        quadratic=np.r_[truth.quadratic,[]if wide_truth is None else wide_truth.quadratic].astype(bool)
        scales=np.r_[truth.scales,[]if wide_truth is None else wide_truth.scales]
        for factor in frame_factors:quadratic=np.r_[quadratic,factor.quadratic];scales=np.r_[scales,factor.scales*getattr(factor,'_manual_centre_mean_scale',1.)]
        if confirmed is not None:
            size=len(confirmed._carriers(working)[0]);quadratic=np.r_[quadratic,np.ones(size,bool)];scales=np.r_[scales,np.ones(size)]
        result=solve_joint_source(evaluate,working,quadratic,scales,stage_path,physical,
            checkpoint_callback=reviewed,checkpoint_identity=stage_identity,checkpoint_arrays=checkpoint_arrays,
            verbose=False,global_schur_count=len(selected),global_schur_indices=selected,
            fixed_parameter_indices=fixed,schur_backtracking=True,source_block_factory=source_blocks,
            schur_view_lm_ratio=1.,schur_backtracking_aware_damping=True,cache_native_source_replay=True,
            source_stage_timing=True,adaptive_source_backtracking=True,positive_parameter_indices=np.intersect1d(pivots[:,2],selected),
            positive_parameter_resolution=1e-6,shared_parameter_bounds=bounds,shared_linear_constraints=domains,
            trial_feasibility_recovery=recover,
            numerical_convergence_guard=lambda value:(truth.report(value)['full_stroke_guard']['maximum_violation_px']<=truth_initial_group_allowance if frame_factors else truth.report(value)['feasible'])and(wide_truth is None or wide_truth.report(value)['feasible']))
        working=result.q.copy();native_after=truth.report(working);wide_after=None if wide_truth is None else wide_truth.report(working)
        phase_stages.append(dict(stage=stage,status=result.status,accepted_iterations=sum(r['accepted']for r in result.history),
            native_guard= native_after['full_stroke_guard'],wide_guard=wide_after,native_cuts=inventory,wide_cuts=wide_inventory,
            numeric_convergence_is_not_hard_feasibility=True,temporary_cuts_are_not_added_FIT_observations=True))
        (args.output/'actual_feasibility_cut_stage_progress.json').write_text(json.dumps(phase_stages,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    if result is None:
        from types import SimpleNamespace
        result=SimpleNamespace(q=working,status='already_complete_human_feasible')
    if args.single_frame_pts is not None and not np.array_equal(result.q[fixed],scope_reference[fixed]):
        raise ValueError('Single-view result changed fixed parameters; no initializer will be exported')
    final_q=layout.canonical_q(result.q)if layout is not None else result.q.copy()
    saved=dict(arrays);saved['q']=final_q;saved['identity_json']=np.asarray(json.dumps(checkpoint_identity))
    initializer=args.output/'complete_Phase_I_initializer_UNACCEPTED.npz';np.savez_compressed(initializer,**saved)
    final=truth.report(result.q)
    wide_final=None if wide_truth is None else wide_truth.report(result.q)
    report=dict(schema='rinkspace.actual-direct-human-cheap-phase-I-diagnostic.v1',parent=parent,
        complete_initializer=dict(path=str(initializer),sha256=sha(initializer),q_sha256=hashlib.sha256(final_q.tobytes()).hexdigest()),
        native_human_truth=final,initial_maximum_primary_violation_px=initial['maximum_violation_px'],
        original_wide_human_truth=wide_final,feasibility_cut_stages=phase_stages,
        phase_I_status=result.status,physical_camera_archive_initial_max_difference=camera_error,
        complete_parameter_count=len(final_q),profile_arrays_preserved=18,original_auxiliary_arrays_preserved=True,
        changed_canonical_parameter_indices=np.flatnonzero(final_q!=parent_q).tolist(),
        single_view_fixed_parameters_preserved=args.single_frame_pts is None or bool(np.array_equal(result.q[fixed],scope_reference[fixed])),
        single_view_infeasible_action=None if args.single_frame_pts is None else 'Retain every guard and use jointly admitted source views; do not release other cameras',
        original_source_UV_sigma_roles_packets_not_modified=True,old_motion_graph_and_J_not_loaded=True,
        old_source_active_quarantine_and_source_guard_validation_pending=True,
        wide_human_TOP_guard_pending_separate_operator=wide_final is None or not wide_final['feasible'],
        all_complete_human_strokes_feasible=final['feasible']and(wide_final is not None and wide_final['feasible']),
        solver='unchanged z190_joint_source_solver.solve_joint_source',fulljoint_started=False,accuracy_accepted=False,
        elapsed_seconds=time.perf_counter()-started)
    (args.output/'actual_Phase_I_result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(result_path=str(args.output/'actual_Phase_I_result.json'),status=result.status,
        primary_maximum_violation_px=final['maximum_violation_px'],complete_native_feasible=final['feasible'],seconds=report['elapsed_seconds'])))


if __name__=='__main__':main()
