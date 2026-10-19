"""Importable complete secondary source factory for shared-scene diagnostics.

No import starts a solve. All actual exposures, original motion and source
roles are preserved. Only previously qualified development outer-ring samples
use explicit source upper/lower operators. Candidate geometry must be bound
with one validated shared loader/rebinder supplied by the caller.
"""
from pathlib import Path
from types import SimpleNamespace
import json
from z190_secondary_nonzero_mount_full_source import prepare,changes,write,ref,IDENTIFIER,SCENE
from shared_F_scene_candidate import load_candidate,rebind
from z190_moving_pivot_source_bundle import MovingPivotSourceBundle
from z190_profiled_motion_schur import SchurMotionKernel
from z190_continuous_source_bundle import SCENE_UNIT_M,SOURCE_SHA
from continuous_central_brown4_optin import enable_projection_backend
import numpy as np,cv2
DEFAULT_SOURCE_ROOT=Path('D:/CodexData/.codex/visualizations/2026/10/02/private-thread-00008/RinkSpace_tmp/source_solver_audit')
TEMP_ROOT=DEFAULT_SOURCE_ROOT.parent
T=Path('D:/CodexData/.codex/visualizations/2026/10/02/private-thread-00009/RinkSpace_tmp')
CAM=T/'z190_full_programme_depth_profile_low8_v1/z190_full_programme_camera_candidate.npz'
AUTH=T/'z190_secondary_source_factor_authority_v1/source_authority_rows.json'

def prepare_full_secondary_core_source(output,candidate_scene_path=None,*,source_root=None,scene_loader=None,world_rebinder=None):
    B=Path(source_root).resolve() if source_root is not None else DEFAULT_SOURCE_ROOT
    scene_loader=load_candidate if scene_loader is None else scene_loader
    world_rebinder=rebind if world_rebinder is None else world_rebinder
    O=Path(output).resolve()
    if not O.is_relative_to(TEMP_ROOT.resolve()):raise ValueError('Bulk source output must stay in this round temporary root')
    O.mkdir(parents=True,exist_ok=False);(O/'original_source_adapter').mkdir();cv2.setNumThreads(1);enable_projection_backend()
    if candidate_scene_path is None:
        original_scene=json.loads(SCENE.read_text(encoding='utf8'));scene=original_scene
        scene_binding=dict(baseline_scene=ref(SCENE),candidate_scene=ref(SCENE),F_column_dx_m=0.,all_four_F_houses_and_concentric_rings_share_translation=True,radii_row_Y_R_houses_Lane_other_paint_board_ads_and_metric_unchanged=True,accuracy_accepted=False)
    else:original_scene,scene,scene_binding=scene_loader(Path(candidate_scene_path),SCENE)
    if ref(CAM)['sha256']!='7625b07752027a857fcc9ea2afa979f2c639152308851958e94bdeee27994e1d':raise ValueError('Production current camera changed')
    old,q,current,views,native,aliases,rows,manifest=prepare(O/'original_source_adapter',CAM,AUTH,warm_current_mount=True);groups={p:list(rr)for p,rr in old.abs_groups}
    ledger=json.loads((B/'secondary_source_interval_RF_v1/source_corner_to_interval_ledger.json').read_text(encoding='utf8'));inactive=[]
    for p,rr in groups.items():
        inactive+=[r for r in rr if r['pts']in(230499584,230507264)and r['id']in('ad7:left_top','ad7:left_bottom')];groups[p]=[r for r in rr if not(r['pts']in(230499584,230507264)and r['id']in('ad7:left_top','ad7:left_bottom'))]
    assert len(inactive)==5
    for rr in ledger['new_development_interval_rows']:
        row=dict(rr)
        for k in('world','uv','normal','normal_interval','sigma'):row[k]=np.array(row[k])
        groups.setdefault(aliases[row['pts']],[]).append(row)
    f2packet=B/'secondary_core_ice_current_review_v1/new_f2_outer_full_source_gradient_qualified_v2.json';f2=json.loads(f2packet.read_text(encoding='utf8'));assert f2['source_visual_inspection_completed'];development=[r for r in f2['records']if r['source_role']=='development'];assert len(development)==2;icegroups=[]
    for rec in development:
        rr=rec['source_qualified_rows'];count=len(rr);row=dict(id='F2_outer_source_qualified_red_transition',pts=rec['source_pts'],split='fit',source_role='new_development',world=np.array(f2['world_curve_xyz']),uv=np.array([r['source_uv']for r in rr]),normal=np.array([r['source_normal_uv']for r in rr]),normal_interval=np.array([r['normal_interval_px']for r in rr]),sigma=np.full(count,2.0*np.sqrt(count)),point_correspondence=False,origin=str(f2packet),physical_identity_authority=rec['physical_identity_authority'],sigma_modelling_decision='2px engineering local source read sigma outside explicit blur interval; sqrt(group scan count) correlation normalization, not statistical confidence')
        row.update(physical_curve_id='F2_outer',canonical_physical_geometry_id='F2_outer',
            physical_curve_id_authority=ref(f2packet))
        assert aliases[row['pts']]==row['pts'];groups.setdefault(row['pts'],[]).append(row);icegroups.append(row)
    cut2packet=B/'secondary_core_ice_current_review_v1/new_cut2_ring_lane_full_source_gradient_qualified_v2.json';cut2=json.loads(cut2packet.read_text(encoding='utf8'));assert cut2['source_visual_inspection_completed'];cut2groups=[]
    for rec in cut2['records']:
        if rec['source_role']!='development':continue
        rr=rec['source_qualified_rows'];count=len(rr);row=dict(id=rec['physical_id']+'_source_qualified_native_transition',pts=rec['source_pts'],split='fit',source_role='new_development',world=np.array(rec['world_curve_xyz']),uv=np.array([r['source_uv']for r in rr]),normal=np.array([r['source_normal_uv']for r in rr]),normal_interval=np.array([r['normal_interval_px']for r in rr]),sigma=np.full(count,2.*np.sqrt(count)),point_correspondence=False,origin=str(cut2packet),sigma_modelling_decision=cut2['engineering_sigma_decision'])
        row.update(physical_curve_id=rec['physical_id'],canonical_physical_geometry_id=rec['physical_id'],
            physical_curve_id_authority=ref(cut2packet))
        assert aliases[row['pts']]==row['pts'];groups.setdefault(row['pts'],[]).append(row);cut2groups.append(row)
    write(O/'new_cut2_ice_source_groups.json',dict(packet=ref(cut2packet),groups=cut2groups,qualified_points=sum(len(r['uv'])for r in cut2groups),new_development_exposures=sorted(set(r['pts']for r in cut2groups)),source_visual_inspection_completed=True))
    extentpacket=B/'secondary_core_ice_current_review_v1/whole_ring_extent_full_source_gradient_qualified.json';extent=json.loads(extentpacket.read_text(encoding='utf8'));extentgroups=[]
    for rec in extent['records']:
        if rec['source_role']!='development':continue
        assert rec['source_visual_inspection_completed'];row=dict(rec['source_row'])
        for k in('world','uv','normal','normal_interval','sigma'):row[k]=np.array(row[k])
        row.update(physical_curve_id=rec['physical_id'],canonical_physical_geometry_id=rec['physical_id'],
            physical_curve_id_authority=ref(extentpacket))
        assert aliases[row['pts']]==row['pts'];groups.setdefault(row['pts'],[]).append(row);extentgroups.append(row)
    write(O/'whole_ring_extent_source_admission.json',dict(packet=ref(extentpacket),development_groups=extentgroups,held_extent_source_not_fit=[r for r in extent['records']if r['source_role']!='development'],source_gradient_supports_uses_complete_ROI=True,source_read_not_camera_derived=True,earlier_peak_plusminus6_intervals_replaced_preserving_history=True))
    # Rebind every admitted original/new factor to the same physical F-column scene.
    # The original authority files, source UV, normal supports, role and sigma stay immutable.
    binding_ledger=[]
    for p,rr in groups.items():
        rebound=[]
        for row in rr:
            moved=world_rebinder(row,original_scene,scene)
            for field in ('uv','normal','normal_interval','sigma'):
                if field in row and not np.array_equal(row[field],moved[field]):raise ValueError('Source fields changed:'+field)
            if row.get('split')!=moved.get('split') or row.get('source_role')!=moved.get('source_role'):raise ValueError('Source role changed')
            if moved is not row:binding_ledger.append(dict(pts=row['pts'],id=row['id'],world_before=row['world'],world_candidate=moved['world']))
            rebound.append(moved)
        groups[p]=rebound
    bound_rows={role:[world_rebinder(row,original_scene,scene)for row in rr]for role,rr in rows.items()}
    write(O/'candidate_scene_world_source_binding.json',dict(binding=scene_binding,fit_F_world_changes=binding_ledger,all_original_UV_roles_sigma_unchanged=True,held_pending_rebound_to_same_physical_scene_not_fit=True,original_roles_count={k:len(v)for k,v in rows.items()}))
    write(O/'all_original_source_current_hard_scene.json',changes(current,views,aliases,rows))
    write(O/'all_original_source_current_candidate_scene_conditional.json',changes(current,views,aliases,bound_rows))
    absgroups=[(int(p),groups[int(p)])for p in old.stamps if groups.get(int(p))];absolute_sigma=np.concatenate([np.repeat(r['sigma'],2)for _,rr in absgroups for r in rr]);base=SimpleNamespace(stamps=old.stamps,index=old.index,a=old.a,b=old.b,ia=old.ia,ib=old.ib,motion_sigma=old.motion_sigma,abs_groups=absgroups,native_residual_sigma=np.r_[old.motion_sigma,absolute_sigma])
    planes=old.finite_planes.copy();finite=list(old.finite_records);finitepacket=B/'secondary_core_ice_current_review_v1/source_qualified_original_ice_motion_rows.json';mat=json.loads(finitepacket.read_text(encoding='utf8'))
    for rec in mat['records']:
        if rec['status']!='qualified_ice_z0':continue
        i=rec['original_global_bundle_index'];assert not old.finite_mask[i];assert int(old.stamps[old.ia[i]])==rec['source_pts']and int(old.stamps[old.ib[i]])==rec['target_pts'];assert np.array_equal(old.a[i],rec['source_uv'])and np.array_equal(old.b[i],rec['target_uv']);planes[i]=[0,0,1,0];finite.append(dict(global_bundle_index=i,surface_kind='ice_plane',plane=[0,0,1,0],source_qualification=ref(finitepacket),record=rec))
    cal=current.calibrations[IDENTIFIER];bundle=MovingPivotSourceBundle(base,cal,SchurMotionKernel(minimum_source_depth_scene_units=1./SCENE_UNIT_M),reference_pts=old.reference_pts,initial_reference_optical_height_scene=old.initial_reference_optical_height_scene,finite_planes=planes,finite_records=finite,scene=scene)
    write(O/'core_source_admission_ledger.json',dict(new_ice_development=icegroups,original_circle_held_pending_untouched=rows['held_rows']+rows['pending_rows'],new_source_reserved_not_fit=[r for r in f2['records']if r['source_role']!='development'],source_plane_binding=finite,original_false_precision_corner_history=ledger,source_UV_modified=False,new_development_UV_derived_from_native_source_not_camera=True))
    report=dict(schema='rinkspace.secondary.fixed-current-mechanics.shared-complete-F-column-core-source-RF.v1',candidate_scene_binding=scene_binding,scene_candidate_only=True,production_scene_changed=False,v4_same_source_objective_new_scene_only=True,new_source_branch_diagnostic_not_refitted=True,new_Lane_or_Ad_source_admission=False,current_complete=ref(CAM),original_source_adapter_manifest=ref(O/'original_source_adapter/input_manifest.json'),scene=scene_binding['candidate_scene'],actual_exposures=len(bundle.stamps),native_presentations=len(native),source_sha256=SOURCE_SHA,original_motion_rows=len(bundle.a),original_motion_UV_sigma_unchanged=True,original_active_fit_factors=262,new_F2_ice_development_groups=2,new_F2_scan_points=sum(len(r['uv'])for r in icegroups),finite_ice_motion_original_rows_newly_bound=4,total_finite_motion_rows=len(finite),current_pivotXYZ=cal.pivot_xyz,current_leverXYZ=cal.lever_xyz,fixed=['pivotXYZ','leverXYZ','PP','k1','fixed shared-F candidate scene','official aliases','all original held/pending roles and UV'],free='all1061 actualR3/effectivef',unknown_source_depth_floor_m=1.,depth_floor_engineering_assumption_not_measured=True,core_source_packet=ref(f2packet),ledger=ref(O/'core_source_admission_ledger.json'),no_rejected_or_stopped_candidate_warm=True,accuracy_accepted=False,video_generated=False)
    write(O/'input_manifest.json',report)
    return dict(source_root=B,world_rebinder=world_rebinder,scene_loader=scene_loader,bundle=bundle,q=q,current=current,views=views,native=native,aliases=aliases,original_source_rows=rows,scene_source_rows=bound_rows,baseline_scene=original_scene,scene=scene,scene_binding=scene_binding,input_manifest=report)


def prepare_full_secondary_qualified_branches(output,candidate_scene_path=None,*,source_root=None,scene_loader=None,world_rebinder=None):
    prepared=prepare_full_secondary_core_source(output,candidate_scene_path,source_root=source_root,scene_loader=scene_loader,world_rebinder=world_rebinder)
    B=prepared['source_root']
    old=prepared['bundle'];packets=[B/'secondary_core_ice_current_review_v1/new_f2_outer_full_source_gradient_qualified_v2.json',B/'secondary_core_ice_current_review_v1/new_cut2_ring_lane_full_source_gradient_qualified_v2.json'];lookup={}
    for packetpath in packets:
        packet=json.loads(packetpath.read_text(encoding='utf8'))
        for rec in packet['records']:
            if rec['source_role']!='development' or rec['physical_id']not in ('F2_outer','R2_outer','R3_outer'):continue
            lookup[(rec['source_pts'],rec['physical_id'])]=(rec,packetpath)
    groups=[];ledger=[];original_points=0;typed_points=0
    for p,rows in old.abs_groups:
        newrows=[]
        for row in rows:
            original_points+=len(row['uv'])
            physical='_'.join(str(row['id']).split('_')[:2]);entry=lookup.get((row['pts'],physical))
            if entry is None or row.get('source_role')!='new_development' or 'source_qualified' not in str(row['id']):
                newrows.append(row);typed_points+=len(row['uv']);continue
            rec,packetpath=entry;source=rec['source_qualified_rows']
            assert len(source)==len(row['uv']);assert np.array_equal(row['uv'],np.array([s['source_uv']for s in source]))
            indices={'upper':[],'lower':[],'horizontal_source_normal':[]}
            for i,s in enumerate(source):
                if s['source_normal_uv']!=[0.,1.]:indices['horizontal_source_normal'].append(i);continue
                if physical=='F2_outer':upper=s['arc']=='upper'
                else:
                    seed=s['seed_index']
                    if seed is None:raise ValueError('Vertical-normal ring source lacks branch authority')
                    upper=seed<(10 if physical=='R2_outer'else 8)
                    if p==229526272 and physical=='R2_outer'and seed==9:upper=False
                indices['upper'if upper else'lower'].append(i)
            covered=[]
            for branch,selected in indices.items():
                if not selected:continue
                typed=dict(row);typed['id']=str(row['id'])+'_'+branch
                for field in ('uv','normal','normal_interval','sigma'):typed[field]=row[field][selected].copy()
                if branch!='horizontal_source_normal':
                    typed.update(curve_observation_kind='axis_scanline_branch',scanline_fixed_axis='x',scanline_branch='min_y'if branch=='upper'else'max_y',must_preserve_upper_branch=branch=='upper',source_branch_operator_authority=ref(packetpath),branch_not_changed_by_nearest_projection=True)
                # Every subgroup keeps sigma=2sqrt(full group N). Splitting a
                # physical exposure does not create independent new evidence.
                assert np.array_equal(typed['sigma'],row['sigma'][selected]);assert np.array_equal(typed['world'],row['world']);assert typed['split']==row['split'] and typed['source_role']==row['source_role']
                newrows.append(typed);covered+=selected;typed_points+=len(selected)
            assert sorted(covered)==list(range(len(source)))
            ledger.append(dict(pts=p,physical_id=physical,old_operator=row.get('curve_observation_kind','nearest_complete_curve'),source_packet=ref(packetpath),source_sigma=row['sigma'],upper_source_indices=indices['upper'],lower_source_indices=indices['lower'],unchanged_horizontal_source_indices=indices['horizontal_source_normal'],source_UV_normals_supports_roles_sigma_preserved=True))
        groups.append((p,newrows))
    assert original_points==typed_points
    sig=np.concatenate([np.repeat(r['sigma'],2)for _,rows in groups for r in rows]);base=SimpleNamespace(stamps=old.stamps,index=old.index,a=old.a,b=old.b,ia=old.ia,ib=old.ib,motion_sigma=old.motion_sigma,abs_groups=groups,native_residual_sigma=np.r_[old.motion_sigma,sig])
    bundle=MovingPivotSourceBundle(base,old.calibration,old.motion_kernel,reference_pts=old.reference_pts,initial_reference_optical_height_scene=old.initial_reference_optical_height_scene,finite_planes=old.finite_planes.copy(),finite_records=old.finite_records,scene=old.scene)
    assert np.array_equal(old.a,bundle.a)and np.array_equal(old.b,bundle.b)and np.array_equal(old.finite_planes,bundle.finite_planes)
    prepared['bundle']=bundle
    report=dict(schema='rinkspace.secondary.exact-source-qualified-upper-lower-operator-adapter.v1',operator_change_only=True,source_measurements_added=False,original_absolute_point_count=original_points,typed_absolute_point_count=typed_points,groups=ledger,full_physical_group_correlation_sigma_preserved=True,original_262_active_35_held_4_pending_untouched=True,new_held_pending_fitted=False,new_native_aliases_unchanged=True,original_motion_rows=len(bundle.a),finite_motion_rows=len(bundle.finite_records),production_changed=False,solve_started=False,accuracy_accepted=False)
    write(Path(output)/'source_branch_operator_adapter.json',report)
    prepared['input_manifest']=dict(prepared['input_manifest'],v4_same_source_objective_new_scene_only=False,new_source_branch_diagnostic_not_refitted=False,qualified_branch_operator_adapter=ref(Path(output)/'source_branch_operator_adapter.json'),source_branch_modelling_change='only existing new qualified development vertical-normal outer-ring rows use explicit upper/lower, unchanged full-group sigma, no original/held/pending admission')
    write(Path(output)/'input_manifest.json',prepared['input_manifest'])
    return prepared


def _numeric_row(raw):
    row=dict(raw)
    for field in ('world','uv','normal','normal_interval','sigma'):
        if field in row:row[field]=np.asarray(row[field],float)
    return row


def source_records(prepared,*,candidate_scene=None,world_rebinder=None,
                   include_development=True,include_pending=True):
    """Return actual fit rows and separately held source checks, without admission.

    The returned dictionaries wrap the immutable source row: ``row`` is the
    numeric observation and ``report_role`` identifies how it is checked.
    Development rows are the actual bundle objective, never a second nearest
    representation. Pending identities remain conditional. New R2 right extent
    is diagnostic held even though its exposure also has development readings.
    """
    B=prepared['source_root'];S=B/'secondary_core_ice_current_review_v1'
    target=prepared['scene'] if candidate_scene is None else candidate_scene
    rebinder=prepared['world_rebinder'] if world_rebinder is None else world_rebinder
    records=[]

    def add(raw,role,base,packet=None,*,fitted=False,conditional=False):
        row=_numeric_row(raw);bound=rebinder(row,base,target)
        for field in ('uv','normal','normal_interval','sigma'):
            if field in row and not np.array_equal(row[field],bound[field]):
                raise ValueError('Reporting rebound changed source '+field)
        if row.get('split')!=bound.get('split') or row.get('source_role')!=bound.get('source_role'):
            raise ValueError('Reporting rebound changed source role')
        records.append(dict(row=bound,report_role=role,entered_fit_objective=fitted,
            conditional_identity=conditional,source_packet=ref(packet) if packet else None,
            reporting_does_not_admit_source=True))

    if include_development:
        for _,rows in prepared['bundle'].abs_groups:
            for row in rows:add(row,'development_actual_objective',prepared['scene'],fitted=True)
    for row in prepared['original_source_rows']['held_rows']:
        add(row,'held_original',prepared['baseline_scene'])
    if include_pending:
        for row in prepared['original_source_rows']['pending_rows']:
            add(row,'pending_original_conditional',prepared['baseline_scene'],conditional=True)

    extent_path=S/'whole_ring_extent_full_source_gradient_qualified.json'
    for rec in json.loads(extent_path.read_text(encoding='utf8'))['records']:
        if rec['source_role']!='development':
            row=dict(rec['source_row'],physical_curve_id=rec['physical_id'],
                canonical_physical_geometry_id=rec['physical_id'],physical_curve_id_authority=ref(extent_path))
            add(row,'held_new_whole_extent',prepared['baseline_scene'],extent_path)
    branch_path=B/'scanline_branch_audit_v3_source_corrected_guarded/source_branch_rows_qualified_for_diagnostic_only.json'
    for row in json.loads(branch_path.read_text(encoding='utf8'))['rows']:
        if row['source_role']!='development':
            # This packet has exactly one reserved group, explicitly inherited
            # from the source-qualified F2 outer packet, not a suffix guess.
            if row['id']!='F2_outer_lower' or row['pts']!=230395904:
                raise ValueError('Unexpected new reserved branch identity')
            row=dict(row,physical_curve_id='F2_outer',canonical_physical_geometry_id='F2_outer',
                physical_curve_id_authority=ref(branch_path),source_branch_operator_authority=ref(branch_path))
            add(row,'reserved_new_source_branch_not_fit',prepared['baseline_scene'],branch_path)
    right_path=B/'scanline_branch_audit_v3_source_corrected_guarded/R2_229526272_right_extent_source_only_diagnostic.json'
    extra=json.loads(right_path.read_text(encoding='utf8'))
    if extra['source_row']['id']!='R2_outer_whole_right_extent' or extra['source_pts']!=229526272:
        raise ValueError('Unexpected source-only R2 right extent identity')
    row=dict(extra['source_row'],physical_curve_id='R2_outer',canonical_physical_geometry_id='R2_outer',
        physical_curve_id_authority=ref(right_path),source_qualification=ref(right_path))
    add(row,'diagnostic_held_new_R2_right_not_blind_not_fit',
        prepared['baseline_scene'],right_path)

    for packetpath in (S/'ad2_ad3_source_whole_visible_boundaries_qualified.json',
                       B.parent/'ice_alignment_review/ad7_source_boundaries_qualified.json'):
        packet=json.loads(packetpath.read_text(encoding='utf8'))
        for rec in packet['records']:
            # The five source-qualified Ad7 development interval groups are
            # already represented by their actual bundle rows above.
            if rec['source_role']=='development':continue
            points=rec.get('source_qualified_rows',rec['rows'])
            if not points:continue
            ends=np.asarray(rec['world_endpoints'],float)
            t=np.linspace(0.,1.,121)[:,None]
            row=dict(id=rec['physical_id'],pts=rec['source_pts'],split='held',
                source_role=rec['source_role'],world=(1-t)*ends[0]+t*ends[1],
                uv=np.asarray([s['source_uv'] for s in points],float),
                normal=np.asarray([s.get('source_normal_uv',[1.,0.]) for s in points],float),
                normal_interval=np.asarray([s['normal_interval_px'] for s in points],float),
                sigma=np.ones(len(points)),point_correspondence=False,
                diagnostic_unit_sigma_not_fit_weight=True,
                blind_independent_physical_group_claim=rec.get('blind_independent_physical_group_claim',False))
            add(row,'held_new_whole_ad_boundary_not_fit',prepared['baseline_scene'],packetpath)
    return records


def report_source_records(prepared,model,views,*,candidate_scene=None,
                          world_rebinder=None,records=None,output_path=None):
    """Evaluate outside-support pixels by physical source/branch/role.

    Camera objects must use the same 1061 canonical source exposures. A domain
    error is retained in the report as a failure, never omitted as a bad point.
    This numerical report is a source diagnostic and cannot replace native
    whole-frame, whole-cut inspection.
    """
    from z190_source_camera_diagnostic import absolute_value_jac15
    from projection import project_central_valid
    from reference_board_pair import BoardIntersectionError
    records=source_records(prepared,candidate_scene=candidate_scene,
        world_rebinder=world_rebinder) if records is None else records
    result=[]
    for rec in records:
        row=rec['row'];pts=int(row['pts']);canonical=prepared['aliases'][pts]
        entry={k:v for k,v in rec.items() if k!='row'}
        entry.update(pts=pts,canonical_pts=canonical,id=row['id'],
            source_role=row.get('source_role'),split=row.get('split'),
            curve_observation_kind=row.get('curve_observation_kind','nearest_complete_curve'),
            scanline_branch=row.get('scanline_branch'),extent_side=row.get('extent_side'),
            source_uv=row['uv'],source_normal=row.get('normal'),
            source_normal_interval=row.get('normal_interval'),source_sigma=row['sigma'])
        camera=model.camera25(views[canonical])
        try:
            residual=absolute_value_jac15(camera,[row],False).reshape(-1,2)*row['sigma'][:,None]
            projected,depth,valid=project_central_valid(camera,row['world'])
            finite=np.isfinite(projected).all(axis=1)
            entry.update(status='evaluated',outside_support_xy_px=residual,
                maximum_euclidean_outside_px=float(np.max(np.linalg.norm(residual,axis=1))),
                median_euclidean_outside_px=float(np.median(np.linalg.norm(residual,axis=1))),
                complete_curve_x_minmax=[float(np.min(projected[finite,0])),float(np.max(projected[finite,0]))] if finite.any() else None,
                complete_curve_all_central_forward=bool((valid & (depth>0.) & finite).all()))
            if 'normal' in row:
                normal_outside=np.abs(np.sum(residual*row['normal'],axis=1))
                entry.update(maximum_normal_outside_px=float(normal_outside.max()),
                    median_normal_outside_px=float(np.median(normal_outside)))
        except BoardIntersectionError as ex:
            entry.update(status='projection_domain_failure',failure_type=type(ex).__name__,failure=str(ex))
        result.append(entry)
    report=dict(schema='rinkspace.secondary.complete-source-per-role-branch-report.v1',
        records=result,all_source_roles_preserved=True,held_pending_admitted=False,
        no_source_failures_hidden=True,production_changed=False,accuracy_accepted=False,
        numerical_report_does_not_replace_native_whole_cut_review=True)
    if output_path is not None:write(Path(output_path),report)
    return report


def review_held_groups(prepared):
    """Root-joint interface: original/new held and pending, bound to seed scene.

    These groups are report-only. In particular the new same-exposure R2 right
    extent never enters ``bundle.abs_groups``. Source ``split`` and
    ``source_role`` remain untouched; the additional review label makes the
    original pending conditional identity explicit to the caller.
    """
    grouped={}
    for rec in source_records(prepared,include_development=False):
        row=dict(rec['row'])
        row.update(review_report_role=rec['report_role'],
            review_conditional_identity=rec['conditional_identity'],
            review_source_packet=rec['source_packet'],review_only_never_admitted=True)
        p=prepared['aliases'][int(row['pts'])]
        grouped.setdefault(p,[]).append(row)
    return [(p,grouped[p]) for p in sorted(grouped)]
