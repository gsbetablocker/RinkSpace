from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
import sys,json,argparse
R=Path('D:/OneDrive/codex_projs/RinkSpace');sys.path[:0]=[str(R/'three_frame_refit/program_tracking'),str(R),str(R/'src')]
from z190_secondary_nonzero_mount_full_source import prepare,changes,write,ref,IDENTIFIER,SCENE
from shared_F_scene_candidate import load_candidate,rebind
from z190_source_camera_diagnostic import source_errors
from z190_moving_pivot_source_bundle import MovingPivotSourceBundle
from z190_low_static_foreground_rf_recovery import fixed_mount_rf_solve
from z190_profiled_motion_schur import SchurMotionKernel
from z190_low8_direct_source_rf_recovery import objective_direction_check
from z190_physical_camera import PhysicalCameraModel,CanonicalView
from z190_continuous_source_bundle import SCENE_UNIT_M,SOURCE_SHA
from continuous_central_brown4_optin import enable_projection_backend
import numpy as np,cv2
B=Path(__file__).resolve().parent
T=Path('D:/CodexData/.codex/visualizations/2026/10/02/private-thread-00009/RinkSpace_tmp')
CAM=T/'z190_full_programme_depth_profile_low8_v1/z190_full_programme_camera_candidate.npz'
AUTH=T/'z190_secondary_source_factor_authority_v1/source_authority_rows.json'

def prepare_full_secondary_core_source(output,candidate_scene_path=None):
    O=Path(output).resolve()
    if not O.is_relative_to(B.parent.resolve()):raise ValueError('Bulk source output must stay in this round temporary root')
    O.mkdir(parents=True,exist_ok=False);(O/'original_source_adapter').mkdir();cv2.setNumThreads(1);enable_projection_backend()
    if candidate_scene_path is None:
        original_scene=json.loads(SCENE.read_text(encoding='utf8'));scene=original_scene
        scene_binding=dict(baseline_scene=ref(SCENE),candidate_scene=ref(SCENE),F_column_dx_m=0.,all_four_F_houses_and_concentric_rings_share_translation=True,radii_row_Y_R_houses_Lane_other_paint_board_ads_and_metric_unchanged=True,accuracy_accepted=False)
    else:original_scene,scene,scene_binding=load_candidate(Path(candidate_scene_path),SCENE)
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
        assert aliases[row['pts']]==row['pts'];groups.setdefault(row['pts'],[]).append(row);icegroups.append(row)
    cut2packet=B/'secondary_core_ice_current_review_v1/new_cut2_ring_lane_full_source_gradient_qualified_v2.json';cut2=json.loads(cut2packet.read_text(encoding='utf8'));assert cut2['source_visual_inspection_completed'];cut2groups=[]
    for rec in cut2['records']:
        if rec['source_role']!='development':continue
        rr=rec['source_qualified_rows'];count=len(rr);row=dict(id=rec['physical_id']+'_source_qualified_native_transition',pts=rec['source_pts'],split='fit',source_role='new_development',world=np.array(rec['world_curve_xyz']),uv=np.array([r['source_uv']for r in rr]),normal=np.array([r['source_normal_uv']for r in rr]),normal_interval=np.array([r['normal_interval_px']for r in rr]),sigma=np.full(count,2.*np.sqrt(count)),point_correspondence=False,origin=str(cut2packet),sigma_modelling_decision=cut2['engineering_sigma_decision'])
        assert aliases[row['pts']]==row['pts'];groups.setdefault(row['pts'],[]).append(row);cut2groups.append(row)
    write(O/'new_cut2_ice_source_groups.json',dict(packet=ref(cut2packet),groups=cut2groups,qualified_points=sum(len(r['uv'])for r in cut2groups),new_development_exposures=sorted(set(r['pts']for r in cut2groups)),source_visual_inspection_completed=True))
    extentpacket=B/'secondary_core_ice_current_review_v1/whole_ring_extent_full_source_gradient_qualified.json';extent=json.loads(extentpacket.read_text(encoding='utf8'));extentgroups=[]
    for rec in extent['records']:
        if rec['source_role']!='development':continue
        assert rec['source_visual_inspection_completed'];row=dict(rec['source_row'])
        for k in('world','uv','normal','normal_interval','sigma'):row[k]=np.array(row[k])
        assert aliases[row['pts']]==row['pts'];groups.setdefault(row['pts'],[]).append(row);extentgroups.append(row)
    write(O/'whole_ring_extent_source_admission.json',dict(packet=ref(extentpacket),development_groups=extentgroups,held_extent_source_not_fit=[r for r in extent['records']if r['source_role']!='development'],source_gradient_supports_uses_complete_ROI=True,source_read_not_camera_derived=True,earlier_peak_plusminus6_intervals_replaced_preserving_history=True))
    # Rebind every admitted original/new factor to the same physical F-column scene.
    # The original authority files, source UV, normal supports, role and sigma stay immutable.
    binding_ledger=[]
    for p,rr in groups.items():
        rebound=[]
        for row in rr:
            moved=rebind(row,original_scene,scene)
            for field in ('uv','normal','normal_interval','sigma'):
                if field in row and not np.array_equal(row[field],moved[field]):raise ValueError('Source fields changed:'+field)
            if row.get('split')!=moved.get('split') or row.get('source_role')!=moved.get('source_role'):raise ValueError('Source role changed')
            if moved is not row:binding_ledger.append(dict(pts=row['pts'],id=row['id'],world_before=row['world'],world_candidate=moved['world']))
            rebound.append(moved)
        groups[p]=rebound
    bound_rows={role:[rebind(row,original_scene,scene)for row in rr]for role,rr in rows.items()}
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
    return dict(bundle=bundle,q=q,current=current,views=views,native=native,aliases=aliases,original_source_rows=rows,scene_source_rows=bound_rows,baseline_scene=original_scene,scene=scene,scene_binding=scene_binding,input_manifest=report)
