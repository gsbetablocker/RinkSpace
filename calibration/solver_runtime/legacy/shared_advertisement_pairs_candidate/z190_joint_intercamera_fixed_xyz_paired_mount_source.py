"""Complete programme source with directly observed fixed camera targets.

The physical installations each have one constant XYZ and nonzero lever.
This isolated candidate also estimates three shared physical print-pair station/width/Z/height charts alongside one shared camera-local lever XYZ plus constant PPx/PPy/k1 increments for each of the
three physical installations, while retaining the complete original sources. Numeric rink relationships remain common. Every
actual exposure retains R3/logf and its original source-motion graph; fresh
main4 source is prepared from production, never from its rejected candidate.
"""
from __future__ import annotations
import argparse,io,json,hashlib,os
from pathlib import Path
from types import SimpleNamespace
import sys
ROOT=Path('D:/OneDrive/codex_projs/RinkSpace');sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2,numpy as np
from scipy.sparse import csr_matrix,vstack,hstack
from z190_joint_rink_midpoint_source import prepare_joint as prepare_original
from z190_joint_shared_F_height_source import N,SCENE,main_input
from z190_joint_fixed_xyz_rink_block import FixedXYZRinkBlock
from z190_joint_rink_dimensions_source import main_held
from z190_secondary_shared_board_height_source import merge_upper_cap_source
from z190_secondary_connected_cap_source import connected_cap_absolute_value_jac15
from z190_secondary_confirmed_nearboard_source import NEAR_KIND
from z190_moving_pivot_source_bundle import MovingPivotSourceBundle
from z190_continuous_source_bundle import SCENE_UNIT_M,SOURCE_SHA
from z190_main_zoom_k1_source_information import read,write,ref,jsonable
from z190_physical_camera import PhysicalCameraModel,CanonicalView
from z190_joint_source_solver import solve_joint_source,make_profile_update
from z190_fixed_target_cross_view_factor import fixed_target_source_factor
from all_installation_optics_candidate.all_installation_optics_graph import (
    optics_diagnostics)
from shared_advertisement_pairs_candidate.paired_ad_graph import (
    append_paired_chart_optics_csr as append_installation_optics_csr)
from installation_mount_candidate.mount_graph import (
    evaluate_external_with_mount, bind_external_main_mount,
    insert_mount_csr, validate_mounts)

INSTALLATIONS={'long_side_main':0,'long_side_secondary':1,'low_end':2}
WARM=N/'low5_repair/full_low_major_ice_continuous_midpoint_v1/actual_source_reviews/0012/complete_low_in_joint_state_UNACCEPTED.npz'
DENSE=N/'low5_repair/persistent_ice_dense_source_v1/persistent_ice_z0_dense_source_QUALIFIED_FROZEN.json'


def upgrade_cap(prepared,*,absolute_evaluator=connected_cap_absolute_value_jac15):
    data=vars(prepared).copy();old=data['bundle']
    fit,held,changes,ledger=merge_upper_cap_source(data)
    groups=[(p,[changes.get(int(r['pts']),r)if r.get('curve_observation_kind')==NEAR_KIND else r for r in rows])for p,rows in old.abs_groups]
    sigma=np.concatenate([np.repeat(r['sigma'],2)for _,rows in groups for r in rows])
    base=SimpleNamespace(stamps=old.stamps,index=old.index,a=old.a,b=old.b,ia=old.ia,ib=old.ib,
        motion_sigma=old.motion_sigma,abs_groups=groups,native_residual_sigma=np.r_[old.motion_sigma,sigma])
    bundle=MovingPivotSourceBundle(base,old.calibration,old.motion_kernel,
        reference_pts=old.reference_pts,initial_reference_optical_height_scene=old.initial_reference_optical_height_scene,
        finite_planes=old.finite_planes,finite_records=old.finite_records,scene=old.scene,
        absolute_evaluator=absolute_evaluator)
    bundle.robust_phase=old.robust_phase;bundle.profile_weights=old.profile_weights
    data.update(bundle=bundle,confirmed_nearboard_fit_rows=fit,confirmed_nearboard_held_rows=held)
    return SimpleNamespace(**data),held,ledger


def prepare(args):
    if getattr(args,'cache_native_source_replay',False)and not getattr(args,'streaming_source',False):
        raise ValueError('Same-state native replay retention requires explicit streaming source')
    if getattr(args,'shared_ad1_14_geometry',False)and not(
            args.centrosymmetric_ads and getattr(args,'shared_lane_geometry',False)):
        raise ValueError('Shared Ad1/14 numeric panel requires explicit Lane q7 and user advertisement symmetry')
    args.output.mkdir(parents=True,exist_ok=False)
    original=prepare_original(args.output/'complete_original_sources',args.warm,args.low_temporal_source)
    geometry=original.geometry;ad_binding=None;ad_symmetry=None;ad_source=None
    if args.centrosymmetric_ads:
        from shared_centrosymmetric_advertisement_geometry import (
            build_centrosymmetric_ad_scene,AdvertisementSourceBinding,corner_xyz,audit_finite_advertisement_domains)
        from shared_rink_numeric_geometry_midpoint import SharedRinkNumericGeometryMidpoint
        oldscene=geometry.base;newscene,ad_symmetry=build_centrosymmetric_ad_scene(oldscene)
        ad_binding=AdvertisementSourceBinding(oldscene,newscene)
        geometry=SharedRinkNumericGeometryMidpoint(newscene)
    if args.ad_corner_source is not None:
        if not args.centrosymmetric_ads:raise ValueError('New Ad14 source needs the shared user-confirmed advertisement pairs')
        from z190_qualified_ad_corner_source import admit_native_ad_corner_intervals
        corners={role:corner_xyz(geometry.base,14,role)for role in ('left_top','right_top','right_bottom','left_bottom')}
        ad_fit,ad_held,ad_source=admit_native_ad_corner_intervals(args.ad_corner_source,original.aliases,corners,
            dict(TL='left_top',TR='right_top',BR='right_bottom',BL='left_bottom'),
            expected_observer='long_side_secondary',
            expected_sha256='34f60c5fbd7c7e3650f4b0b152c47c43575b3bdbd7b7ad12ba027c748371e798')
    ids=original.ids;slots={name:np.arange(5+3*i,8+3*i)for i,name in enumerate(ids)}
    qparts=[np.r_[original.q[:5],np.concatenate([
        np.r_[np.asarray(original.model.calibrations[name].pivot_xyz[:2])*SCENE_UNIT_M,
            original.q[original.slots[name]]]for name in ids])]]
    blocks=[];start=14;held=dict(original.held);cap_ledger=None;finite_ad_audits={}
    ring_qualification=None;ring_unknown=[]
    for oldblock in original.blocks:
        prepared=SimpleNamespace(**vars(oldblock.prepared))
        prepared.q=oldblock.local(original.q)
        if oldblock.name=='secondary':
            secondary_absolute=connected_cap_absolute_value_jac15
            qualification_path=getattr(args,'secondary_original_ring_source_qualification',None)
            if qualification_path is not None:
                from z190_secondary_original_ring_source_qualification import (
                    qualify_before_source_layout_rebuild,qualified_original_ring_absolute_value_jac15)
                prepared,ring_unknown,ring_qualification=qualify_before_source_layout_rebuild(prepared,qualification_path)
                secondary_absolute=qualified_original_ring_absolute_value_jac15
                write(args.output/'original_secondary_ring_source_qualification_admission.json',ring_qualification)
                write(args.output/'original_secondary_ring_unknown_historical_records_reportonly.json',ring_unknown)
            prepared,capheld,cap_ledger=upgrade_cap(prepared,absolute_evaluator=secondary_absolute)
            groups={int(p):[r for r in rows if r.get('curve_observation_kind')!=NEAR_KIND]for p,rows in held['secondary']}
            for row in capheld:groups.setdefault(int(row['pts']),[]).append(row)
            held['secondary']=sorted(groups.items())
        if ad_binding is not None:
            prepared.bundle.abs_groups=[(p,[ad_binding.rebind(r)for r in rows])for p,rows in prepared.bundle.abs_groups]
            held[oldblock.name]=[(p,[ad_binding.rebind(r)for r in rows])for p,rows in held[oldblock.name]]
            finite_ad_audits[oldblock.name]=audit_finite_advertisement_domains(oldscene,geometry.base,prepared.bundle.finite_records)
            prepared.bundle.scene=geometry.base
        if oldblock.name=='secondary'and ad_source is not None:
            groups={int(p):list(rows)for p,rows in prepared.bundle.abs_groups}
            for p,rows in ad_fit:
                if p not in prepared.bundle.index:raise ValueError('Ad14 source does not belong to secondary actual exposure inventory')
                groups.setdefault(p,[]).extend(rows)
            prepared.bundle.abs_groups=sorted(groups.items())
            groups={int(p):list(rows)for p,rows in held['secondary']}
            for p,rows in ad_held:groups.setdefault(p,[]).extend(rows)
            held['secondary']=sorted(groups.items())
            prepared.bundle.native_residual_sigma=np.r_[prepared.bundle.motion_sigma,
                np.concatenate([np.repeat(r['sigma'],2)for _,rows in prepared.bundle.abs_groups for r in rows])]
        block=FixedXYZRinkBlock(oldblock.name,prepared,geometry,
            slots[prepared.bundle.calibration.installation_id],start,
            absolute=secondary_absolute if oldblock.name=='secondary'else oldblock.absolute)
        blocks.append(block);qparts.append(block.q0[4:]);start+=len(block.q0)-4
    fresh=main_input(args.output/'fresh_main4_original_source',4,args.main4_source_admission_cache)
    main4held=main_held(fresh)
    if ad_binding is not None:
        fresh.bundle.abs_groups=[(p,[ad_binding.rebind(r)for r in rows])for p,rows in fresh.bundle.abs_groups]
        main4held=[(p,[ad_binding.rebind(r)for r in rows])for p,rows in main4held]
        finite_ad_audits['main4']=audit_finite_advertisement_domains(oldscene,geometry.base,fresh.bundle.finite_records)
        fresh.bundle.scene=geometry.base
    main4=FixedXYZRinkBlock('main4',fresh,geometry,slots[fresh.bundle.calibration.installation_id],start)
    if len(main4.bundle.stamps)!=800:raise ValueError('Fresh main4 actual exposure inventory changed')
    blocks.append(main4);qparts.append(main4.q0[4:]);held['main4']=main4held
    q=np.concatenate(qparts)
    if sum(len(b.bundle.stamps)for b in blocks)!=8503:raise ValueError('Complete programme actual inventory changed')
    bypts={int(p):b for b in blocks for p in b.bundle.stamps}
    if len(bypts)!=8503:raise ValueError('Actual source exposure duplicated across blocks')
    rows=[];authority=[]
    # Each source ledger normalizer is source-schema specific. It must retain
    # source qualification, raw hashes, roles, intervals and clipped bounds.
    from z190_intercamera_source_admission import load_fixed_panhead_sources
    for path in args.source_packets:
        accepted,manifest=load_fixed_panhead_sources(path,original.aliases,ids)
        rows.extend(accepted);authority.append(manifest)
    fit=[r for r in rows if r['split']=='fit'];seen=set()
    if not all(r['factor_source']['physical_target_authority']['whole_head_mechanical_origin_support']for r in rows):
        raise ValueError('Joint fixed rotation origins require whole pan/tilt head support, not collar-only locators')
    for r in fit:
        p=r['actual_pts'];observer=ids[INSTALLATIONS[r['observer_installation']]]
        if p not in bypts or bypts[p].bundle.calibration.installation_id!=observer:
            raise ValueError('Direct camera source observing installation disagrees with original cut')
        seen.update((observer,ids[INSTALLATIONS[r['target_installation']]]))
    if seen!=set(ids):raise ValueError('Direct fixed camera evidence must cover all three installations')
    external=None;external_start=len(q);external_contract=None
    if args.external_main_initial is not None:
        if not args.centrosymmetric_ads:raise ValueError('External Ad14 observer requires shared advertisement symmetry')
        from z190_external_main_low_source import prepare_external_main_low_source
        external=prepare_external_main_low_source(geometry.base,original.model.calibrations[ids[0]])
        with np.load(args.external_main_initial,allow_pickle=False)as saved:
            if not np.array_equal(saved['canonical_pts'],external.canonical_pts)or saved['q'].shape!=(external.local_size,):
                raise ValueError('External same-window actual exposure/private state inventory changed')
            eq=saved['q'].copy()
        e=external.evaluate(eq,q[slots[ids[0]]],q[slots[ids[2]]],jacobian=False)
        scales=np.ones(len(e.residual));quadratic=np.ones(len(e.residual),bool)
        sources={r['id']:r for r in (*external.upper_rows,*external.track_rows,*external.head_rows,*external.ad14_rows)}
        track_ids={r['id']for r in external.track_rows}
        for row in e.source_row_ledger:
            lo,hi=row['residual_slice'];source=sources.get(row['id'])
            if source is not None:
                sigma=source.get('source_sigma_native_px',source.get('sigma_native_px',source.get('source_sigma_px')))
                if sigma is None:raise ValueError('External source row lost its original native scale')
                scales[lo:hi]=sigma
            if row['id']in track_ids:quadratic[lo:hi]=False
        external_contract=(quadratic,scales,[r['id']for r in e.source_row_ledger])
        q=np.r_[q,eq]
    seed_contract=None;seed_pattern_q=None;seed_lane_q=None;seed_panel_q=None;seed_ring_qualification_sha=None
    if args.joint_state_seed is not None:
        with np.load(args.joint_state_seed,allow_pickle=False)as saved:
            if 'aux__original_ring_source_qualification_sha256'in saved:
                seed_ring_qualification_sha=str(saved['aux__original_ring_source_qualification_sha256'].item())
                if ring_qualification is None or seed_ring_qualification_sha!=ring_qualification['source']['sha256']:
                    raise ValueError('Joint seed original ring source qualification is not explicitly retained')
            pattern_count=9 if getattr(args,'shared_board_patterns',False)else 0
            panel_enabled=getattr(args,'shared_ad1_14_geometry',False)
            seed_has_panel=(panel_enabled and saved['q'].shape==(len(q)+pattern_count+6,))
            seed_has_lanes=(getattr(args,'shared_lane_geometry',False)and
                            (saved['q'].shape==(len(q)+pattern_count+2,)or seed_has_panel))
            expected=dict(block_start=np.asarray([b.start for b in blocks]),
                block_names=np.asarray([b.name for b in blocks]),
                block_lengths=np.asarray([len(b.q0)-4 for b in blocks]),
                canonical_pts=np.concatenate([b.bundle.stamps for b in blocks]),
                geometry_parameter_count=np.array(11 if seed_has_panel else 7 if seed_has_lanes else 5),
                installation_xyz_slots=np.asarray([slots[n]for n in ids]))
            if any('aux__'+key not in saved or not np.array_equal(saved['aux__'+key],value)
                    for key,value in expected.items()):
                raise ValueError('Joint seed source/block/canonical/shared-XYZ inventory changed')
            if external is not None:
                expected.update(external_start=np.array(external_start),external_stop=np.array(len(q)),
                    external_canonical_pts=external.canonical_pts)
                if any('aux__'+key not in saved or not np.array_equal(saved['aux__'+key],value)
                        for key,value in expected.items()):
                    raise ValueError('Joint seed external observer inventory changed')
            seed_has_patterns=(getattr(args,'shared_board_patterns',False)and
                saved['q'].shape in((len(q)+9,),
                    (len(q)+15,)if seed_has_panel else (len(q)+11,)if seed_has_lanes else (len(q)+9,)))
            if (saved['q'].shape!=q.shape and not(seed_has_patterns or seed_has_lanes or seed_has_panel))or not np.isfinite(saved['q']).all():
                raise ValueError('Joint seed must contain the complete finite source state')
            if seed_has_patterns:
                if ('aux__shared_pattern_start'not in saved or int(saved['aux__shared_pattern_start'])!=len(q)or
                        'aux__shared_pattern_chart'not in saved or
                        str(saved['aux__shared_pattern_chart'].item())!='actual_zero_thickness_perimeter_station_Z'):
                    raise ValueError('Joint seed shared print-material inventory/chart changed')
                seed_pattern_q=saved['q'][len(q):len(q)+9].copy()
            if seed_has_lanes:
                expected_indices=np.r_[np.arange(5),np.arange(len(q)+pattern_count,
                                      len(q)+pattern_count+(6 if seed_has_panel else 2))]
                if ('aux__geometry_parameter_indices'not in saved or
                        not np.array_equal(saved['aux__geometry_parameter_indices'],expected_indices)):
                    raise ValueError('Joint seed shared Lane material inventory changed')
                seed_lane_q=saved['q'][len(q)+pattern_count:len(q)+pattern_count+2].copy()
            if seed_has_panel:
                panel_start=len(q)+pattern_count+2
                if ('aux__shared_ad1_14_start'not in saved or int(saved['aux__shared_ad1_14_start'])!=panel_start or
                        'aux__shared_ad1_14_chart'not in saved or
                        str(saved['aux__shared_ad1_14_chart'].item())!='actual_closed_boundary_station_Z_print_panel'):
                    raise ValueError('Joint seed shared physical Ad1/14 panel chart changed')
                seed_panel_q=saved['q'][panel_start:panel_start+4].copy()
            q=saved['q'][:len(q)].copy()
            profiles={k[9:]:saved[k].copy()for k in saved.files if k.startswith('profile__')}
            make_profile_update(blocks).restore_checkpoint(profiles)
            seed_contract=(saved['quadratic'].copy(),saved['scales'].copy())
    state=SimpleNamespace(output=args.output,original=original,geometry=geometry,blocks=blocks,
        q=q,ids=ids,slots=slots,held=held,bypts=bypts,cross_rows=rows,cross_fit=fit,
        cross_source_authority=authority,cap_ledger=cap_ledger,seed_contract=seed_contract,
        original_ring_source_qualification=ring_qualification,seed_ring_qualification_sha=seed_ring_qualification_sha,
        original_ring_unknown_historical_report=ring_unknown,
        advertisement_symmetry=ad_symmetry,advertisement_corner_source=ad_source,finite_advertisement_domain_audits=finite_ad_audits,
        external=external,external_start=external_start,external_stop=len(q),external_contract=external_contract,
        patterns=None,geometry_parameter_indices=np.arange(5,dtype=int),lane_parameter_start=None,
        panel_parameter_start=None)
    if getattr(args,'shared_board_patterns',False):
        if not args.centrosymmetric_ads:raise ValueError('Shared print patterns require the user advertisement symmetry authority')
        from z190_joint_shared_board_pattern_adapter import JointSharedBoardPatterns
        state.patterns=JointSharedBoardPatterns(state,N/'low5_repair/secondary6_Ad13_Ad12_native_patterns_v1',
            initial_q=seed_pattern_q,initial_authority=None if seed_pattern_q is None else ref(args.joint_state_seed))
        state.q=np.r_[state.q,state.patterns.initial_q]
    if getattr(args,'shared_lane_geometry',False):
        from shared_rink_numeric_geometry_lanes import SharedRinkNumericGeometryLanes
        state.lane_parameter_start=len(state.q)
        state.geometry_parameter_indices=np.r_[np.arange(5),np.arange(len(state.q),len(state.q)+2)]
        state.geometry=SharedRinkNumericGeometryLanes(state.geometry.base)
        state.q=np.r_[state.q,np.zeros(2)if seed_lane_q is None else seed_lane_q]
        for block in state.blocks:block.set_shared_geometry(state.geometry,state.geometry_parameter_indices)
    if getattr(args,'shared_ad1_14_geometry',False):
        from shared_rink_numeric_geometry_lanes_ads import SharedRinkNumericGeometryLanesAds
        state.panel_parameter_start=len(state.q)
        state.geometry_parameter_indices=np.r_[state.geometry_parameter_indices,np.arange(len(state.q),len(state.q)+4)]
        state.geometry=SharedRinkNumericGeometryLanesAds(state.geometry.base)
        state.q=np.r_[state.q,state.geometry.panel.initial if seed_panel_q is None else seed_panel_q]
        for block in state.blocks:block.set_shared_geometry(state.geometry,state.geometry_parameter_indices)
    return state


def evaluate_external_state(state,q,*,jacobian=True,held=False,local_override=None):
    """Preserve all original external rows while sharing actual panel worlds."""
    local=q[state.external_start:state.external_stop]if local_override is None else local_override
    observer=q[state.slots[state.ids[0]]];target=q[state.slots[state.ids[2]]]
    if state.panel_parameter_start is None:
        return state.external.evaluate(local,observer,target,jacobian=jacobian,held=held)
    from shared_rink_numeric_geometry_lanes_ads import evaluate_external_shared_ad1_14
    start=state.panel_parameter_start
    return evaluate_external_shared_ad1_14(state.external,local,observer,target,
        state.geometry.panel,q[start:start+4],jacobian=jacobian,held=held)


# Keep the original implementation as a callback. The wrapper must not call
# itself recursively when the external main lens is perturbed.
_snapshot_evaluate_external_state=evaluate_external_state


def evaluate_external_state(state,q,*,jacobian=True,held=False,local_override=None):
    return evaluate_external_with_mount(state,q,jacobian=jacobian,held=held,
        local_override=local_override,evaluator=_snapshot_evaluate_external_state)


def mount_diagnostics(state,q):
    result={}
    for name,chart in state.installation_optics.items():
        block=next(b for b in state.blocks if b.bundle.calibration.installation_id==name)
        current=block.bundle.current_calibration(block.local(q))
        result[name]=dict(parameter_indices=list(chart.mount_indices),
            delta_lever_m=np.asarray(q)[list(chart.mount_indices)].tolist(),
            current_lever_m=(np.asarray(current.lever_xyz)*SCENE_UNIT_M).tolist(),
            current_calibration=current.manifest())
    return result


def run(args):
    state=prepare(args);blocks=state.blocks
    if (not hasattr(state,'installation_optics')or len(state.installation_optics)!=3 or
            len(state.optics_parameter_indices)!=9):
        raise ValueError('Explicit independent all-three-installation optics installation is required before driver/actors')
    if (not hasattr(state,'mount_contract') or len(state.mount_parameter_indices)!=9 or
            state.mount_parameter_indices.tolist()!=list(range(34195,34204)) or
            state.optics_parameter_indices.tolist()!=list(range(34204,34213)) or len(state.q)!=34225 or
            not hasattr(state,'paired_ad_contract') or
            state.paired_ad_parameter_indices.tolist()!=list(range(34213,34225)) or
            len(state.geometry_parameter_indices)!=23):
        raise ValueError('Explicit original34195 plus mount9/optics9/three paired-print geometry12 required')
    validate_mounts(state,state.q)
    from z190_parallel_source_evaluation import ordered_block_evaluations,make_parallel_profile_update
    source_workers=int(getattr(args,'source_workers',1))
    if source_workers<1:raise ValueError('Positive source worker count required')
    source_actors=None
    if getattr(args,'source_processes',False):
        from z190_process_source_evaluation import PersistentSourceActors
        source_actors=PersistentSourceActors(blocks,native_threads=int(getattr(args,'source_actor_threads',3)),q_columns=len(state.q))
        import atexit
        atexit.register(source_actors.close)
    profile_update=(make_parallel_profile_update(blocks,source_workers)if source_actors is None else source_actors.profile_update)
    state.source_actors=source_actors
    soft_board=getattr(args,'pivot_border_policy','outside-soft')=='outside-soft'
    board_prior_count=3 if soft_board else 0
    refs={name:next(b for b in blocks if b.bundle.calibration.installation_id==name)for name in state.ids}
    def cross(q,row,jacobian):
        p=row['actual_pts'];block=state.bypts[p]
        local=block.local(q);cal=block.bundle.current_calibration(local)
        observer=state.ids[INSTALLATIONS[row['observer_installation']]]
        target=state.ids[INSTALLATIONS[row['target_installation']]]
        index=block.start+4*block.bundle.index[p];view=q[index:index+4]
        result=fixed_target_source_factor(row['factor_source'],q[state.slots[observer]],q[state.slots[target]],
            view[:3],view[3],np.asarray(cal.lever_xyz)*SCENE_UNIT_M,
            np.array([1920*cal.principal_x_fraction[0],1080*cal.principal_y_fraction[0]]),cal.k1[0])
        if result.requires_offset_nuisance_in_solver:raise ValueError('This runner supports qualified fixed heads only')
        if not jacobian:return result.residual,result
        matrix=np.zeros((2,len(q)))
        matrix[:,state.slots[observer]]+=result.observer_xyz_m_jacobian
        matrix[:,state.slots[target]]+=result.target_xyz_m_jacobian
        matrix[:,index:index+4]+=result.observer_view_R3_logf_jacobian
        # The complete original source is reevaluated for this observer's
        # installation-wide lens coordinates; XYZ and R/f stay unchanged.
        try:
            for col,width,scale in zip(block.optics_chart.indices,
                    block.optics_chart.physical_steps,(1.,1.,.001)):
                step=float(width/scale)
                plus,minus=q.copy(),q.copy();plus[col]+=step;minus[col]-=step
                matrix[:,col]=(cross(plus,row,False)[0]-cross(minus,row,False)[0])/(2*step)
            for col in block.optics_chart.mount_indices:
                step=1e-7
                plus,minus=q.copy(),q.copy();plus[col]+=step;minus[col]-=step
                matrix[:,col]=(cross(plus,row,False)[0]-cross(minus,row,False)[0])/(2*step)
        finally:
            block.local(q)
        return result.residual,csr_matrix(matrix),result
    quadratic=np.r_[np.concatenate([b.quadratic for b in blocks]),
        np.full(2*len(state.cross_fit),args.physical_volume_loss=='quadratic',bool),
        np.empty(0,bool)if state.external is None else state.external_contract[0],
        np.empty(0,bool)if state.patterns is None else state.patterns.quadratic,np.ones(4+board_prior_count,bool)]
    scales=np.r_[np.concatenate([np.r_[b.bundle.native_residual_sigma,np.ones(len(b.bundle.stamps))]for b in blocks]),
        np.concatenate([r['factor_source']['sigma_native_px']for r in state.cross_fit]),
        np.empty(0)if state.external is None else state.external_contract[1],
        np.empty(0)if state.patterns is None else state.patterns.scales,np.ones(4+board_prior_count)]
    prefix_count=len(scales)-(0 if state.patterns is None else len(state.patterns.scales))-4-board_prior_count
    # A declared new pattern family may warm from the preceding identical
    # source state, with either the historical strict-border priors or the
    # latest three explicit soft-border rows. Preserve the matching family.
    seed_prior_count=4
    if state.seed_contract is not None and len(state.seed_contract[0])==prefix_count+4+board_prior_count:
        seed_prior_count+=board_prior_count
    original_quadratic=np.r_[quadratic[:prefix_count],np.ones(seed_prior_count,bool)]
    original_scales=np.r_[scales[:prefix_count],np.ones(seed_prior_count)]
    if state.seed_contract is not None:
        complete_match=(np.array_equal(quadratic,state.seed_contract[0])and
            np.array_equal(scales,state.seed_contract[1]))
        preceding_match=(np.array_equal(original_quadratic,state.seed_contract[0])and
            np.array_equal(original_scales,state.seed_contract[1]))
        declared_ring_match=False
        if state.original_ring_source_qualification is not None and state.seed_ring_qualification_sha is None:
            from z190_secondary_original_ring_source_qualification import restore_prequalification_contract
            secondary_index=next(i for i,b in enumerate(blocks)if b.name=='secondary')
            absolute_start=sum(len(b.quadratic)for b in blocks[:secondary_index])+blocks[secondary_index].bundle.motion_rows
            previous_q,previous_s=restore_prequalification_contract(quadratic,scales,absolute_start,state.original_ring_source_qualification)
            declared_ring_match=(np.array_equal(previous_q,state.seed_contract[0])and
                np.array_equal(previous_s,state.seed_contract[1]))
        if not(complete_match or preceding_match or declared_ring_match):
            raise ValueError('Joint seed normalized source masks/scales changed')
        state.declared_ring_seed_contract_transition_verified=declared_ring_match
    shared_schur_indices=np.r_[np.arange(14),
        np.empty(0,dtype=int)if state.patterns is None else np.arange(state.patterns.start,state.patterns.start+9),
        state.geometry_parameter_indices[5:],state.mount_parameter_indices,state.optics_parameter_indices]
    if len(set(shared_schur_indices.tolist()))!=59:
        raise ValueError('Retain old29 plus nine shared mount, nine shared optics and twelve shared print coordinates')
    if float(getattr(args,'initial_source_cost_fd_step',1e-7))!=1e-7:
        raise ValueError('This complete59 physical candidate uses the original geometry and mount source-cost FD width 1e-7')
    args.initial_source_cost_fd_step=1e-7
    args.initial_source_cost_fd_steps={int(col):1e-7 for col in shared_schur_indices
        if int(col)not in set(state.optics_parameter_indices.tolist())}
    for chart in state.installation_optics.values():
        args.initial_source_cost_fd_steps.update({int(col):float(width/scale)
            for col,width,scale in zip(chart.indices,chart.physical_steps,(1.,1.,.001))})
    # A numerical cost check may use a narrower source-confirmed local width.
    # This does not alter the residual, its Jacobian, or trial associations.
    for col,width in getattr(args,'source_cost_fd_width_overrides',{}).items():
        if (int(col)not in set(state.optics_parameter_indices.tolist())or
                not np.isfinite(float(width))or float(width)<=0):
            raise ValueError('Explicit positive lens cost-verification FD widths required')
        args.initial_source_cost_fd_steps[int(col)]=float(width)
    requested_fd=getattr(args,'initial_source_fd_columns',None)
    if requested_fd is None:
        initial_fd_columns=np.r_[np.arange(14),state.geometry_parameter_indices[5:],state.mount_parameter_indices,state.optics_parameter_indices]
        initial_pattern_fd_columns=(np.empty(0,dtype=int)if state.patterns is None
                                    else np.arange(state.patterns.start,state.patterns.start+9))
    else:
        requested_fd=np.asarray(requested_fd,dtype=int).reshape(-1)
        if (not len(requested_fd)or len(set(requested_fd.tolist()))!=len(requested_fd)or
                any(int(col)not in set(shared_schur_indices.tolist())for col in requested_fd)):
            raise ValueError('Initial source FD columns must be explicit distinct actual shared parameters')
        pattern_columns=set()if state.patterns is None else set(range(state.patterns.start,state.patterns.start+9))
        initial_fd_columns=np.asarray([col for col in requested_fd if int(col)not in pattern_columns],dtype=int)
        initial_pattern_fd_columns=np.asarray([col for col in requested_fd if int(col)in pattern_columns],dtype=int)
    def iter_evaluations(q,jacobian=True):
        from z190_parallel_pattern_evaluation import prefetch_pattern_frames
        prefetch_pattern_frames(state.patterns,q,jacobian)
        evaluations=(ordered_block_evaluations(blocks,q,jacobian,source_workers)if source_actors is None else source_actors.ordered_evaluations(q,jacobian))
        for b,result in evaluations:
            yield 'programme:'+b.name,result[0]if jacobian else result,result[1]if jacobian else None
            del result
        for row in state.cross_fit:
            result=cross(q,row,jacobian)
            yield 'physical_target:'+row['id'],result[0],result[1]if jacobian else None
            del result
        if state.external is not None:
            result=evaluate_external_state(state,q,jacobian=jacobian)
            if [r['id']for r in result.source_row_ledger]!=state.external_contract[2]:
                raise ValueError('External source residual family/order changed')
            if jacobian:
                prefix=np.zeros((len(result.residual),14))
                prefix[:,state.slots[state.ids[0]]]=result.observer_xyz_m_jacobian
                prefix[:,state.slots[state.ids[2]]]=result.target_xyz_m_jacobian
                matrix=hstack((csr_matrix(prefix),csr_matrix((len(result.residual),state.external_start-14)),
                    csr_matrix(result.local_jacobian),csr_matrix((len(result.residual),len(q)-state.external_stop))),format='csr')
                if state.panel_parameter_start is not None:
                    panelj=csr_matrix(result.shared_panel_jacobian).tocoo()
                    matrix+=csr_matrix((panelj.data,(panelj.row,state.panel_parameter_start+panelj.col)),shape=matrix.shape)
                matrix=append_installation_optics_csr(matrix,result.optics_jacobian,
                    result.optics_parameter_indices,state.optics_contract['original_parameter_count'])
                matrix=insert_mount_csr(matrix,result.mount_jacobian,result.mount_parameter_indices,
                    state.mount_contract['original_parameter_count'])
            yield 'external_main_to_low',result.residual,matrix if jacobian else None
            del result
            if jacobian:del matrix,prefix
        if state.patterns is not None:
            result=state.patterns.evaluate(q,jacobian)
            yield 'shared_board_patterns',result[0]if jacobian else result,result[1]if jacobian else None
            del result
        for name in state.ids:
            b=refs[name];local=b.local(q);cal=b.bundle.current_calibration(local)
            camera,_,vj=cal.camera25_and_jacobians(b.bundle.view(local,b.bundle.reference_pts))
            value=np.array([(camera[2]*SCENE_UNIT_M-1.6)/.18])
            if jacobian:
                row=np.zeros((1,len(q)));row[0,state.slots[name][2]]=1/.18
                pos=b.start+4*b.bundle.index[b.bundle.reference_pts];row[0,pos:pos+4]=vj[2]*SCENE_UNIT_M/.18
                rotation=cv2.Rodrigues(np.asarray(camera[3:6]))[0]
                row[0,b.optics_chart.mount_indices]=rotation.T[2]/.18
            yield 'optical_height:'+name,value,csr_matrix(row)if jacobian else None
        mid,mj=state.geometry.midpoint_prior(q[state.geometry_parameter_indices])
        if jacobian:
            row=np.zeros((1,len(q)));row[:,state.geometry_parameter_indices]=mj
        yield 'common_row_midpoint',mid,csr_matrix(row)if jacobian else None
        if soft_board:
            from z190_fixed_pivot_board_side_support import fixed_pivot_board_side_support
            for name in state.ids:
                residual,localj,_=fixed_pivot_board_side_support(board_polygon_m,q[state.slots[name]],args.pivot_board_soft_sigma_m)
                if jacobian:
                    row=np.zeros((1,len(q)));row[:,state.slots[name]]=localj
                yield 'fixed_pivot_board_side:'+name,residual,csr_matrix(row)if jacobian else None
    def iter_source_blocks(q,jacobian=True):
        from z190_streaming_source_normal import SourceRowBlock
        cursor=0
        for family,residual,matrix in iter_evaluations(q,jacobian):
            stop=cursor+len(residual)
            if stop>len(scales):raise ValueError('Original source block appended uncontracted rows')
            yield SourceRowBlock(residual,matrix,quadratic[cursor:stop],scales[cursor:stop],
                dict(family=family,original_start=cursor,original_stop=stop,source_sha256=SOURCE_SHA))
            cursor=stop
            del residual,matrix
        if cursor!=len(scales):raise ValueError('Original source block inventory omitted rows')
    def evaluate(q,jacobian=True):
        # Compatibility only; actual streaming solving never retains this
        # complete concatenated Jacobian.
        result=list(iter_evaluations(q,jacobian));value=np.concatenate([r[1]for r in result])
        return(value,vstack([r[2]for r in result],format='csr'))if jacobian else value
    polygon=np.ascontiguousarray(np.asarray(state.geometry.base['ice_boundary']['points_xyz'],np.float32)[:,:2])
    board_polygon_m=np.asarray(state.geometry.base['ice_boundary']['points_xyz'],float)[:,:2]*SCENE_UNIT_M
    def guard(q):
        state.geometry.validate_parameters(q[state.geometry_parameter_indices])
        validate_mounts(state,q)
        if not np.isfinite(q).all():raise ValueError('Finite common installation/rink/view state required')
        for name in state.ids:
            xyz=q[state.slots[name]]
            if xyz[2]<=0 or (not soft_board and cv2.pointPolygonTest(polygon,tuple(map(float,xyz[:2]/SCENE_UNIT_M)),False)>0):
                raise ValueError('Fixed tripod pivot must stay behind the physical rink boundary')
        for b in blocks:
            local=b.local(q);f=local[7::4]
            if np.any((f<np.log(100.))|(f>np.log(50000.))):raise ValueError('Physical focal domain')
            b.bundle.validate_finite_extents(b.bundle.cache(local,False))
        for r in state.cross_fit:cross(q,r,False)
        if state.external is not None:
            with bind_external_main_mount(state,q):
                views,_=state.external.unpack(q[state.external_start:state.external_stop])
                if np.any((views[:,3]<np.log(100.))|(views[:,3]>np.log(50000.))):raise ValueError('External physical focal domain')
                if state.panel_parameter_start is None:
                    result=state.external.guard(q[state.external_start:state.external_stop],q[state.slots[state.ids[0]]],q[state.slots[state.ids[2]]])
                else:
                    from shared_rink_numeric_geometry_lanes_ads import bind_external_ad1_14
                    start=state.panel_parameter_start
                    with bind_external_ad1_14(state.external,state.geometry.panel,q[start:start+4]):
                        result=state.external.guard(q[state.external_start:state.external_stop],q[state.slots[state.ids[0]]],q[state.slots[state.ids[2]]])
                if result['valid']is not True:raise ValueError('External shared observer source domain: '+result['reason'])
        if state.patterns is not None:state.patterns.geometry.guard(q[state.patterns.start:state.patterns.start+9])
    def diagnostics(q):
        from z190_fixed_pivot_board_side_support import fixed_pivot_board_side_support
        return dict(shared_geometry_metres=q[state.geometry_parameter_indices].tolist(),fixed_pivot_xyz_m={n:q[state.slots[n]].tolist()for n in state.ids},
            fixed_pivot_board_side={n:fixed_pivot_board_side_support(board_polygon_m,q[state.slots[n]],args.pivot_board_soft_sigma_m)[2]for n in state.ids},
            shared_print_pattern_parameters_m=None if state.patterns is None else q[state.patterns.start:state.patterns.start+9].tolist(),
            all_installation_shared_optics=optics_diagnostics(state,q),
            shared_installation_mounts=mount_diagnostics(state,q))
    review_cache={}
    cache_review=bool(getattr(args,'cache_complete_source_review',source_actors is not None))
    compact_review=bool(getattr(args,'compact_source_review',source_actors is not None))
    def review_write(path,payload):
        if compact_review:
            Path(path).write_text(json.dumps(payload,ensure_ascii=False,default=jsonable,separators=(',',':')),encoding='utf8')
        else:write(path,payload)
    def _source_event_profile_token():
        # Use the original solver v1 identity (prefix/key/dtype/shape/data).
        # This hashes the CURRENT profiles; it never freezes or changes them.
        from z190_joint_source_solver import _frozen_profile_token as solver_profile_token
        return solver_profile_token(profile_update)
    def source_event_state_contract(q):
        fullq=np.ascontiguousarray(q)
        return dict(q_dtype=fullq.dtype.str,q_shape=list(fullq.shape),
            q_sha256=hashlib.sha256(memoryview(fullq).cast('B')).hexdigest(),
            frozen_profile_sha256=_source_event_profile_token())
    def source_review(q,event=None):
        event_contract=source_event_state_contract(q)
        source_stat=Path(args.low_temporal_source).stat()
        key=(q.dtype.str,q.shape,q.tobytes(),source_stat.st_size,source_stat.st_mtime_ns)
        if cache_review and review_cache.get('key')==key:
            payload=dict(review_cache['payload'],source_event_state_contract=event_contract)
            scene=dict(review_cache['scene'],source_event_state_contract=event_contract)
            review_write(state.output/'latest_actual_source_review.json',dict(payload,event=event))
            review_write(state.output/'latest_candidate_scene_UNACCEPTED.json',scene)
            review_cache.update(payload=payload,scene=scene)
            return
        from z190_parallel_pattern_evaluation import prefetch_pattern_frames
        prefetch_pattern_frames(state.patterns,q,False)
        rows=[]
        if source_actors is None:
            for b in blocks:
                rows.extend(b.absolute_records(q));rows.extend(b.absolute_records(q,state.held[b.name]))
        else:
            for b,records in source_actors.ordered_absolute_records(q,state.held):rows.extend(records)
        direct=[]
        for r in state.cross_rows:
            try:
                _,result=cross(q,r,False)
                direct.append(dict(id=r['id'],native_pts=r['source_native_pts'],actual_pts=r['actual_pts'],split=r['split'],
                    observer=r['observer_installation'],target=r['target_installation'],uv_projected_native_px=result.projection.uv_native_px.tolist(),
                    outside_support_native_px=result.outside_support_native_px.tolist(),source_support=r['factor_source']['native_support_bounds_xy_px'],
                    no_body_center_treated_as_pivot=True))
            except ValueError as exc:direct.append(dict(id=r['id'],split=r['split'],domain_error=str(exc)))
        from z190_persistent_ice_source import persistent_ice_source_report
        low=next(b for b in blocks if b.name=='low')
        persistent=persistent_ice_source_report(low.bundle,low.local(q),args.low_temporal_source,state.original.aliases)
        external_review=None
        if state.external is not None:
            with bind_external_main_mount(state,q):
                eq,inference=state.external.infer_held_view_from_neighbours(q[state.external_start:state.external_stop])
                fit=evaluate_external_state(state,q,jacobian=False,local_override=eq)
                held=evaluate_external_state(state,q,jacobian=False,held=True,local_override=eq)
                external_review=dict(source_manifest=state.external.manifest,
                    fit_normalized_residual=fit.residual.tolist(),fit_source_ledger=fit.source_row_ledger,
                    held_normalized_residual=held.residual.tolist(),held_source_ledger=held.source_row_ledger,
                    held_view_inference=inference,programme_view_reused=False,
                    static_world_points_are_unknown_not_absolute_anchors=True)
        payload=dict(source_event_state_contract=event_contract,diagnostics=diagnostics(q),rows=rows,
            explicit_shared_mount_candidate=state.mount_contract,
            explicit_shared_print_pairs_candidate=state.paired_ad_contract,
            direct_fixed_camera_source=direct,persistent_ice_native_fit_and_held=persistent,
            external_main_to_low_source=external_review,
            original_secondary_ring_unknown_historical_records_reportonly=state.original_ring_unknown_historical_report,
            unknown_original_anchor_records_are_excluded_from_held_alignment_and_fit_statistics=True,
            shared_board_patterns=None if state.patterns is None else state.patterns.source_report(q),accuracy_accepted=False)
        review_write(state.output/'latest_actual_source_review.json',dict(payload,event=event))
        scene=state.geometry.scene(q[state.geometry_parameter_indices])
        scene['source_event_state_contract']=event_contract
        scene['explicit_shared_mount_candidate']=state.mount_contract
        scene['current_shared_mounts']=mount_diagnostics(state,q)
        scene['explicit_shared_print_pairs_candidate']=state.paired_ad_contract
        if state.patterns is not None:scene['shared_board_patterns']=state.patterns.descriptor(q)
        review_write(state.output/'latest_candidate_scene_UNACCEPTED.json',scene)
        if cache_review:review_cache.update(key=key,payload=payload,scene=scene)
    manifest=dict(schema='rinkspace.complete-programme.shared-print-pairs-mount-and-optics-source.v1',
        all_installation_shared_optics=state.optics_contract,
        shared_installation_mount_candidate=state.mount_contract,
        physical_parameter_count=34225,physical_shared_parameter_count=59,
        added_physical_shared_parameters=21,added_shared_mount_parameters=9,
        added_shared_print_geometry_parameters=12,
        shared_print_pairs_candidate=state.paired_ad_contract,
        statistical_parameters_not_installed_by_this_driver=True,
        original_source_rows_UV_sigmas_loss_roles_preserved=True,
        feasible_camera_parameter_family_changed_explicitly=True,
        measured_mount_or_hardware_search_range_claim=False,
        additional_mount_prior_rows=0,
        all_installation_shared_optics_parameter_names=[name for installation in state.ids
            for name in state.optics_contract['installations'][installation]['parameter_names']],
        initial_source_cost_fd_steps=args.initial_source_cost_fd_steps,
        source_video_sha256=SOURCE_SHA,warm=ref(args.warm),dense_ice_source=ref(args.low_temporal_source),
        direct_camera_source=state.cross_source_authority,shared_cap_upper_lower_source=state.cap_ledger,
        shared_user_centrosymmetric_advertisements=state.advertisement_symmetry,
        new_visible_Ad14_corner_source=state.advertisement_corner_source,
        shared_visible_print_patterns=None if state.patterns is None else state.patterns.manifest,
        original_finite_advertisement_domain_audits=state.finite_advertisement_domain_audits,
        additional_external_observer=None if state.external is None else dict(source=state.external.manifest,
            initialization=ref(args.external_main_initial),private_actual_exposures=38,private_parameters=state.external.local_size,
            same_fixed_main_and_low_XYZ=True,not_inserted_into_programme_output_inventory=True),
        actual_exposures=8503,blocks=[dict(name=b.name,actual=len(b.bundle.stamps),motion=len(b.bundle.a),
            original_input_manifest=b.prepared.manifest if hasattr(b.prepared,'manifest')else b.prepared.input_manifest)for b in blocks],
        original_secondary_ring_source_qualification=state.original_ring_source_qualification,
        declared_ring_seed_contract_transition_verified=getattr(state,'declared_ring_seed_contract_transition_verified',False),
        global_parameter_names=['Fdx_m','Rdx_m','pitch_delta_m','radius_delta_m','midpoint_delta_m',
            *[n+':pivot_'+a+'_m'for n in state.ids for a in 'xyz']],
        old_approximate_XY_is_initial_estimate_not_survey=True,one_fixed_XYZ_per_physical_installation=True,
        direct_source_is_whole_head_possible_origin_volume_not_exact_axis_measurement=True,
        physical_volume_excess_loss=args.physical_volume_loss,
        physical_volume_loss_reason='A qualified possible-origin volume has zero interior error; exceeding it cannot be explained as an alternative target identity. Original engineering sigma and source uncertainty volume are unchanged.',
        linear_strategy=args.linear_strategy,
        actual_shared_Schur_indices=shared_schur_indices.tolist(),
        initial_source_fd_columns=np.r_[initial_fd_columns,initial_pattern_fd_columns].tolist(),
        geometry_residual_fd_step_m=float(os.environ.get('RINKSPACE_GEOMETRY_FD_STEP_M','1e-5')),
        initial_source_cost_fd_step=float(getattr(args,'initial_source_cost_fd_step',1e-5)),
        initial_source_fd_selection_is_engineering_reuse_not_user_count_requirement=requested_fd is not None,
        unchanged_source_chain_preceding_actual_fd_evidence=None if getattr(args,'initial_source_fd_evidence',None)is None
            else ref(args.initial_source_fd_evidence),
        shared_rink_geometry=state.geometry.manifest(state.q[state.geometry_parameter_indices]),
        shared_ad1_14_numeric_geometry=state.panel_parameter_start is not None,
        shared_ad1_14_parameter_indices=None if state.panel_parameter_start is None else
            list(range(state.panel_parameter_start,state.panel_parameter_start+4)),
        geometry_parameter_indices=state.geometry_parameter_indices.tolist(),
        shared_material_parameters_are_global_not_per_exposure=True,
        pivot_border_policy=getattr(args,'pivot_border_policy','outside-soft'),
        pivot_board_soft_engineering_sigma_m=args.pivot_board_soft_sigma_m,
        user_border_clarification='Fixed panhead point is outside and approximately on the board. Permit the border; use a tolerant condition rather than an exact inside rejection against the working outline.',
        whole_Schur_step_backtracking=getattr(args,'schur_backtracking',False),
        numerical_schur_view_LM_ratio=getattr(args,'schur_view_lm_ratio',0.),
        numerical_view_LM_is_not_a_source_prior=True,
        numerical_schur_backtracking_aware_damping=getattr(args,'schur_backtracking_aware_damping',False),
        numerical_schur_accepted_damping_update_rule='lambda_times_0.3_for_full_step_otherwise_lambda_divided_by_accepted_alpha'
            if getattr(args,'schur_backtracking_aware_damping',False)else'original_lambda_times_0.3',
        numerical_damping_update_does_not_change_source_objective_or_priors=True,
        same_state_native_source_replay_retention=getattr(args,'cache_native_source_replay',False),
        native_source_replay_retention_scope='only generated normalized r/CSR from the current q, frozen profiles and original ordered source contract; released before trial costs',
        native_replay_does_not_copy_prepared_source_graph_or_reuse_initial_normal=True,
        source_stage_timing=getattr(args,'source_stage_timing',False),
        source_workers=source_workers,
        source_processes=source_actors is not None,
        initial_joint_source_state=None if args.joint_state_seed is None else ref(args.joint_state_seed),
        per_frame_camera_translation=False,nonzero_levers_metric_unchanged=False,
        one_nonzero_lever_shared_per_installation_reestimated=True,
        mount_lever_zoom_dependence=False,
        constant_PPxy_k1_variable_for_all_three_physical_installations=True,
        optics_shared_per_installation_not_per_frame=True,
        all_other_original_lens_fields_and_output_mapping_preserved=True,
        fresh_main4_source_from_production=True,old_rejected_main4_candidate_read=False,
        source_UV_roles_original_motion_materials_preserved=True,priors_once=dict(actual_roll=8503,optical_height=3,midpoint=1,fixed_pivot_board_side=board_prior_count),
        source_world_units_and_native_pixels_kept_distinct=True,body_rebound=False,production_changed=False,accuracy_accepted=False)
    resumed_accepted=getattr(args,'accepted_checkpoint_continuation',None)
    if resumed_accepted is not None:
        if args.joint_state_seed is None or not isinstance(resumed_accepted,dict):
            raise ValueError('Accepted checkpoint continuation needs explicit validated provenance')
        if not getattr(args,'streaming_source',False):
            raise ValueError('Accepted checkpoint continuation requires complete streaming source replay')
        manifest['accepted_checkpoint_continuation']=resumed_accepted
    write(state.output/'input_manifest.json',manifest);guard(state.q);source_review(state.q,dict(stage='initial_unaccepted'))
    streaming=getattr(args,'streaming_source',False)
    if streaming:
        from z190_streaming_source_normal import accumulate_source_normal,replay_source_cost
        token_hash=hashlib.sha256()
        for key,value in profile_update.iter_checkpoint_arrays():
            token_hash.update(key.encode());token_hash.update(memoryview(np.ascontiguousarray(value)).cast('B'))
        token=token_hash.hexdigest()
        initial_normal=accumulate_source_normal(lambda:iter_source_blocks(state.q,True),len(state.q),
            expected_row_count=len(scales),frozen_profile_token=token)
        if resumed_accepted is not None:
            expected=float(resumed_accepted['accepted_source_cost'])
            error=abs(initial_normal.cost-expected)
            if error>1e-8:
                raise ValueError('Accepted checkpoint q/profiles no longer replay its original source cost: '+str(error))
            write(state.output/'accepted_checkpoint_source_replay.json',dict(
                expected_accepted_source_cost=expected,actual_source_cost=initial_normal.cost,
                absolute_difference=error,full_source_rows=len(scales),
                original_fresh_start_FD_repeated=False,provenance=resumed_accepted))
        gradient=initial_normal.gradient
        def direction_cost(q):return replay_source_cost(lambda:iter_source_blocks(q,False),initial_normal.contract,frozen_profile_token=token)
        write(state.output/'initial_streaming_source_normal.json',initial_normal.report)
    else:
        residual,jac=evaluate(state.q,True)
        if residual.shape!=scales.shape:raise ValueError('Complete source and once-only prior inventory changed')
        weights=np.minimum(1.,2./np.maximum(abs(residual),1e-12));weights[quadratic]=1.
        gradient=np.asarray(jac.T@(weights*residual)).ravel()
        def direction_cost(q):return source_cost(evaluate(q,False),quadratic)
    checks=[]
    from z190_joint_rink_midpoint_source import source_cost
    for col in ([]if resumed_accepted is not None else initial_fd_columns):
        width=float(args.initial_source_cost_fd_steps[int(col)])
        if not np.isfinite(width)or width<=0:raise ValueError('Positive finite initial source cost FD step required')
        plus,minus=state.q.copy(),state.q.copy();plus[col]+=width;minus[col]-=width
        fd=(direction_cost(plus)-direction_cost(minus))/(2*width)
        error=abs(fd-gradient[col])/max(1.,abs(fd),abs(gradient[col]))
        checks.append(dict(column=int(col),step=width,analytic=float(gradient[col]),source_fd=float(fd),relative_error=float(error)))
        print(json.dumps(dict(stage='actual_complete_intercamera_global_FD',**checks[-1])),flush=True)
    if state.patterns is not None and resumed_accepted is None:
        # The new nine shared coordinates occur only in this declared
        # pattern family. Check their true source cost directly; reevaluating
        # millions of unchanged motion rows would add no derivative evidence.
        for col in initial_pattern_fd_columns:
            width=float(getattr(args,'initial_source_cost_fd_step',1e-5));plus,minus=state.q.copy(),state.q.copy();plus[col]+=width;minus[col]-=width
            rp=state.patterns.evaluate(plus,False);rm=state.patterns.evaluate(minus,False)
            fd=(float(rp@rp)-float(rm@rm))/(4*width)
            error=abs(fd-gradient[col])/max(1.,abs(fd),abs(gradient[col]))
            checks.append(dict(column=int(col),step=width,analytic=float(gradient[col]),source_fd=float(fd),relative_error=float(error),
                family='same_shared_advertisement_print_pattern'))
            print(json.dumps(dict(stage='actual_shared_print_pattern_source_FD',**checks[-1])),flush=True)
    if resumed_accepted is None:
        write(state.output/'actual_global_source_cost_direction_checks.json',checks)
        if max(c['relative_error']for c in checks)>.005:raise ValueError('Complete source XYZ/rink gradient failed')
    if streaming:del initial_normal
    else:del residual,jac
    if not args.solve:return state
    checkpoint_arrays=dict(block_start=np.asarray([b.start for b in blocks]),block_lengths=np.asarray([len(b.q0)-4 for b in blocks]),
        block_names=np.asarray([b.name for b in blocks]),canonical_pts=np.concatenate([b.bundle.stamps for b in blocks]),
        geometry_parameter_count=np.array(len(state.geometry_parameter_indices)),
        geometry_parameter_indices=state.geometry_parameter_indices,
        installation_xyz_slots=np.asarray([state.slots[n]for n in state.ids]),
        shared_scene_json=np.array(json.dumps(state.geometry.base,ensure_ascii=False)))
    if state.external is not None:
        checkpoint_arrays.update(external_start=np.array(state.external_start),external_stop=np.array(state.external_stop),external_canonical_pts=state.external.canonical_pts)
    if state.patterns is not None:
        checkpoint_arrays.update(shared_pattern_start=np.array(state.patterns.start),
            shared_pattern_chart=np.array('actual_zero_thickness_perimeter_station_Z'))
    if state.panel_parameter_start is not None:
        checkpoint_arrays.update(shared_ad1_14_start=np.array(state.panel_parameter_start),
            shared_ad1_14_chart=np.array('actual_closed_boundary_station_Z_print_panel'))
    if state.original_ring_source_qualification is not None:
        checkpoint_arrays.update(original_ring_source_qualification_sha256=np.array(state.original_ring_source_qualification['source']['sha256']),
            original_ring_source_qualification_json=np.array(json.dumps(state.original_ring_source_qualification,ensure_ascii=False)))
    checkpoint_arrays.update(all_installation_optics_parameter_indices=state.optics_parameter_indices,
        all_installation_optics_chart=np.array('dPPx_native_px_dPPy_native_px_dk1_divided_0.001'),
        all_installation_optics_installation_order=np.asarray(state.ids),
        all_installation_optics_parameter_indices_by_installation=np.asarray([
            state.installation_optics[name].indices for name in state.ids]),
        all_installation_optics_contract_json=np.array(json.dumps(state.optics_contract,ensure_ascii=False)))
    checkpoint_arrays.update(shared_mount_parameter_indices=state.mount_parameter_indices,
        shared_mount_parameter_indices_by_installation=np.asarray([
            state.installation_optics[name].mount_indices for name in state.ids]),
        shared_mount_chart=np.array('camera_local_lever_m_increments'),
        shared_mount_contract_json=np.array(json.dumps(state.mount_contract,ensure_ascii=False)))
    fixed_external_held=[]if state.external is None else [state.external_start+4*state.external.index[p]+c
        for p in state.external.held_pts for c in range(4)]
    result=solve_joint_source(evaluate,state.q,quadratic,scales,state.output/'solver',guard,
        update_profiles=profile_update,checkpoint_callback=source_review,
        checkpoint_identity=dict(input_manifest=ref(state.output/'input_manifest.json')),
        checkpoint_arrays=checkpoint_arrays,
        diagnostic_values=diagnostics,
        global_schur_count=len(shared_schur_indices)if args.linear_strategy=='global-schur'else None,
        global_schur_indices=shared_schur_indices if args.linear_strategy=='global-schur'else None,
        source_block_factory=iter_source_blocks if streaming else None,
        fixed_parameter_indices=fixed_external_held,schur_backtracking=getattr(args,'schur_backtracking',False),
        schur_view_lm_ratio=getattr(args,'schur_view_lm_ratio',0.),
        schur_backtracking_aware_damping=getattr(args,'schur_backtracking_aware_damping',False),
        cache_native_source_replay=getattr(args,'cache_native_source_replay',False),
        source_stage_timing=getattr(args,'source_stage_timing',False))
    source_review(result.q,dict(stage='final_unaccepted',status=result.status,stop_reason=result.stop_reason))
    calibrations=dict(state.original.model.calibrations);views=dict(state.original.views)
    # Rebind every one of the six blocks at the final complete state. No
    # previous guard/review perturbation may determine the exported lens.
    exported_optics=optics_diagnostics(state,result.q)
    write(state.output/'final_all_installation_optics_UNACCEPTED.json',exported_optics)
    write(state.output/'final_shared_mounts_UNACCEPTED.json',mount_diagnostics(state,result.q))
    write(state.output/'final_shared_print_pairs_UNACCEPTED.json',
        state.geometry.pair_panels.manifest(result.q[state.paired_ad_parameter_indices]))
    for b in blocks:
        local=b.local(result.q);cal=b.bundle.current_calibration(local);calibrations[cal.installation_id]=cal
        for p in b.bundle.stamps:views[p]=CanonicalView(cal.installation_id,b.bundle.view(local,p))
    target=state.output/'complete_intercamera_fixed_XYZ_camera_UNACCEPTED.npz'
    PhysicalCameraModel(calibrations).save_npz(target,state.original.native,state.original.aliases,views,
        source_sha256=SOURCE_SHA,time_base='1/15360')
    write(state.output/'report.json',dict(status=result.status,stop_reason=result.stop_reason,seconds=result.seconds,
        camera=ref(target),scene=ref(state.output/'latest_candidate_scene_UNACCEPTED.json'),actual=8503,
        accuracy_accepted=False,full_video_generated=False,body_rebound=False))
    return state


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--warm',type=Path,default=WARM);parser.add_argument('--low-temporal-source',type=Path,default=DENSE)
    parser.add_argument('--source-packets',type=Path,nargs='+',required=True);parser.add_argument('--solve',action='store_true')
    parser.add_argument('--main4-source-admission-cache',type=Path)
    parser.add_argument('--physical-volume-loss',choices=('quadratic','huber'),default='quadratic')
    parser.add_argument('--linear-strategy',choices=('original','global-schur'),default='original')
    parser.add_argument('--joint-state-seed',type=Path)
    parser.add_argument('--centrosymmetric-ads',action='store_true')
    parser.add_argument('--ad-corner-source',type=Path)
    parser.add_argument('--external-main-initial',type=Path)
    parser.add_argument('--shared-board-patterns',action='store_true')
    parser.add_argument('--shared-lane-geometry',action='store_true')
    parser.add_argument('--shared-ad1-14-geometry',action='store_true')
    parser.add_argument('--secondary-original-ring-source-qualification',type=Path,
        help='Explicit original development R2/R3 source qualification update: preserve 29 outer transitions and report two unknown tips; default source is unchanged')
    parser.add_argument('--initial-source-fd-columns',type=int,nargs='+')
    parser.add_argument('--initial-source-fd-evidence',type=Path)
    parser.add_argument('--streaming-source',action='store_true')
    parser.add_argument('--cache-native-source-replay',action='store_true',
        help='Opt-in same-q/frozen-profile/original-contract r/CSR retention for one GN/native reference replay; released before trials, no prepared graph copy or initial normal reuse')
    parser.add_argument('--source-stage-timing',action='store_true',
        help='Opt-in perf_counter timings for profile, source normal, GN/reference/native replay and original trial source costs')
    parser.add_argument('--schur-backtracking',action='store_true')
    parser.add_argument('--schur-view-lm-ratio',type=float,choices=(0.,1.),default=0.,
        help='Opt-in numerical view-block LM before shared Schur elimination; zero keeps the existing path, no source prior is added')
    parser.add_argument('--schur-backtracking-aware-damping',action='store_true',
        help='Opt-in next numerical damping update: lambda*.3 after a full step, lambda/alpha after accepted whole-step backtracking; source cost and priors stay unchanged')
    parser.add_argument('--pivot-border-policy',choices=('outside-soft','strict'),default='outside-soft')
    parser.add_argument('--pivot-board-soft-sigma-m',type=float,default=.30)
    run(parser.parse_args())
