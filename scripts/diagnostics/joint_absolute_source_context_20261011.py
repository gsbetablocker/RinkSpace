"""Small current absolute-source factories; no programme motion graph.

This diagnostic owns no source observations and writes no images or models.
It reconstructs the current q8 absolute operators from the same saved inputs.
"""
from pathlib import Path
from types import SimpleNamespace, MethodType
from copy import deepcopy
from collections import defaultdict
from contextlib import contextmanager
import pickle
import json
import time

import audit_saved_absolute_target_response_20261011 as a
np=a.np
from calibration.solver_runtime.source_loss import (
    SourceLossWeights, weighted_source_cost, source_irls_weights, loss_weight_arrays)
from structured_programme_source_candidate.paired_programme_chart import PairedProgrammeCoordinateChart

METADATA=a.H/'joint55_exposure_existing_absolute_metadata_20261011.json'


@contextmanager
def existing_material_source_byte_alias():
    """Reuse one authenticated existing PNG, scoped to a diagnostic loader.

    The original source row/path/SHA remains frozen.  Its original loader
    still reads the returned file and checks that literal SHA independently.
    No pixels are decoded, generated or copied by this adapter.
    """
    from structured_programme_source_candidate import native_board_material_point_source as material_module
    old_reference=material_module._native_frame_reference
    old_png=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp/history/tracking_runs/continuous_camera_model_v15_joint_f2_source_seven_review_once/main_104_066667_pts230708224_ORIGINAL_SOURCE.png').resolve()
    same_png=(a.H/'source_pixel_review_additional/board_top_contact_material_semantics/original_farCONTACT_full_RGB_scan/230708224_RAW_NATIVE.png').resolve()
    expected_png='1eacf19e87d8560ef4b9edd9a17a4beb5f77db6fa2937f092863d57d9854e575'
    if a.sha(same_png)!=expected_png:
        raise ValueError('Existing Ad26 exact-byte source alias SHA changed')
    aliases=[]
    def existing_same_byte_reference(row):
        original_path,expected=old_reference(row)
        if original_path!=old_png:return original_path,expected
        if expected!=expected_png:raise ValueError('Frozen Ad26 source SHA changed')
        aliases.append(dict(original_path=str(original_path),actual_read_path=str(same_png),expected_sha256=expected,
            exact_original_PNG_bytes_reused=True,source_packet_reference_unchanged=True,
            independently_authenticated_alias_sha256=expected_png))
        return same_png,expected
    material_module._native_frame_reference=existing_same_byte_reference
    try:
        yield aliases
    finally:
        material_module._native_frame_reference=old_reference


def load_context():
    started=time.perf_counter()
    options=a.read(a.CONFIG)['source_options'];manifest=a.read(a.RUN/'input_manifest.json')
    directory=a.RUN/'accepted_states'/a.CHILD;identity=a.read(directory/'state_identity.json')
    path=directory/'complete_state_UNACCEPTED.npz'
    assert a.sha(path)==identity['state_npz_sha256']
    keys=('q','scales','quadratic','aux__canonical_pts','aux__block_lengths','aux__block_start','aux__block_names',
          'aux__installation_xyz_slots','aux__current_shared_mount_parameter_indices',
          'aux__current_all_installation_optics_parameter_indices','aux__correlated_roll_prior_contract_json',
          'aux__current_complete_parameter_layout_json','aux__shared_pattern_start','aux__all_installation_optics_contract_json',
          'aux__external_start','aux__external_stop','aux__external_canonical_pts')
    with np.load(path,allow_pickle=False)as z:arrays={k:z[k].copy()for k in keys}
    provider,world,proof=a.current_complete_source_geometry(path,options['--world-init'])
    # Exact public production acceleration: avoid rebuilding the whole export
    # scene/grid/advertisements for each original ice-material FD coordinate.
    from structured_programme_source_candidate.current_ice_material_scene import install_source_ice_material_view
    ice_material_view=install_source_ice_material_view(provider,max_cache_bytes=32*1024**2)
    assert a.sha(identity['camera_path'])==identity['camera_sha256']
    with np.load(identity['camera_path'],allow_pickle=False)as z:cameras={k:z[k].copy()for k in z.files}
    state,camera_error=a.state_camera_adapter(arrays,cameras,provider)
    q=state.evaluation_q(arrays['q']);world_indices=state.geometry_parameter_indices
    scope=manifest['additional_real_native_board_edges'].get('physical_course_scope_authority')
    if scope:state.current_user_physical_course_scope_source=scope['path']
    board=a.NativeBoardEdgeFactor(state,options['--additional-board-source'],len(q))
    ice=a.NativeIceCurveFactor(state,options['--additional-ice-source'],len(q))
    scene=a.read(directory/'scene_UNACCEPTED.json')
    pattern_base=a.SharedPerimeterBoardPatternGeometry(provider.current.base,provider.scene_unit_m,
        facet_authority=scene['shared_board_patterns']['facet_authority'])
    pattern_original=a.SharedBoardPatternSourceFactor(pattern_base)
    owner=SimpleNamespace(start=int(arrays['aux__shared_pattern_start']),state=state,geometry=pattern_base,
        factor=pattern_original,external_factor=pattern_original,_original_course_pattern_geometry=pattern_base,
        _original_course_pattern_factor=pattern_original,_original_course_external_pattern_factor=pattern_original,
        course_world_geometry=provider,fit=[],held=[])
    owner.bind_current_geometry=MethodType(a.CurrentPatterns.bind_current_geometry,owner);state.patterns=owner
    prefix_count=len(q)-4
    with existing_material_source_byte_alias() as source_auth_aliases:
        material=a.NativeBoardMaterialPointFactor(state,options['--additional-board-material-point-source'],prefix_count)
    empty=SimpleNamespace(count=prefix_count,scales=np.empty(0),quadratic=np.empty(0,bool),contract={},updater=None,
        blocks=lambda value,jacobian:iter(()),guard=lambda value:provider.validate_parameters(value[world_indices]))
    placement=a.SourceWithIndependentPatternPlacements(a.SourceWithAdditionalNativeBoardMaterialPoints(empty,material),state,
        'D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/model_constraints/user_advertisement_pairs_20261005.json')
    assert np.array_equal(placement.extend_initial(q[:prefix_count],suffix=q[prefix_count:]),q)
    circle=a.NativeBoardCircleFactor(state,options['--additional-board-circle-source'],prefix_count-6)
    base=SimpleNamespace(count=len(q),scales=np.empty(0),quadratic=np.empty(0,bool),contract={},
        native_board_factor=board,native_ice_factor=ice)
    integration=a.SourceWithNativeBaseCurveIntegration(base,options['--native-base-curve-integration-policy'])
    human=a.HumanStrokeTruth(state,options['--direct-human-truth-authority'],board,ice,parameter_count=len(q))
    human.bind_full_stroke_guard(options['--direct-human-full-stroke-guard'])
    human.bind_stroke_cap_support(options['--direct-human-stroke-cap-support'])
    human.bind_practical_location_envelope(options['--practical-human-location-envelope'])
    centres=a.direct_manual_centre_factors(state,human,options['--additional-board-source'],options['--additional-ice-source'],len(q))
    assert a.sha(a.PACKET)==a.PACKET_SHA256
    packet=pickle.loads(a.PACKET.read_bytes());originals=packet['programme_absolute_rows']
    calls,call_proof=a.load_original_absolute_evaluators()
    corrections={r['original_packet_row_index']:r for r in a.read(options['--original-ice-pixel-correction-source'])['records']}
    retained={}
    for block in manifest['original_absolute_input_compaction']['blocks']:
        for row in block['row_maps']:retained[block['block'],row['group_order'],row['row_order']]=row['kept_original_point_indices']
    def current_original(index):
        item=originals[index];old=item['source_row'];row=deepcopy(old);change=corrections.get(index)
        if change:
            points=change['point_indices'];row['uv']=np.asarray(row['uv'],float).copy();row['uv'][points]=change['corrected_uv']
            if change.get('remove_original_normal_interval'):row.pop('normal_interval',None)
            elif change.get('corrected_normal_interval_px')is not None:
                row['normal_interval']=np.asarray(row['normal_interval'],float).copy();row['normal_interval'][points]=change['corrected_normal_interval_px']
        keep=retained.get((item['block'],item['group_order'],item['row_order']),list(range(len(old['uv']))))
        for key in ('uv','sigma','normal','normal_interval','source_uv'):
            if key in row and len(np.asarray(row[key]))==len(old['uv']):row[key]=np.asarray(row[key])[keep].copy()
        return row,np.asarray(keep,int)
    original_records=[];original_slots={}
    packet_ref=manifest['original_ice_curve_integration']['original_packet']
    for source in manifest['original_ice_curve_integration']['source_rows']:
        index=source['original_packet_row_index'];item=originals[index];row,keep=current_original(index)
        assert keep.tolist()==source['original_point_indices']
        assert a.hashlib.sha256(np.asarray(row['uv'],float).tobytes()).hexdigest()==source['current_active_UV_sha256']
        old=item['source_row'];stored=np.asarray(old['sigma'],float)
        working,prior,_=a.original_working_sigma(old,stored,source.get('original_declared_scan_group_count')or 0)
        name=f"programme:{item['block']}:{item['group_order']}:{item['row_order']}"
        original_records.append(dict(id=name,split='fit',actual_pts=int(item['group_actual_pts']),
            source_video_sha256=manifest['original_ice_curve_integration']['source_video_sha256'],
            physical_curve_id=json.dumps(source['physical_identity'],separators=(',',':')),
            operator_id=json.dumps({k:row.get(k)for k in ('curve_observation_kind','scanline_fixed_axis','scanline_branch','extent_side')},sort_keys=True),
            uv=np.asarray(row['uv'],float),stored_sigma_px=stored[keep],working_sigma_px=working[keep],prior_sampling_multiplier=prior[keep],
            point_integration_mass=a.isolated_anchor_masses(row['uv']),sampling_kind=['measured']*len(keep),source_anchor_identity_authority=packet_ref))
        original_slots[index]=name
    plan=a.compile_curve_sampling(original_records,duplicate_anchor_policy=manifest['original_ice_curve_integration']['duplicate_anchor_policy'])
    assert json.dumps(plan['groups'],sort_keys=True,default=a.plain)==json.dumps(manifest['original_ice_curve_integration']['groups'],sort_keys=True,default=a.plain)
    # Recreate unchanged original-board quadrature from existing detector
    # arrays. Corrected measured RGB anchors already declare singleton mass
    # in the authenticated installed receipt; do not decode PNGs again.
    from calibration.solver_runtime.original_board_curve_integration import (
        OriginalDetectorSupport,declared_working_sigma,reconcile_isolated_anchor_with_declared_trace)
    contract=manifest['original_board_curve_integration']
    for reference in [contract['policy_reference']]+contract['source_detector_references']+contract['authenticated_corrected_RGB_measurement_references']:
        assert a.sha(reference['path'])==reference['sha256']
    board_support=OriginalDetectorSupport(contract['source_video_sha256'],contract['source_station_gap_policy'])
    locators={(int(item['group_actual_pts']),item['group_order'],item['row_order']):index
        for index,item in enumerate(originals)if item['collection_role']=='FIT'}
    assert len(locators)==sum(item['collection_role']=='FIT'for item in originals)
    board_records=[];board_slots={}
    for source in contract['source_rows']:
        index=locators[(source['actual_pts'],source['group_order'],source['row_order'])]
        item=originals[index];row,keep=current_original(index);old=item['source_row']
        assert keep.tolist()==source['original_point_indices']
        assert a.hashlib.sha256(np.asarray(row['uv'],float).tobytes()).hexdigest()==source['current_active_UV_sha256']
        stored=np.broadcast_to(np.asarray(old['sigma'],float),(len(old['uv']),))
        working,prior,basis=declared_working_sigma(old,stored)
        assert basis==source['sigma_decomposition']
        if source['corrected_RGB_source_proof']is not None:
            proof=source['corrected_RGB_source_proof']
            assert proof['continuous_support_claim']is False and proof['excluded_anchor_cells_not_redistributed']is True
            mass=np.ones(len(keep))
        else:mass,_=board_support.masses(old,row['uv'],source['actual_pts'],original_point_indices=keep)
        assert abs(float(mass.sum())-source['source_support_mass'])<1e-9
        name=f"programme:{item['block']}:{item['group_order']}:{item['row_order']}"
        board_records.append(dict(id=name,split='fit',actual_pts=source['actual_pts'],
            source_video_sha256=contract['source_video_sha256'],physical_curve_id=source['physical_curve_id'],
            operator_id=str(row.get('curve_observation_kind','finite_curve_vector_distance')),uv=np.asarray(row['uv'],float),
            stored_sigma_px=stored[keep],working_sigma_px=working[keep],prior_sampling_multiplier=prior[keep],
            point_integration_mass=mass,sampling_kind=['measured']*len(keep),source_support_basis=source['support_basis'],
            source_anchor_identity_authority=contract['duplicate_anchor_policy']['source_anchor_identity_authority']))
        board_slots[index]=name
    reconciliation=reconcile_isolated_anchor_with_declared_trace(board_records)
    assert json.dumps(reconciliation,sort_keys=True,default=a.plain)==json.dumps(contract['shared_source_anchor_support_reconciliation'],sort_keys=True,default=a.plain)
    board_plan=a.compile_curve_sampling(board_records,duplicate_anchor_policy=contract['duplicate_anchor_policy'])
    assert json.dumps(board_plan['groups'],sort_keys=True,default=a.plain)==json.dumps(contract['groups'],sort_keys=True,default=a.plain)
    assert set(original_slots).isdisjoint(board_slots)
    compaction=manifest['original_absolute_input_compaction']
    assert a.sha(compaction['packet_path'])==compaction['packet_sha256']
    compact_source=a.read(compaction['packet_path']);compact_losses={};budgets=defaultdict(float)
    for record in compact_source['records']:
        locator=record['original_locator'];key=(locator['block'],locator['group_actual_pts'],record['source_identity'])
        budgets[key]+=sum(record['retained_point_integration_mass'])
    for record in compact_source['records']:
        index=record['original_entry_index'];row,keep=current_original(index);locator=record['original_locator']
        assert keep.tolist()==record['kept_original_point_indices']
        assert np.array_equal(row['uv'],np.asarray(record['retained_uv']))
        key=(locator['block'],locator['group_actual_pts'],record['source_identity'])
        compact_losses[index]=SourceLossWeights(np.repeat(np.asarray(record['retained_point_integration_mass'])/budgets[key],2),
            np.repeat(np.asarray(record['original_prior_sampling_multiplier'])[keep],2))
    assert set(compact_losses).isdisjoint(set(original_slots)|set(board_slots))
    trial=a.read(a.RUN/'actual_trial_family_cost_pass_0001.json')['families']
    ledgers=defaultdict(list)
    for stored in trial:ledgers[stored['original_row_ledger']['family']].append(stored)
    fixed=a.read(a.RUN/'solver/iterations.json')[0]['streaming_source_linearization']['fixed_parameter_indices']
    fixed_evaluation=state.canonical_parameter_layout.canonical_to_evaluation[np.asarray(fixed,int)]
    return SimpleNamespace(options=options,manifest=manifest,arrays=arrays,path=path,directory=directory,identity=identity,
        provider=provider,world=world,proof=proof,cameras=cameras,state=state,q=q,world_indices=world_indices,
        board=board,ice=ice,material=material,circle=circle,placement=placement,integration=integration,human=human,centres=centres,
        packet=packet,originals=originals,calls=calls,call_proof=call_proof,current_original=current_original,
        original_plan=plan,original_slots=original_slots,ledgers=ledgers,fixed_canonical=np.asarray(fixed,int),
        original_board_plan=board_plan,original_board_slots=board_slots,
        original_compaction_losses=compact_losses,
        fixed_evaluation=fixed_evaluation,camera_error=camera_error,source_auth_aliases=source_auth_aliases,load_seconds=time.perf_counter()-started)


def camera(c,value,at):
    return c.state.bypts[int(at)].bundle.camera(value,int(at))


def native_value(c,factor,value,rows,raw=False):
    cams={r['actual_pts']:camera(c,value,r['actual_pts'])for r in rows}
    if factor is c.material:
        with c.placement.pattern_placement_context(value):
            return factor._residual(value[:factor.parameter_count],[a.raw_row(r)for r in rows]if raw else rows,cameras=cams)
    if factor is c.circle:return factor._residual(value[:factor.parameter_count],[a.raw_row(r)for r in rows]if raw else rows,cameras=cams)
    if factor is c.board:return factor._residual(value,rows,cameras=cams,apply_optical_support=not raw)
    return factor._residual(value,[a.raw_row(r)for r in rows]if raw else rows,cameras=cams)


def wide_factory(c):
    from wide_anchor_candidate.wide_high_source_factor import WORLD_INDICES as old_world
    from wide_anchor_candidate.wide_high_free_optics_source_factor import WideHighFreeOpticsSourceFactor
    from structured_programme_source_candidate.current_wide_source import CurrentWideSource
    optics=json.loads(str(c.arrays['aux__all_installation_optics_contract_json'].item()))
    ids=json.loads(str(c.arrays['aux__correlated_roll_prior_contract_json'].item()))['installation_order'];secondary=ids[1]
    old_state=SimpleNamespace(q=c.arrays['q'][:34225],ids=ids,slots={secondary:np.array([8,9,10])},
        geometry=SimpleNamespace(parameter_count=23,scene=lambda unused:c.provider.current.scene(c.world)),
        geometry_parameter_indices=old_world,installation_optics={secondary:SimpleNamespace(mount_indices=np.array([34198,34199,34200]))},
        optics_contract=optics)
    old=WideHighFreeOpticsSourceFactor(old_state)
    trace=a.R.parents[2]/'evidence/user_contours_20261004/edge1_2e2bd471877d/edge1_source_uv.json'
    # R is .../events/ankara_2026/calibration/runs/<run>; parents[2] is event.
    factor=CurrentWideSource(old,c.state,a.read(trace))
    return factor


def original_loss(c,index):
    if index in c.original_compaction_losses:return c.original_compaction_losses[index]
    if index in c.original_slots:info=c.original_plan['rows'][c.original_slots[index]]
    elif index in c.original_board_slots:info=c.original_board_plan['rows'][c.original_board_slots[index]]
    else:return None
    return SourceLossWeights(np.repeat(info['point_integration_weights'],2),np.repeat(info['working_normalization_multiplier'],2))


def original_quadratic(c,index):
    """Use each retained original row's authenticated production loss mask.

    Programme order is motion, compacted absolute FIT rows, then view-roll
    priors.  The saved mask, scales and ledger jointly certify the slices;
    absolute observations are not uniformly Huber rows.
    """
    cached=getattr(c,'_original_absolute_quadratic_masks',None)
    if cached is None:
        counts={str(name):int(length)//4 for name,length in zip(
            c.arrays['aux__block_names'],c.arrays['aux__block_lengths'])}
        if any(int(length)%4 for length in c.arrays['aux__block_lengths']):
            raise ValueError('Original programme RF4 layout changed')
        cached={}
        for block in c.manifest['original_absolute_input_compaction']['blocks']:
            name=block['block'];ledgers=c.ledgers['programme:'+name]
            if len(ledgers)!=1:raise ValueError('Unique saved programme family required')
            ledger=ledgers[0]['original_row_ledger']
            begin=int(ledger['original_start'])+2*int(block['motion_point_count'])
            end=int(ledger['original_stop'])-counts[name];cursor=begin
            if int(block['compact_start'])!=int(ledger['original_start']):
                raise ValueError('Absolute compaction and saved source ledger disagree')
            for row_index,item in enumerate(c.originals):
                if item['block']!=name or item['collection_role']!='FIT':continue
                row,keep=c.current_original(row_index);size=2*len(keep)
                stop=cursor+size
                if stop>end:raise ValueError('Original absolute row exceeds its saved source slice')
                sigma=np.broadcast_to(np.asarray(row['sigma'],float),(len(keep),))
                if not np.array_equal(c.arrays['scales'][cursor:stop],np.repeat(sigma,2)):
                    raise ValueError('Original absolute row order or stored sigma changed')
                mask=np.asarray(c.arrays['quadratic'][cursor:stop],bool).copy()
                mask.setflags(write=False);cached[row_index]=mask;cursor=stop
            if cursor!=end:raise ValueError('Absolute FIT rows do not exhaust the saved source slice')
        c._original_absolute_quadratic_masks=cached
    if index not in cached:raise ValueError('Retained original FIT row required for loss mask')
    return cached[index]


def original_value(c,index,value,raw=False):
    item=c.originals[index];row,_=c.current_original(index);at=int(item['group_actual_pts'])
    c.provider.register_row(item['source_row'],actual_pts=at,allow_previously_qualified_held_ad2_representation=item['collection_role']=='HELD')
    bound=c.provider.rebind(row,value[c.world_indices])
    return np.asarray(c.calls[item['block']](camera(c,value,at),[a.raw_row(bound)]if raw else[bound],False))


def baseline(c,selected_pts):
    started=time.perf_counter();checks=[];groups=[]
    declarations=[('additional_real_native_ice_curves',c.ice),('additional_real_native_board_edges',c.board),
        ('additional_real_native_board_material_points',c.material),('additional_real_native_board_circles',c.circle)]
    for family,factor in declarations:
        view=factor._qualified_active_fit_view()if hasattr(factor,'_qualified_active_fit_view')else None
        view=factor if view is None else view
        payload=c.integration._payloads.get(family)
        value=native_value(c,factor,c.q,view.rows)
        quadratic=np.ones(len(value),bool)
        cost=weighted_source_cost(value,quadratic,payload);saved=c.ledgers[family][0]['normalized_cost']
        assert abs(cost-saved)<2e-8,(family,cost,saved)
        ledger=c.ledgers[family][0]['original_row_ledger'];start=ledger['original_start'];stop=ledger['original_stop']
        assert np.array_equal(c.arrays['scales'][start:stop],view.scales)
        assert np.array_equal(c.arrays['quadratic'][start:stop],quadratic)
        checks.append(dict(family=family,full_family_scalar_rows=len(value),actual_cost=cost,saved_cost=saved,error=cost-saved))
        weights,norm=loss_weight_arrays(payload,len(value));offset=0
        for row in view.rows:
            count=2*len(row['uv']);lo,hi=offset,offset+count;offset=hi
            if row['actual_pts']not in selected_pts:continue
            loss=None if weights is None else SourceLossWeights(weights[lo:hi],None if norm is None else norm[lo:hi])
            groups.append(dict(family=family,actual_pts=int(row['actual_pts']),id=row['id'],physical_group=a.selected_curve(row),
                row=row,factor=factor,source_kind='native',scalar_rows=count,scales=np.repeat(row['sigma'],2),
                raw=native_value(c,factor,c.q,[row],raw=True),actual=value[lo:hi].copy(),quadratic=np.ones(count,bool),loss=loss))
    family='direct_manual_sparse_correlated_centre_secondary'
    for ordinal,(factor,stored)in enumerate(zip(c.centres,c.ledgers[family])):
        value=factor._residual(c.q,factor.rows,cameras={at:camera(c,c.q,at)for at in factor.blocks})/factor._manual_centre_mean_scale
        cost=weighted_source_cost(value,np.ones(len(value),bool),None);saved=stored['normalized_cost']
        assert abs(cost-saved)<2e-8,(family,ordinal,cost,saved)
        checks.append(dict(family=family,ordinal=ordinal,full_family_scalar_rows=len(value),actual_cost=cost,saved_cost=saved,error=cost-saved))
        offset=0
        for row in factor.rows:
            count=2*len(row['uv']);lo,hi=offset,offset+count;offset=hi
            if row['actual_pts']not in selected_pts:continue
            mean=factor._manual_centre_mean_scale[lo:hi]
            groups.append(dict(family=family,ordinal=ordinal,actual_pts=int(row['actual_pts']),id=row['id'],physical_group=a.selected_curve(row),
                row=row,factor=factor,source_kind='manual_centre',mean_scale=mean,scalar_rows=count,
                scales=np.repeat(row['sigma'],2)*mean,raw=value[lo:hi].copy(),actual=value[lo:hi].copy(),quadratic=np.ones(count,bool),loss=None))
    original_checks=defaultdict(lambda:dict(cost=0.,rows=0,operator_rows=0))
    for index,item in enumerate(c.originals):
        if item['collection_role']!='FIT':continue
        row,keep=c.current_original(index)
        if not len(keep):continue
        at=int(item['group_actual_pts'])
        # Whole main1/main7 absolute sub-costs have saved same-profile receipts.
        if at not in selected_pts and item['block']not in ('main1','main7'):continue
        actual=original_value(c,index,c.q);loss=original_loss(c,index)
        quadratic=original_quadratic(c,index)
        if quadratic.shape!=actual.shape:raise ValueError('Original absolute scalar layout changed')
        cost=weighted_source_cost(actual,quadratic,loss)
        if item['block']in ('main1','main7'):
            proof=original_checks[item['block']];proof['cost']+=cost;proof['rows']+=len(actual);proof['operator_rows']+=1
        if at not in selected_pts:continue
        binding=c.provider._binding(item['source_row'])[1]
        physical=binding.get('name')or'+'.join('Ad'+str(n)for n in sorted({n for n,role in binding.get('ids',[])}))or binding['kind']
        groups.append(dict(family='original_absolute:'+item['block'],actual_pts=at,id=row['id'],physical_group=physical,
            packet_row_index=index,row=row,source_kind='original',scalar_rows=len(actual),scales=np.repeat(row['sigma'],2),
            raw=original_value(c,index,c.q,raw=True),actual=actual,quadratic=quadratic,loss=loss))
    for block,proof in original_checks.items():
        receipt=a.read(a.RUN/f'actual_trial_{block}_source_row_cost_pass_0001.json')
        saved=next(r['actual_after_cost']for r in receipt['actual_source_partition']if r['family']=='absolute_original_FIT')
        # A mismatch is reported rather than silently treating a motion family
        # as an absolute-only baseline; admission semantics must be reconciled.
        checks.append(dict(family='whole_original_absolute:'+block,actual_cost=proof['cost'],saved_cost=saved,
            error=proof['cost']-saved,full_family_scalar_rows=proof['rows'],operator_rows=proof['operator_rows']))
    c.wide=wide_factory(c)
    core=c.state.structured_programme_coordinate_chart.core_count;private=np.asarray(c.state.structured_programme_coordinate_chart.high_indices,int)
    value=c.wide.evaluate(c.q[:core],c.q[private],False)[0]
    gold=np.asarray(a.read(c.directory/'source_review_UNACCEPTED.json')['wide']['source_normalized_residual'])
    assert value.shape==gold.shape==(5808,)
    maximum=float(abs(value-gold).max());assert maximum<1e-10,maximum
    weights=4*c.wide.cost_weights
    assert np.array_equal(np.rint(weights),weights)
    loss=SourceLossWeights(weights);cost=weighted_source_cost(value,c.wide.quadratic,loss)
    saved=c.ledgers['strong_high_wide_source'][0]['normalized_cost'];assert abs(cost-saved)<2e-8,(cost,saved)
    checks.append(dict(family='strong_high_wide_source',unique_scalar_rows=5808,
        numerical_quadrature_scalar_rows=int(weights.sum()),actual_cost=cost,saved_cost=saved,error=cost-saved,
        saved_full_residual_max_error=maximum,private_camera_columns=private.tolist()))
    c.groups=groups
    return dict(checks=checks,groups=len(groups),selected_programme_scalar_rows=sum(g['scalar_rows']for g in groups),
        native_and_manual_full_family_costs_exact=True,wide_full_saved_values_and_cost_exact=True,
        original_absolute_group_baseline_is_current_supported_operator=True,seconds=time.perf_counter()-started)
