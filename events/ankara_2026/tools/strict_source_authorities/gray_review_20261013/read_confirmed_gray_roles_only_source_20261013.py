"""Independent source-only successor. No old pixel or board producer runs.

The complete cached navigation query inventory remains available for competing
materials. Only the three corrected gray source roles receive fresh profiles;
F2/F3 are the two named physical outputs. row3 retains measured pending source
evidence and model display, without an independent graph entity or loss.
"""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,time
ROOT=Path(r'D:\OneDrive\codex_projs\RinkSpace');sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts import admit_programme_source_inputs as shared
import cv2
import numpy as np
from rinkspace.programme_source_stream import FrozenProgrammeSource
from rinkspace.programme_source_measurement_cache import ProgrammeSourceMeasurementCache,restore_exact_navigation_queries
from rinkspace.vision_programme_ice_review import IceProgrammeSourceReader
from rinkspace.vision_ice_source_role_override import effective_source_role_inventory,remeasure_confirmed_source_roles
from rinkspace.vision_ice_temporal_identity import TemporalIceIdentitySelector
from rinkspace.vision_ice_model_context_identity import SameStateIceModelContextIdentity
from rinkspace.vision_ice_single_gray_support_identity import ConfirmedSingleGraySupportModelContextIdentity,F3,DISPLAY_ROW3
from rinkspace.vision_ice_source_guides import ActualIceSourceGuideTracker
from rinkspace.vision_ice_final_precision_graph import PreciseSourceIceMaterialIdentityGraph
from rinkspace.vision_ice_local_clearance import KnownIceMaterialBoxClearance,primary_person_box_components
from rinkspace.native_alias_source_review import compare_material_candidates,check_native_alias_paint_precision
from rinkspace.source_state_spill import ExactSourceStateSpill
from calibration.program_tracking.strict_absolute_source import StrictAbsolutePackWriter,normalize_selected_ice_rows
from calibration.program_tracking.strict_model_scope import file_sha,authenticate_source_reference
from calibration.program_tracking.compact_source_json import pack_source_json,unpack_source_json

HERE=Path(__file__).resolve().parent
F2='house_connection_F2_R2:segment0'
OUTPUT_ROLES=(F2,F3)
MEASURED_ROLES=(DISPLAY_ROW3,F2,F3)
ROLE_FACT=ROOT/'events/ankara_2026/model_constraints/source_material_operator_corrections_straight_connections_20261012.json'
F3_FACT=ROOT/'events/ankara_2026/model_constraints/user_confirmed_F3_R3_B_source_identity_229842176_20261013.json'
PREDECESSOR=HERE/'model_context_pre_single_gray_support_20261013/vision_ice_model_context_identity_predecessor.py'
GUIDE_PREDECESSOR=HERE/'model_context_pre_single_gray_support_20261013/vision_ice_source_guides_predecessor.py'
def ref(p):return dict(path=str(Path(p).resolve()),sha256=file_sha(p))
def plain(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:plain(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [plain(x) for x in v]
    return v
def literal(v):
    if isinstance(v,np.ndarray):return dict(dtype=v.dtype.str,shape=list(v.shape),bytes=v.tobytes().hex())
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:literal(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [literal(x) for x in v]
    return v
def save(p,v):
    with p.open('x',encoding='utf8') as f:json.dump(plain(v),f,ensure_ascii=False,indent=2)

class TwoPhysicalGrayOnlyPrecisionGraph(PreciseSourceIceMaterialIdentityGraph):
    def _nodes(self,rgb,measurement,mask):
        if getattr(self,'_prepared_final_nodes',None) is not None:return self._prepared_final_nodes
        curve=np.asarray(measurement['query_curve'])[np.asarray(measurement['candidate_query'])]
        allowed=np.flatnonzero(np.isin(np.asarray([r['curve_id'] for r in self.inventory])[curve],OUTPUT_ROLES))
        if hasattr(self,'current_field_order_prepared_candidate_ids'):
            allowed=allowed[np.isin(allowed,self.current_field_order_prepared_candidate_ids)]
        view=dict(measurement)
        for name in ('candidate_uv','candidate_sides','candidate_query','candidate_flags','candidate_side_quantiles_uv'):
            if name in view:view[name]=np.asarray(view[name])[allowed]
        frame=super()._nodes(rgb,view,mask)
        frame['ids']=allowed[frame['ids']]
        for key in ('source_pixel_accuracy_pending_candidate_ids',):
            if key in frame:frame[key]=allowed[np.asarray(frame[key],int)].tolist()
        self._latest_final_node_rejections={int(allowed[int(i)]):r for i,r in self._latest_final_node_rejections.items()}
        return frame

class MeasuredGrayBandSourceGuideTracker(ActualIceSourceGuideTracker):
    def _source_support_profile_half_extent_px(self,support):
        if support['curve_id'] not in OUTPUT_ROLES:return super()._source_support_profile_half_extent_px(support)
        center=np.asarray(support['uv'],float);sides=np.asarray(support['sides'],float)
        # A search band contains the actually measured two sides, the public
        # three-pixel RGB exterior samples, and the public eight-pixel source
        # exclusion context beyond each side. This is navigation only; it does
        # not change a precision/contrast/identity acceptance allowance.
        actual_half_width=float(np.max(np.linalg.norm(sides-center[:,None],axis=2)))
        return max(10.,actual_half_width+3.+8.)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--shot-id',default='223403520:R0402_037')
    parser.add_argument('--start',type=int);parser.add_argument('--stop',type=int);parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--compare-default-predecessor',action='store_true');args=parser.parse_args()
    dest=args.output_dir.resolve();dest.mkdir(exist_ok=True)
    output=dest/'confirmed_gray_roles_only_source.npz'
    if output.exists():raise ValueError('Preserve previous source outputs')
    old_manifest=HERE/'source_inventory/source_full_precision_spill_retry3_FROZEN_v1.json'
    old=json.loads(old_manifest.read_text(encoding='utf8'));common=old['common_arguments']
    def option(name):return Path(common[common.index(name)+1])
    programme=FrozenProgrammeSource(option('--identity'),option('--source'),ROOT/'calibration/program_tracking/source_shots.json')
    shot=next(s for s in programme.shots if s['id']==args.shot_id)
    start=args.start if args.start is not None else int(shot['start_pts']);stop=args.stop if args.stop is not None else int(shot['end_pts'])
    expected=programme.actual_inventory[(programme.actual_inventory>=start)&(programme.actual_inventory<stop)]
    native=programme.native_pts[(programme.native_pts>=start)&(programme.native_pts<stop)]
    if not len(expected):raise ValueError('Exact nonempty source window required')
    roles=shared.read_json(option('--roles')) if '--roles' in common else None
    cache=ProgrammeSourceMeasurementCache(option('--measurements'),source_sha256=programme.source_sha256,time_base=programme.time_base,expected_actual_pts=programme.actual_inventory)
    navigation=IceProgrammeSourceReader(programme.scene,source_sha256=programme.source_sha256,time_base=programme.time_base,role_contract=roles,profile_backend='cpu')
    if navigation.inventory!=cache.inventory:raise ValueError('Same model/query inventory must match')
    fact=shared.read_json(ROLE_FACT);overrides={}
    for i,r in enumerate(fact['records']):
        if r['curve_id'] not in MEASURED_ROLES:continue
        overrides[r['curve_id']]=dict(r,source_role_evidence=dict(ref(ROLE_FACT),source_record_index=i))
    effective=effective_source_role_inventory(cache.inventory,overrides)
    # All source query/model names still participate in field-order trees.
    # Non-requested roles have no fresh candidates and cannot be output here.
    context_inventory=[dict(r) for r in effective]
    context_kwargs=dict(state_identity_path=option('--identity'),navigation_authority=ref(option('--model-navigation-authority')),
        source_sha256=programme.source_sha256,time_base=programme.time_base)
    known_refs=[ref(common[i+1]) for i,v in enumerate(common) if v=='--known-source-material-exclusion']
    known=shared.load_known_source_material_exclusions(known_refs);context_kwargs['actual_source_negative_materials']=known
    model_context=ConfirmedSingleGraySupportModelContextIdentity(context_inventory,visible_F3_authority=ref(F3_FACT),**context_kwargs)
    source_F3_fact=shared.read_json(F3_FACT)
    anchor_path=Path(source_F3_fact['source_readout_reference']['path']);authenticate_source_reference(source_F3_fact['source_readout_reference'])
    with np.load(anchor_path,allow_pickle=False) as z:anchor_uv=z['actual_q50_uv'];anchor_binding=json.loads(str(z['source_binding_json'].item()))
    witness=shared.read_json(source_F3_fact['witness_reference']['path'])
    human_seed=dict(curve_id=F3,physical_id='house_connection_F3_R3',actual_pts=anchor_binding['actual_pts'],native_pts=anchor_binding['native_pts'],
        source_sha256=programme.source_sha256,time_base=programme.time_base,shot_id=anchor_binding['shot_id'],
        decoded_native_RGB_sha256=witness['decoded_native_RGB_sha256'],decoded_native_BGR_sha256=anchor_binding['decoded_native_BGR_sha256'],
        navigation_uv=anchor_uv.tolist(),source_material_class='static_painted_ice',surface_and_physical_identity_confirmed=True,
        authority=ref(F3_FACT),is_exact_centerline_pixel_ground_truth=False,
        source_seed_basis='explicit human naming of actually measured visible B stroke; fresh own RGB profiles still required')
    selector=TemporalIceIdentitySelector(effective,source_sha256=programme.source_sha256,time_base=programme.time_base,seed_registry=[human_seed],model_context_identity=model_context)
    graph=TwoPhysicalGrayOnlyPrecisionGraph(effective,source_sha256=programme.source_sha256,time_base=programme.time_base,source_path=str(programme.source_path),device='cpu')
    clearance=KnownIceMaterialBoxClearance(effective,source_sha256=programme.source_sha256,time_base=programme.time_base,seed_registry=())
    guide_tracker=MeasuredGrayBandSourceGuideTracker(effective,source_sha256=programme.source_sha256,time_base=programme.time_base,profile_backend='cpu')
    qualification_path=HERE/'gray_alias_qualified_source_q50_field_order_20261013_v1.json'
    qualified=json.loads(qualification_path.read_text(encoding='utf8'));guides_by_frame={}
    for record in qualified['records']:
        b=record['source_binding'];points=[c['actual_q50_uv'] for c in record['candidates']]
        if len(points)<2:continue
        source_image=HERE/'unresolved_straight_role_originals_20261012'/f"{b['actual_pts']}_RAW_NATIVE.png"
        if not source_image.exists():source_image=HERE/'037_named_house_topology_two_actual_20261012'/f"{b['actual_pts']}_EXACT_RAW_NATIVE.png"
        ownraw=cv2.imread(str(source_image))
        if ownraw is None or hashlib.sha256(ownraw.tobytes()).hexdigest()!=b['decoded_native_BGR_sha256']:raise ValueError('Source guide own original image differs')
        guide=dict(b,native_pts=int(b.get('native_pts',b['actual_pts'])),decoded_native_RGB_sha256=hashlib.sha256(ownraw[:,:,::-1].tobytes()).hexdigest(),
            curve_id=F3,navigation_uv=points,authority=ref(qualification_path),original_source_image_actually_reviewed=True,
            physical_identity_from_model_projection=False,normal_search_half_extent_px=18.,sampling_spacing_px=navigation.spacing,
            source_identity_is_not_admitted_by_this_guide=True)
        guides_by_frame[int(b['actual_pts'])]=[guide]
    default_previous=default_current=default_guide_previous=default_guide_current=None
    if args.compare_default_predecessor:
        spec=importlib.util.spec_from_file_location('rinkspace._gray_source_saved_predecessor',PREDECESSOR);oldmod=importlib.util.module_from_spec(spec);spec.loader.exec_module(oldmod)
        default_previous=oldmod.SameStateIceModelContextIdentity(context_inventory,**context_kwargs)
        default_current=SameStateIceModelContextIdentity(context_inventory,**context_kwargs)
        spec=importlib.util.spec_from_file_location('rinkspace._gray_source_saved_guides_predecessor',GUIDE_PREDECESSOR);oldguides=importlib.util.module_from_spec(spec);spec.loader.exec_module(oldguides)
        guide_kwargs=dict(source_sha256=programme.source_sha256,time_base=programme.time_base,profile_backend='cpu')
        default_guide_previous=oldguides.ActualIceSourceGuideTracker(effective,**guide_kwargs)
        default_guide_current=ActualIceSourceGuideTracker(effective,**guide_kwargs)
    inventory_paths=[Path(common[i+1]) for i,v in enumerate(common) if v=='--legacy-inventory']
    if not inventory_paths:inventory_paths=[option('--measurements').parent.parent/'source_inventory'/n for n in ('original_absolute_source_review_inventory.npz','additional_absolute_source_review_inventory.npz')]
    held=shared.LegacyHeldSourceRoles(inventory_paths)
    masks=shared.PersonMasks(shared.DEFAULT_MASKS,shared.DEFAULT_PEOPLE)
    plan=shared.read_json(option('--shot-plan'));plan=plan.get('shot_plan',plan)
    references={}
    for r in old['references']:
        r=dict(r);p=Path(r['path']).resolve()
        if p.is_relative_to(ROOT) and p.suffix=='.py':r['sha256']=file_sha(p)
        else:authenticate_source_reference(r)
        references[str(p)]=dict(r,path=str(p))
    for p in (Path(__file__),ROLE_FACT,F3_FACT,PREDECESSOR,GUIDE_PREDECESSOR,qualification_path,ROOT/'src/rinkspace/vision_ice_single_gray_support_identity.py',ROOT/'src/rinkspace/vision_ice_model_context_identity.py',ROOT/'src/rinkspace/vision_ice_source_guides.py'):
        r=ref(p);references[r['path']]=r
    manifest=dict(schema='rinkspace.frozen-confirmed-gray-role-only-source-successor.v1',predecessor_manifest_reference=ref(old_manifest),
        shot_id=args.shot_id,start_pts=start,stop_pts=stop,exact_actual_pts=expected.tolist(),exact_native_pts=native.tolist(),
        references=list(references.values()),old_materials_or_board_read_or_output=False,
        complete_query_inventory_retained_as_navigation_and_competition=True,
        measured_code_roles=MEASURED_ROLES,physical_output_roles=OUTPUT_ROLES,
        existing_gray_named_material_neighborhood_px=24.,gray_source_profile_search_window_is_not_pixel_acceptance_tolerance=True,
        source_window_basis='Existing 24px named-material search neighborhood; actual profile flanks/quantiles, competing stations, RGB continuity, scope and final precision remain independent.',
        native_aliases_own_RGB_profiles_required=True,whole_curve_physical_alias_or_endpoints_inferred=False)
    frozen=dest/'gray_role_only_FROZEN.json';save(frozen,manifest)
    source=dict(video_sha256=programme.source_sha256,time_base=programme.time_base,pixel_domain='native1920x1080')
    writer=StrictAbsolutePackWriter(output,source=source,expected_actual_pts=expected,original_references=[ref(frozen)],legacy_pixel_families=(),recovery_journal=dest/'source_admission_recovery.sqlite')
    # Reference the imported source and source facts explicitly, without
    # authenticating mutable Python behind an older successful pack.
    writer.original_references.extend(list(references.values()))
    spill=ExactSourceStateSpill(dest/'lossless_source_state')
    frame_records=[];alias_records=[];aliases=[];last=None;total=Counter();default_comparisons=0
    began=time.perf_counter();lastnotice=began
    def normalize(rows,measurement,binding):
        count=0
        for r in rows:
            if r['curve_id'] not in OUTPUT_ROLES:raise ValueError('Only canonical two physical gray roles can output')
            split=held.split(r,'fit');r=dict(r,source_origin_role=r.get('source_seed_reference',{}).get('source_original_split','fit'))
            result=normalize_selected_ice_rows([dict(r,split=split)],measurement,source_binding=binding,curve_inventory=effective,split=split)
            total['normalizer_rejected_rows']+=len(result['rejected_rows'])
            for row in result['rows']:
                row['vision_training_role']=shared.whole_shot_role(plan,binding);count+=writer.add_row(row,result['review'])
                total[row['physical_material_id']]+=len(row['uv'])
        return count
    def fresh(rgb,binding,template,mask):
        m={k:(v.copy() if isinstance(v,np.ndarray) else deepcopy(v)) for k,v in template.items() if k.startswith('query_') or k in ('curve_status','source_readout_quantization_step_px','source_interpolation_grid_px')}
        m.update(source_binding=binding,decoded_native_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
            candidate_uv=np.empty((0,2),np.float32),candidate_sides=np.empty((0,2,2),np.float32),candidate_query=np.empty(0,np.int32),
            candidate_flags=np.empty(0,np.uint8),candidate_side_quantiles_uv=np.empty((0,2,3,2),np.float64),
            old_cached_candidate_pixels_used_as_current_readouts=False)
        m,inv=remeasure_confirmed_source_roles(rgb,m,mask,cache.inventory,overrides,profile_backend='cpu')
        if inv!=effective:raise ValueError('Source roles changed')
        return m
    cv2.setNumThreads(1)
    for frame in programme.frames(start_pts=start,stop_pts=stop,cpu_threads=1):
        pts=int(frame['actual_pts']);native_pts=int(frame['native_pts']);raw=frame['raw_bgr'];rgb=cv2.cvtColor(raw,cv2.COLOR_BGR2RGB)
        binding=dict(frame['source_binding'],native_pts=native_pts)
        union,mask_status=masks.static_scene_exclusion(native_pts);person=primary_person_box_components(masks,native_pts)
        exclusion=shared.source_exclusion(raw,union)
        scope_measurement=shared.measure_board_source_frame(raw,exclusion,binding)
        bands=scope_measurement.get('ice_scope_boundary_candidates',scope_measurement['boundary_candidates'])
        ice_exclude=shared.conservative_ice_exclusion(exclusion,bands);scope=~shared.conservative_ice_exclusion(np.zeros_like(exclusion),bands)
        negative=shared.exact_source_material_exclusion(binding,known);ice_exclude|=negative;scope&=~negative
        if not frame['representative']:
            if last is None or last['pts']!=pts:raise ValueError('Each alias must follow its own canonical source')
            current=fresh(rgb,binding,last['measurement'],ice_exclude)
            compared=compare_material_candidates(last['measurement'],current,effective)
            checked=check_native_alias_paint_precision(compared,current,rgb,effective)
            alias_index=len(aliases);aliases.append(checked)
            alias_records.append(dict(actual_pts=pts,native_pts=native_pts,decoded_native_BGR_sha256=binding['decoded_native_BGR_sha256'],
                own_RGB_profiles_read=True,new_FIT_observations_created=False,check_array_index=alias_index,
                canonical_point_status_counts=dict(Counter(map(int,checked['canonical_candidate_status'])))))
            continue
        cached=cache.frame(pts,source_binding=binding);restore_exact_navigation_queries(cached,navigation,frame['camera25'])
        # Restrict only the fresh gray search profiles to the already existing
        # material identity neighborhood, so unrelated broad-band dark patches
        # cannot set a misleading local paint background. This changes search
        # navigation, never source acceptance, observed pixels, or old arrays.
        qgray=np.isin(cached['query_curve'],[i for i,r in enumerate(cache.inventory) if r['curve_id'] in MEASURED_ROLES])
        cached['query_interval'][qgray]=np.c_[np.maximum(cached['query_interval'][qgray,0],-24.),np.minimum(cached['query_interval'][qgray,1],24.)]
        measurement=fresh(rgb,binding,cached,ice_exclude)
        if default_guide_previous is not None:
            ga=default_guide_previous.consume(rgb,deepcopy(measurement),ice_exclude,source_ice_scope_mask=scope,guides=guides_by_frame.get(pts,()))
            gb=default_guide_current.consume(rgb,deepcopy(measurement),ice_exclude,source_ice_scope_mask=scope,guides=guides_by_frame.get(pts,()))
            if literal(ga)!=literal(gb):raise ValueError('Default source guide behavior changed from saved predecessor')
        guided=guide_tracker.consume(rgb,measurement,ice_exclude,source_ice_scope_mask=scope,guides=guides_by_frame.get(pts,()))
        measurement=guided['measurement']
        cleared=clearance.consume(rgb,measurement,ice_exclude,source_ice_scope_mask=scope,
            **{k:person[k] for k in ('primary_box_mask','measured_person_mask','other_person_boxes_mask')})
        measurement=cleared['measurement'];ice_exclude=cleared['exclusion_mask']|negative;scope=cleared['source_ice_scope_mask']&~negative
        if default_previous is not None:
            a=default_previous.consume(rgb,deepcopy(measurement),ice_exclude,source_ice_scope_mask=scope)
            b=default_current.consume(rgb,deepcopy(measurement),ice_exclude,source_ice_scope_mask=scope)
            if literal(a)!=literal(b) or literal(default_previous.previous_tracks)!=literal(default_current.previous_tracks):raise ValueError('Default context changed from its saved predecessor')
            default_comparisons+=1
        selected=selector.consume(rgb,measurement,ice_exclude,source_ice_scope_mask=scope)
        if any(r['curve_id'] not in OUTPUT_ROLES for r in selected['rows']):raise ValueError('Display role or old material selected')
        preparation=np.unique(np.concatenate([np.asarray(model_context.previous_tracks[n]['source_candidate_indices'],int)
            for n in OUTPUT_ROLES if n in model_context.previous_tracks])) if model_context.previous_tracks else np.empty(0,int)
        graph.current_field_order_prepared_candidate_ids=preparation
        spill[pts]=dict(measurement=deepcopy(measurement),field_order_prepared_candidate_ids=preparation)
        delayed=graph.consume(rgb,measurement,ice_exclude,source_ice_scope_mask=scope,selected_rows=selected,source_binding=binding)
        graph.preserve_current_source_row_proofs(selected['rows'])
        selector.retain_final_source_rows(rgb,measurement,selected,final_source_node_ids=graph.frames[-1]['ids'])
        clearance.remember_current_source_rows(selected,measurement)
        guide_tracker.remember_selected_source_rows(selected,measurement=measurement)
        if default_guide_previous is not None and not frame_records:
            default_guide_previous.remember_selected_source_rows(selected,measurement=measurement)
            default_guide_current.remember_selected_source_rows(selected,measurement=measurement)
        current_count=normalize(selected['rows'],measurement,binding)
        delayed_count=0
        for row in delayed['rows']:
            prior=int(row['actual_pts']);own=measurement if prior==pts else spill[prior]['measurement']
            delayed_count+=normalize([row],own,row['source_binding'])
        writer.mark_exposure_processed(pts,source_binding=binding);spill.spill_graph_frame(graph.frames[-1])
        counts=Counter(effective[int(c)]['curve_id'] for c in np.asarray(measurement['query_curve'])[measurement['candidate_query']])
        rows_count=Counter()
        for r in selected['rows']:rows_count[r['curve_id']]+=len(r['uv'])
        frame_records.append(dict(actual_pts=pts,native_pts=native_pts,source_binding=binding,source_candidate_count=dict(counts),
            selected_current_point_counts=dict(rows_count),current_written=current_count,delayed_written=delayed_count,
            source_decisions=[d for d in selected['decisions'] if d.get('curve_id') in MEASURED_ROLES],
            model_context_decisions=[d for d in selected['model_context_decisions'] if d.get('curve_id') in MEASURED_ROLES],
            field_order_source_prepared_original_candidate_indices=preparation.tolist(),
            F3_single_source_support_certificate=model_context.latest_single_gray_support_certificate,
            graph_nodes=len(graph.frames[-1]['ids']),graph_final_pixel_rejection_count=len(graph._latest_final_node_rejections)))
        last=dict(pts=pts,measurement=measurement)
        if time.perf_counter()-lastnotice>25 or len(expected)<=4:
            print(json.dumps(dict(stage='gray_roles_only_actual_RGB',actual_pts=pts,processed=len(frame_records),expected=len(expected),points=dict(total),
                source_decisions=frame_records[-1]['source_decisions'],F3_qualified_support=0 if model_context.latest_single_gray_support_certificate is None else len(model_context.latest_single_gray_support_certificate['current_qualified_measured_support'])),ensure_ascii=False),flush=True);lastnotice=time.perf_counter()
    if [r['actual_pts'] for r in frame_records]!=expected.tolist() or len(frame_records)+len(alias_records)!=len(native):raise ValueError('Full exact canonical/alias window was not read')
    graph_path=dest/'single_physical_gray_source_material_graph.npz';graph_receipt=graph.save_compact_state(graph_path)
    writer.original_references.append(dict(path=graph_receipt['path'],sha256=graph_receipt['sha256']))
    if graph_receipt.get('source_precision_pending_reference'):writer.original_references.append(graph_receipt['source_precision_pending_reference'])
    spill_ref=spill.seal();writer.original_references.append(spill_ref)
    alias_path=dest/'gray_native_alias_own_source_readouts.npz';alias_offsets=np.r_[0,np.cumsum([len(a['canonical_candidate_indices']) for a in aliases])].astype(np.int64)
    arrays=dict(offsets=alias_offsets,actual_pts=np.asarray([r['actual_pts'] for r in alias_records]),native_pts=np.asarray([r['native_pts'] for r in alias_records]))
    for k in ('canonical_candidate_indices','canonical_candidate_status','measured_native_uv','measured_native_sides_uv','matched_native_candidate_indices'):
        shape={'measured_native_uv':(0,2),'measured_native_sides_uv':(0,2,2)}.get(k,(0,))
        arrays[k]=np.concatenate([a[k] for a in aliases]) if aliases else np.empty(shape)
    arrays['metadata_json']=json.dumps(dict(own_RGB_alias_records=alias_records,aliases_do_not_create_FIT=True),ensure_ascii=False,separators=(',',':'))
    np.savez_compressed(alias_path,**arrays);writer.original_references.append(ref(alias_path))
    coverage_path=dest/'gray_current_actual_source_coverage_and_pending.json'
    coverage=dict(schema='rinkspace.confirmed-gray-role-only-source-coverage.v1',frame_records=frame_records,alias_records=alias_records,
        full_actual_native_window_read=True,source_profile_roles=MEASURED_ROLES,physical_output_roles=OUTPUT_ROLES,
        old_ice_or_board_pixels_remeasured_or_written=False,default_current_context_bitexact_to_predecessor_exposures=default_comparisons,
        source_weights_Held_roles_scope_pixel_identity_gates_not_relaxed=True,
        source_output_coverage_not_all_visible_points_guaranteed=True,solver_started=False)
    save(coverage_path,coverage);writer.original_references.append(ref(coverage_path))
    exported=writer.finish()
    with np.load(output,allow_pickle=False) as z:
        rows=unpack_source_json(z['rows_json']);final_counts=Counter()
        for r in rows:final_counts[r['physical_material_id']]+=r['point_count']
        if any(n not in OUTPUT_ROLES for n in final_counts):raise ValueError('Final source pack contains a duplicated/old material')
    receipt=dict(exported,manifest_reference=ref(frozen),coverage_reference=ref(coverage_path),source_current_final_point_counts=dict(final_counts),
        exact_actual_count=len(frame_records),own_native_alias_count=len(alias_records),physical_entity_loss_count_is_single_per_measured_support=True,
        old_successful_packs_unchanged=True,source_accuracy_or_whole_visible_coverage_not_claimed=True,solver_started=False,elapsed_seconds=time.perf_counter()-began)
    save(dest/'role_only_source_terminal_receipt.json',receipt);print(json.dumps(receipt,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
