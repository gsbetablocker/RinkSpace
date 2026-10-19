"""Shared original-video replay for strict ice/board source admission.

Reuse pending RGB profiles, recheck actual scope, measure advertisements, and
write only source-confirmed current readouts. No fit, video, PNG library, or
projection-derived pixel truth is produced.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import importlib
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'calibration/program_tracking')]
for task_thread_limit_key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[task_thread_limit_key]='1'

import cv2
import numpy as np

from rinkspace.programme_source_stream import FrozenProgrammeSource
from rinkspace.programme_source_measurement_cache import (
    ProgrammeSourceMeasurementCache,recheck_source_candidate_exclusions,
    supplement_selected_profile_quantiles,restore_exact_navigation_queries)
from rinkspace.vision_ice_temporal_identity import TemporalIceIdentitySelector,registered_user_stroke_seed
from rinkspace.vision_ice_identity_graph import IceSourceMaterialIdentityGraph
from rinkspace.vision_ice_nested_material_identity import SourceHouseLayerIdentityExpander
from rinkspace.vision_ice_local_clearance import primary_person_box_components
from rinkspace.vision_programme_ice_review import IceProgrammeSourceReader
from rinkspace.vision_programme_board_review import BoardProgrammeSourceReader,conservative_ice_exclusion
from calibration.program_tracking.render_result import PersonMasks,DEFAULT_MASKS,DEFAULT_PEOPLE
from calibration.program_tracking.far_rail_source_components import source_exclusion
from calibration.program_tracking.source_board_measurements import measure_board_source_frame
from calibration.program_tracking.strict_absolute_source import (
    StrictAbsolutePackWriter,StrictAbsoluteSourcePack,
    normalize_selected_ice_rows,normalize_selected_board_rows,board_print_target_inventory,
    board_material_point_target_inventory)
from calibration.program_tracking.strict_model_scope import file_sha,ExactSourceScopeReview,REVIEW_SCHEMA
from calibration.program_tracking.compact_source_json import pack_source_json
from calibration.program_tracking.source_material_exclusions import (
    load_known_source_material_exclusions,exact_source_material_exclusion)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _source_json(value):
    if isinstance(value,np.ndarray):return _source_json(value.tolist())
    if isinstance(value,np.generic):return _source_json(value.item())
    if isinstance(value,dict):return {str(k):_source_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_source_json(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def whole_shot_role(plan,binding):
    matches=[s for s in plan['shots'] if s['source_sha256']==binding['source_sha256']
        and Fraction(s['time_base'])==Fraction(binding['time_base'])
        and s['id']==binding['shot_id']]
    if not matches:return 'unlisted_not_training'
    if len(matches)!=1 or matches[0].get('whole_shot') is not True:
        raise ValueError('Every native source exposure needs one original whole-shot role')
    # A development or test shot stays held; only an explicit train shot can
    # train. This source pack does not alter the original human shot plan.
    return matches[0]['split']


def old_source_dispositions(uv,exclusion,bands):
    """Compact physical removal reasons, with unknown separate from exclusion.

    0 unprocessed, 1 pending identity (removed), 2 actual exclusion-mask region
    (removed, not negative truth), 3 visibly above measured board (removed).
    Neither mask geometry nor an unassigned board band certifies admission.
    """
    points=np.asarray(uv,float)
    category=np.ones(len(points),np.uint8)
    finite=np.isfinite(points).all(1)
    pix=np.rint(np.nan_to_num(points,nan=-1.,posinf=-1.,neginf=-1.)).astype(int)
    inside=finite&(pix>=0).all(1)&(pix<[1920,1080]).all(1)
    category[~inside]=2
    category[inside&exclusion[np.clip(pix[:,1],0,1079),np.clip(pix[:,0],0,1919)]]=2
    for band in bands:
        top=np.asarray(band['blue_upper_uv'])
        for a,b in zip(top[:-1],top[1:]):
            if b[0]-a[0]>29:continue
            good=(category==1)&(points[:,0]>=a[0])&(points[:,0]<=b[0])
            height=a[1]+(points[:,0]-a[0])/(b[0]-a[0])*(b[1]-a[1])
            category[good&(points[:,1]<height-8)]=3
    return category


def _reference(path):
    return dict(path=str(Path(path).resolve()),sha256=file_sha(path))


class LegacyHeldSourceRoles:
    """Keep explicit calibration holdouts on their source frame and entity.

    Material patch identities stay distinct from named modeled curves. A
    source feature that cannot be named from its preserved record is never
    guessed from a projected coordinate or renamed into an ice curve.
    """
    def __init__(self,paths):
        self.held=set();self.exact_points=set();self.references=[]
        for path in paths:
            self.references.append(_reference(path))
            with np.load(path,allow_pickle=False) as z:
                rows=json.loads(str(z['rows_json'].item()))
                pts=z['actual_pts'];uv=z['uv'];row_index=z['row_index']
            held=np.asarray([r.get('split') in ('held','holdout') for r in rows],bool)
            for index,row in enumerate(rows):
                if not held[index]:continue
                actual=int(row['actual_pts'])
                names=[row.get(k) for k in ('feature_id','curve_id','physical_material_id') if row.get(k)]
                ad=re.match(r'(?i)^ad(\d+)(?:[_:.]|$)',str(row.get('id','')))
                if ad:names.append('ad'+ad.group(1))
                for name in names:self.held.add((actual,str(name).casefold()))
            for point,timepoint in zip(uv[held[row_index]],pts[held[row_index]]):
                self.exact_points.add((int(timepoint),np.asarray(point,np.float64).tobytes()))

    def include_source_annotation_holdouts(self,seeds):
        # Source authority roles attach to that original frame/feature. A
        # source identity propagated to a different image does not propagate
        # its calibration holdout role to the rest of the camera shot.
        for seed in seeds:
            if seed.get('source_original_split') not in ('held','holdout'):continue
            actual=int(seed['actual_pts'])
            for name in (seed['curve_id'],seed.get('physical_id')):
                if name:self.held.add((actual,str(name).casefold()))

    def split(self,row,default='fit'):
        if row.get('split',default)=='held':return 'held'
        actual=int(row['actual_pts'])
        names=[row.get(k) for k in ('feature_id','curve_id','physical_material_id') if row.get(k)]
        if row.get('advertisement_number') is not None:names.append('ad'+str(row['advertisement_number']))
        if any((actual,str(name).casefold()) in self.held for name in names):return 'held'
        if any((actual,np.asarray(point,np.float64).tobytes()) in self.exact_points for point in row['uv']):return 'held'
        return default


def actual_role_absence_feedback(path,source):
    if path is None:return {}
    with np.load(path,allow_pickle=False) as saved:
        metadata=json.loads(str(saved['metadata_json'].item()))
        rows=json.loads(str(saved['rows_json'].item()));uv=saved['uv']
    if (metadata.get('schema')!='rinkspace.strict-source-role-absence-pixels.v1'
            or metadata['source']!=source or len(rows)!=len(uv)):
        raise ValueError('Source role-absence feedback identity differs')
    proofs=[dict(row,uv=point.tolist()) for row,point in zip(rows,uv)]
    review=ExactSourceScopeReview(dict(schema=REVIEW_SCHEMA,source=source,reviewed_endpoints=proofs))
    by_frame=defaultdict(dict)
    for proof in review.records.values():
        if (proof['decision']!='exclude' or proof.get('reject_only_authority')!='actual_visual_role_absence'
                or proof.get('actual_source_RGB_role_absence_confirmed') is not True
                or proof.get('confidence_used_as_rejection_evidence') is not False
                or proof.get('model_probability_used_as_rejection_evidence') is not False):
            raise ValueError('Unknown/mask/model-probability exclusions are not actual role-absence feedback')
        by_frame[int(proof['actual_pts'])][np.asarray(proof['uv'],np.float64).tobytes()]=proof
    return by_frame


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--identity',required=True,type=Path)
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--measurements',required=True,type=Path)
    parser.add_argument('--current-points',required=True,type=Path)
    parser.add_argument('--ice-seeds',required=True,type=Path)
    parser.add_argument('--registered-user-stroke',action='append',default=[],type=Path)
    parser.add_argument('--source-guide-packet',action='append',default=[],type=Path,
        help='Immutable same-source guide packet; guides navigate actual RGB remeasurement')
    parser.add_argument('--model-navigation-authority',type=Path,
        help='User authority to use this exact frozen grid for RGB-confirmed identity context')
    parser.add_argument('--known-source-material-exclusion',type=Path,action='append',default=[],
        help='Authenticated exact-native model-external source material; never training pixel truth')
    parser.add_argument('--roles',type=Path)
    parser.add_argument('--source-role-overrides',type=Path,action='append',default=[])
    parser.add_argument('--candidate-local-clearance',action='store_true',
        help='Require two actual RGB material views for candidate-local primary-box clearance')
    parser.add_argument('--board-config',type=Path)
    parser.add_argument('--board-reader',default='rinkspace.vision_programme_board_review:BoardProgrammeSourceReader',
        help='Public BoardProgrammeSourceReader subclass, also used for native alias profiles')
    parser.add_argument('--board-only',action='store_true',
        help='Independent board source delta; retain existing ice packs without repeating their material graph')
    parser.add_argument('--legacy-inventory',action='append',default=[],type=Path)
    parser.add_argument('--negative-feedback',type=Path)
    parser.add_argument('--native-alias-review',help='Importable module:function; shared current RGB, no decode or FIT')
    parser.add_argument('--shot-plan',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--pending-material-state-dir',type=Path,
        help='Compact per-shot measured nodes and actual RGB adjacency; no source image library')
    parser.add_argument('--preserve-current-source-proofs',action='store_true',
        help='Persist the original selector temporal proof after strict graph validation')
    parser.add_argument('--spill-source-state',action='store_true',
        help='Keep complete overlay and numeric graph state in this replay Data output, without thinning')
    parser.add_argument('--follow',action='store_true')
    parser.add_argument('--start-pts',type=int)
    parser.add_argument('--stop-pts',type=int)
    parser.add_argument('--supplement-support',choices=('graph','selected'),default='graph')
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--cpu-threads',type=int,default=4,
        help='Decoder and OpenCV threads; set from an actual throughput comparison')
    args=parser.parse_args()
    if args.cpu_threads < 1:raise ValueError('CPU thread count must be positive')
    cv2.setNumThreads(args.cpu_threads)
    if args.device=='cuda':
        import torch
        torch.set_num_threads(1)
        if not torch.cuda.is_available():raise RuntimeError('Source admission requested CUDA but it is unavailable')
    if args.output.exists():raise ValueError('Do not overwrite an earlier strict source pack')
    retired_path=args.output.with_name(args.output.stem+'_retired_old_source_indices.npz')
    native_path=args.output.with_name(args.output.stem+'_native_alias_review.npz')
    if retired_path.exists():raise ValueError('Do not replace earlier source retirement records')
    if native_path.exists():raise ValueError('Do not replace earlier native source review records')
    programme=FrozenProgrammeSource(args.identity,args.source,ROOT/'calibration/program_tracking/source_shots.json')
    cache=ProgrammeSourceMeasurementCache(args.measurements,
        source_sha256=programme.source_sha256,time_base=programme.time_base,
        expected_actual_pts=programme.actual_inventory,follow=args.follow)
    roles=read_json(args.roles) if args.roles else None
    navigation_reader=IceProgrammeSourceReader(programme.scene,source_sha256=programme.source_sha256,
        time_base=programme.time_base,role_contract=roles,profile_backend=args.device)
    if cache.inventory!=navigation_reader.inventory:
        raise ValueError('Frozen scene no longer reproduces cached ice readout roles')
    source_role_overrides={};effective_inventory=cache.inventory
    if args.source_role_overrides:
        from rinkspace.vision_ice_source_role_override import (
            effective_source_role_inventory,remeasure_confirmed_source_roles)
        for override_path in args.source_role_overrides:
            evidence=read_json(override_path)
            if (evidence['source_sha256']!=programme.source_sha256
                    or evidence['time_base']!=programme.time_base):
                raise ValueError('Source material role corrections belong to another original source')
            reference=_reference(override_path)
            for index,record in enumerate(evidence['records']):
                if (record.get('original_source_image_actually_viewed') is not True
                        or record.get('model_projection_or_fit_residual_used_for_role',
                            record.get('model_projection_or_fit_residual_used_for_identity')) is not False):
                    raise ValueError('Source role correction requires actual original source physical evidence')
                if record['curve_id'] in source_role_overrides:
                    raise ValueError('One source material cannot have competing operator corrections')
                source_role_overrides[record['curve_id']]=dict(record,
                    source_role_evidence=dict(reference,source_record_index=index),
                    original_source_role_evidence=record['source_role_evidence'])
        effective_inventory=effective_source_role_inventory(cache.inventory,source_role_overrides)
    seeds=read_json(args.ice_seeds)
    seeds=seeds['records'] if isinstance(seeds,dict) else seeds
    for path in args.registered_user_stroke:
        record=read_json(path);pts=int(record['actual_exposure_pts'])
        shot=next(s for s in programme.shots if int(s['start_pts'])<=pts<int(s['end_pts']))
        seeds.append(registered_user_stroke_seed(path,source_sha256=programme.source_sha256,
            time_base=programme.time_base,shot_id=shot['id'],curve_id=record['feature_id']+':segment0'))
    guide_packets=[read_json(path) for path in args.source_guide_packet]
    guides_by_frame=defaultdict(list)
    for packet in guide_packets:
        for guide in packet.get('guides',()):guides_by_frame[int(guide['actual_pts'])].append(guide)
        seeds.extend(packet.get('identity_seeds',()))
    guide_tracker=None
    if guide_packets:
        from rinkspace.vision_ice_source_guides import ActualIceSourceGuideTracker
        guide_tracker=ActualIceSourceGuideTracker(effective_inventory,source_sha256=programme.source_sha256,
            time_base=programme.time_base,profile_backend=args.device)
    model_context=None
    known_material_exclusions=load_known_source_material_exclusions([
        _reference(path) for path in args.known_source_material_exclusion])
    if args.model_navigation_authority:
        from rinkspace.vision_ice_model_context_identity import SameStateIceModelContextIdentity
        model_context=SameStateIceModelContextIdentity(effective_inventory,state_identity_path=args.identity,
            navigation_authority=_reference(args.model_navigation_authority),
            source_sha256=programme.source_sha256,time_base=programme.time_base,
            actual_source_negative_materials=known_material_exclusions)
    selector=TemporalIceIdentitySelector(effective_inventory,source_sha256=programme.source_sha256,
        time_base=programme.time_base,seed_registry=seeds,model_context_identity=model_context)
    graph=IceSourceMaterialIdentityGraph(effective_inventory,source_sha256=programme.source_sha256,
        time_base=programme.time_base,source_path=str(programme.source_path),device=args.device)
    nested_materials=SourceHouseLayerIdentityExpander(effective_inventory,
        source_sha256=programme.source_sha256,time_base=programme.time_base)
    local_clearance=None
    if args.candidate_local_clearance:
        from rinkspace.vision_ice_local_clearance import KnownIceMaterialBoxClearance
        local_clearance=KnownIceMaterialBoxClearance(effective_inventory,
            source_sha256=programme.source_sha256,time_base=programme.time_base,seed_registry=seeds)
    board_config=read_json(args.board_config) if args.board_config else {}
    print_targets=board_print_target_inventory(programme.scene)
    material_targets=board_material_point_target_inventory(programme.scene)
    module,class_name=args.board_reader.rsplit(':',1)
    board_reader_class=getattr(importlib.import_module(module),class_name)
    if not isinstance(board_reader_class,type) or not issubclass(board_reader_class,BoardProgrammeSourceReader):
        raise ValueError('Board profile reader must use the public source reader contract')
    boards=board_reader_class.from_source_packets(board_config.get('packet_paths',()),
        source_sha256=programme.source_sha256,time_base=programme.time_base,
        shot_manifest=ROOT/'calibration/program_tracking/source_shots.json',
        confirmed_identity_references=board_config.get('confirmed_identity_references',()),
        actual_exposure_pts=programme.actual_inventory,
        native_to_actual=dict(zip(map(int,programme.native_pts),map(int,programme.actual_pts))),
        corner_annotation_paths=board_config.get('corner_annotation_paths',()),
        source_annotation_seeds=board_config.get('source_annotation_seeds',()),device=args.device)
    cv2.setNumThreads(args.cpu_threads)
    plan=read_json(args.shot_plan);plan=plan.get('shot_plan',plan)
    inventory_paths=args.legacy_inventory or [args.measurements.parent.parent/'source_inventory'/name
        for name in ('original_absolute_source_review_inventory.npz','additional_absolute_source_review_inventory.npz')]
    held_roles=LegacyHeldSourceRoles(inventory_paths)
    held_roles.include_source_annotation_holdouts(seeds)
    masks=PersonMasks(DEFAULT_MASKS,DEFAULT_PEOPLE)
    masks.validate_coverage(np.asarray(sorted(masks.boxes),np.int64))
    if not set(map(int,programme.native_pts)).issubset(masks.boxes):
        raise ValueError('Original source person mask archive omits a programme frame')
    with np.load(args.current_points,allow_pickle=False) as frozen:
        old_pts,old_offsets,old_uv=frozen['actual_pts'],frozen['offsets'],frozen['uv']
        frozen_metadata=json.loads(str(frozen['metadata_json'].item()))
    if not np.array_equal(old_pts,programme.actual_inventory):
        raise ValueError('Old pixel index does not cover the frozen source clock')
    source=dict(video_sha256=programme.source_sha256,time_base=programme.time_base,pixel_domain='native1920x1080')
    feedback=actual_role_absence_feedback(args.negative_feedback,source)
    legacy_families=[r['family'] for r in frozen_metadata['family_receipts']
        if not r['family'].endswith(':motion') and
        (r.get('pixel_input_count',0)>0 or r.get('outside_programme_pixel_count',0)>0)]
    immutable_paths=[args.current_points,args.identity,args.ice_seeds,args.shot_plan]
    if args.roles:immutable_paths.append(args.roles)
    immutable_paths.extend(args.source_role_overrides)
    if args.board_config:immutable_paths.append(args.board_config)
    immutable_paths.extend(args.registered_user_stroke)
    immutable_paths.extend(args.source_guide_packet)
    immutable_paths.extend(args.known_source_material_exclusion)
    immutable_paths.extend(Path(record[key]['path']) for record in known_material_exclusions
        for key in ('authority','original_source_image'))
    if args.model_navigation_authority:immutable_paths.append(args.model_navigation_authority)
    immutable_paths.extend(inventory_paths)
    if args.negative_feedback:immutable_paths.append(args.negative_feedback)
    writer=StrictAbsolutePackWriter(args.output,source=source,expected_actual_pts=programme.actual_inventory,
        original_references=[_reference(p) for p in immutable_paths],legacy_pixel_families=legacy_families,
        recovery_journal=args.output.with_name(args.output.stem+'_recovery.sqlite'))
    disposition=np.zeros(len(old_uv),np.uint8)
    old_lookup={int(p):i for i,p in enumerate(old_pts)}
    if args.spill_source_state:
        from rinkspace.source_state_spill import ExactSourceStateSpill
        corrections=ExactSourceStateSpill(args.output.with_name(args.output.stem+'_source_state'))
    else:
        corrections={}
    shot=None;processed=0;last_notice=time.monotonic()
    def log(**values):print(json.dumps(values,ensure_ascii=False,separators=(',',':')),flush=True)
    def source_review_with_known_absence(review,binding):
        known=feedback.get(int(binding['actual_pts']),{})
        if not known:return review
        endpoints=[];changed=False
        for proof in review.records.values():
            absence=known.get(np.asarray(proof['uv'],np.float64).tobytes())
            if absence is not None:
                if (absence['native_pts']!=binding['native_pts']
                        or absence['decoded_BGR_sha256']!=binding['decoded_native_BGR_sha256']):
                    raise ValueError('Known source role absence cannot move to another native image')
                proof=absence;changed=True
            endpoints.append(proof)
        return ExactSourceScopeReview(dict(schema=REVIEW_SCHEMA,source=source,
            reviewed_endpoints=endpoints)) if changed else review
    def add_ice(rows,measurement,binding):
        selected=rows.get('rows',()) if isinstance(rows,dict) else rows
        count=rejected=0
        for row in selected:
            seed=row.get('source_seed_reference',{})
            origin=seed.get('source_original_split','fit')
            split=held_roles.split(row,'fit')
            row=dict(row,source_origin_role=origin)
            result=normalize_selected_ice_rows([dict(row,split=split)],measurement,source_binding=binding,
                curve_inventory=effective_inventory,split=split)
            for normalized in result['rows']:
                normalized['vision_training_role']=whole_shot_role(plan,binding)
                count+=writer.add_row(normalized,source_review_with_known_absence(result['review'],binding))
            rejected+=len(result['rejected_rows'])
        return count,rejected
    totals=defaultdict(int)
    pending_material_receipts=[]
    pending_state_dir=args.pending_material_state_dir or args.output.with_name(args.output.stem+'_pending_material_graphs')
    def save_shot_material_graph():
        if not graph.frames:return
        pending_state_dir.mkdir(parents=True,exist_ok=True)
        identifier=re.sub(r'[^A-Za-z0-9_.-]','_',graph.shot_id)
        path=pending_state_dir/(identifier+'.npz')
        receipt=graph.save_compact_state(path)
        pending_material_receipts.append(receipt)
        writer.original_references.append(dict(path=receipt['path'],sha256=receipt['sha256']))
        if receipt.get('source_precision_pending_reference') is not None:
            writer.original_references.append(dict(receipt['source_precision_pending_reference']))
        overlay_frames=[pts for pts,value in corrections.items()
            if len(value['appended']['candidate_uv']) or value['source_guided_queries']]
        if overlay_frames:
            overlay=pending_state_dir/(identifier+'_source_role_overlay.npz')
            if overlay.exists():raise ValueError('Do not replace an earlier source-role readout overlay')
            counts=[len(corrections[pts]['appended']['candidate_uv']) for pts in overlay_frames]
            original_counts=[len(corrections[pts]['flags'])-count for pts,count in zip(overlay_frames,counts)]
            quantiles=[]
            for pts,count,original_count in zip(overlay_frames,counts,original_counts):
                value=corrections[pts];q=np.full((count,2,3,2),np.nan)
                appended_support=value['supported']>=original_count
                q[value['supported'][appended_support]-original_count]=value['quantiles'][appended_support]
                quantiles.append(q)
            np.savez_compressed(overlay,actual_pts=np.asarray(overlay_frames,np.int64),
                offsets=np.r_[0,np.cumsum(counts)].astype(np.int64),
                original_candidate_counts=np.asarray(original_counts,np.int32),
                **{key:np.concatenate([corrections[pts]['appended'][key] for pts in overlay_frames])
                    for key in ('candidate_uv','candidate_sides','candidate_query')},
                candidate_flags=np.concatenate([corrections[pts]['flags'][start:]
                    for pts,start in zip(overlay_frames,original_counts)]),
                candidate_side_quantiles_uv=np.concatenate(quantiles),
                query_offsets=np.r_[0,np.cumsum([len(corrections[pts]['queries']['query_curve'])
                    for pts in overlay_frames])].astype(np.int64),
                **{key:np.concatenate([corrections[pts]['queries'][key] for pts in overlay_frames])
                    for key in ('query_curve','query_uv','query_normal','query_interval','query_status')},
                source_guided_queries_json=pack_source_json([corrections[pts]['source_guided_queries']
                    for pts in overlay_frames]),
                metadata_json=json.dumps(dict(source=source,shot_id=graph.shot_id,
                    effective_inventory=effective_inventory,source_role_overrides=source_role_overrides,
                    source_material_graph_reference=dict(path=receipt['path'],sha256=receipt['sha256']),
                    actual_source_RGB_remeasurements=True,source_RGB_images_saved=False,
                    unresolved_or_excluded_not_admitted=True),ensure_ascii=False,separators=(',',':')))
            writer.original_references.append(_reference(overlay))
        log(stage='source_material_adjacency_saved',**receipt)
    native_catalog=[];alias_checks=[]
    canonical_raw_bgr=None;canonical_board_rows=();canonical_raw_actual_pts=None
    canonical_original_exclusion=None;canonical_person_components=None
    canonical_current_measurement=None
    canonical_ice_source_rows=()
    alias_hook=None
    if args.native_alias_review:
        module,function=args.native_alias_review.rsplit(':',1)
        alias_hook=getattr(importlib.import_module(module),function)
    for frame in programme.frames(start_pts=args.start_pts,stop_pts=args.stop_pts,cpu_threads=args.cpu_threads):
        binding=dict(frame['source_binding'],native_pts=int(frame['native_pts']))
        pts=int(frame['actual_pts']);raw=frame['raw_bgr']
        forbidden_material=exact_source_material_exclusion(binding,known_material_exclusions)
        union,_=masks.static_scene_exclusion(frame['native_pts'])
        person_components=primary_person_box_components(masks,frame['native_pts'])
        exclusion=source_exclusion(raw,union)
        board_measurement=measure_board_source_frame(raw,exclusion,binding)
        bands=board_measurement['boundary_candidates']
        ice_boundaries=board_measurement.get('ice_scope_boundary_candidates',bands)
        ice_exclusion=conservative_ice_exclusion(exclusion,ice_boundaries)
        ice_scope=~conservative_ice_exclusion(np.zeros_like(exclusion),ice_boundaries)
        ice_exclusion|=forbidden_material;ice_scope&=~forbidden_material
        native_catalog.append(dict(native_pts=binding['native_pts'],actual_pts=pts,
            decoded_BGR_sha256=binding['decoded_native_BGR_sha256'],
            representative=bool(frame['representative']),actual_source_scope_RGB_processed=True,
            canonical_UV_current_native_material_checked=bool(frame['representative'])))
        if not frame['representative']:
            canonical=(canonical_current_measurement if canonical_raw_actual_pts==pts else
                cache.frame(pts,wait_callback=lambda p:log(
                    stage='waiting_for_canonical_source_measurement_chunk',actual_pts=p)))
            if canonical['query_uv'].dtype!=np.float64:
                canonical_camera=programme.cameras[programme.lookup[pts]]
                restore_exact_navigation_queries(canonical,navigation_reader,canonical_camera)
            index=old_lookup[pts];lo,hi=old_offsets[index:index+2]
            if alias_hook is not None:
                check=alias_hook(frame=frame,canonical_measurement=canonical,
                    exclusion_mask=exclusion,source_ice_exclusion_mask=ice_exclusion,
                    source_ice_scope_mask=ice_scope,fresh_board_measurement=board_measurement,
                    navigation_reader=navigation_reader,canonical_old_uv=old_uv[lo:hi],
                    canonical_old_uv_start_index=int(lo),
                    canonical_raw_bgr=canonical_raw_bgr if canonical_raw_actual_pts==pts else None,
                    canonical_board_rows=canonical_board_rows if canonical_raw_actual_pts==pts else (),
                    canonical_exclusion_mask=canonical_original_exclusion if canonical_raw_actual_pts==pts else None,
                    canonical_person_components=canonical_person_components if canonical_raw_actual_pts==pts else None,
                    candidate_local_clearance=local_clearance,
                    primary_person_components=person_components,
                    canonical_ice_source_rows=canonical_ice_source_rows if canonical_raw_actual_pts==pts else (),
                    **({'profile_reader_class':board_reader_class}
                       if board_reader_class is not BoardProgrammeSourceReader else {}))
                if (check.get('native_pts')!=binding['native_pts'] or check.get('actual_pts')!=pts
                        or check.get('decoded_native_BGR_sha256')!=binding['decoded_native_BGR_sha256']
                        or check.get('native_RGB_actually_measured') is not True
                        or check.get('canonical_UV_checked_against_current_native_material') is not True
                        or check.get('new_FIT_observations_created') is not False):
                    raise ValueError('Native alias hook must actually check its own original RGB without new FIT')
                ids=np.asarray(check['canonical_candidate_indices'],np.int32)
                status=np.asarray(check['canonical_candidate_status'],np.uint8)
                uv=np.asarray(check['measured_native_uv'],float).reshape(-1,2)
                sides=np.asarray(check['measured_native_sides_uv'],float).reshape(-1,2,2)
                if (ids.shape!=status.shape or uv.shape!=(len(ids),2) or sides.shape!=(len(ids),2,2)
                        or np.any(ids<0) or np.any(ids>=len(canonical['candidate_uv']))
                        or len(np.unique(ids))!=len(ids) or np.any(status>4)):
                    raise ValueError('Native alias point checks must preserve exact canonical candidate indices')
                alias_checks.append(dict(native_pts=binding['native_pts'],actual_pts=pts,
                    ids=ids,status=status,uv=uv,sides=sides,
                    own_candidate_indices=np.asarray(check.get('matched_native_candidate_indices',
                        np.full(len(ids),-1,np.int32)),np.int32),
                    native_paint_pixel_accuracy_checks=check.get('native_paint_pixel_accuracy_checks',[]),
                    board_native_checks=check.get('board_native_checks',dict(status='board_pending')),
                    native_candidate_local_source_clearance=check.get('native_candidate_local_source_clearance',{}),
                    board_current_native_material_checked=bool(canonical_board_rows) and
                        check.get('board_native_checks',{}).get('board_material_check_pending') is False))
                native_catalog[-1]['canonical_UV_current_native_material_checked']=True
                totals['native_alias_RGB_checked']+=1
            else:
                totals['native_alias_pending_material_check']+=1
            if time.monotonic()-last_notice>=30:
                log(stage='native_alias_scope_review',native_pts=binding['native_pts'],actual_pts=pts,**totals)
                last_notice=time.monotonic()
            continue
        measurement=cache.frame(pts,source_binding=binding,
            wait_callback=lambda p:log(stage='waiting_for_source_measurement_chunk',actual_pts=p))
        if shot!=binding['shot_id']:
            save_shot_material_graph()
            corrections.clear();shot=binding['shot_id']
        recheck_source_candidate_exclusions(measurement,ice_exclusion)
        rgb=cv2.cvtColor(raw,cv2.COLOR_BGR2RGB)
        original_candidate_count=len(measurement['candidate_uv'])
        if source_role_overrides:
            restore_exact_navigation_queries(measurement,navigation_reader,frame['camera25'])
            measurement,roles=remeasure_confirmed_source_roles(rgb,measurement,ice_exclusion,
                cache.inventory,source_role_overrides,profile_backend=args.device)
            if roles!=effective_inventory:raise ValueError('Effective source material role changed during replay')
            measurement['source_role_overrides']=source_role_overrides
        if guide_tracker is not None and not args.board_only:
            restore_exact_navigation_queries(measurement,navigation_reader,frame['camera25'])
            guided=guide_tracker.consume(rgb,measurement,ice_exclusion,
                source_ice_scope_mask=ice_scope,guides=guides_by_frame.get(pts,()))
            measurement=guided['measurement']
            totals['actual_source_guide_queries']+=sum(d.get('source_queries',0) for d in guided['decisions'])
        if local_clearance is not None:
            clear=local_clearance.consume(rgb,measurement,ice_exclusion,
                source_ice_scope_mask=ice_scope,**{key:person_components[key] for key in
                    ('primary_box_mask','measured_person_mask','other_person_boxes_mask')})
            measurement=clear['measurement'];ice_exclusion=clear['exclusion_mask']
            ice_scope=clear['source_ice_scope_mask']
            # A primary-person-box clearance cannot undo a separately
            # authenticated model-external material disposition.
            ice_exclusion|=forbidden_material;ice_scope&=~forbidden_material
            if forbidden_material.any():recheck_source_candidate_exclusions(measurement,ice_exclusion)
            totals['actual_source_candidate_local_clearances']+=len(clear['candidate_local_source_clearance'])
        selected=dict(rows=[],decisions=[]) if args.board_only else selector.consume(
            rgb,measurement,ice_exclusion,source_ice_scope_mask=ice_scope)
        nested=dict(rows=[],decisions=[]) if args.board_only else nested_materials.consume(
            rgb,measurement,ice_exclusion,source_ice_scope_mask=ice_scope,selected_rows=selected)
        selected['rows'].extend(nested['rows'])
        selected['decisions'].extend(nested['decisions'])
        totals['current_source_nested_material_rows']+=len(nested['rows'])
        anchored={r['curve_id'] for r in seeds if int(r['actual_pts'])==pts}
        if processed==0 or anchored:
            log(stage='initial_source_readout',actual_pts=pts,
                fresh_board_bands=len(bands),known_scope_pixels=int(ice_scope.sum()),
                candidate_count=len(measurement['candidate_uv']),
                clear_candidate_count=int(((measurement['candidate_flags']&3)==0).sum()),
                anchored_curve_decisions=[d for d in selected['decisions'] if d['curve_id'] in anchored])
        ids=np.unique(np.concatenate([np.asarray(r['original_candidate_indices'],int)
            for r in selected['rows']])) if selected['rows'] else np.empty(0,int)
        if args.supplement_support=='graph' and not args.board_only:
            ids=graph.source_support_candidate_indices(measurement,ice_exclusion,source_ice_scope_mask=ice_scope)
        if len(ids) and ('candidate_side_quantiles_uv' not in measurement or
                not np.isfinite(measurement['candidate_side_quantiles_uv'][ids]).all()):
            restore_exact_navigation_queries(measurement,navigation_reader,frame['camera25'])
            totals['supplemented_current_source_quantiles']+=supplement_selected_profile_quantiles(
                rgb,measurement,ids,effective_inventory,backend=args.device)
        # Save only corrections/support for this shot, never another source RGB
        # library. Delayed rows normalize against their own original indices.
        support=measurement.get('candidate_side_quantiles_uv')
        supported=np.flatnonzero(np.isfinite(support).all(axis=(1,2,3))) if support is not None else np.empty(0,int)
        corrections[pts]=dict(flags=measurement['candidate_flags'].copy(),supported=supported,
            quantiles=support[supported].copy() if support is not None else np.empty((0,2,3,2)),
            candidate_local_source_clearance=measurement.get('candidate_local_source_clearance',{}),
            queries={key:measurement[key].copy() for key in
                ('query_curve','query_uv','query_normal','query_interval','query_status')},
            source_guided_queries=measurement.get('source_guided_queries'),
            appended={key:measurement[key][original_candidate_count:].copy()
                for key in ('candidate_uv','candidate_sides','candidate_query')})
        delayed=dict(rows=[]) if args.board_only else graph.consume(rgb,measurement,ice_exclusion,
            source_ice_scope_mask=ice_scope,selected_rows=selected,source_binding=binding)
        if args.preserve_current_source_proofs and not args.board_only:
            graph.preserve_current_source_row_proofs(selected['rows'])
        if not args.board_only:
            selector.retain_final_source_rows(rgb,measurement,selected,
                final_source_node_ids=graph.frames[-1]['ids'])
        # The graph successor checks the final measured sides and may narrow
        # selected rows. Rejected pixels cannot become later navigation or
        # local foreground-clearance evidence through the earlier selectors.
        if local_clearance is not None:
            local_clearance.remember_current_source_rows(selected,measurement)
        if guide_tracker is not None and not args.board_only:
            guide_tracker.remember_selected_source_rows(selected,measurement=measurement)
        count,rejected=add_ice(selected,measurement,binding)
        totals['ice_source_pixels']+=count;totals['rejected_selected_rows']+=rejected
        by_frame=defaultdict(list)
        for row in delayed['rows']:by_frame[int(row['actual_pts'])].append(row)
        for prior,rows in by_frame.items():
            own=rows[0]['source_binding']
            prior_measurement=measurement if prior==pts else cache.frame(prior,source_binding=own)
            correction=corrections[prior];ids=correction['supported'];quantiles=correction['quantiles']
            if prior!=pts:
                for key,extra in correction['appended'].items():
                    if len(extra):prior_measurement[key]=np.concatenate([prior_measurement[key],extra])
                prior_measurement.update(correction['queries'])
                if correction['source_guided_queries']:
                    prior_measurement['source_guided_queries']=correction['source_guided_queries']
            prior_measurement['candidate_flags']=correction['flags']
            prior_measurement['candidate_local_source_clearance']=correction['candidate_local_source_clearance']
            if source_role_overrides:
                prior_measurement.update(effective_source_role_inventory=effective_inventory,
                    source_role_override_evidence={name:role['source_role_evidence'] for name,role in source_role_overrides.items()},
                    source_role_overrides=source_role_overrides)
            prior_measurement['candidate_side_quantiles_uv']=np.full(
                (len(prior_measurement['candidate_uv']),2,3,2),np.nan)
            prior_measurement['candidate_side_quantiles_uv'][ids]=quantiles
            count,rejected=add_ice(rows,prior_measurement,own)
            totals['ice_delayed_source_pixels']+=count;totals['rejected_selected_rows']+=rejected
        board=boards.read(raw,exclusion,binding,precomputed_measurement=board_measurement,
            person_components=person_components)
        # Hold exactly the current representative BGR for its immediate native
        # aliases. This is one immutable source image reference, not a library.
        canonical_raw_bgr=raw;canonical_board_rows=board['advertisement_rows']
        canonical_original_exclusion=exclusion.copy()
        canonical_person_components={key:value.copy() if isinstance(value,np.ndarray) else value
            for key,value in person_components.items()}
        canonical_raw_actual_pts=pts
        canonical_current_measurement=measurement
        canonical_ice_source_rows=selected['rows']
        for row in board['advertisement_rows']:
            split=held_roles.split(row)
            normalized=normalize_selected_board_rows([dict(row,split=split)],board,source_binding=binding,
                split=split,board_print_targets=print_targets,board_material_point_targets=material_targets)
            for approved in normalized['rows']:
                approved['vision_training_role']=whole_shot_role(plan,binding)
                totals['board_source_pixels']+=writer.add_row(approved,
                    source_review_with_known_absence(normalized['review'],binding))
            totals['rejected_selected_rows']+=len(normalized['rejected_rows'])
        index=old_lookup[pts];lo,hi=old_offsets[index:index+2]
        disposition[lo:hi]=old_source_dispositions(old_uv[lo:hi],exclusion,bands)
        for local,point in enumerate(old_uv[lo:hi]):
            proof=feedback.get(pts,{}).get(np.asarray(point,np.float64).tobytes())
            if proof is None:continue
            if (proof['native_pts']!=binding['native_pts']
                    or proof['decoded_BGR_sha256']!=binding['decoded_native_BGR_sha256']):
                raise ValueError('Actual role-absence feedback differs from current original RGB')
            disposition[int(lo)+local]=4
        writer.mark_exposure_processed(pts,source_binding=binding)
        if args.spill_source_state and graph.frames and not args.board_only:
            corrections.spill_graph_frame(graph.frames[-1])
        processed+=1
        if time.monotonic()-last_notice>=30 or processed==1:
            log(stage='actual_source_admission',processed_exposures=processed,actual_pts=pts,
                shot_id=shot,**totals);last_notice=time.monotonic()
    save_shot_material_graph()
    if args.spill_source_state:
        writer.original_references.append(corrections.seal())
    receipt=writer.finish()
    # Old UV are removed physically from the new active input. Original UV,
    # family records, source clocks, and locators remain in immutable archives.
    retired=np.flatnonzero(disposition>0).astype(np.int32)
    retirement_metadata=dict(schema='rinkspace.strict-source-old-pixel-retirement.v1',source=source,
        original_pixel_reference=_reference(args.current_points),approved_absolute_inputs=receipt,
        legacy_pixel_families=legacy_families,default_legacy_admission='forbidden',
        original_indices_removed_from_active_input=True,unknown_is_not_negative_training_truth=True,
        source_measurement_processing_is_not_manual_visual_completion=True,
        disposition_labels={'0':'unprocessed_not_approved','1':'pending_identity_removed',
            '2':'current_actual_exclusion_region_removed','3':'visibly_above_measured_board_removed',
            '4':'explicit_current_RGB_role_absence_review_removed'})
    np.savez_compressed(retired_path,retired_original_pixel_indices=retired,
        old_pixel_disposition=disposition,exposure_pts=old_pts,exposure_offsets=old_offsets,
        metadata_json=json.dumps(retirement_metadata,ensure_ascii=False,separators=(',',':')))
    native_metadata=dict(schema='rinkspace.strict-source-native-alias-material-review.v1',source=source,
        source_delta_scope='board_only' if args.board_only else 'ice_and_board',
        earlier_ice_source_input_not_replaced=bool(args.board_only),
        expected_native_pts_reference=_reference(args.identity),
        native_alias_has_separate_source_BGR=True,new_FIT_observations_created=False,
        source_measurement_processing_is_not_manual_visual_completion=True,
        all_native_source_frames_processed=np.array_equal(
            np.asarray([r['native_pts'] for r in native_catalog]),programme.native_pts),
        all_processed_native_material_checked=all(r['canonical_UV_current_native_material_checked'] for r in native_catalog)
            and len(alias_checks)==sum(not r['representative'] for r in native_catalog)
            and all(r['board_current_native_material_checked'] for r in alias_checks),
        all_processed_native_alias_board_material_checked=bool(alias_checks) and
            all(r['board_current_native_material_checked'] for r in alias_checks),
        status_labels={'0':'pending_physical_identity','1':'actual_native_material_support',
            '2':'current_native_scope_excluded','3':'current_native_material_missing','4':'current_native_material_competes'})
    np.savez_compressed(native_path,
        native_pts=np.asarray([r['native_pts'] for r in native_catalog],np.int64),
        actual_pts=np.asarray([r['actual_pts'] for r in native_catalog],np.int64),
        BGR_sha256=np.asarray([r['decoded_BGR_sha256'] for r in native_catalog],dtype='S64'),
        representative=np.asarray([r['representative'] for r in native_catalog],bool),
        native_material_checked=np.asarray([r['canonical_UV_current_native_material_checked'] for r in native_catalog],bool),
        alias_native_pts=np.asarray([r['native_pts'] for r in alias_checks],np.int64),
        alias_actual_pts=np.asarray([r['actual_pts'] for r in alias_checks],np.int64),
        alias_candidate_offsets=np.r_[0,np.cumsum([len(r['ids']) for r in alias_checks])].astype(np.int64),
        canonical_candidate_indices=np.concatenate([r['ids'] for r in alias_checks]) if alias_checks else np.empty(0,np.int32),
        canonical_candidate_status=np.concatenate([r['status'] for r in alias_checks]) if alias_checks else np.empty(0,np.uint8),
        measured_native_uv=np.concatenate([r['uv'] for r in alias_checks]) if alias_checks else np.empty((0,2)),
        measured_native_sides_uv=np.concatenate([r['sides'] for r in alias_checks]) if alias_checks else np.empty((0,2,2)),
        matched_native_candidate_indices=np.concatenate([r['own_candidate_indices'] for r in alias_checks])
            if alias_checks else np.empty(0,np.int32),
        alias_board_checks_json=pack_source_json(_source_json([dict(native_pts=r['native_pts'],
            actual_pts=r['actual_pts'],board_current_native_material_checked=r['board_current_native_material_checked'],
            native_candidate_local_source_clearance=r['native_candidate_local_source_clearance'],
            native_paint_pixel_accuracy_checks=r['native_paint_pixel_accuracy_checks'],
            checks=r['board_native_checks']) for r in alias_checks])),
        metadata_json=json.dumps(native_metadata,ensure_ascii=False,separators=(',',':')))
    pack=StrictAbsoluteSourcePack(receipt['path'],expected_sha256=receipt['sha256'])
    pack.require_complete_legacy_family_gate(frozen_metadata['family_receipts'])
    log(stage='strict_source_pack_saved',absolute_inputs=receipt,
        old_retirement_indices=_reference(retired_path),processed_exposures=processed,
        native_alias_review=_reference(native_path),processed_native_frames=len(native_catalog),
        all_exposures_processed=pack.metadata['all_exposures_processed'],
        source_uncertainty_ready_for_fit=pack.metadata['source_uncertainty_ready_for_fit'],
        solver_started=False,**totals)


if __name__=='__main__':main()
