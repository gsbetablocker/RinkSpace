"""Opt-in source-image pose groups and evidence-limited main-camera clock.

Official exposure identity remains the original conservative mapping. This
sidecar only ties camera pose for directly supported near-repeat native images.
No source PTS, UV observation, source factor or scene parameter is changed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = 'rinkspace.main-source-image-pose-groups.v1'
CLOCK_SCHEMA = 'rinkspace.continuous-exposure-clock.v1'
MAIN_SHOTS = (1, 4, 7, 9)
SOURCE_TICKS_PER_SECOND = 15360.0
OFFICIAL_FRAME_TICKS_NATIVE = 307.2


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _sha(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def _json_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
                                     allow_nan=False).encode('utf8')).hexdigest()


def _source_edges(census: dict, pixels: dict) -> list[dict]:
    _require(census.get('schema')=='rinkspace.main-source-near-repeat-census.v1'
             and pixels.get('schema')=='rinkspace.main-near-repeat-original-pixels.v1',
             'Source-only native image reports differ')
    by_pair={(int(row['from_pts']),int(row['to_pts'])):row for row in pixels['pairs']}
    _require(len(by_pair)==pixels['pair_count']==21,
             'Original-size pixel review must cover all 21 selected pairs')
    edges=[]
    for row in census['pairs']:
        qualifies=(row['previous_pair_status']=='unresolved_due_to_frame_ambiguity'
                   and not row['existing_official_alias']
                   and row['roi_gray_mae']<=.20
                   and row['flow_count']>=30
                   and row['flow_magnitude_p50_px']<=.05
                   and row['flow_magnitude_p95_px']<=.8)
        if not qualifies:
            continue
        key=(int(row['from_pts']),int(row['to_pts']))
        direct=by_pair.get(key)
        _require(direct is not None and direct['shot_index']==row['shot_index'],
                 f'Original full-size frame check is missing for {key}')
        _require(not direct['raw_bgr_bitwise_equal']
                 and direct['full_bgr_mae']<=.40
                 and direct['static_band_bgr_mae']<=.55,
                 f'Near-repeat full original differs more than reviewed scope: {key}')
        _require(row['current_accepted_official_pts'] is None
                 and row['previous_accepted_official_pts'] is None,
                 f'New pose edge must not re-label a matched official exposure: {key}')
        edges.append(dict(
            shot_index=int(row['shot_index']),from_pts=key[0],to_pts=key[1],
            official_exposure_identity='unknown_for_both_native_frames',
            pair_status=row['previous_pair_status'],
            cadence_roi_gray_mae=float(row['roi_gray_mae']),
            raw_flow_count=int(row['flow_count']),
            raw_flow_magnitude_p50_px=float(row['flow_magnitude_p50_px']),
            raw_flow_magnitude_p95_px=float(row['flow_magnitude_p95_px']),
            full_native_bgr_mae=float(direct['full_bgr_mae']),
            native_static_band_bgr_mae=float(direct['static_band_bgr_mae'])))
    _require(len(edges)==len(by_pair)==21 and
             {(e['from_pts'],e['to_pts']) for e in edges}==set(by_pair),
             'The full-size original video review omitted a selected edge')
    return edges


def build_source_pose_sidecar(phase: dict, old_aliases: dict[int,int],
                              census: dict, pixels: dict) -> dict:
    """Build a standalone, uninstalled sidecar from source-only evidence."""
    _require(phase.get('exposure_clock',{}).get('schema')==CLOCK_SCHEMA,
             'Expected audited original continuous exposure clock')
    old_aliases={int(k):int(v) for k,v in old_aliases.items()}
    old_clock={int(k):float(v) for k,v in phase['frame_clock_virtual_pts'].items()}
    _require(old_aliases.keys()==old_clock.keys() and len(old_clock)==6642,
             'Original main-camera frame coverage differs')
    old_records=phase['exposure_clock']['timestamps_by_canonical_source_pts']
    old_pairs={tuple(map(int,p)) for p in phase['valid_temporal_acceleration_pairs']}
    _require(len(old_pairs)==5353 and
             set(old_aliases.values())==set(map(int,old_records)),
             'Original official alias/clock/strict-prior coverage differs')
    source_edges=_source_edges(census,pixels)

    shot_by_pts={}
    for key,record in old_records.items():
        for pts in record['source_pts']:
            pts=int(pts)
            _require(pts not in shot_by_pts and old_aliases[pts]==int(key),
                     'Original official alias record disagrees with native PTS')
            shot_by_pts[pts]=int(record['shot_index'])
    _require(shot_by_pts.keys()==old_aliases.keys(),
             'Original frame binding omits source PTS')

    parent={key:key for key in set(old_aliases.values())}
    def find(key):
        while parent[key]!=key:
            parent[key]=parent[parent[key]]
            key=parent[key]
        return key
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:
            parent[max(a,b)]=min(a,b)
    for edge in source_edges:
        a,b=edge['from_pts'],edge['to_pts']
        _require(b-a==256 and old_aliases[a]!=old_aliases[b]
                 and shot_by_pts[a]==shot_by_pts[b]==edge['shot_index'],
                 'Source-image pose edge is not a distinct adjacent same-shot pair')
        union(old_aliases[a],old_aliases[b])
    pose_aliases={p:find(old_aliases[p]) for p in old_aliases}
    by_rep=defaultdict(list)
    for p,rep in pose_aliases.items():
        by_rep[rep].append(p)
    by_shot=defaultdict(list)
    edge_by_group=defaultdict(list)
    for edge in source_edges:
        edge_by_group[pose_aliases[edge['from_pts']]].append(edge)
    groups=[]
    for rep,natives in by_rep.items():
        natives.sort()
        old_keys=sorted({old_aliases[p] for p in natives})
        shot_id=shot_by_pts[natives[0]]
        _require(all(shot_by_pts[p]==shot_id for p in natives)
                 and rep==natives[0] and
                 all(b-a==256 for a,b in zip(natives,natives[1:])),
                 'Pose group crosses a shot, skips native frames or changes representative')
        accepted={int(old_records[str(k)]['accepted_official_pts']) for k in old_keys
                  if old_records[str(k)]['accepted_official_pts'] is not None}
        _require(len(accepted)<=1,'Pose group joins different official anchors')
        group=dict(shot_index=shot_id,pose_representative_pts=rep,
                   native_source_pts=natives,original_official_canonical_keys=old_keys,
                   official_accepted_pts=(next(iter(accepted)) if accepted else None),
                   new_source_image_edge_count=len(edge_by_group[rep]),
                   pose_group_role=('source_image_near_equivalent_official_identity_unknown'
                                    if edge_by_group[rep] else
                                    'original_official_alias_or_singleton'))
        groups.append(group)
        by_shot[shot_id].append(group)
    _require(len(groups)==len(parent)-21 and sum(g['new_source_image_edge_count']
                 for g in groups)==21,
             'Expected 21 distinct new image-equivalence reductions')
    for shot_id in MAIN_SHOTS:
        by_shot[shot_id].sort(key=lambda g:g['pose_representative_pts'])
        _require(by_shot[shot_id] and all(g['shot_index']==shot_id
                 for g in by_shot[shot_id]),'Main shot group list differs')

    pose_clock={p:old_clock[p] for p in old_clock}
    segments=[]
    for shot_id in MAIN_SHOTS:
        shot_groups=by_shot[shot_id]
        anchors=[i for i,g in enumerate(shot_groups)
                 if g['official_accepted_pts'] is not None]
        _require(bool(anchors),'Each main shot needs an official time anchor')
        affected=set()
        for i,g in enumerate(shot_groups):
            if not g['new_source_image_edge_count']:
                continue
            left=max((a for a in anchors if a<i),default=None)
            right=min((a for a in anchors if a>i),default=None)
            if left is None:
                affected.add((0,right,'estimated_head_distinct_image_count_display_endpoint'))
            elif right is None:
                affected.add((left,len(shot_groups)-1,
                              'estimated_tail_distinct_image_count_display_endpoint'))
            else:
                affected.add((left,right,'between_matched_official_anchors_distinct_image_count'))
        # Distinct anchor intervals do not overlap except at an unchanged anchor.
        for left,right,role in sorted(affected):
            _require(left is not None and right is not None and right>left,
                     'Invalid affected image-time interval')
            g0,g1=shot_groups[left],shot_groups[right]
            native0=g0['native_source_pts'][0]
            native1=g1['native_source_pts'][-1] if role.startswith('estimated_tail') else g1['native_source_pts'][0]
            t0,t1=old_clock[native0],old_clock[native1]
            _require(t1>t0,'Image group endpoints lack positive display-time span')
            for i in range(left,right+1):
                t=(t0 if i==left else t1 if i==right else
                   float(t0+(t1-t0)*(i-left)/(right-left)))
                for pts in shot_groups[i]['native_source_pts']:
                    pose_clock[pts]=t
            segments.append(dict(shot_index=shot_id,
                                 first_pose_representative_pts=g0['pose_representative_pts'],
                                 last_pose_representative_pts=g1['pose_representative_pts'],
                                 endpoint_roles=[('official_matched_anchor' if g0['official_accepted_pts'] is not None else 'old_estimated_display_endpoint'),
                                                 ('official_matched_anchor' if g1['official_accepted_pts'] is not None else 'old_estimated_display_endpoint')],
                                 time_assignment_role=role,
                                 distinct_pose_group_count=right-left+1,
                                 new_source_image_edges=sum(g['new_source_image_edge_count']
                                                            for g in shot_groups[left:right+1]),
                                 first_virtual_pts=t0,last_virtual_pts=t1,
                                 group_interval_ticks=(t1-t0)/(right-left)))
        for group in shot_groups:
            values={pose_clock[p] for p in group['native_source_pts']}
            _require(len(values)==1,'One source-equivalent pose group has unequal times')
        times=[pose_clock[g['native_source_pts'][0]] for g in shot_groups]
        _require(all(b>=a-1e-6 for a,b in zip(times,times[1:])),
                 'New pose clock reverses within a shot')
        support=phase['shot_knots'][str(shot_id)]['pts']
        _require(all(support[3]-1e-7<=t<=support[-4]+1e-7 for t in times),
                 f'New pose clock leaves shot {shot_id} spline support')
        for g in shot_groups:
            if g['official_accepted_pts'] is not None:
                original_anchors={old_clock[p] for p in g['native_source_pts']
                                  if old_records[str(old_aliases[p])]['accepted_official_pts'] is not None}
                _require(len(original_anchors)==1 and
                         pose_clock[g['native_source_pts'][0]]==next(iter(original_anchors)),
                         'Matched official anchor time changed')

    new_records={}
    for group in groups:
        rep=group['pose_representative_pts']
        record=deepcopy(old_records[str(old_aliases[rep])])
        record['source_pts']=group['native_source_pts']
        record['virtual_pts']=pose_clock[rep]
        if group['new_source_image_edge_count']:
            record['status']='estimated_source_image_pose_equivalent_official_identity_unknown'
            record['accepted_official_pts']=None
            record['original_canonical_source_pts']=group['original_official_canonical_keys']
            record['source_image_equivalent_not_official_alias']=True
        elif record['virtual_pts']!=old_clock[rep]:
            record['reestimated_by_source_image_group_ordinal']=True
        new_records[str(rep)]=record
    clock_meta=deepcopy(phase['exposure_clock'])
    clock_meta['schema']=CLOCK_SCHEMA
    clock_meta['source_pts_role']='unchanged original native PTS, factor observation identity and output timestamp'
    clock_meta['estimated_time_role']='source-image pose clock estimate; only original accepted official anchor times retain their official mapping'
    clock_meta['timestamps_by_canonical_source_pts']=new_records
    clock_meta['source_image_pose_group_policy']=dict(
        official_exposure_identity_unchanged=True,
        original_official_aliases_unchanged=True,
        source_image_equivalent_pose_binding_is_not_official_exposure_claim=True,
        affected_time_intervals_only=True,
        tail_display_endpoint_is_old_estimate_not_new_official_evidence=True)
    clock_meta['pose_time_reestimated_segments']=segments
    for shot_id in MAIN_SHOTS:
        report=clock_meta['shots'][str(shot_id)]
        report['canonical_frame_count']=len(by_shot[shot_id])
        report['timestamp_status_counts']=dict(Counter(new_records[str(g['pose_representative_pts'])]['status']
                                                       for g in by_shot[shot_id]))
        report['source_image_pose_group_count']=sum(bool(g['new_source_image_edge_count'])
                                                     for g in by_shot[shot_id])

    valid_pairs=set()
    index_by_shot={shot_id:{g['pose_representative_pts']:i
                            for i,g in enumerate(by_shot[shot_id])} for shot_id in MAIN_SHOTS}
    for a,b in old_pairs:
        pa,pb=pose_aliases[a],pose_aliases[b]
        _require(pa!=pb and shot_by_pts[pa]==shot_by_pts[pb] and
                 index_by_shot[shot_by_pts[pa]][pb]==index_by_shot[shot_by_pts[pa]][pa]+1,
                 'A strict official pair was collapsed or is no longer adjacent')
        _require(abs((pose_clock[pb]-pose_clock[pa])-OFFICIAL_FRAME_TICKS_NATIVE)<=1e-4,
                 'A strict official pair lost its 20ms matched interval')
        valid_pairs.add((pa,pb))
    _require(len(valid_pairs)==len(old_pairs),
             'New pose grouping changed a strict official motion pair')
    for shot_id in MAIN_SHOTS:
        clock_meta['shots'][str(shot_id)]['valid_adjacent_motion_pair_count']=sum(
            shot_by_pts[a]==shot_id for a,b in valid_pairs)

    altered=[p for p in old_clock if pose_clock[p]!=old_clock[p]]
    old_tail=old_clock[231705344]
    nominal_tail=(old_clock[231692800]+OFFICIAL_FRAME_TICKS_NATIVE*
                  (len(by_shot[9])-1-index_by_shot[9][pose_aliases[231692800]]))
    sidecar=dict(
        schema=SCHEMA,source_sha256=phase['source_sha256'],
        original_clock_digest=_json_digest(phase['frame_clock_virtual_pts']),
        original_official_alias_digest=_json_digest({str(p):old_aliases[p]
                                                     for p in sorted(old_aliases)}),
        original_valid_pairs_digest=_json_digest(sorted(map(list,old_pairs))),
        evidence_sha256=dict(census_source_inputs=census['input_sha256'],
                             census_report=_json_digest(census),
                             direct_original_pixels=_json_digest(pixels)),
        original_native_frame_count=len(old_aliases),
        original_official_pose_count=len(parent),
        new_source_image_pair_count=len(source_edges),
        new_pose_count=len(by_rep),
        official_exposure_identity_unchanged=True,
        source_uv_and_factor_observations_unchanged=True,
        original_source_pts_unchanged=True,
        matched_official_anchor_time_unchanged=True,
        selected_source_image_thresholds=dict(cadence_roi_gray_mae_max=.20,
            raw_flow_magnitude_p50_px_max=.05,raw_flow_magnitude_p95_px_max=.8,
            minimum_raw_flow_count=30,full_native_bgr_mae_max=.40,
            native_static_band_bgr_mae_max=.55),
        source_image_pose_edges=source_edges,pose_groups=sorted(groups,key=lambda g:g['pose_representative_pts']),
        official_alias_representative_pts_by_native_pts={str(p):old_aliases[p] for p in sorted(old_aliases)},
        pose_representative_pts_by_native_pts={str(p):pose_aliases[p] for p in sorted(pose_aliases)},
        pose_virtual_pts_by_native_pts={str(p):pose_clock[p] for p in sorted(pose_clock)},
        virtual_clock_changed_native_frame_count=len(altered),
        max_abs_virtual_clock_adjustment_ticks=max((abs(pose_clock[p]-old_clock[p])
                                                      for p in altered),default=0.0),
        affected_time_segments=segments,
        shot09_tail_sensitivity=dict(
            anchor_pts=231692800,last_native_pts=231705344,
            old_estimated_display_endpoint_virtual_pts=old_tail,
            nominal_50fps_distinct_group_extrapolated_endpoint_virtual_pts=nominal_tail,
            nominal_endpoint_excess_over_existing_spline_ticks=nominal_tail-phase['shot_knots']['9']['pts'][-4],
            nominal_endpoint_excess_seconds=(nominal_tail-phase['shot_knots']['9']['pts'][-4])/SOURCE_TICKS_PER_SECOND,
            policy='preserve old estimated display endpoint and allocate its span by distinct image group; no exact exposure-time claim'),
        clock_meta=clock_meta,
        valid_official_motion_pairs=sorted(map(list,valid_pairs)))
    return sidecar


def load_source_pose_groups(path: Path, phase: dict, old_aliases: dict,
                            expected_sha256: str):
    """Return pose_aliases, pose_clock, clock_meta, valid_pairs, metadata.

    `expected_sha256` is required and must be pinned by the caller. This
    loader never modifies the original official alias evidence or graph.
    """
    path=Path(path)
    _require(_sha(path)==expected_sha256,'Pinned source-image pose sidecar changed')
    sidecar=json.loads(path.read_text(encoding='utf8'))
    _require(sidecar.get('schema')==SCHEMA and
             sidecar['source_sha256']==phase['source_sha256'] and
             sidecar['official_exposure_identity_unchanged'] is True and
             sidecar['source_uv_and_factor_observations_unchanged'] is True,
             'Source-image sidecar scope or source identity changed')
    official={int(k):int(v) for k,v in old_aliases.items()}
    _require(_json_digest({str(p):official[p] for p in sorted(official)})==
             sidecar['original_official_alias_digest'] and
             {int(k):int(v) for k,v in sidecar['official_alias_representative_pts_by_native_pts'].items()}==official,
             'Original official alias map changed')
    _require(_json_digest(phase['frame_clock_virtual_pts'])==
             sidecar['original_clock_digest'] and
             _json_digest(sorted(map(list,{tuple(map(int,p)) for p in
                       phase['valid_temporal_acceleration_pairs']})))==
             sidecar['original_valid_pairs_digest'],
             'Original clock or strict official-prior identity changed')
    aliases={int(k):int(v) for k,v in sidecar['pose_representative_pts_by_native_pts'].items()}
    clock={int(k):float(v) for k,v in sidecar['pose_virtual_pts_by_native_pts'].items()}
    _require(aliases.keys()==official.keys()==clock.keys() and
             all(aliases[official[p]]==aliases[p] for p in official) and
             all(clock[p]==clock[aliases[p]] for p in clock),
             'Pose binding omitted frames, split an official alias or assigned unequal group times')
    meta=deepcopy(sidecar['clock_meta'])
    _require(meta['schema']==CLOCK_SCHEMA and
             set(map(int,meta['timestamps_by_canonical_source_pts']))==set(aliases.values()),
             'Pose clock metadata does not cover each canonical key')
    pairs={tuple(map(int,row)) for row in sidecar['valid_official_motion_pairs']}
    _require(len(pairs)==len(phase['valid_temporal_acceleration_pairs']) and
             all(a!=b and clock[b]-clock[a]>0 for a,b in pairs),
             'Official strict motion pairs changed')
    metadata={key:sidecar[key] for key in (
        'schema','source_sha256','original_native_frame_count',
        'original_official_pose_count','new_source_image_pair_count',
        'new_pose_count','official_exposure_identity_unchanged',
        'source_uv_and_factor_observations_unchanged',
        'original_source_pts_unchanged','matched_official_anchor_time_unchanged',
        'virtual_clock_changed_native_frame_count',
        'max_abs_virtual_clock_adjustment_ticks','affected_time_segments',
        'shot09_tail_sensitivity','evidence_sha256')}
    metadata['sidecar_path']=str(path)
    metadata['sidecar_sha256']=expected_sha256
    return aliases,clock,meta,pairs,metadata


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',type=Path,required=True)
    parser.add_argument('--census',type=Path,required=True)
    parser.add_argument('--pixels',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    phase=json.loads(args.phase.read_text(encoding='utf8'))
    census=json.loads(args.census.read_text(encoding='utf8'))
    pixels=json.loads(args.pixels.read_text(encoding='utf8'))
    old_aliases={int(k):int(v) for k,v in census['pairs'][0:0]}  # populated below
    for row in census['pairs']:
        # The official source-only group map is independently reconstructed
        # from confirmed adjacent edges, including all 6,642 native frames.
        for pts in (int(row['from_pts']),int(row['to_pts'])):
            old_aliases.setdefault(pts,pts)
    for row in census['pairs']:
        if row['existing_official_alias']:
            old_aliases[int(row['to_pts'])]=old_aliases[int(row['from_pts'])]
    sidecar=build_source_pose_sidecar(phase,old_aliases,census,pixels)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(sidecar,ensure_ascii=False,indent=2,
                                      allow_nan=False),encoding='utf8')
    digest=_sha(args.output)
    loaded=load_source_pose_groups(args.output,phase,old_aliases,digest)
    print(json.dumps(dict(output=str(args.output),sha256=digest,
                          original_frames=len(loaded[0]),new_poses=len(set(loaded[0].values())),
                          source_image_edges=sidecar['new_source_image_pair_count'],
                          changed_virtual_frames=sidecar['virtual_clock_changed_native_frame_count'],
                          strict_official_pairs=len(loaded[3]))))
