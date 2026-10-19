"""Recover authenticated ice readouts after a final writer failure, without RGB.

Only genuine saved source anchors and their recorded material paths export.
Lost anchor proofs stay pending. The frozen cache supplies exact original
locators, never replacement measurements or identities.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rinkspace.programme_source_measurement_cache import ProgrammeSourceMeasurementCache
from rinkspace.vision_ice_identity_graph import IceSourceMaterialIdentityGraph
from scripts.admit_programme_source_inputs import LegacyHeldSourceRoles, whole_shot_role, read_json
from .strict_absolute_source import StrictAbsolutePackWriter, StrictAbsoluteSourcePack, normalize_selected_ice_rows
from .strict_model_scope import authenticate_source_reference, file_sha


def recover_confirmed_graph(reference, *, measurements, held_inventories, ice_seeds,
        shot_plan, output_path):
    authenticate_source_reference(reference)
    graph=IceSourceMaterialIdentityGraph.load_compact_state(reference['path'],device='cpu')
    packet=read_json(measurements)
    identity=read_json(packet['camera_identity'])
    authenticate_source_reference(dict(path=identity['camera_path'],sha256=identity['camera_sha256']))
    with np.load(identity['camera_path'],allow_pickle=False) as clock:
        expected_pts=np.unique(clock['exposure_representative_pts'])
    cache=ProgrammeSourceMeasurementCache(measurements,source_sha256=graph.source_sha256,
        time_base=graph.time_base,expected_actual_pts=expected_pts)
    if graph.inventory!=cache.inventory:
        raise ValueError('Saved graph role inventory differs from its frozen source cache')
    seeds=read_json(ice_seeds);seeds=seeds.get('records',seeds) if isinstance(seeds,dict) else seeds
    held=LegacyHeldSourceRoles(held_inventories);held.include_source_annotation_holdouts(seeds)
    plan=read_json(shot_plan);plan=plan.get('shot_plan',plan)
    ref=lambda p:dict(path=str(Path(p).resolve()),sha256=file_sha(p))
    source=dict(video_sha256=graph.source_sha256,time_base=graph.time_base,pixel_domain='native1920x1080')
    writer=StrictAbsolutePackWriter(output_path,source=source,expected_actual_pts=cache.expected_pts,
        original_references=[reference,ref(measurements),ref(ice_seeds),ref(shot_plan),*map(ref,held_inventories)],
        legacy_pixel_families=read_json(measurements).get('legacy_pixel_families',()),
        recovery_journal=Path(output_path).with_name(Path(output_path).stem+'_recovery.sqlite'))
    result=graph.export_confirmed_source_rows(source_graph_reference=reference,stream=True)
    measurement=None;current_pts=None;admitted=rejected=0
    for row in result['rows']:
        pts=int(row['actual_pts']);binding=row['source_binding']
        if current_pts!=pts:
            measurement=cache.frame(pts,source_binding=binding);current_pts=pts
            measurement['candidate_side_quantiles_uv']=np.full((len(measurement['candidate_uv']),2,3,2),np.nan)
        ids=np.asarray(row['original_candidate_indices'],np.int64)
        if (not np.array_equal(measurement['candidate_uv'][ids].astype(float),row['raw_candidate_uv'])
                or not np.array_equal(measurement['candidate_sides'][ids].astype(float),row['source_sides_uv'])
                or not np.array_equal(measurement['candidate_query'][ids],row['source_query_indices'])
                or np.any(measurement['candidate_flags'][ids]&1)):
            raise ValueError('Recovered source readout differs from its exact frozen candidate locator')
        # The graph contains only the actual source support nodes admitted to
        # this shot's measured scope. Restore that saved decision for these
        # exact nodes; no other cached candidate or polygon is approved.
        row['source_scope_restored_from_authenticated_actual_graph_nodes']=dict(
            source_graph_reference=reference,original_candidate_indices=ids.tolist(),
            frozen_original_scope_flags=measurement['candidate_flags'][ids].tolist())
        measurement['candidate_flags'][ids]&=np.uint8(253)
        if 'source_side_quantiles_uv' in row:
            measurement['candidate_side_quantiles_uv'][ids]=row['source_side_quantiles_uv']
        measurement['candidate_local_source_clearance']=row.get('candidate_local_source_clearance',{})
        split=held.split(row)
        normalized=normalize_selected_ice_rows([dict(row,split=split)],measurement,
            source_binding=binding,curve_inventory=graph.inventory,split=split)
        for value in normalized['rows']:
            value['vision_training_role']=whole_shot_role(plan,binding)
            admitted+=writer.add_row(value,normalized['review'])
        rejected+=len(normalized['rejected_rows'])
    for frame in graph.frames:
        # Validate every saved frame catalog against the hash-bound frozen
        # cache; this records prior actual decoding, not a new visual review.
        cache.frame(frame['pts'],source_binding=frame['binding'])
        writer.mark_exposure_processed(frame['pts'],source_binding=frame['binding'])
    receipt=writer.finish()
    pack=StrictAbsoluteSourcePack(receipt['path'],expected_sha256=receipt['sha256'])
    return dict(absolute_inputs=receipt,recovered_actual_source_points=admitted,
        rejected_rows=rejected,pending_source_frames=len(result['pending']),
        processed_actual_exposures=len(graph.frames),
        held_source_points=sum(len(r['uv']) for r in pack.held_rows),
        source_scope_is_not_new_manual_review=True,source_RGB_decoded_again=False,
        source_material_matching_repeated=False,source_quantiles_remeasured=False,
        visual_review_complete=False,solver_started=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph',nargs=2,required=True,metavar=('PATH','SHA256'))
    parser.add_argument('--measurements',type=Path,required=True)
    parser.add_argument('--ice-seeds',type=Path,required=True)
    parser.add_argument('--held-inventory',type=Path,action='append',required=True)
    parser.add_argument('--shot-plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(recover_confirmed_graph(dict(path=args.graph[0],sha256=args.graph[1]),
        measurements=args.measurements,held_inventories=args.held_inventory,
        ice_seeds=args.ice_seeds,shot_plan=args.shot_plan,output_path=args.output),separators=(',',':')))


if __name__=='__main__':main()
