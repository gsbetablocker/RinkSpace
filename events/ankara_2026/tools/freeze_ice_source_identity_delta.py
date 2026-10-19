"""Freeze source identity/navigation inputs for a separately reviewed ice delta."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'calibration/program_tracking')]
from calibration.program_tracking.strict_model_scope import authenticate_source_reference


def reference(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return dict(path=str(path), sha256=digest.hexdigest())


def immutable_json(directory, stem, payload):
    encoded = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode('utf8')
    digest = hashlib.sha256(encoded).hexdigest()
    path = directory / f'{stem}_{digest[:16]}.json'
    if path.exists() and path.read_bytes() != encoded:
        raise ValueError('An immutable ice source delta input changed')
    if not path.exists():
        path.write_bytes(encoded)
    return reference(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workroot', type=Path, required=True)
    args = parser.parse_args()
    work = args.workroot.resolve()
    manifest = json.loads((work / 'source_pixels/source_measurements.json').read_text(encoding='utf8'))
    registry = json.loads((work / 'ice_source_identity_seeds.json').read_text(encoding='utf8'))
    if (registry['source_sha256'] != manifest['source_sha256']
            or registry['time_base'] != manifest['time_base']):
        raise ValueError('Source registry belongs to another original source clock')
    known = {v['curve_id'] for v in manifest['inventory']}
    authorities = {}
    for seed in registry['records']:
        if seed['curve_id'] not in known:
            raise ValueError('Source identity seed names an unmodeled feature')
        authority = seed['authority']
        authenticate_source_reference(authority)
        authorities[(authority['path'], authority['sha256'])] = authority
    frozen_registry = immutable_json(work, 'ice_source_identity_seeds_delta_frozen', registry)
    overlays = []
    for pts in (229174016, 229840384, 229987328, 230024192, 230688512):
        path = work / f'actual_nested_source_overlay_{pts}.npz'
        if not path.exists():
            raise FileNotFoundError(path)
        overlays.append(dict(actual_pts=pts, reference=reference(path),
            appended_query_locators_require_overlay=True,
            appended_nodes_exist_in_earlier_pending_graph=False))
    c_packet = reference(work / 'ice_source_guides_C_229246976_e84e91952dec8db2.json')
    payload = dict(schema='rinkspace.frozen-actual-ice-source-identity-delta.v1',
        source_sha256=manifest['source_sha256'], time_base=manifest['time_base'],
        source_measurements=reference(work / 'source_pixels/source_measurements.json'),
        frozen_ice_seed_registry=frozen_registry,
        authenticated_source_identity_authorities=list(authorities.values()),
        modeled_curve_count=len(known), source_anchored_curve_count=len({v['curve_id'] for v in registry['records']}),
        source_identity_record_count=len(registry['records']),
        source_guides=[c_packet], actual_current_source_overlays=overlays,
        shared_source_replay_api=dict(
            tracker='rinkspace.vision_ice_source_guides.ActualIceSourceGuideTracker',
            before_selection='tracker.consume(rgb, measurement, exclusion, source_ice_scope_mask=scope, guides=current_guides)',
            after_selection='tracker.remember_selected_source_rows(selected_rows)',
            native_alias='rinkspace.vision_ice_source_guides.remeasure_actual_source_guided_queries',
            original_query_precision_restoration='rinkspace.programme_source_measurement_cache.restore_exact_navigation_queries'),
        offline_existing_node_api=dict(
            graph='rinkspace.vision_ice_identity_graph.IceSourceMaterialIdentityGraph.load_compact_state',
            apply='graph.apply_confirmed_source_rows(rows, source_scope_review=authenticated_exact_pixel_review)',
            saved_readout_coordinate_domains=['raw_saved_candidate_readout', 'actual_gradient_q50_float64'],
            nearest_locator_or_projected_UV_matching_allowed=False),
        appended_source_queries_need_current_shared_original_RGB_reread=True,
        source_guide_UV_is_search_navigation_only=True,
        original_candidate_flags_require_current_material_admission=True,
        source_identity_is_per_shot_and_requires_actual_adjacent_material_proof=True,
        full_programme_visual_review_complete=False, fitting_started=False)
    receipt = immutable_json(work, 'ice_source_identity_delta', payload)
    print(json.dumps(dict(receipt=receipt, frozen_registry=frozen_registry,
        curves=payload['source_anchored_curve_count'], records=len(registry['records']),
        source_overlays=len(overlays)), ensure_ascii=False))


if __name__ == '__main__':
    main()
