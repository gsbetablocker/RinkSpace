"""Recheck actual ice identity continuity using saved RGB measurements.

One original-video sequence supplies current RGB/LK context; saved source
profiles are reused. No new PNG cache, profile reads, fit or video is made.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'calibration/program_tracking')]
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import av
import cv2
import numpy as np
from calibration.program_tracking.render_result import PersonMasks, DEFAULT_MASKS, DEFAULT_PEOPLE
from rinkspace.vision_ice_temporal_identity import TemporalIceIdentitySelector, registered_user_stroke_seed
from rinkspace.vision_ice_identity_graph import IceSourceMaterialIdentityGraph
from rinkspace.vision_ice_source_guides import ActualIceSourceGuideTracker
from rinkspace.vision_programme_ice_review import IceProgrammeSourceReader
from rinkspace.programme_source_measurement_cache import restore_exact_navigation_queries
from scripts.review_programme_source_inputs import _observed_ice_region, _ice_exclusion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measurements', type=Path, required=True)
    parser.add_argument('--registered-user-stroke', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--diagnostic-decisions', action='store_true')
    parser.add_argument('--inspect-pts', type=int, nargs='*', default=[])
    parser.add_argument('--reread-board', action='store_true',
        help='Read repaired real yellow/blue support from this same original RGB stream')
    parser.add_argument('--graph-anchor-last', action='store_true')
    parser.add_argument('--graph-device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--compare-cpu-graph',action='store_true')
    parser.add_argument('--source-guides',type=Path)
    parser.add_argument('--state-identity',type=Path)
    parser.add_argument('--profile-backend',choices=('cpu','cuda'),default='cpu')
    args = parser.parse_args()
    cv2.setNumThreads(1)
    manifest = json.loads(args.measurements.read_text(encoding='utf8'))
    chunks = []
    for ref in manifest['chunks']:
        if hashlib.sha256(Path(ref['path']).read_bytes()).hexdigest() != ref['sha256']:
            raise ValueError('Saved actual source measurements changed')
        chunks.append(np.load(ref['path'], allow_pickle=False))
    indexed = {}
    for cache in chunks:
        for ordinal, pts in enumerate(cache['actual_pts']):indexed[int(pts)] = (cache, ordinal)
    stamps = sorted(indexed)
    seedpacket = json.loads(args.registered_user_stroke.read_text(encoding='utf-8-sig'))
    seedpts = int(seedpacket['actual_exposure_pts'])
    cache, si = indexed[seedpts]
    seed = registered_user_stroke_seed(args.registered_user_stroke,
        source_sha256=manifest['source_sha256'], time_base=manifest['time_base'],
        shot_id=str(cache['shot_id'][si]), curve_id=seedpacket['feature_id']+':segment0')
    selector = TemporalIceIdentitySelector(manifest['inventory'], source_sha256=manifest['source_sha256'],
        time_base=manifest['time_base'], seed_registry=[seed])
    masks = PersonMasks(DEFAULT_MASKS, DEFAULT_PEOPLE)
    guide_packet=json.loads(args.source_guides.read_text(encoding='utf8')) if args.source_guides else {'guides':[]}
    guide_tracker=navigation_reader=None;camera_by_pts={}
    if args.source_guides:
        identity=json.loads(args.state_identity.read_text(encoding='utf8'))
        scene=json.loads((args.state_identity.parent/'scene_UNACCEPTED.json').read_text(encoding='utf8'))
        workroot=args.source_guides.parent
        roles=json.loads((workroot.parents[3]/'observations/vision_training/paint_role_contract_current.json').read_text(encoding='utf8'))
        navigation_reader=IceProgrammeSourceReader(scene,source_sha256=manifest['source_sha256'],time_base=manifest['time_base'],role_contract=roles)
        with np.load(identity['camera_path'],allow_pickle=False) as camera:
            camera_by_pts={int(p):v.copy() for p,v in zip(camera['pts'],camera['camera25'])}
        guide_tracker=ActualIceSourceGuideTracker(manifest['inventory'],source_sha256=manifest['source_sha256'],time_base=manifest['time_base'],profile_backend=args.profile_backend)
    graph=IceSourceMaterialIdentityGraph(manifest['inventory'],source_sha256=manifest['source_sha256'],
        time_base=manifest['time_base'],source_path=manifest['source_path'],device=args.graph_device) if args.graph_anchor_last else None
    cpu_graph=IceSourceMaterialIdentityGraph(manifest['inventory'],source_sha256=manifest['source_sha256'],
        time_base=manifest['time_base'],source_path=manifest['source_path']) if args.compare_cpu_graph else None
    graph_counts={}
    counts, statuses, current_uv, current_sides, selected_indices = [], [], [], [], []
    offsets = [0]
    seen = set()
    with av.open(manifest['source_path']) as video:
        stream = video.streams.video[0]
        stream.thread_count = 1
        if str(stream.time_base) != manifest['time_base']:
            raise ValueError('Original source time base changed')
        video.seek(stamps[0], stream=stream, backward=True)
        for frame in video.decode(stream):
            pts = frame.pts
            if pts is None or pts < stamps[0]:continue
            if pts > stamps[-1]:break
            if pts not in indexed:continue
            cache, i = indexed[pts]
            rgb = frame.to_ndarray(format='rgb24')
            digest = hashlib.sha256(rgb.tobytes()).hexdigest()
            if digest != str(cache['RGB_sha256'][i]):
                raise ValueError('Original source RGB differs from measured cache')
            measurement = dict(source_binding=dict(source_sha256=manifest['source_sha256'],
                source_pts=int(pts), time_base=manifest['time_base'], shot_id=str(cache['shot_id'][i]),
                native_hw=[1080, 1920]), decoded_native_RGB_sha256=digest)
            for prefix, names in [('query', ['curve', 'uv', 'normal', 'interval', 'status']),
                                  ('candidate', ['uv', 'sides', 'query', 'flags'])]:
                lo, hi = cache[prefix+'_offsets'][i:i+2]
                for name in names:measurement[prefix+'_'+name] = cache[prefix+'_'+name][lo:hi]
            bands = []
            for b in range(*map(int, cache['frame_band_offsets'][i:i+2])):
                lo, hi = cache['band_point_offsets'][b:b+2]
                bands.append(dict(blue_upper_uv=cache['board_upper_uv'][lo:hi],
                                  yellow_lower_uv=cache['board_lower_uv'][lo:hi]))
            excluded = masks.static_scene_exclusion(int(cache['native_pts'][i]))[0]
            if args.reread_board:
                from calibration.program_tracking.far_rail_source_components import source_exclusion
                from calibration.program_tracking.source_board_measurements import measure_board_source_frame
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                excluded = source_exclusion(bgr, excluded)
                exact = dict(source_sha256=manifest['source_sha256'], actual_pts=int(pts),
                    native_pts=int(cache['native_pts'][i]), source_pts=int(pts), time_base=manifest['time_base'],
                    shot_id=str(cache['shot_id'][i]), decoded_native_BGR_sha256=str(cache['BGR_sha256'][i]))
                board=measure_board_source_frame(bgr, excluded, exact)
                bands = board.get('ice_scope_boundary_candidates',board['boundary_candidates'])
            current_mask=_ice_exclusion(excluded,bands);scope=_observed_ice_region(bands)
            if guide_tracker is not None:
                restore_exact_navigation_queries(measurement,navigation_reader,camera_by_pts[int(cache['native_pts'][i])])
                guided=guide_tracker.consume(rgb,measurement,current_mask,source_ice_scope_mask=scope,
                    guides=[v for v in guide_packet['guides'] if v['actual_pts']==pts])
                measurement=guided['measurement']
            result = selector.consume(rgb, measurement, current_mask,source_ice_scope_mask=scope)
            if guide_tracker is not None:guide_tracker.remember_selected_source_rows(result)
            rows = [r for r in result['rows'] if r['curve_id'] == seed['curve_id']]
            decisions = [r for r in result['decisions'] if r['curve_id'] == seed['curve_id']]
            selected = rows[0] if rows else None
            if args.diagnostic_decisions:
                print(json.dumps(decisions[0], ensure_ascii=False))
            if pts in args.inspect_pts:
                ci = [row['curve_id'] for row in manifest['inventory']].index(seed['curve_id'])
                queries = np.asarray(measurement['candidate_query'])
                indices = np.flatnonzero(measurement['query_curve'][queries] == ci)
                print(json.dumps(dict(actual_pts=pts, actual_C_candidate_uv=measurement['candidate_uv'][indices].tolist(),
                    actual_C_candidate_sides=measurement['candidate_sides'][indices].tolist(),
                    candidate_flags=measurement['candidate_flags'][indices].tolist(),
                    actual_C_candidate_query=queries[indices].tolist()), ensure_ascii=False))
                xys = np.rint(measurement['candidate_uv'][indices]).astype(int)
                scope = _observed_ice_region(bands)
                scene_excluded = masks.static_scene_exclusion(int(cache['native_pts'][i]))[0]
                print(json.dumps(dict(actual_pts=pts,
                    actual_board_lower_samples=[np.asarray(b['yellow_lower_uv']).tolist() for b in bands],
                    C_candidate_source_scope_confirmed=scope[xys[:,1],xys[:,0]].tolist(),
                    C_candidate_static_scene_excluded=scene_excluded[xys[:,1],xys[:,0]].tolist()), ensure_ascii=False))
                cv2.imwrite(str(args.output.parent/f'original_source_inspection_{pts}.jpg'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            count = len(selected['uv']) if selected else 0
            if graph is not None:
                graph_result=graph.consume(rgb,measurement,_ice_exclusion(excluded,bands),
                    source_ice_scope_mask=_observed_ice_region(bands),
                    selected_rows=rows if pts==stamps[-1] else ())
                for r in graph_result['rows']:
                    if r['curve_id']==seed['curve_id']:
                        graph_counts[r['actual_pts']]=graph_counts.get(r['actual_pts'],0)+len(r['uv'])
                if cpu_graph is not None:
                    cpu_result=cpu_graph.consume(rgb,measurement,_ice_exclusion(excluded,bands),
                        source_ice_scope_mask=_observed_ice_region(bands),selected_rows=rows if pts==stamps[-1] else ())
                    def signature(result):return {(r['actual_pts'],r['curve_id'],tuple(r['original_candidate_indices']),
                        hashlib.sha256(np.asarray(r['uv'],np.float64).tobytes()).hexdigest()) for r in result['rows']}
                    if signature(cpu_result)!=signature(graph_result):raise ValueError('GPU material matching changed actual source admission')
            counts.append(count); statuses.append(decisions[0]['status'])
            offsets.append(offsets[-1]+count)
            if selected:
                current_uv.extend(selected['uv']); current_sides.extend(selected['candidate_sides'])
                selected_indices.extend(selected['source_candidate_indices'])
            seen.add(pts)
    if seen != set(stamps):raise ValueError('Exact original exposure sequence was not completely decoded')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, actual_pts=np.asarray(stamps, np.int64),
        admitted_point_count=np.asarray(counts, np.int32), frame_status=np.asarray(statuses),
        offsets=np.asarray(offsets, np.int64), selected_uv=np.asarray(current_uv, float).reshape(-1, 2),
        selected_sides=np.asarray(current_sides, float).reshape(-1, 2, 2),
        selected_source_candidate_indices=np.asarray(selected_indices, np.int32),
        source_measurement_manifest_path=str(args.measurements.resolve()),
        registered_user_source_seed_path=str(args.registered_user_stroke.resolve()))
    print(json.dumps(dict(actual_exposures=len(counts), measured_exposures=int(np.count_nonzero(counts)),
        actual_measured_centers=sum(counts), missing=[dict(pts=p, status=s) for p,c,s in zip(stamps,counts,statuses) if not c],
        output=str(args.output)), ensure_ascii=False))
    if graph is not None:
        print(json.dumps(dict(graph_backward_reached_exposures=len(graph_counts),
            graph_actual_side_midpoints=sum(graph_counts.values()),source_pts_point_count=graph_counts,
            source_material_matching_device=args.graph_device,CPU_actual_source_indices_and_UV_equivalent=bool(cpu_graph)),ensure_ascii=False))


if __name__ == '__main__':main()
