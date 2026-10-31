"""Organize existing paired RGB reads of complete painted circle alternatives.

This reuses the all-frame capture. It never re-samples, changes source UV,
assigns physical identity, or promotes a fitted conic to an observation.
"""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import time

import cv2
import numpy as np

from rinkspace.path_policy import guard_project_write
from rinkspace.session.material_rings import source_ring_hypotheses
from rinkspace.session.source_shape_consensus import known_shape_profile_hypotheses
from capture_independent_ice import timed_capture_frames


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    raw = Path(path).read_bytes()
    return json.loads(gzip.decompress(raw) if str(path).endswith('.gz') else raw)


def write(path, value):
    data = (json.dumps(value, ensure_ascii=False, separators=(',', ':'))+'\n').encode('utf8')
    path = guard_project_write(path)
    path.write_bytes(gzip.compress(data, compresslevel=1, mtime=0) if str(path).endswith('.gz') else data)


def run(args):
    index = read(args.root/'index.json')
    binding = read(args.capture/'binding.json')
    paths = binding['native_inventory']
    requested = set(args.features)
    curves = {ci: path for ci, path in enumerate(paths) if path['feature_id'] in requested and path['closed']}
    if {p['feature_id'] for p in curves.values()} != requested:
        raise ValueError('Every requested feature must have a captured closed material path')
    families = {feature: [ci for ci, path in curves.items() if path['feature_id'] == feature]
                for feature in args.features}
    if args.shape_model == 'free-ellipse' and any(len(indices) != 1 for indices in families.values()):
        raise ValueError('Independent free-ellipse diagnostics only accept single-path families')
    source_sha = index['source']['actual_sha256']
    if binding['source_video_sha256'] != source_sha or digest(index['source']['resolved_path']) != source_sha:
        raise ValueError('Original source identity changed')
    source = next(s for s in index['shots'] if s['shot_id'] == args.shot)
    scope = source['processing']
    first = args.first_pts if args.first_pts is not None else scope['first_pts']
    stop = args.stop_pts if args.stop_pts is not None else scope['stop_pts_exclusive']
    if not scope['first_pts'] <= first < stop <= scope['stop_pts_exclusive'] or (first-scope['first_pts']) % 256 or (stop-first) % 256:
        raise ValueError('Range must be actual contiguous native PTS')
    source = dict(source, processing=dict(scope, first_pts=first, last_pts=stop-256, stop_pts_exclusive=stop))
    args.output = guard_project_write(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    folder = args.output/args.shot
    folder.mkdir(exist_ok=True)
    receipt = dict(schema='rinkspace.legacy_paint_circle_source_shapes.v1',
        capture=str(args.capture.resolve()), capture_binding_sha256=digest(args.capture/'binding.json'),
        source_video_sha256=source_sha, shot=args.shot, first_pts=first, stop_pts_exclusive=stop,
        pts=args.pts, features=args.features, code_sha256=digest(__file__),
        algorithm_sha256=digest(source_ring_hypotheses.__code__.co_filename),
        known_shape_algorithm_sha256=digest(known_shape_profile_hypotheses.__code__.co_filename),
        shape_model=args.shape_model,
        source_UV_unchanged=True, fit_eligible=False, physical_identity_confirmed=False,
        conics_are_observations=False)
    receipt_path = args.output/f'{args.shot}_{first}_{stop}_binding.json'
    if receipt_path.exists() and read(receipt_path) != receipt:
        raise ValueError('Existing output binds different inputs')
    write(receipt_path, receipt)
    counts = {p['curve_id']: dict(frames=0, with_hypotheses=0, consensus_frames=0, consensus_readouts=0)
              for p in curves.values()}
    cv2.setNumThreads(1)
    start = time.perf_counter()
    frames = 0
    for _, (pts, image), _ in timed_capture_frames(index, source, args.pts):
        metadata = read(args.capture/args.shot/f'{pts}.json.gz')
        npz = args.capture/args.shot/f'{pts}.npz'
        if digest(npz) != metadata['raw_candidates']['sha256']:
            raise ValueError('Captured candidate arrays changed')
        if hashlib.sha256(image.tobytes()).hexdigest() != metadata['source_BGR_sha256']:
            raise ValueError('Source exposure pixels differ')
        measurement = dict(np.load(npz, allow_pickle=False))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = {}
        for feature, indices in families.items():
            if args.shape_model == 'native-similarity':
                hypotheses = known_shape_profile_hypotheses(measurement, indices, metadata['fragments'])
            else:
                ci = indices[0]
                hypotheses = source_ring_hypotheses(rgb, measurement, ci, metadata['fragments'],
                    band_centers=True, material_signal=curves[ci]['material_signal'])
            # Compare measured alternatives, rather than fabricating missing arc
            # positions. The same public 85% alternative rule is used for houses.
            plausible = [h for h in hypotheses if h['score'] >= .85*hypotheses[0]['score']] if hypotheses else []
            selected = sorted(set.intersection(*(set(h['candidate_ids']) for h in plausible))) if plausible else []
            if selected and not np.all(np.isin(measurement['query_curve'][measurement['candidate_query'][selected]], indices)):
                raise ValueError('Conic references an unrelated source candidate')
            result[feature] = dict(hypotheses=hypotheses, plausible_alternatives=len(plausible),
                consensus_candidate_ids=selected, identity_from_query_name=False)
            for ci in indices:
                count = counts[curves[ci]['curve_id']]
                count['frames'] += 1
                count['with_hypotheses'] += any(any(measurement['query_curve'][measurement['candidate_query'][h['candidate_ids']]] == ci) for h in hypotheses)
                local = [cid for cid in selected if measurement['query_curve'][measurement['candidate_query'][cid]] == ci]
                count['consensus_frames'] += bool(local)
                count['consensus_readouts'] += len(local)
        write(folder/f'{pts}.json.gz', dict(pts=pts, shot=args.shot,
            source_BGR_sha256=metadata['source_BGR_sha256'], source_video_sha256=source_sha,
            measurement_path=str(npz.resolve()), measurement_sha256=metadata['raw_candidates']['sha256'],
            shape_families=result, source_UV_unchanged=True, fit_eligible=False,
            physical_identity_confirmed=False))
        frames += 1
        if frames % 100 == 0:
            print(json.dumps(dict(shot=args.shot, frames=frames, pts=pts, elapsed_s=time.perf_counter()-start)), flush=True)
    summary = dict(binding=receipt, frames=frames, by_curve=counts, elapsed_s=time.perf_counter()-start)
    write(args.output/f'{args.shot}_{first}_{stop}_coverage.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'capture', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--shot', choices=['s07', 's08', 's09'], required=True)
    parser.add_argument('--features', nargs='+', default=['C0', 'A1', 'A2', 'A3', 'A4'])
    parser.add_argument('--shape-model', choices=['native-similarity', 'free-ellipse'], default='native-similarity')
    parser.add_argument('--first-pts', type=int)
    parser.add_argument('--stop-pts', type=int)
    parser.add_argument('--pts', type=int, nargs='+')
    run(parser.parse_args())
