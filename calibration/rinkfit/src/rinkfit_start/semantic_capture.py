"""Source-run legal capture seeds on the same complete physical model.

Dynamic programming uses raw native pixel capture costs, not the correlated W
likelihood. Both undeclared world orientations remain initialization choices.
This module changes only source-indexed station/path seeds in a new directory.
It never changes source UV, identity, finite topology, statistics or geometry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .data import sha256_file
from .native_scene import NativeScene, camera_from_event
from .runtime import atomic_json
from .semantic_order import audit_source_runs, _board_components, _board_lengths, _kind


def monotone_grid_dp(costs, *, closed=False, direction=1):
    """Exact finite-grid non-strict DP, including at most one closed circuit.

    The closed case enumerates every first-node anchor unless an admissible
    lower bound proves it cannot improve the current result. No source row is
    dropped. Returned indices refer to the original grid orientation.
    """
    costs = np.asarray(costs, float)
    if costs.ndim != 2 or not costs.shape[0] or not costs.shape[1]:
        raise ValueError('Nonempty row by finite-station capture cost table required')
    if direction not in (1, -1):
        raise ValueError('An explicit initialization direction candidate is required')
    if direction == -1:
        costs = costs[:, ::-1]
    n, count = costs.shape

    def solve(table, first=None):
        previous = table[0].copy()
        if first is not None:
            previous[:] = np.inf; previous[first] = table[0, first]
        back = np.zeros(table.shape, dtype=np.int64)
        for row in range(1, n):
            prefix = np.minimum.accumulate(previous)
            improvements = previous < np.r_[np.inf, prefix[:-1]]
            choices = np.maximum.accumulate(np.where(improvements, np.arange(len(previous)), 0))
            back[row] = choices
            previous = table[row]+prefix
        last = int(np.argmin(previous)); total = float(previous[last])
        if not np.isfinite(total):
            return None, total
        result = np.empty(n, dtype=np.int64); result[-1] = last
        for row in range(n-1, 0, -1):
            result[row-1] = back[row, result[row]]
        return result, total

    if not closed:
        choice, total = solve(costs)
    else:
        # Each anchor permits exactly count consecutive grid nodes. This is a
        # finite directed arc shorter than one period, with no seam connector.
        other_lower = float(np.sum(np.min(costs[1:], axis=1))) if n > 1 else 0.
        best = np.inf; choice = None
        double = np.tile(costs, (1, 2))
        for anchor in np.argsort(costs[0], kind='stable'):
            if costs[0, anchor]+other_lower >= best:
                break
            local, total = solve(double[:, anchor:anchor+count], first=0)
            if local is not None and total < best:
                best = total; choice = (local+anchor) % count
        total = best
    if choice is not None and direction == -1:
        choice = count-1-choice
    return choice, total


class CaptureDomain:
    def __init__(self, material, paths, branches, variant, scene, board_lengths=None):
        self.material = material; self.paths = paths; self.branches = list(branches)
        self.variant = variant; self.scene = scene; self.board_lengths = board_lengths
        if board_lengths is None:
            self.closed = bool(paths[self.branches[0]].get('closed', False))
            self.label = paths[self.branches[0]]['id']
            self.length = None
        else:
            self.segments = [int(paths[b]['boundary_segment']) for b in self.branches]
            self.lengths = board_lengths[self.segments]
            self.edges = np.r_[0., np.cumsum(self.lengths)]
            self.length = float(self.edges[-1]); self.closed = len(self.segments) == 8
            self.label = 'board/' + ','.join(map(str, self.segments))

    def map(self, coordinate):
        coordinate = np.asarray(coordinate, float)
        if self.board_lengths is None:
            return np.full(coordinate.shape, self.branches[0], dtype=np.int64), coordinate
        length = coordinate*self.length
        index = np.minimum(np.searchsorted(self.edges[1:], length, side='right'), len(self.branches)-1)
        local = (length-self.edges[index])/self.lengths[index]
        return np.asarray(self.branches)[index], local

    def world(self, coordinate):
        branch, local = self.map(coordinate)
        world = np.empty(np.shape(coordinate)+(3,))
        for b in np.unique(branch):
            selected = branch == b
            world[selected] = self.scene.world(self.material, int(b), local[selected])
        return world

    def original_coordinates(self, branches, stations):
        result = []
        for branch, station in zip(branches, stations):
            if int(branch) not in self.branches:
                continue
            if self.board_lengths is None:
                result.append(float(station))
            else:
                index = self.branches.index(int(branch))
                result.append(float((self.edges[index]+station*self.lengths[index])/self.length))
        return result

    def initial_breaks(self, original):
        if self.board_lengths is not None:
            values = self.edges/self.length
        else:
            path = self.paths[self.branches[0]]
            if _kind(path) == 2:
                values = np.linspace(0., 1., max(2, len(path['points'])))
            elif _kind(path) == 1:
                values = np.linspace(0., 1., 9)
            else:
                values = np.asarray([0., 1.])
        return np.unique(np.r_[values, original])


def _domains(material_index, material, variant, scene, board_lengths):
    paths = material['paths']; matching = [i for i, p in enumerate(paths) if p.get('variant_id') == variant]
    board = [i for i in matching if _kind(paths[i]) == 3 and int(paths[i].get('boundary_segment', -1)) >= 0]
    result = []
    if board:
        for component in _board_components(paths, variant):
            branches = [next(i for i in board if int(paths[i]['boundary_segment']) == segment) for segment in component]
            result.append(CaptureDomain(material_index, paths, branches, variant, scene, board_lengths))
    for branch in matching:
        if branch not in board:
            result.append(CaptureDomain(material_index, paths, [branch], variant, scene))
    return result


def capture_grid(domain, camera, times, observed, original, *, spacing_px=4., curvature_px=.1):
    """Adaptive capture grid; its resolution is a seed choice, not fit accuracy."""
    grid = domain.initial_breaks(original)
    while True:
        world = domain.world(grid)
        projected = camera.at(world[None], times[:, None])
        middle = (grid[:-1]+grid[1:])/2
        mid = camera.at(domain.world(middle)[None], times[:, None])
        uv = projected['uv']; delta = np.linalg.norm(np.diff(uv, axis=1), axis=2)
        bend = np.linalg.norm(mid['uv']-(uv[:, 1:]+uv[:, :-1])/2, axis=2)
        supported = projected['valid'][:, 1:] | projected['valid'][:, :-1] | mid['valid']
        split = np.any(supported & ((delta > spacing_px) | (bend > curvature_px)), axis=0)
        split &= middle > grid[:-1]; split &= middle < grid[1:]
        if not np.any(split):
            break
        grid = np.sort(np.r_[grid, middle[split]])
    if domain.closed:
        # The duplicated endpoint is the same physical point, not another turn.
        grid = grid[grid < 1.]
    projected = camera.at(domain.world(grid)[None], times[:, None])
    costs = np.sum((projected['uv']-observed[:, None])**2, axis=2)
    costs[~projected['valid']] = np.inf
    return grid, costs


def _raw_at_rows(scene, camera, materials, branches, stations, times, observed):
    world, support, _ = scene.world_rows(materials, branches, stations)
    out = camera.at(np.where(support[:, None], world, np.array([0., 0., 1.])), times)
    return {'error_px': np.linalg.norm(out['uv']-observed, axis=1), 'valid': support & out['valid']}


def capture_source_runs(run_dir, model_path, seed_dir, output_dir, *, progress=None,
                        capture_all_runs=False):
    start = time.monotonic(); run = Path(run_dir); packet = run/'backend_input'
    model_path, seed, output = Path(model_path), Path(seed_dir), Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Capture candidate already exists; never overwrite a prior seed')
    output.mkdir(parents=True, exist_ok=True)
    model = json.loads(model_path.read_text(encoding='utf-8'))
    semantic_path = run/'semantic_packet/manifest.json'; semantic = json.loads(semantic_path.read_text(encoding='utf-8'))
    n = semantic['observations']; stations = np.fromfile(seed/'station.f64', '<f8'); branches = np.fromfile(seed/'path.u32', '<u4')
    if stations.shape != (n,) or branches.shape != (n,):
        raise ValueError('Capture needs the complete restored row-state seed')
    observed = np.fromfile(packet/'uv.f64', '<f8').reshape(n, 2)
    pts = np.fromfile(packet/'pts.i64', '<i8'); materials = np.fromfile(packet/'material.u32', '<u4')
    variants = np.fromfile(run/'semantic_packet/variant_index.u32', '<u4')
    scene = NativeScene(model); camera = camera_from_event(model); lengths = _board_lengths(model)
    times = (pts-256256)/12800.+camera.tau*(observed[:, 1]/1079.-.5)
    before = audit_source_runs(semantic_path, model, stations, branches)
    atomic_json(output/'source_order_before.json', before)
    result_station = stations.copy(); result_branch = branches.copy()
    runs = [json.loads(line) for line in (run/'semantic_packet/source_runs.jsonl').read_text(encoding='utf-8').splitlines()]
    used = np.zeros(n, dtype=bool); records = []; failures = []
    for source in runs:
        rows = np.asarray(source['source_rows'], int)
        if np.any(used[rows]):
            raise ValueError('Overlapping source runs require a joint capture graph; do not rewrite independent seeds')
        used[rows] = True
    for ordinal, (source, old_audit) in enumerate(zip(runs, before['records'], strict=True)):
        if old_audit['valid'] and not capture_all_runs:
            continue
        rows = np.asarray(source['source_rows'], int); material = int(materials[rows[0]])
        if np.any(materials[rows] != material):
            raise ValueError('A source run cannot cross distinct physical materials')
        variant = semantic['variant_ids'][int(source['variant_index'])]
        if np.any(variants[rows] != int(source['variant_index'])):
            raise ValueError('A source run cannot change immutable row variants')
        domains = _domains(material, model['materials'][material], variant, scene, lengths)
        candidates = []; best = None
        # A legal independently captured continuous seed is an admissible
        # whole-run candidate.  A finite grid must never replace it merely
        # because the grid was evaluated later or has a coarser station chart.
        # All candidates use the SAME raw observed-row capture objective.
        baseline = _raw_at_rows(scene, camera, materials[rows], branches[rows],
            stations[rows], times[rows], observed[rows])
        if old_audit['valid'] and np.all(baseline['valid']):
            total = float(np.sum(baseline['error_px']**2))
            legal_directions = old_audit['legal_direction_candidates']
            candidate = {'domain': 'existing_legal_source_run',
                'direction_seed': int(legal_directions[0]),
                'cost_raw_capture_px2': total, 'grid_points': None,
                'feasible': True, 'continuous_nearest_seed_preserved_if_no_better_grid': True}
            candidates.append(candidate)
            best = total, branches[rows].copy(), stations[rows].copy(), candidate
        for domain in domains:
            original = domain.original_coordinates(branches[rows], stations[rows])
            grid, costs = capture_grid(domain, camera, times[rows], observed[rows], original)
            for direction in old_audit['allowed_direction_candidates']:
                selected, total = monotone_grid_dp(costs, closed=domain.closed, direction=direction)
                candidate = {'domain': domain.label, 'direction_seed': direction, 'cost_raw_capture_px2': total if np.isfinite(total) else None,
                    'grid_points': len(grid), 'feasible': selected is not None}
                candidates.append(candidate)
                if selected is not None and (best is None or total < best[0]):
                    new_branch, new_station = domain.map(grid[selected])
                    best = total, new_branch, new_station, candidate
        record = {'run_id': old_audit['run_id'], 'frame': source['frame'], 'source_rows': rows.tolist(),
            'source_feature_id': source['source_feature_id'], 'variant_id': variant, 'candidates': candidates,
            'world_direction_is_initialization_choice_only': source.get('world_direction') is None,
            'original_failure': old_audit['failure_reason'],
            'original_run_legal': bool(old_audit['valid']),
            'original_capture_all_valid': bool(np.all(baseline['valid'])),
            'original_raw_capture_cost_px2': float(np.sum(baseline['error_px']**2))
                if np.all(baseline['valid']) else None}
        if best is None:
            record['unresolved'] = True; failures.append(old_audit['run_id'])
        else:
            _, new_branch, new_station, selected = best
            result_station[rows] = new_station; result_branch[rows] = new_branch
            record.update(unresolved=False, selected=selected,
                station_before=stations[rows].tolist(), station_after=new_station.tolist(),
                branch_before=branches[rows].tolist(), branch_after=new_branch.tolist())
        records.append(record)
        if progress is not None and len(records) % 100 == 0:
            progress({'stage': 'source_run_capture', 'processed_runs': len(records),
                'requested_runs': len(runs) if capture_all_runs else before['failed_runs'],
                'capture_all_runs': capture_all_runs, 'unresolved': len(failures),
                'elapsed_seconds': time.monotonic()-start})
    after = audit_source_runs(semantic_path, model, result_station, result_branch)
    atomic_json(output/'source_order_after.json', after)
    changed = (result_station != stations) | (result_branch != branches)
    affected = np.flatnonzero(changed)
    old_raw = _raw_at_rows(scene, camera, materials[affected], branches[affected], stations[affected], times[affected], observed[affected])
    new_raw = _raw_at_rows(scene, camera, materials[affected], result_branch[affected], result_station[affected], times[affected], observed[affected])
    result_station.astype('<f8').tofile(output/'station.f64'); result_branch.astype('<u4').tofile(output/'path.u32')
    np.savez_compressed(output/'changed_rows.npz', source_rows=affected, old_station=stations[affected], new_station=result_station[affected],
        old_branch=branches[affected], new_branch=result_branch[affected], capture_error_before_px=old_raw['error_px'],
        capture_error_after_px=new_raw['error_px'], capture_valid_before=old_raw['valid'], capture_valid_after=new_raw['valid'])
    atomic_json(output/'run_capture_comparisons.json', records)
    summary = {'schema': 'rinkfit.source-run-legal-capture-initialization.v1',
        'source_rows': n, 'source_runs': len(runs), 'changed_source_rows': int(changed.sum()),
        'capture_all_runs': bool(capture_all_runs), 'candidate_compared_runs': len(records),
        'existing_legal_seed_selected_runs': sum(r.get('selected',{}).get('domain')=='existing_legal_source_run' for r in records),
        'failed_runs_before': before['failed_runs'], 'failed_runs_after': after['failed_runs'],
        'unresolved_run_ids': after['failed_run_ids'], 'complete_source_run_legality': after['passed'],
        'model': {'path': str(model_path), 'sha256': sha256_file(model_path)},
        'semantic_manifest_sha256': sha256_file(semantic_path),
        'seed_station_sha256': sha256_file(seed/'station.f64'), 'seed_branch_sha256': sha256_file(seed/'path.u32'),
        'station_sha256': sha256_file(output/'station.f64'), 'branch_sha256': sha256_file(output/'path.u32'),
        'capture_error_before_px': {'median': float(np.median(old_raw['error_px'])) if len(affected) else None,
            'max': float(np.max(old_raw['error_px'])) if len(affected) else None},
        'capture_error_after_px': {'median': float(np.median(new_raw['error_px'])) if len(affected) else None,
            'max': float(np.max(new_raw['error_px'])) if len(affected) else None},
        'capture_invalid_before': int(np.count_nonzero(~old_raw['valid'])), 'capture_invalid_after': int(np.count_nonzero(~new_raw['valid'])),
        'objective': 'finite-grid raw native pixel squared distance for initialization only; no W likelihood optimum claimed',
        'projection_clock': 'actual native observed row time for capture only; true implicit RS remains independently required',
        'world_direction_hard_fact_added': False, 'source_UV_ID_material_variant_W_geometry_changed': False,
        'nonrun_rows_byte_unchanged': bool(stations[~used].tobytes() == result_station[~used].tobytes()
            and branches[~used].tobytes() == result_branch[~used].tobytes()),
        'original_seed_overwritten': False, 'physical_fit_success_claimed': False,
        'elapsed_seconds': time.monotonic()-start, 'exporter': {'path': str(Path(__file__).resolve()), 'sha256': sha256_file(__file__)}}
    atomic_json(output/'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True); parser.add_argument('--model', required=True)
    parser.add_argument('--seed-dir', required=True); parser.add_argument('--output-dir', required=True)
    parser.add_argument('--all-runs', action='store_true', help='Compare both DP directions for every proven run, retaining a better legal continuous input seed')
    args = parser.parse_args()
    result = capture_source_runs(args.run_dir, args.model, args.seed_dir, args.output_dir,
        capture_all_runs=args.all_runs,
        progress=lambda value: print(json.dumps(value), flush=True))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
