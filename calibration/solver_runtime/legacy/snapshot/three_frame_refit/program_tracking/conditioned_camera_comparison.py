"""Same-camera-shot fixed/pivot diagnostic, conditioned on a full solution.

The original full graph is reconstructed up to (but never through) update.
Unselected cameras and shared scene nodes are constants at the saved solution.
Both models use the same original pixels, priors and native board factors.
This is a conditional model diagnostic, not a programme candidate or validation.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import numpy as np
import gtsam

import joint_tracking_refine as joint
from backend import IncrementalCameraGraph
from projection import from_state
from rig_motion import add_rig_equality
import rig_parameterization as rig_model
import source_supported_motion as source_support_model
from backend.callback_factory import select_factory
from rig_parameterization import (add_dummy_center_prior, camera25 as derived_camera25,
                                  conditioned_spec, native_pair as derived_native_pair)
from robust_observations import pointwise_huber
from correlated_motion import wrap_correlated_pair
from source_supported_motion import load_supported_motion, grouped_native_pairs


FROZEN_V13_STATES_SHA256 = '00752a8ca8225b40702129d2fd53c985e70ab296a7c6d7cb9a041e6eb1eb6045'
FROZEN_BOARD_PACKET_SHA256 = 'fe4e15a4845b77466a64c602b537ed72aca7a939ec5025ebe89eb157ccc5ebef'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class _Captured(Exception):
    pass


def capture_original(config, output):
    """Retain callback arguments, including warm-frozen reference covariances."""
    source_path = Path(joint.__file__)
    tree = ast.parse(source_path.read_text(encoding='utf8'))
    loops = [(n.lineno, n.end_lineno, ast.unparse(n.iter))
             for n in ast.walk(tree) if isinstance(n, ast.For)]
    captured = {}

    class Capture:
        def __init__(self, *args, **kwargs):
            self.specs = []

        def _record(self, kind, args, kwargs):
            frame = inspect.currentframe().f_back.f_back
            filename = Path(frame.f_code.co_filename).name
            function, line = frame.f_code.co_name, frame.f_lineno
            loop = None
            if filename == source_path.name:
                containing = [(b-a, expression) for a, b, expression in loops
                              if a <= line <= b]
                if containing:
                    loop = min(containing)[1]
            keys = ((int(args[0]),) if kind == 'prior' else
                    tuple(map(int, args[:2])) if kind == 'between' else
                    tuple(map(int, args[0])))
            identity = None
            if filename == 'ad_landmarks.py' and function == 'factor':
                identity = dict(role=frame.f_locals['role'],
                                uv=np.asarray(frame.f_locals['observed']).tolist())
            self.specs.append(dict(kind=kind, keys=keys, args=args, kwargs=kwargs,
                file=filename, function=function, line=line, loop=loop,
                source_identity=identity))

        def add_prior(self, *args, **kwargs):
            self._record('prior', args, kwargs)

        def add_between(self, *args, **kwargs):
            self._record('between', args, kwargs)

        def add_observation(self, *args, **kwargs):
            self._record('observation', args, kwargs)

        def update(self, guesses):
            local = inspect.currentframe().f_back.f_locals
            captured.update(specs=self.specs, guesses=guesses,
                **{name: local[name] for name in
                   ('shots', 'templates', 'mount_keys', 'lens_keys',
                    'source_quality', 'source_hash', 'model_hash',
                    'observation_inputs')})
            raise _Captured()

    original = joint.IncrementalCameraGraph
    joint.IncrementalCameraGraph = Capture
    try:
        # Capture raises before any solver update. Its source replay files may
        # live in the authorized project output; normal solver calls retain the
        # historical external-cache default.
        joint.solve(config, output, allow_project_output=True)
    except _Captured:
        pass
    finally:
        joint.IncrementalCameraGraph = original
    if not captured:
        raise RuntimeError('Graph construction did not reach update')
    return captured


def add_conditioned(graph, spec, active, states):
    """Keep exact callbacks/noise; substitute saved constants for external keys."""
    keys, args, kwargs = spec['keys'], spec['args'], spec['kwargs'].copy()
    if spec['kind'] == 'prior':
        graph.add_prior(*args, **kwargs)
        return
    selected = [i for i, key in enumerate(keys) if key in active]
    kept = [keys[i] for i in selected]
    if spec['kind'] == 'between':
        delta, sigma = np.asarray(args[2]).copy(), args[3]
        fn = lambda q: q[1]-q[0]-delta
        jac = lambda q: [-np.eye(11), np.eye(11)]
        kwargs = dict(residual_dim=11, sigma=sigma)
    else:
        fn = args[1]
        jac = kwargs.pop('jac_fn', None)

    def expand(q):
        local = dict(zip(kept, q))
        return [local[key] if key in local else states[key] for key in keys]

    def residual(q):
        return fn(expand(q))

    def jacobians(q):
        matrices = jac(expand(q))
        return [matrices[i] for i in selected]

    graph.add_observation(kept, residual,
        jac_fn=jacobians if jac is not None else None, **kwargs)


def selected_shots(captured, shot_indices, *, complete_camera_group=False):
    """One physical camera only; multi-shot selection must cover its full programme group."""
    if isinstance(shot_indices, int):
        shot_indices = (shot_indices,)
    requested = tuple(int(index) for index in shot_indices)
    by_index = {int(shot['index']): shot for shot in captured['shots']}
    if not requested or len(set(requested)) != len(requested) or set(requested)-set(by_index):
        raise ValueError(f'Unknown, duplicate, or empty shot selection: {requested}')
    shots = [by_index[index] for index in sorted(requested)]
    camera_ids = {shot['camera_id'] for shot in shots}
    if len(camera_ids) != 1:
        raise ValueError(f'Selected shots mix physical camera IDs: {sorted(camera_ids)}')
    camera_id = shots[0]['camera_id']
    if complete_camera_group:
        all_indices = {int(shot['index']) for shot in captured['shots']
                       if shot['camera_id'] == camera_id}
        if set(requested) != all_indices:
            raise ValueError(f'--shots must include every {camera_id} shot: '
                             f'{sorted(all_indices)}')
    return shots


def prepare(config_path, candidate, manifest_path, shot_indices, output,
            *, complete_camera_group=False):
    config = json.loads(config_path.read_text(encoding='utf8'))
    diag = json.loads((candidate/'diagnostics.json').read_text(encoding='utf8'))
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    if (manifest.get('schema') != 'rinkspace.native-surface-fixed-pivot-frozen-board.v1' or
        digest(config_path) != manifest['v13_config_sha256'] or
        digest(candidate/'diagnostics.json') != manifest['v13_diagnostics_sha256'] or
        digest(candidate/'candidate_scene.json') != manifest['candidate_scene_sha256'] or
        diag.get('configuration_sha256') != manifest['v13_config_sha256'] or
        diag.get('model_sha256') != manifest['candidate_scene_sha256'] or
        diag.get('source_sha256') != manifest['source_sha256'] or
        digest(candidate/'graph_states.npz') != FROZEN_V13_STATES_SHA256):
        raise ValueError('Frozen board input does not match the chosen candidate/config')
    captured = capture_original(config_path, output/'graph_capture')
    if (captured['model_hash'] != diag['base_model_sha256'] or
            captured['model_hash'] != manifest['base_scene_sha256'] or
            captured['source_hash'] != diag['source_sha256'] or
            captured['observation_inputs'] != diag['observation_input_sha256']):
        raise ValueError('Reconstructed graph differs from saved source/model inputs')
    with np.load(candidate/'graph_states.npz', allow_pickle=False) as z:
        if (str(z['source_sha256']) != captured['source_hash'] or
            str(z['model_sha256']) != diag['model_sha256']):
            raise ValueError('Saved state source/model differs')
        states = {int(k): v.copy() for k, v in zip(z['keys'], z['states'])}
    if set(states) != set(captured['guesses']):
        raise ValueError('Full reconstructed graph and saved state keys differ')
    shots = selected_shots(captured, shot_indices,
                           complete_camera_group=complete_camera_group)
    camera_id = shots[0]['camera_id']
    pts = {int(pt) for shot in shots for pt in shot['pts']}
    if len(pts) != sum(len(shot['pts']) for shot in shots):
        raise ValueError('Selected shot PTS overlap')
    mount = captured['mount_keys'][camera_id]
    lens = captured['lens_keys'][camera_id]
    shared = {mount, lens}
    active = pts | shared
    retained, excluded = [], Counter()
    motion_loops = {'links', 'native_links', 'raw_native_links', 'auxiliary_links'}
    for spec in captured['specs']:
        keys = set(spec['keys'])
        # The single-shot legacy scope omits equalities from other shots on
        # its rig. Complete-camera scope has no such shot and keeps any
        # remaining reference factor on this rig with its external key fixed.
        if not (keys & pts or keys <= shared or
                (complete_camera_group and keys & shared)):
            continue
        if (spec['file'] == Path(joint.__file__).name and
                spec['loop'] in motion_loops and keys <= pts):
            excluded['replaced_'+spec['loop']] += 1
            continue
        if spec['function'] == '_mount_equality' and keys <= active:
            excluded['replaced_mount_equality'] += 1
            continue
        retained.append(spec)
    if excluded['replaced_mount_equality'] != len(pts):
        raise ValueError('Did not capture exactly one fixed mount per selected frame')
    scene = json.loads((candidate/'candidate_scene.json').read_text(encoding='utf8'))
    packet_path = Path(manifest['frozen_packet_path'])
    # Pin the explicit frozen payload; this diagnostic is not a new selector.
    if (packet_path.resolve() != manifest_path.with_name('frozen_board_pairs.npz').resolve()
            or manifest['frozen_packet_sha256'] != FROZEN_BOARD_PACKET_SHA256
            or digest(packet_path) != FROZEN_BOARD_PACKET_SHA256):
        raise ValueError('Strict board pixel packet has changed')
    with np.load(packet_path, allow_pickle=False) as z:
        packet = {k: z[k].copy() for k in z.files}
    if (len(packet['pair_pts']) != sum(manifest['frozen_factor_counts'].values())
            or len(packet['xy0']) != manifest['frozen_pixel_count']
            or not np.array_equal(packet['offsets'][[0,-1]], [0, len(packet['xy0'])])
            or np.any(np.diff(packet['offsets']) < 8)):
        raise ValueError('Strict board packet structure differs from frozen manifest')
    selected_indices = {int(shot['index']) for shot in shots}
    for row in manifest['source_files']:
        if row['kind'] == 'native' and row['shot_index'] in selected_indices:
            if digest(row['path']) != row['sha256']:
                raise ValueError('Native source motion cache changed')
    for shot in shots:
        index = int(shot['index'])
        coverage = manifest['coverage_by_shot'][str(index)]
        count = int(np.count_nonzero(packet['shot_index'] == index))
        expected = coverage['native_pair_count']-coverage['native_below8_pairs']
        expected += coverage['auxiliary_pair_count']
        if (coverage['camera_id'] != camera_id or
                coverage['native_frames'] != len(shot['pts']) or count != expected):
            raise ValueError(f'Strict board coverage differs for shot {index}')
    return dict(config=config, captured=captured, states=states, shots=shots,
                shot_indices=tuple(int(shot['index']) for shot in shots),
                camera_id=camera_id,
                active=active, pts=pts, mount=mount, lens=lens, retained=retained,
                excluded=dict(excluded), scene=scene, packet=packet,
                packet_path=packet_path, manifest=manifest, diag=diag)


def native_pairs(data):
    from native_board_pair import NativeBoardPair
    from reference_board_pair import BoardIntersectionError
    packet, captured, states = data['packet'], data['captured'], data['states']
    result, counts, invalid = [], Counter(), []
    counts_by_shot = {}
    components_by_shot = {}
    panels = np.arange(len(data['scene']['board_surface']['bottom_edge_xyz']))
    for shot in data['shots']:
        shot_index = int(shot['index'])
        own_pts = set(map(int, shot['pts']))
        own_counts = Counter()
        edges = []
        for group in grouped_native_pairs(data, shot_index):
            i = group['packet_pair_index']
            a, b = group['pts']
            if a not in own_pts or b not in own_pts:
                raise ValueError(f'Frozen board pair {i} crosses shot {shot_index}')
            source_file_id = group['source_file_id']
            source_file = data['manifest']['source_files'][source_file_id]
            if (source_file['kind'] != 'native' or
                    int(source_file['shot_index']) != shot_index):
                raise ValueError(f'Frozen board pair {i} belongs to another source cache')
            xy0, xy1 = group['xy0'], group['xy1']
            c0 = from_state(states[a], captured['templates'][a])
            c1 = from_state(states[b], captured['templates'][b])
            pair = NativeBoardPair(xy0, xy1, data['scene'], c0, c1,
                                   allowed_panels=panels)
            q = [states[a], states[b]]
            valid, reasons = pair.initial_validity(q)
            failures = [dict(row_index=int(row), reason=reasons[row])
                        for row in np.flatnonzero(~valid)]
            if not failures:
                try:
                    raw = pair._raw_residual(q)
                    matrices = pair._raw_jacobians(q)
                except BoardIntersectionError as exc:
                    failures = [dict(row_index=int(row), reason='jacobian_'+exc.reason)
                                for row in exc.indices]
                else:
                    if (not np.isfinite(raw).all() or
                            any(not np.isfinite(m).all() for m in matrices)):
                        failures = [dict(row_index=int(row), reason='nonfinite_residual_or_jacobian')
                                    for row in range(len(xy0))]
            if failures:
                invalid.append(dict(shot_index=shot_index, packet_pair_index=i,
                                    source_pair_index=group['source_pair_index'],
                                    source_pts=a, target_pts=b, pixel_count=len(xy0),
                                    invalid_rows=failures))
            counts['pixels'] += len(xy0)
            own_counts['pixels'] += len(xy0)
            category = 'adjacent_pairs' if b-a == 256 else 'auxiliary_pairs'
            counts[category] += 1
            own_counts[category] += 1
            edges.append((a, b))
            result.append((a, b, pair))
        counts_by_shot[str(shot_index)] = dict(own_counts)
        components_by_shot[str(shot_index)] = joint._components(shot['pts'], edges)
    reason_counts = dict(Counter(row['reason'] for entry in invalid
                                 for row in entry['invalid_rows']))
    validity = dict(all_selected_pixels_geometry_valid=not invalid,
                    invalid_pair_count=len(invalid), invalid_row_count=sum(
                        len(entry['invalid_rows']) for entry in invalid),
                    reason_counts=reason_counts, pairs=invalid)
    return result, dict(counts), counts_by_shot, components_by_shot, validity


def verify_conditioning(data):
    """Check every retained factor against its unchanged full-state callback."""
    full = IncrementalCameraGraph(11, window_size=len(data['states'])+2)
    small = IncrementalCameraGraph(11, window_size=len(data['active'])+2)
    for spec in data['retained']:
        getattr(full, 'add_'+spec['kind'])(*spec['args'], **spec['kwargs'])
        add_conditioned(small, spec, data['active'], data['states'])
    full_values, small_values = gtsam.Values(), gtsam.Values()
    for k, q in data['states'].items():
        full_values.insert(full._key(k), q)
        if k in data['active']:
            small_values.insert(small._key(k), q)
    errors = []
    jac_errors = []
    checked_coupled_types = set()
    for f, s, spec in zip(full._pending, small._pending, data['retained']):
        errors.append(abs(f.error(full_values)-s.error(small_values)))
        kind = (spec['file'], spec['function'])
        if (set(spec['keys'])-data['active'] and kind not in checked_coupled_types):
            fj, sj = f.linearize(full_values), s.linearize(small_values)
            fa, sa = fj.getA(), sj.getA()
            original_keys = list(fj.keys())
            selected_columns = np.concatenate([np.arange(original_keys.index(k)*11,
                (original_keys.index(k)+1)*11) for k in sj.keys()])
            jac_errors.append(float(np.max(np.abs(fa[:, selected_columns]-sa))))
            jac_errors.append(float(np.max(np.abs(fj.getb()-sj.getb()))))
            checked_coupled_types.add(kind)
    report = dict(factors_checked=len(errors),
        max_cost_difference=float(max(errors, default=0)),
        coupled_factor_types_checked=sorted(map(str, checked_coupled_types)),
        max_linearization_difference=float(max(jac_errors, default=0)))
    if report['max_cost_difference'] > 1e-9 or report['max_linearization_difference'] > 1e-9:
        raise ValueError(f'Conditioned callback changed the retained likelihood: {report}')
    return report


def build_graph(data, pairs, mode, rig_parameterization, *, diagnose_cost=False):
    """Build either unchanged soft-equality or exact-derived diagnostic graph."""
    if mode not in ('fixed', 'pivot') or rig_parameterization not in ('equality', 'derived'):
        raise ValueError('Unknown physical mode or rig parameterization')
    config, active = data['config'], data['active']
    states = {key: data['states'][key].copy() for key in active}
    if rig_parameterization == 'derived':
        for pt in data['pts']:
            states[pt][:3] = 0.
    graph = IncrementalCameraGraph(11, window_size=len(active)+2,
        relinearize_threshold=1e-5, refinement_updates=config['refinement_updates'],
        optimization=config['optimization'], diagnose_cost=diagnose_cost)
    for spec in data['retained']:
        if spec['kind'] == 'prior' and spec['keys'] == (data['mount'],):
            args = list(spec['args'])
            if mode == 'pivot':
                sigma = np.asarray(args[2]).copy()
                sigma[3:6] = config.get('offset_sigma_scene', .6)/100.
                args[2] = sigma
            graph.add_prior(*args, **spec['kwargs'])
        elif rig_parameterization == 'derived' and set(spec['keys']) & data['pts']:
            keys, residual, jac, options = conditioned_spec(
                spec, active, data['states'], data['pts'], data['mount'], mode)
            graph.add_observation(keys, residual, jac_fn=jac, **options)
        else:
            add_conditioned(graph, spec, active, data['states'])
    for pt in sorted(data['pts']):
        if rig_parameterization == 'derived':
            add_dummy_center_prior(graph, pt)
        elif mode == 'pivot':
            add_rig_equality(graph, pt, data['mount'], sigma_scene=.00001)
        else:
            joint._mount_equality(graph, pt, data['mount'])
    quality = data['captured']['source_quality']
    for a, b, pair in pairs:
        sigma = float(config.get('native_pair_sigma_px', 2.))
        if rig_parameterization == 'derived':
            keys, raw_residual, raw_jacobian = derived_native_pair(
                pair, a, b, data['mount'], mode)
        else:
            keys, raw_residual, raw_jacobian = (a, b), pair._raw_residual, pair._raw_jacobians
        residual, jac = quality.wrap(raw_residual, raw_jacobian,
            quality.pair_scales(a, b, pair.xy0, pair.xy1, 'native_motion'))
        residual, jac = pointwise_huber(residual, jac, group_size=2, delta=2*sigma)
        if config.get('native_pair_similarity_noise_px') is not None:
            residual, jac, _ = wrap_correlated_pair(residual, jac, pair.xy1,
                sigma_point_px=sigma, sigma_modes_px=config['native_pair_similarity_noise_px'])
            sigma = 1.
        graph.add_observation(keys, residual, residual_dim=2*len(pair.xy0),
                              jac_fn=jac, sigma=sigma)
    return graph, states


def initial_objective_audit(data, pairs, mode):
    """Evaluate both full conditional objectives at the same saved v13 source state."""
    costs, factor_counts = {}, {}
    for parameterization in ('equality', 'derived'):
        graph, states = build_graph(data, pairs, mode, parameterization)
        values = gtsam.Values()
        for key, state in states.items():
            values.insert(graph._key(key), state)
        costs[parameterization] = float(sum(factor.error(values) for factor in graph._pending))
        factor_counts[parameterization] = len(graph._pending)
    if not all(np.isfinite(value) for value in costs.values()):
        raise ValueError('One initial conditional objective is nonfinite')
    return dict(at='v13_saved_states_with_exact_rig_projection_for_derived',
                old_soft_equality_cost=costs['equality'],
                exact_derived_cost=costs['derived'],
                derived_minus_old_cost=costs['derived']-costs['equality'],
                factor_counts=factor_counts,
                removed_gaussian_rig_equalities=len(data['pts']),
                added_independent_dummy_priors=len(data['pts']),
                old_rig_sigma_state=1e-7,
                semantics='different_objectives; this cost comparison is not an accuracy test')


def solve_one(data, pairs, mode, output, rig_parameterization='equality',
              *, initial_override=None, initial_provenance=None):
    graph, states = build_graph(data, pairs, mode, rig_parameterization,
                                diagnose_cost=True)
    if initial_override is not None:
        if initial_provenance is None or set(initial_override) != set(states):
            raise ValueError('Initial override needs provenance and every active graph key')
        replacement = {}
        for key in states:
            value = np.asarray(initial_override[key], np.float64)
            if value.shape != (11,) or not np.isfinite(value).all():
                raise ValueError(f'Invalid initial override state: {key}')
            replacement[key] = value.copy()
        states = replacement
    count = len(graph._pending)
    t0 = time.perf_counter()
    solved = graph.update(states)
    # Preserve the returned iSAM2 estimate before batch LM. Do not recompute it
    # merely for checkpointing; an invalid trial or optimizer error remains
    # diagnosable without presenting this intermediate state as a final result.
    output.mkdir(parents=True, exist_ok=True)
    post_isam_path = output/'post_isam_states.npz'
    stage_keys = np.asarray(sorted(solved), np.int64)
    np.savez_compressed(post_isam_path, keys=stage_keys,
                        states=np.asarray([solved[k] for k in stage_keys]))
    post_isam_summary = dict(
        stage='post_isam_pre_batch_lm', mode=mode,
        rig_parameterization=rig_parameterization,
        batch_lm_completed=False, candidate_result=False,
        shot_indices=list(data['shot_indices']), camera_id=data['camera_id'],
        source_candidate_model_sha256=data['diag']['model_sha256'],
        frozen_board_packet_sha256=digest(data['packet_path']),
        driver_code_sha256=digest(__file__),
        states_sha256=digest(post_isam_path), states=len(solved), factors=count,
        custom_factor_callback=graph.callback_metadata,
        supported_motion=data.get('supported_motion_metadata'),
        supported_motion_fallback=data.get('supported_motion_fallback_metadata'),
        seconds=time.perf_counter()-t0, update=graph.last_update_summary)
    if initial_override is not None:
        post_isam_summary['initial_override_provenance'] = initial_provenance
    (output/'post_isam_summary.json').write_text(json.dumps(
        post_isam_summary, indent=2), encoding='utf8')
    solved = graph.batch_polish(**data['config']['batch_polish'])
    elapsed = time.perf_counter()-t0
    stamps = np.asarray(sorted(data['pts']), np.int64)
    if rig_parameterization == 'derived':
        cameras = np.asarray([derived_camera25(
            solved[int(p)], solved[data['mount']], data['captured']['templates'][int(p)], mode)
            for p in stamps])
    else:
        cameras = np.asarray([from_state(solved[int(p)], data['captured']['templates'][int(p)])
                              for p in stamps])
    np.savez_compressed(output/'shot_cameras.npz', pts=stamps, camera25=cameras,
        shot_index=np.asarray([next(int(shot['index']) for shot in data['shots']
                                    if int(shot['pts'][0]) <= stamp <= int(shot['pts'][-1]))
                               for stamp in stamps], np.int16),
        source_sha256=data['captured']['source_hash'],
        model_sha256=data['diag']['model_sha256'])
    keys = np.asarray(sorted(solved), np.int64)
    np.savez_compressed(output/'states.npz', keys=keys, states=np.asarray([solved[k] for k in keys]))
    result = dict(mode=mode, rig_parameterization=rig_parameterization,
        rig_parameterization_code_sha256=digest(rig_model.__file__),
        source_candidate_model_sha256=data['diag']['model_sha256'],
        frozen_board_packet_sha256=digest(data['packet_path']),
        removed_gaussian_rig_equalities=(len(data['pts']) if rig_parameterization == 'derived' else 0),
        shot_indices=list(data['shot_indices']),
        camera_id=data['camera_id'], frames=len(stamps), states=len(solved), factors=count,
        seconds=elapsed, update=graph.last_update_summary, batch=graph.last_batch_diagnostics,
        custom_factor_callback=graph.callback_metadata,
        supported_motion=data.get('supported_motion_metadata'),
        supported_motion_fallback=data.get('supported_motion_fallback_metadata'),
        post_isam_states_sha256=digest(post_isam_path),
        mount_state=solved[data['mount']].tolist(), lens_state=solved[data['lens']].tolist(),
        center_span_scene=np.ptp(cameras[:, :3], axis=0).tolist(),
        scope='Conditional same-camera shot diagnostic; all other cameras/ad/ring geometry fixed at v13. '
              'Not an independent validation or complete programme candidate.')
    if initial_override is not None:
        result['initial_override_provenance'] = initial_provenance
    (output/'summary.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--source-support-sidecar', type=Path,
                   help='Add original LK points with audited source-only board-interior support')
    p.add_argument('--source-support-sha256',
                   help='Explicitly frozen SHA256 of the reviewed source-support sidecar')
    p.add_argument('--source-support-fallback', type=Path,
                   help='Original raw LK fallback restricted to zero-support parent pairs')
    p.add_argument('--source-support-fallback-sha256')
    shot_group = p.add_mutually_exclusive_group(required=True)
    shot_group.add_argument('--shot', type=int,
                            help='Legacy one-shot diagnostic, including a subset of a physical camera')
    shot_group.add_argument('--shots', type=int, nargs='+',
                            help='All shots of one physical camera; e.g. --shots 3 5 8')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--run', action='store_true')
    p.add_argument('--mode', choices=('both', 'fixed', 'pivot'), default='both')
    p.add_argument('--rig-parameterization', choices=('equality', 'derived'),
                   default='equality',
                   help='equality retains the historical 1e-7-state Gaussian rig; '
                        'derived defines optical centres exactly from one mount')
    a = p.parse_args()
    if (a.source_support_sidecar is None) != (a.source_support_sha256 is None):
        p.error('--source-support-sidecar and --source-support-sha256 must be provided together')
    if ((a.source_support_fallback is None) != (a.source_support_fallback_sha256 is None)
            or (a.source_support_fallback is not None and a.source_support_sidecar is None)):
        p.error('Fallback requires its SHA256 and the original source-support sidecar')
    shot_indices = (a.shot,) if a.shot is not None else tuple(a.shots)
    a.output.mkdir(parents=True, exist_ok=True)
    _, callback_metadata = select_factory()
    print(json.dumps(dict(stage='conditional_run_start', pid=os.getpid(),
                          output=str(a.output.resolve()), shots=shot_indices,
                          custom_factor_callback=callback_metadata)), flush=True)
    data = prepare(a.config, a.candidate, a.manifest, shot_indices, a.output,
                   complete_camera_group=a.shots is not None)
    if a.source_support_sidecar is not None:
        data['supported_motion'], data['supported_motion_metadata'] = load_supported_motion(
            data, a.source_support_sidecar, a.manifest, a.source_support_sha256)
    if a.source_support_fallback is not None:
        extra, data['supported_motion_fallback_metadata'] = load_supported_motion(
            data, a.source_support_fallback, a.manifest, a.source_support_fallback_sha256)
        data['supported_motion'].extend(extra)
    conditioning_check = verify_conditioning(data)
    pairs, counts, counts_by_shot, components_by_shot, validity = native_pairs(data)
    components = (components_by_shot[str(a.shot)] if a.shot is not None
                  else components_by_shot)
    modes = ('fixed', 'pivot') if a.mode == 'both' else (a.mode,)
    initial_cost = ({mode: initial_objective_audit(data, pairs, mode) for mode in modes}
                    if a.rig_parameterization == 'derived' and
                    not validity['invalid_pair_count'] else None)
    report = dict(schema='rinkspace.conditioned-camera-comparison.v1',
        config_sha256=digest(a.config), candidate_states_sha256=digest(a.candidate/'graph_states.npz'),
        manifest_sha256=digest(a.manifest), packet_sha256=digest(data['packet_path']),
        code_sha256=digest(__file__),
        source_support_loader_sha256=digest(source_support_model.__file__),
        custom_factor_callback=callback_metadata,
        rig_parameterization_code_sha256=digest(rig_model.__file__), shot=a.shot,
        shot_indices=list(data['shot_indices']), camera_id=data['camera_id'],
        complete_camera_group=a.shots is not None, frames=len(data['pts']),
        active_states=len(data['active']),
        rig_parameterization=a.rig_parameterization,
        removed_gaussian_rig_equalities=(len(data['pts'])
            if a.rig_parameterization == 'derived' else 0),
        initial_objective_comparison=initial_cost,
        rig_semantics=('exact_C_equals_P_plus_RT_d_with_dummy_frame_centres; '
            'not_bitwise_equivalent_to_the_old_soft_rig_likelihood'
            if a.rig_parameterization == 'derived' else
            'historical_soft_gaussian_rig_equality'),
        retained_factors=len(data['retained']), excluded=data['excluded'],
        conditioning_check=conditioning_check,
        native_counts=counts, native_counts_by_shot=counts_by_shot,
        supported_motion=data.get('supported_motion_metadata'),
        supported_motion_fallback=data.get('supported_motion_fallback_metadata'),
        components=components, native_input_validity=validity,
        source_uv_changed=False, shared_scene_fixed=True, final_holdouts_used=False,
        full_programme_acceptance=False, optimized=False)
    (a.output/'preflight.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    summary = {**report, 'components': (
        [len(c) for c in components] if a.shot is not None else
        {shot: [len(c) for c in groups] for shot, groups in components.items()}),
        'native_input_validity': {key: value for key, value in validity.items()
                                  if key != 'pairs'}}
    print(json.dumps(summary), flush=True)
    if validity['invalid_pair_count']:
        raise ValueError('Dynamic board initialization is invalid; full indexed reasons '
                         f'are in {a.output/"preflight.json"}. No solver was started.')
    if a.run:
        for mode in modes:
            if (a.output/mode/'summary.json').exists():
                raise ValueError(f'Completed {mode} result exists; do not restart it')
            if (a.output/mode/'post_isam_states.npz').exists():
                raise ValueError(f'Incomplete {mode} stage checkpoint exists; inspect it before restarting')
            print(json.dumps(solve_one(data, pairs, mode, a.output/mode,
                                       a.rig_parameterization)), flush=True)


if __name__ == '__main__':
    main()
