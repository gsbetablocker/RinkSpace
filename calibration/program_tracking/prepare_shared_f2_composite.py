"""Read-only actual-source gate for a two-rig, one-F2 composite graph.

Usage (from the repository root)::

    D:/CodexData/RinkSpace_program_tracking_tmp/gtsam-env/python.exe \
      calibration/program_tracking/prepare_shared_f2_composite.py \
      --report calibration/program_tracking/shared_f2_composite_gate_20260928.json

This reconstructs saved rigs and captured original factors, checks exact
camera/prior rekeying, tests all captured low-end F2 factors with the candidate
scene delta, and constructs/evaluates a tiny real GTSAM composite. It never
calls graph.update(), optimization, or writes into a running solve directory.
Full graph builders may pass their complete pre-update graphs to
``CompositeFactorTape.append_complete_rig``; this gate does not invoke them.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile

from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import gtsam
import numpy as np

from audit_quadratic_result import reconstruct
from audit_time_principal_result import _rig as low_rig
from audit_time_principal_result import _cameras as low_cameras
from continuous_time_principal_optin import reconstruct_time_principal_rigs
from continuous_rig_rekey import SCENE_RING_KEYS, rekey_rig_pair
from conditioned_camera_comparison import capture_original
from ice_refine import _world_curves
from low_end_calibration_transfer_profile import seed_driver
from probe_shared_f2_nested import main_sources, add_main_reference_rig
from shared_markers import RingPixels, ring_world
from shared_scene_calibration import SharedRingScene
from backend.isam2 import IncrementalCameraGraph
from continuous_shared_f2_composite import (
    F2_KEY, CompositeFactorTape, adapt_captured_f2, captured_f2_spec)


TMP = Path('D:/CodexData/RinkSpace_program_tracking_tmp')
ROOT = Path(__file__).resolve().parents[2]
MAIN = TMP/'continuous_camera_model_v13_source_identity/main_long_side_crash_resume'
LOW = TMP/'continuous_camera_model_v11_time_principal/low_end'
CANDIDATE = TMP/'joint_programme_atlas_provenance_full_v13/candidate_scene.json'
ORIGINAL = ROOT/'calibration/deliverables/scene.json'
ORIGINAL_STATES = TMP/'joint_programme_atlas_provenance_full_v13/graph_states.npz'
LOW_F2_PTS = (229984256, 230353920, 230361600, 230368256)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def packet(path):
    with np.load(io.BytesIO(Path(path).read_bytes()), allow_pickle=False) as z:
        keys, states = z['keys'], z['states']
    if (states.shape != (len(keys), 11) or len(set(map(int, keys))) != len(keys)
            or not np.isfinite(states).all()):
        raise ValueError(f'Invalid Vector11 state packet: {path}')
    return {int(k): v.copy() for k, v in zip(keys, states)}


def low_models(directory=LOW):
    directory=Path(directory)
    phase = json.loads((directory/'phase_00.json').read_text(encoding='utf8'))
    states = packet(directory/'states.npz')
    prior = packet(directory/'prior_initializer_states.npz')
    programme = low_rig(phase, low_cameras(directory/'shot_cameras.npz', reference=False),
                        reference=False)
    references = low_rig(phase, low_cameras(directory/'reference_cameras.npz', reference=True),
                         reference=True)
    programme, references = reconstruct_time_principal_rigs(
        programme, references, phase['time_principal'])
    if set(states) != set(prior):
        raise ValueError('Low saved state/prior keysets differ')
    return phase, programme, references, states, prior


def _find_ring_groups(fn):
    """Read the exact RingPixels captured by an original callback closure."""
    seen, found = set(), []
    def walk(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, RingPixels):
            found.append(value)
        elif callable(value) and getattr(value, '__closure__', None):
            for cell in value.__closure__:
                try:
                    walk(cell.cell_contents)
                except ValueError:
                    pass
    walk(fn)
    return found


def _factor_values(keys, states):
    return [np.asarray(states[int(key)], np.float64).copy() for key in keys]


def _gtsam_values(keys, states):
    values = gtsam.Values()
    for key in keys:
        values.insert(gtsam.symbol('x', int(key)), states[int(key)])
    return values


def prepare(report_path: Path):
    main_state_sha = sha(MAIN/'accepted_states.npz')
    main_phase, main_model, main_refs, main_states, main_context = reconstruct(
        MAIN, state_name='accepted_states.npz', require_quadratic=True)
    main_prior = packet(MAIN/'prior_initializer_states.npz')
    low_phase, low_model, low_refs, low_states, low_prior = low_models()
    if (main_phase['graph_factors_with_source_calibration'] != 27481
            or low_phase['graph_factors_with_source_calibration'] != 10819
            or len(main_model.frames) != 6642 or len(low_model.frames) != 2219
            or len(main_refs.frames) != 11 or len(low_refs.frames) != 3
            or set(main_states) != set(main_prior)):
        raise ValueError('Saved complete graph inventory/prior graph differs')
    if main_phase['source_sha256'] != low_phase['source_sha256']:
        raise ValueError('Two camera rigs are not bound to the same source')
    original_scene = json.loads(ORIGINAL.read_text(encoding='utf8'))
    candidate_scene = json.loads(CANDIDATE.read_text(encoding='utf8'))
    original_graph_states = packet(ORIGINAL_STATES)
    q_saved = original_graph_states[F2_KEY]
    if (main_phase['model_sha256'] != sha(CANDIDATE)
            or low_phase['model_sha256'] != sha(CANDIDATE)):
        raise ValueError('The two camera rigs use different candidate scenes')
    released = SharedRingScene(original_scene,
        {'features': ['C0', 'F2', 'R3', 'R4']})
    if released.keys['F2'] != F2_KEY:
        raise ValueError('Shared F2 key mapping changed')
    world_base = _world_curves(original_scene)['F2_outer']
    world_candidate = _world_curves(candidate_scene)['F2_outer']
    metadata = released.features['F2']
    group = RingPixels('F2', 'outer', 0, 'scene', 'F2_outer', str(ORIGINAL),
        np.zeros(25), world_base, metadata['center'], metadata['outer_radius'],
        metadata['middle_radius'], np.zeros((1, 2)), np.ones(1))
    world_error = float(np.max(np.abs(ring_world(group, q_saved)-world_candidate)))
    if world_error > 1e-10:
        raise ValueError('Candidate F2 world already includes a different offset')
    feature = next(f for f in candidate_scene['ice_features'] if f['id'] == 'F2')
    outer = next(r for r in feature['rings'] if r['role'] == 'outer')
    center = np.asarray(outer['center_xyz'][:2], float)
    radius = float(outer['radius'])
    if (np.max(np.abs(metadata['center']+100*q_saved[:2]-center)) > 1e-10
            or abs(metadata['outer_radius']+100*q_saved[2]-radius) > 1e-10):
        raise ValueError('Candidate absolute F2 center/radius is double-offset')

    main_rekey = rekey_rig_pair(main_model, main_refs, main_states, main_prior,
        main_phase, key_start=3_000_000_000, reference_shot_start=11_000,
        reserved_keys=SCENE_RING_KEYS)
    low_rekey = rekey_rig_pair(low_model, low_refs, low_states, low_prior,
        low_phase, key_start=4_000_000_000, reference_shot_start=12_000,
        reserved_keys=set(main_rekey.states)|SCENE_RING_KEYS,
        reserved_shot_ids=set(main_rekey.references.shot_splines))
    if (set(main_rekey.states) & set(low_rekey.states)
            or set(main_rekey.programme.frames) & set(low_rekey.programme.frames)):
        raise ValueError('Two rekeyed camera groups collide')
    main_curves, _, source_frames = main_sources(candidate_scene, center, radius)
    main_with_new_refs = {key: value.copy() for key, value in main_rekey.states.items()}
    new_main_ref_model, new_main_ref_rows = add_main_reference_rig(
        main_rekey.programme, main_with_new_refs, main_curves, source_frames)
    added = set(main_with_new_refs)-set(main_rekey.states)
    if (len(new_main_ref_rows) != 14 or len(added) != 28
            or added & set(low_rekey.states)
            or set(new_main_ref_model.frames) &
            (set(main_rekey.programme.frames)|set(main_rekey.references.frames))):
        raise ValueError('Fourteen new main F2 references do not have 28 new nodes')

    # Source capture constructs exactly the original callback graph, stopping
    # before its first update. Nothing is optimized or added to the source.
    with tempfile.TemporaryDirectory(prefix='rinkspace-f2-capture-') as td:
        with redirect_stdout(io.StringIO()):
            captured = capture_original(seed_driver.CONFIG, Path(td))
    f2_specs = [s for s in captured['specs'] if captured_f2_spec(s)
                and int(s['keys'][0]) in low_model.frames]
    if tuple(sorted(int(s['keys'][0]) for s in f2_specs)) != LOW_F2_PTS:
        raise ValueError('All four original low F2 callbacks were not captured')
    low_delta_states = {**low_rekey.states, F2_KEY: np.zeros(11)}
    old_graph = IncrementalCameraGraph(11, window_size=100)
    replacements = {}
    checks = []
    for spec in sorted(f2_specs, key=lambda s: int(s['keys'][0])):
        pts = int(spec['keys'][0])
        rings = _find_ring_groups(spec['args'][1])
        if (len(rings) != 1 or rings[0].feature_id != 'F2'
                or rings[0].role != 'outer'):
            raise ValueError(f'Captured F2 callback lost exact world group: {pts}')
        captured_world_error = float(np.max(np.abs(
            ring_world(rings[0], q_saved)-world_candidate)))
        if captured_world_error > 1e-10:
            raise ValueError(f'Captured F2 callback world differs at {pts}')
        before = adapt_captured_f2(spec, low_model, set(low_states),
            original_graph_states, q_saved, release_scene=False)
        after = adapt_captured_f2(spec, low_rekey.programme,
            set(low_delta_states), original_graph_states, q_saved,
            release_scene=True)
        before_values = _factor_values(before[0], low_states)
        after_values = _factor_values(after[0], low_delta_states)
        old_r = before[1](before_values)
        new_r = after[1](after_values)
        old_j = before[2](before_values)
        new_j = after[2](after_values)
        if (not np.array_equal(old_r, new_r)
                or before[3] != after[3]
                or len(after[0]) != len(before[0])+1
                or after[0][-1] != F2_KEY
                or any(not np.array_equal(a, b)
                       for a, b in zip(old_j, new_j[:-1]))):
            raise ValueError(f'Zero-release low captured F2 changed: {pts}')
        old_graph.add_observation(before[0], before[1],
            jac_fn=before[2], **before[3])
        replacements[len(old_graph._pending)-1] = after
        checks.append(dict(source_pts=pts, source_uv=len(rings[0].uv),
            residual_dim=int(before[3]['residual_dim']),
            source_sigma_range=[float(np.min(rings[0].sigma)),
                                float(np.max(rings[0].sigma))],
            original_factor_sigma=float(before[3]['sigma']),
            captured_world_max_abs_delta=captured_world_error,
            residual_exact=True, camera_jacobian_exact=True,
            original_callback_and_noise_preserved=True))
    # A real camera prior is part of the mini graph and must survive rekeying.
    first_camera_key = int(old_graph._callbacks[0][0][0])
    old_graph.add_prior(first_camera_key, low_states[first_camera_key],
                        np.full(11, .1))
    tape = CompositeFactorTape()
    composed = tape.append_complete_rig('low_end_mini', old_graph,
        low_rekey.key_mapping, replacements=replacements)
    old_values = _gtsam_values(
        {k for factor in old_graph._pending
         for k in CompositeFactorTape._factor_keys(factor)}, low_states)
    new_values = _gtsam_values(
        {k for factor in tape.factors
         for k in CompositeFactorTape._factor_keys(factor)}, low_delta_states)
    old_costs = [float(f.error(old_values)) for f in old_graph._pending]
    new_costs = [float(f.error(new_values)) for f in tape.factors]
    if not np.allclose(old_costs, new_costs, rtol=0, atol=1e-11):
        raise ValueError('Candidate-zero GTSAM factor costs differ')
    prior_manifest = tape.append_original_f2_scene_prior(q_saved,
        outer_radius=metadata['outer_radius'],
        middle_radius=metadata['middle_radius'])
    if sha(MAIN/'accepted_states.npz') != main_state_sha:
        raise ValueError('Main accepted state changed during the gate')
    report = dict(schema='rinkspace.shared-f2-composite-gate.v1',
        source_sha256=main_phase['source_sha256'],
        original_scene_sha256=sha(ORIGINAL), candidate_scene_sha256=sha(CANDIDATE),
        original_f2_state11=q_saved.tolist(),
        candidate_zero_world_max_abs_delta=world_error,
        main=dict(state_path=str(MAIN/'accepted_states.npz'),
            state_sha256=main_state_sha,
            complete_factor_count=main_phase['graph_factors_with_source_calibration'],
            prior_target_count=len(main_rekey.prior),
            rekey=main_rekey.verification,
            existing_reference_count=len(main_rekey.references.frames),
            new_f2_reference_count=len(new_main_ref_rows),
            new_f2_reference_graph_nodes=len(added),
            new_f2_source_uv=sum(len(x.uv) for x in main_curves.values())),
        low=dict(state_path=str(LOW/'states.npz'),
            state_sha256=sha(LOW/'states.npz'),
            complete_factor_count=low_phase['graph_factors_with_source_calibration'],
            prior_target_count=len(low_rekey.prior),
            rekey=low_rekey.verification,
            captured_f2_original_factors=checks),
        composite=dict(original_complete_factor_count=27481+10819,
            mini_graph=composed,
            mini_old_factor_costs=old_costs,
            mini_candidate_zero_factor_costs=new_costs,
            mini_factor_cost_max_abs_difference=float(max(
                abs(a-b) for a, b in zip(old_costs,new_costs))),
            scene_prior=prior_manifest,
            full_complete_graph_instantiated=False,
            solver_started=False,
            fresh_blind_accuracy_claim=False))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report = prepare(args.report)
    print(json.dumps(dict(report=str(args.report.resolve()),
        main_factors=report['main']['complete_factor_count'],
        low_factors=report['low']['complete_factor_count'],
        low_captured_f2=len(report['low']['captured_f2_original_factors']),
        zero_world_error=report['candidate_zero_world_max_abs_delta'],
        zero_factor_error=report['composite']['mini_factor_cost_max_abs_difference'])))


if __name__ == '__main__':
    main()
