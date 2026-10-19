"""Preflight-only reconstruction and zero-release check of two complete graphs.

This deliberately calls each historical driver's own ``run`` with its
preflight-only switch in a fresh TMP folder. It never invokes solve/update.
The original native managers remain alive until all factors are checked.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time

from runtime.bootstrap import add_geometry_dependencies
from rinkspace.storage import resolve_asset
add_geometry_dependencies()
import gtsam
import numpy as np

import fit_continuous_source_identity_repaired as main_entry
import fit_low_end_temporal_principal as low_entry
from fit_continuous_rig_source_calibrated import release_native_resources
from continuous_camera_solver import values_for
from continuous_shared_f2_composite import (
    F2_KEY, CompositeFactorTape, adapt_captured_f2)
from prepare_shared_f2_composite import (
    TMP, ROOT, MAIN, LOW, ORIGINAL_STATES, packet, low_models, sha)
from native_aware_shared_f2_rebuild import (
    capture_low_builder_context, rebuild_low_complete)
from audit_quadratic_result import reconstruct
from continuous_rig_rekey import SCENE_RING_KEYS, rekey_rig_pair
from conditioned_camera_comparison import capture_original
from low_end_calibration_transfer_profile import seed_driver


def _main_args(output):
    return SimpleNamespace(camera='main_long_side',
        warm_group=TMP/'continuous_camera_model_v12_complete_motion/stopped_before_source_identity_repair',
        warm_state='accepted_states.npz',
        prior_state_cache=TMP/'continuous_camera_model_v4_domain_preflight/main_long_side',
        output=output,
        rail_review=TMP/'far_rail_reviewed_source_v1/far_rail_reviewed_source_selection.json',
        rail_review_sha256='61eb15d384d43c16959375c79f6d560d78b8fc291eefaeea7354644e8210fd5b',
        motion_sidecar=TMP/'continuous_source_motion_requalification_v2/main_hint0_source_review_pending.json',
        motion_sidecar_sha256='345da13109dd394288d4de127575d755743d6e784c5dfba4eb12d00b550296e5',
        motion_review=resolve_asset(ROOT, 'evidence/round10_tail_user_issue_20260928/main_hint0_source_review_v2/agent_reviewed_motion_tail_v2_union.json'),
        motion_review_sha256='2083502f51c97314241ad1ca17ad7a81226b854f22f1993514dcb771dd086ea9',
        native_board_first_hit=True, native_central_domain=True,
        central_edge_brown15=True, native_source_edges=True,
        native_edge_threads=8, reference_curve_workers=8,
        native_trial_values=True, native_affine_priors=True,
        adaptive_source_curves=True, implicit_adaptive_curve_jacobian=True,
        initialize_absolute_sources=False, absolute_initializer_resume_from=None,
        nearboard_source_report=None, surface_review=None,
        iterations=0, preflight_only=True)


def _low_args(output, directory=LOW):
    phase=json.loads((Path(directory)/'phase_00.json').read_text(encoding='utf8'))
    return SimpleNamespace(
        warm_group=Path(phase['warm_estimate']['directory']),
        warm_state='states.npz',
        prior_state_cache=Path(phase['initializer_adaptation']['cache_directory']),
        initializer_map=Path(phase['reference_initializer_provenance']['path']),
        output=output, native_edge_threads=8, reference_curve_workers=8,
        include_user_f1=True, iterations=0, preflight_only=True)


def _observation_family(callback):
    """Read the family captured by the original Python factor wrapper."""
    closure = inspect.getclosurevars(callback).nonlocals
    observation = closure.get('spec')
    if observation is None:
        return 'custom_unknown', None
    residual = observation.residual_fn
    parts = inspect.getclosurevars(residual).nonlocals
    family = parts.get('family')
    source_keys = parts.get('source_keys')
    if family is None:
        defaults = getattr(residual, '__kwdefaults__', None) or {}
        family = defaults.get('family') or residual.__qualname__
    return str(family), tuple(source_keys) if source_keys is not None else None


def _factor_families(graph, manager):
    callbacks = iter(graph._callbacks)
    result = []
    for index, factor in enumerate(graph._pending):
        if index in getattr(manager, '_records', {}):
            result.append(('native_source_edge', int(factor.dim())))
        elif isinstance(factor, gtsam.CustomFactor):
            _, callback = next(callbacks)
            name, _ = _observation_family(callback)
            result.append((name, int(factor.dim())))
        elif isinstance(factor, gtsam.PriorFactorVector):
            result.append(('prior', int(factor.dim())))
        elif isinstance(factor, gtsam.BetweenFactorVector):
            result.append(('between', int(factor.dim())))
        else:
            result.append(('native_affine_prior', int(factor.dim())))
    try:
        next(callbacks)
    except StopIteration:
        return result
    raise ValueError('Custom callback/factor inventory differs')


def _low_f2_indices(graph):
    callbacks = iter(graph._callbacks)
    found = {}
    for index, factor in enumerate(graph._pending):
        if not isinstance(factor, gtsam.CustomFactor):
            continue
        _, cb = next(callbacks)
        family, keys = _observation_family(cb)
        if family == 'shared_markers.py:add_shared_ring_factor' and keys:
            if len(keys) == 2 and int(keys[1]) == F2_KEY:
                found[int(keys[0])] = index
    expected = {229984256,230353920,230361600,230368256}
    if set(found) != expected:
        raise ValueError(f'Original low graph F2 indices differ: {found}')
    return found


def _cost_by_family(factors, families, values):
    summary = defaultdict(lambda: {'factors': 0, 'residual_rows': 0, 'cost': 0.})
    costs = []
    for factor, (family, rows) in zip(factors, families):
        cost = float(factor.error(values))
        if not np.isfinite(cost):
            raise ValueError('Nonfinite factor cost')
        row = summary[family]
        row['factors'] += 1
        row['residual_rows'] += int(rows)
        row['cost'] += cost
        costs.append(cost)
    return costs, dict(summary)


def run(report_path: Path, *, only=None, verify_native_low=False):
    started = time.perf_counter()
    print(json.dumps(dict(stage='start',only=only)),flush=True)
    low_resources, main_resources = {}, {}
    rebuilt_low_resources = {}
    tape = CompositeFactorTape()
    original_state = packet(ORIGINAL_STATES)[F2_KEY]
    with tempfile.TemporaryDirectory(prefix='shared-f2-complete-', dir=TMP) as temp:
        folder = Path(temp)
        print(json.dumps(dict(stage='temp_ready',folder=str(folder))),flush=True)
        try:
            print(json.dumps(dict(stage='low_preflight_start')),flush=True)
            low_context={}
            if verify_native_low:
                with capture_low_builder_context(low_context):
                    low_entry.run(_low_args(folder/'low_end'), low_resources)
            else:
                low_entry.run(_low_args(folder/'low_end'), low_resources)
            print(json.dumps(dict(stage='low_preflight_complete',factors=len(low_resources['graph']._pending))),flush=True)
            low_graph = low_resources['graph']
            low_manager = low_resources['native_edge_manager']
            low_phase, low_model, low_refs, low_states, low_prior = low_models()
            if len(low_graph._pending) != 10819 or set(low_states) != set(low_prior):
                raise ValueError('Complete low graph changed factors or prior keys')
            low_rekey = rekey_rig_pair(low_model, low_refs, low_states, low_prior,
                low_phase, key_start=4_000_000_000,
                reference_shot_start=12_000, reserved_keys=SCENE_RING_KEYS)
            with tempfile.TemporaryDirectory(prefix='shared-f2-original-capture-') as src:
                captured = capture_original(seed_driver.CONFIG, Path(src))
            specs = {int(s['keys'][0]): s for s in captured['specs']
                     if (s['kind']=='observation' and s['file']=='shared_markers.py'
                         and s['function']=='add_shared_ring_factor'
                         and int(s['keys'][1])==F2_KEY
                         and int(s['keys'][0]) in low_model.frames)}
            indices = _low_f2_indices(low_graph)
            low_joint_states = {**low_rekey.states, F2_KEY: np.zeros(11)}
            replacements = {}
            for pts,index in indices.items():
                replacements[index] = adapt_captured_f2(
                    specs[pts],low_rekey.programme,set(low_joint_states),
                    packet(ORIGINAL_STATES),original_state,release_scene=True)
            low_append = tape.append_complete_rig('low_end', low_graph,
                low_rekey.key_mapping,replacements=replacements)
            native_low_gate=None
            if verify_native_low:
                rebuilt_low=rebuild_low_complete(low_context, low_phase,
                    low_model,low_refs,low_states,low_prior,original_state,
                    _low_args(folder/'low_end_native'),
                    resources=rebuilt_low_resources)
                old_values=values_for(low_graph,low_states)
                native_values=values_for(rebuilt_low.graph,rebuilt_low.states)
                old_factor_cost=[float(f.error(old_values))
                                 for f in low_graph._pending]
                new_factor_cost=[float(f.error(native_values))
                                 for f in rebuilt_low.graph._pending]
                if len(old_factor_cost)!=10819 or len(new_factor_cost)!=10819:
                    raise ValueError('Native-aware low rebuild changed factor count')
                native_difference=np.abs(np.asarray(old_factor_cost)-
                                         np.asarray(new_factor_cost))
                if float(np.max(native_difference))>1e-7:
                    raise ValueError('Native-aware low F2 zero delta differs')
                native_low_gate=dict(factors=10819,
                    cost_original=float(sum(old_factor_cost)),
                    cost_rebuilt=float(sum(new_factor_cost)),
                    max_factor_cost_absolute_difference=float(np.max(native_difference)),
                    f2_released_factors=rebuilt_low.f2_released_factor_count,
                    original_native_affine_backend_retained=True,
                    solver_started=False)
                print(json.dumps(dict(stage='native_low_zero_gate',
                                      **native_low_gate)),flush=True)
            if only == 'low':
                outcome=dict(low=low_append,native_low=native_low_gate,
                    elapsed_seconds=time.perf_counter()-started)
                if verify_native_low:
                    report_path.write_text(json.dumps(outcome,indent=2)+'\n',
                                           encoding='utf8')
                return outcome

            main_entry.run(_main_args(folder/'main_long_side'), main_resources)
            main_graph=main_resources['graph']
            main_manager=main_resources['native_edge_manager']
            main_phase,main_model,main_refs,main_states,_=reconstruct(
                MAIN,state_name='accepted_states.npz',require_quadratic=True)
            main_prior=packet(MAIN/'prior_initializer_states.npz')
            if len(main_graph._pending)!=27481 or set(main_states)!=set(main_prior):
                raise ValueError('Complete main graph changed factors or prior keys')
            main_rekey=rekey_rig_pair(main_model,main_refs,main_states,main_prior,
                main_phase,key_start=3_000_000_000,
                reference_shot_start=11_000,
                reserved_keys=set(low_rekey.states)|SCENE_RING_KEYS)
            main_append=tape.append_complete_rig('main_long_side',main_graph,
                main_rekey.key_mapping)
            if len(tape.factors)!=27481+10819:
                raise ValueError('Combined complete factor count differs')
            combined_states={**main_rekey.states,**low_joint_states}
            if len(combined_states)!=len(main_rekey.states)+len(low_joint_states):
                raise ValueError('Composite graph state key collision')
            old_low_values=values_for(low_graph,low_states)
            old_main_values=values_for(main_graph,main_states)
            new_values=values_for(main_graph,combined_states)
            families_low=_factor_families(low_graph,low_manager)
            families_main=_factor_families(main_graph,main_manager)
            old_low_cost, old_low_groups=_cost_by_family(low_graph._pending,
                families_low,old_low_values)
            old_main_cost, old_main_groups=_cost_by_family(main_graph._pending,
                families_main,old_main_values)
            new_cost, new_groups=_cost_by_family(tape.factors,
                families_low+families_main,new_values)
            original_cost=old_low_cost+old_main_cost
            diff=np.abs(np.asarray(original_cost)-np.asarray(new_cost))
            if len(diff)!=38300 or float(np.max(diff))>1e-7:
                raise ValueError('Full composite factor cost differs at zero F2 release')
            report=dict(schema='rinkspace.complete-two-rig-shared-f2-zero-gate.v1',
                main_saved_state_sha256=sha(MAIN/'accepted_states.npz'),
                low_saved_state_sha256=sha(LOW/'states.npz'),
                main=main_append,low=low_append,
                original_complete_factors=38300,
                low_f2_factor_indices={str(k):v for k,v in indices.items()},
                old_objective=float(sum(original_cost)),
                new_zero_delta_objective=float(sum(new_cost)),
                max_factor_cost_absolute_difference=float(np.max(diff)),
                original_cost_by_rig_family={'low_end':old_low_groups,
                                             'main_long_side':old_main_groups},
                composite_cost_by_factor_family=new_groups,
                source_pixel_accuracy_certified=False,
                optimization_started=False,
                elapsed_seconds=time.perf_counter()-started)
            report_path.parent.mkdir(parents=True,exist_ok=True)
            report_path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
            return report
        finally:
            # Native C++ factor owners must die before their session/threads.
            tape.factors.clear(); tape.callbacks.clear()
            gc.collect()
            release_native_resources(main_resources)
            release_native_resources(rebuilt_low_resources)
            release_native_resources(low_resources)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--only-low',action='store_true')
    parser.add_argument('--verify-native-low',action='store_true')
    args=parser.parse_args()
    outcome=run(args.report,only='low' if args.only_low else None,
                verify_native_low=args.verify_native_low)
    print(json.dumps(outcome if args.only_low else dict(
        report=str(args.report.resolve()),
        factors=outcome['original_complete_factors'],
        max_factor_cost_delta=outcome['max_factor_cost_absolute_difference'],
        seconds=outcome['elapsed_seconds'])))


if __name__=='__main__':
    main()
