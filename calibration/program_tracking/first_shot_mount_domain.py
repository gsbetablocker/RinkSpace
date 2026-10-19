"""Release unmeasured photo search ranges without changing the camera/source chart.

Opt-in for a new adapter/run only. No historical facts, binding, checkpoint,
source measurements or active process are rewritten.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
import traceback
import types

import numpy as np

AUTHORIZATION = dict(
    host_id='private-host-00001', thread_id='private-thread-00005',
    turn_id='private-turn-00001',
    user_message_id='private-message-00001',
    user_quote='我好像记得当时我们以最可能的体育转播摄像机假设了一套支架偏心参数？你检查一下是不是这样？如果这不是出自有准确依据（比如Z190实际规格参数）我觉得应该允许放开？',
    scope='Release photo-estimated numeric mount search windows only; retain fixed shared installation and all other physical/source conditions.')


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _first_failure(report):
    if not isinstance(report, dict):
        return {}
    failures = report.get('failures', report.get('unsupported', []))
    first = failures[0] if failures else report
    missing = first.get('missing_queries', [])
    return dict(frame=first.get('index'), native_pts=first.get('native_pts'),
                feature=first.get('source_feature_id', first.get('target_key')),
                target=first.get('target_key'), query=missing[0] if missing else first.get('query_id'))


def install_unmeasured_mount_domain(owner, *, entry_reference=None,
                                    failure_directory=None, merge_print_domains=None):
    """Remove four routes of one non-measured numeric box on this adapter.

    Camera mappings, state vectors, source/raw/J methods and all non-mount guard
    code remain the same. The existing provider print domains remain in the QP.
    """
    if hasattr(owner, '_unmeasured_mount_domain'):
        raise ValueError('Photo mount domain release is already installed')
    state = owner.state
    original = state.original
    facts_ref = deepcopy(original.mount_facts_reference)
    if reference(facts_ref['path']) != facts_ref:
        raise ValueError('Authenticated photo mount facts changed')
    facts = json.loads(Path(facts_ref['path']).read_text(encoding='utf8'))
    mount = facts['rigid_mount_offset']
    if (mount.get('bounds_are_user_working_search_ranges_not_measurements') is not True
            or mount.get('shared_across_same_installation_exposures') is not True
            or mount.get('per_frame_free_optical_centre_enabled') is not False):
        raise ValueError('Explicit unmeasured photo window and shared fixed mount facts required')
    lower = np.asarray(mount['lower_camera_xyz_m'], float)
    upper = np.asarray(mount['upper_camera_xyz_m'], float)
    if not np.array_equal(lower, original.mount_absolute_lower) or not np.array_equal(upper, original.mount_absolute_upper):
        raise ValueError('Active numeric box is not the authenticated photo search window')
    if merge_print_domains is None:
        from print_domain_shared_constraints import merge_mount_and_print
        merge_print_domains = merge_mount_and_print
    q_before = np.asarray(owner.start).copy()
    historical = {int(k): tuple(v) for k, v in original.historical_mount_parameter_bounds.items()}
    if set(historical) != {121, 122, 123}:
        raise ValueError('Explicit original three-coordinate photo-bound inventory required')
    changed = []
    objects = [('complete_state', state), ('original_state', original),
               ('world_state', getattr(getattr(owner, 'bundle', None), 'state', None)),
               ('unmodified_chart_copy', getattr(state, '_unmodified_original', None))]
    seen = set()
    for label, obj in objects:
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        for attr in ('shared_parameter_bounds', 'active_shared_parameter_bounds'):
            if attr not in vars(obj):
                continue
            before = getattr(obj, attr)
            after = dict(before)
            removed = {}
            for column in historical:
                if column in after:
                    if tuple(after[column]) != historical[column]:
                        raise ValueError('Unexpected non-photo bound on a mount coordinate')
                    removed[column] = after.pop(column)
            setattr(obj, attr, after)
            changed.append(dict(state=label, field=attr,
                                removed={str(k): list(v) for k, v in removed.items()},
                                all_other_bounds_retained=True))
        if 'mount_bounds_active' in vars(obj):
            obj.mount_bounds_active = False
    original.mount_bounds_active = False
    if hasattr(owner, 'guard_cache'):
        owner.guard_cache.clear()
    if not np.array_equal(q_before, owner.start):
        raise ValueError('Domain release changed the complete state')
    producer = reference(__file__)
    frozen_refs = [producer] + ([] if entry_reference is None else [deepcopy(entry_reference)])
    for ref in frozen_refs:
        if reference(ref['path']) != ref:
            raise ValueError('Domain release producer reference changed')
    if hasattr(owner, 'recovery_references'):
        owner.recovery_references.extend(frozen_refs)
    receipt = dict(schema='rinkspace.unmeasured-photo-mount-numeric-domain-release.v1',
        producer=producer, entry=entry_reference, authorization=deepcopy(AUTHORIZATION),
        facts=facts_ref, historical_lower_working_m=lower.tolist(), historical_upper_working_m=upper.tolist(),
        numeric_window_was_not_measured=True, historical_facts_and_binding_unchanged=True,
        total_offset_box_enforced=False, independent_d_box_enforced=False,
        shared_parameter_bound_audit=changed, no_new_numeric_bounds_or_priors=True,
        fixed_shared_pivot_two_axis_PP_RF4_row_clock_and_source_chart_retained=True,
        all_non_mount_physical_guards_retained=True, original_print_domain_QP_rows_retained=True,
        source_r_J_UV_ID_sigma_whitening_and_objective_mathematically_unchanged=True,
        same_physical_guard=False, guard_change='Only reject/projection routes of the unmeasured photo numeric window are disabled',
        conditional_source_metric_may_still_be_built_but_never_projects_to_photo_box=True,
        failure_directory=None if failure_directory is None else str(Path(failure_directory)))
    original_identity = owner.checkpoint_identity
    original_guard = owner.guard
    started = time.perf_counter()

    def check_producers():
        for ref in frozen_refs:
            if reference(ref['path']) != ref:
                raise ValueError('Frozen photo-domain release producer changed')

    def linear_constraints(adapter, q, selected):
        check_producers()
        if not np.array_equal(selected, adapter.shared):
            raise ValueError('Complete original shared coordinate order required')
        empty = dict(matrix=np.zeros((0, len(selected))), lower=np.zeros(0),
                     report=dict(kind='unmeasured_photo_numeric_mount_window_released'))
        result = merge_print_domains(adapter.state, q, selected, empty)
        result['report']['unmeasured_photo_numeric_mount_domain'] = receipt
        return result

    def recover(adapter, parent, proposed):
        check_producers()
        parent = adapter.state._q(parent)
        proposed = adapter.state._q(proposed)
        return dict(q=proposed.copy(), report=dict(kind='photo_numeric_box_recovery_disabled',
            modified_parameter_indices=[], all_proposed_coordinates_preserved=True,
            complete_physical_guard_and_actual_source_cost_still_required=True))

    def identity(adapter):
        check_producers()
        result = original_identity()
        result.update(total_old_v4_offset_box_retained=False, independent_old_d_box_removed=True,
            numerical_recovery='Public shared-QP proposal retained; no unmeasured photo-window projection',
            unmeasured_photo_mount_domain=receipt)
        return result

    def guard(adapter, q):
        check_producers()
        try:
            return original_guard(q)
        except Exception as error:
            failure_traceback = traceback.format_exc()
            if failure_directory is not None:
                try:
                    path = Path(failure_directory)
                    path.mkdir(parents=True, exist_ok=True)
                    value = np.asarray(q, float)
                    digest = hashlib.sha256(value.tobytes()).hexdigest()
                    stem = digest[:16]
                    archive = path / (stem + '.npz')
                    if archive.exists():
                        with np.load(archive, allow_pickle=False) as saved:
                            if not np.array_equal(saved['q_augmented'], value):
                                raise ValueError('Failure snapshot short-name collision')
                    else:
                        temporary = path / (stem + '.tmp.npz')
                        try:
                            payload = adapter.complete_state_payload(value)
                        except Exception:
                            payload = dict(q_augmented=value.copy(), **{'aux__'+k:v for k,v in adapter.checkpoint_arrays().items()})
                        np.savez(temporary, **payload)
                        temporary.replace(archive)
                    report = getattr(error, 'report', None)
                    record = dict(schema='rinkspace.actual-trial-physical-guard-failure.v1',
                        qaug_sha256=digest, complete_state=reference(archive),
                        exception_type=type(error).__name__, message=str(error), exception_report=report,
                        traceback=failure_traceback, elapsed_seconds=time.perf_counter()-started,
                        original_guard_called_exactly_once=True, original_exception_reraised=True,
                        acceptance_or_source_unchanged=True, first_failure=_first_failure(report))
                    temporary = path / (stem + '.tmp.json')
                    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=_json)+'\n', encoding='utf8')
                    temporary.replace(path / (stem + '.json'))
                    print(json.dumps(dict(stage='actual_trial_guard_failure_saved',
                        qaug_sha256=digest, state=str(archive), **record['first_failure']), ensure_ascii=False, default=_json), flush=True)
                except Exception as diagnostic_error:
                    try:
                        print(json.dumps(dict(stage='guard_failure_diagnostic_write_failed',
                            error=type(diagnostic_error).__name__, message=str(diagnostic_error),
                            original_exception_type=type(error).__name__)), flush=True)
                    except Exception:
                        pass
            raise

    owner.linear_constraints = types.MethodType(linear_constraints, owner)
    owner.recover = types.MethodType(recover, owner)
    owner.checkpoint_identity = types.MethodType(identity, owner)
    owner.guard = types.MethodType(guard, owner)
    owner._unmeasured_mount_domain = receipt
    return receipt
