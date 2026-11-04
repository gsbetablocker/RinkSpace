"""Profile the complete current body/skate objective on existing prepared inputs.

Unlike profile_body_solver.py, this includes JointContactProblem, compatibility,
ice, source-motion, action and support-image factors. It never initializes a
network prior, generates observations, exports video or replaces an old result.
The default performs ONE fresh Jacobian. A production solve is available only
with --operation solve and an explicit existing --run-policy, in a new directory.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from contextlib import ExitStack
from dataclasses import fields
from datetime import datetime, timezone
import functools
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import threading
import time
from types import SimpleNamespace


PUBLIC_BASELINE = '3d6f51fee68a07706c1456c5e99d28c1d0438908'
PROGRAMME = Path('events/ankara_2026/programmes/okada_mei_sp')


def _helpers():
    path = Path(__file__).with_name('profile_body_solver.py')
    spec = importlib.util.spec_from_file_location('_rink_contact_profile_helpers', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


H = _helpers()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind', choices=('skate', 'action', 'support', 'network', 'current_spin'), required=True)
    p.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--base-root', type=Path, required=True, help='Existing S01 shared-model data root')
    p.add_argument('--case-root', type=Path, required=True, help='Prepared clip directory with input/observations; no preparation is run')
    p.add_argument('--state', type=Path, required=True, help='Existing candidate or initial_attempt NPZ with exact controls and fixed identity')
    p.add_argument('--initial', type=Path, help='Exact saved reference NPZ (P,Q,local_R,pts); required except for --kind skate')
    p.add_argument('--source-evidence', type=Path, help='Exact source_evidence.json for --kind skate')
    p.add_argument('--evidence', type=Path, help='Existing action evidence JSON with clips; required for clip kinds')
    p.add_argument('--clip', help='Exact evidence clips key, e.g. spin_s05')
    p.add_argument('--support', type=Path, help='Exact support document for network/support kinds')
    p.add_argument('--support-profile', type=Path, help='Existing support profile.json with fixed_arc_ids; required for support kind')
    p.add_argument('--grasp-relations', type=Path, help='Existing current_spin grasp_relations.json; read without preparation')
    p.add_argument('--grasp-geometry', type=Path, help='Existing current_spin palm_grip_geometry.json; never regenerated')
    p.add_argument('--arm', choices=('A', 'B'), help='Required for action kind; B retains its existing action factors')
    p.add_argument('--optimize-shared', action=argparse.BooleanOptionalAction, default=None)
    p.add_argument('--operation', choices=('evaluate', 'solve'), default='evaluate')
    p.add_argument('--run-policy', type=Path, help='Required for solve; existing RunPolicy JSON or run_policy.json with policy member')
    p.add_argument('--quality-baseline', type=Path, help='For network/current_spin solve, fixed original GVHMR profile.json; support uses --support-profile')
    p.add_argument('--quality-baseline-result', type=Path, help='Optional fixed original GVHMR initial result for current-mask body statistics; default: profile parent/initial/fitted/result.npz')
    p.add_argument('--jacobian-seconds-hint', type=H.nonnegative_float,
                   help='Initial whole-Jacobian reserve estimate; required for network/support/current_spin solve, from an already measured profile')
    p.add_argument('--output', type=Path, required=True, help='New directory only')
    p.add_argument('--backend', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--compatibility-backend', choices=('cpu', 'cuda_resident'), default='cpu',
                   help='Foot/collar compatibility implementation only; cuda_resident keeps the body, camera and solver on their separately recorded paths')
    p.add_argument('--workers', type=H.positive_int, default=1)
    p.add_argument('--blas-threads', type=H.positive_int, default=1)
    p.add_argument('--native-threads', type=H.positive_int, default=1)
    p.add_argument('--native-library', type=Path, help='Explicit implementation override; frozen camera arrays stay identical')
    p.add_argument('--gpu-batch-size', type=H.positive_int, default=512)
    p.add_argument('--perturbation-batch', type=H.positive_int, default=4)
    p.add_argument('--max-batch-samples', type=H.positive_int, default=4096)
    p.add_argument('--local-derivatives', choices=('analytic', 'finite_difference'),
                   help='CPU derivative path. Unspecified: selected checkout default; an older checkout retains finite differences')
    p.add_argument('--repeats', type=H.positive_int, default=1)
    p.add_argument('--include-fun-jac', action='store_true', help='Also measure a fresh fun followed by same-x jac; adds one full Jacobian per repeat')
    p.add_argument('--compare-snapshot', type=Path)
    p.add_argument('--residual-atol', type=H.nonnegative_float, default=1e-8)
    p.add_argument('--residual-rtol', type=H.nonnegative_float, default=1e-10)
    p.add_argument('--jacobian-atol', type=H.nonnegative_float, default=5e-5)
    p.add_argument('--jacobian-rtol', type=H.nonnegative_float, default=2e-4)
    return p


def validate_args(a):
    if a.backend == 'cuda' and a.workers != 1:
        raise ValueError('CUDA uses one owner; --workers must be 1')
    if a.kind == 'skate':
        if a.source_evidence is None:
            raise ValueError('skate requires the existing --source-evidence JSON')
        if a.case_root.resolve() != a.base_root.resolve():
            raise ValueError('skate uses the shared S01 case; --case-root must equal --base-root')
    elif a.initial is None or a.evidence is None or a.clip is None:
        raise ValueError('clip kinds require --initial, --evidence and --clip; the profiler never reconstructs a prior')
    if a.kind in ('network', 'support', 'current_spin') and a.support is None:
        raise ValueError('network/support/current_spin requires the actual --support document')
    if a.kind == 'current_spin':
        if a.grasp_relations is None or a.grasp_geometry is None:
            raise ValueError('current_spin requires --grasp-relations and --grasp-geometry; no geometry or observations are prepared')
        if a.optimize_shared:
            raise ValueError('current_spin requires frozen shared footwear (--no-optimize-shared)')
    if a.kind == 'support' and a.support_profile is None:
        raise ValueError('support requires --support-profile so previously frozen runner arc IDs are not reselected')
    if a.kind == 'action' and a.arm is None:
        raise ValueError('action requires an explicit --arm A or B')
    if a.operation == 'solve' and a.run_policy is None:
        raise ValueError('solve requires an explicit existing --run-policy; no new fit budget is invented')
    if a.operation == 'solve' and a.kind in ('network', 'current_spin') and a.quality_baseline is None:
        raise ValueError('network/current_spin solve requires --quality-baseline with the fixed GVHMR profile metrics')
    if a.operation == 'solve' and a.kind in ('network', 'support', 'current_spin') and a.jacobian_seconds_hint is None:
        raise ValueError('network/support/current_spin solve requires --jacobian-seconds-hint from an existing measured profile; no hidden pre-evaluation')
    if a.operation == 'solve' and (a.compare_snapshot or a.include_fun_jac or a.repeats != 1):
        raise ValueError('solve is one production call; snapshot comparison/repeats/fun-jac belong to evaluate')
    if a.backend == 'cuda' and a.local_derivatives == 'analytic':
        raise ValueError('The current analytic body path is CPU-only; use --local-derivatives finite_difference for this CUDA adapter')
    if a.output.exists():
        raise FileExistsError('--output must be a new directory; previous inputs and candidates are preserved')
    if a.optimize_shared is None:
        a.optimize_shared = a.kind == 'skate'


def clear_result_caches(problem):
    """Clear all evaluated subproblems, not only the outer Joint cache.

    Preserve prepared model tensors, selected-surface tensors and spline bases.
    The old profiler's window-only recursion misses the current .body child.
    """
    seen = set()
    def clear(obj):
        if obj is None or id(obj) in seen or not hasattr(obj, '__dict__'):
            return
        seen.add(id(obj))
        for key in ('cache', 'cache_x', 'cached', 'cached_x', 'binding_key'):
            if key in vars(obj):
                setattr(obj, key, None)
        for key in ('body', 'source_motion', 'action_factors', 'support_image'):
            clear(vars(obj).get(key))
        for child in vars(obj).get('windows', ()):
            clear(child)
    clear(problem)
    # Drop the previous x's GPU pose workspace too. Immutable model, reference,
    # topology and union tensors stay prepared. fun -> same-x jac reuses only
    # the workspace created by that fresh fun within this measured pair.
    resident=getattr(getattr(getattr(problem,'body',None),'execution',None),'_compatibility_backend',None)
    if resident is not None and hasattr(resident,'clear_result_cache'):
        resident.clear_result_cache()


def execution_config(Config, a):
    """Explicit settings with honest backward compatibility to the baseline."""
    values = dict(backend=a.backend, workers=a.workers, blas_threads=a.blas_threads,
        gpu_batch_size=a.gpu_batch_size, perturbation_batch=a.perturbation_batch,
        max_batch_samples=a.max_batch_samples)
    available = {field.name for field in fields(Config)}
    if 'local_derivatives' in available:
        if a.local_derivatives is not None:
            values['local_derivatives'] = a.local_derivatives
        elif a.backend == 'cuda':
            values['local_derivatives'] = 'finite_difference'
    elif a.local_derivatives == 'analytic':
        raise ValueError('The selected checkout has no analytic local derivative path')
    compatibility = getattr(a, 'compatibility_backend', 'cpu')
    if 'compatibility_backend' in available:
        values['compatibility_backend'] = compatibility
    elif compatibility != 'cpu':
        raise ValueError('The selected checkout has no resident CUDA compatibility path')
    config = Config(**values)
    os.environ['RINKSPACE_BODY_LOCAL_DERIVATIVES'] = getattr(config, 'local_derivatives', 'finite_difference')
    os.environ['RINKSPACE_BODY_COMPATIBILITY_BACKEND'] = getattr(config, 'compatibility_backend', 'cpu')
    return config


def cuda_execution_owners(execution):
    """Independent implementation scopes; CPU body plus GPU skin is mixed."""
    owners = {}
    for scope, attribute in (('compact_body', '_body_backend'), ('compatibility', '_compatibility_backend')):
        owner = getattr(execution, attribute, None)
        if owner is not None and getattr(getattr(owner, 'device', None), 'type', None) == 'cuda':
            owners[scope] = owner
    return owners


def synchronize_execution(execution):
    """Include all active CUDA scopes, even with --backend cpu."""
    durations = {}
    for scope, owner in cuda_execution_owners(execution).items():
        start = time.perf_counter()
        owner.torch.cuda.synchronize(owner.device)
        durations[scope] = time.perf_counter()-start
    return durations


def reset_compatibility_measurement(execution):
    owner = getattr(execution, '_compatibility_backend', None)
    if owner is not None and hasattr(owner, 'reset_measurement'):
        owner.reset_measurement()


def compatibility_measurement(execution):
    """Keep the owner's event/transfer counters intact and clearly scoped."""
    owner = getattr(execution, '_compatibility_backend', None)
    if owner is None:
        return None
    result = dict(implementation_scope='foot/collar compatibility only',
                  measurement_supported=hasattr(owner, 'measurement'))
    if hasattr(owner, 'record'):
        result['implementation'] = deepcopy(owner.record())
    if hasattr(owner, 'measurement'):
        result['measurement'] = deepcopy(owner.measurement())
    return result


def gpu_allocator_records(execution, *, reset=False):
    """Torch memory is device-wide: never add the same GPU for two owners."""
    devices = {}
    for scope, owner in cuda_execution_owners(execution).items():
        key = (id(owner.torch.cuda), str(owner.device))
        if key not in devices:
            devices[key] = dict(owner=owner, scopes=[])
        devices[key]['scopes'].append(scope)
    records = []
    for row in devices.values():
        owner = row['owner']
        if reset:
            owner.torch.cuda.reset_peak_memory_stats(owner.device)
        else:
            records.append(dict(device=str(owner.device), implementation_scopes=row['scopes'],
                peak_allocated=int(owner.torch.cuda.max_memory_allocated(owner.device)),
                peak_reserved=int(owner.torch.cuda.max_memory_reserved(owner.device))))
    return dict(devices=records, peak_allocated=sum(r['peak_allocated'] for r in records),
        peak_reserved=sum(r['peak_reserved'] for r in records),
        scope='PyTorch allocator only, per unique CUDA device; includes all listed implementation scopes, not ownership-attributed memory')


def compare_evaluations(old_residual, old_jacobian, residual, jacobian, a):
    """Compare every sparse union entry; a large block cannot hide a lost row."""
    import numpy as np
    old, current = old_jacobian.tocsr(), jacobian.tocsr()
    r0, r = np.asarray(old_residual), np.asarray(residual)
    if r.shape != r0.shape or current.shape != old.shape:
        raise ValueError('Complete residual/Jacobian layouts differ')
    for name, values in (('baseline residual', r0), ('current residual', r),
                         ('baseline Jacobian', old.data), ('current Jacobian', current.data)):
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite '+name+'; refusing numerical equivalence')
    if not old.has_canonical_format:
        old = old.copy(); old.sum_duplicates(); old.sort_indices()
    if not current.has_canonical_format:
        current = current.copy(); current.sum_duplicates(); current.sort_indices()
    if not np.isfinite(old.data).all() or not np.isfinite(current.data).all():
        raise ValueError('Nonfinite canonical Jacobian; refusing numerical equivalence')
    difference = current-old
    difference.eliminate_zeros()
    norm = float(np.linalg.norm(old.data))
    delta = float(np.linalg.norm(difference.data))
    # The sparse difference contains every unequal union entry, including
    # disappeared nonzeros. Equal entries already satisfy the entrywise bound.
    violations, worst, maximum_ratio = 0, None, 0.
    for start in range(0, current.shape[0], 256):
        block = difference[start:start+256].tocoo()
        if not block.nnz:
            continue
        row, col = block.row+start, block.col
        prior = np.asarray(old[row, col]).ravel()
        actual = np.asarray(current[row, col]).ravel()
        absolute = np.abs(block.data)
        limit = a.jacobian_atol+a.jacobian_rtol*np.maximum(np.abs(prior), np.abs(actual))
        bad = absolute > limit
        violations += int(np.count_nonzero(bad))
        ratio = np.divide(absolute, limit, out=np.full_like(absolute, np.inf), where=limit>0)
        k = int(np.argmax(ratio))
        if ratio[k] > maximum_ratio:
            maximum_ratio = float(ratio[k])
            worst = dict(row=int(row[k]), column=int(col[k]), baseline=float(prior[k]),
                current=float(actual[k]), absolute_error=float(absolute[k]), allowed_error=float(limit[k]),
                normalized_error=float(ratio[k]) if np.isfinite(ratio[k]) else None)
    residual_error = np.abs(r-r0)
    residual_limit = a.residual_atol+a.residual_rtol*np.maximum(np.abs(r0), np.abs(r))
    residual_violations = int(np.count_nonzero(residual_error > residual_limit))
    return dict(residual_max_abs=float(np.max(residual_error, initial=0.)),
        residual_relative_l2=float(np.linalg.norm(r-r0)/max(np.linalg.norm(r0), 1e-15)),
        residual_violation_count=residual_violations,
        jacobian_max_abs=float(np.max(np.abs(difference.data), initial=0.)),
        jacobian_relative_frobenius=delta/max(norm, 1e-15),
        jacobian_entry_violation_count=violations, jacobian_worst_entry=worst,
        jacobian_max_normalized_error=maximum_ratio if np.isfinite(maximum_ratio) else None,
        residual_within_tolerance=residual_violations==0, jacobian_within_tolerance=violations==0,
        thresholds=dict(residual_atol=a.residual_atol, residual_rtol=a.residual_rtol,
            jacobian_entry_atol=a.jacobian_atol, jacobian_entry_rtol=a.jacobian_rtol,
            equation='abs(new-old) <= atol + rtol*max(abs(old),abs(new)); every sparse union entry',
            frobenius_is_summary_only=True))


def snapshot_jacobian(snapshot):
    from scipy.sparse import csr_matrix
    return csr_matrix((snapshot['jac_data'], snapshot['jac_indices'], snapshot['jac_indptr']),
                      shape=tuple(snapshot['jac_shape']))


def compare_cached_evaluation(snapshot, value, a):
    result = compare_evaluations(snapshot['residual'], snapshot_jacobian(snapshot),
                                 value['residual'], value['jac'], a)
    if not result['residual_within_tolerance'] or not result['jacobian_within_tolerance']:
        raise ValueError('Fresh and fun->jac complete evaluations differ: '+json.dumps(result))
    return result


def compare_snapshot(current, baseline_path, a):
    import numpy as np
    with np.load(H._existing(baseline_path), allow_pickle=False) as data:
        old = {key: data[key].copy() for key in data.files}
    before = json.loads(str(old['metadata_json']))
    after = json.loads(str(current['metadata_json']))
    if before['input_fingerprint'] != after['input_fingerprint'] or not np.array_equal(old['x'], current['x']):
        raise ValueError('Cannot compare snapshots: source inputs, objective layout or x differ')
    result = compare_evaluations(old['residual'], snapshot_jacobian(old),
                                current['residual'], snapshot_jacobian(current), a)
    return dict(baseline_snapshot=str(Path(baseline_path).resolve()),
                automatic_backend_selection=False, **result)


class PhaseTrace:
    """Scoped call timing; nested/worker times are explicitly not additive wall.

    Methods are patched on classes. An instance-bound wrapper would be copied
    by shared_column's copy.copy(binding) and could call the WRONG binding.
    All original attributes are restored, including after a failed evaluation.
    """
    def __init__(self):
        self._changes = []
        self._patched = set()
        self._local = threading.local()
        self._lock = threading.Lock()
        self.main_thread = threading.get_ident()
        self.records = {}
        self.lsmr_calls = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for owner, name, had_own, original in reversed(self._changes):
            if had_own:
                setattr(owner, name, original)
            else:
                delattr(owner, name)

    def reset(self):
        with self._lock:
            self.records = {}
            self.lsmr_calls = []

    def patch(self, owner, name, label, *, lsmr=False):
        key = (id(owner), name)
        if key in self._patched or not hasattr(owner, name):
            return
        original = getattr(owner, name)
        if not callable(original):
            return
        had_own = name in vars(owner)
        @functools.wraps(original)
        def measured(*args, **kwargs):
            stack = getattr(self._local, 'stack', None)
            if stack is None:
                stack = self._local.stack = []
            frame = [time.perf_counter(), 0.]
            stack.append(frame)
            failed = False
            result = None
            try:
                result = original(*args, **kwargs)
                return result
            except BaseException:
                failed = True
                raise
            finally:
                elapsed = time.perf_counter() - frame[0]
                stack.pop()
                if stack:
                    stack[-1][1] += elapsed
                main = threading.get_ident() == self.main_thread
                with self._lock:
                    row = self.records.setdefault(label, dict(calls=0, failures=0,
                        inclusive_call_seconds=0., exclusive_call_seconds=0.,
                        main_thread_inclusive_seconds=0., worker_call_seconds_sum=0.))
                    row['calls'] += 1
                    row['failures'] += int(failed)
                    row['inclusive_call_seconds'] += elapsed
                    row['exclusive_call_seconds'] += max(0., elapsed-frame[1])
                    row['main_thread_inclusive_seconds' if main else 'worker_call_seconds_sum'] += elapsed
                    if lsmr and result is not None:
                        self.lsmr_calls.append(dict(wall_seconds=elapsed, istop=int(result[1]),
                            iterations=int(result[2]), normr=float(result[3]), normar=float(result[4]),
                            norma=float(result[5]), conda=float(result[6]), normx=float(result[7])))
        self._changes.append((owner, name, had_own, original))
        self._patched.add(key)
        setattr(owner, name, measured)

    def install(self, problem, *, solve=False):
        from calibration.program_tracking.body_tracking.fullbody_fit import FullBodyProblem
        module = importlib.import_module('calibration.program_tracking.body_tracking.skate_integration')
        body, binding = problem.body, problem.binding
        for owner, name, label in (
            (type(problem), 'evaluate', 'joint_complete'),
            (type(body), 'evaluate', 'body_with_compatibility'),
            (FullBodyProblem, 'evaluate', 'fullbody_base'),
            (type(body), 'compatibility', 'compatibility_forward'),
            (type(binding), 'compatibility_rows', 'compatibility_rows'),
            (type(binding), 'foot_metrics', 'foot_metrics'),
            (type(binding), 'collar_metrics', 'collar_metrics'),
            (module, 'fk_state', 'physics_fk'),
            (module, 'constraints', 'physics_constraints'),
            (module, 'body_constraint_jacobian', 'physics_jacobian'),
            (type(problem), 'shared_column', 'shared_shoe_column'),
            (type(problem.camera), 'project', 'native_projection'),
            (type(problem.camera), 'with_jacobian', 'native_xyz_jacobian'),
        ):
            self.patch(owner, name, label)
        for name in ('source_motion', 'action_factors', 'support_image'):
            child = getattr(problem, name, None)
            if child is not None:
                self.patch(type(child), 'evaluate', name)
        action = getattr(problem, 'action_factors', None)
        if action is not None and type(action).__name__ == 'SpinGraspFactors':
            spin = importlib.import_module(type(action).__module__)
            for owner, name, label in (
                (type(action), 'geometry_at', 'spin_grasp_geometry'),
                (type(action), 'continuous_support', 'spin_continuous_support'),
                (type(action), 'image', 'spin_source_image'),
                (spin, 'palm_local', 'spin_palm_skin'),
                (spin, 'fk_state', 'spin_fk'),
                (spin, 'body_constraint_jacobian', 'spin_support_jacobian')):
                self.patch(owner, name, label)
        # Do not wrap reference.world: optimized paths may certify its exact
        # implementation identity. The compact adapter call proves real use.
        compact = problem.reference.body
        for name in ('forward_rotations', 'forward_native_frames'):
            self.patch(type(compact), name, 'compact_' + name)
        kernel_path = Path(module.__file__).with_name('compact_body_derivatives.py')
        if kernel_path.is_file():
            kernel = importlib.import_module('calibration.program_tracking.body_tracking.compact_body_derivatives')
            cls = kernel.CompactBodyLinearization
            for name, label in (('__init__', 'compact_linearization_prepare'),
                                ('derivative', 'compact_analytic_derivative'),
                                ('kinematic_pair', 'compact_fk_pair')):
                self.patch(cls, name, label)
        surface_path = Path(module.__file__).with_name('selected_surface_linearization.py')
        if surface_path.is_file():
            surface = importlib.import_module('calibration.program_tracking.body_tracking.selected_surface_linearization')
            self.patch(surface.SelectedSurfaceLinearization, '__init__', 'selected_surface_prepare')
            self.patch(surface.SelectedSurfaceLinearization, 'pair', 'selected_surface_pair')
        else:
            # The optimized surface guard certifies the original method.
            # Do not change that identity merely to obtain a call counter.
            self.patch(type(binding.surface), 'sample_vertices', 'surface_sample')
        if solve:
            if getattr(problem, 'stage_quality', None) is not None:
                for name in ('begin', 'finish'):
                    self.patch(type(problem.stage_quality), name, 'stage_quality_' + name)
            trf = importlib.import_module('scipy.optimize._lsq.trf')
            if not hasattr(trf, 'lsmr'):
                raise RuntimeError('This SciPy TRF module exposes no lsmr instrumentation point')
            self.patch(trf, 'lsmr', 'lsmr', lsmr=True)
            self.patch(module, 'least_squares', 'scipy_trf_complete')
            # The profiler owns this isolated process. Restore both module
            # references on exit; the array/checkpoint contents are untouched.
            self.patch(module.np, 'savez_compressed', 'checkpoint_npz_write')
            self.patch(module, 'write_json', 'solve_json_write')

    def snapshot(self):
        with self._lock:
            return json.loads(json.dumps(dict(phases=self.records, lsmr_calls=self.lsmr_calls)))


def validate_fixed_identity(saved, expected, *, native_override=False):
    """Keep model identity strict while allowing an explicit binary comparison."""
    import numpy as np
    from calibration.program_tracking.body_tracking.skate_checkpoint import validate_checkpoint_identity
    validate_checkpoint_identity(saved)
    implementation_change = None
    adjusted = dict(expected)
    if native_override:
        before = json.loads(str(saved['camera_binding_json']))
        actual = json.loads(str(expected['camera_binding_json']))
        old_hash, new_hash = before.get('native_library_sha256'), actual.get('native_library_sha256')
        if old_hash != new_hash:
            implementation_change = dict(saved_native_sha256=old_hash, evaluation_native_sha256=new_hash)
            actual['native_library_sha256'] = old_hash
            adjusted['camera_binding_json'] = np.array(json.dumps(actual, sort_keys=True))
    # Every geometry value, camera-state SHA, unit and source identity still
    # passes the production validator. The changed binary is recorded separately.
    validate_checkpoint_identity(saved, adjusted)
    return implementation_change


def validate_current_spin_inputs(saved, initial, p, files, info, *, native_override=False):
    """Validate the already prepared v2 case without replaying its producers."""
    import numpy as np
    from calibration.program_tracking.body_tracking.exmr_integration import read_case_identity
    from calibration.program_tracking.body_tracking.skate_checkpoint import CAMERA_LOCATION_FIELDS
    case = read_case_identity(saved)
    if case['label'] != 'exmr_hand_nofoot' or case['schema'] != 'rinkspace.network-comparison-case.v2':
        raise ValueError('current_spin requires the complete ExMR body/hand, FootMR-off v2 identity')
    for key, filename in (
        ('case_input_sha256', 'case_input'), ('initial_reference_sha256', 'initial'),
        ('observations_sha256', 'observation'), ('compact_smplx_sha256', 'case_compact'),
        ('export_source_sha256', 'case_export_source'), ('shared_s01_parameters_sha256', 'shared_s01'),
        ('support_document_sha256', 'support'), ('action_evidence_sha256', 'evidence'),
        ('relation_document_sha256', 'grasp_relations'), ('geometry_semantics_sha256', 'grasp_geometry')):
        if case[key] != H.sha256_file(files[filename]):
            raise ValueError('Current spin input identity mismatch: '+key)
    pts = np.asarray(p['pts'], dtype='<i8')
    if (case['source_frames'] != len(pts) or case['source_time_base'] != [1, 12800] or
            case['source_pts_sha256'] != hashlib.sha256(pts.tobytes()).hexdigest()):
        raise ValueError('Current spin source clock/PTS identity mismatch')
    camera = {k: v for k, v in info.items() if k not in CAMERA_LOCATION_FIELDS}
    if native_override:
        camera['native_library_sha256'] = case['camera']['native_library_sha256']
    if camera != case['camera']:
        raise ValueError('Current spin fixed camera identity mismatch')
    if str(saved.get('state_kind', '')) != 'spin_support_v2' or len(saved['active']) != 63:
        raise ValueError('current_spin requires the saved spin_support_v2 state with all 63 body columns')
    required = ('root_curve_schema', 'root_curve_knots', 'root_curve_controls', 'root_curve_degree', 'hand_local_R')
    if any(key not in initial for key in required):
        raise ValueError('Current spin initial reference must retain exact root curve and both hand rotations')
    for key, field in (('grasp_relations_sha256', 'relation_document_sha256'),
                       ('grasp_geometry_sha256', 'geometry_semantics_sha256')):
        if str(saved.get(key, '')) != case[field]:
            raise ValueError('Current spin checkpoint identity mismatch: '+key)
    return case


def current_spin_runtime_document(document, relations):
    """Reproduce make_problem's runtime weights without changing source JSON."""
    runtime = deepcopy(document)
    multiplier = relations['factor_parameters']['shoe_region_weight_multiplier']
    for row in runtime['observations']:
        row['objective_weight'] *= multiplier
    return runtime


def construct_problem(a, execution, log):
    import numpy as np
    from calibration.program_tracking.body_tracking.rebuild_prepare import load
    from calibration.program_tracking.body_tracking.fullbody_state import camera_binding
    from calibration.program_tracking.body_tracking.skate_checkpoint import checkpoint_identity
    from calibration.program_tracking.body_tracking.skate_integration import (
        build, JointContactProblem, restore_checkpoint_layers, checkpoint_state_kind, source_evidence)
    from calibration.program_tracking.body_tracking.ice_soft import IceSoftPolicy

    files = {'state': H._existing(a.state)}
    relations = geometry = None
    saved = load(files['state'])
    required = ('x', 'knots', 'active', 'seconds', 'physics_times', 'multiplier', 'penalty', 'ice_soft_policy_json')
    if any(key not in saved for key in required):
        raise ValueError('The existing state must contain complete current solver/checkpoint fields')
    x = np.asarray(saved['x'], dtype=np.float64).copy()
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError('Expected finite one-dimensional x')
    shared, camera, info, catalog = build(a.project, a.base_root, a.output/'construction')
    files['shared_compact'] = H._existing(a.base_root/'compact_smplx.npz')
    files['shared_input'] = H._existing(a.base_root/'input.npz')
    files['shared_s01'] = H._existing(a.project/PROGRAMME/'skate_s01_1103.npz')
    # --native-library and launcher env are explicit implementation overrides.
    override = bool(a.native_library or os.environ.get('RINKSPACE_BODY_NATIVE_LIBRARY'))
    native_changes = []
    if a.kind == 'skate':
        ref = shared
        files['source_evidence'] = H._existing(a.source_evidence)
        supports = json.loads(files['source_evidence'].read_text(encoding='utf8'))
        observation = a.base_root/'boot_integration_1102/observations.npz'
        evidence = document = None
    else:
        from calibration.program_tracking.body_tracking.action_integration import ClipReference
        files.update(case_input=H._existing(a.case_root/'input.npz'), initial=H._existing(a.initial),
                     evidence=H._existing(a.evidence), case_compact=H._existing(a.case_root/'compact_smplx.npz'))
        p, initial = load(files['case_input']), load(files['initial'])
        np.testing.assert_array_equal(p['pts'], initial['pts'])
        for name in ('shared_betas', 'body_scale'):
            np.testing.assert_array_equal(p[name], shared.p[name])
        if H.sha256_file(files['case_compact']) != H.sha256_file(files['shared_compact']):
            raise ValueError('Prepared clip and shared compact model differ')
        all_evidence = json.loads(files['evidence'].read_text(encoding='utf8'))
        evidence = all_evidence['clips'][a.clip]
        camera, info, _, pose = camera_binding(a.project, SimpleNamespace(p=p), shot_id=evidence['shot_id'])
        p = camera.p
        p.update(camera_rotation=pose['rotation'], camera_centre=pose['centre'], camera_values=pose['values'])
        seed = load(files['shared_s01'])
        if a.kind in ('network', 'current_spin'):
            if a.kind == 'network' and int(saved.get('base_layer_count', 0)) != 0:
                raise ValueError('Network checkpoint has base layers; its original frozen support arc mapping must be supplied by a dedicated adapter')
            np.testing.assert_array_equal(x[:17], seed['x'][:17],
                err_msg='Current network comparison keeps the accepted S01 shared geometry frozen')
        native_changes.append(validate_fixed_identity(seed, shared.fixed_checkpoint_identity, native_override=override))
        shared.binding.x = seed['x'][:17].copy()
        shared.binding.enable_collar_constraints()
        ref = ClipReference(shared, a.case_root, p, initial)
        ref.fixed_checkpoint_identity = checkpoint_identity(ref.binding, a.project, info)
        ref.hashes = {'profile_initial_sha256': H.sha256_file(files['initial']),
                      'shared_s01_parameters_sha256': H.sha256_file(files['shared_s01'])}
        ref.footwear_observations = dict(visible_rows=[], blade_rows=[], landmark_rows=[],
            provenance={'status': 'same prepared clip; explicit support/action factors selected below'})
        if a.kind == 'action' and evidence['shot_id'] == 's01':
            from calibration.program_tracking.body_tracking.fullbody_fit import footwear_rows
            gear, provenance = footwear_rows(a.base_root, np.arange(len(p['pts'])), p['pts'])
            heldout = set(map(int, evidence.get('heldout_pts', [])))
            for name in ('visible_rows', 'blade_rows', 'landmark_rows'):
                ref.footwear_observations[name] = [r for r in getattr(gear, name) if int(r['pts']) not in heldout]
            ref.footwear_observations['provenance'] = provenance
        records = list(evidence.get('supports', []))
        document = None
        if a.kind in ('support', 'network', 'current_spin'):
            files['support'] = H._existing(a.support)
            document = json.loads(files['support'].read_text(encoding='utf8'))
            records += document['supports']
            if a.kind == 'current_spin':
                records = list(document['supports'])
            if a.kind == 'network':
                review_pts = {r['pts'] for r in document['observations'] if r['role'] == 'review'}
                records = [r for r in records if r['pts'] not in review_pts]
        supports = source_evidence(a.case_root, shot_id=evidence['shot_id'], source_pts=p['pts'], records=records)
        observation = a.case_root/'observations.npz'
        if a.kind == 'current_spin':
            from calibration.program_tracking.body_tracking.exmr_integration import CASE_FIELD
            files.update(observation=H._existing(observation),
                case_export_source=H._existing(a.case_root/'export_source.npz'),
                grasp_relations=H._existing(a.grasp_relations), grasp_geometry=H._existing(a.grasp_geometry))
            if a.clip != 'spin_s05' or evidence['shot_id'] != 's05':
                raise ValueError('current_spin uses the existing spin_s05/s05 case')
            case = validate_current_spin_inputs(saved, initial, p, files, info, native_override=override)
            relations = json.loads(files['grasp_relations'].read_text(encoding='utf8'))
            geometry = json.loads(files['grasp_geometry'].read_text(encoding='utf8'))
            ref.grasp_geometry, ref.grasp_relations = geometry, relations
            ref.hashes = {k: case[k] for k in ('native_input_sha256', 'camera_prior_sha256', 'shared_s01_parameters_sha256')}
            ref.fixed_checkpoint_identity[CASE_FIELD] = saved[CASE_FIELD].copy()
    native_changes.append(validate_fixed_identity(saved, ref.fixed_checkpoint_identity, native_override=override))
    ref.binding.x = x[:17].copy()
    if bool(saved.get('collar_constraints_enabled', False)):
        ref.binding.enable_collar_constraints()
    restore_checkpoint_layers(ref, saved, include_primary=False)
    execution.attach(ref)
    files['observation'] = H._existing(observation)
    ice_policy = IceSoftPolicy(**json.loads(str(saved['ice_soft_policy_json'])))
    physical_evidence = [] if a.kind == 'current_spin' else supports
    production_options = dict(record_acceptance_before_jacobian=True) if a.kind == 'current_spin' else {}
    problem = JointContactProblem(ref, camera, files['observation'], tuple(saved['seconds']), physical_evidence,
        a.output/'solve', log, checkpoint_state_kind(saved), optimize_shared=a.optimize_shared, ice_policy=ice_policy,
        **production_options)
    for name in ('knots', 'active'):
        np.testing.assert_array_equal(saved[name], getattr(problem.body.state, name))
    if a.kind == 'current_spin' and not np.array_equal(saved['physics_times'], problem.physics_times):
        problem.set_physics_times(saved['physics_times'])
    np.testing.assert_array_equal(saved['physics_times'], problem.physics_times)
    if x.shape != (problem.size,) or saved['multiplier'].shape != problem.al.multiplier.shape:
        raise ValueError('Checkpoint variables or contact rows differ from the reconstructed objective')
    problem.al.multiplier[:] = saved['multiplier']
    problem.al.penalty = float(saved['penalty'])
    if (not np.isfinite(problem.al.multiplier).all() or np.any(problem.al.multiplier < 0) or
            not np.isfinite(problem.al.penalty) or problem.al.penalty <= 0):
        raise ValueError('Invalid saved augmented-Lagrangian state')
    if a.kind == 'action':
        from calibration.program_tracking.body_tracking.action_live import GlideActionFactors
        problem.action_factors = GlideActionFactors(problem, evidence,
            sigma_mps=evidence['glide_sigma_mps'], enabled=a.arm == 'B')
    runtime_document = current_spin_runtime_document(document, relations) if a.kind == 'current_spin' else document
    if document is not None:
        from calibration.program_tracking.body_tracking.support_image import SupportImageFactors
        ref.footwear_observations['support_regions'] = document['observations']
        arc_ids = None
        if a.kind == 'support' or (a.kind == 'current_spin' and a.support_profile is not None):
            files['support_profile'] = H._existing(a.support_profile)
            arc_ids = json.loads(files['support_profile'].read_text(encoding='utf8'))['fixed_arc_ids']
            validate_frozen_arc_ids(document, arc_ids, problem.binding.packet()['template']['contact_arcs'])
        if a.kind == 'current_spin':
            # Empty is intentional for this case: its old contact pixels are
            # disabled; blade midpoint/endpoint rows need no runner selection.
            arc_ids = {} if arc_ids is None else arc_ids
            validate_frozen_arc_ids(runtime_document, arc_ids, problem.binding.packet()['template']['contact_arcs'])
        problem.support_image = SupportImageFactors(problem, runtime_document, arc_ids=arc_ids)
    if a.kind == 'current_spin':
        from calibration.program_tracking.body_tracking.spin_grasp_factors import SpinGraspFactors
        problem.action_factors = SpinGraspFactors(problem, relations)
        # Preserve immutable source/provenance fields in newly written states.
        problem.checkpoint_extra = {k: saved[k].copy() for k in (
            'hand_pose_schema_version', 'grasp_relations_sha256', 'grasp_geometry_sha256',
            'source_support_track_sha256', 'initialization_provenance_json') if k in saved}
    if a.kind == 'network':
        from calibration.program_tracking.body_tracking.exmr_integration import read_case_identity, CASE_FIELD
        case = read_case_identity(saved)
        if a.clip != 'spin_s05':
            raise ValueError('The current network comparison case uses the explicit spin_s05 evidence')
        for key, filename in (('case_input_sha256', 'case_input'), ('initial_reference_sha256', 'initial'),
                              ('observations_sha256', 'observation'), ('compact_smplx_sha256', 'case_compact'),
                              ('shared_s01_parameters_sha256', 'shared_s01'), ('support_document_sha256', 'support'),
                              ('action_evidence_sha256', 'evidence')):
            if case[key] != H.sha256_file(files[filename]):
                raise ValueError('Comparison-case input identity mismatch: '+key)
        ref.fixed_checkpoint_identity[CASE_FIELD] = saved[CASE_FIELD].copy()
    if a.operation == 'solve' and document is not None:
        from calibration.program_tracking.body_tracking.stage_quality import StageQualityTrace
        from calibration.program_tracking.body_tracking.support_refinement import measure
        baseline_path = a.quality_baseline if a.kind in ('network', 'current_spin') else files['support_profile']
        files['stage_quality_baseline'] = H._existing(baseline_path)
        baseline = read_quality_baseline(a.kind, files['stage_quality_baseline'])
        review = SupportImageFactors(problem, runtime_document, include_review=True,
                                     arc_ids=problem.support_image.arc_ids)
        identity = dict(kind=a.kind, state_sha256=H.sha256_file(files['state']),
            fixed_baseline_sha256=H.sha256_file(files['stage_quality_baseline']),
            source_region_sha256=H.sha256_file(files['support']),
            fit_pts=sorted({r['pts'] for r in document['observations'] if r['role']=='fit'}),
            review_pts=sorted({r['pts'] for r in document['observations'] if r['role']=='review'}))
        problem.stage_quality = StageQualityTrace(a.output/'cross_stage_quality.json',
            identity, baseline, lambda xx, value: measure(problem, review, xx, value))
        if a.kind == 'current_spin':
            result_path = a.quality_baseline_result or files['stage_quality_baseline'].parent/'initial/fitted/result.npz'
            files['stage_quality_baseline_result'] = H._existing(result_path)
            attach_current_mask_baseline(problem, files['stage_quality_baseline_result'], files['stage_quality_baseline'])
    for key in ('model_path', 'config_path'):
        value = getattr(catalog, key, None)
        if value is not None:
            files['catalog_' + key] = H._existing(value)
    for key in ('record', 'state_path'):
        files['camera_' + key] = H._existing(info[key])
    for key in ('review_path', 'admission_path'):
        if key in problem.body.gear_sources:
            files['footwear_' + key] = H._existing(problem.body.gear_sources[key])
    native_info = H.configure_native(camera, info, a)
    if problem.body.execution is not execution:
        raise ValueError('The selected body execution owner is not used by the actual problem')
    if a.backend == 'cuda' and (execution._body_backend is None or execution._body_backend.device.type != 'cuda'):
        raise ValueError('CUDA requested but the actual body has no CUDA adapter')
    compatibility_owner = getattr(execution, '_compatibility_backend', None)
    if (getattr(a, 'compatibility_backend', 'cpu') == 'cuda_resident' and
            (compatibility_owner is None or getattr(compatibility_owner.device, 'type', None) != 'cuda')):
        raise ValueError('Resident CUDA compatibility requested but the actual problem has no CUDA compatibility owner')
    return SimpleNamespace(problem=problem, x=x, reference=ref, camera=camera, camera_info=info,
        files=files, native_info=native_info, native_identity_changes=[v for v in native_changes if v],
        objective=dict(kind=a.kind, state_kind=checkpoint_state_kind(saved), optimize_shared=a.optimize_shared,
            action_arm=a.arm if a.kind == 'action' else None, evidence=evidence, support=document,
            grasp_relations=relations, grasp_geometry=geometry,
            support_runtime=runtime_document if a.kind == 'current_spin' else None,
            support_arc_ids=getattr(getattr(problem, 'support_image', None), 'arc_ids', None),
            supports=supports, physical_evidence=physical_evidence, ice_policy=vars(ice_policy),
            record_acceptance_before_jacobian=problem.record_acceptance_before_jacobian))


def read_quality_baseline(kind, path):
    """Use the caller's frozen cross-stage baseline, never the current trial."""
    record = json.loads(path.read_text(encoding='utf8'))
    if kind in ('network', 'current_spin'):
        if record.get('label') != 'gvhmr':
            raise ValueError('Network quality baseline must be the fixed gvhmr profile')
        baseline = record['metrics']
    elif kind == 'support':
        baseline = record['candidates']['original_A']['measurement']
    else:
        raise ValueError('No stage-quality baseline for this problem kind')
    if not isinstance(baseline, dict) or 'groups' not in baseline:
        raise ValueError('Stage-quality baseline has no complete group measurements')
    return baseline


def attach_current_mask_baseline(problem, result_path, profile_path):
    """Read original G's saved projection in begin(), inside timed solve wall.

    Current admission may differ from the earlier GVHMR profile. Only body
    groups can be recomputed from its saved native projection; historical shoe
    region metrics remain explicitly identified with their original source set.
    """
    trace = problem.stage_quality
    trace.identity['fixed_original_gvhmr_result_sha256'] = H.sha256_file(result_path)
    trace.identity['body_quality_scope'] = '88-92 source seconds; current observation/use mask for original G and current candidate'
    measure = trace.measure
    prepared = False
    def with_original_projection(x, value):
        nonlocal prepared
        if not prepared:
            import numpy as np
            from calibration.program_tracking.body_tracking.rebuild_prepare import load
            from calibration.program_tracking.body_tracking.stage_quality import body_groups
            from calibration.program_tracking.body_tracking.exmr_integration import read_case_identity
            original = load(result_path)
            identity = read_case_identity(original)
            profile_identity = json.loads(json.loads(profile_path.read_text(encoding='utf8'))['comparison_case_identity_json'])
            if (identity['label'] != 'gvhmr' or identity != profile_identity or
                    str(original.get('comparison_stage', '')) != 'common_initial'):
                raise ValueError('Current-mask quality requires the fixed original GVHMR initial result')
            obs, frames = problem.body.obs, problem.body.frames
            clock = obs['pts'][frames]
            lookup = {int(pts): i for i, pts in enumerate(original['pts'])}
            if len(lookup) != len(original['pts']) or any(int(pts) not in lookup for pts in clock):
                raise ValueError('Original GVHMR result does not cover the current native PTS')
            selected = np.array([lookup[int(pts)] for pts in clock], int)
            core = (clock >= 88*12800) & (clock <= 92*12800)
            groups = body_groups(original['COCO_uv'][selected][core], obs['uv'][frames][core],
                                 obs['use'][frames][core], clock[core])
            trace.original_A['original_profile_metrics'] = deepcopy(trace.original_A)
            trace.original_A['groups'].update(groups)
            trace.original_A['body_groups_recomputed_on_current_use_mask'] = True
            trace.original_A['shoe_region_scope'] = 'Original profile source regions; current additional/changed source regions are reported separately and are not an equal-source comparison'
            prepared = True
        return measure(x, value)
    trace.measure = with_original_projection


def validate_frozen_arc_ids(document, arc_ids, arcs):
    """Prevent SupportImageFactors silently choosing missing fit-row arcs."""
    if not isinstance(arc_ids, dict):
        raise ValueError('Support profile has no exact frozen arc-ID mapping')
    required = {row['id'] for row in document['observations']
        if row['kind']=='support_contact_region' and row['role']=='fit' and row['foot'] in (0, 1)
        and row['visibility']=='visible' and row['reliability']>0 and row['contact_state']=='supported'
        and row.get('optimization_enabled', True)}
    missing = sorted(required-set(arc_ids))
    if missing:
        raise ValueError('Frozen support arc mapping omits admitted fit rows: '+str(missing))
    runners = {arc['id'] for arc in arcs if arc['semantic']=='runner'}
    invalid = sorted(key for key, value in arc_ids.items() if value not in runners)
    if invalid:
        raise ValueError('Frozen support arc mapping contains non-runner arc IDs: '+str(invalid))


def numerical_identity(setup, input_files):
    """Hash the actual consumed model/reference arrays, not execution settings."""
    import numpy as np
    problem = setup.problem
    reference = setup.reference
    compact = getattr(reference.body, 'source', reference.body)
    arrays = {}
    def add(prefix, obj, names):
        for name in names:
            value = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
            if value is not None:
                arrays[prefix+'/'+name] = H.array_digest(np.asarray(value))
    add('camera', setup.camera.p, ('rig', 'controls', 'motion_times', 'first_control_time', 'control_dt', 'metres_per_scene_unit'))
    add('compact', compact, ('J', 'shaped', 'posedirs', 'W', 'reg', 'default', 'parents', 'scale'))
    add('surface', reference.binding.surface, ('shaped', 'posedirs', 'weights', 'faces'))
    add('state', problem.body.state, ('knots', 'active', 'seconds'))
    add('reference_exposure', problem.body.ref, ('P', 'Q', 'local_R', 'hand_local_R'))
    add('reference_physics', problem.physics_ref, ('P', 'Q', 'local_R', 'hand_local_R'))
    add('reference_root_curve', getattr(reference, 'p_curve', {}), ('t', 'c', 'k'))
    add('al', problem.al, ('multiplier', 'penalty'))
    arrays['physics_times'] = H.array_digest(problem.physics_times)
    return dict(input_files={key: value['sha256'] for key, value in input_files.items()},
                arrays=arrays, objective=setup.objective, x_sha256=H.array_digest(setup.x),
                interpretation='saved primary controls applied once; saved base layers are immutable reference')


def implementation_records(project):
    result = {}
    for name, module in tuple(sys.modules.items()):
        if name == 'calibration' or name.startswith('calibration.') or name == 'rinkspace' or name.startswith('rinkspace.'):
            path = getattr(module, '__file__', None)
            if path and Path(path).suffix == '.py':
                path = Path(path).resolve()
                if not path.is_relative_to(project):
                    raise ValueError('Mixed checkout module: '+str(path))
                result[path.relative_to(project).as_posix()] = H.sha256_file(path)
    for path in (Path(__file__).resolve(), Path(H.__file__).resolve()):
        result['profiler/'+path.name] = H.sha256_file(path)
    return result


def select_project_sources(project):
    """All numerical imports, including parent packages, belong to --project."""
    project = Path(project).resolve()
    selected = [str(project/'src'), str(project)]
    sys.path[:] = selected+[entry for entry in sys.path if entry not in selected]
    importlib.invalidate_caches()
    # Fail before loading the DLL/model if an embedding interpreter already
    # imported another checkout. Never replace an imported module in place.
    implementation_records(project)


def run_profile(a):
    import numpy as np
    import scipy
    from threadpoolctl import threadpool_info
    BodyExecution, Config, _ = H.execution_classes(a.project, a.backend)
    execution = BodyExecution(execution_config(Config, a))
    setup_started = time.perf_counter()
    with ExitStack() as stack:
        stack.callback(execution.close)
        log = stack.enter_context((a.output/'progress.jsonl').open('w', encoding='utf8', buffering=1))
        # All evaluations and solve callbacks use the same BLAS scope. This is
        # also recorded inside the scope, not after restoring the prior limits.
        stack.enter_context(execution.limits())
        setup = construct_problem(a, execution, log)
        setup_seconds = time.perf_counter()-setup_started
        problem, x = setup.problem, setup.x
        synchronize_execution(execution)
        gpu_allocator_records(execution, reset=True)
        input_files = {key: dict(path=str(path), sha256=H.sha256_file(path)) for key, path in sorted(setup.files.items())}
        identity = numerical_identity(setup, input_files)
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, default=H.json_value).encode()).hexdigest()
        trace = stack.enter_context(PhaseTrace())
        trace.install(problem, solve=a.operation == 'solve')
        report = dict(schema='rinkspace.contact-solver-profile.v1', public_baseline=PUBLIC_BASELINE,
            utc=datetime.now(timezone.utc).isoformat(), operation=a.operation, kind=a.kind,
            input_fingerprint=fingerprint, input_files=input_files, objective=setup.objective,
            execution=execution.record(), actual_body_type=type(setup.reference.body).__name__,
            native_implementation=setup.native_info, native_identity_changes=setup.native_identity_changes,
            implementation_sha256=implementation_records(a.project), setup_seconds=setup_seconds,
            runtime=dict(python=sys.version, executable=sys.executable, numpy=np.__version__, scipy=scipy.__version__,
                platform=platform.platform(), processor=platform.processor(), logical_cpu_count=os.cpu_count(),
                thread_pools_during_measurement=threadpool_info(),
                selected_environment={k:v for k,v in os.environ.items() if k.startswith('RINKSPACE_BODY_') or
                    k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS')}),
            no_input_or_existing_candidate_write=True,
            construction_side_effect='The existing ResultCatalog may create an empty workbench/results3d directory.',
            phase_timing_note='Nested inclusive times overlap. Worker call-seconds can exceed elapsed wall. Do not add these columns or call requested worker count measured utilization.',
            surface_timing_note='surface_sample is instrumented only in checkouts without selected_surface_linearization; optimized checkouts use prepare/pair and compatibility timings so tracing preserves the strict sampling-method guard.',
            gpu_timing_note='Whole-call wall includes explicit synchronization of compact-body and resident-compatibility CUDA owners. PhaseTrace subphases are host wall; compatibility_gpu_measurement separately preserves device CUDA-event computation/transfer time and bytes. Static setup is reported by the owner separately from reset measurement counters.',
            initial_x_sha256=H.array_digest(x), source_frames=problem.body.n,
            active_variables=problem.body.size+(17 if a.optimize_shared else 0), full_layout_variables=problem.size,
            physics_times=len(problem.physics_times), initial_objective=None,
            measured_scope='complete objective evaluation; no solve' if a.operation == 'evaluate' else 'production JointContactProblem.solve including initial evaluation, TRF/LSMR, acceptance, policy, cross-stage quality diagnostics when applicable and checkpoint writes; no preparation or media export')
        H.write_json(a.output/'profile_started.json', report)
        def timed(call, path):
            trace.reset()
            synchronize_execution(execution)
            reset_compatibility_measurement(execution)
            wall_start, cpu_start = time.perf_counter(), time.process_time()
            value = call()
            synchronization_seconds = synchronize_execution(execution)
            wall, cpu = time.perf_counter()-wall_start, time.process_time()-cpu_start
            record = dict(path=path, wall_seconds=wall, process_cpu_seconds=cpu,
                process_average_cpu_cores=cpu/max(wall, 1e-15),
                process_fraction_of_logical_capacity=cpu/max(wall*max(os.cpu_count() or 1,1),1e-15),
                process_lifetime_peak_rss_bytes=H.process_peak_rss_bytes(), **trace.snapshot())
            record['post_evaluation_cuda_synchronization_host_seconds'] = synchronization_seconds
            resident = compatibility_measurement(execution)
            if resident is not None:
                record['compatibility_gpu_measurement'] = resident
            if isinstance(value, dict):
                record['body_phases_after_complete_evaluation'] = dict((problem.body.cache or {}).get('phases', {}))
                record['joint_evaluation_phases'] = value.get('evaluation_phases', value.get('phases'))
            return value, record
        if a.operation == 'solve':
            from calibration.program_tracking.body_tracking.skate_run_policy import RunPolicy
            policy_file = H._existing(a.run_policy)
            policy_data = json.loads(policy_file.read_text(encoding='utf8'))
            policy_data = policy_data.get('policy', policy_data)
            policy = RunPolicy(**policy_data)
            problem.jacobian_seconds_hint = a.jacobian_seconds_hint if a.jacobian_seconds_hint is not None else 0.
            report['run_policy'] = dict(path=str(policy_file), sha256=H.sha256_file(policy_file), policy=vars(policy),
                initial_jacobian_seconds_hint=problem.jacobian_seconds_hint,
                hint_source='explicit command-line estimate' if a.jacobian_seconds_hint is not None else 'production default zero',
                hint_semantics='The production monitor reserves this estimate before Jacobians and only increases it from measured calls; no extra evaluation is performed by the profiler.')
            clear_result_caches(problem)
            result, timing = timed(lambda: problem.solve(x.copy(), policy), 'production_solve')
            final_x, solver_report = result
            report.update(solve=timing, solver_report=solver_report, final_x_sha256=H.array_digest(final_x),
                          accepted_steps=problem.accepted, final_candidate_directory=str(problem.output))
        else:
            timings = []
            snapshot = None
            for repeat in range(a.repeats):
                clear_result_caches(problem)
                value, timing = timed(lambda: problem.evaluate(x, True), 'fresh_complete_jacobian')
                if not np.isfinite(value['residual']).all() or not np.isfinite(value['jac'].data).all():
                    raise ValueError('Nonfinite full objective/Jacobian; refusing a speed result')
                timing['repeat'] = repeat+1
                timings.append(timing)
                snapshot = H.snapshot_arrays(value, x, dict(input_fingerprint=fingerprint, identity=identity))
                report.update(initial_objective=float(value['residual']@value['residual']/2),
                    cost_parts=value['parts'], jacobian_shape=list(value['jac'].shape), jacobian_nnz=int(value['jac'].nnz),
                    gradient_inf=float(np.max(np.abs(value['jac'].T@value['residual']), initial=0.)))
                del value
                if a.include_fun_jac:
                    clear_result_caches(problem)
                    _, fun = timed(lambda: problem.evaluate(x, False), 'fresh_fun')
                    pair, jac = timed(lambda: problem.evaluate(x, True), 'same_x_jac_after_fun')
                    jac['cache_equivalence'] = compare_cached_evaluation(snapshot, pair, a)
                    for rec in (fun, jac):
                        rec['repeat'] = repeat+1
                    timings.extend((fun, jac))
                print(json.dumps(dict(event='complete_profile_repeat', repeat=repeat+1,
                    wall_seconds=timing['wall_seconds'], process_average_cpu_cores=timing['process_average_cpu_cores'])), flush=True)
            report['evaluations'] = timings
            report['median_complete_jacobian_seconds'] = statistics.median(v['wall_seconds'] for v in timings if v['path']=='fresh_complete_jacobian')
            if a.include_fun_jac:
                report['median_fun_then_jac_seconds'] = statistics.median(sum(v['wall_seconds'] for v in timings
                    if v['repeat']==i+1 and v['path'] in ('fresh_fun','same_x_jac_after_fun')) for i in range(a.repeats))
            if a.compare_snapshot:
                report['comparison'] = compare_snapshot(snapshot, a.compare_snapshot, a)
            np.savez_compressed(a.output/'snapshot.npz', **snapshot)
        if cuda_execution_owners(execution):
            report['gpu_memory_bytes'] = gpu_allocator_records(execution)
        report['execution_after_measurement'] = execution.record()
        before = report['execution'].get('derivative_usage', {})
        after = report['execution_after_measurement'].get('derivative_usage', {})
        report['derivative_usage_during_measurement'] = {key: value-before.get(key, 0) for key, value in after.items()}
        measured_calls = report.get('evaluations', [report.get('solve', {})])
        compact_calls = sum(v.get('phases', {}).get('compact_forward_rotations', {}).get('calls', 0) for v in measured_calls)
        compatibility_calls = sum(v.get('compatibility_gpu_measurement', {}).get('measurement', {}).get('calls', 0)
                                  for v in measured_calls)
        body_backend = getattr(execution, '_body_backend', None)
        report['backend_evidence'] = dict(compact_forward_calls_during_measurement=compact_calls,
            cuda_compact_forward_called=bool(body_backend is not None and compact_calls),
            requested_compatibility_backend=getattr(a, 'compatibility_backend', 'cpu'),
            resident_compatibility_calls_during_measurement=compatibility_calls,
            cuda_resident_compatibility_called=compatibility_calls > 0,
            scope='Each CUDA flag applies only to its named implementation scope. Resident compatibility covers native55 FK, selected skin and active +/- compatibility consumers; body derivatives, camera, CSR and TRF/LSMR retain their separately recorded implementations.',
            note='Owner measurement calls exclude static setup; allocated GPU memory or setup H2D alone is not execution evidence.')
        report['process_lifetime_peak_rss_bytes'] = H.process_peak_rss_bytes()
        # Include every lazily loaded optimized kernel and reject any import
        # resolved outside --project, also when first used during evaluation.
        report['implementation_sha256'] = implementation_records(a.project)
        H.write_json(a.output/'profile.json', report)
        return report


def main(argv=None):
    p = parser()
    a = p.parse_args(argv)
    for name, value in vars(a).copy().items():
        if isinstance(value, Path):
            setattr(a, name, value.resolve())
    try:
        validate_args(a)
    except (ValueError, FileExistsError) as exc:
        p.error(str(exc))
    a.output.mkdir(parents=True, exist_ok=False)
    print(json.dumps(dict(main_log=str(a.output/'progress.jsonl'), profile=str(a.output/'profile.json'), operation=a.operation)), flush=True)
    for key, value in dict(BACKEND=a.backend, COMPATIBILITY_BACKEND=a.compatibility_backend,
        WORKERS=a.workers, BLAS_THREADS=a.blas_threads,
        NATIVE_THREADS=a.native_threads, GPU_BATCH_SIZE=a.gpu_batch_size,
        PERTURBATION_BATCH=a.perturbation_batch, MAX_BATCH_SAMPLES=a.max_batch_samples).items():
        os.environ['RINKSPACE_BODY_'+key] = str(value)
    if a.native_library is not None:
        os.environ['RINKSPACE_BODY_NATIVE_LIBRARY'] = str(a.native_library)
    for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[key] = str(a.blas_threads)
    select_project_sources(a.project)
    started = time.perf_counter()
    try:
        # Match the existing Windows launcher: load the chosen native library
        # before FullSurface or the optional Torch adapter imports.
        from rinkspace.core.native import NativeCore
        record = json.loads((a.project/PROGRAMME/'fit_v11/camera_solver_refined_1102.json').read_text(encoding='utf8'))
        selected = a.native_library or os.environ.get('RINKSPACE_BODY_NATIVE_LIBRARY') or record['native_library']['path']
        held_native = NativeCore(selected)
        report = run_profile(a)
        report['launcher_wall_seconds'] = time.perf_counter()-started
        H.write_json(a.output/'profile.json', report)
        print(json.dumps(dict(profile=str(a.output/'profile.json'),
            median_complete_jacobian_seconds=report.get('median_complete_jacobian_seconds'),
            solve_wall_seconds=report.get('solve', {}).get('wall_seconds'), comparison=report.get('comparison'))), flush=True)
        comparison = report.get('comparison', {})
        return 0 if all(comparison.get(k, True) for k in ('residual_within_tolerance','jacobian_within_tolerance')) else 2
    except BaseException as exc:
        H.write_json(a.output/'profile_failed.json', dict(error=repr(exc), operation=a.operation,
            elapsed_seconds=time.perf_counter()-started, prior_results_modified=False))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
