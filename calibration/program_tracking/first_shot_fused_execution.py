"""Explicit parent/spawn same-value execution and one full same-q source proof.

No source equation or solver gate is replaced. Worker entry points are top-level
functions: spawn imports the untouched Data/context modules before calling their
original initializer, so parent dispatch replacement cannot recurse in children.
"""
from copy import deepcopy
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time
from types import MethodType

import numpy as np

_SOURCE_INSTALLATION = None
_GEOMETRY_INSTALLATION = None


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _worker_receipt(installation):
    from . import first_shot_value_execution as value
    return dict(pid=os.getpid(), start_method=multiprocessing.get_start_method(),
                installer=installation, statistics=value.statistics())


def source_worker_init(*args):
    import native_transverse_bundle as runtime
    from . import first_shot_raw_execution as raw
    from . import first_shot_value_execution as value
    global _SOURCE_INSTALLATION
    if runtime.init_worker in (source_worker_init, raw.worker_init):
        raise RuntimeError('Spawn must import the original Data source initializer')
    raw.worker_init(*args)
    _SOURCE_INSTALLATION = value.install()


def source_chunk(task):
    import native_transverse_bundle as runtime
    from . import first_shot_raw_execution as raw
    if runtime.source_chunk in (source_chunk, raw.source_chunk) or _SOURCE_INSTALLATION is None:
        raise RuntimeError('Original spawn source producer and explicit value installation required')
    r, j, receipt = raw.source_chunk(task)
    receipt['same_value_spawn_execution'] = _worker_receipt(_SOURCE_INSTALLATION)
    return r, j, receipt


def geometry_worker_init(*args):
    from . import first_shot_transverse_execution as context
    from . import first_shot_value_execution as value
    global _GEOMETRY_INSTALLATION
    if context._init_context_worker is geometry_worker_init:
        raise RuntimeError('Spawn must import the original geometry context initializer')
    context._init_context_worker(*args)
    _GEOMETRY_INSTALLATION = value.install()


def geometry_sample(task):
    from . import first_shot_transverse_execution as context
    if context._context_sample is geometry_sample or _GEOMETRY_INSTALLATION is None:
        raise RuntimeError('Original geometry sampler and explicit value installation required')
    result, receipt = context._context_sample(task)
    receipt['same_value_spawn_execution'] = _worker_receipt(_GEOMETRY_INSTALLATION)
    return result, receipt


def install_worker_execution(owner, *, parent_installation, entry_reference, validation_references):
    """Call after raw.install, before source-value replay and any parallel work."""
    import native_transverse_bundle as runtime
    from . import first_shot_raw_execution as raw
    from . import first_shot_transverse_execution as context
    from . import first_shot_value_execution as value
    if hasattr(owner, 'fused_execution_receipt'):
        raise ValueError('Fused same-value execution is already installed')
    if owner.pool is not None or owner.bundle._geometry_pool is not None:
        raise ValueError('Fused execution must be selected before either pool opens')
    if runtime.init_worker is not raw.worker_init or runtime.source_chunk is not raw.source_chunk:
        raise ValueError('Install the existing original-cost raw wrapper before fused worker dispatch')
    if (context._init_context_worker.__module__ != context.__name__
            or context._context_sample.__module__ != context.__name__):
        raise ValueError('An unknown geometry context execution override is already installed')
    if parent_installation['producer'] != reference(value.__file__):
        raise ValueError('Parent same-value installation producer changed')
    refs = [reference(__file__), reference(value.__file__), reference(raw.__file__),
            reference(context.__file__), deepcopy(entry_reference)]
    for ref in validation_references:
        if reference(ref['path']) != ref:
            raise ValueError('Same-value validation report changed')
        report = json.loads(Path(ref['path']).read_text(encoding='utf8'))
        if report.get('passed') is not True:
            raise ValueError('A passed same-value/pivot validation report is required')
        for certified in (report.get('producer'), report.get('installer', {}).get('producer')):
            if certified is not None and reference(certified['path']) != certified:
                raise ValueError('Validation report producer differs from the current execution')
    for ref in refs:
        if ref not in owner.source_producers:
            owner.source_producers.append(deepcopy(ref))
    receipt = dict(schema='rinkspace.first-shot-fused-same-value-execution.v1', producers=refs,
        validation_references=deepcopy(validation_references), parent_installation=deepcopy(parent_installation),
        parent_source_and_geometry_each_explicitly_installed=True,
        source_initializer='raw.worker_init then value.install in each fresh spawn',
        geometry_initializer='original context initializer then value.install in each fresh spawn',
        source_sampler='original Data source_chunk through unchanged raw capture wrapper',
        geometry_sampler='original context sample and finite-difference producer',
        parent_dispatch_replacements_only=True, child_reimports_original_functions=True,
        full_same_seed_source_r_J_proof_required=True, full_same_seed_source_r_J_proof=None,
        source_equations_rows_weights_and_solver_acceptance_unchanged=True,
        execution_replay_is_not_a_new_measurement_or_accuracy_validation=True)
    previous_identity = owner.checkpoint_identity

    def identity(adapter):
        result = previous_identity()
        result['fused_same_value_execution'] = deepcopy(receipt)
        return result

    runtime.init_worker = source_worker_init
    runtime.source_chunk = source_chunk
    context._init_context_worker = geometry_worker_init
    context._context_sample = geometry_sample
    owner.checkpoint_identity = MethodType(identity, owner)
    owner.fused_execution_receipt = receipt
    return receipt


def _same_array(actual, expected, label):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if (actual.dtype != expected.dtype or actual.shape != expected.shape
            or not np.array_equal(np.ascontiguousarray(actual).view(np.uint8),
                                  np.ascontiguousarray(expected).view(np.uint8))):
        raise ValueError('Full same-q source proof differs: ' + label)


def _array_sha(value):
    value = np.asarray(value)
    digest = hashlib.sha256(value.dtype.str.encode() + str(value.shape).encode())
    digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _geometry_sample_count(owner):
    pool = owner.bundle._geometry_pool
    return len(getattr(pool, 'receipts', {})) if pool is not None else 0


def certify_seed_source(replay, output):
    """Evaluate every original current r/J row once against its old exact cache.

    The direct Data class call bypasses the seed source/metric replay wrappers.
    Its world(q,True) must hit the authenticated full same-q old world/D cache.
    The solver subsequently keeps using that now re-certified initial r/J.
    """
    import native_transverse_bundle as runtime
    import z190_streaming_source_normal as algebra
    from calibration.solver_runtime.source_loss import SourceLossWeights
    from . import first_shot_value_execution as value
    owner = replay.adapter
    output = Path(output)
    if output.exists():
        raise ValueError('Full same-q source proof must not overwrite existing evidence')
    if getattr(replay, 'world_reference', None) is None:
        raise ValueError('A fully authenticated same-q world/J manifest is required; do not recompute FD')
    replay.require_same_state(owner.start)
    q = replay.q.copy()
    geometry = np.asarray(q[owner.state.world_parameter_indices], float).tobytes()
    cached = owner.transverse_world_cache.get(geometry)
    if cached is None or cached[1] is None:
        raise ValueError('Authenticated complete same-q world derivatives are not installed')
    owner.guard(q)
    worlds, derivatives = owner.world(q, True)
    if worlds is not cached[0] or derivatives is not cached[1]:
        raise ValueError('Full source certification did not select the authenticated world/D cache')
    geometry_before = _geometry_sample_count(owner)
    receipts_before = len(owner.worker_receipts)
    evaluation_before = len(owner.evaluation_receipts)
    started = time.perf_counter()
    records = []
    source_rows = nnz = 0
    try:
        expected = replay.blocks(q, True)
        # CompleteTransverseBundle itself is the unchanged underlying source
        # producer, not the metric wrapper or any authenticated source replay.
        actual = runtime.CompleteTransverseBundle.source_blocks(owner, q, True)
        for ordinal, (fresh, old) in enumerate(zip(actual, expected, strict=True)):
            fr, fj, fm, fs, signature = algebra._checked_block(fresh, len(q), True)
            er, ej, em, es, expected_signature = algebra._checked_block(old, len(q), True)
            single = algebra.SourceBlockContract(len(q), len(er), (expected_signature,),
                                                'authenticated-complete-seed-source-proof')
            algebra._check_replay(signature, 0, single, True)
            for a, b, label in ((fresh.r, old.r, 'r'), (fresh.quadratic, old.quadratic, 'mask'),
                (fresh.scales, old.scales, 'scales'), (fj.data, ej.data, 'CSR data'),
                (fj.indices, ej.indices, 'CSR indices'), (fj.indptr, ej.indptr, 'CSR indptr')):
                _same_array(a, b, f'block {ordinal} ' + label)
            if fj.shape != ej.shape:
                raise ValueError('Same-q full source CSR shape changed')
            encode = dict(sort_keys=True, ensure_ascii=False, allow_nan=False,
                          separators=(',', ':'), default=algebra._json_default)
            if json.dumps(fresh.row_ledger, **encode) != json.dumps(old.row_ledger, **encode):
                raise ValueError('Same-q full source original row ledger changed')
            a, b = fresh.loss_weights, old.loss_weights
            if isinstance(a, SourceLossWeights) and isinstance(b, SourceLossWeights):
                _same_array(a.integration_weights, b.integration_weights, 'loss integration')
                if a.working_normalization_multiplier is None or b.working_normalization_multiplier is None:
                    if a.working_normalization_multiplier is not b.working_normalization_multiplier:
                        raise ValueError('Same-q source loss working normalization changed')
                else:
                    _same_array(a.working_normalization_multiplier, b.working_normalization_multiplier,
                                'loss working normalization')
            elif a is None or b is None:
                if a is not None or b is not None:
                    raise ValueError('Same-q source loss weights changed')
            else:
                _same_array(a, b, 'source loss weights')
            source_rows += len(fr); nnz += fj.nnz
            records.append(dict(block=ordinal, rows=len(fr), J_shape=list(fj.shape), J_nonzeros=fj.nnz,
                r_sha256=_array_sha(fresh.r), J_data_sha256=_array_sha(fj.data),
                J_indices_sha256=_array_sha(fj.indices), J_indptr_sha256=_array_sha(fj.indptr),
                dtype_shape_and_all_bytes_exact=True, original_row_ledger_and_loss_weights_exact=True))
        if source_rows != len(owner.scales) or nnz != replay.manifest['J_nonzeros']:
            raise ValueError('Full same-q source proof did not include every original row/J nonzero')
        replay.require_same_state(q)
        _same_array(q, owner.start, 'complete unchanged accepted seed q')
        if _geometry_sample_count(owner) != geometry_before:
            raise ValueError('Source proof unexpectedly evaluated geometry finite differences')
        workers = owner.worker_receipts[receipts_before:]
        frames = [i for receipt in workers for i in receipt['frames_0based']]
        if frames != list(range(len(owner.bundle.frames))):
            raise ValueError('Fresh source proof did not visit every native frame in order')
        for receipt in workers:
            installation = receipt.get('same_value_spawn_execution')
            if (installation is None or installation['start_method'] != 'spawn'
                    or installation['installer']['producer'] != reference(value.__file__)
                    or receipt.get('world_derivatives_received') is not True
                    or '_same_state_cost_raw' in receipt):
                raise ValueError('Fresh source/J worker did not use the authenticated fused spawn initializer')
        report = dict(schema='rinkspace.fused-original-full-source-same-q-proof.v1', passed=True,
            producer=reference(__file__), qaug_sha256=hashlib.sha256(q.tobytes()).hexdigest(),
            original_source_manifest=replay.manifest_reference, original_world_manifest=replay.world_reference,
            source_rows=source_rows, source_points=len(replay.raw), native_frames=len(owner.bundle.frames),
            parameter_count=len(q), source_blocks=len(records), J_nonzeros=nnz, blocks=records,
            all_r_J_masks_scales_loss_weights_ledgers_dtype_shape_and_bytes_exact=True,
            original_full_data_source_factory_executed_once=True,
            source_projection_executed=True, geometry_FD_executed=False,
            authenticated_same_q_world_D_used=True, source_workers=workers,
            actual_source_worker_pids=sorted({int(receipt['pid']) for receipt in workers}),
            original_evaluations=owner.evaluation_receipts[evaluation_before:],
            original_seed_q_and_checkpoint_unchanged=True,
            next_solver_initial_full_J_uses_authenticated_equal_cache=True,
            no_duplicate_source_projection_for_first_solver_linearization=True,
            downstream_source_value_replay_first_hit_shadow_still_required=True,
            elapsed_seconds=time.perf_counter() - started)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
        owner.fused_execution_receipt['full_same_seed_source_r_J_proof'] = reference(output)
        replay.receipt['fused_original_full_source_same_seed_proof'] = reference(output)
        print(json.dumps(dict(stage='fused_full_original_source_same_seed_bitwise_proof_passed',
            qaug_sha256=report['qaug_sha256'], source_rows=source_rows, J_nonzeros=nnz,
            source_blocks=len(records), elapsed_seconds=report['elapsed_seconds'],
            proof=reference(output))), flush=True)
        return report
    except BaseException as error:
        failure = dict(schema='rinkspace.fused-original-full-source-same-q-proof.v1', passed=False,
            producer=reference(__file__), qaug_sha256=hashlib.sha256(q.tobytes()).hexdigest(),
            type=type(error).__name__, message=str(error), completed_blocks=records,
            elapsed_seconds=time.perf_counter() - started, optimizer_called=False)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(failure, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
        raise
