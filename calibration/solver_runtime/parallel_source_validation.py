"""Authenticate the actor source stream against the already sealed full cache.

The caller installs this before its one complete programme run. The ordinary
cache loader remains the authority for q, J, H/g, source identity and profiles.
Only the candidate residual stream is evaluated, with the original baseline r
read directly from that loaded cache. No second graph or baseline is built.
"""
from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path
import time


def _raw_source(source):
    from structured_programme_source_candidate.current_full_source import CurrentProgrammeSource
    visited = set()
    while source is not None and id(source) not in visited:
        visited.add(id(source))
        if isinstance(source, CurrentProgrammeSource):
            return source
        values = vars(source)
        cap = values.get('cap')
        if isinstance(cap, dict):
            raw = getattr(cap.get('iter_source_blocks'), '__self__', None)
            if isinstance(raw, CurrentProgrammeSource):
                return raw
        source = values.get('parent', values.get('base'))
    raise RuntimeError('Complete source wrapper chain lost the actual programme source owner')


def _array_identity(value):
    import numpy as np
    from z190_streaming_source_normal import _array_digest
    array = np.asarray(value)
    digest = hashlib.sha256()
    _array_digest(digest, array)
    return dict(shape=list(array.shape), dtype=array.dtype.str, sha256=digest.hexdigest())


def validate_loaded_normal(normal, q, source, profile_token):
    import numpy as np
    from z190_joint_source_solver import _frozen_profile_token
    from z190_streaming_source_normal import _checked_block, _check_replay, _finish_replay
    from structured_programme_source_candidate.residual_only_source_actors import ResidualOnlySourceActors
    raw = _raw_source(source)
    if not isinstance(raw.actors, ResidualOnlySourceActors):
        raise RuntimeError('Complete actor validation requires the existing residual-only source actors')
    cache = normal.native_replay_cache
    if cache is None or len(cache.blocks) != len(normal.contract.signatures):
        raise RuntimeError('Authenticated full baseline residual blocks are unavailable')
    before = _frozen_profile_token(source.updater)
    if before != profile_token:
        raise RuntimeError('Original frozen profiles changed before candidate replay')
    original_state = np.asarray(q).copy()
    count = rows = 0
    records = []
    begin = time.perf_counter()
    iterator = iter(source.blocks(q, False))
    try:
        for ordinal, block in enumerate(iterator):
            if ordinal >= len(cache.blocks):
                raise RuntimeError('Candidate source appended an uncontracted block')
            reference = cache.blocks[ordinal]
            r, _, mask, scales, signature = _checked_block(block, len(q), False)
            _check_replay(signature, ordinal, normal.contract, False)
            fields = {}
            for name, candidate, baseline in (('r', r, reference.r),
                    ('quadratic', mask, reference.quadratic), ('scales', scales, reference.scales)):
                left, right = np.asarray(candidate), np.asarray(baseline)
                candidate_identity, baseline_identity = _array_identity(left), _array_identity(right)
                exact = candidate_identity == baseline_identity
                if not exact:
                    error = dict(block=ordinal, family=block.row_ledger.get('family'), field=name,
                        candidate=candidate_identity, baseline=baseline_identity)
                    if left.shape == right.shape and left.size and name != 'quadratic':
                        error['max_absolute_error'] = float(np.max(np.abs(left-right)))
                    raise RuntimeError('Complete original cache replay mismatch: ' + json.dumps(error))
                fields[name] = dict(exact=True, baseline_sha256=baseline_identity['sha256'], **candidate_identity)
            # inventory_sha256 above hashes the complete masks, scales and
            # every original row ledger field with the original public code.
            records.append(dict(ordinal=ordinal, family=block.row_ledger.get('family'), rows=len(r),
                inventory_sha256=signature.inventory_sha256,
                complete_row_ledger_and_native_scales_exact=True, arrays=fields))
            count += 1
            rows += len(r)
    finally:
        close = getattr(iterator, 'close', None)
        if close is not None:
            close()
    _finish_replay(count, rows, normal.contract)
    after = _frozen_profile_token(source.updater)
    if after != before or not np.array_equal(q, original_state):
        raise RuntimeError('Candidate residual replay changed the original frozen q or profile token')
    delegate = raw.actors.delegate
    if any(not actor.get('finite_extent_guard_supported') for actor in delegate._actors):
        raise RuntimeError('An actual resident source actor lacks the complete guard operation')
    return dict(schema='rinkspace.loaded-full-cache-actor-residual-and-inventory-certification.v1',
        passed=True, source_cache_baseline_recomputed=False, source_graph_reconstructed=False,
        original_full_J_H_gradient_cost_retained=True,
        parameters=len(q), blocks=count, rows=rows, records=records,
        q=_array_identity(q), profile_token_before=before, profile_token_after=after,
        candidate_replay_seconds=time.perf_counter()-begin,
        actual_resident_workers=len(delegate._actors),
        actual_actor_pids=[actor['process'].pid for actor in delegate._actors],
        native_threads_per_actor=delegate.native_threads,
        all_actual_workers_support_complete_original_guard=True,
        actual_last_actor_stats=raw.actors.last_stats, accuracy_accepted=False)


def install_validation(rootdir, *, require_residual_actors=True):
    """Install a once-per-loaded-normal validation; write a small Data report."""
    import structured_programme_source_candidate.complete_same_state_replay as loader
    existing = getattr(loader, '_rinkspace_actor_cache_validation', None)
    rootdir = Path(rootdir).resolve()
    if existing is not None:
        if (existing['rootdir'] != str(rootdir) or
                existing['require_residual_actors'] != bool(require_residual_actors)):
            raise RuntimeError('Actor validation was already installed for another output directory')
        return dict(existing, already_installed=True)
    original = loader.load_complete_same_state_normal
    target = rootdir / 'actual_full_cache_actor_residual_guard_capability_certification.json'

    @functools.wraps(original)
    def load_and_certify(directory, q, source, manifest, profile_token, **kwargs):
        normal = original(directory, q, source, manifest, profile_token, **kwargs)
        rootdir.mkdir(parents=True, exist_ok=True)
        if not require_residual_actors and _raw_source(source).actors is None:
            report = dict(schema='rinkspace.serial-source-no-actor-validation-required.v1',
                status='not_required_for_serial_source', actor_validation_performed=False,
                authenticated_original_source_cache=str(Path(directory).resolve()),
                original_strict_cache_loader_completed=True, actual_resident_workers=0,
                original_q_J_H_gradient_cost_and_profile_identity_verified_by_loader=True,
                cache_identity_changed=False, accuracy_accepted=False)
            target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf8')
            print(json.dumps(dict(stage='serial_source_actor_validation_not_required', report=str(target))), flush=True)
            return normal
        try:
            report = validate_loaded_normal(normal, q, source, profile_token)
        except Exception as exc:
            report = dict(schema='rinkspace.loaded-full-cache-actor-residual-and-inventory-certification.v1',
                passed=False, error_type=type(exc).__name__, reason=str(exc),
                candidate_not_admitted=True, original_sealed_cache_and_checkpoint_retained=True)
            target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf8')
            try:
                raw = _raw_source(source)
                if raw.actors is not None:
                    raw.actors.close()
            except Exception:
                pass
            raise RuntimeError('Six-source actor candidate refused; see ' + str(target)) from exc
        report['authenticated_original_source_cache'] = str(Path(directory).resolve())
        target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf8')
        print(json.dumps(dict(stage='actual_complete_actor_source_certification_passed',
            rows=report['rows'], blocks=report['blocks'], candidate_seconds=report['candidate_replay_seconds'],
            report=str(target))), flush=True)
        return normal

    loader.load_complete_same_state_normal = load_and_certify
    receipt = dict(rootdir=str(rootdir), report=str(target), baseline_recomputed=False,
        require_residual_actors=bool(require_residual_actors),
        installs_before_driver_cache_load=True, failed_candidate_aborts_and_closes_only_owned_actors=True)
    loader._rinkspace_actor_cache_validation = receipt
    return receipt


def certify_saved_byte_identities(report_path, output_path):
    """Check saved candidate hashes against baseline bytes without source replay.

    This upgrades earlier np.array_equal reports to strict byte identity,
    including signed zero. It reads only the original r/mask/scales arrays,
    never J, any programme graph or an optimizer state.
    """
    from npz_readonly import mapped_array
    report_path, output_path = Path(report_path), Path(output_path)
    candidate = json.loads(report_path.read_text(encoding='utf8'))
    if candidate.get('passed') is not True:
        raise ValueError('A failed candidate cannot obtain byte certification')
    prefix = Path(candidate['authenticated_original_source_cache']) / 'initial_normal_source_blocks'
    records = []
    for item in candidate['records']:
        filename = prefix / (str(item['ordinal']).zfill(3) + '.npz')
        fields = {}
        for name in ('r', 'quadratic', 'scales'):
            expected = item['arrays'][name]
            expected = {key: expected[key] for key in ('shape', 'dtype', 'sha256')}
            baseline = _array_identity(mapped_array(filename, name))
            if baseline != expected:
                raise ValueError(f'Saved candidate byte mismatch: block {item["ordinal"]} field {name}')
            fields[name] = dict(byte_exact=True, **baseline)
        records.append(dict(ordinal=item['ordinal'], family=item['family'], rows=item['rows'], arrays=fields))
    result = dict(schema='rinkspace.saved-actor-residual-strict-byte-certification.v1', passed=True,
        candidate_report=str(report_path.resolve()),
        candidate_report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
        original_source_cache=candidate['authenticated_original_source_cache'],
        blocks=len(records), rows=sum(item['rows'] for item in records), records=records,
        signed_zero_and_dtype_bytes_checked=True, source_graph_loaded=False,
        candidate_or_baseline_recomputed=False, full_J_loaded=False, accuracy_accepted=False)
    output_path.write_text(json.dumps(result, indent=2), encoding='utf8')
    return result
