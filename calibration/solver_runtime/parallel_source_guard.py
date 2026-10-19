"""Run the existing finite-extent guard in the resident residual actors.

Install before constructing CurrentProgrammeSource or starting source actors.
Only the six independent bundle guard calls are dispatched differently. The
original source worker, backend checks, q chart and guard operations survive.
This candidate deliberately leaves registered source files untouched.
"""
from __future__ import annotations

import hashlib
import inspect
import sys
import textwrap
import time


_WORKER_GUARD = """                elif mode=='finite_extent_guard':
                    local=block.local(command['q']);f=local[7::4]
                    if np.any((f<np.log(100.))|(f>np.log(50000.))):raise ValueError('Physical focal domain')
                    block.bundle.validate_finite_extents(block.bundle.cache(local,False))
                    reply=dict(status='ok',operation='finite_extent_guard',original_complete_bundle_guard=True)
"""


def compile_guarded_source_actor(*, complete_J_shared_bank=False, payload_loader=None):
    """Compile the same original worker, guards and native-backend checks."""
    from calibration.solver_runtime import install
    install()
    from structured_programme_source_candidate.runtime_backend_contract import install_source_worker_backend_contract
    install_source_worker_backend_contract()
    import z190_process_source_evaluation as source

    original = inspect.getsource(source._rinkspace_original_actor_worker)
    unpickle = '        block=pickle.loads(payload);del payload\n'
    review = "                elif mode=='review':\n"
    ready = "        connection.send(dict(status='ready',pid=os.getpid(),name=block.name))\n"
    if any(original.count(marker) != 1 for marker in (unpickle, review, ready)):
        raise RuntimeError('Original source worker guard insertion boundaries changed')
    hydrate = unpickle
    if payload_loader is not None:
        hydrate = '        block=_rinkspace_actor_payload_loader(payload);del payload\n'
    modified = original.replace(unpickle, hydrate +
        '        _rinkspace_finalize_worker_backend(task_root,native_threads,dynamic_modules)\n', 1)
    modified = modified.replace(review, _WORKER_GUARD + review, 1)
    modified = modified.replace(ready,
        "        connection.send(dict(status='ready',pid=os.getpid(),name=block.name,finite_extent_guard_supported=True))\n", 1)
    namespace = dict(vars(source))
    if payload_loader is not None:
        namespace['_rinkspace_actor_payload_loader'] = payload_loader
    if complete_J_shared_bank:
        from calibration.solver_runtime.resident_programme_J_runtime import (
            _worker_complete_J_evaluate, _extend_worker_source)
        modified = _extend_worker_source(modified)
        namespace['_rinkspace_complete_J_evaluate'] = _worker_complete_J_evaluate
    exec(compile(modified, source.__file__ + '::<resident-finite-extent-guard>', 'exec'), namespace)
    return namespace['_actor_worker']


def run_guarded_source_actor(*args, complete_J_shared_bank=False):
    """Project-owned, Windows-spawn-pickleable actor entry point."""
    return compile_guarded_source_actor(complete_J_shared_bank=complete_J_shared_bank)(*args)


def _ordered_source_guards(actors, q):
    """Evaluate every independent bundle; report the first original-order error."""
    import numpy as np
    import z190_process_source_evaluation as source
    actors.start()
    delegate = actors.delegate
    delegate._check_ready()
    if any(not actor.get('finite_extent_guard_supported') for actor in delegate._actors):
        raise RuntimeError('Resident source actors started before finite guard installation')
    delegate._busy = True
    delegate.last_stats = []
    received = sent = 0
    began = time.perf_counter()
    try:
        q = np.asarray(q, float)
        for actor in delegate._actors:
            delegate._send(actor, dict(mode='finite_extent_guard', q=q))
            sent += 1
        for actor in delegate._actors:
            received += 1
            reply = delegate._receive(actor)
            delegate.last_stats.append(dict(block=actor['block'].name, **reply))
    finally:
        # Exactly as original residual/review transport: consume every sent
        # reply before allowing another command, including after a rejection.
        for actor in delegate._actors[received:sent]:
            try:
                delegate._receive(actor)
            except source.SourceActorTransportError:
                raise
            except Exception:
                pass
        delegate._busy = False
        actors.last_stats = list(delegate.last_stats)
        actors.last_guard_seconds = time.perf_counter() - began
        actors._report('original_finite_extent_guards_evaluated')


def install_parallel_source_guard():
    """Install execution hooks before a full graph is built; return the receipt.

    The raw source's existing capture() binds this exact guard, so all later
    wrappers retain their original world/camera/truth/external checks. A serial
    raw source continues to use the original complete bundle loop.
    """
    from calibration.solver_runtime import worker_entry
    import z190_process_source_evaluation as source_workers
    from structured_programme_source_candidate.current_full_source import CurrentProgrammeSource
    from structured_programme_source_candidate.residual_only_source_actors import ResidualOnlySourceActors
    prior = getattr(CurrentProgrammeSource, '_rinkspace_parallel_guard_installation', None)
    if prior is not None:
        return dict(prior, already_installed=True)

    serial_guard = CurrentProgrammeSource.guard
    original = textwrap.dedent(inspect.getsource(serial_guard))
    loop = """    for b in s.blocks:
        local=b.local(q);f=local[7::4]
        if np.any((f<np.log(100.))|(f>np.log(50000.))):raise ValueError('Physical focal domain')
        b.bundle.validate_finite_extents(b.bundle.cache(local,False))
"""
    replacement = """    if isinstance(self.actors,_rinkspace_residual_only_type):
        _rinkspace_ordered_source_guards(self.actors,q)
    else:
""" + textwrap.indent(loop, '    ')
    if original.count(loop) != 1:
        raise RuntimeError('Original complete six-bundle guard loop changed')
    modified = original.replace(loop, replacement, 1)
    namespace = dict(vars(sys.modules[CurrentProgrammeSource.__module__]))
    namespace['_rinkspace_ordered_source_guards'] = _ordered_source_guards
    namespace['_rinkspace_residual_only_type'] = ResidualOnlySourceActors
    exec(compile(modified, inspect.getsourcefile(serial_guard) + '::<parallel-source-guard>', 'exec'), namespace)

    original_receive = source_workers.PersistentSourceActors._receive
    def receive_with_guard_capability(manager, actor):
        answer = original_receive(manager, actor)
        if answer.get('status') == 'ready':
            actor['finite_extent_guard_supported'] = bool(answer.get('finite_extent_guard_supported'))
        return answer
    source_workers.PersistentSourceActors._receive = receive_with_guard_capability
    worker_entry.run_source_actor = run_guarded_source_actor
    CurrentProgrammeSource._rinkspace_original_serial_guard = serial_guard
    CurrentProgrammeSource.guard = namespace['guard']
    # Keep a directly callable interface for the actual same-graph proof.
    ResidualOnlySourceActors.guard = _ordered_source_guards
    receipt = dict(schema='rinkspace.resident-original-bundle-guard-execution-hook.v1',
        original_guard_text_sha256=hashlib.sha256(original.encode()).hexdigest(),
        runtime_guard_text_sha256=hashlib.sha256(modified.encode()).hexdigest(),
        worker_entry=f'{run_guarded_source_actor.__module__}.{run_guarded_source_actor.__name__}',
        independent_original_bundles=True, original_bundle_guard_operations_retained=True,
        dispatches_only_residual_only_actor_instances=True,
        original_family_order_error_selection=True, all_other_guard_code_retained_exactly=True,
        original_profile_and_residual_commands_retained=True, full_J_route_changed=False,
        registered_source_files_modified=False, accuracy_accepted=False)
    CurrentProgrammeSource._rinkspace_parallel_guard_installation = receipt
    return receipt


def certify_residual_and_guard(raw, q, *, guard_trials=()):
    """Certify the original r and full raw guard using the caller's only graph.

    q and supplied trials use the raw source's existing evaluation chart. No
    graph is loaded here and no checkpoint, profile or source array is written.
    The generated invalid focal trial verifies sorted rejection and subsequent
    transport drainage; it is never proposed as an optimizer state.
    """
    import numpy as np
    from structured_programme_source_candidate.current_full_source import CurrentProgrammeSource
    from structured_programme_source_candidate.residual_only_source_actors import ResidualOnlySourceActors
    if not isinstance(raw, CurrentProgrammeSource) or not isinstance(raw.actors, ResidualOnlySourceActors):
        raise ValueError('Actual CurrentProgrammeSource with six residual-only actors required')
    q = np.asarray(q, float)
    original = CurrentProgrammeSource._rinkspace_original_serial_guard
    actors = raw.actors
    profile_before = [(bool(block.bundle.robust_phase), None if block.bundle.profile_weights is None
        else block.bundle.profile_weights.copy()) for block in raw.state.blocks]
    begin = time.perf_counter()
    reference = {block.name: np.asarray(block.evaluate(q, False)).copy() for block in raw.state.blocks}
    serial_r_seconds = time.perf_counter() - begin
    startup = time.perf_counter()
    actors.start()
    actor_startup_seconds = time.perf_counter() - startup
    begin = time.perf_counter()
    residual_records = []
    for block, value in actors.ordered_evaluations(q, False):
        expected = reference.pop(block.name)
        exact = bool(np.array_equal(expected, value))
        if not exact:
            raise RuntimeError('Actual original full residual changed in actor: ' + block.name)
        residual_records.append(dict(block=block.name, rows=len(value), exact=True,
            residual_sha256=hashlib.sha256(value.tobytes()).hexdigest()))
    actor_r_seconds = time.perf_counter() - begin
    if reference:
        raise RuntimeError('An original complete source block was omitted')

    bad_focal = q.copy()
    first = raw.state.blocks[0]
    bad_focal[first.start + 3] = np.log(50.)
    if first.local(bad_focal)[7] != np.log(50.):
        raise RuntimeError('Synthetic focal rejection must use the original local view chart')
    trials = [('current', q)] + [(f'actual_trial_{index}', np.asarray(value, float))
        for index, value in enumerate(guard_trials)] + [('invalid_first_focal', bad_focal), ('current_after_rejection', q)]
    guards = []
    for name, value in trials:
        samples = {}
        for route, fn in (('original_serial', lambda: original(raw, value)),
                          ('resident_parallel', lambda: raw.guard(value))):
            begin = time.perf_counter()
            try:
                fn()
                outcome = dict(valid=True)
            except ValueError as exc:
                outcome = dict(valid=False, error_type=type(exc).__name__, error=str(exc))
            samples[route] = dict(seconds=time.perf_counter()-begin, **outcome)
        lhs, rhs = samples['original_serial'], samples['resident_parallel']
        if lhs['valid'] != rhs['valid']:
            raise RuntimeError('Actual full source guard acceptance changed: ' + name)
        if not lhs['valid'] and lhs['error'] not in rhs['error']:
            raise RuntimeError('First original guard rejection changed: ' + name)
        if name == 'invalid_first_focal' and (lhs['valid'] or lhs['error'] != 'Physical focal domain'):
            raise RuntimeError('Synthetic focal trial did not exercise actor rejection')
        guards.append(dict(name=name, outcomes_equal=True, **samples))
    profile_exact = True
    for block, (robust, value) in zip(raw.state.blocks, profile_before):
        current = block.bundle.profile_weights
        profile_exact &= bool(block.bundle.robust_phase == robust and
            ((value is None and current is None) or
             (value is not None and current is not None and np.array_equal(value, current))))
    if not profile_exact:
        raise RuntimeError('Guard/residual certification changed an original frozen profile')
    return dict(schema='rinkspace.actual-complete-residual-and-resident-guard-certification.v1',
        passed=True, parameter_count=len(q), complete_source_rows=sum(item['rows'] for item in residual_records),
        q_sha256=hashlib.sha256(q.tobytes()).hexdigest(), residuals=residual_records,
        guard_trials=guards, original_frozen_profiles_exact=True,
        actor_startup_seconds=actor_startup_seconds, serial_r_seconds=serial_r_seconds,
        actor_r_seconds=actor_r_seconds, resident_pids=[a['process'].pid for a in actors.delegate._actors],
        full_J_evaluated=False, graph_loaded_by_certifier=False, accuracy_accepted=False)
