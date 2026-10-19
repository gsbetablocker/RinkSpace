"""Persist row-index scheduling hints, never a solution or a KKT certificate.

The original solver already accepts any in-range warm row indices. Every new
problem still builds its original matrix, solves, separates all omitted rows,
and passes the complete original primal/dual KKT gate. A stale hint can only
cost extra work; it cannot authorize a step.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from .original_halfspace_primal import PRIMAL_CERTIFICATE_SCHEMA,original_full_qp_kkt_verified


_RECEIPT = None


def install_persistent_qp_face_hint_runtime(*, cache_directory, report_directory=None):
    global _RECEIPT
    if _RECEIPT is not None:
        return _RECEIPT
    import z190_global_schur_linear_step as qp
    original = qp._linearly_constrained_shared_step
    holder = original.__globals__.get('_linearly_constrained_shared_step')
    if not callable(holder):
        raise RuntimeError('Original shared-QP warm-face storage owner required')
    directory = Path(cache_directory)
    directory.mkdir(parents=True, exist_ok=True)
    scope = os.environ.get('RINKSPACE_SOLVER_DATA_ROOT')
    first = True

    def event(value):
        print(json.dumps(dict(stage='original_QP_persistent_face_hint', **value)), flush=True)
        if report_directory is not None:
            path = Path(report_directory)
            path.mkdir(parents=True, exist_ok=True)
            with (path/'face_hint_events.jsonl').open('a', encoding='utf8') as stream:
                stream.write(json.dumps(value)+'\n')

    @functools.wraps(original)
    def solve(matrix, gradient, lower, upper, constraint_matrix, constraint_lower):
        nonlocal first
        shape = (len(constraint_lower)+int(np.isfinite(lower).sum())+int(np.isfinite(upper).sum()), len(gradient))
        key = hashlib.sha256(json.dumps(dict(scope=scope, shape=shape,
            primal_gate=PRIMAL_CERTIFICATE_SCHEMA), sort_keys=True).encode()).hexdigest()
        path = directory/(key+'.json')
        if first:
            first = False
            try:
                saved = json.loads(path.read_text(encoding='utf8'))
                rows = saved['rows']
                if (saved.get('primal_gate') != PRIMAL_CERTIFICATE_SCHEMA
                        or saved['scope'] != scope or saved['shape'] != list(shape)
                        or type(rows) is not list or any(type(i) is not int or not 0 <= i < shape[0] for i in rows)):
                    raise ValueError('Warm face row-index shape/scope changed')
                holder._last_verified_separation_faces = dict(shape=shape, scope=scope, rows=tuple(sorted(set(rows))))
                event(dict(action='loaded_scheduling_hint', rows=len(rows), shape=shape,
                    prior_solution_or_duals_loaded=False, full_current_problem_KKT_still_required=True))
            except (OSError, ValueError, KeyError, TypeError) as error:
                event(dict(action='original_cold_separation', reason=type(error).__name__))
        result = original(matrix, gradient, lower, upper, constraint_matrix, constraint_lower)
        seed = getattr(holder, '_last_verified_separation_faces', None)
        if isinstance(seed, dict) and tuple(seed.get('shape', ())) == shape and seed.get('scope') == scope:
            # This is just a hint, but save only after the original solver has
            # returned through its final complete KKT check.
            report = result[2]
            if original_full_qp_kkt_verified(report,result[0]):
                payload = dict(schema='rinkspace.original-QP-row-index-scheduling-hint.v1',
                    scope=scope, shape=list(shape), rows=[int(i) for i in seed['rows']],
                    primal_gate=PRIMAL_CERTIFICATE_SCHEMA,
                    solution_or_duals_stored=False, authority='scheduling_only')
                temporary = path.with_suffix('.'+str(os.getpid())+'.tmp')
                try:
                    temporary.write_text(json.dumps(payload)+'\n', encoding='utf8')
                    temporary.replace(path)
                    event(dict(action='saved_after_original_full_KKT', rows=len(seed['rows']), shape=shape))
                except OSError as error:
                    event(dict(action='optional_hint_save_failed', reason=str(error)))
        return result

    qp._linearly_constrained_shared_step = solve
    _RECEIPT = dict(schema='rinkspace.persist-original-QP-row-index-hints.v1',
        cache_directory=str(directory), original_initial_face_union_unchanged=True,
        original_current_H_g_all_constraints_and_full_KKT_unchanged=True,
        prior_q_J_H_step_or_duals_not_loaded=True, hint_is_not_acceptance_evidence=True)
    return _RECEIPT
