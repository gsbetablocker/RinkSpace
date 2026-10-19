"""Observe one original complete fresh cycle; never substitute cached numerics."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import time


def attach_complete_cycle_observer(tree, namespace, destination, variant):
    destination = Path(destination)
    started = None
    completed = False
    start_identity = None
    block_signatures = []

    def signature(value):
        import numpy as np
        a = np.ascontiguousarray(value)
        return dict(dtype=a.dtype.str, shape=list(a.shape),
                    sha256=hashlib.sha256(memoryview(a).cast('B')).hexdigest())

    def begin(q, source, damping, fraction):
        nonlocal started, start_identity
        if started is not None:
            raise RuntimeError('Complete cycle comparison entered twice')
        from z190_joint_source_solver import _frozen_profile_token
        start_identity = dict(q=signature(q), profile=_frozen_profile_token(source.updater),
                              damping=float(damping), initial_fraction=float(fraction))
        started = time.perf_counter()
        print(json.dumps(dict(stage='complete_cycle_comparison_begin', variant=variant,
                              wall_time_ns=time.time_ns(), **start_identity)), flush=True)

    def block_signature(block, ordinal):
        matrix = block.J
        block_signatures.append(dict(ordinal=int(ordinal),
            family=block.row_ledger.get('family'), r=signature(block.r),
            quadratic=signature(block.quadratic), scales=signature(block.scales),
            J_shape=list(matrix.shape), J_data=signature(matrix.data),
            J_indices=signature(matrix.indices), J_indptr=signature(matrix.indptr)))

    def decorate_review(original):
        def review(q, event=None):
            nonlocal completed
            result = original(q, event)
            if not completed and event and event.get('checkpoint_kind') == 'accepted':
                if started is None:
                    raise RuntimeError('Accepted comparison state without complete cycle start')
                report = dict(schema='rinkspace.same-state-complete-fresh-cycle.v1',
                    variant=variant, start=start_identity, accepted_q=signature(q),
                    complete_cycle_seconds=time.perf_counter()-started,
                    original_guard_profile_update_constraints_initial_review_fresh_J_normal_solve_trials_checkpoint_and_accepted_review_included=True,
                    graph_construction_and_startup_source_diagnostics_excluded=True,
                    previous_J_or_normal_reused=False,
                    fresh_source_block_signatures=block_signatures,
                    source_stage_timing_seconds=event.get('source_stage_timing_seconds'),
                    acceptance=event.get('explicit_source_trial_acceptance'),
                    wall_time_ns=time.time_ns(), accuracy_accepted=False)
                (destination/'complete_cycle_comparison.json').write_text(
                    json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
                print(json.dumps(dict(stage='complete_cycle_comparison_accepted',
                    variant=variant, seconds=report['complete_cycle_seconds'],
                    q_sha256=report['accepted_q']['sha256'])), flush=True)
                completed = True
            return result
        return review

    def finished():
        if completed:
            marker = dict(stage='complete_result_saved_and_owned_workers_closed',
                          wall_time_ns=time.time_ns(), variant=variant)
            (destination/'complete_result_saved.json').write_text(
                json.dumps(marker, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
            print(json.dumps(marker), flush=True)

    sites = []
    class Observe(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            self.generic_visit(node)
            if node.name == 'recorded_initial_blocks':
                class Record(ast.NodeTransformer):
                    def visit_Expr(self, expression):
                        if (isinstance(expression.value, ast.Yield)
                                and isinstance(expression.value.value, ast.Name)
                                and expression.value.value.id == 'b'):
                            sites.append('fresh_source_block_signatures')
                            return [ast.parse('_rinkspace_cycle_block(b, ordinal)').body[0], expression]
                        return self.generic_visit(expression)
                node = Record().visit(node)
            if node.name == 'review':
                node.decorator_list.insert(0, ast.Name(id='_rinkspace_cycle_review', ctx=ast.Load()))
                sites.append('original_review')
            return node

        def visit_Expr(self, node):
            self.generic_visit(node)
            # Enter before the original guard, profile preparation, domain,
            # review, full J and original solve. Both variants use this site.
            if (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == 'print' and any(
                        isinstance(n, ast.keyword) and n.arg == 'stage'
                        and isinstance(n.value, ast.Constant)
                        and n.value.value == 'current_source_guard' for n in ast.walk(node))):
                sites.append('before_original_current_source_guard')
                return [ast.parse('_rinkspace_cycle_begin(q, source, initial_damping, continued_line_search_fraction)').body[0], node]
            return node
    tree = Observe().visit(tree)
    if sorted(sites) != ['before_original_current_source_guard', 'fresh_source_block_signatures', 'original_review']:
        raise ValueError('Original complete comparison boundary changed: '+str(sites))
    namespace['_rinkspace_cycle_begin'] = begin
    namespace['_rinkspace_cycle_review'] = decorate_review
    namespace['_rinkspace_cycle_block'] = block_signature
    namespace['_rinkspace_cycle_finished'] = finished
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    main.body.append(ast.parse('_rinkspace_cycle_finished()').body[0])
    ast.fix_missing_locations(tree)
    return tree
