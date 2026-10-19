"""Keep an owned residual actor's role planes on its complete source block.

The parent role certificate and records remain complete.  Only that actor's
private view has fewer records; an original per-stamp ray batch is never split.
"""
from __future__ import annotations

import copy
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np

_SCOPE = None


def _make_view(role, block_name):
    indices = [i for i, record in enumerate(role.records)
               if record['source_role_block'] == block_name]
    if not indices:
        return None, 'No original admitted records for this owned block'
    owned = set(indices)
    remap = {original: local for local, original in enumerate(indices)}
    groups = {}
    for stamp, original_rows in role.source_row_groups.items():
        rows = np.asarray(original_rows)
        relevant = [int(i) in owned for i in rows]
        if not any(relevant):
            continue
        if not all(relevant):
            return None, 'Original ray batch shares a source stamp across blocks: '+str(stamp)
        groups[stamp] = np.asarray([remap[int(i)] for i in rows], dtype=rows.dtype)
    view = copy.copy(role)
    view.records = [role.records[i] for i in indices]
    view.source_uv = np.asarray(role.source_uv)[indices].copy()
    view.indices = np.asarray(role.indices)[indices].copy()
    view.source_row_groups = groups
    view._current_binding_cache = None
    view._facet_course_cache = None
    view.exact_cache_statistics = dict.fromkeys(role.exact_cache_statistics, 0)
    # Synchronization updates this same dictionary in this one worker. Source
    # certificate/contract metadata is retained exactly, without requalification.
    view.source_cameras = role.source_cameras
    if sum(len(rows) for rows in groups.values()) != len(indices):
        return None, 'Original role ray groups do not cover exactly the owned records'
    for stamp, rows in groups.items():
        original = np.asarray(role.source_uv)[role.source_row_groups[stamp]]
        local = view.source_uv[rows]
        if (original.shape != local.shape or original.dtype != local.dtype
                or original.tobytes() != local.tobytes()):
            return None, 'Original complete per-stamp source UV batch bytes changed'
    from .role_view_binding_seed_runtime import seed_owned_role_view_cache
    seed_owned_role_view_cache(role, view, indices)
    return view, None


class _ActorScope:
    def __init__(self, block, *, report_directory=None):
        self.block = block
        self.geometry = block.geometry
        self.role = block._original_motion_far_board_role_binding
        self.original_current_binding = self.role.current_binding
        self.view, self.fallback = _make_view(self.role, block.name)
        self.calls = 0
        self.destination = None if report_directory is None else Path(report_directory).resolve()
        if self.destination is not None:
            self.destination.mkdir(parents=True, exist_ok=True)
        self.receipt = dict(schema='rinkspace.owned-actor-complete-block-role-binding.v1',
            pid=multiprocessing.current_process().pid, block=block.name,
            full_role_records=len(self.role.records),
            owned_role_records=0 if self.view is None else len(self.view.records),
            complete_original_per_stamp_UV_batches_preserved=self.view is not None,
            parent_certificate_contract_and_original_role_records_unchanged=True,
            source_cameras_shared_only_inside_one_private_actor=True,
            independent_current_binding_and_facet_caches=True,
            original_existing17_full_binding_retained=True,
            original_complete_world_admission_before_planes_retained=True,
            original_binding_cache_seed=(None if self.view is None else
                self.view._rinkspace_role_binding_seed_receipt),
            current_q_or_source_math_errors_not_caught=True,
            fallback_reason=self.fallback)
        self.write(dict(stage='owned_actor_role_view_installed', **self.receipt))

    def write(self, value):
        if self.destination is not None:
            value['wall_time_ns'] = time.time_ns()
            with (self.destination/('actor_'+str(self.receipt['pid'])+'.jsonl')).open('a', encoding='utf8') as output:
                output.write(json.dumps(value, sort_keys=True)+'\n')

    def current_binding(self, binder, q):
        if binder is not self.geometry or self.view is None:
            return self.original_current_binding(binder, q)
        # Everything below is the original producer on complete same-stamp
        # batches. Its original domain/math exception must propagate directly.
        bound = self.view.current_binding(binder, q)
        self.calls += 1
        if self.calls == 1:
            self.write(dict(stage='original_role_binding_complete_owned_block',
                block=self.block.name, pid=self.receipt['pid'],
                full_role_records=len(self.role.records), computed_role_records=len(bound['records']),
                original_world_q_sha256=hashlib.sha256(np.asarray(q, float).tobytes()).hexdigest(),
                original_full_world_chain_and_complete_per_stamp_rays_retained=True,
                original_records_arrays_and_plane_J_producers_retained=True,
                parent_or_other_worker_state_shared=False, succeeded=True))
        return bound


def install_owned_actor_role_block_binding_runtime(block, *, report_directory=None):
    """Install after hydration, in the original owned residual actor only."""
    global _SCOPE
    process = multiprocessing.current_process()
    if not process.name.startswith('rink-source-'):
        raise ValueError('Private role block binding may only install in an owned source actor')
    if _SCOPE is not None:
        if _SCOPE.block is not block:
            raise ValueError('One private source actor cannot replace its original owner')
        return _SCOPE.receipt
    from structured_programme_source_candidate import original_motion_far_board_roles as roles
    role = getattr(block, '_original_motion_far_board_role_binding', None)
    if role is None:
        return dict(installed=False, reason='Owned source block has no admitted role binding')
    if type(role) is not roles.OriginalMotionFarBoardRoleBinding:
        return dict(installed=False, reason='Original admitted role owner type required')
    if role is not getattr(block.geometry, '_original_motion_far_board_role_binding', None):
        return dict(installed=False, reason='Original block and geometry role identity required')
    if 'current_binding' in vars(role):
        return dict(installed=False, reason='Original class current_binding required before private instance wrapper')
    try:
        scope = _ActorScope(block, report_directory=report_directory)
    except Exception as exc:
        # This only prepares the private execution view. No current source
        # calculation occurred; preserve the complete original function.
        return dict(installed=False, reason=type(exc).__name__+': '+str(exc))
    role.current_binding = scope.current_binding
    _SCOPE = scope
    scope.receipt.update(installed=True, original_finite_dispatch_and_existing_installed_code_unchanged=True,
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return scope.receipt
