"""Ordered evaluation of independent original programme source bundles.

Only compute scheduling changes. No observations, row order, profiles, source
weights or camera parameters are selected or approximated here. A pool must
never evaluate one mutable bundle twice concurrently.
"""
from concurrent.futures import ThreadPoolExecutor
import os
import numpy as np
from z190_joint_source_solver import _BlockProfileUpdate


def ordered_block_evaluations(blocks, q, jacobian, workers):
    blocks = tuple(blocks)
    workers = int(workers)
    if workers < 1:
        raise ValueError('Positive source worker count required')
    if len({id(b.bundle) for b in blocks}) != len(blocks):
        raise ValueError('Parallel evaluations require distinct source bundles')
    if workers == 1:
        for block in blocks:
            yield block, block.evaluate(q, jacobian)
        return
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='rink-source') as pool:
        futures = [pool.submit(block.evaluate, q, jacobian) for block in blocks]
        for block, future in zip(blocks, futures):
            result = future.result()
            yield block, result
            del result


class ParallelBlockProfileUpdate(_BlockProfileUpdate):
    """Keep the existing checkpoint/token protocol and independent updates."""
    def __init__(self, blocks, workers):
        super().__init__(blocks)
        self.workers = int(workers)
        if self.workers < 1:
            raise ValueError('Positive source worker count required')

    def __call__(self, q):
        motion_only=os.environ.get('RINKSPACE_PROFILE_MOTION_ONLY','0')=='1'
        if self.workers == 1 and not motion_only:
            return super().__call__(q)
        def update(block):
            if motion_only:
                block.bundle.robust_phase=True
                preliminary=block.bundle.profile_motion_residual(block.local(q))
                block.bundle.profile_weights=np.minimum(1.,2./np.maximum(np.abs(preliminary),1e-12)).reshape(-1,2)
                return
            # Reuse the reference formula in the original callback.
            _BlockProfileUpdate((block,))(q)
        if self.workers == 1:
            for block in self.blocks:update(block)
            return
        with ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix='rink-profile') as pool:
            list(pool.map(update, self.blocks))


def make_parallel_profile_update(blocks, workers=1):
    return ParallelBlockProfileUpdate(blocks, workers)
