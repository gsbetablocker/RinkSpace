"""Reusable source-owned solver runtime; install before importing legacy factors."""

from .runtime import install, source_path, status


def configure_native_runtime(*, stage="reusable_runtime", native_threads=3, pattern_workers=None):
    """Verify existing kernels before building factors; never compile."""
    install()
    from structured_programme_source_candidate.runtime_backend_contract import restore_native_backend_contract
    return restore_native_backend_contract(stage=stage, native_threads=native_threads, pattern_workers=pattern_workers)


def solve_joint_source(*args, **kwargs):
    """Reuse the original source LM/Schur interface with caller-owned factors."""
    install()
    from z190_joint_source_solver import solve_joint_source as implementation
    return implementation(*args, **kwargs)


__all__ = ["configure_native_runtime", "install", "solve_joint_source", "source_path", "status"]
