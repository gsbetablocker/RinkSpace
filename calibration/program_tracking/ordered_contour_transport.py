"""Safely transport the known BoardIntersectionError source-domain type.

The historical error stores formatted args that its own constructor cannot
unpickle. Its public and legacy evaluator aliases are the same domain contract;
convert their instances at the worker task boundary without editing producers.
Other exception types keep their original failure semantics.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def _board_error_types():
    from .first_shot_source_kernel import _interfaces
    from .reference_board_pair import BoardIntersectionError
    # The original absolute evaluator imports the historical bare module;
    # direct public callers can hold the package-qualified class instead.
    return tuple(set((_interfaces()[3], BoardIntersectionError)))


def source_chunk(task, *, original_sampler):
    try:
        result = original_sampler(task)
    except ValueError as error:
        if not isinstance(error, _board_error_types()):
            raise
        failure = ValueError(str(error))
        failure.reason = str(error.reason)
        failure.indices = np.asarray(error.indices, np.int32).copy()
        failure.original_exception_type = type(error).__name__
        failure.original_exception_module = type(error).__module__
        failure.worker_domain_rejection = True
        for note in getattr(error, "__notes__", ()):
            failure.add_note(str(note))
        raise failure from None
    path = Path(__file__).resolve()
    receipt = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        transported_domain_type="BoardIntersectionError", all_reasons=True,
        other_exception_types_changed=False)
    if not isinstance(result, tuple) or not isinstance(result[-1], dict):
        raise ValueError("Original source worker final receipt dictionary required")
    return result[:-1]+(dict(result[-1], board_domain_error_transport=receipt),)
