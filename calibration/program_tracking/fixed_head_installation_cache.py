"""Explicit same-value reuse of a fixed head's current installation.

The original installation method still computes every value and derivative.
Only repeated calls at the exact same head chart and shared axes are reused.
Returned arrays remain independent writable copies, as in the original API.
This module does not install itself or alter source data, camera equations,
parameter domains, or solver decisions. Select it at an explicit phase boundary.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np


_head_class = None
_original = None
_current_key = None
_current_value = None
_receipt = None
_calls = 0
_computations = 0


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _copy(value):
    return {key: item.copy() if isinstance(item, np.ndarray) else deepcopy(item)
            for key, item in value.items()}


def _key(head, shared_axes):
    axes = np.zeros(4) if shared_axes is None else np.asarray(shared_axes, dtype=float)
    if axes.shape != (4,) or not np.isfinite(axes).all():
        return None  # Delegate invalid input to the original validation/errors.
    # Include every head chart input used by the original installation method.
    chart = (head.pan_axis_world, head.tilt_axis_camera,
             head._pan_basis, head._tilt_basis,
             head._pan_reference, head._tilt_reference)
    return tuple(np.asarray(value, dtype=float).tobytes() for value in chart) + (axes.tobytes(),)


def _installation(head, shared_axes=None):
    global _current_key, _current_value, _calls, _computations
    _calls += 1
    key = _key(head, shared_axes)
    if key is None:
        _computations += 1
        return _original(head, shared_axes)
    if key != _current_key:
        value = _original(head, shared_axes)
        _computations += 1
        _current_key, _current_value = key, _copy(value)
    return _copy(_current_value)


def install(*, expected_math_reference, head_class=None):
    """Install explicitly in the parent or a declared source worker task."""
    global _head_class, _original, _current_key, _current_value, _receipt
    global _calls, _computations
    if head_class is None:
        from .fixed_pan_tilt import FixedPanTilt
        head_class = FixedPanTilt
    if _head_class is not None:
        if (head_class is not _head_class or head_class.installation is not _installation
                or _receipt['original_math'] != expected_math_reference):
            raise ValueError('A different fixed-head installation execution is already active')
        return deepcopy(_receipt)
    original = head_class.installation
    actual = reference(original.__code__.co_filename)
    if actual != expected_math_reference or original.__name__ != 'installation':
        raise ValueError('The declared original fixed-head installation producer changed')
    _head_class, _original = head_class, original
    _current_key = _current_value = None
    _calls = _computations = 0
    _receipt = dict(schema='rinkspace.fixed-head-current-installation-same-value-execution.v1',
        producer=reference(__file__), original_math=actual,
        original_values_and_all_shared_axis_derivatives_used=True,
        exact_head_chart_and_shared_axis_bytes_key=True,
        only_current_installation_retained=True,
        returned_arrays_are_independent_writable_copies=True,
        invalid_inputs_delegate_to_original_validation=True,
        source_rows_weights_camera_equations_domains_and_solver_unchanged=True,
        requires_actual_source_value_and_J_comparison_before_phase_use=True)
    head_class.installation = _installation
    return deepcopy(_receipt)


def statistics():
    return dict(calls=_calls, original_computations=_computations,
                reused_calls=_calls - _computations,
                current_installation_present=_current_key is not None)


def uninstall():
    """Restore the original method in independent checks or a closed phase."""
    global _head_class, _original, _current_key, _current_value, _receipt
    if _head_class is None:
        raise ValueError('Fixed-head installation execution is not active')
    if _head_class.installation is not _installation:
        raise ValueError('Fixed-head installation execution changed after installation')
    result = statistics()
    _head_class.installation = _original
    _head_class = _original = _current_key = _current_value = _receipt = None
    return result
