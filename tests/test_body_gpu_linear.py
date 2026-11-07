"""Reject invalid initial residuals before Jacobian/GPU work; CPU only."""
import numpy as np
import pytest
import scipy.optimize._lsq.trf as trf

from calibration.program_tracking.body_tracking.body_gpu_linear import gpu_trf


@pytest.mark.parametrize('bad', [np.inf, np.nan], ids=['inf', 'nan'])
def test_nonfinite_initial_residual_is_rejected_before_jacobian_or_trf(monkeypatch, bad):
    calls = []
    saved_scale = trf.compute_jac_scale

    def fun(x):
        calls.append('fun')
        return np.array([0., bad])

    def jac(x):
        pytest.fail('Invalid initial residual reached Jacobian/GPU preparation')

    def solve(*args, **kwargs):
        pytest.fail('Invalid initial residual reached TRF/GPU execution')

    monkeypatch.setattr(trf, 'trf', solve)
    with pytest.raises(ArithmeticError, match='Initial state.*nonfinite residual'):
        gpu_trf(fun, jac, np.zeros(1), np.full(1, -1.), np.ones(1),
                callback=None, options={})
    assert calls == ['fun']
    assert trf.compute_jac_scale is saved_scale
