"""Direct GN preserves bounded TRF behavior and restores SciPy hooks."""
from types import SimpleNamespace
import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import least_squares
import scipy.optimize._lsq.trf as trf
from calibration.program_tracking.body_tracking.body_linear_solver import sparse_direct_trf

def test_bounded_solution_agrees_with_lsmr_and_restores_hooks():
    rng=np.random.default_rng(1105);A=rng.normal(size=(90,12));b=rng.normal(size=90);J=csr_matrix(A)
    args=dict(fun=lambda x:A@x-b,x0=np.zeros(12),jac=lambda x:J,bounds=(-.1,.12),method='trf',tr_solver='lsmr',x_scale='jac',ftol=1e-10,xtol=1e-10,gtol=1e-10,tr_options=dict(atol=1e-9,btol=1e-9))
    old=least_squares(**args);saved=(trf.right_multiplied_operator,trf.regularized_lsq_operator,trf.lsmr);owner=SimpleNamespace(linear_solver_backend='sparse_direct')
    with sparse_direct_trf(owner):new=least_squares(**args)
    assert (trf.right_multiplied_operator,trf.regularized_lsq_operator,trf.lsmr)==saved
    np.testing.assert_allclose(new.x,old.x,atol=2e-6,rtol=2e-5);assert abs(new.cost-old.cost)<1e-8
    assert owner.linear_solver_statistics and not any(r['fallback'] for r in owner.linear_solver_statistics)
