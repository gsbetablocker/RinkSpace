"""Exact repeated RHS reuse and prepared CUDA CSR transpose lifetime."""
import numpy as np
import pytest
from scipy.sparse import csr_matrix

def test_prepared_transpose_and_same_rhs_reuse_do_not_change_normal_model(monkeypatch):
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available():pytest.skip('CUDA hardware check')
    from calibration.program_tracking.body_tracking.body_gpu_linear import GPUJacobian
    dense=np.array([[1.,0,2],[0,3,1],[1,-2,0],[.5,0,.25],[0,1,-1]])
    op=GPUJacobian(csr_matrix(dense));op._prepare_gram()
    assert op._normal_transpose is not None
    calls=[];original=torch.sparse.mm
    def counted(*args,**kwargs):
        calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(torch.sparse,'mm',counted)
    rhs=np.arange(1.,6.)
    first=op.rmatvec(rhs);np.testing.assert_allclose(first,dense.T@rhs,rtol=1e-13,atol=1e-13)
    assert op._normal_transpose is None
    count=len(calls);second=op.rmatvec(rhs.copy());assert len(calls)==count
    np.testing.assert_array_equal(first,second)
    second[0]=1e10
    np.testing.assert_array_equal(op.rmatvec(rhs),first)
    rhs[0]+=.5
    changed=op.rmatvec(rhs);assert len(calls)==count+1
    np.testing.assert_allclose(changed,dense.T@rhs,rtol=1e-13,atol=1e-13)
    objective=op.scaled_normal_objective(rhs,np.ones(3),damp=.07)
    count=len(calls);z=np.array([.2,-.1,.3]);value,gradient=objective(z)
    assert len(calls)==count+1
    np.testing.assert_allclose(value,.5*np.sum((dense@z-rhs)**2)-.5*np.sum(rhs**2)+.5*.07**2*np.sum(z**2),rtol=1e-13,atol=1e-13)
    np.testing.assert_allclose(gradient,dense.T@(dense@z-rhs)+.07**2*z,rtol=1e-13,atol=1e-13)
    op.release();assert op._transpose_argument is None and op._transpose_result is None
    with pytest.raises(RuntimeError,match='Retired'):op.rmatvec(rhs)


@pytest.mark.parametrize('retain_inactive_slots',[True,False])
def test_active_column_window_preserves_products_and_reuses_slots(monkeypatch,retain_inactive_slots):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA hardware check')
    from calibration.program_tracking.body_tracking.body_gpu_linear import DeviceCSRLayout, GPUJacobian

    monkeypatch.setattr(DeviceCSRLayout, 'row_batch', 2)
    layout = DeviceCSRLayout(retain_inactive_slots=retain_inactive_slots)
    dense = np.array([[0., 1., 0., 8.], [0., 0., 0., 9.],
                      [0., 3., 4., 0.], [0., 0., 2., 0.], [0., 0., 0., 0.]])
    changed = dense.copy()
    changed[1, 2] = 2.
    changed[2, 1] = 0.
    rhs = np.arange(1., 6.)
    for index, full in enumerate((dense, dense * 2., changed, dense)):
        expected = full[:, 1:3]
        operator = GPUJacobian(csr_matrix(full), shared_prefix=1, column_stop=3, layout=layout)
        np.testing.assert_allclose(operator @ np.array([2., 3.]), expected @ [2., 3.])
        np.testing.assert_allclose(operator.rmatvec(rhs), expected.T @ rhs)
        np.testing.assert_allclose(operator.scale_inv, np.linalg.norm(expected, axis=0))
        assert operator.transfer['inactive_entries_skipped'] == 2
        assert operator.transfer['host_slice_value_bytes_avoided'] == np.count_nonzero(expected) * 8
        assert operator.transfer['jacobian_structure_reused'] == (index == 1 or (retain_inactive_slots and index == 3))
        operator.release()
