"""The bounded CSR pack preserves every actual weighted residual/column."""
import numpy as np
from scipy.sparse import csr_matrix,vstack,diags
from calibration.program_tracking.body_tracking.weighted_blocks import block,full_columns,pack_weighted,verify


def test_packed_blocks_equal_independent_stack_and_share_final_storage():
    rng=np.random.default_rng(1106)
    blocks=[full_columns(block('body',rng.normal(size=11),csr_matrix(rng.normal(size=(11,7))))),
            full_columns(block('empty',np.empty(0),csr_matrix((0,7)))),
            full_columns(block('physics',rng.normal(size=9),csr_matrix(rng.normal(size=(9,7)))))]
    weights={'body':np.linspace(.025,.45,11),'physics':5.}
    expected_r=np.concatenate([b.residual*np.sqrt(weights.get(b.name,1.)) for b in blocks])
    expected_J=vstack([diags(np.sqrt(np.broadcast_to(weights.get(b.name,1.),b.residual.shape)))@b.jacobian for b in blocks],format='csr')
    r,J,named=pack_weighted(blocks,weights,True)
    np.testing.assert_array_equal(r,expected_r)
    np.testing.assert_allclose(J.toarray(),expected_J.toarray(),rtol=0,atol=0)
    assert verify(named,r,J)['actual_jacobian_checked']
    assert all(np.shares_memory(b.jacobian.data,J.data)for b in named if b.jacobian.nnz)
