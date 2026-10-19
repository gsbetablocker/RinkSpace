"""Covariance marginalization keeps retained source uncertainty, not old W columns."""
import numpy as np
from rinkfit_start.source_audit_revision import covariance_marginal


def test_correlated_source_marginal_is_not_conditional_column_slice():
    covariance=np.array([[4.,2.],[2.,9.]])
    W=np.linalg.inv(np.linalg.cholesky(covariance))
    new,factor,proof=covariance_marginal(W,np.array([True,False]),1e-12)
    # Marginal variance of x is four.  Dropping y from the old whitened
    # residual would instead condition y=0, retaining spurious information.
    assert np.allclose(new.T@new,[[.25]],atol=1e-14)
    assert not np.allclose(W[:,:1].T@W[:,:1],new.T@new)
    assert proof['retained_rank']==1
    assert np.allclose(factor@factor.T,[[4.]],atol=1e-13)


def test_rank_deficient_source_marginal_drops_only_lost_noise_mode():
    W=np.array([[1.,0.,0.],[0.,.5,0.]])
    new,factor,proof=covariance_marginal(W,np.array([False,True,True]),1e-12)
    assert proof['retained_rank']==1
    assert new.shape==(1,2)
    assert np.allclose(new.T@new,[[.25,0.],[0.,0.]],atol=1e-14)
    assert np.allclose(new@factor@factor.T@new.T,np.eye(1),atol=1e-14)
