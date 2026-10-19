"""Physical source modes survive; arithmetic null modes never become precision."""
import numpy as np
import pytest

from calibration.program_tracking.first_shot_source_evidence import _board_curve
from calibration.program_tracking.source_evidence_whitening import latent_whitening_operator


def normal_basis(normals):
    basis = np.zeros((len(normals), 2*len(normals)))
    for i, normal in enumerate(normals):
        basis[i, 2*i:2*i+2] = normal
    return basis


def test_source_product_roundoff_cannot_create_an_overconfident_null_mode():
    # The factor's final bit is an upstream rounding error: its cancellation
    # is amplified by B. A rank tolerance based only on the small B@L output
    # would incorrectly turn this 1e-5 mode into a huge whitening weight.
    sensitivity = np.array([[1e12, 1e12, 1.], [1e12, 1e12, 2.]])
    factor = np.array([[1., 1.], [-1., -1.+np.finfo(float).eps], [1., 2.]])
    product = sensitivity @ factor
    singular = np.linalg.svd(product, compute_uv=False)
    naive = np.finfo(float).eps*max(product.shape)*singular[0]
    assert np.count_nonzero(singular > naive) == 2
    basis = normal_basis([[0., 1.], [0., 1.]])
    operator = latent_whitening_operator(["a", "b"], sensitivity,
        explicit_noise_factor=factor, physical_basis=basis,
        group_id="actual_original_readout_factor", covariance_origin="saved_native_noise")
    assert operator.rank == 1
    assert operator.consumed_point_count == 2
    assert np.all(np.linalg.norm(operator.matrix.reshape(1, 2, 2), axis=(0, 2)) > 0)
    audit = operator.numerical_rank_evidence
    assert audit["multiplication_backward_error_bound"] > singular[1]
    assert audit["source_point_membership_reduced"] is False


@pytest.mark.parametrize("scale", [0.001, 7., 1000.])
def test_native_sigma_scale_changes_precision_without_changing_source_rank(scale):
    sensitivity = np.array([[1., -.25], [1., .5], [1., 1.5]])
    factor = np.array([[.2, .1, -.1], [.1, -.15, .25]])
    basis = normal_basis([[.6, .8], [0., 1.], [-.8, .6]])
    arguments = dict(physical_basis=basis, group_id="same_source", covariance_origin="native_IQR")
    a = latent_whitening_operator(["a", "b", "c"], sensitivity,
        explicit_noise_factor=factor, **arguments)
    b = latent_whitening_operator(["a", "b", "c"], sensitivity,
        explicit_noise_factor=factor*scale, **arguments)
    assert a.rank == b.rank == 2
    np.testing.assert_allclose(b.matrix.T @ b.matrix*scale**2,
        a.matrix.T @ a.matrix, rtol=1e-12, atol=1e-12)
    assert b.numerical_rank_evidence["numerical_rank_threshold"] == pytest.approx(
        a.numerical_rank_evidence["numerical_rank_threshold"]*scale)


def test_nearly_horizontal_quadratic_does_not_condition_source_X_as_noiseless():
    x = np.linspace(0., 900., 20)
    native = np.column_stack((x, 100.+.0001*x+.0000002*x*x))
    indices = [0, 4, 9, 14, 19]
    segment = dict(uv=native[indices].tolist(), source_native_readout_uv=native.tolist(),
        source_native_transition_iqr_px=[2.]*len(native), source_curve_degree=2,
        representative_points_share_line_estimate=True,
        measurement_origin="current_native_joint_smooth_material_curve_fit")
    curve = _board_curve("real_source_quadratic_chart", segment, {})
    normals = [curve.normal(point) for point in native[indices]]
    sensitivity = np.stack([normal @ curve.sensitivity(point)
        for normal, point in zip(normals, native[indices], strict=True)])
    operator = latent_whitening_operator([str(i) for i in indices], sensitivity,
        explicit_noise_factor=curve.noise_factor, physical_basis=normal_basis(normals),
        group_id="five_native_representatives", covariance_origin="actual_transition_IQR")
    assert operator.rank == 4  # Keep the genuine fourth physical shape mode.
    assert np.max(np.abs(operator.matrix)) < 10.
    reasonable_normal_error = np.concatenate([.25*normal for normal in normals])
    assert np.dot(operator.apply(reasonable_normal_error),
                  operator.apply(reasonable_normal_error)) < 10.
    assert curve.metadata["source_coordinate_fit_conditioning_not_interpreted_as_source_precision"]
