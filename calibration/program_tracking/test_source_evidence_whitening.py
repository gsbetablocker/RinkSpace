"""Evidence propagation tests; never run a camera or scene optimization."""
import numpy as np
import pytest

from calibration.program_tracking.source_evidence_whitening import (
    MissingSourceEvidence, SourceLineEvidence, intersection_coordinates_and_jacobian,
    joint_line_corner_operator, latent_whitening_operator, line_evidence_from_readouts,
    line_whitening_operator,
)


def _line(axis=0, coefficients=(.08,12.), group="line"):
    x = np.array([0., 50., 100., 150.])
    y = coefficients[0]*x+coefficients[1]
    uv = np.column_stack((x,y)) if axis == 0 else np.column_stack((y,x))
    return line_evidence_from_readouts(uv, [2.,3.,2.5,4.], coefficients,
        axis=axis, group_id=group, estimator="weighted_linear")


def test_all_line_representatives_consume_rank_two_with_density_invariant_information():
    line = _line()
    coefficient_error = np.array([.005,.7])
    expected = coefficient_error @ np.linalg.solve(line.covariance_ab,coefficient_error)
    for count in (2,7,41):
        x = np.linspace(0,150,count)
        uv = np.column_stack((x,line.coefficients[0]*x+line.coefficients[1]))
        before = uv.copy()
        operator = line_whitening_operator(uv,[f"point/{i}" for i in range(count)],line)
        n = line.normal
        displacement = (np.column_stack((x,np.ones(count))) @ coefficient_error)*n[1]
        raw = (displacement[:,None]*n).ravel()
        assert operator.rank == 2
        assert operator.consumed_point_count == count
        np.testing.assert_array_equal(uv,before)
        assert np.linalg.norm(operator.apply(raw))**2 == pytest.approx(expected,rel=1e-10)
        # The measured source normal uses rx as well as ry.
        assert np.linalg.norm(operator.matrix[:,0::2]) > 0


def test_low_rank_covariance_is_whitened_and_jacobian_uses_same_operator():
    sensitivity = np.array([[1.,0.],[0.,1.],[2.,0.],[0.,3.]])
    covariance = np.array([[2.,.4],[.4,4.]])
    operator = latent_whitening_operator(["a","b"],sensitivity,covariance,
        group_id="native_group",covariance_origin="test_explicit_covariance")
    np.testing.assert_allclose(operator.matrix @ sensitivity @ covariance @ sensitivity.T @ operator.matrix.T,np.eye(2),atol=1e-12)
    residual,jacobian = operator.apply(np.arange(4.),np.arange(12.).reshape(4,3))
    np.testing.assert_allclose(residual,operator.matrix@np.arange(4.))
    np.testing.assert_allclose(jacobian,operator.matrix@np.arange(12.).reshape(4,3))
    assert operator.metadata()["source_point_ids"] == ["a","b"]


def test_intersection_analytic_jacobian_matches_source_coefficient_perturbation():
    h = _line(group="horizontal")
    v = _line(axis=1,coefficients=(.02,40.),group="vertical")
    xy,jacobian = intersection_coordinates_and_jacobian(h,v)
    coefficients = np.r_[h.coefficients,v.coefficients]
    finite = np.empty((2,4))
    for i in range(4):
        step = 1e-6
        high,low = coefficients.copy(),coefficients.copy()
        high[i] += step;low[i] -= step
        def intersection(c):
            hh = SourceLineEvidence(c[:2],h.covariance_ab,0,"h","explicit")
            vv = SourceLineEvidence(c[2:],v.covariance_ab,1,"v","explicit")
            return intersection_coordinates_and_jacobian(hh,vv)[0]
        finite[:,i] = (intersection(high)-intersection(low))/(2*step)
    np.testing.assert_allclose(jacobian,finite,rtol=1e-7,atol=1e-7)
    np.testing.assert_allclose(xy[1],h.coefficients[0]*xy[0]+h.coefficients[1])
    np.testing.assert_allclose(xy[0],v.coefficients[0]*xy[1]+v.coefficients[1])


def test_derived_corner_shares_latent_readouts_without_adding_information():
    h,v = _line(group="h"),_line(axis=1,coefficients=(.02,40.),group="v")
    lines = {"h":h,"v":v}
    members=[];raw=[];noise=np.arange(8.)*.05
    for key,line in lines.items():
        for i,rid in enumerate(line.readout_ids):
            direction = [0.,1.] if line.axis == 0 else [1.,0.]
            members.append(dict(point_id=rid,uv=[i*50.,10.],kind="direct_readout",
                readout_id=rid,displacement_xy_per_unit_noise=direction))
            raw.extend(np.asarray(direction)*noise[len(raw)//2])
    direct = joint_line_corner_operator(members,lines,group_id="source_edges")
    _,derivative = intersection_coordinates_and_jacobian(h,v)
    corner_noise = derivative @ np.r_[h.coefficient_influence@noise[:4],v.coefficient_influence@noise[4:]]
    xy,_ = intersection_coordinates_and_jacobian(h,v)
    members.append(dict(point_id="unchanged_original_corner_id",uv=xy.tolist(),kind="derived_intersection",
                        horizontal_line_id="h",vertical_line_id="v"))
    joint = joint_line_corner_operator(members,lines,group_id="same_source_edges_and_corner")
    assert joint.rank == direct.rank == 8
    assert joint.consumed_point_count == 9
    assert np.linalg.norm(joint.apply(np.r_[raw,corner_noise]))**2 == pytest.approx(np.linalg.norm(direct.apply(raw))**2,rel=1e-10)


def test_missing_scales_or_decorative_member_cannot_become_fit():
    with pytest.raises(MissingSourceEvidence):
        line_evidence_from_readouts([[0.,0.],[1.,1.]],[0.,2.],[1.,0.],
            axis=0,group_id="bad",estimator="weighted_linear")
    with pytest.raises(MissingSourceEvidence):
        line_evidence_from_readouts([[0.,0.],[1.,1.]],[2.,2.],[1.,0.],
            axis=0,group_id="bad",estimator="weighted_soft_l1")
    with pytest.raises(ValueError,match="decorative"):
        latent_whitening_operator(["a","b"],[[1.],[1.],[0.],[0.]],[[2.]],
            group_id="bad",covariance_origin="explicit")


def test_soft_l1_covariance_uses_original_nonunit_source_scales():
    uv=np.array([[0.,2.],[10.,3.],[20.,4.],[30.,5.]])
    ordinary=line_evidence_from_readouts(uv,[2.,3.,4.,5.],[.1,2.],
        axis=0,group_id="ordinary",estimator="weighted_linear")
    robust=line_evidence_from_readouts(uv,[2.,3.,4.,5.],[.1,2.],
        axis=0,group_id="robust",estimator="weighted_soft_l1",soft_l1_f_scale=1.)
    np.testing.assert_allclose(robust.covariance_ab,ordinary.covariance_ab)
    scaled=line_evidence_from_readouts(uv,[4.,6.,8.,10.],[.1,2.],
        axis=0,group_id="scaled",estimator="weighted_linear")
    np.testing.assert_allclose(scaled.covariance_ab,4*ordinary.covariance_ab)
