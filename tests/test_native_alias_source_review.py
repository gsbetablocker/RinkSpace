import numpy as np
import pytest
import hashlib

from rinkspace.native_alias_source_review import compare_material_candidates
from rinkspace.native_alias_source_review import check_native_alias_paint_precision


def measurement(sides, *, operator='paint_center'):
    sides = np.asarray(sides, float).reshape(-1, 2, 2)
    quantiles = np.repeat(sides[:, :, None], 3, axis=2)
    quantiles[:, :, 0, 0] -= .25
    quantiles[:, :, 2, 0] += .25
    return dict(query_curve=np.array([0]), query_uv=np.array([[0., 0.]], np.float32),
        query_normal=np.array([[1., 0.]], np.float32),
        query_interval=np.array([[-128., 128.]], np.float32),
        candidate_query=np.zeros(len(sides), np.int32), candidate_sides=sides,
        candidate_uv=sides.mean(1), candidate_flags=np.zeros(len(sides), np.uint8),
        candidate_side_quantiles_uv=quantiles, source_readout_quantization_step_px=1.)


def test_matching_center_cannot_hide_changed_material_sides():
    canonical = measurement([[[10., 0.], [50., 0.]]])
    alias = measurement([[[20., 0.], [40., 0.]]])
    result = compare_material_candidates(canonical, alias, [{'operator':'paint_center'}])
    assert result['canonical_candidate_status'].tolist() == [3]
    assert np.isnan(result['measured_native_uv']).all()


def test_alias_outputs_its_own_current_interfaces_and_rejects_competition():
    canonical = measurement([[[10., 0.], [50., 0.]]])
    alias = measurement([[[10.75, 0.], [50.75, 0.]]])
    result = compare_material_candidates(canonical, alias, [{'operator':'paint_center'}])
    assert result['canonical_candidate_status'].tolist() == [1]
    np.testing.assert_array_equal(result['measured_native_sides_uv'], alias['candidate_sides'])
    np.testing.assert_array_equal(result['measured_native_uv'], [[30.75, 0.]])
    ambiguous = measurement([[[10.75, 0.], [50.75, 0.]], [[9.5, 0.], [49.5, 0.]]])
    result = compare_material_candidates(canonical, ambiguous, [{'operator':'paint_center'}])
    assert result['canonical_candidate_status'].tolist() == [4]


def test_unknown_role_and_changed_station_never_gain_alias_admission():
    canonical = measurement([[[10., 0.], [50., 0.]]])
    alias = measurement([[[10., 0.], [50., 0.]]])
    result = compare_material_candidates(canonical, alias, [{'operator':'unresolved'}])
    assert result['canonical_candidate_status'].tolist() == [0]
    alias['query_uv'][0, 0] = 4.
    with pytest.raises(ValueError, match='search stations'):
        compare_material_candidates(canonical, alias, [{'operator':'paint_center'}])


def test_alias_colour_support_cannot_be_inherited_from_matching_canonical_sides():
    canonical=measurement([[[6.,6.],[6.,8.]]])
    current=measurement([[[6.,6.],[6.,8.]]])
    inventory=[dict(operator='paint_center',feature_id='C',curve_id='C:segment0',material_signal='red')]
    checked=compare_material_candidates(canonical,current,inventory)
    rgb=np.full((16,16,3),100,np.uint8)
    def bind():
        current['decoded_native_RGB_sha256']=hashlib.sha256(rgb.tobytes()).hexdigest()
        current['source_binding']=dict(native_pts=110,actual_pts=100,
            decoded_native_BGR_sha256=hashlib.sha256(rgb[:,:,::-1].tobytes()).hexdigest())
    bind()
    failed=check_native_alias_paint_precision(checked,current,rgb,inventory)
    assert failed['canonical_candidate_status'].tolist()==[3]
    assert np.isnan(failed['measured_native_uv']).all()
    assert checked['canonical_candidate_status'].tolist()==[1]
    rgb[6:9,:,0]=140
    bind()
    passed=check_native_alias_paint_precision(checked,current,rgb,inventory)
    assert passed['canonical_candidate_status'].tolist()==[1]
    np.testing.assert_array_equal(passed['measured_native_uv'],checked['measured_native_uv'])
    assert np.array_equal(current['candidate_flags'],[0])
