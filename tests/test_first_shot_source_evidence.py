"""Source-only support includes every accepted point without moving its UV."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from calibration.program_tracking import first_shot_source_evidence as evidence
from calibration.program_tracking import official_static_surface_dataset as scope
from calibration.program_tracking.source_evidence_whitening import (
    MissingSourceEvidence, SourceWhiteningOperator, latent_whitening_operator,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "events/ankara_2026/programmes/okada_mei_sp/source_datasets/frame_by_frame_review_20261013/first_shot_remaining_50fps/repair_20261014"
REGISTRY = ROOT / "events/ankara_2026/models/framewise_8a28_first_shot_20261014/repair_20261014/feature_registry.json"


@pytest.fixture
def accepted_frame_201():
    packet_path, binding_path = SOURCE / "frames/000201.json", SOURCE / "bindings/000201.json"
    if not all(path.is_file() for path in (packet_path, binding_path, REGISTRY)):
        pytest.skip("The accepted native frame 201 source fixture is unavailable")
    packet_ref, binding_ref = scope.reference(packet_path), scope.reference(binding_path)
    packet, binding = scope.read_reference(packet_ref), scope.read_reference(binding_ref)
    registry = scope.read_reference(scope.reference(REGISTRY))["targets"]
    features = {feature["id"]: feature for feature in packet["features"]}
    rows = []
    for observation in binding["observations"]:
        feature = features[observation["source_feature_id"]]
        segment = feature["segments"][observation["source_segment_index"]]
        indices = observation["source_vertex_indices"]
        scales, _ = scope.source_scales(segment)
        rows.append(dict(observation, source_segment=segment, source_feature=feature,
            original_vertex_indices=indices, uv=np.asarray(segment["uv"], float)[indices],
            sigma=scales[indices], model_target=registry[observation["target_id"]]))
    native = packet["source"]
    frame = dict(index=201, native_pts=native["native_pts"], packet=packet_ref,
        source_image=scope.reference(native["image_path"]),
        mask=dict(path=native["mask_path"], sha256=native["mask_sha256"]))
    return rows, frame, (packet_ref, binding_ref)


def test_real_native_201_includes_all_edges_corners_lines_and_circles(accepted_frame_201):
    rows, frame, originals = accepted_frame_201
    original_uv = [row["uv"].copy() for row in rows]
    original_bytes = {ref["path"]: Path(ref["path"]).read_bytes() for ref in originals}
    result = evidence.prepare_frame_evidence(rows, frame)
    ids = [point_id for row in rows for point_id in row["source_point_ids"]]
    assert len(ids) == len(set(ids)) == result["point_count"] == 189
    assert set(result["source_scales_by_point_id"]) == set(ids)
    scales = np.asarray(list(result["source_scales_by_point_id"].values()))
    assert np.isfinite(scales).all() and (scales > 0).all()
    assert not np.allclose(scales, 1.)
    assert result["correlated_point_count"] == 158
    assert result["ordinary_point_count"] == 31
    assert len(result["whitening_groups"]) == 13
    consumed = set()
    for group in result["whitening_groups"]:
        operator = SourceWhiteningOperator.from_record(group["operator"])
        indices = group["point_indices"]
        assert tuple(ids[index] for index in indices) == operator.source_point_ids
        assert consumed.isdisjoint(indices)
        consumed.update(indices)
        assert np.isfinite(operator.matrix).all() and operator.rank > 0
        for i in range(operator.consumed_point_count):
            assert np.any(operator.matrix[:, 2*i:2*i+2] != 0)
    assert len(consumed) == result["correlated_point_count"]
    targets = [row["model_target"] for row in rows]
    assert any(target["kind"] == "ad_corner" for target in targets)
    assert any(row["source_feature"].get("source_line_fit") for row in rows)
    assert any(row["source_segment"].get("representative_points_share_circle_family_estimate") for row in rows)
    assert result["source_UV_modified"] is False
    assert result["supplementary_readouts_added_as_FIT_points"] == 0
    for row, uv in zip(rows, original_uv, strict=True):
        np.testing.assert_array_equal(row["uv"], uv)
    for path, original in original_bytes.items():
        assert Path(path).read_bytes() == original


def quadratic_segment(width=2.):
    x = np.linspace(0., 9., 10)
    native = np.column_stack((x, 10. + 0.02*x*x))
    return dict(uv=native[[0, 4, 9]].tolist(),
        representative_points_share_line_estimate=True,
        source_native_readout_uv=native.tolist(), source_native_transition_iqr_px=[width] * len(native),
        source_curve_degree=2, measurement_origin="synthetic_saved_native_quadratic")


def test_native_quadratic_support_has_finite_covariance_and_uses_measured_widths():
    segment = quadratic_segment()
    before = deepcopy(segment)
    curve = evidence._board_curve("synthetic.board.material", segment, {"path": "synthetic native readouts"})
    assert curve.covariance.shape == (6, 6)
    assert np.isfinite(curve.covariance).all()
    np.testing.assert_allclose(curve.covariance, curve.covariance.T, atol=1e-12)
    assert np.linalg.eigvalsh(curve.covariance).min() >= -1e-10
    sensitivity = np.vstack([curve.sensitivity(uv) for uv in segment["uv"]])
    operator = latent_whitening_operator(["left", "middle", "right"], sensitivity, curve.covariance,
        group_id="synthetic_native_material", covariance_origin="recorded_actual_transition_iqr")
    assert 2 <= operator.rank <= 6
    assert operator.consumed_point_count == 3
    assert not np.allclose(curve.covariance, np.eye(6))
    wider = evidence._board_curve("synthetic.board.material", quadratic_segment(4.), {})
    assert np.trace(wider.covariance) / np.trace(curve.covariance) == pytest.approx(4., rel=0.02)
    assert segment == before
    assert curve.metadata["source_sigma_origin"] == "source_native_transition_iqr_px/1.349"
    assert curve.metadata["accepted_UV_modified"] is False
    assert curve.metadata["camera_or_scene_fit_used"] is False


@pytest.mark.parametrize("points", [[], [[1., 2.]], [[1., 2.], [1., 2.]], [[1., 2.], [1., 2.], [1., 2.]]])
def test_no_tangent_can_be_invented_for_missing_or_repeated_native_stations(points):
    with pytest.raises(MissingSourceEvidence):
        evidence._normal(points)


def test_native_quadratic_rejects_missing_uncertainty_instead_of_a_default_scale():
    segment = quadratic_segment()
    segment["source_native_transition_iqr_px"] = []
    with pytest.raises(MissingSourceEvidence):
        evidence._board_curve("synthetic.board.material", segment, {})


@pytest.mark.parametrize("neighbour_depth", [30., 60., 100.])
def test_human_thin_stroke_support_cannot_borrow_a_stronger_neighbouring_band(neighbour_depth):
    offsets = np.arange(-20., 21.)
    triangle = lambda center, width: np.maximum(0., 1.-np.abs(offsets-center)/width)
    # The accepted stroke is centered at zero. Darker unrelated material on
    # both sides cannot become its valley, flank, or background shoulder.
    profile = 200.-12.*triangle(0., 4.)-30.*triangle(-14., 4.)-neighbour_depth*triangle(13., 3.)
    point = np.array([100., 500.]); normal = np.array([0., 1.])
    result = evidence._manual_local_paint_evidence(point, normal, offsets, profile,
        np.ones(len(offsets), bool))
    native = result["source_evidence"]
    np.testing.assert_allclose(native["actual_source_valley_uv"], point)
    np.testing.assert_allclose(native["actual_source_sides_uv"], [[100., 498.], [100., 502.]])
    np.testing.assert_allclose(native["actual_source_own_shoulders_uv"], [[100., 496.], [100., 504.]])
    assert result["sigma_px"] == pytest.approx(2./1.349)
    assert native["sigma_floor_applied"] is False


@pytest.mark.parametrize("frame_index", [1241, 1242])
def test_actual_manual_C0_points_use_their_own_thin_contour_without_moving_clicks(frame_index):
    path = SOURCE / f"frames/{frame_index:06d}.json"
    if not path.is_file():
        pytest.skip("The accepted native manual C0 source fixture is unavailable")
    original_bytes = path.read_bytes()
    packet = scope.read_reference(scope.reference(path))
    source_packet = scope.read_reference(packet["ice_source_readout_packet"])
    source_features = {feature["id"]: feature for feature in source_packet["features"]}
    native = packet["source"]
    frame = dict(source_image=scope.reference(native["image_path"]),
        mask=dict(path=native["mask_path"], sha256=native["mask_sha256"]))
    manual = [f for f in packet["features"] if f.get("manual_source_observation")]
    assert len(manual) == 2
    for feature in manual:
        segment = feature["segments"][0]
        uv = np.asarray(segment["uv"], float)
        row = dict(target_id=feature["target_id"], uv=uv,
                   source_point_ids=segment["point_ids"])
        before_uv = uv.copy(); before_ids = list(row["source_point_ids"])
        sigma, result = evidence._manual_support(row, 0, packet, source_features, frame)
        support = result["native_material_evidence"]
        quantiles = np.asarray(support["actual_source_side_quantiles_uv"])
        assert sigma == pytest.approx(np.max(np.linalg.norm(quantiles[:, 2]-quantiles[:, 0], axis=1))/1.349)
        assert np.max(np.linalg.norm(np.asarray(support["actual_source_sides_uv"])-uv[0], axis=1)) < 8.
        assert np.linalg.norm(np.asarray(support["actual_source_valley_uv"])-uv[0]) <= .5+1e-9
        assert support["sigma_floor_applied"] is False
        assert result["accepted_UV_modified"] is False
        np.testing.assert_array_equal(row["uv"], before_uv)
        assert row["source_point_ids"] == before_ids
    assert path.read_bytes() == original_bytes


@pytest.fixture(scope='module')
def confirmed_identity_sources():
    data_path=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp/source_datasets/first_shot_fit_20261014/numerically_valid_fit.json')
    fix_path=ROOT/'events/ankara_2026/programmes/okada_mei_sp/source_datasets/shot1_fit_20261014/identity_fix2.json'
    if not data_path.is_file() or not fix_path.is_file():pytest.skip('Confirmed identity evidence is unavailable')
    data=scope.read_reference(scope.reference(data_path));ref=scope.reference(fix_path);fix=scope.read_reference(ref)
    registry=scope.read_reference(fix['feature_registry'])['targets'];fixtures={}
    for index in (526,582,583,587):
        record=next(f for f in data['appended_frames'] if f['index']==index)
        packet=scope.read_reference(record['source_packet']);binding=scope.read_reference(record['source_binding'])
        features={f['id']:f for f in packet['features']};rows=[]
        for observation in binding['observations']:
            feature=features[observation['source_feature_id']];segment=feature['segments'][observation['source_segment_index']]
            indices=observation['source_vertex_indices'];scales,_=scope.source_scales(segment)
            rows.append(dict(observation,source_feature=feature,source_segment=segment,original_vertex_indices=indices,
                uv=np.asarray(segment['uv'],float)[indices],sigma=scales[indices],model_target=registry[observation['target_id']]))
        native=packet['source'];frame=dict(index=index,native_pts=native['native_pts'],packet=record['source_packet'],
            source_image=scope.reference(native['image_path']),mask=dict(path=native['mask_path'],sha256=native['mask_sha256']))
        fixtures[index]=(rows,frame,record)
    return fixtures,ref


@pytest.mark.parametrize('index',[526,582,583,587])
def test_confirmed_material_normals_change_only_Q_of_affected_original_source_group(confirmed_identity_sources,index):
    fixtures,ref=confirmed_identity_sources;rows,frame,original=fixtures[index]
    before=deepcopy(original);before_uv=[row['uv'].copy() for row in rows]
    got=evidence.prepare_model_association_frame_evidence(rows,frame,ref,preserved_frame=original)
    assert got['source_scales_by_point_id']==before['source_scales_by_point_id']
    assert original==before
    for row,uv in zip(rows,before_uv,strict=True):np.testing.assert_array_equal(row['uv'],uv)
    changes=got['corrected_evidence']['physical_material_normals_by_point_id']
    assert len(changes)==5
    altered=0
    for old,new in zip(before['whitening_groups'],got['whitening_groups'],strict=True):
        ids=old['operator']['source_point_ids'];affected=set(ids)&set(changes)
        assert old['point_indices']==new['point_indices']
        assert ids==new['operator']['source_point_ids']
        if not affected:
            assert old==new;continue
        altered+=1;a=SourceWhiteningOperator.from_record(old['operator']);b=SourceWhiteningOperator.from_record(new['operator'])
        assert a.rank==b.rank==5
        assert np.max(abs(b.matrix))<1
        correction=new['physical_material_normal_correction']
        assert correction['source_estimator_B_and_noise_factor_L_preserved'] is True
        for j,pid in enumerate(ids):
            assert np.any(b.matrix[:,2*j:2*j+2])
            if pid in changes:
                n=np.array(changes[pid]['normal']);block=b.matrix[:,2*j:2*j+2]
                np.testing.assert_allclose(block@np.array([-n[1],n[0]]),0.,atol=3e-16)
                assert abs(changes[pid]['source_output_to_material_normal_cosine'])>.97
    assert altered==1
    if index==587:assert 'ice.C0:outer/segment/3/vertex/0' not in changes


def test_confirmed_material_support_rejects_any_changed_approved_UV(confirmed_identity_sources):
    fixtures,ref=confirmed_identity_sources;rows,frame,record=fixtures[587]
    changed=deepcopy(rows)
    row=next(row for row in changed if 'ice.C0:outer/segment/0/vertex/0' in row['source_point_ids'])
    row['uv'][0,0]+=.0001
    with pytest.raises(ValueError,match='accepted source address or UV'):
        evidence.prepare_model_association_frame_evidence(changed,frame,ref,preserved_frame=record)
