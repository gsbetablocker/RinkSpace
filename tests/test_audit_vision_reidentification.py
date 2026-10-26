"""Reidentified material profiles must stay normal to the current image edge."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from rinkspace.vision_reidentify import AcceptedSourceReidentifier


def test_topology_pair_cannot_certify_board_face_or_identity(monkeypatch):
    from rinkspace import vision_instances
    rgb = np.full((128, 128, 3), 160, np.uint8)
    binding = dict(source_sha256='a'*64, source_pts=256, time_base='1/12800', shot_id='shot')
    topology = dict(boundary_topology_hypotheses=[dict(kind=kind,
        preferred_region_search_hypothesis=True, same_board_structure_verified=False,
        source_gradient_pixel_runs_y_x_length=[[y, 8, 112]])
        for kind,y in [('board_top',30),('ice_contact',90)]],
        board_contact_pair_hypotheses=[dict(same_board_structure_verified=False)])
    matcher = AcceptedSourceReidentifier.__new__(AcceptedSourceReidentifier)
    matcher.spec = dict(source_sha256='a'*64,event_id='event')
    matcher.spec_reference = dict(path='source-bank.json',sha256='b'*64)
    matcher.last_search = None
    matcher.native_bank_receipt = {}
    matcher.matcher = SimpleNamespace(read=lambda *a,**k: dict(matches=[]))
    matcher.advertisement_guides = lambda *a,**k: ([],{})
    matcher.ice_search = lambda *a,**k: ([],{})
    matcher.geometry = SimpleNamespace(bind=lambda guides: ([dict(g,
        native_geometry=dict(path_id=g['id'])) for g in guides],{}))
    monkeypatch.setattr(vision_instances,'automatic_source_region_prompts',lambda *a,**k: topology)
    guides,report = matcher.search(rgb,binding,reset=True,
        prediction=dict(class_probabilities=np.zeros((1,1,128,128))),classes=('ice_paint',))
    assert len(guides)==2
    assert all(g['source_board_view']=='unresolved' and not g['identity_confirmed']
        and g['identity_authority'] is None for g in guides)
    assert all(len(g['points_uv']) for g in guides)
    assert report['board_search']['topology_pair_alone_certifies_front_face'] is False


@pytest.mark.parametrize('homography', [
    np.array([[1., 2., 0.], [0., 1., 0.], [0., 0., 1.]]),
    np.array([[-1., 0., 110.], [0., 1., 0.], [0., 0., 1.]]),
    np.array([[1., .3, 0.], [.2, 1., 0.], [.002, .001, 1.]]),
])
def test_ice_reidentification_transports_normals_as_covectors(monkeypatch, homography):
    from rinkspace import camera_reuse_localization, vision_instances, vision_programme_board_review
    from calibration.program_tracking import source_board_measurements

    rgb = np.zeros((128, 128, 3), np.uint8)
    binding = dict(source_sha256='a'*64, source_pts=256, time_base='1/12800', shot_id='shot')
    uv = np.array([[10., 20.], [20., 20.], [30., 20.]], np.float32)
    normal = np.tile([0., 1.], (len(uv), 1))
    guide = dict(id='ice.line', kind='ice_paint', points_uv=uv, normals=normal,
        segment_ids=np.zeros(len(uv), int), source_binding=dict(binding, source_pts=0),
        source_guide_exclusion_mask=np.ones(rgb.shape[:2],bool),
        source_exclusion_protected_mask=np.ones(rgb.shape[:2],bool))
    left = np.array([[0., 0.], [40., 0.], [40., 40.], [0., 40.]], np.float32)
    right = cv2.perspectiveTransform(left[None], homography)[0]
    matcher = AcceptedSourceReidentifier.__new__(AcceptedSourceReidentifier)
    matcher.references = {'ref': dict(guides=[guide], reference={}, source_binding=dict(binding, source_pts=0))}
    matcher.geometry = SimpleNamespace(refs=[dict(path='scene', sha256='b'*64)])
    matcher.matcher = SimpleNamespace(contexts=[dict(source_packet_sha256='ref', image=rgb)])
    matcher.ice_descriptors = {'ref': (left, None)}
    monkeypatch.setattr(camera_reuse_localization, 'descriptors', lambda *args: (right, None))
    monkeypatch.setattr(camera_reuse_localization, 'match_features', lambda *args: (left, right, homography))
    monkeypatch.setattr(vision_instances, 'source_correspondence_identity_verified', lambda *args: True)
    monkeypatch.setattr(source_board_measurements, 'measure_board_source_frame',
        lambda *args: dict(ice_scope_boundary_candidates=[]))
    monkeypatch.setattr(vision_programme_board_review, 'conservative_ice_exclusion',
        lambda excluded, candidates: excluded)
    result, _ = matcher.ice_search(rgb, binding,
        dict(matches=[dict(source_packet_sha256='ref', physical_id='Ad1', inlier_count=4, identity_authority={})]),
        prediction=dict(class_probabilities=np.zeros((1, 1, 128, 128))), classes=('ice_paint',))
    assert len(result) == 1
    for field in ('source_guide_exclusion_mask','source_exclusion_protected_mask'):
        assert field not in result[0]
        assert guide[field].all()  # Preserve the immutable reference frame's masks.
    actual = result[0]['normals']
    mapped = cv2.perspectiveTransform(uv[None], homography)[0]
    mapped_tangent = cv2.perspectiveTransform((uv + [1., 0.])[None].astype(np.float32), homography)[0] - mapped
    mapped_positive_side = cv2.perspectiveTransform((uv + [0., .1])[None].astype(np.float32), homography)[0] - mapped
    np.testing.assert_allclose(np.sum(actual * mapped_tangent, axis=1), 0., atol=5e-6)
    assert np.all(np.sum(actual * mapped_positive_side, axis=1) > 0.)
    np.testing.assert_allclose(np.linalg.norm(actual, axis=1), 1., atol=1e-6)


def test_generic_near_board_reader_does_not_load_another_event_config(monkeypatch):
    from rinkspace import vision_near_board
    def forbidden():
        pytest.fail('Unselected event evidence must not be loaded')
    monkeypatch.setattr(vision_near_board, 'configured_near_evidence', forbidden)
    binding = dict(source_sha256='a'*64, source_pts=0, time_base='1/12800', shot_id='other')
    result = vision_near_board.read_near_board(np.zeros((64, 64, 3), np.uint8), binding)
    assert result['source_binding'] == binding
    assert result['exclusion_pixel_count'] == 0


def test_explicit_near_board_evidence_is_not_silently_dropped(tmp_path):
    from rinkspace.vision_near_board import read_near_board
    binding = dict(source_sha256='a'*64, source_pts=0, time_base='1/12800', shot_id='one')
    with pytest.raises(FileNotFoundError):
        read_near_board(np.zeros((64, 64, 3), np.uint8), binding,
            qualified_generation=dict(path=str(tmp_path/'missing.json'), sha256='b'*64))


def test_reader_passes_only_its_selected_near_board_evidence(monkeypatch):
    from test_vision_tracking_frame import reader_and_spies, source_rgb, binding, guide
    from rinkspace import vision_near_board
    reader, _ = reader_and_spies(monkeypatch)
    gray = dict(path='selected-gray.json', sha256='b'*64)
    qualified = dict(path='selected-qualified.json', sha256='c'*64)
    reader.config = dict(near_board_source_evidence=dict(gray_history=gray, official_gray_generation=qualified))
    original = vision_near_board.read_near_board
    calls = []
    def capture(rgb, source_binding, *, gray_history, qualified_generation, **kwargs):
        calls.append((gray_history, qualified_generation))
        return original(rgb, source_binding, **kwargs)
    monkeypatch.setattr(vision_near_board, 'read_near_board', capture)
    reader.read(source_rgb(8), binding(0), guides_2d=[guide()], reset=True, extract_source_components=False)
    assert calls == [(gray, qualified)]

@pytest.mark.parametrize("evidence", [None, {
    "gray_history": {"path": "selected-gray.json", "sha256": "b" * 64},
    "official_gray_generation": {"path": "selected-qualified.json", "sha256": "c" * 64},
}])
def test_advertisement_review_passes_selected_near_board_evidence(monkeypatch, evidence):
    from rinkspace import vision_near_board
    from rinkspace.vision_review import reread_source_advertisements
    binding = dict(source_sha256="a" * 64, source_pts=0, time_base="1/12800", shot_id="one")
    packet = dict(source_binding=binding, instances=[])
    calls = []
    def capture(rgb, actual_binding, **kwargs):
        assert actual_binding is binding
        calls.append(kwargs)
        return dict(source_binding=binding)
    monkeypatch.setattr(vision_near_board, "read_near_board", capture)
    monkeypatch.setattr(vision_near_board, "exclude_top_face_prints", lambda packet, near: packet)
    result = reread_source_advertisements(np.zeros((4, 5, 3), np.uint8), packet,
        SimpleNamespace(reread_rgb=True), near_board_evidence=evidence)
    assert result is packet
    assert calls == [dict(gray_history=(evidence or {}).get("gray_history"),
                          qualified_generation=(evidence or {}).get("official_gray_generation"))]
