"""Intermediate frames skip global detection, but keep current RGB rereading."""
from types import SimpleNamespace
import numpy as np
import pytest

torch = pytest.importorskip('torch')
from rinkspace.vision import SourceRinkReader
from rinkspace import vision_geometry


def reader_and_spies(monkeypatch):
    from rinkspace import vision_near_board
    monkeypatch.setattr(vision_near_board, 'configured_near_evidence', lambda: {})
    calls = {'global': 0, 'guided': 0}
    reader = SourceRinkReader.__new__(SourceRinkReader)
    reader.active_guides = []
    reader.active_binding = None
    reader.active_global_exclusion = None
    module = SimpleNamespace(device='cpu', classes=('background', 'ice_contact'),
                             model_usage=lambda:dict(sha256='b'*64,path='cpu_test_stub'))

    def predict(rgb, binding, reset=False):
        module.current_image = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].float() / 255
        current = {**binding, 'native_hw': list(rgb.shape[:2])}
        module.current_prediction = dict(source_binding=current, prediction_source_binding=current,
                    class_probabilities=torch.zeros((1, 2, *rgb.shape[:2])),
                    memory_update='shot_reset', new_observation=True,
                    model_sha256='b' * 64, source_clock_evidence={}, motion={})
        return module.current_prediction

    def global_search(image, probability, **kwargs):
        calls['global'] += 1
        return dict(source_binding=kwargs['rgb_source_binding'], native_hw=list(image.shape[-2:]),
                    components=[{'source_component_index': 0}])

    def guided(image, probability, guides, **kwargs):
        calls['guided'] += 1
        # Exercise the reader's routing with a deterministic actual-image
        # measurement. Deliberately different guide coordinates cannot become
        # output coordinates just because global detection was skipped.
        x = int(torch.argmax(image[0, 0, 4]))
        return dict(readings=[dict(actual_source_uv=torch.tensor([[x, 4.], [x, 6.]]),
                                  source_indices=torch.tensor([0, 1]))], metadata={})

    module.reset = lambda: None
    module.predict = predict
    reader.vision = module
    monkeypatch.setattr(vision_geometry, 'extract_source_primitives', global_search)
    monkeypatch.setattr(vision_geometry, 'guided_source_reading', guided)
    monkeypatch.setattr(vision_geometry, 'coherent_named_source_reading', lambda reading: reading)
    return reader, calls


def source_rgb(x):
    rgb = np.zeros((10, 12, 3), dtype=np.uint8)
    rgb[:, x, 0] = 255
    return rgb


def binding(pts):
    return dict(source_sha256='a' * 64, source_pts=pts, time_base='1/15360', shot_id='one')


def guide():
    return dict(id='contact', points_uv=[[3., 4.], [3., 6.]])


def test_tracking_frame_rereads_current_image_without_global_detector(monkeypatch):
    reader, calls = reader_and_spies(monkeypatch)
    first = reader.read(source_rgb(8), binding(0), guides_2d=[guide()], reset=True,
                        extract_source_components=False)
    second = reader.read(source_rgb(9), binding(256), guides_2d=[guide()], reset=True,
                         extract_source_components=False)
    assert calls == {'global': 0, 'guided': 2}
    assert first['guided']['readings'][0]['actual_source_uv'].tolist() == [[8., 4.], [8., 6.]]
    assert second['guided']['readings'][0]['actual_source_uv'].tolist() == [[9., 4.], [9., 6.]]
    assert second['source']['components'] == []
    assert not second['source']['full_frame_component_extraction_performed']
    assert second['source']['component_search_status'] == 'not_performed_on_intermediate_tracking_frame'
    assert second['source_binding']['source_pts'] == 256
    assert not second['hint_pixels_used_as_output']


def test_keyframe_default_keeps_original_global_search(monkeypatch):
    reader, calls = reader_and_spies(monkeypatch)
    result = reader.read(source_rgb(8), binding(0), guides_2d=[guide()], reset=True)
    assert calls == {'global': 1, 'guided': 1}
    assert result['source']['full_frame_component_extraction_performed']
    assert result['source']['components'] == [{'source_component_index': 0}]
    assert result['guided']['readings'][0]['actual_source_uv'].tolist() == [[8., 4.], [8., 6.]]


@pytest.mark.parametrize('identity_confirmed', [False, True])
def test_temporal_transport_preserves_source_identity_qualification(monkeypatch, identity_confirmed):
    """Source provenance alone must not name an unverified candidate next frame."""
    reader, _ = reader_and_spies(monkeypatch)
    predict = reader.vision.predict
    reader.vision.last_motion = None

    def continued_prediction(rgb, current_binding, reset=False):
        prediction = predict(rgb, current_binding, reset=reset)
        prediction['memory_update'] = 'shot_reset' if reset else 'image_aligned'
        return prediction

    reader.vision.predict = continued_prediction
    dispatched = []

    def read_current_pixels(image, probability, guides, **kwargs):
        current, = guides
        dispatched.append(current['identity_confirmed'])
        # Isolate dispatch from CUDA localization, preserving the real reader's
        # named-output contract: an unconfirmed expected identity stays unnamed.
        named = current['identity_confirmed'] is True and bool(current.get('identity_authority'))
        x = int(torch.argmax(image[0, 0, 4]))
        return dict(readings=[dict(
            expected_id=current['id'], physical_id=current['id'] if named else None,
            identity_authority=current.get('identity_authority') if named else None,
            actual_source_uv=torch.tensor([[x, 4.], [x, 6.]]),
            source_indices=torch.tensor([0, 1]))], metadata={})

    monkeypatch.setattr(vision_geometry, 'guided_source_reading', read_current_pixels)
    # Same-frame provenance is present even for the unreviewed machine hint.
    source_guide = dict(guide(), origin='source_annotation_search_hint',
        identity_confirmed=identity_confirmed,
        identity_authority={'reference': 'original_source_candidate.json'},
        source_binding=binding(0))
    first = reader.read(source_rgb(8), binding(0), guides_2d=[source_guide],
                        reset=True, extract_source_components=False)
    assert reader.active_guides[0]['identity_confirmed'] is identity_confirmed
    assert reader.active_guides[0]['identity_authority'] == source_guide['identity_authority']
    second = reader.read(source_rgb(9), binding(256), extract_source_components=False)
    expected_physical_id = source_guide['id'] if identity_confirmed else None
    assert dispatched == [identity_confirmed, identity_confirmed]
    assert first['guided']['readings'][0]['physical_id'] == expected_physical_id
    assert second['guided']['readings'][0]['physical_id'] == expected_physical_id
    assert second['guided']['readings'][0]['actual_source_uv'].tolist() == [[9., 4.], [9., 6.]]
    assert reader.active_guides[0]['identity_confirmed'] is identity_confirmed
