"""Source-reader export must not interpret NPZ references as measured UV."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from rinkspace.session import source_read
from rinkspace import vision
from rinkspace.vision_storage import ARRAY_FIELD, hydrate_source_record_arrays


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def test_dense_source_readouts_survive_array_sidecars(tmp_path, monkeypatch):
    import av

    video = tmp_path / 'source.mp4'
    video.write_bytes(b'synthetic decoder fixture')
    scope_path = tmp_path / 'scope.json'
    scope_path.write_text(json.dumps(dict(
        source=dict(registered_path=str(video), sha256=hashlib.sha256(video.read_bytes()).hexdigest(), width=64, height=64),
        source_clock=dict(time_base_num=1, time_base_den=12800, native_pts_step=256),
        event_id='fixture', required_shot_ids=['shot'],
        shots=[dict(id='shot', rig_id='main', processing=dict(first_pts=0, stop_pts_exclusive=256))],
        rig_bindings=dict(ids={'main':'fixture_main_installation'}),
    )), encoding='utf8')
    request = tmp_path / 'request.json'
    request.write_text(json.dumps(dict(programme_scope=str(scope_path), vision_config='fixture',
        sequences=[dict(shot_id='shot', first_pts=0, stop_pts_exclusive=256)])), encoding='utf8')
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    # Each UV array exceeds the production 16 KiB sidecar threshold.
    count = 1100
    uv = np.c_[np.arange(count) % 64, np.arange(count) // 64].astype(np.float64)
    source_result = {}

    class Reader:
        def __init__(self, *args, **kwargs):
            self.native_geometry = SimpleNamespace(refs=[], assert_unchanged=lambda: None)
            self.vision = SimpleNamespace(model_usage=lambda: {})

        def read(self, image, binding, **kwargs):
            result = dict(source_binding=binding, native_geometry={}, memory_update='shot_reset',
                new_observation=True, source_feedback={}, native_material={},
                guided=dict(readings=[dict(expected_id='ice.line', primitive='line',
                    physical_role='painted_stroke_center_axis', actual_source_uv=uv,
                    coherent_source_uv=uv, temporal_flow_supported=np.ones(count, dtype=bool),
                    metadata=dict(source_point_count=count))]),
                near_board=dict(readings=[dict(source_point_count=count, actual_source_uv=uv)]))
            source_result.update(result)
            return result

    class Container:
        streams = SimpleNamespace(video=[SimpleNamespace(time_base=Fraction(1, 12800))])
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def seek(self, *args, **kwargs): pass
        def decode(self, stream):
            yield SimpleNamespace(pts=0, to_ndarray=lambda **kwargs: rgb)

    monkeypatch.setattr(av, 'open', lambda *args, **kwargs: Container())
    monkeypatch.setattr(vision, 'SourceRinkReader', Reader)
    monkeypatch.setattr(vision, '_source_json', _json_value)
    out = tmp_path / 'out'
    receipt = source_read.run_source_read(request, out)
    assert receipt['frames'] == 1
    saved = json.loads((out / 'shot_0.json').read_text())
    assert ARRAY_FIELD in saved['guided']['readings'][0]['actual_source_uv']
    hydrated = hydrate_source_record_arrays(saved, out)
    np.testing.assert_array_equal(hydrated['guided']['readings'][0]['actual_source_uv'], uv)
    report = json.loads((out / 'result.json').read_text())['frames'][0]
    assert report['readings'][0]['read_count'] == count
    assert report['readings'][0]['tracked_count'] == count
    assert report['near_board_points'] == count
    assert (out / report['image']).is_file()
    # Packing and reporting must not mutate the original measured arrays.
    assert source_result['guided']['readings'][0]['actual_source_uv'] is uv


def test_native_file_reference_streams_source_video(tmp_path, monkeypatch):
    from rinkspace.vision_native import file_ref
    path = tmp_path / 'large-source.mp4'
    content = b'a source video can be larger than available RAM' * 10
    path.write_bytes(content)
    def no_read_bytes(self):
        raise AssertionError('Source hashes must stream; do not load a complete video')
    monkeypatch.setattr(Path, 'read_bytes', no_read_bytes)
    assert file_ref(path)['sha256'] == hashlib.sha256(content).hexdigest()


def test_same_path_query_runs_have_distinct_observation_ids_without_changing_uv_or_groups():
    from rinkspace.session.source_observations import qualify_measurement
    context=dict(binding=dict(source_sha256='a'*64,source_pts=512,shot_id='s02'),rgb_sha256='b'*64)
    common=dict(native_path_id='native/line',expected_id='ice.line',source_indices=[0,1],
        guide=dict(normals=[[0.,1.],[0.,1.]]),native_measurement='centreline',
        source_group_id='same_original_correlated_material',support={'actual_readout_evidence':None})
    left=dict(common,query_id='line/left',actual_source_uv=[[10.,20.],[20.,20.]])
    right=dict(common,query_id='line/right',actual_source_uv=[[50.,20.],[60.,20.]])
    authority=dict(identity_confirmed=True,identity_authority={'kind':'source'})
    output=qualify_measurement(context,left,authority)+qualify_measurement(context,right,authority)
    assert len({row.candidate_id for row in output})==4
    assert [row.actual_uv for row in output]==left['actual_source_uv']+right['actual_source_uv']
    assert {row.source_group_id for row in output}=={'same_original_correlated_material'}
    assert [row.candidate_id for row in qualify_measurement(context,left,authority)]==[row.candidate_id for row in output[:2]]


@pytest.mark.parametrize('damage',[None,'uv','pts','path','seed'])
def test_material_graph_identity_qualifies_only_its_exact_current_readout(damage):
    from rinkspace.session.source_observations import qualify_measurement
    binding=dict(source_sha256='a'*64,source_pts=512,time_base='1/12800',shot_id='s02',native_hw=[64,64])
    proof=dict(kind='current_RGB_material_graph_identity',source_binding=dict(binding),
        native_path_id='native/line',source_seed_authority={'reference':'original'},
        current_RGB_measured=True,identity_from_projection_alone=False,decoded_native_RGB_sha256='b'*64)
    member=dict(authority_key='source-row',actual_source_uv=[20.,30.])
    if damage=='uv':member['actual_source_uv']=[20.,31.]
    elif damage=='pts':proof['source_binding']['source_pts']=256
    elif damage=='path':proof['native_path_id']='different/line'
    elif damage=='seed':proof['source_seed_authority']=None
    reading=dict(expected_id='ice.line',native_path_id='native/line',query_id='query',
        actual_source_uv=[[20.,30.],[40.,30.]],source_indices=[2,3],source_group_id='original_group',
        guide=dict(normals=[[0.,1.]]*4),support=dict(actual_readout_evidence=dict(
            identity_authorities={'source-row':proof},identity_by_source_index={'2':member})))
    rows=qualify_measurement(dict(binding=binding,rgb_sha256='b'*64),reading,dict(identity_confirmed=False))
    assert rows[0].identity==('confirmed' if damage is None else 'unresolved')
    assert rows[1].identity=='unresolved'
    assert [c.actual_uv for c in rows]==reading['actual_source_uv']
    assert {c.source_group_id for c in rows}=={'original_group'}
    assert all(c.review['human_scope_extended'] is False for c in rows)


def test_reread_inherits_each_original_dependence_group_without_rewriting_source_points():
    from copy import deepcopy
    from rinkspace.session.source_observations import qualify_measurement
    context=dict(binding=dict(source_sha256='a'*64,source_pts=512,shot_id='s02'),rgb_sha256='b'*64)
    reading=dict(native_path_id='native/line',expected_id='ice.line',source_indices=[1,2,3],
        actual_source_uv=[[10.,20.],[20.,20.],[30.,20.]],
        guide=dict(uv=[[0.,20.],[10.,20.],[20.,20.],[30.,20.]],normals=[[0.,1.]]*4),
        source_group_id='new-frame-search',original_dependence_groups_per_point=['A','A','B',None])
    before=deepcopy(reading)
    rows=qualify_measurement(context,reading,dict(identity_confirmed=True,identity_authority={'kind':'source'}))
    assert [c.source_group_id for c in rows]==['A','B','new-frame-search']
    assert [c.actual_uv for c in rows]==reading['actual_source_uv'] and reading==before
    reading['original_dependence_groups_per_point']=['A']
    with pytest.raises(ValueError,match='align with the source query'):
        qualify_measurement(context,reading,dict(identity_confirmed=True))


@pytest.mark.parametrize('weight_case',['engineering','complete_original_W','incomplete_original_W'])
def test_original_and_reread_pixels_cannot_receive_two_independent_weights(weight_case):
    from copy import deepcopy
    from rinkspace.session.source_observations import SourceCandidate,select_source_observations
    rows=[SourceCandidate(name,'s02',512,{},[x,20.],'line',actual_uv=[x,20.],normal=[0.,1.],
        native_path_id='native/line',identity='confirmed',localization='measured',origin='automatic',
        source_group_id='original_group',evidence=dict(station_initial=.5,precise=name=='reread'))
        for name,x in [('original1',10.),('original2',20.),('reread',10.25)]]
    noise=None
    if weight_case!='engineering':
        w=np.eye(4,dtype='<f8')/3
        noise=dict(groups=[dict(id='original_group',observation_ids=['original1','original2'],
            rank=4,weights=w,allowed_use=['camera_navigation'],matrix_sha256=hashlib.sha256(w.tobytes()).hexdigest())])
        if weight_case=='incomplete_original_W':rows[1].withdrawal=dict(scope='exact point',reason='revoked')
    before=deepcopy(rows)
    selected,complete,receipt=select_source_observations(rows,{'native/line':dict(kind='finite_curve')},noise,'camera_navigation')
    if weight_case=='engineering':assert [r.candidate_id for r in selected]==['reread'] and not complete
    elif weight_case=='complete_original_W':
        assert [r.candidate_id for r in selected]==['original1','original2']
        np.testing.assert_array_equal(complete[0]['weights'],w.reshape(-1))
    else:
        assert not selected and not complete
        assert any(r.get('reason')=='original_correlated_group_not_complete' for r in receipt)
    assert rows==before


@pytest.mark.parametrize('bundled', [False, True])
def test_single_frame_reader_preserves_confirmed_shot_and_bundle_member(tmp_path, monkeypatch, bundled):
    import av
    from rinkspace.vision_geometry import _same_frame

    video = tmp_path / 'source.mp4'
    video.write_bytes(b'exact source fixture')
    rgb = np.full((32, 32, 3), 100, np.uint8)
    sample = dict(source_hash=vision.sha256(video), source_pts=256, time_base='1/12800',
        shot_id='confirmed_shot', frame_size_wh=[32, 32], annotations=[dict(
            feature_id='ice_contact', group_id='user_contact', kind='ice_contact', primitive='free_curve',
            uv=[[10., 20.], [12., 21.], [14., 20.]], state='user_confirmed_visual_supervision',
            authority=dict(human_authority=True))])
    if bundled:
        frame_path = tmp_path / 'frames.npz'
        np.savez(frame_path, exact_frame=rgb, other_frame=np.zeros_like(rgb))
        sample['frame_key'] = 'exact_frame'
    else:
        frame_path = tmp_path / 'frame.png'
        Image.fromarray(rgb).save(frame_path)
    sample['frame_path'] = str(frame_path)
    packet = tmp_path / 'confirmed.json'
    packet.write_text(json.dumps(dict(sample_confirmed=sample)), encoding='utf8')

    class Reader:
        def __init__(self, *args, **kwargs):
            self.config = dict(source_video=str(video))
            self.data = tmp_path

        def read(self, image, binding, *, guides_2d, **kwargs):
            np.testing.assert_array_equal(image, rgb)
            assert len(guides_2d) == 1
            _same_frame(binding, guides_2d[0]['source_binding'])
            return dict(source_binding=binding, model_sha256='b'*64, model_usage={},
                source_clock_evidence={}, guided=dict(readings=[]))

        def draw(self, image, result, ads): return image

    class Container:
        streams = SimpleNamespace(video=[SimpleNamespace(time_base=Fraction(1, 12800))])
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def seek(self, *args, **kwargs): pass
        def decode(self, stream):
            yield SimpleNamespace(pts=256, to_ndarray=lambda **kwargs: rgb)

    monkeypatch.setattr(av, 'open', lambda *args, **kwargs: Container())
    monkeypatch.setattr(vision, 'SourceRinkReader', Reader)
    result = vision.read_frame('fixture', 256, packet)
    assert result['source_binding']['shot_id'] == 'confirmed_shot'
    assert result['source_binding']['source_pts'] == 256


@pytest.mark.parametrize('ranges,step,error', [
    ([(0, 512), (256, 768)], 256, 'overlap'),
    ([(1, 513)], 256, 'aligned'),
    ([(0, 256)], 0, 'positive integer'),
    ([(0, 256)], True, 'positive integer'),
])
def test_source_sequence_scope_fails_before_loading_models(tmp_path, monkeypatch, ranges, step, error):
    video = tmp_path / 'source.mp4'
    video.write_bytes(b'original source fixture')
    scope = tmp_path / 'scope.json'
    scope.write_text(json.dumps(dict(
        source=dict(registered_path=str(video), sha256=vision.sha256(video)),
        source_clock=dict(time_base_num=1, time_base_den=12800, native_pts_step=step),
        shots=[dict(id='shot', processing=dict(first_pts=0, stop_pts_exclusive=1024))],
    )), encoding='utf8')
    request = tmp_path / 'request.json'
    request.write_text(json.dumps(dict(programme_scope=str(scope), vision_config='unused',
        sequences=[dict(shot_id='shot', first_pts=start, stop_pts_exclusive=stop)
                   for start, stop in ranges])), encoding='utf8')

    def must_not_load_model(*args, **kwargs):
        raise AssertionError('Invalid request must fail before model loading or output generation')
    monkeypatch.setattr(vision, 'SourceRinkReader', must_not_load_model)
    with pytest.raises(ValueError, match=error):
        source_read.run_source_read(request, tmp_path / 'out')
