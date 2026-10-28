"""Synthetic media checks source identity/PTS handling, never real-fit accuracy."""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import av
import numpy as np
import pytest
from PIL import Image, ImageFont

from rinkspace.session.print_installation import verified_source_frame
from rinkspace.session import print_installation


def make_source(tmp_path):
    video = tmp_path/'synthetic.mp4'
    with av.open(str(video), 'w', options={'video_track_timescale': '12800'}) as container:
        stream = container.add_stream('libx264', rate=50)
        stream.width, stream.height, stream.pix_fmt = 32, 24, 'yuv420p'
        stream.time_base = Fraction(1, 12800)
        stream.codec_context.time_base = Fraction(1, 12800)
        for index in range(3):
            rgb = np.full((24, 32, 3), [30 + index*70, 45, 55], dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
            frame.pts, frame.time_base = index*256, Fraction(1, 12800)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    camera = {'source_id': 'synthetic_only', 'shot_id': 'test',
        'time_base_num': 1, 'time_base_den': 12800,
        'image_width': 32, 'image_height': 24, 'first_pts': 0, 'stop_pts_exclusive': 768}
    scope = {'source': {'id': 'synthetic_only', 'sha256': hashlib.sha256(video.read_bytes()).hexdigest(),
        'width': 32, 'height': 24}, 'source_clock': {'time_base_num': 1, 'time_base_den': 12800},
        'required_shot_ids': ['test'], 'shots': [{'id': 'test', 'processing': {
        'first_pts': 0, 'stop_pts_exclusive': 768}}]}
    path = tmp_path/'scope.json'
    path.write_text(json.dumps(scope))
    return video, path, scope, camera


def test_registered_video_decodes_the_requested_exposure_without_nearest_frame(tmp_path):
    video, path, scope, camera = make_source(tmp_path)
    image, record = verified_source_frame(camera, video, path, 512)
    assert image.size == (32, 24)
    assert np.asarray(image)[:, :, 0].mean() > 150
    assert record['status'] == 'native_pts_decoded'
    assert record['native_source_verified'] is True
    assert record['pts'] == 512
    assert record['video']['sha256'] == scope['source']['sha256']
    with pytest.raises(ValueError, match='actual source PTS gap'):
        verified_source_frame(camera, video, path, 257)


def test_wrong_video_bytes_cannot_be_attested_by_hashing_them_after_render(tmp_path):
    video, path, scope, camera = make_source(tmp_path)
    video.write_bytes(video.read_bytes() + b'different-public-test-bytes')
    with pytest.raises(ValueError, match='video SHA256 differs'):
        verified_source_frame(camera, video, path, 256)


@pytest.mark.parametrize('change', ['source', 'clock', 'dimensions', 'shot'])
def test_camera_and_registered_source_must_agree(tmp_path, change):
    video, path, scope, camera = make_source(tmp_path)
    camera = deepcopy(camera)
    if change == 'source':
        camera['source_id'] = 'another-source'
    elif change == 'clock':
        camera['time_base_den'] = 1000
    elif change == 'dimensions':
        camera['image_width'] = 64
    else:
        camera['shot_id'] = 'another-shot'
    with pytest.raises(ValueError, match='source|scope'):
        verified_source_frame(camera, video, path, 256)


def test_decode_checks_real_stream_even_if_camera_and_scope_agree_on_wrong_clock(tmp_path):
    video, path, scope, camera = make_source(tmp_path)
    camera['time_base_den'] = scope['source_clock']['time_base_den'] = 1000
    path.write_text(json.dumps(scope))
    with pytest.raises(ValueError, match='dimensions/time_base differ'):
        verified_source_frame(camera, video, path, 256)


def test_arbitrary_image_preview_does_not_claim_a_verified_source_exposure(tmp_path, monkeypatch):
    image_path = tmp_path/'arbitrary.png'
    Image.new('RGB', (32, 24), 'white').save(image_path)
    font = ImageFont.load_default()
    monkeypatch.setattr(print_installation, 'review_font', lambda *a, **k: font)
    canvas = Mock()
    monkeypatch.setattr(print_installation, 'Canvas', Mock(return_value=canvas))
    monkeypatch.setattr(print_installation, 'sample_model', lambda *a, **k: {'shapes': []})
    core = Mock()
    core.project_motion.return_value = {'uv': np.array([[10., 10.], [20., 15.]]),
        'status': np.zeros(2, dtype=int), 'root_equality_px': np.zeros(2)}
    camera = {'image_width': 32, 'image_height': 24, 'source_id': 'declared_only', 'shot_id': 'test',
        'reference_pts': 0, 'time_base_num': 1, 'time_base_den': 12800,
        'rig': [1.,0.,0.,0.,1.,0.,0.,0.,1., 0.,0.,1., 1.,0.,0.,
                0.,0.,0., .1,.1,.1, 0.,0.,0., 100.,2.,
                16.,12.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
        'controls': np.tile([0.,0.,np.log(100.)], (6,1)),
        'first_control_time': -.04, 'control_dt': .02}
    data = {'instance_id': 'print_instance/test', 'paths': [{'id': 'test.top_edge',
        'role': 'panel_outer_boundary', 'xyz': [[0., 0., 0.], [1., 0., 0.]]}]}
    record = print_installation.source_view(data, {}, core, camera, image_path, 256, tmp_path)
    assert record['source_verification']['native_source_verified'] is False
    assert record['source_pts'] is None and record['source_id'] is None
    assert record['projection_pts'] == 256
    labels = '\n'.join(str(call.args[1]) for call in canvas.text.call_args_list)
    assert '源曝光未核实' in labels and '不能作同源验收' in labels
    assert '官方 50fps 原片' not in labels
