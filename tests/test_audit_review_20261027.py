"""Real encoded replay must retain the declared source clock and image size."""
from fractions import Fraction
import hashlib
import json
from types import SimpleNamespace

import av
import numpy as np
import pytest

from rinkspace.review import source_programme as review


@pytest.mark.parametrize('root', ['package', 'data'])
def test_review_scope_uses_the_same_declaring_root_as_source_navigation(tmp_path, root):
    from test_source_navigation_resume import case, saved
    from rinkspace.session.source_programme import navigation_result_target
    owner=tmp_path/'package';owner.mkdir()
    ledger,_,_,geometry,dependencies=case(owner)
    try:
        result,_,_=navigation_result_target(ledger.root,'s02',dependencies())
        saved(result,dependencies())
        job=dict(roots=dict(repo=str(tmp_path/'repo'),data=str(owner)),
            scope_ref=dict(root=root,path='scope.json'),sequences=[dict(shot_id='s02')])
        kwargs={'job_path':owner/'job.json'} if root=='package' else {}
        cameras,refs=review.current_review_navigation(job,ledger,geometry,
            ledger.root/'navigation',{'s02':str(result)},**kwargs)
        np.testing.assert_array_equal(cameras['s02']['controls'],np.ones((4,3)))
        assert refs['s02']['path']==str(result.resolve())
    finally:
        ledger.db.close()


def test_point_overlay_uses_the_original_rational_time_base(monkeypatch):
    labels = []
    monkeypatch.setattr(review.cv2, 'putText', lambda image, text, *a, **kw: labels.append(text))
    review.draw_points(np.zeros((48, 64, 3), np.uint8), dict(candidates=[],
        source_binding=dict(shot_id='s01', source_pts=1500, time_base='1/1000')))
    assert 'source 1.500s' in labels[0]


def test_programme_review_encodes_declared_size_cadence_and_every_pts(tmp_path, monkeypatch):
    video = tmp_path/'source.mp4'
    tb = Fraction(1, 1000)
    pts = [1000, 1040, 1080]
    with av.open(str(video), 'w', options={'video_track_timescale': '1000'}) as output:
        stream = output.add_stream('libx264', rate=25)
        stream.width, stream.height, stream.pix_fmt = 64, 48, 'yuv420p'
        stream.time_base = stream.codec_context.time_base = tb
        for i, stamp in enumerate(pts):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), 60+i*30, np.uint8), format='rgb24')
            frame.pts, frame.time_base = stamp, tb
            for packet in stream.encode(frame): output.mux(packet)
        for packet in stream.encode(): output.mux(packet)
    records = {}
    with av.open(str(video)) as source:
        for frame in source.decode(video=0):
            rgb = frame.to_ndarray(format='rgb24')
            records[frame.pts] = dict(candidates=[], source_binding=dict(
                shot_id='s01', source_pts=frame.pts, time_base='1/1000',
                decoded_native_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()))
    assert sorted(records) == pts
    ledger = SimpleNamespace(db=SimpleNamespace(execute=lambda _: [(p,) for p in pts]),
                             read=lambda shot, stamp: records[stamp])
    cameras = {'s01': {}}
    monkeypatch.setattr(review, 'current_review_navigation', lambda *a, **k: (cameras, {}))
    monkeypatch.setattr(review, 'project', lambda *a, **k: dict(uv=np.array([[10., 10.]]),
        status=np.array([0]), root_equality_px=np.array([0.])))
    missing = []
    def top_down(xyz, cuts, height, missing_count):
        missing.append(missing_count)
        return np.zeros((height, 640, 3), np.uint8)
    monkeypatch.setattr(review, 'top_down', top_down)
    samples = dict(xyz=np.array([[0., 0., 0.]]), cuts=[],
                   missing_entities=[dict(id='unlocalized')])
    geometry = SimpleNamespace(samples=samples, core=None, assert_unchanged=lambda: None)
    job = dict(source_clock=dict(time_base_num=1, time_base_den=1000, native_pts_step=40),
        source=dict(registered_path=str(video), width=64, height=48),
        sequences=[dict(shot_id='s01', first_pts=1000, stop_pts_exclusive=1120)],
        selected_shot_ids=['s01'])
    out = tmp_path/'review'
    result = review.render_programme(job, ledger, geometry, tmp_path, out, lambda _: None)
    assert result['native_exposures'] == 3 and missing == [1]
    for entry, width in zip(result['videos'], (64, 704)):
        assert entry['verification']['native_PTS_equal']
        with av.open(str(out/entry['path'])) as output:
            assert (output.streams.video[0].width, output.streams.video[0].height) == (width, 48)
            decoded = list(output.decode(video=0))
            assert [frame.pts*frame.time_base for frame in decoded] == [Fraction(0), Fraction(1, 25), Fraction(2, 25)]
    # A second request reuses the already verified same-identity output.
    saved = json.loads((out/'review.json').read_text())
    assert review.render_programme(job, ledger, geometry, tmp_path, out, lambda _: None) == saved
