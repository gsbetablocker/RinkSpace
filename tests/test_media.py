from fractions import Fraction
from pathlib import Path
import shutil
import subprocess
import pytest
from rinkspace.media import timestamp_summary, sample_window, index_frames
from rinkspace.io import load_json


def test_vfr_not_average_fps():
    r=timestamp_summary([0,.02,.04,.10,.12])
    assert len(r['delta_histogram'])==2
    assert r['monotonic']


def test_duplicate_timestamp_flagged():
    assert not timestamp_summary([0,.02,.02,.04])['monotonic']


def test_nonfinite_timestamp_rejected():
    with pytest.raises(ValueError):timestamp_summary([0,float('nan')])


@pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'), reason='FFmpeg not installed')
def test_real_decode_source_pts_and_short_window(tmp_path):
    video=tmp_path/'fixture.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=320x180:rate=50:duration=2',
                    '-c:v','libx264','-pix_fmt','yuv420p','-threads','1',str(video)],check=True)
    index_frames(video,tmp_path/'index.json',timeout=60)
    idx=load_json(tmp_path/'index.json')
    assert len(idx['frames'])==100
    r=sample_window(video,tmp_path/'samples',start=.4,duration=.5,every_n=5,width=320,timeout=60)
    assert len(r['samples'])==5
    assert [s['source_time_s'] for s in r['samples']]==pytest.approx([.4,.5,.6,.7,.8])
    assert all(s['source_frame_index'] is None for s in r['samples'])
    for s in r['samples']:
        assert s['source_time_s']==pytest.approx(float(s['pts']*Fraction(s['time_base'])))
