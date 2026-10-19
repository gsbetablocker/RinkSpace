from rinkspace.demo import synthetic_demo


def test_demo_is_explicitly_synthetic(tmp_path):
    d=synthetic_demo(tmp_path)
    assert d['real_video_capability_proven'] is False
    assert (tmp_path/'index.html').exists()
    assert (tmp_path/'mosaic_source_index.npy').exists()
    assert d['heldout']['max_source_px'] < 1
