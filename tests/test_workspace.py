"""Exercise real codec I/O: VFR identities, proxy mapping and exact source seeks."""
from fractions import Fraction
import json

import numpy as np
import pytest
from PIL import Image

av = pytest.importorskip("av")

from rinkspace.workspace import (get_frame, import_source, load_index, load_shots,
                                 load_sources, prepare_source, sample_source_window, save_shot_labels,
                                 source_frame_meta)


@pytest.fixture()
def video_project(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/project.example.json").write_text(json.dumps({"sources": {}}))
    path = tmp_path / "variable.mp4"
    times = [0, 40, 80, 140, 180, 260, 300, 340, 400, 440, 500, 560]
    with av.open(str(path), "w") as out:
        stream = out.add_stream("libx264", rate=25)
        stream.width, stream.height, stream.pix_fmt = 160, 90, "yuv420p"
        stream.time_base = Fraction(1, 1000)
        stream.options = {"crf": "12", "preset": "ultrafast"}
        for n, pts in enumerate(times):
            rgb = np.zeros((90, 160, 3), np.uint8)
            rgb[:] = [220, 20, 20] if n < 6 else [20, 220, 20]
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts, frame.time_base = pts, Fraction(1, 1000)
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    import_source(tmp_path, path, key="test")
    return tmp_path


def test_full_decode_proxy_and_exact_frames_keep_actual_vfr(video_project):
    root = video_project
    meta = prepare_source(root, "test", proxy_width=160, proxy_every_n=2)
    rows = load_index(root, "test")["frames"]
    assert meta["frame_count"] == 12
    assert len({round(rows[n+1]["source_time_s"]-rows[n]["source_time_s"], 4) for n in range(11)}) > 1
    mappings = json.loads((root / "outputs/workspace/test/proxy_mapping.json").read_text())["frames"]
    assert [m["source_frame_index"] for m in mappings] == list(range(0, 12, 2))
    for mapping in mappings:
        source = rows[mapping["source_frame_index"]]
        assert mapping["source_pts"] == source["pts"]
        assert mapping["source_time_s"] == source["source_time_s"]
        assert abs(mapping["proxy_time_s"]-source["source_time_s"]) < 1/90000
    image = np.asarray(Image.open(get_frame(root, "test", 8, width=160)))
    assert image[30, 30, 1] > image[30, 30, 0] + 100
    assert source_frame_meta(root, "test", 8)["pts"] == rows[8]["pts"]
    samples = sample_source_window(root, "test", .12, .45, 2, 160, root / "samples")
    assert all(s["pts"] == rows[s["source_frame_index"]]["pts"] for s in samples["samples"])


def test_shot_corrections_preserve_automatic_result_and_partition(video_project):
    root = video_project
    prepare_source(root, "test", make_proxy=False)
    original = (root / "outputs/workspace/test/shots.automatic.json").read_bytes()
    shots = load_shots(root, "test")["shots"]
    for s in shots:
        s["role"] = "programme"
    corrected = save_shot_labels(root, "test", shots)
    assert corrected["status"] == "user_corrected"
    assert original == (root / "outputs/workspace/test/shots.automatic.json").read_bytes()
    assert all(s["role"] == "programme" for s in load_shots(root, "test")["shots"])
    shots[0]["start_frame"] = 1
    with pytest.raises(ValueError, match="partition"):
        save_shot_labels(root, "test", shots)


def test_reimporting_a_key_never_reuses_another_sources_index(video_project):
    root = video_project
    prepare_source(root, "test", make_proxy=False)
    replacement = root / "replacement.mp4"
    replacement.write_bytes((root / "variable.mp4").read_bytes())
    import_source(root, replacement, key="test")
    assert load_sources(root)[0]["prepared"] is False
    with pytest.raises(ValueError, match="identity changed"):
        get_frame(root, "test", 2)
    # Explicit preparation verifies the same bytes and updates the read-only path.
    prepare_source(root, "test", make_proxy=False)
    assert load_sources(root)[0]["prepared"] is True
