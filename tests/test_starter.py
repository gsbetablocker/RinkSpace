from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]

def test_rink_dimensions_not_invented():
    cfg=json.loads((ROOT/'configs/project.example.json').read_text(encoding='utf-8'))
    assert cfg['rink']['length_m'] is None and cfg['rink']['width_m'] is None
    assert cfg['timing']['allow_frame_over_average_fps_fallback'] is False
