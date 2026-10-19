import importlib.util
import json
from pathlib import Path


def test_temporal_foreground_never_crosses_shot_or_unbounded_gap(tmp_path):
    file = Path(__file__).parents[1] / 'scripts/build_foreground_context_v09.py'
    spec = importlib.util.spec_from_file_location('foreground_context_test', file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = []
    for t, shot, box in [(1., 'a', [10, 20, 100, 200]), (1.2, 'a', None),
                         (1.22, 'b', None), (1.6, 'a', None)]:
        detail = {'expanded_old_seed_box_source_px': box} if box else {}
        rows.append({'source_time_s': t, 'pts': round(t * 100), 'time_base': '1/100',
                     'shot_id': shot, 'source_size': [640, 360],
                     'mask_provenance': {'foreground_details': detail}})
    path = tmp_path / 'proposals.json'
    path.write_text(json.dumps({'source_sha256': 'fixture', 'frames': rows}))
    result = module.build(path)
    assert result['frames'][1]['exclusions']
    assert not result['frames'][2]['exclusions']
    assert not result['frames'][3]['exclusions']
    assert all(x['status'] == 'conservative_temporal_exclusion_not_current_person_measurement'
               for x in result['frames'][1]['exclusions'])
