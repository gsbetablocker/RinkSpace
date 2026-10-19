import numpy as np

from calibration.program_tracking.compact_source_header_inventory import read_minimal_compact_source_headers
from calibration.program_tracking.compact_source_json import pack_source_json
from calibration.solver_runtime.strict_source_operator_inventory import operator_headers


def test_streamed_material_point_targets_retain_exact_ordered_corner_roles(tmp_path):
    roles = ['upper_left_printed_outline_corner', 'bottom_printed_outline_tip']
    rows = [dict(scope='board', physical_material_id='Ad12:material_points', split='held',
                 actual_pts=15360, point_count=2,
                 model_target=dict(kind='shared_board_pattern_material_points',
                                   advertisement_id=12, material_roles=roles),
                 point_correspondence=True)]
    path = tmp_path / 'source.npz'
    np.savez_compressed(path, rows_json=pack_source_json(rows), row_offsets=np.array([0, 2], np.int64))
    headers, offsets = read_minimal_compact_source_headers(path)
    assert headers[0]['model_target'] == rows[0]['model_target']
    assert headers[0]['point_correspondence'] is True
    assert offsets.tolist() == [0, 2]
    extended, _ = operator_headers(path)
    assert extended[0]['model_target'] == rows[0]['model_target']
