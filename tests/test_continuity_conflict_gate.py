import copy

import numpy as np

from rinkspace.continuity_conflict_gate import gate_unresolved_components


POINTS = [[20, 20], [100, 20], [200, 20], [300, 20], [20, 100], [100, 100], [200, 100], [300, 100]]
CHAIN = {'H_reference_undistorted_from_model': np.eye(3).tolist(),
         'H_current_sample_from_reference_raw': np.eye(3).tolist(),
         'reference_sample_size': [960, 540], 'reference_radial_distortion': None}
GRAPH = {'components': [{'id': 'g', 'reason': 'unresolved_anchor_hypotheses_retained'}]}


def frame(index, parent=None, *, conflict=False, anchor=False, primary=0):
    row = {'sample_index': index, 'source_frame_index': index, 'pts': index, 'time_base': '1/50',
           'source_time_s': index / 50, 'source_sha256': 'source', 'shot_id': 'shot',
           'camera_component_id': 'local', 'offline_camera_graph_id': 'g', 'sample_size': [960, 540],
           'status': 'ambiguous' if conflict else 'model_reobserved' if anchor else 'image_tracked',
           'reason': 'unresolved_model_anchor_conflict' if conflict else 'old_publication',
           'projection_chain': None if conflict else copy.deepcopy(CHAIN), 'anchor_id': 'ref',
           'H_image_from_model': None if conflict else np.eye(3).tolist(),
           'H_model_from_image': None if conflict else np.eye(3).tolist(),
           'model_bridge': None if conflict else {'old': True},
           'anchor_lineage': [{'sample_index': index if anchor else primary}],
           'local_registration': {'status': 'candidate', 'H_reference_from_current': np.eye(3).tolist(),
               'edge_keyframe_sample_index': parent, 'fit_current_xy': copy.deepcopy(POINTS),
               'fit_keyframe_xy': copy.deepcopy(POINTS)}}
    if anchor:
        row['model_anchor_candidate'] = {'projection_chain': copy.deepcopy(CHAIN),
            'fit': {'fit_current_xy': copy.deepcopy(POINTS), 'fit_reference_xy': copy.deepcopy(POINTS)}}
    return row


def test_long_edge_cannot_jump_over_conflict_to_old_anchor():
    rows = [frame(0, anchor=True), frame(1, 0, conflict=True), frame(2, 0), frame(3, 2)]
    before = copy.deepcopy(rows)
    result = gate_unresolved_components(rows, GRAPH)
    assert rows == before
    assert [r['status'] for r in result['frames']] == ['model_reobserved', 'ambiguous', 'ambiguous', 'ambiguous']
    assert result['frames'][2]['projection_chain'] is None
    assert result['frames'][2]['model_bridge'] is None
    assert result['frames'][2]['conflict_gate_previous_output']['projection_chain'] == CHAIN
    assert result['diagnostics']['removed_relative_edges']


def test_real_anchor_after_barrier_permits_only_pose_with_that_lineage():
    rows = [frame(0, anchor=True), frame(1, 0, conflict=True), frame(2, 0),
            frame(3, 2, anchor=True), frame(4, 3, primary=3)]
    result = gate_unresolved_components(rows, GRAPH)['frames']
    assert result[2]['status'] == 'ambiguous'  # It still publishes the old-side pose.
    assert result[3]['status'] == 'model_reobserved'
    assert result[4]['status'] == 'image_tracked'


def test_synthetic_pose_without_actual_pairs_cannot_clear_barrier():
    rows = [frame(0, anchor=True), frame(1, 0, conflict=True), frame(2, anchor=True)]
    rows[2]['model_anchor_candidate']['fit'] = {}
    result = gate_unresolved_components(rows, GRAPH)['frames']
    assert result[2]['status'] == 'ambiguous'


def test_component_without_known_conflict_still_preserves_real_anchor_path():
    rows = [frame(0, anchor=True), frame(1, 0), frame(2, 1)]
    assert [r['status'] for r in gate_unresolved_components(rows, GRAPH)['frames']] == [r['status'] for r in rows]


def test_consistent_joint_component_is_not_reclassified():
    rows = [frame(0, anchor=True), frame(1, 0)]
    graph = {'components': [{'id': 'g', 'reason': 'joint_constraints_consistent'}]}
    assert gate_unresolved_components(rows, graph)['frames'] == rows


def test_source_and_shot_do_not_share_recovery_anchor():
    rows = [frame(0, anchor=True), frame(1)]
    rows[1]['source_sha256'] = 'different_source'
    assert gate_unresolved_components(rows, GRAPH)['frames'][1]['status'] == 'ambiguous'
