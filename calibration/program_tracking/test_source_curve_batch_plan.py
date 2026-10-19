from copy import deepcopy
import json

import numpy as np
import pytest

from calibration.program_tracking.compact_source_header_inventory import read_minimal_compact_source_headers
from calibration.program_tracking.compact_source_json import pack_source_json
from calibration.program_tracking.source_curve_batch_plan import compile_source_curve_batch_plan
from calibration.program_tracking.strict_absolute_source import StrictAbsoluteSourcePack
from calibration.program_tracking.test_strict_source_disjoint_merge import _fixture_pack


SOURCE=dict(path='immutable-source.npz',sha256='a'*64)
MODEL=dict(path='immutable-model.json',sha256='b'*64)
Q='c'*64


def test_light_header_reader_keeps_empty_qualified_source_pack(tmp_path):
    path = tmp_path/'empty_qualified.npz'
    np.savez_compressed(path, rows_json=pack_source_json([]), row_offsets=np.asarray([0], np.int64))
    headers, offsets = read_minimal_compact_source_headers(path)
    assert headers == []
    assert offsets.tolist() == [0]


def _row(index=0,**extra):
    row=dict(shot_id='actual-shot',actual_pts=100,native_pts=110,
        scope='ice',physical_material_id='lane3.paint.edge2',
        model_target=dict(kind='current_scene_ice_curve',curve_id='lane3.edge2'),
        observation_semantics='paint_center',split='fit',point_count=2,
        decoded_BGR_sha256='d'*64,original_source_row_index=index,
        source_visible_paths=[[0],[1]],source_operator_array_fields_present=[])
    row.update(extra)
    return row


def _plan(headers,sigma=None,metadata=None):
    if sigma is None:sigma=np.ones(2*len(headers))
    if metadata is None:metadata=dict(expected_actual_pts=[100,200],processed_actual_pts=[100],all_exposures_processed=False)
    return compile_source_curve_batch_plan(headers,np.arange(len(headers)+1,dtype=np.int64)*2,
        sigma,source_reference=SOURCE,model_reference=MODEL,model_q_sha256=Q,source_metadata=metadata)


def test_same_primitive_candidates_keep_interleaved_source_order_paths_and_held_separate():
    headers=[_row(0),_row(1,physical_material_id='lane2.paint.edge2',
        model_target=dict(kind='current_scene_ice_curve',curve_id='lane2.edge2')),
        _row(2,source_visible_paths=[[0,1]]),_row(3,split='held')]
    original=deepcopy(headers)
    sigma=np.asarray([1.,2.,3.,4.,5.,6.,7.,8.])
    result=_plan(headers,sigma)
    assert result['original_source_points']==8 and result['prepared_groups']==3
    group=result['groups'][0]
    assert [r['original_source_row_index'] for r in group['original_rows']]==[0,2]
    assert [r['original_global_point_range'] for r in group['original_rows']]==[[0,2],[4,6]]
    assert [r['independent_visible_source_paths'] for r in group['original_rows']]==[[[0],[1]],[[0,1]]]
    assert result['groups'][2]['source_context']['split']=='held'
    assert result['original_split_point_counts']==dict(fit=6,held=2)
    assert result['original_processed_actual_pts']==[100] and result['all_source_exposures_processed'] is False
    assert headers==original and sigma.tobytes()==np.arange(1.,9.).tobytes()
    assert result['source_points_removed']==0 and result['original_sigma_or_loss_weights_changed'] is False
    assert not any(result['guards'].values())
    assert all(g['execution_ready'] is False and g['solver_FIT_admitted'] is False for g in result['groups'])
    # Header/proof locators reuse the authenticated top-level source reference,
    # rather than duplicating the large input pack or its reference per row.
    json.dumps(result)


def test_missing_support_typed_and_unknown_operators_remain_independent_singletons():
    headers=[_row(0),_row(1),_row(2,source_operator_array_fields_present=['normal_interval']),
        _row(3,curve_observation_kind='observed_extent_endpoint'),
        _row(4,model_target=dict(kind='future_unqualified_operator',curve_id='lane3.edge2'))]
    sigma=np.ones(10);sigma[1]=np.nan;sigma[3]=np.nan
    result=_plan(headers,sigma)
    assert result['prepared_groups']==5
    assert result['missing_working_sigma_points']==2 and result['finite_positive_working_sigma_points']==8
    assert result['status_row_counts']==dict(singleton_pending_original_source_sigma=2,
        singleton_original_typed_operator=2,singleton_unknown_original_operator=1)
    assert all(g['original_row_count']==1 for g in result['groups'])
    assert sum(g['points'] for g in result['groups'])==10


def test_native_binding_and_exact_target_state_are_part_of_candidate_identity():
    result=_plan([_row(0),_row(1,native_pts=111),_row(2,decoded_BGR_sha256='e'*64)])
    assert result['prepared_groups']==3
    with pytest.raises(ValueError,match='another frozen model'):
        _plan([_row(model_target=dict(kind='current_scene_board_print_contour',scene_q_sha256='f'*64))])


@pytest.mark.parametrize('change',[dict(processed_actual_pts=[100,100]),
    dict(processed_actual_pts=[100,999]),dict(all_exposures_processed=True)])
def test_inconsistent_source_clock_is_rejected(change):
    clock=dict(expected_actual_pts=[100,200],processed_actual_pts=[100],all_exposures_processed=False)
    clock.update(change)
    with pytest.raises(ValueError,match='source clock'):_plan([_row()],metadata=clock)


def test_light_header_reader_preserves_literal_board_target_and_operator_guard(tmp_path):
    ref=_fixture_pack(tmp_path/'board',board=True,split='held')
    minimal,offsets=read_minimal_compact_source_headers(ref['path'])
    public=StrictAbsoluteSourcePack(ref['path'],expected_sha256=ref['sha256'])
    assert offsets.tolist()==public.values['row_offsets'].tolist()
    assert minimal[0]['model_target']==public.rows[0]['model_target']==dict(kind='current_scene_board_edge',edge='upper')
    for key,value in minimal[0].items():
        if key not in ('original_source_row_index','source_operator_array_fields_present'):
            assert value==public.rows[0][key],key
    result=compile_source_curve_batch_plan(minimal,offsets,public.values['sigma_px'],
        source_reference=ref,model_reference=MODEL,model_q_sha256=Q,source_metadata=public.metadata)
    assert result['original_split_point_counts']==dict(held=3)
    assert result['groups'][0]['preparation_status']=='singleton_pending_original_source_sigma'
