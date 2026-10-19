from types import SimpleNamespace

import numpy as np
import pytest

from scripts.diagnostics.audit_strict_source_actual_images import gather_actual_rows


def _pack():
    def header(actual,ids):
        return dict(actual_pts=actual,native_pts=actual,decoded_BGR_sha256=str(actual),
            original_candidate_indices=ids,scope='ice',physical_material_id='confirmed-paint',
            model_target=dict(kind='current_scene_ice_curve',curve_id='confirmed'),
            observation_semantics='paint_centerline',split='fit')
    return SimpleNamespace(rows=[header(100,[2,4]),header(200,[0]),header(100,[1])],
        values=dict(row_offsets=np.asarray([0,2,3,4]),original_point_indices=np.asarray([2,4,0,1]),
            uv=np.asarray([[20.,10.],[40.,10.],[300.,200.],[10.,10.]]),
            source_sides_uv=np.arange(16.).reshape(4,2,2)))


def test_interleaved_rows_join_only_their_own_actual_candidate_indices():
    pack=_pack();result,duplicates=gather_actual_rows(pack)
    assert duplicates==0
    assert result[100]['rows']==[0,2] and result[100]['original_candidate_indices'].tolist()==[2,4,1]
    assert result[200]['rows']==[1] and result[200]['original_candidate_indices'].tolist()==[0]
    assert np.array_equal(result[100]['uv'],pack.values['uv'][[0,1,3]])
    assert np.array_equal(result[100]['sides'],pack.values['source_sides_uv'][[0,1,3]])
    # This is the actual alias operator: a status vector must be indexed with
    # its own actual's original IDs, never another frame's [2,4,1] IDs.
    own_status={100:np.asarray([0,1,1,0,0]),200:np.asarray([1])}
    assert result[100]['original_candidate_indices'][own_status[100][result[100]['original_candidate_indices']]==1].tolist()==[2,1]
    assert result[200]['original_candidate_indices'][own_status[200][result[200]['original_candidate_indices']]==1].tolist()==[0]


def test_numeric_original_indices_cannot_silently_change_alias_locator():
    pack=_pack();pack.values['original_point_indices'][2]=4
    with pytest.raises(ValueError,match='original candidate indices differ'):gather_actual_rows(pack)


def test_same_actual_conflicting_image_or_noncanonical_native_is_rejected():
    pack=_pack();pack.rows[2]['decoded_BGR_sha256']='different-image'
    with pytest.raises(ValueError,match='conflicting RGB'):gather_actual_rows(pack)
    pack=_pack();pack.rows[0]['native_pts']=101
    with pytest.raises(ValueError,match='own actual/native clock'):gather_actual_rows(pack)
