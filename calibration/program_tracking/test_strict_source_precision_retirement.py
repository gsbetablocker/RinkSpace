"""Exact source dispositions; fixtures are not programme measurements."""
import json

import numpy as np
import pytest

from calibration.program_tracking.compact_source_json import pack_source_json
from calibration.program_tracking.strict_model_scope import file_sha
from calibration.program_tracking.strict_source_precision_retirement import index_current_source_precision


def fixture_pack(path, rows, uv, offsets, processed):
    np.savez_compressed(path, uv=np.asarray(uv, np.float64),
        row_offsets=np.asarray(offsets, np.int64), rows_json=pack_source_json(rows),
        metadata_json=json.dumps(dict(source={'fixture_only': True},
            physically_compacted_approved_pixels_only=True,
            processed_actual_pts=processed)))
    return dict(path=str(path), sha256=file_sha(path))


def row(pts=100, *, material='C.paint', split='fit', target='C'):
    return dict(actual_pts=pts, feature_id='C', physical_material_id=material,
        split=split, model_target=dict(kind='ice_curve', curve_id=target))


def test_retention_requires_exact_uv_material_and_target(tmp_path):
    old_rows=[row(), row(material='B1.paint'), row(target='other'), row(200), row(300)]
    old=fixture_pack(tmp_path/'old.npz', old_rows,
        [[1, 2], [3, 4], [1, 2], [1, 2], [1, 2], [1, 2]], [0, 2, 3, 4, 5, 6], [100, 200, 300])
    new=fixture_pack(tmp_path/'current.npz', [row()],
        [[1, 2], [3, 4+1e-12]], [0, 2], [100, 200])
    receipt=index_current_source_precision(old, new, tmp_path/'disposition.npz')
    with np.load(receipt['path'], allow_pickle=False) as saved:
        assert saved['old_point_disposition'].tolist()==[0, 1, 2, 2, 2, 3]
        assert saved['exact_retained_current_point_index'].tolist()==[0, -1, -1, -1, -1, -1]
    # A changed measured coordinate remains superseded, even within tiny tolerance.
    assert receipt['counts']==[1, 1, 3, 1]


def test_existing_held_entity_cannot_become_fit_at_new_coordinate(tmp_path):
    old=fixture_pack(tmp_path/'held.npz', [row(split='held')], [[1, 2]], [0, 1], [100])
    new=fixture_pack(tmp_path/'fit.npz', [row()], [[5, 6]], [0, 1], [100])
    target=tmp_path/'disposition.npz'
    with pytest.raises(ValueError, match='HELD'):
        index_current_source_precision(old, new, target)
    assert not target.exists()
