import json

import numpy as np

from scripts.admit_programme_source_inputs import LegacyHeldSourceRoles,whole_shot_role


def test_ml_whole_shot_does_not_set_calibration_split():
    binding=dict(source_sha256='a'*64,time_base='1/15360',native_pts=100,shot_id='whole')
    plan=dict(shots=[dict(id='whole',source_sha256='a'*64,time_base='1/15360',
        start_pts=101,end_pts=200,whole_shot=True,split='development')])
    assert whole_shot_role(plan,binding)=='development'
    # Native boundary refinement preserves the whole-shot role by identity,
    # rather than allocating a shifted boundary exposure to another split.
    assert whole_shot_role(plan,dict(binding,shot_id='unlisted'))=='unlisted_not_training'


def test_original_ad_holdout_covers_new_current_measurements(tmp_path):
    path=tmp_path/'source_inventory.npz'
    rows=[dict(actual_pts=100,id='Ad13_outer_upper_100',split='held')]
    np.savez_compressed(path,actual_pts=np.array([100]),uv=np.array([[10.,20.]]),
        row_index=np.array([0]),rows_json=json.dumps(rows))
    roles=LegacyHeldSourceRoles([path])
    same=dict(actual_pts=100,advertisement_number=13,physical_material_id='Ad13.top_edge',
        uv=np.array([[15.,25.]]))
    assert roles.split(same)=='held'
    assert roles.split(dict(same,actual_pts=200))=='fit'
    assert roles.split(dict(same,advertisement_number=14,physical_material_id='Ad14.top_edge'))=='fit'


def test_source_anchor_holdout_does_not_propagate_to_other_exposures():
    roles=LegacyHeldSourceRoles([])
    roles.include_source_annotation_holdouts([dict(actual_pts=100,curve_id='F1:outer',
        physical_id='F1',source_original_split='holdout')])
    row=dict(actual_pts=100,feature_id='F1',curve_id='F1:middle',uv=[[15.,25.]])
    assert roles.split(row)=='held'
    assert roles.split(dict(row,actual_pts=200))=='fit'
    assert roles.split(dict(row,feature_id='F2',curve_id='F2:middle'))=='fit'
