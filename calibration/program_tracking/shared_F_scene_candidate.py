"""Validate one source-bound shared translation of the complete F houses.

No per-ring, per-frame or installation-dependent scene is admitted. This
validation does not establish accuracy; it protects a reviewable candidate's
geometry and the baseline against accidental mixed-version source binding.
"""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
from z190_main_zoom_k1_source_information import read,ref
from z190_continuous_source_bundle import HARD_SCENE_SHA,SCENE_UNIT_M


def load_candidate(path,baseline_path):
    if ref(baseline_path)['sha256']!=HARD_SCENE_SHA:raise ValueError('Baseline scene changed')
    original=read(baseline_path);candidate=read(path);expected=copy.deepcopy(original)
    old={f['id']:f for f in original['ice_features']};new={f['id']:f for f in candidate['ice_features']}
    delta=new['F1']['center_xyz'][0]-old['F1']['center_xyz'][0]
    for feature in expected['ice_features']:
        if feature['id']not in ('F1','F2','F3','F4'):continue
        feature['center_xyz'][0]+=delta
        for ring in feature.get('rings',[]):ring['center_xyz'][0]+=delta
        actual=new[feature['id']]
        if not np.allclose(actual['center_xyz'],feature['center_xyz'],rtol=0,atol=1e-12):
            raise ValueError('F houses require one shared X translation')
        if any(not np.array_equal(ring['center_xyz'],actual['center_xyz'])for ring in actual.get('rings',[])):
            raise ValueError('F houses must remain concentric in the actual supplied scene')
    if candidate!=expected:raise ValueError('Shared-F candidate changed other modeled geometry or provenance')
    binding=dict(baseline_scene=ref(baseline_path),candidate_scene=ref(path),F_column_dx_m=delta*SCENE_UNIT_M,
        all_four_F_houses_and_concentric_rings_share_translation=True,
        radii_row_Y_R_houses_Lane_other_paint_board_ads_and_metric_unchanged=True,
        accuracy_accepted=False)
    return original,candidate,binding


def rebind(row,original,candidate):
    # The only permitted geometric revision is a common F X translation.
    # Keep every other world array and source UV/role/normal/sigma bit-exact.
    name=str(row['id']).split('_')[0]
    if name not in ('F1','F2','F3','F4'):return row
    old=next(f for f in original['ice_features']if f['id']==name)
    new=next(f for f in candidate['ice_features']if f['id']==name)
    result=dict(row);result['world']=np.asarray(row['world'],float).copy()
    result['world'][:,0]+=new['center_xyz'][0]-old['center_xyz'][0]
    result['world_scene_binding']=dict(feature=name,source_UV_role_and_sigma_changed=False,
        shared_complete_F_column_translation=True)
    return result
