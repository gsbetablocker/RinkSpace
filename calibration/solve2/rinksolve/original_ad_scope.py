"""Exact numerical subscopes of the current fixed-camera advertisement graph."""
from __future__ import annotations
import numpy as np
from .observation_edits import SelectionSnapshot,apply_selection


def scope_one_ad(prepared,model,ad_id):
    if not isinstance(ad_id,str) or not ad_id.startswith('Ad'):raise ValueError('explicit advertisement ID required')
    active=prepared.selection_current_active_rows();curves={c.id:c for c in model.scene.curves}
    owners={k:model.scene.resolved_definition(c,model.state).get('shared_layout_block_id') for k,c in curves.items()}
    def belongs(curve):
        key=owners.get(str(curve));return bool(key and key.split('/')[1]==ad_id)
    bound=prepared.bound_curve_ids(active);keep=np.array([belongs(v) for v in bound]);disabled=active[~keep]
    if not keep.any():raise ValueError('advertisement has no current active source: '+ad_id)
    disabled_set=set(map(int,disabled));groups=[]
    for i,g in enumerate(prepared.manifest['groups']):
        if not g.get('active_statistical'):continue
        lo,hi=prepared.arrays['group_member_offsets'][i:i+2];rr=prepared.arrays['group_member_rows'][lo:hi]
        off=[int(r) in disabled_set for r in rr]
        if any(off):
            if not all(off):raise ValueError('advertisement subproblem would split complete source W: '+g['id'])
            groups.append(g['id'])
    landmarks={f.id:f for f in model.landmarks};supp=[]
    for v in prepared.manifest.get('ad7_fixed_observations',[]):
        feature=landmarks[v['fixed_feature_id']];curve=feature.position.geometry_feature_ids[0]
        if not belongs(curve):supp.append(v['id'])
    result=apply_selection(prepared,SelectionSnapshot(int(prepared.manifest.get('selection_revision',0)),disabled,disabled,tuple(supp),tuple(groups),{}))
    result.manifest.update(numerical_data_scope='original_S01_static_ad/'+ad_id,numerical_stage_id='independent_ad_initialization',
        subproblem_ad_id=ad_id,subproblem_budget_is_temporary=True,final_acceptance_requires_original_whole_ad_graph=True,
        source_admission_changed_by_stage=False,source_UV_PTS_E_W_and_raw_admission_changed=False)
    return result
