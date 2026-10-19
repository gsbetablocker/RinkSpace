from copy import deepcopy

import pytest

from rinkspace import camera_classification as classification


def identity_catalog():
    return {'scope':{'target_rink_only':True},
        'human_evidence':{'physical_identity_decisions':[{'decision_id':'merge','status':'human_confirmed',
            'physical_camera_id':'shared','label':'confirmed position','installation_ids':['i1','i2'],
            'scope':'two reviewed views'}]},
        'installations':[{'installation_id':'i1','physical_camera_id':'p1'}],
        'physical_cameras':[{'physical_camera_id':'p1','installation_ids':['i1']}],
        'shots':[{'classification':{'status':'supported','physical_camera_id':'p1','installation_id':'i1'}}]}


def test_identity_decision_applies_to_available_scope_without_dangling_installations():
    catalog=identity_catalog();original=deepcopy(catalog['human_evidence'])
    classification.apply_physical_identity_decisions(catalog)
    classification.apply_physical_identity_decisions(catalog)
    assert catalog['human_evidence']==original
    assert catalog['physical_cameras'][0]['installation_ids']==['i1']
    assert catalog['physical_cameras'][0]['physical_camera_id']=='shared'
    assert catalog['shots'][0]['classification']['physical_camera_id']=='shared'
    assert catalog['physical_identity_decision_scope']['merge']['unavailable_in_current_scope_installation_ids']==['i2']
    assert len(catalog['historical_physical_camera_records'])==1


def test_identity_decision_outside_scope_does_not_create_empty_camera():
    catalog=identity_catalog();catalog.update(installations=[],physical_cameras=[],shots=[])
    classification.apply_physical_identity_decisions(catalog)
    assert catalog['physical_cameras']==[]
    assert catalog['applied_physical_identity_decision_ids']==[]
    assert catalog['physical_identity_decision_scope']['merge']['status']=='outside_current_scope'


def test_missing_installation_still_errors_without_scene_scope():
    catalog=identity_catalog();catalog.pop('scope')
    with pytest.raises(ValueError,match='unavailable installation'):
        classification.apply_physical_identity_decisions(catalog)


def test_new_timeline_scope_flags_resolve_as_scope_boundaries():
    segment={'source_id':'s','start_frame_index':30,'end_frame_index_exclusive':60,
             'start_pts':30,'end_pts_exclusive':60,'start_seconds':1,'end_seconds':2,
             'scope_start':True,'scope_end':True}
    source={'frame_count':90,'track':{'time_base':'1/30'}}
    assert classification.resolve_interval_boundary(segment,{},source,'start')['status']=='scope_start'
    assert classification.resolve_interval_boundary(segment,{},source,'end')['status']=='scope_end'


def test_metadata_refresh_preserves_projected_scope_boundaries(tmp_path,monkeypatch):
    output=tmp_path/'output'
    boundaries={f'boundary_{side}':{'status':f'scope_{side}','source_pts':p,
                  'source_seconds':p/30,'meaning':'scene scope endpoint','exact_cut_claim':False}
                for side,p in [('start',30),('end',60)]}
    catalog={'sources':[{'source_id':'s','frame_count':90,'duration_seconds':3,'track':{'time_base':'1/30'},
                 'processing_scope':{'target_intervals':[{'start_pts':30,'end_pts_exclusive':60}],
                                     'target_frame_count':30}}],
             'shots':[{'shot_id':'s:30','source_id':'s','start_pts':30,'end_pts':60,'end_pts_exclusive':60,
                       'start_seconds':1,'end_seconds':2,'start_frame_index':30,'end_frame_index_exclusive':60,
                       'frame_count':30,'classification':{'status':'unknown'},**boundaries}],
             'installations':[],'physical_cameras':[]}
    classification.write(output/'catalog.json',catalog)
    classification.write(output/'validation.json',{})
    monkeypatch.setattr(classification,'collect_visual_validation',lambda *_:{'summary':{}})
    result=classification.refresh_catalog_metadata(tmp_path,output)
    saved=classification.read(output/'catalog.json')
    assert result['structural_pass']
    assert result['metadata_refresh']['changed_endpoint_count']==0
    assert {k:saved['shots'][0][k] for k in boundaries}==boundaries
