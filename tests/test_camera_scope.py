from copy import deepcopy

from rinkspace.camera_scope import build_scope, clip_shot, project_catalog, _scene_decision
from rinkspace.camera_classification import validate_catalog


def sample_catalog():
    def shot(start,end,kind):
        return {'source_id':'s','shot_id':f's:{start}','start_pts':start,'end_pts':end,
                'end_pts_exclusive':end,'start_seconds':start/30,'end_seconds':end/30,
                'start_frame_index':start,'end_frame_index_exclusive':end,'frame_count':end-start,
                'interval_type':'shot','time_base':'1/30', 'content':{'kind':kind},
                'observations':[], 'classification':{'status':'unknown','candidate_installation_ids':[],
                                                     'supporting_evidence':[]}}
    return {'sources':[{'source_id':'s','frame_count':90,'duration_seconds':3,
                        'track':{'time_base':'1/30','source_sha256':'test'}}],
            'shots':[shot(0,30,'arena_skating'),shot(30,60,'off_rink_arena_kiss_and_cry'),
                     shot(60,90,'unresolved_content')],
            'installations':[],'physical_cameras':[],'appearance_groups':[]}


def test_unknown_scene_is_not_ignored_or_camera_classified(tmp_path):
    catalog=sample_catalog();scope=build_scope(catalog,tmp_path)
    assert scope['sources']['s']['target_frame_count']==30
    assert scope['sources']['s']['ignored_frame_count']==30
    assert scope['sources']['s']['pending_frame_count']==30
    result=project_catalog(catalog,scope)
    assert [s['shot_id'] for s in result['shots']]==['s:0']
    assert validate_catalog(result)['structural_pass']
    assert len(catalog['shots'])==3  # Historical input was not edited.


def test_validator_rejects_ignored_frames_even_when_total_frame_count_matches(tmp_path):
    catalog=sample_catalog();result=project_catalog(catalog,build_scope(catalog,tmp_path))
    result['shots']=[deepcopy(catalog['shots'][1])]
    assert not validate_catalog(result)['structural_pass']


def test_user_first_rink_time_overrides_false_positive_before_it(tmp_path):
    c=sample_catalog();s=c['shots'][0]
    s.update(source_id='Rd66CneD9io',shot_id='women:before',start_pts=0,end_pts=18000,
             end_pts_exclusive=18000,frame_count=18000,end_frame_index_exclusive=18000,end_seconds=600)
    c['shots']=[s]
    c['sources'][0].update(source_id='Rd66CneD9io',frame_count=18000,duration_seconds=600)
    scope=build_scope(c,tmp_path);row=scope['sources']['Rd66CneD9io']
    first=round(583.5333333333333*30)
    assert row['ignored_intervals']==[{'start_pts':0,'end_pts_exclusive':first}]
    result=project_catalog(c,scope)
    assert result['shots'][0]['start_pts']==first
    assert result['shots'][0]['boundary_start']['status']=='scope_start'
    assert validate_catalog(result)['structural_pass']


def test_white_foreign_rink_reference_does_not_pass_target_gate():
    shot=sample_catalog()['shots'][0]
    shot.update(source_id='2U7RhoKchXA',content={'kind':'unresolved_content'},observations=[{},{}])
    shot['classification']['supporting_evidence']=[{'source_pts':p,'matches':[
        {'reference_id':'2U7RhoKchXA:review:8','inliers':50,'x_span':.8,
         'regions':{'end_wall_windows':8,'grandstand_structure':15}}]} for p in (1,20)]
    assert _scene_decision(shot,([1,20],[.8,.8]))[0]=='ignored'


def test_clipping_keeps_source_time_and_removes_outside_evidence():
    shot=sample_catalog()['shots'][0]
    shot['observations']=[{'source_pts':3},{'source_pts':12}]
    shot['classification']['supporting_evidence']=[{'source_pts':3},{'source_pts':12}]
    result=clip_shot(shot,10,20,{'time_base':'1/30','frame_step_pts':1})
    assert result['start_seconds']==10/30
    assert result['observations']==[{'source_pts':12}]
    assert result['classification']['supporting_evidence']==[{'source_pts':12}]
    assert result['frame_count']==10
