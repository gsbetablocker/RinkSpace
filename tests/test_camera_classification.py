from rinkspace.camera_classification import classify_observations, high_region_support, validate_catalog, integrate_evidence, integrate_context_evidence, resolve_interval_boundary, apply_physical_identity_decisions, write


def test_single_plane_match_never_supports_physical_installation():
    result = {'status':'processed','image_path':'a.jpg','source_time_seconds':1,'matches':[
        {'kind':'context_candidate','high_multiregion_support':False,'candidate_group_ids':['main_long_side']}]}
    assignment = classify_observations('s', [], [result], 10)
    assert assignment['status'] == 'multiple_candidates'
    assert assignment['physical_camera_id'] is None


def test_two_static_planes_required_and_single_frame_not_support():
    assert not high_region_support({'inliers':100,'x_span':.8,'regions':{'end_wall_windows':0,'grandstand_structure':90,'rink_surface_context':10}})
    result = {'status':'processed','image_path':'a.jpg','source_time_seconds':1,'matches':[
        {'kind':'high_overview','high_multiregion_support':True,'candidate_group_ids':[]}]}
    assert classify_observations('s',[],[result],30)['status'] != 'supported'


def test_crossed_source_ends_are_rejected():
    catalog={'sources':[{'source_id':'s','duration_seconds':10}], 'shots':[
        {'source_id':'s','shot_id':'1','start_pts':0,'end_pts':3,'start_seconds':0,'end_seconds':3,'classification':{'status':'unknown'}},
        {'source_id':'s','shot_id':'2','start_pts':4,'end_pts':10,'start_seconds':4,'end_seconds':10,'classification':{'status':'unknown'}}]}
    assert not validate_catalog(catalog)['structural_pass']


def test_human_confirmation_is_adopted_without_matcher_and_stays_scoped(tmp_path):
    output=tmp_path/'outputs';output.mkdir()
    obs=[{'source_id':'2U7RhoKchXA','source_pts':p,'time_base':'1/1','image_path':f'fixture-{p}.jpg'} for p in [3,13]]
    write(output/'human_evidence.json',{'cases':[{'case_id':'fixture','status':'human_confirmed',
        'installation_id':'human-install','physical_camera_id':'human-position','observations':obs}]})
    shots=[]
    for i in range(3):
        c=classify_observations('2U7RhoKchXA',[],[{'status':'processed','image_path':'fixture.jpg','matches':[]}],10)
        shots.append({'source_id':'2U7RhoKchXA','shot_id':str(i),'start_pts':i*10,'end_pts':i*10+10,
                      'start_seconds':i*10,'end_seconds':i*10+10,'observations':[],'classification':c})
    catalog={'shots':shots,'installations':[],'physical_cameras':[],'evidence_files':[]}
    integrate_evidence(catalog,tmp_path,output)
    assert len(catalog['confirmed_view_assignments'])==2
    assert all(x['installation_id']=='human-install' for x in catalog['confirmed_view_assignments'])
    assert shots[0]['classification']['human_confirmed_observations'][0]['automatic_match_required'] is False
    assert shots[2]['classification']['status']=='unknown'
    assert not any(s['classification']['status']=='supported' for s in shots)


def test_context_match_cannot_cross_unresolved_boundary_or_non_target_content(tmp_path):
    output=tmp_path/'outputs';output.mkdir()
    matches=[{'source_id':'s','source_pts':p,'source_time_seconds':p,'image_path':f'{p}.jpg',
              'position_support_eligible':True} for p in [1,8]]
    write(output/'evidence/context_assignments.json',{'references_path':'refs.json','observations':matches})
    shots=[]
    for i,(boundary,content) in enumerate([('unresolved_dissolve','unresolved_content'),
                                         ('hard_cut','remote_interview'),('hard_cut','unresolved_content')]):
        c=classify_observations('s',[],[{'status':'processed','image_path':'fixture.jpg','matches':[]}],10)
        shots.append({'source_id':'s','shot_id':str(i),'start_seconds':0,'end_seconds':10,
            'observations':matches,'classification':c,'content':{'kind':content},'boundary_start':{'status':boundary}})
    catalog={'shots':shots,'installations':[],'physical_cameras':[],'evidence_files':[]}
    integrate_context_evidence(catalog,tmp_path,output)
    assert shots[0]['classification']['status']!='supported'
    assert shots[1]['classification']['status']!='supported'
    assert shots[1]['content']['kind']=='remote_interview'
    assert shots[2]['classification']['status']=='supported'


def test_interior_uncertainty_endpoints_are_not_video_source_edges():
    source={'frame_count':300,'track':{'time_base':'1/30'}}
    segment={'source_id':'s','start_frame_index':30,'end_frame_index_exclusive':60,
        'start_pts':30,'end_pts_exclusive':60,'start_seconds':1,'end_seconds':2,
        'interval_type':'boundary_unresolved','candidate_points':[{'source_pts':45}]}
    start=resolve_interval_boundary(segment,{},source,'start')
    end=resolve_interval_boundary(segment,{},source,'end')
    assert start['status']=='explicit_interval_start'
    assert end['status']=='explicit_interval_end'
    assert start['uncertainty_interval_seconds']==[1,2]
    assert not start['exact_cut_claim']
    segment.update(start_frame_index=0,start_pts=0,start_seconds=0,end_frame_index_exclusive=300,end_pts_exclusive=300,end_seconds=10)
    assert resolve_interval_boundary(segment,{},source,'start')['status']=='source_start'
    assert resolve_interval_boundary(segment,{},source,'end')['status']=='source_end'


def test_human_position_merge_preserves_versions_and_candidate_status():
    catalog={'human_evidence':{'physical_identity_decisions':[{'decision_id':'merge','status':'human_confirmed',
        'physical_camera_id':'shared','label':'高位全景','installation_ids':['i1','i2'],'scope':'two reference views'}]},
        'installations':[{'installation_id':iid,'physical_camera_id':pid} for iid,pid in [('i1','p1'),('i2','p2')]],
        'physical_cameras':[{'physical_camera_id':pid,'installation_ids':[iid]} for iid,pid in [('i1','p1'),('i2','p2')]],
        'shots':[{'classification':{'status':'supported','physical_camera_id':'p1','installation_id':'i1'}},
                 {'classification':{'status':'multiple_candidates','physical_camera_id':None,'candidate_installation_ids':['i2']}}]}
    apply_physical_identity_decisions(catalog)
    apply_physical_identity_decisions(catalog)
    assert len(catalog['physical_cameras'])==1
    assert catalog['physical_cameras'][0]['installation_ids']==['i1','i2']
    assert catalog['shots'][0]['classification']['physical_camera_id']=='shared'
    assert catalog['shots'][0]['classification']['installation_id']=='i1'
    assert catalog['shots'][1]['classification']['status']=='multiple_candidates'
    assert catalog['shots'][1]['classification']['physical_camera_id'] is None
    assert len(catalog['historical_physical_camera_records'])==2
