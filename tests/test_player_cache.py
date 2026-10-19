import copy
from pathlib import Path

import pytest

from rinkspace.io import dump_json, load_json, sha256
from rinkspace.player_cache import LocalizationCache
from rinkspace.storage import resolve_asset


def test_catalog_reads_header_without_parsing_large_frame_diagnostics_and_invalidates_by_stat(tmp_path,monkeypatch):
    import rinkspace.player_cache as module
    path=tmp_path/'outputs/whole_rink_localization/run/localization.json'
    run={'source_id':'athlete','model_id':'rink','model_revision':'v07','algorithm_version':'continuity',
         'selected_window':{'first_pts':100,'last_pts':101,'time_base':'1/50','frame_count':2},
         'status_counts':{'image_tracked':2},'frames':[{'source_time_s':2+i*.02,
            'status':'image_tracked','reobservation_attempts':[{'large_diagnostics':'x'*200000}]} for i in range(2)]}
    dump_json(path,run)
    original=module.load_json
    reads=[]
    def spy(p):
        reads.append(Path(p))
        if Path(p)==path:raise AssertionError('Catalog must not parse the full frame array')
        return original(p)
    monkeypatch.setattr(module,'load_json',spy)
    cache=LocalizationCache(tmp_path)
    first=cache.summary(path)
    assert first['frame_count']==2 and first['start_s']==2 and first['end_s']==2.02
    assert 'frames' not in first
    assert cache.summary(path)==first
    assert reads and path not in reads
    run['source_id']='changed_athlete';dump_json(path,run)
    assert cache.summary(path)['source_id']=='changed_athlete'


def test_player_retains_geometry_pts_and_provenance_but_loads_full_diagnostics_only_on_demand(tmp_path,monkeypatch):
    import rinkspace.player_cache as module
    path=tmp_path/'outputs/whole_rink_localization/run/localization.json'
    frame={'source_frame_index':5,'source_time_s':.1,'pts':5,'time_base':'1/50',
           'scene_observations':{'board_surface':{'points_source_px':[[120,340]],'used_in_camera':True}},
           'status':'ambiguous','reason':'model_anchor_conflict','projection_chain':None,
           'anchor_lineage':[{'reference_id':'old_view','pts':0,'time_base':'1/50'}],
           'reobservation_attempts':[{'large_diagnostics':'x'*100000}],
           'model_anchor_hypotheses':[{'large_fit':'x'*100000}]}
    run={'source_id':'athlete','model_revision':'v07','frames':[frame]};dump_json(path,run)
    before=path.read_bytes();cache=LocalizationCache(tmp_path)
    player=cache.player(path)
    assert player['frames'][0]['anchor_lineage']==frame['anchor_lineage']
    assert player['frames'][0]['projection_chain'] is None
    assert player['frames'][0]['scene_observations']==frame['scene_observations']
    assert 'reobservation_attempts' not in player['frames'][0]
    assert 'model_anchor_hypotheses' not in player['frames'][0]
    original=module.load_json
    def prohibit_full(p):
        assert Path(p)!=path
        return original(p)
    monkeypatch.setattr(module,'load_json',prohibit_full)
    assert cache.player(path)==player
    assert path.read_bytes()==before
    assert load_json(path)==run


def test_display_sidecar_changes_only_view_and_invalidates_cache_without_mutating_pose(tmp_path,monkeypatch):
    import rinkspace.player_cache as module
    import pytest
    path=tmp_path/'outputs/whole_rink_localization/run/localization.json'
    frame={'pts':10,'time_base':'1/50','source_time_s':.2,'status':'image_tracked',
           'H_image_from_model':[[1,0,20],[0,1,30],[0,0,1]],
           'projection_chain':{'camera':{'K':[[800,0,480],[0,800,270],[0,0,1]]}},
           'scene_observations':{'board_surface':{'used_in_camera':True}},
           'ice_polygons_sample':[[[0,300],[960,300],[960,540],[0,540]]],
           'foreground_polygons_sample':[[[200,100],[700,100],[700,540],[200,540]]],
           'display_mask_provenance':{'source':'frozen_conservative_solver_mask'}}
    run={'source_sha256':'video-sha','model_sha256':'model-sha','frames':[frame]}
    dump_json(path,run)
    original_bytes=path.read_bytes();cache=LocalizationCache(tmp_path)
    old=cache.player(path)
    assert old['frames'][0]['foreground_polygons_sample']==frame['foreground_polygons_sample']
    sidecar=path.with_name('display_masks.json')
    display={'schema_version':'rinkspace.display-masks.v1','localization_sha256':sha256(path),
             'source_sha256':'video-sha','model_sha256':'model-sha','frames':[{
                 'pts':10,'time_base':'1/50','ice_polygons_sample':frame['ice_polygons_sample'],
                 'foreground_polygons_sample':[[[410,150],[450,150],[450,500],[410,500]]],
                 'display_mask_provenance':{'source':'display_only_full_body_mask'},
                 'H_image_from_model':[[9,0,0],[0,9,0],[0,0,1]]}]}
    dump_json(sidecar,display)
    changed=cache.player(path)
    view=changed['frames'][0]
    assert view['foreground_polygons_sample']==display['frames'][0]['foreground_polygons_sample']
    for key in ('pts','time_base','status','H_image_from_model','projection_chain','scene_observations'):
        assert view[key]==frame[key]
    assert view['display_mask_provenance']==display['frames'][0]['display_mask_provenance']
    original_load=module.load_json
    with monkeypatch.context() as patch:
        def cached_only(p):
            assert Path(p) not in {path,sidecar},'A cached page must not reparse evidence or its sidecar'
            return original_load(p)
        patch.setattr(module,'load_json',cached_only)
        assert cache.player(path)==changed
    display['frames'][0]['foreground_polygons_sample']=[]
    dump_json(sidecar,display)
    assert cache.player(path)['frames'][0]['foreground_polygons_sample']==[]
    display['model_sha256']='different-model'
    dump_json(sidecar,display)
    with pytest.raises(ValueError,match='identity differs'):
        cache.player(path)
    sidecar.unlink()
    assert cache.player(path)==old
    assert path.read_bytes()==original_bytes


def test_player_preserves_visible_landmark_evidence_and_failures_without_a_pose(tmp_path):
    path=tmp_path/'outputs/whole_rink_localization/run/localization.json'
    observations=[{
        'observation_id':'arc_7','source_pts':2560,'source_time_base':'1/12800',
        'points_source_px':[[420.25,500.5],[438.75,512.0]],
        'visible_segments_source_px':[[[420.25,500.5],[438.75,512.0]],[[480,531],[488,536]]],
        'candidate_model_ids':['R2','R3'],'selected_model_id':None,
        'association_status':'detected_unidentified','used_by_solver':False,
        'identity_basis':['two_equal_rings_not_disambiguated'],'manual_confirmed':False,
        'rejection_stage':'association','rejection_reason':'multiple_model_ids',
        'measurement_uncertainty_source_px':3.5,'group_weight':0,
        'residual_summary_sample_px':None,
    },{'observation_id':'masked_arc','association_status':'masked_out','used_by_solver':False,
       'rejection_stage':'mask','rejection_reason':'foreground_occlusion','mask_fraction':.81}]
    frame={'pts':2560,'time_base':'1/12800','source_time_s':.2,'status':'unavailable',
           'projection_chain':None,'landmark_observations':observations,
           'image_scale_from_source':.5,'landmark_detection_summary':{'detected':1,'masked':1},
           'scene_observations':{'board_surface':{'points_source_px':[[5,6]]}}}
    run={'model_id':'m','model_revision':'v07','source_sha256':'original',
         'landmark_observation_schema':'rinkspace.landmarks.v1','frames':[frame]}
    dump_json(path,run);before=path.read_bytes();cache=LocalizationCache(tmp_path)
    first=cache.player(path)
    assert first['landmark_observation_schema']==run['landmark_observation_schema']
    assert first['frames'][0]==frame
    assert cache.player(path)==first
    assert path.read_bytes()==before


def test_player_omits_rich_landmark_diagnostics_without_losing_any_frame_point_gap_or_identity(tmp_path):
    import json
    from rinkspace.player_cache import CACHE_VERSION
    path=tmp_path/'outputs/whole_rink_localization/run/localization.json'
    points=[[123.125+i*.0625,456.375+i*.03125] for i in range(101)]
    observation={'observation_id':'source:2560:arc_0','source_sha256':'source',
        'source_pts':2560,'source_time_base':'1/12800','source_frame_index':10,
        'kind':'arc_or_line','surface_type':'ice_plane','colour_class':'gray',
        'image_domain':'raw_source','image_scale_from_source':[1,1],
        'points_source_px':points,'visible_segment_ranges':[[0,35],[41,101]],
        'visible_segment_range_convention':'half_open_indices_into_points_source_px',
        'candidate_model_ids':['F2','F3'],'selected_model_id':'F2',
        'association_status':'associated','used_by_solver':True,
        'manual_confirmed':True,'user_confirmed':False,
        'confirmation_actor':'agent_source_reading_not_user_confirmation',
        'not_automatic':True,'coordinate_domain':'native_distorted_source_pixels',
        'rejection_stage':None,'rejection_reason':None,
        'measurement_uncertainty_source_px':4.0,'group_weight':.5,
        'ellipse_candidate':{'large_fit_diagnostics':'x'*50000},
        'brightness_side_evidence':[{'gradients_source_xy':[[1.1,2.2]]*1000}],
        'association_diagnostics':{'all_camera_alternatives':'y'*50000}}
    frames=[{'pts':2560+i*256,'time_base':'1/12800','source_time_s':.2+i*.02,
             'landmark_observations':[copy.deepcopy(observation)]} for i in range(3)]
    run={'source_sha256':'source','landmark_mode':'manual_source_reading_development_diagnostic','frames':frames}
    dump_json(path,run);before=path.read_bytes();cache=LocalizationCache(tmp_path)
    # A populated v4 cache must be invalidated even with unchanged input stats.
    stat=path.stat()
    dump_json(cache._path(path,'player'),{'source_stat':{'version':4,'mtime_ns':stat.st_mtime_ns,
        'bytes':stat.st_size,'display_masks':None},'data':{'frames':[]}})
    player=cache.player(path)
    assert CACHE_VERSION>=5 and len(player['frames'])==len(frames)
    assert player['landmark_mode']==run['landmark_mode']
    for actual,original in zip(player['frames'],frames):
        assert actual['pts']==original['pts']
        row=actual['landmark_observations'][0]
        for key,value in observation.items():
            if key not in ('ellipse_candidate','brightness_side_evidence','association_diagnostics'):
                assert row[key]==value
            else:
                assert key not in row
    assert len(json.dumps(player))<len(json.dumps(run))*.1
    assert path.read_bytes()==before
    assert cache.player(path)==player


@pytest.mark.parametrize('dimensions',[
    {'width':1920,'height':1080},
    {'source_size':[1920,1080]},
    {'width':1920,'height':1080,'source_size':[1920,1080]},
])
def test_missing_projection_scale_uses_explicit_dimensions_only_in_display(tmp_path,monkeypatch,dimensions):
    import rinkspace.player_cache as module
    path=tmp_path/'outputs/run/localization.json'
    frame={'pts':256,'time_base':'1/12800','status':'image_tracked',**dimensions,
           'sample_size':[960,540],'projection_chain':{'current_sample_size':[960,540],
               'H_current_sample_from_reference_raw':[[1,0,0],[0,1,0],[0,0,1]]}}
    run={'model_sha256':'frozen-model','source_sha256':'frozen-source','frames':[frame]}
    untouched=copy.deepcopy(run);dump_json(path,run);before=path.read_bytes()
    cache=LocalizationCache(tmp_path);stat=path.stat()
    dump_json(cache._path(path,'player'),{'source_stat':{'version':7,'mtime_ns':stat.st_mtime_ns,
        'bytes':stat.st_size,'display_masks':None},'data':{'frames':[]}})
    original_load=module.load_json
    monkeypatch.setattr(module,'load_json',lambda p:run if Path(p)==path else original_load(p))
    payload=cache.player(path);view=payload['frames'][0]
    assert module.CACHE_VERSION>=8
    assert view['projection_chain']['H_source_image_from_sample']==[[2,0,0],[0,2,0],[0,0,1]]
    assert view['display_adapter']['status']=='applied'
    assert view['display_adapter']['source_size']==[1920,1080]
    assert view['display_adapter']['inference_unchanged'] is True
    for key in ('pts','time_base','status'):assert view[key]==frame[key]
    for key in ('source_sha256','model_sha256'):assert payload[key]==run[key]
    assert run==untouched  # In particular, the nested original chain was not mutated.
    assert path.read_bytes()==before
    assert cache.player(path)==payload


@pytest.mark.parametrize('dimensions,chain_extra,reason',[
    ({},{},'Missing explicit native'),
    ({'image_width':1920,'image_height':1080},{},'Missing explicit native'),
    ({'width':1920},{},'positive finite integer'),
    ({'width':1920,'height':1080,'source_size':[1280,720]},{},'Conflicting explicit native'),
    ({'source_size':[1920,1080],'source_width':1280,'source_height':720},{},'Conflicting explicit native'),
    ({'source_size':[0,1080]},{},'positive finite integer'),
    ({'source_size':[True,1080]},{},'positive finite integer'),
    ({'source_size':['1920',1080]},{},'positive finite integer'),
    ({'source_size':[1920.5,1080]},{},'positive finite integer'),
    ({'source_size':[1920,1080]},{'current_sample_size':None},'current_sample_size'),
    ({'source_size':[1920,1080]},{'current_sample_size':[960,0]},'positive finite integer'),
    ({'source_size':[1920,1080],'sample_size':[1280,720]},{},'Conflicting explicit current sample'),
])
def test_missing_projection_scale_rejects_unknown_invalid_or_conflicting_sizes(tmp_path,dimensions,chain_extra,reason):
    path=tmp_path/'outputs/run/localization.json'
    chain={'current_sample_size':[960,540],**chain_extra}
    frame={'pts':256,'time_base':'1/12800',**dimensions,'projection_chain':chain}
    dump_json(path,{'frames':[frame]});before=path.read_bytes()
    view=LocalizationCache(tmp_path).player(path)['frames'][0]
    assert view['projection_chain']==chain
    assert 'H_source_image_from_sample' not in view['projection_chain']
    assert view['display_adapter']['status']=='rejected'
    assert reason in view['display_adapter']['reason']
    assert path.read_bytes()==before


def test_projection_scale_already_recorded_is_never_replaced_by_display_adapter(tmp_path):
    path=tmp_path/'outputs/run/localization.json'
    frame={'pts':1,'time_base':'1/50','width':1920,'height':1080,
           'projection_chain':{'current_sample_size':[960,540],
               'H_source_image_from_sample':[[1.5,0,0],[0,1.5,0],[0,0,1]]}}
    dump_json(path,{'frames':[frame]})
    assert LocalizationCache(tmp_path).player(path)['frames'][0]==frame


def test_actual_okada_3044_frozen_candidate_receives_display_scale(tmp_path):
    root=Path(__file__).resolve().parents[1]
    evidence=resolve_asset(root,root/'outputs/whole_rink_localization/okada_camera_reuse_v11_rectified/localization.json')
    if not evidence.is_file():pytest.skip('Local OKADA evidence is not present in this checkout')
    before=sha256(evidence);run=load_json(evidence)
    frame=next(row for row in run['frames'] if abs(row['source_time_s']-30.44)<1e-8)
    assert frame['width']==1920 and frame['height']==1080
    assert frame['projection_chain']['current_sample_size']==[960,540]
    assert frame['projection_chain'].get('H_source_image_from_sample') is None
    path=tmp_path/'outputs/okada/localization.json'
    dump_json(path,{**{k:v for k,v in run.items() if k!='frames'},'frames':[frame]})
    payload=LocalizationCache(tmp_path).player(path);view=payload['frames'][0]
    expected={**frame['projection_chain'],'H_source_image_from_sample':[[2,0,0],[0,2,0],[0,0,1]]}
    assert view['projection_chain']==expected
    assert view['display_adapter']['status']=='applied'
    assert (view['pts'],view['time_base'],view['status'])==(frame['pts'],frame['time_base'],frame['status'])
    for key in ('model_sha256','source_sha256'):assert payload.get(key)==run.get(key)
    assert sha256(evidence)==before
