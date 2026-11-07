"""Explicit old-candidate registration keeps source/result identities separate."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from calibration.program_tracking.body_tracking import programme_player as player


@pytest.fixture
def loc(tmp_path):
    result, comparison = tmp_path/'new.npz', tmp_path/'old.npz'
    result.write_bytes(b'new saved result')
    comparison.write_bytes(b'old saved P Q and equipment')
    value = player.locations(tmp_path, tmp_path/'data', result=result,
        comparison_result=comparison, prepared=tmp_path/'prepared',
        output=tmp_path/'player', config=tmp_path/'config.json')
    value['prepared'].mkdir()
    (value['prepared']/'input.npz').write_bytes(b'input')
    (value['prepared']/'compact_smplx.npz').write_bytes(b'compact')
    value.update(template=tmp_path/'template.json', scope=tmp_path/'scope.json',
                 navigation=tmp_path/'navigation.json')
    player.write(value['template'], dict(schema='test', variants=[dict(id='old_default')], default_variant='old_default'))
    player.write(value['scope'], dict(shots=[]))
    return value


def test_comparison_is_explicit_and_keeps_primary_first_and_default(loc, capsys):
    before = loc['template'].read_bytes()
    primary = player.make_config(dict(loc, comparison_result=None))
    compared = player.make_config(loc)
    assert len(primary['variants']) == 1
    assert primary['source_review_model_body_default'] is True
    assert primary['shot'] == primary['session_id'] == player.SESSION
    assert loc['variant_id'] == player.VARIANT
    assert loc['comparison_variant_id'] == player.COMPARISON_VARIANT
    assert loc['session_id'] == player.SESSION
    assert compared['variants'][0] == primary['variants'][0]
    assert compared['default_variant'] == primary['default_variant'] == player.VARIANT
    old = compared['variants'][1]
    assert old['id'] == player.COMPARISON_VARIANT
    assert old['result'] == str(loc['comparison_result'])
    assert '无本轮钢刃源诊断' in old['label'] and old['blade_review_available'] is False
    player.main(['commands', '--project', str(loc['project']), '--data-root', str(loc['store'].data),
                 '--result', str(loc['result']), '--comparison-result', str(loc['comparison_result'])])
    assert capsys.readouterr().out.count('--comparison-result') == 3
    assert loc['template'].read_bytes() == before
    assert not loc['config'].exists()


@pytest.mark.parametrize('defect', ['pts', 'P', 'Q', 'shoe'])
def test_comparison_checks_its_own_clock_root_and_equipment(loc, monkeypatch, defect):
    n = 8329
    pts = 283136+np.arange(n, dtype=np.int64)*256
    shapes = dict(P_world_mid=(n, 3), native22_world_mid=(n, 22, 3),
        COCO_world_mid=(n, 17, 3), boot_world_mid=(n, 2, 14, 3),
        blade_world_mid=(n, 2, 34, 3), gear_world_mid=(n, 2, 48, 3), G_world_mid=(n, 3))
    saved = {key: np.broadcast_to(0., shape) for key, shape in shapes.items()}
    saved.update(pts=pts, time_base=np.array([1, 12800]),
        Q_world_mid=np.broadcast_to(np.eye(3), (n, 3, 3)),
        local_R_mid=np.broadcast_to(np.eye(3), (n, 21, 3, 3)),
        source_identity_suspect=np.zeros(n, bool), window_mask=np.zeros(n, bool),
        source_lower_body_inferred=np.zeros(n, bool))
    comparison = dict(saved)
    packet = dict(pts=pts, time_base=np.array([1, 12800]))
    assets = {loc['prepared']/'input.npz': packet, loc['result']: saved,
              loc['comparison_result']: comparison}
    monkeypatch.setattr(player, 'load', lambda path: assets[Path(path)])
    monkeypatch.setattr(player, 'required_clock', lambda scope: (pts, np.resize(np.arange(9), n), (1, 12800)))
    player.checked(loc)
    if defect == 'pts': comparison['pts'] = pts+256
    elif defect == 'P': comparison['P_world_mid'] = np.full((n, 3), np.nan)
    elif defect == 'Q': comparison['Q_world_mid'] = np.zeros((n, 3, 3))
    else: comparison['boot_world_mid'] = np.ones((n, 2, 14, 3))
    with pytest.raises((AssertionError, ValueError)):
        player.checked(loc)


@pytest.mark.parametrize('custom', [False, True])
@pytest.mark.parametrize('mapping', ['missing', 'bound', 'wrong_sha', 'wrong_primary_sha'])
def test_registration_binds_old_sha_and_prepares_in_order_without_projection_alias(loc, monkeypatch, mapping, custom):
    from rinkspace.viewer3d import data, prepare
    from rinkspace.viewer3d.blade_review import BladeReview
    if custom:
        loc.update(variant_id='programme_source_1107', comparison_variant_id='programme_v6_1106', session_id='source_1107')
    primary_id, comparison_id = loc['variant_id'], loc['comparison_variant_id']
    pts = np.arange(9, dtype=np.int64)*256+283136
    packet = dict(pts=pts, time_base=np.array([1, 12800]), source_identity_suspect=np.zeros(9, bool))
    monkeypatch.setattr(player, 'checked', lambda where: (packet, {}, np.arange(9)))
    calls = []

    class Catalog:
        def __init__(self, **kwargs):
            self.config = player.read(kwargs['config'])
            self.pts = pts

        def parameters(self, variant): calls.append(('parameters', variant))
        def navigation_points(self): return []
        def variant(self, name): return next(v for v in self.config['variants'] if v['id'] == name)

    monkeypatch.setattr(data, 'ResultCatalog', Catalog)
    monkeypatch.setattr(prepare, 'prepare_variant',
        lambda catalog, variant: calls.append(('prepare', variant)) or loc['output']/variant)
    models = {primary_id: 'new'}
    if mapping != 'missing': models[comparison_id] = 'old'
    arrays = dict(pts=pts)
    for prefix, value in [('new', 10.), ('old', 20.)]:
        rate = np.zeros((len(pts), 2)); rate[0] = np.nan
        arrays.update({prefix+'_blade_uv': np.full((len(pts), 2, 3, 2), value),
            prefix+'_boot_uv': np.full((len(pts), 2, 2, 2), value), prefix+'_axis_rate_rad_s': rate})
    projection = loc['project']/'projection.npz'
    np.savez_compressed(projection, **arrays)
    asset = loc['project']/'blade.json'
    old_sha = player.sha(loc['comparison_result'])
    player.write(asset, dict(schema='rinkspace.blade-player-review.v1', projection_npz=str(projection),
        time_base=[1, 12800], variants=models,
        prediction_binding=dict(result_sha256=player.sha(loc['result'])),
        prediction_bindings={comparison_id: dict(result_sha256='wrong' if mapping == 'wrong_sha' else old_sha),
            **({primary_id: dict(result_sha256='wrong' if mapping == 'wrong_primary_sha' else player.sha(loc['result']))}
               if custom or mapping == 'wrong_primary_sha' else {})}))
    loc['blade_review_asset'] = asset
    if mapping in ('wrong_sha', 'wrong_primary_sha'):
        with pytest.raises(ValueError, match='Comparison blade review' if mapping == 'wrong_sha' else 'primary variant'):
            player.register(loc)
        assert not loc['config'].exists() and not calls
        return
    report = player.register(loc)
    assert report['comparison_result_sha256'] == old_sha
    assert report['comparison_blade_review_available'] == (mapping == 'bound')
    config = player.read(loc['config'])
    assert config['default_variant'] == primary_id
    assert [v['id'] for v in config['variants']] == [primary_id, comparison_id]
    model = BladeReview(asset, assets=loc['store']).at(comparison_id, pts[0])['model']
    if mapping == 'missing':
        assert model is None and '无本轮钢刃源诊断' in config['variants'][1]['label']
    else:
        np.testing.assert_array_equal(model['boot_uv'], np.full((2, 2, 2), 20.))
        assert '独立保存' in config['variants'][1]['label']
    cached = player.prepare_caches(loc)
    assert calls == [('parameters', primary_id), ('parameters', comparison_id),
                     ('prepare', primary_id), ('prepare', comparison_id)]
    assert [v['variant'] for v in cached['variants']] == [primary_id, comparison_id]
    assert cached['session_id'] == config['session_id'] == config['shot'] == loc['session_id']
    tampered = player.read(loc['config'])
    tampered['variants'][0]['result'] = str(loc['comparison_result'])
    player.write(loc['config'], tampered)
    with pytest.raises(ValueError, match='primary configuration'):
        player.prepare_caches(loc)
    player.write(loc['config'], config)
    with pytest.raises(ValueError, match='IDs/session'):
        player.prepare_caches(dict(loc, session_id='other_session'))
    with pytest.raises(FileExistsError, match='registered player IDs/session'):
        player.register(dict(loc, session_id='other_session'))
    assert len(calls) == 4
    assert player.sha(loc['comparison_result']) == old_sha
    loc['comparison_result'].write_bytes(b'changed saved result')
    with pytest.raises(ValueError, match='comparison result differs'):
        player.prepare_caches(loc)
    assert len(calls) == 4


@pytest.mark.parametrize('field', ['variant_id', 'comparison_variant_id', 'session_id'])
@pytest.mark.parametrize('bad', ['', 'slash/id', 'has-dash', '非ASCII'])
def test_stable_identifiers_reject_unsafe_values(tmp_path, field, bad):
    with pytest.raises(ValueError, match='Invalid stable player'):
        player.locations(tmp_path, tmp_path/'data', config=tmp_path/'new.json', **{field: bad})


def test_custom_identifiers_require_explicit_new_config_and_reproduce_commands(loc, capsys):
    kwargs = dict(variant_id='source_1107', comparison_variant_id='v6_1106', session_id='source_session_1107')
    with pytest.raises(ValueError, match='explicit --config'):
        player.locations(loc['project'], loc['store'].data, **kwargs)
    with pytest.raises(ValueError, match='1106 configuration'):
        player.locations(loc['project'], loc['store'].data, config=loc['project']/player.PROGRAMME/player.CONFIG, **kwargs)
    with pytest.raises(ValueError, match='must differ'):
        player.locations(loc['project'], loc['store'].data, config=loc['config'], comparison_result=loc['comparison_result'],
                         variant_id='same', comparison_variant_id='same')
    player.main(['commands', '--project', str(loc['project']), '--data-root', str(loc['store'].data),
        '--config', str(loc['config']), '--variant-id', kwargs['variant_id'],
        '--comparison-variant-id', kwargs['comparison_variant_id'], '--session-id', kwargs['session_id']])
    printed = capsys.readouterr().out
    for flag, value in [('variant-id', kwargs['variant_id']), ('comparison-variant-id', kwargs['comparison_variant_id']), ('session-id', kwargs['session_id'])]:
        assert printed.count('--'+flag+' "'+value+'"') == 3
    assert '-Config "'+str(loc['config'])+'" -Variant "source_1107"' in printed
    assert not loc['config'].exists()


def test_custom_session_isolates_real_catalog_cache_and_workbench_roots(loc, monkeypatch):
    from rinkspace.viewer3d import data
    monkeypatch.setattr(data, 'estimate_uniform_scale', lambda board: 1.)
    loc['output'].mkdir()
    np.savez_compressed(loc['output']/'input.npz', pts=np.array([283136,283392]), time_base=[1,12800], metres_per_scene_unit=1.)
    camera = loc['project']/'camera.npz';np.savez_compressed(camera, board=np.array([.5,.25]))
    model = loc['project']/'model.npz';model.write_bytes(b'local model identity')
    player.write(loc['template'], dict(schema='test', camera_snapshot=str(camera), model_asset=str(model), display_metres_per_scene_unit=1.))
    custom = player.locations(loc['project'], loc['store'].data, prepared=loc['prepared'], result=loc['result'],
        output=loc['output'], config=loc['project']/'custom.json', variant_id='source_1107',
        comparison_variant_id='v6_1106', session_id='source_session_1107')
    custom.update(template=loc['template'], scope=loc['scope'])
    assert custom['navigation'] != loc['navigation']
    roots = []
    for where in (loc, custom):
        player.write(where['config'], player.make_config(where))
        catalog = data.ResultCatalog(project=where['project'], data_root=where['store'].data, config=where['config'])
        assert catalog.config['default_variant'] == where['variant_id']
        roots.append((catalog.cache_root, catalog.session_root))
    assert roots[0][0] != roots[1][0] and roots[0][1] != roots[1][1]
    assert roots[1][0].name == roots[1][1].name == 'source_session_1107'


@pytest.fixture
def diagnostic_case(loc,monkeypatch):
    import hashlib,json
    loc.update(variant_id='primary_1107',comparison_variant_id='before_1106',session_id='review_1107',comparison_result=None)
    n=8329;pts=np.arange(n,dtype=np.int64)*256+283136
    shots=np.concatenate([np.repeat('s'+str(i+1).zfill(2),len(indices))for i,indices in enumerate(np.array_split(np.arange(n),9))])
    monkeypatch.setattr(player,'required_clock',lambda scope:(pts,shots,(1,12800)))
    camera=dict(shots={str(shot):dict(state_sha256='a'*64)for shot in np.unique(shots)},metres_per_scene_unit=60.)
    camera_json=np.array(json.dumps(camera))
    base=dict(pts=pts,source_pts=pts,time_base=np.array([1,12800]),source_shot_ids=shots,
        P_world_mid=np.broadcast_to(0.,(n,3)),native22_world_mid=np.broadcast_to(0.,(n,22,3)),
        COCO_world_mid=np.broadcast_to(0.,(n,17,3)),boot_world_mid=np.broadcast_to(0.,(n,2,14,3)),
        blade_world_mid=np.broadcast_to(0.,(n,2,34,3)),gear_world_mid=np.broadcast_to(0.,(n,2,48,3)),G_world_mid=np.broadcast_to(0.,(n,3)),
        Q_world_mid=np.broadcast_to(np.eye(3),(n,3,3)),local_R_mid=np.broadcast_to(np.eye(3),(n,21,3,3)),
        source_identity_suspect=np.zeros(n,bool),source_lower_body_inferred=np.zeros(n,bool),window_mask=np.zeros(n,bool),
        COCO_uv=np.broadcast_to(0.,(n,17,2)),equipment_uv=np.broadcast_to(0.,(n,2,48,2)),
        shared_betas=np.zeros(10),shared_body_scale=np.array(1.),metres_per_scene_unit=np.array(60.),camera_binding_json=camera_json)
    context=dict(schema='rinkspace.programme-context.v1',source_sha256='b'*64,source_frames=n,first_pts=int(pts[0]),last_pts=int(pts[-1]),time_base=[1,12800],shared_body_id='logical_person_0',metres_per_scene_unit=60.,frozen_camera_sha256='c'*64,world='same frozen scene',no_per_shot_world_alignment=True,
        shots=[dict(shot_id=str(shot),valid_pts=[int(pts[shots==shot][0]),int(pts[shots==shot][-1])],frozen_camera_state_sha256='a'*64)for shot in np.unique(shots)])
    player.write(loc['scope'],dict(source=dict(sha256='b'*64),shots=[]))
    primary=dict(base,programme_context_json=np.array(json.dumps(context)))
    packet=dict(pts=pts,time_base=np.array([1,12800]),shared_betas=np.zeros(10))
    arrays={loc['prepared']/'input.npz':packet,loc['result']:primary}
    records=[]
    for i,scale in enumerate((.8,.9,1.1,1.2)):
        candidate='branch_'+str(i);result=loc['project']/(candidate+'.npz');parameter=loc['project']/(candidate+'_parameter.npz')
        result.write_bytes(('synthetic saved result '+candidate).encode());parameter.write_bytes(('synthetic absolute state '+candidate).encode())
        beta=np.zeros(10);beta[0]=i*.1
        branch_context=dict(context,shared_shape_sha256=hashlib.sha256(np.asarray(np.r_[beta,scale],dtype='<f8').tobytes()).hexdigest(),continuous_state_registry=dict(parameter_sha256=player.sha(parameter)))
        arrays[result]=dict(base,shared_betas=beta,shared_body_scale=np.array(scale),programme_context_json=np.array(json.dumps(branch_context)))
        arrays[parameter]=dict(state_kind=np.array('absolute_motion_v1'),camera_binding_json=camera_json)
        record=loc['project']/(candidate+'.json')
        player.write(record,dict(candidate_id=candidate,variant_id=candidate,result=str(result),result_sha256=player.sha(result),parameter=str(parameter),parameter_sha256=player.sha(parameter),programme_context=branch_context,user_accepted=False,accuracy_accepted=False,totalcost=-1000*i))
        records.append(record)
    loc['diagnostic_records']=records
    monkeypatch.setattr(player,'load',lambda path:arrays[Path(path)])
    return loc,arrays,primary


def test_diagnostic_shapes_can_differ_without_selecting_a_default(diagnostic_case,capsys):
    loc,arrays,primary=diagnostic_case
    bindings=player.checked_diagnostics(loc,primary)
    config=player.make_config(loc,diagnostics=bindings)
    assert len(bindings)==4 and len(config['variants'])==5
    assert config['default_variant']=='primary_1107'
    for index,variant in enumerate(config['variants'][1:]):
        assert variant['id']=='branch_'+str(index) and variant['kind']=='fullbody'
        assert variant['conditional_diagnostic'] and '诊断' in variant['label']
        assert variant['accuracy_accepted'] is False and variant['blade_review_available'] is False
    player.main(['commands','--project',str(loc['project']),'--data-root',str(loc['store'].data),'--config',str(loc['config']),
        '--variant-id','primary_1107','--session-id','review_1107',
        *[value for path in loc['diagnostic_records']for value in ('--diagnostic-record',str(path))]])
    assert capsys.readouterr().out.count('--diagnostic-record')==12
    assert not loc['config'].exists()


@pytest.mark.parametrize('defect',['result_sha','parameter_sha','id','clock','identity','camera'])
def test_diagnostic_records_reject_wrong_sha_id_clock_or_identity(diagnostic_case,defect):
    import json
    loc,arrays,primary=diagnostic_case;path=loc['diagnostic_records'][0];record=player.read(path)
    if defect=='result_sha':Path(record['result']).write_bytes(b'changed result')
    elif defect=='parameter_sha':Path(record['parameter']).write_bytes(b'changed parameter')
    elif defect=='id':record['variant_id']=loc['variant_id'];player.write(path,record)
    elif defect=='clock':arrays[Path(record['result'])]['pts']=arrays[Path(record['result'])]['pts']+256
    else:
        context=record['programme_context']
        if defect=='identity':context['shared_body_id']='another_person'
        else:context['shots'][0]['frozen_camera_state_sha256']='different_camera'
        arrays[Path(record['result'])]['programme_context_json']=np.array(json.dumps(context));player.write(path,record)
    with pytest.raises((ValueError,AssertionError)):player.checked_diagnostics(loc,primary)
    assert not loc['config'].exists()


def test_diagnostics_require_a_new_session_and_cannot_alias_review_predictions(loc):
    with pytest.raises(ValueError,match='new custom config/session'):
        player.locations(loc['project'],loc['store'].data,config=loc['config'],diagnostic_records=[loc['project']/'future.json'])
    entry=dict(variant_id='branch_0',result_sha256='result',parameter_sha256='parameter')
    review=SimpleNamespace(models={'branch_0':{}},variants={'primary':'main','branch_0':'main'})
    asset=dict(prediction_bindings={'branch_0':dict(result_sha256='result',parameter_sha256='parameter')})
    with pytest.raises(ValueError,match='aliases'):player._diagnostic_review_status([entry],asset,review)
    review.variants['branch_0']='own';asset['prediction_bindings']['branch_0']['parameter_sha256']='primary_parameter'
    with pytest.raises(ValueError,match='own result and parameter'):player._diagnostic_review_status([entry],asset,review)


def test_registration_and_cache_prepare_recheck_all_diagnostic_bindings(diagnostic_case,monkeypatch):
    from rinkspace.viewer3d import data,prepare
    loc,arrays,primary=diagnostic_case;calls=[];pts=primary['pts']
    class Catalog:
        def __init__(self,**kwargs):self.config=player.read(kwargs['config']);self.pts=pts
        def variant(self,name):return next(v for v in self.config['variants']if v['id']==name)
        def parameters(self,name):calls.append(('parameters',name))
        def navigation_points(self):return []
    monkeypatch.setattr(data,'ResultCatalog',Catalog)
    monkeypatch.setattr(prepare,'prepare_variant',lambda catalog,name:calls.append(('prepare',name))or loc['output']/name)
    first=player.read(loc['diagnostic_records'][0]);projection=loc['project']/'merged.npz'
    rate=np.zeros((len(pts),2));rate[0]=np.nan;values=dict(pts=pts)
    for prefix in ('main','own'):
        values.update({prefix+'_blade_uv':np.zeros((len(pts),2,3,2)),prefix+'_boot_uv':np.zeros((len(pts),2,2,2)),prefix+'_axis_rate_rad_s':rate})
    np.savez_compressed(projection,**values)
    asset=loc['project']/'merged.json';loc['blade_review_asset']=asset
    player.write(asset,dict(schema='rinkspace.blade-player-review.v1',time_base=[1,12800],projection_npz=str(projection),
        variants={loc['variant_id']:'main','branch_0':'own'},prediction_bindings={loc['variant_id']:dict(result_sha256=player.sha(loc['result'])),
        'branch_0':dict(result_sha256=first['result_sha256'],parameter_sha256=first['parameter_sha256'])}))
    receipt=player.register(loc);config=player.read(loc['config'])
    assert len(receipt['diagnostics'])==4 and config['default_variant']==loc['variant_id']
    assert config['variants'][1]['blade_review_available'] is True
    assert all(v['blade_review_available'] is False for v in config['variants'][2:])
    # Reading an existing small player input is part of idempotent registration.
    arrays[loc['output']/'input.npz']=player.load(loc['prepared']/'input.npz')
    caches=player.prepare_caches(loc)
    assert [item['variant']for item in caches['variants']]==['primary_1107','branch_0','branch_1','branch_2','branch_3']
    assert calls==[('parameters',name)for name in ['primary_1107','branch_0','branch_1','branch_2','branch_3']]+[('prepare',name)for name in ['primary_1107','branch_0','branch_1','branch_2','branch_3']]
    tampered=player.read(loc['config']);tampered['variants'][2]['blade_review_available']=True
    player.write(loc['config'],tampered)
    with pytest.raises(ValueError,match='conditional/review status'):player.prepare_caches(loc)
    player.write(loc['config'],config)
    Path(first['parameter']).write_bytes(b'changed after registration')
    with pytest.raises(ValueError,match='parameter SHA'):player.prepare_caches(loc)
    assert len(calls)==10
