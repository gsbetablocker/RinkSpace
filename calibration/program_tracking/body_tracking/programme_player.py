"""Register/prepare a saved nine-shot player candidate; never fit or open a GUI.

commands works before the result exists. register creates only a small player
input and registration metadata. prepare is an explicit display-cache action.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from .sequence_delivery import PROGRAMME, read, write, sha, guard, load
from .programme_contract import required_clock

VARIANT = 'programme_trajectory'
COMPARISON_VARIANT = 'programme_before_trajectory'
CONFIG = 'viewer3d_programme_trajectory_1106.json'
NAVIGATION = 'programme_trajectory_1106_navigation.json'
SESSION = 'programme_trajectory_1106'
TEMPLATE = 'viewer3d_source_fusion_support_1106.json'


def locations(project, data_root=None, *, prepared=None, result=None, output=None, config=None, blade_review_asset=None, comparison_result=None, variant_id=VARIANT, comparison_variant_id=COMPARISON_VARIANT, session_id=SESSION, diagnostic_records=None):
    for label, value in (('variant_id', variant_id), ('comparison_variant_id', comparison_variant_id), ('session_id', session_id)):
        if not isinstance(value, str) or re.fullmatch(r'[A-Za-z0-9_]+', value) is None:
            raise ValueError('Invalid stable player ' + label + ': use only letters, digits and underscores')
    if comparison_result and variant_id == comparison_variant_id:
        raise ValueError('Primary and comparison variant IDs must differ')
    custom = (variant_id, comparison_variant_id, session_id) != (VARIANT, COMPARISON_VARIANT, SESSION)
    if custom and config is None:
        raise ValueError('Custom player IDs require an explicit --config path')
    diagnostic_records = [Path(path).resolve() for path in (diagnostic_records or [])]
    if diagnostic_records and (session_id == SESSION or config is None):
        raise ValueError('Diagnostic records require a new custom config/session, never the default 1106 session')
    project = Path(project).resolve()
    config_path = Path(config).resolve() if config else project / PROGRAMME / CONFIG
    if custom and config_path == project / PROGRAMME / CONFIG:
        raise ValueError('Preserve the 1106 configuration; choose a new explicit --config path')
    sys.path[:0] = [str(project), str(project / 'src')]
    from rinkspace.storage import AssetStore
    store = AssetStore(project, data_root=data_root)
    root = store.data / 'runs/body_s01_20261029/trajectory_first_1106'
    result_path = Path(result).resolve() if result else None
    return dict(project=project, store=store, variant_id=variant_id,
        comparison_variant_id=comparison_variant_id, session_id=session_id, diagnostic_records=diagnostic_records,
        prepared=Path(prepared).resolve() if prepared else root / 'programme',
        result=result_path,
        comparison_result=Path(comparison_result).resolve() if comparison_result else None,
        output=Path(output).resolve() if output else (result_path.parent.parent / 'player' if result_path else None),
        config=config_path,
        navigation=config_path.with_name(session_id + '_navigation.json') if custom else project / PROGRAMME / NAVIGATION,
        template=project / PROGRAMME / TEMPLATE,
        scope=project / PROGRAMME / 'rebuild_scope.json',
        blade_review_asset=Path(blade_review_asset).resolve() if blade_review_asset else None)


def navigation_document(scope):
    shots = scope['shots']
    cuts = []
    for before, after in zip(shots[:-1], shots[1:]):
        first, last = after['processing']['first_pts'], before['processing']['last_pts']
        cuts.append(dict(label=f"{before['id'].upper()} → {after['id'].upper()} | {first / 12800:.3f}s",
            previous_shot_id=before['id'], shot_id=after['id'], previous_pts=last, pts=first))
    return dict(schema='rinkspace.viewer-source-navigation.v1', time_base=[1, 12800],
        description_zh='八次摄影切镜；切前与切后都定位原片精确PTS，不换算近邻帧。', cuts=cuts)


def make_config(loc, pts=None, *, comparison_review_available=False, diagnostics=None, diagnostic_review_available=()):
    """Use the existing effective scene/model/source paths, without its S05 layers."""
    old = read(loc['template'])
    keys = ('schema', 'event', 'programme', 'scene_root', 'scene_label', 'metric_binding',
            'metric_binding_base', 'model_asset', 'camera_snapshot', 'display_metres_per_scene_unit',
            'source_video', 'source_time_base', 'diagnostic_semantics')
    config = {key: copy.deepcopy(old[key]) for key in keys if key in old}
    config.update(shot=loc['session_id'], source_run=str(loc['output']), session_id=loc['session_id'],
        default_variant=loc['variant_id'], initial_source_pts=int(pts[0]) if pts is not None else 283136,
        initial_view='side', initial_wear_inspection=False, source_identity_field='source_identity_suspect',
        source_navigation=str(loc['navigation']), source_programme_scope=str(loc['scope']),
        source_review_model_body_default=True,
        body_window_scope='九镜头8329曝光的统一连续开发候选；身体、鞋刀与接触准确性未验收。',
        body_scope_label='全节目九镜头连续开发候选；本帧是否参与本轮优化按保存mask显示。',
        footwear_scope_label='装备按本轮保存数组显示；已保存的unknown不由显示端补造。',
        display_outside_window_label='保存的全节目初状态；本帧未参与本轮优化。',
        display_footwear_binding_scope='全节目保存的装备应用mask',
        review_status='全节目开发候选；不表示独立三维、接触或逐曝光视觉准确性通过。',
        initial_visible=dict(surface=True, skeleton=True, shoes=True, trajectory=True, pelvis=True,
                             com=False, blade_material=True, contact=True),
        variants=[dict(id=loc['variant_id'], label='九镜头连续轨迹与全身开发候选（未验收）', kind='fullbody',
            result=str(loc['result']), shell_topology='saved', body_reference_label='programme_initial',
            body_refinement_intervals_seconds=[[22.12, 188.68]])])
    if loc.get('comparison_result'):
        review_label = '本候选独立保存的钢刃诊断' if comparison_review_available else '无本轮钢刃源诊断'
        config['variants'].append(dict(id=loc['comparison_variant_id'],
            label='旧全节目候选（原保存状态；'+review_label+'）', kind='fullbody',
            result=str(loc['comparison_result']), shell_topology='saved',
            body_reference_label='programme_initial',
            source_projection_origin='own saved COCO_uv/equipment_uv',
            blade_review_available=comparison_review_available))
    if diagnostics is None:
        diagnostics = checked_diagnostics(loc)
    for diagnostic in diagnostics:
        available = diagnostic['variant_id'] in diagnostic_review_available
        config['variants'].append(dict(id=diagnostic['variant_id'],
            label='诊断对照 '+diagnostic['candidate_id']+'（未验收；'+('本候选独立保存的钢刃诊断' if available else '无本候选钢刃源诊断')+'）',
            kind='fullbody', result=diagnostic['result'], shell_topology='saved',
            body_reference_label='programme_initial', conditional_diagnostic=True,
            user_accepted=False, accuracy_accepted=False, blade_review_available=available,
            source_projection_origin='own saved COCO_uv/equipment_uv',
            diagnostic_record=diagnostic['record'], diagnostic_record_sha256=diagnostic['record_sha256'],
            parameter=diagnostic['parameter'], parameter_sha256=diagnostic['parameter_sha256']))
    if loc.get('blade_review_asset'):
        config.update(blade_review_asset=str(loc['blade_review_asset']), source_review_model_runner_default=True)
    return config


def checked(loc):
    """Read the actual saved result; no geometry, network, camera fit or writes."""
    if loc['result'] is None:
        raise ValueError('An explicit completed --result is required')
    for path in (loc['result'], loc['prepared'] / 'input.npz', loc['prepared'] / 'compact_smplx.npz'):
        if not path.is_file():
            raise FileNotFoundError('Actual saved programme artifact is absent: ' + str(path))
    packet, saved = load(loc['prepared'] / 'input.npz'), load(loc['result'])
    expected, shots, tb = required_clock(read(loc['scope']))
    pts = np.asarray(saved['pts'])
    np.testing.assert_array_equal(pts, expected)
    np.testing.assert_array_equal(packet['pts'], pts)
    if 'source_pts' in saved:
        np.testing.assert_array_equal(saved['source_pts'], pts)
    if 'source_shot_ids' in saved:
        np.testing.assert_array_equal(saved['source_shot_ids'], shots)
    for item in (packet, saved):
        value = item.get('time_base', item.get('source_time_base'))
        if value is None:
            raise ValueError('Programme input/result must declare its native source time base')
        np.testing.assert_array_equal(value, tb)
    flags = np.asarray(saved['source_identity_suspect'])
    if flags.shape != pts.shape or flags.dtype.kind != 'b':
        raise ValueError('Saved source identity warnings must be boolean on the complete result PTS')
    if len(pts) != 8329 or len(set(shots.tolist())) != 9:
        raise ValueError('Expected all nine shots and 8329 native exposures')
    from rinkspace.viewer3d.data import _array, _rotations, hand_state, footwear_state
    n = len(pts)
    for key, shape in (('P_world_mid', (n, 3)), ('native22_world_mid', (n, 22, 3)),
            ('COCO_world_mid', (n, 17, 3)), ('boot_world_mid', (n, 2, 14, 3)),
            ('blade_world_mid', (n, 2, 34, 3)), ('gear_world_mid', (n, 2, 48, 3)),
            ('G_world_mid', (n, 3))):
        _array(saved[key], shape, key)
    _rotations(saved['Q_world_mid'], (n, 3, 3), 'saved world root')
    _rotations(saved['local_R_mid'], (n, 21, 3, 3), 'saved local rotations')
    for key in ('window_mask', 'source_lower_body_inferred'):
        if np.asarray(saved[key]).shape != (n,) or saved[key].dtype.kind != 'b':
            raise ValueError('Invalid saved programme mask: ' + key)
    np.testing.assert_array_equal(saved['gear_world_mid'][:, :, :14], saved['boot_world_mid'])
    np.testing.assert_array_equal(saved['gear_world_mid'][:, :, 14:], saved['blade_world_mid'])
    hand_state(saved, n, dict(id=loc['variant_id']))
    footwear_state(saved, n, 1.)
    packet = dict(packet, pts=pts.copy(), time_base=np.asarray(tb, np.int64),
                  source_identity_suspect=flags.copy())
    if loc.get('comparison_result'):
        if loc['comparison_result'] == loc['result']:
            raise ValueError('Comparison result must be a distinct saved candidate')
        # Check the old candidate's own complete clock, P/Q and equipment.
        checked(dict(loc, result=loc['comparison_result'], comparison_result=None, variant_id=loc['comparison_variant_id']))
    return packet, saved, shots


def _saved_json(saved, key):
    value=np.asarray(saved.get(key))
    if value.shape!=() or value.dtype.kind!='U':
        raise ValueError('Missing/invalid saved diagnostic '+key)
    document=json.loads(str(value))
    if not isinstance(document,dict):raise ValueError('Expected diagnostic JSON object: '+key)
    return document


def _diagnostic_headers(loc):
    paths=loc.get('diagnostic_records',[])
    if not paths:return []
    if loc['session_id']==SESSION or loc['config']==loc['project']/PROGRAMME/CONFIG:
        raise ValueError('Diagnostic records require a new custom config/session')
    ids={loc['variant_id'],loc['comparison_variant_id']};candidates=set();results={loc['result'],loc.get('comparison_result')}
    headers=[]
    for path in paths:
        path=Path(path).resolve();record=read(path)
        for key in ('candidate_id','variant_id'):
            if not isinstance(record.get(key),str) or re.fullmatch(r'[A-Za-z0-9_]+',record[key]) is None:
                raise ValueError('Invalid diagnostic '+key)
        if record['variant_id'] in ids or record['candidate_id'] in candidates or record['candidate_id'] in {loc['variant_id'],loc['comparison_variant_id']}:
            raise ValueError('Duplicate/colliding diagnostic ID')
        ids.add(record['variant_id']);candidates.add(record['candidate_id'])
        if record.get('user_accepted') is not False or record.get('accuracy_accepted') is not False:
            raise ValueError('Diagnostic records must retain unaccepted conditional status')
        for key in ('result','parameter'):
            value=Path(record[key]);record[key]=str((value if value.is_absolute() else path.parent/value).resolve())
        if Path(record['result']) in results:raise ValueError('Diagnostic result aliases another player candidate')
        results.add(Path(record['result']))
        record.update(record=str(path),record_sha256=sha(path))
        headers.append(record)
    return headers


def _context_identity(context, pts, shots, tb):
    if context.get('schema')!='rinkspace.programme-context.v1' or not isinstance(context.get('shared_body_id'),str) or not context['shared_body_id']:
        raise ValueError('Missing diagnostic body logical identity')
    if (context.get('source_frames'),context.get('first_pts'),context.get('last_pts'),context.get('time_base'))!=(len(pts),int(pts[0]),int(pts[-1]),list(tb)):
        raise ValueError('Diagnostic context source clock differs')
    metric=context.get('metres_per_scene_unit')
    if isinstance(metric,bool) or not isinstance(metric,(int,float)) or not np.isfinite(metric) or metric<=0:
        raise ValueError('Invalid diagnostic scene metric binding')
    actual=context.get('shots',[])
    expected=[dict(shot_id=str(shot),valid_pts=[int(pts[shots==shot][0]),int(pts[shots==shot][-1])])for shot in dict.fromkeys(shots.tolist())]
    if len(actual)!=len(expected) or len(actual)!=9:
        raise ValueError('Diagnostic context must cover all nine frozen shots')
    by_id={row['shot_id']:row for row in actual}
    if len(by_id)!=9:raise ValueError('Duplicate diagnostic context shots')
    for row in expected:
        camera=by_id.get(row['shot_id'],{})
        if camera.get('valid_pts')!=row['valid_pts'] or not camera.get('frozen_camera_state_sha256'):
            raise ValueError('Diagnostic frozen camera/shot coverage differs')
    keys=('source_sha256','shared_body_id','metres_per_scene_unit','frozen_camera_sha256','world','no_per_shot_world_alignment')
    if any(key not in context for key in keys) or context['no_per_shot_world_alignment'] is not True:
        raise ValueError('Incomplete diagnostic common world/camera identity')
    return {**{key:context[key]for key in keys},'shots':sorted(actual,key=lambda row:row['shot_id'])}


def checked_diagnostics(loc, primary_saved=None):
    """Read actual finalized conditional branches; never choose by fit cost."""
    headers=_diagnostic_headers(loc)
    if not headers:return []
    primary_saved=load(loc['result']) if primary_saved is None else primary_saved
    scope=read(loc['scope']);pts,shots,tb=required_clock(scope)
    primary_context=_saved_json(primary_saved,'programme_context_json')
    common=_context_identity(primary_context,pts,shots,tb)
    if common['source_sha256']!=scope['source']['sha256']:
        raise ValueError('Primary context belongs to another source')
    primary_camera=_saved_json(primary_saved,'camera_binding_json')
    if set(primary_camera.get('shots',{}))!=set(shots.tolist()):
        raise ValueError('Primary must bind the nine frozen camera contents')
    expected_beta_shape=np.asarray(primary_saved.get('shared_betas',np.zeros(10))).shape
    bindings=[]
    for record in headers:
        for key in ('result','parameter'):
            if not Path(record[key]).is_file() or sha(record[key])!=record.get(key+'_sha256'):
                raise ValueError('Diagnostic '+key+' SHA differs from formal record')
        _,saved,_=checked(dict(loc,result=Path(record['result']),variant_id=record['variant_id'],comparison_result=None,diagnostic_records=[]))
        context=record.get('programme_context',{})
        if _context_identity(context,pts,shots,tb)!=common:
            raise ValueError('Diagnostic logical body/world/metric/frozen cameras differ from primary')
        if _saved_json(saved,'programme_context_json')!=context:
            raise ValueError('Diagnostic result context differs from its formal record')
        if _saved_json(saved,'camera_binding_json')!=primary_camera:
            raise ValueError('Diagnostic saved frozen camera content differs from primary')
        np.testing.assert_array_equal(saved['source_shot_ids'],shots)
        np.testing.assert_array_equal(saved['time_base'],tb)
        if float(np.asarray(saved['metres_per_scene_unit']))!=common['metres_per_scene_unit']:
            raise ValueError('Diagnostic saved scene metric differs')
        beta=np.asarray(saved['shared_betas']);scale=np.asarray(saved['shared_body_scale'])
        if beta.shape!=expected_beta_shape or beta.ndim!=1 or beta.dtype.kind not in 'fiu' or not np.isfinite(beta).all():
            raise ValueError('Invalid diagnostic shared betas')
        if scale.shape!=() or scale.dtype.kind not in 'fiu' or not np.isfinite(scale) or float(scale)<=0:
            raise ValueError('Invalid diagnostic shared body scale')
        shape_sha=hashlib.sha256(np.asarray(np.r_[beta,scale],dtype='<f8').tobytes()).hexdigest()
        if context.get('shared_shape_sha256')!=shape_sha:
            raise ValueError('Diagnostic shape hash differs from saved betas/scale')
        for key,shape in [('COCO_uv',(len(pts),17,2)),('equipment_uv',(len(pts),2,48,2))]:
            value=np.asarray(saved[key])
            if value.shape!=shape or not np.isfinite(value).all():raise ValueError('Invalid diagnostic own projection: '+key)
        parameter=load(record['parameter'])
        if str(parameter.get('state_kind'))!='absolute_motion_v1' or _saved_json(parameter,'camera_binding_json')!=primary_camera:
            raise ValueError('Diagnostic parameter state/frozen camera differs')
        if context.get('continuous_state_registry',{}).get('parameter_sha256')!=record['parameter_sha256']:
            raise ValueError('Diagnostic context is bound to another parameter')
        bindings.append({key:record[key]for key in ('record','record_sha256','candidate_id','variant_id','result','result_sha256','parameter','parameter_sha256')})
    return bindings


def variant_ids(loc, diagnostics=None):
    ids=[loc['variant_id'],loc['comparison_variant_id']] if loc.get('comparison_result') else [loc['variant_id']]
    return ids+[entry['variant_id']for entry in (_diagnostic_headers(loc) if diagnostics is None else diagnostics)]


def _diagnostic_review_status(diagnostics,asset,review):
    available=[]
    for entry in diagnostics:
        variant=entry['variant_id']
        if variant not in review.models:continue
        binding=asset.get('prediction_bindings',{}).get(variant,{})
        if any(binding.get(key)!=entry[key]for key in ('result_sha256','parameter_sha256')):
            raise ValueError('Diagnostic review must bind its own result and parameter SHA')
        prefix=review.variants[variant]
        if list(review.variants.values()).count(prefix)!=1:
            raise ValueError('Diagnostic review aliases another candidate projection')
        available.append(variant)
    return available


def register(loc):
    """Bind a completed saved candidate without preparing full meshes."""
    packet, saved, shots = checked(loc)
    diagnostics = checked_diagnostics(loc, saved)
    diagnostic_review_available = []
    out = loc['output']
    result_sha = sha(loc['result'])
    comparison_sha = sha(loc['comparison_result']) if loc.get('comparison_result') else None
    comparison_review_available = False
    asset_binding = None
    if loc.get('blade_review_asset'):
        from rinkspace.viewer3d.blade_review import BladeReview
        asset_path = loc['blade_review_asset']
        asset = read(asset_path)
        binding = asset.get('prediction_bindings', {}).get(loc['variant_id'])
        if binding is None and loc['variant_id'] == VARIANT:
            binding = asset.get('prediction_binding', {})
        if (binding or {}).get('result_sha256') != result_sha:
            raise ValueError('Blade review predictions belong to another saved result or primary variant')
        review = BladeReview(asset_path, assets=loc['store'], time_base=(1, 12800))
        if loc['variant_id'] not in review.models or set(review.index) != set(map(int, packet['pts'])):
            raise ValueError('Blade review must cover this variant and every programme PTS')
        if comparison_sha and loc['comparison_variant_id'] in review.models:
            binding = asset.get('prediction_bindings', {}).get(loc['comparison_variant_id'], {})
            if binding.get('result_sha256') != comparison_sha:
                raise ValueError('Comparison blade review predictions belong to another saved result')
            comparison_review_available = True
        diagnostic_review_available = _diagnostic_review_status(diagnostics, asset, review)
        asset_binding = dict(path=str(asset_path), sha256=sha(asset_path))
    config = make_config(loc, packet['pts'], comparison_review_available=comparison_review_available,
                         diagnostics=diagnostics, diagnostic_review_available=diagnostic_review_available)
    receipt = out / 'player_registration.json'
    if receipt.exists():
        previous = read(receipt)
        if previous.get('diagnostics', []) != diagnostics:
            raise FileExistsError('Preserve the registered diagnostic branches; choose --output and --config')
        for key, default in (('variant_id', VARIANT), ('comparison_variant_id', COMPARISON_VARIANT), ('session_id', SESSION)):
            if previous.get(key, default) != loc[key]:
                raise FileExistsError('Preserve the registered player IDs/session; choose --output and --config')
    if receipt.exists() and read(receipt).get('result_sha256') != result_sha:
        raise FileExistsError('Preserve the previous registered candidate; choose --output and --config')
    if receipt.exists() and read(receipt).get('comparison_result_sha256') not in (None, comparison_sha):
        raise FileExistsError('Preserve the previous registered comparison; choose --output and --config')
    for target in (loc['config'], loc['navigation']):
        guard(target, loc['project'])
    if loc['config'].exists():
        old = read(loc['config'])
        if old != config:
            raise FileExistsError('Preserve unrelated existing configuration: ' + str(loc['config']))
    out.mkdir(parents=True, exist_ok=True)
    target = out / 'input.npz'
    if target.exists():
        current = load(target)
        if set(current) != set(packet):
            raise FileExistsError('Preserve an incompatible existing player input: ' + str(target))
        for key in packet:
            np.testing.assert_array_equal(current[key], packet[key], err_msg='Existing player input differs: ' + key)
    else:
        np.savez_compressed(target, **packet)
    compact = out / 'compact_smplx.npz'
    source_compact = loc['prepared'] / 'compact_smplx.npz'
    if compact.exists():
        if sha(compact) != sha(source_compact):
            raise FileExistsError('Existing compact model has a different identity')
    else:
        os.link(source_compact, compact)
    navigation = navigation_document(read(loc['scope']))
    if (loc['prepared']/'action_input_identity.json').is_file():
        doc=read(loc['prepared']/'source_evidence.json')
        contexts=doc['source_episode_review']['whole_programme_coverage_intervals']
        labels={'first_combination':'3F+3T 连跳','axel':'2A','flying_camel':'FCSp 跳入与抓握',
                'lutz':'3Lz','step_sequence':'步法段','layback':'LSp 左足旋',
                'change_foot_spin':'CCoSp 换足旋'}
        navigation['events']=[dict(label=labels[e['episode_id']],start_pts=e['start_pts'],
            end_pts=e['end_pts_inclusive']) for e in contexts if e['episode_id'] in labels]
    write(loc['navigation'], navigation)
    write(loc['config'], config)
    from rinkspace.viewer3d.data import ResultCatalog
    catalog = ResultCatalog(project=loc['project'], data_root=loc['store'].data, config=loc['config'])
    for variant in variant_ids(loc, diagnostics):
        catalog.parameters(variant)
    catalog.navigation_points()
    if navigation.get('events'):
        catalog.navigation_points('events')
    report = dict(schema='rinkspace.programme-player-registration.v1', frames=len(packet['pts']),
        shots={str(s): int(np.count_nonzero(shots == s)) for s in dict.fromkeys(shots.tolist())},
        first_last_pts=packet['pts'][[0, -1]].tolist(), time_base=[1, 12800],
        result=str(loc['result']), result_sha256=result_sha, config=str(loc['config']),
        variant_id=loc['variant_id'], comparison_variant_id=loc['comparison_variant_id'], session_id=loc['session_id'],
        prepared=str(loc['prepared']), player_input=str(target), player_input_sha256=sha(target),
        compact_smplx_sha256=sha(compact), identity_warning_origin='actual saved result',
        source_configuration=dict(path=str(loc['template']), sha256=sha(loc['template'])),
        blade_review_asset=asset_binding, diagnostics=diagnostics,
        no_fit=True, no_network=True, no_gui=True, no_mesh_preparation=True, accuracy_accepted=False)
    if comparison_sha:
        report.update(comparison_variant=loc['comparison_variant_id'],
            comparison_result=str(loc['comparison_result']), comparison_result_sha256=comparison_sha,
            comparison_blade_review_available=comparison_review_available,
            comparison_source_projection_origin='own saved COCO_uv/equipment_uv')
    write(receipt, report)
    return report


def prepare_caches(loc):
    """Explicitly reuse the current mesh-cache preparer after registration."""
    _, saved, _ = checked(loc)
    diagnostics = checked_diagnostics(loc, saved)
    receipt = read(loc['output'] / 'player_registration.json')
    if receipt.get('diagnostics', []) != diagnostics:
        raise ValueError('Diagnostic record/result/parameter differs from registration')
    if receipt['result_sha256'] != sha(loc['result']):
        raise ValueError('Saved result changed after player registration')
    for key, default in (('variant_id', VARIANT), ('comparison_variant_id', COMPARISON_VARIANT), ('session_id', SESSION)):
        if receipt.get(key, default) != loc[key]:
            raise ValueError('Player IDs/session differ from registration')
    comparison_sha = sha(loc['comparison_result']) if loc.get('comparison_result') else None
    if receipt.get('comparison_result_sha256') != comparison_sha:
        raise ValueError('Saved comparison result differs from player registration; pass its registered --comparison-result')
    from rinkspace.viewer3d.data import ResultCatalog
    from rinkspace.viewer3d.prepare import prepare_variant
    catalog = ResultCatalog(project=loc['project'], data_root=loc['store'].data, config=loc['config'])
    if catalog.config.get('session_id') != loc['session_id'] or catalog.config.get('shot') != loc['session_id']:
        raise ValueError('Registered configuration points to another player/cache session')
    if catalog.config.get('default_variant') != loc['variant_id'] or Path(catalog.variant(loc['variant_id'])['result']).resolve() != loc['result']:
        raise ValueError('Registered primary configuration points to another saved result or variant')
    if comparison_sha and Path(catalog.variant(loc['comparison_variant_id'])['result']).resolve() != loc['comparison_result']:
        raise ValueError('Registered comparison configuration points to another saved result')
    expected_ids=variant_ids(loc, diagnostics)
    if [entry['id']for entry in catalog.config['variants']]!=expected_ids:
        raise ValueError('Registered diagnostic variant list differs')
    for entry in diagnostics:
        variant=catalog.variant(entry['variant_id'])
        if Path(variant['result']).resolve()!=Path(entry['result']) or variant.get('parameter_sha256')!=entry['parameter_sha256'] or variant.get('diagnostic_record_sha256')!=entry['record_sha256']:
            raise ValueError('Registered diagnostic configuration points to another candidate')
    if diagnostics:
        available=[];binding=receipt.get('blade_review_asset')
        asset_path=loc.get('blade_review_asset')
        if bool(binding)!=bool(asset_path) or (binding and binding.get('path')!=str(asset_path)):
            raise ValueError('Pass the registered diagnostic --blade-review-asset')
        if asset_path:
            from rinkspace.viewer3d.blade_review import BladeReview
            if binding.get('sha256')!=sha(asset_path):raise ValueError('Diagnostic merged review changed after registration')
            review=BladeReview(asset_path,assets=loc['store'],time_base=(1,12800))
            available=_diagnostic_review_status(diagnostics,read(asset_path),review)
        for entry in diagnostics:
            variant=catalog.variant(entry['variant_id'])
            if (variant.get('kind')!='fullbody' or variant.get('conditional_diagnostic') is not True
                    or variant.get('user_accepted') is not False or variant.get('accuracy_accepted') is not False
                    or variant.get('blade_review_available')!=(entry['variant_id'] in available)):
                raise ValueError('Diagnostic conditional/review status differs from verified bindings')
    caches = []
    for variant in expected_ids:
        path = prepare_variant(catalog, variant)
        caches.append(dict(variant=variant, path=str(path)))
    report = dict(path=caches[0]['path'], frames=len(catalog.pts), variant=loc['variant_id'], session_id=loc['session_id'], no_gui=True)
    if comparison_sha or diagnostics:
        report.update(variants=caches, diagnostics=diagnostics)
    if comparison_sha:
        report.update(comparison_result_sha256=comparison_sha)
    write(loc['output'] / 'display_caches.json', report)
    return report


def commands(loc):
    python = loc['store'].data / 'runtime_tools/vision_cuda/Scripts/python.exe'
    prefix = f'& "{python}" -B -m calibration.program_tracking.body_tracking.programme_player'
    arguments = f' --project "{loc["project"]}" --data-root "{loc["store"].data}"'
    arguments += f' --result "{loc["result"] or "SAVED_RESULT_NPZ"}"'
    arguments += f' --prepared "{loc["prepared"]}" --config "{loc["config"]}"'
    arguments += f' --variant-id "{loc["variant_id"]}" --comparison-variant-id "{loc["comparison_variant_id"]}" --session-id "{loc["session_id"]}"'
    if loc['output'] is not None:
        arguments += f' --output "{loc["output"]}"'
    if loc.get('comparison_result'):
        arguments += f' --comparison-result "{loc["comparison_result"]}"'
    for record in loc.get('diagnostic_records', []):
        arguments += f' --diagnostic-record "{record}"'
    if loc.get('blade_review_asset'):
        arguments += f' --blade-review-asset "{loc["blade_review_asset"]}"'
    for action in ('check', 'register', 'prepare'):
        print(prefix + ' ' + action + arguments)
    print('Explicit GUI entry after preparation: powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "' + str(loc['project'] / 'launchers/START_3D_RESULTS.ps1') + '" -Config "' + str(loc['config']) + '" -Variant "' + loc['variant_id'] + '" -Fresh')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['commands', 'check', 'register', 'prepare'])
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--data-root', type=Path)
    for name in ('prepared', 'result', 'output', 'config', 'blade-review-asset', 'comparison-result'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--diagnostic-record', type=Path, action='append', default=[])
    parser.add_argument('--variant-id', default=VARIANT)
    parser.add_argument('--comparison-variant-id', default=COMPARISON_VARIANT)
    parser.add_argument('--session-id', default=SESSION)
    args = parser.parse_args(argv)
    if args.action != 'commands' and args.result is None:
        parser.error('--result is required; only register an actual completed saved candidate')
    loc = locations(args.project, args.data_root, prepared=args.prepared, result=args.result,
                    output=args.output, config=args.config, blade_review_asset=args.blade_review_asset,
                    comparison_result=args.comparison_result, variant_id=args.variant_id,
                    comparison_variant_id=args.comparison_variant_id, session_id=args.session_id, diagnostic_records=args.diagnostic_record)
    if args.action == 'commands':
        commands(loc)
    elif args.action == 'check':
        packet, saved, shots = checked(loc)
        diagnostics = checked_diagnostics(loc, saved)
        print(json.dumps(dict(frames=len(packet['pts']), shots=len(set(shots.tolist())), diagnostics=diagnostics, no_writes=True)))
    else:
        print(json.dumps(register(loc) if args.action == 'register' else prepare_caches(loc), ensure_ascii=False))


if __name__ == '__main__':
    main()
