"""Proven shape quotient, shared tangent assembly, and explicit state migration."""
from copy import deepcopy
import json
import os
from pathlib import Path

import numpy as np
import pytest

from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import (SceneParameterModel, ParameterState, FactorPlan,
    load_parameter_state, write_parameter_state, migrate_parameter_state, load_parameter_stations, _identity)
from rinkspace.domain.shape_chart import (compile_shape_local_chart, shape_group_definition,
    expand_shape_state, encode_shape_state, migrate_shape_stations, validate_shape_local_chart)
from rinkspace.domain.scene_assets import snapshot_scene
from rinkspace.domain.print_assets import sample_template_paths, sample_instance
from rinkspace.review.render import load_state
from rinkspace.session.dataset import load_prepared, read_json
from rinkspace.session.prepare import prepare
from test_print_measurement import FIELD, make_print_input

ROOT = Path(__file__).resolve().parents[1]
LPATH = 'Ad7.top_12_L_part00'


@pytest.fixture(scope='module')
def asset():
    return read_json(ROOT/'events/ankara_2026/scene/ad7/assets_r4.json')


@pytest.fixture(scope='module')
def chart(asset):
    template = asset['templates'][0]
    internal = template['internal_geometry']
    group = next(g for g in internal['groups'] if LPATH in g['path_ids'])
    return compile_shape_local_chart(shape_group_definition(template, group), internal['coordinate_gauge'])


@pytest.fixture(scope='module')
def model(chart):
    return SceneParameterModel(read_json(FIELD), FIELD, local_charts={chart['parameter_block_id']:chart})


@pytest.fixture(scope='module')
def core():
    library = os.environ.get('RINKSPACE_CORE_LIBRARY')
    if not library:
        pytest.skip('Explicit native library required')
    return NativeCore(library)


def test_exact_l_rank_and_serialized_expansion(chart):
    assert chart['ambient_size'] == 10 and chart['tangent_size'] == 6
    assert [a['rank'] for a in chart['axes']] == [1, 1]
    assert chart['fixed_affine_indices'] == []
    validate_shape_local_chart(chart)
    for axis in chart['axes']:
        b = np.asarray(axis['B'])
        np.testing.assert_allclose(b @ np.ones(4), 0., atol=2e-16)
        np.testing.assert_allclose(b @ b.T, np.eye(len(b)), atol=2e-15)
    np.testing.assert_array_equal(expand_shape_state(chart, np.zeros(6)), np.zeros(10))


def test_tiny_slanted_edge_is_not_rounded_into_axis_alignment(chart):
    definition = deepcopy(chart['definition'])
    definition['paths'][0]['material_uv'][1][1] += 1e-15
    with pytest.raises(ValueError, match='Non-axis-aligned'):
        compile_shape_local_chart(definition, chart['coordinate_gauge'])


def test_true_fixed_material_coordinate_adds_information(chart):
    definition = deepcopy(chart['definition'])
    p = definition['paths'][0]['material_uv']
    definition['fixed_material_points'] = [dict(id='actual/material/mark', path_id=LPATH,
        material_uv=[(p[4][0]+p[5][0])/2, p[4][1]])]
    constrained = compile_shape_local_chart(definition, chart['coordinate_gauge'])
    assert constrained['axes'][0]['rank'] == 2
    assert constrained['tangent_size'] == 7


def test_open_endpoints_and_holes_are_not_dropped():
    outer = dict(id='outer',closed=True,material_uv=[[0,0],[1,0],[1,1],[0,1],[0,0]])
    hole = dict(id='hole',closed=True,parent_contour_key='outer',
        material_uv=[[.2,.3],[.8,.3],[.8,.7],[.2,.7],[.2,.3]])
    open_path = dict(id='open',closed=False,material_uv=[[.4,.5],[.6,.5]])
    group = dict(id='group',parameter_block_id='shape',path_ids=['outer','hole','open'],
        reference_bounds=[0,0,1,1],parameterization='monotone_axis_warp.v1')
    template = dict(paths=[outer,hole,open_path])
    definition = shape_group_definition(template,group)
    c = compile_shape_local_chart(definition,dict(kind='unit_panel_chart'))
    assert .4 in c['axes'][0]['q'] and .6 in c['axes'][0]['q']
    assert .3 in c['axes'][1]['q'] and .7 in c['axes'][1]['q']
    group['path_ids'] = ['outer','open']
    with pytest.raises(ValueError, match='parent/hole'):
        shape_group_definition(template,group)


def test_nonzero_positive_states_keep_turn_domain(chart,core):
    rng = np.random.default_rng(58)
    vertices = chart['definition']['paths'][0]['material_uv']
    bounds = chart['definition']['reference_bounds']
    for _ in range(30):
        old = rng.normal(0,1.5,10); old[:4] *= .02
        new = expand_shape_state(chart, encode_shape_state(chart,old))
        a = core.deform_print_points(vertices,bounds,old)
        b = core.deform_print_points(vertices,bounds,new)
        assert not np.any(a['status']) and not np.any(b['status'])
        np.testing.assert_allclose(a['material_uv'],b['material_uv'],rtol=0,atol=2e-14)


@pytest.mark.parametrize('instance',['Ad7','Ad20'])
def test_local_columns_assemble_at_nonzero_state(model,chart,core,instance):
    identity = chart['parameter_block_id']; state = model.initial_state
    state = state.retract_block(identity,[.012,-.008,.05,-.04,.3,-.23])
    path = 'print_instance/'+instance+'::'+LPATH
    stations = [.041,.217,.381,.527,.719,.913]
    result = model.evaluate_print(path,stations,core,state=state)
    assert result['active_jacobians_by_block'][identity].shape == (6,3,6)
    direction = np.zeros(state.layout['tangent_size'])
    predicted = np.zeros((6,3))
    for block,columns in result['parameter_columns'].items():
        direction[columns] = np.sin(columns+.37)
        predicted += result['active_jacobians_by_block'][block] @ direction[columns]
    epsilon = 2e-7
    plus = model.evaluate_print(path,stations,core,state=state.retract(epsilon*direction),jacobian=False)['xyz']
    minus = model.evaluate_print(path,stations,core,state=state.retract(-epsilon*direction),jacobian=False)['xyz']
    np.testing.assert_allclose(predicted,(plus-minus)/(2*epsilon),rtol=2e-5,atol=2e-8)


def test_shared_chart_independent_installations_and_packed_columns(model,chart):
    state = model.initial_state
    assert len(state.values) == 1946 and len(state.active_columns()) == 1843
    np.testing.assert_array_equal(state.active_columns(),np.arange(1843))
    a,b = [model.factor_plan.columns('print_instance/'+i+'::'+LPATH) for i in ('Ad7','Ad20')]
    identity = chart['parameter_block_id']
    np.testing.assert_array_equal(a[identity],b[identity])
    assert set(a)-set(b) == {'installation/Ad7/shared_placement'}
    plan = model.factor_plan.to_dict()
    entry = next(b for b in plan['blocks'] if b['id'] == identity)
    assert entry['chart_revision'] == chart['revision'] and entry['ambient_size'] == 10
    assert entry['tangent_size'] == 6 and plan['column_space'] == 'packed_tangent'


def test_counterexample_requires_explicit_station_migration(model,chart,core):
    legacy = SceneParameterModel(model.scene,FIELD).initial_state
    identity = chart['parameter_block_id']
    old = legacy[identity].copy(); old[4:7] = np.log(np.array([.25,.25,.35])/.15)
    legacy = legacy.with_blocks({identity:old})
    with pytest.raises(ValueError,match='Noncanonical'):
        model.initial_state.with_blocks({identity:old})
    binding = dict(id='migration/fixture',block_id=identity,path_id=LPATH,
        stations=[.041,.217,.381,.527,.719,.913],pts=1015552,noise_group_id='preserved/group',
        source_uv=[[100.,200.]],W=[[1.,0.]],user_approval='unchanged_fixture')
    state,bindings = migrate_parameter_state(legacy,{identity:chart},station_bindings=[binding],core=core)
    np.testing.assert_allclose(state.local_values(identity),np.zeros(6),atol=1e-14)
    assert {k:v for k,v in bindings[0].items() if k!='stations'} == {k:v for k,v in binding.items() if k!='stations'}
    assert bindings[0]['stations'] != binding['stations']
    for instance in ('Ad7','Ad20'):
        path = 'print_instance/'+instance+'::'+LPATH
        old_model = SceneParameterModel(model.scene,FIELD)
        a = old_model.evaluate_print(path,binding['stations'],core,state=legacy,jacobian=False)['xyz']
        b = model.evaluate_print(path,bindings[0]['stations'],core,state=state,jacobian=False)['xyz']
        np.testing.assert_allclose(a,b,rtol=0,atol=1e-14)


@pytest.mark.parametrize('damage',['offset','matrix','chart'])
def test_rehashed_invalid_local_layout_is_rejected(model,chart,damage):
    layout = model.initial_state.layout
    block = next(b for b in layout['blocks'] if b['id']==chart['parameter_block_id'])
    if damage=='offset': block['tangent_offset'] += 1
    elif damage=='matrix': block['d_ambient_d_local'][4][4] += .1
    else: block['local_chart']['u_basis'][0][0] += .1
    layout.pop('layout_revision');layout['layout_revision']=_identity(layout)
    with pytest.raises(ValueError): ParameterState(layout,model.initial_state.values)


def test_candidate_station_roundtrip_d1_and_complete_review(tmp_path,chart,asset,core):
    scene,_ = snapshot_scene(FIELD,tmp_path/'scene'); field=tmp_path/'scene/scene.json'
    model = SceneParameterModel(scene,field,local_charts={chart['parameter_block_id']:chart})
    state = model.initial_state.retract_block(chart['parameter_block_id'],[.012,-.008,.05,-.04,.3,-.23])
    bindings = [dict(id='station/fixture',block_id=chart['parameter_block_id'],path_id=LPATH,stations=[.23],pts=1015552)]
    write_parameter_state(field.parent,state,name='candidate.json',station_bindings=bindings,
        required_shot_ids=['s%02d'%i for i in range(1,10)],selected_shot_ids=['s05'])
    loaded=load_parameter_state(field.parent/'candidate.json',expected_scene=scene,expected_scene_path=field)
    np.testing.assert_array_equal(loaded.values,state.values)
    assert load_parameter_stations(field.parent/'candidate.json') == bindings
    _,_,samples,_=load_state(field.parent/'candidate.json',core,[])
    assert samples['inventory']['unlocalized_print_instance_ids']==['print_instance/Ad15','print_instance/Ad16']
    template=asset['templates'][0];size=np.exp(state[template['size_class_id']])
    drawn=sample_template_paths(asset,template['id'],core,size,parameter_state=loaded)
    drawn=next(p for p in drawn if p['id']==LPATH)
    result=model.evaluate_print('print_instance/Ad7::'+LPATH,drawn['stations'],core,state=loaded)
    np.testing.assert_allclose(drawn['material_uv'],result['material_uv'],rtol=0,atol=0)
    cut=next(c for c in samples['cuts'] if c['id']=='print_instance/Ad7::'+LPATH)
    assert np.isfinite(samples['xyz'][cut['begin']:cut['end']]).all()


def test_prepare_reload_uses_same_factor_plan_preserves_source_arrays(tmp_path,chart):
    root,out=tmp_path/'input',tmp_path/'prepared'
    expected,_,members,weights,_=make_print_input(root)
    prepare(root/'input.json',root/'scope.json','s01',out,local_charts={chart['parameter_block_id']:chart})
    manifest,arrays=load_prepared(out)
    for key,value in expected.items():np.testing.assert_array_equal(arrays[key],value)
    np.testing.assert_array_equal(arrays['noise_group_members'],members)
    np.testing.assert_array_equal(arrays['noise_group_weights'],weights)
    plan=read_json(out/manifest['factor_plan_ref'])
    assert plan['column_space']=='packed_tangent'
    plan['layout_revision']='wrong'
    (out/manifest['factor_plan_ref']).write_text(json.dumps(plan),encoding='utf-8')
    with pytest.raises(ValueError,match='factor plan'):load_prepared(out)


def test_proven_axis_catalog_uses_same_layout():
    from rinkspace.domain.shape_chart import load_shape_local_charts
    charts = load_shape_local_charts(FIELD.parent.parent / "ad7/local_charts.json")
    assert len(charts) == 5
    assert sorted(c["tangent_size"] for c in charts.values()) == [6, 7, 7, 7, 9]
    current = SceneParameterModel(read_json(FIELD), FIELD, local_charts=charts)
    assert current.initial_state.layout["ambient_size"] == 1946
    assert current.initial_state.layout["tangent_size"] == 1833
    for chart in charts.values():
        assert len(current.factor_plan.columns("print_instance/Ad7::" + chart["path_ids"][0])[chart["parameter_block_id"]]) == chart["tangent_size"]
