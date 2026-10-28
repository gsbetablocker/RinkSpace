"""Whole artwork components must not regain independent glyph deformation."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pytest

from rinkspace.domain.print_assets import PrintBindingError
from rinkspace.domain.print_parameters import parameterize_print_layout, validate_print_parameters
from rinkspace.domain.scene_assets import read_scene_assets

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT/'events/ankara_2026/scene/field/model.json'


def definitions():
    scene = json.loads(SCENE.read_text(encoding='utf-8-sig'))
    plan = json.loads((SCENE.parent.parent/'print_layout_components.json').read_text(encoding='utf-8-sig'))
    by_id = {r['template_id']:r for r in plan['records']}
    return [(ref, assets, {ref['template_id']:by_id[ref['template_id']]['groups']})
            for ref, assets in read_scene_assets(scene, SCENE)]


def test_all_templates_preserve_material_paths_sources_and_installation_unknowns():
    for ref, old, layout in definitions():
        before = deepcopy(old)
        new = parameterize_print_layout(old, layout)
        assert old == before
        for a, b in zip(old['templates'], new['templates']):
            assert a['paths'] == b['paths']
        for key in ('instances', 'observations', 'sources'):
            assert new[key] == old[key]
        base = lambda a: [b for b in a['parameter_blocks'] if b['kind'] in ('print_size','print_placement')]
        for a, b in zip(base(old), base(new)):
            for key in ('id','kind','values','size'):
                assert a[key] == b[key]
        for block in new['parameter_blocks']:
            if block['kind']=='print_shape':
                assert set(block['independent_local_indices']) <= {0,1,2,3}
                assert block['values'][4:]==[0.]*6


@pytest.mark.parametrize('number,word',[ (8,'WYNDHAM'), (10,'SPORTOTO') ])
def test_letter_identity_holes_and_spacing_share_one_word_block(number, word):
    _, old, layout=next(r for r in definitions() if r[0]['advertisement_number']==number)
    new=parameterize_print_layout(old,layout)
    internal=new['templates'][0]['internal_geometry']
    group=next(g for g in internal['groups'] if word in g.get('component_identity','')
               or any(word in name for name in g.get('component_evidence',{}).get('components',[])))
    assert len(group['path_ids']) >= (9 if number==8 else 11)
    bindings=internal['path_bindings']
    assert len({bindings[p]['group_id'] for p in group['path_ids']})==1
    block=next(b for b in new['parameter_blocks'] if b['id']==group['parameter_block_id'])
    assert set(block['independent_local_indices']) <= {0,1,2,3}
    assert group['nonlinear_warp_enabled'] is False


def test_incomplete_duplicate_or_split_hole_layout_fails_instead_of_guessing():
    _, old, layout=next(r for r in definitions() if r[0]['advertisement_number']==8)
    missing=deepcopy(layout);next(iter(missing.values()))[-1]['path_ids'].pop()
    with pytest.raises(PrintBindingError):parameterize_print_layout(old,missing)
    duplicate=deepcopy(layout);g=next(iter(duplicate.values()));g[0]['path_ids'].append(g[-1]['path_ids'][0])
    with pytest.raises(PrintBindingError):parameterize_print_layout(old,duplicate)
    split=deepcopy(layout);g=next(iter(split.values()));hole=next(p for p in old['templates'][0]['paths'] if p.get('parent_contour_key'))['id']
    owner=next(d for d in g if hole in d['path_ids']);owner['path_ids'].remove(hole)
    g.append(dict(id='split',path_ids=[hole],evidence={'kind':'bad'}))
    with pytest.raises(PrintBindingError,match='splits_parent_hole'):parameterize_print_layout(old,split)


def test_layout_contract_rejects_reopened_warp_coordinate():
    _,old,layout=next(r for r in definitions() if r[0]['advertisement_number']==2)
    new=parameterize_print_layout(old,layout)
    block=next(b for b in new['parameter_blocks'] if b['kind']=='print_shape' and b['active'])
    block['independent_local_indices'].append(4);block['tangent_size']+=1
    with pytest.raises(PrintBindingError,match='freedom_contract_changed'):validate_print_parameters(new)


def test_complete_municipal_artwork_cannot_stretch_one_line_into_another():
    ref,old,layout=next(r for r in definitions() if r[0]['advertisement_number']==12)
    members=[p for group in next(iter(layout.values())) for p in group['path_ids']]
    complete={ref['template_id']:[dict(id='Ad12/complete artwork',path_ids=members,
        evidence=dict(kind='same printed municipal emblem and four text lines'))]}
    new=parameterize_print_layout(old,complete)
    template=new['templates'][0]
    ordinary=[g for g in template['internal_geometry']['groups']
              if g['parameterization']=='monotone_axis_warp.v1']
    assert len(ordinary)==1
    assert set(ordinary[0]['path_ids'])==set(members)
    block=next(b for b in new['parameter_blocks'] if b['id']==ordinary[0]['parameter_block_id'])
    # The whole-template map is a gauge already represented by size/placement.
    assert block['independent_local_indices']==[]
    assert block['tangent_size']==0
    assert any(b['kind']=='print_size' and b['active'] for b in new['parameter_blocks'])
    assert any(b['kind']=='print_placement' and b['active'] for b in new['parameter_blocks'])


def test_current_complete_artworks_do_not_reopen_separate_line_or_logo_maps():
    for ref,old,layout in definitions():
        new=parameterize_print_layout(old,layout)
        groups=[g for g in new['templates'][0]['internal_geometry']['groups']
                if g['parameterization']=='monotone_axis_warp.v1']
        assert len(groups)==1, ref['template_id']
        if ref['advertisement_number']==5:
            group=groups[0]
            assert group['fixed_in_template_chart'] is True
            block=next(b for b in new['parameter_blocks'] if b['id']==group['parameter_block_id'])
            assert block['independent_local_indices']==[]


def test_ad26_parallel_constraint_survives_artwork_recompile_and_state_load():
    from rinkspace.domain.parameters import ParameterState, SceneParameterModel
    _, old, layout = next(r for r in definitions() if r[0]['advertisement_number'] == 13)
    new = parameterize_print_layout(old, layout)
    key = 'installation/Ad26/shared_placement'
    for assets in (old, new):
        blocks = {b['id']: b for b in assets['parameter_blocks']}
        assert blocks[key]['independent_local_indices'] == [0, 1]
        assert blocks[key]['values'][2] == np.pi
        assert blocks['installation/Ad13/shared_placement']['independent_local_indices'] == [0, 1, 2]
        assert blocks['size/Ad13_Ad26']['independent_local_indices'] == [0, 1]
    scene = json.loads(SCENE.read_text(encoding='utf-8-sig'))
    state = SceneParameterModel(scene, SCENE, optimize_scene=True).initial_state
    column = state.blocks[key]['offset'] + 2
    assert column not in state.active_columns()
    bad = state.values.copy()
    bad[column] += .01
    with pytest.raises(ValueError, match='Constrained/frozen parameter coordinate changed'):
        ParameterState(state.layout, bad)


def test_physical_panel_domain_requires_explicit_boundary_and_source():
    _,old,layout=next(r for r in definitions() if r[0]['advertisement_number']==6)
    new=parameterize_print_layout(old,layout);template=new['templates'][0]
    boundary=[p['id'] for p in template['paths'] if p.get('role')=='panel_outer_boundary']
    template['material_domain']=dict(kind='physical_unit_panel',boundary_path_ids=boundary,
        evidence=dict(kind='actual_source_poster_boundary'))
    validate_print_parameters(new)
    for change in [dict(evidence={}),dict(boundary_path_ids=boundary[:-1]),dict(kind='inferred_box')]:
        bad=deepcopy(new);bad['templates'][0]['material_domain'].update(change)
        with pytest.raises(PrintBindingError,match='invalid_physical_print_panel_domain'):validate_print_parameters(bad)
