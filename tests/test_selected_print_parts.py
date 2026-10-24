from copy import deepcopy
import pytest
from rinkspace.compat.ad_assets import import_selected_parts
from rinkspace.domain.print_assets import resolve_instance


def selected():
    return {'schema':'rinkspace.print_selection.v1','advertisement_number':1,
        'template':{'id':'template','paths':[
            {'id':'letter_A','material_uv':[[0,0],[1,0],[0,1],[0,0]],'closed':True,'role':'letter_material_boundary'},
            {'id':'A_hole','material_uv':[[.2,.2],[.3,.2],[.2,.3],[.2,.2]],'closed':True,
             'role':'letter_hole_boundary','parent_contour_key':'letter_A'}]},
        'sources':{'video':{'kind':'video_frame','pts':100,'time_base':'1/100'},
                   'artwork':{'kind':'still_image','pts':None,'time_base':None}},
        'observations':[{'id':'original_fragment','source_id':'video','pts':100,'time_base':'1/100',
            'feature_id':'letter_A','instance_id':'print_instance/Ad1','source_native_uv':[[17,21],[18,22]],
            'closed':False,'active_for_fit':False}],
        'source_correlated_groups':[{'id':'original_group','members':['original_fragment']}],
        'shape_evidence':[{'source_id':'artwork','feature_id':'letter_A','programme_observation':False}],
        'provenance':[],'limitations':[],
        'primary_instance':{'id':'print_instance/Ad1','template_id':'template','scene_surface_id':'entity/board.continuous'},
        'pair':{'primary_instance_id':'print_instance/Ad1','paired_instance_id':'print_instance/Ad14',
            'size_block_id':'shared_size','authority':{'kind':'direct_user_confirmation'}},
        'installation_initializations':[
            {'instance_id':'print_instance/Ad1','parameters':[.3,.015,3.14,.08,.01],'provenance':{}},
            {'instance_id':'print_instance/Ad14','parameters':[1.7,.016,3.14,.08,.01],'provenance':{}}]}


def test_open_original_pixels_and_groups_survive_closed_template_import():
    source=selected();saved=deepcopy(source);assets=import_selected_parts(source)
    assert source==saved
    assert assets['observations']==source['observations']
    assert assets['source_correlated_groups']==source['source_correlated_groups']
    assert assets['shape_evidence']==source['shape_evidence']
    assert assets['observations'][0]['closed'] is False
    assert assets['templates'][0]['paths'][0]['closed'] is True
    a=resolve_instance(assets,'print_instance/Ad1');b=resolve_instance(assets,'print_instance/Ad14')
    assert a[2] is b[2] and a[3] is not b[3]
    assert b[0]['source_status']=='no_instance_source_pixels'
    assert 'visibility_expectation' not in b[0]
    assert not assets['fit_executed']


def test_still_artwork_cannot_become_programme_observation():
    source=selected();source['observations'][0]['source_id']='artwork'
    with pytest.raises(ValueError,match='own video clock'):import_selected_parts(source)


def test_import_does_not_authorize_fit_or_include_reference_only_detail():
    source=selected();source['observations'][0]['active_for_fit']=True
    with pytest.raises(ValueError,match='does not authorize fitting'):import_selected_parts(source)
    source=selected();source['template']['paths'][0]['model_inclusion']='reference_only'
    with pytest.raises(ValueError,match='reference-only detail'):import_selected_parts(source)


def test_missing_hole_parent_or_clock_conflict_fails_explicitly():
    source=selected();source['template']['paths'][1].pop('parent_contour_key')
    with pytest.raises(ValueError,match='requires its parent'):import_selected_parts(source)
    source=selected();source['observations'][0]['pts']=101
    with pytest.raises(ValueError,match='clocks differ'):import_selected_parts(source)


def test_analytic_square_carries_shape_rule_without_noisy_polyline_fallback():
    source=selected();path=source['template']['paths'][0]
    path.pop('material_uv');path['geometry']={'kind':'unfolded_square','center_uv':[.15,.5],
        'side_over_panel_height':.8,'authority':'direct_user_confirmation'}
    assets=import_selected_parts(source)
    assert assets['templates'][0]['paths'][0]['geometry']==path['geometry']
    assert 'material_uv' not in assets['templates'][0]['paths'][0]
    path['material_uv']=[[0,0],[1,1]]
    with pytest.raises(ValueError,match='polyline fallback'):import_selected_parts(source)
