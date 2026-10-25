from copy import deepcopy
import math
import pytest
from rinkspace.domain.print_assets import attach_paired_instance,resolve_instance,native_instance_parameters,initialize_instance,PrintBindingError


def source_assets():
    return {'templates':[{'id':'print_template/Ad2','text':'ISU-SKATING.COM',
        'paths':[{'id':'Ad2.print.COM_O','material_uv':[[0.,0.],[1.,0.],[1.,1.],[0.,0.]],'closed':True},
                 {'id':'Ad2.print.COM_O_inner','material_uv':[[.1,.2],[.2,.3]],'parent_contour_key':'Ad2.print.COM_O'}]}],
        'instances':[{'id':'print_instance/Ad2','template_id':'print_template/Ad2','scene_surface_id':'entity/board.continuous'}],
        'observations':[{'id':'measured/Ad2','feature_id':'Ad2.print.COM_O_inner','pts':320000,'uv':[100.,200.],'noise_group_id':'original/group'}],
        'aliases':[{'canonical_observation_id':'measured/Ad2'}],
        'noise_groups':[{'id':'original/group','members':['measured/Ad2']}],'counts':{}}


def bind(value):
    return attach_paired_instance(value,'print_instance/Ad2','print_instance/Ad15',
        'artwork_pair/Ad2-Ad15/size',authority={'kind':'direct_user_confirmation'})


def test_both_instances_resolve_same_template_and_actual_size_block():
    assets=bind(source_assets());a=resolve_instance(assets,'print_instance/Ad2');b=resolve_instance(assets,'print_instance/Ad15')
    assert a[1] is b[1] and a[2] is b[2]
    assert a[3] is not b[3]
    a[2]['values']=[math.log(.08),math.log(.012)]
    # Synthetic initialized placements only exercise common parameter lookup.
    a[3]['values']=[.1,.01,.0];b[3]['values']=[.41,.02,.2]
    before=[native_instance_parameters(assets,i)[3:] for i in ('print_instance/Ad2','print_instance/Ad15')]
    a[2]['values'][0]+=math.log(2.)
    after=[native_instance_parameters(assets,i)[3:] for i in ('print_instance/Ad2','print_instance/Ad15')]
    assert before[0]==before[1] and after[0]==after[1]
    assert after[0][0]==pytest.approx(2*before[0][0])
    assert native_instance_parameters(assets,'print_instance/Ad15')[:3]==[.41,.02,.2]


def test_identity_attachment_creates_no_pixels_or_equality_factors():
    original=source_assets();saved=deepcopy(original);new=bind(original)
    assert original==saved
    for key in ('observations','aliases','noise_groups'):assert new[key]==original[key]
    assert new['templates'][0]['paths']==original['templates'][0]['paths']
    paired=resolve_instance(new,'print_instance/Ad15')
    assert paired[0]['entity_id']=='entity/Ad15'
    assert paired[0]['source_observation_ids']==[] and paired[3]['values'] is None
    assert paired[2]['values'] is None
    assert 'historical_installation_ref' not in paired[0]
    assert 'constraints' not in new and 'residuals' not in new


def test_unlocalized_identity_cannot_be_rendered_as_fake_world_geometry():
    new=bind(source_assets())
    with pytest.raises(PrintBindingError) as e:native_instance_parameters(new,'print_instance/Ad15')
    assert e.value.code=='unlocalized_print_instance'
    resolve_instance(new,'print_instance/Ad2')[3]['values']=[.1,.01,0.]
    with pytest.raises(PrintBindingError) as e:native_instance_parameters(new,'print_instance/Ad2')
    assert e.value.code=='uninitialized_print_size'


def test_repeated_binding_keeps_existing_parameter_values_and_no_duplicates():
    new=bind(source_assets());resolve_instance(new,'print_instance/Ad2')[2]['values']=[-3.,-4.]
    again=bind(new)
    assert again==new
    assert len(again['instances'])==2 and len(again['parameter_blocks'])==3


def test_conflicting_existing_template_is_not_silently_overwritten():
    new=source_assets();new['instances'].append({'id':'print_instance/Ad15','template_id':'another_print'})
    with pytest.raises(PrintBindingError) as e:bind(new)
    assert e.value.code=='conflicting_paired_print_template'


def test_installation_seed_changes_only_shared_size_and_selected_placement():
    old=bind(source_assets());saved=deepcopy(old)
    new=initialize_instance(old,'print_instance/Ad2',[.3,.015,math.pi,.08,.012],
        provenance={'source':'frozen_initializer'})
    assert old==saved
    for key in ('observations','aliases','noise_groups','templates'):assert new[key]==old[key]
    assert native_instance_parameters(new,'print_instance/Ad2')==pytest.approx([.3,.015,math.pi,.08,.012])
    assert resolve_instance(new,'print_instance/Ad15')[2] is resolve_instance(new,'print_instance/Ad2')[2]
    assert resolve_instance(new,'print_instance/Ad15')[3]['values'] is None
    with pytest.raises(PrintBindingError,match='unlocalized_print_instance'):
        native_instance_parameters(new,'print_instance/Ad15')
    with pytest.raises(PrintBindingError,match='print_initialization_conflicts'):
        initialize_instance(new,'print_instance/Ad2',[.4,.015,math.pi,.08,.012],provenance={})
