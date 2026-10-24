"""One current measurement definition; historical aliases are invalid."""
from copy import deepcopy
import pytest
from rinkspace.domain.contracts import ContractError
from rinkspace.domain.measurement import (
    POLICY_ID, MeasurementPolicyError, measurement_rule, required_measurement,
    assess_measurement_binding, validate_feature_bindings,
)


def test_only_named_houses_and_half_ellipse_are_contours():
    for side in ('F','R'):
        for row in range(1,5):
            for suffix in ('center','inner','middle','outer'):
                rule=measurement_rule(f'ice.{side}{row}:{suffix}')
                assert rule.mode=='contour'
                assert rule.accepted_measurements==('contour',)
    for label in ('ice.goal_crease_left','entity/ice.goal_crease_right'):
        assert required_measurement(label)=='contour'


def test_every_other_curve_is_zero_width_midline():
    labels=['ice.C0:outer',*[f'ice.A{i}:outer' for i in range(1,5)],
            'ice.C','ice.B1','ice.lane3:edge2','ice.house_connection_F4_R4:segment0',
            'ice.goal_trapezoid_left_lower_outer:segment0','board.top','Ad7.print.contour:arc0']
    for label in labels:
        rule=measurement_rule(label)
        assert rule.mode=='centreline' and rule.model_width==0.
        assert rule.accepted_measurements==('centreline',)


@pytest.mark.parametrize('old_name',['outer_boundary','inner_boundary','named_material_boundary',
    'half_ellipse_material_boundary','midwidth_material_centreline','centerline','outer_material_boundary'])
def test_old_names_are_never_current_aliases(old_name):
    assert not assess_measurement_binding('ice.C0',old_name)['measurement_compatible']
    assert not assess_measurement_binding('ice.F1',old_name)['measurement_compatible']


def test_source_evidence_cannot_rename_outer_pixels_to_midline():
    evidence={'measurement':'outer_boundary','uv':[501.,603.],'group':'W17'}
    before=deepcopy(evidence)
    assessment=assess_measurement_binding('ice.C0','centreline',evidence)
    assert not assessment['measurement_compatible']
    assert evidence==before
    assert not assessment['source_values_rewritten']
    assert 'original_measurement' not in assessment
    assert 'original_descriptor' not in assessment


def test_only_exact_canonical_meaning_is_valid():
    assert assess_measurement_binding('ice.C0','centreline')['measurement_compatible']
    assert assess_measurement_binding('ice.F1','contour')['measurement_compatible']
    assert assess_measurement_binding('ice.goal_crease_left','contour')['measurement_compatible']
    assert not assess_measurement_binding('ice.C0','contour')['measurement_compatible']
    assert not assess_measurement_binding('ice.F1','centreline')['measurement_compatible']


def test_invalid_inactive_feature_rejects_entire_current_input():
    features=[{'id':'C','label':'ice.C','kind':'finite_curve','measurement':'centreline'},
              {'id':'C0-old','label':'ice.C0','kind':'finite_curve','measurement':'outer_boundary'}]
    before=deepcopy(features)
    for active in ([],['C'],['C0-old']):
        with pytest.raises(MeasurementPolicyError) as exc:
            validate_feature_bindings(features,active)
        assert exc.value.context['conflicts'][0]['feature_id']=='C0-old'
    assert features==before


def test_alternate_current_definition_channels_rejected():
    for channel in ('original_measurement','measurement_definition'):
        feature={'id':'C','label':'ice.C','measurement':'centreline',channel:'outer_boundary'}
        with pytest.raises(MeasurementPolicyError) as exc:
            validate_feature_bindings([feature],[])
        assert exc.value.context['conflicts'][0]['forbidden_definition_channels']==[channel]


def test_fixed_nodes_and_known_policy():
    assert required_measurement('junction/C/B1','fixed_junction')=='material_point'
    features=[{'id':'C','label':'ice.C','measurement':'centreline'}]
    assert validate_feature_bindings(features,[],POLICY_ID)['all_current_features_valid']
    with pytest.raises(MeasurementPolicyError):validate_feature_bindings(features,[],policy_id='v2')
    with pytest.raises(MeasurementPolicyError):validate_feature_bindings([{'id':'opaque','measurement':'centreline'}],[])
    with pytest.raises(ContractError):measurement_rule('ice.F1','guessed_entity')