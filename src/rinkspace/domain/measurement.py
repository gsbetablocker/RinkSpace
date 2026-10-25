"""Ice marking measurements and explicitly bound print material boundaries.

Ice width-centrelines do not redefine printed lettering, holes or panel edges.
Source pixels and historical evidence are separate; old ice names remain invalid.
"""
from __future__ import annotations
from dataclasses import dataclass
import re
from typing import Mapping
from .contracts import ContractError
from .print_assets import PrintCurveBinding

POLICY_ID = 'ankara_2026.curve_measurement.user_20261024.v3'
CANONICAL_MEASUREMENTS = frozenset({'centreline', 'contour', 'material_point'})
FORBIDDEN_DEFINITION_CHANNELS = frozenset({
    'original_measurement', 'original_descriptor', 'legacy_measurement', 'measurement_definition',
    'previous_measurement', 'previous_model_measurement', 'stored_measurement',
})


@dataclass(frozen=True)
class MeasurementRule:
    mode: str
    accepted_measurements: tuple[str, ...]
    model_width: float | None
    policy_id: str = POLICY_ID


class MeasurementPolicyError(ContractError):
    def __init__(self, message: str, context: dict):
        super().__init__(message)
        self.code = 'invalid_measurement_semantics'
        self.message = message
        self.context = context


def _name(label: str) -> str:
    if not isinstance(label, str) or not label:
        raise ContractError('Measurement policy requires an explicit entity label')
    return label.removeprefix('entity/').removeprefix('ice.').split(':', 1)[0]


def measurement_rule(label: str, feature_kind: str = 'finite_curve', *,
                     surface: str | None = None, print_binding: PrintCurveBinding | None = None) -> MeasurementRule:
    """Use an identified print boundary's own meaning, never its label spelling."""
    if print_binding is not None:
        if not isinstance(print_binding, PrintCurveBinding):
            raise ContractError('Print measurement requires a resolved current template/path binding')
        if surface != 'board' or feature_kind != 'finite_curve':
            raise ContractError('Print material boundaries require a finite curve on the board')
        if label not in (print_binding.path_id, print_binding.template_path_id):
            raise ContractError('Print feature label must identify its bound material path')
        return MeasurementRule('contour', ('contour',), None)
    if feature_kind in ('material_point', 'fixed_junction'):
        return MeasurementRule('material_point', ('material_point',), None)
    if feature_kind != 'finite_curve':
        raise ContractError('Unsupported feature kind for measurement policy: '+feature_kind)
    name = _name(label)
    if surface in (None, 'ice') and (re.fullmatch(r'[FR][1-4]', name) or name in ('goal_crease_left', 'goal_crease_right')):
        return MeasurementRule('contour', ('contour',), None)
    return MeasurementRule('centreline', ('centreline',), 0.)


def required_measurement(label: str, feature_kind: str = 'finite_curve', *,
                         surface: str | None = None, print_binding: PrintCurveBinding | None = None) -> str:
    return measurement_rule(label, feature_kind, surface=surface, print_binding=print_binding).mode


def _source_evidence_error(evidence, required):
    if evidence is None:
        return None
    if isinstance(evidence, str):
        declared = evidence
    elif isinstance(evidence, Mapping):
        declared = evidence.get('measurement', evidence.get('declared_measurement', evidence.get('mode')))
        if declared is None:
            return None  # Source identity/provenance alone supplies no alternate definition.
    else:
        return 'Source measurement evidence has an invalid representation'
    if declared != required:
        return 'Source pixels were not established as the required current measurement; revise affected source bindings, never rename those pixels'
    return None


def assess_measurement_binding(label: str, measurement: str, source_measurement_evidence=None,
                               *, feature_kind: str = 'finite_curve', surface: str | None = None,
                               print_binding: PrintCurveBinding | None = None) -> dict:
    """Reject incompatible current meaning or incompatible source evidence.

    This function does not change pixels, source time, IDs, row activity or W.
    Evidence is diagnostic input only, never a second valid feature definition.
    """
    rule = measurement_rule(label, feature_kind, surface=surface, print_binding=print_binding)
    current_valid = measurement == rule.mode
    evidence_error = _source_evidence_error(source_measurement_evidence, rule.mode)
    compatible = current_valid and evidence_error is None
    return {
        'policy_id':rule.policy_id, 'required_measurement':rule.mode,
        'measurement':measurement, 'model_width':rule.model_width,
        'measurement_compatible':compatible,
        'status':'current_definition_valid' if compatible else 'invalid_current_measurement',
        'source_values_rewritten':False,
        'reason':('Current definition is canonical; source identity and quality require their own checks.'
                  if compatible else evidence_error or 'Only the canonical current measurement is allowed; old names and alternate definitions are invalid.'),
    }


def validate_feature_bindings(features, active_feature_ids, policy_id=POLICY_ID, *, print_bindings=None) -> dict:
    """Validate every current feature, including inactive features, or fail.

    There is no inactive-conflict acceptance path. Frozen historical data must
    remain outside this current feature registry. No row or group is altered.
    """
    if policy_id != POLICY_ID:
        raise MeasurementPolicyError('Unknown current measurement policy', {'policy_id':policy_id})
    records = list(features.items()) if isinstance(features, Mapping) else [
        (value.get('id', value.get('feature_id')),value) for value in features]
    registry = {}
    for identity,value in records:
        if not isinstance(value, Mapping) or not isinstance(identity,str) or not identity or identity in registry:
            raise MeasurementPolicyError('Invalid or duplicate feature registry identity', {'feature_id':identity})
        registry[identity]=value
    active=set(active_feature_ids)
    if active-set(registry):
        raise MeasurementPolicyError('Active measurement references absent feature', {'feature_ids':sorted(active-set(registry))})
    resolved = print_bindings or {}
    conflicts=[]
    for identity,feature in registry.items():
        forbidden=sorted(FORBIDDEN_DEFINITION_CHANNELS.intersection(feature))
        if forbidden:
            conflicts.append({'feature_id':identity,'forbidden_definition_channels':forbidden,
                              'reason':'Historical source evidence must not remain as alternate current feature definitions'})
            continue
        label=feature.get('label')
        if not isinstance(label,str) or not label:
            conflicts.append({'feature_id':identity,'reason':'Explicit physical entity label is missing'})
            continue
        binding = resolved.get(identity)
        declaration = feature.get('print_binding')
        if declaration is not None and (binding is None or declaration != binding.as_record()):
            conflicts.append({'feature_id':identity,'reason':'Print boundary is not resolved against current scene assets'})
            continue
        assessment=assess_measurement_binding(label,feature.get('measurement',''),
            feature.get('source_measurement_evidence'),feature_kind=feature.get('kind','finite_curve'),
            surface=feature.get('surface'), print_binding=binding)
        if not assessment['measurement_compatible']:
            conflicts.append({'feature_id':identity,'label':label,**assessment})
    if conflicts:
        raise MeasurementPolicyError('Current input contains invalid measurement definitions; retain source pixels separately and revise bindings/groups before use',
                                     {'policy_id':policy_id,'conflicts':conflicts})
    return {'policy_id':policy_id,'features':len(registry),'active_features':len(active),
            'all_current_features_valid':True,'source_values_rewritten':False}


def validate_scene_measurements(scene, policy_id=POLICY_ID) -> dict:
    """Check actual current paths, including unused paths, without rewriting them."""
    declared = scene.get('measurement_policy_id', policy_id)
    if declared != policy_id:
        raise MeasurementPolicyError('Scene measurement policy differs from current input',
                                     {'scene_policy_id':declared,'policy_id':policy_id})
    paths = scene.get('paths', [])
    features = []
    for path in paths:
        # Geometry kinds (circle, ellipse, native integer, etc.) are distinct
        # from observation kinds. Named ring layers remain separate curves.
        kind = path.get('kind')
        feature_kind = kind if kind in ('material_point', 'fixed_junction') else 'finite_curve'
        features.append({**path, 'kind':feature_kind})
    report = validate_feature_bindings(features, [], policy_id)
    for path, feature in zip(paths, features):
        if required_measurement(feature['label'], feature['kind']) == 'centreline':
            if path.get('model_line_width', 0.) != 0.:
                raise MeasurementPolicyError('Current centreline geometry must have zero model width',
                                             {'path_id':path['id']})
    return report


def validate_observation_binding(record, feature, row_index, *, print_binding=None):
    """Check current row declarations only; never reopen source/provenance assets."""
    if not isinstance(record, Mapping):
        raise MeasurementPolicyError('Observation metadata must be a record', {'source_row':row_index})
    context = {'source_row':row_index, 'feature_id':feature['id']}
    forbidden = sorted(FORBIDDEN_DEFINITION_CHANNELS.intersection(record))
    if forbidden:
        raise MeasurementPolicyError('Current row contains alternate measurement definitions',
                                     {**context,'forbidden_definition_channels':forbidden})
    if record.get('feature_id') != feature['id']:
        raise MeasurementPolicyError('Row metadata and numeric feature binding differ', context)
    if 'print_binding' in record and (print_binding is None or record['print_binding'] != print_binding.as_record()):
        raise MeasurementPolicyError('Observation and feature print bindings differ', context)
    if feature.get('print_binding') is not None:
        if print_binding is None or feature['print_binding'] != print_binding.as_record():
            raise MeasurementPolicyError('Observation print boundary lacks a current resolved binding', context)
    assessment = assess_measurement_binding(feature['label'], record.get('measurement', ''),
        record.get('source_measurement_evidence'), feature_kind=feature.get('kind', 'finite_curve'),
        surface=feature.get('surface'), print_binding=print_binding)
    if not assessment['measurement_compatible']:
        raise MeasurementPolicyError('Current row contains an invalid measurement definition',
                                     {**context, **assessment})
