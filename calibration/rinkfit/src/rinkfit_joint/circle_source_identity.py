"""Source-only identity gate for finite-width painted circles.

A model/demo may locate a search, but a successful material profile cannot
certify which physical circle it is. Keep the old observation and its label
as history when the entity is unknown. This module never reassigns a point
to a neighbouring straight line, and never promotes previously withheld rows.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import numpy as np

from .dense_joint_proofs import _ref, _write, _checked
from .compact_identity_withhold_v2 import _array, _source_identities

# Historical names remain recognizable; current fitting uses only C0 midline.
PAINT_CIRCLES = frozenset(('C0', 'A1', 'A2', 'A3', 'A4'))
CURRENT_PAINT_CIRCLES = frozenset(('C0',))
C0_CURRENT_LABEL = 'ice.C0:middle'
IDENTITY_SCHEMA = 'rinkfit.confirmed_source_circle_identity.v1'
SCOPE_SCHEMA = 'rinkfit.source_circle_identity_withdrawal_scope.v1'


def circle_name(feature):
    """Historical :outer is a label; these five paint curves use midlines."""
    name = str(feature).removeprefix('ice.').split(':', 1)[0]
    return name if name in PAINT_CIRCLES else None



def require_c0_midline_reading(row, *, label=None, uv=None):
    """Check two actual material interfaces; never turn one edge into a center.

    Identity/complete-stroke review is a separate upstream requirement. This
    validates the measured operator at its unchanged source UV and deliberately
    accepts historical C0 labels only for explicit measurement transport.
    """
    name = circle_name(label or row.get('material_label', row.get('physical_entity',
                       row.get('feature_id', row.get('target_id', '')))))
    if name not in CURRENT_PAINT_CIRCLES:
        raise ValueError('A1-A4 are excluded from current fitting; only C0 paired-paint midline is issued')
    profile = row.get('current_native_profile', {})
    pair = profile.get('actual_paired_material_profile', row.get('actual_paired_midline_reading'))
    if (not isinstance(pair, dict) or pair.get('disposition') != 'measured_paired_stroke_midline'
            or not pair.get('source_flags') or any(pair['source_flags'].values())
            or pair.get('intensity_median_or_darkest_peak_used_as_center') is not False):
        raise ValueError('C0 needs both actual paint-to-ice interfaces; single edges cannot issue a midline')
    sides = [pair.get('actual_RGB_entering_interface'), pair.get('actual_RGB_leaving_interface')]
    if any(not isinstance(side, dict) or side.get('disposition') != 'measured_material_transition'
           or not side.get('source_flags') or any(side['source_flags'].values()) for side in sides):
        raise ValueError('C0 has an unresolved material flank')
    if {side.get('material_transition') for side in sides} != {'light_to_dark', 'dark_to_light'}:
        raise ValueError('C0 requires opposite ordered transitions of the same paint stroke')
    flanks = np.asarray([side.get('actual_source_uv') for side in sides], float)
    actual = np.asarray(uv if uv is not None else row.get('source_uv', row.get('uv', pair.get('actual_source_uv'))), float)
    if (flanks.shape != (2, 2) or actual.shape != (2,) or not np.isfinite(flanks).all()
            or not np.isfinite(actual).all() or np.array_equal(flanks[0], flanks[1])):
        raise ValueError('C0 material flanks and observed midpoint must be finite and distinct')
    center = flanks.mean(0)
    if (not np.allclose(actual, center, rtol=0, atol=1e-9)
            or not np.allclose(np.asarray(pair.get('actual_source_uv'), float), center, rtol=0, atol=1e-9)):
        raise ValueError('C0 observation is not the arithmetic midpoint of its two actual material edges')
    if profile:
        if (profile.get('measurement_definition') != 'arithmetic_width_midpoint_of_two_actual_material_edges'
                or profile.get('complete_both_material_sides_visible') is not True
                or not np.allclose(np.asarray(profile.get('two_side_interfaces_uv'), float), flanks, rtol=0, atol=1e-9)):
            raise ValueError('C0 midpoint receipt does not preserve both complete material sides')
    return center

def _validate_identity(row, proof, base):
    if proof.get('schema') != IDENTITY_SCHEMA:
        raise ValueError('An actual source circle identity proof is required')
    for key in ('world_camera_projection_used_as_identity', 'model_residual_used_for_identity',
                'navigation_shape_used_as_identity', 'source_UV_invented'):
        if proof.get(key) is not False:
            raise ValueError('Circle identity must be independent of navigation/model: ' + key)
    name = circle_name(row.get('feature_id', row.get('target_id', '')))
    if (proof.get('entity_id') != name or proof.get('source_geometry') != 'confirmed_circle_arc'
            or proof.get('same_paint_stroke') is not True
            or proof.get('competing_straight_material_resolved') is not True
            or proof.get('source_video_sha256') != row.get('source_video_sha256')
            or proof.get('shot_id') != row.get('shot_id')
            or int(proof.get('native_pts', -1)) != int(row.get('native_pts', -2))):
        raise ValueError('Circle identity proof does not bind this actual source arc')
    rid = row.get('selected_row_id', row.get('source_row_id', row.get('point_id')))
    if rid not in proof.get('source_row_ids', ()):
        raise ValueError('Circle identity proof does not cover this source Query')
    refs = proof.get('actual_source_evidence_refs', ())
    if not refs:
        raise ValueError('Missing actual source arc evidence')
    for ref in refs:
        _checked(ref, base)
    anchor = proof.get('identity_anchor', {})
    if (anchor.get('source_entity_confirmed') is not True
            or anchor.get('kind') not in ('reviewed_original_RGB_circle_arc',
                                          'source_to_source_static_contour_chain')):
        raise ValueError('A source arc anchor, not a model ellipse, must issue identity')
    if anchor['kind'] == 'source_to_source_static_contour_chain':
        if (anchor.get('same_shot_static_source_verified') is not True
                or anchor.get('forward_backward_source_track_verified') is not True
                or anchor.get('no_occlusion_or_material_crossing') is not True
                or not anchor.get('source_chain_receipt')):
            raise ValueError('Unverified source contour transport cannot carry circle identity')
        _checked(anchor['source_chain_receipt'], base)
    return True


def qualify_circle_profile_row(row, *, identity_evidence=None, evidence_base='.'):
    """Separate an observed paint midpoint from its still unproved entity.

    Explicit reviewed straight competitors stay unknown under their old circle
    label. They are not silently relabelled B2/C. Existing unknowns never revive.
    """
    out = deepcopy(row)
    name = circle_name(row.get('feature_id', row.get('target_id', '')))
    if name is None:
        return out
    out['circle_source_identity_gate'] = _ref(__file__)
    out['navigation_is_identity_authority'] = False
    out['original_source_reading_preserved'] = True
    if name not in CURRENT_PAINT_CIRCLES:
        out.update(physical_instance_qualification='unknown', source_fit_eligible=False,
                   source_geometry_eligible=False, qualification_granted=False,
                   qualification_reason_code='A1_A4_removed_from_current_fitting_no_replacement',
                   source_geometry_classification='historical_paint_circle_not_in_current_fit')
        return out
    prior_withheld = (row.get('source_fit_eligible') is False
                      or row.get('physical_instance_qualification') == 'unknown'
                      or row.get('admission') == 0 or row.get('fit_role') == 0)
    reason = 'circle_entity_not_confirmed_by_original_source_arc'
    disposition = 'circle_identity_unknown'
    if row.get('source_geometry_classification') == 'reviewed_straight_material':
        reason = 'circle_label_conflicts_with_reviewed_source_straight_material'
        disposition = 'wrong_circle_entity_straight_candidate'
    elif identity_evidence is not None:
        _validate_identity(row, identity_evidence, Path(evidence_base))
        if not prior_withheld:
            out['physical_instance_qualification'] = 'source_circle_identity_confirmed'
            out['circle_identity_evidence'] = deepcopy(identity_evidence)
            out['source_geometry_classification'] = 'confirmed_circle_arc'
            # Identity alone cannot grant a paint-center observation.
            out.update(source_fit_eligible=False, source_geometry_eligible=False, qualification_granted=False,
                       qualification_reason_code='C0_both_actual_material_flanks_not_yet_measured')
            return out
        reason = 'prior_withheld_source_requires_separate_explicit_readmission'
    out.update(physical_instance_qualification='unknown', source_fit_eligible=False,
               source_geometry_eligible=False, qualification_granted=False,
               qualification_reason_code=reason, source_geometry_classification=disposition)
    return out


def read_confirmed_circle_midlines(rgb, *, rows, identity_evidence, evidence_base='.', **profile_args):
    """Read each actual paint side only after independent circle identification.

    Uses the public finite-side reader; missing or competing flanks stay unknown.
    The source proof covers Query IDs, not generated/interpolated pixel locations.
    """
    from rinkspace.vision_source_shapes import read_source_stroke_midlines
    if len(rows) != len(identity_evidence):
        raise ValueError('Each actual circle Query needs explicit identity evidence')
    qualified = [qualify_circle_profile_row(r, identity_evidence=p, evidence_base=evidence_base)
                 for r, p in zip(rows, identity_evidence)]
    if any(r.get('physical_instance_qualification') != 'source_circle_identity_confirmed'
           for r in qualified):
        raise ValueError('Unconfirmed/withheld circle Query cannot be searched as a fit source')
    measured = read_source_stroke_midlines(rgb, **profile_args)
    if len(measured['readings']) != len(rows):
        raise ValueError('Circle source Query/profile coverage differs')
    result = []
    for row, reading in zip(qualified, measured['readings']):
        row['actual_paired_midline_reading'] = reading
        if reading['disposition'] != 'measured_paired_stroke_midline':
            row.update(source_fit_eligible=False, source_geometry_eligible=False,
                       qualification_reason_code='circle_material_flanks_unresolved')
        else:
            center = require_c0_midline_reading(row, uv=reading['actual_source_uv'])
            row['original_source_uv'] = deepcopy(row.get('source_uv', row.get('uv')))
            row['uv'] = center.tolist()
            row['source_uv'] = center.tolist()
            row['current_uv'] = center.tolist()
            row['source_query_remeasured'] = True
            row['source_observation_generation_requires_explicit_revision'] = True
            row['target_id'] = C0_CURRENT_LABEL
            row['measurement_definition'] = 'arithmetic_width_midpoint_of_two_actual_material_edges'
            row['measurement_operator_resolved'] = 'paint_center'
            row['source_fit_eligible'] = True
            row['source_geometry_eligible'] = True
        result.append(row)
    return dict(schema='rinkfit.confirmed_circle_actual_midlines.v1', rows=result,
                source_binding=measured['source_binding'], RGB_sha256=measured['RGB_sha256'],
                model_projection_used_as_observation=False,
                unknown_source_rows_preserved=True)


def prepare_identity_withdrawal_scope(*, parent_packet, material_entities, shot_ids,
                                      actual_source_review_refs, finding, output):
    """Record every affected raw row; this does not claim a source re-fit.

    Withdrawal is due to a revoked identity issuer over the complete batch,
    not residual magnitude. Already unknown rows remain in the denominator.
    The exact eligible subset is suitable for an immutable qualification child.
    """
    parent = Path(parent_packet).resolve()
    doc = json.loads(parent.read_text(encoding='utf-8-sig'))
    if not actual_source_review_refs or not finding:
        raise ValueError('Explicit source identity review and batch finding required')
    for ref in actual_source_review_refs:
        _checked(ref, parent.parent)
    materials = {int(k): v for k, v in material_entities.items()}
    if any(v not in PAINT_CIRCLES for v in materials.values()):
        raise ValueError('This gate only withdraws named painted circle identity')
    shots = {int(s.removeprefix('s')) - 1: s for s in shot_ids}
    arrays = {k: _array(doc['arrays'][k]) for k in
              ('material_index', 'shot_index', 'native_pts', 'admission', 'fit_role')}
    hit = np.isin(arrays['material_index'], list(materials)) & np.isin(arrays['shot_index'], list(shots))
    affected = np.flatnonzero(hit)
    eligible = affected[(arrays['admission'][affected] == 1) & (arrays['fit_role'][affected] == 1)]
    counts = Counter((shots[int(arrays['shot_index'][i])], materials[int(arrays['material_index'][i])],
                      bool(arrays['admission'][i] and arrays['fit_role'][i])) for i in affected)
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    target = out / 'withdrawn_raw_rows.u64'
    eligible.astype('<u8').tofile(target)
    scope = dict(schema=SCOPE_SCHEMA, parent_packet=_ref(parent), issuer=_ref(__file__),
                 affected_material_entities={str(k): v for k, v in materials.items()},
                 affected_shot_ids=list(shot_ids), raw_rows=int(doc['row_count']),
                 affected_raw_rows=len(affected), newly_unknown_rows=len(eligible),
                 already_unknown_rows=len(affected)-len(eligible),
                 withdrawn_raw_rows=dict(_ref(target), dtype='<u8', shape=[len(eligible)]),
                 by_shot_entity=[dict(shot_id=s, entity=e, eligible_before=active, rows=n)
                                 for (s, e, active), n in sorted(counts.items())],
                 actual_source_review_refs=actual_source_review_refs, finding=finding,
                 confirmed_straight_rows=0, other_rows_circle_identity_unknown=True,
                 original_UV_ID_nativePTS_material_path_retained=True,
                 unknown_rows_readmitted=False, new_B2_or_C_labels_issued=False,
                 model_residual_used_for_qualification=False, source_rows_deleted=0,
                 new_objective_required=True, old_cost_reduction_claimed=False,
                 qualification_packet_applied=False, solver_started=False)
    _write(out / 'scope.json', scope)
    return scope
