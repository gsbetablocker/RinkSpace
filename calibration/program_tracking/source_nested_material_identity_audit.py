"""Find named inner-layer claims inherited from an outer carrier annotation.

This is a review guard, not a source scope authority. A detected claim needs
its own material review; it is never converted into a reflection/negative.
"""
from __future__ import annotations

NESTED_GRADIENT_BASIS = 'same_actual_named_house_and_current_RGB_alternating_nested_material_sequence'


def inherited_nested_material_review_problem(row, point_proof=None, *, point_uv=None):
    """Return an exact-point review reason without changing any source data.

    An annotation that identifies the outer house may navigate its inner
    searches. Alternating real RGB gradients do not establish the named inner
    paint layer, because shadows and reflections also contain such sequences.
    """
    material = row.get('physical_material_id', '')
    parts = material.rsplit(':', 1)
    if len(parts) != 2 or parts[1] not in ('middle', 'inner', 'center'):
        return None
    seed = row.get('source_seed_reference')
    if not isinstance(seed, dict) or seed.get('physical_identity_basis') != NESTED_GRADIENT_BASIS:
        return None
    annotation = seed.get('source_annotation_id', '')
    if not isinstance(annotation, str) or '_outer' not in annotation:
        return None
    proof = point_proof if isinstance(point_proof, dict) else {}
    # A direct point review must name this material and this exact exposure,
    # rather than borrow the outer annotation's general static-ice scope.
    if (proof.get('original_source_image_actually_reviewed') is True
            and proof.get('decision') == 'allow'
            and proof.get('source_feature_physical_identity_confirmed') is True
            and proof.get('physical_material_id') == material
            and proof.get('actual_pts') == row.get('actual_pts')
            and proof.get('native_pts') == row.get('native_pts')
            and proof.get('decoded_BGR_sha256') == row.get('decoded_BGR_sha256')
            and point_uv is not None and proof.get('uv') == list(point_uv)
            and proof.get('model_projection_or_fit_residual_used_for_identity') is False):
        return None
    return 'outer_carrier_and_alternating_RGB_gradients_do_not_independently_identify_this_named_inner_material'
