"""Read actual material evidence across source-reader generations.

This checks measurement semantics only. Named identity, current visibility,
and original video binding remain mandatory independent qualifications.
"""
import numpy as np


def direct_ice_measurement_reason(row):
    """Return None for a direct supported reading, otherwise its unresolved reason."""
    if row.get('source_uv_from_model') is not False or row.get('surface') != 'ice':
        return 'not_original_ice_source_measurement'
    if row.get('source_UV_is_direct_measurement') is False or row.get('status') != 'direct_current_pixel_supported':
        return 'inferred_or_unmeasured_source_coordinate'
    evidence = row.get('pixel_support', {})
    contrast = np.asarray(evidence.get('signed_interface_IQR_contrast', []), float)
    normal = np.asarray(evidence.get('normal', []), float)
    uv = np.asarray(row.get('uv', []), float)
    if contrast.shape != (2,) or not np.isfinite(contrast).all() or np.min(contrast) <= 0:
        return 'missing_signed_material_interface_support'
    if normal.shape != (2,) or not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-8:
        return 'missing_source_profile_normal'
    if uv.shape != (2,) or not np.isfinite(uv).all():
        return 'missing_actual_source_uv'
    definition = row.get('definition', '')
    if 'width_midpoint' in definition:
        sides = np.asarray(row.get('paired_edge_uv'), float)
        quartiles = np.asarray(evidence.get('side_interface_quartiles_uv', []), float)
        if sides.shape != (2, 2) or quartiles.shape != (2, 3, 2) or not np.isfinite(sides).all() or not np.isfinite(quartiles).all():
            return 'missing_two_measured_paint_interfaces'
        if evidence.get('complete_both_material_sides_visible') is False:
            return 'source_explicitly_marks_incomplete_material_band'
        if np.max(np.abs(sides.mean(axis=0) - uv)) > 1e-6:
            return 'coordinate_is_not_actual_material_midpoint'
        if np.linalg.norm(sides[1] - sides[0]) < .25:
            return 'paint_interfaces_not_distinct'
        if evidence.get('material_transition') is not None:
            return 'boundary_operator_conflicts_with_midpoint'
        support = np.concatenate((sides, quartiles.reshape(-1, 2)))
        if (support < [0, 0]).any() or (support > [1919, 1079]).any():
            return 'paint_interfaces_outside_native_exposure'
        return None
    if 'boundary' in definition:
        layer = row.get('curve_id', '').split(':')[-1]
        if layer not in ('inner', 'middle', 'outer'):
            return 'unresolved_named_target_boundary_layer'
        expected = 'light_to_dark' if layer == 'middle' else 'dark_to_light'
        return None if evidence.get('material_transition') == expected else 'wrong_material_transition'
    return 'unresolved_measurement_definition'
