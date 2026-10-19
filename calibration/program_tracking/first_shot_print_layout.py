"""Shared print layouts within each advertisement's existing board canvas.

The eight coordinates describe one projective change of the old rectification
initializer.  They create no observations, priors or per-frame print positions.
Prototype v runs down; the public physical material API runs up.  Source UV and
the advertisement's outer geometry are never changed here.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np


SCHEMA = "rinkspace.first-shot-ad2-print-layout.v1"
GENERIC_SCHEMA = "rinkspace.first-shot-print-layouts.v1"
SUPPORTED_ADVERTISEMENT_IDS = (2, 3, 4, 5, 6, 7)
LAYOUT_PARAMETER_NAMES = ("a", "b", "c", "d", "tx", "ty", "p", "q")
EXTENDED_MATERIAL_DOMAIN = "continuous_board_s_h_relative_to_named_internal_graphic"


class PrintLayoutDomainError(ValueError):
    """A finite layout leaves the actual projective or material domain."""


def _uv(value):
    result = np.asarray(value, dtype=np.float64)
    if (result.ndim != 2 or result.shape[1:] != (2,) or not len(result)
            or not np.isfinite(result).all()):
        raise ValueError("A nonempty finite N-by-2 prototype material UV array is required")
    return result


def validate_layout(layout8, prototype_uv=None):
    """Validate the projective map, without an arbitrary parameter size cap.

    A positive denominator at all four unit-canvas corners excludes an
    interior pole.  Actual prototype samples are checked too, including
    templates whose declared material domain extends beyond that canvas.
    The denominator is affine in UV, so positivity at all prototype vertices
    also guarantees positivity over their full convex hull.  For Ad5 the
    unit region is its internal graphic, not a containing outer poster.
    Containment in a particular template's physical canvas is checked by
    :func:`apply_print_layout`, rather than inferred from parameter size.
    """
    values = np.asarray(layout8, dtype=np.float64)
    if values.shape != (8,):
        raise ValueError("Exactly eight shared print-layout coordinates are required")
    if not np.isfinite(values).all():
        raise PrintLayoutDomainError("Shared print layout must be finite")
    a, b, c, d, tx, ty, p, q = values
    matrix = np.array(((1+a, b, tx), (c, 1+d, ty), (p, q, 1.)), dtype=np.float64)
    # slogdet avoids imposing a size cap merely because det would overflow.
    sign, logabsdet = np.linalg.slogdet(matrix)
    if sign <= 0 or not np.isfinite(logabsdet):
        raise PrintLayoutDomainError("Shared print layout must have positive determinant")
    centered = np.array(((-.5, -.5), (-.5, .5), (.5, -.5), (.5, .5)))
    if prototype_uv is not None:
        centered = np.vstack((centered, _uv(prototype_uv)-.5))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        denominator = 1 + p*centered[:, 0] + q*centered[:, 1]
        numerator = np.column_stack(((1+a)*centered[:, 0]+b*centered[:, 1]+tx,
                                     c*centered[:, 0]+(1+d)*centered[:, 1]+ty))
        mapped = numerator/denominator[:, None]+.5
    if (not np.isfinite(denominator).all() or (denominator <= 0).any()
            or not np.isfinite(mapped).all()):
        raise PrintLayoutDomainError("Shared print layout has a nonpositive pole or nonfinite material map")
    return matrix


def warp_normalized_material_uv(uv, layout8):
    """Return warped v-down UV and its N-by-2-by-8 analytic derivative.

    The centered homography is H=[[1+a,b,tx],[c,1+d,ty],[p,q,1]].
    Zero coordinates preserve the input float64 values exactly.
    """
    original = _uv(uv)
    validate_layout(layout8, original)
    a, b, c, d, tx, ty, p, q = np.asarray(layout8, dtype=np.float64)
    x, y = (original-.5).T
    denominator = 1+p*x+q*y
    nx, ny = (1+a)*x+b*y+tx, c*x+(1+d)*y+ty
    warped = (original.copy() if not np.any(layout8)
              else np.column_stack((nx/denominator+.5, ny/denominator+.5)))
    jacobian = np.zeros((len(original), 2, 8), dtype=np.float64)
    inverse = 1/denominator
    jacobian[:, 0, 0] = x*inverse
    jacobian[:, 0, 1] = y*inverse
    jacobian[:, 0, 4] = inverse
    jacobian[:, 1, 2] = x*inverse
    jacobian[:, 1, 3] = y*inverse
    jacobian[:, 1, 5] = inverse
    jacobian[:, 0, 6] = -nx*x*inverse**2
    jacobian[:, 0, 7] = -nx*y*inverse**2
    jacobian[:, 1, 6] = -ny*x*inverse**2
    jacobian[:, 1, 7] = -ny*y*inverse**2
    if not np.isfinite(jacobian).all():
        raise PrintLayoutDomainError("Shared print-layout derivative is nonfinite")
    return warped, jacobian


def _advertisement_id(value):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or int(value) not in SUPPORTED_ADVERTISEMENT_IDS):
        raise ValueError("Shared print layouts are explicitly supported for Ad2/3/4/5/6/7 only")
    return int(value)


def print_layout_template(templates, advertisement_id):
    """Select this advertisement's material without falling back to another ad."""
    number = _advertisement_id(advertisement_id)
    if not isinstance(templates, dict):
        raise ValueError("Explicit existing print templates are required")
    material = templates.get("advertisements", {}).get(str(number))
    if material is None:
        # Ad2's actual compound template keeps its own contours at the root.
        # Standalone explicitly numbered templates also retain their schema.
        material = (templates if templates.get("advertisement_number") == number
                    or (number == 2 and "advertisement_number" not in templates) else None)
    if (not isinstance(material, dict) or not isinstance(material.get("contours"), dict)
            or ("advertisement_number" in material and material["advertisement_number"] != number)):
        raise ValueError(f"Missing existing Ad{number} print-material template")
    if any("advertisement_id" in record and record["advertisement_id"] != number
           for record in material["contours"].values()):
        raise ValueError(f"An Ad{number} print contour belongs to another advertisement")
    return material


def _ad2_template(templates):
    return print_layout_template(templates, 2)


def apply_print_layout(templates, advertisement_id, layout8):
    """Return a print-only template copy; never edit the source template.

    The identity copy has exactly the original fields and numbers.  Other ads,
    source-native coordinates, contour IDs and outer boundaries are retained.
    Nonidentity copies retain prototype coordinates explicitly for review.
    One homography adjusts the common layout; it does not independently
    repair local shape or piecewise atlas-registration errors.
    """
    number = _advertisement_id(advertisement_id)
    original = print_layout_template(templates, number)
    records = original["contours"]
    if not records:
        raise ValueError(f"The existing Ad{number} print template has no contours")
    prototypes = {key: _uv(record.get("prototype_material_uv", record.get("material_uv")))
                  for key, record in records.items()}
    validate_layout(layout8, np.vstack(list(prototypes.values())))
    result = deepcopy(templates)
    if not np.any(layout8):
        return result
    material = print_layout_template(result, number)
    for key, prototype in prototypes.items():
        record = material["contours"][key]
        warped, _ = warp_normalized_material_uv(prototype, layout8)
        domain = record.get("prototype_material_coordinate_domain", record.get("material_coordinate_domain"))
        if domain != EXTENDED_MATERIAL_DOMAIN and ((warped < 0).any() or (warped > 1).any()):
            raise PrintLayoutDomainError(f"Ad{number} print leaves its declared actual outer canvas: " + key)
        if record.get("closed") and np.array_equal(prototype[0], prototype[-1]):
            warped[-1] = warped[0]
        record["prototype_material_uv"] = prototype.tolist()
        record["prototype_material_coordinate_domain"] = domain
        record["material_uv"] = warped.tolist()
    material["shared_print_layout"] = dict(schema=GENERIC_SCHEMA, advertisement_id=number,
        parameter_names=list(LAYOUT_PARAMETER_NAMES), layout8=np.asarray(layout8, dtype=np.float64).tolist(),
        fitted_from_existing_native_50_points_only=True)
    return result


def apply_ad2_print_layout(templates, layout8):
    """Compatibility wrapper for the original single-Ad2 layout interface."""
    result = apply_print_layout(templates, 2, layout8)
    if np.any(layout8):
        _ad2_template(result)["shared_print_layout"]["schema"] = SCHEMA
    return result


def material_world_layout_jacobian(material_geometry, advertisement_id, prototype_uv,
                                  layout8, geometry_q, *, fd_step=1e-6, closed=False):
    """Chain the layout derivative through the public material lifting API.

    Four batched material calls differentiate the two UV axes, independently
    of how many shared layout parameters there are.  No scene is rebuilt and
    no old source_template_material datum is used.  The returned derivative
    is in the same scene units as feature_world.  World values themselves
    must still come from feature_world to preserve its exact identity path.
    """
    number = _advertisement_id(advertisement_id)
    if not np.isfinite(fd_step) or fd_step <= 0:
        raise ValueError("A positive finite material UV finite-difference step is required")
    original = _uv(prototype_uv)
    warped, layout_jacobian = warp_normalized_material_uv(original, layout8)
    world_uv_jacobian = np.empty((len(warped), 3, 2), dtype=np.float64)
    for axis in range(2):
        minus, plus = warped.copy(), warped.copy()
        minus[:, axis] -= fd_step
        plus[:, axis] += fd_step
        # Prototype v-down becomes physical material v-up exactly once.
        low = np.asarray(material_geometry.material(number, minus[:, 0], 1-minus[:, 1], geometry_q)[0], dtype=np.float64)
        high = np.asarray(material_geometry.material(number, plus[:, 0], 1-plus[:, 1], geometry_q)[0], dtype=np.float64)
        if low.shape != (len(warped), 3) or high.shape != low.shape:
            raise ValueError(f"Public Ad{number} material API returned a different world inventory")
        world_uv_jacobian[:, :, axis] = (high-low)/(2*fd_step)
    result = np.einsum("nci,nip->ncp", world_uv_jacobian, layout_jacobian)
    if not np.isfinite(result).all():
        raise PrintLayoutDomainError(f"Ad{number} material-layout world derivative is nonfinite")
    if closed:
        if np.array_equal(original[0], original[-1]):
            result[-1] = result[0]
        else:
            result = np.concatenate((result, result[:1]), axis=0)
    return result


def ad_print_world_layout_jacobian(material_geometry, scene, target, templates,
                                   layout8, geometry_q, *, fd_step=1e-6,
                                   laid_out_templates=None):
    """Return exact same-model world values and their shared-layout derivative."""
    if not isinstance(target, dict) or target.get("kind") != "ad_print":
        raise ValueError("An explicit existing print target is required")
    number = _advertisement_id(target.get("advertisement_id"))
    from .framewise_model import feature_world
    current = (apply_print_layout(templates, number, layout8) if laid_out_templates is None
               else laid_out_templates)
    world = feature_world(scene, target, current)
    record = print_layout_template(templates, number)["contours"][target["contour_key"]]
    prototype = record.get("prototype_material_uv", record["material_uv"])
    derivative = material_world_layout_jacobian(material_geometry, number, prototype,
        layout8, geometry_q, fd_step=fd_step, closed=bool(record["closed"]))
    if derivative.shape != (len(world), 3, 8):
        raise ValueError(f"The Ad{number} print layout and exact world closure inventories differ")
    return world, derivative


def ad2_print_world_layout_jacobian(material_geometry, scene, target, templates,
                                   layout8, geometry_q, *, fd_step=1e-6,
                                   laid_out_templates=None):
    """Compatibility wrapper preserving the original Ad2 world-value path."""
    if not isinstance(target, dict) or target.get("advertisement_id") != 2:
        raise ValueError("An explicit existing Ad2 print target is required")
    current = (apply_ad2_print_layout(templates, layout8) if laid_out_templates is None
               else laid_out_templates)
    return ad_print_world_layout_jacobian(material_geometry, scene, target, templates,
        layout8, geometry_q, fd_step=fd_step, laid_out_templates=current)
