"""Match the Ankara ice measurement definition to the frozen path labels."""
from __future__ import annotations

import re

import numpy as np

from common import HEIGHT, WIDTH

MIDPOINT = "original_ice_material_width_centre"
CONTOUR = "original_named_ice_material_contour"
HOUSE_CONTOUR = re.compile(r"ice\.[FR][1-4]:(?:centre|center|inner|middle|outer)$")
BLUE_AREAS = {"ice.goal_crease_left", "ice.goal_crease_right"}
ICE_EVIDENCE_FIELDS = ("source_bracket", "source_side_contrast", "ice_measurement_role")


def ice_roles(native):
    """A legacy path ID can end in 'outer' while its current label is a centreline."""
    return {key: CONTOUR if HOUSE_CONTOUR.fullmatch(path["label"]) or path["label"] in BLUE_AREAS
            else MIDPOINT for key, path in native.paths.items() if path["layer"] == "ice"}


def assert_ice_evidence(data, keep, roles):
    """Require a pair of original material flanks for each active line midpoint."""
    paths = data["path"]
    for role in (CONTOUR, MIDPOINT):
        registered = keep & np.isin(paths, [k for k, value in roles.items() if value == role])
        if registered.any() and ("ice_measurement_role" not in data
                                 or not np.all(data["ice_measurement_role"][registered] == role)):
            first = np.flatnonzero(registered)[0]
            raise ValueError(f"Active ice point has no matching material role, {paths[first]} at PTS {data['pts'][first]}")
    midpoint_paths = [path for path, role in roles.items() if role == MIDPOINT]
    selected = keep & np.isin(paths, midpoint_paths)
    if not selected.any():
        return
    if not set(ICE_EVIDENCE_FIELDS).issubset(data.keys()):
        raise ValueError("Active ice width midpoints have no measured source flank pair")
    if not np.all(data["ice_measurement_role"][selected] == MIDPOINT):
        raise ValueError("Active ice point uses the wrong material measurement")
    edges = data["source_bracket"][selected]
    contrast = data["source_side_contrast"][selected]
    if edges.shape != (selected.sum(), 2, 2) or contrast.shape != (selected.sum(), 2):
        raise ValueError("Each ice midpoint must bind precisely two visible flank samples")
    width = np.linalg.norm(edges[:, 1] - edges[:, 0], axis=1)
    on_sensor = ((edges >= 0.) & (edges <= [WIDTH - 1., HEIGHT - 1.])).all(axis=(1, 2))
    centered = np.linalg.norm(data["uv"][selected] - edges.mean(axis=1), axis=1) <= 1.e-7
    good = on_sensor & centered & (width > 0.) & (contrast > 0.).all(axis=1) & np.isfinite(contrast).all(axis=1)
    if not good.all():
        indices = np.flatnonzero(selected)[~good]
        first = indices[0]
        raise ValueError(f"{len(indices)} invalid material flank pairs, first {paths[first]} at PTS {data['pts'][first]}")
