"""Explicit continuous candidates for an already identified physical board edge.

The source point and its original target identity remain unchanged.  A caller
may replace the hard straight/corner segment boundary with an authenticated
coarse, connected range of the same upper or ice-contact edge.  Only existing
scene vertices are used; the range stays open and cannot become a whole ring.
"""
from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping

import numpy as np


SCHEMA = "rinkspace.continuous-board-association.v1"


def _candidate_parts(policy):
    if not isinstance(policy, Mapping) or policy.get("schema") != SCHEMA:
        raise ValueError("An explicit continuous board association policy is required")
    parts = policy.get("candidate_parts")
    if (not isinstance(parts, (list, tuple)) or not parts
            or any(not isinstance(part, str) or not part for part in parts)
            or len(set(parts)) != len(parts)):
        raise ValueError("The continuous board range requires unique named candidate parts")
    return tuple(parts)


def with_continuous_board_association(target, policy):
    """Add matching metadata without replacing the original physical identity."""
    parts = _candidate_parts(policy)
    if (not isinstance(target, Mapping) or target.get("kind") != "board_edge"
            or target.get("role") not in ("top", "contact")):
        raise ValueError("Continuous board association requires an identified board edge role")
    if target.get("part") not in parts:
        raise ValueError("The original board part is outside the source-confirmed continuous range")
    result = deepcopy(dict(target))
    result["association_policy"] = deepcopy(dict(policy))
    return result


def continuous_board_path(path, part_records, policy, *, original_part):
    """Return one finite connected scene path, including its shared end vertex.

    The caller selects the role before entry.  Candidate parts must follow the
    scene's cyclic physical order and segment inventory; a disconnected union,
    invented connector or complete-ring search is rejected.
    """
    names = _candidate_parts(policy)
    if original_part not in names:
        raise ValueError("The original board part is outside the source-confirmed continuous range")
    vertices = np.asarray(path, dtype=np.float64)
    if (vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) < 2
            or not np.isfinite(vertices).all() or np.array_equal(vertices[0], vertices[-1])):
        raise ValueError("Continuous board association requires a finite unclosed scene inventory")
    if not isinstance(part_records, (list, tuple)) or not part_records:
        raise ValueError("The scene has no named physical board parts")
    selected = []
    for name in names:
        matches = [(index, record) for index, record in enumerate(part_records)
                   if isinstance(record, Mapping) and record.get("id") == name]
        if len(matches) != 1:
            raise ValueError("Missing or ambiguous continuous board part: " + name)
        index, record = matches[0]
        first, count = record.get("first_vertex"), record.get("vertex_count")
        if (isinstance(first, bool) or isinstance(count, bool)
                or not isinstance(first, int) or not isinstance(count, int)
                or not 0 <= first < len(vertices) or not 0 < count < len(vertices)):
            raise ValueError("Invalid continuous board part vertex range: " + name)
        selected.append((index, first, count))
    for current, following in zip(selected, selected[1:]):
        if ((current[0] + 1) % len(part_records) != following[0]
                or (current[1] + current[2]) % len(vertices) != following[1]):
            raise ValueError("Continuous board candidates must share consecutive physical seams")
    count = sum(record[2] for record in selected)
    if count >= len(vertices):
        raise ValueError("A source-confirmed continuous board range must remain open, not a whole ring")
    first = selected[0][1]
    return vertices[(first + np.arange(count + 1)) % len(vertices)].copy()
