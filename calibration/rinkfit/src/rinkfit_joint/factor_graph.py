"""Inspect the block structure of a physical, multi-shot rink fit.

This is executable *architecture/preflight* code, not a numerical optimizer.
It neither projects points nor changes observation weights.  In particular,
Schur components are formed from actual factor connectivity: correlated W
groups and source-order factors can connect several station variables.

CLI: python -m rinkfit_joint.factor_graph graph_spec.json --output graph_audit.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA = "rinkfit.joint_factor_graph.v1"
NUISANCE_SCOPES = frozenset({"row", "query"})
VALID_SCOPES = frozenset({"scene", "rig", "optics", "shot", "row", "query", "epigraph"})


def _indexed(items: Iterable[Mapping[str, Any]], label: str) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for item in items:
        key = str(item["id"])
        if key in out:
            raise ValueError(f"Duplicate {label} id: {key}")
        out[key] = item
    return out


def expected_row_blocks(spec: Mapping[str, Any], row: Mapping[str, Any]) -> list[str]:
    """Return an exact dependency union when the caller supplies local support.

    row keys: shot_id, material_id, control_block_ids, nuisance_block_ids.
    `control_block_ids` MUST include the union of spline support over the whole
    allowed rolling-shutter interval, plus any gauge-substitution dependencies.
    The function cannot infer that interval from an unimplemented camera model.
    """
    shots = _indexed(spec["shots"], "shot")
    rigs = _indexed(spec["rigs"], "rig")
    optics = _indexed(spec["optics"], "optics")
    materials = _indexed(spec["materials"], "material")
    shot = shots[str(row["shot_id"])]
    dependencies = (
        list(spec["scene"]["block_ids"])
        + list(rigs[str(shot["rig_id"])]["block_ids"])
        + list(optics[str(shot["optics_id"])]["block_ids"])
        + list(materials[str(row["material_id"])]["block_ids"])
        + list(shot.get("reference_angle_block_ids", ()))
        + list(row["control_block_ids"])
        + list(row.get("nuisance_block_ids", ()))
    )
    return list(dict.fromkeys(map(str, dependencies)))


class _UnionFind:
    def __init__(self, keys: Iterable[str]) -> None:
        self.parent = {k: k for k in keys}

    def find(self, key: str) -> str:
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def elimination_components(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Plan connected local nuisance elimination without breaking full W.

    A component is an elimination unit, not automatically a Ceres parameter
    block.  Use component-local factorization/custom reduction, or explicitly
    pack its variables before selecting it as a Schur group.  A singular local
    Hessian/KKT block requires a stable constrained factorization or retention;
    this topology plan is not a rank or invertibility certificate.
    """
    blocks = _indexed(spec["blocks"], "block")
    local = {
        key for key, b in blocks.items()
        if b["scope"] in NUISANCE_SCOPES and int(b.get("free_size", b["size"])) > 0
    }
    uf = _UnionFind(local)
    for factor in spec["factors"]:
        keys = sorted(local.intersection(map(str, factor["block_ids"])))
        for key in keys[1:]:
            uf.union(keys[0], key)
    groups: dict[str, list[str]] = defaultdict(list)
    for key in sorted(local):
        groups[uf.find(key)].append(key)
    component_factors: dict[str, list[str]] = defaultdict(list)
    retained_neighbors: dict[str, set[str]] = defaultdict(set)
    for factor in spec["factors"]:
        deps = set(map(str, factor["block_ids"]))
        roots = {uf.find(key) for key in deps.intersection(local)}
        if len(roots) > 1:
            raise AssertionError("A factor crosses two supposedly separate nuisance components")
        for root in roots:
            component_factors[root].append(str(factor["id"]))
            retained_neighbors[root].update(deps.difference(local))
    components = []
    for root, keys in sorted(groups.items()):
        components.append({
            "id": f"local/{root}",
            "block_ids": keys,
            "free_scalars": sum(int(blocks[k].get("free_size", blocks[k]["size"])) for k in keys),
            "factor_ids": sorted(component_factors[root]),
            "retained_neighbors": sorted(retained_neighbors[root]),
            "can_treat_individual_blocks_as_schur_independent_set": len(keys) == 1,
        })
    retained = [
        key for key, b in blocks.items()
        if key not in local and int(b.get("free_size", b["size"])) > 0
    ]
    return {
        "schema": "rinkfit.joint_elimination_plan.v1",
        "components": components,
        "component_count": len(components),
        "largest_local_free_scalars": max((c["free_scalars"] for c in components), default=0),
        "retained_block_ids": sorted(retained),
        "retained_free_scalars": sum(int(blocks[k].get("free_size", blocks[k]["size"])) for k in retained),
        "independent_point_schur_assumption_valid": all(len(c["block_ids"]) == 1 for c in components),
        "invertibility_certified": False,
        "numerical_action": "Factor each connected local constrained block; retain it if singular. Do not diagonalize W or delete order/query couplings.",
    }


def validate_joint_graph(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Validate ownership, coupling, and one-statistical-unit-per-source-row.

    The complete factor list should include pixel, domain, RS/query, source
    order, full-W, and physical/motion factors. `active_fit_row_ids` declares
    the exact set that the W partition must cover. A non-statistical factor
    may of course refer to a source row more than once.
    """
    errors: list[str] = []
    warnings: list[str] = []
    if spec.get("schema") != SCHEMA:
        errors.append(f"Expected schema {SCHEMA}")
    try:
        blocks = _indexed(spec["blocks"], "block")
        shots = _indexed(spec["shots"], "shot")
        rigs = _indexed(spec["rigs"], "rig")
        optics = _indexed(spec["optics"], "optics")
        materials = _indexed(spec["materials"], "material")
        factors = _indexed(spec["factors"], "factor")
    except (KeyError, TypeError, ValueError) as exc:
        return {"schema": "rinkfit.joint_graph_audit.v1", "valid": False, "errors": [str(exc)], "warnings": []}
    if len(shots) != int(spec.get("expected_shot_count", 9)):
        errors.append("Shot count differs from the explicit programme contract")
    if len(rigs) != int(spec.get("expected_rig_count", 3)):
        errors.append("Rig count differs from the explicit programme contract")
    scene_blocks = list(map(str, spec.get("scene", {}).get("block_ids", ())))
    if not scene_blocks:
        errors.append("One shared scene must explicitly provide its common block_ids")
    for key, block in blocks.items():
        scope = block.get("scope")
        if scope not in VALID_SCOPES:
            errors.append(f"Unknown block scope {scope}: {key}")
        size, free = int(block.get("size", 0)), int(block.get("free_size", block.get("size", 0)))
        if size <= 0 or not 0 <= free <= size:
            errors.append(f"Invalid ambient/free dimension: {key}")
        owner = str(block.get("owner", ""))
        domain = {"rig": rigs, "optics": optics, "shot": shots}.get(str(scope))
        if domain is not None and owner not in domain:
            errors.append(f"Unknown {scope} block owner: {key} -> {owner}")
    for key in scene_blocks:
        if key not in blocks or blocks[key].get("scope") != "scene":
            errors.append(f"Common scene dependency must be a shared scene block: {key}")
    for shot_id, shot in shots.items():
        if str(shot.get("rig_id")) not in rigs or str(shot.get("optics_id")) not in optics:
            errors.append(f"Unknown rig/optics assigned to {shot_id}")
    for material_id, material in materials.items():
        for key in map(str, material.get("block_ids", ())):
            if key not in blocks or blocks[key].get("scope") != "scene":
                errors.append(f"Material {material_id} must own shared scene geometry, not shot-local geometry: {key}")
    for owner_scope, registry in (("rig", rigs), ("optics", optics)):
        for owner, item in registry.items():
            for key in map(str, item.get("block_ids", ())):
                if key not in blocks or blocks[key].get("scope") != owner_scope or str(blocks[key].get("owner")) != owner:
                    errors.append(f"Invalid {owner_scope} dependency {owner}/{key}")
    active_values = list(map(str, spec.get("active_fit_row_ids", ())))
    active = set(active_values)
    if len(active_values) != len(active):
        errors.append("active_fit_row_ids contains duplicate source identities")
    membership: Counter[str] = Counter()
    unit_ids: set[str] = set()
    touched: Counter[str] = Counter()
    rank_total = 0
    kinds: Counter[str] = Counter()
    for key, factor in factors.items():
        kind = str(factor.get("kind", ""))
        kinds[kind] += 1
        deps = list(map(str, factor.get("block_ids", ())))
        if len(deps) != len(set(deps)):
            errors.append(f"Duplicate parameter pointer within factor {key}")
        missing = set(deps).difference(blocks)
        if missing:
            errors.append(f"Missing blocks in {key}: {sorted(missing)}")
            continue
        touched.update(deps)
        shot_ids = set(map(str, factor.get("shot_ids", ())))
        if not shot_ids.issubset(shots):
            errors.append(f"Unknown shot in factor {key}")
            continue
        assigned_rigs = {str(shots[s]["rig_id"]) for s in shot_ids}
        assigned_optics = {str(shots[s]["optics_id"]) for s in shot_ids}
        for dep in deps:
            block = blocks[dep]
            scope, owner = block.get("scope"), str(block.get("owner", ""))
            if scope == "shot" and owner not in shot_ids:
                errors.append(f"Factor {key} leaks controls from shot {owner}")
            if shot_ids and scope == "rig" and owner not in assigned_rigs:
                errors.append(f"Factor {key} leaks a different rig {owner}")
            if shot_ids and scope == "optics" and owner not in assigned_optics:
                errors.append(f"Factor {key} leaks different optics {owner}")
        if kind in {"motion", "motion_prior", "temporal_regularizer"} and len(shot_ids) != 1:
            errors.append(f"Temporal factor crosses a cut or has no explicit shot: {key}")
        if kind == "statistical":
            group_id = str(factor.get("statistical_group_id", ""))
            if not group_id or group_id in unit_ids:
                errors.append(f"Missing/duplicate statistical group id: {key}")
            unit_ids.add(group_id)
            row_ids = list(map(str, factor.get("row_ids", ())))
            rank = int(factor.get("rank", 0))
            if not row_ids or len(row_ids) != len(set(row_ids)) or not 0 < rank <= 2 * len(row_ids):
                errors.append(f"Invalid full-W row list/rank: {key}")
            if not factor.get("whitening_provenance"):
                errors.append(f"Missing numeric-W provenance: {key}")
            membership.update(row_ids)
            rank_total += rank
    if active:
        uncovered = sorted(row for row in active if membership[row] == 0)
        repeated = sorted(row for row in active if membership[row] > 1)
        unexpected = sorted(set(membership).difference(active))
        if uncovered:
            errors.append(f"{len(uncovered)} active rows absent from statistics; first: {uncovered[:5]}")
        if repeated:
            errors.append(f"{len(repeated)} active rows counted in multiple statistical units; first: {repeated[:5]}")
        if unexpected:
            errors.append(f"{len(unexpected)} non-fit rows enter statistics; first: {unexpected[:5]}")
    else:
        warnings.append("No active_fit_row_ids supplied: full-source W coverage was not checked")
    untouched = [key for key, block in blocks.items() if int(block.get("free_size", block["size"])) and not touched[key]]
    if untouched:
        warnings.append(f"{len(untouched)} free blocks have no factors and must not enter the numerical state")
    report: dict[str, Any] = {
        "schema": "rinkfit.joint_graph_audit.v1", "valid": not errors,
        "errors": errors, "warnings": warnings,
        "shots": len(shots), "rigs": len(rigs), "optics_groups": len(optics),
        "blocks": len(blocks), "factors": len(factors), "factor_kinds": dict(sorted(kinds.items())),
        "free_scalars_by_scope": {
            scope: sum(int(b.get("free_size", b["size"])) for b in blocks.values() if b.get("scope") == scope)
            for scope in sorted(VALID_SCOPES)
        },
        "statistical_units": len(unit_ids), "full_W_residual_rank": rank_total,
        "active_source_rows": len(active), "unobserved_free_block_ids": untouched,
        "gauge_rank_certified": False,
    }
    if not errors:
        report["elimination"] = elimination_components(spec)
    return report


def grouped_half_cost(
    native_errors: Mapping[str, Iterable[float]],
    groups: Iterable[Mapping[str, Any]],
    active_row_ids: Iterable[str],
) -> dict[str, Any]:
    """Reference full-W cost with exact membership; useful for small fixtures.

    Each group has id, row_ids, W (rank x 2N), whitening_provenance. It is
    deliberately dense/stdlib and should not be used as the large production
    evaluator. Scalar sigmas must be materialized as their real 2x2 W upstream.
    Invalid projections must be rejected before this function, not zeroed.
    """
    active_list = list(map(str, active_row_ids))
    active = set(active_list)
    if len(active_list) != len(active):
        raise ValueError("Duplicate active row id")
    seen: Counter[str] = Counter()
    costs: dict[str, float] = {}
    rank = 0
    for group in groups:
        gid = str(group["id"])
        if gid in costs or not group.get("whitening_provenance"):
            raise ValueError(f"Duplicate group or missing W provenance: {gid}")
        ids = list(map(str, group["row_ids"]))
        if len(ids) != len(set(ids)) or not ids or set(ids).difference(active):
            raise ValueError(f"Invalid source membership: {gid}")
        vector: list[float] = []
        for row_id in ids:
            pair = list(map(float, native_errors[row_id]))
            if len(pair) != 2 or not all(map(math.isfinite, pair)):
                raise ValueError(f"Invalid native error for {row_id}")
            vector.extend(pair)
        matrix = group["W"]
        if not 0 < len(matrix) <= len(vector):
            raise ValueError(f"Invalid rank: {gid}")
        whitened = []
        for coeffs in matrix:
            coeffs = list(map(float, coeffs))
            if len(coeffs) != len(vector) or not all(map(math.isfinite, coeffs)):
                raise ValueError(f"Invalid W matrix: {gid}")
            whitened.append(math.fsum(a * b for a, b in zip(coeffs, vector)))
        costs[gid] = 0.5 * math.fsum(value * value for value in whitened)
        rank += len(matrix)
        seen.update(ids)
    if any(seen[row] != 1 for row in active):
        raise ValueError("Every active row must occur in exactly one full-W statistical unit")
    return {"half_cost": math.fsum(costs.values()), "by_group": costs, "rank": rank, "source_rows": len(active)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate_joint_graph(json.loads(args.spec.read_text(encoding="utf-8")))
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    if not report["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
