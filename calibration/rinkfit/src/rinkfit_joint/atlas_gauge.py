"""Generate a new-stage atlas gauge without moving any pixel or world point.

The ad chart remains variable. Four selected latent UV coordinates define its
coordinate convention; no selected point becomes a measured world landmark.
This is a declared new graph / warm start, never an exact-problem resume.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(model, config, arrays, chart_name="ad/23"):
    out, cfg = copy.deepcopy(model), copy.deepcopy(config)
    if "runtime_atlas_gauge_v1" in cfg:
        raise ValueError("Atlas gauge already declared; do not stack it twice")
    ids = [i for i, b in enumerate(model["blocks"]) if b["name"] == chart_name]
    if len(ids) != 1:
        raise ValueError("Exact chart block must exist once")
    chart = ids[0]
    if model["blocks"][chart].get("constant") or chart_name in cfg.get("temporary_runtime_fixed_physical_blocks", []):
        raise ValueError("Requested chart is already fixed; do not add redundant gauges")
    active = (arrays["admission"] == 1) & (arrays["fit_role"] == 1)
    members, rows = set(), []
    for mi, material in enumerate(model["materials"]):
        for pi, path in enumerate(material["paths"]):
            if path.get("advertisement_block") != chart:
                continue
            ii = np.flatnonzero(active & (arrays["material"] == mi) & (arrays["path"] == pi))
            if not len(ii):
                continue
            bi = path.get("geometry_block", -1)
            if (path.get("kind") != 2 or not path.get("physical_point")
                    or path.get("points") != [[0, 0]] or bi < 0):
                raise ValueError("Chart has active fixed-local/curve evidence; not this atlas gauge")
            block = model["blocks"][bi]
            if (not block["name"].startswith("board_texture_shared_uv/") or block.get("constant")
                    or block.get("fixed_indices") != [0, 1, 3, 4]):
                raise ValueError("Expected exactly two unknown source-point chart coordinates")
            members.add(bi); rows.extend(map(int, ii))
    if len(members) < 2:
        raise ValueError("At least two distinct unknown points are required")
    selected = []
    ordered = sorted(members)
    for index, axis in [(2, "u"), (5, "v")]:
        values = np.array([model["blocks"][bi]["values"][index] for bi in ordered])
        lo, hi = int(np.argmin(values)), int(np.argmax(values))
        if values[hi] - values[lo] <= 1e-8:
            raise ValueError("Degenerate chart span cannot define an atlas gauge")
        for position, extremum in [(lo, "minimum"), (hi, "maximum")]:
            bi = ordered[position]
            block = out["blocks"][bi]
            selected.append({"block_index": bi, "block_name": block["name"], "index": index,
                             "axis": axis, "selection": "maximum_spanning_pair_"+extremum,
                             "fixed_chart_value": block["values"][index],
                             "world_coordinate_measured": False})
    selected_by_index = {}
    for row in selected:
        selected_by_index.setdefault(row["block_index"], set()).add(row["index"])
    if "audit_expected_free_physical_blocks" not in cfg:
        raise ValueError("Actual expected scalar stage registry must be provided")
    registry, before, after, changed = [], 0, 0, set()
    for entry in cfg["audit_expected_free_physical_blocks"]:
        row = copy.deepcopy(entry); bi = int(row["block_index"])
        before += int(row["expected_tangent_size"])
        if bi in selected_by_index:
            old = list(row["free_ambient_indices"])
            if not selected_by_index[bi].issubset(old):
                raise ValueError("Selected gauge coordinate not free in declared actual graph")
            row["free_ambient_indices"] = [j for j in old if j not in selected_by_index[bi]]
            row["expected_tangent_size"] = len(row["free_ambient_indices"])
            changed.add(bi)
        after += int(row["expected_tangent_size"])
        if row["expected_tangent_size"]:
            registry.append(row)
    if changed != set(selected_by_index) or before-after != 4:
        raise ValueError("Exactly four actual scalar freedoms must be removed")
    cfg["audit_expected_free_physical_blocks"] = registry
    cfg["runtime_atlas_gauge_v1"] = {
        "schema": "rinkfit.runtime_atlas_gauge.v1",
        "chart_block": chart, "chart_name": chart_name,
        "active_chart_rows": len(rows), "active_latent_blocks": len(members),
        "selected_coordinates": selected,
        "source_model_and_source_identity_unchanged": True,
        "new_stage_warm_start_only": True,
    }
    # An old CP describes a different graph identity after these scalar gauges.
    resume_fields = [k for k in cfg if k.startswith("resume_")]
    for k in resume_fields:
        cfg.pop(k)
    assert model == out
    receipt = {"schema": "rinkspace.explicit_atlas_gauge_transition.v1",
               "chart": chart_name, "chart_block": chart,
               "chart_remains_variable": True,
               "selected_unknown_coordinates": selected,
               "active_rows_using_chart": len(rows), "unknown_point_blocks": len(members),
               "physical_tangent_dimensions_before": before,
               "physical_tangent_dimensions_after": after,
               "all_numeric_block_values_bitwise_unchanged": True,
               "all_material_paths_unchanged": True,
               "entire_source_model_unchanged": True,
               "source_pixels_or_qualification_changed": False,
               "projection_equivalence_basis": "Every source-model byte, material path, clock and auxiliary is unchanged; only the configured runtime Ceres manifold and stage registry change.",
               "old_resume_fields_removed": resume_fields,
               "same_problem_resume_claimed": False,
               "source_qualification_unchanged_and_rechecked": True,
               "native_stage_guard_must_match_new_Ceres_manifolds": True,
               "global_feasible_set_equivalence_claimed": False,
               "limitation": "Local atlas gauge. Monitor original UV bounds and chart-domain inequalities. If a defining point ceases to bracket source extent, use a recorded rechart or absolute board(s,h) coordinates; do not move source UV or fix an inferred world panel as truth."}
    cfg["joint6_atlas_gauge_transition"] = {"schema": receipt["schema"], "new_graph_warm_start_only": True,
                                           "physical_tangent_dimensions_removed": 4}
    return out, cfg, receipt


def main():
    from .io_utils import read_json
    from .dense_joint_packet import _array
    from .observed_primal_restart import write_new, reference
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True);p.add_argument('--packet',required=True);p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();packet=read_json(a.packet);base=Path(a.packet).resolve().parent
    arrays={k:_array(packet['arrays'][v],base) for k,v in [('material','material_index'),('path','path'),('admission','admission'),('fit_role','fit_role')]}
    model,cfg,receipt=build(read_json(a.model),read_json(a.config),arrays)
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    receipt.update(model=reference(a.model),packet=reference(a.packet),configuration=reference(a.config))
    write_new(out/'config.json',cfg);write_new(out/'receipt.json',receipt)
    print(json.dumps(dict(configuration=reference(out/'config.json'),receipt=reference(out/'receipt.json'))))

if __name__=='__main__':main()
