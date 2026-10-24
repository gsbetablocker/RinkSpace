"""One-time data import. No historical solver, renderer or Python module imports.

This produces an UNFITTED structural seed, not a projection-preserving d78
replay. The original model is never edited. Runtime preview reads only the
resulting self-contained JSON; it never reruns this importer.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import statistics


D78_SHA256 = "d78b7012c80664131f88134be0f9f868c1d6742ad7516155fc131b72ca3110e9"


def build_structural_seed(model_path: str | Path, relations_path: str | Path) -> dict:
    path = Path(model_path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != D78_SHA256:
        raise ValueError("import-d78 accepts the recovered d78 model only; use a separately reviewed adapter for other formats/states")
    model = json.loads(raw)
    relation_file = Path(relations_path)
    relation_raw = relation_file.read_bytes()
    relations = json.loads(relation_raw)
    if relations.get("schema") != "rinkspace.user-geometry-relations.current.v1":
        raise ValueError("unsupported geometry-relations format")
    ties = relations["curling_structure"]["confirmed_layout_ties"]
    for key in ("F_and_R_columns", "equal_row_spacing", "equal_Lane_width", "collinear_Lane_painted_right_edges", "left_open_and_no_equal_length"):
        if not ties.get(key):
            raise ValueError(f"missing confirmed structural relation: {key}")
    blocks = [block["values"] for block in model["blocks"]]
    g = blocks[0]
    circles: dict[str, dict[str, tuple[float, float, float]]] = {}
    lanes: dict[int, list[tuple[float, ...]]] = {}
    for material in model["materials"]:
        label = material["label"]
        for branch in material["paths"]:
            if branch["kind"] == 1:
                match = re.fullmatch(r"ice\.(F[1-4]|R[1-4]|C0):(inner|middle|outer)", label)
                if match:
                    q = blocks[branch["geometry_block"]]
                    value = (float(q[0]), float(q[1]), float(q[branch.get("radius_index", 2)]))
                    circles.setdefault(match[1], {})[match[2]] = value
            elif branch["kind"] == 0:
                match = re.fullmatch(r"ice\.lane([1-4]):edge[0-2]", label)
                if match:
                    lanes.setdefault(int(match[1]), []).append(tuple(map(float, blocks[branch["geometry_block"]])))
    houses = {name: rings for name, rings in circles.items() if name != "C0"}
    rows = [(int(name[1:])-1, rings["outer"][1]) for name, rings in houses.items()]
    mean_i = statistics.mean(i for i, _ in rows)
    mean_y = statistics.mean(y for _, y in rows)
    variance_i = sum((i-mean_i)**2 for i, _ in rows)
    if variance_i <= 0:
        raise ValueError("not enough row coverage for a structural seed")
    pitch = sum((i-mean_i)*(y-mean_y) for i, y in rows) / variance_i
    y0 = mean_y - pitch*mean_i
    x_f = statistics.median(r["outer"][0] for n, r in houses.items() if n.startswith("F"))
    x_r = statistics.median(r["outer"][0] for n, r in houses.items() if n.startswith("R"))
    radius = statistics.median(r["outer"][2] for r in houses.values())
    ratios = {}
    for ring in ("middle", "inner"):
        ratios[ring] = statistics.median(r[ring][2]/r["outer"][2] for r in houses.values() if ring in r)
    widths = []
    for segments in lanes.values():
        if len(segments) > 1:
            ys = [0.5*(q[1]+q[3]) for q in segments]
            widths.append(max(ys)-min(ys))
    if not widths:
        raise ValueError("no opposite lane sides available to initialise width")
    support_left = [min(q[k] for q in lanes[i] for k in (0, 2)) for i in range(1, 5)]
    # Stored segment end positions are finite support, not confirmed physical
    # end measurements. This value is an explicit initial candidate only.
    right_seed = statistics.median(max(q[0], q[2]) for segs in lanes.values() for q in segs)
    if right_seed <= x_r:
        right_seed = max(q[k] for segs in lanes.values() for q in segs for k in (0, 2))
    parameters = [x_f, x_r, y0, pitch, radius, ratios["middle"], ratios["inner"], 0.0,
                  statistics.median(widths), right_seed]
    if not all(math.isfinite(x) for x in parameters):
        raise ValueError("non-finite seed")
    return {
        "schema": "rinkspace.scene.seed.v1",
        "scene_id": "ankara_2026.shared_structure_seed",
        "status": "unfitted_structure_candidate",
        "frame": {"frame_id": "d78_L1", "length_unit": "rink_length", "meters_per_unit": None},
        "geometry": {
            "board": {"kind": "polar_superellipse_course", "parameters": [.5, float(g[0])/2, float(g[1]), float(g[2]), float(g[3]), float(g[4])]},
            "curling": {"kind": "paired_equal_rows", "row_count": 4, "parameters": parameters,
                        "left_render_support": support_left, "ring_indices_present": [0, 1, 2],
                        "button_status": "not_imported_no_radius_evidence",
                        "original_houses_present": sorted(houses),
                        "right_edge_position_status": "initial_candidate_from_legacy_support_not_a_measured_endpoint"},
            "centre_circle": {"radius": circles["C0"]["outer"][2], "center": [0.0, 0.0, 0.0]},
        },
        "provenance": {
            "legacy_model_sha256": digest,
            "relations_sha256": hashlib.sha256(relation_raw).hexdigest(),
            "operation": "deterministic least-squares row positions and median shared dimensions",
            "retained_exact": ["board shape", "C0 radius"],
            "changed_by_structure": ["house sizes", "house positions", "lane center ties", "C0 center"],
            "projection_preserving": False,
            "source_measurements_created": 0,
            "unknowns": ["metric scale", "actual lane terminal positions", "button radius", "missing source-frame coverage"],
        },
        "scope": {"implemented": ["shared house/lane construction", "center circle", "continuous board"],
                  "not_yet_imported": ["other hockey marks", "all advertisement templates", "time-dependent camera", "observations"]},
    }


def import_d78(model_path: str | Path, relations_path: str | Path, output: str | Path) -> dict:
    seed = build_structural_seed(model_path, relations_path)
    target = Path(output).resolve()
    if target.exists():
        raise ValueError(f"refusing to overwrite existing seed: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(seed, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return {"scene_seed": str(target), "status": seed["status"], "fit_executed": False,
            "projection_preserving": False, "source_measurements_created": 0}
