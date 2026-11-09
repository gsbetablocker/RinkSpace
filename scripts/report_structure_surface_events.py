"""Compare saved surface scans on the common source clock without rescanning."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import copy
from itertools import combinations
import json
import os
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from threadpoolctl import threadpool_limits

from calibration.program_tracking.body_tracking.absolute_motion_state import absolute_reference, curve_from_packet
from calibration.program_tracking.body_tracking.programme_surface import CollisionSurface
from calibration.program_tracking.body_tracking.rebuild_prepare import digest, load
from calibration.program_tracking.body_tracking.sequence_skate_state import shoe_curve_from_packet
from calibration.program_tracking.body_tracking.skate_physics import fk_state
from calibration.program_tracking.body_tracking.trajectory_backend import load_reference
from rinkspace.path_policy import guard_project_write
from rinkspace.storage import AssetStore


PROJECT = Path(__file__).resolve().parents[1]
RUN = Path("runs/body_s01_20261029/structure_camera_1108")
SAMPLE_KINDS = ("native", "interpolated_midpoint", "interpolated_adaptive")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def file_ref(path):
    return {"path": str(path), "sha256": digest(path)}


def named_path(value):
    name, sep, path = value.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError("Use LABEL=PATH")
    return name, Path(path).resolve(strict=True)


def named_paths(items):
    result = dict(items)
    if len(result) != len(items):
        raise ValueError("Repeated candidate labels")
    return result


def legacy_scan_binding(path, scan, prepared):
    # The older audit wrapper kept the run input and numeric hash in its log.
    records = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    launcher = next(row["command"] for row in records if row["event"] == "body_performance_launcher")
    arguments = {name: Path(launcher[launcher.index("--" + name) + 1]).resolve(strict=True)
                 for name in ("prepared", "parameter", "output")}
    if arguments["output"] != scan.parent or arguments["prepared"] != prepared or "--scan" not in launcher:
        raise ValueError(f"Historical launch does not bind this scan and prepared input: {path}")
    if not any(row["event"] == "complete" for row in records):
        raise ValueError(f"Historical scan lacks completion in {path}")
    receipt = next(row for row in records if row["event"] == "fk_exported")
    return arguments["parameter"], receipt["parameter_sha256"], receipt["camera_sha256"]


def restore_geometry(prepared, parameter):
    saved = load(parameter)
    binding = json.loads(saved["camera_binding_json"].item())
    library = Path(binding["native_library"]).resolve(strict=True)
    if digest(library) != binding["native_library_sha256"]:
        raise ValueError(f"Saved camera runtime differs: {library}")
    variable = "RINKSPACE_BODY_NATIVE_LIBRARY"
    previous = os.environ.get(variable)
    os.environ[variable] = str(library)
    try:
        root, _, info, document, _ = load_reference(PROJECT, prepared)
    finally:
        if previous is None:
            del os.environ[variable]
        else:
            os.environ[variable] = previous
    if info["state_sha256"] != binding["state_sha256"]:
        raise ValueError(f"Prepared reference camera differs from the saved parameter: {parameter}")
    reference = absolute_reference(root, curve_from_packet(saved), root=prepared)
    reference.relative_skate_curve = shoe_curve_from_packet(saved)
    reference.binding = copy(root.binding)
    reference.binding.reference = reference
    reference.binding.x = saved["x"][:17].copy()
    reference.binding._packet = reference.binding._parameter_key = None
    return CollisionSurface(reference, document), reference, info


def inspect_topology(geometry, reference):
    faces = geometry.faces
    edges = faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    edge_ids, inverse, incidence = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True,
                                             return_counts=True)
    directed_balance = np.bincount(inverse, weights=np.sign(edges[:, 1] - edges[:, 0]), minlength=len(edge_ids))
    a, b = edge_ids.T
    adjacency = csr_matrix((np.ones(len(a)), (a, b)), shape=(geometry.vertex_count, geometry.vertex_count))
    components, labels = connected_components(adjacency, directed=False)
    face_component = labels[faces[:, 0]]
    np.testing.assert_array_equal(labels[faces], np.broadcast_to(face_component[:, None], faces.shape))
    edge_component = labels[a]
    sample = reference.times[:1]
    state = fk_state(reference, reference.at(sample), np.zeros((1, 69)))
    vertices = geometry.vertices(state)[0]
    records = {}
    for index in range(components):
        selected = faces[face_component == index]
        points = vertices[labels == index]
        edge_select = edge_component == index
        boundary = int(np.count_nonzero(incidence[edge_select] == 1))
        nonmanifold = int(np.count_nonzero(incidence[edge_select] > 2))
        bad_orientation = int(np.count_nonzero(edge_select & (incidence == 2) & (directed_balance != 0)))
        repeated_faces = int(len(selected) - len(np.unique(np.sort(selected, axis=1), axis=0)))
        triangle = vertices[selected] - points.mean(axis=0)
        degenerate = int(np.count_nonzero(np.linalg.norm(
            np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]), axis=1) <= 1e-15))
        closed = bool(len(selected) and not boundary and not nonmanifold)
        consistent = bool(len(selected) and not nonmanifold and not bad_orientation)
        volume = float(np.einsum("ni,ni->n", triangle[:, 0], np.cross(triangle[:, 1], triangle[:, 2])).sum() / 6)
        if not closed or not consistent or degenerate or repeated_faces:
            orientation = "local_oriented_plane_only"
        elif volume > 1e-15:
            orientation = "positive_closed_winding_at_first_native_sample"
        elif volume < -1e-15:
            orientation = "negative_closed_winding_at_first_native_sample"
        else:
            orientation = "closed_winding_without_resolved_signed_volume"
        parts = np.unique(geometry.parts[face_component == index])
        records[index] = dict(
            component=index, vertices=len(points), faces=len(selected),
            parts=[geometry.part_names[j] for j in parts], boundary_edges=boundary,
            nonmanifold_edges=nonmanifold, same_direction_shared_edges=bad_orientation,
            repeated_faces=repeated_faces, degenerate_faces_at_first_native=degenerate,
            closed_by_edge_incidence=closed, shared_edges_consistently_oriented=consistent,
            signed_volume_m3_at_first_native=volume if closed and consistent else None,
            normal_interpretation=orientation,
        )
    return face_component, records


def validate_clock(packet, reference):
    pts = packet["source_pts"]
    np.testing.assert_array_equal(pts, reference.p["pts"])
    if pts.ndim != 1 or pts.dtype.kind not in "iu" or np.any(np.diff(pts) <= 0):
        raise ValueError("Expected ordered original integer PTS")
    tb = np.asarray(reference.p["time_base"])
    native = pts.astype(float) * tb[0] / tb[1]
    native_times = np.asarray(reference.times)
    base = np.sort(np.r_[native_times, .5 * (native_times[1:] + native_times[:-1])])
    t, seconds = packet["times"], packet["source_seconds"]
    adaptive = packet.get("adaptive_sample", np.zeros(len(t), dtype=bool))
    if (t.shape != seconds.shape or adaptive.shape != t.shape or adaptive.dtype.kind != "b" or
            not np.isfinite(t).all() or not np.isfinite(seconds).all() or np.any(np.diff(t) <= 0)):
        raise ValueError("Invalid scan sample clock")
    np.testing.assert_allclose(t[~adaptive], base, rtol=0, atol=1e-10)
    np.testing.assert_allclose(seconds, t + native[0] - native_times[0], rtol=0, atol=1e-10)
    known = np.flatnonzero(~adaptive)[::2]
    np.testing.assert_allclose(seconds[known], native, rtol=0, atol=1e-10)
    source_index = np.full(len(t), -1, dtype=int)
    source_index[known] = np.arange(len(pts))
    sample_kind = np.where(adaptive, 2, 1)
    sample_kind[known] = 0
    keys = np.rint(seconds * 1e9).astype(np.int64)
    if np.any(np.diff(keys) <= 0):
        raise ValueError("Distinct scan samples cannot share a common nanosecond match key")
    return source_index, sample_kind, keys, tuple(map(int, tb))


def parse_ledger(packet, geometry, receipt):
    part_count = len(geometry.part_names)
    n = len(packet["times"])
    np.testing.assert_array_equal(packet["parts"], geometry.part_names)
    for field in ("crossing_counts", "grip_crossing_counts", "topology_uncertain_candidates"):
        values = packet[field]
        if values.shape != (n,) or values.dtype.kind not in "iu" or np.any(values < 0):
            raise ValueError(f"Invalid per-time scan counts: {field}")
    if (receipt["native_exposures"] != len(packet["source_pts"]) or
            receipt.get("samples", receipt.get("total_samples")) != n or
            receipt["selected_witnesses"] != len(packet["sample"]) or
            receipt["total_crossing_pairs"] != int(packet["crossing_counts"].sum())):
        raise ValueError("Saved scan and count receipt differ")
    if "faces" in receipt and (receipt["faces"] != len(geometry.faces) or receipt["vertices"] != geometry.vertex_count):
        raise ValueError("Scan receipt does not match the reconstructed mesh inventory")
    if ("source_grip_crossing_pairs" in receipt and
            receipt["source_grip_crossing_pairs"] != int(packet["grip_crossing_counts"].sum())):
        raise ValueError("Palm/shoe crossing count differs from receipt")
    ledger = packet["pair_counts"]
    if (ledger.ndim != 2 or ledger.shape[1] != 4 or ledger.dtype.kind not in "iu" or
            np.any(ledger < 0) or np.any(ledger[:, 0] >= n) or np.any(ledger[:, 1] > ledger[:, 2]) or
            np.any(ledger[:, 2] >= part_count) or np.any(ledger[:, 3] <= 0) or
            len(np.unique(ledger[:, :3], axis=0)) != len(ledger)):
        raise ValueError("Invalid sample/semantic-pair crossing ledger")
    sample_total = np.zeros(n, dtype=np.int64)
    np.add.at(sample_total, ledger[:, 0], ledger[:, 3])
    np.testing.assert_array_equal(sample_total, packet["crossing_counts"])
    for field, limit in (("sample", n), ("face_a", len(geometry.faces)), ("face_b", len(geometry.faces)),
                         ("vertex", geometry.vertex_count)):
        values = packet[field]
        if (values.shape != packet["sample"].shape or values.dtype.kind not in "iu" or
                np.any(values < 0) or np.any(values >= limit)):
            raise ValueError(f"Invalid selected witness index: {field}")
    gap = packet["initial_gap"]
    if gap.shape != packet["sample"].shape or not np.isfinite(gap).all() or np.any(gap >= 0):
        raise ValueError("Invalid signed oriented-plane witness")
    for field in ("normal", "bary"):
        value = packet[field]
        if value.shape != (len(gap), 3) or not np.isfinite(value).all():
            raise ValueError(f"Invalid witness vectors: {field}")
    np.testing.assert_allclose(np.linalg.norm(packet["normal"], axis=1), 1, rtol=0, atol=1e-10)
    np.testing.assert_allclose(packet["bary"].sum(1), 1, rtol=0, atol=1e-10)
    np.testing.assert_array_equal(packet["face_vertices"], geometry.faces[packet["face_b"]])
    if not np.all(np.any(geometry.faces[packet["face_a"]] == packet["vertex"][:, None], axis=1)):
        raise ValueError("Witness source endpoint differs from the saved face topology")
    pairs = np.sort(np.column_stack((geometry.parts[packet["face_a"]], geometry.parts[packet["face_b"]])), axis=1)
    counts, witnesses = {}, {}
    for pair in np.unique(ledger[:, 1:3], axis=0):
        key = tuple(geometry.part_names[i] for i in pair)
        matched = np.all(ledger[:, 1:3] == pair, axis=1)
        vec = np.zeros(n, dtype=np.int64)
        vec[ledger[matched, 0]] = ledger[matched, 3]
        counts[key] = vec
        witness_ids = np.flatnonzero(np.all(pairs == pair, axis=1))
        sample = packet["sample"][witness_ids]
        if len(np.unique(sample)) != len(sample) or not np.all(vec[sample] > 0):
            raise ValueError(f"Selected witness must bind one retained semantic pair/time: {key}")
        vec = np.full(n, -1, dtype=np.int64)
        vec[sample] = witness_ids
        witnesses[key] = vec
    if sum(np.count_nonzero(v >= 0) for v in witnesses.values()) != len(gap):
        raise ValueError("Some selected witnesses lack an exact semantic pair/time")
    return counts, witnesses


def scan_case(scan, prepared, parameter, scan_log):
    receipt_path = scan.with_suffix(".json")
    receipt = read(receipt_path)
    schema = receipt["schema"]
    if schema not in ("rinkspace.actual-surface-scan.v1", "rinkspace.adaptive-surface-scan.v1"):
        raise ValueError(f"Unsupported scan receipt: {receipt_path}")
    recorded_camera = receipt.get("camera_sha256")
    origin_log = None
    if "parameter_sha256" not in receipt:
        if scan_log is None:
            raise ValueError(f"Legacy scan requires its launch log: {scan}")
        declared, expected, recorded_camera = legacy_scan_binding(scan_log, scan, prepared)
        origin_log = file_ref(scan_log)
    else:
        declared, expected = receipt.get("parameter"), receipt["parameter_sha256"]
        if schema == "rinkspace.adaptive-surface-scan.v1":
            base = Path(receipt["base_scan"]).resolve(strict=True)
            if digest(base) != receipt["base_scan_sha256"]:
                raise ValueError("Adaptive scan's registered base scan changed")
            base_receipt = read(base.with_suffix(".json"))
            if base_receipt.get("parameter_sha256", expected) != expected:
                raise ValueError("Adaptive and base scan parameter hashes differ")
            declared = declared or base_receipt.get("parameter")
    if parameter is None and declared is None:
        raise ValueError(f"Supply the saved parameter for {scan}")
    parameter = (parameter or Path(declared)).resolve(strict=True)
    parameter_ref = file_ref(parameter)
    if parameter_ref["sha256"] != expected:
        raise ValueError(f"Scan parameter SHA256 differs: {parameter}")
    inputs = dict(input=file_ref(prepared / "input.npz"), evidence=file_ref(prepared / "source_evidence.json"))
    for key, field in (("input", "input_sha256"), ("evidence", "source_document_sha256")):
        if field in receipt and receipt[field] != inputs[key]["sha256"]:
            raise ValueError(f"Prepared scan {key} hash differs: {prepared}")
    geometry, reference, info = restore_geometry(prepared, parameter)
    if recorded_camera != info["state_sha256"]:
        raise ValueError("Prepared camera differs from the original scan camera")
    packet = load(scan)
    native_index, kinds, keys, time_base = validate_clock(packet, reference)
    face_component, topology = inspect_topology(geometry, reference)
    counts, witnesses = parse_ledger(packet, geometry, receipt)
    return dict(packet=packet, native_index=native_index, kinds=kinds, keys=keys,
                time_base=time_base, counts=counts, witnesses=witnesses, face_component=face_component,
                topology=topology, identity=dict(scan=file_ref(scan), receipt=file_ref(receipt_path),
                    legacy_launcher_log=origin_log, prepared=inputs, parameter=parameter_ref,
                    camera_state_sha256=recorded_camera, input_and_evidence_hashes_in_scan_receipt=
                    all(key in receipt for key in ("input_sha256", "source_document_sha256"))))


def sample_counts(kinds):
    return {name: int(np.count_nonzero(kinds == i)) for i, name in enumerate(SAMPLE_KINDS)}


def endpoint(case, index):
    seconds = float(case["packet"]["source_seconds"][index])
    native = case["native_index"][index]
    return dict(source_seconds=seconds, kind=SAMPLE_KINDS[case["kinds"][index]],
                source_pts=int(case["packet"]["source_pts"][native]) if native >= 0 else None)


def continuous_runs(indices):
    return np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1) if len(indices) else []


def signed_stats(case, ids):
    p = case["packet"]
    ids = ids[ids >= 0]
    if not len(ids):
        return dict(selected_witness_samples=0, min_signed_plane_witness_m=None, p95_witness_magnitude_m=None)
    values = p["initial_gap"][ids]
    peak = ids[np.argmin(values)]
    target = int(case["face_component"][p["face_b"][peak]])
    orientations = Counter(case["topology"][int(i)]["normal_interpretation"]
                           for i in case["face_component"][p["face_b"][ids]])
    return dict(
        selected_witness_samples=len(ids), min_signed_plane_witness_m=float(values.min()),
        p95_witness_magnitude_m=float(np.quantile(-values, .95)),
        witness_target_normal_classes=dict(orientations),
        strongest=dict(**endpoint(case, int(p["sample"][peak])), source_face=int(p["face_a"][peak]),
                       target_face=int(p["face_b"][peak]), target_component=target,
                       target_normal_interpretation=case["topology"][target]["normal_interpretation"]),
    )


def interval(case, indices):
    first, last = endpoint(case, indices[0]), endpoint(case, indices[-1])
    return dict(first=first, last=last, sampled_span_seconds=last["source_seconds"] - first["source_seconds"],
                samples=len(indices), sample_kinds=sample_counts(case["kinds"][indices]))


def rank_event(event, evidence):
    return (
        -(evidence["p95_witness_magnitude_m"] or 0.),
        evidence["min_signed_plane_witness_m"] or 0.,
        -event["sampled_span_seconds"],
        event["first"]["source_seconds"],
    )


def case_report(case):
    p = case["packet"]
    ranked = []
    for pair, counts in case["counts"].items():
        for indices in continuous_runs(np.flatnonzero(counts)):
            ranked.append(dict(pair=list(pair), **interval(case, indices),
                               triangle_pair_sample_count=int(counts[indices].sum()),
                               **signed_stats(case, case["witnesses"][pair][indices])))
    ranked.sort(key=lambda row: rank_event(row, row))
    tagged = p["grip_crossing_counts"]
    unknown = p["topology_uncertain_candidates"]
    targets = case["face_component"][p["face_b"]]
    by_target = {int(i): signed_stats(case, np.flatnonzero(targets == i)) for i in np.unique(targets)}
    return dict(
        identity=case["identity"], sample_kinds=sample_counts(case["kinds"]),
        retained_triangle_pair_sample_count=int(p["crossing_counts"].sum()),
        retained_positive_samples=sample_counts(case["kinds"][p["crossing_counts"] > 0]),
        **signed_stats(case, np.arange(len(p["sample"]))), semantic_pair_count=len(case["counts"]),
        ranked_semantic_pair_intervals=ranked,
        conditional_palm_shoe=dict(
            tagged_triangle_pair_sample_count=int(tagged.sum()),
            intervals=[dict(**interval(case, ids), tagged_triangle_pair_sample_count=int(tagged[ids].sum()))
                       for ids in continuous_runs(np.flatnonzero(tagged))],
            retained_deep_tagged_crossings_overlap=True,
        ),
        head_topology_uncertain=dict(broad_phase_candidate_sample_count=int(unknown.sum()),
                                     positive_samples=sample_counts(case["kinds"][unknown > 0])),
        topology=[dict(row, selected_target_witnesses=by_target.get(i, dict(selected_witness_samples=0)))
                  for i, row in case["topology"].items()],
    )


def compare(before, after):
    _, i, j = np.intersect1d(before["keys"], after["keys"], assume_unique=True, return_indices=True)
    np.testing.assert_allclose(before["packet"]["source_seconds"][i], after["packet"]["source_seconds"][j],
                               rtol=0, atol=1e-9)
    np.testing.assert_array_equal(before["kinds"][i], after["kinds"][j])
    transitions = ("disappeared", "retained", "new")
    totals = {status: Counter() for status in transitions}
    result = []
    for pair in sorted(before["counts"].keys() | after["counts"].keys()):
        counts_a = before["counts"].get(pair, np.zeros(len(before["keys"]), dtype=int))[i]
        counts_b = after["counts"].get(pair, np.zeros(len(after["keys"]), dtype=int))[j]
        cases = {}
        masks = ((counts_a > 0) & (counts_b == 0), (counts_a > 0) & (counts_b > 0),
                 (counts_a == 0) & (counts_b > 0))
        for status, mask in zip(transitions, masks, strict=True):
            kinds = sample_counts(before["kinds"][i[mask]])
            totals[status].update(kinds)
            runs = []
            for ids in continuous_runs(np.flatnonzero(mask)):
                record = interval(before, i[ids])
                record["before_triangle_pair_sample_count"] = int(counts_a[ids].sum())
                record["after_triangle_pair_sample_count"] = int(counts_b[ids].sum())
                for label, case, indices in (("before", before, i[ids]), ("after", after, j[ids])):
                    w = case["witnesses"].get(pair)
                    record[label] = signed_stats(case, np.empty(0, dtype=int) if w is None else w[indices])
                runs.append(record)
            runs.sort(key=lambda row: rank_event(row, row["before"] if status == "disappeared" else row["after"]))
            cases[status] = dict(semantic_pair_samples=int(mask.sum()), sample_kinds=kinds, intervals=runs)
        result.append(dict(pair=list(pair), **cases))
    return dict(
        common_samples=len(i), common_sample_kinds=sample_counts(before["kinds"][i]),
        before_only_samples=len(before["keys"]) - len(i), after_only_samples=len(after["keys"]) - len(j),
        semantic_pair_sample_transitions={status: dict(totals[status]) for status in transitions},
        by_semantic_pair=result,
    )


def main():
    store = AssetStore(PROJECT)
    run = store.data / RUN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", type=named_path, default=[], metavar="LABEL=SCAN")
    parser.add_argument("--prepared", type=Path, help="Shared prepared input for explicitly listed candidates")
    parser.add_argument("--candidate-prepared", action="append", type=named_path, default=[], metavar="LABEL=DIR")
    parser.add_argument("--parameter", action="append", type=named_path, default=[], metavar="LABEL=NPZ")
    parser.add_argument("--scan-log", action="append", type=named_path, default=[], metavar="LABEL=LOG")
    parser.add_argument("--out", type=Path, default=run / "out/surface_events_intermediate.json")
    args = parser.parse_args()

    scans = named_paths(args.candidate) if args.candidate else {
        "baseline": run / "baseline_v2/surface_scan.npz",
        "structure_round2": run / "structure_round2/surface/surface_scan.npz",
    }
    prepared = named_paths(args.candidate_prepared)
    params = named_paths(args.parameter)
    logs = named_paths(args.scan_log)
    out = guard_project_write(args.out)
    if not store.configured or not out.is_relative_to(store.data.resolve()) or out.exists():
        raise ValueError("Choose a new JSON report path in the configured Data root")
    if not (prepared.keys() | params.keys() | logs.keys()) <= scans.keys():
        raise ValueError("Unknown label in a candidate override")
    os.environ["RINKSPACE_BODY_BACKEND"] = "cpu"
    cases = {}
    for label, scan in scans.items():
        scan = scan.resolve(strict=True)
        is_baseline = scan == (run / "baseline_v2/surface_scan.npz").resolve()
        default_prepared = run / ("prepared_structure" if is_baseline else "prepared_structure2")
        directory = prepared.get(label, args.prepared or default_prepared).resolve(strict=True)
        log = logs.get(label, run / "baseline_scan_v2.log" if is_baseline else None)
        with threadpool_limits(limits=1):
            cases[label] = scan_case(scan, directory, params.get(label), log)
        print(f"Verified {label}: {len(cases[label]['keys'])} samples", flush=True)
    first = next(iter(cases.values()))
    for case in cases.values():
        np.testing.assert_array_equal(case["packet"]["source_pts"], first["packet"]["source_pts"])
        if case["time_base"] != first["time_base"]:
            raise ValueError("Candidate scan time bases differ")
    report = dict(
        schema="rinkspace.structure-surface-events.v1", source_time_base=list(first["time_base"]),
        source_exposures=len(first["packet"]["source_pts"]),
        source_first_last_pts=first["packet"]["source_pts"][[0, -1]].tolist(),
        classification=dict(
            retained="Non-coplanar actual triangle crossing outside attachment/head exclusions or retained deep source-tagged crossings. Saved counts measure incidence only.",
            conditional_palm_shoe="Source-supported palm/held-shoe face pairs may be tagged during specified intervals. The tagged counts include deep retained crossings; the two count sets overlap and must not be added.",
            intentional="Same-side gear interfaces and own-foot/gear cavity pairs were filtered before crossing. Their occurrence counts are absent from the saved scan.",
            head="Head/head broad-phase candidates are topologically uncertain. The counts do not establish crossing or legality.",
            unknown="Closed-solid containment, coplanar overlap, between-sample events and clinical/physical legality remain unverified.",
            signed_witness="The most negative selected crossing endpoint relative to its oriented target face is a local signed plane witness in conditional metres. Magnitude is not penetration depth.",
            winding="Target normals use original SMPL-X and equipment shell face winding. Open components only define a local half-space untangling witness. They cannot determine closed-solid inside/outside, SDF, or penetration depth. Closed consistent positive winding is a first-sample orientation cue; all-time non-self-intersection, vertex links and outward orientation remain unchecked.",
        ),
        interval_semantics="Consecutive saved samples with the same state. Duration is last minus first sampled time; singleton duration is zero. Comparison intervals use only common time keys. No continuous crossing certification.",
        ranking="Descending p95 magnitude of the selected oriented-plane witnesses, then ascending minimum signed witness and descending sampled span. Triangle-pair count does not rank severity.",
        transition_semantics="Disappear/retain/new refers to a semantic pair's crossing presence at an aligned sampled instant, not to identities of individual face pairs.",
        common_time_match="Source seconds matched by rounded integer nanosecond key and checked to 1 ns.",
        scans={label: case_report(case) for label, case in cases.items()},
        comparisons=[dict(before=a, after=b, **compare(cases[a], cases[b])) for a, b in combinations(cases, 2)],
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf8") as stream:
        json.dump(report, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write(chr(10))
    print(out, flush=True)


if __name__ == "__main__":
    main()
