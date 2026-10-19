"""Opt-in atomic replacement of a derived line by its reviewed native queries.

The fixed native chart and original localization factor define the likelihood.
The forward kernel intersects the physical curve with each saved profile query;
it must not substitute an Euclidean nearest foot. Huber2 remains row-wise on
the new fixed whitening basis, so costs before/after are different objectives.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import numpy as np

from .official_static_surface_dataset import read_reference, reference
from .first_shot_noise_revision import source_members_digest
from .source_evidence_whitening import SourceWhiteningOperator, latent_whitening_operator

SCHEMA = "rinkspace.authenticated-native-line-replacement.v1"


def _check_reference(ref):
    if reference(ref["path"]) != ref:
        raise ValueError("Native line evidence changed")


def _same(a, b, label):
    if not np.array_equal(np.asarray(a), np.asarray(b)):
        raise ValueError("Native line changed " + label)


def _authenticate(item, review, frame, group, binding):
    """Authenticate original pixels, independent identity and the current L."""
    if (item.get("native_material_identity_independently_RGB_reviewed_for_all_existing_query_IDs") is not True
            or review is None):
        raise ValueError("Every native query requires an explicit independent material identity review")
    old = SourceWhiteningOperator.from_record(item["old_operator"])
    current = group["operator"]
    if (item["old_point_indices"] != list(group["point_indices"])
            or old.source_point_ids != current.source_point_ids or old.group_id != current.group_id
            or not np.array_equal(old.matrix, current.matrix)
            or source_members_digest(frame, group["point_indices"]) != item["current_noise_source_members_sha256"]):
        raise ValueError("Native replacement does not match its entire current derived group")
    if item["original_source_packet"] != frame["source_packet"]:
        raise ValueError("Native replacement has a different source packet")
    packet = read_reference(frame["source_packet"])
    if packet["ice_source_readout_packet"] != item["native_readout_packet"]:
        raise ValueError("Native line readout packet differs")
    native = read_reference(item["native_readout_packet"])
    for source in (packet["source"], native["source"]):
        if (int(source["native_pts"]) != int(frame["native_pts"])
                or int(source["actual_pts"]) != int(frame["native_pts"])
                or source["time_base"] != binding["source_clock"]["time_base"]
                or source["source_video_sha256"] != binding["source"]["sha256"]):
            raise ValueError("Native line source exposure or clock changed")
    if packet["source"]["decoded_BGR_sha256"] != native["source"]["decoded_BGR_sha256"]:
        raise ValueError("Native line source pixels differ")
    features = [f for f in native["features"] if f["id"] == item["feature_id"]]
    if len(features) != 1 or not features[0].get("source_line_fit", {}).get("current_frame_only"):
        raise ValueError("One current native line support is required")
    feature = features[0]
    line = feature["source_line_fit"]
    direct = line["direct_readouts"]
    digest = hashlib.sha256(json.dumps(direct, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != item["old_native_support_canonical_sha256"] or digest != review["native_support_sha256"]:
        raise ValueError("Native line final support changed")
    ids = [f"{frame['native_pts']}/{item['feature_id']}/query/{r['source_evidence']['query_index']}" for r in direct]
    if (ids != item["native_source_point_ids"] or ids != item["identity_scope_evidence_query_IDs"]
            or ids != [r["query_id"] for r in review["native_queries"]]
            or len(set(ids)) != len(ids) or len(direct) != item["native_count"]):
        raise ValueError("Native identity review must cover the entire exact original query inventory")
    uv = np.asarray([r["uv"] for r in direct], float)
    queries = np.asarray([r["source_evidence"]["query_uv"] for r in direct], float)
    normals = np.asarray([r["source_evidence"]["profile_normal"] for r in direct], float)
    chart = np.asarray(line["unit_normal"], float)
    _same(chart, item["fixed_chart"]["unit_normal"], "chart normal")
    if not np.isclose(np.linalg.norm(chart), 1., rtol=0, atol=1e-10):
        raise ValueError("Native chart normal must be a recorded unit direction")
    for i, (raw, declared, accepted) in enumerate(zip(direct, item["native"], review["native_queries"], strict=True)):
        evidence = raw["source_evidence"]
        if (evidence.get("actual_source_support_outside_person_and_broadcast_masks") is not True
                or accepted.get("independent_RGB_material_identity_confirmed") is not True
                or accepted.get("actual_source_support_outside_person_and_broadcast_masks") is not True):
            raise ValueError("Unconfirmed or masked native material query cannot enter FIT")
        _same(raw["uv"], declared["original_native_uv"], "original UV")
        _same(raw["uv"], accepted["original_native_uv"], "reviewed UV")
        _same(evidence["query_uv"], declared["original_query_uv"], "query UV")
        _same(evidence["profile_normal"], declared["profile_normal"], "profile normal")
        _same(chart, declared["noise_coordinate_normal"], "native noise coordinate")
        _same(evidence["actual_source_sides_uv"], accepted["actual_source_sides_uv"], "reviewed material sides")
        if declared["query_id"] != ids[i] or declared["source_point_id"] != ids[i]:
            raise ValueError("A new native ID must be its existing query ID")
    factor = item["native_noise_factor"]
    _check_reference(factor["reference"])
    with np.load(factor["reference"]["path"], allow_pickle=False) as saved:
        L = np.asarray(saved[factor["array_key"]], float)
    if (list(L.shape) != factor["shape"] or L.shape[0] != len(ids) or not np.isfinite(L).all()
            or hashlib.sha256(L.tobytes()).hexdigest() != factor["matrix_float64_bytes_sha256"]
            or np.linalg.matrix_rank(L) != len(ids)):
        raise ValueError("Original native localization factor must retain its complete measured row rank")
    noise = group.get("localization_noise_revision")
    if noise is None or not np.array_equal(L, np.asarray(noise["localization_noise_factor"])):
        raise ValueError("Native replacement must preserve the current original localization factor L")
    if item["material_target"].get("kind") != "ice_curve" or feature.get("measurement_operator") != "paint_center":
        raise ValueError("This opt-in supports only the original physical ice material centerline")
    return uv, queries, normals, chart, L, ids


def _native_operator(ids, chart, L, group_id):
    basis = np.zeros((len(ids), 2*len(ids)))
    for i in range(len(ids)):
        basis[i, 2*i:2*i+2] = chart
    return latent_whitening_operator(ids, np.eye(len(ids)), explicit_noise_factor=L,
        physical_basis=basis, group_id=group_id,
        covariance_origin="original_native_chart_localization_factor_without_line_compression")


def certify_native_group(frame, row, group):
    """Coverage guard: verify the active native factor, never rebuild rank-2 W."""
    proof = group["native_line_replacement"]
    item = proof["item"]
    noise = proof["localization_noise_factor"]
    ids = item["native_source_point_ids"]
    _same(row["uv"], [r["original_native_uv"] for r in item["native"]], "active native UV")
    if row["source_point_ids"] != ids or group["operator"].source_point_ids != tuple(ids):
        raise ValueError("Coverage native source membership changed")
    queries = row.get("native_line_queries", {})
    _same(queries.get("query_uv"), [r["original_query_uv"] for r in item["native"]], "active query UV")
    _same(queries.get("profile_normal"), [r["profile_normal"] for r in item["native"]], "active query profile normal")
    expected = _native_operator(ids, item["fixed_chart"]["unit_normal"], np.asarray(noise), group["operator"].group_id)
    if not np.array_equal(expected.matrix, group["operator"].matrix):
        raise ValueError("Native coverage received a different localization factor or whitening basis")
    return dict(native_line_replacement_authenticated=True, native_query_count=len(ids),
        native_noise_coordinate="original_fixed_source_chart_normal_px",
        original_native_support_sha256=item["old_native_support_canonical_sha256"],
        compressed_estimator_restored=False, source_mean_or_width_scale_rebuilt=False)


def revise_native_lines(frames, revision_reference, binding):
    """Replace reviewed whole line groups after existing source/noise revisions."""
    revision = read_reference(revision_reference)
    if revision.get("schema") != SCHEMA:
        raise ValueError("Explicit authenticated native-line replacement schema required")
    base = read_reference(revision["base_binding"])
    if any(base.get(k) != binding.get(k) for k in ("source", "source_clock", "complete_fit_input", "source_revision", "source_localization_noise_revision", "model")):
        raise ValueError("Native line revision belongs to another active source binding")
    recipe = read_reference(revision["recipe"])
    identity = read_reference(revision["identity_review"])
    if (recipe["base_binding"] != revision["base_binding"] or identity["base_binding"] != revision["base_binding"]
            or recipe.get("identity_evidence") != revision["identity_review"]
            or identity.get("schema") != "rinkspace.independent-native-static-material-identity-scope.v1"
            or identity.get("whole_original_group_replacement_only") is not True
            or identity.get("world_camera_model_or_fit_residual_used_for_identity") is not False
            or identity.get("boolean_motion_or_current_identity_flags_used_as_admission") is not False):
        raise ValueError("Native replacement requires a complete independent RGB material identity scope")
    reviews = {(int(r["native_pts"]), r["group_id"]): r for r in identity["groups"]}
    replacements = {}
    for item in recipe["groups"]:
        key = int(item["native_pts"]), item["group_id"]
        if key in replacements:
            raise ValueError("One native group cannot replace a derived factor twice")
        replacements[key] = item
    if set(replacements) != set(reviews) or not replacements:
        raise ValueError("Review and complete replacement groups differ")
    result, used, mappings = [], set(), []
    old_global = new_global = 0
    for original in frames:
        active = {g["operator"].group_id: g for g in original["whitening_groups"]}
        selected = {gid: replacements[(int(original["native_pts"]), gid)] for gid in active
                    if (int(original["native_pts"]), gid) in replacements}
        old_count = sum(g["operator"].rank for g in original["whitening_groups"])
        old_ordinary = 2*(original["source_point_count"]-sum(len(g["point_indices"]) for g in original["whitening_groups"]))
        old_total = old_count+old_ordinary
        if not selected:
            result.append(original)
            mappings.append(dict(frame=original["index"],unchanged=True,old_global_formal_start=old_global,
                new_global_formal_start=new_global,old_formal_rows=old_total,new_formal_rows=old_total))
            old_global += old_total;new_global += old_total;continue
        flattened = [(r,j) for r in original["rows"] for j in range(len(r["uv"]))]
        prepared = {}
        removed = set()
        for gid,item in selected.items():
            group=active[gid];key=int(original["native_pts"]),gid
            values=_authenticate(item,reviews.get(key),original,group,binding)
            old_ids=[flattened[ix][0]["source_point_ids"][flattened[ix][1]] for ix in group["point_indices"]]
            if old_ids!=[r["source_point_id"] for r in item["old_representatives"]]:
                raise ValueError("Derived representative identity/order changed")
            source_rows={id(flattened[ix][0]) for ix in group["point_indices"]}
            whole={i for i,(r,_)in enumerate(flattened)if id(r)in source_rows}
            if whole!=set(group["point_indices"]) or removed.intersection(whole):
                raise ValueError("Native replacement must cover whole original source rows and group")
            for ix in whole:
                r,_=flattened[ix]
                if r["target"]!=item["material_target"] or r["point_correspondence"]:
                    raise ValueError("Native replacement changes its physical curve target")
            prepared[gid]=(item,group,flattened[group["point_indices"][0]][0],values)
            removed.update(whole);used.add(key)
        rows=[];point_map={};cursor=0;old_cursor=0
        for row in original["rows"]:
            indices=range(old_cursor,old_cursor+len(row["uv"]));old_cursor+=len(row["uv"])
            if all(i in removed for i in indices):continue
            if any(i in removed for i in indices):raise ValueError("Partial source row replacement is forbidden")
            rows.append(row)
            for i in indices:point_map[i]=cursor;cursor+=1
        new_groups={}
        for gid,(item,group,oldrow,values)in prepared.items():
            uv,queries,normals,chart,L,ids=values
            if set(ids).intersection(pid for row in rows for pid in row["source_point_ids"]):
                raise ValueError("Native source query would be counted twice")
            row=dict(oldrow,uv=uv.copy(),sigma=np.linalg.norm(L,axis=1),source_point_ids=list(ids),point_correspondence=False,
                native_line_queries=dict(query_uv=queries.copy(),profile_normal=normals.copy(),query_ids=list(ids)),
                measurement_origin="original_reviewed_native_material_queries_without_line_estimator_compression")
            row.pop("point_support_bounds_xy_px",None)
            rows.append(row)
            indices=list(range(cursor,cursor+len(ids)));cursor+=len(ids)
            operator=_native_operator(ids,chart,L,gid)
            new_groups[gid]=dict(operator=operator,point_indices=indices,
                native_line_replacement=dict(revision=revision_reference,recipe=revision["recipe"],identity_review=revision["identity_review"],
                    item=deepcopy(item),localization_noise_factor=L.tolist(),old_operator=group["operator"].to_record()),
                source_evidence=dict(source_feature_id=item["feature_id"],original_native_readout_packet=item["native_readout_packet"],
                    source_noise_coordinate="original_saved_line_chart_normal_px",native_query_intersection_required=True))
        groups=[];blocks=[];oc=nc=0
        for group in original["whitening_groups"]:
            gid=group["operator"].group_id;updated=new_groups.get(gid)
            if updated is None:updated=dict(group,point_indices=[point_map[i]for i in group["point_indices"]])
            old_stop=oc+group["operator"].rank;new_stop=nc+updated["operator"].rank
            block=dict(group_id=gid,replaced=gid in new_groups,old_formal_local_slice=[oc,old_stop],new_formal_local_slice=[nc,new_stop])
            if gid in new_groups:
                block.update(old_representative_ids=list(group["operator"].source_point_ids),new_native_ids=list(updated["operator"].source_point_ids),
                    old_point_indices=list(group["point_indices"]),new_point_indices=list(updated["point_indices"]),new_operator=updated["operator"].to_record())
            blocks.append(block);groups.append(updated);oc=old_stop;nc=new_stop
        new_ordinary=2*(cursor-sum(len(g["point_indices"])for g in groups))
        if new_ordinary!=old_ordinary:raise ValueError("Native replacement changed ungrouped observations")
        blocks.append(dict(group_id=None,replaced=False,ordinary=True,old_formal_local_slice=[oc,oc+old_ordinary],new_formal_local_slice=[nc,nc+new_ordinary]))
        audit=dict(frame=original["index"],unchanged=False,old_global_formal_start=old_global,new_global_formal_start=new_global,
            old_formal_rows=old_total,new_formal_rows=nc+new_ordinary,old_raw_points=original["source_point_count"],new_raw_points=cursor,formal_blocks=blocks)
        frame=dict(original,rows=rows,whitening_groups=groups,source_point_count=cursor,native_line_replacement_audit=audit)
        result.append(frame);mappings.append(audit);old_global+=old_total;new_global+=nc+new_ordinary
    if used!=set(replacements):raise ValueError("A reviewed native group is absent from the current complete FIT")
    from .first_shot_native_line_kernel import PREDICTION_DOMAIN
    return result,dict(reference=revision_reference,replaced_groups=len(used),frames=mappings,
        old_total_formal_rows=old_global,new_total_formal_rows=new_global,original_L_and_native_UV_preserved=True,
        old_representatives_removed_atomically=True,physical_target_or_parameter_count_changed=False,
        prediction_domain=deepcopy(PREDICTION_DOMAIN),
        fixed_noise_basis="original native chart normal; exact saved profile-query curve intersections",
        robust_objective="same normalized per-formal-row Huber2; basis and row count change; cross-objective costs are not comparable")
