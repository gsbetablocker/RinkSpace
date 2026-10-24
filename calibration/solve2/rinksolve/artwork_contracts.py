"""Typed artwork metadata bound to geometry already present in a current model.

This module never replaces geometry, adds parameters, admits observations or
publishes a runtime model.  Source contours, installed content composition and
physical board installation are separate records.  A caller may construct and
persist a detached candidate, then apply its metadata to the latest compatible
physical state at an explicit stage boundary.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import hashlib
import json
import math

import numpy as np

from . import schema as S
from .model import ModelBundle, plain

VERSION = "rinksolve.artwork_content_contract.v1"


@dataclass(frozen=True)
class CurveSourceBinding:
    curve_id: str
    source_component_id: str
    source_feature_id: str
    canonical_curve_id: str
    finite_semantic_arc: str | None
    source_vertex_indices: tuple[int, ...]
    closed: bool
    is_alias: bool


@dataclass(frozen=True)
class FixedCornerBinding:
    world_landmark_id: str
    source_landmark_id: str
    curve_id: str
    source_component_id: str
    source_feature_id: str
    source_vertex_index: int
    source_station: float


@dataclass(frozen=True)
class ContentInstallationLink:
    installation_id: str
    content_layout_id: str
    runtime_group_id: str
    existing_shared_transform_block_id: str
    board_surface_id: str
    mapping_definition: str
    initial_source_to_unwrapped: tuple[float, float, float, float]
    source_pixel_dimensions: tuple[int, int]
    source_normalization: str
    external_banner_extent_status: str


@dataclass(frozen=True)
class ArtworkContentContract:
    artwork_id: str
    model_id: str
    model_revision: int
    base_parameter_sha256: str
    geometry_definition_sha256: str
    source_metadata_path: str
    source_metadata_sha256: str
    template: S.ArtworkTemplate
    coordinates: tuple[S.CoordinateSystem, ...]
    evidence: tuple[S.Evidence, ...]
    layout_groups: tuple[S.ArtworkLayoutGroup, ...]
    layout: S.ArtworkLayout
    curves: tuple[CurveSourceBinding, ...]
    fixed_corners: tuple[FixedCornerBinding, ...]
    installation: ContentInstallationLink


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def geometry_definition_digest(model):
    """Exclude parameter values and artwork metadata, so newer physical states rebase."""
    return _digest({
        "scene": {"id": model.scene.id,
                  "coordinate_system_id": model.scene.coordinate_system_id,
                  "board_block_id": model.scene.board_block_id,
                  "geometry_family": model.scene.geometry_family,
                  "structural_family": asdict(model.scene.structural_family)
                  if model.scene.structural_family else None,
                  "curves": [{"id": c.id, "label": c.label, "entity_id": c.entity_id,
                              "kind": c.kind, "parameter_block_id": c.parameter_block_id,
                              "definition": plain(c.definition),
                              "vertices_are_landmarks": c.vertices_are_landmarks,
                              "semantic_status": c.semantic_status}
                             for c in model.scene.curves]},
        "rigs": [asdict(x) for x in model.rigs],
        "shots": [asdict(x) for x in model.shots],
        "installations": [asdict(x) for x in model.installations],
        "landmarks": [S.to_json_value(x) for x in model.landmarks],
        "runtime_artwork_groups": [{"id": a.id, "groups": [asdict(g) for g in a.groups]}
                                   for a in model.artworks],
        "parameter_blocks": [S.to_json_value(x) for x in model.state.blocks],
    })


def _source_geometry(metadata):
    components, contours = {}, {}
    for component in metadata["components"]:
        if component["id"] in components:
            raise ValueError("duplicate source component identity")
        components[component["id"]] = component
        for contour in component["geometry"]:
            if contour["id"] in contours:
                raise ValueError("duplicate source contour identity")
            uv = np.asarray(contour["uv"], dtype=np.float64)
            if uv.ndim != 2 or uv.shape[1] != 2 or len(uv) < 2 or not np.isfinite(uv).all():
                raise ValueError("invalid source contour")
            contours[contour["id"]] = (component, contour)
    return components, contours


def _artwork(model, artwork_id):
    found = [a for a in model.artworks if a.id == artwork_id]
    if len(found) != 1 or len(found[0].groups) != 1:
        raise ValueError("one existing whole-content runtime group is required")
    return found[0], found[0].groups[0]


def _canonical(model, curve):
    by_id = {c.id: c for c in model.scene.curves}
    seen = set()
    while curve.kind == "curve_alias":
        if curve.id in seen:
            raise ValueError("cyclic curve alias")
        seen.add(curve.id)
        curve = by_id[curve.definition["target_curve_id"]]
    return curve


def build_content_contract(model, source_metadata_path, *, artwork_id, brand,
                           registration, model_metadata_path, extra_evidence=()):
    """Bind an already-used source template; never import a new shape or layout.

    registration is the existing immutable numeric source-to-unwrapped initializer,
    [station_origin, height_origin, log_uniform_scale, angle], not a new physical
    fact.  Its exact source-point correspondence is checked against every current
    HD branch.  Existing runtime board-unit transform blocks remain untouched.
    """
    path = Path(source_metadata_path).resolve()
    metadata = json.loads(path.read_text(encoding="utf-8-sig"))
    source_sha = _sha(path)
    template_id = metadata["id"]
    width, height = int(metadata["raster"]["width"]), int(metadata["raster"]["height"])
    if min(width, height) <= 0:
        raise ValueError("invalid original source dimensions")
    if metadata.get("kind") not in ("component_logo", "composite_logo_source",
                                     "candidate_typography_component", "component_event_logo"):
        raise ValueError("source extent requires an explicit component-content interpretation")
    art, runtime_group = _artwork(model, artwork_id)
    if runtime_group.source_template_id != template_id:
        raise ValueError("source template is not already bound to the current runtime group")
    if runtime_group.layout_block_id is None:
        raise ValueError("existing shared board installation transform required")
    components, contours = _source_geometry(metadata)
    registration = tuple(float(v) for v in registration)
    if len(registration) != 4 or not all(math.isfinite(v) for v in registration):
        raise ValueError("explicit finite existing source registration required")
    model_path = Path(model_metadata_path).resolve()
    if not model_path.is_file():
        raise ValueError("current model metadata evidence missing")
    source_eid = "official_isu_horizontal_source/sha256/" + source_sha if template_id == "isu_horizontal_official" else "artwork_source/sha256/" + source_sha
    source_coord = template_id + "/source_aspect_coordinates"
    content_coord = artwork_id + "/content_coordinates"
    layout_id = artwork_id + "/content_layout"
    content_group_id = artwork_id + "/content_full_lockup"
    norm_description = (
        "Source pixel coordinates (x,y) map to (x/W,y/W), W=" + str(width) +
        ". Origin top-left, x right, y down; equal pixel scale preserves source aspect. "
        "Asset metadata contour uv=(x/W,y/H) therefore maps through diag(1,H/W). "
        "This coordinate box is content, never a physical banner border or size.")
    coordinates = tuple(S.CoordinateSystem(
        cid, S.CoordinateKind.ARTWORK, 2, ("right", "down"),
        (S.UnitRef(S.Unit.NORMALIZED_2D, cid), S.UnitRef(S.Unit.NORMALIZED_2D, cid)),
        norm_description + (" Current content composition is source identity, without new variables."
                            if cid == content_coord else ""))
        for cid in (source_coord, content_coord))
    evidence = [S.Evidence(
        source_eid, S.EvidenceKind.OFFICIAL_ARTWORK, str(path),
        "Existing official extracted full-lockup contour catalog; source threshold "
        "boundaries retain segmentation uncertainty. Does not establish physical "
        "banner margins or per-source video association.", source_sha)]
    for i, vector in enumerate(metadata.get("source_vectors", [])):
        vp = Path(vector["path"])
        if not vp.is_file() or _sha(vp) != vector["sha256"]:
            raise ValueError("current source vector path/hash mismatch")
        evidence.append(S.Evidence(
            template_id + "/source_vector/" + str(i), S.EvidenceKind.OFFICIAL_ARTWORK,
            str(vp), "Official source vector asset, retained independently of extracted raster contours.",
            vector["sha256"]))
    raster_path = Path(metadata["raster"]["path"])
    if not raster_path.is_file():
        raise ValueError("current extracted source raster missing")
    evidence.append(S.Evidence(
        template_id + "/extracted_raster", S.EvidenceKind.OFFICIAL_ARTWORK, str(raster_path),
        "Actual extracted content raster; original pixel dimensions retained. "
        "Its canvas edges are not physical print panel edges.", _sha(raster_path)))
    registration_eid = artwork_id + "/existing_registration_candidate"
    evidence.append(S.Evidence(
        registration_eid, S.EvidenceKind.COMPUTATION,
        str(model_path) + "#provenance/ad7_hd_candidate/registration"
        if template_id == "isu_horizontal_official" else str(model_path),
        "Already-used numeric content-to-board registration candidate. The metadata "
        "binding performs no new fitting and does not claim whole-shot acceptance.",
        _sha(model_path), (source_eid,)))
    evidence.extend(extra_evidence)
    if len({e.id for e in evidence}) != len(evidence):
        raise ValueError("duplicate artwork evidence identity")
    evidence_ids = {e.id for e in evidence}
    shapes = []
    roles = {"outer": S.ContourRole.OUTER, "hole": S.ContourRole.INNER_HOLE,
             "open_stroke": S.ContourRole.OPEN_STROKE}
    for component in metadata["components"]:
        for contour in component["geometry"]:
            role = roles.get(contour["role"])
            if role is None:
                raise ValueError("unknown source contour topology")
            shapes.append(S.ArtworkShape(
                contour["id"], component["kind"], source_eid,
                contour["kind"] == "closed_material_boundary", role, component["id"],
                contour.get("parent_id"), None, component.get("text")))
    curves, corners, source_landmarks = [], [], []
    for curve in model.scene.curves:
        if curve.id not in runtime_group.shape_ids:
            continue
        direct = curve.definition
        if direct.get("source_template_id") != template_id:
            continue
        canonical = _canonical(model, curve)
        d = canonical.definition
        if d.get("source_template_json_sha256") != source_sha:
            raise ValueError("curve source template version/hash mismatch")
        component, contour = contours[d["source_feature_id"]]
        if d.get("semantic_glyph_id") != component["id"]:
            raise ValueError("curve source component identity mismatch")
        if curve.kind == "curve_alias" and (direct.get("source_feature_id") != d["source_feature_id"]
                                             or direct.get("semantic_glyph_id") != component["id"]):
            raise ValueError("alias names a different source contour")
        indices = tuple(int(i) for i in d["source_contour_vertex_indices"])
        if not indices or min(indices) < 0 or max(indices) >= len(contour["uv"]):
            raise ValueError("finite source vertex range invalid")
        curves.append(CurveSourceBinding(
            curve.id, component["id"], contour["id"], canonical.id,
            d.get("finite_semantic_arc"), indices, bool(d["closed"]), curve.kind == "curve_alias"))
        if curve.kind == "curve_alias":
            continue
        for world_id, record in d.get("fixed_artwork_landmarks", {}).items():
            if not record.get("not_arbitrary_contour_sample") or not record.get("source_corner_station_is_not_optimizer_variable"):
                raise ValueError("only existing semantically defined fixed corners may be catalogued")
            if record["source_feature_id"] != contour["id"] or record["source_component_id"] != component["id"]:
                raise ValueError("fixed corner source identity mismatch")
            landmark_id = template_id + "/fixed_corner/" + world_id
            if any(e not in evidence_ids for e in record["evidence_ids"]):
                raise ValueError("existing fixed corner evidence must be supplied, not invented")
            xy = np.asarray(record["source_pixel_xy"], dtype=np.float64)
            vertex = int(record["source_contour_vertex_index"])
            source_xy = np.asarray(contour["uv"][vertex], dtype=np.float64) * [width, height]
            if not np.allclose(xy, source_xy, rtol=0, atol=1e-8):
                raise ValueError("fixed corner is not the recorded source contour point")
            source_landmarks.append(S.ArtworkLandmark(
                landmark_id, (float(xy[0]/width), float(xy[1]/width)),
                record["definition"] + " Existing world identity: " + world_id +
                "; fixed shared position; not a new contour-sampling landmark.",
                tuple(record["evidence_ids"])))
            corners.append(FixedCornerBinding(
                world_id, landmark_id, curve.id, component["id"], contour["id"],
                vertex, float(record["station"])))
    if not curves:
        raise ValueError("no current geometry is already linked to this source template")
    source_features = tuple(s.id for s in shapes)
    if set(source_features) != set(runtime_group.source_feature_ids):
        raise ValueError("runtime source feature catalog differs from actual source contours")
    template = S.ArtworkTemplate(
        template_id, 1, brand, metadata.get("source", {}).get("variant", metadata["name"]),
        source_coord, S.TemplateExtent.COMPONENT_LOGO,
        (S.ArtworkSourceFrame(source_eid, (width, height), (0., 0., float(width), float(height)),
                             "Original content raster viewBox; preserve aspect using x/W,y/W. "
                             "Logo canvas and its margins do not define the physical banner."),),
        (source_eid,), tuple(shapes), tuple(source_landmarks),
        "Complete existing extracted content catalog: " + str(len(components)) +
        " named components, " + str(len(shapes)) + " contours. "
        "Installed whole-banner border/background/margins remain outside this source "
        "content template. Native raw associations and whole-shot fit require their "
        "own evidence; this metadata operation re-verifies neither.",
        (), None)
    layout_group = S.ArtworkLayoutGroup(
        content_group_id, template.id, source_features + tuple(x.id for x in source_landmarks),
        "source_coordinates_identity.v1", (), S.VerificationStatus.UNCONFIRMED,
        (registration_eid,),
        "Current content composition retains the already-used official full lockup. "
        "This identity content stage has no parameters. The existing runtime group "
        + runtime_group.id + " acts later in board-unwrapped coordinates; it is not "
        "a normalized-source layout parameter and is not duplicated here.")
    layout = S.ArtworkLayout(
        layout_id, 1, content_coord, (content_group_id,), (registration_eid,),
        "Physical banner border, margins and background extent are unconfirmed. "
        "Official logo raster/viewBox is only content; no canvas edge is a banner edge.",
        "One cross-exposure shared full-lockup content composition candidate. "
        "Source->content identity->existing numeric installation initializer->existing "
        "shared board transform->continuous 3D board. No new DOF or physical equality.")
    install = [i for i in model.installations if i.shared_layout_block_id == runtime_group.layout_block_id]
    if len(install) != 1:
        raise ValueError("one current content installation relation required")
    installation = ContentInstallationLink(
        install[0].id, layout.id, runtime_group.id, runtime_group.layout_block_id,
        install[0].board_surface_id,
        "Apply the recorded numeric source-to-unwrapped initializer, then the existing "
        "shared similarity about its existing content anchor, then continuous-board xyz. "
        "This metadata describes already stored curve coordinates; it is not a second evaluator.",
        registration, (width, height), norm_description,
        "No physical banner extent is inferred; InstallationModel.template_id and extent are untouched.")
    contract = ArtworkContentContract(
        art.id, model.id, model.revision, model.state.digest(), geometry_definition_digest(model),
        str(path), source_sha, template, coordinates, tuple(evidence), (layout_group,),
        layout, tuple(curves), tuple(corners), installation)
    validate_content_contract(contract, model=model)
    return contract


def validate_content_contract(contract, *, model=None):
    """Validate metadata references and, when supplied, exact current geometry links."""
    coordinate_ids = {c.id for c in contract.coordinates}
    evidence_ids = {e.id for e in contract.evidence}
    if len(coordinate_ids) != len(contract.coordinates) or len(evidence_ids) != len(contract.evidence):
        raise ValueError("duplicate coordinate/evidence identity")
    if contract.template.coordinate_system_id not in coordinate_ids or contract.layout.coordinate_system_id not in coordinate_ids:
        raise ValueError("source/content coordinate registry incomplete")
    for coordinate in contract.coordinates:
        if coordinate.kind is not S.CoordinateKind.ARTWORK or coordinate.dimension != 2 or len(coordinate.units) != 2 or any(u.unit is not S.Unit.NORMALIZED_2D or u.reference_id != coordinate.id for u in coordinate.units):
            raise ValueError("source/content coordinates require explicit normalized 2D units")
    for record in contract.evidence:
        if not set(record.parent_evidence_ids) <= evidence_ids:
            raise ValueError("source evidence dependency missing")
    if not contract.template.source_frames or not set(contract.template.source_evidence_ids) <= evidence_ids:
        raise ValueError("source evidence/viewBox incomplete")
    for frame in contract.template.source_frames:
        if frame.evidence_id not in evidence_ids or min(frame.view_box[2:]) <= 0:
            raise ValueError("source frame definition invalid")
    shapes = {s.id: s for s in contract.template.shapes}
    landmarks = {p.id: p for p in contract.template.landmarks}
    if len(shapes) != len(contract.template.shapes) or len(landmarks) != len(contract.template.landmarks):
        raise ValueError("duplicate source shape/landmark identity")
    if any(s.path_evidence_id not in evidence_ids for s in shapes.values()) or any(not set(p.evidence_ids) <= evidence_ids for p in landmarks.values()):
        raise ValueError("source feature evidence incomplete")
    features = set(shapes) | set(landmarks)
    if len(features) != len(shapes) + len(landmarks):
        raise ValueError("source feature identities overlap")
    if contract.template.extent is S.TemplateExtent.WHOLE_BANNER:
        raise ValueError("a component content contract cannot claim whole banner extent")
    if any(s.topology_role is S.ContourRole.INNER_HOLE and
           (s.parent_shape_id not in shapes or not s.closed or
            shapes[s.parent_shape_id].group_id != s.group_id)
           for s in shapes.values()):
        raise ValueError("source hole topology invalid")
    if len({b.curve_id for b in contract.curves}) != len(contract.curves):
        raise ValueError("duplicate current curve binding")
    if len({b.world_landmark_id for b in contract.fixed_corners}) != len(contract.fixed_corners):
        raise ValueError("duplicate fixed corner binding")
    if set(contract.layout.group_ids) != {g.id for g in contract.layout_groups}:
        raise ValueError("layout group references incomplete")
    for group in contract.layout_groups:
        if group.source_template_id != contract.template.id or not set(group.source_feature_ids) <= features:
            raise ValueError("layout source feature reference invalid")
        if group.transform_model_id != "source_coordinates_identity.v1" or group.parameter_block_ids:
            raise ValueError("metadata content binding cannot add a transform or parameter")
        if group.match_status is S.VerificationStatus.CONFIRMED:
            raise ValueError("metadata construction cannot promote native visual approval")
    error = 0.
    if model is not None:
        if model.id != contract.model_id or model.revision != contract.model_revision or geometry_definition_digest(model) != contract.geometry_definition_sha256:
            raise ValueError("current geometry definition changed; metadata must be reviewed again")
        art, group = _artwork(model, contract.artwork_id)
        if group.id != contract.installation.runtime_group_id or group.layout_block_id != contract.installation.existing_shared_transform_block_id:
            raise ValueError("runtime installation group changed")
        metadata = json.loads(Path(contract.source_metadata_path).read_text(encoding="utf-8-sig"))
        if _sha(contract.source_metadata_path) != contract.source_metadata_sha256:
            raise ValueError("source metadata version changed")
        _, contours = _source_geometry(metadata)
        curve_map = {c.id: c for c in model.scene.curves}
        w, h = contract.installation.source_pixel_dimensions
        s0, z0, log_scale, angle = contract.installation.initial_source_to_unwrapped
        c, s = math.cos(angle), math.sin(angle)
        rotation = np.array([[c,s],[-s,c]])
        for binding in contract.curves:
            current = curve_map[binding.curve_id]
            canonical = _canonical(model, current)
            if canonical.id != binding.canonical_curve_id:
                raise ValueError("current alias target changed")
            definition = canonical.definition
            if binding.is_alias != (current.kind == "curve_alias") or binding.closed != bool(definition["closed"]) or binding.finite_semantic_arc != definition.get("finite_semantic_arc") or binding.source_vertex_indices != tuple(definition["source_contour_vertex_indices"]):
                raise ValueError("finite source interval/topology changed")
            if binding.source_feature_id not in shapes or definition.get("source_feature_id") != binding.source_feature_id or definition.get("semantic_glyph_id") != binding.source_component_id:
                raise ValueError("current finite source feature changed")
            contour = contours[binding.source_feature_id][1]
            xy = np.asarray(contour["uv"], dtype=float)[list(binding.source_vertex_indices)] * [1., h/w]
            stored = np.asarray(definition["unwrapped_station_height"], dtype=float)
            if len(stored) == len(xy)+1 and binding.closed:
                xy = np.vstack((xy, xy[:1]))
            if stored.shape != xy.shape:
                raise ValueError("finite contour point count/closure mismatch")
            expected = np.exp(log_scale)*xy@rotation + [s0,z0]
            error = max(error, float(np.max(np.abs(expected-stored))))
            if error > 5e-14:
                raise ValueError("existing curve is not the declared source registration")
        world_landmarks = {x.id:x for x in model.landmarks}
        for binding in contract.fixed_corners:
            landmark = world_landmarks.get(binding.world_landmark_id)
            if landmark is None or not isinstance(landmark.position, S.StructuralPointPosition) or landmark.position.definition_id != "artwork_contour_corner.v1:"+landmark.id or landmark.position.geometry_feature_ids != (binding.curve_id,):
                raise ValueError("fixed world landmark identity/definition changed")
            if binding.source_landmark_id not in landmarks:
                raise ValueError("fixed source landmark missing")
            record = curve_map[binding.curve_id].definition.get("fixed_artwork_landmarks", {}).get(binding.world_landmark_id)
            if record is None or binding.source_feature_id != record["source_feature_id"] or binding.source_component_id != record["source_component_id"] or binding.source_vertex_index != record["source_contour_vertex_index"] or binding.source_station != float(record["station"]):
                raise ValueError("fixed source corner definition changed")
            expected_uv = np.asarray(record["source_pixel_xy"], dtype=float) / w
            if not np.array_equal(np.asarray(landmarks[binding.source_landmark_id].uv), expected_uv):
                raise ValueError("fixed source corner coordinates changed")
    return {"valid":True, "source_shapes":len(shapes), "fixed_source_landmarks":len(landmarks),
            "current_bound_curves":len(contract.curves),
            "source_to_stored_unwrapped_max_abs":error,
            "physical_parameters_added":0, "observations_added":0,
            "whole_shot_acceptance":False}


def apply_content_contract(model, contract):
    """Return metadata only on the incoming latest physical state; never publish it."""
    validate_content_contract(contract, model=model)
    artworks = []
    for artwork in model.artworks:
        if artwork.id != contract.artwork_id:
            artworks.append(artwork)
        else:
            artworks.append(replace(artwork, template=contract.template,
                                    source_templates=(contract.template,), layout=contract.layout))
    provenance = plain(model.provenance)
    registry = dict(provenance.get("artwork_content_contracts", {}))
    registry[contract.artwork_id] = contract_to_json(contract)
    provenance["artwork_content_contracts"] = registry
    return replace(model, artworks=tuple(artworks), provenance=provenance)


def contract_to_json(contract):
    return {"schema":VERSION, **{
        k:getattr(contract,k) for k in (
            "artwork_id","model_id","model_revision","base_parameter_sha256",
            "geometry_definition_sha256","source_metadata_path","source_metadata_sha256")},
        "template":S.to_json_value(contract.template),
        "coordinates":[S.to_json_value(x) for x in contract.coordinates],
        "evidence":[S.to_json_value(x) for x in contract.evidence],
        "layout_groups":[S.to_json_value(x) for x in contract.layout_groups],
        "layout":S.to_json_value(contract.layout),
        "curves":[asdict(x) for x in contract.curves],
        "fixed_corners":[asdict(x) for x in contract.fixed_corners],
        "installation":asdict(contract.installation)}


def contract_from_json(value):
    if value.get("schema") != VERSION:
        raise ValueError("unknown artwork content contract schema")
    fields = {k:value[k] for k in (
        "artwork_id","model_id","model_revision","base_parameter_sha256",
        "geometry_definition_sha256","source_metadata_path","source_metadata_sha256")}
    curves = tuple(CurveSourceBinding(**{**x,"source_vertex_indices":tuple(x["source_vertex_indices"])})
                   for x in value["curves"])
    installation = dict(value["installation"])
    installation["initial_source_to_unwrapped"] = tuple(installation["initial_source_to_unwrapped"])
    installation["source_pixel_dimensions"] = tuple(installation["source_pixel_dimensions"])
    result = ArtworkContentContract(
        **fields, template=S.from_json_value(value["template"],S.ArtworkTemplate),
        coordinates=tuple(S.from_json_value(x,S.CoordinateSystem) for x in value["coordinates"]),
        evidence=tuple(S.from_json_value(x,S.Evidence) for x in value["evidence"]),
        layout_groups=tuple(S.from_json_value(x,S.ArtworkLayoutGroup) for x in value["layout_groups"]),
        layout=S.from_json_value(value["layout"],S.ArtworkLayout),
        curves=curves, fixed_corners=tuple(FixedCornerBinding(**x) for x in value["fixed_corners"]),
        installation=ContentInstallationLink(**installation))
    validate_content_contract(result)
    return result


def merge_typed_registry(context, contract):
    """Merge into an existing current DataBundle for full schema reference checks."""
    def merge(existing, incoming):
        result = {x.id:x for x in existing}
        for item in incoming:
            prior = result.get(item.id)
            if prior is not None and prior != item:
                raise ValueError("typed registry identity conflict: "+item.id)
            result[item.id] = item
        return tuple(result.values())
    return replace(
        context, coordinate_systems=merge(context.coordinate_systems,contract.coordinates),
        evidence=merge(context.evidence,contract.evidence),
        artworks=merge(context.artworks,(contract.template,)),
        artwork_layout_groups=merge(context.artwork_layout_groups,contract.layout_groups),
        artwork_layouts=merge(context.artwork_layouts,(contract.layout,)))
