"""S2 offline plane-registration graph with original-edge provenance.

All G_i_to_j edges in this module are initialization/diagnostic relations.  Even
an independently estimated pairwise G is removed before the final all-source
physical objective.  A G synthesized from current H values cannot establish
connectivity, contribute to a graph solve, or validate a loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import heapq
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .registration import (
    LocalHomographyChart, RegistrationError, _deep_freeze, _immutable_array,
    _jsonable, _xy, normalize_homography, normalized_dlt, project_points,
)


@dataclass(frozen=True)
class PairwiseEdge:
    source_frame: int
    target_frame: int
    G_i_to_j: np.ndarray
    plane_id: str
    provenance: str
    source_witness_uv: np.ndarray
    original_evidence: bool = False
    evidence_kind: str = "independent_pairwise_registration"
    data_rank: int | None = None
    weight: float = 1.0
    source_record_keys: tuple[Any, ...] = ()
    target_record_keys: tuple[Any, ...] = ()
    target_entity_witness_uv: np.ndarray | None = None
    derived_from_H: bool = False
    edge_id: str = ""
    purpose: str = "initialization_and_consistency_only; absent from final physical objective"

    def __post_init__(self):
        if self.source_frame == self.target_frame:
            raise RegistrationError("A pairwise graph edge needs two different frames")
        if not self.provenance or not self.evidence_kind:
            raise RegistrationError("Every edge needs evidence kind and provenance")
        if not np.isfinite(self.weight) or self.weight <= 0:
            raise RegistrationError("Graph initialization weight must be positive and frozen")
        if self.data_rank is not None and not 0 <= self.data_rank <= 8:
            raise RegistrationError("Pairwise H data rank must lie in [0,8] or be unknown")
        matrix = normalize_homography(self.G_i_to_j)
        witness = _xy(self.source_witness_uv, "source_witness_uv")
        project_points(matrix, witness)
        if self.target_entity_witness_uv is not None:
            target = _xy(self.target_entity_witness_uv, "target_entity_witness_uv")
            if target.shape != witness.shape or self.evidence_kind != "confirmed_entity_correspondence":
                raise RegistrationError("Raw paired target witnesses require confirmed entity-correspondence evidence")
            object.__setattr__(self, "target_entity_witness_uv", _immutable_array(target))
        if self.derived_from_H and self.original_evidence:
            raise RegistrationError("An H-synthesized edge cannot claim original independent pairwise evidence")
        object.__setattr__(self, "source_frame", int(self.source_frame))
        object.__setattr__(self, "target_frame", int(self.target_frame))
        object.__setattr__(self, "plane_id", str(self.plane_id))
        object.__setattr__(self, "G_i_to_j", _immutable_array(matrix))
        object.__setattr__(self, "source_witness_uv", _immutable_array(witness))
        object.__setattr__(self, "source_record_keys", tuple(_deep_freeze(k) for k in self.source_record_keys))
        object.__setattr__(self, "target_record_keys", tuple(_deep_freeze(k) for k in self.target_record_keys))
        if not self.edge_id:
            h = hashlib.sha256(matrix.tobytes()+witness.tobytes()+self.provenance.encode()).hexdigest()[:16]
            object.__setattr__(self, "edge_id", f"G_{self.source_frame}_to_{self.target_frame}_{h}")

    @property
    def usable_original_edge(self) -> bool:
        return self.original_evidence and not self.derived_from_H

    @property
    def target_witness_uv(self) -> np.ndarray:
        if self.target_entity_witness_uv is not None:
            return self.target_entity_witness_uv
        return _immutable_array(project_points(self.G_i_to_j, self.source_witness_uv))

    @classmethod
    def from_entity_correspondences(cls, source_frame: int, target_frame: int,
                                    source_native_uv: Any, target_native_uv: Any, *,
                                    plane_id: str, entity_evidence: str | Sequence[str],
                                    provenance: str, source_record_keys: Sequence[Any] = (),
                                    target_record_keys: Sequence[Any] = (), **kw):
        seed = normalized_dlt(source_native_uv, target_native_uv, entity_evidence=entity_evidence,
                              direction="native_i_to_j")
        return cls(source_frame, target_frame, seed.matrix, plane_id, provenance, source_native_uv,
                   original_evidence=True, evidence_kind="confirmed_entity_correspondence", data_rank=8,
                   source_record_keys=tuple(source_record_keys), target_record_keys=tuple(target_record_keys),
                   target_entity_witness_uv=target_native_uv, **kw)

    def to_dict(self) -> dict:
        return {"edge_id": self.edge_id, "source_frame": self.source_frame, "target_frame": self.target_frame,
                "G_i_to_j": self.G_i_to_j.tolist(), "plane_id": self.plane_id, "provenance": self.provenance,
                "original_evidence": self.original_evidence, "derived_from_H": self.derived_from_H,
                "evidence_kind": self.evidence_kind, "data_rank": self.data_rank, "weight": self.weight,
                "source_witness_uv": self.source_witness_uv.tolist(),
                "target_entity_witness_uv": None if self.target_entity_witness_uv is None else self.target_entity_witness_uv.tolist(),
                "source_record_keys": _jsonable(self.source_record_keys), "target_record_keys": _jsonable(self.target_record_keys),
                "purpose": self.purpose}


@dataclass(frozen=True)
class GraphAnchor:
    frame_id: int
    H_atlas_to_native: np.ndarray
    atlas_id: str
    provenance: str

    def __post_init__(self):
        if not self.atlas_id or not self.provenance:
            raise RegistrationError("An anchor requires an explicit atlas identity and provenance")
        object.__setattr__(self, "frame_id", int(self.frame_id))
        object.__setattr__(self, "H_atlas_to_native", _immutable_array(normalize_homography(self.H_atlas_to_native)))


@dataclass(frozen=True)
class PropagatedCandidate:
    frame_id: int
    origin_anchor_frame: int
    H_atlas_to_native: np.ndarray
    atlas_id: str
    edge_path: tuple[tuple[str, bool], ...]
    path_cost: float
    provenance: tuple[str, ...]

    def __post_init__(self):
        object.__setattr__(self, "H_atlas_to_native", _immutable_array(normalize_homography(self.H_atlas_to_native)))
        object.__setattr__(self, "edge_path", tuple(tuple(x) for x in self.edge_path))
        object.__setattr__(self, "provenance", tuple(self.provenance))


@dataclass(frozen=True)
class GraphFrameState:
    frame_id: int
    component_id: int
    status: str
    H_atlas_to_native: np.ndarray | None
    candidates: tuple[PropagatedCandidate, ...]
    original_edge_ids: tuple[str, ...]

    def __post_init__(self):
        if self.H_atlas_to_native is not None:
            object.__setattr__(self, "H_atlas_to_native", _immutable_array(normalize_homography(self.H_atlas_to_native)))
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "original_edge_ids", tuple(self.original_edge_ids))

    def to_dict(self) -> dict:
        return {"frame_id": self.frame_id, "component_id": self.component_id, "status": self.status,
                "H_atlas_to_native": None if self.H_atlas_to_native is None else self.H_atlas_to_native.tolist(),
                "original_edge_ids": list(self.original_edge_ids),
                "candidates": [{"origin_anchor_frame": c.origin_anchor_frame, "atlas_id": c.atlas_id,
                                "H_atlas_to_native": c.H_atlas_to_native.tolist(), "edge_path": list(c.edge_path),
                                "path_cost": c.path_cost, "provenance": list(c.provenance)} for c in self.candidates]}


@dataclass(frozen=True)
class GraphComponent:
    component_id: int
    frame_ids: tuple[int, ...]
    gauge_frame: int | None
    gauge_mode: str
    atlas_id: str | None
    alignment_status: str
    original_edge_ids: tuple[str, ...]


@dataclass(frozen=True)
class GraphInitialization:
    plane_id: str
    frame_ids: tuple[int, ...]
    frame_states: tuple[GraphFrameState, ...]
    components: tuple[GraphComponent, ...]
    edges: tuple[PairwiseEdge, ...]
    anchors: tuple[GraphAnchor, ...]
    limitations: tuple[str, ...]
    scope: str = "complete plane-graph initialization ledger, not a physical whole-shot state"

    @property
    def frame_ledger(self) -> Mapping[int, GraphFrameState]:
        return MappingProxyType({s.frame_id: s for s in self.frame_states})

    def to_dict(self) -> dict:
        return {"plane_id": self.plane_id, "frame_ids": list(self.frame_ids), "scope": self.scope,
                "frame_ledger": [s.to_dict() for s in self.frame_states],
                "components": [{k: _jsonable(getattr(c, k)) for k in c.__dataclass_fields__} for c in self.components],
                "edges": [e.to_dict() for e in self.edges],
                "anchors": [{"frame_id": a.frame_id, "H_atlas_to_native": a.H_atlas_to_native.tolist(),
                             "atlas_id": a.atlas_id, "provenance": a.provenance} for a in self.anchors],
                "limitations": list(self.limitations)}


@dataclass(frozen=True)
class CycleDiagnostic:
    edge_steps: tuple[tuple[str, bool], ...]
    start_frame: int
    witness_native_uv: np.ndarray
    displacement_native_uv: np.ndarray
    max_displacement_native_px: float
    rms_displacement_native_px: float
    provenance: tuple[str, ...]
    interpretation: str = "original pairwise-edge consistency; shared pixels may be correlated; not independent ground truth"

    def __post_init__(self):
        object.__setattr__(self, "witness_native_uv", _immutable_array(self.witness_native_uv))
        object.__setattr__(self, "displacement_native_uv", _immutable_array(self.displacement_native_uv))


@dataclass(frozen=True)
class GraphRefinement:
    initialization: GraphInitialization
    frame_states: tuple[GraphFrameState, ...]
    status: str
    optimization_log: tuple[Mapping[str, Any], ...]
    initial_native_witness_rms_px: float | None
    final_native_witness_rms_px: float | None
    actual_residual_rank: int | None
    variable_count: int
    weak_directions: np.ndarray
    objective_settings: Mapping[str, Any]
    limitations: tuple[str, ...]
    scope: str = "joint graph initializer only; all G pseudo-factors removed before final physical solve"

    def __post_init__(self):
        object.__setattr__(self, "optimization_log", tuple(_deep_freeze(e) for e in self.optimization_log))
        object.__setattr__(self, "weak_directions", _immutable_array(self.weak_directions))
        object.__setattr__(self, "objective_settings", _deep_freeze(self.objective_settings))

    @property
    def frame_ledger(self) -> Mapping[int, GraphFrameState]:
        return MappingProxyType({s.frame_id: s for s in self.frame_states})

    def to_dict(self) -> dict:
        return {"scope": self.scope, "status": self.status, "plane_id": self.initialization.plane_id,
                "frame_ids": list(self.initialization.frame_ids), "frame_ledger": [s.to_dict() for s in self.frame_states],
                "initial_native_witness_rms_px": self.initial_native_witness_rms_px,
                "final_native_witness_rms_px": self.final_native_witness_rms_px,
                "actual_residual_rank": self.actual_residual_rank, "variable_count": self.variable_count,
                "weak_directions": self.weak_directions.tolist(), "objective_settings": _jsonable(self.objective_settings),
                "optimization_log": _jsonable(self.optimization_log), "limitations": list(self.limitations),
                "initialization": self.initialization.to_dict()}


class RegistrationGraph:
    """One genuine plane and the full frame inventory, including isolated frames."""

    def __init__(self, frame_ids: Sequence[int], plane_id: str, edges: Sequence[PairwiseEdge] = (), *,
                 frame_information_scores: Mapping[int, float] | None = None):
        ids = tuple(int(i) for i in frame_ids)
        if not ids or len(set(ids)) != len(ids):
            raise RegistrationError("Supply the complete unique frame inventory")
        self.frame_ids = ids
        self.plane_id = str(plane_id)
        self.edges = tuple(edges)
        if len({e.edge_id for e in edges}) != len(edges):
            raise RegistrationError("Graph edge IDs must be unique")
        known = set(ids)
        for edge in edges:
            if edge.source_frame not in known or edge.target_frame not in known or edge.plane_id != self.plane_id:
                raise RegistrationError("Edge lies outside this frame inventory or genuine plane")
        self._edge_by_id = {e.edge_id: e for e in self.edges}
        self._original = tuple(e for e in self.edges if e.usable_original_edge)
        self._adj = {i: [] for i in ids}
        for e in self._original:
            self._adj[e.source_frame].append((e.target_frame, e, True))
            self._adj[e.target_frame].append((e.source_frame, e, False))
        self._information = {int(k): float(v) for k, v in (frame_information_scores or {}).items()}
        if set(self._information)-known or not all(np.isfinite(v) for v in self._information.values()):
            raise RegistrationError("Invalid declared frame information score")

    def connected_components(self) -> tuple[tuple[int, ...], ...]:
        """Connectivity uses only actual independently estimated original edges."""
        seen, components = set(), []
        for start in self.frame_ids:
            if start in seen:
                continue
            stack, component = [start], []
            seen.add(start)
            while stack:
                i = stack.pop()
                component.append(i)
                for j, _, _ in self._adj[i]:
                    if j not in seen:
                        seen.add(j); stack.append(j)
            components.append(tuple(sorted(component)))
        return tuple(components)

    def _information_key(self, frame: int):
        points = []
        for _, edge, forward in self._adj[frame]:
            points.append(edge.source_witness_uv if forward else edge.target_witness_uv)
        if points:
            p = np.concatenate(points)
            centered = p-p.mean(0)
            covariance = centered.T @ centered/max(1, len(p))
            coverage = math.sqrt(max(0., float(np.linalg.det(covariance))))
            count = len(p)
        else:
            coverage, count = 0., 0
        return (self._information.get(frame, 0.), coverage, count, -frame)

    def initialize(self, anchors: Sequence[GraphAnchor] = (), *,
                   gauge_mode: str = "component_projective") -> GraphInitialization:
        """Propagate each anchor in both time directions and retain every candidate.

        In a relative atlas, one informative anchor per connected component is
        set exactly to I by a common right-coordinate change.  External metric
        templates instead use ``external_fixed_atlas`` and retain the supplied
        gauge anchor H.  Different unspecified atlases are never silently joined.
        """
        if gauge_mode not in ("component_projective", "external_fixed_atlas"):
            raise RegistrationError("Unknown atlas gauge mode")
        anchors = tuple(anchors)
        if not all(isinstance(a, GraphAnchor) for a in anchors):
            raise RegistrationError("Use GraphAnchor objects with explicit atlas identity/provenance")
        if len({a.frame_id for a in anchors}) != len(anchors) or any(a.frame_id not in self._adj for a in anchors):
            raise RegistrationError("Anchor IDs must be unique and belong to the whole frame inventory")
        by_frame = {a.frame_id: a for a in anchors}
        states, components, effective_anchors = {}, [], []
        limitations = ["Disconnected projective components remain unaligned; H=I is a coordinate gauge, not metric calibration",
                       "Bidirectional propagated H values are conditional initialization, not successful physical tracking",
                       "All graph G/flow/pose pseudo-observations are absent from the final all-source physical objective"]
        if any(not e.usable_original_edge for e in self.edges):
            limitations.append("Stored non-original or H-synthesized edges are excluded from connectivity, propagation, refinement, and loops")
        for component_id, frames in enumerate(self.connected_components()):
            local_anchors = [by_frame[i] for i in frames if i in by_frame]
            edge_ids = tuple(e.edge_id for e in self._original if e.source_frame in frames)
            if not local_anchors and edge_ids and gauge_mode == "component_projective":
                reference = max(frames, key=self._information_key)
                local_anchors = [GraphAnchor(reference, np.eye(3), f"{self.plane_id}:component:{component_id}",
                                             "formal projective coordinate gauge on original connected support; no metric claim")]
            if not local_anchors:
                components.append(GraphComponent(component_id, frames, None, gauge_mode, None,
                                                  "unaligned_and_uninitialized", edge_ids))
                for i in frames:
                    states[i] = GraphFrameState(i, component_id, "unresolved_no_anchor_or_original_support", None, (),
                                               tuple(e.edge_id for _, e, _ in self._adj[i]))
                continue
            if len({a.atlas_id for a in local_anchors}) != 1:
                raise RegistrationError("Anchors in one component need an explicitly common atlas before propagation")
            reference_anchor = max(local_anchors, key=lambda a: self._information_key(a.frame_id))
            reference = reference_anchor.frame_id
            gauge_change = np.linalg.inv(reference_anchor.H_atlas_to_native) if gauge_mode == "component_projective" else np.eye(3)
            effective = [GraphAnchor(a.frame_id, a.H_atlas_to_native @ gauge_change,
                                    a.atlas_id, a.provenance) for a in local_anchors]
            effective_anchors.extend(effective)
            candidates = {i: [] for i in frames}
            for anchor in effective:
                queue = [(0., anchor.frame_id)]
                distance = {anchor.frame_id: 0.}
                best_path = {anchor.frame_id: (anchor.H_atlas_to_native, (), (anchor.provenance,))}
                while queue:
                    cost, i = heapq.heappop(queue)
                    if cost != distance[i]:
                        continue
                    current_h, path, provenance = best_path[i]
                    for j, edge, forward in self._adj[i]:
                        proposed_cost = cost + 1/edge.weight
                        if proposed_cost >= distance.get(j, np.inf):
                            continue
                        g = edge.G_i_to_j if forward else np.linalg.inv(edge.G_i_to_j)
                        proposed_h = normalize_homography(g @ current_h)
                        distance[j] = proposed_cost
                        best_path[j] = (proposed_h, path+((edge.edge_id, forward),), provenance+(edge.provenance,))
                        heapq.heappush(queue, (proposed_cost, j))
                for i, (matrix, path, provenance) in best_path.items():
                    candidates[i].append(PropagatedCandidate(i, anchor.frame_id, matrix, anchor.atlas_id,
                                                             path, distance[i], provenance))
            components.append(GraphComponent(component_id, frames, reference, gauge_mode, reference_anchor.atlas_id,
                                              "component_projective_unaligned" if gauge_mode == "component_projective" else "external_atlas_alignment_declared_by_caller",
                                              edge_ids))
            for i in frames:
                local = tuple(sorted(candidates[i], key=lambda c: (c.path_cost, c.origin_anchor_frame)))
                if not local:
                    status, matrix = "unresolved_no_valid_original_path", None
                else:
                    matrix = local[0].H_atlas_to_native
                    status = "anchor_candidate" if any(a.frame_id == i for a in effective) else "conditional_propagated_candidate"
                    if any(e.data_rank is None or e.data_rank < 8 for _, e, _ in self._adj[i]):
                        status += "_rank_weak_or_unverified"
                states[i] = GraphFrameState(i, component_id, status, matrix, local,
                                           tuple(e.edge_id for _, e, _ in self._adj[i]))
        return GraphInitialization(self.plane_id, self.frame_ids, tuple(states[i] for i in self.frame_ids),
                                   tuple(components), self.edges, tuple(effective_anchors), tuple(limitations))

    def check_cycle(self, edge_steps: Sequence[tuple[str, bool]], *, witness_native_uv: Any = None) -> CycleDiagnostic:
        """Evaluate actual original-edge composition as native point displacement."""
        steps = tuple((str(edge_id), bool(forward)) for edge_id, forward in edge_steps)
        if len(steps) < 3 or len({edge_id for edge_id, _ in steps}) != len(steps):
            raise RegistrationError("A nontrivial cycle needs at least three distinct original edges")
        matrices, provenance = [], []
        start = current = None
        first_edge = None
        for edge_id, forward in steps:
            if edge_id not in self._edge_by_id:
                raise RegistrationError("Cycle references an unknown edge")
            edge = self._edge_by_id[edge_id]
            if not edge.usable_original_edge:
                raise RegistrationError("A loop cannot use synthetic/current-H or non-original edges")
            source, target = (edge.source_frame, edge.target_frame) if forward else (edge.target_frame, edge.source_frame)
            if current is None:
                start, current, first_edge = source, source, edge
            if source != current:
                raise RegistrationError("Cycle edge directions do not form a connected walk")
            matrices.append(edge.G_i_to_j if forward else np.linalg.inv(edge.G_i_to_j))
            provenance.append(edge.provenance)
            current = target
        if current != start:
            raise RegistrationError("Cycle is not closed")
        if witness_native_uv is None:
            witness_native_uv = first_edge.source_witness_uv if steps[0][1] else first_edge.target_witness_uv
        witness = _xy(witness_native_uv, "cycle witness")
        transformed = witness.copy()
        for matrix in matrices:
            transformed = project_points(matrix, transformed)
        displacement = transformed-witness
        d2 = (displacement*displacement).sum(1)
        return CycleDiagnostic(steps, start, witness, displacement, float(np.sqrt(d2.max())),
                               float(np.sqrt(d2.mean())), tuple(provenance))

    def fundamental_cycles(self, max_cycles: int = 128) -> tuple[CycleDiagnostic, ...]:
        """A finite deterministic cycle basis from original edges, never synthesized H."""
        if max_cycles < 1:
            return ()
        tree = {i: [] for i in self.frame_ids}
        parent = {i: i for i in self.frame_ids}
        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        cycles = []
        for edge in self._original:
            i, j = edge.source_frame, edge.target_frame
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj
                tree[i].append((j, edge.edge_id, True))
                tree[j].append((i, edge.edge_id, False))
                continue
            # Traverse the new edge i->j and the real spanning-tree path j->i.
            queue, paths = [j], {j: ()}
            for node in queue:
                if node == i:
                    break
                for neighbor, edge_id, forward in tree[node]:
                    if neighbor not in paths:
                        paths[neighbor] = paths[node]+((edge_id, forward),)
                        queue.append(neighbor)
            steps = ((edge.edge_id, True),)+paths[i]
            if len(steps) >= 3:
                cycles.append(self.check_cycle(steps))
            if len(cycles) >= max_cycles:
                break
        return tuple(cycles)

    def refine(self, initialization: GraphInitialization, *, max_iterations: int = 20,
               engineering_scale_native_px: float = 2., max_parameter_step: float = .15,
               scaled_gradient_tolerance: float = 1e-6, rank_relative_threshold: float = 1e-5) -> GraphRefinement:
        """Joint sparse damped LM over all initialized H, with component gauges fixed.

        Residuals compare native witness positions.  Paired raw entity targets
        are used only when their physical identity is explicitly confirmed;
        otherwise independently estimated G supplies an initialization target.
        These graph targets never enter the final physical source objective.
        """
        if initialization.frame_ids != self.frame_ids or initialization.plane_id != self.plane_id:
            raise RegistrationError("Initialization belongs to a different complete frame inventory or plane")
        if tuple(e.edge_id for e in initialization.edges) != tuple(e.edge_id for e in self.edges):
            raise RegistrationError("Edge evidence changed: rebuild graph initialization first")
        if max_iterations < 0 or engineering_scale_native_px <= 0 or max_parameter_step <= 0:
            raise RegistrationError("Invalid graph refinement limits or scales")
        ledger = initialization.frame_ledger
        matrices0 = {i: s.H_atlas_to_native for i, s in ledger.items() if s.H_atlas_to_native is not None}
        fixed = {c.gauge_frame for c in initialization.components if c.gauge_frame is not None}
        active_edges = tuple(e for e in self._original if e.source_frame in matrices0 and e.target_frame in matrices0)
        active_nodes = {i for e in active_edges for i in (e.source_frame, e.target_frame)}
        variable_nodes = tuple(i for i in self.frame_ids if i in active_nodes and i not in fixed)
        settings = {"objective_id": "native_witness_pairwise_initializer_v1", "gauge_frames": sorted(fixed),
                    "engineering_scale_native_px": engineering_scale_native_px, "max_iterations": max_iterations,
                    "max_parameter_step": max_parameter_step, "scaled_gradient_tolerance": scaled_gradient_tolerance,
                    "edge_normalization": "positive frozen weight/N; engineering initialization only, not measured independent statistics",
                    "rank_relative_threshold": rank_relative_threshold,
                    "derived_from_current_H_edges_used": False, "present_in_final_physical_objective": False}
        limitations = list(initialization.limitations)
        if not active_edges or not variable_nodes:
            limitations.append("No jointly refinable original-edge component; the entire frame ledger remains present")
            return GraphRefinement(initialization, initialization.frame_states, "no_supported_joint_variables", (),
                                   None, None, None, 0, np.empty((0, 0)), settings, tuple(limitations))
        charts = {}
        active_edge_ids = {e.edge_id for e in active_edges}
        for i in variable_nodes:
            native_support = np.concatenate([e.source_witness_uv if forward else e.target_witness_uv
                                             for _, e, forward in self._adj[i] if e.edge_id in active_edge_ids])
            atlas_support = project_points(np.linalg.inv(matrices0[i]), native_support)
            charts[i] = LocalHomographyChart.from_homography(matrices0[i], atlas_support)
        node_column = {i: 8*k for k, i in enumerate(variable_nodes)}
        x = np.zeros(8*len(variable_nodes))
        row_starts = np.r_[0, np.cumsum([2*len(e.source_witness_uv) for e in active_edges])]
        targets = tuple(e.target_witness_uv for e in active_edges)
        factor_scales = tuple(math.sqrt(e.weight/len(e.source_witness_uv))/engineering_scale_native_px for e in active_edges)
        incidence = {i: [k for k, e in enumerate(active_edges) if i in (e.source_frame, e.target_frame)] for i in variable_nodes}

        def matrices(parameters):
            result = dict(matrices0)
            for i in variable_nodes:
                col = node_column[i]
                result[i] = charts[i].matrix(parameters[col:col+8])
            return result

        def edge_residual(k, current, scaled=True):
            edge = active_edges[k]
            # Evaluate inverse and forward denominators on used native witnesses.
            atlas = project_points(np.linalg.inv(current[edge.source_frame]), edge.source_witness_uv)
            predicted = project_points(current[edge.target_frame], atlas)
            residual = (predicted-targets[k]).ravel()
            return residual*factor_scales[k] if scaled else residual

        def residual(parameters):
            current = matrices(parameters)
            return np.concatenate([edge_residual(k, current) for k in range(len(active_edges))])

        def jacobian(parameters, current):
            rows, columns, values = [], [], []
            for i in variable_nodes:
                start = node_column[i]
                local = parameters[start:start+8]
                for dim in range(8):
                    step = 2e-6*max(1., abs(float(local[dim])))
                    for _ in range(14):
                        plus, minus = local.copy(), local.copy()
                        plus[dim] += step; minus[dim] -= step
                        try:
                            pmat, mmat = dict(current), dict(current)
                            pmat[i], mmat[i] = charts[i].matrix(plus), charts[i].matrix(minus)
                            blocks = [(k, (edge_residual(k, pmat)-edge_residual(k, mmat))/(2*step)) for k in incidence[i]]
                            break
                        except (RegistrationError, np.linalg.LinAlgError):
                            step *= .5
                    else:
                        raise RegistrationError("Graph finite-difference domain could not be restored")
                    for k, block in blocks:
                        row = np.arange(row_starts[k], row_starts[k+1])
                        rows.extend(row.tolist()); columns.extend([start+dim]*len(row)); values.extend(block.tolist())
            return sparse.coo_matrix((values, (rows, columns)), shape=(row_starts[-1], len(parameters))).tocsr()

        def raw_rms(parameters):
            current = matrices(parameters)
            raw = np.concatenate([edge_residual(k, current, False).reshape(-1, 2) for k in range(len(active_edges))])
            return float(np.sqrt(np.mean(np.sum(raw*raw, axis=1))))

        initial_rms = raw_rms(x)
        r = residual(x)
        cost, damping, status, history = float(.5*r @ r), 1e-3, "iteration_limit", []
        for iteration in range(max_iterations):
            try:
                jac = jacobian(x, matrices(x))
            except RegistrationError as exc:
                status = "derivative_domain_limited"
                history.append({"iteration": iteration, "status": status, "reason": str(exc)})
                break
            normal, gradient = (jac.T @ jac).tocsc(), np.asarray(jac.T @ r).ravel()
            diagonal = np.maximum(normal.diagonal(), 1e-12)
            scaled_gradient = float(np.max(np.abs(gradient)/np.sqrt(diagonal)))
            if scaled_gradient <= scaled_gradient_tolerance:
                status = "scaled_gradient_converged"
                break
            accepted = False
            for trial in range(16):
                try:
                    step = np.asarray(spsolve(normal+sparse.diags(damping*diagonal, format="csc"), -gradient)).ravel()
                    if not np.isfinite(step).all():
                        raise RegistrationError("Nonfinite graph LM step")
                    largest = float(np.max(np.linalg.norm(step.reshape(-1, 8), axis=1)))
                    if largest > max_parameter_step:
                        step *= max_parameter_step/largest
                    prediction = float(-gradient @ step - .5*step @ (normal @ step))
                    if prediction <= 0:
                        damping *= 5
                        continue
                    trial_r = residual(x+step)
                    trial_cost = float(.5*trial_r @ trial_r)
                except (RegistrationError, np.linalg.LinAlgError, FloatingPointError) as exc:
                    history.append({"iteration": iteration, "trial": trial, "accepted": False,
                                    "reason": "true-domain/factorization: "+str(exc)})
                    damping *= 5
                    continue
                decrease = cost-trial_cost
                ratio = decrease/prediction
                if decrease > 0 and ratio > 1e-4:
                    previous = cost
                    x, r, cost = x+step, trial_r, trial_cost
                    damping = max(1e-12, damping*(.35 if ratio > .75 else 1. if ratio > .25 else 2.))
                    history.append({"iteration": iteration, "accepted": True, "cost": cost,
                                    "native_witness_rms_px": raw_rms(x), "actual_over_predicted": ratio})
                    accepted = True
                    if decrease <= 1e-10*max(1., previous):
                        status = "relative_cost_converged"
                    break
                damping *= 5
            if not accepted:
                status = "trial_or_domain_limited"
                break
            if status == "relative_cost_converged":
                break
        final_matrices = matrices(x)
        final_rank, weak = None, np.empty((0, len(x)))
        # Small graphs get the complete normalized data rank.  Do not turn an
        # expensive incomplete sparse spectrum into a full-rank certificate.
        if len(x) <= 128:
            final_jac = jacobian(x, final_matrices).toarray()
            _, sv, vh = np.linalg.svd(final_jac, full_matrices=final_jac.shape[0] < final_jac.shape[1])
            final_rank = int(np.count_nonzero(sv > (sv[0]*rank_relative_threshold if len(sv) and sv[0] else 0.)))
            weak = vh[final_rank:]
            settings["normalized_data_singular_values"] = sv.tolist()
            if final_rank < len(x):
                status = "underconstrained"
                limitations.append(f"Joint graph actual residual rank is {final_rank}/{len(x)}; weak directions retained")
        else:
            limitations.append("Whole sparse-graph data rank was not fully computed; per-edge/local rank and conditioning remain explicit obligations")
        final_states = []
        for original in initialization.frame_states:
            if original.frame_id in active_nodes:
                refined_status = "joint_graph_gauge_fixed" if original.frame_id in fixed else "joint_graph_conditional_candidate"
                if final_rank is None or final_rank < len(x):
                    refined_status += "_rank_unverified_or_weak"
                final_states.append(replace(original, H_atlas_to_native=final_matrices[original.frame_id], status=refined_status))
            else:
                final_states.append(original)
        return GraphRefinement(initialization, tuple(final_states), status, tuple(history), initial_rms,
                               raw_rms(x), final_rank, len(x), weak, settings, tuple(limitations))
