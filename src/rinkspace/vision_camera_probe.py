"""Read-only camera probes using the actual small native source operators.

The authenticated checkpoint and camera export provide only exposure bindings.
No motion graph, source actors, profiles, normal equations or video are built.
Native factor constructors still validate their original packets and geometry;
their residual methods receive an explicit camera on the frozen complete world.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
import inspect
import re
import sys
from types import SimpleNamespace

import numpy as np

from .vision_camera_initialization import current_complete_source_geometry


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class FrozenCameraSourceProbe:
    """Expose original FIT/HELD rows without promoting or deriving source UV."""

    def __init__(self, state_path, camera_path, auxiliary_path, source_options):
        from calibration.solver_runtime import install
        install()
        from structured_programme_source_candidate.paired_programme_chart import (
            CompleteProgrammeParameterLayout, PairedProgrammeCoordinateChart,
        )
        from structured_programme_source_candidate.native_ice_curve_source import NativeIceCurveFactor
        from structured_programme_source_candidate.native_board_edge_source import NativeBoardEdgeFactor
        from structured_programme_source_candidate.original_review_source import (
            PACKET, PACKET_SHA256, load_original_absolute_evaluators,
        )
        from z190_physical_camera import SharedCalibration, ViewState

        self.state_path, self.camera_path = Path(state_path), Path(camera_path)
        self.binder, self.world, world_provenance = current_complete_source_geometry(state_path, auxiliary_path)
        fields = ("q", "aux__canonical_pts", "aux__block_start", "aux__block_lengths",
                  "aux__block_names", "aux__installation_xyz_slots",
                  "aux__all_installation_optics_installation_order",
                  "aux__current_shared_mount_parameter_indices",
                  "aux__current_all_installation_optics_parameter_indices",
                  "aux__current_shared_world_geometry_parameter_indices",
                  "aux__current_complete_parameter_layout_json")
        with np.load(state_path, allow_pickle=False) as archive:
            state = {key: archive[key].copy() for key in fields}
        with np.load(camera_path, allow_pickle=False) as archive:
            self.camera = {key: archive[key].copy() for key in (
                "pts", "camera25", "exposure_representative_pts", "installation_id",
                "view_state4", "effective_pp_offset_px")}
            self.metadata = json.loads(str(archive["metadata_json"].item()))

        declaration = json.loads(str(state["aux__current_complete_parameter_layout_json"].item()))
        layout = CompleteProgrammeParameterLayout(declaration["original_complete_parameter_count"],
                                                 declaration["additional_world_parameter_count"])
        if layout.manifest() != declaration:
            raise ValueError("Saved complete parameter permutation differs from its original implementation")
        chart = PairedProgrammeCoordinateChart(len(self.world) - 66)
        self.q = layout.evaluation_q(state["q"])
        if not np.array_equal(self.q[np.asarray(chart.world_indices)], self.world):
            raise ValueError("Frozen complete world differs across canonical/evaluation charts")
        pts = self.camera["pts"]
        representatives = self.camera["exposure_representative_pts"]
        aliases = {int(p): int(a) for p, a in zip(pts, representatives)}
        actual_index = {}
        for i, actual in enumerate(representatives):
            actual_index.setdefault(int(actual), i)
        if len(aliases) != len(pts) or set(actual_index) != set(state["aux__canonical_pts"].tolist()):
            raise ValueError("The complete explicit native exposure binding is required")
        self.actual_index, self.aliases = actual_index, aliases
        self.calibrations = {key: SharedCalibration.from_manifest(value)
                             for key, value in self.metadata["calibrations"].items()}
        installation_order = state["aux__all_installation_optics_installation_order"].tolist()
        installation_count = len(installation_order)
        if (installation_count == 0 or len(set(installation_order)) != installation_count
                or state["aux__installation_xyz_slots"].shape != (installation_count, 3)):
            raise ValueError("Explicit unique installation order and one XYZ triplet per installation required")
        optics_slots = state["aux__current_all_installation_optics_parameter_indices"].reshape(installation_count, 3)
        mount_slots = state["aux__current_shared_mount_parameter_indices"].reshape(installation_count, 3)
        bypts = {}
        cursor = 0
        for name, start, length in zip(state["aux__block_names"], state["aux__block_start"],
                                      state["aux__block_lengths"]):
            if int(length) % 4:
                raise ValueError("Original four-scalar view blocks required")
            stamps = state["aux__canonical_pts"][cursor:cursor + int(length) // 4]
            identities = {str(self.camera["installation_id"][actual_index[int(p)]]) for p in stamps}
            if len(identities) != 1:
                raise ValueError("A programme block cannot merge physical installations")
            identity = identities.pop()
            installation = installation_order.index(identity)
            block = SimpleNamespace(name=str(name), start=int(start),
                xyz_slots=state["aux__installation_xyz_slots"][installation].copy(),
                optics_chart=SimpleNamespace(
                    indices=optics_slots[installation].copy(),
                    mount_indices=mount_slots[installation].copy()),
                bundle=SimpleNamespace(index={int(p): i for i, p in enumerate(stamps)}))
            for i, actual in enumerate(stamps):
                actual = int(actual)
                source_camera = self.camera["camera25"][actual_index[actual]]
                source_view = self.camera["view_state4"][actual_index[actual]]
                slot = int(start) + 4 * i
                if (not np.array_equal(self.q[slot:slot + 3], source_view[:3])
                        or self.q[slot + 3] != float(np.mean(source_camera[6:8]))):
                    raise ValueError("Actual RF4 slots do not reproduce the frozen camera")
                rebuilt = self.calibrations[identity].camera25(ViewState.from_state4(source_view,
                    effective_pp_offset_px=self.camera["effective_pp_offset_px"][actual_index[actual]]))
                if not np.array_equal(rebuilt, source_camera):
                    raise ValueError("Exported physical calibration does not reproduce the original camera")
                bypts[actual] = block
            cursor += len(stamps)
        if cursor != len(state["aux__canonical_pts"]):
            raise ValueError("A complete checkpoint exposure block was omitted")

        binding = SimpleNamespace(geometry=self.binder, geometry_parameter_indices=np.asarray(chart.world_indices),
            structured_programme_coordinate_chart=chart, original=SimpleNamespace(aliases=aliases), bypts=bypts,
            current_user_physical_course_scope_source=source_options.get("--physical-course-scope-source"))
        self.native = {
            "native_ice": NativeIceCurveFactor(binding, source_options["--additional-ice-source"], len(self.q)),
            "native_board": NativeBoardEdgeFactor(binding, source_options["--additional-board-source"], len(self.q)),
        }
        if _sha(PACKET) != PACKET_SHA256:
            raise ValueError("The original source material packet changed")
        self.original_rows = pickle.loads(PACKET.read_bytes())["programme_absolute_rows"]
        self.original_calls, original_authorities = load_original_absolute_evaluators()
        self.provenance = dict(state_path=str(self.state_path), state_file_sha256=_sha(state_path),
            q_sha256=hashlib.sha256(state["q"].tobytes()).hexdigest(), camera_path=str(self.camera_path),
            camera_file_sha256=_sha(camera_path), world=world_provenance,
            native_factor_contracts={key: factor.contract for key, factor in self.native.items()},
            native_value_operator_authorities={key: dict(
                module=type(factor).__module__,
                path=str(Path(sys.modules[type(factor).__module__].__file__).resolve()),
                sha256=_sha(sys.modules[type(factor).__module__].__file__),
                residual_signature=str(inspect.signature(factor._residual)),
                frozen_world_signature=str(inspect.signature(factor._world)))
                for key, factor in self.native.items()},
            original_source_packet=dict(path=str(PACKET), sha256=PACKET_SHA256),
            original_absolute_operator_authorities=original_authorities,
            complete_source_graph_or_profiles_loaded=False, existing_source_SPLIT_preserved=True,
            camera_bindings_reconstructed_from_exact_checkpoint_slots=True,
            thin_camera_bindings_used_only_with_original_value_operator_camera_override=True,
            thin_bindings_not_a_production_factor_or_source_Jacobian_interface=True,
            all_exposure_camera_manifest_values_byte_exact=True, accuracy_accepted=False)

    def rows_at(self, actual_pts):
        """Return exact existing rows, with per-factor quadratic/Huber2 policy."""
        from structured_programme_source_candidate.original_review_source import observation_signature
        actual_pts = int(actual_pts)
        if actual_pts not in self.actual_index:
            raise ValueError("The source time must be an authenticated real exposure")
        result = []
        for family, factor in self.native.items():
            for collection_role, rows in (("FIT", factor.rows), ("HELD", factor.held)):
                for row in rows:
                    if row["actual_pts"] == actual_pts:
                        result.append(dict(family=family, collection_role=collection_role,
                            original_split=row["split"], quadratic=True, row=row, factor=factor,
                            source_instance_key=[family, row["id"], actual_pts]))
        for item in self.original_rows:
            if int(item["group_actual_pts"]) != actual_pts:
                continue
            original = item["source_row"]
            signature = observation_signature(original)
            self.binder.register_row(original, actual_pts=actual_pts,
                allow_previously_qualified_held_ad2_representation=item["collection_role"] == "HELD")
            rebound = self.binder.rebind(original, self.world)
            if observation_signature(rebound) != signature:
                raise ValueError("Current material binding changed an original observation")
            binding = self.binder._binding(original)[1]
            result.append(dict(family="original_absolute", collection_role=item["collection_role"],
                original_split=original.get("split"), quadratic=False, row=rebound,
                qualified_material_binding=dict(kind=binding["kind"], ids=binding.get("ids"),
                    name=binding.get("name"), physical_identity=binding["contract"].get("physical_identity")),
                original_call=self.original_calls[item["block"]],
                source_instance_key=[item["block"], item["group_order"], item["row_order"], actual_pts, original["id"]]))
        return result

    def residual(self, item, camera25):
        """Evaluate the unchanged original factor at one explicit camera."""
        camera25 = np.asarray(camera25, dtype=float)
        row = item["row"]
        if item["family"] == "original_absolute":
            return item["original_call"](camera25, [row], False)
        return item["factor"]._residual(self.q, [row], cameras={int(row["actual_pts"]): camera25})

    @staticmethod
    def primitive_key(item):
        """Return one reliable carrier identity, or None when unresolved/multiple.

        Row-ID text never proves different material. The original material
        owner's qualified IDs take precedence over explicit source metadata.
        Unknown carriers must be excluded from HELD independence claims.
        """
        row = item["row"]
        qualified = item.get("qualified_material_binding", {})
        numbers = None
        if qualified.get("kind") == "ad":
            ids = qualified.get("ids")
            if isinstance(ids, (list, tuple)) and ids:
                numbers = [identity[0] for identity in ids
                           if isinstance(identity, (list, tuple)) and len(identity) == 2]
                if len(numbers) != len(ids):
                    return None
        if numbers is None:
            binding = row.get("world_scene_binding", {})
            ids = binding.get("physical_ids")
            if binding.get("kind") == "actual_advertisement_corners" and isinstance(ids, (list, tuple)) and ids:
                matches = [re.fullmatch(r"ad([1-9][0-9]*):(?:left_top|right_top|left_bottom|right_bottom)", value)
                           if isinstance(value, str) else None for value in ids]
                if not all(matches):
                    return None
                numbers = [int(match.group(1)) for match in matches]
        if numbers is not None:
            if (any(isinstance(number, (bool, np.bool_)) or not isinstance(number, (int, np.integer))
                    or not 1 <= number <= 26 for number in numbers) or len(set(numbers)) != 1):
                return None
            return "print:Ad" + str(numbers[0])
        physical = row.get("physical_identity")
        if isinstance(physical, dict) and type(physical.get("ad_id")) is int and 1 <= physical["ad_id"] <= 26:
            return "print:Ad" + str(physical["ad_id"])
        if qualified.get("kind") in ("circle", "line", "carrier", "lane", "crease") and qualified.get("name"):
            return "ice:" + str(qualified["name"])
        if qualified.get("kind") == "course":
            course_identity = qualified.get("physical_identity")
            if course_identity in ("full_board_top", "original_named_board_top",
                                   "original_reviewed_lower_long_side_top", "qualified_lower_long_side_top"):
                return "board:board_top"
            if course_identity in ("full_ice_board_contact", "original_named_ice_board_contact"):
                return "board:ice_board_contact"
        curve = row.get("curve", {})
        kind = curve.get("kind")
        if kind == "house_ring":
            return f"ice:{curve['side']}{curve['row']}"
        if kind in ("current_ice_segment", "hockey_circle", "current_goal_crease"):
            return "ice:" + str(curve["feature_id"])
        if kind == "curling_centreline":
            return "ice:curling_axis_row" + str(curve["row"])
        if kind == "advertisement_boundary":
            return "print:Ad" + str(curve["advertisement_number"])
        if kind in ("board_top", "ice_board_contact"):
            return "board:" + kind
        binding = row.get("world_scene_binding", {})
        if binding.get("kind") == "finite_full_board_top_perimeter":
            return "board:board_top"
        if binding.get("kind") == "finite_full_ice_board_contact_perimeter":
            return "board:ice_board_contact"
        return None
