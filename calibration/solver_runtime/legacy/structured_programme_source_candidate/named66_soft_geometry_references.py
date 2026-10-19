"""Append the inherited programme reference18 on the current named66 world.

Only the existing engineering relation operator changes to the current axes.
The checkpoint row identities, nominal values, working widths and ordering
are retained. They are assumptions, never additional image observations.
"""
from __future__ import annotations
from calibration.solver_runtime.config import data_root as _runtime_data_root, data_module_dir as _runtime_data_module_dir

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.sparse import csr_matrix

from z190_streaming_source_normal import SourceRowBlock
from shared_nominal_geometry_reference_candidate.soft_nominal_geometry_source import (
    VERSION, EngineeringReferenceConfig, GeometryReferenceEvaluation,
    CompleteSourceWithSoftGeometryReferences, _readonly,
)
from simple_wide_perspective_reset_candidate.structured_geometry_reference import (
    ACTIVE_ROWS, build_structured_reference_contract,
    structured_reference_values, structured_reference_jacobian,
)

BASE=_runtime_data_root()
DEFAULT_PARENT_CHECKPOINT = (BASE / 'current_common_hockey_soft_reference_source_freezes'
    / '9a8c1ac9d049' / 'complete_frozen_solver_checkpoint_UNACCEPTED.npz')
DEFAULT_PARENT_SHA256 = '9a8c1ac9d0496b3a6150583924d6d0b162d09244865ded02520bdc9afd1cf171'
CONTRACT_KEY = 'aux__current_soft_geometry_reference_contract_json'
ROW_IDS = (
    'C0_A_common_radius', 'C0_nominal_radius', 'C0_on_current_C',
    'C0_on_actual_boundary_midaxis', 'A_pair_midaxis',
    'A_pair_half_separation', 'A_pair_to_right_goal',
    'A_pair_station_symmetry', 'B1_parallel_C', 'B2_parallel_C',
    'goal_line_left_parallel_C', 'goal_line_right_parallel_C',
    'blue_pair_symmetry', 'blue_pair_separation',
    'left_goal_to_actual_estimated_end', 'right_goal_to_actual_estimated_end',
    'FR_common_outer_radius', 'FR_tee_separation',
)
DEFINITIONS = (
    'All five current hockey radii are one shared radius: structural zero.',
    'Current common five-circle hockey radius in conditional metres.',
    'C0 is the centre of current C supporting carrier: structural zero.',
    'Historical global-Y centre anchor removed; retained zero diagnostic row.',
    'Right A-pair mean lies on the current hockey longitudinal axis: structural zero.',
    'Half A1/A2 separation on the current hockey transverse axis.',
    'Current right A-pair mean normal distance inward from current right goal carrier.',
    'A1/A2 share their current hockey longitudinal station: structural zero.',
    'B1 parallel to current C carrier by construction: structural zero.',
    'B2 parallel to current C carrier by construction: structural zero.',
    'Left goal carrier parallel to current C carrier by construction: structural zero.',
    'Right goal carrier parallel to current C carrier by construction: structural zero.',
    'Current B1/B2 carriers are symmetric about current C: structural zero.',
    'B2 minus B1 projected carrier separation on the current hockey axis.',
    'Current left goal-centre inward distance from the current hockey-axis/current-boundary intersection.',
    'Current right goal-centre inward distance from the current hockey-axis/current-boundary intersection.',
    'One shared outer radius of all eight current F/R houses.',
    'Current paired F/R tee-centre separation on the current curling axis.',
)


def load_inherited_reference_contract(checkpoint=DEFAULT_PARENT_CHECKPOINT):
    """Read the real frozen reference ledger, with file and JSON provenance."""
    path = Path(checkpoint).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if path == DEFAULT_PARENT_CHECKPOINT.resolve() and digest != DEFAULT_PARENT_SHA256:
        raise ValueError('Historical 9a8c reference checkpoint changed')
    with np.load(path, allow_pickle=False) as stored:
        value = stored[CONTRACT_KEY]
        if value.ndim != 0:
            raise ValueError('Scalar historical reference contract required')
        contract = json.loads(str(value.item()))
    provenance = dict(checkpoint=str(path), checkpoint_sha256=digest,
        checkpoint_contract_key=CONTRACT_KEY,
        reference_contract_sha256=hashlib.sha256(json.dumps(contract,
            sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest())
    return contract, provenance


class Named66SoftNominalGeometryFactor:
    """The same reference18, on explicit named66 complete-world columns.

    The current provider parameterizes goal distances from dynamically
    recomputed wall intersections, so these distance derivatives are the
    named distance columns. Boundary derivatives move both goal and wall;
    they do not silently freeze the old rectangular boundary.
    """
    def __init__(self, geometry, geometry_indices, parameter_count, *,
                 inherited_contract=None, inherited_provenance=None,
                 parent_checkpoint=DEFAULT_PARENT_CHECKPOINT, config=None):
        self.geometry = geometry
        self.world_parameter_count = int(geometry.parameter_count)
        if self.world_parameter_count < 66:
            raise ValueError('Actual current named66 geometry prefix required')
        self.parameter_count = int(parameter_count)
        self.geometry_indices = _readonly(geometry_indices, int)
        if (self.geometry_indices.shape != (self.world_parameter_count,) or
            len(set(self.geometry_indices.tolist())) != self.world_parameter_count or
            np.any(self.geometry_indices < 0) or
            np.any(self.geometry_indices >= self.parameter_count)):
            raise ValueError('Unique explicit complete-world parameter map required')
        if inherited_contract is None:
            inherited_contract, inherited_provenance = load_inherited_reference_contract(parent_checkpoint)
        inherited = deepcopy(inherited_contract)
        old_rows = inherited.get('rows', [])
        if (inherited.get('schema') != VERSION or
            inherited.get('new_engineering_reference_rows') != 18 or
            tuple(r.get('id') for r in old_rows) != ROW_IDS):
            raise ValueError('Explicit authenticated historical reference18 ledger required')
        self.config = EngineeringReferenceConfig(**inherited['reference_configuration'])
        if config is not None and (not isinstance(config, EngineeringReferenceConfig) or
                                  asdict(config) != asdict(self.config)):
            raise ValueError('Inherited reference widths/nominals cannot be silently changed')
        if (not self.config.include_curling_tee_and_outer_radius or
            self.config.include_curling_inner_radii or self.config.include_button_lower_bound):
            raise ValueError('Exactly the inherited eighteen engineering references required')
        structured_rows, nominal, scales, current_contract = build_structured_reference_contract(geometry, self.config)
        if (not np.array_equal(nominal, [r['nominal'] for r in old_rows]) or
            not np.array_equal(scales, [r['working_sigma'] for r in old_rows]) or
            any(r.get('loss') != 'quadratic' or
                r.get('role') != 'engineering_reference_not_observation' or
                r.get('unit') != ('rad' if 8 <= i < 12 else 'm')
                for i, r in enumerate(old_rows))):
            raise ValueError('Historical reference identity, target, width, unit or loss mismatch')
        self.unit = float(geometry.scene_unit_m)
        if not np.isfinite(self.unit) or self.unit <= 0:
            raise ValueError('Finite positive inherited conditional scene unit required')
        self.nominal = _readonly(nominal)
        self.scales = _readonly(scales)
        self.quadratic = _readonly(np.ones(18), bool)
        self.row_count = 18
        rows = []
        for i, row in enumerate(old_rows):
            row = deepcopy(row)
            row.update(index=i, inherited_definition=row['definition'],
                definition=DEFINITIONS[i], current_relation_name=structured_rows[i]['name'],
                current_relation_role=structured_rows[i]['role'],
                active_world_column=ACTIVE_ROWS.get(i),
                active_named_parameter=structured_rows[i]['active_named_parameter'],
                current_hockey_axes_not_global_XY=True,
                disabled_historical_origin_anchor=(i == 3),
                structural_zero_is_not_independent_observation=(i not in ACTIVE_ROWS))
            rows.append(row)
        self.rows = tuple(rows)
        self.contract = dict(schema=VERSION,
            adapter_schema='rinkspace.inherited-reference18-on-current-named66.v1',
            original_source_rows_modified=0, new_engineering_reference_rows=18,
            parameter_count=self.parameter_count, world_parameter_count=self.world_parameter_count,
            world_parameter_indices=self.geometry_indices.tolist(),
            reference_configuration=asdict(self.config),
            source_role='explicit_engineering_reference_not_measured_source',
            loss='component quadratic L2',
            inherited_reference_provenance=deepcopy(inherited_provenance),
            inherited_reference_contract=inherited,
            current_axis_reference_contract=current_contract,
            inherited_row_id_order_nominals_widths_units_losses_preserved=True,
            current_relation_operator_adapted_explicitly=True,
            historical_global_Y_origin_anchor_removed=True,
            structural_zero_rows_retained=sorted(set(range(18)) - set(ACTIVE_ROWS)),
            active_soft_length_reference_rows=sorted(ACTIVE_ROWS),
            goal_end_reference_uses_dynamic_current_boundary_axis_intersections=True,
            old_boundary_bounds_not_frozen=True,
            conditional_scene_unit_m=self.unit,
            final_display_metric_uniform_scale_is_separate=True,
            actual_boundary_60x30_transport_installed=False,
            actual_width30_reference_row_installed=False,
            design_dimensions_metres=[self.config.design_length_m, self.config.design_width_m],
            widths_are_working_engineering_choices_not_measurement_uncertainties=True,
            nominal_structure_may_be_overruled_by_original_real_source=True,
            unknown_lane_end_or_missing_observations_added=False,
            sheet_width_reference_installed=False,
            original_profile_arrays_roles_UV_sigma_budget_and_aliases_preserved=True,
            reference_operator_adds_no_parameters=True,
            inherited_active_named_reference_columns_unchanged=True,
            appended_advertisement_locations_not_reference_targets=True,
            must_pass_fresh_complete_source_gate_before_solver=True,
            accuracy_accepted=False, rows=deepcopy(rows))

    def evaluate_world(self, q, jacobian=True):
        q = self.geometry.validate_parameters(q)
        values = structured_reference_values(self.geometry, q)
        raw = _readonly(values - self.nominal)
        residual = _readonly(raw / self.scales)
        j = (_readonly(structured_reference_jacobian(self.geometry, q) /
                       self.scales[:, None]) if jacobian else None)
        return GeometryReferenceEvaluation(raw, residual, j, self.quadratic,
            self.scales, self.rows, dict(cost=float(.5 * (residual @ residual)),
                derivative_is_unique=True, nondifferentiable_reference_branches=[],
                original_source_rows_modified=0, dynamic_boundary_reference=True,
                actual_boundary_60x30_transport_installed=False,
                sheet_width_not_inferred_from_pitch=True))

    def block(self, q, jacobian=True, *, original_start=0):
        q = np.asarray(q, float)
        if q.shape != (self.parameter_count,) or not np.isfinite(q).all():
            raise ValueError('Explicit finite complete programme state required')
        result = self.evaluate_world(q[self.geometry_indices], jacobian)
        matrix = None
        if jacobian:
            rr, cc = np.nonzero(result.jacobian)
            matrix = csr_matrix((result.jacobian[rr, cc],
                (rr, self.geometry_indices[cc])), shape=(18, self.parameter_count))
        ledger = dict(family='soft_nominal_geometry_engineering_references',
            schema=VERSION, adapter_schema=self.contract['adapter_schema'],
            original_start=int(original_start), original_stop=int(original_start) + 18,
            references_are_not_image_observations=True, working_sigma_not_measured=True,
            source_rows=deepcopy(list(self.rows)),
            complete_parameter_count=self.parameter_count,
            world_parameter_indices=self.geometry_indices.tolist())
        return SourceRowBlock(result.residual, matrix, result.quadratic, result.scales, ledger)


def wrap_named66_soft_geometry_references(base, geometry, geometry_indices, **kwargs):
    """Delegate the complete base API and append exactly eighteen rows."""
    factor = Named66SoftNominalGeometryFactor(geometry, geometry_indices,
        int(base.count), **kwargs)
    return CompleteSourceWithSoftGeometryReferences(base, factor)
