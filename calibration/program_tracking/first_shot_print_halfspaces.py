"""Route the unchanged H8 print-canvas/pole domains into the shared QP."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import types

import numpy as np
from scipy.sparse import csr_matrix, issparse, vstack

from .first_shot_print_layout import EXTENDED_MATERIAL_DOMAIN, print_layout_template


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def exact_hull_support(uv):
    """Original row indices of the exact dyadic hull; no geometric tolerance."""
    uv = np.asarray(uv, float)
    unique, first = np.unique(uv, axis=0, return_index=True)
    ratios = [[float(v).as_integer_ratio() for v in point] for point in unique]
    power = max(den.bit_length()-1 for point in ratios for _, den in point)
    points = sorted((numx << (power-(denx.bit_length()-1)),
                     numy << (power-(deny.bit_length()-1)), int(first[i]))
                    for i, ((numx, denx), (numy, deny)) in enumerate(ratios))
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    lower, upper = [], []
    for point in points:
        while len(lower)>1 and cross(lower[-2], lower[-1], point)<=0: lower.pop()
        lower.append(point)
    for point in reversed(points):
        while len(upper)>1 and cross(upper[-2], upper[-1], point)<=0: upper.pop()
        upper.append(point)
    hull = lower[:-1]+upper[:-1]
    # Keep every distinct point for a line/point instead of choosing a jittered hull.
    return np.asarray([p[2] for p in (hull if len(hull)>=3 else points)], int)


class PrintLayoutDomain:
    """Compile exact affine margins of the same immutable prototype vertices."""
    def __init__(self, physical_state, parameter_count):
        original = getattr(physical_state, 'original', physical_state)
        if not hasattr(original, 'print_layout_world_slices'):
            raise ValueError('Original physical print-layout coordinate mapping required')
        self.parameter_count = int(parameter_count)
        world = np.asarray(original.world_parameter_indices, int)
        rr, cc, vv, constants, ledger = [], [], [], [], []

        def append(coefficients, offset, columns):
            row = len(constants)
            for j in np.flatnonzero(coefficients):
                rr.append(row); cc.append(int(columns[j])); vv.append(float(coefficients[j]))
            constants.append(float(offset))

        corners = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        full_vertex_ledger = []
        uncompressed_rows = 0
        for number in original.print_layout_advertisement_ids:
            material = print_layout_template(original.templates, number)
            part = original.print_layout_world_slices[number]
            columns = world[np.arange(part.start, part.stop)]
            if len(columns) != 8 or len(np.unique(columns)) != 8 or np.any(columns >= self.parameter_count):
                raise ValueError('Eight distinct full-state print columns required')
            pole_points, canvas_points = [corners], []
            pole_owners = [dict(contour='unit_canvas_corners', vertex=i) for i in range(4)]
            canvas_owners = []
            uncompressed_rows += 4
            for key, record in material['contours'].items():
                uv = np.asarray(record['material_uv'], float)
                if uv.ndim != 2 or uv.shape[1] != 2 or not len(uv) or not np.isfinite(uv).all():
                    raise ValueError('Finite actual prototype material vertices required')
                extended = record.get('material_coordinate_domain') == EXTENDED_MATERIAL_DOMAIN
                owners = [dict(contour=key, vertex=i) for i in range(len(uv))]
                pole_points.append(uv); pole_owners.extend(owners)
                if not extended:
                    canvas_points.append(uv); canvas_owners.extend(owners)
                uncompressed_rows += len(uv)*(1 if extended else 5)
                full_vertex_ledger.append(dict(advertisement_id=int(number), contour=key,
                    prototype_sha256=_sha(uv), prototype_vertices=len(uv),
                    prototype_material_domain=record.get('material_coordinate_domain'),
                    full_state_columns=columns.tolist(), extended_material=bool(extended)))
            groups = [('pole', np.vstack(pole_points), pole_owners)]
            if canvas_points: groups.append(('outer_canvas', np.vstack(canvas_points), canvas_owners))
            for kind, points, owners in groups:
                support = exact_hull_support(points)
                begin = len(constants)
                for x, y in points[support] - .5:
                    if kind == 'pole':
                        append(np.array([0., 0., 0., 0., 0., 0., x, y]), 1., columns)
                    else:
                        append(np.array([x, y, 0., 0., 1., 0., .5*x, .5*y]), x+.5, columns)
                        append(np.array([-x, -y, 0., 0., -1., 0., .5*x, .5*y]), .5-x, columns)
                        append(np.array([0., 0., x, y, 0., 1., .5*x, .5*y]), y+.5, columns)
                        append(np.array([0., 0., -x, -y, 0., -1., .5*x, .5*y]), .5-y, columns)
                ledger.append(dict(advertisement_id=int(number), domain=kind,
                    original_vertices=len(points), original_vertices_sha256=_sha(points),
                    full_state_columns=columns.tolist(),
                    support_indices=support.tolist(), support_owners=[owners[i] for i in support],
                    row_order=['w'] if kind=='pole' else ['Nx_plus_half_w', 'half_w_minus_Nx', 'Ny_plus_half_w', 'half_w_minus_Ny'],
                    row_slice=[begin, len(constants)]))
        self.matrix = csr_matrix((vv, (rr, cc)), shape=(len(constants), self.parameter_count))
        self.constant = np.asarray(constants, float)
        self.ledger = ledger
        self.full_vertex_ledger = full_vertex_ledger
        self.identity = dict(schema='rinkspace.existing-print-H8-affine-domain.v1',
            row_count=len(constants), parameter_count=self.parameter_count,
            equivalent_uncompressed_row_count=uncompressed_rows,
            original_world_parameter_indices_sha256=_sha(world),
            matrix_data_sha256=_sha(self.matrix.data), matrix_indices_sha256=_sha(self.matrix.indices),
            matrix_indptr_sha256=_sha(self.matrix.indptr), constant_sha256=_sha(self.constant),
            ledger=deepcopy(ledger), full_original_vertex_ledger=deepcopy(full_vertex_ledger),
            support_algorithm='exact dyadic-integer monotone-chain convex hull; degenerate all-unique fallback',
            no_joggle_or_geometric_tolerance=True, no_geometry_unit_conversion=True,
            prototype_and_normalized_outer_canvas_unchanged=True,
            extended_materials_not_constrained_to_outer_canvas=True,
            pole_QP_is_closed_relaxation_of_original_strict_positive_domain=True,
            strict_pole_and_positive_determinant_and_original_material_guards_retained=True,
            positive_pole_epsilon_or_numeric_cap_added=False)

    def margins(self, physical_q):
        q = np.asarray(physical_q, float)
        if q.shape != (self.parameter_count,) or not np.isfinite(q).all():
            raise ValueError('Complete physical q required for print-domain margins')
        return np.asarray(self.matrix @ q).ravel() + self.constant

    def pullback(self, physical_q, chain, selected, fixed):
        selected, fixed = np.asarray(selected, int), np.asarray(fixed, int)
        if chain.shape != (self.parameter_count, self.parameter_count):
            raise ValueError('Complete canonical-to-physical chain required')
        full = (self.matrix @ csr_matrix(chain)).tocsr(); full.eliminate_zeros()
        omitted = np.setdiff1d(np.arange(self.parameter_count), np.r_[selected, fixed])
        if full[:, omitted].nnz:
            raise ValueError('Print domain has an undeclared nonshared canonical dependency')
        return dict(matrix=full[:, selected].tocsr(), lower=-self.margins(physical_q))


def install(owner, *, entry_reference, qa_reference):
    """Explicitly extend only a newly constructed owner's linear-domain callback."""
    refs = [reference(__file__), deepcopy(entry_reference), deepcopy(qa_reference)]
    for ref in refs:
        if reference(ref['path']) != {k: ref[k] for k in ('path', 'sha256')}:
            raise ValueError('Frozen print-halfspace execution reference changed')
    qa = json.loads(Path(qa_reference['path']).read_text(encoding='utf8'))
    if qa.get('passed') is not True or qa.get('producer') != refs[0]:
        raise ValueError('Explicit passed QA for this print-halfspace producer required')
    if hasattr(owner, '_print_H8_domain'):
        raise ValueError('Print H8 halfspaces already installed on this new owner')
    domain = PrintLayoutDomain(owner.programme.state, owner.parameter_count)
    old_constraints = owner.linear_constraints
    old_identity = owner.checkpoint_identity
    old_arrays = owner.checkpoint_arrays
    receipt = dict(schema='rinkspace.explicit-print-H8-QP-domain-execution.v1',
        producers=refs, domain=domain.identity,
        source_guard_recovery_parameters_r_J_or_objective_modified=False,
        original_constraint_prefix_and_current_lower_preserved=True,
        exact_physical_H8_to_canonical_chain_applied=True,
        QP_closed_pole_boundary_not_claimed_guard_feasible=True,
        floating_point_boundary_acceptance_owned_by_unchanged_actual_guard=True)

    def constraints(this, q, selected):
        if not np.array_equal(selected, this.shared):
            raise ValueError('Exact active shared index order required')
        original = old_constraints(q, selected)
        qe, chain = this.effective_q_and_chain(q)
        extra = domain.pullback(qe, chain, selected, this.fixed)
        matrix = (vstack((csr_matrix(original['matrix']), extra['matrix']), format='csr')
                  if issparse(original['matrix']) else np.vstack((original['matrix'], extra['matrix'].toarray())))
        return dict(original, matrix=matrix, lower=np.r_[original['lower'], extra['lower']],
            report=dict(original.get('report', {}),
                original_constraint_prefix_rows=len(original['lower']),
                appended_print_H8_row_slice=[len(original['lower']), len(original['lower'])+len(extra['lower'])],
                explicit_print_H8_domain=domain.identity))

    def identity(this):
        return dict(old_identity(), print_H8_QP_domain=deepcopy(receipt))

    def arrays(this):
        return dict(old_arrays(), print_H8_QP_domain_identity_json=np.asarray(json.dumps(receipt, sort_keys=True)))

    owner._print_H8_domain = domain
    owner.source_producers = deepcopy(owner.source_producers)
    owner._references = deepcopy(owner._references)
    for ref in refs:
        if ref not in owner.source_producers: owner.source_producers.append(deepcopy(ref))
        if ref not in owner._references: owner._references.append(deepcopy(ref))
    owner.linear_constraints = types.MethodType(constraints, owner)
    owner.checkpoint_identity = types.MethodType(identity, owner)
    owner.checkpoint_arrays = types.MethodType(arrays, owner)
    return deepcopy(receipt)
