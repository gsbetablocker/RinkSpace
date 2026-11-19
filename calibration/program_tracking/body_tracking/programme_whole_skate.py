"""Native source outlines, directed outsole observations and support-foot phase."""
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

from .absolute_motion_state import chain_exposure
from .programme_skate_evidence import exposure_times
from .rebuild_exposure_body import moving_projection
from .rebuild_prepare import digest, load
from .sequence_skate_state import perturb_relative
from .skate_physics import fk_state
from .weighted_blocks import block, signature

LEG_COLUMNS = (
    (3, 4, 5, 6, 7, 8, 15, 16, 17, 24, 25, 26),
    (3, 4, 5, 9, 10, 11, 18, 19, 20, 27, 28, 29),
)


def ambiguous_shoe_frames(packet):
    """Mask identities are unproved during overlap and its immediate transitions."""
    admitted=(packet['role']==1)&packet['visible']
    uncertain=~admitted.all(1)|packet.get('partial_boundary',np.zeros_like(admitted)).any(1)
    result=uncertain.copy();shots=packet['shot_ids']
    for offset in range(1,5):
        same=shots[offset:]==shots[:-offset]
        result[offset:]|=uncertain[:-offset]&same
        result[:-offset]|=uncertain[offset:]&same
    return result


def native_point_derivative(problem, base, delta, points, foot, derivative):
    """Differentiate actual upper deformation as well as rigid FK, at fixed associations."""
    native = np.zeros((*derivative.shape[:-1], 69))
    native[..., :3] = derivative
    shoe = np.zeros((*derivative.shape[:-1], 6))

    def column(plus, minus):
        a = points(fk_state(problem.reference, base, plus))
        b = points(fk_state(problem.reference, base, minus))
        return np.einsum("nkpac,nakc->nkpa", derivative, (a - b) / 2.e-6, optimize=True)

    columns = LEG_COLUMNS[foot]
    if getattr(problem, 'analytic_rigid_shoe_derivatives', False):
        from .fullbody_fit import right_jacobian
        pose = fk_state(problem.reference, base, delta)
        world = points(pose).swapaxes(1, 2)
        count = world.shape[0]
        # Hip, knee and root rotations carry the boot rigidly. Only ankle
        # rotation changes the upper cuff relative to its shank and therefore
        # retains the exact deformation finite difference below.
        for joint in (0, *((1, 4) if foot == 0 else (2, 5))):
            first = 3 + 3*joint
            axes = pose['Q'] @ right_jacobian(delta[:, first:first+3])
            pivot = pose['P']
            if joint:
                axes = pose['Q'] @ pose['frames'][:, joint] @ right_jacobian(delta[:, first:first+3])
                pivot = pose['native_world'][:, joint]
            arm = world - pivot.reshape(count, 1, 5, 3)
            vectors = axes.reshape(count, 5, 3, 3).swapaxes(-1, -2)
            movement = np.cross(vectors[:, None], arm[..., None, :])
            native[..., first:first+3] = np.einsum('nkpac,nkadc->nkpad', derivative, movement, optimize=True)
        ankle = 7 + foot
        columns = tuple(range(3+3*ankle, 6+3*ankle))
    for axis in columns:
        plus, minus = delta.copy(), delta.copy()
        plus[:, axis] += 1.e-6
        minus[:, axis] -= 1.e-6
        native[..., axis] = column(plus, minus)
    if not getattr(problem, 'fixed_relative_skates', False):
        for axis in range(3):
            shoe[..., 3 * foot + axis] = column(perturb_relative(delta, foot, axis, 1.e-6),
                                                perturb_relative(delta, foot, axis, -1.e-6))
    return native, shoe


def chain(problem, delta, times, body, shoe):
    derivative = chain_exposure(body, delta)
    derivative += delta.owner.shoe_exposure_chain(shoe, delta.parameters, times)
    return hstack((csr_matrix((derivative.shape[0], 17)), derivative), format="csr")


class WholeSkateSourceFactor:
    """Joint boot and blade silhouette on the common continuous exposure state."""

    name = "whole_skate_source"

    def __init__(self, problem, specification, *, frames=None):
        from experiments.skate_video_v1.body_skate_fit import Assembly

        self.problem = problem
        self.assembly = Assembly(problem.binding)
        path = Path(specification["path"])
        if digest(path) != specification["sha256"]:
            raise ValueError("Whole-skate source roles or contours changed")
        self.packet = load(path)
        self.distance_model = specification.get('distance_model', 'normal_arcsinh')
        self.continuous_edge_matches = specification.get('continuous_edge_matches', False)
        if self.distance_model not in ('normal_arcsinh', 'vector_pseudo_huber'):
            raise ValueError('Unknown whole-skate source distance model')
        if self.continuous_edge_matches and self.distance_model != 'vector_pseudo_huber':
            raise ValueError('Continuous edge matches require the vector pixel objective')
        ref = problem.reference
        np.testing.assert_array_equal(self.packet["source_pts"], ref.p["pts"])
        np.testing.assert_array_equal(self.packet["time_base"], [1, 12800])
        np.testing.assert_array_equal(self.packet["camera_sha256"], problem.camera.info["state_sha256"])
        role = self.packet["role"]
        selected = np.ones(len(role), dtype=bool)
        if frames is not None:
            frames = np.asarray(frames, dtype=int)
            if np.any((frames < 0) | (frames >= len(role))):
                raise ValueError("Whole-skate source frame outside native clock")
            selected[:] = False
            selected[frames] = True
        if specification.get('ambiguous_source_union',False):
            selected &= ~ambiguous_shoe_frames(self.packet)
        self.batches = []
        for foot in (0, 1):
            frames = np.flatnonzero((role[:, foot] == 1) & selected)
            for first in range(0, len(frames), 32):
                group = frames[first:first + 32]
                times = exposure_times(problem, group)
                self.batches.append((foot, group, times, ref.at(times)))
        self.matches = None
        self.refresh_each_state = False
        self.associated_parameters = None
        self.last_distances = None
        self.association_signature = ""
        self.group_provenance = dict(source_pts=ref.p["pts"].tolist(), role="fit",
                                     evidence_sha256=specification["sha256"], native_exposure_nodes=5)
        if self.distance_model != 'normal_arcsinh' or self.continuous_edge_matches:
            self.group_provenance.update(distance_model=self.distance_model,
                                        continuous_edge_matches=self.continuous_edge_matches)

    def _segment_matches(self, pixels, observed, boundary_valid=None, partial=False):
        """Project onto actual boundary edges instead of snapping to vertices."""
        from scipy.spatial import cKDTree
        from experiments.skate_video_v1.inputs import distribute
        outlines = self.assembly.raster(pixels)
        boundary = np.concatenate(outlines)
        pairs = self.assembly.edges
        mids = pixels[pairs].mean(axis=1)
        keep = cKDTree(boundary).query(mids)[0] < 1.55
        pairs = pairs[keep]
        if not len(pairs):
            raise ValueError('No projected shoe boundary edges')

        def closest(query, starts, ends):
            edge = ends-starts
            t = np.clip(np.einsum('qec,ec->qe', query[:, None]-starts, edge)
                        / np.maximum(np.sum(edge*edge, axis=1), 1.e-12), 0., 1.)
            projected = starts[None]+t[..., None]*edge[None]
            selected = np.argmin(np.sum((query[:, None]-projected)**2, axis=-1), axis=1)
            return selected, t[np.arange(len(query)), selected], projected[np.arange(len(query)), selected]

        source = distribute(observed, 80, closed=True)
        if boundary_valid is not None:
            # Qualification addresses the ORIGINAL sampled pixels. Re-sampling
            # the polygon would silently move the valid flags onto other arcs.
            source = np.asarray(observed)
            valid = np.asarray(boundary_valid, bool)
            if valid.shape != (len(source),) or np.count_nonzero(valid) < 8:
                raise ValueError('Insufficient qualified source boundary arcs')
            ids = np.flatnonzero(valid)
            take = np.rint(np.linspace(0, len(ids)-1, 40 if partial else 20)).astype(int)
            target = source[ids[take]]
        else:
            valid = np.ones(len(source), bool)
            target = distribute(observed, 20, closed=True)
        i, a, _ = closest(target, pixels[pairs[:, 0]], pixels[pairs[:, 1]])
        if partial:
            # A visible arc must touch the model, but the model's unseen side
            # must not collapse onto the small visible fragment. In particular,
            # no completed outline is fabricated through the foreground boot.
            return pairs[i], a, target, np.zeros((40, 2))
        perimeter = distribute(max(outlines, key=len), 20, closed=True)
        j, b, material = closest(perimeter, pixels[pairs[:, 0]], pixels[pairs[:, 1]])
        segments = valid & np.roll(valid, -1)
        if np.count_nonzero(segments) < 2:
            raise ValueError('A complete source outline needs resolved adjacent arcs')
        _, _, back = closest(material, source[segments], np.roll(source, -1, axis=0)[segments])
        return np.r_[pairs[i], pairs[j]], np.r_[a, b], np.r_[target, back], np.zeros((40, 2))

    def _points(self, state, foot, selectors):
        return self.assembly.points(state, foot, selectors, (state["R"], state["O"]))

    def associate(self, x):
        matches = []
        for foot, frames, times, base in self.batches:
            delta = self.problem.body.state.increments(x, times)
            pose = fk_state(self.problem.reference, base, delta)
            nodes = self._points(pose, foot, None).reshape(len(frames), 5, -1, 3)
            uv = moving_projection(self.problem.camera, nodes, frames)["uv"]
            associate = self._segment_matches if self.continuous_edge_matches else self.assembly.associations
            if 'boundary_valid' in self.packet:
                if not self.continuous_edge_matches:
                    raise ValueError('Qualified source arcs require continuous boundary matching')
                rows = [self._segment_matches(a,self.packet['region'][frame,foot],
                            self.packet['boundary_valid'][frame,foot],
                            bool(self.packet['partial_boundary'][frame,foot]))
                        for a,frame in zip(uv,frames)]
            else:
                rows = [associate(a, b) for a, b in zip(uv, self.packet["region"][frames, foot])]
            matches.append(tuple(np.stack(values) for values in zip(*rows)))
        self.matches = tuple(matches)
        self.association_signature = signature(*(a for group in matches for a in group[:2]))
        self.associated_parameters = np.array(x, copy=True)

    def checkpoint_fields(self):
        if self.matches is None:
            raise ValueError("Whole-skate correspondences must be frozen before checkpoint")
        return dict(whole_skate_match_frames=np.concatenate([v[1] for v in self.batches]),
                    whole_skate_match_feet=np.concatenate([np.full(len(v[1]),v[0]) for v in self.batches]),
                    whole_skate_match_vertices=np.concatenate([m[0] for m in self.matches]),
                    whole_skate_match_fraction=np.concatenate([m[1] for m in self.matches]),
                    whole_skate_match_target=np.concatenate([m[2] for m in self.matches]),
                    whole_skate_match_normal=np.concatenate([m[3] for m in self.matches]),
                    whole_skate_match_sha256=np.array(signature(*(a for m in self.matches for a in m))))

    def restore_checkpoint(self, packet):
        frames = np.concatenate([v[1] for v in self.batches])
        feet = np.concatenate([np.full(len(v[1]),v[0]) for v in self.batches])
        np.testing.assert_array_equal(packet['whole_skate_match_frames'],frames)
        np.testing.assert_array_equal(packet['whole_skate_match_feet'],feet)
        values = tuple(packet['whole_skate_match_'+k] for k in ('vertices','fraction','target','normal'))
        if any(len(v)!=len(frames) or not np.isfinite(v).all() for v in values):
            raise ValueError("Invalid saved whole-skate association")
        edges, fractions, targets, normals = values
        if (edges.dtype.kind not in 'iu' or edges.shape != (*fractions.shape,2)
                or targets.shape != edges.shape or normals.shape != edges.shape
                or np.any((edges<0)|(edges>=len(self.assembly.vertices)))
                or np.any((fractions<0)|(fractions>1))):
            raise ValueError("Saved whole-skate association changed its geometry")
        stop = np.cumsum([len(v[1]) for v in self.batches])[:-1]
        self.matches = tuple(zip(*(np.split(v,stop) for v in values),strict=True))
        if signature(*(a for m in self.matches for a in m)) != str(packet['whole_skate_match_sha256']):
            raise ValueError("Saved whole-skate association hash changed")
        self.association_signature = signature(*(a for group in self.matches for a in group[:2]))

    def _batch(self, x, batch, match, jacobian):
        p = self.problem
        foot, frames, times, base = batch
        indices, fractions, target, normal = match

        def points(s):
            return self._points(s, foot, (indices, fractions)).reshape(len(frames), 5, -1, 3)

        delta = p.body.state.increments(x, times)
        pose = fk_state(p.reference, base, delta)
        projection = moving_projection(p.camera, points(pose), frames, jacobian=jacobian)
        difference = projection["uv"] - target
        signed = np.sum(difference * normal, axis=-1)
        radius = self.packet["sigma_px"][frames, foot, None]
        if self.distance_model == 'vector_pseudo_huber':
            # Symmetric nearest-boundary pixel distances retain both image
            # components. A normal-only residual can hide a reversed forefoot
            # behind a nearby parallel edge. Associations remain explicit and
            # fixed during each derivative check, not material identities.
            z = difference / radius[..., None]
            u = np.sqrt(1. + np.sum(z*z, axis=-1) / 4.)
            k = np.sqrt(2. / (u + 1.))
            scale = np.sqrt(self.packet['measure_s'][frames, foot, None] / z.shape[1])
            r = scale[..., None] * k[..., None] * z
            J = None
            if jacobian:
                B, S = native_point_derivative(p, base, delta, points, foot,
                                               projection['pixel_jac_nodes'])
                dk = -k / (16. * u * (u + 1.))
                D = k[..., None, None] * np.eye(2) + 2. * dk[..., None, None] * z[..., :, None] * z[..., None, :]
                D *= (scale / radius)[..., None, None]
                body = np.einsum('nkqp,nkpad->nkqad', D, B)
                shoe = np.einsum('nkqp,nkpad->nkqad', D, S)
                J = chain(p, delta, times, body.reshape(len(frames), -1, 5, 69),
                          shoe.reshape(len(frames), -1, 5, 6))
            return r.ravel(), J, np.median(np.linalg.norm(difference, axis=-1), axis=1)
        scaled = signed / radius
        scale = np.sqrt(self.packet["measure_s"][frames, foot, None] / signed.shape[1])
        r = scale * np.arcsinh(scaled)
        J = None
        if jacobian:
            raw = projection["pixel_jac_nodes"]
            B, S = native_point_derivative(p, base, delta, points, foot, raw)
            weight = scale / (radius * np.sqrt(1 + scaled**2))
            metric = weight[..., None] * normal
            J = chain(p, delta, times, np.einsum("nkp,nkpad->nkad", metric, B),
                      np.einsum("nkp,nkpad->nkad", metric, S))
        return r.ravel(), J, np.median(np.linalg.norm(difference, axis=-1), axis=1)

    def evaluate(self, x, jacobian=False):
        if (self.matches is None or (self.refresh_each_state and
                (self.associated_parameters is None or
                 not np.array_equal(self.associated_parameters,x)))):
            self.associate(x)
        values = [self._batch(x, batch, match, jacobian)
                  for batch, match in zip(self.batches, self.matches, strict=True)]
        r = np.concatenate([v[0] for v in values])
        J = vstack([v[1] for v in values], format="csr") if jacobian else None
        by_foot = np.full(self.packet["role"].shape, np.nan)
        for (foot, frames, _, _), value in zip(self.batches, values, strict=True):
            by_foot[frames, foot] = value[2]
        self.last_distances = by_foot
        metrics = {}
        for shot in np.unique(self.packet["shot_ids"]):
            for foot in (0, 1):
                d = by_foot[self.packet["shot_ids"] == shot, foot]
                d = d[np.isfinite(d)]
                if len(d):
                    metrics[f"{shot}:{foot}"] = dict(count=len(d), median_px=float(np.median(d)),
                                                     p90_px=float(np.quantile(d, .90)))
        cost = float(r @ r / 2)
        return dict(residual=r, jac=J, parts={self.name:cost}, metrics={self.name:metrics},
                    image_cost=cost, weighted_blocks=[block(self.name, r, J, self.group_provenance,
                                                           self.association_signature)])

    def check_direction(self, x, frames):
        """Use real native exposures and the held outer correspondence for the derivative check."""
        by_foot = []
        rng = np.random.default_rng(1112)
        for foot_id in (0,1):
            candidates = [(batch,match) for batch,match in zip(self.batches,self.matches,strict=True)
                          if batch[0] == foot_id]
            if not candidates:
                continue
            batch,match = min(candidates,key=lambda v:np.min(np.abs(v[0][1][:,None]-frames)))
            foot, ids, _, _ = batch
            local = np.flatnonzero(np.isin(ids, frames))
            if not len(local):
                local = np.argsort(np.min(np.abs(ids[:,None]-frames),axis=1))[:2]
            group = ids[local]
            times = exposure_times(self.problem, group)
            sample = (foot, group, times, self.problem.reference.at(times))
            selectors = tuple(v[local] for v in match)
            _r, J, _ = self._batch(x, sample, selectors, True)
            state = self.problem.body.state
            support = state.basis(times)
            body = np.unique(support.indices)[:,None]*69 + np.r_[np.arange(3),LEG_COLUMNS[foot]]
            relative = state.body_size + np.unique(state.shoe_curve.basis_sparse(times).indices)[:,None]*6 + 3*foot + np.arange(3)
            direction = np.zeros(len(x))
            columns = np.r_[body.ravel(),relative.ravel()]
            direction[columns] = rng.normal(size=len(columns))
            direction /= np.linalg.norm(direction)
            step = 2.e-5
            a = self._batch(x + step * direction, sample, selectors, False)[0]
            b = self._batch(x - step * direction, sample, selectors, False)[0]
            analytic = J[:, 17:] @ direction
            numeric = (a - b) / (2 * step)
            by_foot.append(dict(frames=self.packet["source_pts"][group].tolist(),foot=int(foot),rows=len(_r),
                active_test_columns=len(columns),native_exposure_nodes=5,
                max_abs=float(np.max(np.abs(analytic-numeric))),
                relative=float(np.linalg.norm(analytic-numeric)/max(np.linalg.norm(numeric),1.e-10))))
        if not by_foot:
            raise ValueError("No admitted native exposure for whole-skate derivative audit")
        return dict(max_abs=max(r["max_abs"] for r in by_foot),
                    relative=max(r["relative"] for r in by_foot),by_foot=by_foot)


class DirectedBootSourceFactor:
    """Signed heel-to-toe image chords from clear original source exposures."""

    name = "directed_boot_source"

    def __init__(self, problem, rows):
        self.problem = problem
        self.rows = rows

    def evaluate(self, x, jacobian=False):
        p = self.problem
        residuals, jacobians, metrics = [], [], []
        shape = p.binding.packet()
        local = np.array([[0., 0., shape["floor"][0]],
                          [shape["xs"][-1], 0., shape["floor"][-1]]])
        pts = p.reference.p["pts"]
        for foot in (0, 1):
            rows = [r for r in self.rows if r["foot"] == foot]
            if not rows:
                continue
            stamps = np.asarray([r["pts"] for r in rows])
            frames = np.searchsorted(pts, stamps)
            np.testing.assert_array_equal(pts[frames], stamps)
            times = exposure_times(p, frames)
            base = p.reference.at(times)
            d = p.body.state.increments(x, times)
            s = fk_state(p.reference, base, d)

            def points(state):
                w = state["O"][:, foot, None] + np.einsum("nij,kj->nki", state["R"][:, foot], local)
                return w.reshape(len(frames), 5, 2, 3)

            pr = moving_projection(p.camera, points(s), frames, jacobian=jacobian)
            vector = pr["uv"][:, 1] - pr["uv"][:, 0]
            expected = np.array([r["toe_uv"] for r in rows]) - np.array([r["heel_uv"] for r in rows])
            # Residuals are a pixel chord, not a normalized direction angle.
            # The same endpoint error then gives a wider angle range for a
            # foreshortened heel-to-toe span.
            sigma = np.asarray([r["sigma_px"] for r in rows])
            scale = np.sqrt(.02) / sigma
            err = vector - expected
            residuals.append((scale[:, None] * err).ravel())
            if jacobian:
                B, S = native_point_derivative(p, base, d, points, foot, pr["pixel_jac_nodes"])
                jacobians.append(chain(p, d, times, scale[:, None, None, None] * (B[:, 1] - B[:, 0]),
                                       scale[:, None, None, None] * (S[:, 1] - S[:, 0])))
            span = np.linalg.norm(expected, axis=-1)
            metrics.extend(dict(pts=int(stamp),foot=foot,distance_px=float(value),
                                source_span_px=float(length),
                                angle_uncertainty_rad=float(np.arctan2(noise,length)))
                           for stamp,value,length,noise in zip(stamps,np.linalg.norm(err,axis=-1),span,sigma,strict=True))
        r = np.concatenate(residuals)
        J = vstack(jacobians, format="csr") if jacobian else None
        cost = float(r @ r / 2)
        return dict(residual=r, jac=J, parts={self.name:cost}, metrics={self.name:metrics},
                    image_cost=cost, weighted_blocks=[block(self.name, r, J, self.rows,
                                                             signature(self.rows))])


class SupportShoePhase:
    """Finite penalty for opposite turn increments during a source-confirmed support phase."""

    name = "support_shoe_phase"

    def __init__(self, problem, episode):
        self.problem = problem
        self.foot = episode["foot"]
        self.direction = episode["turn_direction_sign"]
        if self.foot not in (0, 1) or self.direction not in (-1, 1):
            raise ValueError("A signed supported foot phase requires explicit source identity")
        ref = problem.reference
        pts = ref.p["pts"]
        frames = np.flatnonzero((pts >= episode["start_pts"]) & (pts <= episode["end_pts_inclusive"]))
        self.pts = pts[frames]
        self.times = ref.times[frames]
        self.base = ref.at(self.times)
        self.scale = np.sqrt(np.diff(self.times)) / float(episode["angular_scale_rad"])

    def evaluate(self, x, jacobian=False):
        p = self.problem
        d = p.body.state.increments(x, self.times)
        foot = self.foot

        def axis(delta):
            return fk_state(p.reference, self.base, delta)["R"][:, foot, :2, 0]

        u = axis(d)
        yaw = np.arctan2(u[:, 1], u[:, 0])
        change = np.diff(yaw)
        signed = self.direction * np.arctan2(np.sin(change), np.cos(change))
        r = self.scale * np.minimum(signed, 0.)
        J = None
        if jacobian:
            local = np.zeros((len(u), 1, 69))
            gradient = np.column_stack((-u[:, 1], u[:, 0])) / np.maximum(np.sum(u*u, axis=1)[:, None], 1.e-8)
            for column in LEG_COLUMNS[foot]:
                plus, minus = d.copy(), d.copy()
                plus[:, column] += 1.e-6
                minus[:, column] -= 1.e-6
                local[:, 0, column] = np.sum(gradient * (axis(plus) - axis(minus)), axis=1) / 2.e-6
            J = d.chain(local)
            J = J[1:] - J[:-1]
            J = J.multiply((self.direction * self.scale * (signed < 0))[:, None]).tocsr()
            J = hstack((csr_matrix((len(r), 17)), J), format="csr")
        cost = float(r @ r / 2)
        worst = int(np.argmin(signed))
        return dict(residual=r, jac=J, parts={self.name:cost},
                    metrics={self.name:dict(min_step_deg=float(np.rad2deg(signed[worst])),
                                            worst_pts=int(self.pts[worst + 1]),
                                            reverse_deg=float(np.rad2deg(-np.minimum(signed, 0.).sum())))},
                    weighted_blocks=[block(self.name, r, J, dict(foot=foot, signed_source_direction=self.direction))])
