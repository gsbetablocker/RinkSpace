"""Dynamic finite-board factor for already qualified, actual native pixel pairs.

The factor does not classify image content.  A caller must first establish that
both endpoints track the same static board texture, using source ROI, masks,
forward-backward flow and connected motion evidence; per-point advertisement
names are not required.  A physical panel arc may further restrict the hit.
It intersects each original source pixel anew at the current camera state and
projects that hit into the target image; ``xy1`` is never model-generated.

No invalid-trial penalty is invented here.  The current GTSAM backend has no
retraction boundary hook; an invalid hit raises ``BoardIntersectionError``.
"""
from __future__ import annotations

import cv2
import numpy as np

try:
    from .projection import from_state, project_fast
    from .reference_board_pair import BoardIntersectionError, ReferenceBoardPair
except ImportError:
    from projection import from_state, project_fast
    from reference_board_pair import BoardIntersectionError, ReferenceBoardPair


class NativeBoardPair(ReferenceBoardPair):
    """One-direction actual UV factor on the existing complete board ring.

    ``allowed_panels=None`` allows every panel.  A verified physical print arc
    may instead be supplied without freezing any one segment.  The source and
    target keep independent Brown25 camera templates.
    """

    def __init__(self, xy0, xy1, scene, camera0_template, camera1_template,
                 *, allowed_panels=None, seam_fraction_margin: float = 1e-8):
        panel_count = len(scene["board_surface"]["bottom_edge_xyz"])
        if allowed_panels is None:
            allowed_panels = np.arange(panel_count, dtype=np.int32)
        if not np.isfinite(seam_fraction_margin) or not 0 <= seam_fraction_margin < .5:
            raise ValueError("seam_fraction_margin must be finite in [0, .5)")
        super().__init__(xy0, xy1, scene, allowed_panels,
                         camera0_template, camera1_template)
        self.seam_fraction_margin = float(seam_fraction_margin)
        self.last_jacobian_counts = {"analytic_rows": 0, "numeric_rows": 0}

    def initial_validity(self, states) -> tuple[np.ndarray, tuple[str, ...]]:
        """Report geometry-valid original rows before making a fixed-size factor.

        This is an admission diagnostic, not a source-content/print-identity
        test.  A caller may form a new factor from the returned valid rows at
        initialization; rows must not be dropped inside residual/J callbacks.
        """
        source, target = states
        hits = self.source_hits(source)
        reason = list(hits.reason)
        live = np.flatnonzero(hits.valid)
        if not len(live):
            return np.zeros(len(self.xy0), bool), tuple(reason)
        camera = from_state(target, self.camera1_template)
        uv, depth = project_fast(camera, hits.xyz[live])
        projected = np.isfinite(uv).all(axis=1) & (depth > 0)
        for local in np.flatnonzero(~projected):
            reason[int(live[local])] = "target_projection_invalid"
        live = live[projected]
        if len(live):
            direction = hits.xyz[live]-camera[:3]
            target_hits = self._hits_from_rays(camera[:3], direction)
            for local, why in enumerate(target_hits.reason):
                if why != "ok":
                    reason[int(live[local])] = "target_" + why
                elif np.linalg.norm(target_hits.xyz[local]-hits.xyz[live[local]]) > 1e-4:
                    reason[int(live[local])] = "target_board_occlusion"
        valid = np.asarray([why == "ok" for why in reason], bool)
        return valid, tuple(reason)

    def _same_local_plane(self, baseline, other) -> np.ndarray:
        """An adjacent collinear polygon split is one smooth vertical plane."""
        baseline = np.asarray(baseline, np.int32)
        other = np.asarray(other, np.int32)
        result = baseline == other
        count = len(self._edges)
        adjacent = (((baseline+1) % count == other)
                    | ((other+1) % count == baseline))
        check = np.flatnonzero(~result & adjacent & self._allowed[other])
        if len(check):
            first, second = self._edges[baseline[check]], self._edges[other[check]]
            cross = first[:, 0]*second[:, 1]-first[:, 1]*second[:, 0]
            lengths = np.linalg.norm(first, axis=1)*np.linalg.norm(second, axis=1)
            aligned = ((first*second).sum(axis=1) > 0)
            aligned &= (lengths > 1e-10) & (np.abs(cross) <= 1e-8*lengths)
            result[check] = aligned
        return result

    def _raw_jacobians(self, states):
        """Piecewise analytic zero-Brown source J; guarded numerical fallback.

        The analytic interior formula is the plane-hit differentiation used by
        SurfacePairConstraint, but the current panel is first selected by the
        full board ring.  Brown source rays and points effectively on a seam
        use the ReferenceBoardPair central difference on only those rows.
        A positive interior panel fraction has a valid local analytic J even
        when a finite-difference step would cross a nearby seam.  Target J
        always comes from the independent target Brown projection.
        """
        source, target = (np.asarray(q, np.float64) for q in states)
        _, base_panel, target_jac = self.projected([source, target], True)
        hits = self.source_hits(source)
        self._require(hits)
        count = len(self.xy0)
        source_jac = np.empty((count, 2, 11), np.float64)
        camera = from_state(source, self.camera0_template)
        smooth = (np.max(np.abs(camera[11:25])) <= 1e-12)
        analytic = np.zeros(count, bool)
        if smooth:
            fraction = hits.panel_fraction
            analytic = (np.isfinite(fraction)
                        & (fraction > self.seam_fraction_margin)
                        & (fraction < 1.-self.seam_fraction_margin))
            # A straight side may be split into many scene polygon panels.
            # Crossing such a vertex changes panel ID but not the hit plane.
            at_start = np.isfinite(fraction) & (fraction <= self.seam_fraction_margin)
            at_end = np.isfinite(fraction) & (fraction >= 1.-self.seam_fraction_margin)
            previous = (hits.panel-1) % len(self._edges)
            following = (hits.panel+1) % len(self._edges)
            analytic |= at_start & self._same_local_plane(hits.panel, previous)
            analytic |= at_end & self._same_local_plane(hits.panel, following)
        if np.any(analytic):
            indices = np.flatnonzero(analytic)
            fx, fy = np.exp(camera[6:8])
            qy = (self.xy0[indices, 1]-camera[9])/fy
            qx = (self.xy0[indices, 0]-camera[8]-camera[10]*qy)/fx
            rays_camera = np.column_stack((qx, qy, np.ones(len(indices))))
            rotation, rod_jac = cv2.Rodrigues(np.ascontiguousarray(camera[3:6]))
            if rod_jac.shape == (3, 9):
                drot = rod_jac.reshape(3, 3, 3)
            elif rod_jac.shape == (9, 3):
                drot = rod_jac.T.reshape(3, 3, 3)
            else:
                raise ValueError(f"Unexpected Rodrigues Jacobian shape {rod_jac.shape}")
            rays = rays_camera @ rotation
            edge = self._edges[hits.panel[indices]]
            plane_point = self.bottom[hits.panel[indices]]
            normal = np.column_stack((edge[:, 1], -edge[:, 0], np.zeros(len(edge))))
            denominator = np.einsum("ni,ni->n", rays, normal)
            stable = np.isfinite(denominator) & (np.abs(denominator) > 1e-7)
            if not np.all(stable):
                analytic[indices[~stable]] = False
                indices = indices[stable]
                rays_camera, rays, normal, plane_point = (item[stable] for item in
                    (rays_camera, rays, normal, plane_point))
                qx, qy, denominator = qx[stable], qy[stable], denominator[stable]
            if len(indices):
                distance = np.einsum("ni,ni->n", plane_point-camera[:3], normal)/denominator
                ray_jac = np.zeros((len(indices), 3, 11), np.float64)
                for axis in range(3):
                    ray_jac[:, :, 3+axis] = rays_camera @ drot[axis]
                local = np.zeros((len(indices), 3, 5), np.float64)
                local[:, 0, 0] = -qx
                local[:, 0, 1] = camera[10]*qy/fx
                local[:, 1, 1] = -qy
                local[:, 0, 2] = -1000./fx
                local[:, 0, 3] = 1000.*camera[10]/(fx*fy)
                local[:, 1, 3] = -1000./fy
                local[:, 0, 4] = -1000.*qy/fx
                ray_jac[:, :, 6:] = np.einsum("nap,aj->njp", local, rotation)
                centre_jac = np.zeros((len(indices), 3, 11), np.float64)
                centre_jac[:, :, :3] = 100.*np.eye(3)
                d_num = -np.einsum("na,nap->np", normal, centre_jac)
                d_den = np.einsum("na,nap->np", normal, ray_jac)
                d_distance = (d_num-distance[:, None]*d_den)/denominator[:, None]
                world_jac = (centre_jac + distance[:, None, None]*ray_jac
                             + rays[:, :, None]*d_distance[:, None, :])
                image_world = -target_jac.reshape(count, 2, 11)[indices, :, :3]/100.
                source_jac[indices] = np.einsum("nai,nij->naj", image_world, world_jac)
        numeric = np.flatnonzero(~analytic)
        shrunk = 0
        if len(numeric):
            pixels = self.xy0[numeric]
            for axis in range(11):
                step = 1e-6
                high, low = source.copy(), source.copy()
                high[axis] += step
                low[axis] -= step
                try:
                    high_uv, high_panel = self._project_pixels([high, target], pixels)
                    low_uv, low_panel = self._project_pixels([low, target], pixels)
                except BoardIntersectionError as exc:
                    # A single near-seam row must not force all otherwise
                    # stable Brown rows onto a tiny, noisy difference step.
                    retry = np.ones(len(numeric), bool)
                    retry_reason = exc.reason
                else:
                    retry = (~self._same_local_plane(base_panel[numeric], high_panel)
                             | ~self._same_local_plane(base_panel[numeric], low_panel))
                    retry_reason = "nondifferentiable_panel_transition"
                    source_jac[numeric[~retry], :, axis] = (
                        (high_uv[~retry]-low_uv[~retry])/(2*step))
                for original in numeric[retry]:
                    # The source point may be on one smooth panel even though
                    # 1e-6 state units crosses its neighboring polygon chord.
                    # Shrink both sides until they remain on the base branch.
                    # An exact corner or truly invalid hit still raises.
                    for divisor in (4., 16., 64., 256.):
                        smaller = step/divisor
                        high, low = source.copy(), source.copy()
                        high[axis] += smaller
                        low[axis] -= smaller
                        try:
                            hi_uv, hi_panel = self._project_pixels(
                                [high, target], self.xy0[original:original+1])
                            lo_uv, lo_panel = self._project_pixels(
                                [low, target], self.xy0[original:original+1])
                        except BoardIntersectionError as exc:
                            retry_reason = exc.reason
                            continue
                        if (self._same_local_plane(
                                base_panel[original:original+1], hi_panel)[0]
                                and self._same_local_plane(
                                    base_panel[original:original+1], lo_panel)[0]):
                            source_jac[original, :, axis] = ((hi_uv-lo_uv)/(2*smaller))[0]
                            shrunk += 1
                            break
                    else:
                        raise BoardIntersectionError(retry_reason, [original])
        self.last_jacobian_counts = {"analytic_rows": int(np.sum(analytic)),
                                     "numeric_rows": int(len(numeric)),
                                     "numeric_shrunk_axis_rows": int(shrunk)}
        result = source_jac.reshape(2*count, 11)
        if not np.isfinite(result).all() or not np.isfinite(target_jac).all():
            raise ValueError("Native board pair Jacobian is nonfinite")
        return [result, target_jac]
