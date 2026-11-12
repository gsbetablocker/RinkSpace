"""The experiment's five-node adapter to the registered native Brown camera."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from inputs import PROJECT, Inputs, atomic_json, emit, geometry
from calibration.program_tracking.body_tracking.programme_camera import ProgrammeCamera
from calibration.program_tracking.body_tracking.rebuild_exposure_body import moving_projection


def exp(rotvec):
    return Rotation.from_rotvec(rotvec.reshape(-1, 3)).as_matrix().reshape(*rotvec.shape[:-1], 3, 3)


def log(matrix):
    return Rotation.from_matrix(matrix.reshape(-1, 3, 3)).as_rotvec().reshape(*matrix.shape[:-2], 3)


def inverse(matrix):
    return np.swapaxes(matrix, -1, -2)


def slerp(a, b, u):
    return a @ exp(log(inverse(a) @ b) * u[..., None])


class PoseCurve:
    """Hermite translation and local SQUAD, with state continuity across camera cuts.

    Local interpolation follows the handoff's native_projector.py. Time
    derivatives use the original seconds, including the exposure halo.
    """

    def __init__(self, times, origin, rotation):
        self.t = times
        self.p = origin
        self.r = rotation
        step = np.diff(times)
        if len(times) < 3 or np.any(step <= 0) or not np.allclose(step, step[0], atol=1e-10, rtol=0):
            raise ValueError("Require the verified uniform native programme PTS")
        self.h = step[0]
        self.velocity = np.gradient(origin, self.h, axis=0)
        self.edge = log(inverse(rotation[:-1]) @ rotation[1:])
        if np.any(np.linalg.norm(self.edge, axis=-1) >= np.pi - 1e-5):
            raise ValueError("Unresolved adjacent half-turn branch")
        self.tangent = rotation.copy()
        self.tangent[1:-1] = rotation[1:-1] @ exp(.25 * (self.edge[:-1] - self.edge[1:]))

    def at(self, query):
        query = np.asarray(query)
        k = np.clip(np.searchsorted(self.t, query, side="right") - 1, 0, len(self.t) - 2)
        u = (query - self.t[k]) / self.h
        w = u[..., None, None]
        pos = ((2 * w**3 - 3 * w**2 + 1) * self.p[k] + (w**3 - 2 * w**2 + w) * self.h * self.velocity[k]
               + (-2 * w**3 + 3 * w**2) * self.p[k + 1] + (w**3 - w**2) * self.h * self.velocity[k + 1])
        a = slerp(self.r[k], self.r[k + 1], u[..., None])
        b = slerp(self.tangent[k], self.tangent[k + 1], u[..., None])
        rot = slerp(a, b, (2 * u * (1 - u))[..., None])
        for use, endpoint, edge in ((query < self.t[0], 0, 0),
                                    (query > self.t[-1], -1, -1)):
            dt = query[use] - self.t[endpoint]
            pos[use] = self.p[endpoint] + dt[..., None, None] * self.velocity[endpoint]
            rot[use] = self.r[endpoint] @ exp(dt[..., None, None] * self.edge[edge] / self.h)
        return pos, rot


class Native:
    def __init__(self, inputs):
        self.camera = ProgrammeCamera(PROJECT, inputs.camera_packet, inputs.shots,
                                      record_path=inputs.camera_record)
        if self.camera.info["state_sha256"] != inputs.camera_sha:
            raise ValueError("Incorrect frozen camera")
        self.camera.core.motion_projection_threads = 8
        for camera in self.camera.cameras.values():
            camera.core.motion_projection_threads = 8
        self.time = inputs.times
        self.ratio = float(self.camera.info["length_export_ratio"])
        self.tau = self.camera.readout_times(np.arange(len(self.time)))
        if np.max(abs(self.tau)) * .5 >= np.diff(self.time).min():
            raise ValueError("Five-node exposure exceeds the local trajectory support")
        self.centres = self.camera.midpoint["centre"] * self.camera.info["metres_per_scene_unit"]

    @staticmethod
    def world(pos, rot, local):
        return pos[..., None, :] + np.einsum("...fij,fkj->...fki", rot, local)

    def frozen(self, pos, rot, local, frames, *, global_shutter=False):
        xyz = self.world(pos, rot, local)
        n, f, k, _ = xyz.shape
        if not global_shutter:
            return self.camera.project(xyz.reshape(n, f * k, 3) / self.ratio, frames).reshape(n, f, k, 2)
        uv = np.empty((n, f * k, 2), float)
        for ix, camera, local_frame in self.camera._groups(frames):
            p = camera.p
            rig = p["rig"].copy()
            rig[37] = 0.
            projected = camera.core.project_motion(
                rig, p["controls"], float(p["first_control_time"]), float(p["control_dt"]),
                camera.image_height, np.repeat(p["motion_times"][local_frame], f * k),
                xyz[ix].reshape(-1, 3) / self.ratio / camera.scale)
            if np.any(projected["status"]):
                raise ValueError("Native midpoint projection domain failure")
            uv[ix] = projected["uv"].reshape(len(ix), f * k, 2)
        return uv.reshape(n, f, k, 2)

    def project(self, curve, local, frames):
        query = self.time[frames, None] + self.tau[frames, None] * np.linspace(-.5, .5, 5)
        pos, rot = curve.at(query)
        xyz = self.world(pos, rot, local)
        n, _, f, k, _ = xyz.shape
        answer = moving_projection(self.camera, xyz.reshape(n, 5, f * k, 3) / self.ratio, frames)
        return answer["uv"].reshape(n, f, k, 2)

    def exact_rows(self, curve, local, frames, initial):
        """Independent loop evaluating the local curve at the implicit native row."""
        uv = initial.copy()
        n, f, k, _ = uv.shape
        height = self.camera.image_height
        feet = np.arange(f)[None, :, None]
        for _ in range(32):
            t = self.time[frames, None, None] + self.tau[frames, None, None] * (
                np.clip(uv[..., 1], 0, height - 1) / (height - 1) - .5)
            p, r = curve.at(t)
            # PoseCurve evaluates both physical feet at each query. Select the
            # queried foot and its local point before sending the native call.
            selected_p = np.take_along_axis(p, feet[..., None, None], axis=-2)[..., 0, :]
            selected_r = np.take_along_axis(r, feet[..., None, None, None], axis=-3)[..., 0, :, :]
            xyz = selected_p + np.einsum("nfkij,fkj->nfki", selected_r, local)
            new = self.camera.project(xyz.reshape(n, f * k, 3) / self.ratio, frames).reshape(n, f, k, 2)
            if np.max(np.abs(new - uv)) < 1e-9:
                return new
            uv = new
        raise ValueError("Exact-row reference did not close")


def project_available(native, curve, local, frames):
    """Isolate any failed native row domains without dropping source exposures."""
    try:
        return native.project(curve, local, frames)
    except ValueError:
        if len(frames) == 1:
            return np.full((1, *local.shape[:2], 2), np.nan)
        midpoint = len(frames) // 2
        return np.concatenate((project_available(native, curve, local, frames[:midpoint]),
                               project_available(native, curve, local, frames[midpoint:])))


def stats(delta):
    d = np.linalg.norm(delta, axis=-1).ravel()
    return dict(p50=float(np.median(d)), p95=float(np.percentile(d, 95)), max=float(d.max()))


def closure(output):
    inputs = Inputs()
    shape, p, r = geometry(inputs)
    native = Native(inputs)
    np.testing.assert_allclose(shape["ratio"], native.ratio, rtol=1e-10)
    # Fixed material points on the actual two steel edges. Visible silhouettes
    # are never treated as if their arbitrary sample positions were materials.
    local = shape["edges"][:, :, ::10].reshape(2, -1, 3)
    cuts = np.flatnonzero(inputs.shots[1:] != inputs.shots[:-1]) + 1
    frames = np.unique(np.r_[np.linspace(0, len(p) - 1, 96).astype(int), cuts - 1, cuts])
    curve = PoseCurve(inputs.times, p, r)
    front = native.project(curve, local, frames)
    frozen = native.frozen(p[frames], r[frames], local, frames)
    reference = native.exact_rows(curve, local, frames, frozen)
    global_shutter = native.frozen(p[frames], r[frames], local, frames, global_shutter=True)
    constant_world = native.world(p[frames], r[frames], local).reshape(len(frames), -1, 3) / native.ratio
    static_nodes = np.repeat(constant_world[:, None], 5, axis=1)
    adapter_frozen = moving_projection(native.camera, static_nodes, frames)["uv"].reshape(frozen.shape)
    np.testing.assert_allclose(adapter_frozen, frozen, atol=2e-8, rtol=0)
    row_error = np.linalg.norm(front - reference, axis=-1)
    inside = ((reference >= [0, 0]) & (reference < [1920, 1080])).all(axis=-1)
    worst = np.unravel_index(row_error.argmax(), row_error.shape)
    midp, midr = curve.at(inputs.times)
    np.testing.assert_allclose(midp, p, atol=1e-11, rtol=0)
    np.testing.assert_allclose(midr, r, atol=1e-11, rtol=0)
    record = dict(camera_sha256=inputs.camera_sha, template_sha256=str(shape["template_sha256"]),
                  dll=native.camera.info["native_library"], dll_sha256=native.camera.info["native_library_sha256"],
                  source_pts=inputs.pts[frames].tolist(), exposures=len(frames), projections=int(front[..., 0].size),
                  fixed_node_vs_direct_native_px=stats(adapter_frozen - frozen),
                  five_node_vs_exact_native_row_px=stats(front - reference),
                  inside_image_projections=int(inside.sum()),
                  inside_five_node_vs_exact_px=stats((front - reference)[inside]),
                  components_exceeding_0p05px=int(np.sum(np.abs(front - reference) > .05)),
                  worst_source_pts=int(inputs.pts[frames[worst[0]]]),
                  object_motion_px=stats(front - frozen),
                  camera_rolling_vs_midpoint_px=stats(frozen - global_shutter),
                  readout_ms=(1e3 * np.unique(native.tau)).tolist())
    atomic_json(output, record)
    emit("native_closure", **{k: v for k, v in record.items() if k != "source_pts"})


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--closure", type=Path, required=True)
    closure(parser.parse_args().closure)
