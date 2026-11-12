"""Image-side tests for a boot and the runner mounted to that boot."""

from typing import NamedTuple

import cv2
import numpy as np

from inputs import distribute


def unit(vector):
    return vector / max(np.linalg.norm(vector), 1e-6)


def silhouette(points, faces):
    """Rasterize the boot mesh union at a common oversampled scale."""
    extent = np.ptp(points, axis=0).max()
    xy = (points - points.min(axis=0)) * (120. / extent) + 3.
    raster = np.zeros((128, 128), np.uint8)
    fixed = np.rint(xy * 16).astype(np.int32)
    for face in faces:
        cv2.fillConvexPoly(raster, fixed[face], 1, shift=4)
    contours, _ = cv2.findContours(raster, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    arc = max(contours, key=cv2.contourArea)[:, 0] - 3.
    return distribute(arc * (extent / 120.) + points.min(axis=0), 40, closed=True)


def normalized(points):
    centre = points.mean(axis=-2)
    delta = points - centre[..., None, :]
    radius = np.sqrt(np.mean(np.sum(delta**2, axis=-1), axis=-1))
    return delta / np.maximum(radius[..., None, None], 1e-6), centre, radius


class AttachmentFit(NamedTuple):
    lower: np.ndarray
    upper: np.ndarray
    visible: np.ndarray
    clearance: np.ndarray
    quality: np.ndarray
    radius: float


def exposed_web(model, foot, camera, rails):
    """Find lower-rail stations that can be a visible steel silhouette."""
    geometry = model.geometry
    boot = model.boot[foot] @ camera
    lower = geometry["lower"][foot] @ camera
    points = np.vstack((boot, lower))
    origin = points.min(axis=0)
    factor = 210. / np.ptp(points, axis=0).max()
    xy = np.rint(((points - origin) * factor + 4.) * 16.).astype(np.int32)
    other = np.zeros((220, 220), np.uint8)
    steel = np.zeros_like(other)
    for face in model.faces:
        cv2.fillConvexPoly(other, xy[face], 1, shift=4)
    for face, part in zip(geometry["lower_faces"], geometry["parts"], strict=True):
        cv2.fillConvexPoly(steel if part == "steel_runner" else other,
                          xy[face + len(boot)], 1, shift=4)
    lower, upper = rails
    tangent = np.gradient(lower, axis=1)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=-1, keepdims=True), 1e-9)
    web = upper - lower
    web -= (web * tangent).sum(axis=-1, keepdims=True) * tangent

    def at(image, coords):
        uv = np.rint((coords - origin) * factor + 4.).astype(int)
        uv = np.clip(uv, 0, image.shape[0] - 1)
        return image[uv[..., 1], uv[..., 0]]

    outside = lower - .27 * web
    middle = lower + .55 * web
    lip = lower + .90 * web
    visible = ((at(steel, outside) == 0) & (at(steel, middle) > 0)
               & (at(other, lower) == 0) & (at(other, middle) == 0) & (at(other, lip) == 0))
    clearance = cv2.distanceTransform(1 - other, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return visible, at(clearance, lower) / factor


class BootAttachment:
    """Boot silhouettes and steel from identical views of the saved assembly.

    Similarity alignment in the original pixel plane is a search constraint.
    Depth, blade tilt and visibility remain unresolved at this stage.
    """

    def __init__(self, model):
        self.atlas = []
        azimuths = np.arange(12) * np.pi / 6
        elevations = np.deg2rad([-72, -48, -24, 0, 24, 48, 72])
        rolls = np.deg2rad(np.arange(-180, 180, 20))
        angle = np.arange(16) * np.pi / 8
        self.directions = np.stack((np.cos(angle), np.sin(angle)), axis=-1)
        for f in range(2):
            shell, rails, cuff, roll_cost, visible, clearance = [], [], [], [], [], []
            upper = model.boot[f, :, 2]
            collar = model.boot[f, upper >= np.quantile(upper, .88)].mean(axis=0)
            for elevation in elevations:
                for azimuth in azimuths:
                    ray = np.array([np.cos(elevation) * np.cos(azimuth),
                                    np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])
                    down = -unit(np.array([0., 0., 1.]) - ray[2] * ray)
                    camera = np.stack((np.cross(down, ray), down), axis=-1)
                    outline = silhouette(model.boot[f] @ camera, model.faces)
                    runner = model.runner[f] @ camera
                    exposed, gap = exposed_web(model, f, camera, runner)
                    projected_collar = collar @ camera
                    norm, centre, radius = normalized(outline)
                    for roll in rolls:
                        c, s = np.cos(roll), np.sin(roll)
                        transform = np.array([[c, s], [-s, c]])
                        shell.append(norm @ transform)
                        rails.append(((runner - centre) / radius) @ transform)
                        cuff.append(((projected_collar - centre) / radius) @ transform)
                        roll_cost.append(abs(roll))
                        visible.append(exposed)
                        clearance.append(gap / radius)
            shell, rails, cuff, roll_cost = map(np.asarray, (shell, rails, cuff, roll_cost))
            support = (shell @ self.directions.T).max(axis=1)
            self.atlas.append((shell, rails, cuff, roll_cost, support, np.asarray(visible), np.asarray(clearance)))

    def align(self, f, polygon, ankle=None, shin=None):
        """Keep compatible runner projections based solely on boot evidence."""
        up = unit(shin) if shin is not None else None
        basis = np.stack(([-up[1], up[0]], -up), axis=-1) if up is not None else np.eye(2)
        goal, centre, radius = normalized(polygon[::2] @ basis)
        if radius < 3.:
            return None
        shell, rails, cuff, roll, support, visible, clearance = self.atlas[f]
        observed_support = (goal @ self.directions.T).max(axis=0)
        support_cost = np.mean((support - observed_support)**2, axis=1)
        coarse = support_cost
        # The image ankle is near the cuff/shank but its exact position under
        # the boot is unknown. Do not force the model onto that pixel.
        if up is not None:
            ankle_local = ((ankle @ basis) - centre) / radius
            cuff_gap = np.linalg.norm(cuff - ankle_local, axis=-1)
            coarse = support_cost + .013 * roll**2 + .045 * np.maximum(cuff_gap - .65, 0.)**2
            coarse[roll > np.deg2rad(61)] = np.inf
        pick = np.argpartition(coarse, 31)[:32]
        distance = np.linalg.norm(shell[pick, :, None] - goal[None, None], axis=-1)
        forward = np.min(distance, axis=2)
        reverse = np.min(distance, axis=1)
        quality = (.5 * (np.median(forward, axis=1) + np.median(reverse, axis=1))
                   + .16 * np.quantile(reverse, .8, axis=1) + .32 * coarse[pick])
        good = np.argsort(quality)
        good = good[quality[good] <= min(.26, quality[good[0]] + .065)][:12]
        if not len(good):
            return None
        ix = pick[good]
        projected = (rails[ix] * radius + centre) @ basis.T
        return AttachmentFit(projected[:, 0], projected[:, 1], visible[ix], clearance[ix] * radius, quality[good], radius)


def cutting_profiles(samples, fitted):
    """Bind source pixels to a lower rail and the inward steel web direction."""
    start = fitted.lower[:, :, :-1]
    span = fitted.lower[:, :, 1:] - start
    delta = samples[None, None, :, None, :] - start[:, :, None]
    t = np.clip((delta * span[:, :, None]).sum(-1) /
                np.maximum((span**2).sum(-1)[:, :, None], 1e-6), 0., 1.)
    errors = np.linalg.norm(delta - span[:, :, None] * t[..., None], axis=-1)
    seg = errors.argmin(axis=-1)
    v, side = np.indices(seg.shape[:2])
    v, side = v[..., None], side[..., None]
    pos = t[v, side, np.arange(len(samples)), seg]
    error = errors[v, side, np.arange(len(samples)), seg]
    web = fitted.upper - fitted.lower
    web = web[v, side, seg] * (1.-pos[..., None]) + web[v, side, seg+1] * pos[..., None]
    tangent = span[v, side, seg]
    tangent /= np.maximum(np.linalg.norm(tangent, axis=-1, keepdims=True), 1e-6)
    web -= (web * tangent).sum(axis=-1, keepdims=True) * tangent
    width = np.linalg.norm(web, axis=-1)
    normal = web / np.maximum(width[..., None], 1e-6)
    clearance = np.minimum(fitted.clearance[v, side, seg], fitted.clearance[v, side, seg+1])
    exposed = fitted.visible[v, side, seg] & fitted.visible[v, side, seg+1]
    # A projected rail closer than the uncertainty to the upper lip, holder
    # or sole is unresolved, even if there is a strong edge at that pixel.
    tolerance = max(1.8, .10 * fitted.radius)
    along = np.abs(tangent @ unit(samples[-1] - samples[0]))
    supported = (exposed & (width >= 1.8) & (along >= .83) & (error <= tolerance)
                 & (error + .5 < clearance) & (error <= .85 * width))
    return supported, normal, width, error
