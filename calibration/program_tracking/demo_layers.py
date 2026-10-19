"""Reusable presentation layers for native-clock rink demo renders.

IceLayers returns straight-alpha BGRA in the model's unchanged x/y axes.
The longest texture dimension is texture_width; row zero is maximum world y,
column zero is minimum world x. bounds_xy uses scene units, while input
trajectory positions and all width/radius options use metres. This module
does not alter camera, body, source timing or measurement artifacts.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class IceLayers:
    """Growing coverage and trajectory texture, clipped to the actual ice.

    texture(index) includes trajectory rows 0..index. Calls in increasing order
    increment binary coverage and the old trail rather than rerasterizing the
    complete programme. Repeated indices are exact cache hits. A backwards
    request safely rebuilds the prefix. Invalid positions split visible paths.

    Coverage is a union of the route's radius-metre neighbourhood followed by
    a fixed Gaussian feather. Repeating a stationary position adds no weight.
    There are no markings, grids, labels or statistics in this texture.
    """

    def __init__(
        self, scene, unit, xy_m, times_s, texture_width=1536,
        fade_seconds=2.5, old_alpha=.22, line_width_m=.09,
        coverage_radius_m=1., sigma_m=.7,
        particle_style="markers", heat_palette="amber", heat_update_anchor=None,
    ):
        self.unit = float(unit)
        self.fade_seconds = float(fade_seconds)
        self.old_alpha = float(old_alpha)
        self.line_width_m = float(line_width_m)
        self.coverage_radius_m = float(coverage_radius_m)
        self.sigma_m = float(sigma_m)
        self.particle_style = str(particle_style)
        self.heat_palette = str(heat_palette)
        if self.particle_style not in {"markers", "particles"}:
            raise ValueError("particle_style must be markers or particles")
        if self.heat_palette not in {"amber", "deep_red", "hot_red", "champagne_gold"}:
            raise ValueError("heat_palette must be amber, deep_red, hot_red or champagne_gold")
        if heat_update_anchor is not None and int(heat_update_anchor) != heat_update_anchor:
            raise ValueError("heat_update_anchor must be an integer source-row anchor")
        self.heat_update_anchor = (None if heat_update_anchor is None
                                   else int(heat_update_anchor))
        if not (math.isfinite(self.unit) and self.unit > 0):
            raise ValueError("unit must be positive metres per scene unit")
        if not (math.isfinite(self.fade_seconds) and self.fade_seconds > 0):
            raise ValueError("fade_seconds must be finite and positive")
        if not 0 <= self.old_alpha < .96:
            raise ValueError("old_alpha must lie in [0,.96)")
        if not all(math.isfinite(v) and v > 0 for v in
                   (self.line_width_m, self.coverage_radius_m, self.sigma_m)):
            raise ValueError("line width, coverage radius and sigma must be positive")
        longest = int(texture_width)
        if longest < 64:
            raise ValueError("texture_width must be at least 64 pixels")
        boundary = np.asarray(scene["ice_boundary"]["points_xyz"], dtype=float)
        if (boundary.ndim != 2 or boundary.shape[0] < 3
                or boundary.shape[1] < 2 or not np.isfinite(boundary).all()):
            raise ValueError("Expected a finite ice-boundary polygon in scene units")
        self.boundary_xy = boundary[:, :2].copy()
        xy = np.asarray(xy_m, dtype=float)
        self.times_s = np.asarray(times_s, dtype=float).copy()
        if (xy.ndim != 2 or xy.shape[1] < 2 or len(xy) == 0
                or self.times_s.shape != (len(xy),)
                or not np.isfinite(self.times_s).all()
                or np.any(np.diff(self.times_s) <= 0)):
            raise ValueError("xy_m and strictly increasing times_s must align")
        self.xy_m = xy[:, :2].copy()
        self.valid = np.isfinite(self.xy_m).all(axis=1)
        padding = 1.5
        low = self.boundary_xy.min(axis=0) - padding
        high = self.boundary_xy.max(axis=0) + padding
        span = high-low
        if np.any(span <= 0):
            raise ValueError("Ice bounds must have positive x and y extent")
        scale = (longest-1)/float(span.max())
        width = max(2, int(round(span[0]*scale))+1)
        height = max(2, int(round(span[1]*scale))+1)
        self.bounds_xy = (float(low[0]), float(low[1]),
                          float(high[0]), float(high[1]))
        self.texture_wh = (width, height)
        self._pixels_per_scene_unit = np.array(
            [(width-1)/span[0], (height-1)/span[1]], dtype=float)
        self._pixels_per_metre = float(
            self._pixels_per_scene_unit.mean()/self.unit)
        self._shape = (height, width)
        self._uv = self._map_xy(self.xy_m/self.unit)
        self._boundary_mask = np.zeros(self._shape, np.uint8)
        boundary_uv = self._map_xy(self.boundary_xy)
        cv2.fillPoly(self._boundary_mask,
                     [np.rint(boundary_uv*16).astype(np.int32)],
                     255, cv2.LINE_AA, shift=4)
        self._clip = self._boundary_mask.astype(np.float32)/255.
        self._line_thickness = max(
            1, int(round(self.line_width_m*self._pixels_per_metre)))
        self._coverage_radius_px = max(
            1, int(round(self.coverage_radius_m*self._pixels_per_metre)))
        self._coverage_thickness = 2*self._coverage_radius_px+1
        self._dot_radius_px = max(2, int(round(.075*self._pixels_per_metre)))
        self._blue = np.array([244., 132., 4.], np.float32)
        self._cold = np.array([205., 143., 54.], np.float32)
        self._cold_alpha = .085
        self._warm_light = np.array([45., 202., 255.], np.float32)
        self._warm_deep = np.array([12., 128., 249.], np.float32)
        self._warm_alpha = .22
        if self.heat_palette == "deep_red":
            self._warm_light = np.array([35., 65., 241.], np.float32)
            self._warm_deep = np.array([32., 22., 175.], np.float32)
            self._warm_alpha = .32
        if self.heat_palette == "hot_red":
            self._warm_light = np.array([20., 28., 225.], np.float32)
            self._warm_deep = np.array([16., 6., 135.], np.float32)
            self._warm_alpha = .58
        if self.heat_palette == "champagne_gold":
            self._cold = np.array([220., 176., 67.], np.float32)
            self._cold_alpha = .055
            self._warm_light = np.array([150., 220., 255.], np.float32)
            self._warm_deep = np.array([65., 175., 250.], np.float32)
            self._warm_alpha = .55
        self._dot_colour = np.array([255., 239., 195.], np.float32)
        # Fixed source-time marks, rather than frame-count marks.
        self._dot_times = np.arange(
            self.times_s[0], self.times_s[-1]+1e-9, .25, dtype=float)
        right = np.searchsorted(self.times_s, self._dot_times, side="left")
        right = np.clip(right, 0, len(self.times_s)-1)
        left = np.maximum(right-1, 0)
        dt = self.times_s[right]-self.times_s[left]
        fraction = np.divide(self._dot_times-self.times_s[left], dt,
                             out=np.zeros_like(dt), where=dt > 0)
        self._dot_uv = (self._uv[left]*(1-fraction[:, None])
                        + self._uv[right]*fraction[:, None])
        self._dot_valid = self.valid[left] & self.valid[right]
        if self.particle_style == "particles":
            self._build_particles()
        self._reset()

    def _build_particles(self):
        """Seed decorative plane sparks from an immutable source-time sequence.

        These are presentation decoration, not measured contact or speed data.
        Every birth has fixed irregular timing, world offset, size and lifetime.
        Low displacement thins emission instead of collecting a bright disc.
        """
        random = np.random.default_rng(20261016).random((
            max(1, int(math.ceil(
                (self.times_s[-1]-self.times_s[0])/.012))+2), 10))
        births = self.times_s[0]+np.cumsum(.012+.015*random[:, 0])
        available = births <= self.times_s[-1]
        births, random = births[available], random[available]
        right = np.searchsorted(self.times_s, births, side="left")
        right = np.clip(right, 0, len(self.times_s)-1)
        left = np.maximum(right-1, 0)
        dt = self.times_s[right]-self.times_s[left]
        fraction = np.divide(births-self.times_s[left], dt,
                             out=np.zeros_like(dt), where=dt > 0)
        delta = self.xy_m[right]-self.xy_m[left]
        distance = np.linalg.norm(delta, axis=1)
        direction = np.divide(delta, distance[:, None],
                              out=np.zeros_like(delta),
                              where=distance[:, None] > 1e-8)
        speed = np.divide(distance, dt, out=np.zeros_like(dt), where=dt > 0)
        probability = np.clip(speed/1.5, 0., 1.)
        keep = (self.valid[left] & self.valid[right]
                & (speed > .15) & (random[:, 9] < probability))
        normal = np.column_stack((-direction[:, 1], direction[:, 0]))
        # Narrow irregular scatter around the actual recent route; no screen
        # spraying, no 3D height, and no accumulating particles while stationary.
        across = (random[:, 1]-.5)*.50
        along = (random[:, 2]-.5)*.16
        position = (self.xy_m[left]*(1-fraction[:, None])
                    + self.xy_m[right]*fraction[:, None]
                    + normal*across[:, None]+direction*along[:, None])
        self._particle_times = births[keep]
        self._particle_uv = self._map_xy(position[keep]/self.unit)
        self._particle_lifetimes = .9+.6*random[keep, 3]
        self._particle_gains = .70+.28*random[keep, 4]
        self._particle_radii = np.maximum(4, np.rint(
            (.03+.045*random[keep, 5])*self._pixels_per_metre*16)).astype(int)
        self._particle_angles = 2*math.pi*random[keep, 6]
        self._particle_lengths = (
            .035+.07*random[keep, 7])*self._pixels_per_metre
        self._particle_sliver = random[keep, 8] < .35

    def _particle_mask(self, now):
        raster = np.zeros(self._shape, np.uint8)
        begin = int(np.searchsorted(
            self._particle_times, now-1.8, side="left"))
        end = int(np.searchsorted(self._particle_times, now, side="right"))
        for particle in range(begin, end):
            age = now-self._particle_times[particle]
            remaining = 1-age/self._particle_lifetimes[particle]
            if remaining <= 0:
                continue
            opacity = self._particle_gains[particle]*remaining**1.2
            value = round(float(opacity)*255)
            if value <= 0:
                continue
            point = self._particle_uv[particle]
            if self._particle_sliver[particle]:
                angle = self._particle_angles[particle]
                offset = .5*self._particle_lengths[particle]*np.array([
                    math.cos(angle), math.sin(angle)])
                cv2.line(raster, self._fixed_point(point-offset),
                         self._fixed_point(point+offset), value,
                         1, cv2.LINE_AA, shift=4)
            else:
                cv2.circle(raster, self._fixed_point(point),
                           int(self._particle_radii[particle]), value,
                           -1, cv2.LINE_AA, shift=4)
        return raster

    def _map_xy(self, xy):
        xmin, ymin, xmax, ymax = self.bounds_xy
        xy = np.asarray(xy, float)
        return np.column_stack((
            (xy[:, 0]-xmin)*self._pixels_per_scene_unit[0],
            (ymax-xy[:, 1])*self._pixels_per_scene_unit[1],
        ))

    def _reset(self):
        self._coverage = np.zeros(self._shape, np.uint8)
        self._old_line = np.zeros(self._shape, np.uint8)
        self._coverage_index = -1
        self._old_index = -1
        self._heat_index = -1
        self._heat_premult = None
        self._heat_alpha = None
        self._last_index = -1
        self._last_texture = None
        self._last_summary = None
        self._last_premult = None
        self._last_alpha = None
        self._last_prepared = None

    @staticmethod
    def _fixed_point(uv):
        return tuple(np.rint(uv*16).astype(np.int32).tolist())

    def _stamp_prefix(self, target, start, end, thickness, value):
        """Add every finite route edge, plus isolated valid points, once."""
        for row in range(start, end+1):
            if not self.valid[row]:
                continue
            point = self._fixed_point(self._uv[row])
            if row and self.valid[row-1]:
                cv2.line(target, self._fixed_point(self._uv[row-1]), point,
                         int(value), thickness, cv2.LINE_AA, shift=4)
            else:
                cv2.circle(target, point, max(8, thickness*8),
                           int(value), -1, cv2.LINE_AA, shift=4)

    def _refresh_heat(self, index):
        # Binary union: line anti-aliasing does not accumulate dwell time.
        occupancy = (self._coverage > 0).astype(np.float32)
        sigma = self.sigma_m*self._pixels_per_metre
        field = cv2.GaussianBlur(occupancy, (0, 0), sigmaX=sigma,
                                 sigmaY=sigma, borderType=cv2.BORDER_CONSTANT)
        field = np.clip(field, 0., 1.)
        warm_alpha = (self._warm_alpha*field)[:, :, None]
        cold_alpha = self._cold_alpha
        warm = (self._warm_light[None, None, :]*(1-field[:, :, None])
                + self._warm_deep[None, None, :]*field[:, :, None])
        self._heat_alpha = cold_alpha*(1-warm_alpha)+warm_alpha
        self._heat_premult = (
            self._cold[None, None, :]*cold_alpha*(1-warm_alpha)
            + warm*warm_alpha)
        self._heat_index = index

    def _trail_opacity(self, age_seconds):
        age = np.asarray(age_seconds, float)
        remaining = 1-np.clip(age/self.fade_seconds, 0., 1.)
        return self.old_alpha+(.96-self.old_alpha)*remaining**1.7

    @staticmethod
    def _over(premult, alpha, colour, source_alpha):
        source_alpha = source_alpha[:, :, None]
        premult *= 1-source_alpha
        premult += colour[None, None, :]*source_alpha
        alpha *= 1-source_alpha
        alpha += source_alpha

    def _compose(self, index, summary=False):
        index = int(index)
        summary = bool(summary)
        if not 0 <= index < len(self.times_s):
            raise IndexError("Trajectory index is outside the supplied sequence")
        if index == self._last_index and summary == self._last_summary:
            return self._last_premult, self._last_alpha
        if index < self._last_index:
            self._reset()
        heat_end = index
        if self.heat_update_anchor is not None and index != len(self.times_s)-1:
            heat_end = max(0, self.heat_update_anchor
                           + ((index-self.heat_update_anchor)//5)*5)
            heat_end = min(heat_end, index)
        if heat_end > self._coverage_index:
            self._stamp_prefix(
                self._coverage, self._coverage_index+1, heat_end,
                self._coverage_thickness, 255)
            self._coverage_index = heat_end
        now = self.times_s[index]
        old_end = int(np.searchsorted(
            self.times_s, now-self.fade_seconds, side="right")-1)
        old_end = min(old_end, index)
        if old_end > self._old_index:
            self._stamp_prefix(self._old_line, self._old_index+1,
                               old_end, self._line_thickness,
                               round(self.old_alpha*255))
            self._old_index = old_end
        if self.heat_update_anchor is None:
            refresh = (self._heat_premult is None
                       or index-self._heat_index >= 5
                       or index == len(self.times_s)-1)
        else:
            refresh = self._heat_index != heat_end
        if refresh:
            self._refresh_heat(heat_end)
        premult = self._heat_premult.copy()
        alpha = self._heat_alpha.copy()
        if summary:
            # One uniform full-history line. No birth-time gradient or sparks.
            trail = np.zeros(self._shape, np.uint8)
            self._stamp_prefix(trail, 0, index,
                               self._line_thickness, round(.70*255))
        else:
            trail = self._old_line.copy()
            first_recent = max(0, old_end)
            if first_recent == 0 and self.valid[0]:
                cv2.circle(
                    trail, self._fixed_point(self._uv[0]),
                    max(8, self._line_thickness*8),
                    round(float(self._trail_opacity(now-self.times_s[0]))*255),
                    -1, cv2.LINE_AA, shift=4)
            for row in range(max(1, first_recent+1), index+1):
                if not (self.valid[row-1] and self.valid[row]):
                    if self.valid[row]:
                        cv2.circle(
                            trail, self._fixed_point(self._uv[row]),
                            max(8, self._line_thickness*8),
                            round(float(self._trail_opacity(now-self.times_s[row]))*255),
                            -1, cv2.LINE_AA, shift=4)
                    continue
                midpoint_time = .5*(self.times_s[row-1]+self.times_s[row])
                value = round(float(self._trail_opacity(now-midpoint_time))*255)
                cv2.line(trail, self._fixed_point(self._uv[row-1]),
                         self._fixed_point(self._uv[row]), value,
                         self._line_thickness, cv2.LINE_AA, shift=4)
        self._over(premult, alpha, self._blue,
                   trail.astype(np.float32)/255.)
        if not summary:
            if self.particle_style == "particles":
                dots = self._particle_mask(now)
            else:
                dots = np.zeros(self._shape, np.uint8)
                begin_dot = int(np.searchsorted(
                    self._dot_times, now-self.fade_seconds, side="left"))
                end_dot = int(np.searchsorted(self._dot_times, now, side="right"))
                for dot in range(begin_dot, end_dot):
                    if not self._dot_valid[dot]:
                        continue
                    remaining = 1-np.clip(
                        (now-self._dot_times[dot])/self.fade_seconds, 0., 1.)
                    opacity = .94*remaining**1.6
                    cv2.circle(dots, self._fixed_point(self._dot_uv[dot]),
                               self._dot_radius_px*16, round(opacity*255),
                               -1, cv2.LINE_AA, shift=4)
            self._over(premult, alpha, self._dot_colour,
                       dots.astype(np.float32)/255.)
        self._last_index = index
        self._last_summary = summary
        self._last_premult = premult
        self._last_alpha = alpha
        self._last_texture = None
        self._last_prepared = None
        return premult, alpha

    def texture(self, index, summary=False):
        """Return independent uint8 straight BGRA, optionally a uniform summary."""
        premult, alpha = self._compose(index, summary=summary)
        if self._last_texture is None:
            straight = np.divide(premult, alpha,
                                 out=np.zeros_like(premult), where=alpha > 1e-8)
            clipped_alpha = alpha[:, :, 0]*self._clip
            output = np.empty((*self._shape, 4), np.uint8)
            output[:, :, :3] = np.clip(
                np.rint(straight), 0, 255).astype(np.uint8)
            output[:, :, 3] = np.clip(
                np.rint(clipped_alpha*255), 0, 255).astype(np.uint8)
            output[output[:, :, 3] == 0, :3] = 0
            self._last_texture = output
        return self._last_texture.copy()

    def summary_texture(self, index):
        """Full prefix and unchanged heat, one .70-alpha blue line, no sparks."""
        return self.texture(index, summary=True)

    def texture_prepared(self, index, summary=False):
        """Return demo_outro.PlaneTexture directly in premultiplied float32.

        The cached array is shared and should be treated as read-only. No
        unpremultiply, uint8 quantization or prepare_texture round trip occurs.
        Clipping is already applied to both premultiplied colour and alpha.
        """
        try:
            from .demo_outro import PlaneTexture
        except ImportError:
            from demo_outro import PlaneTexture
        premult, alpha = self._compose(index, summary=summary)
        if self._last_prepared is None:
            prepared = np.empty((*self._shape, 4), np.float32)
            np.multiply(premult, self._clip[:, :, None],
                        out=prepared[:, :, :3])
            prepared[:, :, :3] *= 1./255.
            prepared[:, :, 3] = alpha[:, :, 0]*self._clip
            self._last_prepared = PlaneTexture(np.ascontiguousarray(prepared))
        return self._last_prepared


class Speedometer:
    def __init__(self, size: int = 220, maximum: float = 40):
        if size < 200 or maximum <= 0:
            raise ValueError("size must be >= 200 and maximum must be positive")
        self.size = int(size)
        self.maximum = float(maximum)
        self.margin = 28
        self.center = np.array([self.size / 2, self.size / 2], dtype=float)
        self.radius = self.size * 0.46
        self._rgba = self._build_static()
        self._bgr = self._rgba[:, :, :3][:, :, ::-1].astype(np.float32)
        self._alpha = self._rgba[:, :, 3:4].astype(np.float32) / 255.0
        self._inverse_alpha = 1.0 - self._alpha

    @staticmethod
    def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
        folder = Path("C:/Windows/Fonts")
        choices = ("segoeuib.ttf", "arialbd.ttf") if bold else ("segoeui.ttf", "arial.ttf")
        for name in choices:
            if (folder / name).is_file():
                return ImageFont.truetype(str(folder / name), size)
        return ImageFont.load_default()

    def _point(self, angle: float, radius: float) -> np.ndarray:
        radians = math.radians(angle)
        return self.center + radius * np.array([math.cos(radians), math.sin(radians)])

    def _angle(self, speed: float) -> float:
        return 150.0 + 240.0 * np.clip(speed / self.maximum, 0, 1)

    def _build_static(self) -> np.ndarray:
        scale = 3
        image = Image.new("RGBA", (self.size * scale, self.size * scale), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        center = self.center * scale
        radius = self.radius * scale

        def box(r):
            return (center[0] - r, center[1] - r, center[0] + r, center[1] + r)

        def point(angle, radial):
            return tuple((self._point(angle, radial) * scale).tolist())

        # Dark glass keeps the compact dial readable over bright source ice.
        draw.ellipse(box(radius), fill=(8, 19, 29, 245), outline=(171, 209, 224, 102), width=scale)
        draw.ellipse(box(radius - 3 * scale), outline=(5, 11, 18, 164), width=2 * scale)
        draw.arc(box(radius - 6 * scale), 193, 317, fill=(178, 225, 241, 64), width=scale)
        draw.arc(box(radius - 6 * scale), 13, 137, fill=(22, 60, 78, 130), width=scale)

        dial_radius = self.size * 0.413
        draw.arc(box(dial_radius * scale), 150, 390, fill=(63, 169, 207, 186), width=2 * scale)
        draw.arc(box((dial_radius - 8) * scale), 150, 390, fill=(95, 154, 176, 32), width=scale)

        minor_step = 1.0 if self.maximum <= 50 else 2.0
        major_step = 5.0 if self.maximum <= 50 else 10.0
        values = np.arange(0, self.maximum + minor_step / 2, minor_step)
        for value in values:
            is_major = abs(value / major_step - round(value / major_step)) < 1e-6
            angle = self._angle(float(value))
            inner = self.size * (0.369 if is_major else 0.389)
            outer = self.size * 0.412
            draw.line([point(angle, inner), point(angle, outer)],
                      fill=(236, 246, 250, 240) if is_major else (168, 197, 210, 135),
                      width=(2 if is_major else 1) * scale)

        font = self._font(round(self.size * 0.059 * scale), bold=True)
        labels = list(np.arange(0, self.maximum + major_step / 2, major_step))
        # If an unusual maximum is selected, its endpoint still receives a label.
        if labels[-1] < self.maximum - 1e-6:
            labels.append(self.maximum)
        for value in labels:
            if value > self.maximum:
                continue
            location = point(self._angle(float(value)), self.size * 0.317)
            draw.text(location, f"{value:g}", font=font, fill=(241, 249, 252, 250), anchor="mm")

        # Separate the unit from the brand with deliberately generous spacing.
        unit = self._font(round(self.size * 0.061 * scale), bold=False)
        draw.text((center[0], self.size * 0.682 * scale), "km/h", font=unit,
                  fill=(219, 238, 246, 230), anchor="mm")
        brand = self._font(round(self.size * 0.041 * scale), bold=True)
        brand_text = "R I N K S P A C E"
        draw.text((center[0], self.size * 0.817 * scale), brand_text, font=brand,
                  fill=(152, 205, 225, 204), anchor="mm")
        draw.line([(self.size * 0.40 * scale, self.size * 0.77 * scale),
                   (self.size * 0.60 * scale, self.size * 0.77 * scale)],
                  fill=(71, 167, 201, 124), width=scale)
        return np.asarray(image.resize((self.size, self.size), Image.Resampling.LANCZOS)).copy()

    def _needle_rgba(self, speed: float) -> np.ndarray:
        layer = np.zeros((self.size, self.size, 4), dtype=np.uint8)
        center = self.center
        center_i = tuple(np.rint(center).astype(int))
        finite = math.isfinite(float(speed))
        if finite:
            angle = self._angle(float(speed))
            direction = np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))])
            perpendicular = np.array([-direction[1], direction[0]])
            tip = center + direction * self.size * 0.326
            tail = center - direction * self.size * 0.058
            points = np.rint(np.array([tail + perpendicular * self.size * 0.011,
                                      tip,
                                      tail - perpendicular * self.size * 0.011])).astype(np.int32)
            # Restrained cyan progress rim plus a warm needle give the moving
            # indicator contrast against both ice and the cool dial palette.
            if speed > 0:
                angles = np.linspace(150, angle, max(2, int((angle - 150) / 3) + 1))
                path = np.rint(np.array([self._point(a, self.size * 0.413) for a in angles])).astype(np.int32)
                cv2.polylines(layer, [path], False, (24, 213, 255, 240), 3, cv2.LINE_AA)
            cv2.line(layer, tuple(np.rint(tail).astype(int)), tuple(np.rint(tip).astype(int)),
                     (255, 171, 104, 58), 11, cv2.LINE_AA)
            cv2.fillConvexPoly(layer, points, (255, 174, 103, 255), cv2.LINE_AA)
            cv2.line(layer, center_i, tuple(np.rint(tip).astype(int)),
                     (255, 221, 173, 244), 1, cv2.LINE_AA)
        cv2.circle(layer, center_i, round(self.size * 0.039), (18, 39, 51, 255), -1, cv2.LINE_AA)
        cv2.circle(layer, center_i, round(self.size * 0.039), (71, 177, 211, 246), 2, cv2.LINE_AA)
        cv2.circle(layer, center_i, round(self.size * 0.018), (228, 245, 251, 255), -1, cv2.LINE_AA)
        cv2.circle(layer, (center_i[0] - 2, center_i[1] - 2), max(1, round(self.size * 0.008)),
                   (255, 255, 255, 255), -1, cv2.LINE_AA)
        return layer

    def draw(self, frame: np.ndarray, speed_kmh: float) -> np.ndarray:
        """Overlay at top-right and return *frame*, mutated in place (BGR)."""
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("frame must be an 8-bit three-channel BGR image")
        height, width = frame.shape[:2]
        x = width - self.size - self.margin
        y = self.margin
        if x < 0 or y < 0:
            raise ValueError("frame too small for this dial and margin")
        roi = frame[y:y + self.size, x:x + self.size]
        background = roi.astype(np.float32)
        mixed = background * self._inverse_alpha + self._bgr * self._alpha
        needle = self._needle_rgba(speed_kmh)
        needle_alpha = needle[:, :, 3:4].astype(np.float32) / 255.0
        mixed = mixed * (1 - needle_alpha) + needle[:, :, :3][:, :, ::-1] * needle_alpha
        roi[:] = np.clip(mixed, 0, 255).astype(np.uint8)
        return frame

    def save_preview(self, path: str | Path, speed_kmh: float = 25):
        """Save a transparent PNG of the dial (does not create a video)."""
        base = Image.fromarray(self._rgba, "RGBA")
        needle = Image.fromarray(self._needle_rgba(speed_kmh), "RGBA")
        Image.alpha_composite(base, needle).save(path)

