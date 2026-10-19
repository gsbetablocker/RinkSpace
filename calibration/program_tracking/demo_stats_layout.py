"""Data-page layout and continuous plane unfolding for the native-50 demo.

The presentation camera inherits the exact final model camera.  Its endpoint
is composed directly into the left panel instead of pausing at screen center.
No fitting, source trajectory edit or metric computation occurs in this file.
"""
from __future__ import annotations

from pathlib import Path
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from calibration.program_tracking.demo_outro import OutroTransform, _ease

WH = (1920, 1080)
MAP_RECT = (74, 219, 1234, 725)
CHART_RECT = (106, 840, 1848, 968)
INK = (237, 245, 249)
MUTED = (145, 169, 184)
BLUE = (96, 197, 246)
GOLD = (245, 196, 116)


def _font(size, bold=False, latin=False):
    folder = Path("C:/Windows/Fonts")
    choices = (("bahnschrift.ttf", "segoeuib.ttf") if latin else
               (("msyhbd.ttc", "Dengb.ttf") if bold else ("msyh.ttc", "Deng.ttf")))
    for name in choices:
        if (folder/name).is_file():
            return ImageFont.truetype(str(folder/name), size)
    raise FileNotFoundError("A native Chinese/Latin font is required")


class LeftPanelTransform(OutroTransform):
    def __init__(self, camera25, bounds_xy):
        super().__init__(camera25, bounds_xy, WH, turn_seconds=3.0,
                         fade_seconds=.55, hold_seconds=10.0)
        x0, y0, x1, y1 = MAP_RECT
        b = self.bounds_xy
        focal = float(np.exp(np.mean(self.last_camera25[6:8])))
        self._distance1 = max(focal*(b[2]-b[0])/((x1-x0)*.98),
                              focal*(b[3]-b[1])/((y1-y0)*.98))
        self.end_camera25[:3] = self._center+[0., 0., self._distance1]
        self.end_camera25[8:10] = [(x0+x1)/2., (y0+y1)/2.]
        self.total_seconds = 13.0


class StatsLayout:
    """Transparent fixed layout.  Times are on the programme timer axis."""

    def __init__(self, display, times_s, speed_kmh, excluded_mask,
                 exclusion_intervals):
        self.display = dict(display)
        self.times = np.asarray(times_s, float)
        self.speed = np.asarray(speed_kmh, float)
        self.excluded = np.asarray(excluded_mask, bool)
        self.intervals = list(exclusion_intervals)
        if (self.times.ndim != 1 or self.speed.shape != self.times.shape
                or self.excluded.shape != self.times.shape
                or not len(self.times) or not np.isfinite(self.speed).all()
                or not np.isfinite(self.times).all()):
            raise ValueError("Speed curve must contain aligned finite samples")
        self.duration = float(self.display.get("programme_seconds", 166.0))
        self.average = float(self.display["average_speed_kmh"])
        self.maximum = float(self.display["max_speed_kmh"])
        self.ymax = float(math.ceil(max(30., self.maximum+2)/10)*10)
        self.fonts = {}
        self.static = self._static()
        self.final = self._with_curve(self.static.copy(), 1.0)
        self._static_float = self.static.astype(np.float32)
        self._final_float = self.final.astype(np.float32)

    def font(self, size, bold=False, latin=False):
        key = (size, bold, latin)
        if key not in self.fonts:
            self.fonts[key] = _font(size, bold, latin)
        return self.fonts[key]

    def _static(self):
        page = Image.new("RGBA", WH, (0, 0, 0, 0))
        d = ImageDraw.Draw(page)
        def text(xy, value, size=24, colour=INK, bold=False, latin=False, anchor="lt"):
            d.text(xy, str(value), font=self.font(size, bold, latin),
                   fill=(*colour, 255), anchor=anchor)
        text((72, 52), self.display.get("name_zh", "冈田芽衣"), 52, bold=True)
        text((323, 78), self.display.get("name_latin", "Mei OKADA"), 30,
             colour=(182, 208, 220), latin=True)
        text((74, 133), self.display.get(
            "event_line", "日本 JPN  ·  2026 ISU 青年大奖赛安卡拉站  ·  女子短节目"),
             24, MUTED)
        text((1848, 60), "RINKSPACE", 28, (167, 202, 219), latin=True, anchor="rt")
        text((1848, 101), "PROGRAMME ANALYSIS", 15, (99, 132, 151), latin=True, anchor="rt")
        d.line([(74, 185), (1848, 185)], fill=(83, 113, 132, 112), width=1)
        text((76, 202), "全程轨迹与覆盖", 22, MUTED)
        text((1288, 202), "节目数据", 22, MUTED)
        # Two narrow numeric columns. Units are distinct from the large figures.
        def metric(x, y, label, value, unit="", sub=""):
            text((x, y), label, 23, MUTED)
            text((x, y+38), value, 57, INK, latin=True)
            if unit:
                box = d.textbbox((0, 0), str(value), font=self.font(57, latin=True))
                text((x+box[2]-box[0]+13, y+66), unit, 21, MUTED, latin=True)
            if sub:
                text((x, y+110), sub, 17, (120, 147, 164))
        metric(1288, 252, "节目时间", "2:46")
        metric(1580, 252, "滑行总距离",
               f'{float(self.display["active_distance_m"]):.0f}', "m")
        d.line([(1288, 394), (1848, 394)], fill=(76, 102, 118, 110), width=1)
        metric(1288, 421, "平均滑行速度",
               f'{self.average:.1f}', "km/h",
               f'有效滑行 {float(self.display["active_seconds"]):.1f} 秒')
        metric(1580, 421, "最高滑行速度", f'{self.maximum:.1f}', "km/h")
        d.line([(1288, 565), (1848, 565)], fill=(76, 102, 118, 110), width=1)
        text((1288, 597), "冰面覆盖率", 23, MUTED)
        coverage = f'{float(self.display["coverage_percent"]):.0f}'
        text((1848, 584), coverage+"%", 62, GOLD, latin=True, anchor="rt")
        text((1288, 654), self.display.get(
            "coverage_note", "σ = 1.5 m · 固定阈值"), 18, (143, 158, 168))
        # A low-key legend avoids suggesting that heat is dwelling time.
        d.line([(84, 746), (115, 746)], fill=(*BLUE, 255), width=3)
        text((127, 732), "运动轨迹", 18, MUTED)
        d.rounded_rectangle((258, 738, 283, 754), 4, fill=(*GOLD, 190))
        text((296, 732), "覆盖热区", 18, MUTED)
        d.line([(74, 778), (1848, 778)], fill=(83, 113, 132, 95), width=1)
        text((76, 791), "全程速度", 23, MUTED)
        # Chart labels start below the title; legend occupies the upper right.
        text((1848, 791), "灰色区间：旋转，不计入平均滑行速度",
             18, (137, 155, 168), anchor="rt")
        x0, y0, x1, y1 = CHART_RECT
        # Background excluded bands are present as the graph draws.
        for item in self.intervals:
            a = float(item["start_seconds"]); b = float(item["end_seconds"])
            xa = self.x(a); xb = self.x(b)
            d.rectangle((xa, y0, xb, y1), fill=(140, 152, 165, 25))
        for value in np.arange(0, self.ymax+.1, 10):
            yy = self.y(value)
            d.line([(x0, yy), (x1, yy)], fill=(80, 109, 127, 65), width=1)
            text((x0-16, yy-1), f"{value:g}", 16, (114, 143, 161),
                 latin=True, anchor="rm")
        text((76, 824), "km/h", 15, (114, 143, 161), latin=True, anchor="rm")
        ticks = list(range(0, int(self.duration)+1, 30))
        if ticks[-1] != int(self.duration): ticks.append(int(self.duration))
        for value in ticks:
            xx = self.x(value)
            d.line([(xx, y1), (xx, y1+5)], fill=(109, 142, 162, 180), width=1)
            text((xx, y1+11), f"{value//60}:{value%60:02}", 17,
                 (132, 159, 177), latin=True, anchor="mt")
        mean_y = self.y(self.average)
        for x in range(x0, x1, 14):
            d.line([(x, mean_y), (min(x+6, x1), mean_y)],
                   fill=(230, 190, 121, 125), width=1)
        text((x1, mean_y-8), f"平均 {self.average:.1f} km/h", 16,
             (210, 179, 124), anchor="rb")
        text((960, 1040), "开发中版本 · 所示数据为模型估算，仅供参考",
             20, (127, 150, 166), anchor="mm")
        return np.asarray(page).copy()

    def x(self, t):
        x0, _, x1, _ = CHART_RECT
        return float(x0+(x1-x0)*np.clip(t/self.duration, 0, 1))

    def y(self, speed):
        _, y0, _, y1 = CHART_RECT
        return float(y1-(y1-y0)*np.clip(speed/self.ymax, 0, 1))

    def _with_curve(self, rgba, progress):
        x0, _, x1, _ = CHART_RECT
        limit = self.duration*float(np.clip(progress, 0, 1))
        indices = np.flatnonzero((self.times <= limit) & (self.times >= 0))
        if len(indices) < 2:
            return rgba
        xy = np.column_stack((
            x0+(x1-x0)*np.clip(self.times/self.duration, 0, 1),
            np.array([self.y(v) for v in self.speed])))
        keep = indices[::max(1, int(50/25))]
        if keep[-1] != indices[-1]: keep = np.r_[keep, indices[-1]]
        # Paint in contiguous selected segments, retaining exclusion transitions.
        previous = int(keep[0])
        for selected in keep[1:]:
            selected = int(selected)
            grey = bool(self.excluded[previous] or self.excluded[selected])
            colour = (147, 160, 172, 255) if grey else (*BLUE, 255)
            points = np.rint(xy[[previous, selected]]*16).astype(np.int32)
            cv2.line(rgba, tuple(points[0]), tuple(points[1]), colour, 3,
                     cv2.LINE_AA, shift=4)
            previous = selected
        eligible = np.flatnonzero(~self.excluded)
        if len(eligible):
            peak = int(eligible[np.argmax(self.speed[eligible])])
            if self.times[peak] <= limit:
                p = tuple(np.rint(xy[peak]*16).astype(int))
                cv2.circle(rgba, p, 5*16, (*GOLD, 255), -1, cv2.LINE_AA, shift=4)
                cv2.circle(rgba, p, 8*16, (*GOLD, 110), 1, cv2.LINE_AA, shift=4)
        return rgba

    def composite(self, bgr, elapsed):
        fade = _ease((float(elapsed)-2.35)/.8)
        if fade <= 0: return bgr
        progress = np.clip((float(elapsed)-3.0)/1.0, 0, 1)
        ui = self.final if progress >= 1 else self._with_curve(self.static.copy(), progress)
        alpha = ui[..., 3:4].astype(np.float32)*(fade/255.)
        out = bgr.astype(np.float32)*(1.-alpha)+ui[..., :3][..., ::-1]*alpha
        return np.clip(out+.5, 0, 255).astype(np.uint8)
