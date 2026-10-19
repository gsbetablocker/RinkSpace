"""Image-driven coarse row exclusions for skating foreground on known ice.

This is a bounded classical-image candidate, not a learned person segmenter.
It preserves the caller's ice boundary and excludes screen text separately.
No source-specific coordinates, source identities, or per-frame manual labels
are used. Low-contrast white clothing can remain uncertain and needs review.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class RowForegroundConfig:
    # All dimensions scale with input width relative to the 960px work image.
    horizontal_margin_px: int = 18
    vertical_margin_px: int = 12
    pale_skate_extension_px: int = 56
    pale_skate_horizontal_margin_px: int = 36
    text_margin_px: int = 8
    minimum_component_area_fraction: float = .00035
    minimum_component_height_fraction: float = .045
    maximum_component_width_fraction: float = .55
    maximum_upright_aspect_ratio: float = 3.8

    def __post_init__(self):
        if any(not np.isfinite(value) or value <= 0 for value in asdict(self).values()):
            raise ValueError('Foreground limits must be finite and positive')
        if any(value >= 1 for value in (self.minimum_component_area_fraction,
                self.minimum_component_height_fraction, self.maximum_component_width_fraction)):
            raise ValueError('Foreground fractions must be less than one')


def row_foreground_exclusion(image, ice, *, config=None):
    """Return coarse foreground/usable masks in the supplied image coordinates.

    `ice` must already exclude boards and unknown plane regions. Dark/saturated
    connected clothing seeds receive per-row envelopes and bounded margins.
    A separate downward extension covers pale skates missing from those seeds.
    The extension is uncertainty exclusion, not an inferred foot location.
    """
    config = config or RowForegroundConfig()
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3
            or image.shape[2] != 3):
        raise ValueError('Expected uint8 BGR source image')
    if np.asarray(ice).shape != image.shape[:2] or not np.isfinite(ice).all():
        raise ValueError('Ice domain must be a finite mask in the same image coordinates')
    height, width = image.shape[:2]
    domain = (np.asarray(ice) > 0).astype(np.uint8) * 255
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, saturation, value = cv2.split(hsv)
    bright = (saturation < 75) & (value > 130) & (domain > 0)
    brightness = float(np.percentile(value[bright], 70)) if np.any(bright) else 200.
    colored = (((saturation > 55) & (value < 225))
               | (value < max(94., brightness * .73))).astype(np.uint8) * 255
    colored = cv2.bitwise_and(colored, domain)
    kernel_size = max(3, round(width / 240) | 1)
    colored = cv2.morphologyEx(colored, cv2.MORPH_CLOSE,
                              np.ones((kernel_size, kernel_size), np.uint8), iterations=2)
    # Morphological closing may bridge the edge, but cannot add non-ice seeds.
    colored = cv2.bitwise_and(colored, domain)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(colored)
    foreground = np.zeros_like(domain)
    person_mask = np.zeros_like(domain)
    text_mask = np.zeros_like(domain)
    components = []
    scale = width / 960.
    horizontal = max(3, round(config.horizontal_margin_px * scale))
    vertical = max(3, round(config.vertical_margin_px * scale))
    foot_extension = max(vertical, round(config.pale_skate_extension_px * scale))
    foot_horizontal = max(horizontal, round(config.pale_skate_horizontal_margin_px * scale))
    text_margin = max(3, round(config.text_margin_px * scale))
    for index in range(1, count):
        x, y, w, h, area = (int(number) for number in stats[index])
        text = (y > height * .86 and h > height * .012 and w > width * .035
                and w / max(h, 1) > 1.6 and h < height * .08)
        upright = (h > height * config.minimum_component_height_fraction
                   and .12 < w / max(h, 1) < config.maximum_upright_aspect_ratio
                   and area > height * width * config.minimum_component_area_fraction
                   and w < width * config.maximum_component_width_fraction)
        if not (upright or text):
            continue
        envelope = np.zeros_like(domain)
        seed_rows = []
        for row in range(y, y + h):
            columns = np.flatnonzero(labels[row] == index)
            if len(columns):
                left, right = int(columns[0]), int(columns[-1]) + 1
                envelope[row, left:right] = 255
                seed_rows.append([row, left, right])
        if text:
            expanded = cv2.dilate(envelope, np.ones((2 * text_margin + 1,) * 2, np.uint8))
            text_mask = cv2.bitwise_or(text_mask, expanded)
        else:
            expanded = cv2.dilate(envelope, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (2 * horizontal + 1, 2 * vertical + 1)))
            # A bent leg can end higher than the standing leg: extending only
            # the component's global lowest row misses its pale skate. Extend
            # every observed row down by a bounded distance instead. The
            # bottom-anchored dilation is directional (never shifts upward).
            # This is conservative uncertainty around clothing, not foot
            # detection, and never creates a fixed central-image exclusion.
            lower_uncertainty = cv2.dilate(envelope,
                np.ones((foot_extension + 1, 2 * foot_horizontal + 1), np.uint8),
                anchor=(foot_horizontal, foot_extension))
            expanded = cv2.bitwise_or(expanded, lower_uncertainty)
            person_mask = cv2.bitwise_or(person_mask, expanded)
        foreground = cv2.bitwise_or(foreground, expanded)
        components.append({'component_id': index, 'kind': 'screen_text_candidate' if text else 'upright_clothing_candidate',
            'seed_box_xyxy': [x, y, x + w, y + h], 'seed_area_px': area,
            'observed_row_spans': seed_rows, 'pale_skate_extension_px': 0 if text else foot_extension})
    usable = cv2.bitwise_and(domain, cv2.bitwise_not(foreground))
    row_spans = []
    for row in range(height):
        values = np.r_[False, foreground[row] > 0, False]
        changes = np.flatnonzero(values[1:] != values[:-1])
        if len(changes):
            row_spans.append({'y': row, 'excluded_intervals_x': changes.reshape(-1, 2).tolist()})
    return {'foreground': foreground, 'person': person_mask, 'screen_text': text_mask,
        'usable': usable, 'ice': domain, 'row_spans': row_spans, 'components': components,
        'status': 'heuristic_candidate_requires_visual_review',
        'method': 'observed_clothing_row_envelopes+bounded_margins+pale_skate_uncertainty',
        'precise_person_segmentation': False, 'config': asdict(config),
        'image_width': width, 'image_height': height,
        'foreground_on_ice_pixels': int(np.count_nonzero((foreground > 0) & (domain > 0))),
        'usable_ice_pixels': int(np.count_nonzero(usable))}
