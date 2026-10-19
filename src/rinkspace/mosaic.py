"""Single-epoch, source-selecting plane mosaic. Unobserved pixels remain transparent."""
from __future__ import annotations
import cv2
import numpy as np
from .geometry import homography


class ProvenanceMosaic:
    def __init__(self, width: int, height: int, epoch: str):
        if width <= 0 or height <= 0 or width * height > 16_000_000:
            raise ValueError("Invalid/oversized seed canvas; tile larger maps")
        if not epoch:
            raise ValueError("A background epoch is required")
        self.epoch = epoch
        self.rgb = np.zeros((height, width, 3), np.uint8)
        self.source_index = np.full((height, width), -1, np.int32)
        self.quality = np.full((height, width), -np.inf, np.float32)

    def add(self, image: np.ndarray, mask: np.ndarray, h_canvas_from_image: np.ndarray,
            source_index: int, quality: float, epoch: str) -> None:
        if epoch != self.epoch:
            raise ValueError("Different ice epochs belong to separate mosaic layers")
        if source_index < 0 or not np.isfinite(quality):
            raise ValueError("Invalid source index or quality")
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or mask.shape != image.shape[:2]:
            raise ValueError("Expected uint8 RGB/BGR image and matching binary mask")
        h = homography(h_canvas_from_image)
        # Erosion prevents bilinear colour sampling from masked people/boards at an edge.
        valid = cv2.erode((mask > 0).astype(np.uint8) * 255, np.ones((3, 3), np.uint8))
        wh = self.rgb.shape[1], self.rgb.shape[0]
        warped = cv2.warpPerspective(image, h, wh, flags=cv2.INTER_LINEAR)
        support = cv2.warpPerspective(valid, h, wh, flags=cv2.INTER_NEAREST) > 0
        take = support & (quality > self.quality)
        self.rgb[take] = warped[take]
        self.quality[take] = quality
        self.source_index[take] = source_index

    def rgba(self) -> np.ndarray:
        return np.dstack([self.rgb, (self.source_index >= 0).astype(np.uint8) * 255])
