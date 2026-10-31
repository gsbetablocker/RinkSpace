"""Trainable image evidence and within-shot memory for rink landmarks.

Outputs are predictions, not audited observations. The five default channels
name landmark tasks; num_classes=6 prepends background. Primitive logits are
an auxiliary supervised line/circle/other task. Neither an untrained head nor a camera projection is
pixel ground truth.  Reset ``hidden=None`` at every cut or independent shot.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence
from .storage import AssetStore

import torch
from torch import Tensor, nn
from torch.nn import functional as F


LANDMARK_CLASSES = (
    "ice_contact",
    "board_top",
    "ice_paint",
    "ad_boundary",
    "shadow_or_reflection",
)
PRIMITIVE_CLASSES = ("line", "circle", "other")


def masked_region_loss(
    logits: Tensor,
    primitive_logits: Tensor,
    region_mask: Tensor,
    semantic_class: int,
    primitive_class: int = 2,
) -> tuple[Tensor, dict[str, int]]:
    """Supervise only an explicitly reviewed source-region mask.

    Inputs are semantic logits [B,C,H,W], primitive logits [B,3,H,W], and
    a Boolean/binary mask [B,1,H,W]. Class indices follow the caller's class
    map; for the six-class model background=0 and shadow_or_reflection=5.
    Primitive indices are line=0, circle=1, other=2. This function does not
    infer masks, manufacture labels, or inspect a camera/model projection.
    The supplied region must already have source-image annotation authority.

    Loss is mean semantic CE plus mean primitive CE on selected pixels only.
    Unselected pixels have zero loss gradient; an empty mask returns a
    differentiable zero. Counts describe this supplied mask, not accuracy.
    """
    if logits.ndim != 4 or primitive_logits.ndim != 4 or region_mask.ndim != 4:
        raise ValueError("expected logits [B,C,H,W], primitives [B,3,H,W], mask [B,1,H,W]")
    batch, categories, height, width = logits.shape
    if min(batch, categories, height, width) < 1:
        raise ValueError("logits dimensions must be nonempty")
    if primitive_logits.shape != (batch, 3, height, width):
        raise ValueError("primitive_logits must be [B,3,H,W] aligned with logits")
    if region_mask.shape != (batch, 1, height, width):
        raise ValueError("region_mask must be [B,1,H,W] aligned with source logits")
    if logits.device != primitive_logits.device:
        raise ValueError("semantic and primitive logits must be on the same device")
    if not logits.is_floating_point() or not primitive_logits.is_floating_point():
        raise TypeError("logits must be floating point")
    if not isinstance(semantic_class, int) or not 0 <= semantic_class < categories:
        raise ValueError("semantic_class is outside the supplied class map")
    if not isinstance(primitive_class, int) or not 0 <= primitive_class < 3:
        raise ValueError("primitive_class must be line=0, circle=1, or other=2")
    if not torch.isfinite(logits).all().item() or not torch.isfinite(primitive_logits).all().item():
        raise FloatingPointError("region supervision received nonfinite logits")
    region_mask = region_mask.to(device=logits.device)
    if not torch.isfinite(region_mask).all().item():
        raise ValueError("region_mask must be finite binary source labels")
    if not ((region_mask == 0) | (region_mask == 1)).all().item():
        raise ValueError("region_mask must be binary; fractional/confidence masks are not labels")
    selected = region_mask[:, 0].bool()
    selected_count = int(selected.sum().item())
    counts = {
        "supervised_pixels": selected_count,
        "ignored_pixels": selected.numel() - selected_count,
        "supervised_images": int(selected.flatten(1).any(dim=1).sum().item()),
    }
    if selected_count == 0:
        zero = logits.reshape(-1)[0] * 0.0 + primitive_logits.reshape(-1)[0] * 0.0
        return zero.float(), counts
    # Index first: ignored pixels never become CE targets or contribute to
    # either reduction. float32 preserves stable losses under autocast.
    semantic_scores = logits.permute(0, 2, 3, 1)[selected].float()
    primitive_scores = primitive_logits.permute(0, 2, 3, 1)[selected].float()
    semantic_targets = torch.full(
        (selected_count,), semantic_class, dtype=torch.long, device=logits.device
    )
    primitive_targets = torch.full(
        (selected_count,), primitive_class, dtype=torch.long, device=logits.device
    )
    loss = F.cross_entropy(semantic_scores, semantic_targets) + F.cross_entropy(
        primitive_scores, primitive_targets
    )
    if not torch.isfinite(loss).item():
        raise FloatingPointError("nonfinite reviewed-region loss")
    return loss, counts


def _conv_block(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
    groups = next(g for g in (8, 4, 2, 1) if out_channels % g == 0)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(groups, out_channels),
        nn.SiLU(inplace=True),
    )


class _ConvGRUCell(nn.Module):
    """Spatial memory at one eighth of native image resolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.gates = nn.Conv2d(2 * channels, 2 * channels, 3, padding=1)
        self.candidate = nn.Conv2d(2 * channels, channels, 3, padding=1)

    def forward(self, feature: Tensor, hidden: Tensor | None) -> Tensor:
        if hidden is None:
            hidden = torch.zeros_like(feature)
        elif hidden.shape != feature.shape:
            raise ValueError(
                f"hidden shape {tuple(hidden.shape)} does not match {tuple(feature.shape)}; "
                "reset memory for a new shot or resolution"
            )
        else:
            hidden = hidden.to(device=feature.device, dtype=feature.dtype)
        update, reset = self.gates(torch.cat((feature, hidden), dim=1)).sigmoid().chunk(2, dim=1)
        candidate = self.candidate(torch.cat((feature, reset * hidden), dim=1)).tanh()
        return (1.0 - update) * hidden + update * candidate


class RinkVisionNet(nn.Module):
    """Native-resolution landmark heads with image-driven recurrent context.

    ``images``: RGB float tensor [B,T,3,H,W], in [0,1]. The local high-resolution
    branch always sees the original image. An optional, locally stored SAM2
    vision encoder supplies pretrained multi-scale features; it is not SAM2's
    mask decoder or its video/object memory.

    ``geometry_prior``: optional [B,T,C,H,W] (or one channel), aligned to the
    same source pixels. It may encode projections from an actual camera/rink
    model. Missing projections should be zero. It is an auxiliary INPUT ONLY;
    callers must supply independently audited targets to their loss function.
    Its zero-initialized projection initially leaves image features unchanged.

    ``hidden``: [B,4*width,ceil(H/8),ceil(W/8)] from the preceding contiguous
    chunk of this shot. Memory is not detached automatically; the caller may
    detach between chunks for truncated backpropagation, or reset at a cut.
    """

    def __init__(
        self,
        num_classes: int = 5,
        width: int = 24,
        backbone_path: str | Path | None = None,
        freeze_backbone: bool = True,
        sam2_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if num_classes < len(LANDMARK_CLASSES):
            raise ValueError("num_classes must include all five default landmark tasks")
        if width < 4:
            raise ValueError("width must be at least 4")
        self.num_classes = int(num_classes)
        self.width = int(width)
        self.freeze_backbone = bool(freeze_backbone)
        if backbone_path is not None:
            backbone_path = AssetStore(Path(__file__).resolve().parents[2]).source(Path(backbone_path).resolve())
        self.class_names = (
            LANDMARK_CLASSES if num_classes == 5 else ("background",) + LANDMARK_CLASSES
        ) + tuple(f"extra_{i}" for i in range(6, num_classes))
        self.primitive_names = PRIMITIVE_CLASSES
        self.model_config = {
            "num_classes": self.num_classes,
            "width": self.width,
            "backbone_path": str(backbone_path) if backbone_path is not None else None,
            "freeze_backbone": self.freeze_backbone,
            "sam2_config": sam2_config,
        }

        # This path preserves native pixels, even when pretrained features use
        # resized input. GroupNorm is independent of batch/sequence length.
        self.detail = nn.Sequential(_conv_block(3, width), _conv_block(width, width))
        self.downsample = nn.Sequential(
            _conv_block(width, 2 * width, stride=2),
            _conv_block(2 * width, 4 * width, stride=2),
            _conv_block(4 * width, 4 * width, stride=2),
        )
        self.geometry_projection = nn.Conv2d(num_classes, 4 * width, 1, bias=False)
        nn.init.zeros_(self.geometry_projection.weight)
        self.memory = _ConvGRUCell(4 * width)
        self.context = _conv_block(4 * width, width)
        self.boundary_refine = nn.Sequential(
            _conv_block(2 * width, width), _conv_block(width, width)
        )
        self.semantic_head = nn.Conv2d(width, num_classes, 1)
        self.primitive_head = nn.Sequential(_conv_block(width, width), nn.Conv2d(width, 3, 1))

        self.backbone: nn.Module | None = None
        self.backbone_projections = nn.ModuleList()
        self.backbone_detail: nn.Module | None = None
        self.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        if backbone_path is not None and sam2_config is not None:
            raise ValueError("use a local pretrained path OR a saved SAM2 architecture config")
        if backbone_path is not None or sam2_config is not None:
            if sam2_config is not None:
                # Saved checkpoints carry the architecture and the whole vision
                # encoder state, so reconstruction does not require the original
                # external model directory or any network access. These fresh
                # weights must be replaced by strict state_dict loading.
                from transformers import Sam2VisionConfig, Sam2VisionModel

                self.backbone = Sam2VisionModel(Sam2VisionConfig.from_dict(sam2_config))
            else:
                self.backbone = self._load_local_encoder(backbone_path)
            self.model_config["backbone_path"] = None
            self.model_config["sam2_config"] = self.backbone.config.to_dict()
            self.backbone.requires_grad_(not self.freeze_backbone)
            if self.freeze_backbone:
                self.backbone.eval()
            feature_channels = int(self.backbone.config.fpn_hidden_size)
            feature_levels = int(self.backbone.config.num_feature_levels)
            self.backbone_projections = nn.ModuleList(
                nn.Conv2d(feature_channels, 4 * width, 1) for _ in range(feature_levels)
            )
            self.backbone_detail = nn.Conv2d(feature_channels, width, 1)

    @staticmethod
    def _load_local_encoder(backbone_path: str | Path) -> nn.Module:
        local_path = Path(backbone_path)
        if not local_path.is_dir():
            raise FileNotFoundError(f"local SAM2 checkpoint directory missing: {local_path}")
        # A Sam2VideoModel checkpoint prefixes weights with vision_encoder.
        # Loading it directly as Sam2VisionModel can silently initialize
        # fresh weights. Load the matching class and validate every key.
        from transformers import Sam2VideoModel

        complete_model, loading_info = Sam2VideoModel.from_pretrained(
            str(local_path), local_files_only=True, output_loading_info=True
        )
        problems = {
            key: loading_info.get(key)
            for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
            if loading_info.get(key)
        }
        if problems:
            raise RuntimeError(f"SAM2 checkpoint did not load exactly: {problems}")
        encoder = complete_model.vision_encoder
        del complete_model
        return encoder

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> RinkVisionNet:
        """Reconstruct before strictly loading a saved ``state_dict``."""
        return cls(**config)

    def train(self, mode: bool = True) -> RinkVisionNet:
        super().train(mode)
        if self.backbone is not None and self.freeze_backbone:
            self.backbone.eval()
        return self

    def _pretrained_features(self, frame: Tensor) -> tuple[Tensor, ...] | None:
        if self.backbone is None:
            return None
        # The pretrained encoder's spatial positional embeddings support this
        # tested size. The native branch is never resized to 512 pixels.
        encoder_input = F.interpolate(frame, size=(512, 512), mode="bilinear", align_corners=False)
        encoder_input = (encoder_input - self.rgb_mean) / self.rgb_std
        context = torch.no_grad() if self.freeze_backbone else nullcontext()
        with context:
            result = self.backbone(pixel_values=encoder_input, return_dict=True)
        features = result.fpn_hidden_states
        if len(features) != len(self.backbone_projections):
            raise RuntimeError("SAM2 feature count disagrees with its configuration")
        return tuple(features)

    def encode_context(self, images: Tensor) -> list[Tensor]:
        """Encode full original frames once, for reuse by native crop training.

        RGB input is [B,T,3,fullH,fullW] in [0,1]. Each returned FPN tensor is
        [B,T,C,featureH,featureW], ordered finest first. For the local tiny
        encoder C=256 and the spatial sizes are 128, 64, and 32 squared.
        A frozen encoder returns detached features; an unfrozen encoder keeps
        its gradient graph. This function never encodes or invents labels.
        """
        if self.backbone is None:
            raise RuntimeError("full-frame context requires a configured SAM2 vision encoder")
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError("context images must be [B,T,3,fullH,fullW]")
        if min(images.shape[0], images.shape[1], *images.shape[-2:]) < 1:
            raise ValueError("context images must be nonempty")
        if not images.is_floating_point():
            raise TypeError("context images must be RGB floats in [0,1]")
        if not torch.isfinite(images).all().item():
            raise FloatingPointError("nonfinite full-frame RGB context")
        if ((images < 0) | (images > 1)).any().item():
            raise ValueError("full-frame RGB context must be in [0,1]; no implicit rescaling")
        encoded = [self._pretrained_features(images[:, t]) for t in range(images.shape[1])]
        return [torch.stack([frame[level] for frame in encoded], dim=1)
                for level in range(len(self.backbone_projections))]

    @staticmethod
    def _sample_context_roi(
        feature: Tensor,
        crop: Tensor,
        native_hw: tuple[int, int],
        output_hw: tuple[int, int],
    ) -> Tensor:
        """Sample FPN at the native crop's pixel-centre coordinate locations.

        ``crop`` is [B,4] in source coordinates (x,y,width,height). With
        align_corners=False, a source pixel centre x+0.5 maps to normalized
        2*(x+0.5)/fullW-1 at every FPN scale. Sampling straight to the fusion
        size avoids an intermediate resized-ROI interpolation. Border padding
        matches interpolate's endpoint behavior for a full-frame crop.
        """
        full_height, full_width = native_hw
        out_height, out_width = output_hw
        feature = feature.to(device=crop.device)
        horizontal = (torch.arange(out_width, device=crop.device, dtype=torch.float32) + 0.5) / out_width
        vertical = (torch.arange(out_height, device=crop.device, dtype=torch.float32) + 0.5) / out_height
        source_x = crop[:, 0, None, None] + crop[:, 2, None, None] * horizontal[None, None, :]
        source_y = crop[:, 1, None, None] + crop[:, 3, None, None] * vertical[None, :, None]
        grid_x = (2.0 * source_x / full_width - 1.0).expand(-1, out_height, -1)
        grid_y = (2.0 * source_y / full_height - 1.0).expand(-1, -1, out_width)
        grid = torch.stack((grid_x, grid_y), dim=-1)
        # Grid sampling uses float32 even when the cached encoder outputs are
        # BF16; projection/fusion remains eligible for the caller's autocast.
        return F.grid_sample(feature.float(), grid, mode="bilinear", padding_mode="border", align_corners=False)

    def forward(
        self,
        images: Tensor,
        hidden: Tensor | None = None,
        geometry_prior: Tensor | None = None,
        context_features: Sequence[Tensor] | None = None,
        context_crop: tuple[float, float, float, float] | Tensor | None = None,
        context_native_hw: tuple[int, int] | None = None,
    ) -> dict[str, Tensor]:
        """Read source pixels, optionally with cached full-frame FPN context.

        Cached context has the exact batch/time axes of ``images``. Specify
        source ``context_crop=(x,y,w,h)`` and ``context_native_hw=(fullH,fullW)``
        for native crops; [B,T,4] boxes are also accepted. With no crop box,
        images are the full source frame. Features receive no additional SAM2
        encoding here. The old no-context argument path remains compatible.
        """
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError("images must have shape [B,T,3,H,W]")
        batch, length, _, height, image_width = images.shape
        if batch < 1 or length < 1 or min(height, image_width) < 8:
            raise ValueError("images require nonempty sequences and H,W >= 8")
        if not images.is_floating_point():
            raise TypeError("images must be floating RGB in [0,1], not uint8")
        crops = None
        if context_features is not None:
            if self.backbone is None or len(context_features) != len(self.backbone_projections):
                raise ValueError("cached context must include each configured SAM2 FPN level")
            feature_channels = int(self.backbone.config.fpn_hidden_size)
            for feature in context_features:
                if (feature.ndim != 5 or feature.shape[:3] != (batch, length, feature_channels)
                        or min(feature.shape[-2:]) < 1 or not feature.is_floating_point()):
                    raise ValueError("each cached context level must be [B,T,C,featureH,featureW]")
            if context_crop is not None and context_native_hw is None:
                raise ValueError("source crop context requires explicit context_native_hw")
            native_hw = tuple(context_native_hw or (height, image_width))
            if len(native_hw) != 2 or any(int(v) != v or v < 1 for v in native_hw):
                raise ValueError("context_native_hw must contain positive source (H,W)")
            native_hw = (int(native_hw[0]), int(native_hw[1]))
            box = context_crop if context_crop is not None else (0, 0, native_hw[1], native_hw[0])
            crops = torch.as_tensor(box, device=images.device, dtype=torch.float32)
            if crops.shape == (4,):
                crops = crops[None, None].expand(batch, length, -1)
            elif crops.shape != (batch, length, 4):
                raise ValueError("context_crop must be (x,y,w,h) or [B,T,4]")
            if (not torch.isfinite(crops).all().item() or (crops[..., :2] < 0).any().item()
                    or (crops[..., 2:] <= 0).any().item()
                    or (crops[..., 0] + crops[..., 2] > native_hw[1]).any().item()
                    or (crops[..., 1] + crops[..., 3] > native_hw[0]).any().item()):
                raise ValueError("context crop must lie within the full original source frame")
            # A complete original frame must use the identical interpolation
            # operation as uncached inference. Numerically almost equivalent
            # grid_sample coordinates can change BF16 class decisions downstream.
            # Real source crops still need their native-coordinate ROI sampling.
            full_box = crops.new_tensor((0, 0, native_hw[1], native_hw[0]))
            if (native_hw == (height, image_width)
                    and torch.equal(crops, full_box.expand_as(crops))):
                crops = None
        elif context_crop is not None or context_native_hw is not None:
            raise ValueError("context coordinates require context_features")
        if geometry_prior is not None:
            if (
                geometry_prior.ndim != 5
                or geometry_prior.shape[:2] != images.shape[:2]
                or geometry_prior.shape[-2:] != images.shape[-2:]
                or geometry_prior.shape[2] not in (1, self.num_classes)
            ):
                raise ValueError("geometry_prior must be source-aligned [B,T,1 or C,H,W]")
            geometry_prior = geometry_prior.to(device=images.device, dtype=images.dtype)
            if geometry_prior.shape[2] == 1:
                geometry_prior = geometry_prior.expand(-1, -1, self.num_classes, -1, -1)

        logits, primitives = [], []
        for time_index in range(length):
            frame = images[:, time_index]
            native_detail = self.detail(frame)
            low_feature = self.downsample(native_detail)
            pretrained = (
                tuple(feature[:, time_index].to(device=frame.device) for feature in context_features)
                if context_features is not None else self._pretrained_features(frame)
            )
            if pretrained is not None:
                # torch.stack used by the cache can change channels-last batch
                # strides even when every FPN value is identical. Canonicalize
                # both paths before interpolation/convolution, so precision and
                # kernel selection cannot depend on whether context was cached.
                pretrained = tuple(feature.contiguous() for feature in pretrained)
                projected = [
                    projection(
                        self._sample_context_roi(feature, crops[:, time_index], native_hw, low_feature.shape[-2:])
                        if crops is not None else F.interpolate(
                            feature, size=low_feature.shape[-2:], mode="bilinear", align_corners=False
                        )
                    )
                    for projection, feature in zip(self.backbone_projections, pretrained)
                ]
                low_feature = low_feature + torch.stack(projected, dim=0).mean(dim=0)
                backbone_detail = (
                    self._sample_context_roi(
                        self.backbone_detail(pretrained[0].to(
                            device=frame.device, dtype=self.backbone_detail.weight.dtype
                        )),
                        crops[:, time_index], native_hw, (height, image_width)
                    ).to(dtype=native_detail.dtype) if crops is not None else F.interpolate(
                        self.backbone_detail(pretrained[0]), size=(height, image_width),
                        mode="bilinear", align_corners=False,
                    )
                )
                native_detail = native_detail + backbone_detail
            if geometry_prior is not None:
                low_prior = F.interpolate(
                    geometry_prior[:, time_index], size=low_feature.shape[-2:], mode="bilinear", align_corners=False
                )
                low_feature = low_feature + self.geometry_projection(low_prior)
            hidden = self.memory(low_feature, hidden)
            remembered = F.interpolate(
                self.context(hidden), size=(height, image_width), mode="bilinear", align_corners=False
            )
            refined = self.boundary_refine(torch.cat((native_detail, remembered), dim=1))
            logits.append(self.semantic_head(refined))
            primitives.append(self.primitive_head(refined))
        return {
            "logits": torch.stack(logits, dim=1),
            "primitive_logits": torch.stack(primitives, dim=1),
            "hidden": hidden,
        }
