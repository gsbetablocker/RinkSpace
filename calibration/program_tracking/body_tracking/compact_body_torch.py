"""Optional float64 Torch FK/LBS evaluator for the existing compact body.

The observation model, camera, spline, finite-difference step and solver remain
owned by their existing callers.  This adapter only moves the exact compact
SMPL-X forward calculation to a selected device.  Its public return values are
NumPy arrays, matching ``CompactBody``.  There is no automatic device fallback.

For useful GPU work sizes, the caller should stack several independent pose
perturbations along the sample axis.  ``batch_size`` limits *intermediate* LBS
storage; it does not change the number of observations or exposure nodes.
Use one evaluator owner for CUDA.  Concurrent CPU workers should use the NumPy
body instead of launching many tiny operations into one device.
"""
from __future__ import annotations

from numbers import Integral
import numpy as np
from scipy.spatial.transform import Rotation


class TorchCompactBody:
    """Wrap a frozen ``CompactBody`` without reloading its licensed tensors.

    Parameters
    ----------
    source:
        The existing compact body, already constructed with the selected beta,
        scale, lift and rocker.  Changing those model values after construction
        requires constructing a new adapter.
    device:
        Explicit ``'cuda'``, ``'cuda:N'`` or ``'cpu'``.  CPU is for independent
        implementation checks, not an implicit CUDA fallback.
    batch_size:
        Maximum number of pose samples evaluated in one device batch.  A pose
        sample is one exposure node of one perturbation, not an entire frame.
    """

    def __init__(self, source, *, device="cuda", batch_size=512):
        if isinstance(source, TorchCompactBody):
            raise ValueError("Wrap the original CompactBody, not another adapter")
        if not isinstance(batch_size, Integral) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "TorchCompactBody requires PyTorch in this interpreter; "
                "CUDA was requested explicitly and will not fall back to NumPy"
            ) from exc
        target = torch.device(device)
        if target.type not in ("cpu", "cuda"):
            raise ValueError("Compact body device must be cpu or cuda[:index]")
        if target.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA body backend requested, but PyTorch CUDA is unavailable")
            index = torch.cuda.current_device() if target.index is None else target.index
            if index < 0 or index >= torch.cuda.device_count():
                raise ValueError("CUDA body backend requested an unavailable device index")
            target = torch.device("cuda", index)
        self.source = source
        self.torch = torch
        self.device = target
        self.batch_size = int(batch_size)
        self.dtype = torch.float64
        parents = np.asarray(source.parents, dtype=np.int64)
        if parents.shape != (55,) or any(parents[j] < 0 or parents[j] >= j for j in range(1, 55)):
            raise ValueError("Expected the existing topologically ordered 55-joint compact body")
        self._validate_model(source)

        def tensor(value, dtype=torch.float64):
            # Copy once: later mutations of the source's NumPy arrays cannot
            # change CPU buffers while leaving the CUDA snapshot stale.
            return torch.tensor(np.asarray(value), dtype=dtype, device=target)

        with torch.inference_mode():
            self._J = tensor(source.J)
            self._shaped = tensor(source.shaped)
            self._posedirs = tensor(source.posedirs)
            self._weights = tensor(source.W)
            self._reg_t = tensor(np.asarray(source.reg).T)
            self._blade = tensor(source.blade_relative)
            self._eye = torch.eye(3, dtype=self.dtype, device=target)
            self._feet = tensor([7, 8], dtype=torch.int64)
            self._scale = float(source.scale)
            self._levels = []
            depth = np.zeros(55, dtype=np.int64)
            for j in range(1, 55):
                depth[j] = depth[parents[j]] + 1
            for level in range(1, int(depth.max()) + 1):
                indices = np.flatnonzero(depth == level)
                p = parents[indices]
                self._levels.append((tensor(indices, dtype=torch.int64),
                                     tensor(p, dtype=torch.int64),
                                     tensor(np.asarray(source.J)[indices] - np.asarray(source.J)[p])))
            # Check real execution, not only is_available()/reported capability.
            # Unsupported wheel/device kernels fail here instead of starting a
            # fit that will silently continue on the CPU.
            probe = self._eye @ self._eye
            if not bool(torch.isfinite(probe).all().item()):
                raise RuntimeError("The selected device failed the float64 matrix smoke check")

    @staticmethod
    def _validate_model(source):
        shaped = np.asarray(source.shaped)
        nvertices = shaped.shape[0] if shaped.ndim == 2 else -1
        expected = {
            "J": (55, 3), "shaped": (nvertices, 3),
            "posedirs": (486, 3 * nvertices), "W": (nvertices, 55),
            "reg": (nvertices, 17), "default": (33, 3, 3),
            "blade_relative": (2, 5, 3),
        }
        if nvertices < 1:
            raise ValueError("The compact body has no vertices")
        for name, shape in expected.items():
            a = np.asarray(getattr(source, name))
            if a.shape != shape or not np.isfinite(a).all():
                raise ValueError(f"Invalid compact model tensor {name}: expected finite {shape}, got {a.shape}")
        if not np.isfinite(source.scale):
            raise ValueError("The compact body scale must be finite")

    def __getattr__(self, name):
        # Preserve existing read-only model metadata used by diagnostics and
        # footwear binding; avoid delegating during unpickling/construction.
        source = self.__dict__.get("source")
        if source is None:
            raise AttributeError(name)
        return getattr(source, name)

    def backend_info(self):
        """Small serializable record; does not claim end-to-end GPU fitting."""
        torch = self.torch
        info = dict(backend="torch compact FK/LBS", device=str(self.device),
                    dtype="float64", batch_size=self.batch_size,
                    torch_version=torch.__version__, compiled_cuda=torch.version.cuda,
                    vertices=int(self.source.W.shape[0]), fk_depth_groups=len(self._levels),
                    numpy_return=True, automatic_fallback=False)
        if self.device.type == "cuda":
            info.update(device_name=torch.cuda.get_device_name(self.device),
                        compute_capability=list(torch.cuda.get_device_capability(self.device)),
                        compiled_architectures=torch.cuda.get_arch_list())
        return info

    def forward(self, pose, delta, *, additional_rotations=None, with_frames=False):
        from .rebuild_body_geometry import POSE_DOFS
        pose = np.asarray(pose, dtype=np.float64)
        delta = np.asarray(delta, dtype=np.float64)
        if pose.ndim != 3 or pose.shape[1:] != (21, 3):
            raise ValueError("Expected pose (N,21,3)")
        n = len(pose)
        if delta.shape != (n, len(POSE_DOFS)):
            raise ValueError("The compact pose correction has the wrong shape")
        if not np.isfinite(pose).all() or not np.isfinite(delta).all():
            raise ValueError("Nonfinite body pose or correction")
        local = np.broadcast_to(np.eye(3), (n, 55, 3, 3)).copy()
        if n:
            local[:, 1:22] = Rotation.from_rotvec(pose.reshape(-1, 3)).as_matrix().reshape(n, 21, 3, 3)
        local[:, 22:] = self.source.default
        for d, (j, axis) in enumerate(POSE_DOFS):
            if not n:
                break
            vector = np.zeros((n, 3))
            vector[:, axis] = delta[:, d]
            local[:, j] = local[:, j] @ Rotation.from_rotvec(vector).as_matrix()
        if additional_rotations is not None:
            for j, correction in additional_rotations.items():
                j = int(j)
                correction = np.asarray(correction, dtype=np.float64)
                if j < 0 or j >= 55 or correction.shape != (n, 3) or not np.isfinite(correction).all():
                    raise ValueError("Invalid additional local rotation")
                if n:
                    local[:, j] = local[:, j] @ Rotation.from_rotvec(correction).as_matrix()
        return self._forward_local_rotations(local, with_frames=with_frames)

    def forward_rotations(self, body_local_R, *, with_frames=False):
        body_local_R = np.asarray(body_local_R, dtype=np.float64)
        if body_local_R.ndim != 4 or body_local_R.shape[1:] != (21, 3, 3):
            raise ValueError("Expected (N,21,3,3), excluding world root Q")
        if not np.isfinite(body_local_R).all():
            raise ValueError("Nonfinite body rotation")
        local = np.broadcast_to(np.eye(3), (len(body_local_R), 55, 3, 3)).copy()
        local[:, 1:22] = body_local_R
        local[:, 22:] = self.source.default
        return self._forward_local_rotations(local, with_frames=with_frames)

    def _forward_local_rotations(self, local, *, with_frames=False):
        local = np.asarray(local, dtype=np.float64)
        if local.ndim != 4 or local.shape[1:] != (55, 3, 3) or not np.isfinite(local).all():
            raise ValueError("Expected finite (N,55,3,3) local rotations")
        n = len(local)
        if not n:
            result = (np.empty((0, 17, 3)), np.empty((0, 22, 3)), np.empty((0, 2, 5, 3)))
            return (*result, np.empty((0, 22, 3, 3))) if with_frames else result
        # Outputs are small; intermediate vertices/blended matrices are bounded
        # by batch_size, including when the caller submits all +/- directions.
        outputs = [[], [], []] + ([[]] if with_frames else [])
        torch = self.torch
        with torch.inference_mode():
            for start in range(0, n, self.batch_size):
                node = torch.as_tensor(np.ascontiguousarray(local[start:start + self.batch_size]),
                                       dtype=self.dtype, device=self.device)
                values = self._forward_device(node, with_frames=with_frames)
                # One device-to-host copy/synchronization per chunk, rather
                # than one for each COCO/native/blade/frame output.
                shapes = [tuple(v.shape) for v in values]
                widths = [int(np.prod(s[1:])) for s in shapes]
                flat = torch.cat([v.reshape(len(node), -1) for v in values], dim=1).cpu().numpy()
                offset = 0
                for collection, shape, width in zip(outputs, shapes, widths):
                    collection.append(flat[:, offset:offset + width].reshape(shape).copy())
                    offset += width
        return tuple(np.concatenate(values, axis=0) for values in outputs)

    def _forward_device(self, local, *, with_frames):
        """Exact float64 forward; all tensors remain on self.device here."""
        torch = self.torch
        n = len(local)
        rotation = local.clone()
        position = self._J.unsqueeze(0).expand(n, -1, -1).clone()
        for indices, parents, rest_delta in self._levels:
            parent_R = rotation.index_select(1, parents)
            rotation[:, indices] = parent_R @ local.index_select(1, indices)
            position[:, indices] = position.index_select(1, parents) + (parent_R @ rest_delta[..., None]).squeeze(-1)
        translation = position - (rotation @ self._J[..., None]).squeeze(-1)
        feature = (local[:, 1:] - self._eye).reshape(n, 486)
        posed = self._shaped + (feature @ self._posedirs).reshape(n, -1, 3)
        blend_R = (self._weights @ rotation.reshape(n, 55, 9)).reshape(n, -1, 3, 3)
        blend_t = self._weights @ translation
        vertices = (blend_R @ posed[..., None]).squeeze(-1) + blend_t
        coco = (self._reg_t @ vertices - self._J[0]) * self._scale
        native = (position[:, :22] - self._J[0]) * self._scale
        foot_R = rotation.index_select(1, self._feet)
        blade = native.index_select(1, self._feet).unsqueeze(2) + (foot_R @ self._blade.transpose(-1, -2)).transpose(-1, -2)
        result = (coco, native, blade)
        return (*result, rotation[:, :22]) if with_frames else result
