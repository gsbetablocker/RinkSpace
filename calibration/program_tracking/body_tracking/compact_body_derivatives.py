"""Local SMPL-X derivatives without repeated whole-body finite differences.

The 21 corrected joints use the existing chart ``R_ref @ Exp(delta)``.  FK,
pose blend shapes, all 55 skinning joints and the original COCO regressor are
differentiated in float64.  One column is materialized at a time; no Jacobian
with respect to the programme's spline controls is formed here.

``kinematic_pair`` is a separate exact replay of the original +/- chart.  It
lets existing nonsmooth shoe and skin consumers keep their central-difference
semantics while avoiding unrelated compact COCO skinning.  In particular,
this does not declare non-leg pose blend shapes irrelevant to foot skin.
"""
from __future__ import annotations

from numbers import Integral
import numpy as np
from scipy.spatial.transform import Rotation

from .rebuild_body_geometry import CompactBody
from .hand_state import validate_hand


def _matrices(rotvec):
    value = np.asarray(rotvec, dtype=np.float64)
    if not value.size:
        return np.empty((*value.shape[:-1], 3, 3), dtype=np.float64)
    return Rotation.from_rotvec(value.reshape(-1, 3)).as_matrix().reshape(*value.shape[:-1], 3, 3)


def _skew(value):
    value = np.asarray(value, dtype=np.float64)
    out = np.zeros((*value.shape[:-1], 3, 3), dtype=np.float64)
    out[..., 0, 1] = -value[..., 2]; out[..., 0, 2] = value[..., 1]
    out[..., 1, 0] = value[..., 2]; out[..., 1, 2] = -value[..., 0]
    out[..., 2, 0] = -value[..., 1]; out[..., 2, 1] = value[..., 0]
    return out


def right_jacobian(rotvec):
    """Right SO(3) Jacobian, including the exact zero-angle limit."""
    value = np.asarray(rotvec, dtype=np.float64)
    theta2 = np.sum(value * value, axis=-1)
    theta = np.sqrt(theta2)
    # sinc avoids cancellation in 1-cos(theta); the cubic term needs its
    # Taylor limit near zero.  This is the derivative of Exp(delta+epsilon),
    # not an additional right-increment chart at the current rotation.
    a = .5 * np.sinc(theta / (2 * np.pi)) ** 2
    b = np.empty_like(theta)
    small = theta2 < 1e-6
    q = theta2[small]
    b[small] = 1 / 6 - q / 120 + q * q / 5040 - q * q * q / 362880
    np.divide(theta - np.sin(theta), theta2 * theta,
              out=b, where=~small)
    K = _skew(value)
    return np.eye(3) - a[..., None, None] * K + b[..., None, None] * (K @ K)


def _function_key(method):
    function = getattr(method, '__func__', method)
    # functools.wraps preserves module/qualname even when a custom wrapper
    # changes numerical output. Such consumers must keep their own evaluator.
    if hasattr(function, '__wrapped__'):
        return None
    return (getattr(function, '__module__', ''), getattr(function, '__qualname__', ''))


_PACKAGE = 'calibration.program_tracking.body_tracking.'
_PROXY_WORLDS = {
    (_PACKAGE + 'opening_repair', 'OpeningReference.world'),
    (_PACKAGE + 'opening_motion', 'MotionReference.world'),
}
_BASE_WORLD = (_PACKAGE + 'fullbody_state', 'FrozenReference.world')
_BOUND_WORLDS = {
    (_PACKAGE + 'footwear_binding', 'IntegratedBodyState.world'),
    (_PACKAGE + 'skate_integration', 'SkateReference.world'),
}
_BINDING_EVALUATORS = {
    (_PACKAGE + 'footwear_binding', 'FootwearBinding.evaluate'),
    (_PACKAGE + 'skate_binding', 'SkateBinding.evaluate'),
}


def resolve_reference(reference):
    """Resolve only the audited world implementations, or return ``None``.

    A subclass inheriting the same world method is supported (ClipReference
    does this).  An overridden world/equipment implementation uses the old
    evaluator until its chain has its own numerical check.  No body, binding
    or reference object is replaced or mutated by this resolver.
    """
    seen = set()
    owner = reference
    while id(owner) not in seen:
        seen.add(id(owner))
        key = _function_key(getattr(owner, 'world', None))
        if key in _PROXY_WORLDS:
            owner = vars(owner).get('parent')
            if owner is None:
                return None
            continue
        if key != _BASE_WORLD and key not in _BOUND_WORLDS:
            return None
        # CUDA remains on the CUDA evaluator.  Do not unwrap its NumPy source
        # and then report that CPU work as GPU acceleration.
        if type(getattr(owner, 'body', None)) is not CompactBody:
            return None
        if key in _BOUND_WORLDS:
            binding = getattr(owner, 'binding', None)
            if binding is not None and _function_key(getattr(binding, 'evaluate', None)) not in _BINDING_EVALUATORS:
                return None
        return owner
    return None


def supports_reference(reference):
    return resolve_reference(reference) is not None


def uses_binding(reference):
    return _function_key(getattr(reference, 'world', None)) in _BOUND_WORLDS and getattr(reference, 'binding', None) is not None


class CompactBodyLinearization:
    """One immutable local pose linearization, shared by independent columns.

    ``delta_local_rotvec`` has shape (N,21,3).  Public column numbers are 6..68
    in the existing 69-dimensional physical state.  Reordered/repeated column
    requests do not alter state.  ``include_vertices=False`` prepares FK only
    for a consumer that still evaluates its full-surface skinning itself.

    ``chunk_size`` bounds each temporary skinning contraction.  Base compact
    intermediates scale with the number of exposure samples, never with the
    number of requested derivative columns or spline controls.
    """
    def __init__(self, model, reference_local_R, delta_local_rotvec, *,
                 hand_local_R=None, chunk_size=128, include_vertices=True):
        if type(model) is not CompactBody:
            raise TypeError('The analytic CPU derivative requires the original CompactBody')
        if not isinstance(chunk_size, Integral) or isinstance(chunk_size, bool) or chunk_size < 1:
            raise ValueError('Derivative chunk_size must be a positive integer')
        reference = np.asarray(reference_local_R, dtype=np.float64)
        delta = np.asarray(delta_local_rotvec, dtype=np.float64)
        if reference.ndim != 4 or reference.shape[1:] != (21, 3, 3):
            raise ValueError('Expected reference local rotations (N,21,3,3)')
        if delta.shape != (len(reference), 21, 3) or not np.isfinite(reference).all() or not np.isfinite(delta).all():
            raise ValueError('Expected finite local rotation increments (N,21,3)')
        parents = np.asarray(model.parents)
        if parents.shape != (55,) or any(parents[j] < 0 or parents[j] >= j for j in range(1, 55)):
            raise ValueError('Expected the topologically ordered 55-joint compact model')
        self.model = model
        self.reference_local_R = reference.copy()
        self.delta = delta.copy()
        self.chunk_size = int(chunk_size)
        self.include_vertices = bool(include_vertices)
        self.n = len(reference)
        # These are actual local SO(3) values, already using flat_hand_mean.
        # They are fixed at each sample while the 63 body coordinates vary.
        self.hand_local_R = None if hand_local_R is None else validate_hand(hand_local_R, self.n).copy()
        self.local_R = self.reference_local_R @ _matrices(self.delta)
        self._right = right_jacobian(self.delta)
        local = np.broadcast_to(np.eye(3), (self.n, 55, 3, 3)).copy()
        local[:, 1:22] = self.local_R
        local[:, 22:] = model.default
        if self.hand_local_R is not None:
            local[:, 25:55] = self.hand_local_R.reshape(self.n, 30, 3, 3)
        rotation = local.copy()
        position = np.broadcast_to(model.J, (self.n, 55, 3)).copy()
        for j in range(1, 55):
            parent = int(parents[j])
            rotation[:, j] = rotation[:, parent] @ local[:, j]
            position[:, j] = position[:, parent] + np.einsum(
                'nij,j->ni', rotation[:, parent], model.J[j] - model.J[parent])
        self._rotation55 = rotation
        self._position55 = position
        self.frames = rotation[:, :22]
        self.native = (position[:, :22] - model.J[0]) * model.scale
        self._descendants = np.eye(55, dtype=bool)
        for child in range(1, 55):
            parent = int(parents[child])
            self._descendants[:, child] |= self._descendants[:, parent]
        self._indices = [np.flatnonzero(self._descendants[j]) for j in range(55)]
        self._weights = [np.ascontiguousarray(model.W[:, ids].T) for ids in self._indices]
        self._weight_sums = [weight.sum(axis=0) for weight in self._weights]
        self.coco = None
        if self.include_vertices:
            self._translation = position - np.einsum('njab,jb->nja', rotation, model.J)
            vertices = len(model.W)
            self._posed = np.empty((self.n, vertices, 3), dtype=np.float64)
            self._blend_R = np.empty((self.n, vertices, 3, 3), dtype=np.float64)
            self.coco = np.empty((self.n, 17, 3), dtype=np.float64)
            if self.hand_local_R is None:
                tail = (model.default - np.eye(3)).reshape(297)
                shaped = model.shaped + (tail @ model.posedirs[189:]).reshape(-1, 3)
            weights = np.ascontiguousarray(model.W.T)
            for start in range(0, self.n, self.chunk_size):
                stop = min(start + self.chunk_size, self.n)
                m = stop - start
                if self.hand_local_R is None:
                    feature = (self.local_R[start:stop] - np.eye(3)).reshape(m, 189)
                    posed = shaped + (feature @ model.posedirs[:189]).reshape(m, vertices, 3)
                else:
                    feature = (local[start:stop, 1:] - np.eye(3)).reshape(m, 486)
                    posed = model.shaped + (feature @ model.posedirs).reshape(m, vertices, 3)
                blend_R = (rotation[start:stop].reshape(m, 55, 9).transpose(0, 2, 1)
                           @ weights).transpose(0, 2, 1).reshape(m, vertices, 3, 3)
                blend_t = (self._translation[start:stop].transpose(0, 2, 1) @ weights).transpose(0, 2, 1)
                points = np.einsum('nvab,nvb->nva', blend_R, posed) + blend_t
                self.coco[start:stop] = ((points.transpose(0, 2, 1) @ model.reg).transpose(0, 2, 1) - model.J[0]) * model.scale
                self._posed[start:stop] = posed
                self._blend_R[start:stop] = blend_R

    def _column(self, column):
        if not isinstance(column, Integral) or isinstance(column, bool) or not 6 <= column < 69:
            raise ValueError('Local body derivative column must be an integer in 6..68')
        return (int(column) - 6) // 3 + 1, (int(column) - 6) % 3

    def affects_joints(self, column, joints):
        joint, _ = self._column(column)
        return bool(self._descendants[joint, np.asarray(joints, dtype=int)].any())

    def derivative(self, column, *, kinematics=True):
        """Return dCOCO, dNative, dFrames and dLocal_R in the original units.

        Native joints and frames retain their original 22-joint layout.  The
        COCO derivative includes the fixed hand/face subtrees and every posed
        vertex used by all 17 original regressed points.
        """
        joint, axis = self._column(column)
        local_axis = self._right[:, joint - 1, :, axis]
        world_axis = np.einsum('nij,nj->ni', self._rotation55[:, joint], local_axis)
        dlocal = self.local_R[:, joint - 1] @ _skew(local_axis)
        native = frames = local = None
        if kinematics:
            descendant22 = self._descendants[joint, :22]
            native = np.zeros_like(self.native)
            native[:, descendant22] = np.cross(world_axis[:, None],
                self.native[:, descendant22] - self.native[:, joint, None])
            frames = np.zeros_like(self.frames)
            frames[:, descendant22] = _skew(world_axis)[:, None] @ self.frames[:, descendant22]
            local = np.zeros_like(self.local_R)
            local[:, joint - 1] = dlocal
        coco = None
        if self.include_vertices:
            coco = np.empty_like(self.coco)
            ids = self._indices[joint]
            weights = self._weights[joint]
            total_weight = self._weight_sums[joint]
            posedirs = self.model.posedirs[(joint - 1) * 9:joint * 9]
            for start in range(0, self.n, self.chunk_size):
                stop = min(start + self.chunk_size, self.n)
                m = stop - start
                # Differentiate the weighted rigid transforms by a single
                # global rotation about this joint.  Each vertex's pose-blend
                # derivative is a separate, necessary term below.
                selected_R = self._rotation55[start:stop, ids]
                blend_R = (selected_R.reshape(m, len(ids), 9).transpose(0, 2, 1)
                           @ weights).transpose(0, 2, 1).reshape(m, -1, 3, 3)
                blend_t = (self._translation[start:stop, ids].transpose(0, 2, 1)
                           @ weights).transpose(0, 2, 1)
                relative = (np.einsum('nvab,nvb->nva', blend_R, self._posed[start:stop])
                            + blend_t - total_weight[None, :, None] * self._position55[start:stop, joint, None])
                dposed = (dlocal[start:stop].reshape(m, 9) @ posedirs).reshape(m, -1, 3)
                dvertices = (np.cross(world_axis[start:stop, None], relative)
                             + np.einsum('nvab,nvb->nva', self._blend_R[start:stop], dposed))
                coco[start:stop] = (dvertices.transpose(0, 2, 1) @ self.model.reg).transpose(0, 2, 1) * self.model.scale
        return dict(coco=coco, native=native, frames=frames, local_R=local)

    def kinematic_pair(self, column, step=1e-5):
        """Exact +/- native55 FK, retaining the public native22/frame22 views.

        Hand local rotations remain fixed; their world frames and positions
        follow every perturbed ancestor, including the wrist.  No vertices
        are skinned here.
        """
        joint, axis = self._column(column)
        if not np.isfinite(step) or step <= 0:
            raise ValueError('Finite-difference step must be positive')
        descendants = self._descendants[joint]
        parent = int(self.model.parents[joint])
        result = []
        for sign in (1., -1.):
            change = self.delta[:, joint - 1].copy()
            change[:, axis] += sign * step
            local = self.local_R.copy()
            local[:, joint - 1] = self.reference_local_R[:, joint - 1] @ _matrices(change)
            current_frame = self.frames[:, parent] @ local[:, joint - 1]
            turn = current_frame @ self.frames[:, joint].swapaxes(-1, -2)
            position = self._position55.copy()
            rotation = self._rotation55.copy()
            position[:, descendants] = self._position55[:, joint, None] + np.einsum(
                'nij,nkj->nki', turn, self._position55[:, descendants] - self._position55[:, joint, None])
            rotation[:, descendants] = turn[:, None] @ self._rotation55[:, descendants]
            # Preserve the exact changed joint frame and its unchanged pivot;
            # no orthogonalization or first-order rotation approximation.
            rotation[:, joint] = current_frame
            position[:, joint] = self._position55[:, joint]
            state = dict(native=(position[:, :22] - self.model.J[0]) * self.model.scale,
                         frames=rotation[:, :22], local_R=local,
                         rotation55=rotation, position55=position)
            if self.hand_local_R is not None:
                state['hand_local_R'] = self.hand_local_R.copy()
            result.append(state)
        return tuple(result)
