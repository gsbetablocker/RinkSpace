"""Mutable absolute P/SO(3) curves for RinkSpace's next fitting stage.

Reference kernel, not a drop-in replacement for WindowState. NumPy + SciPy only.
There is NO frozen, noisy reference trajectory added back during evaluation.
The old candidate supplies initial controls and a finite data prior only.

Translation uses a clamped cubic B-spline. Rotations use its cumulative Lie-group
form, with four-control local support. An arbitrary number of turns is supported
provided source sampling resolves the motion. Legacy principal-log curves
retain their pi guard. An explicitly archived SU(2) edge chart permits fitted
control edges to cross pi continuously; missed source turns cannot be inferred.
Never fit a spline to a wrapped Euler yaw or one global rotvec.

The SO(3) block Jacobian below is a slow, independent finite-difference reference
for checking a production sparse/analytic implementation. Do not use it as a
reason to finite-difference the entire programme or full SMPL-X surface.

No weights, contact labels, speed limits, or accuracy thresholds are chosen here.
All distances remain in the caller's declared length unit. Apply the project's
length conversion exactly once before evaluating SI physics.
"""
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import BSpline
from scipy.sparse import csr_matrix, diags
from scipy.spatial.transform import Rotation


def _exp(v):
    v = np.asarray(v, float)
    return Rotation.from_rotvec(v.reshape(-1, 3)).as_matrix().reshape(*v.shape[:-1], 3, 3)


def _log(r):
    r = np.asarray(r, float)
    return Rotation.from_matrix(r.reshape(-1, 3, 3)).as_rotvec().reshape(*r.shape[:-2], 3)


def clamped_cubic_knots(break_seconds):
    """Explicit source-clock breaks; callers choose resolution, not this kernel."""
    b = np.asarray(break_seconds, float)
    if b.ndim != 1 or len(b) < 2 or not np.isfinite(b).all() or np.any(np.diff(b) <= 0):
        raise ValueError("Break times must be finite, unique, and increasing")
    return np.r_[np.repeat(b[0], 4), b[1:-1], np.repeat(b[-1], 4)]


def greville_times(knots):
    k = np.asarray(knots, float)
    return np.array([np.mean(k[i+1:i+4]) for i in range(len(k)-4)])


class RotationChartDomainError(ValueError):
    """An optimizer trial leaves this SO(3) curve's unambiguous log chart."""


@dataclass
class AbsoluteMotionCurve:
    """Controls: Mx3 positions and MxJx3x3 rotations (root is track zero).

    For RinkSpace J=22 means root Q plus 21 local joint rotations. Hands may
    remain their explicitly archived HandMR track in the first integration.
    The reference derivative checker returns right tangent coordinates:
    dR = R Exp(J dcontrol), compatible with current local FK derivatives.
    """
    knots: np.ndarray
    position_controls: np.ndarray
    rotation_controls: np.ndarray
    edge_reference_quaternion: np.ndarray | None = None

    def __post_init__(self):
        self.knots = np.asarray(self.knots, float).copy()
        self.position_controls = np.asarray(self.position_controls, float).copy()
        self.rotation_controls = np.asarray(self.rotation_controls, float).copy()
        m = len(self.knots)-4
        r = self.rotation_controls
        if (self.knots.ndim != 1 or m < 4 or np.any(np.diff(self.knots) < 0)
                or not np.isfinite(self.knots).all()
                or self.position_controls.shape != (m, 3)
                or r.ndim != 4 or r.shape[0] != m or r.shape[2:] != (3, 3)
                or not np.isfinite(self.position_controls).all() or not np.isfinite(r).all()):
            raise ValueError("Invalid cubic curve controls")
        if not (np.all(self.knots[:4] == self.knots[0])
                and np.all(self.knots[-4:] == self.knots[-1])
                and self.knots[0] < self.knots[-1]
                and np.all(np.diff(self.knots[3:-3]) > 0)):
            raise ValueError("Expected a clamped cubic domain with simple internal knots")
        # Numerical SO(3) validation only; these tolerances are not physical gates.
        if (np.max(np.abs(r.swapaxes(-1, -2) @ r - np.eye(3))) > 1e-7
                or np.max(np.abs(np.linalg.det(r)-1.)) > 1e-7):
            raise ValueError("Rotation controls must lie in SO(3)")
        if self.edge_reference_quaternion is not None:
            q=np.asarray(self.edge_reference_quaternion,float)
            if (q.shape!=(m-1,r.shape[1],4) or not np.isfinite(q).all()
                    or np.max(abs(np.linalg.norm(q,axis=-1)-1.))>1e-7):
                raise ValueError('Invalid archived SU(2) edge chart reference')
            self.edge_reference_quaternion=q.copy()
        self.rotation_controls.setflags(write=False)
        if self.edge_reference_quaternion is not None:
            self.edge_reference_quaternion.setflags(write=False)
        self._log_chart_cache=None
        self._check_log_chart()

    def _check_log_chart(self):
        cached=self._log_chart_cache
        rotation,chart=self.rotation_controls,self.edge_reference_quaternion
        if cached is not None and cached[0] is rotation and cached[1] is chart:
            return cached[2]
        steps=self._validate_log_chart()
        if not rotation.flags.writeable and (chart is None or not chart.flags.writeable):
            steps.setflags(write=False)
            self._log_chart_cache=(rotation,chart,steps)
        return steps

    def _validate_log_chart(self):
        relative=self.rotation_controls[:-1].swapaxes(-1,-2)@self.rotation_controls[1:]
        if self.edge_reference_quaternion is None:
            steps=_log(relative)
            if np.any(np.linalg.norm(steps,axis=-1)>=np.pi-1e-5):
                raise RotationChartDomainError('Ambiguous adjacent SO(3) branch: subdivide using the source sequence')
            return steps
        # Select the SU(2) lift relative to the explicitly archived chart.
        # This crosses theta=pi continuously, without changing any rotations
        # or silently choosing a new interpolation branch in a saved state.
        q=Rotation.from_matrix(relative.reshape(-1,3,3)).as_quat().reshape(*relative.shape[:-2],4)
        dot=np.sum(q*self.edge_reference_quaternion,axis=-1)
        if np.any(abs(dot)<1e-7):
            raise RotationChartDomainError('Trial reaches the archived SU(2) edge-chart seam')
        q=np.where((dot<0)[...,None],-q,q)
        length=np.linalg.norm(q[...,:3],axis=-1)
        angle=2*np.arctan2(length,q[...,3])
        if np.any(angle>=2*np.pi-1e-4):
            raise RotationChartDomainError('Trial reaches the singular full-turn edge logarithm')
        scale=np.divide(angle,length,out=np.full_like(angle,2.),where=length>1e-12)
        return q[...,:3]*scale[...,None]

    @classmethod
    def seed_from_reference(cls, reference_at, break_seconds, *, source_seconds):
        """Sample current state once as initialization; it is not a fixed prior.

        This is an initial approximation, NOT a source-accuracy-preserving fit.
        Measure its native reprojection before/after optimization. For fidelity,
        initialize/fine-fit these controls against the existing saved sequence
        with finite weights; do not silently replace accepted production states.

        source_seconds must contain the full available native sequence, not a
        downsampled control grid. A control interval with observed rotation-path
        length >= pi is rejected as under-resolved; increase control resolution.
        This is a representation check, not a physiological speed limit. Even
        this check cannot recover turns already missed by the source sampling.
        """
        k = clamped_cubic_knots(break_seconds)
        t = greville_times(k)
        native = np.asarray(source_seconds, float)
        if (native.ndim != 1 or len(native) < 2 or not np.isfinite(native).all()
                or np.any(np.diff(native) <= 0)):
            raise ValueError("Supply the full ordered native source clock for rotation-path checks")
        dense = np.unique(np.r_[t, native[(native >= t[0]) & (native <= t[-1])]])
        dense_state = reference_at(dense)
        dense_rot = np.concatenate((np.asarray(dense_state['Q'])[:, None], np.asarray(dense_state['local_R'])), axis=1)
        angles = np.linalg.norm(_log(dense_rot[:-1].swapaxes(-1, -2) @ dense_rot[1:]), axis=-1)
        for a, b in zip(t[:-1], t[1:]):
            interval = (dense[:-1] >= a) & (dense[1:] <= b)
            if np.any(angles[interval].sum(axis=0) >= np.pi-1e-5):
                raise ValueError("Control interval misses observed rotation path; subdivide before seeding")
        s = reference_at(t)
        rotations = np.concatenate((np.asarray(s['Q'])[:, None], np.asarray(s['local_R'])), axis=1)
        return cls(k, s['P'], rotations)

    def basis_sparse(self, seconds, derivative=0):
        """Exact cubic basis derivatives with at most four supported controls.

        Each differentiation maps adjacent coefficients by
        ``c'_i = k * (c_{i+1} - c_i) / (t_{i+k+1} - t_{i+1})``.
        Applying that sparse map to the lower-degree design matrix avoids an
        M-by-M identity coefficient array, including for derivative orders 1..3.
        """
        t = np.atleast_1d(np.asarray(seconds, float))
        if (t.ndim != 1 or not np.isfinite(t).all()
                or np.any(t < self.knots[3]) or np.any(t > self.knots[-4])):
            raise ValueError("Evaluation requires source times inside the archived domain, including camera readout padding")
        if derivative not in (0, 1, 2, 3):
            raise ValueError("Unsupported derivative")
        if len(t) == 0:
            return csr_matrix((0, len(self.position_controls)), dtype=float)
        derivative = int(derivative)
        knots = self.knots
        degree = 3
        coefficients = None
        for _ in range(derivative):
            count = len(knots)-degree-1
            scale = degree/(knots[degree+1:degree+count]-knots[1:count])
            difference = diags((-scale, scale), (0, 1), shape=(count-1, count), format='csr')
            coefficients = difference if coefficients is None else difference @ coefficients
            knots = knots[1:-1]
            degree -= 1
        basis = csr_matrix(BSpline.design_matrix(t, knots, degree, extrapolate=False))
        if coefficients is not None:
            basis = basis @ coefficients
        basis.eliminate_zeros()
        return basis

    def basis(self, seconds, derivative=0):
        """Dense compatibility interface; production callers can use CSR."""
        return self.basis_sparse(seconds, derivative).toarray()

    def evaluate(self, seconds):
        b = self.basis(seconds)
        steps = self._check_log_chart()
        r = self.rotation_controls
        out = np.empty((len(b), r.shape[1], 3, 3))
        for n, row in enumerate(b):
            nz = np.flatnonzero(row != 0.)
            first, last = int(nz[0]), int(nz[-1])
            out[n] = r[first]
            # Cumulative basis: beta_j = sum of ordinary basis weights j..last.
            for j in range(first+1, last+1):
                beta = float(row[j:last+1].sum())
                out[n] = out[n] @ _exp(beta*steps[j-1])
        return dict(P=b @ self.position_controls, Q=out[:, 0], local_R=out[:, 1:], rotations=out)

    def position_derivative(self, seconds, order):
        return self.basis_sparse(seconds, order) @ self.position_controls

    def retracted(self, position_increment, rotation_increment):
        """New optimization state, with per-control right SO(3) increments."""
        dp = np.asarray(position_increment, float)
        dr = np.asarray(rotation_increment, float)
        if dp.shape != self.position_controls.shape or dr.shape != self.rotation_controls.shape[:-2]+(3,):
            raise ValueError("Increment shape differs from absolute control layout")
        return AbsoluteMotionCurve(self.knots, self.position_controls+dp, self.rotation_controls @ _exp(dr),self.edge_reference_quaternion)

    def rotation_control_jacobian(self, second, track=0, epsilon=1e-6):
        """Return {control_index: 3x3 right-tangent block}, for one time/track.

        Only the at-most-four supported control orientations are perturbed.
        This is a reference checker, not the production batched Jacobian.
        """
        if not 0 <= track < self.rotation_controls.shape[1] or epsilon <= 0:
            raise ValueError("Invalid rotation track or numerical difference step")
        row = self.basis([second])[0]
        controls = np.flatnonzero(row != 0.)
        base = self.evaluate([second])['rotations'][0, track]
        blocks = {}
        for c in controls:
            block = np.empty((3, 3))
            for axis in range(3):
                dr = np.zeros(self.rotation_controls.shape[:-2]+(3,))
                dr[c, track, axis] = epsilon
                plus = self.retracted(np.zeros_like(self.position_controls), dr).evaluate([second])['rotations'][0, track]
                minus = self.retracted(np.zeros_like(self.position_controls), -dr).evaluate([second])['rotations'][0, track]
                block[:, axis] = (_log(base.T @ plus)-_log(base.T @ minus))/(2*epsilon)
            blocks[int(c)] = block
        return blocks

    def packet(self):
        """Archive exact controls; export and player must evaluate this same state."""
        packet=dict(state_schema=np.array('rinkspace.absolute-motion-reference.v1' if self.edge_reference_quaternion is None else 'rinkspace.absolute-motion-reference.v2'),
                    knots=self.knots.copy(), position_controls=self.position_controls.copy(),
                    rotation_controls=self.rotation_controls.copy())
        if self.edge_reference_quaternion is not None:
            packet['edge_reference_quaternion']=self.edge_reference_quaternion.copy()
        return packet


def actual_angular_motion(rotations, seconds):
    """Actual world/parent-frame omega and its change, not correction derivatives.

    R must map segment-local coordinates into a common frame (world for Q or
    segment world R; parent coordinates for local R). Segment-local velocities
    from different parents must not be mixed into whole-body angular momentum.
    Returns interval velocities and midpoint acceleration. Phase masks belong
    to the caller. No constant-speed or maximum-speed acceptance rule is used.
    """
    r = np.asarray(rotations, float)
    t = np.asarray(seconds, float)
    if (r.shape != (len(t), 3, 3) or len(t) < 3
            or not np.isfinite(t).all() or np.any(np.diff(t) <= 0)
            or not np.isfinite(r).all()):
        raise ValueError("Expected increasing time and aligned rotations")
    if (np.max(np.abs(r.swapaxes(-1, -2) @ r-np.eye(3))) > 1e-7
            or np.max(np.abs(np.linalg.det(r)-1.)) > 1e-7):
        raise ValueError("Input rotations must already be SO(3), not silently projected")
    dt = np.diff(t)
    midpoint = (t[:-1]+t[1:])/2
    omega = _log(r[1:] @ r[:-1].swapaxes(-1, -2))/dt[:, None]
    alpha = np.diff(omega, axis=0)/np.diff(midpoint)[:, None]
    return dict(interval_time=midpoint, omega=omega,
                acceleration_time=(midpoint[:-1]+midpoint[1:])/2, alpha=alpha)
