"""Optional source integration weights outside the existing Huber2 loss."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numpy as np


def checked_loss_weights(weights, rows):
    if weights is None:
        return None
    weights = np.asarray(weights, float)
    if weights.shape != (rows,) or not np.isfinite(weights).all() or np.any(weights < 0.):
        raise ValueError('Finite nonnegative per-scalar source loss weights required')
    return weights


@dataclass(frozen=True)
class SourceLossWeights:
    """Once-checked immutable sampling integration and prior scale removal.

    Immutable bytes back the arrays; their identity is hashed once. Streaming
    passes need no additional full-source policy scan or Jacobian copy.
    """
    integration_weights: object
    working_normalization_multiplier: object = None

    def __post_init__(self):
        weights = np.asarray(self.integration_weights, dtype=np.float64)
        if weights.ndim != 1:
            raise ValueError('One-dimensional source integration weights required')
        checked_loss_weights(weights, len(weights))
        weights = np.frombuffer(weights.tobytes(), dtype=np.float64)
        normalization = self.working_normalization_multiplier
        if normalization is not None:
            normalization = np.asarray(normalization, dtype=np.float64)
            if (normalization.shape != weights.shape or not np.isfinite(normalization).all()
                    or np.any(normalization <= 0.)):
                raise ValueError('Finite positive explicit working normalization required')
            normalization = np.frombuffer(normalization.tobytes(), dtype=np.float64)
        object.__setattr__(self, 'integration_weights', weights)
        object.__setattr__(self, 'working_normalization_multiplier', normalization)
        digest = hashlib.sha256(b'rinkspace.source-integration-and-working-scale.v1')
        digest.update(str(weights.shape).encode('ascii')); digest.update(weights.data)
        if normalization is not None: digest.update(normalization.data)
        object.__setattr__(self, 'identity_digest', digest.digest())

    def __reduce__(self):
        # Pickle/worker transport must reconstruct immutable backing bytes.
        return type(self), (self.integration_weights, self.working_normalization_multiplier)


def loss_weight_arrays(loss_weights, rows):
    if isinstance(loss_weights, SourceLossWeights):
        if loss_weights.integration_weights.shape != (rows,):
            raise ValueError('Source integration scalar layout changed')
        return loss_weights.integration_weights, loss_weights.working_normalization_multiplier
    return checked_loss_weights(loss_weights, rows), None


def weighted_source_cost(residual, quadratic, loss_weights):
    """Original component Huber2/quadratic, then fixed integration weight."""
    residual = np.asarray(residual, float)
    quadratic = np.asarray(quadratic, bool)
    weights, normalization = loss_weight_arrays(loss_weights, len(residual))
    if residual.ndim != 1 or quadratic.shape != residual.shape or not np.isfinite(residual).all():
        raise ValueError('Finite original source residual and quadratic row inventory required')
    if normalization is not None:
        residual = residual * normalization
    absolute = abs(residual)
    components = np.where(absolute <= 2., .5 * residual**2, 2. * absolute - 2.)
    components[quadratic] = .5 * residual[quadratic]**2
    return float(components.sum() if weights is None else weights @ components)


def source_irls_weights(residual, quadratic, loss_weights=None):
    """Original IRLS coefficients, multiplied after robust classification."""
    residual = np.asarray(residual, float)
    quadratic = np.asarray(quadratic, bool)
    integration, normalization = loss_weight_arrays(loss_weights, len(residual))
    working = residual if normalization is None else residual * normalization
    weights = np.minimum(1., 2. / np.maximum(abs(working), 1e-12))
    weights[quadratic] = 1.
    if normalization is not None:
        weights *= normalization**2
    if integration is not None:
        weights *= integration
    return weights


def source_loss_checkpoint_arrays(loss_weights, rows):
    """Optional block payload; old unweighted NPZ layouts remain unchanged."""
    if loss_weights is None: return {}
    integration, normalization = loss_weight_arrays(loss_weights, rows)
    values = dict(source_loss_integration_weights=integration)
    if normalization is not None:
        values['source_loss_working_normalization_multiplier'] = normalization
    return values


def source_loss_from_checkpoint(stored):
    if 'source_loss_integration_weights' not in stored:
        if 'source_loss_working_normalization_multiplier' in stored:
            raise ValueError('Source loss checkpoint normalization lacks integration')
        return None
    return SourceLossWeights(stored['source_loss_integration_weights'],
        stored['source_loss_working_normalization_multiplier'] if 'source_loss_working_normalization_multiplier' in stored else None)
