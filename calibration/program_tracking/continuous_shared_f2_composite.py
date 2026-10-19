"""Opt-in composition of saved camera rigs around one released F2 scene delta.

No optimizer is started here. Captured original callbacks retain their pixel
weights, robust transforms, and factor noise. The new graph state is a delta
from the candidate scene, while the captured callback still receives its
original absolute displacement from the released scene.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import gtsam
import numpy as np

from continuous_brown_adapter import adapt_factor
from fit_continuous_camera import raw_spec


F2_KEY = 2001
F2_ORIGINAL_FILE = 'shared_markers.py'
F2_ORIGINAL_FUNCTION = 'add_shared_ring_factor'


def _state11(value, label):
    result = np.asarray(value, np.float64)
    if result.shape != (11,) or not np.isfinite(result).all():
        raise ValueError(f'{label} must be finite Vector11')
    return result


def captured_f2_spec(spec):
    """Check the original graph provenance rather than infer identity by PTS."""
    return (spec['kind'] == 'observation'
            and spec['file'] == F2_ORIGINAL_FILE
            and spec['function'] == F2_ORIGINAL_FUNCTION
            and len(spec['keys']) == 2
            and int(spec['keys'][1]) == F2_KEY)


def shift_absolute_f2_callback(spec, original_scene_state):
    """Translate only the scene argument from candidate-relative to original.

    Derivative d(q_original+delta)/d(delta) is identity. No source callback,
    robust weight, sigma, or camera argument is approximated or rebuilt.
    """
    if not captured_f2_spec(spec):
        raise ValueError('This is not a captured original F2 observation')
    base = _state11(original_scene_state, 'original F2 scene state').copy()
    raw, jac, options = raw_spec(spec)
    if jac is None:
        raise ValueError('Captured F2 factor lacks its original Jacobian')
    original_keys = tuple(map(int, spec['keys']))
    def absolute(q):
        if len(q) != 2:
            raise ValueError('Captured F2 callback needs camera and scene')
        return [q[0], _state11(q[1], 'F2 candidate delta') + base]
    def residual(q):
        return raw(absolute(q))
    def derivative(q):
        return jac(absolute(q))
    return original_keys, residual, derivative, options


def adapt_captured_f2(spec, model, active_keys, original_fixed_states,
                      original_scene_state, *, release_scene):
    """Adapt the exact captured factor to a Brown15 rig, with optional F2 delta.

    With release_scene=False this is the unchanged camera-only original factor.
    With release_scene=True, graph key 2001 represents candidate-scene delta.
    """
    if not captured_f2_spec(spec):
        raise ValueError('Expected original F2 factor')
    active = set(map(int, active_keys))
    if release_scene:
        if F2_KEY not in active:
            raise ValueError('Released F2 scene key must be active')
        keys, raw, jac, options = shift_absolute_f2_callback(
            spec, original_scene_state)
    else:
        if F2_KEY in active:
            raise ValueError('Fixed F2 scene key must not be active')
        keys = tuple(map(int, spec['keys']))
        raw, jac, options = raw_spec(spec)
    fixed = dict(original_fixed_states)
    fixed[F2_KEY] = _state11(original_scene_state, 'original F2 scene state')
    result = adapt_factor(keys, raw, jac, options, active, fixed, model)
    if result is None:
        raise ValueError('F2 factor lost its camera and scene dependencies')
    return result


def scene_prior_in_candidate_coordinates(original_scene_state,
                                         sigma: np.ndarray):
    """Original zero-centered prior is centered at -q_saved in delta space."""
    base = _state11(original_scene_state, 'original F2 scene state')
    width = _state11(sigma, 'scene prior sigma')
    if np.any(width <= 0):
        raise ValueError('Scene prior sigma must be positive')
    return -base.copy(), width.copy()


@dataclass
class RekeyedNativeObjective:
    """Exact cost-only bridge for a native factor with internal old-key lookup.

    This is deliberately not a GTSAM NonlinearFactor and cannot be optimized.
    Production must rebuild native factors from each rekeyed rig model.
    """
    original: object
    old_keys: tuple[int, ...]
    new_keys: tuple[int, ...]

    def keys(self):
        return tuple(gtsam.symbol('x', key) for key in self.new_keys)

    def dim(self):
        return self.original.dim()

    def error(self, values):
        old_values = gtsam.Values()
        for old_key, new_key in zip(self.old_keys, self.new_keys):
            old_values.insert(gtsam.symbol('x', old_key),
                              values.atVector(gtsam.symbol('x', new_key)))
        return self.original.error(old_values)


@dataclass
class CompositeFactorTape:
    """Relabel complete pre-built rig graphs without changing their factors.

    This is an executable *objective gate* for existing full graph builders.
    It copies every prior, between factor, and custom observation, including
    all existing camera priors. A specific original F2 factor can be replaced
    by its adapted candidate-delta factor at its checked index. Native C++
    factors use cost-only proxies because their internal key lookup is not
    changed by GTSAM's generic rekey. This tape cannot be optimized. It
    deliberately does not call update() or optimize().
    """
    factors: list
    callbacks: list
    names: list

    def __init__(self):
        self.factors, self.callbacks, self.names = [], [], []

    @staticmethod
    def _symbol(key):
        return gtsam.symbol('x', int(key))

    @staticmethod
    def _factor_keys(factor):
        return tuple(int(gtsam.Symbol(key).index()) for key in factor.keys())

    def append_complete_rig(self, name, graph, key_mapping: Mapping[int, int],
                            *, replacements=None):
        """Append all factors; replacements map old factor index to adapted tuple.

        ``replacements[i]`` is (keys, residual, jacobian, options), in *old*
        graph-key coordinates. It must evaluate exactly like old factor at
        candidate delta zero before this method is called.
        """
        replacements = replacements or {}
        callback_iter = iter(graph._callbacks)
        before = len(self.factors)
        seen = set()
        for index, factor in enumerate(graph._pending):
            old_keys = self._factor_keys(factor)
            new_keys = tuple(key_mapping.get(key, key) for key in old_keys)
            if len(set(new_keys)) != len(new_keys):
                raise ValueError(f'Rekeyed factor {index} merged graph keys')
            if isinstance(factor, gtsam.CustomFactor):
                cb_keys, old_callback = next(callback_iter)
                if tuple(cb_keys) != old_keys:
                    raise ValueError(f'Custom factor/callback ordering differs at {index}')
                if index in replacements:
                    raw_keys, residual, jacobian, options = replacements[index]
                    target_keys = tuple(key_mapping.get(int(key), int(key))
                                        for key in raw_keys)
                    noise = factor.noiseModel()
                    if int(options['residual_dim']) != int(factor.dim()):
                        raise ValueError('Replacement F2 dimension differs')
                    # The replacement is already source weighted and robust;
                    # retain the actual old factor's GTSAM noise object too.
                    def callback(_factor, values, matrices, ids=target_keys,
                                 fn=residual, jf=jacobian):
                        q = [np.asarray(values.atVector(self._symbol(k)),
                                        np.float64).copy() for k in ids]
                        out = np.asarray(fn(q), np.float64).reshape(-1)
                        if matrices is not None:
                            for j, block in enumerate(jf(q)):
                                matrices[j] = np.asfortranarray(block)
                        return out
                    new_factor = graph._custom_factor(noise,
                        [self._symbol(k) for k in target_keys], callback)
                    new_keys = target_keys
                    seen.add(index)
                else:
                    def callback(_factor, values, matrices, ids=old_keys,
                                 targets=new_keys, fn=old_callback,
                                 previous=factor):
                        old_values = gtsam.Values()
                        for old_key, new_key in zip(ids, targets):
                            old_values.insert(self._symbol(old_key),
                                              values.atVector(self._symbol(new_key)))
                        return fn(previous, old_values, matrices)
                    new_factor = graph._custom_factor(factor.noiseModel(),
                        [self._symbol(k) for k in new_keys], callback)
                self.callbacks.append(callback)
            elif isinstance(factor, gtsam.PriorFactorVector):
                if index in replacements:
                    raise ValueError('A prior cannot be replaced by an F2 observation')
                new_factor = gtsam.PriorFactorVector(
                    self._symbol(new_keys[0]), factor.prior(), factor.noiseModel())
            elif isinstance(factor, gtsam.BetweenFactorVector):
                if index in replacements:
                    raise ValueError('A between factor cannot be replaced by F2')
                new_factor = gtsam.BetweenFactorVector(
                    self._symbol(new_keys[0]), self._symbol(new_keys[1]),
                    factor.measured(), factor.noiseModel())
            else:
                if index in replacements:
                    raise ValueError('Only captured custom F2 may be replaced')
                new_factor = RekeyedNativeObjective(factor, old_keys, new_keys)
            self.factors.append(new_factor)
            self.names.append((name, index))
        try:
            next(callback_iter)
        except StopIteration:
            pass
        else:
            raise ValueError('Unused source custom callbacks')
        if seen != set(replacements):
            raise ValueError('One or more F2 replacement indices were not used')
        return dict(name=name, original_factors=len(graph._pending),
                    appended_factors=len(self.factors)-before,
                    candidate_f2_replacements=len(seen))

    def append_original_f2_scene_prior(self, original_scene_state,
                                       *, outer_radius, middle_radius,
                                       center_sigma_scene=1.,
                                       radius_sigma_scene=.3):
        """Retain released-scene gauge when F2 becomes an active candidate delta.

        The source graph's prior target was zero in released-scene coordinates.
        Candidate-relative coordinates therefore require target ``-q_saved``.
        The radius guard also evaluates ``outer_base+100*(q_saved+delta)[2]``.
        """
        base = _state11(original_scene_state, 'original F2 state').copy()
        sigma = np.r_[[center_sigma_scene/100.]*2,
                      radius_sigma_scene/100., np.full(8, 1e-9)]
        target, sigma = scene_prior_in_candidate_coordinates(base, sigma)
        symbol = self._symbol(F2_KEY)
        noise = gtsam.noiseModel.Diagonal.Sigmas(sigma)
        self.factors.append(gtsam.PriorFactorVector(symbol, target, noise))
        self.names.append(('shared_f2_original_prior', 0))
        def guard(_factor, values, matrices):
            radius = float(outer_radius)+100.*(
                base[2]+values.atVector(symbol)[2])
            result = np.array([max(0., float(middle_radius)+.05-radius)])
            if matrices is not None:
                jac = np.zeros((1, 11))
                if radius < float(middle_radius)+.05:
                    jac[0, 2] = -100.
                matrices[0] = np.asfortranarray(jac)
            return result
        guard_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([.001]))
        self.factors.append(gtsam.CustomFactor(guard_noise, [symbol], guard))
        self.callbacks.append(guard)
        self.names.append(('shared_f2_original_radius_guard', 0))
        return dict(scene_key=F2_KEY, prior_target=target.tolist(),
                    prior_sigma=sigma.tolist(),
                    released_outer_radius=float(outer_radius),
                    released_middle_radius=float(middle_radius))

