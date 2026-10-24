"""Gauge-free physical print coordinates, with the original allowed domain.

A print-only ad has no observed outer installation rectangle. Its installation
is an unobservable coordinate chart. Optimize six physical coefficients per
print block: (s_u,s_v,s_0,h_u,h_v,h_0), and no ad chart coefficients.

The legacy evaluator remains unchanged: decode creates a legal legacy witness
and marks the five witness ad blocks constant. These storage values may change
between evaluations; they are never independent optimizer variables. The
original box image is an existential domain, not the old q box at a fixed ad.
"""
from __future__ import annotations
from copy import deepcopy
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from .native_scene import NativeScene, parameter_values

VERSION = 'rinkfit.physical_print_affine_chart.v1'


def effective_affine(ad, q):
    ad = np.asarray(ad, float); q = np.asarray(q, float)
    out = q * np.array([ad[1]] * 3 + [ad[3] - ad[2]] * 3)
    out[[2, 5]] += ad[[0, 2]]
    return out


def local_affine(ad, physical):
    ad = np.asarray(ad, float); out = np.array(physical, float, copy=True)
    if ad[1] <= 0 or ad[3] <= ad[2]:
        raise ValueError('Noninvertible advertisement witness chart')
    out[[2, 5]] -= ad[[0, 2]]
    return out / np.array([ad[1]] * 3 + [ad[3] - ad[2]] * 3)


def _box_inequalities(A, b, row, lo, hi, offset=0.):
    # lo <= row @ witness + offset <= hi
    if np.isfinite(hi): A.append(row); b.append(hi - offset)
    if np.isfinite(lo): A.append(-np.asarray(row)); b.append(offset - lo)


class PrintOnlyAdChart:
    """Reduced physical coordinates and exact legacy-domain witness decoder.

    pack/unpack and tangent/pullback_jacobian are the solver API. Every trial
    must pass unpack(validate=True). Bounds on the former print q coordinates
    MUST NOT be reused as bounds on these physical coordinates. The LP witness
    constraints below express the complete old box image, including shared
    installation bounds, positive finite support, and sourced canvas bounds.
    The legacy q determinant threshold is checked on a legal common witness.
    """
    def __init__(self, model, ad_ids=(8, 9, 10, 11, 12), *, determinant_min=1e-6):
        self.template = deepcopy(model)
        self.determinant_min = float(determinant_min)
        if not np.isfinite(self.determinant_min) or self.determinant_min <= 0: raise ValueError('Positive original production determinant threshold required')
        self.offset = np.r_[0, np.cumsum([len(b['values']) for b in model['blocks']])]
        names = {b['name']: i for i, b in enumerate(model['blocks'])}
        self.ads = {}; self.print_ad = {}; self.paths = {}
        current_scope=model.get('current_input_scope',{}).get('schema')=='rinkfit.current_advertisement_input.v1'
        chart_materials=[m for m in model['materials'] if not(current_scope and m.get('unused_legacy_witness',False))]
        for ad_id in ad_ids:
            name = 'ad/' + str(ad_id)
            if name not in names: raise ValueError('Missing print-only chart ' + name)
            ai = names[name]; paths = []
            for material in chart_materials:
                for path in material['paths']:
                    if path.get('advertisement_block') == ai:
                        if path.get('kind') not in (2, 'board_path') or path.get('geometry_block', -1) < 0:
                            raise ValueError('Cannot remove an observed installation: ' + name)
                        qi = int(path['geometry_block'])
                        if len(model['blocks'][qi]['values']) != 6: raise ValueError('Expected print affine6')
                        if qi in self.print_ad and self.print_ad[qi] != ai:
                            raise ValueError('Print block shared across different physical ads')
                        self.print_ad[qi] = ai; self.paths.setdefault(qi, []).append(path); paths.append(path)
            if not paths: raise ValueError('No finite print paths for ' + name)
            self.ads[ai] = sorted({int(p['geometry_block']) for p in paths})
        # An affine shared with a non-target ad would invalidate the quotient.
        for material in chart_materials:
            for path in material['paths']:
                qi = int(path.get('geometry_block', -1))
                if qi in self.print_ad and path.get('advertisement_block') != self.print_ad[qi]:
                    raise ValueError('Print affine has a non-chart use')
        old_free = []
        for bi, block in enumerate(model['blocks']):
            if block.get('constant', False): continue
            old_free.extend(int(self.offset[bi] + j) for j in range(len(block['values']))
                            if j not in block.get('fixed_indices', []))
        self.old_free = np.asarray(old_free, int)
        removed = {int(self.offset[i] + j) for i in self.ads for j in range(4)}
        self.free = np.asarray([j for j in old_free if j not in removed], int)
        self.column = {int(j): i for i, j in enumerate(self.free)}
        for qi in self.print_ad:
            if any(int(self.offset[qi] + j) not in self.column for j in range(6)):
                raise ValueError('Print-only affine must retain its six physical freedoms')
        self.removed_free = len(self.old_free) - len(self.free)

    def pack(self, model=None, state=None):
        model = self.template if model is None else model
        values = parameter_values(model, state)
        flat = np.concatenate(values); x = flat[self.free].copy()
        for qi, ai in self.print_ad.items():
            ids = [self.column[int(self.offset[qi] + j)] for j in range(6)]
            x[ids] = effective_affine(values[ai], values[qi])
        return x

    def _physical(self, x, qi):
        return np.asarray(x)[[self.column[int(self.offset[qi] + j)] for j in range(6)]]

    def _system(self, ai, physical, perimeter, height, vertical):
        """A*witness <= rhs; witness is (s0,span) or (bottom,top-bottom)."""
        definition = self.template['blocks'][ai]
        lo = np.asarray(definition.get('lower', [-np.inf] * 4), float)
        hi = np.asarray(definition.get('upper', [np.inf] * 4), float)
        A = []; rhs = []
        if not vertical:
            _box_inequalities(A, rhs, np.array([1., 0.]), lo[0], hi[0])
            _box_inequalities(A, rhs, np.array([0., 1.]), max(0., lo[1]), min(perimeter, hi[1]))
        else:
            _box_inequalities(A, rhs, np.array([1., 0.]), max(0., lo[2]), hi[2])
            _box_inequalities(A, rhs, np.array([1., 1.]), lo[3], min(height, hi[3]))
            _box_inequalities(A, rhs, np.array([0., 1.]), 0., np.inf)
        for qi in self.ads[ai]:
            qdef = self.template['blocks'][qi]
            qlo = np.asarray(qdef.get('lower', [-np.inf] * 6), float)
            qhi = np.asarray(qdef.get('upper', [np.inf] * 6), float)
            eff = physical[qi]; start = 3 if vertical else 0
            for j in range(3):
                # qlo * scale <= eff - origin(if translation) <= qhi * scale
                c = np.array([1. if j == 2 else 0., qlo[start + j]])
                if np.isfinite(qlo[start + j]): A.append(c); rhs.append(eff[start + j])
                c = np.array([-1. if j == 2 else 0., -qhi[start + j]])
                if np.isfinite(qhi[start + j]): A.append(c); rhs.append(-eff[start + j])
            for path in self.paths[qi]:
                points = np.asarray(path['points'], float)
                h = points @ eff[start:start + 2] + eff[start + 2]
                if vertical and (np.min(h) < -1e-12 or np.max(h) > height + 1e-12):
                    raise ValueError('Physical finite print support leaves board')
                if 'physical_canvas_bounds' in path:
                    if not path.get('physical_canvas_provenance'):
                        raise ValueError('Canvas bounds lack physical provenance')
                    bounds = np.asarray(path['physical_canvas_bounds'], float)
                    j = int(vertical)
                    for z in h:
                        A.append(np.array([1., bounds[0, j]])); rhs.append(z)
                        A.append(np.array([-1., -bounds[1, j]])); rhs.append(-z)
        return np.asarray(A, float).reshape(-1, 2), np.asarray(rhs, float)

    @staticmethod
    def _solve_witness(A, rhs, preferred):
        preferred = np.asarray(preferred, float)
        # The strict numerical audit is performed after inverse conversion.
        if preferred[1] > 0 and np.all(A @ preferred <= rhs + 2e-13):
            return preferred, 'preserved_current_witness'
        result = linprog([0., 1.], A_ub=A, b_ub=rhs, bounds=[(None, None)] * 2,
                         method='highs', options={'dual_feasibility_tolerance': 1e-9,
                                                  'primal_feasibility_tolerance': 1e-9})
        if not result.success or result.x[1] <= 0:
            raise ValueError('Physical affine has no legal original chart witness')
        return result.x, 'minimum_scale_LP_witness'

    def unpack(self, x, *, validate=True, preferred_model=None, return_domain=False):
        x = np.asarray(x, float)
        if x.shape != (len(self.free),) or not np.isfinite(x).all():
            raise ValueError('Incomplete or nonfinite reduced physical coordinates')
        model = deepcopy(self.template); flat = np.concatenate([b['values'] for b in model['blocks']]).astype(float)
        flat[self.free] = x
        for i, block in enumerate(model['blocks']): block['values'] = flat[self.offset[i]:self.offset[i + 1]].tolist()
        # Geometry dimensions may move, so perimeter and height are current.
        boundary = NativeScene(model); preferred = self.template if preferred_model is None else preferred_model
        domain = []
        for ai, qids in self.ads.items():
            physical = {qi: self._physical(x, qi) for qi in qids}
            old = np.asarray(preferred['blocks'][ai]['values'], float)
            H, hr = self._system(ai, physical, boundary.perimeter, boundary.height, False)
            V, vr = self._system(ai, physical, boundary.perimeter, boundary.height, True)
            hp, hmethod = self._solve_witness(H, hr, old[:2])
            vp, vmethod = self._solve_witness(V, vr, [old[2], old[3] - old[2]])
            det = np.array([a[0] * a[4] - a[1] * a[3] for a in physical.values()])
            # If a preferred witness is determinant-infeasible, the independent
            # minimum horizontal/vertical scales minimize their product too.
            if np.any(det < self.determinant_min * hp[1] * vp[1]):
                hp, hmethod = self._solve_witness(H, hr, [0., -1.])
                vp, vmethod = self._solve_witness(V, vr, [0., -1.])
            if np.any(det < self.determinant_min * hp[1] * vp[1]):
                raise ValueError('Original positive print determinant domain violated')
            witness = np.array([hp[0], hp[1], vp[0], vp[0] + vp[1]])
            model['blocks'][ai]['values'] = witness.tolist()
            model['blocks'][ai]['constant'] = True
            for qi in qids: model['blocks'][qi]['values'] = local_affine(witness, physical[qi]).tolist()
            domain.append(dict(ad=model['blocks'][ai]['name'], witness=witness.tolist(),
                               horizontal_method=hmethod, vertical_method=vmethod,
                               max_horizontal_inequality=float(np.max(H @ hp - hr, initial=0.)),
                               max_vertical_inequality=float(np.max(V @ vp - vr, initial=0.))))
        model.setdefault('provenance', {})['print_only_ad_chart'] = dict(
            version=VERSION, removed_free_scalars=self.removed_free,
            physical_coordinates='per print: s_u,s_v,s_0,h_u,h_v,h_0',
            allowed_domain='exact image of original ad/affine boxes; common LP witness; legacy determinant and finite material support',
            storage_ad_values='dependent domain witness only; not frozen physical advertisement placement',
            requires_trial_domain_decoder=True)
        if validate:
            result = NativeScene(model).check_physics(tolerance=1e-9)
            if not result['passed']: raise ValueError('Decoded full physical state violates original domain: ' + ','.join(result['violations']))
        return (model, domain) if return_domain else model

    def tangent(self, model=None):
        """raw legacy differential = T @ reduced physical differential.

        This is a horizontal lift with each decoded witness held fixed. A
        change of witness adds pure gauge directions, so J@T is exactly the
        derivative for geometric/projection residuals. It is NOT the derivative
        of a raw chart-bound constraint: those require the domain decoder.
        """
        model = self.template if model is None else model
        values = parameter_values(model); weights = np.ones(len(self.free))
        for qi, ai in self.print_ad.items():
            ad = values[ai]
            ids = [self.column[int(self.offset[qi] + j)] for j in range(6)]
            weights[ids] = 1. / np.array([ad[1]] * 3 + [ad[3] - ad[2]] * 3)
        return sparse.csr_matrix((weights, (self.free, np.arange(len(self.free)))),
                                 shape=(int(self.offset[-1]), len(self.free)))

    def pullback_jacobian(self, J, model=None):
        """Use on the raw ambient geometric/projection Jacobian, before scaling."""
        if J.shape[1] != int(self.offset[-1]): raise ValueError('Expected full raw ambient Jacobian')
        return J @ self.tangent(model)

    def report(self):
        return dict(version=VERSION, raw_stored_scalars=int(self.offset[-1]),
                    old_free_scalars=len(self.old_free), reduced_free_scalars=len(self.free),
                    eliminated_chart_scalars=self.removed_free,
                    ads=[self.template['blocks'][i]['name'] for i in self.ads],
                    print_affine_blocks=len(self.print_ad), original_production_determinant_min=self.determinant_min,
                    domain='original box-image feasibility, determinant and finite-support checks; no reused fixed-chart affine box',
                    engine='unchanged NativeScene/NativeCamera or existing C++ evaluator after unpack')
