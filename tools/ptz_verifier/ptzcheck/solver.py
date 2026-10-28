"""Image-only rotation/zoom models; never reads production geometry or pose."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from collections import deque
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix
from scipy.linalg import svd
from .geometry import symmetric_errors, error_stats, nearest_rotation, K, homography_points
from .io import write_json

MODELS = {'square_center': (), 'square_pp': ('cx_fraction', 'cy_fraction'),
          'free_pp': ('cx_fraction', 'cy_fraction', 'log_aspect'),
          'free_pp_radial': ('cx_fraction', 'cy_fraction', 'log_aspect', 'kappa')}

def subset(data, mask): return {k: v[mask] for k, v in data.items()}

@dataclass
class Layout:
    n: int
    width: int
    height: int
    family: str
    fixed_eta: float | None = None
    fixed_optics: np.ndarray | None = None
    anchors: list[int] | None = None

    def __post_init__(self):
        if self.family not in MODELS and self.family != 'fixed_optics': raise ValueError(self.family)
        self.anchors = list(self.anchors or [0])
        if len(set(self.anchors)) != len(self.anchors) or any(a < 0 or a >= self.n for a in self.anchors): raise ValueError('Invalid rotation anchors')
        self.rot_frames = [i for i in range(self.n) if i not in self.anchors]
        self.rot_index = {i: k for k, i in enumerate(self.rot_frames)}
        self.nr = 3*len(self.rot_frames)
        self.nf = 0 if self.fixed_optics is not None else self.n
        self.globals = [g for g in MODELS.get(self.family, ()) if not (g == 'log_aspect' and self.fixed_eta is not None)]
        self.names = [f'R{i}_{axis}' for i in self.rot_frames for axis in 'xyz']
        self.names += [f'log_f_over_width_{i}' for i in range(self.n)] if self.nf else []
        self.names += self.globals
        self.size = len(self.names)

    def unpack(self, x):
        R = np.repeat(np.eye(3)[None], self.n, axis=0)
        R[self.rot_frames] = Rotation.from_rotvec(x[:self.nr].reshape(-1, 3)).as_matrix()
        if self.fixed_optics is not None: return self.fixed_optics, R
        g = dict(zip(self.globals, x[self.nr+self.nf:]))
        f = self.width*np.exp(x[self.nr:self.nr+self.nf])
        eta = g.get('log_aspect', self.fixed_eta if self.fixed_eta is not None else 0.)
        optics = np.zeros((self.n, 8))
        optics[:, 0] = f*np.exp(eta); optics[:, 1] = f
        optics[:, 2] = g.get('cx_fraction', (self.width-1)/(2*self.width))*self.width
        optics[:, 3] = g.get('cy_fraction', (self.height-1)/(2*self.height))*self.height
        # kappa = Brown k1*(W/fy)^2. Diagnostic family, NOT a measured Sony law.
        optics[:, 4] = g.get('kappa', 0.)*(f/self.width)**2
        return optics, R

    def bounds(self):
        lo = np.full(self.size, -2*np.pi); hi = -lo
        if self.nf:
            lo[self.nr:self.nr+self.nf] = np.log(.2)
            hi[self.nr:self.nr+self.nf] = np.log(30.)
        for j, name in enumerate(self.globals, self.nr+self.nf):
            if name in ('cx_fraction', 'cy_fraction'): lo[j], hi[j] = .05, .95
            elif name == 'log_aspect': lo[j], hi[j] = np.log(.5), np.log(2.)
            elif name == 'kappa': lo[j], hi[j] = -.12, .4
        return lo, hi

    def sparsity(self, data):
        s = lil_matrix((4*len(data['i']), self.size), dtype=np.int8)
        for k, (i, j) in enumerate(zip(data['i'], data['j'])):
            cols = list(range(self.nr+self.nf, self.size))
            for f in (i, j):
                if f in self.rot_index: cols.extend(range(3*self.rot_index[f], 3*self.rot_index[f]+3))
                if self.nf: cols.append(self.nr+f)
            s[4*k:4*k+4, cols] = 1
        return s.tocsr()


def initial(layout, meta, focal_widths, pp_shift=0.):
    """Image-homography initializer; no production camera information."""
    n, w, h = layout.n, layout.width, layout.height
    x = np.zeros(layout.size)
    if layout.nf: x[layout.nr:layout.nr+layout.nf] = np.log(focal_widths)
    for k, name in enumerate(layout.globals, layout.nr+layout.nf):
        if name == 'cx_fraction': x[k] = .5 + pp_shift
        if name == 'cy_fraction': x[k] = .5
    rows, rhs = [], []
    centre = np.array([[w/2, h/2], [w/2+1, h/2], [w/2, h/2+1]])
    adj = [[] for _ in range(n)]
    for e in meta['edges']:
        i, j, H = e['i'], e['j'], np.asarray(e['H'])
        q = homography_points(H, centre)
        A = np.column_stack((q[1]-q[0], q[2]-q[0]))
        scale = np.sqrt(abs(np.linalg.det(A)))
        if not np.isfinite(scale) or scale <= 0: continue
        row = np.zeros(n); row[j], row[i] = 1, -1
        rows.append(row); rhs.append(np.log(np.clip(scale, .1, 10.)))
        adj[i].append((j, H)); adj[j].append((i, np.linalg.inv(H)))
    for a in layout.anchors:
        anchor = np.zeros(n); anchor[a] = 1
        rows.append(anchor); rhs.append(0.)
    ratios = np.linalg.lstsq(np.asarray(rows), rhs, rcond=None)[0]
    if layout.nf: x[layout.nr:layout.nr+layout.nf] += ratios
    lo, hi = layout.bounds(); x = np.clip(x, lo+1e-7, hi-1e-7)
    optics, _ = layout.unpack(x)
    R = np.repeat(np.eye(3)[None], n, axis=0); seen = set(layout.anchors); todo = deque(layout.anchors)
    while todo:
        i = todo.popleft()
        for j, H in adj[i]:
            if j in seen: continue
            Rij = nearest_rotation(np.linalg.solve(K(optics[j]), H @ K(optics[i])))
            R[j] = Rij @ R[i]; seen.add(j); todo.append(j)
    if len(seen) != n: raise ValueError('Disconnected training graph')
    x[:layout.nr] = Rotation.from_matrix(R[layout.rot_frames]).as_rotvec().ravel()
    return x


def fit_one(layout, meta, data, *, focal_widths=3., pp_shift=0., max_nfev=180, start=None):
    train = subset(data, (~data['test']) & data['screened'])
    x0 = initial(layout, meta, focal_widths, pp_shift) if start is None else np.asarray(start)
    lo, hi = layout.bounds(); x0 = np.clip(x0, lo+1e-9, hi-1e-9)
    last_log = [time.perf_counter()]
    def fun(x):
        if time.perf_counter()-last_log[0] > 10:
            print(f'  {layout.family}: optimizer working; bounded to {max_nfev} evaluations', flush=True)
            last_log[0] = time.perf_counter()
        optics, rotations = layout.unpack(x)
        r, valid = symmetric_errors(train, optics, rotations)
        r = np.nan_to_num(r, nan=1e5, posinf=1e5, neginf=-1e5)
        r[~valid] = np.clip(r[~valid], -1e5, 1e5) + 1e4
        return r.ravel()
    then = time.perf_counter()
    result = least_squares(fun, x0, bounds=(lo, hi), jac_sparsity=layout.sparsity(train),
        method='trf', tr_solver='lsmr', loss='soft_l1', f_scale=2., x_scale='jac',
        max_nfev=max_nfev, ftol=1e-8, xtol=1e-9, gtol=1e-6,
        tr_options={'atol': 1e-8, 'btol': 1e-8, 'maxiter': 300})
    optics, rotations = layout.unpack(result.x)
    all_r, valid = symmetric_errors(data, optics, rotations)
    tr = ~data['test'] & data['screened']; te = data['test']; gated = te & data['screened']
    low = np.flatnonzero((result.x-lo) < 1e-4*(hi-lo))
    high = np.flatnonzero((hi-result.x) < 1e-4*(hi-lo))
    summary = dict(family=layout.family, fixed_eta=layout.fixed_eta,
        initial_focal_widths=focal_widths, initial_pp_shift=pp_shift,
        train_robust_cost=float(result.cost), optimizer_success=bool(result.success),
        optimizer_status=int(result.status), optimizer_message=result.message,
        nfev=int(result.nfev), optimality=float(result.optimality), elapsed_s=time.perf_counter()-then,
        train=error_stats(all_r[tr], valid[tr]), holdout_all=error_stats(all_r[te], valid[te]),
        holdout_screened=error_stats(all_r[gated], valid[gated]),
        bounds_hit=[dict(name=layout.names[i], side='lower') for i in low] +
                   [dict(name=layout.names[i], side='upper') for i in high],
        fx_over_fy=float(np.median(optics[:, 0]/optics[:, 1])),
        cx_range_px=[float(optics[:, 2].min()), float(optics[:, 2].max())],
        cy_range_px=[float(optics[:, 3].min()), float(optics[:, 3].max())],
        fy_range_px=[float(optics[:, 1].min()), float(optics[:, 1].max())],
        fitted_globals=dict(zip(layout.globals, result.x[layout.nr+layout.nf:].tolist())),
        physical_acceptance=False, rolling_shutter_compensated=False, translation_modelled=False)
    return dict(summary=summary, x=result.x, optics=optics, rotations=rotations,
                residual=all_r, valid=valid, result=result, layout=layout)


def observability(fit):
    """Local scaled singular values, NOT an IID confidence interval or global proof."""
    layout, opt = fit['layout'], fit['result']
    J = opt.jac.toarray() if hasattr(opt.jac, 'toarray') else opt.jac
    scale = np.full(layout.size, .03)
    scale[layout.nr:layout.nr+layout.nf] = .1
    for k, name in enumerate(layout.globals, layout.nr+layout.nf):
        scale[k] = .1 if name == 'log_aspect' else .05
    A = J*scale
    s = np.linalg.svd(A, compute_uv=False)
    out = dict(kind='local_robust_jacobian_scaled_svd_not_confidence_interval',
        scale=scale, full_singular_values=s, parameter_names=layout.names,
        full_condition=float(s[0]/max(s[-1], 1e-15)), parameter_count=layout.size)
    ng = len(layout.globals)
    if ng:
        N, G = A[:, :-ng], A[:, -ng:]
        U, sn, _ = svd(N, full_matrices=False, check_finite=False)
        good = sn > sn[0]*1e-10
        P = G-U[:, good]@(U[:, good].T@G)
        _, sg, vg = svd(P, full_matrices=False, check_finite=False)
        out.update(shared_names=layout.globals, shared_projected_singular_values=sg,
            shared_condition=float(sg[0]/max(sg[-1], 1e-15)),
            weakest_shared_direction=dict(zip(layout.globals, vg[-1].tolist())))
    return out


def fit_families(graph, out, *, models=None, seeds=(1., 3., 8.), max_nfev=180):
    data, meta = graph['data'], graph['meta']; n = len(meta['frames'])
    result = {}
    for family in (models or ['square_center', 'free_pp', 'free_pp_radial']):
        layout = Layout(n, meta['width'], meta['height'], family, anchors=meta.get('anchors')); runs = []
        for k, seed in enumerate(seeds):
            print(f'fit {family}: seed {seed:g} widths (no production inputs)', flush=True)
            fit = fit_one(layout, meta, data, focal_widths=seed, max_nfev=max_nfev)
            runs.append(fit)
            print(f'  cost={fit["summary"]["train_robust_cost"]:.3f}; status={fit["summary"]["optimizer_status"]}', flush=True)
            write_json(out/f'{family}_start_{k}.json', fit['summary'])
        best = min(runs, key=lambda r: r['summary']['train_robust_cost'])
        summaries = [r['summary'].copy() for r in runs]
        best['summary']['starts'] = summaries
        near = [r for r in runs if r['summary']['train_robust_cost'] <= best['summary']['train_robust_cost']*1.02+1e-8]
        best['summary']['near_best_start_aspect_range'] = [min(r['summary']['fx_over_fy'] for r in near), max(r['summary']['fx_over_fy'] for r in near)]
        obs = observability(best); write_json(out/f'{family}_observability.json', obs)
        best['summary']['local_condition'] = obs['full_condition']
        if 'shared_condition' in obs: best['summary']['shared_condition'] = obs['shared_condition']
        np.savez_compressed(out/f'{family}_fit.npz', x=best['x'], optics=best['optics'],
            rotations=best['rotations'], residual=best['residual'], valid=best['valid'])
        write_json(out/f'{family}.json', best['summary']); result[family] = best
    return result


def profile_aspect(graph, family, best, values, out, max_nfev=180):
    """Fix eta and refit ALL remaining parameters; not a frozen-cost sweep."""
    meta, data = graph['meta'], graph['data']; rows = []
    for aspect in values:
        if aspect <= 0: raise ValueError('Aspect must be positive')
        layout = Layout(len(meta['frames']), meta['width'], meta['height'], family, np.log(aspect), anchors=meta.get('anchors'))
        old = dict(zip(best['layout'].names, best['x']))
        start = np.array([old[name] for name in layout.names])
        print(f'profile fx/fy={aspect}: refit nuisance parameters', flush=True)
        fit = fit_one(layout, meta, data, max_nfev=max_nfev, start=start)
        rows.append(fit['summary'])
    write_json(out/'aspect_profile.json', dict(rows=rows, family=family,
        interpretation='Local nuisance-refitted diagnostic, not confidence interval; optimizer status matters.'))
    return rows
