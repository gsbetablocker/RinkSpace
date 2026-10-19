"""Branch adapter of the full-source Phase-I primary to physical print affines.

The old complete-state Jacobian, W, original A, sensor rows and print SOC guard
are all pulled back by the same rank-4447 lift. Ad8--12 have no solver columns.
Legacy boxes/guards at the current dependent witness define a declared LOCAL
INNER numerical chart; they are not asserted to be the whole old box image.
Every nonlinear proposal is decoded through the exact original-domain LP API.
"""
from pathlib import Path
import argparse, hashlib, json, sys, time
import numpy as np
from scipy import sparse
import clarabel
from rinkfit_start.ad_chart import PrintOnlyAdChart


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p): return dict(path=str(Path(p).resolve()), sha256=sha(p))
def read(p): return json.loads(Path(p).read_text(encoding='utf8'))
def verify(d, base=None):
    p = Path(d.get('path', d.get('file')))
    if not p.is_absolute(): p = Path(base) / p
    if sha(p) != d['sha256']: raise ValueError('Changed immutable input ' + str(p))
    return p

def run(input_path, output, solve=False, release=None, secondary_from=None, time_multiple=None,time_window=None):
    started = time.perf_counter(); IP = Path(input_path).resolve(); spec = read(IP)
    O = Path(output).resolve(); O.mkdir(parents=True, exist_ok=False)
    executed_source=O/'primary_source.py';executed_source.write_bytes(Path(__file__).read_bytes())
    required = ['linearization', 'transport', 'original_A', 'rhs_bounds', 'group_W_source',
                'full_norm_source', 'station_TR_manifest', 'station_TR_limiter_source', 'print_det_guard_source']
    paths = {k: verify(spec[k]) for k in required}
    for key in ['group_W_source', 'full_norm_source', 'station_TR_limiter_source', 'print_det_guard_source']:
        sys.path.insert(0, str(paths[key].parent))
    from group_W_station import assemble, save
    from station_curvature_TR_chart import apply_station_curvature_TR
    from source_affine_norm_constraints import load_all_source_affine_norms
    from print_det_convex_guard import from_full_norm_operator
    Q = paths['linearization']; T = paths['transport']; tr = read(T); CP = verify(tr['checkpoint']); cp = read(CP)
    N = int(tr['raw_rows']); requested = np.arange(N, dtype='<u8'); rp = O / 'source_rows.u64'; rp.write_bytes(requested.tobytes())
    selection = dict(schema='rinkfit.station_column_selection.v1', checkpoint=ref(CP), transport=ref(T),
                     source_rows=dict(file=rp.name, dtype='<u8', shape=[N], sha256=sha(rp)))
    SEL = O / 'selection.json'; SEL.write_text(json.dumps(selection, indent=2), encoding='utf8')
    old = assemble(Q, T, selection=SEL, station_seed=verify(spec['station_seed']) if spec.get('station_seed') else None,
                   station_trust=spec.get('station_trust'))
    old, trust_info = apply_station_curvature_TR(old, paths['station_TR_manifest'], spec.get('station_TR_array_key', 'radius'))
    chart = PrintOnlyAdChart(old.model); physical0 = chart.pack(); full_n = len(chart.free); no = len(old.free); k = len(old.selected)
    if full_n != no - chart.removed_free or chart.removed_free not in (0, 20): raise ValueError('The print-only chart must remove every independent ad coordinate or inherit their already-dependent storage')
    if any(any(int(chart.offset[ai]+j) in set(chart.free)for j in range(4))for ai in chart.ads): raise ValueError('A dependent advertisement chart leaked into solver coordinates')
    from rinkfit_start.camera_spline_chart import time_chart_from_model
    time_chart=time_chart_from_model(old.model,time_multiple,time_window);time_info=None
    if time_chart is not None:
        C,coordinate_scales,coordinate_names,time_info=time_chart.compose_physical(chart,old.model)
    else:
        C=sparse.eye(full_n,format='csr');coordinate_scales=old.scale[chart.free].copy();coordinate_names=[str(x)for x in chart.free]
        for qi,ai in chart.print_ad.items():
            ad=np.asarray(old.model['blocks'][ai]['values']);span=np.array([ad[1]]*3+[ad[3]-ad[2]]*3)
            ids=[chart.column[int(chart.offset[qi]+j)]for j in range(6)];coordinate_scales[ids]*=span
    released=np.arange(C.shape[1])
    if release is not None:
        if release not in ('opening_ad2','time_ad2'):raise ValueError('Unknown declared physical release')
        ai2=next(i for i,b in enumerate(old.model['blocks'])if b['name']=='ad/2');allowed={ai2}
        for material in old.model['materials']:
            for path in material['paths']:
                if path.get('advertisement_block')==ai2 and path.get('geometry_block',-1)>=0:allowed.add(int(path['geometry_block']))
        for bi,b in enumerate(old.model['blocks']):
            if b['name'].startswith('control/')and(release=='time_ad2'or int(b['name'].split('/')[1])<=310):allowed.add(bi)
        bis=np.searchsorted(chart.offset[1:],chart.free,side='right');allowed_rows=np.flatnonzero(np.isin(bis,list(allowed)))
        released=np.unique(C[allowed_rows].indices)
    n=len(released);reduced_scale=coordinate_scales[released]
    physical_map=(C[:,released]@sparse.diags(reduced_scale)).tocsr()
    lift=(sparse.diags(1./old.scale[old.free])@chart.tangent(old.model)[old.free]@physical_map).tocsr()
    if lift.shape!=(no,n)or np.any(np.asarray(abs(lift).sum(axis=0)).ravel()==0):raise ValueError('Singular reduced physical chart')
    norms = load_all_source_affine_norms(Q, T, chart.offset, old.scale, old.free)
    guard = from_full_norm_operator(norms, old.free, old.scale)
    # Every physical derivative and every determinant guard uses the same lift.
    Jnorm = (norms.J @ lift).tocsr(); JW = (old.Jphysical @ lift).tocsc()
    sensorJ = (old.sensor_Jphysical @ lift).tocsr(); guard.G = (guard.G @ lift).tocsr(); guard.B = (guard.B @ lift).tocsr()
    Aold = sparse.load_npz(paths['original_A']).tocsr(); bounds = np.load(paths['rhs_bounds']); rhs = bounds['rhs']
    if Aold.shape != (len(rhs), no + 1): raise ValueError('Mismatched original full-source physical master')
    A = sparse.hstack([Aold[:, :no] @ lift, Aold[:, no:]], format='csr')
    trust=float(spec['trust_scaled']);rows=norms.rows;m=len(rows);qref=max(1.,float(cp['b']))
    if time_chart is None:
        old_col={int(a):i for i,a in enumerate(old.free)};retained=np.array([old_col[int(a)]for a in chart.free[released]])
        diag=np.asarray(lift[retained,np.arange(n)]).ravel()
        lower_physical=old.lower_physical[retained]/diag;upper_physical=old.upper_physical[retained]/diag
    else:
        # A coarse coordinate is a genuine linear combination of fine blocks.
        # Retain every ORIGINAL fine box and original scaled trust as linear
        # rows. Per-coarse-coordinate +/-trust is an additional declared local
        # numerical inner chart, not an unreported physical motion prior.
        fine_lo=np.maximum(old.lower_physical,-trust);fine_hi=np.minimum(old.upper_physical,trust)
        boxA=sparse.vstack([lift,-lift],format='csr');boxrhs=np.r_[fine_hi,-fine_lo]
        A=sparse.vstack([A,sparse.hstack([boxA,sparse.csr_matrix((len(boxrhs),1))])],format='csr');rhs=np.r_[rhs,boxrhs]
        lower_physical=np.full(n,-trust);upper_physical=np.full(n,trust)
    PK = verify(old.authority['packet']); pk = read(PK)
    def ar(desc, base): return np.fromfile(verify(desc, base), desc['dtype']).reshape(desc['shape'])
    E = ar(pk['arrays']['engineering_budget_px'], PK.parent)
    tan = ar(tr['arrays']['station_UV_right'], T.parent).reshape(N, 2)
    rawuv = (2 * rows[:, None] + np.arange(2)).ravel()
    Ts = sparse.diags(np.repeat(1. / E[rows], 2)) @ old.T[rawuv]
    J = sparse.hstack([Jnorm, Ts], format='csr'); rseed = norms.residual + tan[rows] * old.station_seed[rows, None] / E[rows, None]
    lowb = float(bounds['lower'][-1]); highb = float(bounds['upper'][-1])
    lo = np.r_[np.maximum(lower_physical, -trust), old.station_lower, lowb]
    hi = np.r_[np.minimum(upper_physical, trust), old.station_upper, highb]
    units = np.r_[np.full(n, trust), np.maximum(abs(old.station_lower), abs(old.station_upper)), qref]
    if np.any(units <= 0): raise ValueError('Noninvertible station/physical units')
    expanded = sparse.hstack([A[:, :n], sparse.csr_matrix((len(rhs), k)), A[:, n:]], format='csr')
    Al = (expanded @ sparse.diags(units)).tocsc(); bl = rhs.copy()
    sensor = sparse.hstack([sensorJ, old.sensor_Jstation, sparse.csr_matrix((len(old.sensor_rows), 1))], format='csr')
    V = (sensor @ sparse.diags(units)).tocsc(); sr = old.sensor_row0
    VS = sparse.vstack([V[:, :-1], -V[:, :-1]], format='csr'); vrhs = np.r_[1079 - sr, sr]
    zlo = lo[:-1] / units[:-1]; zhi = hi[:-1] / units[:-1]
    estimate = VS.maximum(0) @ zhi + VS.minimum(0) @ zlo; mag = abs(VS) @ np.maximum(abs(zlo), abs(zhi))
    terms = np.diff(VS.indptr); gamma = (2 * terms + 8) * np.finfo(float).eps; gamma = gamma / (1 - gamma)
    enclosure = np.nextafter(estimate + gamma * mag / (1 - gamma), np.inf); keep = np.flatnonzero(enclosure > vrhs)
    Al = sparse.vstack([Al, sparse.hstack([VS[keep], sparse.csr_matrix((len(keep), 1))])], format='csc'); bl = np.r_[bl, vrhs[keep]]
    lowz = lo / units; hiz = hi / units; eye = sparse.eye(n + k + 1, format='csc'); ui = np.flatnonzero(np.isfinite(hiz))
    Al = sparse.vstack([Al, -eye, eye[ui]], format='csc'); bl = np.r_[bl, -lowz, hiz[ui]]
    gA, gb, gdims = guard.clarabel_soc6(np.zeros(n), units[:n], k + 1)
    JC = (J @ sparse.diags(units[:-1] / np.sqrt(qref))).tocsr()
    bi = sparse.csr_matrix((np.full(m, -1.), (np.arange(m), np.full(m, n + k))), shape=(m, n + k + 1))
    jr = sparse.hstack([-2 * JC, sparse.csr_matrix((2 * m, 1))], format='csr')
    stacked = sparse.vstack([bi, jr, bi], format='csr')
    perm = np.column_stack([np.arange(m), m + 2 * np.arange(m), m + 2 * np.arange(m) + 1, 3 * m + np.arange(m)]).ravel()
    soc = stacked[perm].tocsc(); cb = np.column_stack([np.ones(m), 2 * rseed / np.sqrt(qref), -np.ones(m)]).ravel()
    AA = sparse.vstack([Al, gA, soc], format='csc'); bb = np.r_[bl, gb, cb]
    cones = [clarabel.NonnegativeConeT(len(bl))] + [clarabel.SecondOrderConeT(int(d)) for d in gdims] + [clarabel.SecondOrderConeT(4) for _ in rows]
    objective = np.zeros(n + k + 1); objective[-1] = 1.; P = sparse.csc_matrix((len(objective), len(objective)))
    sparse.save_npz(O / 'primary_A.npz', AA); sparse.save_npz(O / 'old_scaled_from_physical_scaled.npz', lift)
    sparse.save_npz(O / 'full_original_W_Jphysical.npz', JW)
    sparse.save_npz(O/'full_original_W_Jstation.npz',old.Jstation)
    sparse.save_npz(O/'full_physical_from_solver_scaled.npz',physical_map)
    np.save(O/'original_W_residual.npy',old.rW)
    np.save(O/'station_source_rows.npy',old.selected)
    np.save(O/'station_seed_full.npy',old.station_seed)
    np.save(O/'norm_source_rows.npy',rows)
    np.savez_compressed(O / 'primary_arrays.npz', rhs=bb, objective=objective, units=units,
                        reduced_physical_scale=reduced_scale, physical0=physical0,
                        released_full_physical_columns=released, lower_native=lo, upper_native=hi, cone_dimensions=np.asarray([int(c.dim) for c in cones]))
    assembly = dict(schema='rinkfit.full_S_primary_physical_print_chart.v1', input=ref(IP), source=ref(executed_source),canonical_source_path=str(Path(__file__).resolve()),
        chart_source=ref(Path(__file__).parents[2] / 'src/rinkfit_start/ad_chart.py'),time_chart_source=ref(Path(__file__).parents[2] / 'src/rinkfit_start/camera_spline_chart.py')if time_chart is not None else None,authority=old.authority,
        time_subspace=time_info,coordinate_names=[coordinate_names[i]for i in released],coordinates=chart.report(), original_physical_columns=no, physical_columns=n, full_reduced_physical_coordinate_count=full_n, declared_release=release, station_columns=k,
        epigraph_columns=1, all_active_source_SOC4=m, original_print_SOC6=len(gdims), original_W_rank=len(old.rW),
        rows=AA.shape[0], columns=AA.shape[1], nnz=AA.nnz,
        local_linear_domain='Current dependent-witness old boxes/scene/determinant guards pulled back by the same physical-affine lift; conservative inner numerical chart. Full old physical box image is checked by the LP decoder at each trial.',
        all_full_J_W_scene_sensor_and_detguard_pulled_back=True,
        print_physical_affine_shape_and_placement_remain_free=True,
        exact_global_domain_not_claimed_for_local_linear_subproblem=True,
        station_TR_application=trust_info, assembly_seconds=time.perf_counter() - started, no_Solve=not solve)
    (O / 'assembly.json').write_text(json.dumps(assembly, ensure_ascii=False, indent=2), encoding='utf8')
    print('ASSEMBLY', json.dumps({key: assembly[key] for key in ['physical_columns','station_columns','rows','columns','nnz','assembly_seconds']}), flush=True)
    if not solve and secondary_from is None: return assembly
    if secondary_from is not None:
        if time_chart is not None:raise ValueError('Use the generic saved-matrix conditional W adapter for a coarse time chart; legacy ambient proposal dumps are unsupported')
        from copy import copy, deepcopy
        sys.path.insert(0, str(verify(spec['secondary_source']).parent))
        from finite_visible_W_secondary import solve as solve_W
        PD = Path(secondary_from).resolve(); proposal = np.load(PD / 'proposal.npz')
        if not np.array_equal(proposal['released_full_physical_columns'], released): raise ValueError('Secondary release differs from primary')
        primal = proposal['native_primal']; d0 = primal[:n]; eta0 = primal[n:n+k]
        q0 = np.einsum('ij,ij->i', rseed + (J @ primal[:-1]).reshape(m,2), rseed + (J @ primal[:-1]).reshape(m,2))
        bcol = np.asarray(A[:,-1].toarray()).ravel(); negative = bcol < 0
        required_b = float(np.max(((A[:,:n]@d0)-rhs)[negative]/(-bcol[negative]),initial=0.))
        working_b = np.nextafter(max(float(q0.max()),float(primal[-1]),required_b),np.inf)
        cw = copy(old); cw.free = chart.free[released].copy(); cw.Jphysical = JW; cw.sensor_Jphysical = sensorJ
        cw.lower_physical = lower_physical; cw.upper_physical = upper_physical
        cw.station_seed = old.station_seed.copy(); cw.station_seed[old.selected] += eta0
        cw.station_lower = old.station_lower - eta0; cw.station_upper = old.station_upper - eta0
        cw.rW = old.rW + old.Jstation @ eta0; cw.sensor_row0 = old.sensor_row0 + old.sensor_Jstation @ eta0
        cw.authority = deepcopy(old.authority); cw.authority['physical_print_chart'] = dict(source=ref(__file__),primary=ref(PD/'report.json'),released_physical_columns=n)
        nw = copy(norms); nw.J = Jnorm
        result = solve_W(cw,nw,E,d0,working_b,A,rhs,trust,O/'W',scene_guard=guard)
        summary = dict(schema='rinkfit.physical_print_chart_W_secondary_branch.v1',assembly=ref(O/'assembly.json'),
            secondary_report=ref(O/'W/report.json'),original_parent_W_half_cost=float(.5*old.rW@old.rW),
            primary_proposal_W_half_cost=float(read(PD/'report.json')['predicted_W_half_cost']),working_b=float(working_b),
            all_source_count=m,physical_columns=n,station_columns=k,accepted_nonlinear_candidate=False)
        if result.get('final'):
            final=result['final']; pz=np.load(O/'W/proposal_step.npz');raw=old.parent_values+pz['physical_delta']
            mm=deepcopy(old.model)
            for bi,block in enumerate(mm['blocks']):block['values']=raw[chart.offset[bi]:chart.offset[bi+1]].tolist()
            decoded,dom=chart.unpack(chart.pack(mm),return_domain=True)
            mp=O/'W/model.json';mp.write_text(json.dumps(decoded,ensure_ascii=False,indent=2),encoding='utf8')
            sp=O/'W/station.f64';s=ar(cp['row_state_files']['station'],CP.parent)+pz['station_delta_full'];s.astype('<f8').tofile(sp)
            aux={key:dict(path=str(verify(desc,CP.parent)),sha256=desc['sha256'],dtype=desc['dtype'],shape=desc['shape'])for key,desc in cp['row_state_files'].items()}
            aux['station']=dict(path=str(sp),sha256=sha(sp),dtype='<f8',shape=[N])
            cm=dict(schema='rinkfit.physical_print_chart_trial_seed.v1',parent_checkpoint=ref(CP),model=ref(mp),row_state_files=aux,
                fraction=1.,requires_fresh_true_RS_profile_and_all_station_evaluation=True,accepted_nonlinear_candidate=False)
            cmP=O/'W/candidate_manifest.json';cmP.write_text(json.dumps(cm,indent=2),encoding='utf8')
            summary.update(candidate_manifest=ref(cmP),exact_domain=dom,predicted_W_half_cost=final['predicted_W_half_cost'],
                W_change_from_original_parent=final['predicted_W_half_cost']-float(.5*old.rW@old.rW),
                W_change_from_primary=final['predicted_W_half_cost']-summary['primary_proposal_W_half_cost'])
        (O/'report.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
        print('W_DONE',json.dumps(summary,ensure_ascii=False),flush=True);return summary
    settings = clarabel.DefaultSettings(); settings.verbose = True; settings.max_threads = 1; settings.max_iter = 100
    settings.tol_gap_abs = 1e-10; settings.tol_gap_rel = 1e-10; settings.tol_feas = 1e-10
    tsolve = time.perf_counter(); solver = clarabel.DefaultSolver(P, objective, AA, bb, cones, settings); sol = solver.solve()
    report = dict(assembly=ref(O / 'assembly.json'), raw_solver_status=str(sol.status), iterations=int(sol.iterations),
                  solve_seconds=time.perf_counter() - tsolve, accepted_nonlinear_candidate=False,
                  original_PhaseI_or_PhaseII_optimality_claim=False)
    if sol.x is not None:
        z = np.asarray(sol.x); step = units * z; d = step[:n]; eta = step[n:n + k]
        full_delta=np.asarray(physical_map@d).ravel()
        predicted = rseed + (J @ step[:-1]).reshape(m, 2); q = np.einsum('ij,ij->i', predicted, predicted)
        wr = old.rW + JW @ d + old.Jstation @ eta
        np.savez_compressed(O / 'proposal.npz', solver_primal=z, native_primal=step,
                            reduced_physical_delta=full_delta, physical_coordinates=physical0 + full_delta, released_full_physical_columns=released,
                            station_eta=eta, station_rows=old.selected)
        witnesses = []
        f = 1.
        while f > 0.:
            try:
                decoded, domain = chart.unpack(physical0 + f * full_delta, return_domain=True)
                if time_chart is not None:time_chart.recover(decoded)
                file = O / ('decoded_fraction_' + str(f).replace('.', '_') + '.json')
                file.write_text(json.dumps(decoded, ensure_ascii=False, indent=2), encoding='utf8')
                station_new = ar(cp['row_state_files']['station'], CP.parent).copy() + f * old.station_seed
                station_new[old.selected] += f * eta
                spath = O / 'candidate_station.f64'; station_new.astype('<f8').tofile(spath)
                auxiliaries = {key:dict(path=str(verify(desc,CP.parent)),sha256=desc['sha256'],dtype=desc['dtype'],shape=desc['shape']) for key,desc in cp['row_state_files'].items()}
                auxiliaries['station'] = dict(path=str(spath),sha256=sha(spath),dtype='<f8',shape=[N])
                cm = dict(schema='rinkfit.physical_print_chart_trial_seed.v1',parent_checkpoint=ref(CP),model=ref(file),row_state_files=auxiliaries,fraction=f,requires_fresh_true_RS_profile_and_all_station_evaluation=True,accepted_nonlinear_candidate=False)
                (O/'candidate_manifest.json').write_text(json.dumps(cm,indent=2),encoding='utf8')
                witnesses.append(dict(fraction=f, passed=True, model=ref(file), domain=domain)); break
            except ValueError as exc:
                witnesses.append(dict(fraction=f, passed=False, reason=str(exc))); f *= .5
        print_cols = np.asarray([chart.column[int(chart.offset[qi] + j)] for qi in chart.print_ad for j in range(6)])
        print_translate = np.asarray([chart.column[int(chart.offset[qi] + j)] for qi in chart.print_ad for j in [2,5]])
        report.update(physical_columns=n, original_physical_columns=no,
                      all_source_affine_qmax=float(q.max()), baseline_b=float(cp['b']),
                      predicted_W_half_cost=float(.5 * wr @ wr), baseline_W_half_cost=float(.5 * old.rW @ old.rW),
                      maximum_print_affine_physical_step_L=float(np.max(abs(full_delta[print_cols]))),
                      maximum_print_placement_step_L=float(np.max(abs(full_delta[print_translate]))),
                      nonzero_print_physical_coordinates=int(np.count_nonzero(full_delta[print_cols])),
                      exact_domain_decoder_trial_witnesses=witnesses,
                      current_linear_box_violations=int(np.count_nonzero((step < lo) | (step > hi))),
                      determinant_guard=guard.audit(d))
    (O / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    print('DONE', json.dumps(report, ensure_ascii=False), flush=True); return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--input', required=True)
    p.add_argument('--output', required=True); p.add_argument('--solve', action='store_true'); p.add_argument('--release',choices=['opening_ad2','time_ad2']); p.add_argument('--secondary-from');p.add_argument('--time-multiple',type=int,choices=[2,4,8]);p.add_argument('--time-window',type=float,nargs=2); a = p.parse_args()
    run(a.input, a.output, a.solve, a.release, a.secondary_from,a.time_multiple,a.time_window)
