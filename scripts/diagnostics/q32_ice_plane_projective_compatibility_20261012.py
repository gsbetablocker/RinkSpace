"""Isolate planar source compatibility from the existing rigid-camera restriction.

This is a conditional diagnostic, not a camera or world-model replacement.
The same finite source curves, typed operators, loss masks and budgets are used.
An eight-coordinate projective map of the ice plane composes with the original
camera projection; Brown distortion stays fixed. No full programme is loaded.
"""
from types import SimpleNamespace, MethodType
import json
import time
import q32_frozen_camera_source_context_20261012 as context
from scipy.optimize import least_squares

s = context.s
np = s.np
PTS = (229702144, 231678464)


def main():
    start = time.perf_counter()
    complete = context.prepare(PTS)
    result = dict(schema='rinkspace.fixed-ice-plane-projective-compatibility.v1',
        state_identity=s.ref(complete.directory / 'state_identity.json'),
        script=s.ref(__file__), source_context=s.ref(context.__file__),
        prepare_seconds=complete.prepare_seconds, modes=[],
        scope='Projective plane compatibility with fixed current Brown distortion. This does not establish a physical shared camera, global feasibility, or independent accuracy.',
        no_production_model_written=True, original_source_observations_unchanged=True)
    for pts in PTS:
        records = [dict(r, row=dict(r['row'])) for r in complete.records
                   if r['actual_pts'] == pts and np.all(np.asarray(r['world'])[:, 2] == 0.)]
        assert records
        c = SimpleNamespace(**dict(vars(complete), records=records))
        cameras = {pts: complete.baseline_cameras[pts]}
        loss = context.loss(c)
        baseline = context.residual(c, cameras)
        baseline_rows = context.metrics(c, cameras)
        original_methods = {}
        mapped = {}
        for r in records:
            if r['family'] != 'original_absolute':
                factor = r['factor']
                original_methods.setdefault(id(factor), (factor, factor._world))
                def cached(self, q, curve=None):
                    return mapped[id(self), json.dumps(curve, sort_keys=True)]
                factor._world = MethodType(cached, factor)
        calls = 0
        def apply(p):
            nonlocal calls
            calls += 1
            H = np.array([[1+p[0], p[1], p[2]], [p[3], 1+p[4], p[5]], [p[6], p[7], 1.]])
            for r in records:
                old = r['world']
                if np.all(p == 0.):
                    world = old
                else:
                    hom = np.c_[old[:, :2] / 50., np.ones(len(old))] @ H.T
                    if not np.isfinite(hom).all() or np.any(hom[:, 2] <= 0.):
                        raise ValueError('Projective map crosses infinity within a complete source curve')
                    world = np.c_[50. * hom[:, :2] / hom[:, 2, None], np.zeros(len(old))]
                if r['family'] == 'original_absolute':
                    r['row']['world'] = world
                else:
                    mapped[id(r['factor']), json.dumps(r['row']['curve'], sort_keys=True)] = world
            return H
        def residual(p):
            apply(p)
            return context.residual(c, cameras)
        record = dict(actual_pts=pts, source_rows=len(records), UV_count=sum(len(r['row']['uv']) for r in records),
            baseline_rows=baseline_rows, baseline_cost=float(.5 * loss(baseline**2)[0].sum()))
        t = time.perf_counter()
        try:
            assert np.array_equal(residual(np.zeros(8)), baseline)
            solved = least_squares(residual, np.zeros(8), jac='3-point', x_scale='jac', loss=loss, tr_solver='exact')
            H = apply(solved.x)
            rows = context.metrics(c, cameras)
            cost = sum(r['weighted_current_cost'] for r in rows)
            assert np.isclose(cost, solved.cost, rtol=1e-12, atol=1e-12)
            record.update(success=bool(solved.success), termination=solved.message,
                nfev=int(solved.nfev), njev=int(solved.njev), cost=cost, final_rows=rows,
                plane_homography_in_50_scene_unit_chart=H,
                robust_weighted_singular_values=np.linalg.svd(solved.jac, compute_uv=False),
                optimality=float(solved.optimality))
        except Exception as error:
            record.update(success=False, error_type=type(error).__name__, error=str(error))
        finally:
            for factor, method in original_methods.values():
                factor._world = method
        record.update(seconds=time.perf_counter()-t, actual_value_calls=calls)
        result['modes'].append(record)
        print(json.dumps({k:v for k,v in record.items() if k not in ('baseline_rows','final_rows','plane_homography_in_50_scene_unit_chart','robust_weighted_singular_values')}, default=s.plain), flush=True)
    result['total_seconds'] = time.perf_counter() - start
    output = s.DIAG / 'q32_ice_plane_projective_compatibility_20261012.json'
    output.write_text(json.dumps(result, default=s.plain, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(s.ref(output)), flush=True)


if __name__ == '__main__':
    main()
