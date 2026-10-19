"""Whole-source equivalence check for the physical print-affine quotient.

Does not fit, replace observations, or adopt a production checkpoint. Uses the
full preserved CP auxiliaries, true implicit RS roots, and original grouped W.
"""
from pathlib import Path
import argparse, hashlib, json, time
import numpy as np
from scipy import sparse
from rinkfit_start.ad_chart import PrintOnlyAdChart, effective_affine
from rinkfit_start.native_scene import NativeScene, camera_from_event


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf8'))
def ref(path): return dict(path=str(Path(path).resolve()), sha256=sha(path))
def verify(d, base=None):
    p = Path(d.get('path', d.get('file')))
    if not p.is_absolute(): p = Path(base) / p
    if sha(p) != d['sha256']: raise ValueError('Changed immutable source ' + str(p))
    return p

def array(d, base):
    p = verify(d, base); a = np.fromfile(p, d['dtype'])
    return a.reshape(d['shape'])


def whole_W(error, sem, semantic_path):
    base = semantic_path.parent
    active = array(sem['active_source_mask'], base).astype(bool)
    members = array(sem['arrays']['group_rows'], base).astype(int)
    weights = array(sem['arrays']['weights'], base)
    sigma = array(sem['arrays']['source_sigma'], base)
    independent = array(sem['arrays']['ordinary_rows'], base).astype(int)
    independent = independent[active[independent]]
    statpath = verify(sem['active_statistics'], base); stats = read(statpath)
    replacements = {x['group_id']: x for x in stats['replacements']}
    partition = np.zeros(len(error), int); cost = 0.; rank = 0
    for line in verify(sem['assets']['groups.jsonl'], base).read_text(encoding='utf8').splitlines():
        g = json.loads(line); rr = members[g['member_start']:g['member_start'] + g['member_count']]
        W = weights[g['weight_start']:g['weight_start'] + g['rank'] * 2 * len(rr)].reshape(g['rank'], 2 * len(rr))
        if g['group_id'] in replacements:
            r = replacements[g['group_id']]; rr = np.asarray(r['source_rows'], int); W = array(r['W'], statpath.parent)
        z = W @ error[rr].ravel(); cost += .5 * float(z @ z); rank += len(z); partition[rr] += 1
    z = error[independent] / sigma[independent, None]
    cost += .5 * float(np.sum(z * z)); rank += z.size; partition[independent] += 1
    if not np.array_equal(partition, active.astype(int)): raise ValueError('Original W partition changed')
    return dict(half_cost=cost, rank=rank, partition_exactly_once=True)


def run(checkpoint, authority, output):
    begin = time.perf_counter(); CP = Path(checkpoint).resolve(); cp = read(CP)
    AD = Path(authority).resolve(); ad = read(AD)
    M = verify(cp['model']); model = read(M)
    PK = verify(ad['packet']); packet = read(PK); SE = verify(ad['semantic']); sem = read(SE)
    chart = PrintOnlyAdChart(model); x = chart.pack(); newmodel, domain = chart.unpack(x, return_domain=True)
    O = Path(output).resolve(); O.mkdir(parents=True, exist_ok=False)
    (O / 'model.json').write_text(json.dumps(newmodel, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    np.save(O / 'physical_coordinates.npy', x)
    sparse.save_npz(O / 'ambient_from_physical.npz', chart.tangent(newmodel))
    before = NativeScene(model); after = NativeScene(newmodel)
    coeff_difference = 0.; path_difference = 0.; paths = 0; samples = 0; all_paths = 0
    for qi, ai in chart.print_ad.items():
        a = effective_affine(model['blocks'][ai]['values'], model['blocks'][qi]['values'])
        b = effective_affine(newmodel['blocks'][ai]['values'], newmodel['blocks'][qi]['values'])
        coeff_difference = max(coeff_difference, float(np.max(abs(a - b))))
    for mi, material in enumerate(model['materials']):
        for pi, p in enumerate(material['paths']):
            all_paths += 1
            if p.get('advertisement_block') not in chart.ads: continue
            n = len(p['points']); s = np.linspace(0., 1., 4 * (n - 1) + 1)
            a = before.world(mi, pi, s); b = after.world(mi, pi, s)
            path_difference = max(path_difference, float(np.max(abs(a - b)))); paths += 1; samples += len(s)
    source_uv = array(packet['arrays']['native_uv'], PK.parent)
    pts = array(packet['arrays']['native_pts'], PK.parent)
    material = array(sem['resolved_material_index'], SE.parent).astype(int)
    active = array(sem['active_source_mask'], SE.parent).astype(bool)
    kind = array(sem['arrays']['kind'], SE.parent)
    branch = array(cp['row_state_files']['branch'], CP.parent).astype(int)
    station = array(cp['row_state_files']['station'], CP.parent)
    saved_path = verify(cp['native_predictions'])
    with np.load(saved_path, allow_pickle=False) as z: saved = {k: z[k] for k in z.files}
    if not np.array_equal(np.flatnonzero(~active), [25537]): raise ValueError('Source exclusion changed')
    selected_materials = np.asarray([any(p.get('advertisement_block') in chart.ads for p in m['paths']) for m in model['materials']])
    affected = active & selected_materials[material]
    if np.any(affected & (kind == 2)): raise ValueError('A target ad unexpectedly has profile semantics')
    rows = np.flatnonzero(active & (kind != 2)); camera = camera_from_event(model)
    pre = saved['uv'].copy(); post = pre.copy(); maximum_world = 0.; invalid_before = 0; invalid_after = 0
    max_eq_before = 0.; max_eq_after = 0.; unbracketed = 0; uncertified = 0
    for start in range(0, len(rows), 20000):
        rr = rows[start:start + 20000]; t = (pts[rr] - 256256) / 12800.
        wa, oka, _ = before.world_rows(material[rr], branch[rr], station[rr])
        wb, okb, _ = after.world_rows(material[rr], branch[rr], station[rr])
        maximum_world = max(maximum_world, float(np.max(abs(wa - wb))))
        pa = camera.project_rs(wa, t); pb = camera.project_rs(wb, t)
        pre[rr] = pa['uv']; post[rr] = pb['uv']
        invalid_before += int(np.count_nonzero(~(oka & pa['valid'])))
        invalid_after += int(np.count_nonzero(~(okb & pb['valid'])))
        unbracketed += int(np.count_nonzero(~pa['bracketed']) + np.count_nonzero(~pb['bracketed']))
        uncertified += int(np.count_nonzero(~pa['certificate']['certified']) + np.count_nonzero(~pb['certificate']['certified']))
        max_eq_before = max(max_eq_before, float(np.max(abs(pa['root_equality_px']))))
        max_eq_after = max(max_eq_after, float(np.max(abs(pb['root_equality_px']))))
        print(json.dumps(dict(replayed_rows=min(start + 20000, len(rows)), total=len(rows))), flush=True)
    Wpre = whole_W(pre - source_uv, sem, SE); Wpost = whole_W(post - source_uv, sem, SE)
    delta = abs(post - pre)
    np.savez_compressed(O / 'whole_source_equivalence.npz', source_rows=np.arange(len(active)),
                        active_source_mask=active, before_uv=pre, after_uv=post,
                        affected_source_rows=np.flatnonzero(affected), station=station, branch=branch)
    report = dict(schema='rinkfit.whole_source_print_chart_equivalence.v1',
        checkpoint=ref(CP), original_model=ref(M), new_model=ref(O / 'model.json'),
        authority=ref(AD), source=ref(__file__), chart_source=ref(Path(__file__).parents[2] / 'src/rinkfit_start/ad_chart.py'),
        coordinates=chart.report(), domain_witnesses=domain,
        all_model_paths=all_paths, affected_finite_paths=paths, vertex_and_interior_samples=samples,
        maximum_physical_affine_coefficient_difference_L=coeff_difference,
        maximum_finite_path_world_difference_L=path_difference,
        raw_rows=len(active), active_rows=int(active.sum()), affected_active_rows=int(affected.sum()),
        ordinary_entity_rows_replayed=len(rows), profile_rows_retained=int(np.sum(active & (kind == 2))),
        profile_scope='All 961 ice profiles have unchanged camera and finite material. Their saved exact query predictions retained for original W; no new global-profile claim.',
        maximum_all_replayed_world_difference_L=maximum_world,
        maximum_all_projection_component_difference_px=float(np.max(delta[active])),
        maximum_affected_projection_component_difference_px=float(np.max(delta[affected])),
        maximum_before_vs_saved_projection_difference_px=float(np.max(abs(pre[rows] - saved['uv'][rows]))),
        invalid_before=invalid_before, invalid_after=invalid_after, unbracketed_both=unbracketed,
        uncertified_both=uncertified, max_RS_equality_before_px=max_eq_before, max_RS_equality_after_px=max_eq_after,
        original_W_before=Wpre, original_W_after=Wpost,
        W_half_cost_difference=Wpost['half_cost'] - Wpre['half_cost'],
        original_scene=before.check_physics(), new_scene=after.check_physics(),
        all_original_UV_PTS_labels_W_E_exclusion_station_branch_preserved=True,
        accepted_physical_improvement=False, elapsed_seconds=time.perf_counter() - begin)
    if report['maximum_all_projection_component_difference_px'] > 1e-6 or invalid_before or invalid_after:
        raise ValueError('Unexpected quotient geometry/projection discrepancy')
    (O / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('original_scene', 'new_scene')}, ensure_ascii=False), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True); parser.add_argument('--authority', required=True)
    parser.add_argument('--output', required=True); args = parser.parse_args()
    run(args.checkpoint, args.authority, args.output)
