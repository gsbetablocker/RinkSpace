"""Independent complete-state audit of the opt-in true-RS source-UV capture mode.

This is a consumer of immutable checkpoints, not a solver or a qualification
issuer. It never profiles a stored station, moves a source UV, discards failures,
or treats the native status bitmaps as independent evidence.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time
import numpy as np
from .accepted_aux_prefix import AUX, _cpp_hex, _quoted, _quoted_fields, _prefix_hash, _runtime_group_digest, _runtime_query_digest
from .io_utils import file_sha256
from .r_inner_prefix_v1 import array, checked
from .joint_model import as_event_view
from .source_quality_compare_v1 import _camera, _windows
from rinkfit_start.native_scene import NativeScene

OBJECTIVE_POLICIES = {
    'rinkfit.sourceUV_trueRS_finite_capture.v1': 'sourceUV_trueRS_capture_v1: ',
    'rinkfit.sourceUV_trueRS_finite_capture.v2': 'sourceUV_trueRS_capture_v2: ',
}
SCHEMA = 'rinkfit.first4_true_RS_capture_independent_audit.v1'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def ref(p):
    p = Path(p).resolve()
    return dict(file=str(p), sha256=file_sha256(p))


def metric(error, valid):
    e = np.asarray(error, float)
    good = np.asarray(valid, bool) & np.isfinite(e)
    z = e[good]
    return dict(denominator=len(e), valid=int(good.sum()), failed=int((~good).sum()),
        full_RMS_px=float(np.sqrt(np.mean(z*z))) if len(e) and good.all() else None,
        available_RMS_px=float(np.sqrt(np.mean(z*z))) if len(z) else None,
        available_P95_px=float(np.quantile(z, .95)) if len(z) else None,
        available_max_px=float(z.max()) if len(z) else None)


def verify_checkpoint(path, packet_hash, model, aux, active, queries, binding):
    """Stream every serialized block/map and independently recompute native FNV."""
    h = 14695981039346656037
    def canonical(s):
        nonlocal h
        for v in s.encode('utf8'):
            h = ((h ^ v)*1099511628211) & ((1 << 64)-1)
    names = set()
    numerical_enclosures = {}
    found = {k: set() for k in ('station', 'row', 'alpha')}
    with Path(path).open(encoding='utf8') as stream:
        def line():
            s = stream.readline()
            if not s:
                raise ValueError('truncated complete checkpoint')
            return s.rstrip('\r\n')
        def count():
            v = int(line())
            if v < 0:
                raise ValueError('negative checkpoint allocation')
            return v
        if line() != 'RINKFIT_AL_CHECKPOINT_V1':
            raise ValueError('unsupported checkpoint')
        fingerprint = _quoted_fields(line())
        if len(fingerprint) != 1:
            raise ValueError('missing runtime fingerprint')
        phase = line().split()
        if len(phase) != 7 or int(phase[0]) != 2 or float(phase[4]) != 0:
            raise ValueError('wrong checkpoint phase or retained AL rho')
        if line() != '0' or line() != '0' or count() != 1:
            raise ValueError('old AL duals, best states, or missing current state')
        token = _quoted_fields(line())
        if token != [binding['state_token']]:
            raise ValueError('capture state token mismatch')
        identity = _quoted_fields(line())
        if len(identity) != 6 or identity[0] != packet_hash or identity[5] != binding['objective_id'] or identity[5] not in OBJECTIVE_POLICIES:
            raise ValueError('complete checkpoint packet/objective mismatch')
        canonical(' '.join(_quoted(v) for v in identity)+'\n')
        size = count()
        canonical(str(size)+'\n')
        if size < len(model['blocks']):
            raise ValueError('partial physical checkpoint')
        for index in range(size):
            s = line()
            name, end = json.JSONDecoder().raw_decode(s)
            flag = s[end:].strip()
            v = line().split()
            length = int(v[0])
            values = [float(x) for x in v[1:]]
            if not name or name in names or flag not in ('0', '1') or length != len(values) or not length or not all(map(math.isfinite, values)):
                raise ValueError('invalid complete checkpoint parameter')
            names.add(name)
            canonical(_quoted(name)+' '+flag+' '+str(length)+''.join(' '+_cpp_hex(x) for x in values)+'\n')
            if index < len(model['blocks']):
                block = model['blocks'][index]
                if name != block['name'] or flag != '0' or values != block['values']:
                    raise ValueError('serialized physical model differs at '+name)
            elif name.startswith(('station/', 'row/', 'alpha/')):
                kind, raw = name.split('/')
                i = int(raw)
                if not 0 <= i < len(active) or not active[i] or length != 1 or values[0] != aux[kind][i]:
                    raise ValueError('checkpoint auxiliary differs at '+name)
                found[kind].add(i)
            elif name.startswith('rig/') and name.endswith('/numerical_zoom_enclosure'):
                rid = name.split('/')[1]
                if rid not in model['rigs'] or length != 2 or values[0] > values[1] or flag != '0':
                    raise ValueError('invalid numerical zoom enclosure '+name)
                numerical_enclosures[rid] = values
            elif name != 'budget_sq' or values != [1.] or flag != '0':
                raise ValueError('unrecognized capture checkpoint parameter '+name)
        maps = []
        for _ in range(2):
            length = count()
            canonical(str(length)+'\n')
            data = {}
            previous = None
            for _ in range(length):
                fields = _quoted_fields(line())
                if len(fields) != 2 or fields[0] in data or (previous is not None and fields[0] <= previous):
                    raise ValueError('unordered/duplicate complete checkpoint map')
                previous = fields[0]
                data[fields[0]] = fields[1]
                canonical(_quoted(fields[0])+' '+_quoted(fields[1])+'\n')
            maps.append(data)
        if count() != 0 or line() != 'END_RINKFIT_AL_CHECKPOINT' or stream.read().strip():
            raise ValueError('old best state or trailing checkpoint data')
    if format(h, '016x') != token[0]:
        raise ValueError('complete state FNV token corrupted')
    ids = set(map(int, np.flatnonzero(active)))
    if found['row'] != ids or found['alpha'] != (ids & set(queries)):
        raise ValueError('complete current row/alpha set differs')
    if maps[0].get('joint_packet_sha256') != packet_hash or not maps[0].get('RS_profile_policy', '').startswith(OBJECTIVE_POLICIES[binding['objective_id']]):
        raise ValueError('complete mode/discrete packet identity differs')
    position = 0
    for run in maps[0].get('finite_material_paths_RLE_u32_v1', '').split(';'):
        if not run:
            continue
        count_text, value = run.split(':')
        count_value, value = int(count_text), int(value)
        if count_value <= 0 or position+count_value > len(active) or not np.all(aux['path'][position:position+count_value] == value):
            raise ValueError('complete finite path map differs')
        position += count_value
    if position != len(active):
        raise ValueError('incomplete finite path map')
    return dict(state_token=token[0], objective_id=identity[5], runtime_fingerprint=fingerprint[0],
        identity_fields=identity, complete_parameters=len(names), station_rows=found['station'],
        row_rows=len(found['row']), alpha_rows=len(found['alpha']), all_parameter_values_verified=True,
        FNV_token_independently_recomputed=True, old_AL_duals_present=False,
        numerical_zoom_enclosures=numerical_enclosures)


def query_point(q, alpha):
    a = np.asarray(q['alpha'], float)
    if not np.isfinite(alpha) or len(a) < 2 or alpha < a[0] or alpha > a[-1]:
        return np.full(2, np.nan), False
    if q.get('denominator'):
        z = (alpha-q.get('alpha_origin', 0.))/q.get('alpha_scale', 1.)
        d = np.polynomial.polynomial.polyval(z, q['denominator'])
        if not np.isfinite(d) or abs(d) < 1e-14:
            return np.full(2, np.nan), False
        uv = np.array([np.polynomial.polynomial.polyval(z, q['numerator_x']),
            np.polynomial.polynomial.polyval(z, q['numerator_y'])])/d
    else:
        uv = np.array([np.interp(alpha, a, np.asarray(q['native_uv'])[:, k]) for k in (0, 1)])
    return uv, bool(np.isfinite(uv).all())


def audit_model_block_closed_bounds(model):
    """Check every serialized physical coordinate against its exact closed box.

    In particular a physical-point path still consumes its affine body q; a
    valid scalar station does not certify that different parameter domain.
    Bounds are the model's declared numerical/material chart, not new source
    qualification, metric truth, a sampled domain, or an expanded tolerance.
    """
    violations = []
    bad_blocks = []
    coordinates = 0
    for bi, block in enumerate(model['blocks']):
        name = block['name']
        try:
            value = np.asarray(block['values'], float)
            lower = np.asarray(block['lower'], float)
            upper = np.asarray(block['upper'], float)
        except (KeyError, TypeError, ValueError):
            violations.append(dict(block_index=bi, block_name=name, coordinate_index=None,
                reason='missing_or_nonnumeric_closed_block_bounds'))
            bad_blocks.append(bi)
            continue
        if value.ndim != 1 or not len(value) or lower.shape != value.shape or upper.shape != value.shape:
            violations.append(dict(block_index=bi, block_name=name, coordinate_index=None,
                reason='closed_block_bounds_shape_mismatch'))
            bad_blocks.append(bi)
            continue
        coordinates += len(value)
        finite = np.isfinite(value) & np.isfinite(lower) & np.isfinite(upper)
        bad = ~finite | (lower > upper) | (value < lower) | (value > upper)
        if not np.any(bad):
            continue
        bad_blocks.append(bi)
        for coordinate in np.flatnonzero(bad):
            def scalar(x):
                return float(x) if np.isfinite(x) else str(float(x))
            violations.append(dict(block_index=bi, block_name=name, coordinate_index=int(coordinate),
                value=scalar(value[coordinate]), lower=scalar(lower[coordinate]), upper=scalar(upper[coordinate]),
                reason='nonfinite_closed_block_coordinate_or_bounds' if not finite[coordinate] else
                    'unordered_closed_block_bounds' if lower[coordinate] > upper[coordinate] else
                    'below_closed_lower_bound' if value[coordinate] < lower[coordinate] else 'above_closed_upper_bound'))
    return dict(schema='rinkfit.independent_model_block_closed_bounds.v1',
        blocks_checked=len(model['blocks']), coordinates_checked=coordinates,
        tolerance=0., bounds_are_closed=True, domain_expanded=False,
        passed=not bad_blocks, violating_block_indices=bad_blocks, violations=violations,
        numerical_material_chart_is_not_new_source_qualification=True)


def finite_world(model, packet, arrays, aux, selected):
    """Finite chart, row-specific domains and existing source-width operator."""
    n = len(selected)
    world = np.full((n, 3), np.nan)
    legal = np.zeros(n, bool)
    reasons = np.full(n, 'not_evaluated', dtype='U100')
    scene = NativeScene(as_event_view(model, packet['shot_ids'][0]))
    scene.audit_closed_block_bounds = audit_model_block_closed_bounds(model)
    invalid_blocks = set(scene.audit_closed_block_bounds['violating_block_indices'])
    domains = {int(d['row_index']): d for d in packet.get('source_station_domains', {}).get('rows', [])}
    for i in np.flatnonzero(selected):
        m, p, s = int(arrays['material_index'][i]), int(aux['path'][i]), float(aux['station'][i])
        try:
            path = model['materials'][m]['paths'][p]
            if m < 0 or p < 0:
                raise IndexError()
            dependencies = [int(path[k]) for k in ('geometry_block', 'advertisement_block') if int(path.get(k, -1)) >= 0]
            if any(bi in invalid_blocks for bi in dependencies):
                reasons[i] = 'material_dependency_outside_closed_block_bounds'
                continue
            allowed = packet.get('allowed_paths', {}).get(str(i), None)
            if allowed is not None and p not in allowed:
                reasons[i] = 'outside_source_allowed_path'
                continue
            intervals = path.get('allowed_station_intervals', [[0., 1.]])
            d = domains.get(int(i))
            if d:
                if p != int(d['path_index']):
                    reasons[i] = 'outside_source_station_domain_path'
                    continue
                intervals = [[max(a[0], b[0]), min(a[1], b[1])] for a in intervals for b in d['allowed_station_intervals'] if max(a[0], b[0]) <= min(a[1], b[1])]
            if not np.isfinite(s) or not any(lo <= s <= hi for lo, hi in intervals):
                reasons[i] = 'outside_finite_station_domain'
                continue
            legal[i] = True
            reasons[i] = ''
        except (IndexError, KeyError, TypeError):
            reasons[i] = 'unknown_material_or_path'
    for mi, pi in sorted(set(zip(arrays['material_index'][legal].tolist(), aux['path'][legal].tolist()))):
        ids = np.flatnonzero(legal & (arrays['material_index'] == mi) & (aux['path'] == pi))
        try:
            world[ids] = scene.world(int(mi), int(pi), aux['station'][ids])
        except (ValueError, IndexError, KeyError):
            legal[ids] = False
            reasons[ids] = 'material_world_evaluation_failed'
    if np.any(selected & (arrays.get('stroke_pair_index', np.full(n, -1)) >= 0)):
        from .measurement_widths import apply_source_paint_interfaces
        # This existing independent operator verifies the typed two-flank binding.
        apply_source_paint_interfaces(model, packet, world, selected, arrays['material_index'], aux['path'], aux['station'],
            pair_index=arrays.get('stroke_pair_index'), source_side=arrays.get('stroke_source_side'))
    return scene, world, legal, reasons


def project_all(model, packet, arrays, aux, selected):
    n = len(selected)
    scene, world, finite, reason = finite_world(model, packet, arrays, aux, selected)
    predicted = np.full((n, 2), np.nan)
    row = np.full(n, np.nan)
    equality = np.full(n, np.nan)
    valid = np.zeros(n, bool)
    root_exists = np.zeros(n, bool)
    certificate = np.zeros(n, bool)
    saved_uv = np.full((n, 2), np.nan)
    saved_equality = np.full(n, np.nan)
    contraction = np.full(n, np.nan)
    optics = {}
    for si, sid in enumerate(packet['shot_ids']):
        ids = np.flatnonzero(selected & finite & (arrays['shot_index'] == si))
        camera = _camera(model, sid)
        optics[sid] = camera.optical_domain_certificate()
        if not len(ids):
            continue
        times = (arrays['native_pts'][ids]-model['shots'][sid]['time_origin_pts'])/12800.
        result = camera.project_rs(world[ids], times)
        explicit = camera.at(world[ids], times+camera.tau*(aux['row'][ids]/1079.-.5))
        predicted[ids], row[ids], equality[ids] = result['uv'], result['row'], result['root_equality_px']
        valid[ids], root_exists[ids], certificate[ids] = result['valid'], result['bracketed'], result['certificate']['certified']
        contraction[ids] = result['certificate']['contraction_bound']
        saved_uv[ids] = explicit['uv']
        saved_equality[ids] = aux['row'][ids]-explicit['uv'][:, 1]
        reason[ids[~result['bracketed']]] = 'RS_no_legal_sensor_bracket'
        reason[ids[result['bracketed'] & ~result['certificate']['certified']]] = 'whole_readout_RS_or_Brown_uncertified'
        reason[ids[result['bracketed'] & result['certificate']['certified'] & ~result['valid']]] = 'RS_root_equality_or_projection_invalid'
    return dict(world=world, finite=finite, predicted=predicted, row=row, root_equality=equality,
        valid=valid & finite, reason=reason, bracketed=root_exists, whole_interval_certificate=certificate,
        saved_uv=saved_uv, saved_equality=saved_equality, contraction=contraction,
        optical_domain_by_shot=optics, whole_scene_domain=scene.check_physics(),
        model_block_closed_bounds=scene.audit_closed_block_bounds)


def audit(packet_path, candidate, output, *, baseline_import=None, native_domain_receipt=None):
    started = time.perf_counter()
    pp, cpdir, out = Path(packet_path).resolve(), Path(candidate).resolve(), Path(output).resolve()
    if out.exists():
        raise FileExistsError('immutable audit output already exists')
    packet, model = read(pp), read(cpdir/'model.json')
    if packet['shot_ids'] != ['s01', 's02', 's03', 's04'] or set(model['shots']) != set(packet['shot_ids']) or len(model['rigs']) != 3:
        raise ValueError('explicit first four shots / three shared rigs required')
    seen = set()
    arrays = {k: array(packet, k, pp.parent, seen) for k in packet['arrays']}
    n = packet['row_count']
    if any(len(a) != n for a in arrays.values()):
        raise ValueError('incomplete source arrays')
    active = (arrays['admission'] == 1) & (arrays['fit_role'] == 1)
    pending = (arrays['admission'] == 1) & (arrays['fit_role'] != 1)
    digest = file_sha256(pp)
    binding = read(cpdir/'source_uv_capture_state_binding.json')
    if binding.get('schema') != 'rinkfit.sourceUV_trueRS_capture_state_binding.v1' or binding['objective_id'] not in OBJECTIVE_POLICIES or binding['packet_sha256'] != digest or binding['model_sha256'] != file_sha256(cpdir/'model.json'):
        raise ValueError('capture model/packet/objective binding differs')
    if binding.get('row_alpha_optimization_dof') != 0 or not binding.get('stored_valid_rows_are_derived_true_RS_roots') or binding.get('formal_Query_closed') is not False:
        raise ValueError('capture mode semantics differ')
    am = read(cpdir/'explicit_nuisance/manifest.json')
    if am['packet_sha256'] != digest or am['raw_rows'] != n:
        raise ValueError('actual auxiliary generation differs')
    aux, dependencies = {}, [ref(pp), ref(cpdir/'model.json'), ref(cpdir/'engineering_checkpoint.json'), ref(cpdir/'source_uv_capture_state_binding.json'), ref(cpdir/'explicit_nuisance/manifest.json')]
    for k, (name, dtype) in AUX.items():
        f = cpdir/'explicit_nuisance'/name
        if file_sha256(f) != am['files'][name]['sha256'] or f.stat().st_size != n*np.dtype(dtype).itemsize:
            raise ValueError('changed actual CP auxiliary '+k)
        aux[k] = np.memmap(f, mode='r', dtype=dtype, shape=(n,))
        if not np.isfinite(aux[k]).all():
            raise ValueError('nonfinite CP auxiliary '+k)
        dependencies.append(ref(f))
    qmap = {int(q['row_index']): q for q in packet['queries']}
    if len(qmap) != len(packet['queries']):
        raise ValueError('duplicate Query row')
    proof = verify_checkpoint(cpdir/'engineering_checkpoint.json', digest, model, aux, active, qmap, binding)
    if proof.pop('station_rows') != set(map(int, np.flatnonzero(active & (arrays['kind'] != 1)))):
        raise ValueError('complete station parameter set differs from curve source rows')
    group_hash, group_count = _runtime_group_digest(packet, pp.parent, seen)
    dependencies.extend(dict(file=str(p), sha256=h) for p, h in sorted(seen))
    print('complete CP token, every physical block, source array, W and auxiliary verified', flush=True)
    state = project_all(model, packet, arrays, aux, active)
    # Inactive/pending rows are retained in all denominators. They are never
    # promoted through a numerical replay or supplied with invented CP values.
    error = state['predicted']-arrays['uv']
    raw = np.linalg.norm(error, axis=1)
    qmask = np.zeros(n, bool)
    qmask[list(qmap)] = True
    normal_ok = np.isfinite(arrays['profile_normal']).all(axis=1) & (np.linalg.norm(arrays['profile_normal'], axis=1) > 0)
    normal = np.abs(np.sum(error*arrays['profile_normal'], axis=1))
    operator = np.where(qmask, normal, raw)
    opvalid = state['valid'] & (~qmask | normal_ok)
    labels = np.full(n, 'UNASSIGNED', dtype=object)
    for m in np.unique(arrays['material_index']):
        if 0 <= int(m) < len(model['materials']):
            labels[arrays['material_index'] == m] = model['materials'][int(m)]['label']
    def cohort(sel):
        return dict(raw_source_2D_UV=metric(raw[sel], state['valid'][sel]),
            source_operator_diagnostic=metric(operator[sel], opvalid[sel]),
            fixed_entity_2D_UV=metric(raw[sel & (arrays['kind'] == 1)], state['valid'][sel & (arrays['kind'] == 1)]),
            source_Query_normal_diagnostic=metric(normal[sel & qmask], state['valid'][sel & qmask] & normal_ok[sel & qmask]),
            finite_station_or_material_failed=int((sel & ~state['finite']).sum()),
            material_parameter_closed_bound_failed=int((sel & (state['reason'] == 'material_dependency_outside_closed_block_bounds')).sum()),
            no_legal_RS_bracket=int((sel & state['finite'] & ~state['bracketed']).sum()),
            RS_whole_interval_uncertified=int((sel & state['finite'] & ~state['whole_interval_certificate']).sum()),
            failure_reasons=dict(Counter(state['reason'][sel & ~state['valid']].tolist())))
    byshot = {}
    for si, sid in enumerate(packet['shot_ids']):
        sel = active & (arrays['shot_index'] == si)
        byshot[sid] = cohort(sel)
        scope = model['shots'][sid]['scope']
        pts = np.arange(scope['first_pts'], scope['last_pts']+256, 256, dtype=np.int64)
        at = np.searchsorted(pts, arrays['native_pts'][sel])
        if np.any(at >= len(pts)) or not np.array_equal(pts[at], arrays['native_pts'][sel]):
            raise ValueError('source clock outside its actual native shot')
        sse, den, bad = np.zeros(len(pts)), np.zeros(len(pts), int), np.zeros(len(pts), int)
        good = opvalid[sel] & np.isfinite(operator[sel])
        np.add.at(den, at, 1)
        np.add.at(bad, at, ~good)
        np.add.at(sse, at, np.where(good, operator[sel]**2, 0.))
        byshot[sid].update(native_frames=len(pts), source_exposures=int((den > 0).sum()),
            temporal_source_operator=_windows(pts, sse, den, bad, den > 0),
            pending_source_rows=int((pending & (arrays['shot_index'] == si)).sum()))
    query_error, query_legal = np.full(n, np.nan), np.zeros(n, bool)
    for i, q in qmap.items():
        value, ok = query_point(q, aux['alpha'][i])
        query_legal[i] = ok
        if ok:
            query_error[i] = np.linalg.norm(state['predicted'][i]-value)
    statuses = {}
    for name in ('RS_status.u8', 'formal_Query_root_status.u8'):
        f = cpdir/'derived_auxiliary_binding'/name
        v = np.fromfile(f, 'u1')
        if v.shape != (n,) or np.any(~np.isin(v, [0, 1, 2])):
            raise ValueError('invalid native derived status bitmap')
        statuses[name] = v
        dependencies.append(ref(f))
    nrpath = cpdir/'derived_auxiliary_binding/receipt.json'
    nr = read(nrpath)
    dependencies.append(ref(nrpath))
    if nr != binding['actual_derived_row_status']:
        raise ValueError('native boundary receipt differs from bound copy')
    for key, value in [('active_rows', int(active.sum())), ('valid_RS_rows', int((active & (statuses['RS_status.u8'] == 1)).sum())), ('failed_RS_rows_retained', int((active & (statuses['RS_status.u8'] == 2)).sum())), ('formal_Query_rows', int((active & qmask).sum())), ('formal_Query_roots_valid', int((active & qmask & (statuses['formal_Query_root_status.u8'] == 1)).sum())), ('formal_Query_roots_failed_retained', int((active & qmask & (statuses['formal_Query_root_status.u8'] == 2)).sum()))]:
        if nr[key] != value:
            raise ValueError('native diagnostic bitmap disagrees with receipt '+key)
    native_g = None
    if native_domain_receipt:
        dp = Path(native_domain_receipt).resolve()
        native_g = read(dp)
        if native_g['state_token'] != binding['state_token']:
            raise ValueError('native g receipt belongs to another complete state')
        dependencies.append(ref(dp))
    baseline = None
    if baseline_import:
        bp = Path(baseline_import).resolve()
        bm = read(bp)
        dependencies.append(ref(bp))
        oldmodel_path = checked(bm['baseline_model'], bp.parent, seen)
        oldmodel = read(oldmodel_path)
        extraction_path = checked(bm['original_source_and_explicit_full_cp_proof'], bp.parent, seen)
        extraction = read(extraction_path)
        oldpacket_path = checked(extraction['packet'], extraction_path.parent, seen)
        oldpacket = read(oldpacket_path)
        N = int(bm['original_rows'])
        for key in ('source_dataset_ids', 'source_row_ids', 'source_family_ids'):
            if packet[key][:N] != oldpacket[key]:
                raise ValueError('original S01 source identity prefix differs')
        for key in oldpacket['arrays']:
            if key not in AUX and _prefix_hash(arrays[key][:N]) != extraction['source_arrays_sha256'][key]:
                raise ValueError('original S01 measurement array differs '+key)
        if _runtime_query_digest([q for q in packet['queries'] if q['row_index'] < N]) != extraction['original_queries_sha256']:
            raise ValueError('original S01 frozen Query differs')
        baux = {k: np.array(v, copy=True) for k, v in aux.items()}
        files = bm['files']
        ids = np.memmap(checked(files['rows.u64'], bp.parent, seen), dtype='<u8', mode='r', shape=tuple(files['rows.u64']['shape'])).astype(int)
        if not np.array_equal(ids, np.flatnonzero(active & (arrays['shot_index'] == 0))) or len(ids) != bm['preserved_active_rows']:
            raise ValueError('original S01 active membership differs')
        for key, (name, dtype) in AUX.items():
            baux[key][ids] = np.memmap(checked(files[name], bp.parent, seen), dtype=dtype, mode='r', shape=tuple(files[name]['shape']))
        base_selected = np.zeros(n, bool)
        base_selected[ids] = True
        before = project_all(oldmodel, packet, arrays, baux, base_selected)
        berror = before['predicted'][ids]-arrays['uv'][ids]
        braw = np.linalg.norm(berror, axis=1)
        bop = braw.copy()
        use = qmask[ids]
        bop[use] = np.abs(np.sum(berror[use]*arrays['profile_normal'][ids[use]], axis=1))
        baseline = dict(manifest=ref(bp), baseline_model=ref(oldmodel_path), complete_original_CP_proof=bm['complete_baseline_state'],
            active_rows=len(ids), original_UV_ID_PTS_Query_unchanged=True,
            before_raw_source_2D_UV=metric(braw, before['valid'][ids]), before_source_operator=metric(bop, before['valid'][ids]),
            after=cohort(base_selected), prediction_change_2D=metric(np.linalg.norm(state['predicted'][ids]-before['predicted'][ids], axis=1), state['valid'][ids] & before['valid'][ids]),
            maximum_absolute_world_coordinate_change_L=float(np.nanmax(np.abs(state['world'][ids]-before['world'][ids]))),
            stations_changed=int(np.count_nonzero(aux['station'][ids] != baux['station'][ids])),
            paths_changed=int(np.count_nonzero(aux['path'][ids] != baux['path'][ids])), shared_main_forward_assumed_unchanged=False)
        dependencies.extend([ref(oldmodel_path), ref(extraction_path), ref(oldpacket_path)])
    out.mkdir(parents=True)
    exported = {}
    for name, value in [('raw_station_RS_predicted_uv.f64', state['predicted'].astype('<f8')), ('raw_station_RS_error_native_uv.f64', error.astype('<f8')), ('raw_station_RS_valid.u8', state['valid'].astype('u1')), ('world.f64', state['world'].astype('<f8')), ('independent_RS_row.f64', state['row'].astype('<f8')), ('independent_RS_equality.f64', state['root_equality'].astype('<f8')), ('finite_station_valid.u8', state['finite'].astype('u1')), ('saved_Query_closure_error.f64', query_error.astype('<f8')), ('saved_Query_alpha_valid.u8', query_legal.astype('u1'))]:
        value.tofile(out/name)
        exported[name] = dict(**ref(out/name), dtype=value.dtype.str, shape=list(value.shape))
    bad = []
    for i in np.flatnonzero(active & ~state['valid']):
        bad.append(dict(row_index=int(i), source_row_id=packet['source_row_ids'][i], shot=packet['shot_ids'][int(arrays['shot_index'][i])], native_pts=int(arrays['native_pts'][i]), entity=str(labels[i]), kind=int(arrays['kind'][i]), source_uv=arrays['uv'][i].tolist(), station=float(aux['station'][i]), path=int(aux['path'][i]), reason=str(state['reason'][i])))
    accepted_w = sum(int(g['rank']) for g in packet['groups'])+2*len(packet.get('independent_rows', []))
    report = dict(schema=SCHEMA, packet=ref(pp), candidate_directory=str(cpdir), model=ref(cpdir/'model.json'), complete_CP=proof,
        original_UV_and_failure_denominators_preserved=True, raw_rows=n, active_rows=int(active.sum()),
        source_good_numeric_pending_rows=int(pending.sum()), source_not_admitted_rows=int((arrays['admission'] != 1).sum()),
        unknown_material_active_rows=int((active & (labels == 'UNASSIGNED')).sum()),
        all=cohort(active), by_shot=byshot,
        by_type={str(int(k)): cohort(active & (arrays['kind'] == k)) for k in np.unique(arrays['kind'][active])},
        by_entity={str(k): cohort(active & (labels == k)) for k in sorted(set(labels[active].tolist()))},
        independent_valid_rows=int((active & state['valid']).sum()), independent_invalid_rows=bad,
        maximum_saved_row_to_independent_root_difference_px=float(np.max(np.abs(aux['row'][active]-state['row'][active]), initial=0.)),
        maximum_saved_RS_equality_px=float(np.max(np.abs(state['saved_equality'][active]), initial=0.)),
        native_RS_status_mismatch_rows=np.flatnonzero(active & ((statuses['RS_status.u8'] == 1) != state['valid'])).tolist(),
        native_bitmaps_used_for_independent_validity=False,
        formal_Query_debt=dict(total=int((active & qmask).sum()), native_branch_local_numerical_root_valid=nr['formal_Query_roots_valid'],
            native_branch_local_numerical_root_failed_retained=nr['formal_Query_roots_failed_retained'],
            native_root_status_bitmaps_rehashed=True, native_numerical_root_counts_independently_certified=False,
            actual_saved_station_alpha_2D_closure=metric(query_error[active & qmask], state['valid'][active & qmask] & query_legal[active & qmask]),
            actual_saved_station_alpha_closed_within_1e_minus7=int((active & qmask & query_legal & state['valid'] & (query_error <= 1e-7)).sum()),
            alpha_optimized=False, root_candidates_materialized_into_CP=False, formal_Query_closed=False,
            native_local_root_bitmap_is_not_saved_Query_closure_or_global_uniqueness=True),
        original_S01_full_d78_comparison=baseline,
        original_W_likelihood_consumed=False, original_W_rank_preserved_and_hashed=accepted_w,
        original_W_group_digest=group_hash, original_W_group_count=group_count,
        source_Query_normal_is_diagnostic_only_in_coarse_2D_objective=True,
        independent_whole_scene_domain=state['whole_scene_domain'], independent_optical_domain_by_shot=state['optical_domain_by_shot'],
        independent_model_block_closed_bounds=state['model_block_closed_bounds'],
        active_path_geometry_and_advertisement_closed_bounds_consumed=True,
        physical_point_station_does_not_replace_body_parameter_domain=True,
        native_physical_g_receipt=native_g, native_g_visibility_independently_reimplemented=False,
        full_physical_visibility_acceptance_claimed=False, formal_Phase_II_complete=False, quality_accepted=False,
        elapsed_seconds=time.perf_counter()-started, files=exported,
        dependencies=dependencies, producer=ref(__file__), independent_sources=[ref(Path(__file__).parent/'joint_model.py'),ref(Path(__file__).parent/'source_quality_compare_v1.py'),ref(Path(__file__).parents[1]/'rinkfit_start/native_camera.py'),ref(Path(__file__).parents[1]/'rinkfit_start/native_scene.py')])
    (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--packet', required=True)
    ap.add_argument('--candidate', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--baseline-import')
    ap.add_argument('--native-domain-receipt')
    a = ap.parse_args(argv)
    report = audit(a.packet, a.candidate, a.output, baseline_import=a.baseline_import, native_domain_receipt=a.native_domain_receipt)
    print(json.dumps(dict(active_rows=report['active_rows'], independently_valid=report['independent_valid_rows'], full_2D_RMS_px=report['all']['raw_source_2D_UV']['full_RMS_px'], report=str(Path(a.output)/'report.json')), ensure_ascii=False))


if __name__ == '__main__':
    main()
