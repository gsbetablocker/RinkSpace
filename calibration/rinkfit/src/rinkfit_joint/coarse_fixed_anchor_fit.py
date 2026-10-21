"""Physical coarse capture from original fixed-entity UV, with independent RS.

No station/Query alpha is optimized. Small physical yaw/pitch/zoom offsets and
the shared non-main head rotation move a real native model into source capture.
Original source rows/UV and the complete scene/material/S01 state stay intact.
This engineering initializer is followed by the unchanged complete packet.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from .io_utils import read_json, write_json, file_sha256, object_sha256
from .joint_model import as_event_view
from .native_baseline_init import _head
from .screen_to_native_seed import _stats
from rinkfit_start.native_scene import NativeScene, camera_from_event
from rinkfit_start.native_camera import rotation


def initialize_from_fixed_anchors(model_file, source_arrays_file, source_metadata_file, output):
    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError('Use a new immutable coarse physical candidate')
    model = read_json(model_file)
    original = copy.deepcopy(model)
    arrays = np.load(source_arrays_file, allow_pickle=False)
    metadata = read_json(source_metadata_file)
    source_rows = np.flatnonzero(arrays['kind'] == 1)
    source_uv = arrays['uv'][source_rows].copy()
    source_pts = arrays['native_pts'][source_rows]
    source_shots = arrays['shot_index'][source_rows]
    E = arrays['E'][source_rows]
    if not np.isfinite(E).all() or np.any(E <= 0):
        raise ValueError('Original engineering scales must be positive')
    scene = NativeScene(as_event_view(model, 's01'))
    world = []
    for row in source_rows:
        material, branch = int(arrays['material_index'][row]), int(arrays['path'][row])
        path = scene.path(material, branch)
        if not path.get('physical_point') or len(path.get('points', [])) != 1:
            raise ValueError('A fixed anchor must be one explicit physical entity')
        world.append(scene.world(material, branch, 0.))
    world = np.asarray(world)
    before = np.full(source_uv.shape, np.nan)
    after = before.copy()
    reports = {}
    frozen = object_sha256(dict(scene=model['blocks'][model['scene_block']], materials=model['materials'],
                                main=model['rigs']['main'], S01=model['shots']['s01'],
                                S01_blocks=[model['blocks'][i] for i in model['shots']['s01']['control_blocks']],
                                packed=[model['blocks'][c['block']] for c in model['shots']['s01'].get('control_coordinates', [])],
                                main_blocks=[model['blocks'][model['rigs']['main'][k]] for k in ('head_block', 'optics_block')]))
    for rid in ('secondary', 'end', 'main'):
        sids = [sid for sid, shot in model['shots'].items() if shot['rig_id'] == rid and sid != 's01']
        refs, lo, hi, scale = [], [], [], []
        rig = model['rigs'][rid]
        hb = model['blocks'][rig['head_block']]
        base_head = np.asarray(hb['values']).copy()
        if rid != 'main':
            for j in range(3):
                refs.append(('head', None, j))
                lo.append(hb['lower'][j]-base_head[j])
                hi.append(hb['upper'][j]-base_head[j])
                scale.append(.1)
        cameras = {}
        for sid in sids:
            shot = model['shots'][sid]
            reference = model['blocks'][shot['reference_angles_block']]
            if not reference.get('constant', False):
                for j in range(2):
                    refs.append(('angle', sid, j))
                    lo.append(reference['lower'][j]-reference['values'][j])
                    hi.append(reference['upper'][j]-reference['values'][j])
                    scale.append(.1)
            controls = [model['blocks'][i] for i in shot['control_blocks']]
            refs.append(('zoom', sid, 2))
            lo.append(max(b['lower'][2]-b['values'][2] for b in controls))
            hi.append(min(b['upper'][2]-b['values'][2] for b in controls))
            scale.append(1.)
            index = metadata['shot_ids'].index(sid)
            rows = np.flatnonzero(source_shots == index)
            camera = camera_from_event(as_event_view(model, sid))
            cameras[sid] = dict(camera=camera, rows=rows,
                                controls=camera.controls.copy(),
                                reference=np.asarray(reference['values']).copy(),
                                time=(source_pts[rows]-shot['time_origin_pts'])/12800.)
        selected = np.concatenate([cameras[sid]['rows'] for sid in sids])
        lower, upper, scales = map(np.asarray, (lo, hi, scale))

        def evaluate(x):
            head = base_head.copy()
            angles = {sid:v['reference'].copy() for sid,v in cameras.items()}
            zoom = {sid:0. for sid in sids}
            for (kind, sid, j), value in zip(refs, x):
                if kind == 'head':
                    head[j] += value
                elif kind == 'angle':
                    angles[sid][j] += value
                else:
                    zoom[sid] = value
            R0, n, b = _head(head, rig['camera'])
            values = []
            invalid = 0
            for sid in sids:
                entry = cameras[sid]
                cam = entry['camera']
                cam.R0 = rotation(b, angles[sid][1])@R0@rotation(n, angles[sid][0])
                cam.n, cam.b = n, b
                cam.controls[:, 2] = entry['controls'][:, 2]+zoom[sid]
                rows = entry['rows']
                result = cam.project_rs(world[rows], entry['time'])
                invalid += int((~result['valid']).sum())
                values.append(result['uv'])
            return np.concatenate(values), invalid

        x = np.zeros(len(refs))
        predicted, invalid = evaluate(x)
        before[selected] = predicted
        if invalid:
            reports[rid] = dict(started=False, reason='fixed_anchor_native_RS_or_domain_invalid_at_initial_state',
                                 invalid=invalid, denominator=len(selected), dropped_rows=0)
            after[selected] = predicted
            continue
        def residual(pred):
            return ((pred-source_uv[selected])/E[selected, None]).ravel()
        current = residual(predicted)
        initial_cost = cost = float(current@current/2)
        # The offset problem is tiny and dense. The bound-aware exact TRF
        # subproblem handles its head/reference gauge far better than clipping
        # a full unconstrained normal step at an active parameter bound.
        from scipy.optimize import least_squares
        invalid_trials = [0]
        rejection_value = 1e12*(1+np.linalg.norm(current))
        def source_residual(parameters):
            value, bad = evaluate(parameters)
            if bad:
                invalid_trials[0] += 1
                # Reject the entire trial: no row is removed, assigned zero, or
                # reported as a successful subset of the frozen anchor set.
                return np.full(len(current), rejection_value)
            return residual(value)
        result = least_squares(source_residual, x, bounds=(lower, upper),
                               x_scale=scales, ftol=1e-8, xtol=1e-10, gtol=1e-7)
        x = result.x
        cost = float(result.fun@result.fun/2)
        accepted = None
        evaluations = result.nfev
        message = result.message
        predicted, invalid = evaluate(x)
        if invalid:
            raise ValueError('Accepted coarse source candidate lost fixed-anchor RS roots')
        after[selected] = predicted
        for (kind, sid, j), value in zip(refs, x):
            if kind == 'head':
                hb['values'][j] += float(value)
            elif kind == 'angle':
                model['blocks'][model['shots'][sid]['reference_angles_block']]['values'][j] += float(value)
            else:
                for bi in model['shots'][sid]['control_blocks']:
                    model['blocks'][bi]['values'][2] += float(value)
        reports[rid] = dict(started=True, denominator=len(selected), cost_before=initial_cost,
                             cost_after=cost, accepted_steps=accepted, evaluations=evaluations,
                             message=message, solver_success=bool(result.success), rejected_invalid_RS_trials=invalid_trials[0],
                             changed_physical_coordinates=len(refs),
                             original_source_E_used_as='engineering scale only, not covariance or formal W',
                             source_UV_shifted=False, station_optimized=False,
                             source_rows_dropped=0, final_invalid=invalid)
    # S01 source is still evaluated, even though this initializer does not move it.
    sid = 's01';rows = np.flatnonzero(source_shots == 0)
    cam = camera_from_event(as_event_view(model, sid))
    t = (source_pts[rows]-model['shots'][sid]['time_origin_pts'])/12800.
    same = cam.project_rs(world[rows], t)
    before[rows] = after[rows] = same['uv']
    now = object_sha256(dict(scene=model['blocks'][model['scene_block']], materials=model['materials'],
                             main=model['rigs']['main'], S01=model['shots']['s01'],
                             S01_blocks=[model['blocks'][i] for i in model['shots']['s01']['control_blocks']],
                             packed=[model['blocks'][c['block']] for c in model['shots']['s01'].get('control_coordinates', [])],
                             main_blocks=[model['blocks'][model['rigs']['main'][k]] for k in ('head_block', 'optics_block')]))
    if frozen != now:
        raise ValueError('Complete S01/scene/material/main physical dependency changed')
    shot_metrics = {}
    for si, sid in enumerate(metadata['shot_ids']):
        mask = source_shots == si
        shot_metrics[sid] = dict(source_rows=int(mask.sum()),
                                 before_error_px=_stats(np.linalg.norm(before[mask]-source_uv[mask], axis=1)),
                                 after_error_px=_stats(np.linalg.norm(after[mask]-source_uv[mask], axis=1)))
    model['fixed_anchor_source_initializer'] = dict(schema='rinkfit.fixed_source_physical_coarse_initializer.v1',
        parent_model_sha256=file_sha256(model_file), source_arrays_sha256=file_sha256(source_arrays_file),
        source_metadata_sha256=file_sha256(source_metadata_file), accepted=False,
        full_packet_source_refinement_required=True, all_original_source_rows_retained=True,
        projection_generated_measurements=0, fixed_entity_station_sliding=False,
        S01_complete_dependency_preserved_sha256=frozen)
    out.mkdir(parents=True)
    write_json(out/'model.json', model)
    np.savez_compressed(out/'source_projection.npz', source_raw_index=arrays['raw_index'][source_rows],
                        source_uv=source_uv, before_uv=before, after_uv=after,
                        native_pts=source_pts, shot_index=source_shots)
    report = dict(schema='rinkfit.fixed_source_physical_coarse_report.v1',
                   model=dict(path=str(out/'model.json'), sha256=file_sha256(out/'model.json')),
                   rigs=reports, shots=shot_metrics, source_fit_accepted=False,
                   engineering_initializer_not_formal_W_objective=True,
                   original_fixed_source_UV_not_shifted=True, full_packet_rows_not_deleted=True,
                   next_step='recapture source-supported finite stations, then consume the complete next-generation packet')
    write_json(out/'report.json', report)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--source-arrays', required=True)
    p.add_argument('--source-metadata', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    print(json.dumps(initialize_from_fixed_anchors(a.model, a.source_arrays, a.source_metadata, a.out)))


if __name__ == '__main__':
    main()
