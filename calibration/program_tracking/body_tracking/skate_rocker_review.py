"""Conditional 7/8 ft rocker comparison on one frozen body/shared state."""
from copy import copy
from dataclasses import asdict, replace
from pathlib import Path
import json

import numpy as np

from .boot_integration import camera_centres
from .fullbody_fit import footwear_rows
from .rebuild_exposure_body import moving_projection
from .rebuild_prepare import digest
from .skate_review import (CORE, _array_identity, _identical, _jsonable,
                           _max_error, _physics, _pixels)


def _rocker_reference(ref, radius_m):
    # SkateReference.__getattr__ makes copy.copy(ref) unsafe during unpickling.
    trial = object.__new__(type(ref));trial.__dict__.update(ref.__dict__)
    binding = copy(ref.binding)
    binding.base_parameters = replace(ref.binding.base_parameters, main_rocker_m=radius_m)
    binding.main_rocker_ft = radius_m/.3048
    binding.x = ref.binding.x.copy()
    binding._parameter_key = None;binding._packet = None
    trial.binding = binding;binding.reference = trial
    return trial


def compare_rockers(root, out, ref, camera, info, packet, evidence):
    """Change only the factory rocker candidate, without fitting or selection.

    Both candidates share every source exposure, body state, camera state,
    17 fitted shared controls, anatomical registration and upper/cavity shape.
    New image geometry is reevaluated at the five existing exposure nodes.
    """
    root, out = Path(root), Path(out);out.mkdir(parents=True, exist_ok=True)
    pts = np.asarray(packet['pts']);times = np.asarray(ref.times);layout = ref.binding.layout
    if not _identical(pts, ref.p['pts']) or not np.array_equal(packet['time_base'], [1, 12800]):
        raise ValueError('Rocker comparison requires the same saved source clock')
    if not np.array_equal(pts, np.arange(283136, 575489, 256)):
        raise ValueError('Rocker comparison requires all 1143 S01 exposures')
    if np.asarray(ref.binding.x).shape != (17,):raise ValueError('Expected the same 17 shared controls')
    original = ref.binding
    saved = dict(x=original.x.copy(), R0=original.R0.copy(), t0=original.t0.copy(),
                 base=asdict(original.base_parameters), key=original._parameter_key, cache=original._packet)
    trials = {name: _rocker_reference(ref, radius) for name, radius in (('7ft', 2.1336), ('8ft', 2.4384))}
    packets = {name: dict(packet) for name in trials};arrays = {name: dict(gear=[], uv=[]) for name in trials}
    ratio = float(info['length_export_ratio']);shutter = float(camera.p['rig'][37])
    body_errors = {key: 0. for key in ('P_world_mid', 'Q_world_mid', 'local_R_mid')}
    projection_errors = {name: 0. for name in trials}
    # The reference used for body evaluation is itself a clone; its template
    # cache may change without touching the caller's binding or reference.
    body_ref = trials['7ft']
    for first in range(0, len(pts), 32):
        frames = np.arange(first, min(first+32, len(pts)));t = times[frames]
        mid = body_ref.world(body_ref.at(t))
        node_t = (t[:, None]+np.linspace(-.5, .5, 5)*shutter).ravel()
        nodes = body_ref.world(body_ref.at(node_t))
        local_mid = np.einsum('nji,nkj->nki', mid['Q'], mid['native']-mid['P'][:, None])
        local_nodes = np.einsum('nji,nkj->nki', nodes['Q'], nodes['native']-nodes['P'][:, None])
        for target, key, scale in (('P_world_mid', 'P', ratio), ('Q_world_mid', 'Q', 1.), ('local_R_mid', 'local_R', 1.)):
            error = _max_error(packet[target][frames], mid[key]*scale)
            if error is None:raise ValueError('Missing/nonfinite shared body state: '+target)
            body_errors[target] = max(body_errors[target], error)
        for name, trial in trials.items():
            binding = trial.binding
            gm = binding.evaluate(mid['P'], mid['Q'], local_mid, mid['frames'])['gear']
            gn = binding.evaluate(nodes['P'], nodes['Q'], local_nodes, nodes['frames'])['gear']
            world = np.concatenate((gn.reshape(-1, layout.equipment_count, 3), nodes['coco']), axis=1)
            uv = moving_projection(camera, world.reshape(len(frames), 5, layout.point_count, 3), frames)['uv']
            arrays[name]['gear'].append(gm*ratio)
            arrays[name]['uv'].append(uv[:, :layout.equipment_count].reshape(len(frames), len(layout.sides), layout.per_foot, 2))
            error = _max_error(packet['COCO_uv'][frames], uv[:, layout.body_landmarks])
            if error is None:raise ValueError('Nonfinite shared body projection')
            projection_errors[name] = max(projection_errors[name], error)
    if max(body_errors.values()) > 1e-9:
        raise ValueError('Supplied reference and exported body packet describe different states')
    rows, source = footwear_rows(root, np.arange(len(pts)), pts)
    if len(rows.blade_rows) != 7:raise ValueError('The fixed seven-curve source batch changed')
    centres = camera_centres(camera, np.arange(len(pts)))*ratio
    comparisons = {};core = (pts >= round(CORE[0]*12800)) & (pts <= round(CORE[1]*12800))
    for name, trial in trials.items():
        data = packets[name]
        data['gear_world_mid'] = np.concatenate(arrays[name]['gear'])
        data['equipment_uv'] = np.concatenate(arrays[name]['uv'])
        physical, report = _physics(trial, data, pts, evidence)
        reviewed = [r for r in report['source_reviewed_support'] if CORE[0] <= r.get('source_seconds', -np.inf) <= CORE[1]]
        heights = [r['eligible_minimum_height_m'] for r in reviewed if 'eligible_minimum_height_m' in r]
        comparisons[name] = dict(main_rocker_m=trial.binding.base_parameters.main_rocker_m,
            image_comparison=_pixels(data, rows, layout, trial.binding.adapter_faces, centres),
            physical_source_50Hz=report,
            core_35_20_to_36_80=dict(source_exposures=int(core.sum()),
                max_nonpenetration_violation_m=float(np.maximum(-physical['clearance'][core], 0.).max()),
                reviewed_support_exposures=len(reviewed),
                max_support_band_violation_m=max((r.get('support_band_violation_m', np.inf) for r in reviewed), default=None),
                minimum_reviewed_support_height_m=min(heights, default=None),
                maximum_reviewed_support_height_m=max(heights, default=None),
                max_absolute_reviewed_support_gap_m=max(map(abs, heights), default=None)),
            body_P_Q_local_R_bytes_unchanged=all(_identical(data[k], packet[k]) for k in body_errors),
            same_body_projection_max_error_px=projection_errors[name])
    first, second = (trials[name].binding for name in ('7ft', '8ft'))
    a, b = first.packet(), second.packet()
    invariant = dict(same_shared_17_controls=_identical(first.x, second.x),
        same_anatomical_R0=_identical(first.R0, second.R0), same_anatomical_t0=_identical(first.t0, second.t0),
        same_upper=_identical(a['upper'], b['upper']), same_inner_cavity=_identical(a['inner'], b['inner']),
        same_shell_adapter=_identical(a['adapter'][:layout.shell_count], b['adapter'][:layout.shell_count]),
        same_upper_faces=_identical(first.faces, second.faces))
    unchanged = (ref.binding is original and _identical(original.x, saved['x']) and
        _identical(original.R0, saved['R0']) and _identical(original.t0, saved['t0']) and
        asdict(original.base_parameters) == saved['base'] and original._parameter_key == saved['key'] and
        original._packet is saved['cache'])
    if not unchanged or not all(invariant.values()):raise ValueError('Rocker review changed a frozen body/binding/upper field')
    result = _jsonable(dict(schema='rinkspace.same-state-rocker-comparison.v1',
        interpretation='Conditional factory-radius candidates only; no refit, model selection, true-model identification or user acceptance.',
        user_accepted=False, selected_candidate=None, source_time_base=[1, 12800],
        source_exposures=len(pts), source_seconds=[pts[0]/12800., pts[-1]/12800.],
        observation_sources=source, common_denominator=dict(blade_curves=len(rows.blade_rows), blade_samples_per_curve=24,
            heel_toe_points=2*len(rows.landmark_rows), contour_source_samples=sum(len(r['observed']) for r in rows.visible_rows)),
        frozen_state=dict(shared_controls=_array_identity(original.x), body_replay_max_errors=body_errors,
            camera=info, invariants=invariant, caller_reference_and_binding_unchanged=unchanged),
        comparisons=comparisons, implementation_sha256=digest(__file__)))
    (out/'rocker_comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    return result
