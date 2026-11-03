"""Keep the compact programme parameters in OneDrive, large evidence in Data."""
from pathlib import Path
import json
import numpy as np
from .rebuild_prepare import load, digest


def deliver(project, output):
    project, output = Path(project), Path(output)
    programme = project/'events/ankara_2026/programmes/okada_mei_sp'
    parameter_path = programme/'body_fullbody_1102.npz'
    binding_path = programme/'body_fullbody_1102.json'
    for path in (parameter_path, binding_path):
        absolute = str(path.resolve())
        if len(absolute.encode('utf-16-le'))//2 >= 260 or len(path.name.encode('utf-16-le'))//2 > 255:
            raise ValueError('OneDrive path exceeds the project length contract')
    read = lambda path: json.loads(Path(path).read_text(encoding='utf8'))
    ledger = read(output/'ledger.json')
    camera = read(output/'camera_binding.json')
    reference = load(output/'reference/result.npz')
    arrays = dict(source_pts=reference['source_pts'], time_base=reference['time_base'],
                  shared_geometry=reference['shared_geometry'],
                  metres_per_scene_unit=reference['metres_per_scene_unit'],
                  body_length_ratio=reference['body_length_ratio'],
                  reference_hashes_json=reference['reference_hashes_json'],
                  camera_binding_json=reference['camera_binding_json'])
    windows = {}
    for mode in ('legacy', 'full'):
        windows[mode] = []
        for index in range(1, 6):
            directory = output/mode/f'w{index}'
            state = load(directory/'result.npz')
            config, report = read(directory/'config.json'), read(directory/'result.json')
            for key in ('x', 'knots', 'active', 'seconds', 'taper'):
                arrays[f'{mode}_w{index}_{key}'] = state[key]
            windows[mode].append(dict(window=index, seconds=state['seconds'].tolist(),
                path=str(directory/'result.npz'), sha256=digest(directory/'result.npz'),
                config=config, report=report))
    np.savez_compressed(parameter_path, **arrays)
    # Verify the compact save, including nonzero rotation controls.
    reloaded = load(parameter_path)
    for key, value in arrays.items():
        np.testing.assert_array_equal(reloaded[key], value)
    evidence = {}
    metric_path = output/'metrics.json'
    metrics_current = False
    if metric_path.is_file():
        metrics = read(metric_path)
        metrics_current = all(metrics['sources']['exports'][mode]['sha256'] == digest(output/mode/'result.npz')
                              for mode in ('reference', 'legacy', 'full'))
    for name in ('metrics.json', 'results.txt', 'state_checks.json', 'derivative_checks.json',
                 'visual_review.json', 'validation_summary.json', 'player_check/checks.json'):
        path = output/name
        if name == 'player_check/checks.json' and path.is_file():
            check = read(path)
            if not check.get('passed') or any(check.get('source_sha256',{}).get(str((output/mode/'result.npz').resolve())) != digest(output/mode/'result.npz')
                    for mode in ('reference', 'legacy', 'full')):
                continue
        if path.is_file() and (name not in ('metrics.json', 'results.txt') or metrics_current):
            evidence[name] = dict(path=str(path), sha256=digest(path))
    candidates = {}
    for mode in ('reference', 'legacy', 'full'):
        path = output/mode/'result.npz'
        candidates[mode] = dict(path=str(path), sha256=digest(path),
                                export=read(path.parent/'export.json'))
    record = dict(schema='rinkspace.programme-body-short-windows.v1',
        event='ankara_2026', programme='okada_mei_sp', shot='s01',
        status='runnable development candidate; not user accepted or independent 3D validation',
        source_time_base=[1,12800], available_source_seconds=[22.12,44.96],
        optimized_source_seconds=[w['seconds'] for w in windows['full']],
        optimized_source_frames=445, unchanged_reference_frames=698,
        complete_programme_body_tracking=False,
        camera=camera, scene=dict(path=str(output/'review_scene.npz'), sha256=digest(output/'review_scene.npz')),
        metric_scale='conditional uniform native scene scale, not independently measured',
        compact_parameters=dict(path=str(parameter_path), sha256=digest(parameter_path)),
        state_contract=dict(global_root='Q_ref @ Exp(root_delta)', pelvis='P_ref + delta_P',
            local_rotations='21 local matrices R_ref @ Exp(delta_local); body root identity',
            reference='original pose @ B9 @ leg correction, folded exactly once',
            continuous_time='cubic local controls, C2 quintic attachment to the unchanged reference',
            ownership='one state drives full mesh, skeleton, shoes and blades; common midpoint separate from row-time projection',
            contact='unknown; no ice snapping or inferred physical left/right blade identity'),
        observation_ledger=dict(path=str(output/'ledger.json'), sha256=digest(output/'ledger.json'),
            observations=str(output/'observations.npz'), observations_sha256=digest(output/'observations.npz'),
            changes=ledger['changes'], changed_points=ledger['changed_points'], excluded_points=ledger.get('excluded_points',0),
            decisions=read(output/'source_review/decisions.json'),
            previous_revision=str(output/'history/observations_v1')),
        reference_hashes=json.loads(str(reference['reference_hashes_json'])),
        windows=windows, candidates=candidates, evidence=evidence,
        acceptance_assessment=metrics.get('acceptance_assessment') if metrics_current else None,
        viewer_config=str(programme/'viewer3d_fullbody_1102.json'),
        replay_entry=str(project/'scripts/run_fullbody_round.py'),
        limitations=['Five short development windows only; outside windows replay J1_s2.',
                     'Pixel error improvements do not establish depth, physical contact or 3D accuracy.',
                     'Assistant source review is not human acceptance; uncertainty and failures are preserved.',
                     'Stopping reasons and nonconverged gradients are retained per window.'])
    binding_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf8')
    return dict(binding=str(binding_path), compact_parameters=str(parameter_path), evidence=evidence)
