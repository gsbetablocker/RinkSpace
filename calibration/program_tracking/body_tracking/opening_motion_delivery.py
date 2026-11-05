"""Deliver/replay the dynamic opening without changing the fitted objective."""
from pathlib import Path
import argparse
import json

import numpy as np

from rinkspace.path_policy import guard_project_write
from rinkspace.viewer3d.data import ResultCatalog
from rinkspace.viewer3d.prepare import prepare_variant
from .opening_motion import build, MotionWindow, MotionProblem, START, END
from .rebuild_prepare import load, digest
from .fullbody_round import write_json
from .opening_motion_review import jsonable


def hand_diagnostics(reference, problem, old, new, length_ratio):
    selected = (new['pts'] >= round(START*12800)) & (new['pts'] <= round(END*12800))
    pts = new['pts'][selected]
    ids = np.r_[problem.waist_ids, problem.palm_ids.ravel()]
    output = dict(schema='rinkspace.right-hand-waist-model-diagnostic.v1', source_pts=pts,
        source_time_base=[1, 12800], palm_vertex_ids=problem.palm_ids, waist_vertex_ids=problem.waist_ids,
        definition='Three averaged palm skin patches to nearest sampled right waist surface; model distances, not measured contact truth',
        held_seconds=[22.12, 23.10], relation_released_seconds=[23.10, 23.40], candidates={})
    for name, packet in [('old_opening', old), ('new_motion', new)]:
        _, joints, _, frames = reference.body.forward_rotations(packet['local_R_mid'][selected], with_frames=True)
        vertices = reference.binding.surface.sample_vertices(joints, frames, vertex_ids=ids, double_precision=True)*length_ratio
        waist = vertices[:, :len(problem.waist_ids)]
        palms = vertices[:, len(problem.waist_ids):].reshape(len(pts), 3, 8, 3).mean(axis=2)
        distance = np.linalg.norm(palms[:, :, None]-waist[:, None], axis=-1).min(axis=2)
        held = pts <= round(23.10*12800)
        output['candidates'][name] = dict(per_palm_nearest_waist_distance_m=distance,
            held_median_m=float(np.median(distance[held])), held_p95_m=float(np.percentile(distance[held], 95)),
            held_max_m=float(distance[held].max()), first_palm_distances_m=distance[0])
    return output


def run(project, root, output, *, builder=build, problem_type=MotionProblem,
        stem='opening_motion_1103', variant_id='motion1103',
        label='新候选：扶腰、下沉起身与动态转身', window_type=MotionWindow):
    project, root, output = map(Path, (project, root, output))
    contract = json.loads((output/'inputs.json').read_text(encoding='utf8'))
    fit = load(output/'fit/result.npz')
    result = load(output/'fitted/result.npz')
    report = json.loads((output/'motion_report.json').read_text(encoding='utf8'))
    reference, camera, info, _ = builder(project, root, output)
    # New descriptive camera metadata must not masquerade as a changed frozen
    # numerical binding. Every field captured by the fit remains exact.
    if any(key not in info or info[key] != value for key, value in contract['camera'].items()):
        raise ValueError('Frozen camera identity changed')
    if digest(root/'boot_integration_1102/fitted/result.npz') != contract['parent_sha256']:
        raise ValueError('Boot parent changed')
    if digest(root/'opening_repair_1102/fit/result.npz') != contract['translation_seed_sha256']:
        raise ValueError('Old translation seed changed')
    contact = problem_type(reference, camera, output/'observations.npz')
    programme = project/'events/ankara_2026/programmes/okada_mei_sp'
    parameter_path = guard_project_write(programme/(stem+'.npz'))
    record_path = guard_project_write(programme/(stem+'.json'))
    compact = dict(reference.packet, **{k: fit[k] for k in ('x', 'knots', 'active', 'seconds', 'taper')})
    compact.update(preparation_local_R=reference.preparation['local_R'],
                   preparation_P=reference.preparation['P'], preparation_Q=reference.preparation['Q'],
                   waist_vertex_ids=contact.waist_ids, palm_vertex_ids=contact.palm_ids,
                   input_contract_json=np.array(json.dumps(contract, sort_keys=True)))
    if parameter_path.is_file():
        previous = load(parameter_path)
        if previous.keys() != compact.keys() or any(not np.array_equal(previous[k], v) for k, v in compact.items()):
            raise ValueError('Existing compact motion parameters differ; preserve the candidate')
    else:
        np.savez_compressed(parameter_path, **compact)
    # Reconstruct from the OneDrive motion samples and controls, not the result
    # packet's evaluated poses. The frozen parent and old translation seed are
    # declared dependencies, with their current identities checked above.
    saved = load(parameter_path)
    reference.packet = {k: saved[k] for k in reference.packet}
    reference.preparation = dict(local_R=saved['preparation_local_R'],
                                 P=saved['preparation_P'], Q=saved['preparation_Q'])
    from .opening_motion import interpolate_motion
    reference.initial_raw_Q, first = interpolate_motion(reference.packet, [START])
    reference.initial_raw_local = first[0]
    state = window_type(reference, (START, END), 'full')
    np.testing.assert_array_equal(state.knots, saved['knots'])
    np.testing.assert_array_equal(state.active, saved['active'])
    reference.refinements.append((state, saved['x']))
    mapping = [('P', 'P_world_mid', info['length_export_ratio']), ('Q', 'Q_world', 1.),
        ('local_R', 'local_R_mid', 1.), ('native', 'native22_world_mid', info['length_export_ratio']),
        ('gear', 'gear_world_mid', info['length_export_ratio'])]
    errors = {key: 0. for key, _, _ in mapping}
    for start in range(0, len(reference.times), 32):
        stop = min(start+32, len(reference.times))
        current = reference.world(reference.at(reference.times[start:stop]))
        for key, target, ratio in mapping:
            error = float(np.max(abs(current[key]*ratio-result[target][start:stop])))
            errors[key] = max(errors[key], error)
    obs, parent_obs = load(output/'observations.npz'), load(root/'boot_integration_1102/observations.npz')
    # Missing observations contain NaNs. Compare their exact saved array bytes
    # as well, rather than letting NaN != NaN reject an unchanged input.
    observations_equal = obs.keys() == parent_obs.keys() and all(
        obs[k].shape == parent_obs[k].shape and obs[k].dtype == parent_obs[k].dtype
        and obs[k].tobytes() == parent_obs[k].tobytes() for k in obs)
    inferred = np.array_equal(result['source_lower_body_inferred'], reference.p['lower_body_inferred'])
    checks = dict(schema='rinkspace.opening-motion-replay.v1',
        one_drive_parameter_replay_max_absolute=errors,
        source_pts_preserved=bool(np.array_equal(result['pts'], reference.p['pts'])),
        observations_unchanged=bool(observations_equal), lower_body_inference_preserved=bool(inferred),
        after_25_exact=report['after_25_boot_parent']['all_exact_equal'],
        camera_unchanged=True, parent_unchanged=True,
        all_float_arrays_finite=all(np.isfinite(v).all() for v in result.values() if v.dtype.kind in 'fc'),
        derivative_checks=json.loads((output/'derivative_checks.json').read_text(encoding='utf8')))
    checks['passed'] = (max(errors.values()) <= 2e-12 and
        all(checks[k] for k in ('source_pts_preserved', 'observations_unchanged',
            'lower_body_inference_preserved', 'after_25_exact', 'all_float_arrays_finite')))
    write_json(output/'verification.json', checks)
    if not checks['passed']:
        raise ValueError('Motion replay/invariants failed; viewer binding not changed')
    if variant_id == 'hand1103':
        dependency = contract['base_timing_parameters']
        if digest(dependency['path']) != dependency['sha256']:
            raise ValueError('Frozen OneDrive body timing parameters changed')
        base = load(root/'opening_timing_1103/fitted/result.npz')
        other = np.setdiff1d(np.arange(21), np.array(contract['hand_polish_native_joints'])-1)
        after_hand = result['pts'] >= round(contract['hand_polish_seconds'][1]*12800)
        scope = dict(schema='rinkspace.opening-hand-scope.v1',
            P_exact=np.array_equal(base['P_world_mid'], result['P_world_mid']),
            Q_exact=np.array_equal(base['Q_world'], result['Q_world']),
            gear_exact=np.array_equal(base['gear_world_mid'], result['gear_world_mid']),
            other_local_R_exact=np.array_equal(base['local_R_mid'][:, other], result['local_R_mid'][:, other]),
            after_hand_end_exact=all(np.array_equal(base[key][after_hand], result[key][after_hand])
                for key in ('P_world_mid', 'Q_world', 'local_R_mid', 'native22_world_mid', 'gear_world_mid')))
        scope['passed'] = all(scope[k] for k in ('P_exact', 'Q_exact', 'gear_exact',
            'other_local_R_exact', 'after_hand_end_exact'))
        write_json(output/'arm_scope_verification.json', scope)
        if not scope['passed']:
            raise ValueError('Arm polish changed the frozen timing/body state; viewer binding not changed')
    hand = hand_diagnostics(reference, contact, load(root/'opening_repair_1102/fitted/result.npz'), result, info['length_export_ratio'])
    write_json(output/'hand_waist_report.json', jsonable(hand))
    text_path = output/'results.txt'
    text = text_path.read_text(encoding='utf8').split('\n右手掌与腰部表面补充诊断')[0]
    text += '\n右手掌与腰部表面补充诊断（模型几何，不是实测接触真值）：\n'
    for name, row in hand['candidates'].items():
        text += f"{name} 22.12–23.10s 三掌部patch最近腰表面距离median/P95/max：{row['held_median_m']:.6g}/{row['held_p95_m']:.6g}/{row['held_max_m']:.6g} 条件m\n"
    text_path.write_text(text, encoding='utf8')
    config_path = programme/'viewer3d_fullbody_1102.json'
    config = json.loads(config_path.read_text(encoding='utf8'))
    variant = dict(id=variant_id, label=label, kind='fullbody',
        result=str(output.relative_to(root)/'fitted/result.npz').replace('\\', '/'),
        shell_topology='saved', body_reference_label='boot_integration1102',
        body_refinement_intervals_seconds=[[START, END]])
    config['variants'] = [v for v in config['variants'] if v['id'] != variant['id']]+[variant]
    for old in config['variants']:
        if old['id'] == 'opening1102':
            old['label'] = '旧版：固定姿态与近静止条件（失败对照）'
        if variant_id in ('timing1103', 'hand1103') and old['id'] == 'motion1103':
            old['label'] = '失败候选：下探过早与开头1.11米平移'
        if variant_id == 'hand1103' and old['id'] == 'timing1103':
            old['label'] = '对照：时序已修正，右手待精修'
    config.update(default_variant=variant_id, initial_source_pts=283136,
                  comparison_variants=(['opening1102', 'timing1103', variant_id] if variant_id == 'hand1103'
                    else ['opening1102', 'motion1103', variant_id] if variant_id == 'timing1103'
                    else ['opening1102', variant_id]))
    write_json(config_path, config)
    entry = guard_project_write(project / 'launchers' / 'START_3D_COMPARISON.cmd')
    entry.parent.mkdir(parents=True,exist_ok=True)
    entry.write_text('@echo off\ncall "%~dp0START_3D_RESULTS.cmd" -Config "%~dp0..\\events\\ankara_2026\\programmes\\okada_mei_sp\\viewer3d_fullbody_1102.json" -Variant '+variant_id+' -Fresh %*\n', encoding='utf8')
    catalog = ResultCatalog(project=project, data_root=root.parents[1], config=config_path)
    cache = prepare_variant(catalog, catalog.variant(variant_id), reference.binding.surface)
    evidence = {}
    for name in ('inputs.json', 'ledger.json', 'fit/config.json', 'fit/result.json',
            'derivative_checks.json', 'motion_report.json', 'hand_waist_report.json', 'results.txt', 'verification.json',
            'source_motion_comparison.png', 'visual_diagnostics.json', 'motion_curves.png',
            'motion_visual_review.json', 'player_check/checks.json', 'side_check/checks.json',
            'timing_report.json', 'arm_scope_verification.json'):
        path = output/name
        if path.is_file():
            evidence[name] = dict(path=str(path), sha256=digest(path))
    record = dict(schema='rinkspace.opening-motion-binding.v1', event='ankara_2026',
        programme='okada_mei_sp', source_seconds=[START, END], source_time_base=[1, 12800],
        affected_exposures=145, lower_body_inferred_exposures=int(reference.p['lower_body_inferred'].sum()),
        input_contract=contract, camera=info, parameters=dict(path=str(parameter_path), sha256=digest(parameter_path)),
        result=dict(path=str(output/'fitted/result.npz'), sha256=digest(output/'fitted/result.npz')),
        translation_seed='Previous opening trajectory is only the translation reference; no actual zero velocity or fixed-position factor',
        articulation='Raw cached source-mapped joint/root SO(3); inferred preparation legs retain dynamic raw increments and release by24.6s',
        wrist='Dynamic cached wrist plus optimized right wrist; source-confirmed sliding palm/waist relation released23.10-23.40s; fingers not directly measured',
        blade_plane='1mm soft minimum and C1 penetration transition; weak support envelope, no identified physical contact point',
        boundary='C2 attachment at25s; all five body/gear arrays exactly equal to boot parent from25s inclusive',
        evidence=evidence, cache=str(cache), viewer_config=str(config_path), viewer_entry=str(entry),
        replay_verification=checks, accuracy_accepted=False, independent_3d_truth=False,
        implementation={str(path): digest(path) for path in (Path(__file__),
            Path(__file__).with_name('opening_motion.py'), Path(__file__).with_name('opening_motion_review.py'),
            project/'scripts/run_opening_motion.py', project/'scripts/deliver_opening_motion.py')})
    if variant_id in ('timing1103', 'hand1103'):
        record.update(schema='rinkspace.opening-timing-binding.v1',
            quiet_phase_seconds=contract['quiet_phase_seconds'],
            user_motion_correction=contract['user_correction'],
            rejected_dynamic_candidate=str(root/'opening_motion_1103/fitted/result.npz'))
        record['articulation'] = ('Source-confirmed quiet P/Q/trunk/legs until22.90, with freely fitted common opening state; '
            'arms/wrists/head stay dynamic. Body increments after22.90 release into raw motion by24.60.')
        record['blade_plane'] += '; quiet-phase support envelope scale1cm, physical blade station unknown'
        record['quiet_body_time_equality'] = dict(position_scale_m=contract['quiet_position_scale_m'],
            rotation_scale_deg=contract['quiet_rotation_scale_deg'],
            interpretation='Soft time-equality only during source-confirmed quiet phase; no fixed world location or later zero-velocity condition')
        record['implementation'].update({str(path): digest(path) for path in (
            Path(__file__).with_name('opening_timing.py'),
            project/'scripts/run_opening_timing.py', project/'scripts/deliver_opening_timing.py')})
        review_code = Path(__file__).with_name('opening_timing_review.py')
        if review_code.is_file():
            record['implementation'][str(review_code)] = digest(review_code)
    if variant_id == 'hand1103':
        record.update(schema='rinkspace.opening-hand-on-timing-binding.v1',
            body_timing_parameters=contract['base_timing_parameters'],
            hand_polish_seconds=contract['hand_polish_seconds'],
            hand_polish_native_joints=contract['hand_polish_native_joints'],
            arm_scope_verification=scope,
            hand_contact_patch_rest_y_m=contract['hand_contact_patch_rest_y_m'])
        record['wrist'] = ('Right shoulder/elbow/wrist-only polish on frozen source-timed body; '
            'palm can slide over right waist and upper hip, relation released23.10-23.40s; fingers default')
        record['implementation'].update({str(path): digest(path) for path in (
            Path(__file__).with_name('opening_hand_polish.py'),
            project/'scripts/run_opening_hand.py', project/'scripts/deliver_opening_hand.py')})
    write_json(record_path, record)
    return dict(binding=str(record_path), parameters=str(parameter_path), cache=str(cache),
                verification_passed=True, replay_errors=errors)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.project, args.root, args.output), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
