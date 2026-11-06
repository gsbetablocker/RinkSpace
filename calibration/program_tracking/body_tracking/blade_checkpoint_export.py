"""Export one current rejected checkpoint for comparison, without fitting.

The archived checkpoint and CASE identity remain unchanged. The final absolute
curve is restored once; saved x is never reapplied to that final curve.
"""
from pathlib import Path
import argparse
import copy
import io
import json
import os
import time

import numpy as np

from rinkspace.path_policy import guard_project_write
from .blade_current_reference import restore_current_reference
from .blade_projection import SavedBladeProjection
from .blade_boot_refit import _mass, _source_quality_groups
from .body_execution import BodyExecution, BodyExecutionConfig
from .exmr_integration import CASE_FIELD, save_packet
from .rebuild_prepare import digest
from .skate_integration import source_evidence
from .skate_physics import fk_state
from .skate_template import profile_height
from .spin_continuous_refit import ContinuousJoint
from .support_image import SupportImageFactors


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _identity(path):
    path = Path(path)
    return dict(path=str(path), sha256=digest(path), bytes=path.stat().st_size)


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def export(project, stage, checkpoint, output, parameter_path):
    """Export current-case state; preserve input/CASE/parameter identities.

    ContinuousJoint is constructed only to supply its actual body ledger and
    source-factor admission. Neither its evaluate nor solve is called. Source
    region metrics alone are evaluated at zero deltas around the saved final
    curve. No historical builder, prepare, inference or parent replay runs.
    """
    started = time.perf_counter()
    project, stage, checkpoint, output, parameter_path = map(
        lambda p: Path(p).resolve(), (project, stage, checkpoint, output, parameter_path))
    guard_project_write(parameter_path, project_root=project)
    if output.exists() or parameter_path.exists():
        raise FileExistsError('Rejected export never overwrites an output or parameter archive')
    source_case = stage/'common/exmr_hand_nofoot'
    source_summary = _read(stage/'run_summary.json')
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not source_summary.get('stop', {}).get('quality_failed'):
        raise ValueError('This comparison export requires a recorded optimizer quality rejection')
    reference, camera, info, saved = restore_current_reference(project, stage, checkpoint, seed=False)
    if reference.current_restore_info['pose_seed'] != 'saved_final':
        raise ValueError('Expected authoritative saved-final absolute curve')
    original_case_identity = str(saved[CASE_FIELD])
    case = output/'common/exmr_hand_nofoot'
    case.mkdir(parents=True)
    for name in ('input.npz', 'initial.npz', 'compact_smplx.npz', 'observations.npz', 'export_source.npz'):
        os.link(source_case/name, case/name)
    for name in ('grasp_relations.json', 'palm_grip_geometry.json', 'exmr_body_hand_nofoot.npz',
                 'support_and_free_shoe_regions.json', 'source_evidence.json', 'mass_geometry.npz'):
        if (stage/name).is_file():
            os.link(stage/name, output/name)
    # The archive is byte-for-byte the attempt, not a rewritten zero vector or
    # packet created from the retained parent. Its original CASE stays intact.
    with checkpoint.open('rb') as source, parameter_path.open('xb') as target:
        target.write(source.read())
    if checkpoint.read_bytes() != parameter_path.read_bytes():
        raise ValueError('Archived rejected checkpoint bytes changed')
    execution = BodyExecution(BodyExecutionConfig(workers=1, blas_threads=1))
    execution.attach(reference)
    document = reference.source_document
    context = tuple(source_summary['context_seconds'])
    core = tuple(source_summary['core_seconds'])
    accepted = int(source_summary['accepted_steps'])
    try:
        with execution.limits():
            supports = source_evidence(source_case, shot_id=document['shot_id'], source_pts=reference.p['pts'], records=document['supports'])
            problem = ContinuousJoint(reference, camera, reference.observations_path, context, supports,
                                      output/'not_a_solve', io.StringIO(), 'absolute_motion_v1',
                                      optimize_shared=False, ice_policy=reference.ice_policy)
            runtime = copy.deepcopy(document)
            multiplier = float(reference.grasp_relations['factor_parameters']['shoe_region_weight_multiplier'])
            for row in runtime['observations']:
                row['objective_weight'] *= multiplier
            for row in runtime.get('boot_axis_observations', []):
                row['objective_weight'] *= multiplier
            problem.support_image = SupportImageFactors(problem, runtime, arc_ids={})
            review = SupportImageFactors(problem, runtime, include_review=True,
                                         arc_ids=source_summary['source_regions_before'].get('fixed_arc_ids', {}))
            # This is source-factor projection only, not JointContactProblem
            # evaluate. Its zero increments apply to the already final curve.
            source_after = review.evaluate(np.zeros(problem.body.size), jacobian=False)['metrics']
            packet = save_packet(problem, info, supports, document, case, status='joint_fit')
            mid = reference.at(reference.times)
            zero = np.zeros((len(reference.times), 69))
            state = fk_state(reference, mid, zero)
            mass, mass_path = _mass(reference, stage)
            centres, _ = mass.evaluate(mid, zero)
            p = reference.binding.packet()['parameters']
            xs = np.array([-p.runner_rear_extension_m,
                           .5*(-p.runner_rear_extension_m+p.sole_length_m-p.runner_front_inset_m),
                           p.sole_length_m-p.runner_front_inset_m])
            local = np.c_[xs, np.zeros(3), profile_height(xs, p)]
            state_metadata = {key:np.asarray(value).copy() for key, value in saved.items()
                              if key.startswith('absolute_') or key in
                              ('state_schema_version', 'state_kind', 'no_inherited_high_frequency_curve')}
            packet.update(state_metadata)
            provenance = dict(source_stage=str(stage), source_checkpoint=_identity(checkpoint),
                              parameter_archive=_identity(parameter_path), original_case_identity_preserved=True,
                              restore_seed=False, saved_x_reapplied=False, mass_geometry=_identity(mass_path),
                              current_input_hardlinks=True, source_runtime_weight_multiplier=multiplier,
                              ContinuousJoint_evaluate_called=False, solve_called=False,
                              source_factor_zero_increment_projection_only=True,
                              restore_info=reference.current_restore_info)
            packet.update(com_world_mid=mass.com(centres),
                          blade_material_world_mid=(state['O'][:, :, None]+np.einsum('nfij,kj->nfki', state['R'], local))*reference.binding.ratio,
                          absolute_parameter_path=np.array(str(parameter_path)),
                          comparison_stage=np.array('rejected_attempt_replay'),
                          optimizer_quality_rejected=np.array(True), accepted_step=np.array(accepted),
                          accepted_steps=np.array(accepted), user_accepted=np.array(False),
                          accuracy_accepted=np.array(False), no_new_fit=np.array(True),
                          rejected_checkpoint_export_provenance_json=np.array(json.dumps(provenance, sort_keys=True)),
                          rejected_attempt_quality_json=np.array(json.dumps(source_summary['stop'], sort_keys=True)))
            if str(packet[CASE_FIELD]) != original_case_identity:
                raise ValueError('Rejected checkpoint CASE identity changed')
            result_path = case/'fitted/result.npz'
            np.savez_compressed(result_path, **packet)
    finally:
        execution.close()
    # The original-stage query reads the rejected checkpoint directly. The
    # independent new-stage query must reproduce it from its copied archive.
    before = SavedBladeProjection(project, stage, checkpoint)
    after = SavedBladeProjection(project, output, parameter_path)
    observed = np.array(sorted({int(row['pts']) for row in document['observations']
                                if int(row['pts']) in before._frames
                                and core[0] <= row['pts']/12800. <= core[1]}), dtype=np.int64)
    pool = observed if len(observed) else before.pts
    pts = pool[np.unique(np.linspace(0, len(pool)-1, 3).astype(int))]
    direct = before.project_points(pts, 1, np.zeros((1, 3)), include_body=True)
    replay = after.project_points(pts, 1, np.zeros((1, 3)), include_body=True)
    indices = np.array([before._frames[int(value)] for value in pts])
    comparison = dict(pts=pts.tolist(), exported_COCO_uv=packet['COCO_uv'][indices].tolist(),
                      checkpoint_direct_COCO_uv=direct['COCO_uv'].tolist(),
                      new_stage_query_COCO_uv=replay['COCO_uv'].tolist(),
                      export_vs_direct_coco_uv_max_abs_px=float(np.max(abs(packet['COCO_uv'][indices]-direct['COCO_uv']))),
                      archive_vs_direct_coco_uv_max_abs_px=float(np.max(abs(replay['COCO_uv']-direct['COCO_uv']))),
                      archive_vs_direct_R_max_abs=float(np.max(abs(replay['R']-direct['R']))))
    if max(comparison['export_vs_direct_coco_uv_max_abs_px'], comparison['archive_vs_direct_coco_uv_max_abs_px']) > 2e-8:
        raise ValueError('Rejected final-state COCO projection is inconsistent')
    with np.load(source_summary['result'], allow_pickle=False) as retained:
        difference = {name:float(np.max(abs(packet[name]-retained[name]))) for name in
                      ('P_world_mid', 'Q_world_mid', 'local_R_mid', 'native22_world_mid', 'skate_R_world_mid', 'COCO_uv')}
    if not max(difference.values()) > 1e-12:
        raise ValueError('Rejected attempt unexpectedly equals the retained parent')
    metadata = dict(schema='rinkspace.current-rejected-checkpoint-export.v1', **provenance,
                    result=_identity(result_path), state_consistency=comparison,
                    rejected_vs_retained_max_abs=difference, comparison_stage='rejected_attempt_replay',
                    optimizer_quality_rejected=True, accepted_step=accepted, no_new_fit=True,
                    user_accepted=False, accuracy_accepted=False,
                    export_wall_seconds=time.perf_counter()-started,
                    source_original_run_summary=_identity(stage/'run_summary.json'),
                    consistency_is_accuracy_proof=False)
    _write(output/'export_metadata.json', metadata)
    latest = source_summary['stop']['latest']
    summary = dict(parameter=str(parameter_path), result=str(result_path),
                   source_frames=len(reference.times), core_seconds=list(core), context_seconds=list(context),
                   active_control_seconds=source_summary.get('active_control_seconds', list(context)),
                   comparison_stage='rejected_attempt_replay', optimizer_quality_rejected=True,
                   accepted_steps=accepted, accepted_step=accepted, retained_step=None,
                   stop=source_summary['stop'], wall_seconds=metadata['export_wall_seconds'],
                   wall_seconds_semantics='read-only export elapsed; original fitting wall time recorded separately',
                   original_fit_wall_seconds=source_summary['wall_seconds'],
                   original_fit_retained_step=source_summary['retained_step'],
                   source_regions_before=source_summary['source_regions_before'], source_regions_after=source_after,
                   source_quality_groups_before=_source_quality_groups(source_summary['source_regions_before'], core, context),
                   source_quality_groups_after=_source_quality_groups(source_after, core, context),
                   initial_parts=source_summary.get('initial_parts'), final_parts=None,
                   final_parts_status='Not recomputed: ContinuousJoint evaluate was not called',
                   image_cost_before=source_summary['stop']['baseline']['image_cost'], image_cost_after=latest['image_cost'],
                   joint_cost_before=source_summary['stop']['baseline']['joint_cost'], joint_cost_after=latest['joint_cost'],
                   pose_quality_monitor_cost=latest.get('pose_cost'),
                   no_new_fit=True, user_accepted=False, accuracy_accepted=False,
                   numerical_convergence=False, not_default=True, export_metadata=str(output/'export_metadata.json'))
    _write(output/'run_summary.json', summary)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('project', 'stage', 'checkpoint', 'output', 'parameter-path'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    value = export(args.project, args.stage, args.checkpoint, args.output, args.parameter_path)
    print(json.dumps(dict(output=value['result']['path'], consistency=value['state_consistency'],
                          rejected_vs_retained=value['rejected_vs_retained_max_abs']), ensure_ascii=False))


if __name__ == '__main__':
    main()
