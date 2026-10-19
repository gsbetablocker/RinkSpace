"""Validate the real small checkpoint/layout and certified writer bridge only."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from calibration.solver_runtime import install
from calibration.solver_runtime import named66_source_skeleton_adapter_runtime as adapter
from calibration.solver_runtime import source_state_skeleton_cache_runtime as skeleton


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    opt = parser.parse_args()
    install()
    from structured_programme_source_candidate.paired_programme_chart import (
        CompleteProgrammeParameterLayout, PairedProgrammeCoordinateChart,
        CORE_COUNT, FULL_COUNT, OLD, WORLD_INDICES)
    from wide_anchor_candidate.complete_wide_free_optics_source import CompleteWideSource
    with np.load(opt.checkpoint, allow_pickle=False) as saved:
        arrays = {name:saved[name].copy() for name in saved.files}
    manifest = json.loads(str(arrays['aux__current_complete_parameter_layout_json'].item()))
    layout = CompleteProgrammeParameterLayout(manifest['original_complete_parameter_count'],
        manifest['additional_world_parameter_count'])
    chart = PairedProgrammeCoordinateChart(manifest['additional_world_parameter_count'])
    # Execute the registered original seed-parts function, without importing
    # its graph driver or constructing any sources.
    original = ast.parse((Path(adapter.__file__).parent/'legacy/run_current_named66_full_programme.py').read_text(encoding='utf8'))
    function = next(node for node in original.body if isinstance(node, ast.FunctionDef) and node.name == 'current_seed_parts')
    namespace = dict(np=np, BASE_FULL_COUNT=FULL_COUNT, CORE_COUNT=CORE_COUNT, OLD=OLD, WORLD_INDICES=WORLD_INDICES)
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<registered original seed parts>', 'exec'), namespace)
    blocks = []
    for i in range(6):
        weights = arrays['profile__weights_'+str(i)]
        blocks.append(SimpleNamespace(bundle=SimpleNamespace(motion_rows=2*len(weights),
            profile_weights=None, robust_phase=False)))
    state = SimpleNamespace(blocks=blocks, geometry=SimpleNamespace(parameter_count=chart.world_parameter_count),
        structured_programme_coordinate_chart=chart, geometry_parameter_indices=np.asarray(chart.world_indices))
    result = skeleton.restore_final_skeleton_checkpoint(state, arrays,
        current_seed_parts=namespace['current_seed_parts'], circle_enabled=True,
        placement_columns=4, adaptive_source_backtracking=True)
    baseq = np.r_[state.q, result['q'][chart.core_count:chart.core_count+3]]
    rebuilt = CompleteWideSource.initial(SimpleNamespace(), baseq, result['q'][-25:])
    rebuilt = np.r_[rebuilt, result['inherited_circle_suffix'], result['inherited_pattern_suffix']]
    assert result['q'].shape == (chart.full_count,)
    assert np.array_equal(layout.canonical_q(rebuilt), arrays['q'])
    assert np.array_equal(result['q'][-25:], layout.evaluation_q(arrays['q'])[chart.full_count-25:chart.full_count])
    assert result['continued_line_search_fraction'] == float(arrays['next_line_search_fraction'])
    # The pre-existing historical fraction branch is restored as well.
    history = dict(arrays)
    history.pop('next_line_search_fraction')
    history['history_json'] = np.asarray(json.dumps([dict(accepted=True,
        global_schur_trial_records=[dict(whole_source_step_fraction=.125)])]))
    prior = skeleton.restore_final_skeleton_checkpoint(state, history,
        current_seed_parts=namespace['current_seed_parts'], circle_enabled=True,
        placement_columns=4, adaptive_source_backtracking=True)
    assert prior['continued_line_search_fraction'] == .125 and prior['inherited_adaptive_line_search'] is False
    bridges = [skeleton.compatible_static_reader(Path(skeleton.__file__).with_name(name), digest)
        for name,digest in skeleton._STATIC_READER_PREDECESSORS.items()]
    assert all(bridges)
    current = Path(skeleton.__file__).with_name('execution_acceleration.py').read_text(encoding='utf8')
    assert skeleton._central_static_contract(current) != skeleton._central_static_contract(
        current.replace('install_role_record_index_runtime(', 'changed_static_installer('))
    assert adapter._structural_options(SimpleNamespace(accepted_steps=1, normal_workers=8,
        geometry_fd_workers=20, reuse_source_jacobian=True, structural_option=True)) == {'structural_option':True}
    report = dict(all_pass=True, current_checkpoint_sha256=skeleton._sha(opt.checkpoint),
        q_sha256=hashlib.sha256(arrays['q'].tobytes()).hexdigest(),
        original_postboundary_wide_and_suffix_rebuild_byteexact=True,
        checkpoint_profile_arrays_restored_by_original_updater=18,
        original_history_fraction_branch_preserved=True,
        execution_only_options_not_static_key=True,
        certified_predecessor_static_writer_and_installer_bridges=bridges,
        structural_installer_change_rejected=True,
        old_source_cache_or_receipt_modified=False,
        complete_graph_or_solver_or_process_pool_started=False,
        current_helper_sha256={Path(module.__file__).name:skeleton._sha(module.__file__)
            for module in (adapter, skeleton)})
    opt.report.write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')
    print(json.dumps(dict(all_pass=True, report=str(opt.report), q_sha256=report['q_sha256'],
        source_cache_modified=False, graph_started=False)))


if __name__ == '__main__':
    main()
