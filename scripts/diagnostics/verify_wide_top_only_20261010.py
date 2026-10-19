"""New TOP-only candidate vs saved real wide constraints; no full source graph."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
from calibration.solver_runtime import configure_native_runtime
from calibration.solver_runtime.wide_top_only_candidate import qualified_wide_top_world, install_wide_top_only


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def exact(lhs, rhs):
    if isinstance(lhs, np.ndarray):
        return isinstance(rhs, np.ndarray) and lhs.shape == rhs.shape and lhs.dtype == rhs.dtype and lhs.tobytes() == rhs.tobytes()
    if isinstance(lhs, dict):
        return isinstance(rhs, dict) and lhs.keys() == rhs.keys() and all(exact(lhs[k], rhs[k]) for k in lhs)
    if isinstance(lhs, (list, tuple)):
        return type(lhs) is type(rhs) and len(lhs) == len(rhs) and all(exact(a, b) for a, b in zip(lhs, rhs))
    return type(lhs) is type(rhs) and lhs == rhs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state-directory', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--saved-proof-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    configure_native_runtime(stage='wide_top_only_real_array_check', native_threads=1)
    from rinkspace.vision_camera_initialization import current_complete_source_geometry
    from structured_programme_source_candidate.current_wide_source import CurrentWideSource
    from structured_programme_source_candidate.human_stroke_truth import WideStrokeTruth
    from structured_programme_source_candidate.paired_programme_chart import CompleteProgrammeParameterLayout
    from npz_readonly import mapped_array

    state_path = args.state_directory / 'complete_state_UNACCEPTED.npz'
    options = dict(read(args.config)['source_options'])
    provider, world, provenance = current_complete_source_geometry(state_path, options['--world-init'])
    with np.load(state_path, allow_pickle=False) as saved:
        q = saved['q'].copy()
        layout_packet = json.loads(str(saved['aux__current_complete_parameter_layout_json'].item()))
        camera_indices = saved['aux__current_wide_camera_parameter_indices'].copy()
    layout = CompleteProgrammeParameterLayout(int(layout_packet['original_complete_parameter_count']),
                                              int(layout_packet['additional_world_parameter_count']))
    eq = layout.evaluation_q(q)
    scene = read(args.state_directory / 'scene_UNACCEPTED.json')
    cameras = [row['camera25'] for row in scene['frame_cameras'] if row.get('source_pts') == 56218880]
    if len(cameras) != 1:
        raise ValueError('Actual independent wide camera archive required')
    authority = read(options['--original-wide-human-truth-authority'])
    trace = read(authority['trace_packet_reference']['path'])
    factor = CurrentWideSource.__new__(CurrentWideSource)
    factor.geometry = provider.current
    factor.qualified_scene = provider.qualified_scene
    factor.geometry_parameter_indices = layout.canonical_to_evaluation[
        np.asarray(provenance['actual_world_parameter_indices'], int)]
    factor.observations = []
    factor.ad_packet = dict(rows=[])
    factor.template_camera25 = np.asarray(cameras[0], float)
    factor.trace_packet = trace
    factor.trace_uv = [np.asarray(row['observed_centres_source_px'], float) for row in trace['contours']]
    truth = WideStrokeTruth(factor, options['--original-wide-human-truth-authority'], len(q),
                            layout.canonical_to_evaluation[camera_indices])
    truth.bind_practical_location_envelope(options['--practical-human-location-envelope'])
    original_queries = truth._queries_for_records
    proof = dict(schema='rinkspace.actual-wide-top-only-candidate.v1', complete_q_sha256=hashlib.sha256(q.tobytes()).hexdigest(),
        state=dict(path=str(state_path), sha256=sha(state_path)), config=dict(path=str(args.config), sha256=sha(args.config)),
        helper=dict(path=str(ROOT / 'calibration/solver_runtime/wide_top_only_candidate.py'),
                    sha256=sha(ROOT / 'calibration/solver_runtime/wide_top_only_candidate.py')),
        source_geometry_provenance=provenance, full_programme_motion_graph_loaded=False,
        normal_or_source_J_loaded=False, previous_world_cache_benchmark_repeated=False,
        old_complete_constraint_build_repeated=False, checks=[])
    samples = [('nominal', eq.copy())]
    for logical in (44, 45, 49):
        column = int(truth.world_indices[logical])
        for sign in (-1, 1):
            value = eq.copy(); value[column] += sign * 1e-6
            samples.append((f'world_{logical}_{sign}', value))
    for sample_name, value in samples:
        at = value[truth.world_indices]
        begin = time.perf_counter(); old_top = factor._curve_worlds(at)[1]; old_seconds = time.perf_counter() - begin
        begin = time.perf_counter(); new_top = qualified_wide_top_world(factor, at); new_seconds = time.perf_counter() - begin
        if not exact(old_top, new_top):
            raise AssertionError('Actual original TOP arrays differ: ' + sample_name)
        old_queries = original_queries(value, truth.records)
        from calibration.solver_runtime.wide_top_only_candidate import _queries_for_records_top_only
        new_queries = _queries_for_records_top_only(truth, value, truth.records)
        if not exact(old_queries, new_queries):
            raise AssertionError('Actual original public query fields differ: ' + sample_name)
        proof['checks'].append(dict(sample=sample_name, top_shape=list(new_top.shape),
            top_float64_bytes_equal=True, all_public_query_fields_equal=True,
            original_top_only_seconds=old_seconds, candidate_top_only_seconds=new_seconds))
    proof['candidate_installation'] = install_wide_top_only(truth)
    fixture = args.saved_proof_directory / 'actual_q4511_all82_original_priority_LP_inputs.npz'
    with np.load(fixture, allow_pickle=False) as saved:
        if not exact(saved['q'], q):
            raise ValueError('Saved original wide matrix has a different complete state')
        selected = saved['selected_evaluation'].copy()
    limits = read(args.state_directory / 'truth_restoration_state.json')['families']['wide']['limits']
    begin = time.perf_counter()
    candidate = truth.constraints(eq, selected, group_limits=np.asarray(limits), return_step_carrier_response=True)
    seconds = time.perf_counter() - begin
    receipt = read(args.saved_proof_directory / 'actual_q4511_truth_constraints_timing_receipt.json')
    sl = receipt['family_row_slices']['wide']
    matrix = mapped_array(fixture, 'constraint_matrix')[sl['start']:sl['stop']]
    with np.load(args.saved_proof_directory / 'old_truth_family_lower_and_carrier_responses.npz', allow_pickle=False) as saved:
        pairs = dict(matrix=(matrix, candidate['matrix']), lower=(saved['wide_lower'], candidate['lower']),
            nominal_carrier=(saved['wide_nominal_carrier'], candidate['step_carrier_response']['nominal_carrier']),
            retained_face_mask=(saved['wide_retained_face_mask'], candidate['step_carrier_response']['retained_face_mask']))
        proof['saved_original_comparison'] = {name: dict(shape=list(rhs.shape), byte_exact=exact(lhs, rhs))
                                            for name, (lhs, rhs) in pairs.items()}
    if not all(row['byte_exact'] for row in proof['saved_original_comparison'].values()):
        raise AssertionError('Original full-column wide constraint arrays differ')
    old_report = read(args.saved_proof_directory / 'old_wide_constraints_report.json')['report']
    if not exact(old_report, candidate['report']):
        raise AssertionError('Original wide source constraint report differs')
    proof.update(passed=True, candidate_complete_wide_constraint_seconds=seconds,
        original_constraint_report_equal=True, source_pixels_support_and_FD_stencils_unchanged=True,
        full_iteration_speedup_claim=False, accuracy_accepted=False)
    path = args.output / 'actual_q4511_wide_top_only_saved_original_PASS.json'
    path.write_text(json.dumps(proof, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(dict(passed=True, proof=str(path), candidate_wide_constraint_seconds=seconds,
                         samples=len(samples), rows=len(candidate['lower']))), flush=True)


if __name__ == '__main__':
    main()
