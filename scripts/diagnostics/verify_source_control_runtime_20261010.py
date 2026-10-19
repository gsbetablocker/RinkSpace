"""Exercise actual small IPM and complete source-loop control/save paths."""
from pathlib import Path
import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
import hashlib
import io
import json
import os
import sys
from types import SimpleNamespace

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import clarabel
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, diags
from calibration.solver_runtime import install
from calibration.solver_runtime.optional_soc_runtime import install_optional_SOC_numerical_failure_route
from calibration.solver_runtime.solver_control_runtime import (
    SourceControlStop, install_solver_control_runtime, source_control_context)
from calibration.solver_runtime.trial_precheck_runtime import with_trial_precheck


def digest(value):
    return hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast("B")).hexdigest()


def write_control(output, cache):
    payload = dict(stop_branch=True, reason="explicit diagnostic control inside IPM",
                   save_current_linearization_cache=cache)
    temporary = output / "control.json.tmp"
    temporary.write_text(json.dumps(payload), encoding="utf8")
    temporary.replace(output / "control.json")


def instrument_poll(context, armed, cache):
    original_poll = context.poll
    events = []

    def poll(info=None):
        if info is not None:
            events.append(int(info.iterations))
            if armed[0] and info.iterations >= 1:
                # This is solely a deterministic test trigger. Production
                # always responds to the user's explicit control file.
                write_control(context.output, cache)
                armed[0] = False
        return original_poll(info)

    context.poll = poll
    return events


class Profiles:
    def __init__(self):
        self.weights = np.array([[0.4, 0.6]])
        self.calls = 0

    def __call__(self, q):
        self.calls += 1
        self.weights = np.array([[0.4 + 0.1 * q[0], 0.6 + 0.1 * q[1]]])

    def iter_checkpoint_arrays(self):
        yield "weights_0", self.weights
        yield "present_0", np.array(True)
        yield "robust_0", np.array(True)

    def checkpoint_arrays(self):
        return {key: value.copy() for key, value in self.iter_checkpoint_arrays()}

    def restore_checkpoint(self, arrays):
        self.weights = arrays["weights_0"].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix("").exists():
        raise FileExistsError(args.output)
    cases = args.output.with_suffix("")
    cases.mkdir(parents=True)
    install()
    import z190_global_schur_linear_step as qp
    import z190_joint_source_solver as joint
    from z190_streaming_source_normal import SourceRowBlock
    raw_constructor = clarabel.DefaultSolver
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_threads = 1
    P, g = csc_matrix(np.eye(2)), np.array([-2., 0.])
    A, b = csc_matrix([[1., 0.], [-1., 0.]]), np.array([1., 0.])

    def raw_solution(constructor):
        return constructor(P, g, A, b, [clarabel.NonnegativeConeT(2)], settings).solve()

    raw = raw_solution(raw_constructor)
    installation = install_solver_control_runtime()
    route = install_optional_SOC_numerical_failure_route()
    log = io.StringIO()
    checks = []
    no_stop = cases / "no_stop"
    no_stop.mkdir()
    with source_control_context(no_stop, reader=lambda output: None):
        controlled = raw_solution(clarabel.DefaultSolver)
    for field in ("x", "z", "s"):
        assert digest(np.asarray(getattr(raw, field))) == digest(np.asarray(getattr(controlled, field)))
    assert raw.obj_val == controlled.obj_val and raw.status == controlled.status
    checks.append(dict(case="actual_IPM_without_stop", solution_x_z_s_bitexact=True,
                       objective_bitexact=True, status=str(raw.status)))

    # A real cancellation inside a strict optional SOC must cross both
    # numerical handlers unchanged, with no full-diagonal/affine retry.
    optional = cases / "optional_SOC"
    optional.mkdir()
    H, gradient = np.eye(2), np.array([-2., 0.])
    lo = np.full(2, -np.inf)
    hi = -lo
    matrix = np.array([[1., 0.], [-1., 0.]])
    lower = np.array([0., -1.])
    with redirect_stdout(log):
        step, _, KKT = qp._linearly_constrained_shared_step(H, gradient, lo, hi, matrix, lower)
    prior = SimpleNamespace(step=step.copy(), report={"shared_linear_domain_step": KKT})

    def solve_optional(H, g, diagonal, **options):
        x, _, check = qp._linearly_constrained_shared_step(
            H, g, lo, hi, options["global_step_constraint_matrix"], options["global_step_constraint_lower"])
        return SimpleNamespace(step=x, report={"shared_linear_domain_step": check})

    optional_log = io.StringIO()
    with source_control_context(optional) as context:
        polls = instrument_poll(context, [True], False)
        try:
            with redirect_stdout(optional_log):
                joint._sequential_measured_shared_source_curvature(
                    np.zeros(2), prior, np.arange(2), dict(lower=lower),
                    dict(global_step_constraint_matrix=matrix, global_step_constraint_lower=lower),
                    H, gradient, np.ones(2),
                    lambda *args: dict(lower=np.array([0.1, -0.5])), solve_optional)
        except SourceControlStop as interruption:
            stop = interruption.stop
        else:
            raise AssertionError("Actual optional SOC did not propagate SourceControlStop")
    events = [json.loads(line) for line in optional_log.getvalue().splitlines()]
    phases = [event["QP_phase"] for event in events if event.get("stage") == "QP_backend_begin"]
    assert phases == ["restricted_affine"] and any(value >= 1 for value in polls)
    assert np.array_equal(prior.step, step)
    checks.append(dict(case="actual_IPM_cancel_crosses_QP_and_SOC", requested_stop=stop,
                       observed_IPM_iterations=polls, backend_phases=phases,
                       previous_verified_direction_bitexact=True))
    log.write(optional_log.getvalue())

    scheduled = with_trial_precheck(joint.solve_joint_source)
    target = np.array([2., 0.25, 1.])
    masks, scales = np.ones(3, bool), np.ones(3)
    for cache in (False, True):
        output = cases / ("source_loop_cache_stop" if cache else "source_loop_stop")
        output.mkdir()
        profiles = Profiles()
        armed = [False]
        accepted_q = []
        linearizations = []
        writer_calls = []
        profile_at_control = []

        def evaluate(q, jacobian):
            factors = np.r_[profiles.weights.ravel(), 1.]
            residual = (q - target) * factors
            return (residual, diags(factors).tocsr()) if jacobian else residual

        def blocks(q, jacobian):
            result = evaluate(q, jacobian)
            if jacobian:
                linearizations.append(dict(q_sha256=digest(q),
                    profile_weights_sha256=digest(profiles.weights),
                    profile_update_count=profiles.calls))
            yield SourceRowBlock(result[0] if jacobian else result,
                result[1] if jacobian else None, masks, scales,
                dict(family="unchanged_small_source", original_start=0, original_stop=3,
                     source_sha256=digest(target)))

        def domain(q, selected):
            return dict(matrix=np.array([[-1., 0.], [1., 0.]]),
                        lower=np.array([q[0] - 0.5, -q[0]]))

        def checkpoint(q, event):
            if event["checkpoint_kind"] == "accepted":
                accepted_q.append(q.copy())
                armed[0] = True
            return None

        def writer(q, normal):
            assert normal.native_replay_cache is not None
            assert np.array_equal(q, accepted_q[-1])
            record = dict(q_sha256=digest(q), profile_weights_sha256=digest(profiles.weights),
                          profile_update_count=profiles.calls,
                          frozen_profile_token=normal.contract.frozen_profile_token,
                          full_original_rows=normal.contract.row_count,
                          source_contract=asdict(normal.contract))
            np.savez(output / "requested_current_cache.npz", q=q,
                     profile_weights=profiles.weights, H=normal.normal.toarray(),
                     gradient=normal.gradient, cost=np.asarray(normal.cost))
            writer_calls.append(record)
            return record

        with source_control_context(output) as context:
            original_poll = context.poll

            def observe(info=None):
                if info is not None and armed[0] and info.iterations >= 1:
                    profile_at_control.append(dict(weights=profiles.weights.copy(), count=profiles.calls))
                    write_control(output, cache)
                    armed[0] = False
                return original_poll(info)

            context.poll = observe
            with redirect_stdout(log):
                result = scheduled(evaluate, np.zeros(3), masks, scales, output,
                    lambda q: bool(-1e-10 <= q[0] <= 0.5 + 1e-10),
                    update_profiles=profiles, checkpoint_callback=checkpoint,
                    checkpoint_identity=dict(source="fixed_small_source"),
                    checkpoint_arrays=dict(source_rows=np.arange(3)),
                    global_schur_count=2, global_schur_indices=np.array([0, 1]),
                    schur_backtracking=True, shared_linear_constraints=domain,
                    source_block_factory=blocks if cache else None,
                    cache_native_source_replay=cache, reuse_source_jacobian=cache,
                    save_current_linearization_cache=writer if cache else None,
                    source_stage_timing=True,
                    source_trial_precheck=lambda parent, trial, **kwargs: dict(
                        action="continue", acceptance_authority=False, reason="all original checks required"))
        assert result.status == "stopped_by_control" and len(result.history) == 1
        assert np.array_equal(result.q, accepted_q[-1])
        assert len(profile_at_control) == 1
        with np.load(result.checkpoint, allow_pickle=False) as saved:
            assert np.array_equal(saved["q"], accepted_q[-1])
            assert np.array_equal(saved["profile__weights_0"], profile_at_control[0]["weights"])
        if cache:
            assert len(writer_calls) == 1
            assert writer_calls[0]["profile_update_count"] == profile_at_control[0]["count"]
            assert writer_calls[0]["profile_weights_sha256"] == digest(profile_at_control[0]["weights"])
            assert linearizations[-1]["q_sha256"] == digest(accepted_q[-1])
            assert linearizations[-1]["profile_update_count"] == profile_at_control[0]["count"]
            assert result.stop_reason["current_linearization_cache"] == writer_calls[0]
        else:
            assert not writer_calls
        checks.append(dict(case="original_source_loop_cache_stop" if cache else "original_source_loop_stop",
                           status=result.status, accepted_history_count=len(result.history),
                           last_accepted_q_preserved_bitexact=True, checkpoint=str(result.checkpoint),
                           current_profile_preserved_bitexact=True,
                           profile_update_count=profiles.calls, cache_writer_calls=writer_calls,
                           fresh_source_linearizations=linearizations))

    outside = raw_solution(clarabel.DefaultSolver)
    assert digest(np.asarray(outside.x)) == digest(np.asarray(raw.x))
    checks.append(dict(case="next_QP_after_control_scope", original_solution_bitexact=True))
    args.output.write_text(json.dumps(dict(
        schema="rinkspace.actual-small-IPM-source-control-integration.v1",
        installation=installation, optional_SOC_route=route,
        scheduled_source_interface=scheduled.execution_change, checks=checks,
        production_numeric_time_or_iteration_limit_added=False,
        actual_programme_graph_or_old_full_QP_baseline_loaded=False),
        ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    (cases / "actual_stage_logs.jsonl").write_text(log.getvalue(), encoding="utf8")
    print(json.dumps(dict(output=str(args.output), checks=checks)), flush=True)


if __name__ == "__main__":
    main()
