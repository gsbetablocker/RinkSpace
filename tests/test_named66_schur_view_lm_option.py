"""Exercise the real CLI boundary without constructing a programme source graph."""
import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csc_matrix

from calibration.solver_runtime import install

install()
from z190_joint_source_solver import schur_trial_linear_step
from z190_global_schur_linear_step import solve_global_schur_step


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "calibration/solver_runtime/legacy/run_current_named66_full_programme.py"
REPLAY = ROOT / "calibration/solver_runtime/legacy/structured_programme_source_candidate/complete_same_state_replay.py"


def driver_tree():
    return ast.parse(DRIVER.read_text(encoding="utf8"))


def parse_options(monkeypatch, *arguments):
    main = next(node for node in driver_tree().body if isinstance(node, ast.FunctionDef) and node.name == "main")
    start = next(i for i, node in enumerate(main.body) if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "parser" for t in node.targets))
    stop = next(i for i, node in enumerate(main.body) if isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "parse_args")
    # Execute the actual parser statements, then return before source preparation.
    body = deepcopy(main.body[start:stop + 1]) + [ast.Return(ast.Name("opt", ast.Load()))]
    definition = ast.FunctionDef(name="parse_only", args=ast.arguments(posonlyargs=[], args=[],
        kwonlyargs=[], kw_defaults=[], defaults=[]), body=body, decorator_list=[])
    namespace = dict(argparse=argparse, Path=Path, __doc__="named66 interface test")
    exec(compile(ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[])),
                 str(DRIVER), "exec"), namespace)
    monkeypatch.setattr("sys.argv", ["named66", "--output", "unused-test-output", *arguments])
    return namespace["parse_only"]()


@pytest.mark.parametrize("arguments, expected", [((), 1.), (("--schur-view-lm-ratio", "0"), 0.),
                                                  (("--schur-view-lm-ratio", "1"), 1.)])
def test_cli_default_and_explicit_routes_reach_actual_solver_keyword(monkeypatch, arguments, expected):
    opt = parse_options(monkeypatch, *arguments)
    calls = [node for node in ast.walk(driver_tree()) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "solve_joint_source"]
    assert len(calls) == 1
    keyword = next(k for k in calls[0].keywords if k.arg == "schur_view_lm_ratio")
    actual = eval(compile(ast.Expression(keyword.value), str(DRIVER), "eval"), {"opt": opt})
    assert actual == opt.schur_view_lm_ratio == expected


@pytest.mark.parametrize("value", ["0.5", "2", "nan"])
def test_cli_rejects_unsupported_ratios_before_source_build(monkeypatch, value):
    with pytest.raises(SystemExit) as error:
        parse_options(monkeypatch, "--schur-view-lm-ratio", value)
    assert error.value.code == 2


def numerical_manifest(ratio):
    assignment = next(node for node in ast.walk(driver_tree()) if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                and t.value.id == "damping_continuation"
                and isinstance(t.slice, ast.Constant) and t.slice.value == "numerical_runtime_options"
                for t in node.targets))
    namespace = dict(opt=SimpleNamespace(schur_view_lm_ratio=ratio),
                     damping_continuation=dict(inherited=True, value=.001))
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(DRIVER), "exec"), namespace)
    return namespace["damping_continuation"]


def test_numeric_option_changes_checkpoint_provenance_but_not_source_math_identity():
    tree = ast.parse(REPLAY.read_text(encoding="utf8"))
    names = {"IDENTITY_KEY", "PROVENANCE_KEYS", "EXECUTION_ENV_KEYS"}
    nodes = [node for node in tree.body if isinstance(node, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)]
    nodes += [node for node in tree.body if isinstance(node, ast.FunctionDef)
              and node.name in ("_plain", "_semantic_manifest")]
    namespace = dict(deepcopy=deepcopy, json=json, np=np)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(REPLAY), "exec"), namespace)
    semantic = namespace["_semantic_manifest"]
    source = dict(source_contract={"rows": 12, "sigma": [3.]}, highest_direct_human_hard_truth={"cap": 0.})
    before = dict(source, optimizer_damping_continuation=numerical_manifest(1.))
    after = dict(source, optimizer_damping_continuation=numerical_manifest(0.))
    assert before != after and semantic(before) == semantic(after) == source
    assert after["optimizer_damping_continuation"]["numerical_runtime_options"]["affects_next_numerical_step_only"]
    # Unknown source/cap changes retain the original fail-closed comparison.
    changed = deepcopy(after)
    changed["highest_direct_human_hard_truth"]["cap"] = 1.
    assert semantic(changed) != semantic(before)
    registered = json.loads((ROOT / "calibration/solver_runtime/source_manifest.json").read_text(encoding="utf8"))
    assert registered["modules"]["run_current_named66_full_programme"]["sha256"] == hashlib.sha256(DRIVER.read_bytes()).hexdigest()


@pytest.mark.parametrize("ratio", [0., 1.])
def test_existing_backend_routes_preserve_source_normal_and_gradient(ratio):
    H = csc_matrix(np.array([[6., 1., .2, .3], [1., 5., .4, .1],
                            [.2, .4, 4., .5], [.3, .1, .5, 3.]]))
    gradient = np.array([.3, -.2, .5, -.6])
    source_diagonal = np.maximum(H.diagonal(), 1.)
    original_H, original_g = H.toarray().copy(), gradient.copy()
    damping = .1
    actual = schur_trial_linear_step(H, gradient, source_diagonal, global_count=2,
                                    damping=damping, schur_view_lm_ratio=ratio)
    expected_H = original_H.copy()
    expected_H[np.arange(2, 4), np.arange(2, 4)] += damping * ratio * source_diagonal[2:]
    expected = solve_global_schur_step(csc_matrix(expected_H), gradient, global_count=2,
        global_damping=damping, numerical_ridge_diagonal=1e-8 * source_diagonal,
        global_damping_diagonal_floor=1.)
    np.testing.assert_array_equal(H.toarray(), original_H)
    np.testing.assert_array_equal(gradient, original_g)
    np.testing.assert_allclose(actual.step, expected.step, rtol=0., atol=2e-15)
