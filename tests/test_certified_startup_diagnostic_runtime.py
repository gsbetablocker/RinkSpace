"""AST and evidence admission only; no source graph or derivative evaluation."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from calibration.solver_runtime import install
from calibration.solver_runtime import certified_startup_diagnostic_runtime as runtime

install()


def _fixture(tmp_path, monkeypatch):
    import calibration.solver_runtime.runtime as registered
    original = '''def main():
    if opt.reuse_complete_linearization is None:
        print(dict(stage="actual_all_small_source_values_and_jacobians"))
        expensive_independent_diagnostic()
    source.guard(q)
    original_fresh_normal()
'''
    code = tmp_path / 'original.py'
    code.write_text(original, encoding='utf8')
    producer = dict(path=str(code), sha256=hashlib.sha256(code.read_bytes()).hexdigest())
    facts = dict(source_contract=dict(source_rows=2, source_sha256='unchanged'), full_parameters=3)
    evidence = dict(directory=str(tmp_path / 'evidence'), files=dict(original=producer),
                    source_facts=facts, q_sha256='prior_q', frozen_profile_token='prior_profile',
                    original_main_sha256=hashlib.sha256(original.encode('utf8')).hexdigest())
    monkeypatch.setattr(runtime, '_read_evidence', lambda _: evidence)
    monkeypatch.setattr(runtime, '_producer_files', lambda: [producer])
    monkeypatch.setattr(runtime, '_source_facts', lambda manifest, q: manifest)
    monkeypatch.setattr(registered, 'source_path', lambda _: code)
    namespace = {}
    tree = runtime.attach_certified_startup_diagnostic_reuse(
        ast.parse(original), namespace, tmp_path / 'receipt', tmp_path / 'evidence')
    opt = SimpleNamespace(camera_seed=Path('same_complete_seed.npz'), reuse_wide_linearization=None)
    source = SimpleNamespace(scales=np.ones(2))
    return original, tree, namespace, facts, opt, source, code


def test_only_diagnostic_condition_changes_and_guard_stays(tmp_path, monkeypatch):
    original, tree, namespace, facts, opt, source, _ = _fixture(tmp_path, monkeypatch)
    old = ast.parse(original)
    before = old.body[0].body
    after = tree.body[0].body
    assert ast.dump(before[0].body[0]) == ast.dump(after[0].body[0])
    assert ast.dump(before[0].body[1]) == ast.dump(after[0].body[1])
    assert [ast.dump(n) for n in before[1:]] == [ast.dump(n) for n in after[1:]]
    assert namespace['_rinkspace_reuse_startup_derivatives'](opt, facts, np.arange(3.), source)
    assert not any('_rinkspace' in ast.dump(n) for n in after[1:])


def test_changed_source_and_changed_producer_run_original(tmp_path, monkeypatch):
    _, _, namespace, facts, opt, source, producer = _fixture(tmp_path, monkeypatch)
    changed = dict(facts, source_contract=dict(source_rows=2, source_sha256='new_source'))
    gate = namespace['_rinkspace_reuse_startup_derivatives']
    assert not gate(opt, changed, np.arange(3.), source)
    report = json.loads((tmp_path / 'receipt/startup_diagnostic_reuse.json').read_text())
    assert report['changed_source_or_domain_fields'] == ['source_contract.source_sha256']
    producer.write_text('changed producer\n', encoding='utf8')
    assert not gate(opt, facts, np.arange(3.), source)
    report = json.loads((tmp_path / 'receipt/startup_diagnostic_reuse.json').read_text())
    assert 'changed or missing' in report['reason']


def test_no_seed_or_explicit_wide_replay_keep_original(tmp_path, monkeypatch):
    _, _, namespace, facts, opt, source, _ = _fixture(tmp_path, monkeypatch)
    gate = namespace['_rinkspace_reuse_startup_derivatives']
    opt.camera_seed = None
    assert not gate(opt, facts, np.arange(3.), source)
    opt.camera_seed = Path('seed.npz')
    opt.reuse_wide_linearization = Path('wide.npz')
    assert not gate(opt, facts, np.arange(3.), source)


def test_missing_evidence_falls_back_and_boundary_mismatch_is_fatal(tmp_path, monkeypatch):
    original, _, _, _, _, _, _ = _fixture(tmp_path, monkeypatch)
    def missing(_):
        raise FileNotFoundError('original evidence missing')
    monkeypatch.setattr(runtime, '_read_evidence', missing)
    namespace = {}
    runtime.attach_certified_startup_diagnostic_reuse(ast.parse(original), namespace,
        tmp_path / 'missing', tmp_path / 'evidence')
    assert not namespace['_rinkspace_reuse_startup_derivatives'](
        SimpleNamespace(camera_seed='seed', reuse_wide_linearization=None), {},
        np.arange(3.), SimpleNamespace(scales=np.ones(2)))
    with pytest.raises(ValueError, match='boundary changed'):
        runtime.attach_certified_startup_diagnostic_reuse(ast.parse('def main(): pass'), {},
            tmp_path / 'wrong', tmp_path / 'evidence')


def test_original_registered_diagnostic_body_is_byte_identical(tmp_path, monkeypatch):
    from calibration.solver_runtime.runtime import source_path
    original = ast.parse(source_path('run_current_named66_full_programme').read_text(encoding='utf8'))
    before = ast.dump(original, include_attributes=False)
    monkeypatch.setattr(runtime, '_read_evidence', lambda _: (_ for _ in ()).throw(FileNotFoundError('no evidence')))
    tree = runtime.attach_certified_startup_diagnostic_reuse(original, {}, tmp_path / 'original', tmp_path)
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.If) and isinstance(n.test, ast.BoolOp)
             and '_rinkspace_reuse_startup_derivatives' in ast.dump(n.test)]
    assert len(nodes) == 1
    nodes[0].test = nodes[0].test.values[0]
    assert ast.dump(tree, include_attributes=False) == before
