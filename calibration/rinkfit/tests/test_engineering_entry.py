"""Regression for the new production dispatch; no solver or source certification is mocked as accuracy."""
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rinkfit_joint import joint_solver

def test_engineering_dispatch_keeps_source_guard_and_feedback(tmp_path, monkeypatch):
    executable = tmp_path / 'joint_fit.exe'
    executable.write_bytes(b'test executable fixture')
    digest = hashlib.sha256(executable.read_bytes()).hexdigest()
    (tmp_path / 'recipe.json').write_text(json.dumps(dict(executable_sha256=digest, runtime_bin_paths=[])))
    model = tmp_path / 'model.json'; model.write_text('{}')
    packet = tmp_path / 'packet.json'; packet.write_text('{}')
    out = tmp_path / 'run'
    seen = []
    monkeypatch.setattr(joint_solver, 'assert_source_identity_current', lambda p: seen.append(('qualified',p)))
    import rinkfit_joint.feedback_checkpoint as feedback
    class Observer:
        def __init__(self, output, source): seen.append(('observer', output, source))
        def __enter__(self): return self
        def __exit__(self, *args): return False
    monkeypatch.setattr(feedback, 'CheckpointObserver', Observer)
    def execute(command, **kwargs):
        seen.append(('command',command));out.mkdir()
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(joint_solver.subprocess, 'run', execute)
    assert joint_solver.execute_explicit(executable,model,packet,out,engineering_recovery=True)==0
    assert seen[0]==('qualified',packet)
    assert any(r[0]=='observer' for r in seen)
    command=next(r[1]for r in seen if r[0]=='command')
    assert command[1]=='--engineering-recovery'
    receipt=json.loads((out/'execution.json').read_text())
    assert receipt['objective_id']=='engineering_recovery_native_pixel_v1'
    assert receipt['engineering_recovery'] and not receipt['certificate_in_native_and_formal_receipts']

def test_engineering_does_not_accept_audit_or_guidance_alias(tmp_path, monkeypatch):
    executable=tmp_path/'joint_fit.exe';executable.write_bytes(b'test')
    (tmp_path/'recipe.json').write_text(json.dumps(dict(executable_sha256=hashlib.sha256(b'test').hexdigest(),runtime_bin_paths=[])))
    monkeypatch.setattr(joint_solver,'assert_source_identity_current',lambda _:None)
    with pytest.raises(ValueError,match='independent production objective'):
        joint_solver.execute_explicit(executable,tmp_path/'m',tmp_path/'p',tmp_path/'out',engineering_recovery=True,audit_only=True)


def test_engineering_checkpoint_feedback_does_not_claim_formal_acceptance(tmp_path):
    from rinkfit_joint.feedback_checkpoint import checkpoint_receipt_path
    receipt=tmp_path/'engineering_checkpoint_receipt.json'
    receipt.write_text(json.dumps(dict(objective_id='engineering_recovery_native_pixel_v1', formal_acceptance_claimed=False)))
    assert checkpoint_receipt_path(tmp_path)==receipt
    receipt.write_text(json.dumps(dict(objective_id='engineering_recovery_native_pixel_v1', formal_acceptance_claimed=True)))
    with pytest.raises(ValueError, match='objective identity'):
        checkpoint_receipt_path(tmp_path)


def test_engineering_checkpoint_rejects_ambiguous_objectives(tmp_path):
    from rinkfit_joint.feedback_checkpoint import checkpoint_receipt_path
    (tmp_path/'formal_receipt.json').write_text('{}')
    (tmp_path/'engineering_checkpoint_receipt.json').write_text('{}')
    with pytest.raises(ValueError, match='Ambiguous'):
        checkpoint_receipt_path(tmp_path)
