"""Fixed checkpoint identity and dispatch only: no real model, solve or UI."""
import json
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from calibration.program_tracking.body_tracking import skate_integration as integration
from calibration.program_tracking.body_tracking import skate_delivery as delivery
from calibration.program_tracking.body_tracking.skate_checkpoint import (
    IDENTITY_FIELDS, checkpoint_identity, validate_checkpoint_identity,
)
from calibration.program_tracking.body_tracking.skate_template import SkateParameters


def files(root):
    return {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}


def case(root):
    project, data, out = root/'project', root/'data', root/'output'
    programme = project/'events/ankara_2026/programmes/okada_mei_sp'
    programme.mkdir(parents=True); data.mkdir(); (out/'fit').mkdir(parents=True)
    for name in ('opening_hand_1103.npz', 'opening_timing_1103.npz'):
        (programme/name).write_bytes(('synthetic parent '+name).encode())
    binding = SimpleNamespace(x=np.zeros(17), base_parameters=SkateParameters(),
        R0=np.tile(np.eye(3), (2, 1, 1)), t0=np.zeros((2, 3)),
        collar_constraints_enabled=False, record=lambda: {})
    info = dict(state_sha256='a'*64, native_library_sha256='b'*64,
        length_export_ratio=1., source_time_base=[1, 12800], shot={'id': 'synthetic'},
        state_path=str(project/'camera.npz'), native_library=str(data/'native.dll'))
    identity = checkpoint_identity(binding, project, info)
    saved = dict(identity, x=np.zeros(86), knots=np.zeros(8), active=np.arange(69),
        seconds=np.array([22.11, 44.97]), state_kind=np.array('continuous_shot'))
    checkpoint = out/'fit/candidate.npz'
    np.savez_compressed(checkpoint, **saved)
    ref = SimpleNamespace(binding=binding, fixed_checkpoint_identity=identity,
        p={'pts': np.array([283136, 283392])}, times=np.array([0., .02]), new_refinements=[])
    for name in ('skate_contact_1103.npz', 'viewer3d_fullbody_1102.json'):
        (programme/name).write_bytes(b'existing accepted asset')
    (project/'START_3D_COMPARISON.cmd').write_bytes(b'existing launcher')
    (out/'source_evidence.json').write_text('[]', encoding='utf8')
    (out/'binding.json').write_bytes(b'existing binding')
    return SimpleNamespace(project=project, data=data, out=out, checkpoint=checkpoint,
        binding=binding, ref=ref, saved=saved, identity=identity, info=info)


CHANGES = ('rocker', 'R_FB_initial', 't_FB_initial_solver',
    'parent_hand_parameters_sha256', 'parent_timing_parameters_sha256',
    'template_version', 'binding_version', 'camera_binding_json')


def change_saved(c, key):
    if key == 'rocker':
        p = json.loads(str(c.saved['base_skate_parameters_json']))
        p['main_rocker_m'] = 8*.3048
        c.saved['base_skate_parameters_json'] = np.array(json.dumps(p))
        expected = 'base_skate_parameters_json'
    elif key in ('R_FB_initial', 't_FB_initial_solver'):
        c.saved[key] = c.saved[key].copy(); c.saved[key].flat[0] += .001
        expected = key
    elif key == 'camera_binding_json':
        info = dict(c.info, state_sha256='c'*64)
        c.saved[key] = np.array(json.dumps(info)); expected = key
    else:
        c.saved[key] = np.array('different identity'); expected = key
    np.savez_compressed(c.checkpoint, **c.saved)
    return expected


def dispatch(monkeypatch, c):
    calls = []
    def forbidden(*a, **kw):
        raise AssertionError('Unexpected fitting/provider path')
    def build(*a):
        calls.append('build')
        return c.ref, None, c.info, None
    def problem(*a):
        calls.append('problem')
        return SimpleNamespace(body=SimpleNamespace(state=SimpleNamespace(
            knots=c.saved['knots'], active=c.saved['active'])), state_kind='continuous_shot')
    monkeypatch.setattr(integration, 'build', build)
    monkeypatch.setattr(integration, 'source_evidence', forbidden)
    monkeypatch.setattr(integration, 'SharedRegistration', forbidden)
    monkeypatch.setattr(integration, 'least_squares', forbidden)
    monkeypatch.setattr(integration, 'JointContactProblem', problem)
    monkeypatch.setattr(integration, 'restore_checkpoint_layers', lambda *a: calls.append('layers'))
    from calibration.program_tracking.body_tracking import skate_diagnostics
    monkeypatch.setattr(skate_diagnostics, 'inspect_checkpoint', lambda *a: calls.append('inspect') or {})
    monkeypatch.setattr(delivery, 'deliver', lambda *a, **kw: calls.append('deliver'))
    return calls


def run(c, *extra):
    integration.main(['--project', str(c.project), '--root', str(c.data),
        '--output', str(c.out), *extra])


@pytest.mark.parametrize('arguments', [
    ('--stage', 'export'), ('--checkpoint', 'missing.npz'),
    ('--stage', 'fit', '--checkpoint', 'missing.npz'),
    ('--checkpoint', 'missing.npz', '--base-checkpoint', 'seed.npz'),
])
def test_readonly_request_never_falls_through_to_build_or_registration(tmp_path, monkeypatch, arguments):
    c = case(tmp_path); calls = dispatch(monkeypatch, c); before = files(tmp_path)
    with pytest.raises((FileNotFoundError, SystemExit)):
        run(c, *arguments)
    assert not calls and files(tmp_path) == before


def test_disappearing_checkpoint_never_changes_dispatch_mode(tmp_path, monkeypatch):
    c = case(tmp_path); calls = dispatch(monkeypatch, c)
    def build(*args):
        calls.append('build'); c.checkpoint.unlink()
        return c.ref, None, c.info, None
    monkeypatch.setattr(integration, 'build', build)
    before = files(tmp_path); before.pop(c.checkpoint.relative_to(tmp_path))
    with pytest.raises(FileNotFoundError):
        run(c, '--checkpoint', str(c.checkpoint))
    assert calls == ['build'] and files(tmp_path) == before


@pytest.mark.parametrize('key', CHANGES)
def test_replay_rejects_fixed_mismatch_before_diagnosis_or_any_write(tmp_path, monkeypatch, key):
    c = case(tmp_path); expected = change_saved(c, key); calls = dispatch(monkeypatch, c)
    before = files(tmp_path)
    with pytest.raises(ValueError, match=expected):
        run(c, '--checkpoint', str(c.checkpoint), '--stage', 'export')
    assert calls == ['build']
    assert files(tmp_path) == before


@pytest.mark.parametrize('key', CHANGES)
def test_deliver_rejects_before_export_or_overwriting_existing_assets(tmp_path, monkeypatch, key):
    c = case(tmp_path); expected = change_saved(c, key)
    def forbidden(*a):
        raise AssertionError('Export reached before fixed identity rejection')
    monkeypatch.setattr(delivery, 'export_candidate', forbidden)
    before = files(tmp_path)
    with pytest.raises(ValueError, match=expected):
        delivery.deliver(c.project, c.data, c.out, c.ref, None, c.info, [], None,
            {'candidate_checkpoint': str(c.checkpoint)})
    assert files(tmp_path) == before


def test_bare_checkpoint_and_unbound_adjacent_sidecars_are_not_identity(tmp_path, monkeypatch):
    c = case(tmp_path)
    c.saved = {k: v for k, v in c.saved.items() if k not in IDENTITY_FIELDS}
    np.savez_compressed(c.checkpoint, **c.saved)
    for name in ('inputs.json', 'registration_inputs.json', 'binding_initial.json', 'fit/kernel_manifest.json'):
        (c.out/name).write_text(json.dumps({'rocker_ft': 7., 'note': 'adjacent, no checkpoint SHA binding'}))
    calls = dispatch(monkeypatch, c); before = files(tmp_path)
    with pytest.raises(ValueError, match='missing fields: base_skate_parameters_json, R_FB_initial, t_FB_initial_solver'):
        run(c, '--checkpoint', str(c.checkpoint))
    assert calls == [] and files(tmp_path) == before


def test_moving_identical_parent_files_and_camera_paths_can_replay(tmp_path, monkeypatch):
    old = case(tmp_path/'old'); moved = tmp_path/'moved'
    shutil.copytree(tmp_path/'old', moved)
    c = old
    c.project, c.data, c.out = moved/'project', moved/'data', moved/'output'
    c.checkpoint = c.out/'fit/candidate.npz'
    c.info = dict(c.info, state_path=str(c.project/'camera.npz'), native_library=str(c.data/'native.dll'))
    c.ref.fixed_checkpoint_identity = checkpoint_identity(c.binding, c.project, c.info)
    calls = dispatch(monkeypatch, c)
    run(c, '--checkpoint', str(c.checkpoint), '--stage', 'export')
    assert calls == ['build', 'layers', 'problem', 'inspect', 'deliver']


def test_shared_controls_cannot_change_before_delivery(tmp_path, monkeypatch):
    c = case(tmp_path); c.binding.x[0] = .001; before = files(tmp_path)
    with pytest.raises(ValueError, match='shared_x'):
        delivery.deliver(c.project, c.data, c.out, c.ref, None, c.info, [], None,
            {'candidate_checkpoint': str(c.checkpoint)})
    assert files(tmp_path) == before


def test_future_checkpoint_fields_keep_existing_fixed_identity(tmp_path):
    c = case(tmp_path); problem = object.__new__(integration.JointContactProblem)
    problem.reference = c.ref; problem.saved_base_layers = {'base_layer_count': np.array(0)}
    problem.body = SimpleNamespace(state=SimpleNamespace(knots=c.saved['knots'],
        active=c.saved['active'], seconds=c.saved['seconds']))
    problem.physics_times = np.array([0., .02]); problem.binding = c.binding
    problem.al = SimpleNamespace(multiplier=np.zeros(2), penalty=10.)
    problem.state_kind = 'continuous_shot'
    problem.ice_policy = integration.IceSoftPolicy(.005, .05, 1., .025, .08, 1.)
    packet = problem.checkpoint_fields(c.saved['x'])
    validate_checkpoint_identity(packet, c.identity)
    for key in IDENTITY_FIELDS:
        assert packet[key].dtype == c.identity[key].dtype
        assert packet[key].tobytes() == c.identity[key].tobytes()
