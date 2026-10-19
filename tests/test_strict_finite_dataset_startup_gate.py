from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from calibration.program_tracking.strict_model_scope import file_sha
from calibration.program_tracking.strict_source_fit_dataset import SCHEMA
from calibration.solver_runtime.strict_model_scope_inputs import require_solver_scope_policy


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding='utf8')
    return dict(path=str(path), sha256=file_sha(path))


@pytest.fixture
def configured(tmp_path):
    source = dict(video_sha256='a' * 64, time_base='1/15360', pixel_domain='native1920x1080')
    model = write_json(tmp_path / 'model.json', dict(q_sha256='b' * 64))
    dataset = write_json(tmp_path / 'dataset.json',
        dict(schema=SCHEMA, source=source, frozen_model_reference=model))
    motion = write_json(tmp_path / 'motion.json',
        dict(schema='rinkspace.strict-model-scope-original-motion-inputs.v1',
             unknown_or_unreviewed_default='excluded', source=source))
    policy = dict(schema='rinkspace.strict-finite-dataset-startup-policy.v2',
        allowed_static_surfaces=['ice', 'board'], unknown_or_unreviewed_default='excluded',
        fit_dataset=dataset, motion_inputs=motion)
    config = dict(strict_model_scope_policy=policy,
        source_options={'--strict-source-fit-dataset': dataset['path'],
                        '--strict-motion-inputs': motion['path']})
    path = tmp_path / 'config.json'
    write_json(path, config)
    return SimpleNamespace(path=path, config=config, dataset=dataset, motion=motion, model=model)


def save(c):
    write_json(c.path, c.config)


def test_configured_paths_hashes_and_runtime_implementation_are_required(configured):
    result = require_solver_scope_policy(configured.path, require_policy=True)
    assert result['fit_dataset'] == configured.dataset
    assert result['motion_inputs'] == configured.motion
    assert result['runtime_full_inventory_validation_required'] is True


def test_missing_policy_and_pending_legacy_sidecar_cannot_fall_back_to_old_pixels(configured):
    configured.config.pop('strict_model_scope_policy')
    save(configured)
    with pytest.raises(ValueError, match='禁止无策略回退旧输入'):
        require_solver_scope_policy(configured.path, require_policy=True)
    configured.config['strict_model_scope_policy'] = dict(
        schema='rinkspace.strict-model-scope-startup-policy.v1',
        allowed_static_surfaces=['ice', 'board'], unknown_or_unreviewed_default='excluded',
        motion_inputs=configured.motion, all_pixel_families_routed_to_strict_input=False,
        source_review_ready_for_fit=False)
    save(configured)
    with pytest.raises(ValueError, match='禁止旧输入继续拟合'):
        require_solver_scope_policy(configured.path, require_policy=True)


@pytest.mark.parametrize('reference', ['dataset', 'motion', 'model'])
def test_any_authenticated_input_reference_byte_change_refuses_startup(configured, reference):
    ref = getattr(configured, reference)
    with Path(ref['path']).open('a', encoding='utf8') as stream:
        stream.write(' ')
    with pytest.raises(ValueError, match='bytes changed'):
        require_solver_scope_policy(configured.path, require_policy=True)


@pytest.mark.parametrize('option', ['--strict-source-fit-dataset', '--strict-motion-inputs'])
def test_configuration_route_cannot_select_an_unbound_pixel_manifest(configured, option):
    configured.config['source_options'][option] = str(configured.path.parent / 'other.npz')
    save(configured)
    with pytest.raises(ValueError, match='not explicitly configured'):
        require_solver_scope_policy(configured.path, require_policy=True)


def test_different_video_or_model_external_scope_is_rejected(configured):
    motion = json.loads(Path(configured.motion['path']).read_text(encoding='utf8'))
    motion['source']['video_sha256'] = 'c' * 64
    configured.config['strict_model_scope_policy']['motion_inputs'] = write_json(
        Path(configured.motion['path']), motion)
    save(configured)
    with pytest.raises(ValueError, match='source identities differ'):
        require_solver_scope_policy(configured.path, require_policy=True)
    configured.config['strict_model_scope_policy']['allowed_static_surfaces'].append('stands')
    save(configured)
    with pytest.raises(ValueError, match='scope policy is invalid'):
        require_solver_scope_policy(configured.path, require_policy=True)


@pytest.fixture
def cli(configured, monkeypatch):
    module = importlib.import_module('calibration.solver_runtime.__main__')
    calls = []
    monkeypatch.setitem(sys.modules, 'run_current_named66_full_programme',
        SimpleNamespace(main=lambda: calls.append(list(sys.argv))))
    relocation = importlib.import_module('calibration.solver_runtime.source_cache_output_relocation')
    monkeypatch.setattr(relocation, 'install_output_relocation_compatibility', lambda: None)
    monkeypatch.delenv('RINKSPACE_BENCHMARK_DIAGNOSTIC_DIR', raising=False)

    def invoke(arguments):
        monkeypatch.setattr(sys, 'argv', ['test-entry', 'named66', '--input-config',
                                        str(configured.path), *arguments])
        module.main()

    required = ['--strict-source-fit-dataset', configured.dataset['path'],
                '--strict-motion-inputs', configured.motion['path']]
    return SimpleNamespace(invoke=invoke, calls=calls, required=required)


def test_actual_cli_reaches_driver_only_after_both_authenticated_routes(cli):
    cli.invoke(cli.required)
    assert len(cli.calls) == 1


@pytest.mark.parametrize('bad', ['missing_dataset', 'wrong_dataset', 'wrong_motion',
    'duplicate_dataset', 'duplicate_motion', 'duplicate_equal_form'])
def test_actual_cli_wrong_or_duplicate_final_route_never_reaches_driver(cli, configured, bad):
    arguments = list(cli.required)
    if bad == 'missing_dataset':
        arguments = arguments[2:]
    elif bad == 'wrong_dataset':
        arguments[1] = str(configured.path.parent / 'other.json')
    elif bad == 'wrong_motion':
        arguments[3] = str(configured.path.parent / 'other.json')
    elif bad == 'duplicate_dataset':
        arguments.extend(['--strict-source-fit-dataset', str(configured.path.parent / 'other.json')])
    elif bad == 'duplicate_motion':
        arguments.extend(['--strict-motion-inputs', str(configured.path.parent / 'other.json')])
    else:
        arguments.append('--strict-source-fit-dataset=' + str(configured.path.parent / 'other.json'))
    with pytest.raises(SystemExit, match='source route'):
        cli.invoke(arguments)
    assert not cli.calls
