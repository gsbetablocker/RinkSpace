"""Installed source authority is restored on the authenticated cache-hit route."""
import ast
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from calibration.solver_runtime import named66_source_skeleton_adapter_runtime as adapter
from calibration.solver_runtime import source_state_skeleton_cache_runtime as skeleton
from calibration.solver_runtime.original_measured_transition_locations import CONTRACT_KEY, MODE, SCHEMA


@pytest.fixture
def installed(tmp_path):
    path = tmp_path/'authority.json'; path.write_text('{"original_observations":true}', encoding='utf8')
    contract = dict(schema=SCHEMA+'.installed', observation=MODE,
                    packet_reference=dict(path=str(path.resolve()), sha256=skeleton._sha(path)),
                    records=[dict(original_packet_row_index=7, points=3)])
    state = SimpleNamespace(**{CONTRACT_KEY: contract})
    opt = SimpleNamespace(original_measured_transition_location_source=path)
    return state, opt, contract


def test_restored_local_is_exact_already_installed_contract_without_row_changes(installed):
    state, opt, contract = installed
    state.groups = [[dict(uv=[[1., 2.]], normal_interval=[[-4., 4.]])]]
    saved = deepcopy(state.groups)
    values = dict(state=state)
    adapter._restore_original_measured_location_local(values, opt)
    assert values['original_measured_locations'] is contract
    assert state.groups == saved
    assert getattr(state, CONTRACT_KEY) is contract


def test_disabled_mode_restores_none_and_rejects_unrequested_installed_authority(installed):
    state, opt, _ = installed
    opt.original_measured_transition_location_source = None
    with pytest.raises(skeleton.SkeletonCacheRejected, match='Unrequested'):
        adapter._restore_original_measured_location_local(dict(state=state), opt)
    values = dict(state=SimpleNamespace())
    adapter._restore_original_measured_location_local(values, opt)
    assert values['original_measured_locations'] is None


@pytest.mark.parametrize('change', ['missing', 'schema', 'mode', 'path', 'bytes'])
def test_current_source_authority_must_match_exact_installed_state(installed, tmp_path, change):
    state, opt, contract = installed
    if change == 'missing':
        delattr(state, CONTRACT_KEY)
    elif change == 'schema':
        contract['schema'] = 'old'
    elif change == 'mode':
        contract['observation'] = 'normal_interval'
    elif change == 'path':
        other = tmp_path/'other.json'; other.write_bytes(opt.original_measured_transition_location_source.read_bytes())
        opt.original_measured_transition_location_source = other
    else:
        opt.original_measured_transition_location_source.write_text('{"changed":true}', encoding='utf8')
    with pytest.raises(skeleton.SkeletonCacheRejected, match='authority differs'):
        adapter._restore_original_measured_location_local(dict(state=state), opt)


def test_original_driver_cold_and_hit_routes_bind_same_live_out_local(installed, monkeypatch):
    state, opt, contract = installed
    calls = []
    values = {name: None for name in (*skeleton.CAPTURE_STATIC_LOCAL_NAMES,
              *skeleton.RESTORED_DRIVER_LOCAL_NAMES, *adapter._MISSING_LOCALS, 'capold', 'geometry', 'contract')}
    values['state'] = state
    class TinyAdapter:
        def __init__(self, **kwargs):
            pass
        def load(self, current, output):
            if calls[-1] == 'cold':
                return None
            adapter._restore_original_measured_location_local(values, current)
            return values
        @contextmanager
        def capture_input_reads(self, current):
            yield
        def save(self, current, options):
            assert current['original_measured_locations'] is contract
    monkeypatch.setattr(adapter, 'Named66SourceSkeletonAdapter', TinyAdapter)
    source = '''
def main():
    state, args, template, parent = prepare(DEFAULT_SEED, out)
    original_measured_locations = None
    if opt.original_measured_transition_location_source is not None:
        original_measured_locations = installer(state, opt.original_measured_transition_location_source)
    active_backend_report = restore_native_backend_contract(stage='before_source_actor_construction')
    return original_measured_locations
'''
    def install(current, path):
        calls.append('installer')
        return contract
    namespace = dict(current_seed_parts=lambda *args: None, DEFAULT_SEED=None, out=Path('.'), opt=opt,
        prepare=lambda *args: (state, None, None, None), installer=install,
        restore_native_backend_contract=lambda **kwargs: None)
    tree = adapter.attach_named66_source_skeleton_reuse(ast.parse(source), namespace,
        cache_directory='.', report_directory='.', declared_source_roots=('authority',),
        complete_static_input_closure_confirmed=True)
    exec(compile(tree, '<actual_skeleton_AST_hit_local>', 'exec'), namespace)
    calls.append('cold')
    assert namespace['main']() is contract
    assert calls == ['cold', 'installer']
    calls.append('hit')
    assert namespace['main']() is contract
    assert calls == ['cold', 'installer', 'hit']


@pytest.mark.parametrize('name', ['named66_source_skeleton_adapter_runtime.py', 'source_state_skeleton_cache_runtime.py'])
def test_exact_existing_predecessor_keeps_unchanged_writer_AST(name):
    root = Path(skeleton.__file__).resolve().parent
    from calibration.solver_runtime.config import data_root
    previous = data_root()/skeleton._MEASURED_LOCATION_READER_SNAPSHOT/'execution_sources'/name
    if not previous.is_file():
        pytest.skip('Exact local predecessor is in the established Data execution archive')
    expected = skeleton._MEASURED_LOCATION_READER_PREDECESSORS[name]
    result = skeleton.compatible_static_reader(root/name, expected)
    assert result['predecessor_source_snapshot'] == str(previous)
    assert result['measured_location_reader_local_restore']
    assert not result['original_read_and_registered_producer_SHA_exempted']
    assert skeleton.compatible_static_reader(root/name, '0'*64) is None


def test_bridge_cannot_admit_writer_or_original_source_change(monkeypatch):
    root = Path(skeleton.__file__).resolve().parent
    from calibration.solver_runtime.config import data_root
    name = 'source_state_skeleton_cache_runtime.py'
    previous = data_root()/skeleton._MEASURED_LOCATION_READER_SNAPSHOT/'execution_sources'/name
    if not previous.is_file():
        pytest.skip('Exact local predecessor is in Data')
    original_read = Path.read_text
    def changed(path, *args, **kwargs):
        result = original_read(path, *args, **kwargs)
        if path == root/name:
            result = result.replace("'motion_input_compaction',", "'motion_input_compaction', 'changed_source',")
        return result
    monkeypatch.setattr(Path, 'read_text', changed)
    expected = skeleton._MEASURED_LOCATION_READER_PREDECESSORS[name]
    with pytest.raises(skeleton.SkeletonCacheRejected, match='writer/installer contract changed'):
        skeleton.compatible_static_reader(root/name, expected)
    assert skeleton.compatible_static_reader(previous, expected) is None
