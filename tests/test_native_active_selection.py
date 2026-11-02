"""Select existing initialization directions without a DLL or a real fit."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]/'events/ankara_2026/programmes/okada_mei_sp'
sys.path.insert(0, str(HERE))
import fit_native_multicamera as fit
from rinkspace.domain.camera_coupling import build_multicamera_directions
from rinkspace.domain.camera_experiment import CAMERA_NAMES, _SCALAR_STORAGE_INDEX
from rinkspace.domain.ice_actual_chart import NOMINAL_WITNESS_INDICES


PROFILE_INDICES = tuple(range(15))+tuple(range(20, 31))
RADIAL_NAMES = ['k10', 'k11', 'k20', 'k21', 'wide_k1', 'wide_k2']
ICE_IDS = ['scene/test/curling_shared', 'scene/test/hockey_shared',
           'scene/test/ice_tolerance']


class State:
    def __init__(self):
        definitions = [
            ('scene/test/board', 'board_shape', 0, 6, range(1, 6)),
            (ICE_IDS[0], 'shared_geometry', 6, 8, range(8)),
            (ICE_IDS[1], 'shared_geometry', 14, 31, range(31)),
            (ICE_IDS[2], 'shared_geometry', 45, 107, range(107)),
            ('print/ad4', 'print_shape', 152, 4, [1, 3]),
            ('print/unobserved', 'print_shape', 156, 2, [0, 1]),
            ('print/inactive', 'print_shape', 158, 1, []),
        ]
        self.blocks = {identity: dict(id=identity, kind=kind, offset=offset,
            size=size, independent_local_indices=tuple(active), available=True,
            active=bool(active)) for identity, kind, offset, size, active in definitions}
        self.values = np.ones(159)*.1
        self.values[9] = .05
        self.values[45:152] = 0.
        self.layout = {}

    def active_columns(self, ids):
        return np.asarray([b['offset']+k for identity, b in self.blocks.items()
            if identity in ids for k in b['independent_local_indices']], dtype=int)


class Compiled(Exception):
    pass


@pytest.fixture
def compile_run(tmp_path, monkeypatch):
    """Run the adapter through direction compilation, then stop before evaluation."""
    state = State()
    shots = [dict(id=f's{i+1:02}', camera_index=c, row_begin=2*i,
        row_stop=2*i+2, control_begin=4*i, control_stop=4*i+4,
        first_pts=0, last_pts=1, reference_pts=0, time_base_num=1,
        time_base_den=1, first_control_time=0.)
        for i, c in enumerate([0, 1, 2, 0, 2, 1, 0, 2, 0])]
    manifest = dict(shots=shots, cameras=[dict(camera_definition='camera.json')]*3,
        control_dt=1., first_control_time=0., max_readout=0.,
        native_library='unused.dll', native_library_sha256='sha',
        point_input_sha256='sha', scene_path='scene.json', scene_file_sha256='sha')
    z = dict(pts=np.arange(18), path_indices=np.zeros(18, int),
        kind=np.ones(18, int), source_rows=np.arange(18),
        uv=np.arange(36, dtype=float).reshape(18, 2))
    rigs = np.zeros((3, 43))
    rigs[:, :9] = np.eye(3).ravel()
    rigs[:, 9:12] = [0, 0, 1]
    rigs[:, 12:15] = [1, 0, 0]
    rigs[:, 24] = 5000.
    rigs[:, 25] = 1.
    initial = dict(controls=np.zeros((36, 3)), rig=rigs,
        scene_values=state.values.copy(), stations=np.full(18, .5),
        source_rows=z['source_rows'])
    mapping = dict(path_ids=['print/instance::edge'])
    captured = {}
    library = SimpleNamespace()

    def read(path):
        name = Path(path).name
        if name == 'joint_manifest.json':
            return manifest
        if name == 'mapping.json':
            return mapping
        if 'metric_scale' in name:
            return dict(converted_span_xy_m=[60.])
        return {}

    class Timeline:
        first = 0.
        dt = 1.

        def __init__(self, *args):
            pass

        def row_times(self, pts):
            return np.zeros(len(pts))

        def local_exposure_time(self, shot, pts):
            return (np.asarray(pts)-shot['reference_pts'])*shot['time_base_num']/shot['time_base_den']

        def assert_support_separation(self, *args):
            pass

        def directions(self, *, continue_endpoint_cubics, sparse):
            raw = np.eye(108, 9)
            if sparse:
                from scipy.sparse import csr_matrix
                raw = csr_matrix(raw)
            else:
                raw = raw.reshape(36, 3, 9)
            return raw, [(s['id'], 'pan', 0) for s in shots]

    class Resident:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def configure_camera_parameters(self, **kwargs):
            captured['camera'] = kwargs

        def evaluate_errors(self):
            raise Compiled

    class Native:
        scene = dict(shared_geometry_options=127)
        paths = {'print/instance::edge': dict(kind=1, layer='advertisement')}
        ice_relative_limit = .01

        def __init__(self, *args, **kwargs):
            pass

        def resident(self, **kwargs):
            captured['resident'] = kwargs
            return Resident()

    monkeypatch.setattr(fit.prepared, 'read', read)
    monkeypatch.setattr(fit.prepared, 'digest', lambda path: 'sha')
    monkeypatch.setattr(fit.prepared, 'npz', lambda path:
        z if Path(path).name == 'point_observations.npz' else initial)
    monkeypatch.setattr(fit.prepared, 'validate_snapshot', lambda *args: None)
    monkeypatch.setattr(fit, 'SceneParameterModel', lambda *args, **kwargs:
        SimpleNamespace(initial_state=state, dependencies=lambda path: {'print/ad4'}))
    monkeypatch.setattr(fit, 'ParameterState', lambda *args: state)
    monkeypatch.setattr(fit, 'NativeCore', lambda *args: SimpleNamespace(_lib=library))
    monkeypatch.setattr(fit, 'ShotMotionTimeline', Timeline)
    monkeypatch.setattr(fit, 'NativeSceneGPU', Native)
    monkeypatch.setattr(fit, 'continue_exposure_halos', lambda controls, *args: controls)
    monkeypatch.setattr(fit, '_load_joint_camera_profile', lambda path, rig:
        (rig.copy(), PROFILE_INDICES, dict(profile={}, profile_path=str(path),
            profile_file_sha256='sha')))
    monkeypatch.setattr(fit.single, 'mount_configuration', lambda *args:
        (np.full(43, -np.inf), np.full(43, np.inf), {}, Path('ranges.json'), {}))
    monkeypatch.setenv('RINKSPACE_MOTION_DIRECTION_STORAGE', 'sparse')
    args = SimpleNamespace(phase='joint', resume=None, run_root=tmp_path,
        output=tmp_path/'output', evaluate_only=False)

    def run(**kwargs):
        for name, value in kwargs.items():
            setattr(args, name, value)
        with pytest.raises(Compiled):
            fit.solve(args)
        return json.loads((args.output/'activity.json').read_text(encoding='utf8'))

    return SimpleNamespace(run=run, args=args, captured=captured,
        initial=initial, state=state, z=z, library=library, manifest=manifest)


def test_default_compilation_keeps_all_existing_rows_and_directions(compile_run):
    run = compile_run
    activity = run.run()
    assert activity['initialization_only'] is False
    assert 'active_selection' not in activity
    assert activity['motion_columns'] == 9
    assert activity['scene_columns'] == 153
    assert activity['retained_columns'] == 240
    assert activity['active_camera_chart_indices_per_rig'] == [list(PROFILE_INDICES)]*3
    assert activity['selected_shot_ids'] == [f's{i+1:02}' for i in range(9)]
    assert activity['rows'] == activity['all_rows'] == 18
    np.testing.assert_array_equal(run.captured['resident']['observed'], run.z['uv'])
    np.testing.assert_array_equal(run.captured['resident']['controls'], run.initial['controls'])
    expected, _ = build_multicamera_directions(run.initial['rig'], [PROFILE_INDICES]*3,
        np.r_[np.array([.001]*6+[.001]*3+[.00025]*3+[.00025]*3+
            [1., 1., .001, 1., 1., .03, .03, .1, .1, .003, .003, .001]), .05, .1, .003, .003])
    np.testing.assert_array_equal(run.captured['camera']['directions'][:, 9:87], expected)
    assert not run.captured['camera']['directions'][:, :9].any()
    assert not run.captured['camera']['directions'][:, 87:].any()


@pytest.mark.parametrize('storage', ['sparse', 'dense'])
def test_explicit_selection_preserves_nine_shot_objective_and_freezes_other_columns(compile_run, monkeypatch, storage):
    run = compile_run
    monkeypatch.setenv('RINKSPACE_MOTION_DIRECTION_STORAGE', storage)
    run.library.rs_gpu_ice_actual_chart_version = 1
    activity = run.run(freeze_motion=True, camera_parameter_names=RADIAL_NAMES,
        scene_block_ids=['print/ad4'])
    assert activity['initialization_only'] is True
    assert activity['motion_columns'] == activity['motion_direction_values'] == 0
    assert activity['unique_camera_columns'] == 18
    assert activity['scene_columns'] == 2
    assert activity['retained_columns'] == 20
    assert activity['ice_chart']['version'] == 'legacy'
    selection = activity['active_selection']
    assert selection['all_prepared_rows_retained']
    assert selection['all_source_rows_preserved']
    assert selection['raw_pixel_objective_unchanged']
    assert selection['source_observations_or_weights_changed'] is False
    assert activity['rows'] == activity['all_rows'] == 18
    np.testing.assert_array_equal(run.captured['resident']['observed'], run.z['uv'])
    sd = run.captured['resident']['scene_directions']
    assert set(np.flatnonzero(np.any(sd != 0, axis=1))) == {153, 155}
    assert set(selection['frozen_scene_columns']) == set(range(159))-{153, 155}
    for camera, frozen_names in enumerate(selection['frozen_camera_scalar_names_per_rig']):
        assert set(frozen_names) == set(_SCALAR_STORAGE_INDEX)-set(RADIAL_NAMES)
        rd = run.captured['camera']['directions'][31*camera:31*(camera+1)]
        assert set(np.flatnonzero(np.any(rd != 0, axis=1))) == {CAMERA_NAMES.index(n) for n in RADIAL_NAMES}


def test_explicit_empty_camera_and_scene_lists_freeze_them(compile_run):
    activity = compile_run.run(camera_parameter_names=[], scene_block_ids=[])
    assert activity['initialization_only']
    assert activity['retained_columns'] == activity['motion_columns'] == 9
    assert activity['scene_columns'] == activity['unique_camera_columns'] == 0
    assert not compile_run.captured['resident']['scene_directions'].any()
    assert not compile_run.captured['camera']['directions'].any()


def test_all_frozen_rejects_zero_global_columns_before_native_binding(compile_run):
    run = compile_run
    run.args.freeze_motion = True
    run.args.camera_parameter_names = []
    run.args.scene_block_ids = []
    with pytest.raises(ValueError, match='no global parameter columns'):
        fit.solve(run.args)
    assert 'resident' not in run.captured


@pytest.mark.parametrize('names,match', [(['missing'], 'Unknown'), (['cx'], 'no active')])
def test_camera_selection_rejects_unknown_and_inactive_profile_names(names, match):
    with pytest.raises(ValueError, match=match):
        fit.select_camera_parameter_names([PROFILE_INDICES]*3, names)


def test_pose_phase_cannot_enable_a_name_inactive_in_current_profile(compile_run, monkeypatch):
    run = compile_run
    monkeypatch.setattr(fit, '_load_joint_camera_profile', lambda path, rig:
        (rig.copy(), (20,), dict(profile={}, profile_path=str(path),
            profile_file_sha256='sha')))
    run.args.phase = 'camera_pose'
    run.args.camera_parameter_names = ['R0_x']
    with pytest.raises(ValueError, match='no active profile/phase directions'):
        fit.solve(run.args)
    assert 'resident' not in run.captured


@pytest.mark.parametrize('ids,match', [(['missing'], 'Unknown'),
    (['print/unobserved'], 'no active dependency'), (['print/inactive'], 'no active dependency')])
def test_scene_selection_rejects_unknown_and_non_dependent_or_inactive_blocks(ids, match):
    state = State()
    with pytest.raises(ValueError, match=match):
        fit.select_scene_block_ids(state, state.active_columns(['print/ad4']), 'joint', ids)


def test_selection_never_reorders_existing_directions_or_opens_new_ones():
    indices = [PROFILE_INDICES, ()]
    assert fit.select_camera_parameter_names(indices, None) is indices
    assert fit.select_camera_parameter_names(indices, ['wide_k2', 'k10']) == [(20, 28), ()]
    with pytest.raises(ValueError, match='no active'):
        fit.select_camera_parameter_names([()], ['k10'])
    state = State()
    active = state.active_columns(['print/ad4', ICE_IDS[0]])[::-1]
    assert fit.select_scene_block_ids(state, active, 'joint', None) is active
    np.testing.assert_array_equal(fit.select_scene_block_ids(state, active, 'joint',
        ['print/ad4', ICE_IDS[0]]), active)
    with pytest.raises(ValueError, match='phase=joint'):
        fit.select_scene_block_ids(state, active, 'motion', [])


def test_shared_camera_selection_cannot_open_only_one_linked_camera():
    rigs = np.zeros((2, 43))
    rigs[:, 24] = [10000., 5000.]
    rigs[:, 38] = np.log([8000., 4000.])
    coupling = dict(groups=[dict(id='radial', camera_indices=[0, 1],
        shared_names=RADIAL_NAMES, delivery_magnifications=[2., 1.],
        reference_focal_base_pixels=5000., wide_boundary_base_log_f=np.log(4000.))])
    half = fit.select_camera_parameter_names([PROFILE_INDICES, ()], RADIAL_NAMES)
    with pytest.raises(ValueError, match='all linked cameras'):
        build_multicamera_directions(rigs, half, np.ones(31), coupling)
    both = fit.select_camera_parameter_names([PROFILE_INDICES]*2, RADIAL_NAMES)
    directions, metadata = build_multicamera_directions(rigs, both, np.ones(31), coupling)
    assert directions.shape == (62, 6)
    assert metadata['unique_camera_columns'] == 6
    np.testing.assert_array_equal(directions[:31], directions[31:])


def test_explicit_non_ice_selection_keeps_actual_chart_request_error(compile_run):
    run = compile_run
    run.library.rs_gpu_ice_actual_chart_version = 1
    run.args.ice_chart = 'actual'
    run.args.scene_block_ids = ['print/ad4']
    with pytest.raises(ValueError, match='Actual ice chart is unavailable'):
        fit.solve(run.args)


def test_actual_chart_witness_exports_are_excluded_from_freeze_check(compile_run):
    run = compile_run
    run.library.rs_gpu_ice_actual_chart_version = 1
    activity = run.run(scene_block_ids=ICE_IDS)
    assert activity['ice_chart']['version'] == 'actual_x_log_radius_v1'
    frozen = activity['active_selection']['frozen_scene_columns']
    assert not set(NOMINAL_WITNESS_INDICES) & set(frozen)
    assert set(range(152, 159)) <= set(frozen)
    current = {key: value.copy() for key, value in run.initial.items()}
    current['scene_values'][NOMINAL_WITNESS_INDICES] += .001
    fit.validate_active_selection_checkpoint(current, run.initial, frozen_scene_columns=frozen)


def test_checkpoint_enforces_motion_bytes_and_unselected_scene_and_scalars(compile_run):
    initial = compile_run.initial
    frozen_scalars = [[name for name in _SCALAR_STORAGE_INDEX if name not in RADIAL_NAMES]]*3
    options = dict(freeze_motion=True, frozen_scene_columns=[0, 1, 2],
        frozen_camera_scalars=frozen_scalars)
    current = {key: value.copy() for key, value in initial.items()}
    current['scene_values'][153] += .01
    current['rig'][0, _SCALAR_STORAGE_INDEX['k10']] += .01
    fit.validate_active_selection_checkpoint(current, initial, **options)
    current['controls'][0, 0] = -0.
    with pytest.raises(RuntimeError, match='Frozen motion controls'):
        fit.validate_active_selection_checkpoint(current, initial, **options)
    current['controls'] = initial['controls'].copy()
    current['scene_values'][1] += .01
    with pytest.raises(RuntimeError, match='Unselected scene'):
        fit.validate_active_selection_checkpoint(current, initial, **options)
    current['scene_values'][1] = initial['scene_values'][1]
    current['rig'][0, _SCALAR_STORAGE_INDEX['cx']] += 1.
    with pytest.raises(RuntimeError, match='Unselected camera scalar'):
        fit.validate_active_selection_checkpoint(current, initial, **options)


def test_structural_per_camera_subfamilies_keep_full_joint_and_preserve_zeroed_terms(compile_run):
    run=compile_run
    all_names=[CAMERA_NAMES[k] for k in PROFILE_INDICES]
    reduced=[n for n in all_names if n!='wide_k2']
    run.manifest['camera_parameterization']=dict(profile_path='profile.json',profile_sha256='sha',active_names_per_rig=[reduced,reduced,all_names])
    activity=run.run()
    assert activity['initialization_only'] is False
    assert activity['unique_camera_columns']==3*len(PROFILE_INDICES)-2
    assert activity['scene_columns']==153 and activity['motion_columns']==9
    assert activity['rows']==activity['all_rows']==18
    rd=run.captured['camera']['directions'];axis=CAMERA_NAMES.index('wide_k2')
    np.testing.assert_array_equal(rd[[axis,31+axis]],0.)
    assert np.any(rd[62+axis])
    frozen=activity['structurally_frozen_camera_scalar_names_per_rig']
    assert all('wide_k2' in x for x in frozen[:2]) and 'wide_k2' not in frozen[2]
    changed={k:v.copy() for k,v in run.initial.items()};changed['rig'][0,_SCALAR_STORAGE_INDEX['wide_k2']]=.01
    with pytest.raises(RuntimeError,match='Unselected camera scalar'):
        fit.validate_active_selection_checkpoint(changed,run.initial,frozen_camera_scalars=frozen)
