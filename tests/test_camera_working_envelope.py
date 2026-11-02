"""Explicit mount-envelope binding without loading a DLL or starting a fit."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]/'events/ankara_2026/programmes/okada_mei_sp'
sys.path.insert(0, str(HERE))
import prepare_native_multicamera as prepared
import rebind_camera_coupling as rebind


@pytest.fixture
def envelope(tmp_path):
    default = prepared.single.mount_configuration(1., np.zeros(43))
    path = tmp_path/'envelope.json'
    fact = deepcopy(default[4])
    fact['status'] = 'CPU binding test; existing numerical ranges unchanged'
    prepared.write(path, fact)
    return path


def binding(path):
    return dict(path=str(path.resolve()), sha256=prepared.digest(path))


def test_absent_binding_keeps_the_existing_default():
    assert prepared.camera_working_envelope_path({}) is None
    implicit = prepared.single.mount_configuration(1., np.zeros(43))
    explicit_none = prepared.single.mount_configuration(1., np.zeros(43), None)
    assert implicit[3].name == 'z190_mount_working_ranges_20261026.json'
    for a, b in zip(implicit[:2], explicit_none[:2]):
        np.testing.assert_array_equal(a, b)
    assert implicit[2:] == explicit_none[2:]
    ranges = implicit[4]['working_ranges']
    np.testing.assert_array_equal(implicit[0][18:21],
        ranges['complete_tilt_axis_to_projection_centre_camera_m']['lower'])


def test_explicit_binding_reads_the_verified_file(envelope):
    selected = prepared.camera_working_envelope_path(
        dict(camera_working_envelope=binding(envelope)))
    bounds = prepared.single.mount_configuration(1., np.zeros(43), selected)
    assert bounds[3] == envelope
    assert bounds[4]['status'].startswith('CPU binding test')


def test_modified_bound_content_is_rejected(envelope):
    manifest = dict(camera_working_envelope=binding(envelope))
    envelope.write_bytes(envelope.read_bytes()+b'\n')
    with pytest.raises(ValueError, match='checksum differs'):
        prepared.camera_working_envelope_path(manifest)


@pytest.mark.parametrize('value', [None, {}, dict(path='relative.json', sha256='0'*64),
    dict(path='D:/file.json', sha256='not-a-sha256')])
def test_malformed_binding_is_rejected(value):
    with pytest.raises(ValueError, match='[Cc]amera working envelope'):
        prepared.camera_working_envelope_path(dict(camera_working_envelope=value))


def test_missing_bound_file_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        prepared.camera_working_envelope_path(dict(camera_working_envelope=
            dict(path=str(tmp_path/'missing.json'), sha256='0'*64)))


@pytest.fixture
def parent(tmp_path, monkeypatch):
    root = tmp_path/'parent'
    root.mkdir()
    layout = dict(scene_content_revision='scene', layout_revision='layout', size=1)
    mapping = dict(path_ids=['fixed'])
    scene = root/'scene.json'
    prepared.write(scene, {})
    profile = HERE/'camera_experiments/joint/square_center_pp_joint.json'
    points = dict(pts=np.array([256], dtype=np.int64), kind=np.array([0], np.int32),
        uv=np.array([[1., 2.]]), source_rows=np.array([3], dtype=np.int64),
        observation_ids=np.array(['source']), stations=np.array([-0.]),
        path_indices=np.array([0], np.int32))
    np.savez(root/'point_observations.npz', **points)
    manifest = dict(shots=[dict(id='s01', control_stop=2)], cameras=[dict(id='main')],
        control_dt=.1, first_control_time=0., max_readout=0.,
        point_input_sha256=prepared.digest(root/'point_observations.npz'),
        scene_path=str(scene), native_library_sha256='provenance',
        camera_parameterization=dict(profile_path=str(profile),
            profile_sha256=prepared.digest(profile)))
    rig = np.zeros((1, 43))
    rig[0, 26:28] = [959.5, 539.5]
    state = dict(rig=rig, controls=np.array([[-0., 0., 1.], [0., 0., 1.]]),
        stations=points['stations'], source_rows=points['source_rows'],
        scene_values=np.array([-0.]))
    prepared.bind_snapshot(state, layout, mapping, manifest)
    np.savez(root/'state.npz', **state)
    prepared.write(root/'joint_manifest.json', manifest)
    prepared.write(root/'mapping.json', mapping)
    model = SimpleNamespace(initial_state=SimpleNamespace(layout=layout))
    monkeypatch.setattr(rebind, 'SceneParameterModel', lambda *args, **kwargs: model)
    coupling = root/'coupling.json'
    prepared.write(coupling, dict(schema='rinkspace.camera_coupling.v1', id='test',
        hardware_calibrated=False, basis='CPU binding test', groups=[]))
    return SimpleNamespace(root=root, layout=layout, manifest=manifest, mapping=mapping,
        points=points, state=state, coupling=coupling)


def test_snapshot_identity_changes_only_when_envelope_is_explicit(parent, envelope):
    p = parent
    previous = prepared.snapshot_identity(p.layout, p.mapping, p.manifest)
    assert 'camera_working_envelope' not in previous
    revised = deepcopy(p.manifest)
    revised['camera_working_envelope'] = binding(envelope)
    current = prepared.snapshot_identity(p.layout, p.mapping, revised)
    assert current.pop('camera_working_envelope') == binding(envelope)
    assert current == previous
    with pytest.raises(ValueError, match='dependencies differ'):
        prepared.validate_snapshot(p.state, p.layout, p.mapping, revised, p.points)


@pytest.mark.parametrize('explicit', [False, True])
def test_rebind_preserves_every_numeric_and_source_byte(parent, envelope, tmp_path, explicit):
    p = parent
    output = tmp_path/'rebound'
    rebind.main(SimpleNamespace(prepared=p.root, state=p.root/'state.npz',
        coupling=p.coupling, profile=None, working_envelope=envelope if explicit else None,
        output=output))
    result = prepared.read(output/'joint_manifest.json')
    if explicit:
        assert result['camera_working_envelope'] == binding(envelope)
    else:
        assert 'camera_working_envelope' not in result
    saved = prepared.npz(output/'initial_pixels.npz')
    for key, array in p.state.items():
        if key == 'joint_numeric_dependencies_json':
            continue
        assert saved[key].dtype == array.dtype
        assert saved[key].shape == array.shape
        assert saved[key].tobytes() == array.tobytes()
    assert (output/'point_observations.npz').read_bytes() == (p.root/'point_observations.npz').read_bytes()
    prepared.validate_snapshot(saved, p.layout, p.mapping, result, p.points)


@pytest.mark.parametrize('fact', [[], {}, dict(schema='rinkspace.assistant_camera_working_envelope.v1')])
def test_rebind_rejects_invalid_envelope_format_before_writing(parent, tmp_path, fact):
    path = tmp_path/'invalid.json'
    prepared.write(path, fact)
    output = tmp_path/'rejected'
    with pytest.raises(ValueError, match='camera working envelope format'):
        rebind.main(SimpleNamespace(prepared=parent.root, state=parent.root/'state.npz',
            coupling=parent.coupling, profile=None, working_envelope=path, output=output))
    assert not output.exists()
