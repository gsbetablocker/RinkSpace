"""Regressions for the source-to-solver paths exercised by real programme jobs."""
from copy import deepcopy
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _candidate(name, uv=(12., 20.), group=None):
    from rinkspace.session.source_observations import SourceCandidate
    return SourceCandidate(name, 'shot', 100, {}, list(uv), 'line',
        actual_uv=list(uv), normal=[0., 1.], native_path_id='native/line',
        identity='confirmed', localization='measured', origin='automatic',
        source_group_id=group, evidence=dict(station_initial=.5))


@pytest.mark.parametrize('size,uv,accepted', [((64, 48), (70., 20.), False),
    ((3840, 2160), (2500., 1200.), True)])
def test_export_selection_uses_declared_source_dimensions(size, uv, accepted):
    from rinkspace.session.source_observations import select_source_observations
    rows, _, omitted = select_source_observations([_candidate('a', uv)],
        {'native/line': dict(kind='finite_curve')}, None, 'camera_navigation',
        image_size=size)
    assert bool(rows) is accepted
    if not accepted:
        assert omitted[0]['reason'] == 'outside_original_image'


def _job(tmp_path, *, admission=None):
    video = tmp_path/'source.bin'; video.write_bytes(b'original source fixture')
    source = dict(id='source', sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
        registered_path=str(video), width=64, height=48)
    clock = dict(time_base_num=1, time_base_den=1000, native_pts_step=20)
    scope = dict(source=source, source_clock=clock, event_id='event',
        required_shot_ids=['shot'], rig_bindings=dict(ids={'main':'main_installation'}),
        shots=[dict(id='shot', rig_id='main', processing=dict(
            first_pts=100, stop_pts_exclusive=160, native_frame_count=3))])
    config = dict(final_model='unused', flow_weights='unused', device='cpu')
    if admission is not None: config['enable_candidate_admission'] = admission
    refs = {}
    for name, value in [('scope_ref', scope), ('scene_ref', {}), ('source_routes_ref', {}),
            ('navigation_ref', {}), ('vision_config_ref', config), ('source_reidentification_ref', {})]:
        target = tmp_path/(name+'.json'); target.write_text(json.dumps(value))
        refs[name] = dict(root='repo', path=target.name,
            sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    job = dict(roots=dict(repo=str(tmp_path), data=str(tmp_path)), source=source,
        source_clock=clock, selected_shot_ids=['shot'], sequences=[dict(shot_id='shot',
            first_pts=100, stop_pts_exclusive=160, native_frame_count=3)], **refs)
    path = tmp_path/'job.json'; path.write_text(json.dumps(job))
    return path, job, scope


def test_verified_programme_uses_its_declared_count_and_native_clock(tmp_path):
    from rinkspace.session.source_programme import _verified_job
    path, job, _ = _job(tmp_path)
    assert _verified_job(path)[1] == job


@pytest.mark.parametrize('damage', ['count', 'clock', 'duplicate', 'out_of_scope'])
def test_verified_programme_rejects_inconsistent_scope(tmp_path, damage):
    from rinkspace.session.source_programme import _verified_job
    path, job, _ = _job(tmp_path)
    if damage == 'count': job['sequences'][0]['native_frame_count'] = 4
    elif damage == 'clock': job['source_clock']['time_base_den'] = 12800
    elif damage == 'duplicate': job['sequences'].append(deepcopy(job['sequences'][0]))
    else: job['sequences'][0]['stop_pts_exclusive'] = 180
    path.write_text(json.dumps(job))
    with pytest.raises(ValueError): _verified_job(path)


@pytest.mark.parametrize('admission', [None, False, True])
def test_programme_does_not_override_candidate_admission_configuration(tmp_path, monkeypatch, admission):
    from rinkspace.session import source_programme
    from rinkspace import vision, vision_native
    path, job, _ = _job(tmp_path, admission=admission)
    monkeypatch.setattr(source_programme, '_verified_job', lambda _: (path, job))
    lib = SimpleNamespace(**{name: object() for name in
        ('rs_g1_create','rs_g3_attach','rs_g4_solve','rs_g2_sparse_info')})
    monkeypatch.setattr(vision_native, 'NativeSourceGeometry',
        lambda *a: SimpleNamespace(core=SimpleNamespace(_lib=lib)))
    seen = {}
    class ReaderReached(Exception): pass
    def reader(config_path, **kwargs):
        seen.update(json.loads(Path(config_path).read_text()))
        raise ReaderReached()
    monkeypatch.setattr(vision, 'SourceRinkReader', reader)
    with pytest.raises(ReaderReached): source_programme.run_programme(path, tmp_path/'out')
    assert seen.get('enable_candidate_admission') is admission


def test_source_reader_does_not_overwrite_previous_frame_evidence(tmp_path):
    from rinkspace.session.source_read import run_source_read
    path, job, _ = _job(tmp_path)
    request = tmp_path/'request.json'
    request.write_text(json.dumps(dict(programme_scope=str(tmp_path/job['scope_ref']['path']),
        vision_config='unused', sequences=job['sequences'])))
    out = tmp_path/'out'; out.mkdir()
    evidence = out/'shot_100.json'; evidence.write_bytes(b'previous automatic source evidence')
    with pytest.raises(FileExistsError): run_source_read(request, out)
    assert evidence.read_bytes() == b'previous automatic source evidence'


@pytest.mark.parametrize('component', ['ledger', 'source_index'])
def test_authenticated_job_revision_can_resume_again_without_its_ancestor(tmp_path, component):
    from rinkspace.session.source_programme import ProgrammeLedger
    from rinkspace.session.source_inputs import SourceEvidenceIndex
    path, job, _ = _job(tmp_path)
    routes=tmp_path/job['source_routes_ref']['path']
    routes.write_text(json.dumps(dict(schema='rinkspace.handoff.source_routes.v1',source_groups={})))
    job['source_routes_ref']['sha256']=hashlib.sha256(routes.read_bytes()).hexdigest()
    path.write_text(json.dumps(job))
    second=tmp_path/'second.json'; second.write_text(json.dumps(dict(job, intent='revised implementation')))
    third=tmp_path/'third.json'; third.write_text(json.dumps(dict(job, intent='next revised implementation')))
    def opened(current, previous=None):
        if component=='ledger':
            return ProgrammeLedger(tmp_path/'ledger',current,previous_job=previous)
        return SourceEvidenceIndex(current,tmp_path/'source.sqlite',reuse_from_job=previous)
    initial=opened(path)
    key='job_sha256' if component=='ledger' else 'input_identity'
    original=initial.db.execute('select value from meta where key=?',(key,)).fetchone()[0]
    initial.db.close()
    migrated=opened(second,path); migrated.db.close()
    resumed=opened(second); resumed.db.close()
    next_revision=opened(third,second)
    # Preserve the initial source/job identity as immutable history.
    assert next_revision.db.execute('select value from meta where key=?',(key,)).fetchone()[0]==original
    next_revision.db.close()
    changed=dict(job, source=dict(job['source'],sha256='f'*64))
    invalid=tmp_path/'invalid.json'; invalid.write_text(json.dumps(changed))
    with pytest.raises(ValueError): opened(invalid,third)


def _programme_module(name):
    root = Path(__file__).resolve().parents[1]
    path = root/'events/ankara_2026/programmes/okada_mei_sp'/f'{name}.py'
    spec = importlib.util.spec_from_file_location('audit_'+name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('resume', [False, True])
def test_station_initializer_round_trips_and_keeps_snapshot_time_grid(tmp_path, monkeypatch, resume):
    module = _programme_module('initialize_native_s01')
    prep = module.prep
    monkeypatch.setattr(prep, 'RUN', tmp_path)
    monkeypatch.setattr(prep, 'CAMERA', tmp_path/'camera.json')
    camera = dict(controls_ref='controls.npy', rig=[0.]*38, source_id='source',
        reference_pts=100, time_base_num=1, time_base_den=1000, image_width=64, image_height=48,
        first_control_time=-1., control_dt=1.)
    prep.write(prep.CAMERA, camera); np.save(tmp_path/'controls.npy', np.zeros((4,3)))
    monkeypatch.setattr(prep, 'SCENE', tmp_path/'scene.json')
    prep.write(prep.SCENE, dict(scene_id='fixture', board=[1., .5]))
    model=module.SceneParameterModel(prep.read(prep.SCENE),prep.SCENE,optimize_scene=True)
    mapping=dict(path_ids=['ice/path'],scene_revision=model.initial_state.layout['scene_content_revision'],
        camera_sha256=prep.digest(prep.CAMERA),control_sha256=prep.digest(tmp_path/'controls.npy'))
    prep.write(tmp_path/'mapping.json', mapping)
    points = dict(source_rows=np.array([1,2]), path_indices=np.array([0,0]),
        stations=np.array([.25,.1]), pts=np.array([100,100]), uv=np.array([[25.,20.],[40.,20.]]),
        kind=np.array([0,1]), observation_ids=np.array(['fixed','curve']))
    np.savez(tmp_path/'point_observations.npz', **points)
    seen = []
    class Resident:
        def __init__(self, **kwargs): self.kwargs = kwargs
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def evaluate(self, backend):
            k = self.kwargs; values=np.zeros((len(k['stations']),11))
            values[:,3]=np.asarray(k['stations'])*100.; values[:,4]=20.
            values[:,9:11]=values[:,3:5]-k['observed']
            return dict(values=values, status=np.zeros(len(values),np.int32))
    class Geometry:
        paths={'ice/path':dict(kind=2)}; scene_revision='current'
        def __init__(self, *args, **kwargs): pass
        def resident(self, **kwargs):
            seen.append(dict(kwargs['camera'])); return Resident(**kwargs)
    monkeypatch.setattr(module, 'NativeCore', lambda _: object())
    monkeypatch.setattr(module, 'NativeSceneGPU', Geometry)
    state_file = None
    if resume:
        state_file=tmp_path/'snapshot.npz'
        snapshot=dict(rig=np.zeros(38), controls=np.zeros((4,3)), scene_values=model.initial_state.values,
            stations=points['stations'], source_rows=points['source_rows'],
            point_input_sha256=np.array(prep.digest(tmp_path/'point_observations.npz')),
            motion_grid_first_control_time=np.array(-2.), motion_grid_control_dt=np.array(.5))
        prep.bind_snapshot_identity(snapshot,model.initial_state.layout,camera,mapping)
        np.savez(state_file,**snapshot)
    module.initialize(state_file)
    loaded, _, stations=prep.load_pixel_input(tmp_path)
    np.testing.assert_array_equal(loaded['uv'], points['uv'])
    np.testing.assert_allclose(stations, [.25,.4])
    if resume:
        assert all(c['first_control_time']==-2. and c['control_dt']==.5 for c in seen)



def _snapshot_fixture(tmp_path, monkeypatch):
    module=_programme_module('initialize_native_s01');prep=module.prep
    monkeypatch.setattr(prep,'RUN',tmp_path);monkeypatch.setattr(prep,'CAMERA',tmp_path/'camera.json')
    monkeypatch.setattr(prep,'SCENE',tmp_path/'scene.json')
    camera=dict(controls_ref='controls.npy',rig=[0.]*38,source_id='source',reference_pts=100,
        time_base_num=1,time_base_den=1000,image_width=64,image_height=48,
        first_control_time=-1.,control_dt=1.)
    prep.write(prep.CAMERA,camera);np.save(tmp_path/'controls.npy',np.zeros((4,3)))
    prep.write(prep.SCENE,dict(scene_id='original',board=[1.,.5]))
    model=module.SceneParameterModel(prep.read(prep.SCENE),prep.SCENE,optimize_scene=True)
    mapping=dict(path_ids=['ice/path','ice/other'],scene_revision=model.initial_state.layout['scene_content_revision'],
        camera_sha256=prep.digest(prep.CAMERA),control_sha256=prep.digest(tmp_path/'controls.npy'))
    prep.write(tmp_path/'mapping.json',mapping)
    points=dict(source_rows=np.array([10,11]),path_indices=np.array([0,0]),stations=np.array([.25,.4]),
        pts=np.array([100,100]),uv=np.array([[25.,20.],[40.,20.]]),kind=np.array([0,1]),observation_ids=np.array(['fixed','curve']))
    np.savez(tmp_path/'point_observations.npz',**points)
    state=dict(rig=np.zeros(38),controls=np.zeros((4,3)),scene_values=model.initial_state.values,
        source_rows=points['source_rows'],stations=points['stations'].copy(),
        point_input_sha256=np.array(prep.digest(tmp_path/'point_observations.npz')))
    prep.bind_snapshot_identity(state,model.initial_state.layout,camera,mapping)
    return module,model,camera,mapping,points,state


@pytest.mark.parametrize('damage',['scene','source_clock','path_registry','grid','legacy'])
def test_native_initializer_rejects_same_size_stale_snapshot_before_native_use(tmp_path,monkeypatch,damage):
    module,model,camera,mapping,points,state=_snapshot_fixture(tmp_path,monkeypatch);prep=module.prep
    if damage=='scene':
        # Same number of doubles but different block identities and scene semantics.
        prep.write(prep.SCENE,dict(scene_id='different',board=[1.,.5]))
        new=module.SceneParameterModel(prep.read(prep.SCENE),prep.SCENE,optimize_scene=True)
        assert new.initial_state.values.shape==state['scene_values'].shape
        mapping['scene_revision']=new.initial_state.layout['scene_content_revision']
    elif damage=='source_clock':
        camera['time_base_den']=2000;prep.write(prep.CAMERA,camera)
        mapping['camera_sha256']=prep.digest(prep.CAMERA)
    elif damage=='path_registry': mapping['path_ids'].reverse()
    elif damage=='grid': state['motion_grid_control_dt']=np.array(.5)
    else: state.pop(prep.SNAPSHOT_IDENTITY_FIELD)
    prep.write(tmp_path/'mapping.json',mapping)
    path=tmp_path/'accepted.npz';np.savez(path,**state);original=path.read_bytes();native_calls=[]
    monkeypatch.setattr(module,'NativeCore',lambda _:native_calls.append('core'))
    class Resident:
        def __init__(self,**kwargs):self.kwargs=kwargs
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def evaluate(self,_):
            values=np.zeros((len(self.kwargs['stations']),11));values[:,3]=np.asarray(self.kwargs['stations'])*100.;values[:,4]=20.
            return dict(values=values,status=np.zeros(len(values),np.int32))
    class Geometry:
        paths={name:dict(kind=2) for name in mapping['path_ids']};scene_revision='fixture'
        def __init__(self,*a,**kw):native_calls.append('geometry')
        def resident(self,**kwargs):return Resident(**kwargs)
    monkeypatch.setattr(module,'NativeSceneGPU',Geometry)
    match='需要与当时模型/映射核对迁移' if damage=='legacy' else 'dependencies changed'
    with pytest.raises(ValueError,match=match):module.initialize(path)
    assert not native_calls
    assert path.read_bytes()==original
    assert not (tmp_path/'initialized_points.npz').exists()


def test_native_snapshot_dependency_binding_allows_corrected_library_replay(tmp_path,monkeypatch):
    module,model,camera,mapping,points,state=_snapshot_fixture(tmp_path,monkeypatch);prep=module.prep
    library=tmp_path/'native.so';library.write_bytes(b'old native implementation')
    prep.bind_snapshot_identity(state,model.initial_state.layout,camera,mapping,native_library=library)
    original_provenance=str(state['native_library_provenance_sha256'])
    library.write_bytes(b'corrected native implementation')
    validated=prep.validate_snapshot_identity(state,model.initial_state.layout,camera,mapping,points=points,
        point_input_sha256=prep.digest(tmp_path/'point_observations.npz'))
    assert validated==camera
    assert original_provenance!=prep.digest(library)


@pytest.mark.parametrize('damage',['source_rows','fixed_station'])
def test_native_snapshot_dependency_binding_preserves_point_identity(tmp_path,monkeypatch,damage):
    module,model,camera,mapping,points,state=_snapshot_fixture(tmp_path,monkeypatch);prep=module.prep
    if damage=='source_rows':state['source_rows']=state['source_rows']+1
    else:state['stations']=state['stations']+.1
    with pytest.raises(ValueError):prep.validate_snapshot_identity(state,model.initial_state.layout,camera,mapping,points=points)


@pytest.mark.parametrize('hw,uv,accepted',[([48,64],(70.,20.),False),([2160,3840],(2500.,1200.),True)])
def test_source_query_bounds_use_decoded_native_image_size(hw,uv,accepted):
    from rinkspace.session.source_observations import source_queries
    rows=[_candidate(str(i),(uv[0]+i,uv[1]),group='line') for i in range(2)]
    binding=dict(shot_id='shot',source_pts=100,native_hw=hw)
    result=source_queries(rows,binding,SimpleNamespace(bind=lambda values:(values,{})))
    assert bool(result) is accepted
    if accepted:np.testing.assert_array_equal(result[0]['points_uv'],[row.actual_uv for row in rows])
