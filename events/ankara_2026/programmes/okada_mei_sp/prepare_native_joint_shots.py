"""Import S04 observations alongside the current accepted S01 candidate.

Data supplies identity, PTS and UV only. All numeric scene/camera values come
from the selected current native state. S04's existing native trajectory is
only a starting guess, not a new calibration authority.
"""
from pathlib import Path
import json,gzip,hashlib,io,importlib.util,collections
import numpy as np
from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import SceneParameterModel,ParameterState
from rinkspace.domain.motion_chart import UniformMotionChart
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline,refine_uniform_controls
from rinkspace.session.gpu_geometry import NativeSceneGPU

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('s01prep',HERE/'prepare_native_s01.py');prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
RUN=Path(r'D:\CodexData\RinkSpace_Data\runs\native_joint_s01_s04_20261027')
SOURCE=RUN/'source_s04'
BASE=Path(r'D:\CodexData\RinkSpace_Data\runs\native_joint_s01_20261026\sensor_projection_fix\current')
LIBRARY=prep.LIBRARY
FIELDS={'source_rows','observation_ids','uv','pts','kind','stations','path_indices'}

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def npz(p):
    with np.load(io.BytesIO(Path(p).read_bytes()),allow_pickle=False) as f:return dict(f)
def records(p):
    with gzip.open(p,'rt',encoding='utf-8-sig') as f:
        for line in f:yield json.loads(line)

def joint_snapshot_identity(layout,mapping,manifest):
    # Bind the physical meaning and each independent shot clock. A corrected
    # library is provenance, not a reason to reinterpret saved coordinates.
    return dict(schema='rinkspace.joint_numeric_state_dependencies.v1',
        scene_revision=layout['scene_content_revision'],layout_revision=layout['layout_revision'],
        paths_sha256=hashlib.sha256(json.dumps(mapping['path_ids'],separators=(',',':')).encode()).hexdigest(),
        point_input_sha256=manifest['point_input_sha256'],shots=manifest['shots'],
        first_control_time=manifest['first_control_time'],control_dt=manifest['control_dt'],
        max_readout=manifest['max_readout'],
        camera_definition_sha256=digest(manifest['shared_camera_definition']))


def bind_joint_snapshot(snapshot,layout,mapping,manifest):
    snapshot['joint_numeric_dependencies_json']=np.array(json.dumps(joint_snapshot_identity(layout,mapping,manifest),sort_keys=True,separators=(',',':')))
    snapshot['native_library_provenance_sha256']=np.array(manifest['native_library_sha256'])
    return snapshot


def validate_joint_snapshot(snapshot,layout,mapping,manifest,points):
    key='joint_numeric_dependencies_json'
    if key not in snapshot or json.loads(str(snapshot[key]))!=joint_snapshot_identity(layout,mapping,manifest):
        raise ValueError('Joint snapshot dependencies do not match the selected input, native geometry and independent shot clocks')
    if (str(snapshot['point_input_sha256'])!=manifest['point_input_sha256'] or
            not np.array_equal(snapshot['source_rows'],points['source_rows'])):
        raise ValueError('Joint snapshot point identity mismatch')
    sizes={'controls':(manifest['shots'][-1]['control_stop'],3),'stations':points['stations'].shape,'scene_values':(layout['size'],),'rig':(43,)}
    for key,shape in sizes.items():
        if snapshot[key].shape!=shape or not np.isfinite(snapshot[key]).all():raise ValueError('Invalid joint state array: '+key)
    fixed=points['kind']==0
    if not np.array_equal(snapshot['stations'][fixed],points['stations'][fixed]):raise ValueError('Fixed material point station changed')


def audit_s04(z,m,native):
    if set(z)!=FIELDS:raise ValueError('Point input must contain only seven plain fields')
    if len(set(z['observation_ids']))!=len(z['pts']):raise ValueError('Repeated observation identity')
    if len(set(z['source_rows']))!=len(z['pts']):raise ValueError('Repeated source row')
    if not np.array_equal(np.unique(z['pts']),np.arange(745984,948992,256)):raise ValueError('Incomplete or wrong S04 clock')
    if not np.isfinite(z['uv']).all() or not np.isin(z['kind'],[0,1]).all():raise ValueError('Invalid plain point')
    if m['scene_revision']!=native.scene_revision:raise ValueError('Package belongs to a different current scene')
    if set(m['path_ids'])-native.paths.keys():raise ValueError('Native path missing')
    lookup={str(k):i for i,k in enumerate(z['observation_ids'])};seen=set();counts=collections.Counter()
    sources=collections.defaultdict(dict);board_refs=collections.defaultdict(list)
    withdrawn=set(read(SOURCE/'evidence/correction_sources.json')['withdrawals'])
    for group,path in [('ice',SOURCE/'evidence/selected_ice_records.jsonl.gz'),('board',SOURCE/'evidence/selected_board_records.jsonl.gz'),('advertisement',SOURCE/'supplied_ads/evidence/selected_point_records.jsonl.gz')]:
        for r in records(path):
            if group=='ice':key=r['observation_key'];pt=r['native_pts'];pid=r['native_path_id']
            elif group=='board':key='board:'+r['source_row_id'];pt=r['native_pts'];pid=r['native_path_id']
            else:key=r['stable_id'];pt=r['pts'];pid=r['path_id']
            i=lookup[key]
            if i in seen:raise ValueError('Evidence duplicates a row')
            if pt!=int(z['pts'][i]) or pid!=m['path_ids'][z['path_indices'][i]] or not np.array_equal(z['uv'][i],r['uv']):
                raise ValueError('Point/evidence identity, time, path or original pixel differs: '+key)
            if group=='advertisement' and z['kind'][i]!=r['kind']:raise ValueError('Ad entity kind differs')
            if group!='advertisement' and z['kind'][i]!=1:raise ValueError('Ice/board point is not a curve sample')
            if group=='ice':
                if r['source_row_id'] in withdrawn or r.get('source_parent_row_id') in withdrawn:raise ValueError('Withdrawn ice read reintroduced')
                ref=r['source_ref'];sources[ref['path']][ref['jsonl_line_0based']]=r
            if group=='board':board_refs[r['source_ref']['path']].append(r)
            seen.add(i);counts[group]+=1
    if len(seen)!=len(z['pts']):raise ValueError('Source evidence does not cover every supplied point')
    original_ice=0
    for path,wanted in sources.items():
        for line,r in enumerate(records(path)):
            if line not in wanted:continue
            q=wanted.pop(line)
            if any(r[k]!=q[k] for k in ('source_row_id','native_pts','uv')):raise ValueError('Original ice source differs')
            original_ice+=1
        if wanted:raise ValueError('Original ice source line absent')
    original_board=0
    for path,wanted in board_refs.items():
        original=read(path)
        for q in wanted:
            r=original['rows'][q['source_ref']['row_index']]
            if r['uv']!=q['uv']:raise ValueError('Original board pixels differ')
            original_board+=1
    return dict(counts=dict(counts),every_row_matches_package_evidence=True,original_ice_rows_rechecked=original_ice,
        original_board_rows_rechecked=original_board,withdrawn_ice_ids_reintroduced=0,
        weighted_or_uncertainty_fields_in_fit=False,source_quality_metadata_enters_objective=False,
        visual_acceptance=False)

def prepare():
    if (RUN/'joint_manifest.json').exists():raise FileExistsError('Prepared joint input already exists; do not overwrite a running fit')
    base_state=BASE/'accepted_pixels_latest.npz';s01_input=BASE/'point_observations.npz'
    before={str(p):digest(p) for p in (base_state,s01_input,prep.SCENE,prep.CAMERA)}
    s=npz(base_state);a=npz(s01_input);ma=read(BASE/'mapping.json')
    b=npz(SOURCE/'points/point_observations.npz');mb=read(SOURCE/'points/mapping.json')
    model=SceneParameterModel(read(prep.SCENE),prep.SCENE,optimize_scene=True)
    state=ParameterState(model.initial_state.layout,s['scene_values']);core=NativeCore(LIBRARY)
    native=NativeSceneGPU(prep.SCENE,core,parameter_state=state)
    audit=audit_s04(b,mb,native);print(json.dumps(audit),flush=True)
    if str(s['point_input_sha256'])!=digest(s01_input) or not np.array_equal(s['source_rows'],a['source_rows']):raise ValueError('S01 checkpoint/input mismatch')
    ca=prep.camera_for_snapshot(read(prep.CAMERA),s);cb=read(HERE/'ad_cameras/s04.json')
    dt=float(ca['control_dt']);bc,bfirst=refine_uniform_controls(np.load(HERE/'ad_cameras'/cb['controls_ref']),cb['first_control_time'],cb['control_dt'],dt)
    # Align the S04 fitted 0.2 s basis with its old 0.4 s knot origin. The
    # dropped outer coefficients are outside all source/readout support.
    bc=bc[1:-1];bfirst+=dt
    chart=UniformMotionChart.create(len(bc),bfirst,dt,10,0.)
    initialized=chart.initialize(bc)
    if np.max(np.abs(initialized-bc))>1e-9:raise ValueError('S04 refinement unexpectedly changes its old trajectory')
    bc=initialized
    shots=[];cbegin=rbegin=0
    for identity,z,controls,camera,first,stride,reference in [
        ('s01',a,s['controls'],ca,ca['first_control_time'],int(s['motion_knot_stride']),float(s['motion_reference_time'])),
        ('s04',b,bc,cb,bfirst,10,0.)]:
        shots.append(dict(id=identity,row_begin=rbegin,row_stop=rbegin+len(z['pts']),control_begin=cbegin,control_stop=cbegin+len(controls),
            reference_pts=camera['reference_pts'],time_base_num=camera['time_base_num'],time_base_den=camera['time_base_den'],
            first_control_time=first,knot_stride=stride,motion_reference_time=reference,first_pts=int(z['pts'].min()),last_pts=int(z['pts'].max()),stop_pts_exclusive=int(z['pts'].max()+256)))
        cbegin+=len(controls);rbegin+=len(z['pts'])
    timeline=ShotMotionTimeline(shots,dt);original_pts=np.r_[a['pts'],b['pts']];timeline.assert_support_separation(original_pts,timeline.max_readout())
    paths=list(dict.fromkeys(ma['path_ids']+mb['path_ids']));index={p:i for i,p in enumerate(paths)}
    z={k:np.concatenate([a[k],b[k]],axis=0) for k in FIELDS if k!='path_indices'}
    z['path_indices']=np.r_[[index[ma['path_ids'][i]] for i in a['path_indices']],[index[mb['path_ids'][i]] for i in b['path_indices']]].astype(np.int32)
    out=RUN/'point_observations.npz';np.savez(out,**z)
    for k in FIELDS-{'path_indices'}:
        if not np.array_equal(z[k][:len(a['pts'])],a[k]):raise ValueError('S01 raw point changed: '+k)
    write(RUN/'mapping.json',dict(path_ids=paths,scene_revision=native.scene_revision))
    manifest=dict(schema='rinkspace.shared_camera_disjoint_shots.v1',shots=shots,control_dt=dt,first_control_time=0.,
        point_input_sha256=digest(out),scene_path=str(prep.SCENE),scene_file_sha256=digest(prep.SCENE),scene_revision=native.scene_revision,
        shared_camera_definition=str(prep.CAMERA),shared_camera_rig_source=str(base_state),shared_camera_rig_source_sha256=before[str(base_state)],
        native_library=str(LIBRARY),native_library_sha256=digest(LIBRARY),max_readout=timeline.max_readout(),
        source_inputs={'s01':dict(path=str(s01_input),sha256=digest(s01_input)),'s04':dict(path=str(SOURCE/'points/point_observations.npz'),sha256=digest(SOURCE/'points/point_observations.npz'))},
        initial_s04_motion_source=str(HERE/'ad_cameras/s04.json'),initial_s04_motion_is_seed_only=True,
        source_pts_unchanged=True,shot_motion_continuity_across_edits=False,shared_installation_and_optics=True,
        objective='0.5*sum((predicted_uv-original_uv)^2); every retained point contributes equally',s04_import_audit=audit)
    initial=dict(controls=np.r_[s['controls'],bc],rig=s['rig'],scene_values=s['scene_values'],
        stations=np.r_[s['stations'],npz(SOURCE/'points/initialized_points.npz')['stations']],
        source_rows=z['source_rows'],point_input_sha256=np.array(digest(out)))
    np.savez(RUN/'initial_pixels.npz',**initial)
    for p,sha in before.items():
        if digest(p)!=sha:raise ValueError('An existing source or native authority was changed')
    write(RUN/'joint_manifest.json',manifest)
    print(json.dumps(dict(rows=len(z['pts']),shots=shots,input_sha256=digest(out),s01_unchanged=True)),flush=True)

if __name__=='__main__':prepare()
