"""Rebind an explicit camera-sharing hypothesis without changing source/state."""
from pathlib import Path
from copy import deepcopy
import argparse,json
import numpy as np
import prepare_native_multicamera as prepared
from prepare_spatial_refinement import _write_prepared
from rinkspace.domain.parameters import SceneParameterModel
from rinkspace.domain.camera_coupling import read_camera_coupling,build_multicamera_directions
from rinkspace.domain.camera_experiment import load_camera_experiment
from rinkspace.path_policy import guard_project_write


def main(a):
    out=guard_project_write(a.output)
    if out.exists():raise FileExistsError(out)
    m=prepared.read(a.prepared/'joint_manifest.json');mapping=prepared.read(a.prepared/'mapping.json')
    points=prepared.npz(a.prepared/'point_observations.npz');state=prepared.npz(a.state)
    model=SceneParameterModel(prepared.read(m['scene_path']),Path(m['scene_path']),optimize_scene=True)
    prepared.validate_snapshot(state,model.initial_state.layout,mapping,m,points)
    prepared.camera_working_envelope_path(m)
    if prepared.digest(a.prepared/'point_observations.npz')!=m['point_input_sha256']:raise ValueError('Current points changed')
    coupling=read_camera_coupling(a.coupling);previous_profile=Path(m['camera_parameterization']['profile_path'])
    if prepared.digest(previous_profile)!=m['camera_parameterization']['profile_sha256']:raise ValueError('Camera profile changed')
    profile=a.profile if a.profile is not None else previous_profile
    selections=[]
    for rig in state['rig']:
        copied,indices,_=load_camera_experiment(profile,rig)
        if not np.array_equal(copied,rig):raise ValueError('Current state is outside its camera profile')
        selections.append(indices)
    _,layout=build_multicamera_directions(state['rig'],selections,np.ones(31),coupling)
    target=deepcopy(m);previous=deepcopy(target['camera_parameterization'])
    target['camera_parameterization'].update(coupling_path=str(a.coupling.resolve()),coupling_sha256=prepared.digest(a.coupling))
    target['camera_parameterization'].update(profile_path=str(profile.resolve()),profile_sha256=prepared.digest(profile))
    if getattr(a,'working_envelope',None) is not None:
        path=a.working_envelope.resolve()
        target['camera_working_envelope']=dict(path=str(path),sha256=prepared.digest(path))
        prepared.camera_working_envelope_path(target)
        fact=prepared.read(path)
        if not isinstance(fact,dict) or fact.get('schema')!='rinkspace.assistant_camera_working_envelope.v1':raise ValueError('Unsupported camera working envelope format')
        try:prepared.single.mount_configuration(1.,state['rig'][0],path)
        except (KeyError,TypeError,IndexError) as error:raise ValueError('Invalid camera working envelope format') from error
    receipt=dict(schema='rinkspace.camera_coupling_rebind.v1',parent_prepared=str(a.prepared.resolve()),
        parent_state=dict(path=str(a.state.resolve()),sha256=prepared.digest(a.state)),
        previous_camera_parameterization=previous,new_camera_parameterization=target['camera_parameterization'],
        previous_camera_working_envelope=m.get('camera_working_envelope'),new_camera_working_envelope=target.get('camera_working_envelope'),
        selected_camera_column_count=layout['unique_camera_columns'],point_input_sha256=m['point_input_sha256'],
        all_numeric_arrays_unchanged=True,all_source_rows_UV_PTS_unchanged=True,scene_geometry_unchanged=True,
        hypothesis_status='candidate_requires_full_source_fit_and_visual_review',optimizer_started=False)
    result,mapped=_write_prepared(out,points,state['stations'],m['shots'],state,target,mapping,model,receipt,
        link_input=a.prepared/'point_observations.npz')
    if 'spatial_refinement_preparation' in m:result['spatial_refinement_preparation']=m['spatial_refinement_preparation']
    else:result.pop('spatial_refinement_preparation',None)
    result['camera_coupling_rebind']=receipt;prepared.write(out/'joint_manifest.json',result)
    saved=prepared.npz(out/'initial_pixels.npz')
    for key in state:
        if key!='joint_numeric_dependencies_json' and (saved[key].dtype!=state[key].dtype or saved[key].shape!=state[key].shape or saved[key].tobytes()!=state[key].tobytes()):raise ValueError('Numeric/source state changed: '+key)
    if prepared.digest(out/'point_observations.npz')!=m['point_input_sha256']:raise ValueError('Source input bytes changed')
    prepared.validate_snapshot(saved,model.initial_state.layout,mapped,result,points)
    prepared.write(out/'coupling_rebind.json',receipt)
    print(json.dumps(dict(output=str(out),camera_columns=layout['unique_camera_columns'],rows=len(points['pts']),numeric_arrays_unchanged=True)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','state','coupling','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--profile',type=Path,help='Existing profile whose fixed fields exactly preserve the numerical seed')
    p.add_argument('--working-envelope',type=Path,help='Explicit camera working-envelope JSON to bind without changing the numerical seed')
    main(p.parse_args())
