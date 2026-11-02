"""Bind corrected fixed print contours while retaining a complete joint state."""
from pathlib import Path
from copy import deepcopy
import argparse,json
import numpy as np
import prepare_native_multicamera as prepared
from prepare_spatial_refinement import _write_prepared
from build_ad3_completion import layout_without_content_identity
from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import SceneParameterModel,ParameterState
from rinkspace.session.gpu_geometry import NativeSceneGPU
from rinkspace.path_policy import guard_project_write


def main(a):
    root=a.prepared.resolve();out=guard_project_write(a.output.resolve())
    if out.exists() or out.is_relative_to(root):raise FileExistsError(out)
    manifest=prepared.read(root/'joint_manifest.json');mapping=prepared.read(root/'mapping.json')
    state=prepared.npz(a.state);points=prepared.npz(root/'point_observations.npz')
    old_path=Path(manifest['scene_path']);old=prepared.read(old_path);new=prepared.read(a.scene)
    if {k:v for k,v in old.items() if k!='advertisement_assets'}!={k:v for k,v in new.items() if k!='advertisement_assets'}:
        raise ValueError('Print-reference rebind may not alter ice/board/camera scene definitions')
    for previous,candidate in zip(old['advertisement_assets'],new['advertisement_assets'],strict=True):
        if previous.get('advertisement_number')!=4 and previous!=candidate:
            raise ValueError('Unrelated print asset changed')
    old_model=SceneParameterModel(old,old_path,optimize_scene=True)
    model=SceneParameterModel(new,a.scene,optimize_scene=True)
    if layout_without_content_identity(old_model.initial_state.layout)!=layout_without_content_identity(model.initial_state.layout):
        raise ValueError('Print correction changed the parameter layout')
    prepared.validate_snapshot(state,old_model.initial_state.layout,mapping,manifest,points)
    if prepared.digest(root/'point_observations.npz')!=manifest['point_input_sha256']:raise ValueError('Source points changed')
    library_sha=prepared.digest(a.native_library)
    if library_sha!=str(state['native_library_provenance_sha256']):raise ValueError('Frozen DLL changed')
    previous_values=state['scene_values'].copy();changed=[]
    for name in a.reset_block:
        block=next(b for b in model.initial_state.layout['blocks'] if b['id']==name)
        if block['kind']!='print_shape':raise ValueError('Only the corrected shared print shape may reset')
        start=block['offset'];stop=start+block['size']
        state['scene_values'][start:stop]=model.initial_state.values[start:stop]
        changed.append(dict(block=name,offset=start,before=previous_values[start:stop].tolist(),
            after=state['scene_values'][start:stop].tolist()))
    native=NativeSceneGPU(a.scene,NativeCore(a.native_library),
        parameter_state=ParameterState(model.initial_state.layout,state['scene_values']))
    if not set(mapping['path_ids'])<=set(native.paths):raise ValueError('Current observation paths disappeared')
    target=deepcopy(manifest);target.update(scene_path=str(a.scene.resolve()),scene_file_sha256=prepared.digest(a.scene),
        scene_revision=native.scene_revision,native_library=str(a.native_library.resolve()),native_library_sha256=library_sha)
    mapped=deepcopy(mapping);mapped['scene_revision']=native.scene_revision
    receipt=dict(schema='rinkspace.fixed_print_reference_rebind.v1',parent_root=str(root),
        parent_state=dict(path=str(a.state.resolve()),sha256=prepared.digest(a.state)),
        previous_scene=dict(path=str(old_path),sha256=manifest['scene_file_sha256']),
        candidate_scene=dict(path=str(a.scene.resolve()),sha256=target['scene_file_sha256']),
        reset_shared_shape_blocks=changed,point_input_sha256=manifest['point_input_sha256'],
        source_UV_PTS_identity_unchanged=True,ice_and_board_geometry_unchanged=True,
        rig_and_motion_unchanged=True,path_ids_and_vertex_stations_unchanged=True,
        no_individual_glyph_degrees_of_freedom=True,fit_executed=False)
    updated,mapped=_write_prepared(out,points,state['stations'],manifest['shots'],state,target,mapped,model,
        receipt,link_input=root/'point_observations.npz')
    if 'spatial_refinement_preparation' in manifest:updated['spatial_refinement_preparation']=manifest['spatial_refinement_preparation']
    else:updated.pop('spatial_refinement_preparation',None)
    updated['fixed_print_reference_rebind']=receipt
    prepared.write(out/'joint_manifest.json',updated)
    saved=prepared.npz(out/'initial_pixels.npz')
    prepared.validate_snapshot(saved,model.initial_state.layout,mapped,updated,points)
    for key in ['controls','stations','board','layout','rig','source_rows']:
        if not np.array_equal(state[key],saved[key]):raise ValueError('Unexpected numeric change: '+key)
    if prepared.digest(out/'point_observations.npz')!=manifest['point_input_sha256']:raise ValueError('Source bytes changed')
    prepared.write(out/'print_reference_rebind.json',receipt)
    print(json.dumps(dict(output=str(out),rows=len(points['pts']),reset_blocks=changed,fit_executed=False)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','state','scene','native-library','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--reset-block',action='append',required=True)
    main(p.parse_args())
