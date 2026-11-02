"""Apply explicit source-image identity findings without rewriting measurements."""
from pathlib import Path
from copy import deepcopy
import argparse,json
import numpy as np
import prepare_native_multicamera as prepared
from prepare_spatial_refinement import _write_prepared
from rinkspace.domain.parameters import SceneParameterModel
from rinkspace.path_policy import guard_project_write


def main(a):
    out=guard_project_write(a.output.resolve())
    if out.exists():raise FileExistsError(out)
    manifest=prepared.read(a.prepared/'joint_manifest.json');mapping=prepared.read(a.prepared/'mapping.json')
    points=prepared.npz(a.prepared/'point_observations.npz');state=prepared.npz(a.state)
    model=SceneParameterModel(prepared.read(manifest['scene_path']),Path(manifest['scene_path']),optimize_scene=True)
    prepared.validate_snapshot(state,model.initial_state.layout,mapping,manifest,points)
    if prepared.digest(a.prepared/'point_observations.npz')!=manifest['point_input_sha256']:raise ValueError('Current input changed')
    review=prepared.read(a.review);entries=review['rows'];by_id={str(v):i for i,v in enumerate(points['observation_ids'])}
    if len(by_id)!=len(points['pts']):raise ValueError('Duplicate current observation identity')
    keep=np.ones(len(points['pts']),bool);excluded=[];seen=set()
    for row in entries:
        identity=row['observation_id']
        if identity in seen or identity not in by_id:raise ValueError('Review identity not unique/current')
        seen.add(identity);i=by_id[identity]
        if int(points['pts'][i])!=row['pts'] or not np.array_equal(points['uv'][i],row['original_uv']):raise ValueError('Reviewed source UV/PTS changed')
        status=row['status']
        if status not in ('identity_compatible','wrong_entity_or_role','ambiguous_not_fit'):raise ValueError('Unrecognized review status')
        if status!='identity_compatible':
            keep[i]=False;excluded.append({**row,'current_parent_row':i,'source_row':int(points['source_rows'][i]),'path_id':mapping['path_ids'][points['path_indices'][i]]})
    repaired={k:v[keep] for k,v in points.items()};shots=deepcopy(manifest['shots']);begin=0
    for shot in shots:
        original=points['pts'][shot['row_begin']:shot['row_stop']];count=int(np.count_nonzero(keep[shot['row_begin']:shot['row_stop']]))
        shot['row_begin']=begin;shot['row_stop']=begin+count;begin+=count
        now=repaired['pts'][shot['row_begin']:shot['row_stop']]
        if not np.array_equal(np.unique(original),np.unique(now)):raise ValueError('Source exposure coverage unexpectedly changed')
    state={k:v.copy() for k,v in state.items()};state['stations']=state['stations'][keep];state['source_rows']=state['source_rows'][keep]
    receipt=dict(schema='rinkspace.camera_source_identity_repair.v1',parent_prepared=str(a.prepared.resolve()),parent_state=str(a.state.resolve()),parent_state_sha256=prepared.digest(a.state),source_review=str(a.review.resolve()),source_review_sha256=prepared.digest(a.review),parent_point_input_sha256=manifest['point_input_sha256'],parent_rows=len(keep),retained_rows=int(keep.sum()),excluded_rows=len(excluded),all_retained_fields_byte_preserved=True,source_uv_pts_never_rewritten=True,all_numeric_globals_unchanged=True,user_pixel_acceptance_not_changed=True,fit_started=False,excluded=excluded)
    target,mapped=_write_prepared(out,repaired,state['stations'],shots,state,manifest,mapping,model,receipt)
    if 'spatial_refinement_preparation' in manifest:target['spatial_refinement_preparation']=manifest['spatial_refinement_preparation']
    else:target.pop('spatial_refinement_preparation',None)
    target['camera_source_identity_repair']=receipt;prepared.write(out/'joint_manifest.json',target)
    saved=prepared.npz(out/'point_observations.npz')
    for key in points:
        if saved[key].tobytes()!=points[key][keep].tobytes():raise RuntimeError('Retained input bytes changed: '+key)
    prepared.write(out/'source_identity_repair.json',receipt)
    print(json.dumps(dict(output=str(out),retained=len(saved['pts']),excluded=len(excluded),point_input_sha256=target['point_input_sha256'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','state','review','output'):p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
