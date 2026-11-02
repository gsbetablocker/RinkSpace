"""Test whether one fixed artwork correction explains cross-view source edges.

Read-only local linear diagnosis, not a production fit or a new accepted state.
All residuals use independently re-read RGB transitions and their source normals.
The four columns are the existing shared artwork parameters, not per-letter DOFs.
"""
from pathlib import Path
from collections import defaultdict
import argparse,json
import numpy as np
import prepare_native_multicamera as prepared
from audit_ad4_source_correspondence import inverse_photo,source_station
from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import SceneParameterModel,ParameterState
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
from rinkspace.domain.print_assets import evaluate_print_path
from rinkspace.path_policy import guard_project_write


def summarize(v):
    v=np.asarray(v,float)
    return dict(rows=len(v),median_absolute=float(np.median(abs(v))),p95_absolute=float(np.percentile(abs(v),95)),rms=float(np.sqrt(np.mean(v*v))))


def main(a):
    out=guard_project_write(a.output)
    if out.exists():raise FileExistsError(out)
    m=prepared.read(a.prepared/'joint_manifest.json');mapping=prepared.read(a.prepared/'mapping.json');state=prepared.npz(a.state)
    source_points=prepared.npz(a.prepared/'point_observations.npz');scene_path=Path(m['scene_path']);scene=prepared.read(scene_path)
    model=SceneParameterModel(scene,scene_path,optimize_scene=True)
    prepared.validate_snapshot(state,model.initial_state.layout,mapping,m,source_points)
    if prepared.digest(a.native_library)!=str(state['native_library_provenance_sha256']):raise ValueError('DLL differs')
    ref=next(x for x in scene['advertisement_assets'] if x['advertisement_number']==4)
    asset=prepared.read((scene_path.parent/ref['path']).resolve())
    parameters=ParameterState(model.initial_state.layout,state['scene_values']);board=parameters.apply_scene(scene)['board']
    template=next(t for t in asset['templates'] if t['id']==ref['template_id'])
    group=template['internal_geometry']['groups'][0];shape=group['parameter_block_id']
    block=parameters.blocks[shape]
    if list(block['independent_local_indices'])!=[0,1,2,3]:raise ValueError('Expected one four-coordinate artwork map')
    core=NativeCore(a.native_library);timeline=ShotMotionTimeline(m['shots'],m['control_dt'],m['first_control_time'])
    evidence=prepared.read(a.sources);frames=defaultdict(list);receipts=[]
    for item in evidence:
        atlas=prepared.read(item['reference']);paths={p['path_id']:p for p in atlas['paths']}
        saved=prepared.read(item['readouts']);H=np.asarray(saved['registration']['H'])
        if saved['registration']['status']!='source_pattern_registered':raise ValueError('Unregistered source evidence')
        for row in saved['rows']:
            name=row['path_id'].split('::')[-1];path=paths[name]
            position=inverse_photo(H,row['search_uv']);c=source_station(np.asarray(path['pixels']),position,path['source_closed'],0.)
            if c['source_edge_distance_px']>1e-6:raise ValueError('Source search lost its original contour')
            frames[(item['shot'],saved['pts'])].append(dict(word=item['word'],path=name,station=c['station'],uv=row['uv'],normal=row['normal']))
        receipts.append(dict(**item,readouts_sha256=prepared.digest(item['readouts']),reference_sha256=prepared.digest(item['reference'])))
    reports=[];derivative_step=1e-6
    optical_storage=[31,32,33,34,39,40]
    for (shot_id,pt),rows in sorted(frames.items()):
        shot=next(s for s in m['shots'] if s['id']==shot_id);rig=state['rig'][shot['camera_index']]
        clock=(pt-shot['reference_pts'])/12800+timeline.offset(shot)
        n=len(rows);xyz=np.empty((n,3));jx=np.empty((n,3,4));normals=np.asarray([r['normal'] for r in rows]);uv=np.asarray([r['uv'] for r in rows])
        for name in sorted({r['path'] for r in rows}):
            ix=np.asarray([i for i,r in enumerate(rows) if r['path']==name]);stations=[rows[i]['station'] for i in ix]
            result=evaluate_print_path(asset,'print_instance/Ad4',name,board,core,stations,parameter_state=parameters,jacobian=True)
            xyz[ix]=result['xyz'];jx[ix]=result['jacobians_by_block'][shape][:,:,:4]
        def project(world):
            p=core.project_motion(rig,state['controls'],timeline.first,timeline.dt,1080,clock,world)
            if np.any(p['status']):raise ValueError('Source diagnostic projection invalid')
            return p['uv']
        predicted=project(xyz);residual=predicted-uv;normal_residual=np.einsum('ni,ni->n',residual,normals)
        jac=np.empty((n,4))
        for j in range(4):
            dp=(project(xyz+derivative_step*jx[:,:,j])-project(xyz-derivative_step*jx[:,:,j]))/(2*derivative_step)
            jac[:,j]=np.einsum('ni,ni->n',dp,normals)
        optical_jac=np.empty((n,len(optical_storage)))
        for j,index in enumerate(optical_storage):
            hi=rig.copy();lo=rig.copy();hi[index]+=1e-5;lo[index]-=1e-5
            high=core.project_motion(hi,state['controls'],timeline.first,timeline.dt,1080,clock,xyz)
            low=core.project_motion(lo,state['controls'],timeline.first,timeline.dt,1080,clock,xyz)
            if np.any(high['status']) or np.any(low['status']):raise ValueError('Invalid optical derivative probe')
            optical_jac[:,j]=np.einsum('ni,ni->n',(high['uv']-low['uv'])/2e-5,normals)
        delta,_,rank,singular=np.linalg.lstsq(jac,-normal_residual,rcond=None)
        cam=core.evaluate_motion_camera(rig,state['controls'],timeline.first,timeline.dt,[clock])
        report=dict(shot=shot_id,pts=int(pt),source_seconds=pt/12800,focal_pixels=float(cam['fx'][0]),
            source_words=sorted({r['word'] for r in rows}),before=summarize(normal_residual),
            own_view_linear_artwork_correction=delta.tolist(),own_view_linear_after=summarize(normal_residual+jac@delta),
            column_rank=int(rank),singular_values=singular.tolist(),
            words={word:dict(signed_uv_median=np.median(residual[[r['word']==word for r in rows]],0).tolist(),
                            source_normal=summarize(normal_residual[[r['word']==word for r in rows]])) for word in sorted({r['word'] for r in rows})})
        reports.append((report,jac,normal_residual,rows,optical_jac))
    J=np.concatenate([x[1] for x in reports]);r=np.concatenate([x[2] for x in reports])
    common,_,rank,singular=np.linalg.lstsq(J,-r,rcond=None)
    # S01 spans both separated words in the same frame; S04 constrains JGP at a
    # second pan/zoom. Their joint correction is tested on the later whole views.
    front=[x for x in reports if x[0]['shot'] in ('s01','s04')]
    jf=np.concatenate([x[1] for x in front]);rf=np.concatenate([x[2] for x in front])
    front_delta,_,front_rank,_=np.linalg.lstsq(jf,-rf,rcond=None)
    combined=np.concatenate([np.c_[x[1],x[4]] for x in reports])
    column_norms=np.linalg.norm(combined,axis=0)
    combined_solution,_,combined_rank,combined_singular=np.linalg.lstsq(combined/column_norms,-r,rcond=None)
    combined_solution=combined_solution/column_norms
    for report,jac,residual,rows,optical_jac in reports:
        report['one_shared_linear_correction_after']=summarize(residual+jac@common)
        report['shared_artwork_and_existing_radial_linear_after']=summarize(residual+np.c_[jac,optical_jac]@combined_solution)
        report['front_shots_linear_correction_after']=summarize(residual+jac@front_delta)
        for word in report['words']:
            ix=np.asarray([r['word']==word for r in rows])
            report['words'][word]['one_shared_linear_after']=summarize((residual+jac@common)[ix])
            report['words'][word]['front_shots_linear_after']=summarize((residual+jac@front_delta)[ix])
    output=dict(schema='rinkspace.ad4_cross_view_source_geometry_diagnosis.v1',
        state=dict(path=str(a.state),sha256=prepared.digest(a.state)),source_evidence=receipts,
        artifact_parameter_block=shape,parameter_names=['translate_u','translate_v','log_scale_u','log_scale_v'],
        one_shared_linear_correction=common.tolist(),one_shared_column_rank=int(rank),
        existing_radial_parameter_names=['k10','k11','k20','k21','wide_k1','wide_k2'],
        shared_artwork_and_existing_radial_linear_correction=combined_solution.tolist(),
        shared_artwork_and_existing_radial_column_rank=int(combined_rank),
        front_shots_linear_correction=front_delta.tolist(),front_shots_column_rank=int(front_rank),
        frames=[x[0] for x in reports],finite_difference_step=derivative_step,
        optimizer_started=False,state_modified=False,source_pixels_modified=False,
        limits=['This is a first-order explanatory regression with the current camera and installation held fixed.',
                'Photographic queries identify local source edges, not exact permanent material points; source-normal errors are used.',
                'A failed common correction does not identify which camera or surface assumption is wrong.',
                'These frames were selected for diagnosis and are not independent final-validation evidence.'])
    prepared.write(out,output)
    print(json.dumps(dict(output=str(out),shared=common.tolist(),front=front_delta.tolist(),
        frames=[dict(shot=x[0]['shot'],time=x[0]['source_seconds'],focal=x[0]['focal_pixels'],before=x[0]['before']['rms'],
                     after=x[0]['one_shared_linear_correction_after']['rms'],own=x[0]['own_view_linear_after']['rms'],
                     existing_optics=x[0]['shared_artwork_and_existing_radial_linear_after']['rms']) for x in reports])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','state','native-library','sources','output'):p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
