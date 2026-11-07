"""Source-clock programme receipts and a separate final display packet.

No fitting, camera/scale correction, alignment or replacement of old results.
"""
from pathlib import Path
import argparse,json,re,shutil
import numpy as np
from scipy.spatial.transform import Rotation
from .rebuild_prepare import load,digest
from .fullbody_round import write_json
from .programme_contract import required_clock,boundary_receipt
from .trajectory_context import ProgrammeContext
from .programme_camera import ProgrammeCamera


def distribution(value):
    a=np.asarray(value);a=a[np.isfinite(a)]
    return dict(count=len(a),minimum=float(a.min()),median=float(np.median(a)),p95=float(np.percentile(a,95)),maximum=float(a.max()))if len(a)else dict(count=0)


DEFAULT_CANDIDATE_ID='programme_trajectory_1106'


def finalization_paths(project,stage,*,candidate_id=DEFAULT_CANDIDATE_ID,output=None,viewer_config=None):
    """Validate new destinations without creating a result, config or directory."""
    if not isinstance(candidate_id,str) or re.fullmatch(r'[A-Za-z0-9_]+',candidate_id) is None:
        raise ValueError('Invalid candidate_id: use only ASCII letters, digits and underscores')
    project,stage=Path(project).resolve(),Path(stage).resolve()
    programme=project/'events/ankara_2026/programmes/okada_mei_sp'
    out=Path(output).resolve() if output is not None else stage/'final'
    from rinkspace.storage import AssetStore
    data_root=AssetStore(project).data.resolve()
    if out==data_root or not out.is_relative_to(data_root):
        raise ValueError('Final output must be a new directory under the configured Data root')
    target,record=programme/(candidate_id+'.npz'),programme/(candidate_id+'.json')
    from rinkspace.path_policy import guard_project_write
    for path in (target,record):
        guard_project_write(path,project_root=project)
        if path.exists():raise FileExistsError('Preserve existing formal candidate '+str(path))
    if out.exists():raise FileExistsError('Preserve existing final output '+str(out)+'; choose a new --output')
    config=str(viewer_config) if viewer_config is not None else 'events/ankara_2026/programmes/okada_mei_sp/viewer3d_'+candidate_id+'.json'
    return dict(output=out,parameter=target,record=record,viewer_config=config,
        candidate_id=candidate_id,variant_id='programme_trajectory' if candidate_id==DEFAULT_CANDIDATE_ID else candidate_id)


def finalize(project,prepared,stage,*,candidate_id=DEFAULT_CANDIDATE_ID,output=None,viewer_config=None):
    project,prepared,stage=map(Path,(project,prepared,stage))
    paths=finalization_paths(project,stage,candidate_id=candidate_id,output=output,viewer_config=viewer_config)
    out,target,record=paths['output'],paths['parameter'],paths['record']
    raw=stage/'fitted/result.npz';z=load(raw);p=load(prepared/'input.npz');obs=load(prepared/'observations.npz')
    scope_path=project/'events/ankara_2026/programmes/okada_mei_sp/rebuild_scope.json';scope=json.loads(scope_path.read_text(encoding='utf-8'));pts,shots,tb=required_clock(scope)
    np.testing.assert_array_equal(z['pts'],pts);np.testing.assert_array_equal(p['pts'],pts);np.testing.assert_array_equal(z['time_base'],tb)
    context=ProgrammeContext.from_scope(scope_path,body_id='ankara_2026/okada_mei_sp/logical_source_person_0',shape=np.r_[p['shared_betas'],p['body_scale']],metres_per_scene_unit=float(z['metres_per_scene_unit']),equipment_sha256=digest(prepared/'fixed_binding.npz'),camera_sha256=digest(prepared/'camera_context.json'))
    packet=context.packet()
    # The absolute curve is active over this whole source clock. The legacy
    # correction-layer mask is empty for an absolute state, not a frozen body.
    parameter=load(stage/'solve/candidate.npz')
    if str(parameter['state_kind'])!='absolute_motion_v1':raise ValueError('Expected the actual saved full absolute body state')
    out.mkdir(parents=True,exist_ok=False);(out/'fitted').mkdir()
    camera_record=json.loads((prepared/'camera_context.json').read_text(encoding='utf-8'))
    packet.update(shots=[dict(shot_id=s['id'],valid_pts=[s['processing']['first_pts'],s['processing']['last_pts']],
        frozen_camera_state_sha256=camera_record['shots'][s['id']]['state_sha256'])for s in scope['shots']],
        source_evidence=dict(path=str(prepared/'source_evidence.json'),sha256=digest(prepared/'source_evidence.json')),
        continuous_state_registry=dict(kind=str(parameter['state_kind']),parameter_sha256=digest(stage/'solve/candidate.npz'),
            source_clock_sha256=digest(prepared/'input.npz'),root_and_body_control_count=int(len(parameter['absolute_knots'])-4),
            per_frame_identity_reset=False,shoe_translation='attached via saved body FK and one shared binding'))
    write_json(out/'programme_context.json',packet)
    z.update(window_mask=np.ones(len(pts),bool),body_refinement_window_mask=np.ones(len(pts),bool),
        programme_context_json=np.array(json.dumps(packet,sort_keys=True)),source_shot_ids=shots,
        shared_betas=p['shared_betas'],shared_body_scale=p['body_scale'],
        source_uv_native=obs['uv'],source_observation_use=obs['use'],
        final_metadata_provenance_json=np.array(json.dumps(dict(parent_result=str(raw),parent_result_sha256=digest(raw),coordinates_unchanged=True,absolute_active_clock_corrected=True))))
    destination=out/'fitted/result.npz';np.savez_compressed(destination,**z)
    clock=boundary_receipt(scope,pts,z['P_world_mid'],velocity_world=z['P_velocity_world_mps']);write_json(out/'clock_and_cuts.json',clock)
    t=pts/12800.;parents=np.array([-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19]);j=z['native22_world_mid'];lengths=np.linalg.norm(j[:,1:]-j[:,parents[1:]],axis=-1)
    metrics=dict(whole_frames=len(pts),whole_seconds=t[[0,-1]].tolist(),source_observations=int(obs['use'].sum()),
        exposures_without_any_body_observation=int((~obs['use'].any(1)).sum()),source_identity_warnings=int(z['source_identity_suspect'].sum()),
        bone_length_frame_range_m=float(np.max(np.ptp(lengths,axis=0))),shared_identity_and_scale=context.packet(),
        pelvis_step_m=distribution(np.linalg.norm(np.diff(z['P_world_mid'],axis=0),axis=-1)),
        pelvis_speed_mps=distribution(np.linalg.norm(z['P_velocity_world_mps'],axis=-1)),
        raw_ice_reference_diagnostics_not_truth=dict(clearance_m=distribution(z['skate_clearance_m_mid']),support_gap_m=distribution(z['skate_support_gap_m_mid'])),
        interpretation='Numeric outliers rank review locations; no universal speed/ice accuracy gate',accuracy_accepted=False)
    prep=json.loads((prepared/'preparation.json').read_text(encoding='utf-8'))
    edge_height=z['skate_edge_world_mid'][...,2].min(axis=(2,3));worst=np.unravel_index(np.argmin(edge_height),edge_height.shape)
    metrics['raw_ice_reference_diagnostics_not_truth'].update(runner_edge_height_m=distribution(edge_height),
        minimum_runner_exposure=dict(pts=int(pts[worst[0]]),seconds=float(pts[worst[0]]/12800.),foot=int(worst[1]),height_m=float(edge_height[worst])),
        meaning='saved template geometry relative to reference Z=0, not direct real penetration/contact truth')
    old_s01=Path(prep['old_S01_result']);old=load(old_s01);ix=np.searchsorted(pts,old['pts'])
    if digest(old_s01)!=prep['old_S01_result_sha256']:raise ValueError('Accepted old S01 changed')
    metrics['accepted_S01']=dict(original_preserved=True,original_sha256=digest(old_s01),new_candidate_P_change_m=distribution(np.linalg.norm(z['P_world_mid'][ix]-old['P_world_mid'],axis=-1)),new_candidate_root_rotation_change_degrees=distribution(np.degrees(Rotation.from_matrix(old['Q_world_mid'].swapaxes(-1,-2)@z['Q_world_mid'][ix]).magnitude())))
    spin=(t>=88)&(t<=92)
    metrics['spin88_92']=dict(pelvis_XY_span_m=np.ptp(z['P_world_mid'][spin,:2],axis=0).tolist(),right_fixed_blade_reference_XY_span_m=np.ptp(z['skate_reference_world_mid'][spin,1,:2],axis=0).tolist(),meaning='fixed model material reference; not observed contact point')
    # Check the same native per-shot camera against saved per-row world points.
    camera=ProgrammeCamera(project,p,shots);ratio=float(z['body_length_ratio']);closure=[]
    for first in range(0,len(pts),128):
        ids=np.arange(first,min(first+128,len(pts)));uv=camera.project(z['world_at_row'][ids]/ratio,ids)
        closure.append(abs(uv[:,96:]-z['COCO_uv'][ids]))
    metrics['native_saved_row_projection_closure_px']=distribution(np.concatenate(closure))
    np.testing.assert_allclose(z['G_world_mid'][:,:2],z['P_world_mid'][:,:2],rtol=0,atol=1e-12);np.testing.assert_array_equal(z['G_world_mid'][:,2],0.)
    write_json(out/'trajectory_metrics.json',metrics)
    _figures(prepared,out,z,obs,prep)
    # Recheck immediately before publication as well as before any work.
    for path in (target,record):
        if path.exists():raise FileExistsError('Preserve existing formal candidate '+str(path))
    with (stage/'solve/candidate.npz').open('rb') as src,target.open('xb') as dst:
        shutil.copyfileobj(src,dst)
    formal_record=dict(parameter=str(target),parameter_sha256=digest(target),result=str(destination),result_sha256=digest(destination),source_frames=len(pts),source_time_base=list(tb),scope_sha256=digest(scope_path),programme_context=packet,user_accepted=False,accuracy_accepted=False,candidate_id=paths['candidate_id'],viewer_config=paths['viewer_config'],variant_id=paths['variant_id'])
    with record.open('x',encoding='utf8') as stream:
        stream.write(json.dumps(formal_record,ensure_ascii=False,indent=2)+'\n')
    return dict(result=str(destination),parameter=str(target),metrics=str(out/'trajectory_metrics.json'),review=str(out/'source_review_grid.jpg'))


def _figures(prepared,out,z,obs,prep):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image,ImageDraw
    from rinkspace.viewer3d.source import SourceVideo
    initial=load(prepared/'initial.npz');P=z['P_world_mid'];t=z['pts']/12800.;ratio=float(z['body_length_ratio'])
    fig,ax=plt.subplots(figsize=(12,6));ax.plot(initial['P'][:,0]*ratio,initial['P'][:,1]*ratio,color='0.7',lw=.7,label='whole-programme initial')
    for shot in np.unique(z['source_shot_ids']):
        selected=z['source_shot_ids']==shot;ax.plot(P[selected,0],P[selected,1],lw=1.,label=str(shot))
    ax.set(xlabel='world X (m)',ylabel='world Y (m)',title='G = pelvis vertical projection; same frozen world, no per-shot offsets');ax.axis('equal');ax.grid();ax.legend(ncol=5);fig.tight_layout();fig.savefig(out/'programme_G.png',dpi=140);plt.close(fig)
    links=[(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
    times=[22.12,44.98,58.28,74.14,81.62,87.64,101.66,140.92];source=SourceVideo(prep['source_video'],(1,12800));panels=[];rows=[]
    try:
        for sec in times:
            pt=round(sec*12800);i=int(np.searchsorted(z['pts'],pt))
            if int(z['pts'][i])!=pt:raise ValueError('Review point is not an actual source exposure')
            image=Image.fromarray(source.read(pt));d=ImageDraw.Draw(image)
            for a,b in links:d.line([tuple(z['COCO_uv'][i,a]),tuple(z['COCO_uv'][i,b])],fill=(255,70,210),width=3)
            for k,good in enumerate(obs['use'][i]):
                if good:
                    x,y=obs['uv'][i,k];d.ellipse((x-3,y-3,x+3,y+3),fill=(0,255,255))
            # Equipment UV is model prediction, not manually observed steel.
            gear=z['equipment_uv'][i]
            for f in(0,1):
                for a,b in((0,1),(1,2),(2,3),(3,0)):d.line([tuple(gear[f,a]),tuple(gear[f,b])],fill=(255,180,30),width=2)
            image.thumbnail((960,540));tile=Image.new('RGB',(960,580),(25,25,25));tile.paste(image,(0,40));draw=ImageDraw.Draw(tile);draw.text((8,7),f'{sec:.2f}s | {z["source_shot_ids"][i]} | magenta=model body; cyan=2D network estimates; orange=model observation adapters',(240,240,240));tile.save(out/f'source_{pt}.jpg',quality=90);panels.append(tile);rows.append(dict(pts=pt,seconds=sec,shot=str(z['source_shot_ids'][i]),image=str(out/f'source_{pt}.jpg'),status='development review; full blade source evidence is separate'))
    finally:source.close()
    sheet=Image.new('RGB',(1920,2320))
    for k,panel in enumerate(panels):sheet.paste(panel,((k%2)*960,(k//2)*580))
    sheet.save(out/'source_review_grid.jpg',quality=90);write_json(out/'source_review_points.json',rows)


def main(argv=None):
    ap=argparse.ArgumentParser()
    for key in('project','prepared','stage'):ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--candidate-id',default=DEFAULT_CANDIDATE_ID)
    ap.add_argument('--output',type=Path)
    ap.add_argument('--viewer-config',type=Path)
    a=ap.parse_args(argv);print(json.dumps(finalize(a.project,a.prepared,a.stage,candidate_id=a.candidate_id,output=a.output,viewer_config=a.viewer_config)))

if __name__=='__main__':main()
