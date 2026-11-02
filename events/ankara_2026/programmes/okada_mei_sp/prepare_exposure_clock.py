"""Apply an explicit exposure-clock hypothesis while transferring the current view."""
from pathlib import Path
from copy import deepcopy
import argparse,json,numpy as np
import prepare_native_multicamera as p
from prepare_spatial_refinement import _write_prepared
from rinkspace.domain.parameters import SceneParameterModel
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline,continue_exposure_halos
from rinkspace.domain.motion_chart import UniformMotionChart,cubic_samples
from rinkspace.path_policy import guard_project_write


def main(a):
    out=guard_project_write(a.output.resolve())
    if out.exists():raise FileExistsError(out)
    m=p.read(a.prepared/'joint_manifest.json');mapping=p.read(a.prepared/'mapping.json');z=p.npz(a.prepared/'point_observations.npz');v=p.npz(a.state);model=SceneParameterModel(p.read(m['scene_path']),Path(m['scene_path']),optimize_scene=True);p.validate_snapshot(v,model.initial_state.layout,mapping,m,z)
    if p.digest(a.prepared/'point_observations.npz')!=m['point_input_sha256']:raise ValueError('Current source input changed')
    result={k:x.copy() for k,x in v.items()};motion_source=v
    if a.motion_source:
        motion_source=p.npz(a.motion_source)
        for key in ('rig','scene_values','board','layout'):
            if not np.array_equal(motion_source[key],v[key]):raise ValueError('Motion transfer source changed '+key)
        if motion_source['controls'].shape!=v['controls'].shape:raise ValueError('Motion storage layout differs')
        result['controls']=motion_source['controls'].copy()
    target=deepcopy(m);fact=p.read(a.clock);changed=[]
    for shot in target['shots']:
        if shot['id'] not in fact['shots']:continue
        if shot.get('exposure_pts_offsets'):raise ValueError('Preserve existing explicit exposure clock')
        offsets=deepcopy(fact['shots'][shot['id']]);step=fact['source_step_pts']
        for item in offsets:
            if (item['first_pts']-shot['first_pts'])%step or item['offset_pts']%step:raise ValueError('Exposure correction is not on source frame grid')
        shot['exposure_pts_offsets']=offsets;changed.append(shot['id'])
    tl=ShotMotionTimeline(target['shots'],target['control_dt'],target['first_control_time']);notes=[]
    for old,shot in zip(m['shots'],target['shots'],strict=True):
        if shot['id'] not in changed:continue
        count=shot['control_stop']-shot['control_begin'];sl=slice(shot['control_begin'],shot['control_stop']);pts=np.arange(shot['first_pts'],shot['last_pts']+fact['source_step_pts'],fact['source_step_pts'],dtype=np.int64);before=(pts-shot['reference_pts'])*shot['time_base_num']/shot['time_base_den'];after=tl.local_exposure_time(shot,pts);oldidx,oldw=cubic_samples(count,shot['first_control_time'],tl.dt,before);wanted=np.einsum('ij,ijk->ik',oldw,motion_source['controls'][sl][oldidx]);chart=UniformMotionChart.create(count,shot['first_control_time'],tl.dt,shot['knot_stride'],shot['motion_reference_time']);raw=np.zeros((count,3,chart.refinement.shape[1]));raw[:,0,:]=chart.refinement;R=continue_exposure_halos(raw,shot['first_control_time'],tl.dt,[after[0],after[-1]])[:,0,:];idx,w=cubic_samples(count,shot['first_control_time'],tl.dt,after);B=np.einsum('ij,ijk->ik',w,R[idx]);coefficients=np.linalg.lstsq(B,wanted,rcond=None)[0];result['controls'][sl]=R@coefficients;error=B@coefficients-wanted;notes.append(dict(shot=shot['id'],source_exposures=len(pts),effective_extra_seconds=float(after[-1]-before[-1]),max_motion_coordinate_transfer_difference=np.max(abs(error),axis=0).tolist(),exposure_steps=shot['exposure_pts_offsets']))
    tl.assert_support_separation(z['pts'],target['max_readout'])
    target['source_exposure_clock']=dict(path=str(a.clock.resolve()),sha256=p.digest(a.clock),source_PTS_preserved=True,conditional_working_clock=True)
    receipt=dict(schema='rinkspace.prepared_exposure_clock.v1',parent_prepared=str(a.prepared),parent_state=str(a.state),parent_state_sha256=p.digest(a.state),motion_source=str(a.motion_source) if a.motion_source else str(a.state),motion_source_sha256=p.digest(a.motion_source or a.state),clock=target['source_exposure_clock'],source_uv_pts_observation_fields_unchanged=True,scene_and_rig_unchanged=True,old_source_numeric_stations_preserved=True,changed_shots=notes,new_camera_parameters=0,new_geometry_parameters=0,transfer_is_approximate=True,fit_started=False)
    saved,mapped=_write_prepared(out,z,result['stations'],target['shots'],result,target,mapping,model,receipt,link_input=a.prepared/'point_observations.npz')
    if 'spatial_refinement_preparation' in m:saved['spatial_refinement_preparation']=m['spatial_refinement_preparation']
    else:saved.pop('spatial_refinement_preparation',None)
    p.write(out/'joint_manifest.json',saved);p.write(out/'exposure_clock_preparation.json',receipt)
    check=p.npz(out/'initial_pixels.npz');p.validate_snapshot(check,model.initial_state.layout,mapped,saved,z)
    for key in ('rig','scene_values','board','layout'):
        if not np.array_equal(check[key],v[key]):raise ValueError('Exposure migration changed '+key)
    print(json.dumps(dict(output=str(out),point_input_sha256=saved['point_input_sha256'],changed_shots=notes,maximum_readout_support=tl.max_readout())),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','state','clock','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--motion-source',type=Path);main(parser.parse_args())
