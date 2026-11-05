"""Build new display-only runner comparison; preserve old player assets."""
from pathlib import Path
import copy, hashlib, json, os, sys, time
import numpy as np

P=Path('D:/OneDrive/codex_projs/RinkSpace');D=Path('D:/CodexData/RinkSpace_Data')
BASE=D/'runs/body_s01_20261029';OLD=BASE/'blade_orientation_1105';OUT=BASE/'blade_support_profile_1105'
sys.path[:0]=[str(P),str(P/'src')]
os.environ['RINKSPACE_BODY_NATIVE_LIBRARY']=str(D/'runs/body_perf_1103/native_cpu/rinkspace_core.dll')
from calibration.program_tracking.body_tracking.blade_projection import SavedBladeProjection
from calibration.program_tracking.body_tracking.skate_physics import fk_state,batch_features
from calibration.program_tracking.body_tracking.skate_template import profile_height
from rinkspace.viewer3d.blade_review import BladeReview
from rinkspace.viewer3d.data import ResultCatalog
from rinkspace.viewer3d.prepare import prepare_variant


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def serial(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    raise TypeError(type(v).__name__)
def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=serial)+'\n',encoding='utf-8')
def guard(path):
    path=Path(path).resolve()
    if len(str(path).encode('utf-16-le'))//2>=260 or any(len(x.encode('utf-16-le'))//2>255 for x in path.parts):raise ValueError(path)
    return path
def angle(a,b):return float(np.degrees(np.arccos(np.clip(abs(a@b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-16),-1,1))))


def main():
    started=time.perf_counter();asset=read(OLD/'player_blade_review.json');old_asset=copy.deepcopy(asset)
    assembly=read(BASE/'blade_observability_1105/blade_observability_audit.json')['factor_assembly'];defs={}
    for label,vid in [('A','continuous_source'),('B','continuous_flight'),('C6','continuous_refined'),('local_281','local_281')]:
        c=assembly['candidates'][label];defs[vid]=dict(prefix=label,stage=c['stage_root'],parameter=c['input_files'][-1]['path'],result=c['result']['path'])
    for folder,vid,prefix in [('entry_81_2_82_0','boot_axis_entry','boot_entry'),('support_87_6_90_0','boot_axis_support','boot_support')]:
        s=read(OLD/folder/'run_summary.json');defs[vid]=dict(prefix=prefix,stage=str(OLD/folder),parameter=s['parameter'],result=s['result'])
    result_summary=read(OUT/'runner_support_run/run_summary.json')
    defs['runner_support']=dict(prefix='runner',stage=str(OUT/'runner_support_run'),parameter=result_summary['parameter'],result=result_summary['result'])
    with np.load(OLD/'player_blade_projection.npz',allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    old_keys=set(arrays);pts=arrays['pts'];assert len(pts)==951 and np.array_equal(np.diff(pts),np.full(950,256))
    support=(pts>=round(86.8*12800))&(pts<=round(92.46*12800));assert support.sum()==284
    ix=np.arange(len(pts))[:,None];feet=np.arange(2)[None];profiles={'pts':pts.copy()};summaries={};runner_uv=None;outside_errors={};parent_result=defs['boot_axis_entry']['result']
    result_keys=('P_world_mid','Q_world_mid','local_R_mid','native22_world_mid','skate_R_world_mid','skate_origin_world_mid')
    with np.load(parent_result,allow_pickle=False) as z:parent_arrays={k:z[k].copy() for k in result_keys}
    queries={}
    for vid,definition in defs.items():
        query=SavedBladeProjection(P,definition['stage'],definition['parameter']);queries[vid]=query
        np.testing.assert_array_equal(query.pts,pts)
        binding=query.binding;packet=binding.packet();p=packet['parameters'];ref=query.reference
        state=fk_state(ref,ref.at(query.times),np.zeros((len(pts),69)))
        features=batch_features(binding,state['R'],state['O']);height=features['height'];q=features['local_point']
        eligible=set(packet['template']['support_feature_ids']['glide'])
        runner_ids=np.asarray([i for i,name in enumerate(features['ids']) if name in eligible],int)
        pick_ids=np.asarray([i for i,semantic in enumerate(features['semantics']) if semantic=='toe_pick'],int)
        assert len(runner_ids)>0 and all(features['semantics'][i]=='runner' for i in runner_ids) and len(pick_ids)>0
        rb=runner_ids[np.argmin(height[...,runner_ids],axis=-1)];pb=pick_ids[np.argmin(height[...,pick_ids],axis=-1)]
        h_runner=height[ix,feet,rb];h_pick=height[ix,feet,pb];q_runner=q[ix,feet,rb]
        xrear=-p.runner_rear_extension_m;xfront=p.sole_length_m-p.runner_front_inset_m
        xs=np.array([xrear,(xrear+xfront)/2,xfront]);stations=np.c_[xs,np.zeros(3),profile_height(xs,p)]
        station_world=(state['O'][:,:,None]+np.einsum('nfij,kj->nfki',state['R'],stations))*binding.ratio
        station_h=station_world[...,2];edge_h=station_h-np.abs(state['R'][...,2,1,None])*p.blade_width_m*binding.ratio/2
        pitch=np.degrees(np.arctan2(state['R'][...,2,0],np.hypot(state['R'][...,0,0],state['R'][...,1,0])))
        roll=np.degrees(np.arctan2(state['R'][...,2,1],state['R'][...,2,2]));fraction=(q_runner[...,0]-xrear)/(xfront-xrear)
        endpoint=np.zeros((len(pts),2),np.int8);tol=64*np.finfo(float).eps*max(1.,abs(xrear),abs(xfront))
        endpoint[np.abs(q_runner[...,0]-xrear)<=tol]=-1;endpoint[np.abs(q_runner[...,0]-xfront)<=tol]=1
        raw_h=height;raw_semantics=list(features['semantics']);raw_ids=list(features['ids'])
        with np.load(definition['result'],allow_pickle=False) as z:
            result_pts=z['pts'] if 'pts' in z else z['source_pts'];np.testing.assert_array_equal(result_pts,pts)
            np.testing.assert_allclose(state['R'],z['skate_R_world_mid'],rtol=0,atol=2e-10)
            np.testing.assert_allclose(state['O']*binding.ratio,z['skate_origin_world_mid'],rtol=0,atol=2e-10)
            np.testing.assert_allclose(station_world,z['blade_material_world_mid'],rtol=0,atol=2e-10)
            if 'skate_upper_world_mid' in z:
                raw_h=np.concatenate([raw_h,np.min(z['skate_upper_world_mid'][...,2],axis=-1)[...,None]],axis=-1)
                raw_semantics.append('solid_guard');raw_ids.append('upper_shell_guard')
            if vid=='runner_support':
                outside=~support
                for key in result_keys:
                    outside_errors[key]=dict(array_equal=bool(np.array_equal(z[key][outside],parent_arrays[key][outside])),max_abs_error=float(np.max(abs(z[key][outside]-parent_arrays[key][outside]))))
                blade=z['blade_material_world_mid'];axis=blade[:,:,2]-blade[:,:,0];axis/=np.linalg.norm(axis,axis=-1,keepdims=True)
                rates=np.full((len(pts),2),np.nan);rates[1:]=np.arccos(np.clip(np.sum(axis[1:]*axis[:-1],axis=-1),-1,1))/.02
                uv_points=np.r_[stations,packet['adapter'][[0,7]].mean(0)[None],packet['adapter'][[1,8]].mean(0)[None]]
                runner_uv=np.stack([query.project_points(pts,foot,uv_points)['uv'] for foot in (0,1)],axis=1)
                arrays['runner_blade_uv']=runner_uv[:,:,:3];arrays['runner_boot_uv']=runner_uv[:,:,3:];arrays['runner_axis_rate_rad_s']=rates
        raw_best=np.argmin(raw_h,axis=-1);raw_lowest=np.asarray(raw_semantics)[raw_best]
        prefix=definition['prefix'];new_arrays=dict(pitch_degrees=pitch,roll_degrees=roll,runner_min_x_fraction=fraction,station_height_m=station_h,runner_minus_pick_m=h_runner-h_pick)
        for suffix,value in new_arrays.items():arrays[prefix+'_'+suffix]=value
        extra=dict(new_arrays,runner_min_height_m=h_runner,toe_pick_min_height_m=h_pick,station_lower_edge_height_m=edge_h,whole_runner_endpoint=endpoint,runner_min_local_point_solver=q_runner,runner_feature=np.asarray(features['ids'])[rb],raw_lowest_semantic=raw_lowest,raw_lowest_feature=np.asarray(raw_ids)[raw_best],raw_lowest_height_m=np.min(raw_h,axis=-1))
        for suffix,value in extra.items():profiles[prefix+'_'+suffix]=value
        foot=1;ep=endpoint[support,foot]
        summaries[vid]=dict(frames=951,support_seconds=[86.8,92.46],support_foot=foot,support_exposures=int(support.sum()),pitch_degrees_min_q25_median_q75_max=np.quantile(pitch[support,foot],[0,.25,.5,.75,1]),roll_degrees_min_median_max=np.quantile(roll[support,foot],[0,.5,1]),whole_runner_rear_endpoint_frames=int(np.sum(ep==-1)),whole_runner_front_endpoint_frames=int(np.sum(ep==1)),runner_interior_frames=int(np.sum(ep==0)),raw_toe_pick_lowest_frames=int(np.sum(raw_lowest[support,foot]=='toe_pick')),runner_min_height_m_min_median_max=np.quantile(h_runner[support,foot],[0,.5,1]),runner_minus_pick_m_min_median_max=np.quantile((h_runner-h_pick)[support,foot],[0,.5,1]),station_height_m_min_median_max=np.quantile(station_h[support,foot],[0,.5,1],axis=0),station_lower_edge_height_m_min_median_max=np.quantile(edge_h[support,foot],[0,.5,1],axis=0),source_pts_equal=True,geometry_matches_saved=True)
        print(json.dumps(dict(event='profile_ready',variant=vid,frames=951,support_frames=284,front=summaries[vid]['whole_runner_front_endpoint_frames'],rear=summaries[vid]['whole_runner_rear_endpoint_frames'])),flush=True)
    assert all(v['array_equal'] for v in outside_errors.values())
    parent=queries['boot_axis_entry'];runner=queries['runner_support']
    np.testing.assert_array_equal(parent.parameter['x'][:17],runner.parameter['x'][:17])
    assert parent.camera_info==runner.camera_info
    metrics={}
    def metric(pt,foot):return metrics.setdefault((pt,foot),dict(variant='runner_support',pts=int(pt),foot=int(foot),boot_axis_error_deg=None,conditional_blade_axis_error_deg=None))
    lookup={int(pt):i for i,pt in enumerate(pts)}
    for row in asset['source_boot_axes']:
        pt=int(row['pts']);foot=int(row['foot']);i=lookup[pt];m=metric(pt,foot)
        m.update(boot_axis_error_deg=angle(runner_uv[i,foot,4]-runner_uv[i,foot,3],np.asarray(row['toe_uv'])-row['heel_uv']),source_role=row.get('role'),source_uncertainty='existing finite boot contours; not certified blade material or contact')
    conditional_pts={(int(row['pts']),int(row['foot'])) for row in asset['axis_metrics'] if row['variant']=='boot_axis_entry' and row.get('conditional_blade_axis_error_deg') is not None}
    for pt,foot in sorted(conditional_pts):
        ep={r['blade_endpoint']:r for r in asset['source_regions'] if r['pts']==pt and r.get('foot')==foot and r.get('kind')=='blade_endpoint_region' and r.get('visibility')=='visible'}
        if not {'front','rear'}.issubset(ep):continue
        i=lookup[pt];m=metric(pt,foot);m.update(conditional_blade_axis_error_deg=angle(runner_uv[i,foot,2]-runner_uv[i,foot,0],np.asarray(ep['front']['centre_uv'])-ep['rear']['centre_uv']),source_uncertainty='existing conditional finite endpoint-region centres; not certified material')
    asset['variants']['runner_support']='runner';asset['axis_metrics'].extend(metrics.values());asset['projection_npz']='player_blade_projection.npz'
    asset['profile_semantics']=dict(pitch='atan2(R_zx,hypot(R_xx,R_yx)); positive toe axis upward',roll='atan2(R_zy,R_zz)',station_height_m='fixed template rear/arithmetic-x-mid/front centreline height above raw nominal reference plane; conditional metres, not contact',runner_min_x_fraction='glide runner arcs only; 0=rear 1=front, toe-pick excluded',runner_minus_pick_m='minimum glide-runner height minus minimum of all toe-pick groups; conditional metres',exact_material_contact='unknown',no_new_source_observations=True)
    for key in ('source_boot_axes','source_regions','source_visibility','source_steel_segments'):assert asset[key]==old_asset[key]
    projection_path=OUT/'player_blade_projection.npz';profile_path=OUT/'profile_arrays.npz';review_path=OUT/'player_blade_review.json'
    np.savez_compressed(projection_path,**arrays);np.savez_compressed(profile_path,**profiles);write(review_path,asset)
    with np.load(OLD/'player_blade_projection.npz',allow_pickle=False) as old,np.load(projection_path,allow_pickle=False) as new:
        for key in old_keys:np.testing.assert_array_equal(new[key],old[key])
    viewer_dir=P/'events/ankara_2026/programmes/okada_mei_sp';config_path=guard(viewer_dir/'viewer3d_runner_support_1105.json');launcher=guard(P/'launchers/START_3D_RUNNER_SUPPORT.cmd')
    if config_path.exists() or launcher.exists():raise FileExistsError('Preserve existing runner config/launcher')
    config=read(viewer_dir/'viewer3d_boot_axis_1105.json');old_variants=copy.deepcopy(config['variants'])
    config['variants'].append(dict(id='runner_support',label='runner支撑有限先验（8步，仍有端部支撑，未验收）',kind='fullbody',result=result_summary['result'],shell_topology='saved',body_reference_label='exmr_hand_nofoot',required_saved_diagnostics=['com_world_mid','blade_material_world_mid','skate_witness_world_mid'],body_refinement_intervals_seconds=[[81.2,82.0],[86.8,92.46]]))
    config.update(blade_review_asset=str(review_path),comparison_variants=['boot_axis_entry','runner_support'],initial_source_pts=1121792,session_id='runner_support_1105',review_status='One bounded runner-support prior candidate; endpoint support remains and physical/source accuracy is unaccepted; previous default retained')
    assert config['default_variant']=='local_281' and config['variants'][:-1]==old_variants
    write(config_path,config)
    launcher_text=(P/'launchers/START_3D_BOOT_AXIS.cmd').read_text(encoding='utf-8').replace('viewer3d_boot_axis_1105.json','viewer3d_runner_support_1105.json').replace('-Variant boot_axis_entry','-Variant runner_support')
    launcher.write_bytes(launcher_text.replace('\r\n','\n').replace('\n','\r\n').encode('utf-8'))
    review=BladeReview(review_path,time_base=[1,12800]);assert len(review.index)==951 and len(review.models)==7
    for vid in defs:
        for pt in pts:assert review.at(vid,int(pt))['model'] is not None
    manifest=dict(schema='rinkspace.runner-player-preparation.v1',no_fit=True,no_inference=True,no_gui=True,no_video=True,accuracy_accepted=False,user_accepted=False,source_time_base=[1,12800],source_frames=951,source_first_last_pts=pts[[0,-1]],support_exposures=284,source_sections_exactly_preserved=True,old_projection_arrays_exactly_preserved=True,old_variant_count=6,new_variant_count=7,config_old_variants_preserved=True,config_old_default='local_281',launcher_explicit_variant='runner_support',definitions=defs,profile_summaries=summaries,outside_core_state_equality=outside_errors,shared_first_17_exactly_equal=True,camera_exactly_equal=True,new_axis_metric_rows=len(metrics),config=str(config_path),launcher=str(launcher),player_review=dict(path=str(review_path),sha256=sha(review_path)),player_projection=dict(path=str(projection_path),sha256=sha(projection_path)),profile_arrays=dict(path=str(profile_path),sha256=sha(profile_path)),station_semantics=asset['profile_semantics'])
    manifest_path=OUT/'player_runner_preparation.json';write(manifest_path,manifest)
    print(json.dumps(dict(event='prepare_new_variant_cache',config=str(config_path),manifest=str(manifest_path))),flush=True)
    catalog=ResultCatalog(project=P,data_root=D,config=config_path);np.testing.assert_array_equal(catalog.pts,pts)
    cache=prepare_variant(catalog,'runner_support')
    metadata=read(cache/'metadata.json');assert metadata['frames']==951 and metadata['accuracy_accepted'] is False
    with np.load(cache/'state.npz',allow_pickle=False) as z:np.testing.assert_array_equal(z['pts'],pts)
    manifest.update(cache=dict(path=str(cache),metadata=str(cache/'metadata.json'),frames=951,source_pts_equal=True,only_new_variant_prepared=True),preparation_seconds=time.perf_counter()-started)
    write(manifest_path,manifest)
    print(json.dumps(dict(event='runner_player_ready',config=str(config_path),launcher=str(launcher),cache=str(cache),manifest=str(manifest_path),profile_arrays=str(profile_path),seconds=manifest['preparation_seconds'])),flush=True)

if __name__=='__main__':main()
