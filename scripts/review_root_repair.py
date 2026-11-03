"""Audit the saved root repair without inference, optimization or GUI."""
from pathlib import Path
import argparse,json,sys
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from calibration.program_tracking.body_tracking.rebuild_prepare import load,digest
from calibration.program_tracking.body_tracking.fullbody_round import write_json
from calibration.program_tracking.body_tracking.footwear_pixel_factors import nearest
from calibration.program_tracking.body_tracking.footwear_fit import sample_polyline


def stats(x):
    x=np.asarray(x);x=x[np.isfinite(x)]
    return dict(count=int(len(x)),median=float(np.median(x)),p95=float(np.quantile(x,.95)),maximum=float(x.max())) if len(x)else dict(count=0)


def selected(run,clip,arm):
    folder=run/clip/'root_repair'/arm
    if (folder/'reuse.json').exists():return run/clip/'root_repair/A'
    pose=folder.with_name(arm+'_pose')
    if (pose/'metrics.json').exists() and json.loads((pose/'metrics.json').read_text())['run']['run_policy']['accepted']>0:return pose
    return folder


def metrics(z,inp,obs,ledger,clip):
    np.testing.assert_array_equal(z['pts'],inp['pts']);np.testing.assert_array_equal(z['source_pts'],inp['pts'])
    np.testing.assert_array_equal(z['G_world_mid'][:,:2],z['P_world_mid'][:,:2]);np.testing.assert_array_equal(z['G_world_mid'][:,2],0.)
    P=z['P_world_mid'];t=z['pts']/12800.;dt=np.diff(t);velocity=np.diff(P,axis=0)/dt[:,None]
    acceleration=np.diff(velocity,axis=0)/((dt[1:]+dt[:-1])/2)[:,None]
    error=np.linalg.norm(z['COCO_uv']-inp['source_uv'],axis=2)
    Q=z['Q_world_mid'];angle=np.rad2deg(np.linalg.norm(Rotation.from_matrix(Q[1:]@Q[:-1].transpose(0,2,1)).as_rotvec(),axis=1))
    curves=[]
    for row in ledger['rows']:
        if row['clip']!=clip:continue
        i=int(np.flatnonzero(z['pts']==row['pts'])[0]);uv=z['equipment_uv'][i,row['foot']]
        points=sample_polyline(row['uv'],24);ds=[nearest(points,line)[0]for line in (uv[14:31],uv[31:48])]
        d=min(ds,key=lambda a:a@a)
        curves.append(dict(pts=int(row['pts']),seconds=row['pts']/12800.,physical_foot=row['foot'],curve_error_px=stats(d),sigma_px=row['sigma_px'],role=row['role'],blade_side_unknown=True))
    out=dict(path_xy_m=float(np.linalg.norm(np.diff(P[:,:2],axis=0),axis=1).sum()),range_xy_m=np.ptp(P[:,:2],axis=0).tolist(),
        speed_xy_mps=stats(np.linalg.norm(velocity[:,:2],axis=1)),acceleration_xy_mps2=stats(np.linalg.norm(acceleration[:,:2],axis=1)),
        body_image_px=stats(error[inp['source_qualified']]),body_fit_px=stats(error[obs['use']]),
        ankle_proxy_px=stats(error[:,15:17][inp['source_qualified'][:,15:17]]),
        maximum_root_angle_step_deg=float(angle.max()),minimum_raw_witness_height_m=float(z['skate_witness_world_mid'][:,:,2].min()),
        minimum_semantic_margin_m=float(z['skate_clearance_m_mid'].min()),maximum_cavity_violation_m=float(z['foot_cavity_max_violation_m_mid'].max()),
        support_exposures=[dict(pts=int(z['pts'][i]),seconds=z['pts'][i]/12800.,raw_height_m=float(h))for i,h in enumerate(z['skate_support_gap_m_mid'])if np.isfinite(h)],
        heldout_shoe_curves=curves,unknown_contact_foot_frames=int(np.count_nonzero(z['skate_contact_mode_mid']=='unknown')),
        state_pts_consistent=True,pelvis_projection_is_world_P_xy=True,independent_3d_truth=False)
    if clip=='spin_s05':
        m=(t>=82)&(t<=92.9);q=P[m,:2];out['landed_spin_82_92p9']=dict(range_xy_m=np.ptp(q,axis=0).tolist(),path_xy_m=float(np.linalg.norm(np.diff(q,axis=0),axis=1).sum()))
    return out


def review(run):
    run=Path(run);out=run/'pelvis_drift_diagnostic';out.mkdir(exist_ok=True)
    ledger=json.loads((run/'heldout_shoe_review.json').read_text());report={};arrays={}
    for clip in ('jump_s02','step_s01','spin_s05'):
        inp=load(run/clip/'input.npz');obs=load(run/clip/'observations.npz');arms={};arrays[clip]={}
        for arm in ('A','B'):
            folder=selected(run,clip,arm);old=load(run/clip/arm/'fitted/result.npz');new=load(folder/'fitted/result.npz')
            oldm=metrics(old,inp,obs,ledger,clip);newm=metrics(new,inp,obs,ledger,clip)
            root_run=json.loads((run/clip/'root_repair/A'/'metrics.json').read_text())if clip=='spin_s05'else json.loads((run/clip/'root_repair'/arm/'metrics.json').read_text())
            pose_folder=run/clip/'root_repair'/(arm+'_pose');pose_run=json.loads((pose_folder/'metrics.json').read_text())if (pose_folder/'metrics.json').exists()else None
            arms[arm]=dict(previous=oldm,candidate=newm,result=str(folder/'fitted/result.npz'),result_sha256=digest(folder/'fitted/result.npz'),
                image_median_change_fraction=newm['body_image_px']['median']/oldm['body_image_px']['median']-1,
                image_regression=True if newm['body_image_px']['median']>oldm['body_image_px']['median']else False,
                root_run=root_run['run']['run_policy'],pose_run=None if pose_run is None else pose_run['run']['run_policy'],
                reused_A=clip=='spin_s05' and arm=='B',user_accepted=False,physical_accuracy_accepted=False)
            if arm=='A':arrays[clip]=dict(old=old,new=new)
        report[clip]=arms
    result=dict(schema='rinkspace.root-repair-review.v1',clips=report,additional_support_rows=0,
        diagnosis='Framewise apparent-body-size depth was fitted into the root reference; correction-only regularization preserved it. Camera/cache/PTS bindings tested consistent; external world accuracy remains conditional.',
        correction='Conditional proxy/runner or accepted-S01 root initialization; actual-root acceleration; complete-clip root correction, then bounded pose-only pass. Same finite ice policy, no fixed spin centre or speed cap.',
        interpretation='Ankle proxy, runner witness and source blade curve remain separate. Raw model heights are not contact truth. Smaller root path and image cost do not prove physics.',
        remaining='Image projection regresses, inherited root SO(3) jumps remain, source support is sparse/uncertain; no candidate passes complete quality review. No automatic additional fitting.',
        heldout_unknown=ledger['unknown'])
    write_json(out/'root_repair_review.json',result)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(17,4.4))
    for ax,clip in zip(axes[:3],report):
        for key,color in [('old','#d36b2b'),('new','#178559')]:
            P=arrays[clip][key]['P_world_mid'];ax.plot(P[:,0],P[:,1],color=color,lw=1,label=key)
        ax.set_title(clip);ax.set_xlabel('conditional world x / m');ax.set_ylabel('world y / m');ax.axis('equal');ax.grid(alpha=.3);ax.legend()
    ax=axes[3];clip='spin_s05'
    for key,color in [('old','#d36b2b'),('new','#178559')]:
        z=arrays[clip][key];m=(z['pts']/12800.>=82)&(z['pts']/12800.<=92.9);P=z['P_world_mid'][m];ax.plot(P[:,0],P[:,1],color=color,lw=1,label=key)
    ax.set_title('Spin 82.00-92.90s');ax.set_xlabel('conditional world x / m');ax.axis('equal');ax.grid(alpha=.3);ax.legend()
    fig.tight_layout();fig.savefig(out/'root_projection_old_new.png',dpi=150);plt.close(fig)
    return result


def verify_parameters(run):
    from calibration.program_tracking.body_tracking.action_integration import initialize_clip,ClipReference
    from calibration.program_tracking.body_tracking.skate_integration import restore_checkpoint_layers
    from calibration.program_tracking.body_tracking.skate_checkpoint import validate_checkpoint_identity
    from rinkspace.viewer3d.data import ResultCatalog
    run=Path(run);base=run.parent;programme=ROOT/'events/ankara_2026/programmes/okada_mei_sp';report={}
    evidence=json.loads((programme/'action_evidence_1103.json').read_text())['clips']
    for clip in ('jump_s02','step_s01','spin_s05'):
        e=evidence[clip]
        shared,camera,info=initialize_clip(ROOT,base,run/clip,run/'source_cache'/f'{clip}.native50.npz',
            run/clip/'prior/camera_prior.npz',e['shot_id'],heldout_pts=e.get('heldout_pts',[]))
        catalog=ResultCatalog(project=ROOT,data_root=base.parents[1],config=programme/f'viewer3d_action_{clip}_1103.json');checks={}
        for arm in ('A','B'):
            folder=selected(run,clip,arm);actual_arm='A' if clip=='spin_s05' else arm
            suffix='_root_pose_1103.npz' if folder.name.endswith('_pose')else '_root_1103.npz'
            params=load(programme/f'action_{clip}_{actual_arm}{suffix}');validate_checkpoint_identity(params,shared.fixed_checkpoint_identity)
            initial_file=ROOT/str(params['initial_state_file']);assert digest(initial_file)==str(params['initial_state_sha256'])
            ref=ClipReference(shared,run/clip,shared.p,load(initial_file));restore_checkpoint_layers(ref,params,include_primary=True)
            saved=load(folder/'fitted/result.npz');replay=ref.at(ref.times)
            errors=dict(P_m=float(np.max(abs(replay['P']*info['length_export_ratio']-saved['P_world_mid']))),
                Q=float(np.max(abs(replay['Q']-saved['Q_world_mid']))),local_R=float(np.max(abs(replay['local_R']-saved['local_R_mid']))))
            assert max(errors.values())<1e-11
            cache_path,identity=catalog.cache_path(arm+'_root');cache=load(cache_path/'state.npz');metadata=json.loads((cache_path/'metadata.json').read_text())
            assert metadata['identity']==identity
            for key,target in [('pts','pts'),('P','P_world_mid'),('G','G_world_mid'),('gear','gear_world_mid')]:np.testing.assert_array_equal(cache[key],saved[target])
            np.testing.assert_allclose(cache['coco'],saved['COCO_world_mid'],rtol=0,atol=1e-11)
            checks[arm]=dict(parameter_fixed_identity_valid=True,initial_reference_sha256_valid=True,replay_maximum_error=errors,
                cache=str(cache_path),cache_state_and_pts_match=True,common_body_shoe_P_G_coco_match=True)
        report[clip]=checks
    manifest=json.loads((base.parents[1]/'maintenance/integration_s01_20261103/preserved_baseline.json').read_text())
    immutable={path:digest(path)==record['sha256']for path,record in manifest.items()if Path(path).name!='viewer3d_fullbody_1102.json'}
    assert len(immutable)==5 and all(immutable.values())
    main_config=json.loads((programme/'viewer3d_fullbody_1102.json').read_text());assert main_config['default_variant']=='skate_s01_1103'
    result=dict(clips=report,immutable_satisfied_baseline=immutable,main_S01_default='skate_s01_1103',
        main_config_note='The early preservation manifest predates the previous authorized addition of the S01 integration comparison; its JSON hash is not an immutable-baseline assertion.')
    write_json(run/'pelvis_drift_diagnostic/parameter_cache_validation.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--verify-parameters',action='store_true');a=p.parse_args();r=review(a.run)
    if a.verify_parameters:verify_parameters(a.run)
    for c,v in r['clips'].items():
        for arm,m in v.items():print(c,arm,json.dumps(dict(old_path=m['previous']['path_xy_m'],new_path=m['candidate']['path_xy_m'],old_image_median=m['previous']['body_image_px']['median'],new_image_median=m['candidate']['body_image_px']['median'])))
