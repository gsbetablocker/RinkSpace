"""Read saved whole-event candidates; group source checks without solving."""
from pathlib import Path
import argparse,json,sys
import numpy as np
from scipy.spatial.transform import Rotation
PROJECT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(PROJECT),str(PROJECT/'src')]
from calibration.program_tracking.body_tracking.rebuild_prepare import load,digest
from calibration.program_tracking.body_tracking.footwear_pixel_factors import nearest
from calibration.program_tracking.body_tracking.footwear_fit import sample_polyline
from calibration.program_tracking.body_tracking.fullbody_round import write_json


def quantiles(values):
    v=np.asarray(values);v=v[np.isfinite(v)]
    return dict(count=int(v.size),median=float(np.median(v)),p95=float(np.quantile(v,.95)),maximum=float(v.max()))if v.size else dict(count=0,median=None,p95=None,maximum=None)


def review(root,render=False):
    root=Path(root);ledger=json.loads((root/'heldout_shoe_review.json').read_text(encoding='utf-8'));report={}
    for clip in ('jump_s02','step_s01','spin_s05'):
        folder=root/clip
        if not all((folder/arm/'fitted/result.npz').is_file()for arm in ('A','B')):continue
        inp=load(folder/'input.npz');obs=load(folder/'observations.npz');prior=load(folder/'initial.npz');variants={};arrays={}
        for arm in ('A','B'):
            z=load(folder/arm/'fitted/result.npz');params=load(PROJECT/'events/ankara_2026/programmes/okada_mei_sp'/('action_'+clip+'_'+arm+'_1103.npz'));arrays[arm]=z
            np.testing.assert_array_equal(z['pts'],inp['pts']);np.testing.assert_array_equal(z['source_pts'],inp['pts'])
            np.testing.assert_array_equal(z['initialization_sha256'],digest(folder/'initial.npz'))
            np.testing.assert_array_equal(params['x'][:17],load(PROJECT/'events/ankara_2026/programmes/okada_mei_sp/skate_s01_1103.npz')['x'][:17])
            Q=z['Q_world_mid'];np.testing.assert_allclose(Q@Q.transpose(0,2,1),np.broadcast_to(np.eye(3),Q.shape),atol=1e-10)
            L=z['local_R_mid'];np.testing.assert_allclose(L@L.swapaxes(-1,-2),np.broadcast_to(np.eye(3),L.shape),atol=1e-10)
            raw=z['skate_witness_world_mid'][:,:,2];clear=z['skate_clearance_m_mid'];idx=np.unravel_index(int(np.argmin(raw)),raw.shape)
            distances=np.linalg.norm(z['COCO_uv']-inp['source_uv'],axis=2);valid=inp['source_qualified']
            initial_steps=np.rad2deg(np.linalg.norm(Rotation.from_matrix(prior['Q'][1:]@prior['Q'][:-1].transpose(0,2,1)).as_rotvec(),axis=1))
            final_steps=np.rad2deg(np.linalg.norm(Rotation.from_matrix(Q[1:]@Q[:-1].transpose(0,2,1)).as_rotvec(),axis=1));j=int(final_steps.argmax())
            grouped=[]
            for row in ledger['rows']:
                if row['clip']!=clip:continue
                f=int(np.flatnonzero(z['pts']==row['pts'])[0]);foot=row['foot'];points=sample_polyline(row['uv'],24)
                uv=z['equipment_uv'][f,foot];candidates=[uv[14:31],uv[31:48]]
                ds=[nearest(points,line)[0]for line in candidates];side=int(np.argmin([v@v for v in ds]));d=ds[side]
                grouped.append(dict(pts=int(row['pts']),source_seconds=row['pts']/12800.,physical_foot=foot,side_candidate=side,source_side_confirmed=False,source_points_in_curve=len(row['uv']),resampled_curve_points=24,curve_statistics=quantiles(d),sigma_px=row['sigma_px'],role=row['role']))
            run=json.loads((folder/arm/'metrics.json').read_text(encoding='utf-8'))
            variants[arm]=dict(source_exposures=len(z['pts']),native_cadence_pts=np.unique(np.diff(z['pts'])).tolist(),state_pts_equal=True,shared_shape_and_attachment_equal_to_s01=True,initialization_sha256=str(z['initialization_sha256']),body_observation_error_px=quantiles(distances[valid]),body_fit_error_px=quantiles(distances[obs['use']]),body_heldout_error_px=quantiles(distances[valid&~obs['use']]),ankle_detector_proxy_error_px=quantiles(distances[:,15:17][valid[:,15:17]]),minimum_raw_model_witness_height_m=float(raw.min()),minimum_semantic_margin_m=float(clear.min()),worst_raw_height=dict(pts=int(z['pts'][idx[0]]),source_seconds=float(z['pts'][idx[0]]/12800.),physical_foot=int(idx[1]),source_phase=str(z['source_action_phase_mid'][idx])),support_raw_height_m=quantiles(z['skate_support_gap_m_mid']),source_support_gaps=[dict(pts=int(z['pts'][i]),source_seconds=float(z['pts'][i]/12800.),raw_height_m=float(h))for i,h in enumerate(z['skate_support_gap_m_mid'])if np.isfinite(h)],maximum_root_step_deg=float(final_steps.max()),initial_maximum_root_step_deg=float(initial_steps.max()),root_step_source_pts=z['pts'][j:j+2].tolist(),maximum_cavity_breach_m=float(z['foot_cavity_max_violation_m_mid'].max()),contact_unknown_foot_frames=int(np.count_nonzero(z['skate_contact_mode_mid']=='unknown')),heldout_blade_curves=grouped,run_stop=run['run']['stop'],objective_parts=run['run']['parts'],action=run['action_diagnostics'],user_accepted=False,numerical_convergence=run['run']['numerical_convergence'])
        np.testing.assert_array_equal(arrays['A']['initialization_sha256'],arrays['B']['initialization_sha256'])
        delta=dict(root_position_maximum_m=float(np.linalg.norm(arrays['A']['P_world_mid']-arrays['B']['P_world_mid'],axis=1).max()),shoe_edge_maximum_m=float(np.linalg.norm(arrays['A']['skate_edge_world_mid']-arrays['B']['skate_edge_world_mid'],axis=-1).max()))
        report[clip]=dict(variants=variants,AB_difference=delta,selection='preserve A as comparison default; B remains an unaccepted candidate',interpretation='raw height / semantic margin are reference-model diagnostics, not actual contact truth; source image errors never replace ice review')
        if render:render_curves(root,clip,ledger,arrays)
    result=dict(schema='rinkspace.action-ab-review.v1',clips=report,heldout_unknown=ledger['unknown'],measurement_status=ledger['measurement_status'],independent_3d_truth=False,flight_COM=dict(status='not evaluated on real clips',missing='admitted segment mass distribution and segment mass-centre model; pure mass-weighted COM/ballistic kernel tested only',pelvis_substitute_used=False),spin_policy='source spin phase disables stable-glide factor; no uniform yaw or frozen centre')
    write_json(root/'ab_review.json',result)
    return result


def render_curves(root,clip,ledger,arrays):
    import av
    from PIL import Image,ImageDraw
    contract=json.loads((root/clip/'contract.json').read_text(encoding='utf-8'))
    source=Path(contract['source']['path'])
    if digest(source)!=contract['source']['sha256']:raise ValueError('Source video differs from prepared clip')
    out=root/'review';out.mkdir(exist_ok=True)
    for row in ledger['rows']:
        if row['clip']!=clip:continue
        pts=row['pts'];foot=row['foot'];curves={}
        for arm,z in arrays.items():
            i=int(np.flatnonzero(z['pts']==pts)[0]);curves[arm]=z['equipment_uv'][i,foot,14:]
        points=np.concatenate([np.asarray(row['uv']),*curves.values()]);lo=np.floor(points.min(0)-35).astype(int);hi=np.ceil(points.max(0)+35).astype(int);lo=np.maximum(lo,[0,0]);hi=np.minimum(hi,[1920,1080])
        with av.open(str(source))as c:
            st=c.streams.video[0];c.seek(pts,stream=st,backward=True)
            for frame in c.decode(st):
                if frame.pts!=pts:continue
                im=Image.fromarray(frame.to_ndarray(format='rgb24')).crop((*lo,*hi));break
        panels=[]
        for name in ('source','A','B'):
            panel=im.copy();d=ImageDraw.Draw(panel)
            d.line([tuple(v-lo)for v in np.asarray(row['uv'])],fill=(255,240,0),width=2)
            if name!='source':
                color=(255,100,20)if name=='A'else(10,240,130)
                for line in (curves[name][:17],curves[name][17:]):d.line([tuple(v-lo)for v in line],fill=color,width=1)
            panel=panel.resize((panel.width*3,panel.height*3));title=Image.new('RGB',(panel.width,panel.height+30),'white');title.paste(panel,(0,30));ImageDraw.Draw(title).text((8,8),f'{name} / {pts/12800:.2f}s / native PTS {pts}',fill='black');panels.append(title)
        canvas=Image.new('RGB',(sum(p.width for p in panels),max(p.height for p in panels)),'white');x=0
        for panel in panels:canvas.paste(panel,(x,0));x+=panel.width
        canvas.save(out/f'{clip}_{pts}_shoe_AB.png')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--render',action='store_true');a=p.parse_args();r=review(a.root,a.render)
    for clip,v in r['clips'].items():
        print(json.dumps(dict(clip=clip,variants={arm:dict(raw_minimum_m=q['minimum_raw_model_witness_height_m'],heldout_curves=q['heldout_blade_curves'],root_step_deg=q['maximum_root_step_deg'])for arm,q in v['variants'].items()})))

if __name__=='__main__':main()
