"""Read saved continuous candidates on one frozen ledger/native camera."""
from pathlib import Path
from types import SimpleNamespace
import json,os
import numpy as np
from scipy.spatial.transform import Rotation
from .rebuild_prepare import load,digest
from .absolute_motion_curve import _log
from .absolute_motion_state import evaluate_curve
from .spin_continuous_refit import restore_absolute,serial
from .support_root_placement import restore_parent
from .support_image import SupportImageFactors
from .stage_quality import error_summary
from .fullbody_round import write_json
from .centroidal_physics_diagnostics import centroidal_momentum,second_difference_operator

PROGRAMME=Path('events/ankara_2026/programmes/okada_mei_sp')

def describe(packet,obs):
    pts=packet['source_pts'];np.testing.assert_array_equal(pts,obs['pts']);out={}
    error=np.linalg.norm(packet['COCO_uv']-obs['uv'],axis=-1);sec=pts/12800.
    for name,(a,b) in {'whole':(76,95),'entry':(80.8,82.4),'support_reversal':(84,86),'held':(88,92),'release':(92.4,93.2)}.items():
        window=(sec>=a)&(sec<=b);mask=obs['use']&window[:,None]
        by={}
        for group,ix in [('body',np.arange(17)),('left_lower_body',[11,13,15]),('right_lower_body',[12,14,16])]:
            e=error[:,ix];use=mask[:,ix];source=np.broadcast_to(pts[:,None],e.shape)
            by[group]=error_summary(e[use],source[use])
        out[name]=by
    at={int(p):i for i,p in enumerate(pts)};R=packet['local_R_mid'][[at[round(v*12800)] for v in [89.32,89.36]],20]
    out['non_grip_right_wrist_40ms_degrees']=float(np.rad2deg(np.linalg.norm(_log(R[0].T@R[1]))))
    R=packet['skate_R_world_mid'][[at[round(v*12800)] for v in [85.56,85.58]],1]
    out['support_shoe_85_56_to_85_58_world_omega_rad_s']=(_log(R[1]@R[0].T)/.02).tolist()
    out['raw_detector_81_40_errors_px']=error[at[1041920]].tolist();out['raw_detector_81_40_six_sides_unknown']=[9,10,13,14,15,16]
    q=packet['Q_world_mid'];omega=_log(q[1:]@q[:-1].swapaxes(-1,-2))/.02;stable=(sec[:-1]>=88)&(sec[1:]<=92)
    out['held_root_world_omega_change_rms_rad_s2']=float(np.sqrt(np.mean((np.diff(omega,axis=0)[stable[1:]&stable[:-1]]/.02)**2)))
    return out

class ZeroState:
    active=np.arange(69);d=69;m=1;size=69;seconds=(76,95)
    def basis(self,t,derivative=0):return np.ones((len(np.atleast_1d(t)),1)) if not derivative else np.zeros((len(np.atleast_1d(t)),1))
    def increments(self,x,t):return np.zeros((len(np.atleast_1d(t)),69))

def region_measure(ref,camera,doc):
    body=SimpleNamespace(state=ZeroState(),tau=float(camera.p['rig'][37]))
    p=SimpleNamespace(reference=ref,camera=camera,binding=ref.binding,body=body,optimize_shared=False,size=86)
    rows=SupportImageFactors(p,doc,include_review=True,arc_ids={}).evaluate(np.zeros(69))['metrics']['observations']
    result={}
    for role in ['fit','review']:
        for foot in [0,1]:
            for kind in ['boot_marker_region','blade_midpoint_region','blade_endpoint_region','support_contact_region']:
                chosen=[r for r in rows if r['role']==role and r['foot']==foot and r['kind']==kind]
                result[f'{role}_{kind}_foot{foot}']=dict(outside_region_px=error_summary(np.array([r['outside_region_px'] for r in chosen]),np.array([r['pts'] for r in chosen])),centre_distance_px=error_summary(np.array([r['centre_error_px'] for r in chosen]),np.array([r['pts'] for r in chosen])),source_rows=len(chosen),prediction_geometries=sorted({r.get('prediction_geometry','actual_current_geometry') for r in chosen}),actual_geometry_outside_region_px=error_summary(np.array([r.get('actual_geometry_outside_region_px',r['outside_region_px']) for r in chosen]),np.array([r['pts'] for r in chosen])),actual_geometry_centre_distance_px=error_summary(np.array([r.get('actual_geometry_centre_distance_px',r['centre_error_px']) for r in chosen]),np.array([r['pts'] for r in chosen])))
    return dict(groups=result,rows=rows,region_is_not_probability_covariance=True,review_is_development_not_independent_truth=True)


def compare(project,base,root):
    project,base,root=map(Path,(project,base,root));os.environ['RINKSPACE_BODY_NATIVE_LIBRARY']=str(base.parent/'body_perf_1103/native_cpu/rinkspace_core.dll');obs=load(root/'common/exmr_hand_nofoot/observations.npz');doc=json.loads((root/'source_evidence.json').read_text(encoding='utf-8'))
    variants={'gravity6':(base/'spin_entry_gravity_1104',project/PROGRAMME/'spin_entry_gravity_1104.npz'),'support8':(base/'spin_grasp_1104/support_joint_v2',project/PROGRAMME/'spin_support_foot_1104.npz')}
    result={};packets={}
    for label in ['gravity6','support8','continuous_source','continuous_flight']:
        if label in variants:
            case,param=variants[label];ref,camera,_,_=restore_parent(project,base,case,root/'review_geometry'/label,param)
            path=base/('spin_support_foot_1104' if label=='support8' else 'spin_entry_gravity_1104')/'common/exmr_hand_nofoot/fitted/result.npz'
        else:
            stage='source_stage' if label=='continuous_source' else 'flight_stage';s=json.loads((root/stage/'run_summary.json').read_text(encoding='utf-8'));path=Path(s['result']);ref,camera,_,_=restore_absolute(project,base,root/'review_geometry'/label,Path(s['parameter']))
        packet=load(path);packets[label]=packet;result[label]=dict(result_path=str(path),result_sha256=digest(path),body_and_actual_motion=describe(packet,obs),same_evidence_source_regions=region_measure(ref,camera,doc))
    cfg=json.loads((project/PROGRAMME/'viewer3d_spin_entry_motion_1104.json').read_text(encoding='utf-8'))
    from rinkspace.storage import AssetStore
    store=AssetStore(project)
    for row in cfg['variants']:
        if row['id'] not in ['gvhmr_fixed','exmr_prefoot_old','hand_nofoot_initial']:continue
        path=store.resolve(row['result'],must_exist=True);packet=load(path)
        if np.array_equal(packet['source_pts'],obs['pts']):result[row['id']]=dict(result_path=str(path),body_and_actual_motion=describe(packet,obs),new_endpoint_regions='not recomputed for old network controls; no invented point mapping')
    value=dict(schema='rinkspace.continuous-same-ledger-comparison.v1',source_time_base=[1,12800],observations_sha256=digest(root/'common/exmr_hand_nofoot/observations.npz'),source_document_sha256=digest(root/'source_evidence.json'),source_frames=len(obs['pts']),actual_admitted_body_points=int(obs['use'].sum()),
        variants=result,baseline_attribution='cross_stage_quality.original_A is the new absolute-seed reprojection at stage A, fixed unchanged for B; the gravity6 and support8 same-ledger comparisons are explicitly above, not overwritten by seed improvements',accuracy_accepted=False)
    write_json(root/'same_evidence_comparison.json',value)
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(14,9));colors={'gravity6':'#db8034','support8':'#888888','continuous_source':'#3c83c5','continuous_flight':'#189a6c'}
    for label,p in packets.items():
        source=p['source_pts']/12800.;c=colors[label];axs[0,0].plot(p['P_world_mid'][:,0],p['P_world_mid'][:,1],label=label,color=c)
        omega=_log(p['Q_world_mid'][1:]@p['Q_world_mid'][:-1].swapaxes(-1,-2))/.02;mask=(source[1:]>=84)&(source[1:]<=93.2);axs[0,1].plot(source[1:][mask],omega[mask,2],color=c)
        delta=np.rad2deg(np.linalg.norm(_log(p['local_R_mid'][:-1,20].swapaxes(-1,-2)@p['local_R_mid'][1:,20]),axis=-1));mask=(source[1:]>=89.18)&(source[1:]<=89.54);axs[1,0].plot(source[1:][mask],delta[mask],color=c)
        R=p['skate_R_world_mid'][:,1];w=_log(R[1:]@R[:-1].swapaxes(-1,-2))/.02;mask=(source[1:]>=85.4)&(source[1:]<=85.8);axs[1,1].plot(source[1:][mask],w[mask,2],color=c)
    for ax,title in zip(axs.flat,['Saved pelvis XY (not COM), all951 source exposures','Actual root world omega_z (rad/s)','Non-grip right wrist degrees per native20ms','Support-shoe world omega_z (rad/s)']):ax.set_title(title);ax.grid(alpha=.2)
    axs[0,0].axis('equal');axs[0,0].legend();fig.tight_layout();fig.savefig(root/'continuous_motion_comparison.png',dpi=150);plt.close(fig)
    return value


def native_stills(project,base,root,*,stages=('source_stage','flight_stage')):
    project,base,root=map(Path,(project,base,root));os.environ['RINKSPACE_BODY_NATIVE_LIBRARY']=str(base.parent/'body_perf_1103/native_cpu/rinkspace_core.dll');target=root/('native_comparison' if len(stages)==2 else 'native_source_stage');target.mkdir(exist_ok=False)
    from .spin_support_review import project_mesh
    from .native_mesh_overlay import paint
    from rinkspace.viewer3d.source import SourceVideo
    from PIL import Image,ImageFont
    refs=[];labels=[];packets=[]
    old,camera,_,_=restore_parent(project,base,base/'spin_entry_gravity_1104',target/'old_geometry',project/PROGRAMME/'spin_entry_gravity_1104.npz');refs.append(old);labels.append('原 gravity6（保留）');packets.append(load(base/'spin_entry_gravity_1104/common/exmr_hand_nofoot/fitted/result.npz'))
    for stage in stages:
        s=json.loads((root/stage/'run_summary.json').read_text(encoding='utf-8'));ref,cam,_,_=restore_absolute(project,base,target/stage,Path(s['parameter']))
        for key in ['rig','controls','motion_times']:np.testing.assert_array_equal(camera.p[key],cam.p[key])
        refs.append(ref);labels.append(('A 连续源约束' if stage=='source_stage' else 'B 有限腾空物理')+'（待审核）');packets.append(load(s['result']))
    seconds=[81.4,85.56,85.58,89.32,89.36,92.46,92.8,93.2];wanted=np.rint(np.array(seconds)*12800).astype('int64');frames=np.searchsorted(old.p['pts'],wanted);np.testing.assert_array_equal(old.p['pts'][frames],wanted)
    for ref,packet in zip(refs,packets):
        s=ref.at(ref.times[frames]);np.testing.assert_allclose(s['P']*ref.binding.ratio,packet['P_world_mid'][frames],atol=2e-10,rtol=0);np.testing.assert_allclose(s['local_R'],packet['local_R_mid'][frames],atol=2e-10,rtol=0);np.testing.assert_allclose(s['Q'],packet['Q_world_mid'][frames],atol=2e-10,rtol=0)
    projected=[project_mesh(ref,camera,frames) for ref in refs]
    source=json.loads((base/'action_integration_1103/spin_s05/contract.json').read_text(encoding='utf-8'))['source'];reader=SourceVideo(source['path'],(1,12800));font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',23);records=[]
    try:
        for i,pt in enumerate(wanted):
            raw=Image.fromarray(reader.read(int(pt)));canvas=Image.new('RGB',(1920*len(refs),1080))
            for j,(ref,view,label) in enumerate(zip(refs,projected,labels)):
                mesh,shoe,_,_,packet,nlower=view;su=shoe[i];parts=[su[0,:nlower],su[0,nlower:],su[1,:nlower],su[1,nlower:]];faces=np.asarray(packet['template']['faces'],np.int32)
                im=paint(raw,mesh[i],ref.binding.surface.faces,parts,[faces,ref.binding.faces,faces,ref.binding.faces],label,font,int(pt));canvas.paste(im,(1920*j,0))
            p=target/f'compare_{int(pt)}.png';canvas.save(p);records.append(dict(pts=int(pt),seconds=int(pt)/12800.,path=str(p),sha256=digest(p)))
    finally:reader.close()
    write_json(target/'projection_and_clock.json',dict(source=source,frames=records,time_base=[1,12800],native_camera_equal=True,five_nodes=True,parameter_and_saved_states_equal=True,panels=labels,no_geometry_or_pixel_alignment=True,no_video=True))
    return records


def mass_sensitivity(root):
    root=Path(root)
    for stage in ['source_stage','flight_stage']:
        dest=root/stage;n=load(dest/'centroidal_physics.npz');C=n['segment_com_world_m'];R=n['segment_R_world'];i=n['rates_source_indices'];t=(n['physics_pts']-n['physics_pts'][0])/12800.;D2=second_difference_operator(t);m=n['segment_mass_fractions'];ids=n['segment_native_joint_ids'];count=np.count_nonzero(ids>=0);rows=[];arrays={}
        base_com=n['com_world_m'];baseH=n['specific_angular_momentum_world_m2ps'][i];valid=n['derivative_stencil_valid'][1:-1]
        for tag,lower,shoe in [('leg_low_skate_low',.8,.01),('leg_high_skate_high',1.2,.03)]:
            body=m[:count]*np.where(np.isin(ids[:count],[1,2,4,5,7,8,10,11]),lower,1.);body/=body.sum();mass=np.r_[body,shoe,shoe];mass/=mass.sum();COM=np.einsum('s,nsc->nc',mass,C)
            H=centroidal_momentum(centres_world_m=C[i],velocity_world_mps=n['segment_velocity_world_mps'],rotation_world_from_segment=R[i],omega_world_radps=n['segment_omega_world_radps'],segment_mass_fractions=mass,unit_mass_inertia_segment_m2=n['segment_unit_mass_inertia_local_m2'])['specific_angular_momentum_world_m2ps']
            fdelta=D2@(COM-base_com);hd=H-baseH;arrays[tag+'_com_world_m']=COM;arrays[tag+'_specific_H_world_m2ps']=H;arrays[tag+'_external_force_per_mass_change_m_s2']=fdelta
            rows.append(dict(variant=tag,lower_mass_multiplier=lower,relative_skate_mass=shoe,max_COM_change_m=float(np.linalg.norm(COM-base_com,axis=1).max()),max_specific_H_change_m2ps=float(np.linalg.norm(hd[valid],axis=1).max()),max_external_force_per_mass_change_m_s2=float(np.linalg.norm(fdelta[valid],axis=1).max()),max_adjacent_angular_impulse_change_m2ps=float(np.linalg.norm(np.diff(hd,axis=0)[valid[:-1]&valid[1:]],axis=1).max())))
        arrays.update(physics_pts=n['physics_pts'],H_source_indices=i,derivative_stencil_valid=n['derivative_stencil_valid']);np.savez_compressed(dest/'mass_sensitivity.npz',**arrays);write_json(dest/'mass_sensitivity.json',dict(variants=rows,assumptions_unmeasured=True,source_support_episode_inferred_not_per_frame_ground_truth=True,raw_reference_plane_height_not_actual_contact_truth=True))
    return True
