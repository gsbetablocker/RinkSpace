"""Read saved R/O, fixed skate template and current factor documents only.

No body reconstruction, model inference, optimizer, image/video or GUI calls.
"""
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT = Path('D:/OneDrive/codex_projs/RinkSpace')
BASE = Path('D:/CodexData/RinkSpace_Data/runs/body_s01_20261029')
OUT = BASE/'blade_support_profile_1105'
sys.path.insert(0, str(PROJECT))
from calibration.program_tracking.body_tracking.skate_template import SkateParameters, build_skate, profile_height


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def py(v):
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, np.generic): return v.item()
    raise TypeError(type(v).__name__)


def admitted(row):
    return (row['role']=='fit' and row['foot'] in (0,1) and row['visibility']=='visible'
        and row['reliability']>0 and row.get('optimization_enabled',True)
        and (row['kind']!='support_contact_region' or row['contact_state']=='supported'))


def trapezoid(t, singleton=.04):
    if len(t)==1: return np.full(1,singleton)
    w=np.empty(len(t));w[0]=(t[1]-t[0])/2;w[-1]=(t[-1]-t[-2])/2
    w[1:-1]=(t[2:]-t[:-2])/2
    return w


def angle_region(row):
    heel,toe,hh,th=[np.asarray(row[k],float) for k in ['heel_uv','toe_uv','heel_half_extent_uv','toe_half_extent_uv']]
    d=toe-heel;centre=np.arctan2(d[1],d[0]);corners=np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
    v=(toe+corners[:,None]*th)-(heel+corners[None,:]*hh)
    a=np.arctan2(v[...,1],v[...,0])-centre
    a=.5*np.arctan2(np.sin(2*a),np.cos(2*a))
    return float(np.max(np.abs(a)))


def inventory(name, root):
    saved=np.load(root/'solve/candidate.npz',allow_pickle=False)
    packet=np.load(root/'common/exmr_hand_nofoot/fitted/result.npz',allow_pickle=False)
    pts=packet['pts'];origin=pts[0]/12800.;seconds=saved['seconds'];t=pts/12800.
    document=read_json(root/'support_and_free_shoe_regions.json');relations=read_json(root/'grasp_relations.json')
    cfg=relations['factor_parameters'];mult=cfg['shoe_region_weight_multiplier'];policy=json.loads(str(saved['ice_soft_policy_json']))
    # This reproduces SpinGraspFactors.__init__ sampling; phase-boundary rows are
    # unioned after the context selection, exactly as in the implementation.
    chosen=np.flatnonzero((t>=seconds[0])&(t<=seconds[1]))[::2]
    boundary_pts=[round(v*12800) for v in relations['phase_boundaries_seconds']]
    chosen=np.unique(np.r_[chosen,np.flatnonzero(np.isin(pts,boundary_pts))])
    sample_pts=pts[chosen];mask=np.zeros(len(chosen),bool);phase=np.zeros(len(chosen))
    for interval in relations['support_intervals']:
        m=(sample_pts>=interval['start_pts'])&(sample_pts<=interval['end_pts'])
        mask|=m;phase[m]=interval['weight']
    continuous_pts=sample_pts[mask];dt=trapezoid(continuous_pts/12800.)
    coef=np.sqrt(cfg['continuous_support']['weight']*dt*phase[mask])/cfg['continuous_support']['scale_m']
    physics_source_times=saved['physics_times']+origin
    all_support=[r for r in document['supports'] if r.get('mode','glide') not in ('airborne','unknown')]
    discrete=[r for r in all_support if np.any(abs(physics_source_times-r['pts']/12800.)<1e-8)]
    image=[r for r in document['observations'] if admitted(r)]
    image_support=[r for r in image if r['kind']=='support_contact_region']
    planar=[r for r in image_support if r.get('prediction_geometry')=='nominal_ice_projection_of_current_lowest_runner']
    active_seconds=saved['local_continuous_support_seconds'] if 'local_continuous_support_seconds' in saved else seconds
    in_local=lambda p: active_seconds[0]<=p/12800.<=active_seconds[1]
    axis=[]
    for r in document.get('boot_axis_observations',[]):
        half=angle_region(r);enabled=r['role']=='fit' and r['visibility']=='visible' and r.get('optimization_enabled',True)
        axis.append(dict(id=r['id'],pts=r['pts'],role=r['role'],enabled=enabled,within_active_local_support=in_local(r['pts']),
            source_weight=r['objective_weight'],reliability=r['reliability'],runtime_objective_weight=r['objective_weight']*mult,
            combined_weight=r['objective_weight']*mult*r['reliability'],half_angle_rad=half,
            half_angle_degrees=np.rad2deg(half),residual_coefficient_per_rad=np.sqrt(r['objective_weight']*mult*r['reliability'])/half))
    run_summary=read_json(root/'run_summary.json')
    saved_image_diagnostics=[]
    if 'source_regions_after' in run_summary:
        for r in run_summary['source_regions_after']['observations']:
            if r['kind']=='support_contact_region' and r.get('optimization_enabled'):
                saved_image_diagnostics.append({k:r[k] for k in ['id','pts','prediction_geometry','outside_region_px','actual_geometry_outside_region_px','uv','actual_geometry_uv']})
    return dict(root=str(root),candidate=str(root/'solve/candidate.npz'),candidate_sha256=digest(root/'solve/candidate.npz'),
        source_document=str(root/'support_and_free_shoe_regions.json'),context_seconds=seconds,active_local_support_seconds=active_seconds,
        source_frames=len(pts),physics_time_rows=len(saved['physics_times']),ice_soft_policy=policy,factor_parameters=cfg,
        source_support=dict(document_rows=len(all_support),document_pts=[r['pts'] for r in all_support],
            actual_physics_support_pts=[r['pts'] for r in discrete],actual_unique_pts=len(set(r['pts'] for r in discrete)),
            actual_groups=len(discrete),groups=[dict(id=r['id'],pts=r['pts'],mode=r.get('mode','glide'),feet=r.get('feet',[0,1]),
                source_support_weight_s=r['source_support_weight_s'],support_residual_coefficient_per_m=np.sqrt(r['source_support_weight_s']*policy['support_weight'])/policy['support_scale_m']) for r in discrete]),
        continuous_support=dict(actual_unique_pts=len(continuous_pts),pts=continuous_pts,quadrature_seconds=dt,
            coefficient_per_m=coef,within_context_count=int(np.sum((continuous_pts/12800.>=seconds[0])&(continuous_pts/12800.<=seconds[1]))),
            within_active_local_support_count=sum(in_local(p) for p in continuous_pts),
            phase_boundaries_union_after_context_selection=True,foot_hardcoded_by_current_implementation=1),
        image_support=dict(all_enabled_image_rows=len(image),actual_groups=len(image_support),actual_unique_pts=len(set(r['pts'] for r in image_support)),
            pts=[r['pts'] for r in image_support],nominal_ice_groups=len(planar),nominal_ice_pts=[r['pts'] for r in planar],
            within_active_local_support_count=sum(in_local(r['pts']) for r in image_support),robust_delta=document['robust_delta'],
            runtime_weight_multiplier=mult,rows=[dict(id=r['id'],pts=r['pts'],foot=r['foot'],source_weight=r['objective_weight'],
                reliability=r['reliability'],runtime_weight=r['objective_weight']*mult,combined_weight=r['objective_weight']*mult*r['reliability'],
                ellipse_semiaxes_px=np.sqrt(np.linalg.eigvalsh(r['region_shape_px2'])),prediction_geometry=r.get('prediction_geometry','actual_current_geometry'),
                arc_selection=r.get('arc_selection','fixed_initial_arc')) for r in image_support]),
        boot_axes=dict(enabled_groups=sum(r['enabled'] for r in axis),enabled_unique_pts=len(set(r['pts'] for r in axis if r['enabled'])),rows=axis),
        saved_after_nominal_vs_actual_image_diagnostics=saved_image_diagnostics)


def profile(name, path):
    root=path.parents[3]
    with np.load(path,allow_pickle=False) as z:
        pts=z['pts'];R=z['skate_R_world_mid'];O=z['skate_origin_world_mid']
        binding=json.loads(str(z['footwear_binding_json']))
        base_path=root/'solve/candidate.npz'
        with np.load(base_path,allow_pickle=False) as s:
            params=json.loads(str(s['base_skate_parameters_json']))
        params.update(binding['shape_parameters'])
        p=SkateParameters(**params);template=build_skate(p)
        arcs=template['contact_arcs'];normal=R[...,2,:];heights=[];points=[];feature_ids=[];semantic=[];endflags=[]
        for a in arcs:
            c=np.asarray(a['center']);ac=np.asarray(a['axis_cos']);asi=np.asarray(a['axis_sin'])
            A=normal@ac;B=normal@asi;lo,hi=a['theta_start_rad'],a['theta_end_rad']
            interior=np.arctan2(-B,-A);interior+=2*np.pi*np.round(((lo+hi)/2-interior)/(2*np.pi))
            ts=np.stack([np.full_like(A,lo),np.full_like(A,hi),np.clip(interior,lo,hi)],-1)
            iz=np.argmin(A[...,None]*np.cos(ts)+B[...,None]*np.sin(ts),axis=-1)
            theta=np.take_along_axis(ts,iz[...,None],axis=-1)[...,0]
            q=c+a['radius_m']*(np.cos(theta)[...,None]*ac+np.sin(theta)[...,None]*asi)
            heights.append(O[...,2]+np.sum(normal*q,axis=-1));points.append(q);feature_ids.append(a['id']);semantic.append('runner')
            # Exact analytic minimizer at the WHOLE runner rear/front boundary;
            # joining two arcs is not classified as a whole-runner endpoint.
            flags=np.zeros_like(iz,dtype=np.int8)
            if a['id'].endswith('_main'): flags=np.where(theta==lo,-1,flags)
            if a['id'].endswith('_secondary'): flags=np.where(theta==hi,1,flags)
            endflags.append(flags)
        ah=np.stack(heights,-1);aq=np.stack(points,-2);af=np.stack(endflags,-1)
        best=np.argmin(ah,axis=-1);ix=np.arange(len(pts))[:,None];ft=np.arange(2)[None]
        hmin=ah[ix,ft,best];qmin=aq[ix,ft,best];endpoint=af[ix,ft,best]
        for g in template['contact_point_groups']:
            v=np.asarray(g['points']);zz=np.einsum('nfj,vj->nfv',normal,v);j=np.argmin(zz,axis=-1);q=v[j]
            heights.append(O[...,2]+np.sum(normal*q,axis=-1));points.append(q);feature_ids.append(g['id']);semantic.append(g['semantic'])
        # The saved upper vertices carry the actual deformed shell. This uses
        # them directly rather than constructing body/FK/deformation objects.
        upper=z['skate_upper_world_mid'];heights.append(np.min(upper[...,2],axis=-1));feature_ids.append('upper_shell_guard');semantic.append('solid_guard')
        full_h=np.stack(heights,-1);raw_best=np.argmin(full_h,axis=-1)
        allowance=np.array([{'runner':.002,'toe_pick':.005,'solid_guard':0.}[s] for s in semantic])
        adjusted_best=np.argmin(full_h+allowance,axis=-1)
        xrear=-p.runner_rear_extension_m;xfront=p.sole_length_m-p.runner_front_inset_m;xs=np.array([xrear,(xrear+xfront)/2,xfront]);qs=np.c_[xs,np.zeros(3),profile_height(xs,p)]
        centreline=O[:,:,None]+np.einsum('nfij,kj->nfki',R,qs)
        edge_h=centreline[...,2]-np.abs(R[...,2,1,None])*p.blade_width_m/2
        recon=0.
        if 'blade_material_world_mid' in z:
            recon=float(np.max(abs(centreline-z['blade_material_world_mid'])))
        pitch=np.rad2deg(np.arctan2(R[...,2,0],np.hypot(R[...,0,0],R[...,1,0])))
        roll=np.rad2deg(np.arctan2(R[...,2,1],R[...,2,2]))
        saved=dict(pts=pts,seconds=pts/12800.,pitch_degrees=pitch,roll_degrees=roll,runner_min_height_m=hmin,
            runner_min_local_point_m=qmin,runner_min_x_normalized=(qmin[...,0]-xrear)/(xfront-xrear),
            runner_min_arc=np.asarray([a['id'] for a in arcs])[best],whole_runner_endpoint=endpoint,
            rear_mid_front_centreline_height_m=centreline[...,2],rear_mid_front_lower_edge_height_m=edge_h,
            raw_lowest_feature=np.asarray(feature_ids)[raw_best],raw_lowest_semantic=np.asarray(semantic)[raw_best],raw_lowest_height_m=np.min(full_h,axis=-1),
            adjusted_nonpenetration_feature=np.asarray(feature_ids)[adjusted_best],adjusted_nonpenetration_semantic=np.asarray(semantic)[adjusted_best],
            adjusted_nonpenetration_margin_m=np.min(full_h+allowance,axis=-1))
        np.savez_compressed(OUT/(name+'_whole951_profile.npz'),**saved)
        field=['candidate','pts','seconds','foot','pitch_degrees','roll_degrees','runner_min_height_m','runner_min_x_normalized','runner_min_arc','whole_runner_endpoint',
            'rear_height_m','mid_height_m','front_height_m','raw_lowest_feature','raw_lowest_semantic','raw_lowest_height_m','adjusted_nonpenetration_feature','adjusted_nonpenetration_margin_m']
        csvrows=[]
        for i,pt in enumerate(pts):
            for foot in range(2):
                csvrows.append([name,int(pt),pt/12800.,foot,pitch[i,foot],roll[i,foot],hmin[i,foot],saved['runner_min_x_normalized'][i,foot],saved['runner_min_arc'][i,foot],int(endpoint[i,foot]),
                    *edge_h[i,foot],saved['raw_lowest_feature'][i,foot],saved['raw_lowest_semantic'][i,foot],saved['raw_lowest_height_m'][i,foot],saved['adjusted_nonpenetration_feature'][i,foot],saved['adjusted_nonpenetration_margin_m'][i,foot]])
        with (OUT/(name+'_whole951_profile.csv')).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.writer(f);w.writerow(field);w.writerows(csvrows)
        target=[85.2,87.64,89.56,91.98,92.46];specific=[]
        for sec in target:
            i=int(np.flatnonzero(pts==round(sec*12800))[0]);foot=1
            specific.append(dict(pts=int(pts[i]),seconds=sec,foot=foot,pitch_degrees=pitch[i,foot],roll_degrees=roll[i,foot],
                runner_min_height_m=hmin[i,foot],runner_min_x_normalized=saved['runner_min_x_normalized'][i,foot],runner_min_arc=saved['runner_min_arc'][i,foot],
                whole_runner_endpoint=int(endpoint[i,foot]),rear_mid_front_lower_edge_height_m=edge_h[i,foot],
                raw_lowest_semantic=saved['raw_lowest_semantic'][i,foot],raw_lowest_feature=saved['raw_lowest_feature'][i,foot],raw_lowest_height_m=saved['raw_lowest_height_m'][i,foot]))
        support=(pts>=round(86.8*12800))&(pts<=round(92.46*12800));summary={}
        for foot in range(2):
            ss=endpoint[support,foot]
            summary[str(foot)]=dict(frames=int(support.sum()),pitch_degrees_min_median_max=np.quantile(pitch[support,foot],[0,.5,1]),roll_degrees_min_median_max=np.quantile(roll[support,foot],[0,.5,1]),
                whole_runner_rear_endpoint_frames=int(np.sum(ss==-1)),whole_runner_front_endpoint_frames=int(np.sum(ss==1)),interior_frames=int(np.sum(ss==0)),
                runner_min_height_m_min_median_max=np.quantile(hmin[support,foot],[0,.5,1]),
                runner_min_x_normalized_min_median_max=np.quantile(saved['runner_min_x_normalized'][support,foot],[0,.5,1]),
                rear_mid_front_lower_edge_height_m_min_median_max=np.quantile(edge_h[support,foot],[0,.5,1],axis=0),
                raw_lowest_semantics=dict(Counter(saved['raw_lowest_semantic'][support,foot])),
                adjusted_nonpenetration_semantics=dict(Counter(saved['adjusted_nonpenetration_semantic'][support,foot])))
        return dict(path=str(path),sha256=digest(path),frames=len(pts),time_base=z['time_base'],first_last_pts=pts[[0,-1]],
            template_parameters_metric=params,template_condition='saved fixed shared geometry; generic engineering family, not athlete measurement',
            saved_blade_material_reconstruction_max_abs_m=recon,specific_right_support_foot=specific,support_86_8_to_92_46=summary,
            profile_npz=str(OUT/(name+'_whole951_profile.npz')),profile_csv=str(OUT/(name+'_whole951_profile.csv')))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    parents=read_json(BASE/'blade_observability_1105/identity_audit.json')['parents']
    paths={r['candidate']:Path(r['path']) for r in parents}
    paths.update(entry=BASE/'blade_orientation_1105/entry_81_2_82_0/common/exmr_hand_nofoot/fitted/result.npz',
        support=BASE/'blade_orientation_1105/support_87_6_90_0/common/exmr_hand_nofoot/fitted/result.npz')
    profiles={name:profile(name,path) for name,path in paths.items()}
    inventories={name:inventory(name,paths[name].parents[3]) for name in ['local_281','entry','support']}
    code_files=['skate_physics.py','ice_soft.py','spin_grasp_factors.py','support_image.py','boot_axis_factors.py','blade_boot_refit.py','spin_continuous_refit.py','spin_continuous_continue.py','skate_template.py']
    code_identity={f:digest(PROJECT/'calibration/program_tracking/body_tracking'/f) for f in code_files}
    report=dict(schema='rinkspace.blade-support-constraint-mechanism.v1',read_only_scope='saved arrays and fixed template; no optimizer, full body build, inference, images, videos or GUI',
        current_code_sha256=code_identity,
        metric_status='conditional exported metric scale; numeric model heights do not establish physical ice penetration or source acceptance',
        geometry_definitions=dict(pitch_degrees='atan2(R_zx,hypot(R_xx,R_yx)); positive toe axis points upward',
            roll_degrees='atan2(R_zy,R_zz); positive local-left edge points upward; kept separate from pitch',
            x_normalized='(x+runner_rear_extension)/(sole_length-runner_front_inset+runner_rear_extension)',
            whole_runner_endpoint='-1 rear, +1 front, 0 interior; exact analytic arc parameter bound, not physical mm threshold; rocker join is interior',
            rear_mid_front='rear/end, arithmetic x midpoint, front/end of runner arcs; lower of the two running edges at each x; toe-pick excluded',
            lowest_geometry='raw lowest among exact arcs, supplied point groups and actual saved deformed upper vertices; adjusted nonpenetration selection separately includes semantic allowances'),
        mechanisms=[
            dict(factor='skate_physics.constraints nonpenetration',source='calibration/program_tracking/body_tracking/skate_physics.py:89',
                formula='one row per foot: min_feature(raw_height_m + semantic_allowance_m)',allowance_m=dict(runner=.002,toe_pick=.005,solid_guard=0.),
                blind_to='one scalar lowest witness does not constrain fore/aft contact distribution; lifting any nonwinning runner end can be free',
                normalization_m=.005,legacy_feasibility_tolerance_m=.001,not_current_ice_delivery_gate=True),
            dict(factor='source support -> ice_soft_rows',source='calibration/program_tracking/body_tracking/ice_soft.py:74',
                formula='sqrt(source_weight_s*weight)/scale * max(min_allowed_runner_height - support_hover_deadzone,0)',
                blind_to='only lowest eligible feature (or lowest foot if feet=[0,1]); support lower row omitted; no end-to-end length/distribution contact residual',
                source_support_band_m=[-.002,.003],band_note='constraints exposes legacy band but soft adapter uses raw upper-row values and current .025 m deadzone; .003 m is not active soft hover cutoff'),
            dict(factor='ice_soft_rows penetration',formula='sqrt(time_weight*weight)/scale * max(-(raw_height+allowance)-penetration_deadzone,0)',
                implied_zero_pressure_raw_height_lower_m=dict(runner=-.007,toe_pick=-.010,solid_guard=-.005),
                note='finite engineering objective only, not actual-ice or delivery acceptance limits'),
            dict(factor='SpinGraspFactors.continuous_support',source='calibration/program_tracking/body_tracking/spin_grasp_factors.py:153',
                formula='sqrt(20*trapezoid_seconds*phase_weight)/.08 * max(min_glide_runner_height-.025,0)',
                blind_to='hardcoded right foot=1; one minimum across glide arcs; no contact length, contact distribution, selected local x, pitch/roll or end-lift penalty'),
            dict(factor='SupportImageFactors.geometry',source='calibration/program_tracking/body_tracking/support_image.py:50',
                actual='projects selected actual point using native five-node moving projection',nominal='sets current lowest runner world z=0 before projection and zeros z derivative',
                blind_to='nominal branch hides actual witness height in objective; actual_geometry_uv/outside_region is reported only, not residual; neither branch constrains runner span touching ice'),
            dict(factor='BootAxisFactors',source='calibration/program_tracking/body_tracking/boot_axis_factors.py:31',
                formula='sqrt(runtime_objective_weight*reliability)/half_angle_rad * sign(angle) * max(abs(unoriented_2D_angle)-half_angle_rad,0)',
                geometry='heel average adapter [0,7]; toe average adapter [1,8]; actual five-node source projection; no new shoe state',
                blind_to='image axis only; no absolute image position/axis length in this term, no direct 3D pitch/roll or contact-height/distribution constraint; 180 degree symmetric finite source angular band')],
        runtime_inventories=inventories,profiles=profiles,
        limits=['Geometry statistics are model diagnostics under current template and conditional scale, not new physical angle/cm acceptance rules.',
            'Source support, continuous interval support and image regions use different PTS sets; neither objective value nor low witness height proves correct blade placement.',
            'No actual optimization-factor perturbation was run here; parent task performs grouped real-factor cost perturbations.'])
    (OUT/'constraint_mechanism.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=py),encoding='utf-8')
    print(json.dumps(dict(output=str(OUT/'constraint_mechanism.json'),counts={k:dict(source_support=v['source_support']['actual_unique_pts'],continuous=v['continuous_support']['actual_unique_pts'],continuous_local=v['continuous_support']['within_active_local_support_count'],image_support=v['image_support']['actual_unique_pts'],axes=v['boot_axes']['enabled_unique_pts']) for k,v in inventories.items()},support_right_summary={k:v['support_86_8_to_92_46']['1'] for k,v in profiles.items()}),ensure_ascii=False,default=py))


if __name__=='__main__': main()
