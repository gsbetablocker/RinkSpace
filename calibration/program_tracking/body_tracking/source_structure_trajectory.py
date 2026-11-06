"""Read-only whole-sequence physical runner export and source-range diagnostics.

All world curves come from the saved continuous body plus ankle-pivoted shoes.
A lowest steel witness is a moving geometry diagnostic, not a material point or
source contact truth. Sparse source regions and unresolved anatomical identities
remain separate from the model trajectory at every native exposure. No fitting,
network inference, source-video decoding or GUI is performed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .blade_boot_refit import _mass
from .blade_contact_modes import blade_axes_diagnostics
from .blade_projection import SavedBladeProjection
from .exmr_integration import read_case_identity
from .sequence_skate_state import SequenceState
from .skate_checkpoint import validate_checkpoint_identity
from .skate_physics import fk_state
from .skate_template import profile_height
from .source_skate_structure_factors import FullSkateStructureFactors, UnassignedSkateSetFactors
from .source_support_locations import SourceSupportLocationFactors, _runner_witness


SCHEMA='rinkspace.source-structure-trajectory.v1'
CASE=Path('common/exmr_hand_nofoot')
CURVE_LABELS=('lower_negative_y','lower_positive_y','upper_negative_y','upper_positive_y')
PROBE_SECONDS=(76.,78.4,80.4,81.6,81.62,87.64,92.44,93.5,95.)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return str(value)
    raise TypeError(type(value).__name__)
def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=serial,allow_nan=False)+'\n',encoding='utf8')
def load_result(path):
    required=('pts','source_pts','time_base','camera_binding_json','comparison_case_identity_json',
        'P_world_mid','Q_world_mid','local_R_mid','native22_world_mid','skate_R_world_mid',
        'skate_origin_world_mid','COCO_uv','com_world_mid')
    optional=('absolute_position_controls','absolute_rotation_controls','absolute_knots',
        'relative_skate_rotation_controls','relative_skate_knots','runner_edges_world_mid',
        'skate_edge_world_mid','blade_material_world_mid','source_structure_integration_sha256',
        'source_complete_structure_observation_count','source_support_location_in_optimization',
        'source_unassigned_association_json','absolute_parameter_path')
    with np.load(path,allow_pickle=False)as z:
        missing=[key for key in required if key not in z]
        if missing:raise ValueError('Actual result missing '+', '.join(missing))
        return {key:z[key].copy()for key in required+optional if key in z}


def source_masks(pts,structures,integration):
    """Source visibility is not a model association or a measured 3-D label."""
    lookup={int(p):i for i,p in enumerate(pts)};n=len(pts)
    count=np.zeros((n,2),np.int32);assigned=np.zeros((n,2),bool)
    unassigned=np.zeros(n,bool);support=np.zeros((n,2),bool)
    for row in integration['full_source_skate_structures']['rows']:
        count[lookup[int(row['pts'])],int(row['foot'])]+=1
    for row in structures['frames']:
        if row.get('foot')in (0,1):assigned[lookup[int(row['pts'])],int(row['foot'])]=True
    for row in structures.get('unassigned_identity_observations',[]):unassigned[lookup[int(row['pts'])]]=True
    for row in integration['source_support_evidence']:support[lookup[int(row['pts'])],int(row['foot'])]=True
    status=np.full((n,2),'inferred_between_source_exposures',dtype='U64')
    status[assigned]='source_regions_anatomically_assigned'
    status[unassigned]='source_regions_anatomy_unassigned'
    return dict(source_feature_count=count,source_assigned_region_mask=assigned,
        source_unassigned_exposure_mask=unassigned,source_support_location_mask=support,
        source_region_status=status,model_pose_status=np.full((n,2),'inferred_joint_continuous',dtype='U32'))


def jump_summary(points,pts):
    """Largest Euclidean displacement across an actual native 20-ms interval."""
    points=np.asarray(points,float);pts=np.asarray(pts)
    if points.shape[0]!=len(pts)or points.shape[-1]!=3 or not np.isfinite(points).all():
        raise ValueError('Finite world XYZ on the native clock required')
    step=np.diff(pts)/12800.;valid=np.isclose(step,.02,rtol=0,atol=1e-12)
    if not valid.any():return dict(available=False,reason='no_native_20ms_intervals')
    delta=np.diff(points,axis=0);distance=np.linalg.norm(delta,axis=-1)
    scored=np.where(valid.reshape((-1,)+(1,)*(distance.ndim-1)),distance,-np.inf)
    index=np.unravel_index(int(np.argmax(scored)),scored.shape);i=index[0]
    return dict(available=True,max_displacement_conditional_m=float(distance[index]),
        from_pts=int(pts[i]),to_pts=int(pts[i+1]),from_source_seconds=float(pts[i]/12800.),
        to_source_seconds=float(pts[i+1]/12800.),delta_seconds=float(step[i]),
        item_indices=list(index[1:]),delta_xyz_conditional_m=delta[index].tolist(),
        intervals_compared=int(valid.sum()),interpretation='model displacement, not source-measured speed or accuracy')


def witness_span(points,pts,foot):
    select=(pts>=round(88*12800))&(pts<=round(92*12800));xy=np.asarray(points)[select,foot,:2]
    if not len(xy):return dict(available=False)
    from scipy.spatial.distance import pdist
    return dict(model_foot=foot,source_seconds=[88.,92.],samples=len(xy),
        xy_min_conditional_m=xy.min(axis=0).tolist(),xy_max_conditional_m=xy.max(axis=0).tolist(),
        xy_axis_span_conditional_m=np.ptp(xy,axis=0).tolist(),
        maximum_pairwise_xy_distance_conditional_m=float(pdist(xy).max())if len(xy)>1 else 0.,
        witness='moving exact minimum on finite steel arcs; not a fixed material/contact point')


def geometry(query,stage):
    pts=np.asarray(query.pts);n=len(pts)
    if pts.dtype.kind not in 'iu' or pts.shape!=(951,) or not np.array_equal(np.diff(pts),np.full(950,256)):
        raise ValueError('Expected all 951 unique native exposures at 20 ms')
    np.testing.assert_array_equal(pts[[0,-1]],[972800,1216000])
    validate_checkpoint_identity(query.parameter)
    info=query.camera_info;saved_info=json.loads(str(query.parameter['camera_binding_json']))
    for key in ('state_sha256','metres_per_scene_unit','solver_metres_per_scene_unit','length_export_ratio','source_time_base'):
        if info[key]!=saved_info[key]:raise ValueError('Frozen camera identity/units changed: '+key)
    if info['shot']!=saved_info['shot']:raise ValueError('Frozen native shot identity changed')
    np.testing.assert_allclose(info['metres_per_scene_unit']/query.camera.scale,query.ratio,rtol=1e-12,atol=0)
    identity=read_case_identity(query.parameter)
    for field,name in [('case_input_sha256','input.npz'),('compact_smplx_sha256','compact_smplx.npz'),('initial_reference_sha256','initial.npz')]:
        if identity[field]!=sha(Path(stage)/CASE/name):raise ValueError('Current case input differs from saved '+field)
    state=fk_state(query.reference,query.reference.at(query.times),np.zeros((n,69)))
    packet=query.binding.packet();lower=np.asarray(packet['edges']);p=packet['parameters']
    upper=lower+[0.,0.,p.blade_web_height_m];curves=np.concatenate((lower,upper),axis=0)
    # Template, body and O are solver lengths. Export positions multiply by
    # the single global ratio once; R and native pixel coordinates never do.
    world=(state['O'][:,:,None,None,:]+np.einsum('nfij,ekj->nfeki',state['R'],curves))*query.ratio
    rear=-p.runner_rear_extension_m;front=p.sole_length_m-p.runner_front_inset_m
    xs=np.array([rear,(rear+front)/2,front]);materials=np.c_[xs,np.zeros(3),profile_height(xs,p)]
    material_world=(state['O'][:,:,None]+np.einsum('nfij,kj->nfki',state['R'],materials))*query.ratio
    w=_runner_witness(query.binding,state['R'].reshape(-1,3,3),state['O'].reshape(-1,3))
    feature=np.asarray(w['feature_ids'])[w['feature_index']].reshape(n,2)
    switches=np.zeros((n,2),bool);switches[1:]=feature[1:]!=feature[:-1]
    mass,mass_path=_mass(query.reference,Path(stage));C,_=mass.evaluate(query.reference.at(query.times),np.zeros((n,69)))
    out=dict(origin_world_conditional_m=state['O']*query.ratio,R_world=state['R'],
        runner_lower_world_conditional_m=world[:,:,:2],runner_upper_world_conditional_m=world[:,:,2:],
        material_station_world_conditional_m=material_world,com_world_conditional_m=mass.com(C),
        exact_runner_witness_world_conditional_m=w['world'].reshape(n,2,3)*query.ratio,
        exact_runner_witness_feature=feature,exact_runner_witness_feature_switch=switches,
        exact_runner_witness_local_solver=w['local'].reshape(n,2,3),
        exact_runner_witness_theta_rad=w['theta'].reshape(n,2),
        exact_runner_witness_stationary_interior=w['interior'].reshape(n,2),
        runner_next_feature_height_gap_conditional_m=w['next_feature_height_gap_solver'].reshape(n,2)*query.ratio,
        **blade_axes_diagnostics(state['R']))
    return out,dict(curves=curves,materials=materials,state=state,mass_path=mass_path)


def verify_result(query,result,model,local):
    pts=query.pts
    for key in ('pts','source_pts'):np.testing.assert_array_equal(result[key],pts)
    np.testing.assert_array_equal(result['time_base'],[1,12800])
    if read_case_identity(result)!=read_case_identity(query.parameter):raise ValueError('Parameter/result case identities differ')
    if json.loads(str(result['camera_binding_json']))['state_sha256']!=query.camera_info['state_sha256']:
        raise ValueError('Result frozen camera differs')
    state=local['state'];checks={}
    for key,prediction in [('P_world_mid',state['P']*query.ratio),('Q_world_mid',state['Q']),
        ('local_R_mid',state['local_R']),('native22_world_mid',state['native_world']*query.ratio),
        ('skate_R_world_mid',model['R_world']),('skate_origin_world_mid',model['origin_world_conditional_m']),
        ('com_world_mid',model['com_world_conditional_m'])]:
        np.testing.assert_allclose(prediction,result[key],rtol=0,atol=2e-9,err_msg=key)
        checks[key]=float(np.max(np.abs(prediction-result[key])))
    for key in ('absolute_position_controls','absolute_rotation_controls','absolute_knots',
                'relative_skate_rotation_controls','relative_skate_knots'):
        if key in result:np.testing.assert_allclose(result[key],query.parameter[key],rtol=0,atol=2e-12,err_msg=key)
    for key,prediction in [('runner_edges_world_mid',model['runner_lower_world_conditional_m']),
        ('skate_edge_world_mid',model['runner_lower_world_conditional_m']),
        ('blade_material_world_mid',model['material_station_world_conditional_m'])]:
        if key in result:
            np.testing.assert_allclose(prediction,result[key],rtol=0,atol=2e-9,err_msg=key)
            checks[key]=float(np.max(np.abs(prediction-result[key])))
    return checks


def project_native(query,local,indices,model,result=None):
    pts=query.pts[indices];curves=local['curves'];materials=local['materials']
    points=np.concatenate((curves.reshape(-1,3),materials,np.zeros((1,3))),axis=0)
    uv=[];material_uv=[];origin_uv=[];closure={}
    for foot in (0,1):
        out=query.project_points(pts,foot,points,include_body=result is not None and foot==0)
        uv.append(out['uv'][:,:324].reshape(len(pts),4,81,2))
        material_uv.append(out['uv'][:,324:327]);origin_uv.append(out['uv'][:,327])
        if result is not None and foot==0:
            np.testing.assert_allclose(out['COCO_uv'],result['COCO_uv'][indices],rtol=0,atol=2e-6)
            closure['maximum_COCO_native_pixel_difference']=float(np.max(np.abs(out['COCO_uv']-result['COCO_uv'][indices])))
    return dict(projection_pts=pts.copy(),runner_curves_native_uv=np.stack(uv,axis=1),
        material_station_native_uv=np.stack(material_uv,axis=1),origin_native_uv=np.stack(origin_uv,axis=1)),closure


def factor_diagnostics(query,stage,integration):
    ref=query.reference;ref.grasp_relations=read(Path(stage)/'grasp_relations.json')
    state=SequenceState(ref,(76.,95.));body=SimpleNamespace(state=state,size=state.size,tau=float(query.camera.p['rig'][37]))
    problem=SimpleNamespace(reference=ref,binding=query.binding,camera=query.camera,body=body,size=17+state.size,optimize_shared=False)
    x=np.zeros(state.size)
    assigned=FullSkateStructureFactors(problem,integration['full_source_skate_structures']).evaluate(x,False)
    unassigned=UnassignedSkateSetFactors(problem,integration.get('unassigned_source_skate_sets',[])).evaluate(x,False)
    budget=integration['source_structure_budget']
    support=SourceSupportLocationFactors(problem,integration['source_support_locations'],
        support_evidence=integration['source_support_evidence'],**budget).evaluate(x,False)
    extent=None
    for key in ('complete_runner_extents','complete_runner_extent','complete_runner_extent_rows'):
        document=integration.get(key)
        if document:
            from .complete_runner_extent import CompleteRunnerExtentFactors
            extent=CompleteRunnerExtentFactors(problem,dict(rows=document)if isinstance(document,list)else document).evaluate(x,False)
            break
    return dict(feature_rows=assigned['metrics'].get('visible_runner_contours',[])+unassigned['metrics'].get('visible_unassigned_skate_contours',[]),
        support_rows=support['metrics']['source_support_locations'],
        recomputed_unassigned_model_associations=unassigned['metrics']['source_unassigned_associations'],
        reverse_complete_extent=[]if extent is None else extent['metrics']['complete_runner_extents'],
        same_frame_distance_semantics='Existing factor evaluate(False), same native source regions, latent geometry match, no independent acceptance score')


def export_csv(path,pts,models,masks):
    header=['candidate','pts','source_seconds','model_foot','model_pose_status','source_region_status','source_feature_count',
        'source_support_xy_admitted','source_anatomical_identity_unassigned']
    vectors=['origin','F','L','U','rear_material','mid_material','front_material','moving_runner_witness','model_com']
    header += [name+'_'+axis+'_conditional_m'if name not in ('F','L','U')else name+'_'+axis for name in vectors for axis in 'xyz']
    header += ['model_fore_aft_pitch_degrees','model_edge_bank_degrees','runner_witness_feature','witness_feature_switch']
    with Path(path).open('w',encoding='utf-8-sig',newline='')as f:
        writer=csv.writer(f);writer.writerow(header)
        for candidate,model in models.items():
            for i,pt in enumerate(pts):
                for foot in (0,1):
                    vectors=[model['origin_world_conditional_m'][i,foot],model['forward_world'][i,foot],model['lateral_world'][i,foot],model['up_world'][i,foot],
                        *model['material_station_world_conditional_m'][i,foot],model['exact_runner_witness_world_conditional_m'][i,foot],model['com_world_conditional_m'][i]]
                    writer.writerow([candidate,int(pt),float(pt/12800.),foot,masks['model_pose_status'][i,foot],masks['source_region_status'][i,foot],int(masks['source_feature_count'][i,foot]),
                        bool(masks['source_support_location_mask'][i,foot]),bool(masks['source_unassigned_exposure_mask'][i]),
                        *[float(value)for vector in vectors for value in vector],float(model['fore_aft_pitch_degrees'][i,foot]),float(model['edge_bank_degrees'][i,foot]),
                        str(model['exact_runner_witness_feature'][i,foot]),bool(model['exact_runner_witness_feature_switch'][i,foot])])


def feature_csv(path,diagnostics,integration):
    header=['candidate','pts','source_seconds','source_row_id','structure_feature','source_status','source_role','model_foot',
        'source_anatomical_foot','foot_is_model_association','points','pixel_uncertainty_radius_px','mean_distance_px','p95_distance_px',
        'max_distance_px','mean_outside_source_band_px','max_outside_source_band_px','points_outside_band','evaluation_status']
    with Path(path).open('w',encoding='utf-8-sig',newline='')as f:
        writer=csv.DictWriter(f,fieldnames=header);writer.writeheader()
        for candidate,value in diagnostics.items():
            for row in value['feature_rows']:
                dist=np.asarray(row['distance_px']);outside=np.asarray(row['outside_visible_band_px']);latent=row.get('foot_is_model_association',False)
                writer.writerow(dict(candidate=candidate,pts=row['pts'],source_seconds=row['pts']/12800.,source_row_id=row.get('id'),
                    structure_feature=row['structure_feature'],source_status=row['source_status'],source_role=row['role'],model_foot=row['foot'],
                    source_anatomical_foot=''if latent else row['foot'],foot_is_model_association=latent,points=len(dist),
                    pixel_uncertainty_radius_px=row['pixel_uncertainty_radius_px'],mean_distance_px=float(dist.mean()),p95_distance_px=float(np.percentile(dist,95)),
                    max_distance_px=float(dist.max()),mean_outside_source_band_px=float(outside.mean()),max_outside_source_band_px=float(outside.max()),
                    points_outside_band=int(np.sum(outside>0)),evaluation_status='same_source_factor_distance_not_acceptance'))
            for row in value['reverse_complete_extent']:
                dist=np.asarray(row['model_to_source_distance_px']).ravel();outside=np.asarray(row['outside_complete_source_band_px']).ravel()
                writer.writerow(dict(candidate=candidate,pts=row['pts'],source_seconds=row['pts']/12800.,source_row_id=row.get('id'),
                    structure_feature='runner_lower_edge_complete_reverse_extent',source_status=row['source_status'],source_role=row['role'],
                    model_foot=row['foot'],source_anatomical_foot=row['foot'],foot_is_model_association=False,points=len(dist),
                    pixel_uncertainty_radius_px=row['pixel_uncertainty_radius_px'],mean_distance_px=float(dist.mean()),p95_distance_px=float(np.percentile(dist,95)),
                    max_distance_px=float(dist.max()),mean_outside_source_band_px=float(outside.mean()),max_outside_source_band_px=float(outside.max()),
                    points_outside_band=int(np.sum(outside>0)),evaluation_status='same_source_reverse_extent_factor_not_acceptance'))
            for row in integration.get('excluded_features',[]):
                writer.writerow(dict(candidate=candidate,pts=row['pts'],source_seconds=row['pts']/12800.,structure_feature=row['feature'],
                    source_anatomical_foot=row.get('foot'),evaluation_status='not_evaluated: '+row['reason']))


def plots(output,pts,models,masks,diagnostics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Polygon
    colours={'parent':'#3465a4','current':'#df7412'};seconds=pts/12800.
    fig,axes=plt.subplots(2,2,figsize=(14,10),constrained_layout=True)
    for foot in (0,1):
        for column,zoom in enumerate((False,True)):
            ax=axes[foot,column];select=(seconds>=88)&(seconds<=92)if zoom else np.ones(len(pts),bool)
            for candidate,model in models.items():
                xy=model['exact_runner_witness_world_conditional_m'][:,foot,:2]
                ax.plot(*xy[select].T,color=colours[candidate],lw=1.2,label=candidate+' moving exact steel minimum')
            xy=models['current']['exact_runner_witness_world_conditional_m'][:,foot,:2]
            marked=select&masks['source_assigned_region_mask'][:,foot];between=select&~masks['source_assigned_region_mask'][:,foot]&~masks['source_unassigned_exposure_mask']
            ax.scatter(*xy[between].T,s=3,color='#888888',alpha=.35,label='model at unlabelled source exposures')
            ax.scatter(*xy[marked].T,s=15,facecolors='none',edgecolors='#187c3c',label='model at anatomically assigned source regions')
            unknown=select&masks['source_unassigned_exposure_mask']
            ax.scatter(*xy[unknown].T,s=30,marker='s',facecolors='none',edgecolors='#bb2222',label='source anatomical identity unassigned')
            for row in diagnostics['current']['support_rows']:
                if row['foot']!=foot or zoom and not 88<=row['pts']/12800.<=92:continue
                hull=np.asarray(row['source_range_convex_hull_xy_conditional_m'])
                ax.add_patch(Polygon(hull,closed=True,facecolor='#27aa59',edgecolor='#187c3c',alpha=.12))
            ax.set(title=f'Model foot {foot} / '+('88-92 s'if zoom else'76-95 s'),xlabel='X conditional m',ylabel='Y conditional m');ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels()
    handles.append(Patch(facecolor='#27aa59',edgecolor='#187c3c',alpha=.2));labels.append('sampled source pixel-range hull / frozen camera + Z0')
    fig.legend(handles,labels,loc='outside lower center',ncol=2,fontsize=8)
    fig.suptitle('Moving steel geometry minima and sparse camera/Z0-conditioned source hulls\nNo whole-blade Z0 footprint and no measured contact/material-point trajectory')
    xy_path=output/'runner_witness_xy.png';fig.savefig(xy_path,dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(13,7),sharex=True,constrained_layout=True)
    for ax,key,title in zip(axes,['fore_aft_pitch_degrees','edge_bank_degrees'],['Forward-axis pitch','Lateral edge bank']):
        for candidate,model in models.items():
            for foot in (0,1):
                ax.plot(seconds,model[key][:,foot],color=colours[candidate],ls='--'if foot==0 else'-',lw=1.,label=f'{candidate} model foot {foot}')
        for t in seconds[masks['source_unassigned_exposure_mask']]:ax.axvline(t,color='#bb2222',lw=.6,alpha=.3)
        ax.set(ylabel=title+' / degrees');ax.grid(alpha=.2)
    axes[0].legend(ncol=2,fontsize=8);axes[1].set_xlabel('Native source time / seconds')
    fig.suptitle('Actual saved model blade orientation relative to ice normal; model diagnostics, not source contact truth\nRed vertical lines: source anatomical identity unassigned')
    angle_path=output/'runner_model_angles.png';fig.savefig(angle_path,dpi=160);plt.close(fig)
    return [xy_path,angle_path]


def run(project,stage,parameter,result,parent_stage,parent_parameter,structures,integration,output,*,probe=False):
    project,stage,parameter,result,parent_stage,parent_parameter,structures,integration,output=map(lambda p:Path(p).resolve(),
        (project,stage,parameter,result,parent_stage,parent_parameter,structures,integration,output))
    from rinkspace.storage import AssetStore
    from rinkspace.path_policy import guard_project_write
    if not output.is_relative_to(AssetStore(project).data):raise ValueError('Trajectory products must stay in the configured Data root')
    names=['probe_check.json']if probe else ['trajectory.npz','trajectory.csv','feature_distances.csv','factor_diagnostics.json','trajectory_review.json','runner_witness_xy.png','runner_model_angles.png']
    for name in names:
        target=guard_project_write(output/name,project_root=project)
        if target.exists():raise FileExistsError('Preserve existing trajectory artifact: '+str(target))
    sources=dict(parameter=parameter,parent_parameter=parent_parameter,structures=structures,integration=integration)
    if result.is_file():sources['result']=result
    elif not probe:raise FileNotFoundError('Actual final result is required: '+str(result))
    hashes={key:sha(path)for key,path in sources.items()}
    source=read(structures);fit=read(integration)
    if source['time_base']!=[1,12800]or fit['time_base']!=[1,12800]or source['source_sha256']!=fit['source_sha256']:
        raise ValueError('Source structures and integration native identity differ')
    if fit['full_source_skate_structures'].get('source_document_sha256')!=hashes['structures']:
        raise ValueError('Integration is not bound to the selected current source structures')
    parent_result=parent_stage/CASE/'fitted/result.npz';sources['parent_result']=parent_result;hashes['parent_result']=sha(parent_result)
    queries={'parent':SavedBladeProjection(project,parent_stage,parent_parameter),'current':SavedBladeProjection(project,stage,parameter)}
    pts=queries['current'].pts;np.testing.assert_array_equal(queries['parent'].pts,pts)
    current_saved=queries['current'].parameter
    if str(current_saved.get('source_structure_integration_sha256',''))!=hashes['integration']:
        raise ValueError('Current saved physical state is bound to a different source integration')
    masks=source_masks(pts,source,fit);models={};locals={};results={'parent':load_result(parent_result)};checks={};unresolved=[]
    if result.is_file():results['current']=load_result(result)
    else:unresolved.append('Live probe has no final result: final parameter/result identity and export closure are unverified')
    indices=np.array([np.flatnonzero(pts==round(t*12800))[0]for t in PROBE_SECONDS])if probe else np.arange(len(pts))
    for candidate,query in queries.items():
        candidate_stage=parent_stage if candidate=='parent'else stage
        model,local=geometry(query,candidate_stage);models[candidate]=model;locals[candidate]=local
        checks[candidate]=verify_result(query,results[candidate],model,local)if candidate in results else dict(final_result_unavailable=True)
        projection,closure=project_native(query,local,indices,model,results.get(candidate));model.update(projection);checks[candidate].update(closure)
        checks[candidate]['native_projected_exposures']=len(indices)
        for key,prediction in [('source_complete_structure_observation_count',masks['source_feature_count']),('source_support_location_in_optimization',masks['source_support_location_mask'])]:
            if candidate=='current'and candidate in results and key in results[candidate]:np.testing.assert_array_equal(results[candidate][key],prediction)
        if candidate in results and 'source_unassigned_association_json'in results[candidate]:
            model['saved_source_unassigned_association_json']=results[candidate]['source_unassigned_association_json'].copy()
        else:model['saved_source_unassigned_association_json']=np.array('[]')
    statistics={candidate:dict(origin_maximum_20ms_jump=jump_summary(model['origin_world_conditional_m'],pts),
        COM_proxy_maximum_20ms_jump=jump_summary(model['com_world_conditional_m'],pts),
        fixed_material_maximum_20ms_jump=jump_summary(model['material_station_world_conditional_m'],pts),
        moving_runner_witness_XY_span_88_92=[witness_span(model['exact_runner_witness_world_conditional_m'],pts,foot)for foot in(0,1)])for candidate,model in models.items()}
    diagnostics={}if probe else {candidate:factor_diagnostics(query,parent_stage if candidate=='parent'else stage,fit)for candidate,query in queries.items()}
    for key,path in sources.items():
        if sha(path)!=hashes[key]:raise ValueError('Input changed during read-only trajectory export: '+str(path))
    report=dict(schema=SCHEMA,probe_only=probe,inputs={key:dict(path=str(path),sha256=hashes[key])for key,path in sources.items()},
        implementation=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),source_sha256=source['source_sha256'],
        frames=len(pts),feet=2,time_base=[1,12800],source_seconds=[float(pts[0]/12800.),float(pts[-1]/12800.)],
        native_projection='Existing SavedBladeProjection, two batched physical-foot queries, native five-node implicit-row camera; no per-frame model construction',
        length_units=queries['current'].camera_info,world_conversion='Solver O/template/body lengths times length_export_ratio once; exported COM proxy already conditional metres',
        result_closure_checks=checks,statistics=statistics,source_assigned_foot_exposures=int(masks['source_assigned_region_mask'].sum()),
        source_unassigned_identity_exposures=int(masks['source_unassigned_exposure_mask'].sum()),support_location_rows=len(fit['source_support_evidence']),
        source_anatomical_identity='Original unassigned contours stay unassigned; saved/recomputed optimizer model association is not source identity',
        saved_unassigned_model_associations={candidate:json.loads(str(model['saved_source_unassigned_association_json']))for candidate,model in models.items()},
        unresolved=unresolved+['Camera/scene/scale covariance is unavailable; source hulls preserve finite sampled pixel disks with convexification',
            'Exact lowest steel witness can change arc/side; its trajectory is not a fixed material or measured contact trajectory',
            'COM is the archived body/shoe mass proxy, not measured true COM',
            'Partial, unknown, occluded and unmatched source features remain unresolved; factor distances do not establish acceptance'],
        no_fit_performed=True,no_GUI=True,no_video_generated=True,accuracy_accepted=False,user_accepted=False)
    output.mkdir(parents=True,exist_ok=True)
    if probe:
        write(output/'probe_check.json',report);return report
    arrays=dict(pts=pts.copy(),source_pts=pts.copy(),time_base=np.array([1,12800]),curve_labels=np.asarray(CURVE_LABELS),
        material_station_labels=np.array(['rear','mid','front']),metadata_json=np.array(json.dumps(report,ensure_ascii=False,default=serial)),**masks)
    for candidate,model in models.items():arrays.update({candidate+'_'+key:value for key,value in model.items()})
    np.savez_compressed(output/'trajectory.npz',**arrays);export_csv(output/'trajectory.csv',pts,models,masks)
    feature_csv(output/'feature_distances.csv',diagnostics,fit);write(output/'factor_diagnostics.json',dict(candidates=diagnostics,excluded_source_features=fit.get('excluded_features',[]),source_identity_unassigned=source.get('unassigned_identity_observations',[])))
    pictures=plots(output,pts,models,masks,diagnostics)
    report['artifacts']={name:dict(path=str(output/name),sha256=sha(output/name))for name in names if name!='trajectory_review.json'}
    report['plots']=[str(p)for p in pictures];write(output/'trajectory_review.json',report);return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('project','stage','parameter','result','parent-stage','parent-parameter','structures','integration','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--probe',action='store_true',help='Read-only live checkpoint check; no formal result/trajectory export')
    args=parser.parse_args();report=run(**vars(args))
    print(json.dumps(dict(output=str(args.output),probe_only=args.probe,frames=report['frames'],result_closure_checks=report['result_closure_checks'],accuracy_accepted=False),ensure_ascii=False))


if __name__=='__main__':main()
