"""Prototype exact chart re-encoding into existing kind2 print-affine semantics.

Each physical ad's captured kind6 contours become fixed local points sharing
one ordinary six-coefficient affine. Its off-diagonal coefficients remain fixed
at zero, leaving four common placement scalars. Orphaned old vertex blocks are
kept immutable in value and made constant. This changes no source observation.
"""
from __future__ import annotations
import argparse,copy,json
from pathlib import Path
from collections import Counter
import numpy as np
from rinkfit_start.native_scene import NativeScene
from .joint_model import as_event_view,validate_physical_domain
from .io_utils import read_json,write_json,object_sha256
from .source_ad22_capture_v2 import ref
from .print_chart_domain import decode_common_witness


def extract_points(model,path):
    b=model['blocks'][path['geometry_block']];q=np.asarray(b['values'],float)
    indices=path.get('geometry_indices');q=q[indices]if indices else q[int(path.get('geometry_offset',0)):]
    if q.size<2 or q.size%2:raise ValueError('invalid_existing_kind6_vertex_coordinates')
    return q.reshape(-1,2)


def convert(model,advertisement_ids=(1,4,24)):
    result=copy.deepcopy(model);updates=[];freezes=[];groups=[];claimed={};names={b['name']:i for i,b in enumerate(model['blocks'])}
    for ad in advertisement_ids:
        ai=names['ad/'+str(ad)];installation=np.asarray(model['blocks'][ai]['values'],float);width=installation[1];height=installation[3]-installation[2]
        if min(width,height)<=0:raise ValueError('nonpositive_existing_same_ad_chart')
        members=[(mi,pi,p)for mi,m in enumerate(model['materials'])if m['label'].startswith('Ad'+str(ad)+'.')for pi,p in enumerate(m['paths'])if p['kind']==6]
        if not members:raise ValueError('same_physical_ad_has_no_captured_kind6_group')
        bi=len(result['blocks']);name='print/group/Ad'+str(ad)+'/new_native'
        if name in names:raise ValueError('common_placement_block_already_exists')
        # Same ordinary six-coefficient bounds/scales used by current print
        # affines. The two fixed zero off-diagonals are scalar constraints.
        block=dict(name=name,values=[1.,0.,0.,0.,1.,0.],lower=[-10.]*6,upper=[10.]*6,
            scales=[1.,1.,.1,1.,1.,.1],constant=False,fixed_indices=[1,3])
        result['blocks'].append(block);local_all=[];old_blocks=set();paths=[]
        for mi,pi,p in members:
            oldbi=p['geometry_block'];old_blocks.add(oldbi)
            if oldbi in claimed and claimed[oldbi]!=ad:raise ValueError('original_vertex_block_shared_between_physical_ads')
            claimed[oldbi]=ad;points=extract_points(model,p);local=(points-installation[[0,2]])/np.array([width,height])
            n=len(points);closed=bool(p.get('closed',False));edges=n if closed else max(n-1,0)
            fixed=np.vstack([local,local[0]])if closed else local.copy();local_all.extend(local.tolist())
            new=copy.deepcopy(p);new.update(kind=2,geometry_block=bi,advertisement_block=ai,points=fixed.tolist())
            new.pop('geometry_indices',None);new.pop('geometry_offset',None)
            new['provenance']=json.dumps(dict(original_provenance=p.get('provenance'),
                representation='exact_captured_absolute_course_height_to_existing_same_ad_chart',
                source_UV_ID_station_and_physical_entity_unchanged=True,
                existing_ad_block=model['blocks'][ai]['name'],shared_group_affine_block=name,
                fixed_off_diagonal_affine_indices=[1,3],original_kind6_closed=closed,
                appended_first_local_vertex=closed,original_edges=edges,
                local_points_are_chart_coordinates_not_new_world_measurements=True),ensure_ascii=False)
            result['materials'][mi]['paths'][pi]=new
            rec=dict(material_index=mi,path_index=pi,label=model['materials'][mi]['label'],expected_path=copy.deepcopy(p),path=new,
                     original_geometry_block=oldbi,original_vertices=n,original_edges=edges,new_fixed_vertices=len(fixed),
                     new_kind2_edges=max(len(fixed)-1,0),closed=closed)
            if rec['new_kind2_edges']!=edges:raise ValueError('finite_station_edge_count_changed')
            updates.append(rec);paths.append(dict(material_index=mi,path_index=pi,original_vertices=n,closed=closed,original_edges=edges,new_edges=rec['new_kind2_edges']))
        for oldbi in sorted(old_blocks):
            for mi,m in enumerate(model['materials']):
                for pi,p in enumerate(m['paths']):
                    if p.get('geometry_block')==oldbi and (mi,pi)not in {(mii,pii)for mii,pii,pp in members}:
                        raise ValueError('orphan_vertex_block_has_a_surviving_other_material_user')
            original=model['blocks'][oldbi];result['blocks'][oldbi]['constant']=True
            freezes.append(dict(block_index=oldbi,block_name=original['name'],expected_values=original['values'],expected_constant=original.get('constant',False),constant=True))
        local_all=np.asarray(local_all);external=[dict(material_index=mi,path_index=pi,label=m['label'])for mi,m in enumerate(model['materials'])for pi,p in enumerate(m['paths'])if p.get('advertisement_block')==ai and not m['label'].startswith('Ad'+str(ad)+'.print.')]
        groups.append(dict(ad=ad,advertisement_block=ai,advertisement_name=model['blocks'][ai]['name'],existing_installation_values=installation.tolist(),existing_installation_constant=model['blocks'][ai].get('constant',False),new_affine_block_index=bi,new_affine_block=block,free_scalar_indices=[0,2,4,5],fixed_zero_scalar_indices=[1,3],group_material_paths=len(members),orphaned_vertex_blocks=len(old_blocks),local_coordinate_min=local_all.min(0).tolist(),local_coordinate_max=local_all.max(0).tolist(),vertices_outside_unit_rectangle=int(np.any((local_all<0)|(local_all>1),axis=1).sum()),external_entity_paths=external,physical_vertex_height_bounds=[float(installation[2]+height*local_all[:,1].min()),float(installation[2]+height*local_all[:,1].max())],unit_rectangle_containment_not_claimed=True,paths=paths))
    for u in freezes:
        if result['blocks'][u['block_index']]['values']!=u['expected_values']:raise ValueError('orphan_vertex_values_changed')
    old_names=[b['name']for b in model['blocks']]
    if [b['name']for b in result['blocks'][:len(old_names)]]!=old_names:raise ValueError('original_block_indices_changed')
    overlay=dict(schema='rinkfit.prototype.shared_print_group_chart_reencoding.v1',base_model_object_sha256=object_sha256(model),original_block_count=len(model['blocks']),new_blocks=[g['new_affine_block']for g in groups],existing_installation_value_guards=[dict(block_index=g['advertisement_block'],expected_values=g['existing_installation_values'])for g in groups],orphaned_vertex_block_updates=freezes,material_path_updates=updates,groups=groups,source_ID_UV_native_PTS_station_changed=False,scene_head_oldprint_and_Ad22_unchanged=True,source_identity_or_measured_panel_bounds_created=False,production_packet_not_rebound=True)
    return result,overlay


def apply_converter_overlay(model,overlay):
    if len(model['blocks'])!=overlay['original_block_count']:raise ValueError('prototype_chart_base_block_count_changed')
    result=copy.deepcopy(model)
    for g in overlay['existing_installation_value_guards']:
        if result['blocks'][g['block_index']]['values']!=g['expected_values']:raise ValueError('same_ad_installation_chart_changed_recompute_converter')
    for u in overlay['orphaned_vertex_block_updates']:
        b=result['blocks'][u['block_index']]
        if b['name']!=u['block_name']or b['values']!=u['expected_values']or b.get('constant',False)!=u['expected_constant']:raise ValueError('captured_vertex_block_changed')
        b['constant']=u['constant']
    result['blocks'].extend(copy.deepcopy(overlay['new_blocks']))
    for u in overlay['material_path_updates']:
        mi,pi=u['material_index'],u['path_index']
        if result['materials'][mi]['paths'][pi]!=u['expected_path']:raise ValueError('captured_material_path_changed')
        result['materials'][mi]['paths'][pi]=copy.deepcopy(u['path'])
    return result


def run(model_file,output):
    out=Path(output)
    if out.exists():raise ValueError('immutable prototype chart output exists')
    model=read_json(model_file);result,overlay=convert(model);copy_result=apply_converter_overlay(model,overlay)
    if result!=copy_result:raise ValueError('converter_overlay_consumer_not_equivalent')
    domain=validate_physical_domain(result)
    if not domain['passed']:raise ValueError('prototype_kind2_production_physical_domain_failed:'+','.join(domain['violations']))
    adids=tuple(a['advertisement_id']for a in model.get('print_only_domain_witness',{}).get('advertisements',[]));_,chart=decode_common_witness(result,advertisement_ids=adids)
    if not chart['local_original_domain_equivalence_certified']:raise ValueError('existing_original_chart_domain_failed')
    S0=NativeScene(as_event_view(model,next(iter(model['shots']))));S1=NativeScene(as_event_view(result,next(iter(result['shots']))));checks=[];maximum=0.;vertices=0;closed=0
    for u in overlay['material_path_updates']:
        mi,pi,n,edges=u['material_index'],u['path_index'],u['original_vertices'],u['original_edges'];tv=np.arange(n)/edges if edges else np.zeros(n);s=np.unique(np.r_[tv,np.linspace(0,1,max(33,edges*8+1)),1.]);X0=S0.world(mi,pi,s);X1=S1.world(mi,pi,s);error=float(np.max(np.abs(X0-X1)));maximum=max(maximum,error);vertices+=n;closed+=u['closed'];checks.append(dict(material_index=mi,path_index=pi,original_vertices=n,original_edges=edges,new_kind2_edges=u['new_kind2_edges'],closed=u['closed'],full_original_vertex_stations=tv.tolist(),sampled_stations=len(s),max_world_difference_L=error))
    if maximum>1e-12:raise ValueError('kind2_reencoding_changes_finite_curve_world_geometry')
    out.mkdir(parents=True);overlay['base_model']=ref(model_file);write_json(out/'converter_overlay.json',overlay);write_json(out/'model_prototype.json',result);write_json(out/'production_physical_domain.json',domain);write_json(out/'existing_print_chart_domain.json',chart)
    report=dict(schema='rinkfit.prototype.shared_print_group_chart_equivalence.v1',base_model=ref(model_file),prototype=ref(out/'model_prototype.json'),converter_overlay=ref(out/'converter_overlay.json'),producer=ref(__file__),groups=overlay['groups'],groups_count=len(overlay['groups']),group_free_scalars=12,orphan_vertex_blocks_fixed=len(overlay['orphaned_vertex_block_updates']),material_paths_converted=len(checks),closed_paths_first_vertex_appended=closed,all_original_vertices=vertices,all_station_edge_counts_preserved=True,max_world_difference_all_vertices_and_dense_stations_L=maximum,finite_curve_checks=checks,production_physical_domain_passed=domain['passed'],existing_original_print_chart_domain_certified=chart['local_original_domain_equivalence_certified'],new_group_identity_affines_have_strict_original_standard_coefficient_boxes=True,new_local_points_not_clamped_to_unit_rectangle=True,source_station_ID_UV_and_PTS_unchanged=True,scene_head_oldprint_Ad22_parameters_unchanged=True,prototype_not_production_packet_or_stage=True,full_source_station_forward_and_S01_proofs_pending=True,limits=['Existing kind2 local point coordinates are not clipped to [0,1]. Ad4 and Ad24 retain their captured geometry even where their old numerical installation charts do not contain the points in the unit rectangle. This is not a new claim that the installation is a measured outer panel.','Only fixed local contours and one ordinary affine per physical ad are used; the old independent vertex blocks are constant and unreferenced.','The free family intentionally permits only common four-scalar placement, not the old independent vertex deformation space.','The current pipeline packet is not rebound. Native CPU/GPU consumers and next-stage registration must be checked independently.'])
    write_json(out/'representation_QA.json',report);print(json.dumps(dict(prototype=ref(out/'model_prototype.json'),overlay=ref(out/'converter_overlay.json'),groups=3,paths=len(checks),closed=closed,vertices=vertices,max_world_difference_L=maximum,physical_domain_passed=domain['passed'])))
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.model,a.out)
if __name__=='__main__':main()
