"""Append reviewed RGB glyph edges on exact finite parent-contour aliases.

Shared geometry, its derivatives and all old source measurements are preserved.
Only source-confirmed local edges can choose these finite supports.
"""
from pathlib import Path
from copy import deepcopy
import argparse,json,numpy as np
import prepare_native_multicamera as p
from prepare_ad4_stroke_support import split_arcs,containing_arc
from audit_ad4_source_correspondence import closed_vertices,source_station
from prepare_spatial_refinement import collect_additions,revise_observations,_initialize_touched,_preserve_old_rows,_write_prepared
from build_ad3_completion import layout_without_content_identity
from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import SceneParameterModel,ParameterState
from rinkspace.domain.print_assets import evaluate_print_path
from rinkspace.session.gpu_geometry import NativeSceneGPU
from rinkspace.path_policy import guard_project_write


def main(a):
    for path in (a.output,a.asset_output,a.scene_output):
        guard_project_write(path.resolve())
        if path.exists():raise FileExistsError(path)
    m=p.read(a.prepared/'joint_manifest.json');mapping=p.read(a.prepared/'mapping.json');points=p.npz(a.prepared/'point_observations.npz');state=p.npz(a.state)
    scene=p.read(m['scene_path']);old_model=SceneParameterModel(scene,Path(m['scene_path']),optimize_scene=True)
    p.validate_snapshot(state,old_model.initial_state.layout,mapping,m,points)
    if p.digest(a.prepared/'point_observations.npz')!=m['point_input_sha256']:raise ValueError('Current input changed')
    ref=next(r for r in scene['advertisement_assets'] if r['advertisement_number']==4);old_asset=p.read(ref['path']);asset=deepcopy(old_asset);template=next(t for t in asset['templates'] if t['id']==ref['template_id']);existing={r['id']:r for r in template['paths']}
    source=p.read(a.source/'readouts.json');atlas=p.read(a.source/'word_reference.json');reference={r['path_id']:r for r in atlas['paths']};review=p.read(a.source/'source_review.json');review_by={(r['pts'],r['short_index']):r for r in review['rows']};accepted=[];bindings=[];used={};partitions={}
    for row in source['rows']:
        verdict=review_by[row['pts'],row['short_index']]
        for name in ('pts','short_index','path_id','uv','source_query_reference_uv'):
            if not np.array_equal(row[name],verdict[name]):raise ValueError('Reviewed source identity changed: '+name)
        if verdict['status']!='identity_compatible' or verdict.get('reference_same_arc_review')!='compatible':continue
        if any(row['source_profile_flags'].values()) or row.get('model_UV_synthesized') is not False:raise ValueError('Ambiguous/synthetic source reading')
        name=row['path_id'].split('::',1)[1]
        if 'JUNIOR_' not in name or '_inner/' in name or name not in reference or name not in existing:raise ValueError('Existing exterior JUNIOR reference required')
        parent=existing[name];photograph=reference[name]
        if photograph.get('source_closed') is not True or len(photograph['pixels'])!=len(parent['material_uv']):raise ValueError('Original source/glyph vertex identity changed')
        if name not in partitions:
            partition=deepcopy(parent);partition['shape_provenance']={'source_native_boundary':photograph['pixels']};partitions[name]=split_arcs(partition,1.25)
        raw=closed_vertices(photograph['pixels'],True);q=source_station(raw,row['source_query_reference_uv'],True,0.)
        if q['source_edge_distance_px']>1e-7:raise ValueError('Reference query is not on declared edge')
        child,u=containing_arc(partitions[name],q['station']*(len(raw)-1));used[child['id']]=child
        new=deepcopy(row);new.update(path_id='print_instance/Ad4::'+child['id'],parent_path_id=row['path_id'],station=u,kind=1,fit_eligible=True,source_review_path=str(a.source/'source_review.json'),source_review_sha256=p.digest(a.source/'source_review.json'),source_review_disposition='same_visible_physical_edge_and_material_role_confirmed',source_review_reason=verdict['reason'],user_visual_acceptance=False,pixel_accuracy_accepted=False);accepted.append(new);bindings.append(dict(pts=row['pts'],short_index=row['short_index'],parent_path_id=row['path_id'],finite_path_id=new['path_id'],initial_free_station=u,reference_query_uv=row['source_query_reference_uv'],actual_source_uv=row['uv'],reference_edge_distance_px=q['source_edge_distance_px']))
    if not accepted:raise ValueError('No individually reviewed glyph edges')
    children=list(used.values());internal=template['internal_geometry']
    for child in children:
        if child['id'] in existing:raise ValueError('Preserve existing finite alias')
        binding=internal['path_bindings'][child['source_support_parent']]
        if binding['kind']!='shape_group':raise ValueError('Existing shared shape group required')
        internal['path_bindings'][child['id']]=deepcopy(binding);group=next(g for g in internal['groups'] if g['id']==binding['group_id']);group['path_ids'].append(child['id'])
        if internal.get('layout_policy'):next(g for g in internal['layout_policy']['groups'] if g['id']==group['component_identity'])['path_ids'].append(child['id'])
    template['paths'].extend(children);template['reviewed_glyph_support_revision']=dict(parent_asset_sha256=ref['sha256'],source_reference=str(a.source/'word_reference.json'),source_reference_sha256=p.digest(a.source/'word_reference.json'),source_review=str(a.source/'source_review.json'),source_review_sha256=p.digest(a.source/'source_review.json'),exact_parent_vertices=True,no_new_geometry_parameters=True,source_uv_pts_unchanged=True)
    p.write(a.asset_output,asset);ref.update(path=str(a.asset_output.resolve()),sha256=p.digest(a.asset_output));p.write(a.scene_output,scene)
    model=SceneParameterModel(scene,a.scene_output,optimize_scene=True)
    if layout_without_content_identity(model.initial_state.layout)!=layout_without_content_identity(old_model.initial_state.layout):raise ValueError('Finite supports changed parameter layout')
    core=NativeCore(m['native_library']);ps=ParameterState(model.initial_state.layout,state['scene_values']);old_ps=ParameterState(old_model.initial_state.layout,state['scene_values']);board=ps.apply_scene(scene)['board'];geometry_error=0.;derivative_error=0.
    for child in children:
        support=child['source_support'];u=np.linspace(0,1,7);parent_u=((support['parent_segment_begin']+u*(support['parent_segment_stop_unwrapped']-support['parent_segment_begin']))%support['parent_segment_count'])/support['parent_segment_count'];old=evaluate_print_path(old_asset,'print_instance/Ad4',child['source_support_parent'],board,core,parent_u,parameter_state=old_ps);new=evaluate_print_path(asset,'print_instance/Ad4',child['id'],board,core,u,parameter_state=ps);geometry_error=max(geometry_error,float(np.max(abs(old['xyz']-new['xyz']))))
        for block in old['jacobians_by_block']:derivative_error=max(derivative_error,float(np.max(abs(old['jacobians_by_block'][block]-new['jacobians_by_block'][block]))))
    if geometry_error>1e-12 or derivative_error>1e-10:raise ValueError('Finite supports changed geometry or its derivatives')
    mapped=deepcopy(mapping);mapped['path_ids']+=['print_instance/Ad4::'+c['id'] for c in children];native=NativeSceneGPU(a.scene_output,core,parameter_state=ps)
    if not set(mapped['path_ids'])<=set(native.paths):raise ValueError('Native finite support unresolved')
    mapped['scene_revision']=native.scene_revision;target=deepcopy(m);target.update(scene_path=str(a.scene_output.resolve()),scene_file_sha256=p.digest(a.scene_output),scene_revision=native.scene_revision)
    links=a.source/'qualified_source_links.json';p.write(links,dict(source_readouts=dict(path=str(a.source/'readouts.json'),sha256=p.digest(a.source/'readouts.json')),source_review=dict(path=str(a.source/'source_review.json'),sha256=p.digest(a.source/'source_review.json')),source_profiles=[r['source_profile_proof'] for r in accepted],existing_asset=dict(path=ref['path'],sha256=ref['sha256']),source_UV_synthesized=False,fit_eligible=True,admission_basis='actual RGB flags plus individually reviewed same local physical glyph edge; exact finite aliases share original geometry'))
    packet=a.source/'qualified_packet.json';p.write(packet,dict(shot='s08',fit_eligible=True,source_UV_synthesized=False,source_packet=str(a.source/'word_reference.json'),source_packet_sha256=p.digest(a.source/'word_reference.json'),source_links=str(links),source_links_sha256=p.digest(links),accepted=accepted,user_visual_acceptance=False,pixel_accuracy_accepted=False))
    additions,aliases,packets=collect_additions([packet],points,target,mapped['path_ids']);new,numeric,shots,touched,decisions,added,origins=revise_observations(points,state['stations'],target,mapped['path_ids'],dict(decisions=[],added_observations=additions))
    if decisions:raise ValueError('Append-only source repair cannot change old observations')
    notes=_initialize_touched(new,numeric,shots,touched,state,target,mapped,model);_preserve_old_rows(new,points,numeric,state['stations'],origins)
    receipt=dict(schema='rinkspace.reviewed_glyph_source_support.v1',parent_prepared=str(a.prepared),parent_state=str(a.state),parent_state_sha256=p.digest(a.state),parent_input_sha256=m['point_input_sha256'],added_rows=len(additions),new_finite_aliases=len(children),new_geometry_parameters=0,max_parent_child_xyz_difference=geometry_error,max_shared_derivative_difference=derivative_error,old_fields_and_numeric_stations_byte_preserved=True,all_numeric_globals_unchanged=True,source_uv_pts_unchanged=True,source_packets=packets,duplicate_aliases=aliases,bindings=bindings,station_initialization=notes,fit_started=False)
    result,mapped=_write_prepared(a.output,new,numeric,shots,state,target,mapped,model,receipt)
    if 'spatial_refinement_preparation' in m:result['spatial_refinement_preparation']=m['spatial_refinement_preparation']
    else:result.pop('spatial_refinement_preparation',None)
    result['reviewed_glyph_source_support']=dict(receipt_path=str(a.output/'source_glyph_preparation.json'),source_packet=str(packet),source_packet_sha256=p.digest(packet),new_geometry_parameters=0,added_rows=len(additions));p.write(a.output/'joint_manifest.json',result);p.write(a.output/'source_glyph_preparation.json',receipt)
    print(json.dumps(dict(output=str(a.output),added_rows=len(additions),total_rows=len(new['pts']),point_input_sha256=result['point_input_sha256'],new_finite_aliases=len(children),geometry_error=geometry_error,derivative_error=derivative_error)),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','state','source','asset-output','scene-output','output'):parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
