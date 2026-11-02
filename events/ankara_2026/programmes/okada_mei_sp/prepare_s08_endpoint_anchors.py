"""Append reviewed current-RGB endpoints on already-existing finite print arcs."""
from pathlib import Path
from copy import deepcopy
import argparse,json
import numpy as np
import prepare_native_multicamera as p
from prepare_ad4_stroke_support import split_arcs,containing_arc
from audit_ad4_source_correspondence import closed_vertices,source_station
from prepare_spatial_refinement import collect_additions,revise_observations,_initialize_touched,_preserve_old_rows,_write_prepared
from rinkspace.domain.parameters import SceneParameterModel
from rinkspace.path_policy import guard_project_write


def main(a):
    out=guard_project_write(a.output.resolve())
    if out.exists():raise FileExistsError(out)
    m=p.read(a.prepared/'joint_manifest.json');mapping=p.read(a.prepared/'mapping.json');points=p.npz(a.prepared/'point_observations.npz');state=p.npz(a.state)
    if p.digest(a.prepared/'point_observations.npz')!=m['point_input_sha256']:raise ValueError('Current input changed')
    model=SceneParameterModel(p.read(m['scene_path']),Path(m['scene_path']),optimize_scene=True);p.validate_snapshot(state,model.initial_state.layout,mapping,m,points)
    source=p.read(a.source/'readouts.json');reference=p.read(a.source/'word_reference.json');reference_paths={r['path_id']:r for r in reference['paths']};review=p.read(a.source/'source_review.json');review_by={(r['pts'],r['short_index']):r for r in review['rows']}
    scene=p.read(m['scene_path']);ref=next(r for r in scene['advertisement_assets'] if r['advertisement_number']==4);asset=p.read(ref['path']);template=next(t for t in asset['templates'] if t['id']==ref['template_id'])
    parents={r['id']:r for r in template['paths'] if '.ANKARA.' in r['id'] and not r.get('source_support_parent')};existing={r['id']:r for r in template['paths']};arcs={n:split_arcs(r,1.25) for n,r in parents.items()}
    accepted=[];bindings=[];skipped=[]
    for row in source['rows']:
        qualified=review_by[row['pts'],row['short_index']]
        if not np.array_equal(row['uv'],qualified['uv']) or row['path_id']!=qualified['path_id'] or not np.array_equal(row['source_query_reference_uv'],qualified['source_query_reference_uv']):raise ValueError('Reviewed source reading changed')
        if qualified['status']!='identity_compatible' or qualified.get('reference_same_arc_review')!='compatible':continue
        if any(row['source_profile_flags'].values()) or row.get('model_UV_synthesized') is not False:raise ValueError('Ambiguous/synthetic source reading')
        parent=row['path_id'].split('::',1)[1]
        if a.existing_glyph_paths:
            if parent not in reference_paths or parent not in existing:raise ValueError('Existing source glyph required')
            photographed=reference_paths[parent];child=existing[parent]
            if photographed.get('source_closed') is not True or child.get('closed') is not True or 'JUNIOR_' not in parent or '_inner/' in parent:raise ValueError('Only the selected existing JUNIOR exterior contours are supported')
            if len(photographed['pixels'])!=len(child['material_uv']):raise ValueError('Original source/glyph vertex identity changed')
            raw=closed_vertices(photographed['pixels'],True);correspondence=source_station(raw,row['source_query_reference_uv'],True,0.);station=correspondence['station'];name=row['path_id']
        else:
            raw=closed_vertices(parents[parent]['shape_provenance']['source_native_boundary'],True);correspondence=source_station(raw,row['source_query_reference_uv'],True,0.)
            child,station=containing_arc(arcs[parent],correspondence['station']*(len(raw)-1));name='print_instance/Ad4::'+child['id']
        if correspondence['source_edge_distance_px']>1e-7:raise ValueError('Reference query is not on its declared edge')
        if name not in mapping['path_ids'] or child['id'] not in existing:
            if a.existing_arcs_only:
                skipped.append(dict(pts=row['pts'],short_index=row['short_index'],reason='No existing finite native arc; source identity alone does not create geometry',finite_path_id=name));continue
            raise ValueError('Existing finite arc required; no new geometry is added')
        if a.minimum_existing_arc_distance>0:
            prior=(points['pts']==row['pts'])&(points['path_indices']==mapping['path_ids'].index(name))
            if prior.any() and np.min(np.linalg.norm(points['uv'][prior]-np.asarray(row['uv']),axis=1))<a.minimum_existing_arc_distance:
                skipped.append(dict(pts=row['pts'],short_index=row['short_index'],reason='Existing source point on the same finite arc already covers this local boundary',finite_path_id=name,minimum_spacing_px=a.minimum_existing_arc_distance));continue
        current=existing[child['id']]
        if not a.existing_glyph_paths and (current['source_support_parent']!=parent or not np.array_equal(current['material_uv'],child['material_uv'])):raise ValueError('Existing finite arc differs from its original parent geometry')
        new=deepcopy(row);new.update(path_id=name,parent_path_id=row['path_id'],station=station,kind=1,fit_eligible=True,source_review_disposition='same_visible_physical_edge_and_material_role_confirmed',source_review_reason=qualified['reason'],source_review_path=str(a.source/'source_review.json'),source_review_sha256=p.digest(a.source/'source_review.json'),user_visual_acceptance=False,pixel_accuracy_accepted=False)
        accepted.append(new);bindings.append(dict(pts=row['pts'],short_index=row['short_index'],parent_path_id=row['path_id'],finite_path_id=name,initial_free_station=station,source_reference_query_uv=row['source_query_reference_uv'],actual_source_uv=row['uv'],reference_edge_distance_px=correspondence['source_edge_distance_px']))
    if not accepted:raise ValueError('No reviewed source edges qualify')
    links=a.source/'qualified_source_links.json';p.write(links,dict(source_readouts=dict(path=str(a.source/'readouts.json'),sha256=p.digest(a.source/'readouts.json')),source_review=dict(path=str(a.source/'source_review.json'),sha256=p.digest(a.source/'source_review.json')),source_profiles=[r['source_profile_proof'] for r in accepted],existing_asset=dict(path=ref['path'],sha256=ref['sha256']),source_UV_synthesized=False,admission_basis='current RGB flags plus individually reviewed same physical edge; query only selects an already existing finite arc',fit_eligible=True))
    packet=a.source/'qualified_packet.json';p.write(packet,dict(shot='s08',fit_eligible=True,source_UV_synthesized=False,source_packet=str(a.source/'word_reference.json'),source_packet_sha256=p.digest(a.source/'word_reference.json'),source_links=str(links),source_links_sha256=p.digest(links),accepted=accepted,user_visual_acceptance=False,pixel_accuracy_accepted=False))
    additions,aliases,packets=collect_additions([packet],points,m,mapping['path_ids']);new,numeric,shots,touched,decisions,added,origins=revise_observations(points,state['stations'],m,mapping['path_ids'],dict(decisions=[],added_observations=additions))
    if decisions:raise ValueError('Append-only source repair cannot alter existing observations')
    notes=_initialize_touched(new,numeric,shots,touched,state,m,mapping,model);_preserve_old_rows(new,points,numeric,state['stations'],origins)
    receipt=dict(schema='rinkspace.endpoint_source_anchor_preparation.v1',parent_prepared=str(a.prepared),parent_state=str(a.state),parent_state_sha256=p.digest(a.state),parent_input_sha256=m['point_input_sha256'],source_packets=packets,added_rows=len(additions),existing_glyph_paths=a.existing_glyph_paths,skipped_source_candidates=skipped,minimum_existing_arc_distance_px=a.minimum_existing_arc_distance,duplicate_aliases=aliases,bindings=bindings,station_initialization=notes,old_fields_byte_preserved=True,scene_geometry_unchanged=True,numeric_global_arrays_unchanged=True,new_geometry_parameters=0,all_parent_observations_retained=True,no_source_weights=True,fit_started=False)
    target,mapped=_write_prepared(out,new,numeric,shots,state,m,mapping,model,receipt)
    if 'spatial_refinement_preparation' in m:target['spatial_refinement_preparation']=m['spatial_refinement_preparation']
    else:target.pop('spatial_refinement_preparation',None)
    target['endpoint_source_anchors']=dict(receipt_path=str(out/'source_anchor_preparation.json'),source_packet=str(packet),source_packet_sha256=p.digest(packet),added_rows=len(additions),existing_geometry_only=True)
    p.write(out/'joint_manifest.json',target);p.write(out/'source_anchor_preparation.json',receipt)
    print(json.dumps(dict(output=str(out),added=len(additions),total_rows=len(new['pts']),point_input_sha256=target['point_input_sha256'],bindings=bindings)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','state','source','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--existing-glyph-paths',action='store_true',help='Bind reviewed exterior edges to already existing individual JUNIOR glyphs')
    parser.add_argument('--existing-arcs-only',action='store_true',help='Record and omit newly reviewed edges without an existing finite native path')
    parser.add_argument('--minimum-existing-arc-distance',type=float,default=0.,help='Avoid counting nearby readings of the same existing edge twice')
    args=parser.parse_args()
    if not np.isfinite(args.minimum_existing_arc_distance) or args.minimum_existing_arc_distance<0:parser.error('Minimum distance cannot be negative')
    main(args)
