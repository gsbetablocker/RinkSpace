"""Verify real source seed exposures against saved pixels and fresh board scope."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'calibration/program_tracking')]
from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import av
import cv2
import numpy as np
from rinkspace.vision_ice_temporal_identity import TemporalIceIdentitySelector,_point_polyline_distance
from rinkspace.vision_ice_nested_material_identity import SourceHouseLayerIdentityExpander
from rinkspace.vision_ice_source_role_override import remeasure_confirmed_source_roles
from rinkspace.programme_source_measurement_cache import recheck_source_candidate_exclusions,restore_exact_navigation_queries
from rinkspace.vision_programme_ice_review import IceProgrammeSourceReader
from rinkspace.vision_ice_local_clearance import KnownIceMaterialBoxClearance,primary_person_box_components
from calibration.program_tracking.render_result import PersonMasks,DEFAULT_MASKS,DEFAULT_PEOPLE
from calibration.program_tracking.source_board_measurements import measure_board_source_frame
from calibration.program_tracking.far_rail_source_components import source_exclusion
from scripts.review_programme_source_inputs import _ice_exclusion,_observed_ice_region


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workroot',type=Path,required=True)
    parser.add_argument('--pts',type=int,nargs='+',required=True)
    parser.add_argument('--inspect',action='store_true')
    parser.add_argument('--nested-house',action='store_true')
    parser.add_argument('--role-corrections',type=Path)
    parser.add_argument('--save-source-seeds',action='store_true')
    parser.add_argument('--local-clearance',action='store_true')
    args=parser.parse_args();cv2.setNumThreads(1)
    manifest=json.loads((args.workroot/'source_pixels/source_measurements.json').read_text(encoding='utf8'))
    registry=json.loads((args.workroot/'ice_source_identity_seeds.json').read_text(encoding='utf8'))
    identity_path=Path(manifest['camera_identity']);identity=json.loads(identity_path.read_text(encoding='utf8'))
    scene=json.loads((identity_path.parent/'scene_UNACCEPTED.json').read_text(encoding='utf8'))
    roles=json.loads((args.workroot.parents[3]/'observations/vision_training/paint_role_contract_current.json').read_text(encoding='utf8'))
    navigation_reader=IceProgrammeSourceReader(scene,source_sha256=manifest['source_sha256'],time_base=manifest['time_base'],role_contract=roles)
    with np.load(identity['camera_path'],allow_pickle=False) as camera:
        camera_by_pts={int(p):v.copy() for p,v in zip(camera['pts'],camera['camera25'])}
    indexed={};handles=[]
    for path in sorted((args.workroot/'source_pixels').glob('source_pixels_*.npz')):
        data=np.load(path,allow_pickle=False)
        for ordinal,pts in enumerate(data['actual_pts']):
            if int(pts) in args.pts:indexed[int(pts)]=(data,ordinal,path)
        if any(row[0] is data for row in indexed.values()):handles.append(data)
        else:data.close()
        if len(indexed)==len(set(args.pts)):break
    if set(indexed)!=set(args.pts):raise ValueError('Requested original exposure measurements missing')
    masks=PersonMasks(DEFAULT_MASKS,DEFAULT_PEOPLE)
    outputs=[]
    local_clearance=KnownIceMaterialBoxClearance(manifest['inventory'],source_sha256=manifest['source_sha256'],
        time_base=manifest['time_base'],seed_registry=registry) if args.local_clearance else None
    new_seeds=[]
    clearance_records=[]
    role_data=json.loads(args.role_corrections.read_text(encoding='utf8')) if args.role_corrections else {'records':[]}
    continuous_selector=None
    with av.open(manifest['source_path']) as video:
        stream=video.streams.video[0];stream.thread_count=1
        video.seek(min(args.pts),stream=stream,backward=True)
        for source in video.decode(stream):
            pts=source.pts
            if pts is None or pts<min(args.pts):continue
            if pts>max(args.pts):break
            if pts not in indexed:continue
            data,i,path=indexed[pts]
            rgb=source.to_ndarray(format='rgb24');bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            digest=hashlib.sha256(rgb.tobytes()).hexdigest()
            if digest!=str(data['RGB_sha256'][i]):raise ValueError('Saved source RGB differs')
            binding=dict(source_sha256=manifest['source_sha256'],source_pts=int(pts),actual_pts=int(pts),
                native_pts=int(data['native_pts'][i]),time_base=manifest['time_base'],shot_id=str(data['shot_id'][i]),
                decoded_native_BGR_sha256=str(data['BGR_sha256'][i]),native_hw=[1080,1920])
            measurement=dict(source_binding=binding,decoded_native_RGB_sha256=digest)
            for prefix,names in [('query',['curve','uv','normal','interval','status']),('candidate',['uv','sides','query','flags'])]:
                start,end=data[prefix+'_offsets'][i:i+2]
                for name in names:measurement[prefix+'_'+name]=data[prefix+'_'+name][start:end]
            restore_exact_navigation_queries(measurement,navigation_reader,camera_by_pts[binding['native_pts']])
            original_q,original_c=len(measurement['query_curve']),len(measurement['candidate_uv'])
            exclusion=source_exclusion(bgr,masks.static_scene_exclusion(binding['native_pts'])[0])
            board=measure_board_source_frame(bgr,exclusion,binding)
            bands=board.get('ice_scope_boundary_candidates',board['boundary_candidates'])
            scope=_observed_ice_region(bands);ice_mask=_ice_exclusion(exclusion,bands)
            recheck_source_candidate_exclusions(measurement,ice_mask)
            if local_clearance is not None:
                parts=primary_person_box_components(masks,binding['native_pts'])
                clearance=local_clearance.consume(rgb,measurement,ice_mask,source_ice_scope_mask=scope,
                    primary_box_mask=parts['primary_box_mask'],measured_person_mask=parts['measured_person_mask'],
                    other_person_boxes_mask=parts['other_person_boxes_mask'])
                measurement=clearance['measurement'];ice_mask=clearance['exclusion_mask'];scope=clearance['source_ice_scope_mask']
                clearance_records.extend(dict(proof,original_candidate_index=index,
                    retained_original_source_uv=np.asarray(measurement['candidate_uv'][index]).tolist())
                    for index,proof in clearance['candidate_local_source_clearance'].items())
                print(json.dumps(dict(actual_pts=pts,source_local_clearance=clearance['decisions'],
                    primary_box_index=parts['primary_box_index'],clearance_candidates=len(clearance['candidate_local_source_clearance'])),ensure_ascii=False))
            inventory=manifest['inventory'];extra_seeds=[]
            if args.role_corrections:
                current={v['curve_id']:v for v in role_data['records']}
                measurement,inventory=remeasure_confirmed_source_roles(rgb,measurement,ice_mask,inventory,current)
                authority=dict(kind='assistant_actual_original_RGB_physical_material_identity_review',path=str(args.role_corrections.resolve()),
                    sha256=hashlib.sha256(args.role_corrections.read_bytes()).hexdigest())
                for role in role_data['records']:
                    if role['actual_pts']!=pts:continue
                    if role['decoded_native_BGR_sha256']!=hashlib.sha256(bgr.tobytes()).hexdigest():raise ValueError('Role source proof differs from original video')
                    extra_seeds.append(dict(curve_id=role['curve_id'],physical_id=role['feature_id'],actual_pts=pts,native_pts=pts,
                        source_sha256=manifest['source_sha256'],time_base=manifest['time_base'],shot_id=binding['shot_id'],
                        decoded_native_RGB_sha256=digest,navigation_uv=role['navigation_uv'],
                        source_material_class='static_painted_ice',surface_and_physical_identity_confirmed=True,
                        authority=authority,physical_identity_basis=role['source_physical_identity_basis'],
                        is_exact_centerline_pixel_ground_truth=False))
            if args.local_clearance and continuous_selector is not None:
                selector=continuous_selector
            else:
                selector=TemporalIceIdentitySelector(inventory,source_sha256=manifest['source_sha256'],
                    time_base=manifest['time_base'],seed_registry=registry)
                if args.local_clearance:continuous_selector=selector
            result=selector.consume(rgb,measurement,ice_mask,seeds=extra_seeds,source_ice_scope_mask=scope)
            if args.nested_house:
                expanded=SourceHouseLayerIdentityExpander(inventory,source_sha256=manifest['source_sha256'],
                    time_base=manifest['time_base']).consume(rgb,measurement,ice_mask,source_ice_scope_mask=scope,selected_rows=result)
                print(json.dumps(dict(actual_pts=pts,nested_material_decisions=expanded['decisions']),ensure_ascii=False))
                result['rows'].extend(expanded['rows'])
                new_seeds.extend(r['source_seed_reference'] for r in expanded['rows'])
            if len(measurement['candidate_uv'])>original_c:
                arrays={name:np.asarray(measurement[name])[original_q:] for name in ('query_curve','query_uv','query_normal','query_interval','query_status')}
                arrays.update({name:np.asarray(measurement[name])[original_c:] for name in
                    ('candidate_uv','candidate_sides','candidate_query','candidate_flags','candidate_side_quantiles_uv')})
                arrays['metadata_json']=json.dumps(dict(source_binding=binding,decoded_native_RGB_sha256=digest,
                    immutable_original_cache_path=str(path),original_actual_exposure_ordinal=i,original_query_prefix_length=original_q,
                    original_candidate_prefix_length=original_c,source_guided_queries=measurement['source_guided_queries'],
                    actual_selected_source_rows=result['rows'],candidate_local_source_clearance=measurement.get('candidate_local_source_clearance',{})),
                    ensure_ascii=False,separators=(',',':'))
                np.savez_compressed(args.workroot/f'actual_nested_source_overlay_{pts}.npz',**arrays)
            if local_clearance is not None:local_clearance.remember_current_source_rows(result,measurement)
            new_seeds.extend(r['source_seed_reference'] for r in result['rows'] if r['curve_id'] in {s['curve_id'] for s in extra_seeds})
            seed_names={r['curve_id'] for r in registry['records'] if r['actual_pts']==pts}
            seed_names.update(r['curve_id'] for r in extra_seeds)
            decisions=[r for r in result['decisions'] if r['curve_id'] in seed_names]
            output=dict(actual_pts=pts,seed_decisions=decisions,
                admitted_curves=[dict(curve_id=r['curve_id'],actual_points=len(r['uv'])) for r in result['rows']])
            print(json.dumps(output,ensure_ascii=False));outputs.append(output)
            if args.inspect:
                cv2.imwrite(str(args.workroot/f'ice_identity_seed_original_{pts}.jpg'),bgr)
                print(json.dumps(dict(actual_pts=pts,actual_board_bands=[dict(xrange=np.asarray(b['yellow_lower_uv'])[[0,-1],0].tolist(),
                    yrange=[float(np.asarray(b['yellow_lower_uv'])[:,1].min()),float(np.asarray(b['yellow_lower_uv'])[:,1].max())],
                    points=len(b['yellow_lower_uv'])) for b in bands]),ensure_ascii=False))
                for name in seed_names:
                    ci=[r['curve_id'] for r in manifest['inventory']].index(name)
                    cands=np.flatnonzero(measurement['query_curve'][measurement['candidate_query']]==ci)
                    current_seed=next(s for s in registry['records'] if s['actual_pts']==pts and s['curve_id']==name)
                    distance=_point_polyline_distance(measurement['candidate_uv'][cands],current_seed['navigation_uv'])
                    cands=cands[distance<40.]
                    pix=np.rint(measurement['candidate_uv'][cands]).astype(int)
                    print(json.dumps(dict(curve_id=name,actual_pts=pts,
                        candidate_uv=measurement['candidate_uv'][cands].tolist(),flags=measurement['candidate_flags'][cands].tolist(),
                        source_scope=scope[pix[:,1],pix[:,0]].tolist()),ensure_ascii=False))
    for data in handles:data.close()
    (args.workroot/'ice_identity_seed_verification.json').write_text(json.dumps(outputs,ensure_ascii=False,separators=(',',':')),encoding='utf8')
    if args.local_clearance:
        np.savez_compressed(args.workroot/f'ice_source_material_local_clearance_{min(args.pts)}_{max(args.pts)}.npz',
            actual_pts=np.array([v['actual_pts'] for v in clearance_records],np.int64),
            original_candidate_index=np.array([v['original_candidate_index'] for v in clearance_records],np.int32),
            uv=np.array([v['retained_original_source_uv'] for v in clearance_records],np.float64).reshape(-1,2),
            records_json=json.dumps(clearance_records,ensure_ascii=False,separators=(',',':')),
            all_programme_visual_review_complete=False)
    if args.save_source_seeds and new_seeds:
        merged={(v['curve_id'],v['actual_pts']):v for v in new_seeds}
        payload=dict(schema='rinkspace.actual-source-nested-material-and-event-role-confirmations.v1',
            source_sha256=manifest['source_sha256'],time_base=manifest['time_base'],records=list(merged.values()),
            full_programme_visual_review_complete=False)
        encoded=json.dumps(payload,ensure_ascii=False,separators=(',',':')).encode('utf8')
        content_hash=hashlib.sha256(encoded).hexdigest()
        authority=args.workroot/f'ice_source_identity_confirmations_{min(args.pts)}_{max(args.pts)}_{content_hash[:16]}.json'
        if authority.exists() and authority.read_bytes()!=encoded:raise ValueError('An immutable source identity authority changed')
        if not authority.exists():authority.write_bytes(encoded)
        reference=dict(kind='assistant_actual_original_RGB_physical_material_identity_review',path=str(authority.resolve()),
            sha256=hashlib.sha256(authority.read_bytes()).hexdigest())
        retained={(v['curve_id'],v['actual_pts']):v for v in registry['records']}
        retained.update({key:dict(v,authority=reference) for key,v in merged.items()})
        registry['records']=list(retained.values())
        registry['records'].sort(key=lambda v:(v['actual_pts'],v['curve_id']))
        earliest={}
        for row in registry['records']:earliest.setdefault((row['shot_id'],row['curve_id']),row)
        registry['earliest_curve_shot_anchors']=[dict(shot_id=shot,curve_id=curve,actual_pts=row['actual_pts'],authority=row['authority'])
            for (shot,curve),row in earliest.items()]
        (args.workroot/'ice_source_identity_seeds.json').write_text(json.dumps(registry,ensure_ascii=False,separators=(',',':')),encoding='utf8')
        print(json.dumps(dict(saved_actual_source_supplement=len(merged),registered_curves=len({v['curve_id'] for v in registry['records']})),ensure_ascii=False))


if __name__=='__main__':main()
