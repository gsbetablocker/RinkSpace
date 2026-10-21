"""Read candidate same-corner continuations from qualified native source anchors.

This module owns no model, H, world coordinates, or new admission authority.
Actual two-edge readouts remain candidates until their exact physical entity
continuity is reviewed. Failure never becomes a nearby texture correspondence.
"""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import av
import cv2
import numpy as np
from .io_utils import read_json, read_jsonl, write_json, file_sha256, object_sha256
from .native_entity_tracklets import track_material_corner
from .source_texture_tracklets import _source_mask
from .source_worker import _channels, _remeasure


def _ref(path):
    p=Path(path).resolve()
    return dict(file=str(p),sha256=file_sha256(p))


def _write(stream,row):
    stream.write(json.dumps(row,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')


def acquire_candidates(scope_file,roles_file,anchor_inventory,actor_ledgers,output):
    out=Path(output).resolve()
    if out.exists():raise ValueError('New immutable native entity candidate generation required')
    scope=read_json(scope_file);source=scope['source'];inventory=read_json(anchor_inventory)
    if source['time_base']!='1/12800' or source['native_fps']!='50/1' or file_sha256(source['path'])!=source['sha256']:
        raise ValueError('Actual official 50fps native programme source required')
    roles=read_json(roles_file)['frame_roles'];shots={s['id']:s for s in scope['shots']}
    def verified(ref):
        p=Path(ref.get('file',ref.get('path')))
        if file_sha256(p)!=ref['sha256']:raise ValueError('Changed exact packet-bound entity anchor evidence: '+str(p))
        return p
    if inventory.get('new_exposure_qualification_issued')is not False:raise ValueError('Candidate input must not claim new-exposure qualification')
    verified(inventory['reader_module']);verified(inventory['source_packet'])
    anchor_rows=list(read_jsonl(verified(inventory['source_rows_ref'])))
    if len(anchor_rows)!=inventory['exact_qualified_existing_rows']:raise ValueError('Qualified anchor inventory denominator differs')
    clock={p:s['id']for s in scope['shots']if s['id']!='s01'for p in range(s['first_pts'],s['last_pts']+1,256)}
    ledgers={};ledgerrefs=[]
    for path in actor_ledgers:
        ledgerrefs.append(_ref(path))
        for row in read_jsonl(path):
            pts=int(row['native_pts'])
            if pts in ledgers and ledgers[pts]!=row:raise ValueError('Independent actor ledgers disagree at one exposure')
            ledgers[pts]=row
    anchors=defaultdict(list);anchor_keys={};duplicate_anchors=[]
    for a in sorted(anchor_rows,key=lambda a:(not a['source_row'].get('original_source_is_actual_manual_click',False),a['raw_index'])):
        row=a['source_row'];receipt=a['qualification_receipt'];pts=int(row['native_pts']);sid=row['shot_id']
        if (sid=='s01' or clock.get(pts)!=sid or roles.get(str(pts),{}).get('role')!='development_candidate'
            or receipt.get('new_actual_source_geometry_qualified')is not True or not receipt.get('physical_point')
            or not receipt.get('entity_station_fixed') or receipt.get('entity_kind')!=1
            or receipt['source_row_id']!=row['source_row_id'] or receipt['native_pts']!=pts
            or receipt['original_UV_preserved']!=row['source_uv']):
            raise ValueError('Entity candidate anchor is outside exact qualified native source scope')
        mean=object_sha256(dict(source_row_id=row['source_row_id'],native_pts=pts,
            current_2D_UV=row['current_uv'],estimator='native_two_edge_quadratic_intersection_v1'))
        if mean!=receipt['current_2D_source_mean_sha256'] or a['packet_uv']!=row['current_uv']:
            raise ValueError('Entity candidate anchor is not the actual packet-bound qualified measurement')
        edges=row['source_measurement_audit']['current_two_material_edges']
        if not edges['valid'] or edges['current_intersection_uv']!=row['current_uv'] or len(edges['edge_fits'])!=2:
            raise ValueError('Entity anchor lacks two current actual material edges')
        mi=int(receipt['existing_material_index']);path=receipt['existing_point_path'];key=(sid,mi,path,pts)
        if key in anchor_keys:
            duplicate_anchors.append(dict(shot_id=sid,native_pts=pts,existing_material_index=mi,existing_point_path=path,
                retained_original_source_row_id=anchor_keys[key]['source_row_id'],retained_raw_index=anchor_keys[key]['raw_index'],
                additional_original_source_row_id=row['source_row_id'],additional_raw_index=a['raw_index'],
                original_UV_versions_preserved_in_inventory=True,new_independent_weight_added=False))
            continue
        anchor_keys[key]=a;anchors[pts].append(a)
    if not anchors:raise ValueError('No exact qualified source anchors; no model fallback is permitted')
    out.mkdir(parents=True);(out/'ownership_crops').mkdir()
    active={};finished=[];counts=Counter();terminals=Counter();episodes=Counter();previous=None;oldpts=None;oldsid=None
    seen=set();all_pairs=defaultdict(set);candidate_rows=0;source_frames=0;last_decode=None
    cv2.setNumThreads(2)
    def crop(rgb,uv,track,stage):
        x,y=map(int,np.rint(uv));h,w=rgb.shape[:2];bounds=[max(0,x-64),max(0,y-64),min(w,x+65),min(h,y+65)]
        p=out/'ownership_crops'/(track['short_id']+'_'+stage+'.png')
        cv2.imwrite(str(p),rgb[bounds[1]:bounds[3],bounds[0]:bounds[2],::-1])
        return dict(**_ref(p),native_crop_xyxy=bounds,current_rgb_sha256=track['last_rgb_sha256'],
            native_pts=track['last_pts'],unmarked_current_source_pixels=True)
    def finish(key,reason,pts=None,failure=None):
        t=active.pop(key);terminals[reason]+=1
        record=dict(track_id=t['track_id'],shot_id=key[0],existing_material_index=key[1],existing_point_path=key[2],
            target_id=t['target_id'],first_pts=t['first_pts'],last_pts=t['last_pts'],rows=t['rows'],
            qualified_anchor_source_row_id=t['anchor']['source_row']['source_row_id'],
            anchor_qualification=t['anchor']['qualification_manifest'],ownership_seed=t['ownership_seed'],
            terminal_reason=reason,terminal_native_pts=pts,terminal_source_operator=failure,
            continuation_material_qualification='pending_exact_entity_continuity_review',source_geometry_eligible=False)
        if previous is not None and oldpts==t['last_pts']:record['ownership_last']=crop(previous,t['uv'],t,'last')
        finished.append(record)
    def emit(handle,t,uv,digest,diagnostic,operator,anchor_at_exposure=None):
        nonlocal candidate_rows
        key=t['key'];pts=t['last_pts'];identity=(key,pts)
        if identity in seen:raise ValueError('Same physical entity/exposure was counted twice')
        seen.add(identity)
        if t['rows'] and pts-t['previous_pts']==256:all_pairs[key[0]].add(t['previous_pts'])
        row=dict(track_id=t['track_id'],physical_track_id=t['track_id'],shot_id=key[0],native_pts=pts,time_base=[1,12800],
            current_uv=np.asarray(uv).tolist(),source_uv=np.asarray(uv).tolist(),target_id=t['target_id'],
            existing_material_index=key[1],existing_point_path=key[2],physical_point=True,entity_station_fixed=True,
            source_video_sha256=source['sha256'],current_rgb_sha256=digest,fit_role='development',
            source_mean_pixels_from_model=False,world_or_H_used=False,point_correspondence=True,
            qualification_anchor=t['anchor']['qualification_manifest'],
            qualified_anchor_source_row_id=t['anchor']['source_row']['source_row_id'],
            qualified_anchor_original_UV=t['anchor']['source_row']['source_uv'],
            inherited_physical_entity_is_candidate_not_new_source_authority=True,
            source_geometry_eligible=False,solver_eligible=False,statistical_objective_eligible=False,
            numeric_statistics_ready=False,source_precision_status='engineering_no_covariance',
            two_direction_tensor_diagnostic=diagnostic,current_source_operator=operator,
            actor_and_graphics_exclusion='same native RGB independent actor ledger and existing source mask policy',
            original_packet_measurement_priority=anchor_at_exposure is not None,
            original_packet_anchor=anchor_at_exposure)
        _write(handle,row);candidate_rows+=1;counts[key[0]]+=1;t['rows']+=1;t['previous_pts']=pts
    with gzip.open(out/'candidate_rows.jsonl.gz','wt',encoding='utf8')as handle,av.open(source['path'])as con:
        stream=con.streams.video[0]
        if str(stream.time_base)!='1/12800':raise ValueError('Wrong native decoder clock')
        for frame in con.decode(stream):
            pts=int(frame.pts)
            if pts not in clock:
                if pts>max(clock):break
                continue
            sid=clock[pts];last_decode=pts
            if not active and pts not in anchors and pts>max(anchors):break
            if not active and pts not in anchors:previous=None;oldpts=None;oldsid=None;continue
            if sid!=oldsid or oldpts is None or pts!=oldpts+256:
                for key in list(active):finish(key,'native_cut_or_discontinuous_source_clock',pts)
                previous=None
            ledger=ledgers.get(pts)
            if roles.get(str(pts),{}).get('role')!='development_candidate' or ledger is None or ledger.get('role')!='development_candidate' or ledger['shot_id']!=sid:
                for key in list(active):finish(key,'guard_heldout_or_missing_independent_current_actor_ledger',pts)
                previous=None;oldpts=None;oldsid=None;continue
            rgb=frame.to_ndarray(format='rgb24');mask,digest=_source_mask(rgb,ledger,source,pts,sid)
            channels=_channels(rgb);source_frames+=1
            current_anchors={}
            for a in anchors.get(pts,[]):
                r=a['source_row'];receipt=a['qualification_receipt'];key=(sid,int(receipt['existing_material_index']),receipt['existing_point_path'])
                if receipt.get('current_native_RGB_sha256')!=digest:raise ValueError('Qualified source corner anchor RGB differs from actual current decoder')
                current_anchors[key]=a
            for key in list(active):
                t=active[key]
                if key in current_anchors:
                    a=current_anchors[key];uv=np.asarray(a['packet_uv'],float);diagnostic=_remeasure(uv,[0,0],channels,'texture',corner=True)
                    edges=a['source_row']['source_measurement_audit']['current_two_material_edges']
                    operator=dict(valid=True,operator_version='native_two_edge_quadratic_intersection_v1',source_operator=edges,
                        reason='original_packet_qualified_source_measurement_retained',tracking_UV_did_not_replace_original=True)
                else:
                    a=None;operator=track_material_corner(previous,rgb,t['uv'],t['edge_tangents'],mask,current_channels=channels)
                    if not operator.get('valid'):finish(key,operator['reason'],pts,operator);continue
                    uv=np.asarray(operator['current_uv'],float);diagnostic=_remeasure(uv,[0,0],channels,'texture',corner=True)
                if diagnostic['status']!='entity_corner_two_direction_support':finish(key,'current_source_point_has_no_local_two_direction_information',pts,diagnostic);continue
                t['uv']=uv;t['edge_tangents']=[f['tangent']for f in operator['source_operator']['edge_fits']]
                t['last_pts']=pts;t['last_rgb_sha256']=digest
                emit(handle,t,uv,digest,diagnostic,operator,dict(source_row_id=a['source_row']['source_row_id'],packet_uv=a['packet_uv'],source_uv=a['source_row']['source_uv'])if a else None)
            for key,a in current_anchors.items():
                if key in active:continue
                uv=np.asarray(a['packet_uv'],float);u,v=np.rint(uv).astype(int)
                if not (50<=u<1870 and 50<=v<1030)or mask[max(0,v-48):min(1080,v+49),max(0,u-48):min(1920,u+49)].any():
                    terminals['qualified_anchor_current_support_excluded_or_image_edge']+=1;continue
                diagnostic=_remeasure(uv,[0,0],channels,'texture',corner=True)
                if diagnostic['status']!='entity_corner_two_direction_support':terminals['qualified_anchor_local_2D_information_unavailable_for_propagation']+=1;continue
                episodes[key]+=1;short=f'{sid}_{key[1]}_{episodes[key]}';tid='native_entity/'+short
                edges=a['source_row']['source_measurement_audit']['current_two_material_edges']
                t=dict(key=key,track_id=tid,short_id=short,target_id=a['source_row']['target_id'],anchor=a,uv=uv,
                    edge_tangents=[f['tangent']for f in edges['edge_fits']],first_pts=pts,last_pts=pts,
                    previous_pts=None,last_rgb_sha256=digest,rows=0)
                t['ownership_seed']=crop(rgb,uv,t,'seed');active[key]=t
                operator=dict(valid=True,operator_version='native_two_edge_quadratic_intersection_v1',source_operator=edges,
                    reason='original_packet_qualified_source_measurement_retained',tracking_UV_did_not_replace_original=True)
                emit(handle,t,uv,digest,diagnostic,operator,dict(source_row_id=a['source_row']['source_row_id'],packet_uv=a['packet_uv'],source_uv=a['source_row']['source_uv']))
            previous=rgb;oldpts=pts;oldsid=sid
    for key in list(active):finish(key,'source_end_or_scope_end',last_decode)
    write_json(out/'tracks.json',finished)
    coverage={}
    for sid,s in shots.items():
        if sid=='s01':continue
        possible=(s['last_pts']-s['first_pts'])//256;measured=len(all_pairs[sid])
        coverage[sid]=dict(possible_native_pairs=possible,candidate_source_pairs=measured,
            independently_material_qualified_new_pairs=0,unknown_qualified_source_motion_pairs=possible,
            candidate_rows=counts[sid],candidates_do_not_promote_unknown=True)
    manifest=dict(schema='rinkfit.native_fixed_entity_tracklets_candidate_generation.v1',source_video=source,
        scope=_ref(scope_file),roles=_ref(roles_file),anchor_inventory=_ref(anchor_inventory),
        packet_bound_anchor_reader=inventory['reader_module'],original_source_packet=inventory['source_packet'],
        independent_source_bound_actor_ledgers=ledgerrefs,candidate_rows=_ref(out/'candidate_rows.jsonl.gz'),
        tracks=_ref(out/'tracks.json'),rows=candidate_rows,track_episodes=len(finished),by_shot=dict(counts),
        candidate_motion_coverage=coverage,terminal_reasons=dict(terminals),decoded_source_frames_used=source_frames,
        duplicate_same_entity_exposure_anchor_versions_not_reweighted=duplicate_anchors,
        reader_modules=[_ref(__file__),_ref(Path(__file__).with_name('native_entity_tracklets.py'))],
        source_clock='official native 50fps PTS 1/12800 step256',original_anchor_measurements_unchanged=True,
        same_physical_entity_native_PTS_deduplicated=True,model_or_H_used=False,new_world_coordinates_created=False,
        source_geometry_qualification_issued=False,statistical_covariance_issued=False,independent_accuracy_claimed=False,
        limitations=['Numerical two-edge continuation is candidate evidence; exact entity continuity still requires source review.',
            'Single-edge, texture-only, excluded, or missing-role continuations terminate rather than inherit a world corner.',
            'Only current source RGB is measured; there is no cut/gap interpolation or model-filled trajectory.'])
    write_json(out/'manifest.json',manifest);return manifest


def main():
    p=argparse.ArgumentParser();p.add_argument('--scope',required=True);p.add_argument('--roles',required=True)
    p.add_argument('--anchors',required=True);p.add_argument('--actor-ledger',action='append',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();print(json.dumps(acquire_candidates(a.scope,a.roles,a.anchors,a.actor_ledger,a.out),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
