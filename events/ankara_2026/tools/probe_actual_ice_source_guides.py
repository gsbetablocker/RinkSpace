"""Read supplied original-source guides in exact source exposures, without FIT."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'calibration/program_tracking')]
from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import av
import cv2
import numpy as np
from rinkspace.vision_ice_source_guides import ActualIceSourceGuideTracker
from rinkspace.vision_ice_temporal_identity import TemporalIceIdentitySelector
from rinkspace.vision_programme_ice_review import IceProgrammeSourceReader
from rinkspace.programme_source_measurement_cache import restore_exact_navigation_queries,recheck_source_candidate_exclusions
from calibration.program_tracking.render_result import PersonMasks,DEFAULT_MASKS,DEFAULT_PEOPLE
from calibration.program_tracking.source_board_measurements import measure_board_source_frame
from calibration.program_tracking.far_rail_source_components import source_exclusion
from scripts.review_programme_source_inputs import _observed_ice_region,_ice_exclusion


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workroot',required=True,type=Path)
    p.add_argument('--source-guides',required=True,type=Path)
    p.add_argument('--pts',required=True,nargs='+',type=int)
    p.add_argument('--profile-backend',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--model-context-authority',type=Path)
    p.add_argument('--without-source-seeds',action='store_true')
    p.add_argument('--without-source-guides',action='store_true')
    p.add_argument('--all-curves',action='store_true')
    p.add_argument('--validate-source-graph',action='store_true')
    p.add_argument('--cpu-threads',type=int,default=1)
    args=p.parse_args();cv2.setNumThreads(args.cpu_threads)
    if args.output.exists():raise ValueError('Do not overwrite an earlier source guide readout')
    manifest=json.loads((args.workroot/'source_pixels/source_measurements.json').read_text(encoding='utf8'))
    packet=json.loads(args.source_guides.read_text(encoding='utf8'))
    identity_path=Path(manifest['camera_identity']);identity=json.loads(identity_path.read_text(encoding='utf8'))
    scene=json.loads((identity_path.parent/'scene_UNACCEPTED.json').read_text(encoding='utf8'))
    roles=json.loads((args.workroot.parents[3]/'observations/vision_training/paint_role_contract_current.json').read_text(encoding='utf8'))
    reader=IceProgrammeSourceReader(scene,source_sha256=manifest['source_sha256'],time_base=manifest['time_base'],role_contract=roles)
    with np.load(identity['camera_path']) as c:camera={int(p):q.copy() for p,q in zip(c['pts'],c['camera25'])}
    indexed={};handles=[]
    for path in sorted((args.workroot/'source_pixels').glob('source_pixels_*.npz')):
        cache=np.load(path,allow_pickle=False)
        for i,stamp in enumerate(cache['actual_pts']):
            if int(stamp) in args.pts:indexed[int(stamp)]=(cache,i,path)
        if any(v[0] is cache for v in indexed.values()):handles.append(cache)
        else:cache.close()
        if len(indexed)==len(set(args.pts)):break
    if set(indexed)!=set(args.pts):raise ValueError('Requested exact exposures are not in the source inventory')
    tracker=ActualIceSourceGuideTracker(manifest['inventory'],source_sha256=manifest['source_sha256'],time_base=manifest['time_base'],profile_backend=args.profile_backend)
    context=None
    if args.model_context_authority:
        from rinkspace.vision_ice_model_context_identity import SameStateIceModelContextIdentity
        authority=dict(path=str(args.model_context_authority.resolve()),sha256=hashlib.sha256(args.model_context_authority.read_bytes()).hexdigest())
        context=SameStateIceModelContextIdentity(manifest['inventory'],state_identity_path=identity_path,
            navigation_authority=authority,source_sha256=manifest['source_sha256'],time_base=manifest['time_base'])
    selector=TemporalIceIdentitySelector(manifest['inventory'],source_sha256=manifest['source_sha256'],time_base=manifest['time_base'],
        seed_registry=() if args.without_source_seeds else packet.get('identity_seeds',()),model_context_identity=context)
    graph=None;source_validation=[]
    if args.validate_source_graph:
        from rinkspace.vision_ice_identity_graph import IceSourceMaterialIdentityGraph
        from calibration.program_tracking.strict_absolute_source import normalize_selected_ice_rows
        graph=IceSourceMaterialIdentityGraph(manifest['inventory'],source_sha256=manifest['source_sha256'],time_base=manifest['time_base'],
            source_path=manifest['source_path'],device=args.profile_backend if args.profile_backend=='cpu' else 'cuda')
    masks=PersonMasks(DEFAULT_MASKS,DEFAULT_PEOPLE);rows=[];seen=set();saved=[]
    with av.open(manifest['source_path']) as container:
        stream=container.streams.video[0];stream.thread_count=args.cpu_threads
        container.seek(min(args.pts),stream=stream,backward=True)
        for source in container.decode(stream):
            pts=source.pts
            if pts is None or pts<min(args.pts):continue
            if pts>max(args.pts):break
            if pts not in indexed:continue
            cache,i,path=indexed[pts];rgb=source.to_ndarray(format='rgb24');bgr=rgb[:,:,::-1].copy()
            started=time.perf_counter()
            digest=hashlib.sha256(rgb.tobytes()).hexdigest()
            if digest!=str(cache['RGB_sha256'][i]):raise ValueError('Source RGB differs from frozen measured exposure')
            binding=dict(source_sha256=manifest['source_sha256'],time_base=manifest['time_base'],source_pts=int(pts),actual_pts=int(pts),
                native_pts=int(cache['native_pts'][i]),shot_id=str(cache['shot_id'][i]),native_hw=[1080,1920],decoded_native_BGR_sha256=str(cache['BGR_sha256'][i]))
            m=dict(source_binding=binding,decoded_native_RGB_sha256=digest)
            for prefix,names in [('query',('curve','uv','normal','interval','status')),('candidate',('uv','sides','query','flags'))]:
                lo,hi=cache[prefix+'_offsets'][i:i+2]
                for name in names:m[prefix+'_'+name]=cache[prefix+'_'+name][lo:hi]
            lo,hi=cache['candidate_offsets'][i:i+2]
            if 'candidate_side_quantiles_uv' in cache:m['candidate_side_quantiles_uv']=cache['candidate_side_quantiles_uv'][lo:hi]
            restore_exact_navigation_queries(m,reader,camera[binding['native_pts']]);nq,nc=len(m['query_curve']),len(m['candidate_uv'])
            excluded=source_exclusion(bgr,masks.static_scene_exclusion(binding['native_pts'])[0])
            board=measure_board_source_frame(bgr,excluded,binding);bands=board.get('ice_scope_boundary_candidates',board['boundary_candidates'])
            mask=_ice_exclusion(excluded,bands);scope=_observed_ice_region(bands)
            recheck_source_candidate_exclusions(m,mask)
            guided=tracker.consume(rgb,m,mask,source_ice_scope_mask=scope,
                guides=[] if args.without_source_guides else [g for g in packet['guides'] if g['actual_pts']==pts]);m=guided['measurement']
            result=selector.consume(rgb,m,mask,source_ice_scope_mask=scope);tracker.remember_selected_source_rows(result)
            if graph is not None:
                delayed=graph.consume(rgb,m,mask,source_ice_scope_mask=scope,selected_rows=result)
                preserved=graph.preserve_current_source_row_proofs(result)
                normalized=normalize_selected_ice_rows(result,m,source_binding=dict(binding,source_path=manifest['source_path']),
                    curve_inventory=manifest['inventory'])
                source_validation.append(dict(actual_pts=pts,preserved_rows=preserved['preserved_current_source_rows'],
                    normalized_rows=len(normalized['rows']),rejected=normalized['rejected_rows'],
                    delayed_actual_source_rows=[dict(actual_pts=r['actual_pts'],curve_id=r['curve_id'],
                        model_navigation_context=r.get('model_projection_used_as_navigation_and_identity_context'),
                        temporal_kind=r['temporal_evidence']['kind']) for r in delayed['rows']]))
                if normalized['rejected_rows']:raise ValueError('Actual same-source grid selector row failed the public source gate')
            wanted={g['curve_id'] for g in packet['guides']};current=result['rows'] if args.all_curves else [r for r in result['rows'] if r['curve_id'] in wanted]
            print(json.dumps(dict(actual_pts=pts,curves=[dict(curve_id=r['curve_id'],actual_readouts=len(r['uv'])) for r in current],
                decisions=[r for r in result['decisions'] if r['curve_id'] in wanted],
                actual_source_processing_s=round(time.perf_counter()-started,4),
                actual_model_context_source_seeds=len(context.previous_tracks) if context is not None else None),ensure_ascii=False))
            if 'candidate_side_quantiles_uv' not in m:
                m['candidate_side_quantiles_uv']=np.full((len(m['candidate_uv']),2,3,2),np.nan)
            arrays={k:np.asarray(m[k])[nq:] for k in ('query_curve','query_uv','query_normal','query_interval','query_status')}
            arrays.update({k:np.asarray(m[k])[nc:] for k in ('candidate_uv','candidate_sides','candidate_query','candidate_flags','candidate_side_quantiles_uv')})
            info=dict(source_binding=binding,decoded_native_RGB_sha256=digest,immutable_original_cache_path=str(path),original_actual_exposure_ordinal=i,
                original_query_prefix_length=nq,original_candidate_prefix_length=nc,source_guided_queries=m.get('source_guided_queries'),actual_selected_source_rows=current,
                model_context_decisions=result.get('model_context_decisions',()),actual_source_processing_s=time.perf_counter()-started)
            saved.append((pts,arrays,info));rows.extend(current);seen.add(pts)
    if seen!=set(args.pts):raise ValueError('Requested exact original source sequence was not fully read')
    packed={f'{pts}_{k}':v for pts,arrays,info in saved for k,v in arrays.items()}
    packed['metadata_json']=json.dumps(dict(schema='rinkspace.actual-source-guide-probe.v1',frames=[info for _,_,info in saved],
        actual_source_rows=rows,source_guide_packet=dict(path=str(args.source_guides.resolve()),sha256=hashlib.sha256(args.source_guides.read_bytes()).hexdigest()),
        source_graph_validation=source_validation,
        full_programme_visual_review_complete=False,fitting_started=False),ensure_ascii=False,separators=(',',':'))
    np.savez_compressed(args.output,**packed)
    if graph is not None:
        receipt=graph.save_compact_state(args.output.with_name(args.output.stem+'_material_graph.npz'))
        print(json.dumps(dict(source_graph_validation=source_validation,source_graph_receipt=receipt),ensure_ascii=False))
    for cache in handles:cache.close()


if __name__=='__main__':main()
