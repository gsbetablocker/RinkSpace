"""Source-only full-cut5 original FIT/independent-board-track overlap candidates.

Uses the existing low-source audit's numeric ledgers and conservative gates.
No camera projection, scene, source guard, residual, graph, J or solver is read.
Candidates remain unknown material until both actual RGB endpoints are reviewed.
"""
from pathlib import Path
import argparse, collections, hashlib, json, os, sys
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
sys.dont_write_bytecode=True
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import av, cv2, numpy as np
from scipy.spatial import cKDTree
from audit_original_low_motion_far_board import (
    ADMISSION, SELECTED, RAW, RETIREMENT, ICE, TRACKS, PEOPLE, DATA, sha, read)

cv2.setNumThreads(1)
PREP=DATA/'calibration/runs/solver_acceleration_20261003/prepared_full8503_1791035775270492900/complete_original_sources'

def numeric(path, keys):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k].copy() for k in keys}

def source_people_clear(points, stamp, people):
    if stamp not in people:
        return np.zeros(len(points),bool)
    keep=np.ones(len(points),bool)
    for box in people[stamp]:
        x0,y0,x1,y1=np.asarray(box[:4])*2
        dx=max(48.,.4*(x1-x0));dy=48.
        keep &= ~((points[:,0]>=x0-dx)&(points[:,0]<=x1+dx)&
                  (points[:,1]>=y0-dy)&(points[:,1]<=y1+dy))
    return keep

def source_UI_clear(points):
    # The identical native fixed UI rectangles used by the existing source
    # board navigator. These are pixel roles, never a world/print boundary.
    return ~(((points[:,0]<585)&(points[:,1]<265))|
             ((points[:,0]>=1660)&(points[:,0]<1895)&(points[:,1]>=930)&(points[:,1]<1020)))

def refs(path): return dict(path=str(path),sha256=sha(path))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--clock-state',type=Path,required=True,
                    help='Read only actual low stamps and the original typed-ice index contract')
    ap.add_argument('--existing-role-packet',type=Path,required=True,
                    help='Source-only admitted low53 packet, never imported as a runtime binding')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--representative-bin-seconds',type=float,default=5.)
    ap.add_argument('--numeric-only',action='store_true')
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if not np.isfinite(a.representative_bin_seconds) or a.representative_bin_seconds<=0:
        raise ValueError('Positive source-time representative spacing required')
    retirement=read(RETIREMENT)
    for path,key in ((ADMISSION,'original_foreground_admission_npz'),
                     (SELECTED,'original_selected_rows_npz'),(RAW,'original_motion_npz')):
        assert sha(path)==retirement[key]['sha256']
    admission=numeric(ADMISSION,('active_original_global_index','source_uv_a','source_uv_b',
        'source_endpoint_index_a','source_endpoint_index_b','original_sigma_xy'))
    selected=numeric(SELECTED,('native_pair_pts','offsets','original_global_row_indices'))
    raw=numeric(RAW,('pair_pts','offsets','xy0','xy1'))
    tracks=numeric(TRACKS,('source_pts','target_pts','shot_index','offsets','uv0','uv1','forward_backward_error_px'))
    track_receipt_path=TRACKS.parent/'source_extraction_receipt.json'
    track_receipt=read(track_receipt_path)
    assert track_receipt['source_UV_selected_by_model'] is False
    assert track_receipt['no_cross_cut_pairs'] is True
    assert track_receipt['source_board_navigation']=='existing_native_RGB_cap_and_yellow_foot_strip'
    assert track_receipt['missing_people_masks']==0
    assert track_receipt['source_sha256_from_existing_verified_ledger']==retirement['source_video_sha256']
    saved=numeric(a.clock_state,('aux__block_lengths','aux__block_names','aux__canonical_pts',
                               'aux__qualified_original_motion_ice_contract_json'))
    lengths=saved['aux__block_lengths']//4
    bi=list(saved['aux__block_names']).index('low');lo=int(lengths[:bi].sum())
    stamps=saved['aux__canonical_pts'][lo:lo+int(lengths[bi])]
    assert len(stamps)==1851 and bi==3
    active=admission['active_original_global_index'];ids=selected['original_global_row_indices']
    full=31987+np.arange(len(ids));remap=np.full(len(admission['source_uv_a']),-1,int)
    remap[active]=np.arange(len(active))
    assert len(active)==434991 and len(ids)==319568
    assert np.array_equal(admission['source_uv_a'][full],raw['xy0'][ids])
    assert np.array_equal(admission['source_uv_b'][full],raw['xy1'][ids])
    aliasref=retirement['source_actual_exposure_alias_authority']
    assert sha(aliasref['path'])==aliasref['sha256']
    aliasz=numeric(aliasref['path'],('pts','exposure_representative_pts'))
    aliases=dict(zip(map(int,aliasz['pts']),map(int,aliasz['exposure_representative_pts'])))
    rawpair=np.repeat(np.arange(len(raw['pair_pts'])),np.diff(raw['offsets']))[ids]
    assert np.array_equal(selected['native_pair_pts'],raw['pair_pts'])
    aa=stamps[admission['source_endpoint_index_a'][full]]
    bb=stamps[admission['source_endpoint_index_b'][full]]
    assert np.array_equal(aa,[aliases[int(raw['pair_pts'][j,0])] for j in rawpair])
    assert np.array_equal(bb,[aliases[int(raw['pair_pts'][j,1])] for j in rawpair])
    known_ice={int(r['original_global_index']) for r in read(ICE)['qualified_rows'] if r['split']=='fit'}
    retired={int(r['current_bundle_motion_row']) for r in retirement['records']}
    original_typed_path=PREP/'low_source/low_source_factory/finite_material_source_binding.json'
    original_typed=read(original_typed_path)
    typed={int(r['global_bundle_index']) for r in original_typed}
    ice_contract=json.loads(str(saved['aux__qualified_original_motion_ice_contract_json'].item()))
    typed_ice={int(i) for i in ice_contract['global_bundle_indices']}
    role=read(a.existing_role_packet)
    if role.get('source_packets'):
        lowrefs=[r for r in role['source_packets'] if 'low' in str(r['path']).lower()]
        if len(lowrefs)!=1:raise ValueError('Provide the direct admitted low53 packet')
        ref=lowrefs[0];assert sha(ref['path'])==ref['sha256'];role=read(ref['path'])
    existing={int(r.get('current_original_prefix_bundle_index',r.get('original_global_bundle_index')))
              for r in role['records']}
    assert len(original_typed)==274 and len(typed_ice)==3 and len(existing)==53
    unavailable=typed|typed_ice|existing|retired
    people={int(r['pts']):r['boxes'] for r in read(PEOPLE)['frames']}
    lookup={}
    for ti,(pa,pb,shot) in enumerate(zip(tracks['source_pts'],tracks['target_pts'],tracks['shot_index'])):
        if int(shot)!=5:continue
        pair=(int(pa),int(pb))
        if pair in lookup:raise ValueError('Duplicate independent actual cut5 pair')
        lookup[pair]=ti
    bypair=[np.flatnonzero(rawpair==j) for j in range(len(raw['pair_pts']))]
    records=[];groups=[];total=collections.Counter();start=int(min(aa.min(),bb.min()))
    width=int(round(a.representative_bin_seconds*15360))
    for j,pair in enumerate(raw['pair_pts']):
        pa,pb=map(lambda p:aliases[int(p)],pair)
        rows=bypair[j];prefix=remap[full[rows]];active_rows=rows[prefix>=0]
        g=dict(original_raw_pair_index=j,source_native_pair_pts=list(map(int,pair)),source_pts=pa,target_pts=pb,
               original_selected_FIT_count=len(rows),active_original_FIT_count=len(active_rows),
               source_time_seconds_from_cut5_start=(pa-start)/15360.,candidate_count=0)
        total['selected_original_FIT']+=len(rows);total['active_original_FIT']+=len(active_rows)
        if pa==pb:
            g['status']='same_actual_exposure_not_motion';groups.append(g);continue
        usable=[int(ri) for ri in active_rows if int(remap[full[ri]]) not in unavailable and int(ids[ri]) not in known_ice]
        rows=np.asarray(usable,int);g['untyped_nonretired_not_previously_qualified_ice_count']=len(rows)
        total['untyped_nonretired_not_previously_qualified_ice']+=len(rows)
        reverse=False;ti=lookup.get((pa,pb))
        if ti is None:ti=lookup.get((pb,pa));reverse=True
        if ti is None:
            g['status']='independent_source_actual_pair_missing';groups.append(g);continue
        tl,th=map(int,tracks['offsets'][ti:ti+2]);ta,tb=tracks['uv0'][tl:th],tracks['uv1'][tl:th]
        if reverse:ta,tb=tb,ta
        g.update(independent_actual_pair_index=ti,independent_pair_reversed=reverse,independent_source_track_count=len(ta))
        if not len(ta) or not len(rows):
            g['status']='independent_source_board_navigation_sparse_or_no_original_unknowns';groups.append(g);continue
        p,q=raw['xy0'][ids[rows]],raw['xy1'][ids[rows]]
        _,near=cKDTree(np.c_[ta,tb]).query(np.c_[p,q])
        ea=np.linalg.norm(p-ta[near],axis=1);eb=np.linalg.norm(q-tb[near],axis=1)
        overlap=(ea<=2.)&(eb<=2.)
        pc=source_people_clear(p,pa,people)&source_people_clear(q,pb,people)
        ui=source_UI_clear(p)&source_UI_clear(q)
        scope=(p[:,1]<800)&(q[:,1]<800)
        keep=overlap&pc&ui&scope
        g.update(double_endpoint_overlap_count=int(overlap.sum()),
            source_people_or_missing_people_gate_excluded=int((overlap&~pc).sum()),
            fixed_UI_excluded=int((overlap&~ui).sum()),source_y_candidate_scope_excluded=int((overlap&~scope).sum()),
            people_boxes_available_both_endpoints=pa in people and pb in people,
            candidate_count=int(keep.sum()),status='source_only_candidates_pending_RGB' if keep.any() else 'no_qualified_numeric_overlap')
        total['double_endpoint_overlap']+=int(overlap.sum())
        for ri,k in zip(rows[keep],np.flatnonzero(keep)):
            fulli=int(full[ri]);prefixi=int(remap[fulli]);rawi=int(ids[ri])
            records.append(dict(original_raw_cut5_point_index=rawi,original_raw_pair_index=j,
                original_raw_pair_local_index=rawi-int(raw['offsets'][j]),original_selected_cut5_row=int(ri),
                original_full_low_bundle_row=fulli,current_original_prefix_bundle_index=prefixi,
                source_pts=pa,target_pts=pb,source_native_pair_pts=list(map(int,pair)),
                source_uv=p[k].tolist(),target_uv=q[k].tolist(),source_sigma_px=admission['original_sigma_xy'][fulli].tolist(),
                deep_LK_endpoint_difference_px=[float(ea[k]),float(eb[k])],independent_track_index=int(tl+near[k]),
                independent_actual_pair_index=ti,independent_pair_reversed=reverse,
                independent_deep_LK_FB_native_px=float(tracks['forward_backward_error_px'][tl+near[k]]),
                role_status='original_unknown_depth_FIT; both_native_RGB_material_inspection_pending',
                both_end_material_reviewed=False,source_only_people_clear=True,source_fixed_UI_clear=True,
                known_ice_or_existing_typed_index=False,source_y_less_than800_is_not_material_identity=True,
                board_navigation_does_not_exclude_all_nearfield_equipment=True,
                material_role_not_inferred_from_model_residual=True))
        groups.append(g)
    assert len({r['current_original_prefix_bundle_index'] for r in records})==len(records)
    bins=[]
    for b in range((max(max(g['source_pts'],g['target_pts']) for g in groups)-start)//width+1):
        rr=[g for g in groups if (g['source_pts']-start)//width==b]
        if not rr:continue
        bins.append(dict(bin_index=b,source_lower_pts=start+b*width,source_upper_pts=start+(b+1)*width,
            original_pair_count=len(rr),candidate_pairs=sum(g['candidate_count']>0 for g in rr),
            candidate_points=sum(g['candidate_count'] for g in rr),
            untyped_original_points=sum(g.get('untyped_nonretired_not_previously_qualified_ice_count',0) for g in rr),
            independent_pair_missing=sum(g['status']=='independent_source_actual_pair_missing' for g in rr)))
    picked=[]
    for b in bins:
        available=[g for g in groups if g['candidate_count'] and b['source_lower_pts']<=g['source_pts']<b['source_upper_pts']]
        if available:
            middle=(b['source_lower_pts']+b['source_upper_pts'])/2
            picked.append(min(available,key=lambda g:(abs(g['source_pts']-middle),g['original_raw_pair_index'])))
    receipt=[]
    if not a.numeric_only:
        wanted={p for g in picked for p in (g['source_pts'],g['target_pts'])};frames={}
        with av.open(retirement['source_video']) as container:
            stream=container.streams.video[0];stream.thread_count=1;assert str(stream.time_base)=='1/15360'
            if wanted:container.seek(min(wanted),stream=stream,backward=True,any_frame=False)
            for frame in container.decode(stream):
                if not wanted or frame.pts is not None and frame.pts>max(wanted):break
                if frame.pts in wanted:frames[int(frame.pts)]=frame.to_ndarray(format='bgr24')
                if set(frames)==wanted:break
        if set(frames)!=wanted:raise ValueError('Exact representative actual RGB unavailable')
        for group in picked:
            rr=[r for r in records if r['original_raw_pair_index']==group['original_raw_pair_index']]
            files=[];views=[]
            for pts,key in ((group['source_pts'],'source_uv'),(group['target_pts'],'target_uv')):
                image=frames[pts];out=a.output/f'raw_native_{pts}.png'
                assert cv2.imwrite(str(out),image)
                files.append(dict(actual_pts=pts,path=str(out),sha256=sha(out),decoded_BGR_sha256=hashlib.sha256(image.tobytes()).hexdigest()))
                view=image.copy()
                for r in rr:cv2.circle(view,tuple(np.rint(r[key]).astype(int)),4,(40,255,40),1)
                views.append(cv2.resize(view,(960,540)))
            overview=a.output/f'pair{group["original_raw_pair_index"]}_source_only_candidate_navigation.png'
            assert cv2.imwrite(str(overview),np.hstack(views))
            receipt.append(dict(**group,source_RGB=files,overview=str(overview),
                visual_material_inspection_status='pending; navigation overview cannot qualify every point'))
    report=dict(schema='rinkspace.original-low-FIT-full-cut5-source-only-board-overlap-candidate.v1',
        status='numeric_source_overlap_candidates_only_not_admitted',source_video_sha256=retirement['source_video_sha256'],
        time_base='1/15360',block='low',shot_index=5,selection='All original cut5 native pairs; no near-target interval and no residual/model selection',
        original_source_ledger=refs(ADMISSION),original_selected_rows=refs(SELECTED),original_raw_motion=refs(RAW),
        clock_alias_authority=aliasref,clock_state=dict(path=str(a.clock_state),read_arrays=list(saved),q_or_camera_values_read=False),
        independent_v2_tracks=refs(TRACKS),independent_v2_source_receipt=refs(track_receipt_path),
        original_ice_source=refs(ICE),people_boxes=refs(PEOPLE),static_UI_retirement=refs(RETIREMENT),
        original_typed_prefix_contract=refs(original_typed_path),existing_low53_role_packet=refs(a.existing_role_packet),
        gates=dict(double_endpoint_limit_native_px=2.,same_actual_exposure_excluded=True,
            conservative_people_box_native_padding=dict(horizontal='max(48, .4*bbox width)',vertical=48),
            missing_people_box_frame_is_rejected=True,source_fixed_UI_rectangles_same_as_existing_navigation=True,
            native_y_less_than800_is_only_candidate_scope=True,source_RGB_cap_and_yellow_strip_navigation_inherited=True,
            nearfield_equipment_and_remaining_ice_cannot_be_certified_numerically=True),
        prefix_original_UV_sigma_PTS_and_indices_bitexact=True,original_prefix_count=len(active),cut5_selected_original_count=len(ids),
        existing_typed_indices_excluded=dict(old274=sorted(typed),original_ice3=sorted(typed_ice),admitted_low53=sorted(existing)),
        summary=dict(total,source_candidate_count=len(records),candidate_pairs=sum(g['candidate_count']>0 for g in groups),
            total_original_raw_pairs=len(groups),source_first_actual_pts=start,
            source_last_actual_pts=max(max(g['source_pts'],g['target_pts']) for g in groups)),
        temporal_bins=bins,all_original_pair_coverage=groups,records=records,representative_source_RGB=receipt,
        representative_selection='Closest candidate pair to the centre of each fixed source-time bin; source time only, no residual sorting',
        source_measurements_or_roles_modified=False,new_source_rows=0,HELD_promoted=False,
        candidates_are_still_original_unknown_depth_FIT=True,source_guard_geometry_projection_graph_J_solver_loaded=False,
        accuracy_accepted=False)
    out=a.output/'original_low_cut5_full_source_overlap_candidates_v2.json'
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(dict(path=str(out),sha256=sha(out),summary=report['summary'],temporal_bins=bins,
                         representative_pairs=[(g['source_pts'],g['target_pts']) for g in picked])),flush=True)

if __name__=='__main__':main()
