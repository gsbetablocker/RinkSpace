"""Repair specifically witnessed S04 material/identity failures using source RGB."""
from pathlib import Path
import collections
import gzip
import itertools
import json
import sys
import time
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from review_points_s04 import OUT,ADS,read,write,rows,CurrentNativeGrid
from ice_points_s04 import source_domain,REC
from rinkspace.vision_board_reader import _unexcluded_pixels
from ad_points_s04 import configure


def emit(stream,record):stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')


def main():
    cv2.setNumThreads(2);grid=CurrentNativeGrid();r,t,m,o,s,ats,_=configure()
    ats+=read(ADS/'source_review/additional_source_atlases.json')['atlases'];doc=read(ADS/'source_review/source_identity_charts.json')
    refs={int(ai):r.reference(ats[int(ai)])for ai in doc['records']if doc['records'][ai]}
    oldinput=dict(np.load(OUT/'points_repaired/point_observations.npz'));active=set(oldinput['observation_ids'].tolist())
    issues={c['observation_id']:c for c in rows(OUT/'evidence/nonice_checks.jsonl.gz')if c['issues']}
    board=collections.defaultdict(list)
    for c in rows(OUT/'evidence/selected_board_records.jsonl.gz'):board[c['native_pts']].append(c)
    counts=collections.Counter();start=time.monotonic();nextrow=int(oldinput['source_rows'].max())+1
    with gzip.open(OUT/'evidence/nonice_repaired_records.jsonl.gz','wt',encoding='utf8')as dest,gzip.open(OUT/'evidence/nonice_repair_changes.jsonl.gz','wt',encoding='utf8')as ledger:
        for frame,(pt,iterator)in enumerate(itertools.groupby(rows(ADS/'evidence/selected_point_records.jsonl.gz'),key=lambda c:c['pts'])):
            source=[c for c in iterator if c['stable_id']in active];old={c['stable_id']:c for c in source}
            bad=[c for c in source if c['stable_id']in issues];affected={c['path_id']for c in bad}
            im,gray,mask=r.current(pt);bp=read(REC/f'dense_board_v1/qualified_frames/{pt}.json')
            graphics,border,_,_=source_domain(im,mask,bp);exclude=(mask>0)|graphics
            upper=np.asarray(bp['roles']['board.top']['trace_uv']);upper=upper[np.argsort(upper[:,0])]
            def visible(uv):
                x,y=uv
                return _unexcluded_pixels(exclude,np.asarray(uv)[None])[0] and np.interp(x,upper[:,0],upper[:,1])<=y<=np.interp(x,np.arange(1920),border)
            accepted=[dict(c)for c in source if c['stable_id']not in issues];fresh=[]
            if bad:
                variants=[]
                for ai,ref in refs.items():
                    chart=doc['records'][str(ai)].get(str(pt))
                    if chart is None:continue
                    for v in m.chart_paths(ref,chart,im,gray,mask):
                        if o.complete_named_source_reference(v,ref)or any(c['path_id'].split('::')[1]==v['path']and issues[c['stable_id']]['current_observability'].get('identity_reference_atlas')==ai for c in bad):variants.append(v)
                def record(v,qi,profile,uv,sid,decision):
                    nonlocal nextrow
                    c=dict(source_row=nextrow,stable_id=sid,pts=pt,uv=list(map(float,uv)),path_id=f'print_instance/Ad{ats[v["atlas_id"]]["ad"]}::'+v['path'],kind=1,station=.5,decision=decision,
                        identity=dict(atlas_id=v['atlas_id'],source_path_index=v['source_path_index'],source_contour_sha256=v['source_contour_sha256'],source_chart_pts=pt,reference_arc_fraction=float(v['s'][qi]),current_neighborhood_correlation=float(v['ncc'][qi]),unique_native_entity=True),
                        source_profile=dict(search_uv=np.asarray(profile['_query']).tolist(),normal=v['n'][qi].tolist(),offsets=np.asarray(profile['source_q10_q50_q90_normal_offsets_px']).tolist(),search_radius_px=profile['source_search_radius_px'],flags=profile['source_flags']),
                        source_BGR_sha256=r.VIS[pt]['source_BGR_sha256'],native_geometry_used_for_identity=False,human_acceptance=False)
                    nextrow+=1;return c
                for c in bad:
                    if c['kind']==0:continue
                    candidates=[]
                    for v in variants:
                        d,qi=v['tree'].query(c['uv']);qi=int(qi)
                        if v['valid'][qi]:candidates.append((float(d),v,qi))
                    candidates.sort(key=lambda v:v[0])
                    if not candidates:continue
                    d,v,qi=candidates[0];width=np.ptp(c['source_profile']['offsets']);limit=max(4.,width/2+2.)
                    distinct=[q for q in candidates[1:]if q[1]['path']!=v['path']]
                    if d>limit or(distinct and distinct[0][0]<d+2.):continue
                    p=m.read_profiles(im,r.VIS[pt],np.asarray([c['uv']]),v['n'][qi:qi+1])[0];p['_query']=c['uv']
                    off=np.asarray(p['source_q10_q50_q90_normal_offsets_px'])
                    if m.profile_reason(p)is not None or not(off[0]-1<=0<=off[2]+1)or not visible(c['uv']):continue
                    q=record(v,qi,p,c['uv'],c['stable_id'],'include_original_pixel');q['source_row']=c['source_row'];q['original_native_path_id']=c['path_id'];fresh.append(q);affected.add(q['path_id'])
                for v in variants:
                    pid=f'print_instance/Ad{ats[v["atlas_id"]]["ad"]}::'+v['path']
                    if pid not in affected:continue
                    near=np.array([c['uv']for c in accepted+fresh if c['path_id']==pid]).reshape(-1,2)
                    qi=[k for k in range(1,len(v['q'])-1,4)if v['valid'][k]and v['ncc'][k]>=.05 and(not len(near)or np.linalg.norm(near-v['q'][k],axis=1).min()>=3.)]
                    if not qi:continue
                    profiles=m.read_profiles(im,r.VIS[pt],v['q'][qi],v['n'][qi])
                    for k,p in zip(qi,profiles):
                        if m.profile_reason(p)is not None:continue
                        uv=np.asarray(p['actual_source_uv']);p['_query']=v['q'][k]
                        if not visible(uv):continue
                        sid=f's04_complete_ad_RGB/{pt}/{v["atlas_id"]}/{v["source_path_index"]}/{k}'
                        fresh.append(record(v,k,p,uv,sid,'new_actual_source_boundary'))
                if fresh:
                    fresh,_=o.frame_check(pt,fresh,doc,refs);accepted.extend(c for c in fresh if c['decision']!='not_admitted')
                by=collections.defaultdict(list)
                for c in accepted:by[c['path_id']].append(c)
                accepted=[]
                for pid,rr in by.items():
                    if pid not in affected:accepted+=rr;continue
                    keep,why=s.select_path(pt,rr,doc,ats)
                    accepted.extend(rr[i]for i in sorted(keep))
            current={c['stable_id']:c for c in accepted}
            for c in source:
                q=current.get(c['stable_id'])
                if q is None or q['path_id']!=c['path_id']:
                    emit(ledger,dict(family='advertisement',observation_id=c['stable_id'],pts=pt,original_UV=c['uv'],original_native_path_id=c['path_id'],
                        action='source_supported_identity_rebinding'if q else'replaced_by_same_frame_actual_RGB_or_redundant_support',
                        replacement_native_path_id=q['path_id']if q else None,source_issues=issues.get(c['stable_id'],{}).get('issues',[]),original_record_file=str(ADS/'evidence/selected_point_records.jsonl.gz'),human_acceptance=False))
            for c in accepted:emit(dest,dict(c,family='advertisement'));counts['advertisement']+=1;counts['new_ads']+=c['stable_id']not in old;counts['rebound_ads']+=c['stable_id']in old and c['path_id']!=old[c['stable_id']]['path_id']
            clear=[];badboard=[]
            for c in board[pt]:(badboard if 'board:'+c['source_row_id']in issues else clear).append(c)
            used={c['source_row_id']for c in clear};pool=[]
            for j,c in enumerate(bp['rows']):
                if not c.get('pixels_measured_in_this_frame')or not c.get('pixel_support',{}).get('current_pixel_profile_reread'):continue
                support=np.vstack([c['uv'],c['pixel_support'].get('position_interval_endpoints_uv',[c['uv']])])
                if not _unexcluded_pixels(exclude,support).all():continue
                pool.append((j,c))
            for c in badboard:
                options=[(np.linalg.norm(np.asarray(q['uv'])-c['uv']),j,q)for j,q in pool if q['source_row_id']not in used and q['segment_id']==c['segment_id']and all(np.linalg.norm(np.asarray(q['uv'])-v['uv'])>=2.5 for v in clear if v['role']==q['role'])]
                replacement=None
                if options:
                    _,j,q=min(options,key=lambda v:v[0]);replacement=dict(q,native_path_id=c['native_path_id'],visible_span_id=q['segment_id'],
                        source_BGR_sha256=r.VIS[pt]['source_BGR_sha256'],source_ref=dict(path=str(REC/f'dense_board_v1/qualified_frames/{pt}.json'),row_index=j))
                    used.add(q['source_row_id']);clear.append(replacement);counts['board_replacements']+=1
                emit(ledger,dict(family='board',observation_id='board:'+c['source_row_id'],pts=pt,original_UV=c['uv'],action='visible_same_frame_material_support_replacement'if replacement else'no_clear_material_support_in_same_source_span',replacement_source_id=replacement['source_row_id']if replacement else None,source_issues=issues['board:'+c['source_row_id']]['issues'],human_acceptance=False))
            for c in clear:emit(dest,dict(c,family='board'));counts['board']+=1
            if frame%100==0 or frame==792:print(json.dumps(dict(frames=frame+1,total=793,counts=dict(counts),seconds=time.monotonic()-start)),flush=True)
    grid.unchanged();write(OUT/'evidence/nonice_repair_summary.json',dict(counts=dict(counts),frames=793,native_authority=grid.receipt,original_source_packages_modified=False,human_acceptance=False))


if __name__=='__main__':main()
