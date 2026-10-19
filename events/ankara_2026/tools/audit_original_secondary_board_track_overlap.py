"""Whole-source original FIT queries vs independent deep-LK board evidence.

It proposes no solver role and copies no UV. Coincidence at both real native
endpoints is a source-only review candidate, requiring raw RGB inspection.
"""
from pathlib import Path
import argparse,hashlib,json,os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import numpy as np
SOURCE_GRAPH_SHA='71af8c55d5598ba51a59e5f8ac8a6d0f86ed893153a18593d109a3e033f89ae4'
TRACKS_SHA='7193fd7152cd3d08625c34c7e61c77d0bea2866b68bf98a60ed4d65cdb72a6e9'
PEOPLE_SHA='d621c67b219458adef692731a4ab68095829b97b349a08ef3b6abc3c81c98c63'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
    parser=argparse.ArgumentParser()
    for name in ('source-graph','tracks','people','inventory','output'):parser.add_argument('--'+name,required=True,type=Path)
    a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if sha(a.source_graph)!=SOURCE_GRAPH_SHA or sha(a.tracks)!=TRACKS_SHA or sha(a.people)!=PEOPLE_SHA:
        raise ValueError('Exact original graph, preserved v2 source tracks and original people evidence required')
    with np.load(a.source_graph,allow_pickle=False)as z:source={k:z[k].copy()for k in z.files}
    with np.load(a.tracks,allow_pickle=False)as z:tracks={k:z[k].copy()for k in z.files}
    people={int(r['pts']):r['boxes']for r in read(a.people)['frames']}
    lookup={(int(pa),int(pb)):i for i,(pa,pb)in enumerate(zip(tracks['source_pts'],tracks['target_pts']))}
    inventory=read(a.inventory);cut_by_pair={r['original_pair_index']:r['shot']for r in inventory['pairs']}
    records=[];pairs=[];sample_by_cut={}
    for j,(pa,pb)in enumerate(source['pair_pts']):
        pa,pb=int(pa),int(pb);lo,hi=map(int,source['offset'][j:j+2]);p,q=source['source_uv'][lo:hi],source['target_uv'][lo:hi]
        same=(pa,pb)in lookup;reverse=(pb,pa)in lookup
        row=dict(original_pair_index=j,shot=cut_by_pair[j],source_pts=pa,target_pts=pb,original_point_count=hi-lo,
            independent_adjacent_pair_present=same or reverse,source_only_candidates=0)
        if not same and not reverse:pairs.append(row);continue
        k=lookup[(pa,pb)if same else(pb,pa)];t0,t1=map(int,tracks['offsets'][k:k+2])
        pp,qq=(tracks['uv0'][t0:t1],tracks['uv1'][t0:t1])if same else(tracks['uv1'][t0:t1],tracks['uv0'][t0:t1])
        row['independent_v2_pair_index']=k;row['independent_board_points']=len(pp)
        if not len(pp):pairs.append(row);continue
        e0=np.linalg.norm(p[:,None,:]-pp[None,:,:],axis=2);e1=np.linalg.norm(q[:,None,:]-qq[None,:,:],axis=2)
        idx=np.argmin(e0*e0+e1*e1,axis=1);ii=np.arange(len(p))
        aerr=e0[ii,idx];berr=e1[ii,idx];kept=(aerr<=2.)&(berr<=2.)
        for stamp,uv in ((pa,p),(pb,q)):
            if stamp not in people:kept[:]=False;row['missing_actual_people_boxes']=True;continue
            for box in people[stamp]:
                x0,y0,x1,y1=np.asarray(box[:4],float)*2.;pad=max(48.,.4*(x1-x0))
                kept&=~((uv[:,0]>=x0-pad)&(uv[:,0]<=x1+pad)&(uv[:,1]>=y0-48.)&(uv[:,1]<=y1+48.))
        for local in np.flatnonzero(kept):
            ti=int(idx[local]);records.append(dict(original_global_bundle_index=lo+int(local),original_pair_index=j,
                original_local_index=int(local),shot=cut_by_pair[j],source_pts=pa,target_pts=pb,
                original_source_uv=p[local].tolist(),original_target_uv=q[local].tolist(),
                independent_v2_pair_index=k,independent_v2_global_point_index=t0+ti,
                source_endpoint_coincidence_native_px=float(aerr[local]),target_endpoint_coincidence_native_px=float(berr[local]),
                independent_deep_LK_FB_native_px=float(tracks['forward_backward_error_px'][t0+ti]),
                source_only_people_clear=True,source_role_status='pending_raw_RGB_pointwise_material_inspection',
                original_UV_PTS_sigma_profile_or_solver_not_modified=True))
        row['source_only_candidates']=int(kept.sum());pairs.append(row)
        if kept.any():sample_by_cut.setdefault(cut_by_pair[j],[]).append(row)
    summaries=[]
    for cut in sorted(cut_by_pair.values()):
        if any(r['shot']==cut for r in summaries):continue
        rr=[r for r in pairs if r['shot']==cut];cc=[r for r in records if r['shot']==cut]
        qualified=[r for r in rr if r['source_only_candidates']]
        ordered=sorted(qualified,key=lambda r:min(r['source_pts'],r['target_pts']))
        sample=[]
        if ordered:
            for target in np.linspace(0,len(ordered)-1,min(3,len(ordered))).astype(int):sample.append(ordered[int(target)])
        summaries.append(dict(shot=cut,actual_original_pairs=len(rr),actual_original_points=sum(r['original_point_count']for r in rr),
            exact_independent_adjacent_pairs=sum(r['independent_adjacent_pair_present']for r in rr),
            pairs_with_source_point_coincidence=len(qualified),source_only_candidate_original_points=len(cc),
            raw_RGB_representative_review_pairs=sample))
    report=dict(schema='rinkspace.original-secondary-actual-point-independent-board-source-overlap-candidates.v1',
        source_graph=dict(path=str(a.source_graph),sha256=SOURCE_GRAPH_SHA),independent_board_tracks=dict(path=str(a.tracks),sha256=TRACKS_SHA),
        people=dict(path=str(a.people),sha256=PEOPLE_SHA),original_pair_count=len(pairs),original_point_count=len(source['source_uv']),
        native_endpoint_coincidence_gate_px=2.,original_source_UV_used_without_nearest_replacement=True,
        source_only_candidate_is_not_material_admission=True,source_role_operator_or_weights_not_modified=True,
        model_or_solver_not_loaded=True,records=records,pairs=pairs,per_shot=summaries)
    target=a.output/'original_secondary_independent_board_overlap_candidates_v1.json'
    target.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(path=str(target),sha256=sha(target),candidate_original_points=len(records),per_shot=summaries)),flush=True)
if __name__=='__main__':main()
