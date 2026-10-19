"""Decode twelve exact native RGBs for pointwise source-role review only."""
from pathlib import Path
import argparse,hashlib,json,os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import av,cv2,numpy as np
cv2.setNumThreads(1)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def inspect_all_candidates(a,candidate):
    """Stream exact source pairs once; retain crops and decoded-byte receipts.

    Selection depends only on original source candidates and shot/time. No
    camera, residual or world model participates in this material inspection.
    """
    selected=[r for r in candidate['records'] if (not a.shot or r['shot'] in a.shot)
        and (not a.context_for_index or int(r['original_global_bundle_index']) in a.context_for_index)]
    if not selected:raise ValueError('No source candidates for declared shots')
    graph=candidate['source_graph'];assert sha(graph['path'])==graph['sha256']
    with np.load(graph['path'],allow_pickle=False)as z:
        for row in selected:
            index=int(row['original_global_bundle_index'])
            assert np.array_equal(z['source_uv'][index],row['original_source_uv'])
            assert np.array_equal(z['target_uv'][index],row['original_target_uv'])
    bypair={}
    for r in selected:bypair.setdefault(int(r['original_pair_index']),[]).append(r)
    ends={};uses={}
    for j,rows in bypair.items():
        pair=rows[0];pa,pb=int(pair['source_pts']),int(pair['target_pts'])
        if pa==pb:raise ValueError('Repeated exposure cannot qualify a motion pair')
        ends.setdefault(max(pa,pb),[]).append(j)
        for stamp in (pa,pb):uses[stamp]=uses.get(stamp,0)+1
    tiles=[];receipts=[];decoded=[];images={};sheets=[];pending={}
    def flush_sheet():
        if not tiles:return
        path=a.output/f'pointwise_source_sheet_{len(sheets):03d}.png'
        assert cv2.imwrite(str(path),np.vstack([t[1]for t in tiles]))
        sheets.append(dict(path=str(path),sha256=sha(path),original_global_bundle_indices=[t[0]for t in tiles]))
        tiles.clear()
    with av.open(str(a.source_video))as container:
        stream=container.streams.video[0];stream.thread_count=1
        assert str(stream.time_base)=='1/15360'
        container.seek(min(uses)-15360,stream=stream,backward=True,any_frame=False)
        for frame in container.decode(stream):
            stamp=frame.pts
            if stamp is None:continue
            if stamp>max(uses):break
            if stamp not in uses:continue
            raw=frame.to_ndarray(format='bgr24');assert raw.shape==(1080,1920,3)
            if a.context_for_index:
                view=raw.copy()
                for row in selected:
                    key='original_source_uv'if row['source_pts']==stamp else 'original_target_uv'if row['target_pts']==stamp else None
                    if key is None:continue
                    at=tuple(np.rint(row[key]).astype(int));cv2.circle(view,at,7,(40,255,40),2)
                    cv2.putText(view,str(row['original_global_bundle_index']),(at[0]+8,at[1]-8),cv2.FONT_HERSHEY_SIMPLEX,.55,(40,255,40),1)
                assert cv2.imwrite(str(a.output/f'source_context_{stamp}.png'),view)
            images[stamp]=raw
            decoded.append(dict(actual_pts=int(stamp),decoded_BGR_sha256=hashlib.sha256(raw.tobytes()).hexdigest()))
            for j in ends.get(stamp,[]):
                rows=bypair[j];pa,pb=int(rows[0]['source_pts']),int(rows[0]['target_pts'])
                if pa not in images or pb not in images:raise ValueError('Source pair endpoint unavailable')
                for row in rows:
                    panels=[];windows=[]
                    for native,key in ((pa,'original_source_uv'),(pb,'original_target_uv')):
                        xy=np.rint(row[key]).astype(int)
                        x0=int(np.clip(xy[0]-100,0,1720));y0=int(np.clip(xy[1]-75,0,930))
                        panel=images[native][y0:y0+150,x0:x0+200].copy()
                        cv2.circle(panel,(int(xy[0])-x0,int(xy[1])-y0),4,(40,255,40),1)
                        panels.append(panel);windows.append(dict(actual_pts=native,source_crop_xywh=[x0,y0,200,150]))
                    number=int(row['original_global_bundle_index'])
                    path=a.output/f'original_{number}_both_native_RGB.png'
                    point=np.hstack(panels);assert cv2.imwrite(str(path),point)
                    label=np.zeros((28,400,3),np.uint8)
                    cv2.putText(label,f'original FIT {number} cut{row["shot"]} {pa}->{pb}',(4,18),cv2.FONT_HERSHEY_SIMPLEX,.39,(255,255,255),1)
                    tiles.append((number,np.vstack((label,point))))
                    if len(tiles)==8:flush_sheet()
                    receipts.append(dict(original_global_bundle_index=number,original_pair_index=j,shot=row['shot'],
                        source_pts=pa,target_pts=pb,source_uv=row['original_source_uv'],target_uv=row['original_target_uv'],
                        path=str(path),sha256=sha(path),windows=windows,material_review_status='pending'))
                for native in (pa,pb):
                    uses[native]-=1
                    if uses[native]==0:images.pop(native)
    flush_sheet()
    if len(receipts)!=len(selected):raise ValueError('Not all original source queries decoded')
    report=dict(schema='rinkspace.actual-original-secondary-board-pointwise-all-source-RGB.v1',
        source_video_path=str(a.source_video),source_video_verified_ledger_sha256='f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891',
        candidate_packet=dict(path=str(a.candidates),sha256=sha(a.candidates)),time_base='1/15360',
        original_numeric_source_graph=graph,original_query_UV_bitexact_checked=True,
        selected_shots=a.shot or [2,6],selection='all original source-only candidates in declared shots; no model error selection',
        original_UV_PTS_sigma_solver_roles_changed=False,pointwise_crops=receipts,sheets=sheets,
        exact_source_decoded_frame_receipts=decoded,whole_original_motion_reclassified=False,
        source_only_material_review_pending=True,solver_or_geometry_not_imported=True)
    target=a.output/'all_candidate_actual_RGB_receipt_v1.json';target.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(path=str(target),pairs=len(bypair),original_queries=len(receipts),sheets=len(sheets))),flush=True)

def main():
    parser=argparse.ArgumentParser()
    for name in ('candidates','source-video','output'):parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--all-candidate-pairs',action='store_true')
    parser.add_argument('--shot',type=int,action='append',choices=(2,6))
    parser.add_argument('--context-for-index',type=int,action='append')
    a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    candidate=json.loads(a.candidates.read_text(encoding='utf8'))
    if a.all_candidate_pairs:
        inspect_all_candidates(a,candidate);return
    selected=[p for s in candidate['per_shot']for p in s['raw_RGB_representative_review_pairs']]
    graph=candidate['source_graph'];assert sha(graph['path'])==graph['sha256']
    with np.load(graph['path'],allow_pickle=False)as z:uv0=z['source_uv'].copy();uv1=z['target_uv'].copy();offset=z['offset'].copy()
    receipts=[]
    with av.open(str(a.source_video))as container:
        stream=container.streams.video[0];stream.thread_count=1;assert str(stream.time_base)=='1/15360'
        for pair in selected:
            j=pair['original_pair_index'];pa,pb=pair['source_pts'],pair['target_pts'];lo,hi=map(int,offset[j:j+2])
            wanted={pa,pb};frames={};container.seek(min(wanted),stream=stream,backward=True,any_frame=False)
            for frame in container.decode(stream):
                if frame.pts is not None and frame.pts>max(wanted):break
                if frame.pts in wanted:frames[frame.pts]=frame.to_ndarray(format='bgr24')
                if set(frames)==wanted:break
            if set(frames)!=wanted:raise ValueError(('Exact source PTS missing',wanted-set(frames)))
            assert all(im.shape==(1080,1920,3)for im in frames.values())
            candidates=[r for r in candidate['records']if r['original_pair_index']==j]
            images=[];source_files=[];crops=[]
            for stamp,uv in ((pa,uv0[lo:hi]),(pb,uv1[lo:hi])):
                raw=frames[stamp];path=a.output/f'raw_native_{stamp}.png'
                if not path.exists():assert cv2.imwrite(str(path),raw)
                source_files.append(dict(actual_pts=stamp,source_time_seconds=stamp/15360.,path=str(path),
                    png_sha256=sha(path),original_decoded_BGR_sha256=hashlib.sha256(raw.tobytes()).hexdigest()))
                display=raw.copy()
                for local,p in enumerate(uv):cv2.circle(display,tuple(np.rint(p).astype(int)),3,(0,100,240),1)
                for row in candidates:
                    point=np.asarray(row['original_source_uv']if stamp==pa else row['original_target_uv'])
                    at=tuple(np.rint(point).astype(int));cv2.circle(display,at,7,(40,255,40),2)
                    cv2.putText(display,str(row['original_global_bundle_index']),(at[0]+8,at[1]-8),cv2.FONT_HERSHEY_SIMPLEX,.42,(40,255,40),1)
                small=cv2.resize(display,(960,540),interpolation=cv2.INTER_AREA)
                cv2.putText(small,f'cut{pair["shot"]} native {stamp}',(10,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),1)
                images.append(small)
            overview=a.output/f'pair{j}_original_FIT_source_points.png';assert cv2.imwrite(str(overview),np.hstack(images))
            for row in candidates:
                panels=[]
                for stamp,key in ((pa,'original_source_uv'),(pb,'original_target_uv')):
                    p=np.rint(row[key]).astype(int);x0=int(np.clip(p[0]-100,0,1720));y0=int(np.clip(p[1]-75,0,930))
                    panel=frames[stamp][y0:y0+150,x0:x0+200].copy()
                    cv2.circle(panel,(int(p[0])-x0,int(p[1])-y0),4,(40,255,40),1)
                    panels.append(panel)
                path=a.output/f'original_{row["original_global_bundle_index"]}_both_native_RGB.png'
                assert cv2.imwrite(str(path),np.hstack(panels));crops.append(dict(global_index=row['original_global_bundle_index'],path=str(path),sha256=sha(path)))
            receipts.append(dict(**pair,original_native_source_files=source_files,source_only_FIT_queries_overview=str(overview),
                pointwise_native_RGB_crops=crops,visual_inspection_status='pending',solver_or_geometry_not_imported=True))
    report=dict(schema='rinkspace.actual-original-secondary-board-source-overlap-representative-native-RGB.v1',
        source_video_path=str(a.source_video),source_video_verified_ledger_sha256='f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891',
        candidate_packet=dict(path=str(a.candidates),sha256=sha(a.candidates)),time_base='1/15360',
        original_UV_or_solver_roles_not_modified=True,only_six_representative_pairs_extracted=True,records=receipts)
    target=a.output/'representative_actual_RGB_receipt_v1.json';target.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(path=str(target),pairs=len(receipts),pointwise_crops=sum(len(r['pointwise_native_RGB_crops'])for r in receipts))),flush=True)
if __name__=='__main__':main()
