"""Exact original low FIT candidates: native advertisement-band review sheets.

Legacy mode displays both-endpoint source region candidates. The explicit
original-LK inventory mode also displays missing/ambiguous region navigation
for actual material review; those measurements remain unknown. This producer
reads source arrays and source RGB; it cannot admit roles or load a model/solver.
"""
from pathlib import Path
import argparse, collections, hashlib, json, os, sys, time
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT),str(ROOT/'src')];sys.dont_write_bytecode=True
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import av,cv2,numpy as np
from audit_original_low_motion_far_board import ADMISSION,RETIREMENT,sha,read
from audit_original_low_motion_board_overlap import ORIGINAL_LK_MODE, original_candidate_records
cv2.setNumThreads(1)

def arr(path,keys):
    with np.load(path,allow_pickle=False)as z:return {k:z[k].copy()for k in keys}
def ref(path):return dict(path=str(path),sha256=sha(path))

def label_points(image, records, key, y0):
    view=image.copy();occupied=[]
    for ordinal,r in enumerate(records):
        x,y=np.rint(r[key]).astype(int);y-=y0
        cv2.circle(view,(int(x),int(y)),4,(0,255,0),1)
        offsets=((7,-8),(7,18),(-24,-8),(-24,18),(30,-8),(30,18),(-44,-8),(-44,18),(7,-28),(7,38))
        candidates=[(int(np.clip(x+dx,2,view.shape[1]-35)),int(np.clip(y+dy,20,view.shape[0]-4)))for dx,dy in offsets]
        tx,ty=min(candidates,key=lambda p:sum(abs(p[0]-px)<27 and abs(p[1]-py)<20 for px,py in occupied))
        occupied.append((tx,ty));cv2.line(view,(int(x),int(y)),(tx,ty-7),(0,190,0),1)
        cv2.putText(view,str(ordinal),(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(view,str(ordinal),(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,.6,(50,255,50),1,cv2.LINE_AA)
    return view

def pair_image(pair,records,frames,bounds):
    low=[];high=[]
    for pts,key in ((pair['source_pts'],'source_uv'),(pair['target_pts'],'target_uv')):
        uv=np.asarray([r[key]for r in records]);x=np.rint(uv[:,0]).astype(int)
        low.append(float(uv[:,1].min())-80)
        high.append(float(uv[:,1].max())+80)
        if pts in bounds:
            top,bottom=bounds[pts]
            assert np.all(top[x]>=0)
            low[-1]=min(low[-1],float(top[x].min())-65)
            high[-1]=max(high[-1],float(bottom[x].max())+65)
    y0=max(0,int(np.floor(min(low))));y1=min(1080,int(np.ceil(max(high))))
    a=label_points(frames[pair['source_pts']][y0:y1],records,'source_uv',y0)
    b=label_points(frames[pair['target_pts']][y0:y1],records,'target_uv',y0)
    legend_rows=(len(records)+15)//16;h=y1-y0
    canvas=np.full((40+h+legend_rows*26+8,3840,3),245,np.uint8)
    canvas[40:40+h,:1920]=a;canvas[40:40+h,1920:]=b
    title=f"rawpair {pair['original_raw_pair_index']} | source {pair['source_pts']} -> target {pair['target_pts']} | {len(records)} original points | material PENDING"
    cv2.putText(canvas,title,(15,27),cv2.FONT_HERSHEY_SIMPLEX,.8,(15,15,15),2,cv2.LINE_AA)
    for ordinal,r in enumerate(records):
        col=ordinal%16;row=ordinal//16
        text=f"{ordinal} = {r['current_original_prefix_bundle_index']}"
        cv2.putText(canvas,text,(15+col*240,40+h+22+row*26),cv2.FONT_HERSHEY_SIMPLEX,.65,(10,10,10),1,cv2.LINE_AA)
    return canvas,[0,y0,1920,y1-y0]


def review_original_lk_inventory(a):
    """Review unchanged original points, never whole-ROI material promotion."""
    candidate=read(a.candidates);retirement=read(RETIREMENT)
    if candidate.get('candidate_mode')!=ORIGINAL_LK_MODE:
        raise ValueError('Explicit original LK both-RGB inventory required')
    if candidate['source_video_sha256']!=retirement['source_video_sha256']:
        raise ValueError('Original candidate source video changed')
    for key in ('original_source_ledger','original_selected_rows','original_raw_motion','clock_alias_authority'):
        if sha(candidate[key]['path'])!=candidate[key]['sha256']:
            raise ValueError(f'Original source evidence changed: {key}')
    if Path(candidate['original_source_ledger']['path']).resolve()!=Path(ADMISSION).resolve():
        raise ValueError('Original source ledger is not the authenticated low ledger')
    original=arr(ADMISSION,('active_original_global_index','source_uv_a','source_uv_b','source_endpoint_index_a','source_endpoint_index_b','original_sigma_xy'))
    selected=arr(candidate['original_selected_rows']['path'],('original_global_row_indices','native_pair_pts'))
    raw=arr(candidate['original_raw_motion']['path'],('pair_pts','offsets','xy0','xy1'))
    az=arr(candidate['clock_alias_authority']['path'],('pts','exposure_representative_pts'))
    aliases=dict(zip(map(int,az['pts']),map(int,az['exposure_representative_pts'])))
    clock=arr(candidate['clock_state']['path'],('aux__block_names','aux__block_lengths','aux__canonical_pts'))
    lengths=clock['aux__block_lengths']//4;bi=list(clock['aux__block_names']).index('low');lo=int(lengths[:bi].sum())
    stamps=clock['aux__canonical_pts'][lo:lo+int(lengths[bi])]
    wanted_pairs=set(a.pair_id) if a.pair_id else None
    grouped=collections.defaultdict(list)
    for ordinal,r in enumerate(original_candidate_records(candidate)):
        pair=int(r['original_raw_pair_index'])
        if wanted_pairs is not None and pair not in wanted_pairs:continue
        index=int(r['current_original_prefix_bundle_index']);full=int(original['active_original_global_index'][index])
        row=int(r['original_selected_row']);rawi=int(r['original_raw_point_index'])
        assert full==r['original_full_low_bundle_row']==int(candidate['full_original_cut_offset'])+row
        assert int(selected['original_global_row_indices'][row])==rawi
        assert np.array_equal(original['source_uv_a'][full],r['source_uv'])
        assert np.array_equal(original['source_uv_b'][full],r['target_uv'])
        assert np.array_equal(original['original_sigma_xy'][full],r['source_sigma_px'])
        assert np.array_equal(raw['xy0'][rawi],r['source_uv']) and np.array_equal(raw['xy1'][rawi],r['target_uv'])
        assert np.array_equal(raw['pair_pts'][pair],r['source_native_pair_pts'])
        assert np.array_equal(selected['native_pair_pts'][pair],raw['pair_pts'][pair])
        assert int(raw['offsets'][pair])<=rawi<int(raw['offsets'][pair+1])
        assert rawi-int(raw['offsets'][pair])==int(r['original_raw_pair_local_index'])
        assert int(stamps[original['source_endpoint_index_a'][full]])==r['source_pts']==aliases[int(raw['pair_pts'][pair,0])]
        assert int(stamps[original['source_endpoint_index_b'][full]])==r['target_pts']==aliases[int(raw['pair_pts'][pair,1])]
        grouped[pair].append(dict(r,original_candidate_ordinal=ordinal))
    if not grouped or wanted_pairs is not None and set(grouped)!=wanted_pairs:
        raise ValueError('Requested original LK pair has no review inventory; do not fabricate pixels')
    pairs=[]
    for ordinal,pid in enumerate(sorted(grouped)):
        first=grouped[pid][0]
        pairs.append(dict(pair_review_ordinal=ordinal,original_raw_pair_index=pid,
            source_pts=first['source_pts'],target_pts=first['target_pts'],
            source_native_pair_pts=first['source_native_pair_pts'],original_point_count=len(grouped[pid]),
            sheet_index=ordinal//3,sheet_pair_row=ordinal%3))
    wanted={p[k] for p in pairs for k in ('source_pts','target_pts')}
    frames={};needed=collections.Counter(p[k]for p in pairs for k in ('source_pts','target_pts'))
    pages=[];point_map=[];decoded=[];views=[];page_pairs=[];pending=0;started=time.perf_counter()
    out=a.output/'actual_original_LK_both_RGB_material_review_receipt_v2.json'
    if out.exists():raise FileExistsError('Preserve the existing actual original RGB review receipt')
    with av.open(retirement['source_video'])as container:
        stream=container.streams.video[0];stream.thread_count=1
        assert str(stream.time_base)=='1/15360'
        container.seek(min(wanted),stream=stream,backward=True,any_frame=False)
        for frame in container.decode(stream):
            if frame.pts is not None and frame.pts>max(wanted):break
            if frame.pts not in wanted:continue
            pts=int(frame.pts);image=frame.to_ndarray(format='bgr24')
            one=dict(actual_pts=pts,decoded_BGR_sha256=hashlib.sha256(image.tobytes()).hexdigest())
            if a.save_lossless_raw:
                path=a.output/f'{pts}_RAW_NATIVE.png'
                if path.exists():
                    previous=cv2.imread(str(path),cv2.IMREAD_COLOR)
                    if previous is None or not np.array_equal(previous,image):raise ValueError('Preserve changed historical RGB')
                else:assert cv2.imwrite(str(path),image)
                one['source_raw_frame']=dict(ref(path),native_pts=pts,source_time_base='1/15360',
                    native_bgr24_sha256=one['decoded_BGR_sha256'],native_size_wh=[1920,1080],exact_original_decode=True)
            decoded.append(one);frames[pts]=image
            while pending<len(pairs) and all(pairs[pending][k]in frames for k in ('source_pts','target_pts')):
                pair=pairs[pending];rr=grouped[pair['original_raw_pair_index']]
                view,crop=pair_image(pair,rr,frames,{})
                pair['native_source_crop_xywh']=crop;pair['native_target_crop_xywh']=crop
                views.append(view);page_pairs.append(pair)
                for ordinal,r in enumerate(rr):
                    point_map.append(dict(r,pair_review_ordinal=pending,pair_point_ordinal=ordinal,
                        sheet_index=pair['sheet_index'],sheet_pair_row=pair['sheet_pair_row'],
                        material_identity_review='unknown; both actual RGB endpoints pending',
                        automatically_material_admitted=False))
                for key in ('source_pts','target_pts'):
                    p=pair[key];needed[p]-=1
                    if needed[p]==0:frames.pop(p,None)
                pending+=1
                if len(views)==3 or pending==len(pairs):
                    name=a.output/f'original_LK_both_RGB_material_pending_sheet_{pair["sheet_index"]:03d}.png'
                    panel=np.vstack(views)
                    if name.exists():raise FileExistsError('Preserve an existing original RGB sheet')
                    assert cv2.imwrite(str(name),panel)
                    pages.append(dict(ref(name),sheet_index=pair['sheet_index'],native_rendered_hw=list(panel.shape[:2]),
                        original_raw_pair_indices=[p['original_raw_pair_index']for p in page_pairs],
                        point_count=sum(p['original_point_count']for p in page_pairs)))
                    views=[];page_pairs=[]
    assert pending==len(pairs) and {r['actual_pts']for r in decoded}==wanted
    report=dict(schema='rinkspace.original-LK-FIT-both-native-RGB-material-review-receipt.v2',
        status='actual_RGB_material_review_pending_not_admitted',source_candidates=ref(a.candidates),
        original_admission=ref(ADMISSION),source_clock_state=candidate['clock_state'],
        source_video=retirement['source_video'],source_video_sha256_from_existing_verified_ledger=retirement['source_video_sha256'],
        time_base='1/15360',shot_index=candidate['shot_index'],exact_decoded_source_RGB_receipts=decoded,
        actual_source_exposures=len(decoded),actual_source_pairs=len(pairs),selected_original_points=len(point_map),
        requested_original_pair_ids=sorted(wanted_pairs) if wanted_pairs else 'all original LK candidate pairs',
        sheets=pages,all_pairs=pairs,point_to_pair_sheet=point_map,
        original_UV_prefix_raw_index_sigma_native_actual_PTS_bitexact=True,
        existing_source_region_reason_bits_preserved=True,
        both_endpoint_original_RGB_material_certificate_required=True,
        selection='Existing original LK measurements; no second detector, model, cost or fitted plane selection',
        source_rows_UV_sigma_PTS_weights_runtime_target_modified=False,automatically_material_admitted_points=0,
        full_native_width_preserved=True,original_native_RGB_crop_pixels_not_resized_or_warped=True,
        no_solver_runtime_geometry_camera_residual_J_or_fullgraph_loaded=True,
        script=ref(Path(__file__).resolve()),wall_seconds=time.perf_counter()-started)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(dict(path=str(out),sha256=sha(out),sheets=len(pages),points=len(point_map),seconds=report['wall_seconds'])),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidates',type=Path,required=True)
    ap.add_argument('--regions',type=Path);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--original-LK-inventory',action='store_true',help='Review original LK without requiring a second corner/colour-strip navigation')
    ap.add_argument('--pair-id',type=int,action='append',help='Original raw pair identity to inspect; omitted means whole cut')
    ap.add_argument('--save-lossless-raw',action='store_true',help='Save each requested exact original native PNG once')
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.original_LK_inventory:
        review_original_lk_inventory(a)
        return
    if not a.regions or a.pair_id or a.save_lossless_raw:
        raise ValueError('Legacy mode requires regions and preserves its frozen full-region review format')
    candidate=read(a.candidates);region=read(a.regions);retirement=read(RETIREMENT)
    shot=int(candidate.get('shot_index',5))
    if shot not in (5,8):raise ValueError('Only authorized low cut5/cut8 source candidates are supported')
    assert region['original_candidates']==ref(a.candidates)
    assert sha(region['mask_archive']['path'])==region['mask_archive']['sha256']
    assert sha(ADMISSION)==retirement['original_foreground_admission_npz']['sha256']
    masks=arr(region['mask_archive']['path'],('actual_pts','directly_observed_interior_top','directly_observed_interior_bottom',
        'point_endpoint_reason_bits','original_source_index','point_both_endpoint_region_candidate'))
    assert np.array_equal(masks['point_both_endpoint_region_candidate'],np.all(masks['point_endpoint_reason_bits']==0,axis=1))
    original=arr(ADMISSION,('active_original_global_index','source_uv_a','source_uv_b','source_endpoint_index_a','source_endpoint_index_b','original_sigma_xy'))
    if shot==8:
        for key in ('original_source_ledger','original_selected_rows','original_raw_motion','clock_alias_authority'):
            assert sha(candidate[key]['path'])==candidate[key]['sha256']
        assert Path(candidate['original_source_ledger']['path']).resolve()==Path(ADMISSION).resolve()
        selected=arr(candidate['original_selected_rows']['path'],('original_global_row_indices','native_pair_pts'))
        raw=arr(candidate['original_raw_motion']['path'],('pair_pts','offsets','xy0','xy1'))
        aliases_raw=arr(candidate['clock_alias_authority']['path'],('pts','exposure_representative_pts'))
        aliases=dict(zip(map(int,aliases_raw['pts']),map(int,aliases_raw['exposure_representative_pts'])))
    clock=arr(candidate['clock_state']['path'],('aux__block_names','aux__block_lengths','aux__canonical_pts'))
    lengths=clock['aux__block_lengths']//4;bi=list(clock['aux__block_names']).index('low');lo=int(lengths[:bi].sum())
    stamps=clock['aux__canonical_pts'][lo:lo+int(lengths[bi])];assert len(stamps)==1851
    records=[]
    for i,r in enumerate(candidate['records']):
        index=int(r['current_original_prefix_bundle_index']);full=int(original['active_original_global_index'][index])
        assert index==int(masks['original_source_index'][i]) and full==r['original_full_low_bundle_row']
        assert np.array_equal(original['source_uv_a'][full],r['source_uv'])
        assert np.array_equal(original['source_uv_b'][full],r['target_uv'])
        assert np.array_equal(original['original_sigma_xy'][full],r['source_sigma_px'])
        assert int(stamps[original['source_endpoint_index_a'][full]])==r['source_pts']
        assert int(stamps[original['source_endpoint_index_b'][full]])==r['target_pts']
        if shot==8:
            row=int(r['original_selected_row']);rawi=int(r['original_raw_point_index']);pair=int(r['original_raw_pair_index'])
            assert int(selected['original_global_row_indices'][row])==rawi
            assert full==int(candidate['full_original_cut_offset'])+row
            assert np.array_equal(raw['xy0'][rawi],r['source_uv']) and np.array_equal(raw['xy1'][rawi],r['target_uv'])
            assert np.array_equal(raw['pair_pts'][pair],r['source_native_pair_pts'])
            assert np.array_equal(selected['native_pair_pts'][pair],raw['pair_pts'][pair])
            assert int(raw['offsets'][pair])<=rawi<int(raw['offsets'][pair+1])
            assert rawi-int(raw['offsets'][pair])==int(r['original_raw_pair_local_index'])
            assert aliases[int(raw['pair_pts'][pair,0])]==r['source_pts']
            assert aliases[int(raw['pair_pts'][pair,1])]==r['target_pts']
        assert bool(region['point_masks'][i]['directly_observed_region_both_endpoints'])==bool(masks['point_both_endpoint_region_candidate'][i])
        if masks['point_both_endpoint_region_candidate'][i]:records.append(dict(r,original_candidate_ordinal=i))
    point_count=len(records)
    assert point_count>0
    if shot==5:assert point_count==9288
    out=a.output/f'actual_original_cut{shot}_{point_count}_region_candidates_RGB_receipt_v1.json'
    if out.exists():raise FileExistsError('Review receipt already exists; preserve historical evidence')
    grouped=collections.defaultdict(list)
    for r in records:grouped[int(r['original_raw_pair_index'])].append(r)
    pair_ids=sorted(grouped)
    if shot==5:assert len(pair_ids)==780
    exposure_refs={int(r['actual_pts']):r for r in region['source_RGB_receipts']}
    if shot==5:assert len(exposure_refs)==845
    sheet_count=(len(pair_ids)+2)//3
    region_bounds={int(p):(t,b)for p,t,b in zip(masks['actual_pts'],masks['directly_observed_interior_top'],masks['directly_observed_interior_bottom'])}
    frames={};needed=collections.Counter(r[k]for r in records for k in ('source_pts','target_pts'))
    pairs=[]
    for j,pid in enumerate(pair_ids):
        rr=grouped[pid];one=rr[0]
        pairs.append(dict(pair_review_ordinal=j,original_raw_pair_index=pid,source_pts=one['source_pts'],target_pts=one['target_pts'],
            source_native_pair_pts=one['source_native_pair_pts'],original_point_count=len(rr),sheet_index=j//3,sheet_pair_row=j%3))
    pending=0;pages=[];point_map=[];decoded=[];views=[];page_pairs=[];started=time.perf_counter()
    with av.open(retirement['source_video'])as container:
        stream=container.streams.video[0];stream.thread_count=1;assert str(stream.time_base)=='1/15360'
        container.seek(min(exposure_refs),stream=stream,backward=True,any_frame=False)
        for frame in container.decode(stream):
            if frame.pts is not None and frame.pts>max(exposure_refs):break
            if frame.pts not in exposure_refs:continue
            pts=int(frame.pts);image=frame.to_ndarray(format='bgr24');digest=hashlib.sha256(image.tobytes()).hexdigest()
            assert digest==exposure_refs[pts]['decoded_BGR_sha256'];decoded.append(dict(actual_pts=pts,decoded_BGR_sha256=digest))
            if pts in needed:frames[pts]=image
            while pending<len(pairs) and all(pairs[pending][k] in frames for k in ('source_pts','target_pts')):
                pair=pairs[pending];rr=grouped[pair['original_raw_pair_index']]
                view,crop=pair_image(pair,rr,frames,region_bounds);views.append(view)
                pair['native_source_crop_xywh']=crop;pair['native_target_crop_xywh']=crop;page_pairs.append(pair)
                for ord_,r in enumerate(rr):point_map.append(dict(r,pair_review_ordinal=pending,pair_point_ordinal=ord_,
                    sheet_index=pair['sheet_index'],sheet_pair_row=pair['sheet_pair_row'],material_identity_review='pending'))
                for r in rr:
                    for k in ('source_pts','target_pts'):
                        p=r[k];needed[p]-=1
                        if needed[p]==0:frames.pop(p,None)
                pending+=1
                if len(views)==3 or pending==len(pairs):
                    pi=pair['sheet_index'];name=a.output/f'original_cut{shot}_board_region_sheet_{pi:03d}.png'
                    panel=np.vstack(views);assert cv2.imwrite(str(name),panel)
                    pages.append(dict(sheet_index=pi,path=str(name),sha256=sha(name),native_rendered_hw=list(panel.shape[:2]),
                        original_pair_review_ordinals=[p['pair_review_ordinal']for p in page_pairs],
                        original_raw_pair_indices=[p['original_raw_pair_index']for p in page_pairs],point_count=sum(p['original_point_count']for p in page_pairs)))
                    views=[];page_pairs=[]
            if len(decoded)%100==0:print(json.dumps(dict(exposures=len(decoded),total=len(exposure_refs),sheets=len(pages),seconds=time.perf_counter()-started)),flush=True)
    assert set(r['actual_pts']for r in decoded)==set(exposure_refs) and pending==len(pair_ids) and len(pages)==sheet_count and len(point_map)==point_count
    one_third=(sheet_count+2)//3;two_thirds=min(2*one_third,sheet_count)
    assignment=dict(ad_evidence=[0,one_third-1],source_review=[one_third,two_thirds-1],root=[two_thirds,sheet_count-1])
    if shot==5:assert assignment==dict(ad_evidence=[0,86],source_review=[87,173],root=[174,259])
    report=dict(schema=f'rinkspace.original-low-cut{shot}-region-candidate-actual-RGB-review-sheets.v1',status='source_material_review_pending',
        source_candidates=ref(a.candidates),source_region_masks=ref(a.regions),source_mask_archive=region['mask_archive'],
        original_admission=ref(ADMISSION),source_clock_state= candidate['clock_state'],source_video=retirement['source_video'],
        source_video_sha256_from_existing_verified_ledger=retirement['source_video_sha256'],time_base='1/15360',
        actual_source_exposures=len(exposure_refs),exact_decoded_source_RGB_receipts=decoded,
        original_UV_prefix_index_sigma_PTS_bitexact=True,selected_original_points=point_count,actual_source_pairs=len(pair_ids),sheets=pages,
        all_pairs=pairs,point_to_pair_sheet=point_map,
        review_assignment=assignment,
        original_native_RGB_crop_pixels_not_resized_or_warped=True,minimum_context_native_px=65,
        full_native_width_preserved=True,one_source_and_target_band_per_pair_not_per_point_PNG_copies=True,
        selection='Every original candidate with both endpoint source reason bits zero; no model/residual/guard sorting',
        no_solver_runtime_geometry_camera_residual_J_or_fullgraph_loaded=True,
        source_rows_UV_sigma_PTS_weights_and_runtime_target_modified=False,automatically_material_admitted_points=0,
        script=ref(Path(__file__).resolve()),wall_seconds=time.perf_counter()-started)
    if shot!=5:
        report.update(shot_index=shot,original_raw_selected_prefix_UV_sigma_native_actual_PTS_bitexact=True,
            original_selected_rows=candidate['original_selected_rows'],original_raw_motion=candidate['original_raw_motion'],
            original_actual_exposure_alias_authority=candidate['clock_alias_authority'])
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(dict(path=str(out),sha256=sha(out),sheets=len(pages),points=len(point_map),seconds=report['wall_seconds'])),flush=True)

if __name__=='__main__':main()
