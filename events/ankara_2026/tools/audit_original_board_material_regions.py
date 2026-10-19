"""Source-only, directly observed per-column board-region navigation.

This is a review mask, never a solver material-role admission. A colour strip
does not prove that an interior pixel is on the board rather than an occluder.
No source UV, source role, camera, world, residual or solver is modified/read.
"""
from pathlib import Path
import argparse, collections, hashlib, json, os, sys, time
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
sys.dont_write_bytecode = True
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import av, cv2, numpy as np
from audit_original_low_motion_board_overlap import original_candidate_records, ORIGINAL_LK_MODE
cv2.setNumThreads(1)

DATA = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026')
RETIREMENT = DATA/'calibration/runs/spin_source_long_baseline_20261005/actual50_jitter_source_diagnostic_20261005/spin_micro_broadcast_overlay_original_motion_role_retirement_SOURCE_ONLY_v1.json'
PEOPLE = Path('D:/CodexData/RinkSpace_program_tracking_tmp/people/boxes.json')
REASONS = {1:'cap_missing_or_not_a_long_narrow_colour_component',
           2:'foot_missing_or_not_a_long_narrow_colour_component',
           4:'multiple_direct_cap_colour_runs', 8:'multiple_direct_foot_colour_runs',
           16:'cap_foot_order_or_gap_invalid',
           32:'observed_column_neighbourhood_not_continuous',
           64:'point_in_conservative_people_or_fixed_UI_mask',
           128:'point_outside_directly_observed_interior',
           256:'people_frame_missing', 512:'point_outside_native_RGB'}
CONFIG = dict(native_hw=[1080,1920], source_y_search_range=[255,935],
    cap_HSV_hue=[90,135], cap_HSV_saturation_min=60,
    foot_HSV_hue=[8,45], foot_HSV_saturation_min=30,
    foot_BGR_blue_plus15_below_min_green_red=True,
    component_min_native_width=160, component_min_area_pixels=300,
    component_max_height_over_width=.15, column_run_max_height_native_px=40,
    directly_observed_continuity_radius_native_px=8,
    continuity_window_is_17_adjacent_links_requiring_18_observed_columns=True,
    actual_observed_column_offsets_native_px=[-9,8],
    adjacent_column_max_edge_delta_native_px=3,
    interior_edge_margin_native_px=6, allowed_cap_foot_gap_native_px=[30,600],
    missing_columns_not_interpolated=True, missing_columns_not_extrapolated=True,
    morphological_closing_or_gap_filling=False,
    conservative_people_padding_native_px=dict(horizontal='max(48, .4*box width)',vertical=48),
    fixed_UI_rectangles_native_xyxy=[[0,0,585,265],[1660,930,1895,1020]],
    numerical_settings_are_navigation_assumptions_not_measured_world_geometry=True)

def read(path): return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ref(path): return dict(path=str(path),sha256=sha(path))

def excluded_pixels(boxes, shape):
    mask = np.zeros(shape, bool)
    for box in boxes:
        x0,y0,x1,y1 = np.asarray(box[:4],float)*2
        dx=max(48.,.4*(x1-x0));dy=48.
        xa=max(0,int(np.floor(x0-dx)));xb=min(shape[1],int(np.ceil(x1+dx))+1)
        ya=max(0,int(np.floor(y0-dy)));yb=min(shape[0],int(np.ceil(y1+dy))+1)
        mask[ya:yb,xa:xb] = True
    mask[:265,:585] = True
    mask[930:1020,1660:1895] = True
    return mask

def _long_narrow_components(colour):
    # No closing: the selected component contains only actual colour pixels.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(colour.astype(np.uint8),8)
    keep = np.zeros(n,bool)
    if n>1:
        width=stats[1:,cv2.CC_STAT_WIDTH];height=stats[1:,cv2.CC_STAT_HEIGHT]
        keep[1:] = ((width>=CONFIG['component_min_native_width']) &
                    (stats[1:,cv2.CC_STAT_AREA]>=CONFIG['component_min_area_pixels']) &
                    (height<=width*CONFIG['component_max_height_over_width']))
    return keep[labels]

def _observed_runs(colour):
    start=colour & ~np.vstack((np.zeros((1,colour.shape[1]),bool),colour[:-1]))
    stop=colour & ~np.vstack((colour[1:],np.zeros((1,colour.shape[1]),bool)))
    counts=start.sum(0)
    first=np.argmax(start,axis=0).astype(np.int16)
    last=(colour.shape[0]-1-np.argmax(stop[::-1],axis=0)).astype(np.int16)
    valid=(counts==1)&(last-first+1<=CONFIG['column_run_max_height_native_px'])
    return first,last,counts,valid

def observed_column_region(bgr, boxes, *, source_y_search_range=None):
    """Return bounds with evidence in this exact RGB column, never a fit line."""
    if bgr.shape[:2] != tuple(CONFIG['native_hw']): raise ValueError('Native RGB dimensions changed')
    excluded=excluded_pixels(boxes,bgr.shape[:2]);usable=~excluded
    lower, upper = (CONFIG['source_y_search_range'] if source_y_search_range is None else source_y_search_range)
    if (type(lower) is not int or type(upper) is not int
            or not 0 <= lower < upper <= bgr.shape[0]):
        raise ValueError('Explicit original native RGB search rows required')
    usable[:lower] = False;usable[upper:] = False
    hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
    blue=(hsv[:,:,0]>=90)&(hsv[:,:,0]<=135)&(hsv[:,:,1]>=60)&usable
    yellow=(hsv[:,:,0]>=8)&(hsv[:,:,0]<=45)&(hsv[:,:,1]>=30)&usable
    yellow &= bgr[:,:,0].astype(np.int16)+15 < np.minimum(bgr[:,:,1],bgr[:,:,2])
    blue=_long_narrow_components(blue);yellow=_long_narrow_components(yellow)
    cap_upper,cap,cap_count,cap_valid=_observed_runs(blue)
    foot,foot_lower,foot_count,foot_valid=_observed_runs(yellow)
    reasons=np.zeros(bgr.shape[1],np.uint16)
    reasons[cap_count==0] |= 1;reasons[foot_count==0] |= 2
    reasons[cap_count>1] |= 4;reasons[foot_count>1] |= 8
    reasons[(cap_count==1)&~cap_valid] |= 1
    reasons[(foot_count==1)&~foot_valid] |= 2
    gap=foot.astype(int)-cap.astype(int)
    ordered=(gap>=30)&(gap<=600)
    reasons[cap_valid&foot_valid&~ordered] |= 16
    unique=cap_valid&foot_valid&ordered
    adjacent=np.zeros(len(unique),bool)
    adjacent[1:] = (unique[1:]&unique[:-1] &
        (np.abs(np.diff(cap.astype(int)))<=3) & (np.abs(np.diff(foot.astype(int)))<=3))
    # Seventeen adjacent links require eighteen observed columns (-9..+8).
    # Every column contains its own cap AND foot. A missing column is not filled.
    radius=CONFIG['directly_observed_continuity_radius_native_px']
    continuity=np.convolve(adjacent.astype(int),np.ones(2*radius+1,int),'same') == 2*radius+1
    reasons[unique&~continuity] |= 32
    valid=unique&continuity
    top=np.where(valid,cap+6,-1).astype(np.int16)
    bottom=np.where(valid,foot-6,-1).astype(np.int16)
    return dict(top=top,bottom=bottom,column_reason=reasons,excluded=excluded,
                cap_observed_upper=cap_upper,cap_observed_lower=cap,
                foot_observed_upper=foot,foot_observed_lower=foot_lower,
                directly_supported_columns=int(valid.sum()))

def endpoint_region_status(uv, bounds, people_available):
    points=np.asarray(uv,float);pix=np.rint(points).astype(int)
    reasons=np.zeros(len(points),np.uint16)
    native=np.isfinite(points).all(1)&(pix[:,0]>=0)&(pix[:,0]<1920)&(pix[:,1]>=0)&(pix[:,1]<1080)
    reasons[~native] |= 512
    if not people_available:
        reasons[:] |= 256
        return reasons
    k=np.flatnonzero(native);x,y=pix[k].T
    reasons[k] |= bounds['column_reason'][x]
    reasons[k[bounds['excluded'][y,x]]] |= 64
    inside=(bounds['top'][x]>=0)&(points[k,1]>bounds['top'][x])&(points[k,1]<bounds['bottom'][x])
    reasons[k[~inside]] |= 128
    return reasons

def point_record(record):
    secondary='original_source_uv' in record
    return dict(index=int(record.get('current_original_prefix_bundle_index',record.get('original_global_bundle_index'))),
        source_pts=int(record['source_pts']),target_pts=int(record['target_pts']),
        source_uv=record['original_source_uv'] if secondary else record['source_uv'],
        target_uv=record['original_target_uv'] if secondary else record['target_uv'])

def material_fact_fields(record):
    """Keep supplied human identity distinct from this algorithm's navigation."""
    group=record['test_group']
    pending=group in ('pending_original_low_cut5','pending_original_low_cut8')
    fact={'manually_confirmed_far_board':'confirmed_static_current_far_board_printed_material',
          'manually_rejected_nearfield_equipment':'confirmed_nearfield_equipment_or_judges_table_not_far_board'}.get(group)
    if pending:fact='not_individually_material_reviewed'
    if fact is None:raise ValueError('Unknown input human material status')
    return dict(input_human_material_identity=fact,
        algorithm_navigation_status='directly_observed_interior_at_both_endpoints' if record['directly_observed_region_both_endpoints'] else 'missing_or_ambiguous_source_region_evidence',
        physical_role_status=fact if not pending else 'unknown; both-source material/occluder review pending',
        existing_human_material_fact_or_solver_role_changed_by_region_mask=False,
        automatically_material_qualified_or_admitted=False)

def clarify_existing_report(path,output):
    """Metadata repair only: authenticate and keep all masks/bounds unchanged."""
    report=read(path)
    if report['schema']!='rinkspace.original-FIT-source-only-observed-column-board-region-review.v1':
        raise ValueError('Expected preserved v1 source-region report')
    assert sha(report['mask_archive']['path'])==report['mask_archive']['sha256']
    legacy_config=dict(CONFIG)
    legacy_config.pop('continuity_window_is_17_adjacent_links_requiring_18_observed_columns')
    legacy_config.pop('actual_observed_column_offsets_native_px')
    assert report['configuration']==legacy_config
    for record in report['point_masks']:
        record.update(material_fact_fields(record))
        record.pop('material_qualified_or_admitted',None)
    report.update(schema='rinkspace.original-FIT-source-only-observed-column-board-region-review.v2',
        configuration=CONFIG,authenticated_numerical_parent_review=ref(path),
        clarification_changes_only=True,
        source_RGB_bounds_masks_point_reason_bits_and_gate_values_changed=False,
        human_positive_navigation_unknown_does_not_reject_or_retire_the_source=True,
        algorithm_interior_candidate_does_not_override_human_negative_identity=True,
        clarification_script=ref(Path(__file__).resolve()))
    testref=report.get('reused_same_configuration_source_region_test')
    if testref:
        clarified=Path(testref['path']).with_name('observed_source_board_region_review_v2.json')
        if clarified.exists():report['clarified_same_configuration_human_test']=ref(clarified)
    out=output/'observed_source_board_region_review_v2.json'
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(dict(path=str(out),sha256=sha(out),summary=report['summary'],
        numeric_arrays_reused_bitexact=report['mask_archive'],clarification_only=True)),flush=True)

def overlay(bgr,bounds,records,endpoint,status):
    out=bgr.copy();valid=bounds['top']>=0
    # Transparent interior visualises only this exact frame's supported columns.
    nav=np.zeros(bgr.shape[:2],bool)
    for x in np.flatnonzero(valid): nav[int(bounds['top'][x]):int(bounds['bottom'][x])+1,x]=True
    nav &= ~bounds['excluded']
    out[nav]=(out[nav]*.78+np.asarray([0,160,0])*.22).astype(np.uint8)
    xs=np.flatnonzero(valid)
    out[bounds['top'][xs],xs]=(0,255,255);out[bounds['bottom'][xs],xs]=(0,255,255)
    for key,colour in (('cap_observed_upper',(255,180,0)),('cap_observed_lower',(255,0,255)),
                       ('foot_observed_upper',(0,180,255)),('foot_observed_lower',(255,255,0))):
        out[bounds[key][xs],xs]=colour
    for record,reason in zip(records,status):
        xy=tuple(np.rint(record[endpoint+'_uv']).astype(int))
        cv2.circle(out,xy,4,(30,240,30) if reason==0 else (20,30,255),1)
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--candidates',type=Path)
    ap.add_argument('--qualification',type=Path,action='append',default=[])
    ap.add_argument('--reuse-source-region-test',type=Path,
                    help='Carry the same RGB configuration test and reuse its exact representative overlays')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--clarify-existing-report',type=Path,
                    help='Write v2 human-fact metadata without decoding RGB or changing numerical masks')
    ap.add_argument('--representative-only',action='store_true',help='Test fixed source-time representatives and all qualification endpoints first')
    ap.add_argument('--full-native-colour-search',action='store_true',
                    help='Use the full actual RGB height; a fixed y255 floor is not a material boundary')
    ap.add_argument('--pair-id',type=int,action='append',
                    help='Inspect these original pair identities; omitted means all existing candidate pairs')
    ap.add_argument('--continuous-region-sheets',action='store_true',
                    help='Save continuous native board-band review sheets, without per-frame full RGB duplicates')
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.clarify_existing_report:
        clarify_existing_report(a.clarify_existing_report,a.output)
        return
    if not a.candidates:raise ValueError('Candidates required when not clarifying an existing report')
    existing=read(a.candidates);retirement=read(RETIREMENT)
    assert existing['source_video_sha256']==retirement['source_video_sha256']
    shot=int(existing.get('shot_index',5))
    if shot not in (3,5,8):raise ValueError('Original low cut3/cut5/cut8 source candidates are supported')
    pending_group=f'pending_original_low_cut{shot}'
    representatives=existing.get('representative_source_RGB',existing.get('representative_pair_navigation',[]))
    prior=None;prior_frames={};prior_views={}
    if a.reuse_source_region_test:
        if a.full_native_colour_search or a.pair_id:
            raise ValueError('Changed source navigation scope cannot reuse the old numerical region test')
        prior=read(a.reuse_source_region_test)
        assert prior['schema'] in ('rinkspace.original-FIT-source-only-observed-column-board-region-review.v1',
                                   'rinkspace.original-FIT-source-only-observed-column-board-region-review.v2')
        comparable=dict(CONFIG)
        if prior['schema'].endswith('.v1'):
            comparable.pop('continuity_window_is_17_adjacent_links_requiring_18_observed_columns')
            comparable.pop('actual_observed_column_offsets_native_px')
        assert prior['configuration']==comparable and prior['original_candidates']==ref(a.candidates)
        prior_frames={r['actual_pts']:r for r in prior['source_RGB_receipts']}
        prior_views={r['actual_pts']:r for r in prior['representative_source_RGB_overlays']}
        for view in prior_views.values():assert sha(view['path'])==view['sha256']
    people={int(r['pts']):r['boxes'] for r in read(PEOPLE)['frames']}
    requested_pairs=set(a.pair_id) if a.pair_id else None
    records=[dict(point_record(r),test_group=pending_group,source_record_ordinal=i,
                  original_raw_pair_index=r['original_raw_pair_index'])
             for i,r in enumerate(original_candidate_records(existing))
             if requested_pairs is None or r['original_raw_pair_index'] in requested_pairs]
    if requested_pairs is not None and {r['original_raw_pair_index']for r in records}!=requested_pairs:
        raise ValueError('A requested original pair is absent; do not fabricate source material')
    selected_pairs={g['original_raw_pair_index'] for g in representatives}
    if a.representative_only:
        records=[r for r in records if existing['records'][r['source_record_ordinal']]['original_raw_pair_index'] in selected_pairs]
    for path in a.qualification:
        qual=read(path)
        records += [dict(point_record(r),test_group='manually_confirmed_far_board',qualification=str(path))
                    for r in qual['admitted_source_only_rows']]
        records += [dict(point_record(r),test_group='manually_rejected_nearfield_equipment',qualification=str(path))
                    for r in qual.get('rejected_far_board_roles',[])]
    wanted={r[key] for r in records for key in ('source_pts','target_pts')}
    exposure_rows=collections.defaultdict(list)
    for i,r in enumerate(records):
        for endpoint in ('source','target'):exposure_rows[r[endpoint+'_pts']].append((i,endpoint))
    representative_pts={int(f['actual_pts']) for g in representatives for f in g.get('source_RGB',[])}
    representative_pts|={int(g[key]) for g in representatives if 'source_RGB' not in g for key in ('source_pts','target_pts')}
    # Human-rejected source contexts are also mandatory review images.
    representative_pts |= {r[key] for r in records if r['test_group']=='manually_rejected_nearfield_equipment' for key in ('source_pts','target_pts')}
    if requested_pairs is not None:
        representative_pts |= wanted
    endpoints=np.zeros((len(records),2),np.uint16);coltop=[];colbottom=[];colreason=[];readpts=[]
    colour_runs={key:[]for key in ('cap_observed_upper','cap_observed_lower','foot_observed_upper','foot_observed_lower')}
    rgb_receipts=[];source_counts=[];overviews=[];start=time.perf_counter()
    continuous_panels=[];continuous_frames=[];continuous_pages=[];continuous_mapping=[]
    with av.open(retirement['source_video']) as container:
        stream=container.streams.video[0];stream.thread_count=1
        if str(stream.time_base)!='1/15360':raise ValueError('Source time base changed')
        container.seek(min(wanted),stream=stream,backward=True,any_frame=False)
        for frame in container.decode(stream):
            if frame.pts is not None and frame.pts>max(wanted):break
            if frame.pts not in wanted:continue
            pts=int(frame.pts);bgr=frame.to_ndarray(format='bgr24')
            decoded_sha=hashlib.sha256(bgr.tobytes()).hexdigest()
            if pts in prior_frames:assert decoded_sha==prior_frames[pts]['decoded_BGR_sha256']
            bounds=observed_column_region(bgr,people.get(pts,[]),
                source_y_search_range=[0,bgr.shape[0]] if a.full_native_colour_search else None)
            readpts.append(pts);coltop.append(bounds['top']);colbottom.append(bounds['bottom']);colreason.append(bounds['column_reason'])
            for key in colour_runs:
                colour_runs[key].append(np.where(bounds['top']>=0,bounds[key],-1).astype(np.int16))
            source_counts.append(dict(actual_pts=pts,directly_supported_columns=bounds['directly_supported_columns'],people_available=pts in people))
            for endpoint in ('source','target'):
                indices=[i for i,k in exposure_rows[pts] if k==endpoint]
                if indices:
                    reasons=endpoint_region_status([records[i][endpoint+'_uv'] for i in indices],bounds,pts in people)
                    endpoints[indices,0 if endpoint=='source' else 1]=reasons
            if pts in representative_pts:
                rr=[records[i] for i,k in exposure_rows[pts]]
                # Endpoint keys are copied only for displaying, never source mutations.
                display=[dict(r,source_uv=r[k+'_uv']) for r,(_,k) in zip(rr,exposure_rows[pts])]
                status=[int(endpoints[i,0 if k=='source' else 1]) for i,k in exposure_rows[pts]]
                if pts in prior_views and prior_views[pts]['point_count']==len(rr):
                    overviews.append(dict(prior_views[pts],exact_source_RGB_and_configuration_reused=True))
                else:
                    image=overlay(bgr,bounds,display,'source',status)
                    name=a.output/f'source_only_observed_column_mask_{pts}.png'
                    assert cv2.imwrite(str(name),image)
                    overviews.append(dict(actual_pts=pts,path=str(name),sha256=sha(name),point_count=len(rr),
                        manually_rejected_nearfield_present=any(r['test_group']=='manually_rejected_nearfield_equipment' for r in rr)))
            if a.continuous_region_sheets:
                valid=bounds['top']>=0
                if valid.any():
                    y0=max(0,int(bounds['cap_observed_upper'][valid].min())-65)
                    y1=min(1080,int(bounds['foot_observed_lower'][valid].max())+65)
                else:
                    # No measured colour region: show the complete original
                    # context, not a guessed or horizontally extended band.
                    y0,y1=0,1080
                # Region-role review, not new per-point image measurements.
                view=overlay(bgr,bounds,[],'source',[])[y0:y1]
                panel=np.full((view.shape[0]+36,view.shape[1],3),245,np.uint8)
                panel[36:]=view
                title=f"PTS {pts} | native crop y {y0}:{y1} | observed columns {int(valid.sum())} | people frame {'present' if pts in people else 'MISSING: all points UNKNOWN'}"
                cv2.putText(panel,title,(10,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(15,15,15),1,cv2.LINE_AA)
                frame_record=dict(actual_pts=pts,decoded_BGR_sha256=decoded_sha,
                    native_crop_xywh=[0,y0,1920,y1-y0],native_pixels_not_resized_or_warped=True,
                    people_frame_available=pts in people,directly_supported_columns=int(valid.sum()))
                continuous_panels.append(panel);continuous_frames.append(frame_record)
                if len(continuous_panels)==6 or len(readpts)==len(wanted):
                    page_index=len(continuous_pages)
                    path=a.output/f'continuous_native_RGB_material_regions_page_{page_index:03d}.png'
                    if path.exists():raise FileExistsError('Preserve the historical continuous RGB region page')
                    canvas=np.vstack(continuous_panels);assert cv2.imwrite(str(path),canvas)
                    page=dict(ref(path),page_index=page_index,actual_pts=[r['actual_pts']for r in continuous_frames],
                        native_rendered_hw=list(canvas.shape[:2]))
                    continuous_pages.append(page)
                    for row,one in enumerate(continuous_frames):
                        continuous_mapping.append(dict(one,page_index=page_index,page_row=row))
                    continuous_panels=[];continuous_frames=[]
            rgb_receipts.append(dict(actual_pts=pts,decoded_BGR_sha256=decoded_sha))
            if len(readpts)%100==0:print(json.dumps(dict(exposures=len(readpts),requested=len(wanted),seconds=time.perf_counter()-start)),flush=True)
    if set(readpts)!=wanted:raise ValueError(('Missing exact actual RGB',sorted(wanted-set(readpts))))
    result=[]
    for r,er in zip(records,endpoints):
        one=dict(r,source_endpoint_reason_bits=int(er[0]),target_endpoint_reason_bits=int(er[1]),
            directly_observed_region_both_endpoints=bool(np.all(er==0)))
        one.update(material_fact_fields(one));result.append(one)
    groups={}
    for group in sorted({r['test_group'] for r in records}):
        rows=np.asarray([i for i,r in enumerate(records) if r['test_group']==group]);rr=endpoints[rows]
        groups[group]=dict(original_point_count=len(rows),directly_observed_region_both_endpoints=int(np.all(rr==0,axis=1).sum()),
            at_least_one_endpoint_unknown=int(np.any(rr!=0,axis=1).sum()),
            endpoint_reason_counts_can_overlap={name:int(np.count_nonzero(rr&bit)) for bit,name in REASONS.items()})
    pair_groups=collections.defaultdict(list)
    for i,r in enumerate(records):
        if r['test_group']==pending_group:pair_groups[r['original_raw_pair_index']].append(i)
    pair_review=[]
    for pair,indices in sorted(pair_groups.items()):
        rr=endpoints[indices];r=records[indices[0]]
        pair_review.append(dict(original_raw_pair_index=pair,source_pts=r['source_pts'],target_pts=r['target_pts'],
            original_candidate_points=len(indices),directly_observed_region_both_endpoints=int(np.all(rr==0,axis=1).sum()),
            unknown_region_point_count=int(np.any(rr!=0,axis=1).sum()),
            source_endpoint_unknown=int(np.count_nonzero(rr[:,0])),target_endpoint_unknown=int(np.count_nonzero(rr[:,1]))))
    array_path=a.output/'observed_source_column_bounds_and_point_masks_v1.npz'
    np.savez_compressed(array_path,actual_pts=np.asarray(readpts,np.int64),
        directly_observed_interior_top=np.asarray(coltop,np.int16),directly_observed_interior_bottom=np.asarray(colbottom,np.int16),
        column_reason_bits=np.asarray(colreason,np.uint16),point_endpoint_reason_bits=endpoints,
        original_source_index=np.asarray([r['index'] for r in records],np.int64),
        point_both_endpoint_region_candidate=np.all(endpoints==0,axis=1),
        **{key:np.asarray(value,np.int16)for key,value in colour_runs.items()})
    report=dict(schema='rinkspace.original-FIT-source-only-observed-column-board-region-review.v2',
        status='source_region_masks_for_human_review_not_material_admission',
        original_candidates=ref(a.candidates),manual_positive_and_negative_test_packets=list(map(ref,a.qualification)),
        source_video=retirement['source_video'],source_video_sha256_from_existing_verified_ledger=retirement['source_video_sha256'],
        people_boxes=ref(PEOPLE),script=ref(Path(__file__).resolve()),
        configuration=dict(CONFIG,source_y_search_range=[0,1080]) if a.full_native_colour_search else CONFIG,
        test_mode='fixed_source_time_representatives_plus_manual_test_endpoints' if a.representative_only else f'all_original_low_cut{shot}_overlap_candidates_plus_manual_test_endpoints',
        reason_bits=REASONS,summary=groups,actual_exposures_read=len(readpts),source_RGB_receipts=rgb_receipts,
        per_exposure_region_support=source_counts,point_masks=result,mask_archive=ref(array_path),representative_source_RGB_overlays=overviews,
        per_original_pair_region_review=pair_review,
        reused_same_configuration_source_region_test=ref(a.reuse_source_region_test) if a.reuse_source_region_test else None,
        reused_manual_source_test_summary=prior['summary'] if prior else None,
        source_UV_sigma_PTS_weights_rows_and_existing_solver_roles_modified=False,new_source_rows=0,HELD_promoted=False,
        model_camera_geometry_guard_residual_J_graph_solver_loaded=False,
        all_interior_pixels_cannot_be_proven_board_by_two_boundaries=True,
        human_positive_navigation_unknown_does_not_reject_or_retire_the_source=True,
        algorithm_interior_candidate_does_not_override_human_negative_identity=True,
        unmasked_equipment_or_people_detection_misses_remain_possible=True,
        automatically_material_qualified_point_count=0,accuracy_accepted=False,wall_seconds=time.perf_counter()-start)
    report.update(observed_colour_run_edges_are_not_precise_TOP_or_contact_measurements=True,
        directly_observed_interior_basis='blue colour-run lower + inherited6px; yellow colour-run upper - inherited6px',
        observed_colour_run_overlay_BGR=dict(cap_observed_upper=[255,180,0],cap_observed_lower=[255,0,255],
            foot_observed_upper=[0,180,255],foot_observed_lower=[255,255,0]),
        missing_columns_and_mixed_or_multiple_colour_runs_preserved_unknown=True)
    if a.continuous_region_sheets:
        report.update(continuous_actual_RGB_region_pages=continuous_pages,
            continuous_actual_RGB_region_frame_mapping=continuous_mapping,
            review_sheet_frame_count_is_presentation_only_not_source_weight_or_point_limit=True,
            continuous_regions_are_pending_material_role_review_not_automatic_qualification=True)
    if requested_pairs is not None:
        report.update(requested_original_pair_ids=sorted(requested_pairs),
            test_mode='requested_original_pairs_actual_RGB_region_navigation_for_material_review')
    if a.full_native_colour_search:
        report.update(source_y_search_range_basis='full actual native RGB height; no fitted camera/board boundary',
            original_LK_points_do_not_require_corner_redetection=True,
            observed_region_candidate_does_not_automatically_establish_material_role=True)
    if shot!=5:report['shot_index']=shot
    out=a.output/'observed_source_board_region_review_v2.json'
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(dict(path=str(out),sha256=sha(out),summary=groups,actual_exposures=len(readpts),seconds=report['wall_seconds'])),flush=True)

if __name__=='__main__':main()
