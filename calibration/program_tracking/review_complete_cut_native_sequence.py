"""All actual exposures of complete cuts, from exact cached original pixels.

Reuses the immutable scene compositor and its ordered shared-memory drawing
ring. Produces contact pages and explicit native detail checks, never a video.
Missing historical PNGs can explicitly be decoded in memory from the exact
original video and checked against their recorded BGR hashes. No fitting,
camera fill/smoothing, source cache writing or observed point creation.
"""
from __future__ import annotations
import argparse, hashlib, json, os, time
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(Path(__file__).parent)]
import render_complete_problem_cuts_grid_only as grid

DEFAULT_CACHE=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/history/work_sessions/'
    '20261002_01a0fd26-33ba-7873-9aa9-bbd9d2ca1328/source_solver_audit/'
    'secondary_complete_native_lossless_cache_v1/complete_original_native_cache_manifest.json')

def _cache_inventory(path,source_sha,source_path):
    cache=json.loads(path.read_text(encoding='utf8'))
    if cache.get('schema')=='rinkspace.complete-secondary-lossless-native-source-cache.v1':
        if cache['source']['sha256']!=source_sha or cache['native_timebase']!='1/15360':
            raise ValueError('Cached originals have a different source identity/clock')
        if grid.STORAGE.source(cache['source']['path'])!=source_path:
            raise ValueError('Cached original source path differs')
        records=[]
        for row in cache['frames']:
            original=Path(row['raw_png']);resolved=grid.STORAGE.resolve(original)
            # A cache's historical relocation is accepted only by exact PNG
            # and decoded BGR SHA below, never by its filename alone.
            if not resolved.is_file():resolved=path.parent/original.name
            records.append(dict(native_pts=int(row['native_pts']),actual_pts=int(row['actual_pts']),
                path=str(resolved),png_sha256=row['raw_png_sha256'],native_bgr_sha256=row['source_BGR_sha256'],
                historical_raw_png=str(original)))
    elif 'source_sha256_from_timeline'in cache and 'frames'in cache:
        if cache['source_sha256_from_timeline']!=source_sha or grid.STORAGE.source(cache['source_path'])!=source_path:
            raise ValueError('Cached native checks have a different source video')
        records=[dict(native_pts=int(r['pts']),actual_pts=int(r.get('actual_pts',r['pts'])),
            path=str(grid.STORAGE.source(r['path'])),png_sha256=grid.sha(grid.STORAGE.source(r['path'])),
            native_bgr_sha256=r['native_sha256'])for r in cache['frames']]
    else:raise ValueError('Supply an authenticated original native source cache')
    bypts={r['native_pts']:r for r in records}
    if len(bypts)!=len(records):raise ValueError('Cache repeats a native source presentation')
    return bypts

def _read_original(row):
    path=Path(row['path'])
    if not path.is_file()or grid.sha(path)!=row['png_sha256']:
        raise ValueError('Exact cached original PNG changed: '+str(row['native_pts']))
    pixels=grid.cv2.imread(str(path))
    if (pixels is None or pixels.shape!=grid.movie.NATIVE_SHAPE or
            hashlib.sha256(pixels.tobytes()).hexdigest()!=row['native_bgr_sha256']):
        raise ValueError('Exact cached original native BGR changed: '+str(row['native_pts']))
    return pixels

def run(args):
    args.output=grid.STORAGE.resolve(args.output)
    if not args.output.resolve().is_relative_to(Path('D:/CodexData/RinkSpace_Data').resolve()):
        raise ValueError('Bulk sequence review belongs in the existing Data root')
    if args.output.exists():raise FileExistsError(args.output)
    args.state_identity=grid.STORAGE.resolve(args.state_identity)
    args.source=grid.STORAGE.source(args.source)
    identity=json.loads(args.state_identity.read_text(encoding='utf8'))
    view=grid.FrozenProblemCutGridView(identity['camera_path'],args.state_identity.parent/'scene_UNACCEPTED.json',
        args.source,args.masks,args.people,args.state_identity)
    cache=_cache_inventory(args.source_cache_manifest,
        view.source_metadata['sha256_from_authoritative_source_timeline'],args.source)
    native=[];actual=[];cuts=[];scopes=[]
    for cut in sorted(set(args.cut)):
        if cut not in view.cuts:raise ValueError('Cut outside the complete programme')
        pts=view.cut_pts(cut);reps=grid.np.asarray([view.aliases[int(p)]for p in pts],grid.np.int64)
        unique=grid.np.unique(reps);start,end=view.cuts[cut]
        native.extend(map(int,pts));actual.extend(map(int,reps));cuts.extend([cut]*len(pts))
        scopes.append(dict(cut=cut,native_start_pts=start,native_stop_pts_exclusive=end,
            source_start=grid.source_time(start),source_last=grid.source_time(int(pts[-1])),
            source_stop_exclusive=grid.source_time(end),complete_cut=True,native_presentations=len(pts),actual_exposures=len(unique)))
    selected=sorted(set(actual));missing=set(selected)-cache.keys()
    if missing:raise ValueError('Existing original cache omits actual exposures: '+str(sorted(missing)[:10]))
    source_rows=[]
    for pts in selected:
        row=cache[pts]
        if row['actual_pts']!=view.aliases[pts]or pts!=view.aliases[pts]:
            raise ValueError('Actual exposure clock differs from original cache: '+str(pts))
        exists=Path(row['path']).is_file()
        if not exists and not args.decode_missing_source:raise FileNotFoundError(row['path'])
        source_rows.append(dict(row,source_time=grid.source_time(pts),
            pixel_read_method='existing_original_PNG'if exists else'exact_original_video_in_memory',
            original_native_presentations=[p for p,a in zip(native,actual)if a==pts]))
    missing_pixels=[row['native_pts']for row in source_rows
                    if row['pixel_read_method']=='exact_original_video_in_memory']
    if missing_pixels:
        if grid.sha(args.source)!=view.source_metadata['sha256_from_authoritative_source_timeline']:
            raise ValueError('Original video differs from the historical source cache')
        view.source_metadata['file_rehashed_in_this_renderer']=True
    details={view.aliases[int(p)]for p in args.detail_pts}
    if not details.issubset(selected):raise ValueError('Detail source check lies outside the complete selected cuts')
    args.output.mkdir(parents=True)
    clock=args.output/'complete_selected_cut_native_actual_clock.npz'
    grid.np.savez_compressed(clock,native_pts=grid.np.asarray(native,grid.np.int64),
        actual_pts=grid.np.asarray(actual,grid.np.int64),cut_index=grid.np.asarray(cuts,grid.np.int16),
        actual_exposure_pts=grid.np.asarray(selected,grid.np.int64),source_time_base=grid.np.array('1/15360'))
    receipt=dict(schema='rinkspace.complete-cut-immutable-native-RGB-actual-exposure-sequence-review.v1',
        immutable_state_identity=grid.reference(args.state_identity),q_sha256=identity['q_sha256'],
        state_camera_scene_refs=view.input_refs,source=view.source_metadata,scopes=scopes,
        native_presentations=len(native),actual_exposures=len(selected),clock=grid.reference(clock),
        source_cache_manifest=grid.reference(args.source_cache_manifest),source_rows=source_rows,
        source_plan_ice_inventory=view.renderer.path_counts,advertisement_ids=view.plan.advertisement_ids,
        all65_ice_paths_all26_advertisements_boundaries_and_numbers_enabled=True,
        same_complete_solver_scene_on_every_frame=True,model_geometry_hidden_by_alignment_or_confidence=False,
        original_measured_person_occlusion_masks_reused=True,no_camera_smoothing_or_interpolation=True,
        new_original_frames_decoded=0,new_source_images_written=0,new_source_or_FIT_or_HELD_points=0,
        current_source_points=None if view.current_points is None else view.current_points.reference,
        body_loaded_or_fitted=False,video_generated=False,accuracy_accepted=False,
        source_images_may_be_development_material_not_claimed_independent_heldout=True,
        prepare_only=args.prepare_only,pages=[],native_detail_checks=[])
    if args.prepare_only:
        view.assert_frozen_inputs();grid.write(args.output/'continuous_actual_exposure_review.json',receipt)
        print(json.dumps(dict(stage='prepared',output=str(args.output),scopes=scopes,cache_missing=0)),flush=True)
        return receipt

    # Contact pages retain complete source+plan canvases. Native details use
    # the compositor's full resolution; source PNG/BGR remain separately exact.
    started=time.perf_counter();page=[];source_index={pts:i for i,pts in enumerate(selected)}
    tile_w=grid.movie.SIZE[0]//2;tile_h=grid.movie.SIZE[1]//2
    def save_page():
        if not page:return
        index=len(receipt['pages'])+1
        canvas=grid.np.full((tile_h*3,tile_w*2,3),20,grid.np.uint8)
        for i,(_,tile,_)in enumerate(page):canvas[(i//2)*tile_h:(i//2+1)*tile_h,(i%2)*tile_w:(i%2+1)*tile_w]=tile
        path=args.output/f'complete_actual_exposure_review_page{index:03d}.png'
        if not grid.cv2.imwrite(str(path),canvas):raise OSError(path)
        receipt['pages'].append(dict(page=index,**grid.reference(path),
            actual_pts=[pts for pts,_,_ in page],source_times=[grid.source_time(pts)for pts,_,_ in page],
            source_rows=[source_rows[source_index[pts]]for pts,_,_ in page],
            exact_native_canvas_sha256=[digest for _,_,digest in page],complete_geometry_continuously_enabled=True))
        page.clear()
    def frames():
        decoded=iter(grid.decode_selected(args.source,missing_pixels))if missing_pixels else iter(())
        for row in source_rows:
            pts=row['native_pts']
            if row['pixel_read_method']=='existing_original_PNG':pixels=_read_original(row)
            else:
                decoded_pts,pixels=next(decoded)
                if (decoded_pts!=pts or pixels.shape!=grid.movie.NATIVE_SHAPE or
                        hashlib.sha256(pixels.tobytes()).hexdigest()!=row['native_bgr_sha256']):
                    raise ValueError('Original decoded pixels differ from the recorded native source: '+str(pts))
                receipt['new_original_frames_decoded']+=1
            yield pts,pixels
        if next(decoded,None)is not None:raise ValueError('Original decoder returned unrequested source pixels')
    pipeline=None;first_last=[]
    def consume(pts,image,digest):
        if pts!=selected[len(first_last)]:raise ValueError('Actual source order changed')
        first_last.append(pts)
        if pts in details:
            path=args.output/f'native_all_model_{pts}.png'
            if not grid.cv2.imwrite(str(path),image):raise OSError(path)
            receipt['native_detail_checks'].append(dict(pts=pts,source_time=grid.source_time(pts),**grid.reference(path),native_canvas_sha256=digest))
        tile=grid.cv2.resize(image,(tile_w,tile_h),interpolation=grid.cv2.INTER_AREA)
        page.append((pts,tile,digest))
        if len(page)==6:save_page()
    if args.draw_workers>1:
        with grid.GridOnlyOrderedSharedDraw(view,args.draw_workers,args.draw_slots,args.draw_chunk_size,True)as ring:
            for pts,image,digest in ring.frames(frames()):consume(pts,image,digest)
            pipeline=ring.stats.copy()
    else:
        for pts,pixels in frames():
            image=view.draw(pixels,pts);consume(pts,image,hashlib.sha256(image).hexdigest())
        pipeline=dict(mode='serial',workers=1)
    save_page()
    if first_last!=selected:raise ValueError('Sequence omits or repeats actual source exposures')
    view.assert_frozen_inputs()
    if grid.reference(args.source_cache_manifest)!=receipt['source_cache_manifest']:
        raise ValueError('Historical source cache manifest changed during review')
    receipt.update(drawing_pipeline=pipeline,complete_actual_exposure_order_preserved=True,
        all_selected_original_PNG_and_BGR_hashes_authenticated=not missing_pixels,
        all_selected_original_BGR_hashes_authenticated=True,
        native_all_model_canvas_digest=hashlib.sha256(''.join(d for p in receipt['pages']for d in p['exact_native_canvas_sha256']).encode()).hexdigest(),
        elapsed_seconds=time.perf_counter()-started,all_pages_actually_visually_reviewed=False,
        complete_rendered_sequence_alone_does_not_establish_absolute_accuracy=True)
    grid.write(args.output/'continuous_actual_exposure_review.json',receipt)
    print(json.dumps(dict(stage='complete_cached_actual_sequence',output=str(args.output),
        actual_exposures=len(selected),pages=len(receipt['pages']),native_detail_pts=sorted(details),
        elapsed_seconds=receipt['elapsed_seconds'],video_generated=False)),flush=True)
    return receipt

def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--state-identity',required=True,type=Path)
    cli.add_argument('--output',required=True,type=Path)
    cli.add_argument('--cut',nargs='+',type=int,default=[6])
    cli.add_argument('--source-cache-manifest',type=Path,default=DEFAULT_CACHE)
    cli.add_argument('--source',type=Path,default=grid.DEFAULT_SOURCE)
    cli.add_argument('--masks',type=Path,default=grid.DEFAULT_MASKS)
    cli.add_argument('--people',type=Path,default=grid.DEFAULT_PEOPLE)
    cli.add_argument('--detail-pts',nargs='*',type=int,default=[])
    cli.add_argument('--draw-workers',type=int,default=min(2,os.cpu_count()or 1))
    cli.add_argument('--draw-chunk-size',type=int,default=4)
    cli.add_argument('--draw-slots',type=int)
    cli.add_argument('--prepare-only',action='store_true')
    cli.add_argument('--decode-missing-source',action='store_true',
        help='Decode absent historical PNGs in memory and require their original native BGR hashes; never write a source cache')
    run(cli.parse_args())

if __name__=='__main__':main()
