"""Render all nine native source shots from one immutable solver state.

Reuses the complete-cut grid compositor, exact clock checks and ordered shared
pixel ring. No body, camera smoothing, fitting or frame/geometry selection.
Without --video only the clock, inventory and requested source checks are made.
"""
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(Path(__file__).parent)]
import render_complete_problem_cuts_grid_only as grid


def source_check_frames(args,selected,source_sha256):
    if args.source_check_cache is None:
        yield from grid.decode_selected(args.source,selected)
        return
    cache=json.loads(args.source_check_cache.read_text(encoding='utf8'))
    if (cache.get('source_sha256_from_timeline')!=source_sha256
            or grid.STORAGE.source(cache['source_path'])!=args.source):
        raise ValueError('Cached original pixels belong to a different source video')
    rows={int(row['pts']):row for row in cache['frames']}
    if not set(selected).issubset(rows):raise ValueError('Cached native source checks are incomplete')
    for pts in selected:
        row=rows[pts];pixels=grid.cv2.imread(str(grid.STORAGE.source(row['path'])))
        if (pixels is None or pixels.shape!=grid.movie.NATIVE_SHAPE
                or hashlib.sha256(pixels.tobytes()).hexdigest()!=row['native_sha256']):
            raise ValueError('Cached actual native BGR pixels changed: '+str(pts))
        yield pts,pixels


def run(args):
    args.source=grid.STORAGE.source(args.source)
    args.state_identity=grid.STORAGE.resolve(args.state_identity)
    args.output=grid.STORAGE.resolve(args.output)
    if not args.output.resolve().is_relative_to(Path('D:/CodexData/RinkSpace_Data').resolve()):
        raise ValueError('Programme media and clock outputs must remain in the existing Data root')
    if args.output.exists():raise FileExistsError(args.output)
    identity=json.loads(args.state_identity.read_text(encoding='utf8'))
    view=grid.FrozenProblemCutGridView(identity['camera_path'],
        args.state_identity.parent/'scene_UNACCEPTED.json',args.source,
        args.masks,args.people,args.state_identity)
    native=[];indices=[];scopes=[]
    for cut,(start,end)in sorted(view.cuts.items()):
        pts=view.cut_pts(cut);native.extend(pts);indices.extend([cut]*len(pts))
        scopes.append(dict(cut=cut,start_pts=start,end_pts_exclusive=end,
            source_start=grid.source_time(start),source_end_exclusive=grid.source_time(end),
            native_presentations=len(pts),actual_exposures=len({view.aliases[int(p)]for p in pts}),
            source_last_native_pts=int(pts[-1]),complete_cut=True))
    native=grid.np.asarray(native,grid.np.int64)
    if not grid.np.array_equal(native,view.pts):
        raise ValueError('All complete shots must cover the exact ordered native programme once')
    requested=args.check_pts
    if not requested and args.source_check_cache is not None:
        requested=[int(row['pts'])for row in json.loads(args.source_check_cache.read_text(encoding='utf8'))['frames']]
    selected=sorted(set(requested))
    if not set(selected).issubset(set(map(int,native))):
        raise ValueError('Source check PTS must belong to the complete programme')
    args.output.mkdir(parents=True)
    clock=args.output/'complete_programme_native_actual_clock.npz'
    actual=grid.np.asarray([view.aliases[int(p)]for p in native],grid.np.int64)
    grid.np.savez_compressed(clock,native_pts=native,actual_pts=actual,
        cut_index=grid.np.asarray(indices,grid.np.int16),source_time_base=grid.np.array('1/15360'),
        native_rate=grid.np.array('60/1'))
    frames=[];checks=[];serial=[]
    for pts,source in source_check_frames(args,selected,view.source_metadata['sha256_from_authoritative_source_timeline']):
        image=view.draw(source,pts);target=args.output/f'native_all_model_{pts}.png'
        if not grid.cv2.imwrite(str(target),image):raise OSError(target)
        checks.append(dict(image=str(target),**view.last_metrics))
        serial.append(hashlib.sha256(image).hexdigest())
        if args.check_parallel:frames.append((pts,source))
    parallel=None
    if args.check_parallel:
        if not frames or args.draw_workers<=1:
            raise ValueError('Parallel pixel verification needs requested source checks and multiple workers')
        with grid.GridOnlyOrderedSharedDraw(view,args.draw_workers,args.draw_slots,
                args.draw_chunk_size,True)as ring:
            observed=[(pts,digest)for pts,_,digest in ring.frames(frames)]
            statistics=ring.stats.copy()
        if observed!=list(zip(selected,serial)):
            raise ValueError('Same-source serial and parallel canvases or native order differ')
        parallel=dict(exact_serial_process_canvas_sha256=True,source_pts=selected,pipeline=statistics)
    video=None
    if args.video:
        video=grid.render_video(args,view,0,native,filename='whole_programme_grid_UNACCEPTED.mp4')
    view.assert_frozen_inputs()
    receipt=dict(schema='rinkspace.complete-programme-immutable-grid-review.v1',
        complete_q_sha256=identity['q_sha256'],state_identity=grid.reference(args.state_identity),
        complete_state_camera_scene_binding=view.input_refs,source=view.source_metadata,scopes=scopes,
        native_presentations=len(native),actual_exposures=len(set(map(int,actual))),
        native_start_pts=int(native[0]),native_stop_pts_exclusive=int(native[-1])+256,
        native_pts_sha256=hashlib.sha256(native.tobytes()).hexdigest(),clock=grid.reference(clock),
        source_and_plan_all_geometry=view.renderer.path_counts,advertisement_ids=view.plan.advertisement_ids,
        every_frame_projects_same_complete_scene=True,source_clock_and_aliases_reused=True,
        geometry_hidden_by_alignment_or_confidence=False,camera_smoothing=False,
        body_input_loaded=False,body_fitted=False,body_P_G_or_trajectory_drawn=False,
        source_check_cache=None if args.source_check_cache is None else grid.reference(args.source_check_cache),
        scene_unit_m_for_board_grid_only=view.scene_unit_m,verified_metric_scale=False,
        small_checks=checks,small_parallel_check=parallel,video=video,
        fullprogramme_video_generated=video is not None,accuracy_accepted=False,
        small_checks_do_not_validate_complete_shots=True)
    grid.write(args.output/'grid_only_review_entry.json',receipt)
    print(json.dumps(dict(output=str(args.output),native_presentations=len(native),
        actual_exposures=receipt['actual_exposures'],small_checks=len(checks),
        parallel_pixel_exact=parallel is not None,video_generated=video is not None)),flush=True)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-identity',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--source',type=Path,default=grid.DEFAULT_SOURCE)
    parser.add_argument('--masks',type=Path,default=grid.DEFAULT_MASKS)
    parser.add_argument('--people',type=Path,default=grid.DEFAULT_PEOPLE)
    parser.add_argument('--check-pts',type=int,nargs='*',default=[])
    parser.add_argument('--source-check-cache',type=Path,
        help='Reuse SHA-matched lossless native source checks; defaults to every cached PTS when --check-pts is omitted')
    parser.add_argument('--check-parallel',action='store_true')
    parser.add_argument('--draw-workers',type=int,default=min(4,os.cpu_count()or 1))
    parser.add_argument('--draw-chunk-size',type=int,default=4)
    parser.add_argument('--draw-slots',type=int)
    parser.add_argument('--video',action='store_true')
    run(parser.parse_args())


if __name__=='__main__':main()
