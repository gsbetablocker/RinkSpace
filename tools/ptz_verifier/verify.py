#!/usr/bin/env python3
"""Standalone read-only PTZ verification CLI; never launches the RinkSpace solver."""
from __future__ import annotations
import os
# Avoid uncontrolled BLAS oversubscription; respect explicit user overrides.
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'2')
import argparse, json, platform, sys, time, traceback
from pathlib import Path
from fractions import Fraction
import numpy as np
import scipy, cv2
from ptzcheck import __version__
from ptzcheck.io import (fresh_output, prepare_bundle, probe_video, select_frames, decode_selected,
                         write_json, assert_unchanged, sha256)
from ptzcheck.features import build_graph, load_graph
from ptzcheck.solver import fit_families, profile_aspect, MODELS
from ptzcheck.compare import compare_current
from ptzcheck.report import render_report


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    bundle=sub.add_parser('bundle',help='Read the S01 audit ZIP/directory, no production imports')
    bundle.add_argument('--bundle',required=True,type=Path)
    video=sub.add_parser('video',help='One explicitly identified continuous camera shot')
    video.add_argument('--video',required=True,type=Path)
    video.add_argument('--start-pts',required=True,type=int)
    video.add_argument('--end-pts',required=True,type=int)
    video.add_argument('--time-base',required=True,help='Native source time base, e.g. 1/12800')
    video.add_argument('--rig-id',required=True)
    video.add_argument('--shot-id',required=True)
    refit=sub.add_parser('refit',help='Reuse an existing graph; no video decoding or matching')
    refit.add_argument('--graph',required=True,type=Path)
    refit.add_argument('--camera-npz',type=Path,help='Explicit 1ce8ae1 N x 24 camera export at matching native PTS')
    for q in (bundle,video):
        q.add_argument('--stride-seconds',type=float,default=.2)
        q.add_argument('--max-frames',type=int,default=116)
        q.add_argument('--feature-width',type=int,default=1920)
        q.add_argument('--max-matches-per-edge',type=int,default=160)
        q.add_argument('--component',default='all',help='all, largest, or zero-based component index')
        q.add_argument('--masks',type=Path,help='JSON normalized exclude_rects; default S01 score/timer/central-skater masks')
        q.add_argument('--ffmpeg',default='ffmpeg')
        q.add_argument('--ffprobe',default='ffprobe')
    for q in (bundle,video,refit):
        q.add_argument('--out',required=True,type=Path,help='Fresh output directory; overwrite forbidden')
        q.add_argument('--models',nargs='+',choices=sorted(MODELS),default=['square_center','free_pp','free_pp_radial'])
        q.add_argument('--seeds',nargs='+',type=float,default=[1.,3.,8.],help='Initial focal / image-width, no current-camera seeds')
        q.add_argument('--max-nfev',type=int,default=180)
        q.add_argument('--profile-aspect',nargs='*',type=float,default=[])
        q.add_argument('--no-current-comparison',action='store_true')
    return p


def run(args):
    if args.max_nfev < 1 or not args.seeds or any(s<=0 for s in args.seeds): raise ValueError('Invalid solver budget/seeds')
    source=getattr(args,'bundle',None) or getattr(args,'video',None) or getattr(args,'graph',None)
    out=fresh_output(args.out,[source]); hashes={}; camera=None; then=time.perf_counter()
    provenance=dict(version=__version__,argv=sys.argv[1:],python=platform.python_version(),numpy=np.__version__,
        scipy=scipy.__version__,opencv=cv2.__version__,production_solver_imported=False,
        production_state_modified=False,calibration_assumption='static scene, common optical centre, global shutter approximation')
    try:
        if args.command=='refit':
            graph=load_graph(args.graph)
            for name in ('graph.json','matches.npz'):
                hashes[str((args.graph/name).resolve())]=sha256(args.graph/name)
            write_json(out/'graph.json',graph['meta'])
            np.savez_compressed(out/'matches.npz',**graph['data'])
            camera=args.camera_npz
        else:
            if args.command=='bundle':
                video,clock,camera,hashes=prepare_bundle(args.bundle,out)
                start,end=clock['first_source_pts'],clock['last_source_pts']
                provenance.update(rig_id='S01 physical camera only',shot_id='S01',bundle=str(args.bundle.resolve()))
            else:
                video=args.video.resolve(strict=True);clock=None
                hashes[str(video)]=sha256(video);start,end=args.start_pts,args.end_pts
                provenance.update(rig_id=args.rig_id,shot_id=args.shot_id)
            probe=probe_video(video,args.ffprobe)
            if args.command=='video' and Fraction(probe['time_base'])!=Fraction(args.time_base):
                raise ValueError('Requested time base differs from video stream')
            if clock and Fraction(probe['time_base'])!=Fraction(clock['clip_time_base']):
                raise ValueError('Clip time base differs from source clock')
            if clock:
                offset=int(clock.get('source_pts_equals_clip_pts_plus',0));start-=offset;end-=offset
            indices=select_frames(probe,start,end,args.stride_seconds,args.max_frames)
            print(f'Decode {len(indices)} native-PTS frames; no FPS-to-time conversion',flush=True)
            frames=decode_selected(video,probe,indices,out/'frames',clock,args.ffmpeg)
            masks=None
            if args.masks:
                masks=json.loads(args.masks.read_text(encoding='utf-8'))['exclude_rects']
                hashes[str(args.masks.resolve())]=sha256(args.masks)
            write_json(out/'source.json',dict(probe=probe,frames=frames,source_video_sha256=sha256(video)))
            graph=build_graph(frames,probe['width'],probe['height'],out,max_width=args.feature_width,
                masks=masks,max_per_edge=args.max_matches_per_edge,component=args.component)
        provenance['input_hashes_before']=hashes
        write_json(out/'run_config.json',provenance)
        print(f'Fit {len(graph["meta"]["frames"])} frames; independent rotation anchor for every disconnected component.',flush=True)
        fits=fit_families(graph,out,models=args.models,seeds=args.seeds,max_nfev=args.max_nfev)
        # This is the first point at which production camera values are loaded.
        comp=None
        if camera is not None and not args.no_current_comparison:
            comp=compare_current(graph,camera,out,max_nfev=args.max_nfev)
        profile=None
        if args.profile_aspect:
            family='free_pp_radial' if 'free_pp_radial' in fits else 'free_pp'
            if family not in fits: raise ValueError('Aspect profile requires free_pp or free_pp_radial')
            profile=profile_aspect(graph,family,fits[family],args.profile_aspect,out,args.max_nfev)
        assert_unchanged(hashes);provenance['inputs_unchanged']=True
        provenance['wall_seconds']=time.perf_counter()-then
        render_report(graph,fits,out,comp,profile,provenance)
        write_json(out/'run_config.json',provenance)
        print(f'Done: {out/"report.html"}\nNo physical-camera certificate; no production writes.',flush=True)
    except BaseException as e:
        write_json(out/'failure.json',dict(error=type(e).__name__,message=str(e),
            traceback=traceback.format_exc(),production_state_modified=False))
        raise

if __name__=='__main__':
    try: run(parser().parse_args())
    except KeyboardInterrupt: print('Stopped. Existing independent diagnostics retained.',file=sys.stderr);sys.exit(130)
    except Exception as e: print(f'ERROR: {e}',file=sys.stderr);sys.exit(2)
