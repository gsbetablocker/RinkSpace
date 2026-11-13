"""Produce the exact-source local HTD comparison video and verify its PTS."""
from argparse import ArgumentParser
from fractions import Fraction
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np

from experiments.skate_video_v1.body_skate_fit import Assembly
from experiments.skate_video_v1.deliver_body_skate_fit import bounds, display_mesh, panel_crop, projection_batch, put, tint
from experiments.skate_video_v1.inputs import PROGRAMME, PROJECT, atomic_json, digest, emit, read
from experiments.skate_video_v1.programme_motion_followup import bind
from rinkspace.review.render import validate_video
from rinkspace.viewer3d.source import SourceVideo

sys.path.insert(0,str(PROJECT/"experiments/skate_video_v1"))
from render_review import render
sys.path.pop(0)


def frames_video(source, cam, refs, shape, pts, rows, clock):
    mesh,tri=display_mesh(refs[0].binding.surface)
    assembly=[Assembly(r.binding) for r in refs]
    desc=("Source","Current fit","PVA 2D only","PVA 2D + relative V/A")
    problem=SimpleNamespace(times=clock,camera=cam)
    for sub in np.array_split(rows,max(1,int(np.ceil(len(rows)/12)))):
        result=[projection_batch(problem,r,sub,a,mesh) for r,a in zip(refs,assembly,strict=True)]
        for i,f in enumerate(sub):
            stamp=int(pts[f])
            raw=cv2.cvtColor(source.read(stamp),cv2.COLOR_RGB2BGR)
            over=[raw]
            for (b,s),a in zip(result,assembly,strict=True):
                frame=raw.copy()
                tint(frame,b[i],tri,(132,228,244),.22)
                for j,color in enumerate(((105,238,129),(234,182,83))):
                    tint(frame,s[i,j],a.faces,color,.30)
                over.append(frame)
            box=bounds([b[i] for b,s in result]+[s[i,j] for b,s in result for j in range(2)],29,raw.shape)
            canvas=np.full((shape[1],shape[0],3),18,np.uint8)
            width=shape[0]//len(over)
            for j,(name,image) in enumerate(zip(desc,over,strict=True)):
                start=j*width
                view=canvas[46:-47,start:start+width-3]
                panel_crop(view,image,box)
                put(canvas,name,start+16,30,.72)
            sec=stamp/12800
            clock_label=f"{int(sec//60):02}:{sec%60:06.3f}"
            put(canvas,f"S05  |  Source {clock_label}  |  PTS {stamp}",16,shape[1]-16,.65)
            put(canvas,"Ankle-to-skate model and source camera shared",shape[0]-550,shape[1]-16,.56)
            yield canvas


def main(args):
    cv2.setNumThreads(2)
    files=[args.current,args.entry/"pva_2d.npz",args.entry/"pva_relative_va.npz"]
    inputs=[bind(args.prepared,args.evidence,p) for p in files]
    refs=[a[0] for a in inputs]
    ref,cam,info,_,_=inputs[0]
    pts=ref.p["pts"]
    for r,c,i,_,_ in inputs[1:]:
        assert i["state_sha256"]==info["state_sha256"]
        np.testing.assert_array_equal(r.p["pts"],pts)
        np.testing.assert_array_equal(r.absolute_curve.position_controls,ref.absolute_curve.position_controls)
        np.testing.assert_array_equal(r.binding.x,ref.binding.x)
    tb=info["source_time_base"]
    dt=float(Fraction(*tb))
    times=pts*dt
    rows=np.flatnonzero((times>=80.90-1e-8)&(times<=81.90+1e-8))
    clock=ref.times[:,None]+cam.readout_times(np.arange(len(pts)))[:,None]*np.linspace(-.5,.5,5)
    video=read(PROGRAMME/"rebuild_scope.json")["source"]
    args.output.mkdir(exist_ok=True,parents=True)
    segments=dict(pts=pts[rows],times=times[rows],tb=tb,media=video["registered_path"])
    source=SourceVideo(video["registered_path"],tb)
    shape=(2400,720)
    name="s05_flying_entry_htd_review.mp4"
    try:
        render(SimpleNamespace(**segments),args.output,name,frames_video(source,cam,refs,shape,pts,rows,clock),shape)
    finally:
        source.close()
    target=args.output/name
    receipt=validate_video(target,pts[rows],Fraction(*tb))
    atomic_json(target.with_suffix(".json"),dict(path=str(target),file_sha256=digest(target),
         source_seconds=times[rows[[0,-1]]].tolist(),source_pts=pts[rows[[0,-1]]].tolist(),source_time_base=tb,
         source_sha256=video["sha256"],camera_sha256=info["state_sha256"],
         parameter_shas=[digest(p) for p in files],**receipt))
    emit("entry_video_verified",**receipt,source_seconds=times[rows[[0,-1]]].tolist())


if __name__=="__main__":
    cli=ArgumentParser(description=__doc__)
    for k in ("prepared","evidence","current","entry","output"):
        cli.add_argument("--"+k,type=Path,required=True)
    main(cli.parse_args())
