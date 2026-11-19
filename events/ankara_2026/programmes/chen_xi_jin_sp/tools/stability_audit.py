"""Compare exact camera projection with stationary source movement at 50 Hz."""
from __future__ import annotations

import argparse
import importlib.util
from types import SimpleNamespace
from time import monotonic

import numpy as np

from common import JOINT, OUTPUT, frozen_scene, observation_packet, read, save, scope, source_pts, write
from measure import DT, FIRST, load_camera, motion_clock
from stability_tracks import ji

_FILE=JOINT/"stability_1113/measure.py"
_SPEC=importlib.util.spec_from_file_location("rinkspace_stability_measure",_FILE)
temporal=importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(temporal)


def load_tracks(path, record, shot, reuse=()):
    with np.load(path,allow_pickle=False) as cache:
        data={k:cache[k] for k in cache.files}
    np.testing.assert_array_equal(data["pts"],source_pts(shot))
    assert data["source_sha256"].item()==record["source"]["sha256"]
    assert data["geometry_sha256"].item()==record["geometry_binding"]["value_sha256"]
    def clearly_off_graphics(uv):
        x,y=uv.T
        return ((y>158.)|(x>590.))&((y>292.)|(x>451.))&((y<928.)|(x<1618.))
    if bool(data.get("graphics_mask_per_exposure",False)):
        # These tracklets already kept their full patch off the detected
        # broadcast overlay at the anchor and every tracked exposure.
        data["usable"]=np.ones(len(data["uv"]),bool)
    else:
        data["usable"]=clearly_off_graphics(data["uv"])&clearly_off_graphics(data["left_uv"])
        data["usable"]&=clearly_off_graphics(data["anchor_uv"])
    if reuse:
        right=data["pts"][data["index"]]
        beginning=data["pts"][data["start"]]
        for lo,hi in reuse:
            touches=(right>=lo)&(beginning<=hi)
            data["role"][touches]=0
    return data


def predict(native, shot, data, camera):
    pts,c,rig=load_camera(shot,camera)
    d=SimpleNamespace(times=motion_clock(shot,camera,pts),first=FIRST,dt=DT)
    rigs=np.broadcast_to(rig,(len(pts),len(rig)))
    return temporal.predictions(native.core,native,d,c,rigs,data)


def paint_normals(data,path):
    normal=np.full_like(data["uv"],np.nan,dtype=float)
    with np.load(path,allow_pickle=False) as z:
        vector=z["normal"]
        length=np.linalg.norm(vector,axis=1)
        good=z["eligible"]&(length>.7)&~np.char.startswith(z["family"],"Ad")&~np.char.startswith(z["family"],"board.")
        points=z["uv"][good]
        directions=vector[good]/length[good,None]
        idx=z["idx"][good]
    ice=np.flatnonzero(data["family"]==1)
    for i in np.intersect1d(idx,data["index"][ice]):
        k=ice[data["index"][ice]==i]
        use=idx==i
        d=points[use][None,:,:]-data["uv"][k,None,:]
        d2=np.einsum("mni,mni->mn",d,d)
        closest=np.argmin(d2,axis=1)
        ok=d2[np.arange(len(k)),closest]<16.**2
        normal[k[ok]]=directions[use][closest[ok]]
    return normal


def metrics(data,result,selection,normal=None):
    ii=data["index"]
    eligible=selection&data["usable"]&(data["quality"]<17.)
    series={}
    for family,material in enumerate(("board","ice")):
        for suffix,lag,quantity in (("adjacent",1,"displacement"),("lag8",8,"displacement"),("jitter",1,"jitter")):
            choose=eligible&(data["family"]==family)&(data["lag"]==lag)
            e=result[quantity]
            frame=temporal.frame_errors(ii,np.linalg.norm(e,axis=1),choose,len(data["pts"]),
                                         minimum=6 if material=="board" else 4)
            series[material+"_"+suffix]=frame
            if material=="ice" and normal is not None:
                err=np.abs(np.einsum("ni,ni->n",e,normal))
                series["paint_normal_"+suffix]=temporal.frame_errors(ii,err,choose,len(data["pts"]),minimum=3)
    return series


def summarize(pts, series):
    summary={}
    for key,values in series.items():
        aggregate=temporal.stat(values)
        indices=np.flatnonzero(np.isfinite(values))
        largest=indices[np.argsort(-values[indices])[:6]]
        aggregate["worst_pts"]=[int(pts[k]) for k in largest]
        summary[key]=aggregate
    return summary


def audit(record, camera, tracks, folder, *, previous_cache=None, held=False, shots=None, paint=None,reuse=None):
    native=frozen_scene(record,verify=True)
    folder.mkdir(exist_ok=True)
    reports={}
    aggregate={}
    for shot in record["shots"]:
        sid=shot["id"]
        if shots is not None and sid not in shots:
            continue
        started=monotonic()
        data=load_tracks(observation_packet(tracks,sid),record,shot,reuse.get(sid,[]) if reuse else ())
        normal=paint_normals(data,observation_packet(paint,sid)) if paint is not None else None
        output=predict(native,shot,data,camera)
        spatial="role" in data
        role=data["role"] if spatial else data["fold"]
        selections={"fit_and_development": role==0 if spatial else role!=3}
        if held:
            selections["reserved"]=role==3
            selections["all_source"]=np.ones(len(data["index"]),bool)
        prev=None
        if previous_cache is not None:
            with np.load(previous_cache/f"{sid}.npz",allow_pickle=False) as past:
                prev={k:past[k] for k in ("displacement","jitter")}
        report=dict(source_frame_count=len(data["pts"]),raw_rows=len(data["index"]),
                    invalid_native_projection_rows=output["failures"],groups={})
        all_series={}
        for group,mask in selections.items():
            values=metrics(data,output,mask,normal)
            report["groups"][group]={"current":summarize(data["pts"],values)}
            all_series.update({group+"_"+k:v for k,v in values.items()})
            aggregate.setdefault(group,{}).setdefault("current",[]).append(values)
            if prev is not None:
                before=metrics(data,prev,mask,normal)
                report["groups"][group]["baseline"]=summarize(data["pts"],before)
                aggregate[group].setdefault("baseline",[]).append(before)
        cached={k:v for k,v in output.items() if isinstance(v,np.ndarray)}
        save(folder/f"{sid}.npz",pts=data["pts"],**cached,**all_series)
        report["seconds"]=monotonic()-started
        reports[sid]=report
        current=report["groups"]["fit_and_development"]["current"]
        print(sid,"raw visible board",[(key,current[key].get("p95"),current[key]["frames"])
              for key in ("board_adjacent","board_jitter","board_lag8")],
              "seconds",round(report["seconds"],2),flush=True)
    overall={group:{kind:{key:temporal.stat(np.concatenate([x[key] for x in sets]))
                              for key in sets[0]} for kind,sets in cand.items()}
             for group,cand in aggregate.items()}
    doc=dict(camera=str(camera),source_tracks=str(tracks),
             baseline_cache=str(previous_cache) if previous_cache else None,
             source_frame_units="1920x1080 native pixels, one median error per supported exposure",
             definitions=dict(adjacent="20 ms model displacement minus static source displacement",
                 jitter="model three-frame second difference minus source second difference",
                 lag8="160 ms model displacement minus static source displacement"),
             roles=("stationary surface strips reserved with spatial gaps around them; tracklets cross frame blocks"
                    if spatial else "source tracklets confined to a 28-frame block; phase 3 reserved"),
             paint_normal_source=str(paint) if paint is not None else None,
             reserved_regions_reused_for_repairs=reuse or {},
             overall=overall,shots=reports)
    write(folder/"motion.json",doc)
    return doc


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--camera",required=True)
    p.add_argument("--tracks",default="steady_tracks")
    p.add_argument("--out",required=True)
    p.add_argument("--baseline-cache")
    p.add_argument("--held",action="store_true")
    p.add_argument("--shots",nargs="+")
    p.add_argument("--paint-observations")
    p.add_argument("--reused-windows")
    a=p.parse_args()
    audit(scope(),OUTPUT/a.camera,OUTPUT/a.tracks,OUTPUT/a.out,
          previous_cache=OUTPUT/a.baseline_cache if a.baseline_cache else None,held=a.held,shots=a.shots,
          paint=OUTPUT/a.paint_observations if a.paint_observations else None,
          reuse=read(a.reused_windows) if a.reused_windows else None)


if __name__=="__main__":
    main()
