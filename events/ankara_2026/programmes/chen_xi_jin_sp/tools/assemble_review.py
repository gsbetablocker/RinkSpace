"""Replace a corrected review prefix at a verified AVC IDR boundary."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
from pathlib import Path

import av
import numpy as np

from common import OUTPUT, TB, observation_packet, read, scope, sha, source_pts, write
from measure import clock
from rinkspace.review.render import validate_video

NAME = "chen_xi_jin_ankara_grid_1080p50.mp4"


def is_idr(packet, config):
    if not packet.is_keyframe or len(config)<5 or config[0]!=1:
        return False
    length_bytes=(config[4]&3)+1
    data=bytes(packet)
    i=0
    while i+length_bytes<len(data):
        size=int.from_bytes(data[i:i+length_bytes],"big")
        i+=length_bytes
        if size<=0 or i+size>len(data):
            raise ValueError("Invalid AVC packet framing")
        if data[i]&31==5:
            return True
        i+=size
    return False


def find_boundary(video, at_least, origin):
    with av.open(str(video)) as mux:
        track=mux.streams.video[0]
        if track.time_base!=TB or track.codec_context.name!="h264":
            raise ValueError("Previous review is not AVC with the native clock")
        for item in mux.demux(track):
            if item.pts is not None and item.pts>=at_least-origin and is_idr(item,track.codec_context.extradata):
                return item.pts+origin
    raise ValueError("No verified AVC IDR after the changed source frames")


def tail_packets(mux, track, start):
    at_boundary=False
    for p in mux.demux(track):
        if p.dts is None or p.pts is None:
            continue
        if not at_boundary:
            if p.pts!=start or not is_idr(p,track.codec_context.extradata):
                continue
            at_boundary=True
        if p.pts<start:
            raise ValueError("A copied GOP refers to an earlier display frame")
        yield p


def check_tail_state(record, old, current, boundary):
    with np.load(old/"camera.npz",allow_pickle=False) as a, np.load(current/"camera.npz",allow_pickle=False) as b:
        for shot in record["shots"]:
            sid=shot["id"]
            pts=source_pts(shot)
            if pts[-1]>=boundary:
                for k in ("_pts","_controls","_rig"):
                    np.testing.assert_array_equal(a[sid+k],b[sid+k])
                k=sid+"_exposure_times"
                nominal=clock(pts)
                np.testing.assert_array_equal(a[k] if k in a else nominal,b[k] if k in b else nominal)
    with np.load(old/"active_rows.npz",allow_pickle=False) as a, np.load(current/"active_rows.npz",allow_pickle=False) as b:
        keep_a,keep_b=(x["pts"]>=boundary for x in (a,b))
        optional=tuple(k for k in ("idx","path") if k in a and k in b)
        for key in ("shot","pts","uv",*optional):
            np.testing.assert_array_equal(a[key][keep_a],b[key][keep_b])


def check_tail_observations(record, head, tail, boundary):
    for field in ("source_bound_foreground","drawing_subpixel_bits"):
        if head.get(field)!=tail.get(field):
            raise ValueError(f"Spliced reviews differ in {field}")
    current,previous=(r.get("source_observations") for r in (head,tail))
    if (current is None)!=(previous is None):
        raise ValueError("Spliced reviews use different original point modes")
    if current is None:
        return
    for shot in record["shots"]:
        sid=shot["id"]
        if source_pts(shot)[-1]<boundary:
            continue
        a,b=(observation_packet(Path(p),sid) for p in (current,previous))
        if a.resolve()!=b.resolve() and sha(a)!=sha(b):
            raise ValueError(f"Original points changed in the copied portion of {sid}")


def copied_packet_signature(packet):
    return (int(packet.pts),int(packet.dts),hashlib.sha256(bytes(packet)).digest())


def join_streams(prefix, previous, temp, clock):
    copied=[]
    with av.open(str(prefix)) as head, av.open(str(previous)) as tail, av.open(
            str(temp),"w",options={"movflags":"+faststart","video_track_timescale":"12800"}) as out:
        x,y=head.streams.video[0],tail.streams.video[0]
        if (x.time_base!=TB or y.time_base!=TB or x.codec_context.extradata!=y.codec_context.extradata
                or (x.width,x.height)!=(y.width,y.height)):
            raise ValueError("Replacement AVC stream parameters differ")
        dest=out.add_stream_from_template(x)
        dest.time_base=TB
        last=None
        for packet in head.demux(x):
            if packet.dts is None or packet.pts is None:
                continue
            if not (0<=packet.pts<clock):
                raise ValueError("Replacement prefix displays outside its promised range")
            last=packet.dts
            packet.stream=dest
            out.mux(packet)
        if last is None:
            raise ValueError("Replacement prefix contains no encoded packets")
        for packet in tail_packets(tail,y,clock):
            if packet.dts<=last:
                raise ValueError("DTS overlap at the AVC join boundary")
            last=packet.dts
            copied.append(copied_packet_signature(packet))
            packet.stream=dest
            out.mux(packet)
    with av.open(str(temp)) as combined:
        signatures=[copied_packet_signature(p) for p in tail_packets(combined,combined.streams.video[0],clock)]
    if not copied or signatures!=copied:
        raise ValueError("Encoded tail packets changed during the join")
    return len(copied)


def build_coverage(prefix, previous, candidate, target, boundary, expected):
    entries=[]
    inputs=((prefix,lambda p:p<boundary),(previous,lambda p:p>=boundary))
    for part,select in inputs:
        with Path(part["coverage"]).open(newline="",encoding="utf-8") as stream:
            entries.extend(r for r in csv.DictReader(stream) if select(int(r["source_pts"])))
    np.testing.assert_array_equal([int(r["source_pts"]) for r in entries],expected)
    by=defaultdict(list)
    for row in entries:
        if int(row["visible_vertices"])<1 or int(row["camera_status"])!=0:
            raise ValueError("Review coverage contains an unprojected exposure")
        by[row["shot"]].append(row)
    with np.load(candidate/"active_rows.npz",allow_pickle=False) as a:
        pts,count=np.unique(a["pts"],return_counts=True)
        ledger=dict(zip(pts,count,strict=True))
    np.testing.assert_array_equal([int(r["active_original_uv_count"]) for r in entries],
                                  [ledger.get(p,0) for p in expected])
    with target.open("x",newline="",encoding="utf-8") as out:
        writer=csv.DictWriter(out,fieldnames=entries[0].keys())
        writer.writeheader()
        writer.writerows(entries)
    result={}
    for shot,rows in by.items():
        training=[int(r["active_original_uv_count"]) for r in rows]
        result[shot]=dict(frames=len(rows),visible_grid_frames=len(rows),active_original_uv=sum(training),
             frames_with_uv=sum(v>0 for v in training),cpu_gpu_max_px=max(
                 part["shots"].get(shot,{}).get("cpu_gpu_max_px",0.) for part,_ in inputs),
             native_foreground_mattes=sum(r["athlete_matte_available"]=="True" for r in rows))
        if "shown_original_uv_count" in rows[0]:
            for layer in ("original","ice","board","advertisement","reserved"):
                field=f"shown_{layer}_uv"
                result[shot][field]=sum(int(r[field+"_count"]) for r in rows)
            result[shot]["frames_with_shown_uv"]=sum(int(r["shown_original_uv_count"])>0 for r in rows)
    return result


def assemble(prefix, previous, candidate, folder):
    record=scope()
    pts=np.concatenate([source_pts(s) for s in record["shots"]])
    head,tail=(read(d/"review.json") for d in (prefix,previous))
    boundary=head["stop_before_pts"]
    if any(head[key]!=tail[key] for key in ("source_sha256","geometry_value_sha256","inventory")):
        raise ValueError("The rendered prefix and copied tail use different source or scene")
    prior=Path(tail["candidate"])
    check_tail_state(record,prior,candidate,boundary)
    check_tail_observations(record,head,tail,boundary)
    folder.mkdir(exist_ok=False)
    temp,final=folder/"programme.writing.mp4",folder/NAME
    packets=join_streams(Path(head["path"]),Path(tail["path"]),temp,int(boundary-pts[0]))
    achieved=validate_video(temp,pts,TB)
    receipt=folder/"coverage.csv"
    counts=build_coverage(head,tail,candidate,receipt,boundary,pts)
    temp.replace(final)
    report=dict(path=str(final),sha256=sha(final),bytes=final.stat().st_size,source_clock="1/12800",
         native_frame_count=len(pts),first_pts=int(pts[0]),last_pts=int(pts[-1]),candidate=str(candidate),
         inventory=tail["inventory"],encoded_clock_validation=achieved,coverage=str(receipt),shots=counts,
         source_sha256=tail["source_sha256"],geometry_value_sha256=tail["geometry_value_sha256"],
         drawing_subpixel_bits=head.get("drawing_subpixel_bits"),
         source_bound_foreground=head.get("source_bound_foreground"),
         source_observations=head.get("source_observations"),
         prefix_receipt=str(prefix/"review.json"),copied_tail_receipt=str(previous/"review.json"),
         tail_first_source_pts=boundary,encoded_tail_packets_verified=packets)
    write(folder/"review.json",report)
    print(final,achieved,"copied identical tail packets",packets,flush=True)


def main():
    args=argparse.ArgumentParser(description=__doc__)
    args.add_argument("--previous",required=True)
    args.add_argument("--prefix")
    args.add_argument("--candidate")
    args.add_argument("--out")
    args.add_argument("--find-idr",action="store_true")
    a=args.parse_args()
    previous=OUTPUT/a.previous
    if a.find_idr:
        r=scope()
        print(find_boundary(previous/NAME,r["shots"][1]["processing"]["first_pts"],
                            r["processing_scope"]["first_pts"]))
    else:
        assemble(OUTPUT/a.prefix,previous,OUTPUT/a.candidate,OUTPUT/a.out)


if __name__=="__main__":
    main()
