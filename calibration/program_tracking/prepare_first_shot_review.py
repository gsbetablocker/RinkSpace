"""Prepare remaining official50 first-shot native images and 8a28 navigation.

The source clock is always official native PTS. Legacy cameras only navigate
source feature identity. Existing accepted frames 1--200 are never written.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "calibration/program_tracking")]
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import av
import cv2
import numpy as np
from rinkspace.programme_source_stream import file_sha
from calibration.program_tracking.prepare_continuation_frame_sources import read, timestamp, array_sha

PARENT = ROOT / "events/ankara_2026/programmes/okada_mei_sp/source_datasets/frame_by_frame_review_20261013"
DATA_PARENT = Path("D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp/results/frame_by_frame_review_20261013")
DATA = DATA_PARENT / "first_shot_remaining_50fps"
OUT = PARENT / "first_shot_remaining_50fps"
FIRST_PTS = 256256
PRIORITIES = [201, 250, 300, 400, 500, 600, 800, 1000, 1200, 1248]


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    os.replace(temp,path)


def shot_scope():
    opening = read(PARENT / "opening_100_50fps/navigation_mapping.json")
    ledger = Path(opening["mapping_evidence"]["path"])
    if file_sha(ledger) != opening["mapping_evidence"]["sha256"]:
        raise ValueError("Alignment evidence changed")
    rows = [dict(json.loads(line), line_1based=i+1) for i, line in enumerate(ledger.read_text("utf8").splitlines())]
    first = [r for r in rows if int(r["shot_index"]) == 1]
    second = next(r for r in rows if int(r["shot_index"]) == 2)
    last_pts = int(first[-1]["accepted_official_pts"])
    if int(second["accepted_official_pts"]) != last_pts + 256:
        raise ValueError("First-shot boundary not consecutive")
    last_index = (last_pts-FIRST_PTS)//256+1
    return opening, first, second, np.arange(201, last_index+1, dtype=np.int64)


def decoded(path, start_pts, stop_pts):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 2
        if str(stream.time_base) != "1/12800" or (stream.width, stream.height) != (1920, 1080):
            raise ValueError("Official source clock or dimensions changed")
        container.seek(int(start_pts), stream=stream, backward=True)
        expected = int(start_pts)
        for frame in container.decode(stream):
            if frame.pts is None or frame.pts < start_pts:
                continue
            if frame.pts >= stop_pts:
                break
            if int(frame.pts) != expected:
                raise ValueError(f"Native exposure omitted {frame.pts} != {expected}")
            yield int(frame.pts), frame.to_ndarray(format="bgr24")
            expected += 256
            if expected == stop_pts:
                break
        if expected != stop_pts:
            raise ValueError(f"Incomplete native decode {expected}/{stop_pts}")


def prepare_sources(priority_only=False):
    started = time.perf_counter()
    opening, ledger, second, indices = shot_scope()
    canonical = read(DATA_PARENT / "frame_000200_50fps/source_binding.json")
    source_path = Path(canonical["source"])
    if file_sha(source_path) != canonical["source_video_sha256"]:
        raise ValueError("Official source bytes changed")
    for name in ("frames", "masks_current", "navigation", "source_bindings", "mask_provenance"):
        (DATA / name).mkdir(parents=True, exist_ok=True)
    wanted = sorted(set(PRIORITIES) & set(map(int, indices))) if priority_only else list(map(int, indices))
    exposures = ((i, FIRST_PTS+(i-1)*256, next(decoded(source_path, FIRST_PTS+(i-1)*256, FIRST_PTS+i*256))[1]) for i in wanted) if priority_only else (
        ((pts-FIRST_PTS)//256+1, pts, bgr) for pts, bgr in decoded(source_path, FIRST_PTS+200*256, FIRST_PTS+int(indices[-1])*256))
    rows = []
    for ordinal, (index, pts, bgr) in enumerate(exposures, 1):
        path = DATA / "frames" / f"{index:06d}.png"
        existing = cv2.imread(str(path)) if path.is_file() else None
        if existing is None or not np.array_equal(existing, bgr):
            if not cv2.imwrite(str(path), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                raise OSError(path)
        if not np.array_equal(cv2.imread(str(path)), bgr):
            raise ValueError("PNG roundtrip changed native RGB")
        rgb_sha = array_sha(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        binding = dict(schema="rinkspace.exact-single-native-source-frame.v1", event_id="ankara_2026", programme_id="okada_mei_sp",
            programme_frame_index_1based=index, native_frame_index_1based=pts//256+1, native_frame_index_0based=pts//256,
            native_pts=pts, actual_pts=pts, time_base="1/12800", source_time=timestamp(pts), source_time_seconds=pts/12800,
            source=str(source_path), source_video_sha256=canonical["source_video_sha256"], image_path=str(path), png_sha256=file_sha(path),
            decoded_BGR_sha256=array_sha(bgr), decoded_RGB_sha256=rgb_sha, native_hw=[1080, 1920], original_PTS_verified=True,
            PNG_roundtrip_lossless_verified=True, optimization_performed=False)
        binding_path = DATA / "source_bindings" / f"{index:06d}.json"
        write(binding_path, binding)
        rows.append(dict(index=index, source=binding, image_path=str(path), path=str(path), mask_path=str(DATA/"masks_current"/f"{index:06d}.png"),
            native_pts=pts, source_pts=pts, source_time_base="1/12800", source_time=timestamp(pts), source_time_seconds=pts/12800,
            native_frame_index_0based=pts//256, png_sha256=binding["png_sha256"], decoded_native_BGR_sha256=binding["decoded_BGR_sha256"],
            decoded_native_RGB_sha256=rgb_sha, native_hw=[1080, 1920], exact_native_PTS_and_lossless_PNG_verified=True,
            source_binding=dict(path=str(binding_path), sha256=file_sha(binding_path))))
        if priority_only or ordinal % 100 == 0:
            print(json.dumps(dict(stage="native_source_ready", index=index, completed=ordinal, total=len(wanted), elapsed=time.perf_counter()-started)), flush=True)
    manifest = dict(schema="rinkspace.exact-native-consecutive-source-frames.v1", event_id="ankara_2026", programme_id="okada_mei_sp",
        source=read(DATA_PARENT/"opening_100_50fps/frame_manifest.json")["source"],
        source_selection_reference=read(DATA_PARENT/"opening_100_50fps/frame_manifest.json")["source_selection_reference"],
        programme_indices=wanted, start_pts=FIRST_PTS+(wanted[0]-1)*256, last_pts=FIRST_PTS+(wanted[-1]-1)*256,
        stop_pts_exclusive=FIRST_PTS+wanted[-1]*256, native_pts_step=256, frame_count=len(rows), original_PTS_order_verified=True,
        native_exposures_all_decoded=not priority_only, PNG_roundtrip_lossless_verified=True, first200_modified=False,
        source_points_extracted=False, optimization_performed=False, legacy60_source_only_navigation=True,
        first_shot=dict(first_programme_index=1, last_programme_index=int(indices[-1]), start_official_pts=FIRST_PTS,
            last_official_pts=int(ledger[-1]["accepted_official_pts"]), next_shot_official_pts=int(second["accepted_official_pts"]),
            boundary_authority="accepted unique pixel matches at adjacent official exposures; current source boundary contact sheet"),
        partial_priority_preparation=priority_only,
        decoder=dict(library="PyAV", version=av.__version__, cpu_threads=2, temporal_resampling=False, spatial_resizing_of_source_PNGs=False),
        frames=rows, source_prepare_elapsed_seconds=time.perf_counter()-started)
    write(DATA/"frame_manifest.json", manifest)
    tiles=[]
    for index in PRIORITIES:
        path=DATA/"frames"/f"{index:06d}.png"
        if not path.is_file(): continue
        tile=cv2.resize(cv2.imread(str(path)),(384,216));cv2.putText(tile,f"{index} {timestamp(FIRST_PTS+(index-1)*256)}",(4,17),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1);tiles.append(tile)
    sheet=np.zeros((216*((len(tiles)+4)//5),1920,3),np.uint8)
    for i,tile in enumerate(tiles): y,x=divmod(i,5);sheet[y*216:(y+1)*216,x*384:(x+1)*384]=tile
    cv2.imwrite(str(DATA/"source_contact_sheet.png"),sheet)
    cut_tiles=[]
    for index in (200, 201, int(indices[-1])-1, int(indices[-1]), int(indices[-1])+1, int(indices[-1])+2):
        pts=FIRST_PTS+(index-1)*256;bgr=next(decoded(source_path,pts,pts+256))[1]
        tile=cv2.resize(bgr,(480,270));cv2.putText(tile,f"Frame {index} PTS {pts}",(4,20),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),1);cut_tiles.append(tile)
    cv2.imwrite(str(DATA/"first_shot_cut_verification.png"),np.vstack([np.hstack(cut_tiles[:3]),np.hstack(cut_tiles[3:])]))
    print(json.dumps(dict(stage="sources_ready", frames=len(rows), partial=priority_only, first_shot_end=int(indices[-1]), manifest=str(DATA/"frame_manifest.json"))),flush=True)
    return manifest


def mapping():
    opening, ledger_rows, second, indices = shot_scope()
    accepted={}
    for row in ledger_rows:
        stamp=row.get("accepted_official_pts")
        if stamp is not None and row["match_status"]=="unique_pixel_match": accepted.setdefault(int(stamp),[]).append(row)
    anchor_pts=np.array(sorted(accepted),np.int64)
    archive=Path(opening["model_seed"]["inherited_camera_archive"]["path"])
    if file_sha(archive)!=opening["model_seed"]["inherited_camera_archive"]["sha256"]: raise ValueError("Inherited cameras changed")
    with np.load(archive,allow_pickle=False) as saved: cameras={k:saved[k].copy() for k in saved.files}
    lookup={int(p):i for i,p in enumerate(cameras["pts"])}
    frames=[]; archive_indices=[];pts_array=FIRST_PTS+(indices-1)*256
    for ordinal,(index,pts) in enumerate(zip(indices,pts_array)):
        exact=accepted.get(int(pts),[]);candidate=[r for r in ledger_rows if r.get("best_official_pts_candidate")==int(pts)];neighbors=None;predicted=None
        if exact: row=min(exact,key=lambda r:r["full_pts"]);status="confirmed_pixel_match"
        else:
            prior=anchor_pts[anchor_pts<pts];later=anchor_pts[anchor_pts>pts]
            if not len(prior) or not len(later): raise ValueError("Missing same-shot accepted mapping neighbors")
            left,right=int(prior[-1]),int(later[0]);lp=min(accepted[left],key=lambda r:r["full_pts"])["full_pts"];rp=min(accepted[right],key=lambda r:r["full_pts"])["full_pts"]
            predicted=float(lp+(int(pts)-left)/(right-left)*(rp-lp));neighbors=[dict(official_pts=left,legacy60_pts=lp),dict(official_pts=right,legacy60_pts=rp)]
            if not candidate: candidate=[r for r in ledger_rows if lp<=r["full_pts"]<=rp];status="inferred_navigation_from_accepted_neighbor_bracket_missing_exact_pixel_candidate"
            else: status="inferred_navigation_from_unaccepted_pixel_candidate_with_accepted_neighbors"
            row=min(candidate,key=lambda r:(abs(r["full_pts"]-predicted),r["full_pts"]))
        ci=lookup[int(row["full_pts"])];archive_indices.append(ci)
        frames.append(dict(sequence_index_1based=int(index),programme_frame_index_1based=int(index),official_pts=int(pts),official_time_base="1/12800",official_time_seconds=int(pts)/12800,
            official_frame_index_0based=int(pts)//256,legacy60_pts=int(row["full_pts"]),legacy60_time_base="1/15360",legacy60_time_seconds=int(row["full_pts"])/15360,legacy60_shot_index=1,
            mapping_status=status,pixel_correspondence_confirmed=bool(exact),navigation_only=True,
            ledger_evidence={k:row.get(k) for k in ("line_1based","match_status","accepted_official_pts","best_official_pts_candidate","best_roi_gray_mae","unique_margin_mae")},
            accepted_neighbor_anchors=neighbors,neighbor_interpolated_legacy60_pts_for_candidate_selection=predicted,camera_seed_index_0based=ordinal,
            inherited_camera_archive_row_0based=ci,camera_exposure_representative_legacy60_pts=int(cameras["exposure_representative_pts"][ci]),camera25_float64_sha256=array_sha(cameras["camera25"][ci])))
    ai=np.array(archive_indices,np.int64)
    DATA.mkdir(parents=True,exist_ok=True)
    np.savez(DATA/"camera_seeds.npz",programme_indices=indices,pts=pts_array,legacy60_pts=np.array([r["legacy60_pts"] for r in frames],np.int64),
        inherited_camera_archive_indices=ai,camera25=cameras["camera25"][ai],installation_id=cameras["installation_id"][ai],view_state4=cameras["view_state4"][ai],exposure_representative_pts=cameras["exposure_representative_pts"][ai])
    result=deepcopy(opening);result.pop("legacy_foreground_cache",None)
    result.pop("official_representation_verification",None)
    result.update(schema="rinkspace.official50-first-shot-navigation-mapping.v1",purpose="201–1248 official source frames; inherited 8a28 cameras for navigation only; no fitting or source clock substitution.",
        scope=dict(first_programme_index=201,last_programme_index=int(indices[-1]),frame_count=len(frames),first_official_pts=int(pts_array[0]),last_official_pts=int(pts_array[-1]),end_official_pts_exclusive=int(pts_array[-1]+256)),frames=frames,
        summary={s:sum(r["mapping_status"]==s for r in frames) for s in sorted({r["mapping_status"] for r in frames})},first200_modified=False)
    result["model_seed"]["camera_seeds"]=dict(path=str(DATA/"camera_seeds.npz"),sha256=file_sha(DATA/"camera_seeds.npz"));result["model_seed"]["camera25_shape"]=[len(indices),25]
    result["mapping_policy"]["missing_exact_candidate"]="Closest existing same-shot ledger row in accepted neighbor bracket; navigation inference only."
    write(OUT/"navigation_mapping.json",result)
    print(json.dumps(dict(stage="mapping_ready",summary=result["summary"],camera_shape=[len(indices),25])),flush=True)
    return result


def verify_official_representation():
    """Measure exact-P TS pixel equality over the entire newly prepared range."""
    started=time.perf_counter();manifest=read(DATA/"frame_manifest.json")
    if manifest.get("partial_priority_preparation") or manifest["frame_count"]!=1048:
        raise ValueError("Official representation verification requires all1048 current source exposures")
    canonical=read(DATA_PARENT/"frame_000200_50fps/source_binding.json")
    representation=canonical["ledger_official_representation"];path=Path(representation["path"])
    if file_sha(path)!=representation["sha256"]:raise ValueError("Ledger official representation bytes changed")
    by={r["native_pts"]:r for r in manifest["frames"]};records=[]
    for pts,bgr in decoded(path,manifest["start_pts"],manifest["stop_pts_exclusive"]):
        row=by[pts];actual_sha=array_sha(bgr)
        if actual_sha!=row["decoded_native_BGR_sha256"]:
            raise ValueError(f"Current local official versus ledger-bound official exposure differs at programme frame {row['index']} PTS{pts}")
        if file_sha(row["image_path"])!=row["png_sha256"]:raise ValueError("Current native source PNG changed before representation verification")
        records.append(dict(programme_frame_index_1based=row["index"],native_pts=pts,current_source_decoded_BGR_sha256=row["decoded_native_BGR_sha256"],
            ledger_official_representation_decoded_BGR_sha256=actual_sha,decoded_pixels_identical=True))
        if len(records)%200==0:print(json.dumps(dict(stage="official_representation_compared",completed=len(records))),flush=True)
    if len(records)!=1048:raise ValueError("Incomplete official representation comparison")
    result=dict(schema="rinkspace.exact-current-vs-ledger-official-representation-verification.v1",scope="Consecutive official50 programme frames201..1248 nativePTS307456..575488 inclusive",
        frame_count=len(records),first_programme_index=201,last_programme_index=1248,start_pts=307456,last_pts=575488,time_base="1/12800",native_hw=[1080,1920],
        current_primary_source=dict(path=manifest["source"]["path"],sha256=manifest["source"]["sha256"]),ledger_official_representation=representation,
        checked_via="PyAV exact-PTS BGR of ledger-bound official copy compared framewise with current native source decoded_BGR SHA256; PNG SHA independently rechecked",
        all_1048_decoded_pixels_identical=True,whole_video_equality_claimed=False,records=records,
        frame_manifest=dict(path=str(DATA/"frame_manifest.json"),sha256=file_sha(DATA/"frame_manifest.json")),elapsed_seconds=time.perf_counter()-started)
    verification=DATA/"official_representation_verification.json";write(verification,result)
    nav=read(OUT/"navigation_mapping.json")
    nav["official_representation_verification"]={k:v for k,v in result.items() if k!="records"}
    nav["official_representation_verification"]["per_frame_verification"]=dict(path=str(verification),sha256=file_sha(verification))
    for row in nav["frames"]:row["primary_official50_vs_ledger_official50_BGR_identical"]=True
    write(OUT/"navigation_mapping.json",nav)
    print(json.dumps(dict(stage="official_representation_verified",frames=len(records),all_native_BGR_identical=True,elapsed=result["elapsed_seconds"])),flush=True)
    return result


def navigation(indices=None, permit_missing_mask=False):
    from calibration.program_tracking.render_result import SceneRenderer
    mapping_value=read(OUT/"navigation_mapping.json");scene_record=mapping_value["model_seed"]["scene"]
    if file_sha(scene_record["path"])!=scene_record["sha256"]: raise ValueError("8a28 scene changed")
    renderer=SceneRenderer(read(scene_record["path"]),draw_grid=True,draw_advertisements=True)
    with np.load(DATA/"camera_seeds.npz",allow_pickle=False) as saved: cameras=saved["camera25"].copy()
    selected=None if indices is None else set(indices);count=0
    for row in mapping_value["frames"]:
        index=row["programme_frame_index_1based"]
        if selected is not None and index not in selected: continue
        source_path=DATA/"frames"/f"{index:06d}.png";mask_path=DATA/"masks_current"/f"{index:06d}.png"
        if not source_path.is_file() or (not mask_path.is_file() and not permit_missing_mask): continue
        raw=cv2.imread(str(source_path));mask=cv2.imread(str(mask_path),0)>0 if mask_path.is_file() else np.zeros(raw.shape[:2],bool)
        rendered,metrics=renderer.draw(raw,cameras[row["camera_seed_index_0based"]],person_mask=mask,title=None)
        if not np.array_equal(rendered[mask],raw[mask]): raise ValueError("Navigation overwrote person")
        path=DATA/"navigation"/f"{index:06d}.png";cv2.imwrite(str(path),rendered,[cv2.IMWRITE_PNG_COMPRESSION,1]);count+=1
        row["navigation_overlay"]=dict(path=str(path),sha256=file_sha(path),navigation_only=True,current_primary_official50_background=True,
            current_source_mask_applied=mask_path.is_file(),current_mask_sha256=file_sha(mask_path) if mask_path.is_file() else None,camera_seed_refitted=False,
            all_original_feature_paths_and_advertisement_ids_enabled=True,all_mask_pixels_preserved_from_source=True,body_or_trajectory_drawn=False,path_counts=renderer.path_counts)
        if count%100==0: print(json.dumps(dict(stage="navigation_rendered",completed=count,index=index)),flush=True)
    overlays=[r["navigation_overlay"] for r in mapping_value["frames"] if "navigation_overlay" in r]
    provenance=DATA/"masks_current/current_mask_manifest.json"
    mapping_value["navigation_rendering"]=dict(interface="calibration.program_tracking.render_result.SceneRenderer",
        inventory_path_counts=renderer.path_counts,advertisement_ids=[int(ad["user_number"]) for ad in read(scene_record["path"]).get("advertisements",[])],
        output_directory=str(DATA/"navigation"),frame_count=len(overlays),point_extraction_performed=False,optimizer_run=False,
        body_or_trajectory_drawn=False,status="8a28_navigation_seed_overlay_not_fitted_to_current_source_dataset",
        all_png_roundtrips_verified=True,all_current_mask_pixels_preserved_from_source=all(r["all_mask_pixels_preserved_from_source"] for r in overlays),
        occlusion_provenance=dict(current_mask_manifest=dict(path=str(provenance),sha256=file_sha(provenance)) if provenance.is_file() else None,
            current_original_exposures=len(overlays),old_navigation_masks_used=False,prior_frame_memory_used=False,current_source_inference_per_frame=True,
            current_source_conservative_exclusion_retained=True,additional_mask_dilation_or_resize=False,masks_are_exclusion_candidates_not_pixel_ground_truth=True))
    write(OUT/"navigation_mapping.json",mapping_value)
    print(json.dumps(dict(stage="navigation_ready",count=count)),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=["priority","sources","mapping","navigation","representation"],default="priority")
    parser.add_argument("--indices",type=int,nargs="*")
    parser.add_argument("--permit-missing-mask",action="store_true")
    args=parser.parse_args()
    if args.stage=="priority": mapping();prepare_sources(priority_only=True);navigation(PRIORITIES,permit_missing_mask=True)
    elif args.stage=="sources": prepare_sources();verify_official_representation()
    elif args.stage=="mapping": mapping()
    elif args.stage=="representation":verify_official_representation()
    else: navigation(args.indices,permit_missing_mask=args.permit_missing_mask)
