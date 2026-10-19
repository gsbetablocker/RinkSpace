"""Prepare exact official50 continuation frames and inherited navigation.

All saved source PNGs use their actual decoded native PTS. Legacy correspondence
only chooses navigation cameras; this module does not measure or optimize rink
points. Current-frame SAM2 exclusion is prepared by the companion module.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "calibration/program_tracking")]
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import av
import cv2
import numpy as np
from rinkspace.programme_source_stream import file_sha

PARENT = ROOT / "events/ankara_2026/programmes/okada_mei_sp/source_datasets/frame_by_frame_review_20261013"
DATA_PARENT = Path("D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp/results/frame_by_frame_review_20261013")
DATA = DATA_PARENT / "continuation_101_200_50fps"
OUT = PARENT / "continuation_101_200_50fps"
INDICES = np.arange(101, 201, dtype=np.int64)
PTS = 256256 + (INDICES - 1) * 256

def read(path): return json.loads(Path(path).read_text(encoding="utf8"))
def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
def array_sha(value): return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
def timestamp(pts):
    ms = int(pts) * 1000 // 12800
    h, ms = divmod(ms, 3600000); m, ms = divmod(ms, 60000); s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def exact_frames(path):
    ordinal = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        if str(stream.time_base) != "1/12800" or (stream.width, stream.height) != (1920, 1080):
            raise ValueError("Official source clock or dimensions changed")
        container.seek(int(PTS[0]), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None or frame.pts < int(PTS[0]): continue
            if ordinal == len(PTS): break
            if int(frame.pts) != int(PTS[ordinal]): raise ValueError("Requested native exposure omitted")
            bgr = frame.to_ndarray(format="bgr24")
            if bgr.shape != (1080, 1920, 3): raise ValueError("Native source dimensions changed")
            yield int(INDICES[ordinal]), int(frame.pts), bgr
            ordinal += 1
            if ordinal == len(PTS): break
    if ordinal != 100: raise ValueError(f"Incomplete source decode {ordinal}/100")

def source_rows():
    started = time.perf_counter()
    canonical = read(DATA_PARENT / "frame_000200_50fps/source_binding.json")
    opening = read(DATA_PARENT / "opening_100_50fps/frame_manifest.json")
    source_path = Path(canonical["source"])
    if file_sha(source_path) != canonical["source_video_sha256"]: raise ValueError("Official source bytes changed")
    for name in ("frames", "masks_current", "navigation", "source_bindings", "mask_provenance"):
        (DATA / name).mkdir(parents=True, exist_ok=True)
    rows = []
    for index, pts, bgr in exact_frames(source_path):
        path = DATA / "frames" / f"{index:06d}.png"
        if index == 200:
            if array_sha(bgr) != canonical["decoded_BGR_sha256"]: raise ValueError("Canonical200 decoded pixels differ")
            shutil.copyfile(canonical["image_path"], path)
            shutil.copyfile(DATA_PARENT / "frame_000200_50fps/person_mask.png", DATA / "masks_current/000200.png")
        elif not cv2.imwrite(str(path), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1]): raise OSError(path)
        if not np.array_equal(cv2.imread(str(path)), bgr): raise ValueError("PNG roundtrip changed native RGB")
        rgb_sha = array_sha(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        binding = dict(schema="rinkspace.exact-single-native-source-frame.v1", event_id="ankara_2026",
            programme_id="okada_mei_sp", programme_frame_index_1based=index,
            native_frame_index_1based=pts//256+1, native_frame_index_0based=pts//256,
            native_pts=pts, actual_pts=pts, time_base="1/12800", source_time=timestamp(pts),
            source_time_seconds=pts/12800, source=str(source_path),
            source_video_sha256=canonical["source_video_sha256"], image_path=str(path),
            png_sha256=file_sha(path), decoded_BGR_sha256=array_sha(bgr), decoded_RGB_sha256=rgb_sha,
            native_hw=[1080,1920], original_PTS_verified=True, PNG_roundtrip_lossless_verified=True,
            optimization_performed=False)
        binding_path = DATA / "source_bindings" / f"{index:06d}.json"
        write(binding_path, binding)
        rows.append(dict(index=index, source=binding, image_path=str(path), path=str(path),
            mask_path=str(DATA / "masks_current" / f"{index:06d}.png"),
            native_pts=pts, source_pts=pts, source_time_base="1/12800", source_time=timestamp(pts),
            source_time_seconds=pts/12800, native_frame_index_0based=pts//256,
            png_sha256=binding["png_sha256"], decoded_native_BGR_sha256=binding["decoded_BGR_sha256"],
            decoded_native_RGB_sha256=rgb_sha, native_hw=[1080,1920],
            exact_native_PTS_and_lossless_PNG_verified=True,
            source_binding=dict(path=str(binding_path),sha256=file_sha(binding_path))))
    representation = Path(canonical["ledger_official_representation"]["path"])
    if file_sha(representation) != canonical["ledger_official_representation"]["sha256"]:
        raise ValueError("Ledger official representation bytes changed")
    for index, pts, bgr in exact_frames(representation):
        if array_sha(bgr) != rows[index-101]["decoded_native_BGR_sha256"]:
            raise ValueError("Primary versus ledger official frame bytes differ")
        rows[index-101]["ledger_official_representation_same_frame_BGR_identical"] = True
        rows[index-101]["source"]["ledger_official_representation_same_frame_BGR_identical"] = True
    manifest = dict(schema="rinkspace.exact-native-consecutive-source-frames.v1", event_id="ankara_2026",
        programme_id="okada_mei_sp", source=opening["source"], source_selection_reference=opening["source_selection_reference"],
        programme_indices=INDICES.tolist(), start_pts=int(PTS[0]),last_pts=int(PTS[-1]),stop_pts_exclusive=int(PTS[-1]+256),
        native_pts_step=256, frame_count=100, original_PTS_order_verified=True,
        native_exposures_all_decoded=True, PNG_roundtrip_lossless_verified=True,
        canonical200_reused_unchanged=True, first100_modified=False, processing201_performed=False,
        source_points_extracted=False, optimization_performed=False,
        decoder=dict(library="PyAV",version=av.__version__,cpu_threads=1,temporal_resampling=False,
            spatial_resizing_of_source_PNGs=False,consecutive_decode=True), frames=rows,
        source_prepare_elapsed_seconds=time.perf_counter()-started)
    write(DATA / "frame_manifest.json",manifest)
    sample=[101,106,111,117,122,127,132,137,143,148,153,158,164,169,174,179,184,190,195,200]
    tiles=[]
    for index in sample:
        tile=cv2.resize(cv2.imread(str(DATA / "frames" / f"{index:06d}.png")),(384,216))
        cv2.putText(tile,f"{index}  {timestamp(int(PTS[index-101]))}",(4,17),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1)
        tiles.append(tile)
    sheet=np.concatenate([np.concatenate(tiles[i:i+5],axis=1)for i in range(0,20,5)],axis=0)
    cv2.imwrite(str(DATA / "source_contact_sheet.png"),sheet)
    print(json.dumps(dict(stage="sources_ready",frames=100,manifest=str(DATA/"frame_manifest.json"),elapsed=manifest["source_prepare_elapsed_seconds"])),flush=True)
    return manifest

def mapping():
    opening=read(PARENT/"opening_100_50fps/navigation_mapping.json")
    ledger=Path(opening["mapping_evidence"]["path"])
    if file_sha(ledger)!=opening["mapping_evidence"]["sha256"]:raise ValueError("Alignment evidence changed")
    ledger_rows=[dict(json.loads(line),line_1based=i+1) for i,line in enumerate(ledger.read_text(encoding="utf8").splitlines())]
    ledger_rows=[r for r in ledger_rows if int(r["shot_index"])==1]
    accepted={}
    for row in ledger_rows:
        stamp=row.get("accepted_official_pts")
        if stamp is not None and row["match_status"]=="unique_pixel_match": accepted.setdefault(int(stamp),[]).append(row)
    anchor_pts=np.array(sorted(accepted),np.int64)
    archive=Path(opening["model_seed"]["inherited_camera_archive"]["path"])
    if file_sha(archive)!=opening["model_seed"]["inherited_camera_archive"]["sha256"]:raise ValueError("Inherited cameras changed")
    with np.load(archive,allow_pickle=False)as saved:
        cameras={k:saved[k].copy()for k in saved.files}
    camera_lookup={int(p):i for i,p in enumerate(cameras["pts"])}
    frames=[];archive_indices=[]
    for ordinal,(index,pts)in enumerate(zip(INDICES,PTS)):
        exact=accepted.get(int(pts),[])
        candidate=[r for r in ledger_rows if r.get("best_official_pts_candidate")==int(pts)]
        neighbors=None;predicted=None
        if exact:
            row=min(exact,key=lambda r:r["full_pts"]);status="confirmed_pixel_match"
        else:
            prior=anchor_pts[anchor_pts<pts];later=anchor_pts[anchor_pts>pts]
            if not len(prior) or not len(later):raise ValueError("Missing same-shot accepted mapping neighbors")
            left,right=int(prior[-1]),int(later[0])
            lp=min(accepted[left],key=lambda r:r["full_pts"])["full_pts"]
            rp=min(accepted[right],key=lambda r:r["full_pts"])["full_pts"]
            predicted=float(lp+(int(pts)-left)/(right-left)*(rp-lp))
            neighbors=[dict(official_pts=left,legacy60_pts=lp),dict(official_pts=right,legacy60_pts=rp)]
            if not candidate:
                candidate=[r for r in ledger_rows if lp<=r["full_pts"]<=rp]
                status="inferred_navigation_from_accepted_neighbor_bracket_missing_exact_pixel_candidate"
            else:status="inferred_navigation_from_unaccepted_pixel_candidate_with_accepted_neighbors"
            row=min(candidate,key=lambda r:(abs(r["full_pts"]-predicted),r["full_pts"]))
        ci=camera_lookup[int(row["full_pts"])];archive_indices.append(ci)
        frames.append(dict(sequence_index_1based=int(index),programme_frame_index_1based=int(index),
            official_pts=int(pts),official_time_base="1/12800",official_time_seconds=int(pts)/12800,
            official_frame_index_0based=int(pts)//256,legacy60_pts=int(row["full_pts"]),legacy60_time_base="1/15360",
            legacy60_time_seconds=int(row["full_pts"])/15360,legacy60_shot_index=1,
            mapping_status=status,pixel_correspondence_confirmed=bool(exact),navigation_only=True,
            ledger_evidence=dict(line_1based=row["line_1based"],match_status=row["match_status"],
                accepted_official_pts=row.get("accepted_official_pts"),best_official_pts_candidate=row["best_official_pts_candidate"],
                best_roi_gray_mae=row["best_roi_gray_mae"],unique_margin_mae=row["unique_margin_mae"]),
            accepted_neighbor_anchors=neighbors,neighbor_interpolated_legacy60_pts_for_candidate_selection=predicted,
            camera_seed_index_0based=ordinal,inherited_camera_archive_row_0based=ci,
            camera_exposure_representative_legacy60_pts=int(cameras["exposure_representative_pts"][ci]),
            camera25_float64_sha256=array_sha(cameras["camera25"][ci]),
            primary_official50_vs_ledger_official50_BGR_identical=True))
    ai=np.array(archive_indices,np.int64)
    np.savez(DATA/"camera_seeds.npz",programme_indices=INDICES,pts=PTS,
        legacy60_pts=np.array([r["legacy60_pts"]for r in frames],np.int64),
        inherited_camera_archive_indices=ai,camera25=cameras["camera25"][ai],
        installation_id=cameras["installation_id"][ai],view_state4=cameras["view_state4"][ai],
        exposure_representative_pts=cameras["exposure_representative_pts"][ai])
    counts={status:sum(r["mapping_status"]==status for r in frames)for status in sorted({r["mapping_status"]for r in frames})}
    result=deepcopy(opening)
    result.update(schema="rinkspace.official50-continuation-navigation-mapping.v1",
        purpose="101–200 official source frames; inherited cameras for navigation only; no fitting or source-clock substitution.",
        scope=dict(first_programme_index=101,last_programme_index=200,frame_count=100,
            first_official_pts=int(PTS[0]),last_official_pts=int(PTS[-1]),end_official_pts_exclusive=int(PTS[-1]+256)),
        frames=frames,summary=counts,first100_modified=False,processing201_performed=False)
    result.pop("legacy_foreground_cache",None)
    result["model_seed"]["camera_seeds"]=dict(path=str(DATA/"camera_seeds.npz"),sha256=file_sha(DATA/"camera_seeds.npz"))
    result["mapping_policy"]["missing_exact_candidate"]="Use closest existing same-shot ledger row within accepted neighbor bracket; inference only."
    write(OUT/"navigation_mapping.json",result)
    print(json.dumps(dict(stage="mapping_ready",summary=counts,camera_shape=list(cameras["camera25"][ai].shape))),flush=True)
    return result

def navigation(indices=None):
    from calibration.program_tracking.render_result import SceneRenderer
    mapping_value=read(OUT/"navigation_mapping.json")
    scene_record=mapping_value["model_seed"]["scene"]
    if file_sha(scene_record["path"])!=scene_record["sha256"]:raise ValueError("8a28 scene changed")
    renderer=SceneRenderer(read(scene_record["path"]),draw_grid=True,draw_advertisements=True)
    with np.load(DATA/"camera_seeds.npz",allow_pickle=False)as saved: cameras=saved["camera25"].copy()
    selected=set(map(int,INDICES)) if indices is None else set(indices)
    for row in mapping_value["frames"]:
        index=row["programme_frame_index_1based"]
        if index not in selected:continue
        mask_path=DATA/"masks_current"/f"{index:06d}.png"
        if not mask_path.is_file():continue
        raw=cv2.imread(str(DATA/"frames"/f"{index:06d}.png"));mask=cv2.imread(str(mask_path),0)>0
        rendered,metrics=renderer.draw(raw,cameras[index-101],person_mask=mask,title=None)
        if not np.array_equal(rendered[mask],raw[mask]):raise ValueError("Navigation overwrote current person")
        path=DATA/"navigation"/f"{index:06d}.png";cv2.imwrite(str(path),rendered,[cv2.IMWRITE_PNG_COMPRESSION,1])
        row["navigation_overlay"]=dict(path=str(path),sha256=file_sha(path),navigation_only=True,
            current_primary_official50_background=True,current_source_mask_applied=True,current_mask_sha256=file_sha(mask_path),
            camera_seed_refitted=False,all_original_feature_paths_and_advertisement_ids_enabled=True,
            all_mask_pixels_preserved_from_source=True,body_or_trajectory_drawn=False,path_counts=renderer.path_counts)
    write(OUT/"navigation_mapping.json",mapping_value)
    print(json.dumps(dict(stage="navigation_ready",count=sum("navigation_overlay"in r for r in mapping_value["frames"]))),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--navigation-only",action="store_true")
    args=parser.parse_args()
    if args.navigation_only:navigation()
    else:source_rows();mapping()
