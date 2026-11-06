"""Prepare exact-PTS source-only navigation crops for a continuous body sequence.

No network, fitting, projection, foot identity assignment, segmentation or contact
inference is performed. Source UV, source roles and historic failures stay intact.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def bounds_for(points, *, margin=0.0, aspect=None):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if not len(points) or not np.isfinite(points).all():
        raise ValueError("ROI needs finite source-image navigation coordinates")
    lo, hi = points.min(axis=0), points.max(axis=0)
    size = np.maximum(hi - lo, 1.0) + 2 * float(margin)
    if aspect is not None:
        size[0] = max(size[0], size[1] * aspect)
        size[1] = max(size[1], size[0] / aspect)
    centre = (lo + hi) * .5
    return np.r_[np.floor(centre - size / 2), np.ceil(centre + size / 2)].astype(np.int32)


def source_crop(image: Image.Image, bounds, output_size):
    return np.asarray(image.crop(tuple(map(int, bounds))).resize(output_size, Image.Resampling.LANCZOS))


def clipped(bounds, width, height):
    return [max(0, int(bounds[0])), max(0, int(bounds[1])), min(width, int(bounds[2])), min(height, int(bounds[3]))]


def image_reference_index(documents):
    """Reuse references; original PNGs are neither copied nor rewritten."""
    by_pts = {}
    for path, document in documents:
        for row in document.get("frames", []):
            source = row.get("source_frame")
            if source:
                by_pts.setdefault(int(row["pts"]), []).append({
                    "path": source,
                    "sha256": row.get("source_png_sha256", row.get("source_frame_sha256")),
                    "source_rgb_sha256": row.get("source_rgb_sha256"),
                    "document": str(path),
                })
    return by_pts


def anchor_index(path, document):
    anchors, by_pts = [], {}
    for collection in ("observations", "source_evidence_rows", "boot_axis_observations", "additional_direct_review_unknowns"):
        for source_row in document.get(collection, []):
            row = dict(source_row)
            if "pts" not in row:
                continue
            record = {
                "document": str(path), "collection": collection,
                "row_id": row.get("id"), "pts": int(row["pts"]),
                "source_record": row,
                "source_coordinates": "existing source pixels/regions; preserve original semantics and uncertainty",
                "foot_identity_status": "historic foot0/foot1 association proposal, not independently established anatomy",
                "new_inference": False,
            }
            if row.get("visibility") == "unknown" or (row.get("centre_uv") is None and collection != "boot_axis_observations"):
                record["evidence_status"] = "historical_ambiguous_or_unknown_source_review"
            else:
                record["evidence_status"] = "historical_source_review; original semantic scope and provenance retained"
            index = len(anchors)
            anchors.append(record)
            by_pts.setdefault(int(row["pts"]), []).append(index)
    return anchors, by_pts


def sheet_indices(pts):
    # Coarse intervals are half-open, except final95s. Last overview uses 25
    # actual exposures distributed over91-95s rather than duplicating81/86/91.
    specs = [
        ("sequence_76_81.png", 76, 81, .20, False),
        ("sequence_81_86.png", 81, 86, .20, False),
        ("sequence_86_91.png", 86, 91, .20, False),
        ("sequence_91_95.png", 91, 95, None, True),
        ("dense_81_2_82_0.png", 81.2, 82.0, .04, True),
        ("dense_87_6_90_0.png", 87.6, 90.0, .10, True),
        ("dense_92_94.png", 92, 94, .10, True),
    ]
    for name, begin, end, step, inclusive in specs:
        first = int(round(begin * 12800))
        last = int(round(end * 12800))
        if step is None:
            candidates = np.flatnonzero((pts >= first) & (pts <= last))
            selected = candidates[np.rint(np.linspace(0, len(candidates)-1, 25)).astype(int)]
        else:
            stride = int(round(step * 12800))
            targets = np.arange(first, last + (1 if inclusive else 0), stride, dtype=np.int64)
            selected = np.searchsorted(pts, targets)
            if not np.array_equal(pts[selected], targets):
                raise ValueError("Contact sheet target does not exist on native source clock")
        yield name, selected, {"requested_seconds": [begin, end], "requested_stride_seconds": step,
                               "interval_endpoint_included": inclusive, "layout": [5, 5]}


def build_sheets(output, pts, body_rgb, feet_rgb, body_method, feet_method):
    font_path = Path("C:/Windows/Fonts/consola.ttf")
    font = ImageFont.truetype(str(font_path), 17) if font_path.exists() else ImageFont.load_default()
    small = ImageFont.truetype(str(font_path), 13) if font_path.exists() else font
    records = []
    cell_w, cell_h, header = 508, 290, 74
    for name, selected, info in sheet_indices(pts):
        sheet = Image.new("RGB", (5*cell_w, 5*cell_h+header), (24, 28, 35))
        draw = ImageDraw.Draw(sheet)
        draw.text((10, 8), f"SOURCE ONLY | {info['requested_seconds'][0]:g}-{info['requested_seconds'][1]:g}s | native1/12800, 1920x1080", font=font, fill="white")
        draw.text((10, 31), "Body + two ROI proposals; foot0/1 labels do not establish left/right anatomy, blade material or contact.", font=font, fill=(255, 215, 120))
        draw.text((10, 53), "Crops navigate source pixels; bbox fallback can show the same wider person in both proposals. No model overlays.", font=small, fill=(210, 220, 230))
        cells = []
        for cell, index in enumerate(selected):
            index = int(index)
            x, y = (cell % 5)*cell_w, header+(cell//5)*cell_h
            draw.rectangle((x+1, y+1, x+cell_w-2, y+cell_h-2), outline=(90,100,115))
            draw.text((x+7,y+6), f"{pts[index]/12800:.2f}s PTS{int(pts[index])} i{index}",font=font,fill="white")
            sheet.paste(Image.fromarray(body_rgb[index]), (x+7,y+33))
            for foot in range(2):
                fx = x+174+166*foot
                draw.text((fx,y+36), f"ROI proposal{foot}",font=small,fill=(255,215,120))
                sheet.paste(Image.fromarray(feet_rgb[index,foot]),(fx,y+59))
                tag = str(feet_method[index,foot])
                draw.text((fx,y+225), tag[:22],font=small,fill=(215,225,235))
                if "bbox" in tag:
                    draw.text((fx,y+243), "wide fallback",font=small,fill=(255,150,130))
            cells.append({"cell_index": cell,"sequence_index":index,"pts":int(pts[index]),"source_seconds":float(pts[index]/12800),
                          "body_navigation":str(body_method[index]),"feet_navigation":[str(v) for v in feet_method[index]]})
        path = output/name
        sheet.save(path)
        records.append({**info,"path":str(path),"sha256":sha256(path),"dimensions":list(sheet.size),"frames":cells,
                        "actual_pts":[int(pts[i]) for i in selected],"native_source_pixels_only":True})
    return records


def prepare(args):
    project = Path(args.project).resolve()
    sys.path[:0] = [str(project), str(project/"src")]
    from rinkspace.storage import AssetStore
    from rinkspace.viewer3d.source import SourceVideo

    store = AssetStore(project, data_root=args.data_root)
    output = store.resolve(args.output)
    if not output.is_relative_to(store.data) or output.is_relative_to(project):
        raise ValueError("All image/cache products must be within the registered external Data root")
    output.mkdir(parents=True, exist_ok=True)
    programme = project/"events/ankara_2026/programmes/okada_mei_sp"
    base = store.data/"runs/body_s01_20261029"
    case = base/"blade_modes_1105/spin_safe/common/exmr_hand_nofoot"
    paths = {
        "viewer_binding": programme/"viewer3d_runner_support_1105.json",
        "seed_parameter": programme/"spin_modes_spin_safe_1105.npz",
        "source_observations": case/"observations.npz", "camera_input":case/"input.npz",
        "body_seed_result":case/"fitted/result.npz",
        "boot_source_document":base/"blade_orientation_1105/source_boot_axis_document.json",
        "historic_source_index":base/"blade_observability_1105/dual_source_manifest.json",
        "latest_source_review":base/"blade_modes_1105/source_review.json",
    }
    records = {key:{"path":str(path),"sha256":sha256(path),"bytes":path.stat().st_size} for key,path in paths.items()}
    viewer = read_json(paths["viewer_binding"])
    source = store.source(viewer["source_video"])
    tb = tuple(viewer["source_time_base"])
    expected_sha = "e8b53966d56e72f3842d92c3ecbe6e9cc7e4670b7d0c9bd12f58fd2cff5ad57d"
    historic = read_json(paths["historic_source_index"])
    latest = read_json(paths["latest_source_review"])
    if historic["source_sha256"] != expected_sha or latest["source_video_sha256_recomputed"] != expected_sha:
        raise ValueError("Historical source identities disagree")
    if tb != (1,12800):
        raise ValueError("Unexpected native clock")
    source_bytes=source.stat().st_size
    if source_bytes != 65678427:
        raise ValueError("Current source byte count differs from previously verified video")
    with np.load(paths["source_observations"],allow_pickle=False) as z:
        observations={key:z[key].copy() for key in z.files}
    pts=observations["pts"]
    expected=np.arange(972800,1216000+1,256,dtype=np.int64)
    if not np.array_equal(pts,expected):
        raise ValueError("Source observation clock is not the complete951 native exposures")
    with np.load(paths["camera_input"],allow_pickle=False) as z:
        if not np.array_equal(z["pts"],pts):raise ValueError("Camera clock mismatch")
        bbox=z["source_bbox"].copy()
        lower_inferred=z["lower_body_inferred"].copy()
        identity_suspect=z["source_identity_suspect"].copy()
        camera_fields={key:{"shape":list(z[key].shape),"dtype":str(z[key].dtype)} for key in ("camera_rotation","camera_centre","camera_values")}
    with np.load(paths["body_seed_result"],allow_pickle=False) as z:
        if not np.array_equal(z["source_pts"],pts) or not np.array_equal(z["time_base"],tb):raise ValueError("Body clock mismatch")
        camera_binding=json.loads(str(z["camera_binding_json"]))
        body_state_schema=str(z["state_schema_version"])
    uv=observations["uv"]
    sigma=observations["sigma_px"]
    active=observations["use"]
    valid=active & np.isfinite(uv).all(-1) & np.isfinite(sigma) & (sigma>0)
    if uv.shape!=(951,17,2):raise ValueError("Unexpected source COCO shape")
    bootdoc=read_json(paths["boot_source_document"])
    if bootdoc["source_sha256"]!=expected_sha:raise ValueError("Boot source identity mismatch")
    anchors,anchor_by_pts=anchor_index(paths["boot_source_document"],bootdoc)
    shoe_by_pt_foot={}
    for row in bootdoc.get("observations",[]):
        if row.get("feature")=="shoe_body" and row.get("centre_uv") is not None:
            shoe_by_pt_foot.setdefault((int(row["pts"]),row.get("foot")),[]).append(row)
    old_refs=image_reference_index([(paths["historic_source_index"],historic),(paths["latest_source_review"],latest)])
    body_rgb=np.empty((951,240,160,3),np.uint8)
    feet_rgb=np.empty((951,2,160,160,3),np.uint8)
    body_bounds=np.empty((951,4),np.int32)
    feet_bounds=np.empty((951,2,4),np.int32)
    body_method=np.empty(951,dtype="U48")
    feet_method=np.empty((951,2),dtype="U48")
    clocks=[]
    navigation_inferred=[]
    bbox_ok=np.isfinite(bbox).all(1)&(bbox[:,2]>bbox[:,0])&(bbox[:,3]>bbox[:,1])
    real_bbox_indices=np.flatnonzero(bbox_ok)
    reader=SourceVideo(source,tb)
    if reader.stream.time_base != Fraction(*tb):raise ValueError("Video native time base mismatch")
    stream={"time_base":[reader.stream.time_base.numerator,reader.stream.time_base.denominator],
            "width":reader.stream.codec_context.width,"height":reader.stream.codec_context.height,"average_rate":str(reader.stream.average_rate)}
    if (stream["width"],stream["height"])!=(1920,1080):raise ValueError("Source dimensions mismatch")
    for index,pt in enumerate(pts):
        rgb=reader.read(int(pt))
        if rgb.shape!=(1080,1920,3):raise ValueError("Decoded source dimensions mismatch")
        im=Image.fromarray(rgb)
        bb=np.asarray(bbox[index],float)
        bb_valid=np.isfinite(bb).all() and bb[2]>bb[0] and bb[3]>bb[1]
        points=uv[index,valid[index]]
        if bb_valid:
            points=np.r_[points,bb.reshape(2,2)]
            body_method[index]="source_uv_and_detector_bbox" if valid[index].any() else "source_detector_bbox_fallback"
        elif len(points):body_method[index]="source_qualified_uv_extent"
        else:
            before=real_bbox_indices[real_bbox_indices<index]
            after=real_bbox_indices[real_bbox_indices>index]
            if not len(before) or not len(after):
                points=np.asarray([[0,0],[1920,1080]],float)
                neighbours=[]
                method="source_full_frame_navigation_fallback"
            else:
                neighbours=[int(before[-1]),int(after[0])]
                points=bbox[neighbours].reshape(-1,2)
                method="adjacent_source_bbox_union_inferred"
            body_method[index]=method
            navigation_inferred.append({"sequence_index":index,"pts":int(pt),"navigation_status":"inferred_navigation_only",
                                        "method":method,"source_neighbour_indices":neighbours,
                                        "source_neighbour_pts":[int(pts[i]) for i in neighbours],
                                        "measured_source_uv_remains_missing":True,
                                        "new_observation":False,"identity_or_contact_claim":False})
        body_bounds[index]=bounds_for(points,margin=40,aspect=2/3)
        body_rgb[index]=source_crop(im,body_bounds[index],(160,240))
        for foot in (0,1):
            joint=15+foot
            if valid[index,joint]:
                bounds=bounds_for(uv[index,joint],margin=112,aspect=1)
                method="source_qualified_ankle_uv"
            elif shoe_by_pt_foot.get((int(pt),foot)):
                row=shoe_by_pt_foot[(int(pt),foot)][-1]
                bounds=bounds_for(row["centre_uv"],margin=112,aspect=1)
                method="exact_pts_source_shoe_region"
            else:
                bounds=bounds_for(points,margin=60,aspect=1)
                method="adjacent_bbox_union_inferred" if "inferred" in str(body_method[index]) else "source_person_bbox_wide_fallback"
            feet_bounds[index,foot]=bounds
            feet_method[index,foot]=method
            feet_rgb[index,foot]=source_crop(im,bounds,(160,160))
        rgb_sha=hashlib.sha256(rgb.tobytes()).hexdigest()
        refs=old_refs.get(int(pt),[])
        for ref in refs:
            if ref.get("source_rgb_sha256") and ref["source_rgb_sha256"]!=rgb_sha:
                raise ValueError(f"Historic exact-PTS sourceRGB mismatch at {pt}")
        clocks.append({"sequence_index":index,"pts":int(pt),"source_seconds":float(Fraction(int(pt),12800)),
                       "decoded_exact_pts":True,"decoded_source_rgb_sha256":rgb_sha,"camera_input_index":index,
                       "body_seed_index":index,"source_observation_index":index,"source_boot_anchor_indices":anchor_by_pts.get(int(pt),[]),
                       "historic_source_image_references":refs,"body_navigation_bounds_native_xyxy":body_bounds[index].tolist(),
                       "body_navigation_bounds_clipped_xyxy":clipped(body_bounds[index],1920,1080),"body_navigation_method":str(body_method[index]),
                       "feet_navigation_bounds_native_xyxy":feet_bounds[index].tolist(),
                       "feet_navigation_bounds_clipped_xyxy":[clipped(q,1920,1080) for q in feet_bounds[index]],
                       "feet_navigation_methods":[str(q) for q in feet_method[index]],
                       "navigation_status":"inferred_navigation_only" if "inferred" in str(body_method[index]) else "derived_from_source_pixels",
                       "effective_source_joint_count":int(valid[index].sum()),"source_identity_suspect":bool(identity_suspect[index]),
                       "source_lower_body_inferred":bool(lower_inferred[index])})
        if (index+1)%200==0:print(json.dumps({"decoded_exact_exposures":index+1,"last_pts":int(pt)}),flush=True)
    reader.close()
    sheets=build_sheets(output,pts,body_rgb,feet_rgb,body_method,feet_method)
    print(json.dumps({"CONTACT_SHEETS_READY":[s["path"] for s in sheets]},ensure_ascii=False),flush=True)
    cache=output/"source_roi_cache.npz"
    # Keep original source values, flags and association-history arrays verbatim.
    arrays={"source_observations_"+key:value for key,value in observations.items()}
    arrays.update(pts=pts,time_base=np.asarray(tb,dtype=np.int64),source_sha256=np.asarray(expected_sha),
                  source_native_size=np.asarray([1920,1080],dtype=np.int32),body_rgb=body_rgb,feet_rgb=feet_rgb,
                  body_bounds_native_xyxy=body_bounds,feet_bounds_native_xyxy=feet_bounds,
                  body_navigation_method=body_method,feet_navigation_method=feet_method,
                  effective_source_uv_valid=valid,source_detector_bbox_native_xyxy=bbox,
                  source_lower_body_inferred=lower_inferred,source_identity_suspect=identity_suspect,
                  feet_proposal_coco_joint_index=np.asarray([15,16],np.int32))
    np.savez_compressed(cache,**arrays)
    definitions={key:{"shape":list(value.shape),"dtype":str(value.dtype)} for key,value in arrays.items()}
    script=Path(__file__).resolve()
    manifest={"schema":"rinkspace.continuous-source-sequence.v1","created_at":datetime.now(timezone.utc).isoformat(),
              "scope":{"shot_id":"s05","first_last_seconds":[76,95],"frames":951,"first_last_pts":[int(pts[0]),int(pts[-1])],"time_base":list(tb),
                       "denominator":"all951 native source exposures, not interpolated or duplicate display frames"},
              "source_video":{"path":str(source),"sha256":expected_sha,"sha256_status":"reuse previously recomputed source identity; no source-video rehash in this navigation task", "bytes":source_bytes,
                              "identity_evidence":[records["historic_source_index"],records["latest_source_review"]],"stream":stream,"decoded_all_exact_native_pts":True},
              "inputs":records,"camera":{"input_array_fields":camera_fields,"binding_from_saved_body":camera_binding,"role":"frozen camera index only; never used for source ROI navigation"},
              "body":{"state_schema":body_state_schema,"role":"existing seed index only; no model projections used as observations"},
              "source_anchor_observed_records":anchors,"source_anchor_inferred_records":[],
              "navigation_inferred_records":navigation_inferred,
              "inference_note":"Three missing source-bbox exposures use the union of nearest earlier/later real source bboxes, as expressly authorized for navigation. This creates no body, contact, anatomy, edge or material-point observation. Historical source rows retain original statuses and uncertainties.",
              "navigation_semantics":{"observed":"source detector UV/sigma/use and existing exact-PTS source review rows retained verbatim",
                  "derived":"crop bounds, resizing and bbox fallbacks computed only for image navigation; not measurements or segmentation",
                  "foot_labels":"proposal0/1 refer to inherited COCO ankle indices15/16 or historical foot labels; they are not anatomical truth",
                  "effective_source_uv":"use=True AND finite UV AND finite positive sigma_px; excludes invalid/excluded source points",
                  "fallback":"exact-PTS source shoe_body region first, then expanded same-frame source person bbox/qualified UV. Three fully missing exposures use nearest bracketing source-bbox union, marked inferred_navigation_only; no model projections",
                  "body_roi":"union of source detector bbox and qualified source UV,40px margin,2:3 native bounds; allows padding outside image",
                  "foot_roi":"224px square centred on qualified source ankle or same-frame source shoe region; wide source person fallback if missing",
                  "resampling":"PIL LANCZOS from original RGB pixels; no sharpening, segmentation or overlays; out-of-image padding is black"},
              "cache":{"path":str(cache),"sha256":sha256(cache),"bytes":cache.stat().st_size,"arrays":definitions},
              "navigation_counts":{"effective_joint_counts":valid.sum(0).tolist(),"body_methods":dict(Counter(map(str,body_method))),
                                   "foot_methods":dict(Counter(map(str,feet_method.reshape(-1))))},
              "contact_sheets":sheets,"clock":clocks,"code":{"path":str(script),"sha256":sha256(script)},
              "previous_attempt_failures":[{"stage":"source_navigation","pts":1150208,"reason":"all qualified UV and source_bbox unavailable; first attempt stopped before writing products","resolution":"authorized bracketing real source-bbox union for navigation only; missing source measurements retained"}],
              "execution":{"network_inference":False,"fit":False,"model_projection":False,"new_fullsize_source_pngs":0,"source_video_copied":False,"video_generated":False}}
    target=output/"source_sequence.json"
    write_json(target,manifest)
    # Meaningful integrity check on products: all crops/PTS definitions and PNG decoding.
    with np.load(cache,allow_pickle=False) as saved:
        assert np.array_equal(saved["pts"],pts)
        assert np.array_equal(saved["source_observations_uv"],uv,equal_nan=True)
        assert saved["body_rgb"].shape==(951,240,160,3)
        assert saved["feet_rgb"].shape==(951,2,160,160,3)
    for sheet in sheets:
        with Image.open(sheet["path"]) as im:im.verify()
    print(json.dumps({"manifest":str(target),"cache":str(cache),"cache_bytes":cache.stat().st_size,"frames":951,"sheets":len(sheets),"verified":True},ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project",required=True)
    parser.add_argument("--data-root",required=True)
    parser.add_argument("--output",required=True)
    prepare(parser.parse_args())


if __name__=="__main__":
    main()
