"""Workbench runner for the frozen affine candidate, with local fitting audits.

No learned model or download is required. UI corrections and whole-group
exclusions are applied per source sample; masks remain visually reviewable
heuristic candidates. Protected input values cannot affect mask heuristics or
the estimator: replace them with a fixed neutral colour before mask detection,
then apply the exclusion again after all manual corrections.
"""
from __future__ import annotations

from collections import Counter
import copy
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .io import dump_json
from .local_registration import ice_mask, _regions_mask, render_fixed_frames
from .photometric_registration import PhotometricRegistrar, PhotometricConfig, ALGORITHM_VERSION


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render_camera_frames(source, output, result, max_pixels):
    """Choose one finite observed reference chart per component for display only.

    Original matrices/source references remain untouched. If a component cannot
    fit this fixed chart, its display is unavailable; we never remove its frames
    or project across infinity merely to create an attractive preview.
    """
    display_record=copy.deepcopy(result)
    width,height=result['decoded_dimensions']
    corners=np.array([[0.,0.],[width,0.],[width,height],[0.,height]])
    groups={}; gauges={}; rejected=[]
    for row in display_record['frames']:
        if row['status']=='candidate':groups.setdefault(row['graph_component_id'],[]).append(row)
    for component,rows in groups.items():
        middle=rows[len(rows)//2]
        gauge=np.linalg.inv(np.asarray(middle['H_reference_from_current']))
        transformed=[gauge@np.asarray(r['H_reference_from_current']) for r in rows]
        safe=True
        for h in transformed:
            den=(np.c_[corners,np.ones(4)]@h.T)[:,2]
            if not np.isfinite(h).all() or min(den)*max(den)<=0 or min(abs(den))<1e-10:
                safe=False;break
        if not safe:
            rejected.append(component)
            for row in rows:row['status']='unavailable'
            continue
        gauges[component]=(gauge,middle['sample_index'])
        for row,h in zip(rows,transformed):row['H_reference_from_current']=(h/h[2,2]).tolist()
    canvases=render_fixed_frames(source,output,display_record,max_pixels)
    for component,canvas in canvases.items():
        gauge,index=gauges[component]
        canvas['H_display_from_reference']=(np.asarray(canvas['H_display_from_reference'])@gauge).tolist()
        canvas['display_reference_bounds']=canvas.pop('reference_bounds')
        canvas['display_gauge_from_original_reference']=gauge.tolist()
        canvas['display_gauge_source_sample']=index
    for original,display in zip(result['frames'],display_record['frames']):
        for field in ('fixed_path','fixed_image','H_display_from_current'):
            if field in display:original[field]=display[field]
        if original.get('graph_component_id') in rejected:
            original['fixed_display_status']='unavailable_whole_component_reference_chart_horizon'
    result['display_rejected_components']=rejected
    return canvases


def register_photometric_sequence(sample_dir, out_dir, config=None, exclusions=None,
                                 corrections=None, shot_breaks=None, render=True,
                                 freeze_lock=None, backend="photometric_affine"):
    """Same sequence inputs as the baseline; estimator parameters stay frozen."""
    source, output = Path(sample_dir), Path(out_dir)
    settings = dict(config or {})
    corrections = settings.pop("corrections", corrections) or {}
    exclusions = settings.pop("exclusions", exclusions) or {}
    if settings:
        raise ValueError("Frozen affine backend does not accept parameter overrides: " + ", ".join(sorted(settings)))
    if backend not in ("photometric_affine", "rotation_zoom_experimental"):
        raise ValueError("Unknown photometric backend")
    version = ALGORITHM_VERSION
    module_name = "photometric_registration.py"
    params = PhotometricConfig(motion_model="affine")
    tracker = PhotometricRegistrar(params)
    if backend == "rotation_zoom_experimental":
        from .rotation_photometric import RotationPhotometricRegistrar, RotationPhotometricConfig, ALGORITHM_VERSION as rotation_version
        params = RotationPhotometricConfig(initial_focal_width_ratio=4., max_keyframes=40)
        tracker = RotationPhotometricRegistrar(params)
        tracker.maximum_fixed_keyframe_trials = 40
        version, module_name = rotation_version, "rotation_photometric.py"
        if freeze_lock is not None:
            raise ValueError("Experimental rotation model cannot use the affine holdout lock")
    package = Path(__file__).parent
    dependency_hashes = {"src/rinkspace/"+name:_sha(package/name) for name in (
        "masked_photometric.py", "photometric_registration.py", "continuous_registration.py",
        "local_registration.py", "geometry.py")}
    if backend == "rotation_zoom_experimental":
        dependency_hashes["src/rinkspace/rotation_photometric.py"] = _sha(package/module_name)
    freeze = {"verified": False, "reason": "no_local_lock_supplied"}
    if freeze_lock is not None:
        lock_path = Path(freeze_lock)
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        locked = lock["algorithm"]
        mismatches = [name for name,digest in dependency_hashes.items()
                      if locked["source_sha256"].get(name) != digest]
        if locked["config"] != asdict(params) or locked["sparse_initializer_fit_config"] != asdict(tracker.fit_config):
            mismatches.append("estimator_config")
        if mismatches:
            raise ValueError("Frozen affine implementation differs from lock: " + ", ".join(mismatches))
        freeze = {"verified": True, "lock_sha256": _sha(lock_path),
                  "scope": "estimator modules and parameters; workbench mask inputs are separately recorded"}
    record = json.loads((source/"samples.json").read_text(encoding="utf-8"))
    if not record.get("samples"):
        raise ValueError("No actual decoded source frames in requested window")
    for name in ("masks", "protected", "foreground", "fixed", "overlays", "footprints"):
        (output/name).mkdir(parents=True, exist_ok=True)
    rows, audit_files = [], []
    cuts = set(shot_breaks or [])
    for sample in record["samples"]:
        image = cv2.imread(str(source/sample["image"]))
        if image is None:
            raise ValueError(f"Cannot decode source sample {sample['image']}")
        key = str(sample["sample_index"])
        regions = exclusions.get("*", []) + exclusions.get(key, [])
        correction = corrections.get(key, corrections.get("*", {}))
        protected = _regions_mask(image.shape[:2], regions)
        detection_input = image.copy()
        detection_input[protected > 0] = [200, 200, 200]
        masks = ice_mask(detection_input, correction)
        masks["usable"][protected > 0] = 0
        masks["protected"] = protected
        masks["usable_fraction"] = float(np.mean(masks["usable"] > 0))
        row = tracker.push(image, sample, masks, int(sample["sample_index"]) in cuts)
        row.update(protected_regions=regions, mask_correction=correction,
                   mask_input_policy="protected_values_constant_200_before_heuristics_and_excluded_after_corrections",
                   independent_validation=False, formal_acceptance=False)
        for folder,field in (("masks","mask_path"),("protected","protected_mask_path"),("foreground","foreground_mask_path")):
            path = f"{folder}/{int(sample['sample_index']):06d}.png"
            cv2.imwrite(str(output/path), masks["usable" if folder=="masks" else folder])
            row[field] = path
        audit_path = f"footprints/{int(sample['sample_index']):06d}.json.gz"
        with (output/audit_path).open("wb") as f:
            with gzip.GzipFile(fileobj=f, mode="wb", mtime=0) as gz:
                gz.write(json.dumps(tracker.last_photometric_audit, separators=(",",":"), allow_nan=False).encode("utf-8"))
        row["photometric_audit_path"] = audit_path
        audit_files.append({"sample_index": sample["sample_index"], "path": audit_path,
                            "sha256": _sha(output/audit_path),
                            "attempted_edge_count": len(tracker.last_photometric_audit["attempts"])})
        overlay = image.copy()
        overlay[masks["usable"] == 0] = (overlay[masks["usable"] == 0]*.3).astype(np.uint8)
        overlay[protected > 0] = np.clip(overlay[protected > 0]*.4 + [100,0,100],0,255).astype(np.uint8)
        if row["support_polygon"]:
            cv2.polylines(overlay,[np.rint(row["support_polygon"]).astype(np.int32)],True,(0,220,80),1)
        for xy in row["fit_current_xy"]:
            cv2.circle(overlay,tuple(np.rint(xy).astype(int)),1,(0,200,255),-1)
        cv2.putText(overlay,f"PTS {sample['pts']} | {row['status']} | photo samples, NOT feature inliers",
                    (10,23),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,255),1,cv2.LINE_AA)
        path = f"overlays/{int(sample['sample_index']):06d}.jpg"
        cv2.imwrite(str(output/path),overlay,[cv2.IMWRITE_JPEG_QUALITY,87])
        row["overlay_path"] = row["mask_image"] = path
        rows.append(row)
    configuration = {"photometric":asdict(params),"sparse_initializer_fit":asdict(tracker.fit_config),
                     "corrections":corrections,"exclusions":exclusions,"shot_breaks":sorted(cuts),
                     "mask_policy":"protected_constant_before_heuristics_raw_exclusion_then_estimator_safe_erosion"}
    result = {"schema_version":"1.0", "algorithm":version, "backend":backend,
        "configuration_sha256":hashlib.sha256(json.dumps(configuration,sort_keys=True).encode()).hexdigest(),
        "algorithm_source_sha256":dependency_hashes["src/rinkspace/"+module_name],
        "dependency_source_sha256":dependency_hashes,"sequence_runner_sha256":_sha(__file__),
        "freeze_verification":freeze,"source_sha256":record.get("source_sha256"),
        "source_filename":record.get("source_filename"),"sample_directory":str(source.resolve()),
        "decoded_dimensions":[image.shape[1],image.shape[0]],"config":configuration,
        "sampling":{"every_n":record.get("every_n",1),"pts_policy":"actual_decoded_source_pts",
                    "source_frame_count":len(rows)},"coordinate_space":"local_projective","unit":"reference_px",
        "motion_model":params.motion_model,"independent_validation":False,"formal_acceptance":False,
        "map_usable_for_metric_output":False,
        "quality_note":"Photometric candidate; sampling points are not RANSAC inliers or independent checks; masks require source review. Rotation/zoom mode additionally assumes fixed optical centre, centred principal point and initial focal length four image widths; none is camera calibration.",
        "full_fitting_audit":"footprints/manifest.json; all gzip iteration audits remain local, not included in compact review.zip",
        "frames":rows,"summary":{"sample_count":len(rows),
            "candidate_count":sum(r["status"]=="candidate" and not r.get("reference_frame_only") for r in rows),
            "reference_count":sum(bool(r.get("reference_frame_only")) for r in rows),
            "unavailable_count":sum(r["status"]!="candidate" for r in rows),
            "component_count":len({r["graph_component_id"] for r in rows if r["graph_component_id"] is not None}),
            "reasons":dict(Counter(r["reason"] for r in rows))}}
    dump_json(output/"footprints/manifest.json",{"schema":"rinkspace.photometric-footprints.v1",
        "source_sha256":record.get("source_sha256"),"configuration_sha256":result["configuration_sha256"],
        "files":audit_files,"independent_validation":False,"local_full_audit":True})
    if render:
        renderer=render_camera_frames if backend=='rotation_zoom_experimental' else render_fixed_frames
        result["canvases"] = renderer(source,output,result,tracker.fit_config.max_canvas_pixels)
    dump_json(output/"registration.json",result)
    return result
