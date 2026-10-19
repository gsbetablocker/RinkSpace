"""Visible, reproducible registration trial on a short single-shot sample folder."""
from __future__ import annotations
from pathlib import Path
import html
import shutil
import subprocess
import cv2
import numpy as np
from .io import load_json, dump_json
from .registration import masked_pair, adjacent_lk
from .geometry import project


def register_window(samples: str | Path, mask_path: str | Path, output: str | Path, method: str = "adjacent-lk") -> dict:
    folder, out = Path(samples), Path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Use an empty run directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    frames = out / "frames"
    frames.mkdir()
    source = load_json(folder / "samples.json")
    records = source["samples"]
    if not records:
        raise ValueError("No samples")
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    reference = cv2.imread(str(folder / records[0]["image"]))
    if mask is None or reference is None or mask.shape != reference.shape[:2]:
        raise ValueError("Missing image or mask has wrong dimensions")
    height, width = reference.shape[:2]
    cv2.imwrite(str(out / "reference.jpg"), reference)
    cv2.imwrite(str(out / "mask.png"), mask)
    painted = reference.copy()
    painted[mask == 0] = painted[mask == 0] // 4
    cv2.imwrite(str(out / "masked_reference.jpg"), painted)
    results = []
    previous = reference
    cumulative = np.eye(3)
    if method not in {"adjacent-lk", "direct-sift"}:
        raise ValueError("Unknown registration method")
    for i, sample in enumerate(records):
        image = cv2.imread(str(folder / sample["image"]))
        if image is None:
            raise ValueError(f"Unreadable sample: {sample['image']}")
        if i == 0:
            r = {"status": "reference_identity", "H_reference_from_current": np.eye(3).tolist(),
                 "independent_validation": False, "map_usable_for_metric_output": False}
            render = image.copy()
        else:
            r = (adjacent_lk(previous, image, mask, mask) if method == "adjacent-lk"
                 else masked_pair(reference, image, mask, mask))
            visible = (r["status"] == "candidate" and r["train_inliers"] >= 12
                       and r["diagnostic_p95_reference_px"] <= 8
                       and r["inlier_hull_fraction_of_full_frame"] >= 0.015)
            visible = visible and (method != "adjacent-lk" or cumulative is not None)
            r["preview_gate_passed"] = bool(visible)
            render = np.zeros_like(image)
            if visible:
                h = np.asarray(r["H_reference_from_current"])
                if method == "adjacent-lk":
                    r["H_previous_from_current"] = h.tolist()
                    cumulative = cumulative @ h
                    cumulative = cumulative / np.linalg.norm(cumulative)
                    h = cumulative
                    r["H_reference_from_current"] = h.tolist()
                    r["reference_is_previous_sample"] = False
                    r["diagnostic_error_reference"] = "previous_sample_not_global_reference"
                    r["chain_length"] = i
                render = cv2.warpPerspective(image, h, (width, height))
            else:
                if method == "adjacent-lk":
                    cumulative = None
                cv2.putText(render, "UNAVAILABLE - map candidate rejected", (20, height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1, cv2.LINE_AA)
        r.update({k: sample.get(k) for k in ("sample_index", "source_frame_index", "pts", "time_base", "source_time_s")})
        results.append(r)
        previous = image
        # Grid is a display reference only; it does not establish metric geometry.
        for x in range(0, width, width // 8):
            cv2.line(render, (x, 0), (x, height), (180, 180, 180), 1)
        for y in range(0, height, max(1, height // 5)):
            cv2.line(render, (0, y), (width, y), (180, 180, 180), 1)
        panel = np.zeros((height + 62, width * 2, 3), np.uint8)
        panel[62:, :width] = image
        panel[62:, width:] = render
        cv2.putText(panel, f"SOURCE  t={sample['source_time_s']:.3f}s", (16, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, .72, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(panel, "ICE-PLANE CANDIDATE / NOT METRIC / NO 3D CORRECTION", (width + 15, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, .60, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(panel, "Fixed exclusion mask: example fixture, not automatic person masking", (16, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (190, 190, 190), 1, cv2.LINE_AA)
        cv2.imwrite(str(frames / f"comparison_{i:06d}.jpg"), panel)
    # Fixed union canvas for the whole clip: retain the skater instead of cropping
    # motion away at the first frame's boundaries. This is display-only scaling.
    corners = np.array([[0,0],[width,0],[width,height],[0,height]], dtype=float)
    extents = [corners]
    for r in results[1:]:
        if r.get("preview_gate_passed"):
            extents.append(project(r["H_reference_from_current"], corners))
    cloud = np.concatenate(extents)
    low, high = cloud.min(axis=0), cloud.max(axis=0)
    span = high - low
    if not np.all(np.isfinite(span)) or np.max(span) > 20 * max(width,height):
        raise ValueError("Unstable expanded canvas; inspect rejected geometry before rendering")
    scale = min((width-24)/span[0], (height-24)/span[1])
    offset = np.array([width,height])/2 - scale*(low+high)/2
    display = np.array([[scale,0,offset[0]],[0,scale,offset[1]],[0,0,1]])
    for i, (sample, r) in enumerate(zip(records, results)):
        image = cv2.imread(str(folder / sample["image"]))
        render = np.zeros_like(image)
        if i == 0 or r.get("preview_gate_passed"):
            h = display @ np.asarray(r["H_reference_from_current"])
            render = cv2.warpPerspective(image,h,(width,height))
        else:
            cv2.putText(render,"UNAVAILABLE - no usable candidate continuation",(20,height//2),
                        cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1,cv2.LINE_AA)
        for x in range(0,width,max(1,width//8)):
            cv2.line(render,(x,0),(x,height),(125,125,125),1)
        for y in range(0,height,max(1,height//5)):
            cv2.line(render,(0,y),(width,y),(125,125,125),1)
        panel=np.zeros((height+62,width*2,3),np.uint8)
        panel[62:,:width]=image;panel[62:,width:]=render
        cv2.putText(panel,f"SOURCE t={sample['source_time_s']:.3f}s",(16,28),
                    cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(panel,"FIXED UNION CANVAS / ICE-PLANE CANDIDATE",(width+15,28),
                    cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(panel,"Example exclusion mask. No independent accuracy validation. Not metric or 3D.",(16,52),
                    cv2.FONT_HERSHEY_SIMPLEX,.5,(200,200,200),1,cv2.LINE_AA)
        cv2.imwrite(str(frames/f"comparison_{i:06d}.jpg"),panel)
    delta = np.diff([x["source_time_s"] for x in records])
    video_created = False
    if len(delta) and np.allclose(delta, delta[0], atol=1e-5) and delta[0] > 0 and shutil.which("ffmpeg"):
        cmd = ["ffmpeg", "-v", "error", "-framerate", str(1 / delta[0]), "-i", str(frames / "comparison_%06d.jpg"),
               "-c:v", "libx264", "-preset", "fast", "-crf", "24", "-pix_fmt", "yuv420p", "-threads", "2",
               str(out / "comparison.mp4")]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if p.returncode:
            (out / "video_error.txt").write_text(p.stderr, encoding="utf-8")
        else:
            video_created = True
    summary = {"method": method, "fixed_display_H_from_reference": display.tolist(), "source_sha256": source["source_sha256"], "source_filename": source["source_filename"],
               "reference_time_s": records[0]["source_time_s"], "sample_count": len(results),
               "candidate_preview_count_excluding_reference": sum(r.get("preview_gate_passed", False) for r in results),
               "independently_accepted_map_count": 0, "video_created": video_created,
               "status": "real_video_registration_trial_not_feature_completion", "samples": results}
    dump_json(out / "registration.json", summary)
    picks = sorted(set([0, len(results)//3, 2*len(results)//3, len(results)-1]))
    report = """<!doctype html><meta charset='utf-8'><title>RinkSpace real-video seed</title>
<style>body{max-width:1300px;margin:28px auto;font:17px/1.6 system-ui;background:#f6f7fa;color:#19283b}img,video{width:100%}section{background:white;padding:22px;margin:18px 0}code{background:#eee}</style>
<h1>真实视频：单镜头配准试验</h1><p>这是启动代码的候选预览，不是已验收的稳像、全场地图或速度测量。</p>
<section><h2>原视频 / 固定冰面参照候选</h2>"""
    if video_created:
        report += "<video src='comparison.mp4' controls muted loop></video>"
    report += f"<p>源文件：{html.escape(source['source_filename'])}；{len(results)} 个采样。独立验收通过的映射：0。</p></section>"
    report += "<section><h2>配准使用的区域</h2><img src='masked_reference.jpg'><p>变暗部分不参与特征检测/跟踪拟合。这是特定短片段的粗排除掩膜；生产版需要逐帧冰面与动态对象掩膜。</p></section>"
    report += "<section><h2>时序截图</h2>" + "".join(f"<img src='frames/comparison_{i:06d}.jpg'>" for i in picks) + "</section>"
    report += "<section><h2>限制</h2><p>按自动匹配分出的检查点仍可能共享误匹配，不能替代独立标记的静止冰面点。逐帧串接还可能累积漂移，局部残差小不证明全局正确。以上结果全部保留 candidate 状态；没有角度、米制路线或速度读数。</p><a href='registration.json'>原始诊断数据</a></section>"
    (out / "index.html").write_text(report, encoding="utf-8")
    return {k: v for k, v in summary.items() if k != "samples"}
