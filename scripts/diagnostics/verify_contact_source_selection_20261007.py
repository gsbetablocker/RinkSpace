"""Verify final contact selection on authenticated RGB and frozen GPU weights."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from rinkspace.vision_network import RinkVisionNet
from rinkspace.vision import confirmed_source_guides
from rinkspace import vision_geometry as vg


def summary(values):
    a = np.asarray(values, dtype=float)
    return dict(count=len(a), median=float(np.median(a)) if len(a) else None,
                p90=float(np.quantile(a, .9)) if len(a) else None,
                maximum=float(a.max()) if len(a) else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    args = parser.parse_args()
    torch.set_num_threads(2)
    path = Path(r"D:/CodexData/RinkSpace_Data/events/ankara_2026/observations/vision_training/rink_vision_candidate_11200_user_iceedge2.pt")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    network = RinkVisionNet(**saved["model_config"]).cuda().eval()
    network.load_state_dict(saved["state_dict"], strict=True)
    classes = saved["classes"]
    sample = next(s for s in saved["source_manifest"]["samples"] if s["source_pts"] == 228372480)
    rgbpath = Path(sample["frame_path"])
    if hashlib.sha256(rgbpath.read_bytes()).hexdigest() != sample["frame_sha256"]:
        raise ValueError("Authenticated source frame changed")
    rgb = np.asarray(Image.open(rgbpath).convert("RGB"))
    native = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].cuda().float() / 255
    x, y, w, h = 1280, 256, 512, 512
    guides = []
    for original in confirmed_source_guides(sample):
        if not str(original.get("source_group_id", "")).startswith("user_contact_lower_board_base_strip_native228372480_crop1430_1640_v1"):
            continue
        guide = copy.deepcopy(original)
        points = np.asarray(guide["points_uv"])
        keep = np.flatnonzero((points[:, 0] >= x + 25) & (points[:, 0] < x + w - 25)
            & (points[:, 1] >= y + 25) & (points[:, 1] < y + h - 25))[::4]
        for key in ["points_uv", "normals", "segment_ids", "original_source_point_indices"]:
            if key in guide:
                guide[key] = np.asarray(guide[key])[keep].tolist()
        guide["segment_ids"] = [0] * len(keep)
        guides.append(guide)
    binding = guides[0]["source_binding"]
    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"):
            context = network.encode_context(native[:, None])
            result = network(native[:, :, y:y+h, x:x+w][:, None], context_features=context,
                context_crop=(x, y, w, h), context_native_hw=rgb.shape[:2])
        probabilities = result["logits"][:, 0].float().softmax(1)
        full = torch.full((1, len(classes), *rgb.shape[:2]), 1 / len(classes), device="cuda")
        full[:, :, y:y+h, x:x+w] = probabilities
        original = vg.guided_source_reading(native, full, guides, classes=classes,
            rgb_source_binding=binding, probability_source_binding=binding,
            guide_source_binding=binding, probabilities_conditioned_on_geometry=False)
        final = vg.coherent_named_source_reading(original)
        row = original["readings"][0]
        revised = final["readings"][0]
        oldidx = row["source_indices"].cpu().numpy()
        newidx = revised["coherent_source_indices"].cpu().numpy()
        olduv = row["actual_source_uv"].cpu().numpy()
        newuv = revised["coherent_source_uv"].cpu().numpy()
        common, oldat, newat = np.intersect1d(oldidx, newidx, return_indices=True)
        human = np.asarray(guides[0]["points_uv"])
        before = np.linalg.norm(olduv[oldat] - human[common], axis=1)
        after = np.linalg.norm(newuv[newat] - human[common], axis=1)
        uniform = copy.deepcopy(original)
        uniform["readings"][0]["source_peak_candidates"]["typed_probability"].fill_(1 / len(classes))
        fallback = vg.coherent_named_source_reading(uniform)["readings"][0]
        assert torch.equal(fallback["coherent_source_uv"], row["actual_source_uv"])
        assert torch.equal(fallback["coherent_source_indices"], row["source_indices"])
        assert torch.equal(revised["confirmed_source_uv"], row["confirmed_source_uv"])
        assert len(revised["coherent_support"]["normal_interval_px"]) == len(newidx)
        print(json.dumps(dict(before=summary(before), after=summary(after),
            improved=int((after < before).sum()), same=int((after == before).sum()),
            coherent_flags=revised["coherent_flags"])), flush=True)
        if not np.all(after <= before) or np.median(after) >= 2:
            raise ValueError("Real source common-position correction failed")
        report = dict(schema="rinkspace.final-contact-semantic-source-selection-check.v1",
            source_pts=228372480, source_image=str(rgbpath),
            model_path=str(path), device="cuda", precision=args.precision, trained=False,
            common_original_source_positions=len(common), before=summary(before), after=summary(after),
            original_final_count=len(oldidx), revised_final_count=len(newidx),
            improved_common_positions=int((after < before).sum()),
            unchanged_common_positions=int((after == before).sum()),
            flat_semantic_RGB_fallback_bitexact=True, original_human_UV_preserved_bitexact=True,
            final_support_matches_final_selected_UV=True,
            extra_source_positions=summary(np.linalg.norm(newuv[~np.isin(newidx, oldidx)] - human[newidx[~np.isin(newidx, oldidx)]], axis=1)),
            limitation="Frozen training-source local mechanism verification, not whole-shot or held-out accuracy")
        output = ROOT / "events/ankara_2026/diagnostics/contact_source_selection_20261007.json"
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report))


if __name__ == "__main__":
    main()
