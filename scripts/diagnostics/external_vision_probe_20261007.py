"""Independent, CPU-only causal probe; never changes production code or weights.

Run with the project's configured GPU Python (Torch is required), but every
tensor and model remains on CPU. The sole experimental compatibility override
replaces vision_geometry._cuda_tensor's device admission with CPU admission;
all source-reading calculations, quality filters and identity assignment are
the production implementations. Original native RGB and source bindings stay
unchanged. Results are a local causal experiment, not full-video validation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from rinkspace.vision_network import RinkVisionNet
from rinkspace.vision import SourceRinkReader, confirmed_source_guides, _annotations_in_crop, interval_supervision, primitive_supervision
from rinkspace.vision_supervision import confirmed_position_interval_localization_loss
from rinkspace import vision_geometry as vg

DATA = pathlib.Path(r"D:\CodexData\RinkSpace_Data\events\ankara_2026\observations\vision_training")


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def cpu_admission(value, *, dimensions=None):
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
        raise ValueError("Independent probe accepts CPU tensors only")
    if dimensions is not None and value.ndim != dimensions:
        raise ValueError("Independent probe retains production dimension check")
    return value


def number_summary(values):
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    return dict(count=len(a), median=float(np.median(a)) if len(a) else None,
                p90=float(np.quantile(a, .9)) if len(a) else None,
                maximum=float(a.max()) if len(a) else None)


def tensor_summary(a):
    return number_summary(a.detach().cpu().numpy())


def sampled_probabilities(probabilities, uv, crop):
    xy = torch.as_tensor(np.asarray(uv)-np.asarray(crop[:2]), dtype=torch.float32)
    h, w = probabilities.shape[-2:]
    grid = xy * xy.new_tensor([2/(w-1), 2/(h-1)])-1
    return F.grid_sample(probabilities, grid[None, :, None], align_corners=True)[0, :, :, 0].T


def compare_readings(a, b):
    return dict(
        source_indices_equal=torch.equal(a["source_indices"], b["source_indices"]),
        selected_native_UV_bitexact=torch.equal(a["actual_source_uv"], b["actual_source_uv"]),
        candidate_native_UV_bitexact=torch.allclose(a["source_peak_candidates"]["actual_peak_uv"],
                    b["source_peak_candidates"]["actual_peak_uv"], rtol=0, atol=0, equal_nan=True),
        physical_id_equal=a["physical_id"] == b["physical_id"],
        accepted_count=len(b["source_indices"]),
    )


def draw_panel(rgb, crop, guides, reading, probability, title):
    x, y, w, h = crop
    raw = Image.fromarray(rgb[y:y+h, x:x+w]).convert("RGB")
    marks = raw.copy(); d = ImageDraw.Draw(marks)
    for guide in guides:
        points = np.asarray(guide["points_uv"])-[x, y]
        for u, v in points:
            d.ellipse((u-1, v-1, u+1, v+1), fill=(255, 0, 0))
    for item in reading:
        for u, v in item["actual_source_uv"].cpu().numpy()-[x, y]:
            d.ellipse((u-1, v-1, u+1, v+1), fill=(0, 255, 255))
    p = probability.detach().cpu().numpy()
    heat = np.stack((p*255, np.zeros_like(p), (1-p)*255), -1).astype(np.uint8)
    heat = Image.blend(raw, Image.fromarray(heat), .48)
    panel = Image.new("RGB", (3*w, h+46), (18, 24, 30))
    panel.paste(raw, (0, 46)); panel.paste(marks, (w, 46)); panel.paste(heat, (2*w, 46))
    d = ImageDraw.Draw(panel)
    d.text((8, 6), title, fill="white")
    d.text((8, 24), "Original RGB", fill="white")
    d.text((w+8, 24), "Red: human positions; cyan: production RGB reader", fill="white")
    d.text((2*w+8, 24), "Neural class probability: red high; blue low", fill="white")
    return panel


def postprocess_only(out):
    """Replay production final selection without executing the network again.

    The first probe saved exact equality of the RGB peak coordinates and selected
    UVs for each original probability ablation. Reconstruct that source-only
    evidence from the authenticated RGB. coherent_named_source_reading consumes
    RGB peaks/width/quality/profile membership, not typed_probability or model
    disagreements. Run every recorded variant, replacing ONLY those unused
    probability diagnostics. Also directly recompute uniform/all-shadow inputs.
    """
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    report_path=out/"vision_causal_probe.json"
    report=json.loads(report_path.read_text(encoding="utf-8"))
    before_digest=sha(report_path)
    saved=torch.load(report["model_path"],map_location="cpu",weights_only=True)
    classes=saved["classes"]
    vg._cuda_tensor=cpu_admission
    final_report=dict(network_forward_calls=0,checkpoint_loaded_for_source_manifest_only=True,
        replay_basis="Authenticated original RGB and previously measured invariant RGB peak evidence; no neural forward rerun",
        probability_fields_in_final_shape_selection="typed_probability and model_disagreement are unused by production coherent_named_source_reading",
        source_record_before_postprocess_sha256=before_digest,
        actual_call_chain=["vision_review.py:1648 reader.read", "vision.py:1119 guided_source_reading",
            "vision.py:1127 coherent_named_source_reading", "vision.py:1138 coherent_source_indices",
            "vision.py:1139 coherent_source_uv", "vision.py:1171 draw coherent_source_uv",
            "vision_review.py:305 application_source_record coherent_source_uv"],cases=[])
    for case in report["cases"]:
        pts=case["source_pts"];x,y,w,h=case["crop_native_xywh"]
        sample=next(s for s in saved["source_manifest"]["samples"] if s["source_pts"]==pts)
        if sha(sample["frame_path"])!=case["source_image_sha256"]:raise ValueError("Source image changed")
        rgb=np.asarray(Image.open(sample["frame_path"]).convert("RGB"))
        native=torch.from_numpy(rgb.copy()).permute(2,0,1)[None].float()/255
        prefix=("user_contact_lower_board_base_strip_native228372480_crop1430_1640_v1"
                if pts==228372480 else "F3_outer_user_230507776")
        guides=[]
        for original in confirmed_source_guides(sample,role_definitions=case["experimental_role_definitions"]):
            if not str(original.get("source_group_id","")).startswith(prefix):continue
            g=copy.deepcopy(original);p=np.asarray(g["points_uv"])
            keep=np.flatnonzero((p[:,0]>=x+25)&(p[:,0]<x+w-25)&(p[:,1]>=y+25)&(p[:,1]<y+h-25))[::4]
            if not len(keep):continue
            for key in ["points_uv","normals","segment_ids","original_source_point_indices"]:
                if key in g:g[key]=np.asarray(g[key])[keep].tolist()
            g["segment_ids"]=[0]*len(keep);guides.append(g)
        binding=guides[0]["source_binding"]
        uniform=torch.full((1,len(classes),*rgb.shape[:2]),1/len(classes))
        base=vg.guided_source_reading(native,uniform,guides,classes=classes,
            rgb_source_binding=binding,probability_source_binding=binding,guide_source_binding=binding,
            probabilities_conditioned_on_geometry=False)
        final_case=dict(source_pts=pts,probability_variants={},pairwise_candidate_ranking_comparison={})
        baseline=None
        for label,saved_rows in case["final_RGB_reader_ablation"].items():
            for original,row in zip(saved_rows,base["readings"]):
                if not np.array_equal(np.asarray(original["selected_native_UV"]),row["actual_source_uv"].numpy()):
                    raise ValueError("Reconstructed RGB selected UV differs from measured original probe")
                if not original["candidate_native_UV_bitexact"]:raise ValueError("Original candidate pool invariance was not measured")
            packet=copy.deepcopy(base)
            for row in packet["readings"]:
                # These neural diagnostics were different in the original live
                # probe. Their arbitrary replacement tests absence of dependence
                # in the final production RGB shape selector, without inventing
                # another real network prediction.
                pool=row["source_peak_candidates"]
                pool["typed_probability"]=torch.full_like(pool["typed_probability"],
                    .999 if label=="real" else 0. if label=="all_shadow_probabilities" else .123)
                row["model_disagreement"]={"experimental_unused_neural_diagnostic_variant":label}
            coherent=vg.coherent_named_source_reading(packet)
            if baseline is None:baseline=coherent
            rows=[]
            for prior,row,g in zip(baseline["readings"],coherent["readings"],guides):
                indices=row["coherent_source_indices"]
                uv=row["coherent_source_uv"]
                points=np.asarray(g["points_uv"])[indices.numpy()]
                rows.append(dict(final_coherent_indices_equal_baseline=torch.equal(indices,prior["coherent_source_indices"]),
                    final_coherent_UV_bitexact_baseline=torch.equal(uv,prior["coherent_source_uv"]),
                    final_differs_from_precoherent_indices=not torch.equal(indices,row["source_indices"]),
                    final_differs_from_precoherent_UV=not torch.equal(uv,row["actual_source_uv"]),
                    coherent_status=row["coherent_flags"]["status"], final_count=len(indices),
                    final_source_indices=indices.numpy().tolist(),final_native_UV=uv.numpy().tolist(),
                    final_distance_to_original_human_drawn_UV_px=number_summary(np.linalg.norm(uv.numpy()-points,axis=1))))
            final_case["probability_variants"][label]=rows
        # Independently remeasure the extreme case from real RGB, then execute
        # final coherence. This does not use reconstructed neural diagnostics.
        all_shadow=torch.zeros_like(uniform);all_shadow[:,classes.index("shadow_or_reflection")]=1
        direct=vg.guided_source_reading(native,all_shadow,guides,classes=classes,
            rgb_source_binding=binding,probability_source_binding=binding,guide_source_binding=binding,
            probabilities_conditioned_on_geometry=False)
        direct=vg.coherent_named_source_reading(direct)
        final_case["direct_all_shadow_vs_uniform_final_bitexact"]=[
            torch.equal(a["coherent_source_uv"],b["coherent_source_uv"])
            and torch.equal(a["coherent_source_indices"],b["coherent_source_indices"])
            for a,b in zip(baseline["readings"],direct["readings"])]
        for label,alternative_rows in case["candidate_ranking_counterfactual"].items():
            stats=[]
            for old,alt,g in zip(baseline["readings"],alternative_rows,guides):
                old_idx=old["coherent_source_indices"].numpy();old_uv=old["coherent_source_uv"].numpy()
                alt_idx=np.asarray(alt["source_indices"]);alt_uv=np.asarray(alt["selected_native_UV"])
                common,old_at,new_at=np.intersect1d(old_idx,alt_idx,return_indices=True)
                human=np.asarray(g["points_uv"])
                old_error=np.linalg.norm(old_uv[old_at]-human[common],axis=1)
                new_error=np.linalg.norm(alt_uv[new_at]-human[common],axis=1)
                added=np.flatnonzero(~np.isin(alt_idx,old_idx))
                stats.append(dict(common_source_positions=len(common),
                    original_final_error_common_px=number_summary(old_error),
                    alternative_error_common_px=number_summary(new_error),
                    improved_positions=int((new_error<old_error-1e-6).sum()),
                    worsened_positions=int((new_error>old_error+1e-6).sum()),
                    equal_positions=int((np.abs(new_error-old_error)<=1e-6).sum()),
                    additional_positions=len(added),additional_position_indices=alt_idx[added].tolist(),
                    additional_error_px=number_summary(np.linalg.norm(alt_uv[added]-human[alt_idx[added]],axis=1)),
                    original_final_positions_missing_in_alternative=int((~np.isin(old_idx,alt_idx)).sum())))
            final_case["pairwise_candidate_ranking_comparison"][label]=stats
        renamed=copy.deepcopy(base)
        for row in renamed["readings"]:row["expected_id"]=row["physical_id"]="COUNTERFACTUAL_WRONG_PHYSICAL_OBJECT"
        renamed=vg.coherent_named_source_reading(renamed)
        final_case["renamed_ID_final_UV_bitexact"]=[torch.equal(a["coherent_source_uv"],b["coherent_source_uv"])
            for a,b in zip(baseline["readings"],renamed["readings"])]
        actual_draw=SourceRinkReader.draw(rgb,dict(source={"components":[]},guided=baseline))
        Image.fromarray(actual_draw[y:y+h,x:x+w]).save(out/f"production_final_draw_{pts}.png")
        old_panel=Image.open(out/f"vision_causal_probe_{pts}.png").convert("RGB")
        middle=Image.fromarray(rgb[y:y+h,x:x+w]).convert("RGB");d=ImageDraw.Draw(middle)
        for g,row in zip(guides,baseline["readings"]):
            for u,v in np.asarray(g["points_uv"])-[x,y]:d.ellipse((u-1,v-1,u+1,v+1),fill=(255,0,0))
            for u,v in row["coherent_source_uv"].numpy()-[x,y]:d.ellipse((u-1,v-1,u+1,v+1),fill=(0,255,255))
        old_panel.paste(middle,(w,46));d=ImageDraw.Draw(old_panel)
        d.rectangle((w,22,2*w-1,44),fill=(18,24,30))
        d.text((w+8,24),"Red: human; cyan: FINAL production coherent UV",fill="white")
        old_panel.save(out/f"vision_causal_probe_final_{pts}.png")
        final_report["cases"].append(final_case)
        print(json.dumps(dict(source_pts=pts,final=[{k:v for k,v in a.items() if k not in ("final_source_indices","final_native_UV")} for a in final_case["probability_variants"]["real"]],
            common_comparison=final_case["pairwise_candidate_ranking_comparison"]["real"]),ensure_ascii=False),flush=True)
    final_report["script_sha256"]=sha(__file__)
    report["prior_field_final_RGB_reader_ablation_is_pre_coherent_stage_only"]=True
    report["postprocess_probe"]=final_report
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    (out/"vision_causal_final_postprocess.json").write_text(json.dumps(final_report,ensure_ascii=False,indent=2),encoding="utf-8")
    print("Production final postprocessing completed without neural inference",flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--postprocess-only",action="store_true")
    args = parser.parse_args()
    out = pathlib.Path(args.output); out.mkdir(parents=True, exist_ok=True)
    if args.postprocess_only:
        postprocess_only(out)
        return
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.manual_seed(20261007)
    model_path = DATA / "rink_vision_candidate_11200_user_iceedge2.pt"
    print("Loading frozen production-reading checkpoint on CPU", flush=True)
    saved = torch.load(model_path, map_location="cpu", weights_only=True)
    network = RinkVisionNet(**saved["model_config"])
    network.load_state_dict(saved["state_dict"], strict=True)
    network.eval(); classes = saved["classes"]
    report = dict(schema="rinkspace.independent-vision-causal-probe.v1", model_path=str(model_path),
        model_sha256=sha(model_path), step=saved["training_report"]["latest_step"],
        device="cpu", dtype="float32", torch_threads=2, cuda_model_or_tensor_created=False,
        production_geometry_only_device_validator_overridden=True,
        production_RGB_reader_calculation_and_filters_unchanged=True,
        weights_trained_or_modified=False, source_RGB_or_label_files_modified=False,
        limitations=["Two local source crops, not whole-video or held-out accuracy.",
                    "CPU FP32 forward differs from production CUDA BF16 numerically.",
                    "Pixel/identity ablations isolate the existing guided reader; global proposal generation may depend on neural probabilities.",
                    "Human drawings supply reference positions and guide identities; selected UV errors remain distances to drawings, not new ground truth."],
        source_code_sha256={str(p.relative_to(ROOT)): sha(p) for p in [
            ROOT/"src/rinkspace/vision_network.py", ROOT/"src/rinkspace/vision.py",
            ROOT/"src/rinkspace/vision_geometry.py", ROOT/"src/rinkspace/vision_supervision.py"]}, cases=[])
    vg._cuda_tensor = cpu_admission
    all_panels = []
    settings = [(228372480, (1280, 256, 512, 512), "user_contact_lower_board_base_strip_native228372480_crop1430_1640_v1", "contact"),
                (230507776, (0, 512, 512, 512), "F3_outer_user_230507776", "F3_outer")]
    for pts, crop, prefix, name in settings:
        started = time.monotonic()
        sample = next(s for s in saved["source_manifest"]["samples"] if s["source_pts"] == pts)
        source_path = pathlib.Path(sample["frame_path"])
        if sha(source_path) != sample["frame_sha256"]:
            raise ValueError("Frozen checkpoint source image hash differs")
        rgb = np.asarray(Image.open(source_path).convert("RGB"))
        native = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].float()/255
        x, y, w, h = crop
        patch = native[:, :, y:y+h, x:x+w][:, None]
        print(f"Source {pts}: CPU encoding actual full native context and local {name} ablations", flush=True)
        with torch.inference_mode():
            context = network.encode_context(native[:, None])
            zero_context = network.encode_context(torch.zeros_like(native[:, None]))
            common = dict(context_crop=crop, context_native_hw=rgb.shape[:2])
            real = network(patch, context_features=context, **common)
            zero = network(torch.zeros_like(patch), context_features=zero_context, **common)
            shuffled = network(patch.flip(-1), context_features=[f.flip(-1) for f in context], **common)
            zero_hidden = network(patch, hidden=torch.zeros_like(real["hidden"]), context_features=context, **common)
            wrong_hidden = network(patch, hidden=zero["hidden"], context_features=context, **common)
            outputs = dict(real=real, zero_image=zero, flipped_image_and_context=shuffled,
                           explicit_zero_hidden=zero_hidden, wrong_image_hidden=wrong_hidden)
            probs = {k:v["logits"][:, 0].float().softmax(1) for k,v in outputs.items()}
            target = classes.index("ice_contact" if name == "contact" else "ice_paint")
            case = dict(source_pts=pts, source_time_seconds=pts/15360, source_image=str(source_path),
                source_image_sha256=sha(source_path), native_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                crop_native_xywh=list(crop), neural_ablation={}, final_RGB_reader_ablation={},
                target_class=classes[target], guides=[])
            selected = [a for a in _annotations_in_crop(sample, (x,y), (h,w)) if a["group_id"].startswith(prefix)]
            if not selected: raise ValueError("No original annotated positions in crop")
            uv_native = np.concatenate([a["uv"] for a in selected])+[x,y]
            baseline_prob = probs["real"]
            for label, probability in probs.items():
                point_p = sampled_probabilities(probability, uv_native, crop)
                case["neural_ablation"][label] = dict(
                    crop_argmax_changed_fraction=float((probability.argmax(1)!=baseline_prob.argmax(1)).float().mean()),
                    mean_absolute_probability_difference=float((probability-baseline_prob).abs().mean()),
                    target_probability_at_original_human_UV=tensor_summary(point_p[:,target]),
                    original_human_UV_target_class_fraction=float((point_p.argmax(1)==target).float().mean()),
                    target_class_crop_pixel_fraction=float((probability.argmax(1)==target).float().mean()))
            # Distinguish class correctness at annotated points from sharp
            # localization. Measure original normals; never fit a new edge.
            normals=np.concatenate([a["normal"] for a in selected])
            offsets=np.arange(-24.,24.01,.5)
            profile_uv=uv_native[:,None,:]+normals[:,None,:]*offsets[None,:,None]
            profile_prob=sampled_probabilities(baseline_prob,profile_uv.reshape(-1,2),crop)[:,target].reshape(len(uv_native),-1).numpy()
            local_logits=real["logits"][:,0]
            xy=torch.from_numpy(profile_uv.reshape(-1,2)-np.asarray(crop[:2])).float()
            grid=xy*xy.new_tensor([2/(w-1),2/(h-1)])-1
            scores=F.grid_sample(local_logits,grid[None,:,None],align_corners=True)[0,:,:,0].T
            rest=[i for i in range(len(classes)) if i!=target]
            odds=(scores[:,target]-torch.logsumexp(scores[:,rest],dim=1)).reshape(len(uv_native),-1).numpy()
            peak=offsets[odds.argmax(1)]
            widths={}
            for threshold in [.5,.9]:
                lengths=[];clipped=0
                for row in profile_prob:
                    centre=len(offsets)//2
                    if row[centre]<threshold:lengths.append(0.);continue
                    left=right=centre
                    while left>0 and row[left-1]>=threshold:left-=1
                    while right<len(offsets)-1 and row[right+1]>=threshold:right+=1
                    lengths.append((right-left+1)*.5)
                    clipped+=left==0 or right==len(offsets)-1
                widths[str(threshold)]=dict(width_native_px=number_summary(lengths),normal_range_clipped_count=clipped)
            case["neural_localization_shape"]=dict(normal_scan_range_px=[-24,24],step_px=.5,
                contiguous_target_probability_band_containing_original_UV=widths,
                raw_class_vs_rest_log_odds_peak_distance_from_human_px=number_summary(np.abs(peak)),
                raw_class_vs_rest_log_odds_peak_signed_offset_px=number_summary(peak),
                class_probability_is_not_boundary_accuracy=True)
            default_packet=confirmed_source_guides(sample, return_packet=True)
            case["default_guide_adapter_result"]=dict(guide_count=len(default_packet["guides"]),
                unresolved=default_packet["unresolved"])
            roles={"F3:outer": "filled_disc_boundary"} if name=="F3_outer" else None
            case["experimental_role_definitions"]=roles
            supplied_guides=confirmed_source_guides(sample,role_definitions=roles)
            guides = [g for g in supplied_guides if str(g.get("source_group_id", "")).startswith(prefix)]
            if not guides:
                guides = [g for g in supplied_guides if name=="F3_outer" and g["id"]=="F3:outer"]
            measured_guides = []
            for guide in guides:
                guide=copy.deepcopy(guide); points=np.asarray(guide["points_uv"])
                keep=np.flatnonzero((points[:,0]>=x+25)&(points[:,0]<x+w-25)&(points[:,1]>=y+25)&(points[:,1]<y+h-25))[::4]
                if not len(keep): continue
                for key in ["points_uv","normals","segment_ids","original_source_point_indices"]:
                    if key in guide: guide[key]=np.asarray(guide[key])[keep].tolist()
                guide["segment_ids"]=[0]*len(keep)
                measured_guides.append(guide)
                case["guides"].append(dict(id=guide["id"],source_group_id=guide.get("source_group_id"),
                    original_positions=len(points), measured_positions=len(keep), physical_role=guide["physical_role"]))
            if not measured_guides: raise ValueError("No guides selected")
            binding=measured_guides[0]["source_binding"]
            fields=dict(probs)
            fields["uniform_probabilities"]=torch.full_like(baseline_prob, 1/len(classes))
            fields["all_shadow_probabilities"]=torch.zeros_like(baseline_prob)
            fields["all_shadow_probabilities"][:,classes.index("shadow_or_reflection")]=1
            results={}
            for label, probability in fields.items():
                full=torch.full((1,len(classes),*rgb.shape[:2]),1/len(classes))
                full[:,:,y:y+h,x:x+w]=probability
                results[label]=vg.guided_source_reading(native,full,measured_guides,classes=classes,
                    rgb_source_binding=binding,probability_source_binding=binding,guide_source_binding=binding,
                    probabilities_conditioned_on_geometry=False)["readings"]
            for label, rows in results.items():
                details=[]
                for base,row,guide in zip(results["real"],rows,measured_guides):
                    points=np.asarray(guide["points_uv"])
                    indices=row["source_indices"].numpy()
                    details.append(dict(**compare_readings(base,row),
                        distance_to_original_human_drawn_UV_px=number_summary(
                            np.linalg.norm(row["actual_source_uv"].numpy()-points[indices],axis=1)),
                        neural_type_weak_count=int(row["model_disagreement"]["type_weak"].sum()),
                        neural_shadow_high_count=int(row["model_disagreement"]["shadow_high"].sum()),
                        physical_id=row["physical_id"],selected_native_UV=row["actual_source_uv"].numpy().tolist()))
                case["final_RGB_reader_ablation"][label]=details
            # Independent minimal alternative: retain only production observed
            # RGB peaks/width validity, but use the learned class evidence to
            # rank competing material transitions. No projected/human UV is
            # used as ranking distance. This is an ablation, not a promotion.
            case["candidate_ranking_counterfactual"]={}
            for label in ["real","zero_image","flipped_image_and_context","wrong_image_hidden"]:
                ranking=[]
                for base,guide in zip(results["real"],measured_guides):
                    pool=base["source_peak_candidates"]
                    locations=pool["actual_peak_uv"]
                    sources=pool["source_indices"]
                    acceptable=pool["source_valid"] & ~pool["blur"]
                    candidate_prob=sampled_probabilities(probs[label],locations.numpy(),crop)[:,target]
                    index_list=[];location_list=[];evidence=[]
                    for source_index in range(len(guide["points_uv"])):
                        choices=torch.nonzero((sources==source_index)&acceptable,as_tuple=False).flatten()
                        if not len(choices):continue
                        winner=choices[candidate_prob[choices].argmax()]
                        index_list.append(source_index);location_list.append(locations[winner].tolist())
                        evidence.append(float(candidate_prob[winner]))
                    picked=np.asarray(location_list).reshape(-1,2)
                    original=np.asarray(guide["points_uv"])[index_list]
                    ranking.append(dict(accepted_count=len(index_list), source_indices=index_list,
                        selected_native_UV=picked.tolist(),
                        distance_to_original_human_drawn_UV_px=number_summary(np.linalg.norm(picked-original,axis=1)),
                        learned_probability_at_selected_RGB_peak=number_summary(evidence),
                        human_positions_used_as_ranking_distance=False, physical_ID_inferred_by_ranking=False))
                case["candidate_ranking_counterfactual"][label]=ranking
            # Counterfactual ID changes only annotation strings, never RGB, UV,
            # classes, network predictions or numeric training loss targets.
            changed_guides=copy.deepcopy(measured_guides)
            for g in changed_guides:g["id"]="COUNTERFACTUAL_WRONG_PHYSICAL_OBJECT"
            full=torch.full((1,len(classes),*rgb.shape[:2]),1/len(classes))
            full[:,:,y:y+h,x:x+w]=baseline_prob
            renamed=vg.guided_source_reading(native,full,changed_guides,classes=classes,
                rgb_source_binding=binding,probability_source_binding=binding,guide_source_binding=binding,
                probabilities_conditioned_on_geometry=False)["readings"]
            altered=copy.deepcopy(selected)
            for a in altered:a["feature_id"]="COUNTERFACTUAL_WRONG_PHYSICAL_OBJECT"
            loss_real,_=interval_supervision(real["logits"][:,0],selected,classes)
            loss_wrong,_=interval_supervision(real["logits"][:,0],altered,classes)
            primitive_real=primitive_supervision(real["primitive_logits"][:,0],selected)
            primitive_wrong=primitive_supervision(real["primitive_logits"][:,0],altered)
            localization_real,_=confirmed_position_interval_localization_loss(real["logits"][:,0],selected,classes,
                                    original_sample=sample,crop_origin_xy=(x,y))
            altered_sample=copy.deepcopy(sample)
            for a in altered_sample["annotations"]:a["feature_id"]="COUNTERFACTUAL_WRONG_PHYSICAL_OBJECT"
            localization_wrong,_=confirmed_position_interval_localization_loss(real["logits"][:,0],altered,classes,
                                    original_sample=altered_sample,crop_origin_xy=(x,y))
            case["physical_identity_counterfactual"]=dict(
                original_physical_ids=[r["physical_id"] for r in results["real"]],
                renamed_physical_ids=[r["physical_id"] for r in renamed],
                numeric_UV_unchanged=[compare_readings(a,b) for a,b in zip(results["real"],renamed)],
                semantic_loss_original=float(loss_real),semantic_loss_wrong_identity=float(loss_wrong),
                primitive_loss_original=float(primitive_real),primitive_loss_wrong_identity=float(primitive_wrong),
                localization_loss_original=float(localization_real),localization_loss_wrong_identity=float(localization_wrong))
            panel=draw_panel(rgb,crop,measured_guides,results["real"],baseline_prob[0,target],
                              f"PTS {pts}; {name}; CPU actual network / original RGB reader")
            panel.save(out/f"vision_causal_probe_{pts}.png")
            all_panels.append(panel)
            case["elapsed_seconds"]=time.monotonic()-started
            report["cases"].append(case)
            del outputs, probs, results, real, zero, shuffled, wrong_hidden, zero_hidden, context, zero_context
        (out/"vision_causal_probe.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
        print(json.dumps(dict(source_pts=pts,seconds=case["elapsed_seconds"],
            uniform_final=[{k:v for k,v in r.items() if k!="selected_native_UV"} for r in case["final_RGB_reader_ablation"]["uniform_probabilities"]],
            neural_reranking=[{k:v for k,v in r.items() if k not in ("selected_native_UV","source_indices")} for r in case["candidate_ranking_counterfactual"]["real"]],
            identity_counterfactual=case["physical_identity_counterfactual"]),ensure_ascii=False),flush=True)
    sheet=Image.new("RGB",(max(p.width for p in all_panels),sum(p.height for p in all_panels)),(18,24,30))
    y=0
    for panel in all_panels:sheet.paste(panel,(0,y));y+=panel.height
    sheet.save(out/"vision_causal_probe_original_RGB.png")
    report["diagnostic_image"]=str(out/"vision_causal_probe_original_RGB.png")
    report["script_sha256"]=sha(__file__)
    (out/"vision_causal_probe.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print("Completed independent CPU probe",out,flush=True)


if __name__=="__main__":main()
