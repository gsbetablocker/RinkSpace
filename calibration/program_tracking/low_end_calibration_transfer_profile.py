"""Test shot03 dual-edge C/lens transfer over all three complete low_end shots.

The original conditioned 11199-factor graph, source UV, source-supported
motion, priors, quality and robust rules stay fixed. Only each frame's R3 and
mean(logfx, logfy) vary; the dual119 mount C and lens ratio/pp/skew are exact
constants. Default invocation only preflights. ``--run`` requires the matching
saved preflight and performs one restricted solve. This is a conditional
transfer test, not evidence that the frozen camera calibration is true.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time

import gtsam
import numpy as np

import conditioned_camera_comparison as driver
from scipy.optimize import least_squares
import conditioned_shot03_lens19_seed as seed_driver
import shot03_fixed_calibration_profile as profile119
from backend import IncrementalCameraGraph
from projection import from_state, project_fast
from rig_parameterization import camera25 as derived_camera25, physical_frame
from source_supported_motion import load_supported_motion


TMP = seed_driver.TMP
OUTPUT = TMP / "low_end_dual_calibration_transfer_profile_v1"
DUAL = TMP / "shot03_board_edges_interval_derived_v1/fixed"
DUAL_SUMMARY_SHA256 = "3bc2868b2624167692a56be879a9929703c8208a3675d095b41b9c0117bd29c5"
DUAL_STATES_SHA256 = "2d639f850cf2cf9e2fe9fbb6ce5cf367281639f8e7759437ca5687bdabac97d2"
BASE = TMP / "conditioned_low_end_source_support_v2/fixed"
BASE_SUMMARY_SHA256 = "2a789b43cb61e292a978db240707d48f4a177d747da1adf2c842d7c073b130db"
BASE_STATES_SHA256 = "cbd9718a518888ffd9ee1f46be5f9cfda54603a4d816e552c44f784370addf18"
SUPPORT = TMP / "source_board_support_v2/reviewed_shot05/source_board_motion_support_reviewed.json"
SUPPORT_SHA256 = "23a99eca7ca14f3bedd2ab13bafda409e8e9f209bdd363e7b966e99881271c32"
FALLBACK = TMP / "source_board_support_v2/reviewed_shot05/raw_zero_pair_fallback_reviewed.json"
FALLBACK_SHA256 = "1abb893e2d42e585404a44ccaa64e4ddee54426550de388a5c8b2eca2ca5e3a4"
FAILED_INITIAL_GEOMETRY = OUTPUT / "initial_native_geometry.json"
FAILED_INITIAL_GEOMETRY_SHA256 = "c3b8638320b183f56dfad1f7f71aa9790d523141128397706a28345fa8a85470"
ADAPTED_INITIAL = OUTPUT / "adapted_initial_4d_states.npz"
ADAPTATION_REPORT = OUTPUT / "initial_adaptation.json"
EXPECTED_SHOTS = (3, 5, 8)
EXPECTED_COUNTS = {3:(119,118,8132), 5:(1653,1652,72376), 8:(447,446,24323)}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_state_packet(path: Path, expected_sha: str, count: int) -> dict[int,np.ndarray]:
    require(driver.digest(path) == expected_sha, f"Frozen state packet changed: {path}")
    with np.load(path, allow_pickle=False) as packet:
        keys = np.asarray(packet["keys"], np.int64)
        states = np.asarray(packet["states"], np.float64)
    require(keys.shape == (count,) and states.shape == (count,11)
            and len(set(map(int,keys))) == count and np.isfinite(states).all(),
            f"Invalid frozen Vector11 states: {path}")
    return {int(k):v.copy() for k,v in zip(keys,states)}


class TransferProfile:
    """Affine 4D frame embedding with two fixed original Vector11 rig nodes."""

    def __init__(self, data: dict, baseline: dict[int,np.ndarray],
                 dual: dict[int,np.ndarray], dual_summary: dict):
        self.frames = tuple(sorted(map(int,data["pts"])))
        self.frame_set = set(self.frames)
        self.mount, self.lens = int(data["mount"]), int(data["lens"])
        self.baseline = baseline
        require(len(self.frames) == 2219 and len(data["active"]) == 2221
                and set(baseline) == set(data["active"])
                and set(dual) == set(map(int,data["shots"][0]["pts"])) | {self.mount,self.lens},
                "Transfer states do not cover one complete low_end camera group")
        self.constants = {self.mount:dual[self.mount].copy(),
                          self.lens:dual[self.lens].copy()}
        require(np.max(np.abs(dual[self.mount]-dual_summary["mount_state"])) < 1e-12
                and np.max(np.abs(dual[self.lens]-dual_summary["lens_state"])) < 1e-12,
                "Dual summary C/lens differs from frozen solved states")
        self.C = self.constants[self.mount][:3]*100.
        self.ratio = float(self.constants[self.lens][6])
        self.common = self.constants[self.lens][8:11]*1000.
        self.T = np.zeros((11,4))
        self.T[3:6,:3] = np.eye(3)
        self.T[6:8,3] = 1.
        self.offset = np.zeros(11)
        self.offset[6:8] = (self.ratio/2.,-self.ratio/2.)
        self.offset[8:11] = self.common/1000.
        # The old all-shot and dual119 R/mean-logf estimates set only initial
        # values. Their prior targets and every original observation stay intact.
        self.initial4 = {}
        self.initial_source_counts = Counter()
        for stamp in self.frames:
            source = dual if stamp in dual else baseline
            label = "dual119" if stamp in dual else "source_support_v2_2219"
            self.initial_source_counts[label] += 1
            old = source[stamp]
            self.initial4[stamp] = np.r_[old[3:6], np.mean(old[6:8])]
        require(self.initial_source_counts == {"dual119":119,
                "source_support_v2_2219":2100}, "Unexpected transfer initial sources")

    def frame(self, q: np.ndarray) -> np.ndarray:
        value = np.asarray(q,np.float64)
        require(value.shape == (4,) and np.isfinite(value).all(),
                "Transfer frame must have 4 finite variables")
        return self.offset+self.T@value

    def expand(self, keys, reduced_states):
        values = iter(reduced_states)
        return [self.frame(next(values)) if key in self.frame_set
                else self.constants[key] for key in keys]

    def full_states(self, frames4: dict[int,np.ndarray]):
        return {**self.constants,
                **{stamp:self.frame(frames4[stamp]) for stamp in self.frames}}


def load_inputs(output: Path):
    require(driver.digest(DUAL/"summary.json") == DUAL_SUMMARY_SHA256
            and driver.digest(BASE/"summary.json") == BASE_SUMMARY_SHA256,
            "A reviewed dual119 or source-v2 full summary changed")
    dual_summary = json.loads((DUAL/"summary.json").read_text(encoding="utf8"))
    base_summary = json.loads((BASE/"summary.json").read_text(encoding="utf8"))
    require(dual_summary["shot_indices"] == [3] and dual_summary["frames"] == 119
            and dual_summary["states"] == 121 and dual_summary["factors"] == 633
            and base_summary["shot_indices"] == [3,5,8]
            and base_summary["frames"] == 2219 and base_summary["states"] == 2221
            and base_summary["factors"] == 11199
            and dual_summary["source_candidate_model_sha256"] ==
                base_summary["source_candidate_model_sha256"]
            and base_summary["supported_motion"]["sha256"] == SUPPORT_SHA256
            and base_summary["supported_motion_fallback"]["sha256"] == FALLBACK_SHA256,
            "Reviewed original/dual solve source, scene or support differs")
    baseline = load_state_packet(BASE/"states.npz",BASE_STATES_SHA256,2221)
    dual = load_state_packet(DUAL/"states.npz",DUAL_STATES_SHA256,121)
    data = driver.prepare(seed_driver.CONFIG,seed_driver.CANDIDATE,
                          seed_driver.MANIFEST,EXPECTED_SHOTS,output,
                          complete_camera_group=True)
    require(data["shot_indices"] == EXPECTED_SHOTS and data["camera_id"] == "low_end"
            and len(data["pts"]) == 2219 and len(data["active"]) == 2221
            and data["captured"]["source_hash"] == data["diag"]["source_sha256"]
            and data["diag"]["model_sha256"] ==
                dual_summary["source_candidate_model_sha256"],
            "Prepared low_end group differs from dual source/scene")
    data["supported_motion"],data["supported_motion_metadata"] = load_supported_motion(
        data,SUPPORT,seed_driver.MANIFEST,SUPPORT_SHA256)
    fallback,data["supported_motion_fallback_metadata"] = load_supported_motion(
        data,FALLBACK,seed_driver.MANIFEST,FALLBACK_SHA256)
    data["supported_motion"].extend(fallback)
    require(data["supported_motion_metadata"]["pair_count"] == 248
            and data["supported_motion_metadata"]["point_count"] == 3761
            and data["supported_motion_fallback_metadata"]["pair_count"] == 6
            and data["supported_motion_fallback_metadata"]["point_count"] == 54,
            "Frozen v2 254-pair supported-motion repair changed")
    pairs,counts,by_shot,_,validity = driver.native_pairs(data)
    require(counts == {"pixels":104831,"adjacent_pairs":2216}
            and len(pairs) == 2216 and validity["invalid_pair_count"] == 0
            and all((len(shot["pts"]),by_shot[str(shot["index"])]["adjacent_pairs"],
                     by_shot[str(shot["index"])]["pixels"]) == EXPECTED_COUNTS[shot["index"]]
                    for shot in data["shots"]),
            "Original full2219 native motion or board validity changed")
    profile = TransferProfile(data,baseline,dual,dual_summary)
    return data,pairs,profile,dual_summary,base_summary


def old_camera_projection_targets(data,pairs,profile: TransferProfile):
    """Use old solved cameras only to choose visible scene points for an initializer."""
    old_mount = profile.baseline[profile.mount]
    old_physical = {stamp:physical_frame(profile.baseline[stamp],old_mount,"fixed")[0]
                    for stamp in profile.frames}
    shot_by_pts = {int(stamp):int(shot["index"]) for shot in data["shots"]
                   for stamp in shot["pts"]}
    cloud = {stamp:[] for stamp in profile.frames if shot_by_pts[stamp] in (5,8)}
    targets = {stamp:[] for stamp in cloud}
    pair_for_frame = {}
    old_visibility_failures = []
    for pair_index,(a,b,pair) in enumerate(pairs):
        if a not in cloud and b not in cloud:
            continue
        pair_for_frame.setdefault(a,pair)
        pair_for_frame.setdefault(b,pair)
        hits=pair.source_hits(old_physical[a])
        if not np.all(hits.valid):
            bad=np.flatnonzero(~hits.valid)
            old_visibility_failures.append(dict(pair_index=pair_index,
                pts=[a,b],first_row=int(bad[0]),reason=hits.reason[int(bad[0])]))
            continue
        xyz=hits.xyz
        for stamp,template in ((a,pair.camera0_template),(b,pair.camera1_template)):
            if stamp not in cloud:
                continue
            old_camera=from_state(old_physical[stamp],template)
            projected,depth=project_fast(old_camera,xyz)
            visible=(np.isfinite(projected).all(axis=1)&(depth>0.)
                     &(projected[:,0]>=0.)&(projected[:,0]<1920.)
                     &(projected[:,1]>=0.)&(projected[:,1]<1080.))
            if np.any(visible):
                cloud[stamp].append(xyz[visible].copy())
                targets[stamp].append(projected[visible].copy())
    require(not old_visibility_failures,
            f"Old solved camera no longer yields native source board hits: "
            f"{old_visibility_failures[:4]}")
    # A few reviewed native links contain only one or two board points. Borrow
    # scene XYZ selected by neighboring source UV, but keep a point only if the
    # old solved camera for this frame sees that board face in the image. This
    # is still a synthetic old-view target used solely for initialization.
    supplemental = {}
    frames_by_shot={index:[int(p) for p in shot["pts"]]
                    for shot in data["shots"]
                    for index in [int(shot["index"])]}
    for stamp in cloud:
        current=sum(len(part) for part in cloud[stamp])
        if current>=16:
            continue
        camera=from_state(old_physical[stamp],data["captured"]["templates"][stamp])
        same_shot=frames_by_shot[shot_by_pts[stamp]]
        for neighbor in sorted(same_shot,key=lambda p:(abs(p-stamp),p)):
            if neighbor==stamp or not cloud[neighbor]:
                continue
            xyz=np.concatenate(cloud[neighbor])
            projected,depth=project_fast(camera,xyz)
            visible=(np.isfinite(projected).all(axis=1)&(depth>0.)
                     &(projected[:,0]>=0.)&(projected[:,0]<1920.)
                     &(projected[:,1]>=0.)&(projected[:,1]<1080.))
            if np.any(visible):
                tested=np.flatnonzero(visible)
                rays=xyz[tested]-camera[:3]
                board_hits=pair_for_frame[stamp]._hits_from_rays(camera[:3],rays)
                visible[tested] &= (board_hits.valid &
                    (np.linalg.norm(board_hits.xyz-xyz[tested],axis=1)<1e-4))
            if np.any(visible):
                cloud[stamp].append(xyz[visible].copy())
                targets[stamp].append(projected[visible].copy())
                supplemental[stamp]=supplemental.get(stamp,0)+int(visible.sum())
                current+=int(visible.sum())
            if current>=16:
                break
    result={}
    for stamp in cloud:
        require(cloud[stamp],f"No old visible board points for PTS {stamp}")
        xyz=np.concatenate(cloud[stamp]); uv=np.concatenate(targets[stamp])
        require(len(xyz)>=8 and xyz.shape==(len(uv),3) and uv.shape[1]==2
                and np.isfinite(xyz).all() and np.isfinite(uv).all(),
                f"Too few valid old projected board anchors at {stamp}")
        result[stamp]=(xyz,uv)
    require(len(result)==2100,"Old-view board targets missed a full low_end shot")
    return result,dict(frames_supplemented=len(supplemental),
                       supplemental_old_visible_xyz=int(sum(supplemental.values())),
                       max_supplement_per_frame=int(max(supplemental.values(),default=0)))


def fit_old_view(stamp:int, xyz:np.ndarray, old_uv:np.ndarray,
                 profile:TransferProfile, template:np.ndarray):
    """Optimize only an initializer to preserve the old camera's projected FOV."""
    start=profile.initial4[stamp].copy()
    mount=profile.constants[profile.mount]
    cache={}
    def evaluate(q):
        key=np.asarray(q,np.float64).tobytes()
        if cache.get("key")!=key:
            physical=physical_frame(profile.frame(q),mount,"fixed")[0]
            camera=from_state(physical,template)
            uv,depth,jac=project_fast(camera,xyz,with_jacobian=True)
            if not np.isfinite(uv).all() or not np.isfinite(jac).all() or np.any(depth<=0.):
                raise ValueError(f"New calibration projection invalid at PTS {stamp}")
            cache.clear()
            cache.update(key=key,residual=(uv-old_uv).ravel(),jacobian=jac@profile.T)
        return cache
    before=evaluate(start)["residual"].reshape(-1,2)
    solved=least_squares(lambda q:evaluate(q)["residual"],start,
        jac=lambda q:evaluate(q)["jacobian"],method="lm",max_nfev=50,
        ftol=1e-10,xtol=1e-10,gtol=1e-10)
    require(np.isfinite(solved.x).all() and np.isfinite(solved.fun).all(),
            f"Old-view initializer is nonfinite at PTS {stamp}")
    after=evaluate(solved.x)["residual"].reshape(-1,2)
    singular=np.linalg.svd(solved.jac,compute_uv=False)
    rank=int(np.sum(singular>singular[0]*1e-8))
    require(rank==4,f"Old-view initializer has deficient 4D geometry at {stamp}: {singular}")
    info=dict(pts=stamp,old_visible_board_xyz=len(xyz),
        old_camera_projection_target_only=True,source_pixels_not_fit_targets=True,
        initial_projection_p95_px=float(np.percentile(np.linalg.norm(before,axis=1),95)),
        adapted_projection_p95_px=float(np.percentile(np.linalg.norm(after,axis=1),95)),
        adapted_projection_max_px=float(np.max(np.linalg.norm(after,axis=1))),
        rvec_delta_deg=float(np.degrees(np.linalg.norm(solved.x[:3]-start[:3]))),
        mean_logf_delta=float(solved.x[3]-start[3]),
        jacobian_rank=rank,nfev=int(solved.nfev),status=int(solved.status),
        scipy_success=bool(solved.success))
    return solved.x.copy(),info


def load_or_build_adaptation(data,pairs,profile:TransferProfile,*,allow_build:bool):
    """Freeze an old-view-only initializer once; subsequent runs only reload it."""
    require(driver.digest(FAILED_INITIAL_GEOMETRY)==FAILED_INITIAL_GEOMETRY_SHA256,
            "Original 3934-no-hit initial geometry report must stay untouched")
    expected=dict(dual_summary_sha256=DUAL_SUMMARY_SHA256,
        dual_states_sha256=DUAL_STATES_SHA256,
        old_full_summary_sha256=BASE_SUMMARY_SHA256,
        old_full_states_sha256=BASE_STATES_SHA256,
        original_failed_geometry_sha256=FAILED_INITIAL_GEOMETRY_SHA256,
        source_support_sha256=SUPPORT_SHA256,
        source_support_fallback_sha256=FALLBACK_SHA256,
        fixed_scene_sha256=data["diag"]["model_sha256"],
        original_source_sha256=data["captured"]["source_hash"])
    if ADAPTATION_REPORT.exists():
        require(ADAPTED_INITIAL.exists(),"Adapted initializer report has no NPZ")
        report=json.loads(ADAPTATION_REPORT.read_text(encoding="utf8"))
        require(report["schema"]=="rinkspace.low-end-old-view-initializer.v1"
                and report["input_sha256"]==expected
                and report["output_npz_sha256"]==driver.digest(ADAPTED_INITIAL)
                and report["projection_targets_are_only_initialization"] is True
                and report["original_uv_and_graph_unchanged"] is True,
                "Saved old-view initializer identity/provenance changed")
        with np.load(ADAPTED_INITIAL,allow_pickle=False) as packet:
            stamps=np.asarray(packet["pts"],np.int64)
            q4=np.asarray(packet["q4"],np.float64)
        require(np.array_equal(stamps,np.asarray(profile.frames))
                and q4.shape==(2219,4) and np.isfinite(q4).all(),
                "Cached old-view initializer PTS/4D states changed")
        profile.initial4={int(p):q.copy() for p,q in zip(stamps,q4)}
        return report
    require(allow_build and not ADAPTED_INITIAL.exists(),
            "Run cannot rebuild or overwrite the old-view initializer")
    targets,supplement=old_camera_projection_targets(data,pairs,profile)
    by_source={}
    for (a,b,pair) in pairs:
        by_source.setdefault(a,[]).append(pair)
    failed=json.loads(FAILED_INITIAL_GEOMETRY.read_text(encoding="utf8"))
    prototype=[]
    selected=[]
    for item in failed["first_failures"]:
        stamp=int(item["source_pts"])
        if stamp in targets and stamp not in selected:
            selected.append(stamp)
        if len(selected)==5:
            break
    require(len(selected)==5,"Original failure report lacks five prototype PTS")
    fitted={}
    for stamp in selected:
        q,info=fit_old_view(stamp,*targets[stamp],profile,
            data["captured"]["templates"][stamp])
        fitted[stamp]=q
        physical=physical_frame(profile.frame(q),profile.constants[profile.mount],"fixed")[0]
        source_failures=[]
        for pair in by_source.get(stamp,[]):
            hits=pair.source_hits(physical)
            source_failures += [dict(point_index=int(i),source_uv=pair.xy0[i].tolist(),
                                     reason=hits.reason[int(i)])
                                for i in np.flatnonzero(~hits.valid)[:8]]
        prototype.append({**info,"source_board_hit_failures":source_failures[:8]})
    prototype_path=OUTPUT/"initializer_prototype.json"
    require(not prototype_path.exists(),"Old-view prototype already exists")
    prototype_path.write_text(json.dumps(dict(schema="rinkspace.low-end-old-view-prototype.v1",
        input_sha256=expected,pts=selected,rows=prototype,
        no_original_uv_removed=True,solver_started=False),indent=2),encoding="utf8")
    require(all(not row["source_board_hit_failures"] for row in prototype),
            f"Five-frame old-view initializer still misses source board: {prototype_path}")
    metrics=list(prototype)
    for stamp in profile.frames:
        if stamp not in targets:
            continue
        if stamp not in fitted:
            q,info=fit_old_view(stamp,*targets[stamp],profile,
                data["captured"]["templates"][stamp])
            fitted[stamp]=q
            metrics.append(info)
        profile.initial4[stamp]=fitted[stamp]
    require(len(fitted)==2100,"Old-view initializer missed shot05/08 frames")
    stamps=np.asarray(profile.frames,np.int64)
    q4=np.asarray([profile.initial4[int(p)] for p in stamps])
    np.savez_compressed(ADAPTED_INITIAL,pts=stamps,q4=q4,
        source_sha256=data["captured"]["source_hash"],
        model_sha256=data["diag"]["model_sha256"],
        method="old_solved_camera_board_hit_projection_targets_only")
    counts=np.asarray([row["old_visible_board_xyz"] for row in metrics],int)
    before=np.asarray([row["initial_projection_p95_px"] for row in metrics],float)
    after=np.asarray([row["adapted_projection_p95_px"] for row in metrics],float)
    report=dict(schema="rinkspace.low-end-old-view-initializer.v1",
        input_sha256=expected,output_npz_path=str(ADAPTED_INITIAL),
        output_npz_sha256=driver.digest(ADAPTED_INITIAL),
        prototype_path=str(prototype_path),prototype_sha256=driver.digest(prototype_path),
        prototype_pts=selected,frames_from_dual119=119,
        frames_adapted_from_old_full=2100,
        visible_board_xyz_per_frame=[int(counts.min()),float(np.median(counts)),int(counts.max())],
        projection_p95_before_over_frames_px=[float(np.percentile(before,50)),float(np.percentile(before,95)),float(before.max())],
        projection_p95_after_over_frames_px=[float(np.percentile(after,50)),float(np.percentile(after,95)),float(after.max())],
        projection_targets_are_only_initialization=True,
        old_solved_cameras_define_target_projections=True,
        source_pixels_only_select_old_visible_board_xyz=True,
        sparse_frame_neighbor_xyz=supplement,
        original_uv_and_graph_unchanged=True,solver_started=False)
    ADAPTATION_REPORT.write_text(json.dumps(report,indent=2),encoding="utf8")
    return report


def new_calibration_geometry(pairs, profile: TransferProfile, data:dict) -> dict:
    """Evaluate every source/target board ray at the proposed initial cameras."""
    mount = profile.constants[profile.mount]
    physical = {stamp:physical_frame(profile.frame(profile.initial4[stamp]),
                               mount,"fixed")[0] for stamp in profile.frames}
    failures = {3:[],5:[],8:[]}
    reason_counts = Counter()
    per_shot = {shot:Counter() for shot in (3,5,8)}
    shot_by_pts={int(stamp):int(shot["index"]) for shot in data["shots"]
                 for stamp in shot["pts"]}
    checked = 0
    for pair_index,(a,b,pair) in enumerate(pairs):
        shot=shot_by_pts[a]
        require(shot==shot_by_pts[b],"Native pair crossed a physical shot")
        per_shot[shot]["pairs"] += 1
        per_shot[shot]["original_uv"] += len(pair.xy0)
        try:
            valid,reasons = pair.initial_validity([physical[a],physical[b]])
        except Exception as exc:
            failures[shot].append(dict(pair_index=pair_index,source_pts=a,target_pts=b,
                reason="initial_validity_exception",exception_type=type(exc).__name__,
                detail=str(exc)[:400]))
            reason_counts["initial_validity_exception"] += len(pair.xy0)
            per_shot[shot]["invalid_rows"] += len(pair.xy0)
            per_shot[shot]["invalid_pairs"] += 1
            continue
        require(valid.shape == (len(pair.xy0),),
                f"Native validity dimension changed: {a}/{b}")
        checked += len(valid)
        bad = np.flatnonzero(~valid)
        per_shot[shot]["invalid_rows"] += len(bad)
        per_shot[shot]["invalid_pairs"] += bool(len(bad))
        for local in bad:
            reason_counts[reasons[int(local)]] += 1
            if len(failures[shot]) < 12:
                failures[shot].append(dict(pair_index=pair_index,source_pts=a,target_pts=b,
                    point_index=int(local),source_uv=pair.xy0[local].tolist(),
                    target_uv=pair.xy1[local].tolist(),reason=reasons[int(local)]))
    result = dict(pair_count=len(pairs),original_uv_count=104831,
        rows_checked=checked,invalid_rows=int(sum(reason_counts.values())),
        invalid_reason_counts=dict(reason_counts),
        by_shot={str(k):dict(v) for k,v in per_shot.items()},
        first_failures_by_shot={str(k):v for k,v in failures.items()},
        no_rows_dropped=True,geometry_valid=(not reason_counts))
    return result


def original_graph(data,pairs):
    original_class = driver.IncrementalCameraGraph
    driver.IncrementalCameraGraph = profile119.RecordingGraph
    try:
        graph,_ = driver.build_graph(data,pairs,"fixed","derived")
    finally:
        driver.IncrementalCameraGraph = original_class
    require(len(graph._pending) == len(graph.ledger) == 11199,
            "Original 11199-factor graph or ledger order changed")
    return graph


def reduced_graph(data,original,profile: TransferProfile):
    config = data["config"]
    graph = IncrementalCameraGraph(4,window_size=len(profile.frames)+2,
        relinearize_threshold=1e-5,refinement_updates=config["refinement_updates"],
        optimization=config["optimization"],diagnose_cost=True)
    variable_map = {}
    constants = []
    for index,spec in enumerate(original.ledger):
        if profile119.pure_constant(index,spec,data,profile):
            constants.append(index)
            continue
        active = tuple(k for k in spec["keys"] if k in profile.frame_set)
        require(active,f"Unclassified constant factor: {index}")
        raw,raw_jac,options = profile119.observation(spec)
        def residual(q,keys=spec["keys"],fn=raw):
            return fn(profile.expand(keys,q))
        def jacobian(q,keys=spec["keys"],fn=raw_jac):
            matrices = fn(profile.expand(keys,q))
            return [np.asarray(matrices[i],float)@profile.T
                    for i,key in enumerate(keys) if key in profile.frame_set]
        graph.add_observation(active,residual,
            jac_fn=jacobian if raw_jac is not None else None,**options)
        variable_map[index] = len(graph._pending)-1
    require(len(constants)+len(variable_map) == 11199
            and len(graph._pending) == len(variable_map),
            "Transfer graph lost an original factor")
    return graph,variable_map,constants


def equivalence(original,reduced,variable_map,constants,profile,data):
    # Reuse the reviewed 119-frame equality/constant-perturbation implementation;
    # its computations are size-independent and only its display count is fixed.
    result = profile119.verify_factors(original,reduced,variable_map,constants,profile,data)
    require(len(original._pending) == 11199 and len(variable_map)+len(constants) == 11199,
            "Full objective factor count changed")
    result["original_factors"] = len(original._pending)
    require(result["max_per_factor_cost_difference"] < 1e-8
            and result["max_constant_factor_cost_perturbation"] < 1e-8,
            "Original and 4D objective costs differ")
    return result


def representative_jacobians(data,pairs,original,reduced,variable_map,profile):
    """Check fixed representative source/ad/motion factor chain rules."""
    retained = len(data["retained"])
    native_start = retained+len(profile.frames)
    by_shot = {int(stamp):int(shot["index"]) for shot in data["shots"]
               for stamp in shot["pts"]}
    native = {shot:next(native_start+j for j,(a,b,_) in enumerate(pairs)
                        if by_shot[a] == shot)
              for shot in (3,5,8)}
    selected = {}
    for shot in (3,5,8):
        for name,sigma in (("top",2.),("contact",3.)):
            selected[f"shot{shot:02d}_{name}"] = next(
                i for i,s in enumerate(data["retained"])
                if (s["file"],s["function"]) ==
                   ("track_sequence.py","_add_source_edge_factors")
                and by_shot[int(s["keys"][0])] == shot
                and float(s["kwargs"]["sigma"]) == sigma and i in variable_map)
        selected[f"shot{shot:02d}_ad"] = next(
            i for i,s in enumerate(data["retained"])
            if (s["file"],s["function"]) == ("ad_landmarks.py","factor")
            and by_shot[int(s["keys"][0])] == shot and i in variable_map)
        selected[f"shot{shot:02d}_native"] = native[shot]
    full_values = profile119.values(original,profile.full_states(profile.initial4))
    small_values = profile119.values(reduced,profile.initial4)
    report = {}
    for label,index in selected.items():
        require(index in variable_map,f"Representative factor became constant: {label}")
        spec = original.ledger[index]
        keys = tuple(k for k in spec["keys"] if k in profile.frame_set)
        reduced_factor = reduced._pending[variable_map[index]]
        original_linear = original._pending[index].linearize(full_values)
        reduced_linear = reduced_factor.linearize(small_values)
        original_A = original_linear.getA()
        original_keys = list(original_linear.keys())
        mapped_A = np.hstack([original_A[:,
            original_keys.index(original._key(key))*11:
            (original_keys.index(original._key(key))+1)*11]@profile.T for key in keys])
        A_error = float(np.max(np.abs(mapped_A-reduced_linear.getA())))
        b_error = float(np.max(np.abs(original_linear.getb()-reduced_linear.getb())))
        require(A_error < 1e-8 and b_error < 1e-8,
                f"Whitened A@T/b mismatch for {label}: {A_error}/{b_error}")
        raw,raw_jac,_ = profile119.observation(spec)
        q = [profile.initial4[key].copy() for key in keys]
        analytic = [np.asarray(j,float)@profile.T for key,j in
                    zip(spec["keys"],raw_jac(profile.expand(spec["keys"],q)))
                    if key in profile.frame_set]
        steps = []
        for h in (1e-6,1e-7,1e-8):
            largest = 0.
            largest_relative = 0.
            worst = None
            for block in range(len(q)):
                for column in range(4):
                    hi=[v.copy() for v in q]; lo=[v.copy() for v in q]
                    hi[block][column] += h; lo[block][column] -= h
                    numeric=(np.asarray(raw(profile.expand(spec["keys"],hi)),float)
                            -np.asarray(raw(profile.expand(spec["keys"],lo)),float))/(2*h)
                    difference=np.abs(numeric-analytic[block][:,column])
                    row=int(np.argmax(difference))
                    if float(difference[row]) > largest:
                        largest=float(difference[row])
                        largest_relative=float(difference[row]/
                            (1.+abs(analytic[block][row,column])))
                        worst=dict(state_key=int(keys[block]),column=column,row=row)
            steps.append(dict(h=h,max_absolute_error=largest,
                              worst_relative_error=largest_relative,worst=worst))
        require(steps[1]["max_absolute_error"] < 1e-3
                and steps[1]["worst_relative_error"] < 1e-4,
                f"Representative 4D finite difference failed: {label}/{steps}")
        report[label] = dict(original_index=index,keys=list(keys),
            whitened_A_chain_error=A_error,whitened_b_error=b_error,
            fd=steps)
    return report


def preflight(output: Path, *, allow_build_initializer: bool):
    require(output.resolve().is_relative_to(TMP.resolve()),
            "Transfer profile output must remain in project tracking TMP")
    data,pairs,profile,dual_summary,base_summary = load_inputs(output)
    adaptation = load_or_build_adaptation(data,pairs,profile,
                                          allow_build=allow_build_initializer)
    geometry = new_calibration_geometry(pairs,profile,data)
    geometry_path = output/"adapted_initial_native_geometry.json"
    if not geometry_path.exists():
        geometry_path.write_text(json.dumps(geometry,indent=2),encoding="utf8")
    else:
        require(json.loads(geometry_path.read_text(encoding="utf8")) == geometry,
                "Saved initial board geometry differs")
    require(geometry["geometry_valid"],
            f"New calibration invalidates original native board rows: {geometry_path}")
    original = original_graph(data,pairs)
    reduced,variable_map,constants = reduced_graph(data,original,profile)
    equality = equivalence(original,reduced,variable_map,constants,profile,data)
    jacobians = representative_jacobians(data,pairs,original,reduced,
                                        variable_map,profile)
    phase = dict(schema="rinkspace.low-end-dual-calibration-transfer-profile.v1",
        scope="Complete low_end shots 03/05/08, fixed dual119 C/ratio/pp/skew; only per-frame R3 and meanlogf free",
        conditional_calibration_not_accepted=True,source_uv_changed=False,
        original_priors_quality_huber_and_Brown_templates_unchanged=True,
        original_external_scene_nodes_fixed_at_v13=True,
        frames=2219,active_original_nodes=2221,original_factors=11199,
        original_native_pairs=2216,original_native_source_pixels=104831,
        fixed_C_scene=profile.C.tolist(),fixed_log_fx_over_fy=profile.ratio,
        fixed_cx_cy_skew_px=profile.common.tolist(),
        initializer_frame_sources=dict(profile.initial_source_counts),
        source_sha256=data["captured"]["source_hash"],
        scene_sha256=data["diag"]["model_sha256"],
        support=data["supported_motion_metadata"],
        fallback=data["supported_motion_fallback_metadata"],
        initial_adaptation=adaptation,
        hashes=dict(config=driver.digest(seed_driver.CONFIG),
            candidate_scene=driver.digest(seed_driver.CANDIDATE/"candidate_scene.json"),
            manifest=driver.digest(seed_driver.MANIFEST),
            dual_summary=DUAL_SUMMARY_SHA256,dual_states=DUAL_STATES_SHA256,
            base_summary=BASE_SUMMARY_SHA256,base_states=BASE_STATES_SHA256,
            support=SUPPORT_SHA256,fallback=FALLBACK_SHA256,
            adapted_initial_states=driver.digest(ADAPTED_INITIAL),
            initial_adaptation_report=driver.digest(ADAPTATION_REPORT),
            original_failed_native_geometry=FAILED_INITIAL_GEOMETRY_SHA256,
            native_geometry=driver.digest(geometry_path),
            driver=driver.digest(driver.__file__),
            original_profile_code=driver.digest(profile119.__file__),
            script=driver.digest(__file__)),
        new_calibration_native_geometry=geometry,
        factor_error_equivalence=equality,
        chain_rule_jacobian_checks=jacobians,
        solver_started=False,final_holdout_read=False)
    return data,pairs,profile,original,reduced,equality,phase


def run(data,profile,original,reduced,equality,phase,output):
    target = output/"profile"
    require(not target.exists(),"Existing transfer solve needs review")
    target.mkdir()
    start=time.perf_counter()
    variable_factors=list(reduced._pending)
    solved=reduced.update(profile.initial4)
    stamps=np.asarray(profile.frames,np.int64)
    post=target/"post_isam_4d_states.npz"
    np.savez_compressed(post,pts=stamps,q4=np.asarray([solved[int(p)] for p in stamps]))
    (target/"post_isam_summary.json").write_text(json.dumps(dict(
        stage="post_isam_pre_batch_lm",candidate_result=False,
        phase_00_sha256=driver.digest(output/"phase_00.json"),
        states_sha256=driver.digest(post),graph_update=reduced.last_update_summary,
        seconds=time.perf_counter()-start),indent=2),encoding="utf8")
    solved=reduced.batch_polish(**data["config"]["batch_polish"])
    shot_by_pts={int(stamp):int(shot["index"]) for shot in data["shots"]
                 for stamp in shot["pts"]}
    cameras=np.asarray([derived_camera25(profile.frame(solved[int(p)]),
        profile.constants[profile.mount],data["captured"]["templates"][int(p)],
        "fixed") for p in stamps])
    camera_path=target/"shot_cameras.npz"
    np.savez_compressed(camera_path,pts=stamps,
        shot_index=np.asarray([shot_by_pts[int(p)] for p in stamps],np.int16),
        camera25=cameras,source_sha256=data["captured"]["source_hash"],
        model_sha256=data["diag"]["model_sha256"])
    states4=target/"states4.npz"
    np.savez_compressed(states4,pts=stamps,
                        q4=np.asarray([solved[int(p)] for p in stamps]))
    expanded=profile.full_states(solved)
    keys=np.asarray(sorted(expanded),np.int64)
    expanded_path=target/"expanded_original_states11.npz"
    np.savez_compressed(expanded_path,keys=keys,
        states=np.asarray([expanded[int(k)] for k in keys]),
        source_sha256=data["captured"]["source_hash"],
        model_sha256=data["diag"]["model_sha256"],
        semantics="4D_restricted_transfer_expanded_into_original_2221_nodes")
    full_values=profile119.values(original,expanded)
    small_values=profile119.values(reduced,solved)
    original_total=float(sum(f.error(full_values) for f in original._pending))
    variable_total=float(sum(f.error(small_values) for f in variable_factors))
    constant_total=float(equality["pure_constant_total"])
    require(abs(original_total-variable_total-constant_total) < 1e-5,
            "Final original 11199 cost differs from variable plus constant factors")
    batch_cost=reduced.last_batch_diagnostics.get("final_nonlinear_cost")
    require(batch_cost is not None and abs(float(batch_cost)-variable_total) < 1e-5,
            "Final solver cost differs from original-factor variable cost")
    summary=dict(schema="rinkspace.low-end-dual-calibration-transfer-profile-result.v1",
        scope=phase["scope"],conditional_calibration_not_accepted=True,
        phase_00_sha256=driver.digest(output/"phase_00.json"),
        frames=2219,original_factors=11199,variable_factors=len(variable_factors),
        pure_constant_cost=constant_total,source_sha256=phase["source_sha256"],
        scene_sha256=phase["scene_sha256"],fixed_C_scene=profile.C.tolist(),
        fixed_log_fx_over_fy=profile.ratio,fixed_cx_cy_skew_px=profile.common.tolist(),
        graph_update=reduced.last_update_summary,batch=reduced.last_batch_diagnostics,
        final_original_factor_total=original_total,
        final_variable_factor_total=variable_total,
        final_constant_factor_total=constant_total,
        max_final_total_difference=abs(original_total-variable_total-constant_total),
        camera_sha256=driver.digest(camera_path),states4_sha256=driver.digest(states4),
        expanded_states_sha256=driver.digest(expanded_path),
        post_isam_sha256=driver.digest(post),seconds=time.perf_counter()-start,
        process_id=os.getpid(),not_programme_acceptance=True)
    path=target/"summary.json"
    path.write_text(json.dumps(summary,indent=2),encoding="utf8")
    print(json.dumps(dict(phase="solve_complete",summary=str(path),
        sha256=driver.digest(path),comparable_original_factor_cost=original_total)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=OUTPUT)
    parser.add_argument("--run",action="store_true")
    args=parser.parse_args()
    output=args.output
    require(output.resolve().is_relative_to(TMP.resolve()),
            "Transfer output must stay in project tracking TMP")
    if args.run:
        require((output/"phase_00.json").is_file(),
                "Run requires a reviewed matching preflight")
    else:
        require(not (output/"phase_00.json").exists(),
                "Preflight already exists; use a new output directory")
    output.mkdir(parents=True,exist_ok=True)
    data,pairs,profile,original,reduced,equality,phase=preflight(
        output,allow_build_initializer=not args.run)
    phase_path=output/"phase_00.json"
    if args.run:
        require(json.loads(phase_path.read_text(encoding="utf8")) == phase,
                "Frozen transfer phase differs from current source/code/objective")
    else:
        phase_path.write_text(json.dumps(phase,indent=2),encoding="utf8")
    print(json.dumps(dict(phase="preflight",path=str(phase_path),
        sha256=driver.digest(phase_path),geometry=phase["new_calibration_native_geometry"],
        factor_equivalence=equality["max_per_factor_cost_difference"],
        variable_factors=len(reduced._pending),constant_factors=11199-len(reduced._pending))),flush=True)
    if args.run:
        run(data,profile,original,reduced,equality,phase,output)


if __name__ == "__main__":
    main()
