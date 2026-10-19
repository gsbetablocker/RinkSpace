"""Reserve new native images before any template/camera work touches their pixels.

Reservations are not automatically independent test data: the source/template
lineage worker must first rule out historical use. They do immediately prevent
the new training/flow/template jobs from consuming these windows.
"""
from __future__ import annotations
from collections import Counter
from .io_utils import object_sha256


def reserve_native_windows(scope, selected_frames, *, seed="okada-joint9-20261018-v1",
                           window_frames=10, guard_frames=5, fraction=.2):
    if not seed or window_frames<1 or guard_frames<0 or not 0<fraction<1:
        raise ValueError("Invalid validation reservation policy")
    known={f["native_pts"] for f in selected_frames}
    windows=[];unavailable=[]
    for s in scope["shots"]:
        if s["id"]=="s01":continue
        candidates=[]
        for lo in range(s["first_pts"],s["last_pts"]+1,window_frames*256):
            hi=min(lo+(window_frames-1)*256,s["last_pts"])
            if hi-lo<4*256:continue
            guard_lo=max(s["first_pts"],lo-guard_frames*256)
            guard_hi=min(s["last_pts"],hi+guard_frames*256)
            if any(guard_lo<=p<=guard_hi for p in known):continue
            digest=object_sha256([seed,s["id"],lo,hi])
            candidates.append({"shot_id":s["id"],"first_pts":lo,"last_pts":hi,
                "guard_first_pts":guard_lo,"guard_last_pts":guard_hi,
                "draw":int(digest[:16],16)/2**64,"reservation_id":digest[:20]})
        chosen=[c for c in candidates if c["draw"]<fraction]
        if not chosen and candidates: chosen=[min(candidates,key=lambda c:c["draw"])]
        if not chosen: unavailable.append(s["id"])
        for c in chosen:
            c.pop("draw")
            c.update(status="reserved_pending_historical_lineage_audit",
                use_for_training_or_templates=False,
                track_through_window_for_training=False,
                test_after_physical_state_frozen=True)
            windows.append(c)
    roles={p:"development_candidate" for s in scope["shots"]
           for p in range(s["first_pts"],s["last_pts"]+1,256)}
    for s in scope["shots"]:
        if s["id"]=="s01":
            for p in range(s["first_pts"],s["last_pts"]+1,256):roles[p]="historical_regression"
    for w in windows:
        for p in range(w["guard_first_pts"],w["guard_last_pts"]+1,256):
            if roles[p]=="development_candidate":roles[p]="validation_guard"
    for w in windows:
        for p in range(w["first_pts"],w["last_pts"]+1,256):roles[p]="reserved_pending_lineage"
    assert all(roles[p] not in {"validation_guard","reserved_pending_lineage"} for p in known)
    result={"schema":"rinkfit.native_validation_reservation.v1","seed":seed,
        "window_frames":window_frames,"guard_frames":guard_frames,"fraction_of_eligible_windows":fraction,
        "windows":windows,"frame_role_counts":dict(Counter(roles.values())),
        "shots_without_eligible_reservation":unavailable,"independence_certified":False,
        "source_family_locked_roles_status":"must finalise before training/template/camera initialisation",
        "historical_lineage_failure_action":"mark as development; reserve truly unused evidence before numerical tuning",
        "no_validation_pixels_used_in_this_architecture_run":True}
    result["reservation_sha256"]=object_sha256(result)
    return result,roles


def split_jobs_by_reservation(jobs, roles):
    result=[]
    for job in jobs:
        start=job["first_pts"];role=roles[start]
        ends=[]
        for pts in range(start+256,job["last_pts"]+257,256):
            if pts>job["last_pts"] or roles[pts]!=role:
                part={**job,"id":f'{job["shot_id"]}/{start}-{pts-256}/{role}',
                    "first_pts":start,"last_pts":pts-256,"frame_count":(pts-start)//256,
                    "reservation_role":role,"use_pixels_for_training_or_templates":role in {
                        "development_candidate","historical_regression"},
                    "training_flow_may_cross_reservation":False}
                if role in {"reserved_pending_lineage","validation_guard"}:
                    part["action"]="defer_source_measurement_until_lineage_locked_and_model_frozen"
                    part["status"]="reserved_do_not_use_for_training"
                # Context anchors are not allowed to lure a future bidirectional
                # flow worker through a held-out window after interval splitting.
                for key, endpoint in (("left_anchor_pts",start),("right_anchor_pts",pts-256)):
                    anchor=part.get(key)
                    if anchor is not None and any(roles.get(p)!=role for p in
                            range(min(anchor,endpoint),max(anchor,endpoint)+1,256)):
                        part["unusable_cross_reservation_"+key]=anchor
                        part[key]=None
                        part["restart_source_detection_on_this_side"]=True
                result.append(part)
                start=pts
                if pts<=job["last_pts"]:role=roles[pts]
    if sum(j["frame_count"] for j in result)!=sum(j["frame_count"] for j in jobs):
        raise AssertionError("Reservation split changed programme coverage")
    return result
