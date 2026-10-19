"""Immutable frame protection and whole measurement-family development roles."""
from pathlib import Path
import argparse,gzip,json
from collections import Counter
from .io_utils import read_jsonl,write_json,object_sha256,file_sha256

DENY_TRAIN={"reserved_pending_lineage","validation_guard","sealed_validation"}

class FrameRoleGuard:
    def __init__(self,manifest):
        self.document=json.loads(Path(manifest).read_text(encoding="utf8"))
        self.roles={int(k):v for k,v in self.document["frame_roles"].items()}
    def assert_training_pixels(self,pts,*,shot_id=None):
        items=list(pts)
        for p in items:
            if isinstance(p,bool) or not isinstance(p,int) or p not in self.roles:
                raise ValueError("Unknown or non-native exposure requested")
            v=self.roles[p]
            if v["role"] in DENY_TRAIN: raise ValueError(f"Protected source exposure {p}: {v['role']}")
            if shot_id is not None and v["shot_id"]!=shot_id: raise ValueError("Cross-cut source consumption")
        return True
    def assert_training_track(self,first,last,*,shot_id):
        if last<first or (last-first)%256: raise ValueError("Invalid track PTS interval")
        return self.assert_training_pixels(range(first,last+1,256),shot_id=shot_id)

def lock_development_sources(normalized,schedule,out):
    normalized=Path(normalized);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    frames=list(read_jsonl(schedule));roles={}
    for f in frames:
        p=int(f["native_pts"]); role=f["validation_reservation_role"]
        if f["shot_id"]=="s01":role="historical_regression"
        roles[p]={"shot_id":f["shot_id"],"role":role}
    if len(roles)!=8434:raise ValueError("Incomplete programme scope")
    protection={"schema":"rinkfit.joint_source_frame_protection.v1","source_video_sha256":"e8b53966d56e72f3842d92c3ecbe6e9cc7e4670b7d0c9bd12f58fd2cff5ad57d",
        "frame_roles":{str(k):v for k,v in sorted(roles.items())},"historical_lineage_audit_complete":False,
        "independence_certified":False,"no_reserved_or_guard_pixels_for_training":True,
        "role_counts":dict(Counter(v["role"] for v in roles.values()))}
    expected={"historical_regression":1248,"development_candidate":6006,"validation_guard":560,"reserved_pending_lineage":620}
    if protection["role_counts"]!=expected:raise ValueError("Reservation schedule changed")
    write_json(out/"frame_roles.json",protection)
    guard=FrameRoleGuard(out/"frame_roles.json");locked={};families={}
    for row in read_jsonl(normalized/"observations.jsonl.gz"):
        rid=row["source_row_id"];p=int(row["native_pts"])
        guard.assert_training_pixels([p],shot_id=row["shot_id"])
        family=object_sha256([row["shot_id"],p,row["source_packet_sha256"],row["source_feature_id"]])
        if rid in locked:raise ValueError("Duplicate stable source identity")
        locked[rid]={"source_row_id":rid,"fit_role":"development","validation_family_id":family,
            "source_packet_sha256":row["source_packet_sha256"],"native_pts":p,
            "independence_claimed":False,"whole_feature_family_locked":True}
        families.setdefault(family,[]).append(rid)
    if len(locked)!=110350:raise ValueError("New source ledger incomplete")
    with gzip.open(out/"locked_roles.json.gz","wt",encoding="utf8") as f:
        json.dump(locked,f,ensure_ascii=False,separators=(",",":"))
    result={"schema":"rinkfit.joint_source_role_lock.v1","source_rows":len(locked),"measurement_feature_families":len(families),
        "all_selected_source_roles":"development","whole_family_no_random_representative_split":True,
        "protected_frames_pending_lineage":620,"guard_frames":560,"s01_historical_regression_frames":1248,
        "frame_roles":str(out/"frame_roles.json"),"frame_roles_sha256":file_sha256(out/"frame_roles.json"),
        "locked_roles":str(out/"locked_roles.json.gz"),"locked_roles_sha256":file_sha256(out/"locked_roles.json.gz"),
        "independence_certified":False,"reserved_historical_lineage_audit_pending":True,
        "training_consumers_must_use":"FrameRoleGuard.assert_training_pixels / assert_training_track"}
    write_json(out/"manifest.json",result);return result

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--normalized",required=True);p.add_argument("--schedule",required=True);p.add_argument("--out",required=True)
    a=p.parse_args();print(json.dumps(lock_development_sources(a.normalized,a.schedule,a.out),ensure_ascii=False))
