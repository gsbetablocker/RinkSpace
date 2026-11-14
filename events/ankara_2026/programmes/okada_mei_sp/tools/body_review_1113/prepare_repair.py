from pathlib import Path
import hashlib, json, os
import numpy as np

run=Path(os.environ["RINKSPACE_AUDIT_RUN"])
out=Path(os.environ.get("RINKSPACE_OKADA_BODY_REVIEW", run.parent/"audit_af48_1113")).resolve()
def read(p):return json.loads(Path(p).read_text("utf8"))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")
original=run/"surface_ready/source_evidence_detail.json"
prepared=run/"prepared_refined"
parent=run/"opening_direction_1113/refit/solve/candidate.npz"
receipt=read(out/"parent_surface/surface.json")
assert receipt["parameter_sha256"]==sha(parent)
assert receipt["cache_sha256"]==sha(out/"parent_surface/surface.npz")
doc=read(original)
review=read(out/"source_review/index.json")
findings={
 47.28:"后侧鞋与近侧鞋局部重叠，鞋底与钢片不能当完整轮廓",
 47.36:"后侧鞋可见鞋口和后跟弧，前段被近侧鞋挡住",
 47.44:"后侧鞋可见后跟与鞋口，前部遮挡，蓝色回投偏到近侧鞋",
 47.52:"两鞋仍重叠且运动模糊，只可核对后侧鞋口位置",
 57.06:"白鞋轮廓清楚，刀刃较模糊，近横向可辨",
 57.12:"脚已转向端视并模糊，不能赋精确刃方向和接触侧",
 69.38:"可辨右向露出金属和下部轮廓，接触不确定",
 69.44:"低位鞋有显著倾斜，露出钢片部分可辨，不能从画面分辨刃侧",
 69.50:"鞋姿继续转变，前后向受端视和模糊限制",
 84.64:"支撑鞋由近正视转向侧视，钢片方向弱",
 84.76:"支撑鞋侧视，前后和下缘可见",
 84.82:"同一支撑鞋保持侧视，近端刀齿与冰面影像部分混合",
 84.94:"同一支撑鞋开始由侧面转向端视",
 84.98:"接近端视，鞋底横向展开明显缩短",
 85.00:"支撑鞋近端视，横向刀投影与原片不符",
 85.04:"端视偏斜，支撑鞋前后仍不确定",
 89.04:"侧视可见支撑鞋钢片和后跟，接冰侧不确定",
 92.04:"侧视可见装配和钢片，源片支持局部接近冰面",
 97.06:"侧视清晰，鞋帮、鞋底、两安装点和钢片可分辨",
 100.84:"部分正侧视，刀架结构可见，局部受另一只鞋模糊影响",
 101.64:"切镜前两只鞋均可辨，右脚回投靠向相邻鞋，需要连续复核",
 101.66:"切镜后右脚接近端视且模糊，只给可见整体中心区间",
 101.72:"端视可见后跟和钢片小区域，不提供精确前后方向",
 119.30:"侧视斜抬，实际钢片低于蓝色回投",
 119.36:"两鞋掠过重叠，不能以同帧区域完整推定姿态",
 119.40:"近侧轮廓受模糊和亮钢反射影响，保留身份不确定",
 157.26:"右脚跟踪圈到落冰的另一鞋，右鞋实际抬起",
 158.26:"同 anchor 的抬起鞋位置符合右脚轨迹，模糊不支持分刃",
 158.44:"同 anchor 的抬起鞋与小腿相连，方向可辨度低",
 172.88:"快速腿摆造成横向模糊，只可检查整体区域",
 173.22:"同段快速腿摆，尾尖钢片与背景混合",
}
for row in review["rows"]:
    t=row["seconds"]
    if t in findings:
        row["visible_source"]=findings[t]
    elif t<26:
        row["visible_source"]="开场足部出画或随后入画，24.74至24.76未见与模型平移脉冲相符的突然移动"
    else:
        row["visible_source"]="跳入旋转的双腿和手臂可见，快速姿态变化，留作二维开发对照"
    row["role"]="development"
review["independent_final_check"]=0
review["canonical_excluded"]=[
    dict(pts=round(t*12800),foot=f,reason=why)
    for t,f,why in [(157.26,1,"wrong skate instance"),(47.44,0,"far skate partially hidden"),
                     (47.52,0,"far skate partially hidden"),(119.36,1,"crossing and occlusion"),
                     (119.40,1,"crossing and motion blur")]
]
review["roles_preserved"]={"157.26_right":0,"158.26_right":1,"158.44_right":1}
write(out/"evidence_amendment.json",review)
focus=[(22.12,25.18),(56.94,57.26),(69.20,69.66),(81.30,92.86),
       (47.20,47.60),(96.94,97.18),(100.68,101.84),(119.22,119.54),
       (44.82,45.16),(56.20,56.46),(111.24,111.54),(136.02,136.34),
       (159.26,159.72),(172.74,173.66)]
legs=[-1,0,1,2,3,4,5,6,7,8,9,10,11,12]
tracks=[list(range(-1,22)),legs,legs,legs]+[legs]*(len(focus)-4)
tb=[1,12800]
model=read(prepared/"model_identity.json")
spec=dict(schema="rinkspace.okada-scoped-repair.v1",parent_sha256=sha(parent),
          parent_evidence=str(original),parent_evidence_sha256=sha(original),
          model_path=model["path"],model_sha256=model["sha256"],
          source_video_sha256=doc["source_sha256"],source_time_base=tb,
          parent_result=str(parent.parents[1]/"fitted/result.npz"),
          parent_result_sha256="9df455f41f036d24b7ca2c3b57c0cc9e4ece026818a60d1a77d8ea54781fa6b3",
          focus_pts=[[round(x*12800) for x in window] for window in focus],
          tracks_by_window=tracks,opening_pts=[283136,322304],
          opening_acceleration_scale_mps2=18.,
          ice=dict(scale_m=.02,tolerance_m=dict(runner=.009,toe_pick=.012,solid=.009,upper=.010),
                   rationale="Existing model finite ice band plus conditional scene/assembly margin. No new support classification."),
          weights=dict(programme_component_ice=3.,programme_root_curvature=1.,actual_model_com_acceleration=.18),
          guard=dict(body_median_increase_px=.2,body_p90_increase_px=.5,torso_p90_increase_px=.7,
                     maximum_new_trusted_source_sigmas=3.,component_height_numerical_m=.0004,
                     anatomy_new_component_budget=.045,anatomy_peak_budget=.06,cavity_peak_budget_m=.0015,
                     movement_budget_ratio=1.08,world_angular_step_budget_deg=3.,
                     opening_root_peak_mps2=110.,opening_com_peak_mps2=65.),
          amendment=str(out/"evidence_amendment.json"),amendment_sha256=sha(out/"evidence_amendment.json"))
doc["scoped_candidate_import"]=spec
doc["trajectory_reference"]="model_com"
doc["com_motion_policy"]=dict(quadrature="native_source_centres",acceleration_scale_mps2=30.)
s=doc["programme_structure"]
s.update(surface_witnesses=str(out/"parent_surface/surface.npz"),surface_witnesses_sha256=receipt["cache_sha256"])
for key in ("archived_source_direction","numerical_trust_seed"):
    s["solver_step"].pop(key,None)
outpath=out/"revised_source.json"
write(outpath,doc)
write(out/"policy.json",spec)
print("Revised objective",sha(outpath),"with",len(focus),"native windows")
