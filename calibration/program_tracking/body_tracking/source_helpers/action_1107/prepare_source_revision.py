from pathlib import Path
import hashlib, json
from calibration.program_tracking.body_tracking.programme_action_prepare import prepare
from rinkspace.path_policy import guard_project_write
BASE=Path(r"D:/CodexData/RinkSpace_Data/runs/body_s01_20261029/trajectory_action_1107")
ROOT=Path(r"D:/CodexData/RinkSpace_Data/runs/body_s01_20261029/trajectory_action_1107/action_completion_1107")
PROJECT=Path(r"D:/OneDrive/codex_projs/RinkSpace")
PROGRAMME=PROJECT/'events/ankara_2026/programmes/okada_mei_sp'
LEDGER=PROGRAMME/'programme_action_ledger_1107.json'
OUT=PROGRAMME/'programme_action_ledger_finish_1107.json'
ledger=json.loads(LEDGER.read_text('utf8'))
receipt=json.loads((ROOT/'native_steel_crops.json').read_text('utf8'))
def pts(s):return round(s*12800)
def episode(id,kind,first,last,evidence,description,*,foot=None,status='inferred'):
    return dict(episode_id=id,kind=kind,source_id=ledger['source_id'],time_base=[1,12800],
                start_pts=pts(first),end_pts_inclusive=pts(last),boundary_uncertainty={},
                per_foot_state={k:('supported' if foot==i else 'unknown') for i,k in enumerate(('left','right'))},
                direction_hypotheses=[],evidence_pts=[pts(x) for x in evidence],
                evidence_source=[str(ROOT/'steps_source/review.json'),str(ROOT/'native_steel_crops.json')],
                evidence_status=status,confidence_semantics='原片时序和可见冰刀；具体用刃、切换边界及接触材料点保留不确定性',
                foot=foot,role='fit' if kind!='contact_transition_uncertain' else 'review',
                optimization_enabled=kind!='contact_transition_uncertain',description_zh=description)
def support(id,first,last,evidence,note,foot):
    return dict(episode(id,'support_candidate',first,last,evidence,note,foot=foot),contact_mode='normal_runner',factor_reliability=.55,
                source_edge='unknown')
new=[
    support('steps_lunge_recovery_left',147.36,147.78,[147.38,147.54,147.70],'弓步起身后，左脚连续支撑，右脚后抬。',0),
    support('steps_transfer_right',149.42,149.64,[149.46,149.62],'面向镜头转出时右脚承重，左脚外展。',1),
    episode('steps_transfer_ambiguous','contact_transition_uncertain',149.66,149.86,[149.78],'双鞋近冰，换足和蹬冰的确切曝光不确定。',status='ambiguous'),
    support('steps_transfer_left',149.88,150.64,[149.94,150.10,150.26,150.58],'换到左脚前侧滑行，右脚摆到身后；末端鞋开始转向。',0),
]
ledger['episodes'].extend(new)
by_id={e['episode_id']:e for e in ledger['episodes']}
def phase(id,first,last,reliability):
    return dict(support_episode_id=id,start_pts=pts(first),end_pts_inclusive=pts(last),reliability=reliability,edge=None,travel=None)
def relation(id,first,last,evidence,note,phases,names):
    row=episode(id,'step_relation',first,last,evidence,note)
    row.update(step_phases=phases,step_hypotheses=names)
    ledger['episodes'].append(row)
relation('steps_lunge_and_recover',146.22,147.78,[146.26,146.90,147.38,147.54,147.70],
         '左足弓步滑行与左足起身相连，跨接触疑问处仅保留连续状态。',
         [phase('steps_lunge_left',146.22,146.92,.40),phase('steps_lunge_recovery_left',147.36,147.78,.65)],
         ['glide','three_turn'])
relation('steps_weight_transfer',149.42,150.64,[149.46,149.62,149.78,149.94,150.10,150.26,150.58],
         '右足换左足。入/出段滑行关系可用，内外刃和具体步法名留多个假设。',
         [phase('steps_transfer_right',149.42,149.64,.5),phase('steps_transfer_left',149.88,150.64,.65)],
         ['transfer_glide','mohawk','choctaw'])
def observe(t,id,features):
    e=by_id[id];p=pts(t); f=e['foot']; info=receipt[str(p)]
    if p not in e['evidence_pts']:e['evidence_pts'].append(p);e['evidence_pts'].sort()
    name='step' if id.startswith('steps') else 'LSp' if id.startswith('lsp') else 'CCoSp'
    file=f'{name}_steel_{2 if (t==184.14 or (name=="step" and t>=149.94)) else 1}.jpg'
    crop=info['crop']
    use_band=any(feature=='runner_lower_edge' for feature,_,_ in features)
    fraction=(.76 if use_band else 1.)/len(features)
    for feature,points,radius in features:
        assert all(crop[0]<=x<crop[2] and crop[1]<=y<crop[3] for x,y in points),(t,crop,points)
        ledger['skate_regions'].append(dict(id=f'finish_{p}_{feature}',pts=p,foot=f,role='fit',
            optimization_enabled=True,evidence_status='observed',source_status='ambiguous',
            structure_feature=feature,polyline_uv=points,pixel_uncertainty_radius_px=radius,
            contour_frame_budget=3.2,feature_budget_fraction=fraction,source_anatomical_foot=f,
            source_track_id=id,source_identity_provenance='原片连续人体侧别链推断',
            source_order='ambiguous',visibility='partial',completeness='partial',
            source_endpoints={'material_front':None,'material_rear':None},
            source_annotation=dict(points_uv=points,closed=False,status='ambiguous',
              description_zh='同帧实际可见局部，运动模糊与反光纳入区域宽度。'),
            evidence_source=str(ROOT/file),source_rgb_sha256=info['rgb_sha256']))
        if feature=='runner_lower_edge':
            e.setdefault('native_support_bands',[]).append(dict(pts=p,foot=f,polyline_uv=points,radius_px=radius+2,
                objective_weight=.24*3.2,semantics='原片可见承重鞋下沿候选带；固定材料点与反光边界未定'))
def sole(t,id,uv,radius=5.5):
    observe(t,id,[('boot_sole',uv,radius)])
def steel(t,id,uv,radius=5.5,sole_uv=None):
    features=[('runner_lower_edge',uv,radius)]
    if sole_uv:features.append(('boot_sole',sole_uv,radius))
    observe(t,id,features)
sole(157.62,'lsp_left_runner',[[880,870],[897,870],[913,870]])
sole(158.58,'lsp_left_runner',[[921,902],[939,903],[954,903]])
sole(159.70,'lsp_left_runner',[[981,928],[1000,928],[1016,927]])
sole(161.56,'lsp_left_grab_runner',[[894,930],[911,931],[928,931]],6.)
sole(162.74,'lsp_left_grab_runner',[[941,933],[957,933],[977,931]])
steel(173.80,'ccosp_left_camel',[[924,934],[943,937],[962,936]],6.,[[927,925],[948,925],[968,925]])
steel(174.20,'ccosp_left_camel',[[895,948],[920,948],[948,946],[973,943]],6.,[[894,933],[908,933],[922,933]])
steel(177.38,'ccosp_left_sit',[[886,932],[910,935],[933,936],[956,933]],6.,[[873,921],[892,922],[913,920]])
steel(182.70,'ccosp_right_runner',[[857,924],[889,926],[924,925],[951,923]],5.5,[[912,913],[934,913],[952,911]])
steel(184.14,'ccosp_right_runner',[[819,924],[847,927],[873,926],[894,923]],5.5,[[851,914],[875,915],[892,913]])
steel(147.54,'steps_lunge_recovery_left',[[902,935],[935,937],[968,938],[998,934]],5.,[[949,919],[969,920],[990,918]])
steel(147.70,'steps_lunge_recovery_left',[[867,941],[905,945],[946,945],[980,941]],5.,[[936,923],[955,923],[976,923]])
steel(149.46,'steps_transfer_right',[[790,858],[811,861],[835,859]],6.)
steel(149.62,'steps_transfer_right',[[754,853],[782,855],[806,853]],6.)
steel(149.94,'steps_transfer_left',[[873,850],[902,851],[928,851],[960,848]],5.,[[919,838],[939,838],[957,837]])
steel(150.10,'steps_transfer_left',[[831,850],[858,852],[897,851],[925,848]],5.,[[883,841],[904,840],[921,839]])
steel(150.26,'steps_transfer_left',[[817,851],[846,854],[879,853],[910,850]],5.,[[869,840],[891,841],[908,839]])
steel(150.58,'steps_transfer_left',[[766,876],[792,879],[823,878],[851,875]],5.,[[821,863],[837,863],[851,861]])
ledger['source_review']['supplement']=dict(parent_ledger=str(LEDGER),parent_sha256=hashlib.sha256(LEDGER.read_bytes()).hexdigest(),
    steps_source=str(ROOT/'steps_source/review.json'),native_steel_crop_receipt=str(ROOT/'native_steel_crops.json'),
    observed_structures_added=sum(r['id'].startswith('finish_') for r in ledger['skate_regions']))
guard_project_write(OUT,project_root=PROJECT)
with OUT.open('x',encoding='utf8') as f:
    json.dump(ledger,f,ensure_ascii=False,indent=2)
    f.write('\n')
previous=BASE.parent/'trajectory_continue_1107/programme_source_temporal'
print(prepare(previous,OUT,ROOT/'prepared','physics'))
