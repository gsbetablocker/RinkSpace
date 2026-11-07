from pathlib import Path
import json
import hashlib

root=Path.cwd()
data=Path(r'D:/CodexData/RinkSpace_Data/runs/body_s01_20261029')
parent=data/'trajectory_continue_1107/programme_source_temporal/source_evidence.json'
raw=parent.read_bytes()
old=json.loads(raw)
sid=old['source_id']
review=data/'trajectory_action_1107'
tb=[1,12800]
pts=lambda seconds:int(round(seconds*12800))
episodes=[]
def row(name,kind,start,stop,evidence,source,*,status='observed',fit=False,foot=None,
        boundary=None,description='',**extra):
    states={'left':'unknown','right':'unknown'}
    if foot in (0,1):
        states[('left','right')[foot]]='supported' if kind=='support_candidate' else 'lifted' if kind=='free_foot_off_ice_candidate' else 'related'
    if kind=='flight_visible_candidate':
        states={'left':'airborne','right':'airborne'}
    return dict(episode_id=name,kind=kind,source_id=sid,time_base=tb,start_pts=pts(start),
        end_pts_inclusive=pts(stop),boundary_uncertainty=boundary or {},
        per_foot_state=states,foot=foot,direction_hypotheses=extra.pop('direction_hypotheses',[]),
        evidence_pts=list(map(pts,evidence)),evidence_source=source,evidence_status=status,
        confidence_semantics='可见姿态或局部像素，边界与左右链传播显式登记；无校准概率',
        role='fit' if fit else 'review',optimization_enabled=fit,description_zh=description,**extra)
def add(*args,**kwargs):
    episode=row(*args,**kwargs)
    episodes.append(episode)
    return episode
def flight(name,start,stop,evidence,source,leave,land):
    add(name,'flight_visible_candidate',start,stop,evidence,source,fit=True,
        boundary={'takeoff_pts_interval':list(map(pts,leave)),'landing_pts_interval':list(map(pts,land))},
        description='原片可见双鞋离冰的保守内部。前后不清曝光作为过渡，不串接两次腾空。')
    for kind,interval in [('takeoff',leave),('landing',land)]:
        add(name+'_'+kind,'contact_transition_uncertain',*interval,interval,source,status='ambiguous',
            description='刀与冰重叠或运动模糊，保留接触过渡区间。')
def support(name,foot,start,stop,evidence,source,reliability,description):
    return add(name,'support_candidate',start,stop,evidence,source,status='inferred',fit=True,foot=foot,
        boundary={'at_endpoint':'近似，超出所见清晰支持的曝光不外推'},
        factor_reliability=reliability,contact_mode='normal_runner',source_edge='unknown',description=description)
def lifted(name,foot,start,stop,evidence,source):
    add(name,'free_foot_off_ice_candidate',start,stop,evidence,source,status='inferred',fit=True,foot=foot,
        description='按可见两腿和鞋随时间跟踪，部分曝光身份为传播。')
def path(name,start,stop,face,source,sign,notes,view_margin=55.):
    add(name,'rotation_source_path',start,stop,[a for a,_ in face],source,status='inferred',fit=True,
        turn_direction_sign=sign,direction_hypotheses=[sign] if sign else [-1,1],
        facing_observations=[dict(pts=pts(t),view=v,tolerance_deg=view_margin) for t,v in face],
        direction_scale_radps=4.,description=notes)
combo=['review_combo_firstagent_1107','review_combo_edge_a_1107','source_overview']
axel=['review_axel_1107_fine','review_axel_takeoff_1107','review_axel_landing_1107']
flying=['review_flying_contact_1107','review_spin_sample_1107','review_support_exit_1107']
spin=['review_spins_native_child1107','review_spins_orientation_child1107','review_spins_transfer_child1107','source_overview']
flight('jump_3f',48.22,48.70,[48.22,48.30,48.42,48.50,48.62,48.70],combo,(48.12,48.22),(48.70,48.80))
flight('jump_3t',49.66,50.02,[49.66,49.78,49.90,49.98,50.02],combo,(49.52,49.66),(50.02,50.12))
flight('jump_2a',77.88,78.14,[77.88,77.92,78.00,78.04,78.08,78.12,78.14],axel,(77.68,77.76),(78.14,78.20))
flight('jump_3lz',117.32,117.72,[117.32,117.36,117.48,117.56,117.60,117.64,117.72],['review_lutz'],(117.20,117.32),(117.72,117.88))
support('combo_between_right',1,48.98,49.26,[48.98,49.10,49.26],combo,.55,'首跳后右足滑行，左脚伸出。两跳之间有实际接触。')
lifted('combo_between_left',0,49.02,49.22,[49.10],combo)
support('combo_exit_right',1,50.34,50.90,[50.34,50.50,50.90],combo,.55,'第二跳右足吸收并滑出，左腿展开。')
support('axel_left_approach',0,77.62,77.68,[77.68],axel,.4,'起跳前左足近冰，保留蹬起和具体刃侧的不确定性。')
support('axel_exit_right',1,78.52,78.90,[78.60,78.80],axel,.5,'右脚单足滑出，另一脚伸向后侧。')
support('lutz_exit_right',1,118.04,118.88,[118.08,118.24,118.56,118.88],['review_lutz'],.6,'屈膝落冰、右刀滑出，自由左腿从前向后打开。')
lifted('lutz_exit_left',0,118.04,118.88,[118.08,118.56,118.88],['review_lutz'])
add('lutz_pick_unresolved','possible_point_contact',117.20,117.30,[117.24],['review_lutz'],status='ambiguous',
    description='冰上反光和鞋重叠，只能定位起跳前末段，不把刀齿材料或精确接触脚写成真值。')
path('flip_rotation',48.22,48.70,[(48.30,'back'),(48.42,'front'),(48.50,'back'),(48.62,'front')],
    combo,0,'原图胸背交替，绝对方向候选仍保留。')
path('toe_rotation',49.66,50.02,[(49.78,'back'),(49.90,'front'),(49.98,'back')],
    combo,0,'原图胸背交替，中间姿态、圈数与精确轴未固定。')
path('axel_rotation',77.80,78.14,[(77.84,'front'),(77.88,'screen_right'),(77.92,'back'),
    (77.96,'back'),(78.00,'screen_left'),(78.04,'front'),(78.08,'screen_right'),(78.12,'back')],
    axel,1,'正面→画面右→背面→画面左→正面。方位用宽锥，不设900度或恒定转速。',48.)
path('lutz_rotation',117.32,117.72,[(117.32,'screen_left'),(117.36,'front'),(117.48,'back'),
    (117.56,'screen_left'),(117.60,'front'),(117.64,'screen_right'),(117.72,'back')],
    ['review_lutz'],1,'花饰正面与裸背交替，左侧→正面→右侧。按独立腾空处理。',55.)
path('flying_camel_path',81.62,92.72,[(82.28,'front'),(82.60,'back'),(82.96,'front'),(87.08,'front'),
    (87.36,'back'),(87.64,'front')],flying,1,'燕式旋至提刀高位，来源方向为连续可见姿态推断，躯干倾斜用宽角域。',73.)
# Low lunge in the step sequence. Support inferred from visible knee over shoe;
# the rear shoe may skim, so there is no "free-foot" assertion.
support('steps_lunge_left',0,146.22,146.92,[146.52,146.92],['source_overview/sheet_09.jpg'],.35,
    '正面低弓步中左腿在前承载、右腿向后伸展；后刀是否同时接冰保留未知。')
add('step_possible_hop','possible_flight_unresolved',133.56,133.82,[133.72],['source_overview/sheet_08.jpg'],
    status='ambiguous',description='低身交叉过渡，稀疏图不能区分短腾空和拖刃。')
lsp=support('lsp_left_runner',0,156.54,160.44,[156.66,157.32,158.26,159.38],spin,.57,'左足侧倾/后仰保持。')
lsp2=support('lsp_left_grab_runner',0,160.90,165.30,[161.32,163.86,164.82],spin,.57,'左足继续支撑，右足抬起被抓。')
lifted('lsp_right_free',1,161.50,165.26,[161.72,163.86,164.82],spin)
add('lsp_possible_double_touch','contact_transition_uncertain',160.46,160.88,[160.52,160.72,160.88],spin,
    status='ambiguous',description='两鞋接近且模糊，左右链继续，可能短暂双接触。')
add('lsp_grasp','hand_to_raised_shoe_grasp_candidate',161.32,165.26,[161.72,163.86,164.82],spin,
    status='inferred',foot=1,description='可见向自由右鞋伸手并保持；手指材料位置未确认。')
add('lsp_release','grasp_release_transition',165.32,165.88,[165.52,165.72],spin,status='ambiguous',
    description='释放并恢复滑行。')
path('lsp_turn_path',156.50,165.30,[(156.50,'back'),(156.66,'front'),(156.90,'back')],
    spin,0,'左脚为主，持续转圈；绝对方向以对称路径候选处理。',73.)
cc1=support('ccosp_left_camel',0,172.08,174.58,[172.20,172.76,173.48,174.20],spin,.63,'燕式左足支撑。')
cc2=support('ccosp_left_sit',0,174.76,179.02,[175.38,175.58,176.90,178.50,179.02],spin,.65,'左足蹲转；自由右鞋近冰，但不把模型最低脚作支撑。')
lifted('ccosp_right_free_camel',1,172.14,174.56,[172.20,173.48],spin)
add('ccosp_sit_change','posture_transition',174.60,174.74,[174.62,174.72],spin,status='inferred',
    description='燕式过渡低位，保留速度与姿势改变。')
add('ccosp_change_foot','contact_transition_uncertain',179.04,179.24,[179.04,179.16,179.18,179.22],spin,
    status='ambiguous',boundary={'old_left_clear_pts':pts(179.02),'new_right_clear_pts':pts(179.26)},
    description='原图换位模糊，不能核实精确接触或是否短离冰。')
cc3=support('ccosp_right_runner',1,179.26,184.52,[179.26,179.62,181.18,181.84,184.14],spin,.64,'换足后右足支撑，后半左自由脚高举。')
lifted('ccosp_left_held',0,180.60,184.52,[180.92,181.84,183.32,184.52],spin)
add('ccosp_left_grasp','hand_to_raised_shoe_grasp_candidate',180.52,184.52,[180.92,181.84,183.32],spin,
    status='inferred',foot=0,description='左自由鞋被握持。手指与握点材料未确定。')
add('ccosp_release','grasp_release_transition',184.54,184.92,[184.52,184.92],spin,status='ambiguous',
    description='高位释放、降腿及滑出。')
path('ccosp_first_direction',172.08,179.02,[(175.38,'front'),(175.48,'back'),(175.52,'back'),
    (175.58,'back'),(175.66,'front'),(175.72,'front')],spin,0,
    '保守源顺序可排除把单帧模型异常当真实方向切换；左右转向仍按连续性候选。',70.)
path('ccosp_second_direction',179.26,184.52,[(181.10,'back'),(181.34,'front'),(181.50,'back')],
    spin,0,'右足稳定旋后高提腿，保持实际变速并允许中心移动。',68.)
# Clear visible portions of the steel and boot/post outlines only. Coordinates
# were read in the original image from the labelled source crops.
steel=[
 (158.26,0,lsp,[[885,914],[896,914],[907,914],[917,913]],4.5,[]),
 (159.38,0,lsp,[[972,941],[985,943],[998,942],[1010,941],[1022,941]],4.5,
  [('boot_sole',[[969,928],[980,928],[991,929],[1000,929]],5.),
   ('front_stanchion',[[966,929],[965,934],[964,938]],5.)]),
 (163.86,0,lsp2,[[976,957],[992,957],[1008,954],[1022,952],[1034,950]],5.,
  [('boot_sole',[[968,943],[981,942],[995,941],[1008,940]],5.)]),
 (173.48,0,cc1,[[972,957],[989,958],[1007,957],[1023,955],[1039,952]],5.,
  [('boot_sole',[[971,940],[988,940],[1004,940],[1014,939]],5.),
   ('rear_stanchion',[[1055,932],[1056,939],[1057,946]],6.)]),
 (176.90,0,cc2,[[901,943],[917,944],[933,944],[948,943],[958,941]],5.5,[]),
 (178.50,0,cc2,[[923,917],[937,917],[951,917],[965,916]],5.5,[]),
 (179.26,1,cc3,[[1020,917],[1034,917],[1049,916],[1063,915],[1074,915]],5.5,[]),
 (181.84,1,cc3,[[921,920],[934,919],[947,919],[959,918]],5.,
  [('boot_sole',[[935,906],[946,905],[958,905],[966,905]],5.5),
   ('front_stanchion',[[966,906],[967,911],[967,917]],6.)])]
regions=[]
for t,foot,episode,uv,radius,extra in steel:
    frame_budget=3.2
    contours=[('runner_lower_edge',uv,radius)]+extra
    contact_share=.24
    fraction=(1.-contact_share)/len(contours)
    for feature,points,band in contours:
        regions.append(dict(id='s09_'+str(pts(t))+'_'+feature, pts=pts(t),foot=foot,
            role='fit',optimization_enabled=True,evidence_status='observed',source_status='ambiguous',
            structure_feature=feature,polyline_uv=points,pixel_uncertainty_radius_px=band,
            contour_frame_budget=frame_budget,feature_budget_fraction=fraction,
            source_anatomical_foot=foot,source_track_id=episode['episode_id'],
            source_identity_provenance='原片连续人体侧别链推断，非用户确认',
            source_order='ambiguous',visibility='partial',completeness='partial',
            source_endpoints={'material_front':None,'material_rear':None},
            source_annotation=dict(points_uv=points,closed=False,status='ambiguous',
                description_zh='原片可见局部边界，端点、反光和整刀接触位置不确定。'),
            evidence_source=spin[0]))
    episode.setdefault('native_support_bands',[]).append(dict(pts=pts(t),foot=foot,
        polyline_uv=uv,radius_px=radius+2.,objective_weight=frame_budget*contact_share,
        semantics='同一像素预算的条件支撑带，不是固定材料点'))
timeline=[
 ('opening_pose',22.12,23.30,'开场近景；脚及冰面截断。'),
 ('opening_movement',23.32,45.30,'转体、压步、屈伸及起跳前滑行。'),
 ('first_combination',45.32,50.12,'助滑与3F+3T，中间有右足滑出与再次起跳。'),
 ('connection_glides',50.14,77.38,'组合跳滑出、连续走弧换步及Axel助滑。'),
 ('axel',77.40,78.92,'2A进入、腾空、吸收和出跳。'),
 ('camel_entry_connection',78.94,81.28,'连步、低弓步和入旋准备。'),
 ('flying_camel',81.30,92.86,'跳入燕式旋、抓足、高位保持、释放。'),
 ('after_camel_link',92.88,116.58,'出旋后低位交叉、持续滑行及3Lz进入。'),
 ('lutz',116.60,119.10,'3Lz助滑、点冰候选、腾空和滑出。'),
 ('step_sequence',119.12,155.90,'StSq阶段范围候选；单足转体、换足、交叉、深弓步。'),
 ('layback',155.92,165.88,'LSp4入口、左足后仰旋、右鞋抓握及释放。'),
 ('between_spins',165.90,171.70,'出旋后助滑与进入组合旋。'),
 ('change_foot_spin',171.72,184.92,'CCoSp4左足燕式及蹲姿，换右足后高提自由左脚并释放。'),
 ('closing',184.94,188.68,'出旋滑行、手臂动作、结束姿态。')]
coverage=[row(i,'programme_context',a,b,[],['source_overview'],status='inferred',description=note)
          for i,a,b,note in timeline]
inherited={}
for e in old['source_episode_review']['episodes']:
    eid=e['episode_id']
    inherited[eid]=dict(source_id=sid,time_base=tb,
        evidence_source=flying if eid.startswith(('entry','spin','grasp','support_release')) else ['inherited source_episode_review'],
        evidence_status='inferred',confidence_semantics='保留既有源关联及当前原片复查，非接触真值',
        evidence_pts=[e['start_pts'],e['end_pts_inclusive']],direction_hypotheses=[])
    if eid=='entry_flight_core':
        inherited[eid]['boundary_uncertainty']=dict(takeoff_pts_interval=[pts(81.28),pts(81.36)],
            landing_pts_interval=[pts(81.48),pts(81.58)])
    if e['kind']=='support_candidate':
        inherited[eid]['contact_mode']='normal_runner'
payload=dict(schema='rinkspace.programme-source-action-ledger.v1',source_id=sid,time_base=tb,
    scope_pts=[283136,2415104],exposures=8329,parent_evidence_sha256=hashlib.sha256(raw).hexdigest(),
    source_review=dict(overview='全节目原生PTS稀疏顺序图，约0.4秒一幅；重点跳跃及换足查看细图，非全片逐曝光接触判读',
        review_root_data_relative='runs/body_s01_20261029/trajectory_action_1107',
        official_order_source='https://www.isuresults.com/results/season2627/jgptur2026/FSKWSINGLES-JUNIOR----QUAL000100--_JudgesDetailsperSkater.pdf',
        official_order=['3F+3T','2A','FCSp4','3Lz','StSq4','LSp4','CCoSp4'],
        unresolved=['S01开头脚截断','组合跳点冰材料、着冰首曝光','133.72步法跳候选',
                    '换足细节及内外刃','无独立冰面接触力或三维体形测量']),
    coverage_intervals=coverage,episodes=episodes,inherited_episode_details=inherited,
    skate_regions=regions)
out=root/'events/ankara_2026/programmes/okada_mei_sp/programme_action_ledger_1107.json'
with out.open('x',encoding='utf8',newline='\n') as f:
    json.dump(payload,f,ensure_ascii=False,indent=2)
    f.write('\n')
print(out,len(episodes),len(regions))
