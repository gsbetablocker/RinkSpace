from calibration.program_tracking.extract_first_shot_ice import *
ex=Extractor(DATA/'repair_20261014',ROOT/'events/ankara_2026/models/framewise_8a28_first_shot_20261014/repair_20261014');ex.use_circle_cache=False
for i in [293,328,642,768,840,946,975,1054]:
 s=ex.rows[i]['source'];b=cv2.imread(str(DATA/f'frames/{i:06d}.png'));m=cv2.imread(s['mask_path'],0);bc,_=current_broadcast_exclusion(b);dom,_=current_ice_visible_domain(i);n=NativeProfiles(cv2.cvtColor(b,cv2.COLOR_BGR2RGB),s,(m>0)|bc|dom);f,d=ex.nested_house_features(i,n)
 print(i,[(cid,x['point_count'],x.get('joint_group_navigation',{}).get('observable_translation_rank'),x.get('joint_group_navigation',{}).get('shared_source_affine_parameters')) for cid,x in f.items() if x.get('joint_group_navigation',{}).get('observable_translation_rank')])
