from calibration.program_tracking.extract_first_shot_ice import *
from calibration.program_tracking.native_ice_short_arc_repair import short_native_outer_arc
ex=Extractor(DATA/'repair_20261014');i=840;s=ex.rows[i]['source'];b=cv2.imread(str(DATA/f'frames/{i:06d}.png'));m=cv2.imread(s['mask_path'],0);bc,_=current_broadcast_exclusion(b);dom,_=current_ice_visible_domain(i);n=NativeProfiles(cv2.cvtColor(b,cv2.COLOR_BGR2RGB),s,(m>0)|bc|dom);r,d=short_native_outer_arc(ex.house_reader,'F1',i,n);print(d);print(r)
