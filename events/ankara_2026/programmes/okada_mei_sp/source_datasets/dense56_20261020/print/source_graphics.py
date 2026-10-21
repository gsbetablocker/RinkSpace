"""Current source surface/graphics qualification, using the accepted first-shot timer reader."""
import sys,json,hashlib
from pathlib import Path
import numpy as np,cv2
P=Path(r'D:/OneDrive/codex_projs/RinkSpace');sys.path.insert(0,str(P))
from calibration.program_tracking.extract_first_shot_ice import current_broadcast_exclusion

def current_graphics(bgr):
 graphics,timer=current_broadcast_exclusion(bgr)
 # This source's technical-score panel below the two permanent score rows.
 # The source coordinates only locate the layout; current colored header,
 # green execution field and neutral body must all be present.
 header=bgr[94:124,72:395].astype(float);sat=header.max(2)-header.min(2)
 header_fraction=float(((sat>55)&(header.max(2)>100)).mean())
 execution=bgr[187:220,234:396].astype(float)
 green_fraction=float(((execution[:,:,1]>execution[:,:,0]+30)&(execution[:,:,1]>execution[:,:,2]+30)&(execution[:,:,1]>120)).mean())
 body=bgr[125:185,72:395].astype(float)
 neutral_fraction=float(((body.min(2)>155)&(body.max(2)-body.min(2)<55)).mean())
 panel=header_fraction>.25 and green_fraction>.15 and neutral_fraction>.35
 if panel:graphics[90:257,60:400]=True
 return graphics,dict(timer=timer,technical_panel=dict(present=bool(panel),header_saturated_fraction=header_fraction,execution_green_fraction=green_fraction,body_neutral_fraction=neutral_fraction,source_layout_roi_only=True,rectangle_xyxy=[60,90,400,257]if panel else None))
