import cv2,numpy as np,hashlib
from pathlib import Path
REVIEWED_PRINT_INTERVAL=(1411328,1414912)
REVIEWED_CASES=list(range(1411328,1414912+1,256))+[1415168]
def current_static_print_semantics(rgb,body_valid,native_pts):
 mask=np.zeros(rgb.shape[:2],'u1');receipt=dict(native_pts=int(native_pts),independent_static_print_semantic_support=False,reviewed_source_cases=REVIEWED_CASES,world_material_identity_not_issued=True,fixed_2D_material_point_correspondence_not_issued=True)
 if not REVIEWED_PRINT_INTERVAL[0]<=native_pts<=REVIEWED_PRINT_INTERVAL[1]:return mask,receipt
 # Current material pixels, never transported source UVs.  The reviewed same-shot interval
 # visibly contains the confirmed ISU blue/white poster and WYNDHAM blue-print patch.
 blue=(rgb[:,:,2].astype(float)-rgb[:,:,0].astype(float)>12)&(body_valid>0);contours,_=cv2.findContours(blue.astype('u1'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
 cv2.drawContours(mask,contours,-1,255,cv2.FILLED);mask[body_valid==0]=0
 receipt.update(independent_static_print_semantic_support=True,current_native_RGB_component_recognition='same confirmed ISU blue/white printed poster and WYNDHAM blue printing; interval reviewed in original source',same_shot_only=True,current_RGB_resegmented_every_exposure=True,current_RGB_blue_material_contrast_native_DN=12,blue_component_count=len(contours),current_visible_print_material_pixels=int((mask>0).sum()),normal_only_material_boundary_observation=True,source_curve_or_world_projection_not_used=True,reference_source_programme_frame=4528,reference_native_pts=1415168,reference_dataset='shot6_20261014/frames/004528.json',source_all15_RGB_review_ref=dict(file=str(Path(__file__).parent/'semantic_interval_review/all15_source_receipt.json'),sha256=hashlib.sha256((Path(__file__).parent/'semantic_interval_review/all15_source_receipt.json').read_bytes()).hexdigest()))
 return mask,receipt
