"""Append multi-time source-reviewed visible goal-crease strokes, never model UV."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import cv2
import numpy as np

DATA=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026')
BASE=DATA/'observations/vision_training/continuous_review/cut6_clear_source_constraints/complete_source_clear_anchor_append_20261007'
SOURCE=DATA/'diagnostics/goal_crease_shared_geometry_20261007'
PARENT=BASE/'ice_clear_anchors_user_lane2_lane3_traced_lane2_v2_20261007.json'
OUTPUT=BASE/'ice_clear_anchors_lane23_trace_and_goal_crease_v2_20261007.json'

# Native RGB reviewed directly at 1920x1080 with a coordinate grid. These
# describe visible red boundary strokes bordering the blue-filled D shape.
# Occluded portions, screen overlays and out-of-frame arcs are not sampled.
# No camera, world projection or old fitted source is used to select points.
READINGS={
 230955776:('fit',[[100,826],[250,832],[400,836],[550,848],[680,865],[800,885],
                     [480,1037],[580,1020],[680,1001],[780,977],[850,959]]),
 230967552:('fit',[[1000,837],[1150,850],[1300,861],[1450,877],[1550,897],
                     [1200,1045],[1300,1039],[1400,1025],[1500,1005],[1600,977]]),
 230979840:('holdout',[[1150,907],[1250,904],[1350,912],[1450,919],
                         [1550,925],[1650,933]])}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    original=json.loads(PARENT.read_text(encoding='utf8'))
    packet=deepcopy(original)
    for pts,(split,points) in READINGS.items():
        path=SOURCE/f'native_{pts}.png'
        image=cv2.imread(str(path))
        if image is None or image.shape!=(1080,1920,3):raise ValueError(path)
        row=dict(id=f'source_review_goal_crease_left_{pts}_20261007',pts=pts,
            feature_id='goal_crease_left',curve=dict(kind='current_goal_crease',feature_id='goal_crease_left'),
            split=split,uv=points,sigma_px=[12.]*len(points),
            geometry_role='visible_red_boundary_stroke_of_blue_filled_goal_crease',
            source_raw_frame=str(path),source_file_sha256=sha(path),
            source_native_BGR_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
            source_RGB_actual_measurement=True,sourceUV_origin='manual_native_RGB_review_by_agent',
            user_confirmed_source_axis=False,
            physical_identity_authority=dict(kind='same_shot_blue_filled_goal_D_continuity_and_user_identified_baseline_failure',
                human_confirmed_exact_pixels=False,model_projection_used_for_identity_or_pixel_selection=False),
            source_optical_normal_support_authorized=False,
            sigma_semantics='12 native pixel engineering uncertainty for broad motion-blurred red stroke; not calibrated variance or physical paint width',
            no_occluded_or_cropped_boundary_invented=True,whole_video_accuracy_accepted=False)
        packet['rows'].append(row)
        for i,uv in enumerate(points):
            cv2.drawMarker(image,tuple(uv),(0,220,255),cv2.MARKER_CROSS,13,1)
            cv2.putText(image,str(i),tuple(np.array(uv)+[6,-5]),0,.4,(0,0,0),1)
        cv2.imwrite(str(SOURCE/f'review_points_{pts}.png'),image)
    assert packet['rows'][:len(original['rows'])]==original['rows']
    packet.update(status='SOURCE_CANDIDATE_ALL_OLD_ROWS_PLUS_MULTI_TIME_GOAL_CREASE',
        source_installed_in_solver=False,source_goal_crease_append=dict(parent=str(PARENT),
        parent_sha256=sha(PARENT),added_rows=len(READINGS),all_original_rows_preserved=True,
        shared_geometry_and_all_original_camera_source_required=True))
    OUTPUT.write_text(json.dumps(packet,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(path=str(OUTPUT),sha256=sha(OUTPUT),new_fit_points=21,new_held_points=6)))

if __name__=='__main__':main()
