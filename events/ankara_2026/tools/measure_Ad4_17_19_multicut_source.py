"""Read actual independently viewed print boundaries, with optical support.

Ad4/17 are complete event-name groups. Internal purple/blue component joins
are not their right/left external boundaries. Ad18 is ink-only, so this packet
does not invent a white-board rectangle around it.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'calibration/program_tracking')]
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2
import numpy as np
from rinkspace.vision_source_shapes import read_source_material_transition_profiles

DATA=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026')
BASE=DATA/'observations/vision_training/continuous_review/Ad4_17_18_19_source_absolute_20261007'
MOTION=DATA/'programmes/okada_mei_sp/results/aligned_complete_20261007_478afaf77615/nine_shot_motion_regression_v2/native_board_motion_source_frames'
SOURCE_SHA='f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def serial(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:serial(x)for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [serial(x)for x in v]
    return v

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    cases=[
      (230469376,4,'fit','left_vertical',[[1438,y]for y in (535,580,625)],[1,0],[-20,20],'light_to_dark'),
      (230469376,4,'fit','top_edge',[[1550,515],[1750,516]],[0,1],[-18,18],'light_to_dark'),
      (230469376,4,'fit','bottom_edge',[[1550,660],[1750,666]],[0,1],[-20,20],'dark_to_light'),
      (230474752,4,'holdout','left_vertical',[[1818,y]for y in (540,590,645)],[1,0],[-25,25],'light_to_dark'),
      (230474752,4,'holdout','top_edge',[[1860,515],[1890,516]],[0,1],[-18,18],'light_to_dark'),
      (230474752,4,'holdout','bottom_edge',[[1860,681],[1890,682]],[0,1],[-25,25],'dark_to_light'),
      (230797568,4,'fit','left_vertical',[[138,y]for y in (435,460,485)],[1,0],[-18,18],'light_to_dark'),
      (230797568,4,'fit','top_edge',[[230,416],[650,415],[1600,415]],[0,1],[-14,14],'light_to_dark'),
      (230797568,4,'fit','bottom_edge',[[230,503],[650,503],[1600,508]],[0,1],[-14,14],'dark_to_light'),
      (230925312,4,'holdout','left_vertical',[[710,y]for y in (430,470,510)],[1,0],[-18,18],'light_to_dark'),
      (230925312,4,'holdout','top_edge',[[800,405],[1350,405]],[0,1],[-18,18],'light_to_dark'),
      (230925312,4,'holdout','bottom_edge',[[800,550],[1350,540]],[0,1],[-20,20],'dark_to_light'),
      (230022144,17,'fit','left_vertical',[[1497,y]for y in (420,450,490)],[1,0],[-10,10],'light_to_dark'),
      (230022144,17,'fit','top_edge',[[1540,406],[1580,412],[1780,439],[1860,448]],[0,1],[-12,12],'light_to_dark'),
      (230022144,17,'fit','bottom_edge',[[1540,521],[1580,531],[1780,590],[1860,613]],[0,1],[-12,12],'dark_to_light'),
      (230022144,19,'fit','left_vertical',[[1320,y]for y in (390,420,440)],[1,0],[-8,8],'light_to_dark'),
      (230022144,19,'fit','right_vertical',[[1363,y]for y in (390,430,450)],[1,0],[-8,8],'dark_to_light'),
      (230022144,19,'fit','top_edge',[[1332,379],[1350,382]],[0,1],[-8,8],'light_to_dark'),
      (230022144,19,'fit','bottom_edge',[[1332,451],[1350,457]],[0,1],[-8,8],'dark_to_light'),
    ]
    shots=json.loads((ROOT/'calibration/program_tracking/source_shots.json').read_text(encoding='utf8'))['shots'][1:10]
    rows=[];rejected=[];images={}
    for pts,n,split,edge,seeds,normal,interval,sign in cases:
        raw=BASE/f'raw_native_{pts}.png'
        if not raw.exists():raw=MOTION/f'native_{pts}.png'
        if pts not in images:
            bgr=cv2.imread(str(raw))
            if bgr is None:raise FileNotFoundError(raw)
            images[pts]=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        shot=next(s for s in shots if s['start_pts']<=pts<s['end_pts'])
        binding=dict(source_sha256=SOURCE_SHA,source_pts=pts,time_base='1/15360',shot_id=shot['id'],native_hw=[1080,1920],image_domain='native_brown_pixels')
        read=read_source_material_transition_profiles(images[pts],binding=binding,search_points_uv=seeds,normals=[normal]*len(seeds),
            search_intervals_px=[interval]*len(seeds),transition=sign,keep_profiles=True)
        good=[r for r in read['readings']if r['disposition']=='measured_material_transition']
        rejected.extend(dict(pts=pts,advertisement_number=n,edge=edge,**serial(r))for r in read['readings']if r['disposition']!='measured_material_transition')
        if not good:continue
        rows.append(dict(id=f'actual_multicut_native{pts}_Ad{n}_{edge}_source',pts=pts,feature_id=f'Ad{n}',split=split,
            curve=dict(kind='advertisement_boundary',advertisement_number=n,edge=edge,extent_scope='external_border'),
            geometry_role='complete_event_name_group_external_border'if n in (4,17)else'printed_colour_panel_external_border',
            uv=[r['actual_source_uv'].tolist()for r in good],normal=[r['source_search_normal'].tolist()for r in good],
            normal_interval_px=[r['observed_normal_interval_px'].tolist()for r in good],
            sigma_px=[max(4.,float(np.ptp(r['observed_normal_interval_px']))/2.)for r in good],
            sigma_semantics='Existing4px working floor or half actual optical support width; not calibrated covariance',
            point_correspondence=False,curve_observation_kind='normal_interval',
            source_coordinate_is_scanline_context_only=True,model_projection_used_to_generate_or_select_uv=False,
            physical_identity_authority=dict(source='Existing user permanent identity catalog, actual independent raw native view, and latest user same-PTS failure confirmation',
                catalog_path=str(ROOT/'configs/rink_model/ankara_2026_advertisements_user_confirmed_20260924.json'),
                number=n,printed_extent=('ANKARA magenta + white figure-skating logo + blue JUNIOR GRAND PRIX full group; internal component joins excluded'if n in(4,17)else'actual red Oriental Bio full background'),
                actual_raw_pixels_viewed_before_declaring_source_search=True,missing_edges_not_invented=True),
            source_video_sha256=SOURCE_SHA,time_base='1/15360',source_raw_frame=str(raw),source_file_sha256=sha(raw),
            source_native_RGB_sha256=read['RGB_sha256'],measured_RGB_sources=serial(good)))
    report=dict(schema='rinkspace.source-qualified-native-Ad4-Ad17-Ad19-multicut-source.v1',source_video_sha256=SOURCE_SHA,time_base='1/15360',
        rows=rows,rejected_source_queries=rejected,source_appearance_not_model_projection_selected=True,
        model_parameters_or_old_source_sigma_UV_roles_modified=False,existing_HELD_not_promoted=True,
        Ad18=dict(identity='MEIWA',scope='internal_graphic_only',source_frame=230022144,
            actual_native_ink_visible=True,no_external_white_board_rectangle_invented=True,
            status='existing_material_corner_or_glyph_role_must_be_verified_before_typed_internal_ink_source'),
        candidate_only=True,solver_admitted=False,accuracy_accepted=False)
    if a.output.exists():raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps(dict(path=str(a.output),rows=len(rows),FIT=sum(len(r['uv'])for r in rows if r['split']=='fit'),HELD=sum(len(r['uv'])for r in rows if r['split']=='holdout'),rejected=len(rejected)),ensure_ascii=False))

if __name__=='__main__':main()
