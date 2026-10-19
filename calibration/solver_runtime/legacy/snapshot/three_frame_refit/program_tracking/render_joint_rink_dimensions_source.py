"""Whole native source frames of an unaccepted shared rink/camera state."""
from __future__ import annotations
import argparse,io,hashlib,json
from dataclasses import replace
from pathlib import Path
import sys
ROOT=Path('D:/OneDrive/codex_projs/RinkSpace');sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2,numpy as np
from z190_main_cut_depth_profile_recovery import CURRENT
from z190_joint_shared_F_height_source import SCENE
from z190_physical_camera import PhysicalCameraModel,CanonicalView,ViewState
from z190_main_zoom_k1_source_information import read,write,ref
from z190_continuous_source_bundle import SCENE_UNIT_M,SOURCE_SHA
from shared_rink_numeric_geometry import SharedRinkNumericGeometry
from continuous_central_brown4_optin import enable_projection_backend
from render_result import SceneRenderer,PersonMasks,DEFAULT_MASKS,DEFAULT_PEOPLE,decode_selected
from three_frame_refit.program_tracking.body_tracking.pelvis_proxy_programme import PlanView,source_time


def render(args):
    enable_projection_backend();cv2.setNumThreads(1);args.output.mkdir(parents=True,exist_ok=False)
    data=args.checkpoint.read_bytes()
    with np.load(io.BytesIO(data),allow_pickle=False)as z:
        q=z['q'].copy();starts=z['aux__block_start'];lengths=z['aux__block_lengths'];names=z['aux__block_names'];pts=z['aux__canonical_pts']
        ng=int(z['aux__geometry_parameter_count'].item())if 'aux__geometry_parameter_count'in z.files else 4
        geometry_indices=z['aux__geometry_parameter_indices'].copy()if 'aux__geometry_parameter_indices'in z.files else np.arange(ng)
        xyz=z['aux__installation_xyz_slots'].copy()if 'aux__installation_xyz_slots'in z.files else None
        embedded_base=json.loads(str(z['aux__shared_scene_json'].item()))if 'aux__shared_scene_json'in z.files else None
        pattern_start=int(z['aux__shared_pattern_start'].item())if 'aux__shared_pattern_start'in z.files else None
        panel_start=int(z['aux__shared_ad1_14_start'].item())if 'aux__shared_ad1_14_start'in z.files else None
        panel_chart=str(z['aux__shared_ad1_14_chart'].item())if 'aux__shared_ad1_14_chart'in z.files else None
    expected=['main1','main7','secondary','low','main9']+(['main4']if xyz is not None else [])
    if list(names)!=expected or ng not in (4,5,7,11)or geometry_indices.shape!=(ng,)or len(q)<ng+3:
        raise ValueError('Explicit shared dimensions checkpoint inventory required')
    original,native,aliases,oldviews=PhysicalCameraModel.load_npz(CURRENT)
    cals=dict(original.calibrations);ids=sorted(cals)
    if xyz is not None:
        if ng not in(5,7,11)or not np.array_equal(xyz,np.arange(5,14).reshape(3,3)):raise ValueError('Explicit three-installation XYZ slots required')
        for name,slots in zip(ids,xyz):cals[name]=replace(cals[name],pivot_xyz=tuple(q[slots]/SCENE_UNIT_M))
    else:
        for i,name in enumerate(ids):cals[name]=replace(cals[name],pivot_xyz=(*cals[name].pivot_xyz[:2],q[ng+i]/SCENE_UNIT_M))
    candidate=PhysicalCameraModel(cals);views=dict(oldviews);cursor=0
    for start,length in zip(starts,lengths):
        count=int(length)//4;stamps=pts[cursor:cursor+count];cursor+=count
        if len(stamps)!=count or length!=count*4:raise ValueError('Complete actual-exposure inventory changed')
        for i,p in enumerate(stamps):
            state=q[start+4*i:start+4*i+4];views[int(p)]=CanonicalView(oldviews[int(p)].installation_id,ViewState(tuple(state[:3]),float(state[3]-np.log(1920.))))
    if cursor!=len(pts):raise ValueError('Unexpected actual exposure metadata')
    if ng==11:
        if (panel_start is None or panel_chart!='actual_closed_boundary_station_Z_print_panel' or
                not np.array_equal(geometry_indices[:5],np.arange(5))or
                not np.array_equal(geometry_indices[-4:],np.arange(panel_start,panel_start+4))):
            raise ValueError('Explicit shared actual Ad1/14 panel world chart/indices required')
        from shared_rink_numeric_geometry_lanes_ads import SharedRinkNumericGeometryLanesAds
        geometry_class=SharedRinkNumericGeometryLanesAds
    elif ng==7:
        from shared_rink_numeric_geometry_lanes import SharedRinkNumericGeometryLanes
        geometry_class=SharedRinkNumericGeometryLanes
    elif ng==5:
        from shared_rink_numeric_geometry_midpoint import SharedRinkNumericGeometryMidpoint
        geometry_class=SharedRinkNumericGeometryMidpoint
    else:geometry_class=SharedRinkNumericGeometry
    base=read(SCENE)if embedded_base is None else embedded_base
    geometry=geometry_class(base);scene=geometry.scene(q[geometry_indices])
    if pattern_start is not None:
        from z190_shared_board_pattern_source import SharedPerimeterBoardPatternGeometry
        pattern_geometry=SharedPerimeterBoardPatternGeometry(scene,SCENE_UNIT_M,
            facet_authority=dict(kind='frozen_checkpoint_shared_source_identity',checkpoint=str(args.checkpoint)))
        scene['shared_board_patterns']=pattern_geometry.descriptor(q[pattern_start:pattern_start+9])
    # Current default uses original hard93c, not the diagnostic F base.
    from z190_main_cut_depth_profile_recovery import MANIFEST
    hard=read(Path(read(MANIFEST)['scene']['path']))
    renderers=[SceneRenderer(hard),SceneRenderer(scene)];plans=[PlanView(hard,0.),PlanView(scene,0.)]
    if any(r.path_counts!=p.path_counts for r,p in zip(renderers,plans)):
        raise ValueError('Source and plan must show the same complete modeled lines and ads')
    arrays=dict(q=q,block_start=starts,block_lengths=lengths,block_names=names,canonical_pts=pts,geometry_parameter_count=np.array(ng))
    arrays['geometry_parameter_indices']=geometry_indices
    if xyz is not None:arrays['installation_xyz_slots']=xyz
    if embedded_base is not None:arrays['shared_scene_json']=np.array(json.dumps(base,ensure_ascii=False))
    if pattern_start is not None:arrays['shared_pattern_start']=np.array(pattern_start)
    np.savez_compressed(args.output/'frozen_source_checkpoint.npz',**arrays)
    write(args.output/'frozen_scene_UNACCEPTED.json',scene)
    base_reference=ref(SCENE)
    if embedded_base is not None:
        write(args.output/'frozen_shared_base_scene.json',base)
        base_reference=ref(args.output/'frozen_shared_base_scene.json')
    masks=PersonMasks(DEFAULT_MASKS,DEFAULT_PEOPLE);records=[]
    for pts,source in decode_selected(ROOT/'data/input/oxXu1avhrvs.mp4',sorted(set(args.pts))):
        p=aliases[pts];mask,status=masks.get(pts,require_sam2=True);sourcefile=args.output/f'{pts}_source_whole.jpg'
        cv2.imwrite(str(sourcefile),source,[cv2.IMWRITE_JPEG_QUALITY,96])
        cams=[original.camera25(oldviews[p]),candidate.camera25(views[p])]
        for name,renderer,plan,cam in zip(('current','candidate'),renderers,plans,cams):
            left,metrics=renderer.draw(source,cam,mask);right=plan.base.copy();right[510:]=248
            geometry_label=f'共同行Y delta {q[4]:+.3f}m'if ng in(5,7)else '共同数值尺寸＋三台固定云台'
            lines=[f'原片 {source_time(pts)}','全部网格／靶圈／Lane／广告',
                geometry_label if name=='candidate'else'当前相机7625／hard93c',
                f'F/R dx {q[0]:+.3f}/{q[1]:+.3f}m',f'行距/半径 delta {q[2]:+.3f}/{q[3]:+.3f}m',
                '未验收；实物互拍＋三台固定XYZ'if xyz is not None else'未验收；主4尚未重解']
            for i,line in enumerate(lines):right=plan.text(right,(18,530+56*i),line,small=True)
            canvas=np.full((1160,2580,3),22,np.uint8);canvas[65:1145,:1920]=left;canvas[65:1145,1920:]=right
            canvas=plan.text(canvas,(18,12),f'{name} | 原片 {source_time(pts)} | PTS {pts}',color=(245,245,245))
            target=args.output/f'{pts}_{name}_all_model.jpg';cv2.imwrite(str(target),canvas,[cv2.IMWRITE_JPEG_QUALITY,94])
            records.append(dict(pts=pts,actual_pts=p,source_time=source_time(pts),variant=name,image=str(target),source_whole=str(sourcefile),metrics=metrics,mask_status=status))
    write(args.output/'review.json',dict(checkpoint=ref(args.checkpoint),frozen_bytes_sha256=hashlib.sha256(data).hexdigest(),
        current=ref(CURRENT),source_sha256=SOURCE_SHA,scene_reference=base_reference,geometry_parameter_count=ng,same_complete_model_projected_and_plan=True,
        original_and_candidate_each_show_their_complete_shared_model=True,
        installation_xyz_slots=None if xyz is None else xyz.tolist(),accuracy_accepted=False,full_programme_video_generated=False,records=records))
    print(dict(output=str(args.output),frames=len(records)//2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--pts',type=int,nargs='+',required=True);render(parser.parse_args())
