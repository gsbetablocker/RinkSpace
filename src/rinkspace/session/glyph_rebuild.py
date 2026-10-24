"""Rebuild known printed text from native frames, and review the same paths.

Text content is a user fact. Only the ink interfaces and their common chart are
estimated here. No alternative font, background texture, or scene fit is used.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw

from ..vision_print_glyphs import extract_glyphs,validate_holes,panel_chart,transform
from ..vision_glyph_chart import derive_text_chart
from ..core.native import NativeCore
from .curling_preview import Canvas,write,GRAY


def _read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _native_shapes(paths,core):
    result=[]
    for path in paths:
        uv=np.asarray(path['material_uv'],float)
        vertices=np.c_[uv[:,0],1-uv[:,1],np.zeros(len(uv))]
        if path['closed'] and not np.array_equal(vertices[0],vertices[-1]):
            vertices=np.vstack((vertices,vertices[0]))
        q=np.asarray(core.sample_path([],4,[],vertices.tolist(),np.linspace(0,1,len(vertices))))
        result.append({**path,'native_material_xyz':q.tolist()})
    return result


def _paint(shapes,size=(2400,520),*,clip=None):
    """Rasterize only native-sampled letters and the panel outer contour."""
    ss=3;im=Image.new('RGB',(size[0]*ss,size[1]*ss),'#eef3f2');d=ImageDraw.Draw(im)
    if clip is None:lo=np.array([-.025,-.10]);hi=np.array([1.025,1.10])
    else:lo=np.array(clip[0]);hi=np.array(clip[1])
    # One uniform display scale for both axes in the declared 6:1 panel chart.
    aspect=6.;span=(hi-lo)*[aspect,1.];scale=min((size[0]-30)/span[0],(size[1]-30)/span[1])
    offset=(np.array(size)-span*scale)/2
    def px(uv):return (offset+(np.asarray(uv)-lo)*[aspect,1.]*scale)*ss
    panel=px([[0,0],[1,1]]);d.rectangle(tuple(panel.flatten()),fill='#813a6b')
    for path in sorted(shapes,key=lambda p:bool(p.get('parent_contour_key'))):
        q=np.asarray(path['native_material_xyz']);uv=np.c_[q[:,0],1-q[:,1]]
        p=[tuple(t) for t in px(uv)]
        if path['role']=='panel_outer_boundary':d.line(p,fill='#355b62',width=2*ss)
        else:d.polygon(p,fill='#813a6b' if path.get('parent_contour_key') else '#fffdf8')
    return im.resize(size,Image.Resampling.LANCZOS)


def _overlay(image,paths,H_inverse,roi,size=(2300,400)):
    x,y,X,Y=roi;crop=Image.fromarray(cv2.cvtColor(image[y:Y,x:X],cv2.COLOR_BGR2RGB))
    scale=min(size[0]/crop.width,size[1]/crop.height)
    target=(round(crop.width*scale),round(crop.height*scale));plain=crop.resize(target,Image.Resampling.LANCZOS)
    canvas=Image.new('RGB',size,'#eef3f2');off=(np.array(size)-target)//2;canvas.paste(plain,tuple(off));overlay=canvas.copy();d=ImageDraw.Draw(overlay)
    for path in paths:
        if path['role']=='panel_outer_boundary':continue
        uv=transform(path['material_uv'],H_inverse);q=(uv-[x,y])*scale+off
        d.line([tuple(v) for v in q],fill='#cc4dc1' if path.get('parent_contour_key') else '#2fda91',width=2,joint='curve')
    return canvas,overlay


def rebuild(input_path,library_path,out_dir):
    config=_read(input_path);out=Path(out_dir).resolve()
    if out.exists():raise FileExistsError(str(out))
    out.mkdir(parents=True)
    source=config['primary'];reference=_read(config['panel_reference'])
    if _sha(source['frame_path'])!=source['frame_sha256']:raise ValueError('selected glyph frame changed')
    primary=cv2.imread(source['frame_path']);reference_image=cv2.imread(reference['source_image']['path'])
    measured=extract_glyphs(primary,config['primary_roi'],config['component_order'],threshold=config['threshold'])
    ref=extract_glyphs(reference_image,config['reference_roi'],config['component_order'],threshold=config['reference_threshold'])
    for value in (measured,ref):validate_holes(value['paths'],config['hole_parents'])
    h=derive_text_chart(measured['paths'],config['cap_keys'],config['vertical_keys'])
    h_ref=derive_text_chart(ref['paths'],config['cap_keys'],config['vertical_keys'])
    H_text=np.asarray(h['H']);H_ref=np.asarray(h_ref['H'])
    def width(value,H):return float(transform(np.concatenate([p['source_native_uv'] for p in value['paths'].values()]),H)[:,0].max())
    w,width_ref=width(measured,H_text),width(ref,H_ref)
    panel=panel_chart(reference['copied_existing_source_features'])
    # A single global source-to-panel initializer. The source text is never
    # split into individually positioned or stretched characters.
    H=np.asarray(panel['H'])@np.linalg.inv(H_ref)@np.diag([width_ref/w,1.,1.])@H_text
    H/=H[2,2]
    # The two photographed I glyphs are the same plain vertical bar. Estimate
    # their common stroke width from the clear middle portions; shared cap and
    # baseline are typographic geometry, never edits of the original pixels.
    bars={}
    for key in config.get('shared_bar_glyphs',[]):
        p=transform(measured['paths'][key]['source_native_uv'],H_text)
        p=p[(p[:,1]>.2)&(p[:,1]<.8)];middle=np.median(p[:,0])
        left=float(np.median(p[p[:,0]<middle,0]));right=float(np.median(p[p[:,0]>=middle,0]))
        bars[key]=(left,right)
    bar_width=float(np.median([b-a for a,b in bars.values()])) if bars else None
    paths=[]
    for key,p in measured['paths'].items():
        source_geometry=np.asarray(p['template_source_uv'])
        text_geometry=transform(source_geometry,H_text)
        if key in bars:
            centre=sum(bars[key])/2
            text_geometry=np.array([[centre-bar_width/2,0],[centre+bar_width/2,0],
                                    [centre+bar_width/2,1],[centre-bar_width/2,1],[centre-bar_width/2,0]])
        elif key in config['cap_keys']:
            for level in (0.,1.):
                close=np.abs(text_geometry[:,1]-level)<.012
                text_geometry[close,1]=level
        source_geometry=transform(text_geometry,np.linalg.inv(H_text))
        q=transform(source_geometry,H)
        paths.append({'id':'Ad2.print.'+key,'original_id':key,'closed':True,
            'role':p['role'],'parent_contour_key':p['parent_contour_key'],
            'material_uv':q.tolist(),'source_native_uv':deepcopy(p['source_native_uv']),
            'template_source_uv':source_geometry.tolist(),
            'source':deepcopy(source),'source_to_material_homography':H.tolist()})
    borders={'top_edge':[[0,0],[1,0]],'right_vertical':[[1,0],[1,1]],
             'bottom_edge':[[1,1],[0,1]],'left_vertical':[[0,1],[0,0]]}
    for key,uv in borders.items():
        evidence=panel['edge_evidence'][key]
        paths.append({'id':'Ad2.'+key,'original_id':key,'closed':False,
            'role':'panel_outer_boundary','parent_contour_key':None,'material_uv':uv,
            'source_native_uv':evidence['native_readout_uv'],
            'source':{'video_sha256':reference['source_sha256'],'native_pts':reference['native_pts'],
                'time_base':reference['time_base'],'frame_path':reference['source_image']['path'],
                'frame_sha256':reference['source_image']['sha256']},'source_feature':evidence['raw_feature'],
            'source_to_material_homography':panel['H'],'derived_corners_are_observations':False})
    result={'schema':'rinkspace.source_print_rebuild.v1','advertisement_number':2,
        'text':config['text'],'text_content_authority':'direct_user_confirmation',
        'text_content_confirmed':True,'paths':paths,'background_texture_paths':0,
        'letter_outer_paths':len(config['component_order']),'letter_hole_paths':len(config['hole_parents']),
        'panel_outer_paths':4,'source':source,'extraction':measured['extraction'],
        'text_chart':h,'panel_chart_reference':panel,'panel_source':reference,
        'source_to_material_homography':H.tolist(),'single_global_character_layout':True,
        'typographic_geometry':{'common_cap_baseline':config['cap_keys'],
            'shared_vertical_bar_glyphs':config.get('shared_bar_glyphs',[]),
            'bar_width_cap_units':bar_width,'source_pixel_arrays_changed':False},
        'alternative_font_used':False,'physical_dimensions_fitted':False,
        'template_shape_review':'source_rebuilt_pending_user_visual_review',
        'scene_fit_executed':False,'old_observations_rewritten':False}
    template_path=Path(config['template_output']).resolve()
    if template_path.exists():raise FileExistsError(str(template_path))
    template_path.parent.mkdir(parents=True,exist_ok=True);write(template_path,result)
    core=NativeCore(library_path);shapes=_native_shapes(paths,core)
    template=_paint(shapes);template.save(out/'template.png')
    source_plain,source_overlay=_overlay(primary,paths,np.linalg.inv(H),config['primary_roi'])
    source_plain.save(out/'source.png');source_overlay.save(out/'source_overlay.png')
    # A second exposure contains the whole panel. Only its four material edges
    # and text were used; no background-pattern features are read.
    ref_plain,ref_overlay=_overlay(reference_image,paths,np.linalg.inv(np.asarray(panel['H'])),config['reference_roi'])
    ref_plain.save(out/'reference.png');ref_overlay.save(out/'reference_overlay.png')
    details=[]
    for key in config['hole_parents']:
        selected=[p for p in shapes if p['id'] in ('Ad2.print.'+key,'Ad2.print.'+key+'_inner')]
        q=np.concatenate([p['material_uv'] for p in selected]);lo=q.min(0)-[.01,.035];hi=q.max(0)+[.01,.035]
        detail=_paint(selected,(650,450),clip=(lo,hi));detail.save(out/(key+'_detail.png'));details.append((key,detail))
    card=Canvas((2400,2100));card.text((55,50),'Ad2 · ISU-SKATING.COM · 字形与外轮廓重建',40)
    card.text((55,104),'按用户确定的文字建模；15 个字母/符号外轮廓＋A/O 两个内孔＋4 条广告外边界。',25)
    card.text((55,148),'全场原片 03:33:59.900 · PTS 197220864 · 1/15360；背景纹理不进入模板。',25,GRAY)
    card.text((55,208),'清晰原片：保留真实字形和字距',28);card.image.paste(source_plain,(50,230))
    card.text((55,664),'新模板：同一 native 核心生成；文字与外轮廓，全行共用透视校正',28)
    card.image.paste(template,(0,700))
    card.text((55,1250),'A 内孔',29);card.text((835,1250),'O 内孔',29)
    for (key,detail),x in zip(details,(55,835)):card.image.paste(detail,(x,1280))
    card.text((55,1790),'原片叠加：绿色为文字外轮廓，紫色为内孔',28)
    _,overlay=_overlay(primary,paths,np.linalg.inv(H),config['primary_roi'],(2300,220))
    card.image.paste(overlay,(50,1820));card.image.save(out/'review.png')
    write(out/'native_geometry.json',{'same_native_core':True,'paths':shapes})
    receipt={'template':str(template_path),'card':str(out/'review.png'),
        'template_image':str(out/'template.png'),'source_overlay':str(out/'source_overlay.png'),
        'reference_overlay':str(out/'reference_overlay.png'),'native_geometry':str(out/'native_geometry.json'),
        'text':config['text'],'text_content_confirmed':True,'texture_paths':0,
        'scene_fit_executed':False,'user_visual_review':'pending'}
    write(out/'result.json',receipt)
    return receipt
