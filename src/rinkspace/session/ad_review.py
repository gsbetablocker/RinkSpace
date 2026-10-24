"""Source-pixel comparison cards and native-sampled advertisement templates.

No new text, inverse camera projection, contour extraction, or fitting occurs.
Raster resizing below is display only; original UV and source clocks are kept.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..core.native import NativeCore
from .curling_preview import Canvas, INK, GRAY, BG, write

COLORS = {'historical_template_evidence':'#148c6f',
          'current_programme_source_contour':'#ec8f28',
          'current_programme_source_glyph':'#148c6f'}
HOLE = '#d34cce'


def _fit(image, size):
    scale = min(size[0]/image.width, size[1]/image.height)
    target = tuple(max(1,round(v*scale)) for v in image.size)
    return image.resize(target, Image.Resampling.LANCZOS), scale


def source_panels(rows, prefix, out, title):
    """Render only each row's own stored native pixels on its actual frame."""
    frame = Path(rows[0]['source_frame_path'])
    if any(r['source_frame_path'] != str(frame) or r['pts'] != rows[0]['pts']
           or r['source_sha256'] != rows[0]['source_sha256'] for r in rows):
        raise ValueError('mixed-source rows cannot be drawn on one source image')
    payload = frame.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if any(r.get('source_frame_sha256') and r['source_frame_sha256'] != digest for r in rows):
        raise ValueError('declared native source-frame bytes changed: '+str(frame))
    image = Image.open(frame).convert('RGB')
    points = np.concatenate([np.asarray(r['source_native_uv'],float) for r in rows])
    lo = np.maximum(np.floor(points.min(0)-16).astype(int),0)
    hi = np.minimum(np.ceil(points.max(0)+16).astype(int), image.size)
    if np.any(points.min(0)<0) or np.any(points.max(0)>np.asarray(image.size)-1):
        raise ValueError('source UV is outside the original native frame')
    box = tuple(map(int,np.r_[lo,hi]))
    crop = image.crop(box)
    # Equal scales and crop rectangles for the plain/overlay pair.
    enlarged, scale = _fit(crop,(1120,285))
    plain = Image.new('RGB',(1120,300),BG)
    offset=((1120-enlarged.width)//2,(300-enlarged.height)//2)
    plain.paste(enlarged,offset)
    overlay=plain.copy(); pen=ImageDraw.Draw(overlay)
    identifiers=[]
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    for index,row in enumerate(rows):
        identity=f'{prefix}{index+1:02d}'
        p=(np.asarray(row['source_native_uv'])-lo)*scale+offset
        color=HOLE if 'inner' in row['feature_id'] else COLORS[row['evidence_role']]
        line=p.tolist()
        if row['closed'] and not np.array_equal(p[0],p[-1]):
            line.append(p[0].tolist())
        pen.line([tuple(v) for v in line],fill=color,width=2,joint='curve')
        # Compact IDs use an adjacent full-name legend. Never cover the source
        # in the separate, identically scaled plain crop.
        at=np.mean(p,axis=0)
        bounds=pen.textbbox(tuple(at),identity,font=font,anchor='mm')
        pen.rectangle((bounds[0]-2,bounds[1]-1,bounds[2]+2,bounds[3]+1),fill='#142827')
        pen.text(tuple(at),identity,font=font,anchor='mm',fill=color)
        identifiers.append({'display_id':identity,'feature_id':row['feature_id'],
                            'observation_id':row['id'],'closed':row['closed']})
    plain_path=out/f'{prefix}_source.png';overlay_path=out/f'{prefix}_overlay.png'
    plain.save(plain_path);overlay.save(overlay_path)
    return {'title':title,'prefix':prefix,'plain':str(plain_path),'overlay':str(overlay_path),
            'source_frame':str(frame),'source_frame_sha256':digest,'native_size':list(image.size),
            'crop_native_xyxy':list(box),'display_scale':scale,'display_offset':list(offset),
            'source_sha256':rows[0]['source_sha256'],'pts':rows[0]['pts'],
            'time_base':rows[0]['time_base'],
            'source_seconds':float(rows[0]['pts']*Fraction(rows[0]['time_base'])),
            'identifiers':identifiers,'native_vertex_count':sum(len(r['source_native_uv']) for r in rows)}


def template_panel(assets, core, out):
    """Sample the recorded source charts with the same native finite-path API."""
    canvas=Canvas((2300,480))
    canvas.draw.rectangle((30,40,2270,430),fill='#713861')
    shapes=[]
    wh=assets['templates'][0].get('display_reference_size')
    if not wh or len(wh)!=2 or min(wh)<=0:
        raise ValueError('recorded template display reference dimensions are required')
    reference=np.asarray(wh,float)
    scale=min(2180/reference[0],360/reference[1])
    span=reference*scale
    rect=np.array([1150.,235.])-span/2
    for path in assets['templates'][0]['paths']:
        uv=np.asarray(path['material_uv'],float)
        # Template v-down to display coordinates; these are dimensionless
        # source charts, not measured surface sizes or a new installation.
        vertices=np.column_stack((uv[:,0],1-uv[:,1],np.zeros(len(uv))))
        if path['closed'] and not np.array_equal(vertices[0],vertices[-1]):
            vertices=np.vstack((vertices,vertices[0]))
        xyz=np.asarray(core.sample_path([],4,[],vertices.tolist(),np.linspace(0.,1.,len(vertices))))
        p=rect+np.column_stack((xyz[:,0],1-xyz[:,1]))*span
        ink=HOLE if 'inner' in path['id'] else '#f1c05d' if 'shot2_native_component' in path['id'] else '#f8f5ee'
        canvas.line(p,ink,2)
        shapes.append({'id':path['id'],'kind':4,'closed':path['closed'],
                       'source_observation_id':path['source_observation_id'],
                       'xyz_material':xyz.tolist(),'vertices_are_source_observations':False})
    canvas.text((45,456),'旧低清轮廓初值，仅供追溯；显示比例已修正，尚未获得正确的高清字形模板。',25)
    path=out/'template_native.png';canvas.image.save(path)
    write(out/'template_geometry.json',{'native_generated':True,'coordinate_unit':'source_material_chart',
         'physical_placement_estimated':False,'display_width_height_is_physical_aspect':False,
         'display_reference_size':list(wh),'display_scale_xy':[float(scale)]*2,
         'shape_status':'legacy_coarse_source_initializer_not_accepted_artwork','shapes':shapes})
    return str(path)


def render(assets_path, library_path, out_dir):
    assets_path=Path(assets_path).resolve();assets=json.loads(assets_path.read_text(encoding='utf8'))
    out=Path(out_dir).resolve()
    if out.exists():
        raise FileExistsError('Advertisement review output already exists: '+str(out))
    out.mkdir(parents=True)
    core=NativeCore(library_path)
    if assets['templates'][0].get('shape_status')=='source_rebuilt_pending_user_visual_review':
        from .glyph_rebuild import _native_shapes,_paint
        shapes=_native_shapes(assets['templates'][0]['paths'],core)
        target=out/'template_native.png';_paint(shapes).save(target)
        write(out/'native_geometry.json',{'native_generated':True,'paths':shapes})
        page=out/'index.html'
        page.write_text('<!doctype html><meta charset="utf-8"><title>Ad2 ISU-SKATING.COM</title>'
            '<h1>ISU-SKATING.COM</h1><p>当前模型的文字与广告外轮廓；A/O内孔保留，无背景纹理。</p>'
            '<img style="max-width:100%" src="template_native.png">',encoding='utf8')
        receipt={'schema':'rinkspace.ad_asset_review.v1','assets':str(assets_path),'card':str(target),
            'page':str(page),'counts':assets['counts'],'fit_executed':False,
            'shape_status':assets['templates'][0]['shape_status'],'background_texture_paths':0}
        write(out/'review.json',receipt)
        return receipt
    groups=[('historical_template_evidence','H','旧来源 · 17 字形 · 只作模板依据'),
            ('current_programme_source_contour','N','官方 S02 · 15 条新增蓝纹材料边界'),
            ('current_programme_source_glyph','Q','官方 S01 · 高清完整字行 · 17 轮廓 / 2808 点')]
    panels=[]
    for role,prefix,title in groups:
        rows=[r for r in assets['observations'] if r['evidence_role']==role]
        if rows:
            panels.append(source_panels(rows,prefix,out,title))
    template=template_panel(assets,core,out)
    card=Canvas((2400,2440))
    card.text((55,52),'D1 · Ad2 来源与模板接入审核',42)
    card.text((55,106),'原 UV / 原生 PTS 保留；17 份继承副本只登记别名。绿色：字形边界；紫色：内孔；橙色：S02 蓝纹。',25)
    card.text((55,150),'下列高清字形已绑定同名模板作为优先像素依据；尚未重新拟合字形或安装位置。',25,GRAY)
    for index,panel in enumerate(panels):
        y=207+index*510
        card.text((55,y),panel['title'],29)
        t=f"PTS {panel['pts']} · time_base {panel['time_base']} · 源 {panel['source_seconds']:.3f}s"
        card.text((55,y+37),t,23,GRAY)
        card.text((55,y+72),'纯原片裁图',23)
        card.text((1235,y+72),'同裁剪 / 同缩放叠加原轮廓与 ID',23)
        card.image.paste(Image.open(panel['plain']),(55,y+90))
        card.image.paste(Image.open(panel['overlay']),(1235,y+90))
        legends=[f"{item['display_id']} {item['feature_id'].removeprefix('Ad2.print.').replace('shot2_native_component','c')}" for item in panel['identifiers']]
        for j in range(0,len(legends),6):
            card.text((55,y+412+(j//6)*27),'    '.join(legends[j:j+6]),20)
    bottom=1760
    card.text((55,bottom),'旧轮廓追溯图（非正确高清模板）：保持记录中的显示比例，不额外拉宽字形',28)
    card.image.paste(Image.open(template),(50,bottom+35))
    card.text((55,2340),'来源归位不代表字形模板完成。保留原片读数，继续核对与现场同版的高清原稿；未启动求解。',25,GRAY)
    card_path=out/'ad2_sources.png';card.image.save(card_path)
    rows=[]
    for p in panels:
        rows.append('<h2>'+html.escape(p['title'])+'</h2><p>PTS '+str(p['pts'])+' / '+p['time_base']+' · '+f"{p['source_seconds']:.3f}s"+'</p><div class="pair"><img src="'+Path(p['plain']).name+'"><img src="'+Path(p['overlay']).name+'"></div><p>'+html.escape('；'.join(i['display_id']+' = '+i['feature_id'] for i in p['identifiers']))+'</p>')
    content='<!doctype html><meta charset="utf-8"><title>Ad2来源审核</title><style>body{font:18px/1.6 system-ui;margin:28px;background:#edf2f2;color:#173a3b}img{max-width:100%}.pair{display:flex;gap:12px}.pair img{width:49%;object-fit:contain}p{max-width:1800px}</style><h1>D1 · Ad2 原片与既有模板</h1><p>逐记录原来源；所有轮廓均是已有读数。图审不等于新拟合完成。</p>'+''.join(rows)+'<h2>同一native核心的既有模板初值</h2><img src="template_native.png"><p>'+html.escape('；'.join(assets['limitations']))+'</p>'
    (out/'index.html').write_text(content,encoding='utf8')
    receipt={'schema':'rinkspace.ad_asset_review.v1','assets':str(assets_path),
             'library':str(Path(library_path).resolve()),'panels':panels,'card':str(card_path),
             'template':template,'page':str(out/'index.html'),'counts':assets['counts'],
             'fit_executed':False,'whole_advertisement_user_approval_inferred':False}
    write(out/'review.json',receipt)
    return receipt
