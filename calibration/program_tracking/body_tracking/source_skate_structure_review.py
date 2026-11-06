"""Render direct source skate structures and assemble the exact source review.

No model shoe pose or body fit enters these drawings. Native source pixels,
feature interpretations and camera-conditional footprints remain distinct.
"""
from pathlib import Path
import argparse,hashlib,json,math
import numpy as np
from PIL import Image,ImageDraw,ImageFont

COLORS={'boot_outline':(30,165,205),'boot_sole':(225,125,20),'heel_block':(180,80,15),
 'blade_assembly_outline':(135,65,180),'runner_lower_edge':(20,175,55),'runner_upper_edge':(20,115,65),
 'toe_pick':(220,40,45),'rear_extension':(50,80,215),'front_stanchion':(160,65,175),
 'rear_stanchion':(160,65,175),'diagonal_brace':(170,65,140),'ice_interface':(190,145,0)}
NAMES={'boot_outline':'鞋体外形','boot_sole':'鞋底','heel_block':'鞋跟块','blade_assembly_outline':'刀体/刀架外形',
 'runner_lower_edge':'完整钢刃下缘','runner_upper_edge':'钢刃带上缘','toe_pick':'前端刀齿区域','rear_extension':'刀尾外伸',
 'front_stanchion':'前支撑','rear_stanchion':'后支撑','diagonal_brace':'斜撑','ice_interface':'冰面交界候选'}
STATUS={'observed':'直接描线','inferred':'邻帧推断','ambiguous':'局部分界待核','occluded':'遮挡','unknown':'未定'}

def font(size):return ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',size)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def draw_path(pen,points,color,width=2,closed=False,style='observed'):
    p=np.asarray(points,float)
    if closed:p=np.r_[p,p[:1]]
    if len(p)<2:return
    if style=='observed':pen.line([tuple(x)for x in p],fill=color,width=width);return
    # Interpretations remain visible as dashed outlines, never hidden as unknown.
    for a,b in zip(p[:-1],p[1:]):
        length=float(np.linalg.norm(b-a));count=max(1,int(math.ceil(length/7)))
        for j in range(count):
            if j%2==0:pen.line([tuple(a+(b-a)*j/count),tuple(a+(b-a)*min(j+1,count)/count)],fill=color,width=width)

def render_frame(entry,record,output):
    image=Image.open(entry['native_crop']).convert('RGB')
    if sha(entry['native_crop'])!=entry['native_crop_sha256']:raise ValueError('Changed native source crop')
    if record.get('source_crop_sha256')!=entry['native_crop_sha256']:raise ValueError('Annotation/crop identity differs')
    box=np.asarray(entry['native_bounds_xyxy'],float);scale=3;W,H=image.size;pad=24;cw=W*scale;ch=H*scale
    top=105;legend_y=top+ch+18;features=record['features'];legend_rows=int(math.ceil(len(features)/3));height=legend_y+legend_rows*32+64
    canvas=Image.new('RGB',(3*(cw+pad)+pad,height),'white');draw=ImageDraw.Draw(canvas)
    draw.text((pad,7),f"源图鞋刀结构描线 · {entry['source_seconds']:.2f}秒 · PTS {entry['pts']} · 无人体/鞋刀模型",font=font(28),fill='black')
    draw.text((pad,46),'实线：直接描线；虚线：局部分界或邻帧推断。图中接冰候选线与真实接触结论分别保存。',font=font(20),fill='black')
    labels=['原生原片（仅3倍最近邻放大）','原片＋源结构描线','源图结构线稿（遮挡处保留缺失）']
    zoom=image.resize((cw,ch),Image.Resampling.NEAREST)
    panels=[zoom.copy(),zoom.copy(),Image.new('RGB',(cw,ch),(250,250,246))]
    for i,label in enumerate(labels):draw.text((pad+i*(cw+pad),76),label,font=font(20),fill='black')
    for index,(key,f) in enumerate(features.items()):
        p=np.asarray(f.get('points_uv',[]),float);status=f.get('status','unknown');color=COLORS.get(key,(95,95,95))
        name=NAMES.get(key,'刀架开口'+key.rsplit('_',1)[-1] if key.startswith('frame_opening_') else key)
        if p.size:
            if p.ndim!=2 or p.shape[1]!=2 or not np.isfinite(p).all():raise ValueError('Invalid native source feature '+key)
            if np.any(p<box[:2]-1) or np.any(p>box[2:]+1):raise ValueError('Source feature outside crop '+key)
            xy=(p-box[:2])*scale
            for j in (1,2):draw_path(ImageDraw.Draw(panels[j]),xy,color,width=2 if j==1 else 3,closed=bool(f.get('closed')),style=status)
        row=index//3;col=index%3;xx=pad+col*(cw+pad);yy=legend_y+row*32
        draw.line([(xx,yy+13),(xx+34,yy+13)],fill=color,width=4)
        draw.text((xx+42,yy),name+' · '+STATUS.get(status,status),font=font(19),fill='black')
    for j,panel in enumerate(panels):canvas.paste(panel,(pad+j*(cw+pad),top))
    draw.text((pad,height-42),'这些是源像素描线候选，尚未经过用户目视验收；刀齿簇轮廓不代表已分辨每一颗齿。',font=font(19),fill='black')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);canvas.save(output)
    return dict(pts=entry['pts'],path=str(output),sha256=sha(output),features=len(features))

def assemble(inputs,annotation_paths,output):
    source=read(inputs);lookup={int(r['pts']):r for r in source['frames']};frames=[];paths=[]
    for path in annotation_paths:
        d=read(path);paths.append(dict(path=str(path),sha256=sha(path)))
        for r in d['frames']:
            entry=lookup[int(r['pts'])];r=dict(r);r['source_seconds']=entry['source_seconds'];r['native_bounds_xyxy']=entry['native_bounds_xyxy']
            if r.get('source_crop_sha256')!=entry['native_crop_sha256']:raise ValueError('Source crop hash mismatch')
            r.update(historical_role=r.get('role'),role='development_review',human_confirmed=False,optimization_enabled=False,
                source_crop_path=entry['native_crop'],source_frame_rgb_sha256=entry['native_frame_rgb_sha256'],context_source_pts=entry['context_pts'])
            frames.append(r)
    if len(set(r['pts']for r in frames))!=len(frames):raise ValueError('Duplicate source frame')
    frames.sort(key=lambda r:r['pts']);doc=dict(schema='rinkspace.source_skate_structures_v1',source_sha256=source['source_video']['sha256'],
        source_video=source['source_video'],time_base=source['time_base'],native_size=[1920,1080],annotation_sources=paths,
        source_only=True,model_projection_used=False,fit_run=False,accuracy_accepted=False,
        review_scope='Direct complete visible2D structures; interpretations/occlusion retained, no generic3D material identity',frames=frames)
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(doc,ensure_ascii=False,indent=2),encoding='utf-8');return doc

def render_ice_map(entry,record,mapped,output):
    """Source pixels and camera-only rink coordinates, with no body geometry."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    import io
    feature=mapped['features']['ice_interface'];rep=feature['auxiliary_representative']['mapping']
    uv=np.asarray(rep['points_uv'][0]);xy=np.asarray(rep['xyz_conditional_m'][0])[:2]
    radius=float(rep['sampled_xy_max_offset_conditional_m'][0])
    boundary=np.asarray(rep['sampled_pixel_disk_boundary_xy_conditional_m'][0])
    lower=np.asarray(mapped['features']['runner_lower_edge']['mapping']['xyz_conditional_m'])[:,:2]
    interface=np.asarray(feature['mapping']['xyz_conditional_m'])[:,:2]
    photo=Image.open(entry['native_crop']).convert('RGBA');box=np.asarray(entry['native_bounds_xyxy'],float);scale=3;size=(photo.width*scale,photo.height*scale)
    photo=photo.resize(size,Image.Resampling.NEAREST);grid=Image.new('RGBA',size);gd=ImageDraw.Draw(grid)
    for row in mapped['auxiliary_grid']['lines']:
        points=(np.asarray(row['points_uv'])-box[:2])*scale
        gd.line([tuple(p)for p in points],fill=(75,90,100,105),width=2)
    # Keep auxiliary ground lines behind the directly traced source silhouette.
    mask=Image.new('L',size);md=ImageDraw.Draw(mask)
    for key in ['boot_outline','heel_block','blade_assembly_outline']:
        f=record['features'].get(key,{});points=np.asarray(f.get('points_uv',[]))
        if len(points)>=3:md.polygon([tuple(p)for p in ((points-box[:2])*scale)],fill=255)
    for key,f in record['features'].items():
        if key.startswith('frame_opening_') and f.get('status')=='observed':md.polygon([tuple(p)for p in ((np.asarray(f['points_uv'])-box[:2])*scale)],fill=0)
    a=np.asarray(grid.getchannel('A')).astype(float)*(1-np.asarray(mask)/255.);grid.putalpha(Image.fromarray(a.astype('uint8')))
    photo=Image.alpha_composite(photo,grid);pd=ImageDraw.Draw(photo)
    for key,color in [('runner_lower_edge',(20,175,55,255)),('ice_interface',(230,165,0,255))]:
        f=record['features'][key];points=(np.asarray(f['points_uv'])-box[:2])*scale
        draw_path(pd,points,color,3,False,f.get('status','unknown'))
    point=(uv-box[:2])*scale;pd.ellipse([point[0]-6,point[1]-6,point[0]+6,point[1]+6],fill=(225,30,40,255))
    fp=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');fig,ax=plt.subplots(figsize=(7.5,7.5),dpi=110)
    ax.fill(boundary[:,0],boundary[:,1],color='#78aee8',alpha=.22,label='源像素范围的冰面映射')
    ax.plot(lower[:,0],lower[:,1],'-',color='#14af37',lw=2,label='完整钢刃下缘的冰面足迹')
    ax.plot(interface[:,0],interface[:,1],'--',color='#dba500',lw=2,label='源交界候选的冰面足迹')
    ax.plot(xy[0],xy[1],'o',color='#df1e28',markersize=7,label='源交界候选中点')
    extent=np.concatenate((lower,interface,boundary),axis=0)
    lo=extent.min(axis=0);hi=extent.max(axis=0);centre=(lo+hi)/2
    half_span=max(float((hi-lo).max())*1.15,.80)/2
    ax.set_xlim(centre[0]-half_span,centre[0]+half_span);ax.set_ylim(centre[1]-half_span,centre[1]+half_span);ax.set_aspect('equal',adjustable='box');ax.grid(True,color='.85');ax.set_xlabel('X (m)');ax.set_ylabel('Y (m)')
    ax.set_title(f'X={xy[0]:.3f} m  /  Y={xy[1]:.3f} m\n源像素范围最大偏移约 {radius*100:.1f} cm',fontproperties=fp)
    ax.legend(prop=fp,loc='upper left',fontsize=9);fig.tight_layout();stream=io.BytesIO();fig.savefig(stream,format='png');plt.close(fig);stream.seek(0);plot=Image.open(stream).convert('RGB');plot=plot.resize(size)
    canvas=Image.new('RGB',(size[0]*2+72,size[1]+165),'white');d=ImageDraw.Draw(canvas)
    d.text((20,8),f"源图与现有冰面坐标 · {entry['source_seconds']:.2f}秒 · PTS {entry['pts']}",font=font(28),fill='black')
    d.text((20,48),'左：同帧原片与0.25米辅助坐标网格；右：对应的冰面落点。没有使用模型鞋位置。',font=font(19),fill='black')
    canvas.paste(photo.convert('RGB'),(24,82));canvas.paste(plot,(size[0]+48,82))
    d.text((20,size[1]+98),'这些是按当前相机、场地尺度与Z=0回算的坐标；阴影/界线分界及标定误差不能被数值闭环消除。',font=font(18),fill='black')
    d.text((20,size[1]+129),'绿色为完整下缘的视线落冰面足迹，不把整条曲刃都宣称为真正触冰；黄色为独立保存的源交界候选。',font=font(18),fill='black')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);canvas.save(output)
    return dict(pts=entry['pts'],path=str(output),sha256=sha(output),source_midpoint_uv=uv.tolist(),xy_conditional_m=xy.tolist(),pixel_radius_max_offset_m=radius)


def build_review(inputs,annotations,contacts,output):
    import csv,html
    inputs=Path(inputs);annotations=Path(annotations);contacts=Path(contacts);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    source=read(inputs);doc=read(annotations);mapped=read(contacts);lookup={r['pts']:r for r in source['frames']};ml={r['pts']:r for r in mapped['frames']}
    if set(ml)!=set(r['pts']for r in doc['frames']):raise ValueError('Source and plane mapping PTS differ')
    images=[];maps=[]
    for row in doc['frames']:
        entry=lookup[row['pts']];images.append(render_frame(entry,row,output/'structures'/f"source_structure_{row['pts']}.png"));maps.append(render_ice_map(entry,row,ml[row['pts']],output/'ice'/f"source_ice_{row['pts']}.png"))
    with (output/'ice_locations.csv').open('w',encoding='utf-8-sig',newline='')as f:
        writer=csv.writer(f);writer.writerow(['pts','source_seconds','source_midpoint_u','source_midpoint_v','X_current_scale_m','Y_current_scale_m','source_pixel_radius_max_offset_m','source_interface_status'])
        for row in maps:writer.writerow([row['pts'],row['pts']/12800.,*row['source_midpoint_uv'],*row['xy_conditional_m'],row['pixel_radius_max_offset_m'],ml[row['pts']]['features']['ice_interface']['source_status']])
    page=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>22帧源图鞋刀结构与冰面定位</title><style>body{font:17px system-ui;background:#20252d;color:#eef0f3;margin:24px}a{color:#8bceff}section{margin:28px 0;padding:16px;background:#303741}img{width:100%;height:auto}summary{cursor:pointer}nav{line-height:2.2;position:sticky;top:0;background:#20252d}</style><h1>22帧源图鞋刀结构与冰面定位</h1><p>三联图：原生原片 / 完整源图描线 / 同坐标线稿。没有使用模型生成源轮廓。本轮仅核对源图与回算坐标，没有运行人体求解。</p><p>实线为直接描线，虚线保留局部分界或推断。冰面坐标基于当前相机/场地；像素范围未包含相机、场景与尺度标定误差。</p><nav>']
    for row in images:page.append(f'<a href="#{row["pts"]}">{row["pts"]/12800.:.2f}秒</a>　')
    page.append('</nav>')
    for image,maprow in zip(images,maps):
        pts=image['pts'];a=Path(image['path']).relative_to(output).as_posix();b=Path(maprow['path']).relative_to(output).as_posix()
        page.append(f'<section id="{pts}"><h2>{pts/12800.:.2f}秒 · PTS {pts}</h2><a href="{a}"><img loading="lazy" src="{a}" alt="原片、完整描线与线稿"></a><details><summary>冰面位置 X={maprow["xy_conditional_m"][0]:.3f}m / Y={maprow["xy_conditional_m"][1]:.3f}m</summary><a href="{b}"><img loading="lazy" src="{b}" alt="同帧冰面坐标回算"></a></details></section>')
    page.append('</html>');(output/'index.html').write_text(''.join(page),encoding='utf-8')
    report=dict(source_frames=len(images),source_features=sum(len(r['features'])for r in doc['frames']),structures=images,ice_maps=maps,source_annotation_sha256=sha(annotations),plane_mapping_sha256=sha(contacts),fit_run=False,gui_opened=False,visual_acceptance=False)
    (output/'review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--annotations',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--render',type=Path);p.add_argument('--contacts',type=Path);p.add_argument('--review-dir',type=Path)
    a=p.parse_args();doc=assemble(a.inputs,a.annotations,a.output)
    if a.render:
        lookup={int(r['pts']):r for r in read(a.inputs)['frames']};items=[render_frame(lookup[r['pts']],r,a.render/f"source_structure_{r['pts']}.png")for r in doc['frames']]
        (a.render/'images.json').write_text(json.dumps(items,ensure_ascii=False,indent=2),encoding='utf-8')
    if a.contacts and a.review_dir:build_review(a.inputs,a.output,a.contacts,a.review_dir)
    print(json.dumps(dict(frames=len(doc['frames']),output=str(a.output)),ensure_ascii=False))
if __name__=='__main__':main()
