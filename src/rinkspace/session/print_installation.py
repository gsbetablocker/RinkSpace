"""Static installation review through the shared native print/course/camera.

No optimization, source extraction or historical preparation is performed.
"""
from __future__ import annotations
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from PIL import Image, ImageDraw
from ..core.native import NativeCore
from ..domain.print_assets import sample_instance, resolve_instance
from ..domain.field_model import sample_model
from ..review.fonts import review_font
from .board_preview import Canvas, _mapping, _project_display, BG, INK

TEAL='#00ba9d'
HOLE='#ec9243'
NAVY='#173d68'
EDGE='#a245a3'


def _is_hole(path):
    return path['role'] in ('letter_hole_boundary','logo_hole_boundary')


def _edge_offset(path,distance=25):
    side=path.get('boundary_side',path['id'].rsplit('.',1)[-1])
    return {'top_edge':[0,-distance],'right_vertical':[distance,0],
            'bottom_edge':[0,distance],'left_vertical':[-distance,0]}.get(side,[distance,0])


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def ref(path):
    path=Path(path).resolve()
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def verified_source_frame(camera, video_path, scope_path, pts):
    """Decode one registered native exposure, without trusting a PNG caption.

    A frozen source scope supplies the expected video identity. The video bytes,
    decoded time base, dimensions and actual frame PTS must all agree. This is
    a review operation, not a new observation or solver-start qualification chain.
    """
    if type(pts) is not int:
        raise ValueError('D3 requires native integer PTS')
    scope = read(scope_path)
    source, clock = scope['source'], scope['source_clock']
    matches = [shot for shot in scope['shots'] if shot['id'] == camera['shot_id']]
    if len(matches) != 1 or camera['shot_id'] not in scope['required_shot_ids']:
        raise ValueError('D3 camera shot is absent or ambiguous in source scope')
    expected_tb = Fraction(clock['time_base_num'], clock['time_base_den'])
    camera_tb = Fraction(camera['time_base_num'], camera['time_base_den'])
    if (source['id'] != camera['source_id'] or expected_tb != camera_tb or
            (source['width'], source['height']) != (camera['image_width'], camera['image_height'])):
        raise ValueError('D3 camera and registered source identity/clock/dimensions differ')
    window = matches[0]['processing']
    if not (window['first_pts'] <= pts < window['stop_pts_exclusive'] and
            camera['first_pts'] <= pts < camera['stop_pts_exclusive']):
        raise ValueError('D3 PTS outside registered shot or camera')
    expected = source.get('sha256', '')
    if not re.fullmatch(r'[0-9a-f]{64}', expected):
        raise ValueError('D3 requires the registered original video SHA256')
    path = Path(video_path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError('D3 original video SHA256 differs from registered source')
    # Reuse the complete-review decoder's exact-PTS checks. No frame/fps seek,
    # nearest-frame fallback, rescaling or interpolation is used here.
    from ..review.render import source_frames
    decoded = list(source_frames(path, {**source, 'time_base_num': expected_tb.numerator,
        'time_base_den': expected_tb.denominator}, [pts]))
    if len(decoded) != 1 or decoded[0][0] != pts:
        raise ValueError('D3 did not decode the requested native exposure')
    rgb = np.ascontiguousarray(decoded[0][1][:, :, ::-1])
    return Image.fromarray(rgb), {
        'status': 'native_pts_decoded', 'native_source_verified': True,
        'video': {'path': str(path), 'sha256': expected}, 'scope': ref(scope_path),
        'source_id': source['id'], 'pts': pts,
        'time_base_num': expected_tb.numerator, 'time_base_den': expected_tb.denominator,
        'decoded_RGB_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(),
        'verification_meaning': 'registered video bytes and native exposure; no visual-fit approval',
    }


def bound_scene_assets(scene, scene_path, assets_path, instance_id):
    """The selected print and context must come from this scene's exact assets."""
    from ..domain.scene_assets import read_scene_assets
    registered = read_scene_assets(scene, scene_path)
    by_instance = {}
    for reference, assets in registered:
        for instance in _selected_instances(reference, assets):
            identity = instance['id']
            if identity in by_instance:
                raise ValueError('duplicate print instance in current scene: '+identity)
            by_instance[identity] = reference
    if instance_id not in by_instance:
        raise ValueError('selected print instance is not bound to current scene: '+instance_id)
    selected = ref(assets_path)
    if selected['sha256'] != by_instance[instance_id]['sha256']:
        raise ValueError('selected print asset differs from current scene binding: '+instance_id)
    return registered


def _selected_instances(reference, assets):
    """Iterate only the identities admitted by a verified scene reference.

    One asset can retain an omitted paired instance, and disjoint references can
    select different instances from the same asset. Neither expands the scene.
    """
    by_id = {instance['id']: instance for instance in assets['instances']}
    selected = reference.get('instance_ids',
                             [reference['instance_id']] if 'instance_id' in reference else [])
    return [by_id[identity] for identity in selected]


def _paint_shapes(canvas,paths,xy,*,fill=False):
    for path in sorted(paths,key=_is_hole):
        points=xy(path)
        hole=_is_hole(path)
        if fill and path.get('closed') and path.get('display_fill')!='none':
            canvas.draw.polygon([tuple(q) for q in points],fill=BG if hole else NAVY)
        canvas.line(points,HOLE if hole else NAVY if fill else TEAL,2)


def unfolded(data,out):
    # Dense, multi-row words need a separate identifier legend. Putting every
    # hole ID into the inter-row gap erased neighbouring letter strokes.
    legend=len(data['paths'])>24
    legend_rows=(len(data['paths'])+3)//4
    c=Canvas((2400,810+(75+29*legend_rows if legend else 0)))
    label=data['instance_id'].rsplit('/',1)[-1]
    c.text((65,55),f'D3 · {label} 展开面（从冰内观看）',43)
    c.text((65,114),'当前所选轮廓与内孔 → 同一 native 安装 chart；没有重新打字、替换字体或读取背景纹理。',27)
    # Display station right-to-left, as on this upper board from the ice.
    points=np.vstack([p['sz'] for p in data['paths']])
    mapping=_mapping(points*[-1.,1.],(100,210,2200,345))
    xy=lambda p:mapping(np.asarray(p['sz'])*[-1.,1.])
    _paint_shapes(c,data['paths'],xy,fill=True)
    by_id={p['id']:p for p in data['paths']}
    for i,path in enumerate(data['paths'],1):
        if legend:
            name=path['id'].removeprefix(label+'.print.').removeprefix(label+'.')
            c.text((65+(i-1)%4*575,870+(i-1)//4*29),
                f'{i:02}  {name}',19,HOLE if _is_hole(path) else INK)
            continue
        q=xy(path);at=(q.min(axis=0)+q.max(axis=0))/2
        if path['role']=='panel_outer_boundary':
            offset=_edge_offset(path)
            c.text(at+offset,f'{i:02}',21,INK,'mm',True)
            continue
        if _is_hole(path):
            parent=by_id.get(path.get('parent_contour_key'));support=xy(parent) if parent else q
            label_y=support[:,1].max()+18+(i%2)*12
        elif path.get('geometry',{}).get('kind')=='unfolded_square':
            label_y=q[:,1].min()+22
        else:label_y=q[:,1].min()-18-(i%2)*12
        c.text((at[0],label_y),f'{i:02}',18,HOLE if _is_hole(path) else INK,'mm',True)
    p=data['parameters']
    c.text((65,660),f'共享尺寸初值：宽 {p[3]:.8f} L；高 {p[4]:.8f} L。全部尺寸和模板布局尚未重新拟合。',27,'#ad7225')
    c.text((65,708),f'横轴 s 向左增加，纵轴 z 向上；{sum(_is_hole(p) for p in data["paths"])} 条内孔独立保留；开放轮廓不补闭合边。',25)
    missing=' / '.join(data['unlocalized_related_instances']) or '无'
    c.text((65,758),f'未定位的同模板实例：{missing}。' + ('其位置仍为空，没有生成三维位置或源像素。' if data['unlocalized_related_instances'] else '两实例共用模板和尺寸，安装位置各自独立。'),25)
    if legend:
        c.text((65,823),'实体编号索引（对应源图、三维图与 geometry.json；文字区不覆盖编号）',24)
    c.image.save(out/'unfolded.png')


def board_view(data,board,core,out):
    c=Canvas((2400,1580))
    label=data['instance_id'].rsplit('/',1)[-1]
    c.text((65,55),f'D3 · 同一 {label} 沿连续三维围挡',43)
    c.text((65,113),'全部字形、内孔和边界的 XYZ 来自 native kind5 → 连续 B(s,z)；无空间平面矩形替代。',27)
    c.text((65,162),'原安装初值，尚未重新拟合。主图高度按真实比例显示；左上角仅作全场位置指示。',26,'#ad7225')
    sz=np.vstack([p['sz'] for p in data['paths']]);a,b=sz[:,0].min(),sz[:,0].max()
    station=np.linspace(a-.07,b+.03,801)
    low=np.asarray(core.board_points(board,station,np.zeros_like(station)))
    high=np.asarray(core.board_points(board,station,np.full_like(station,board[5])))
    # Display from inside the rink along this installation, including opposite
    # board instances. The view basis changes; native world XYZ does not.
    mid=float((a+b)/2)
    pair=np.asarray(core.board_points(board,[mid-1e-5,mid+1e-5],[0.,0.]))
    tangent=pair[1,:2]-pair[0,:2];tangent/=np.linalg.norm(tangent)
    right=-tangent;inward=np.array([tangent[1],-tangent[0]])
    if np.dot(inward,-pair[0,:2])<0:inward=-inward
    yaw=np.deg2rad(-12.);elevation=np.deg2rad(33.)
    def project(q):
        q=np.asarray(q);horizontal=q[:,:2]@right;depth=q[:,:2]@inward
        # Inward depth points toward the observer. Viewed from above the ice,
        # it projects downward; world height still projects upward.
        return np.c_[np.cos(yaw)*horizontal+np.sin(yaw)*depth,
            -np.sin(elevation)*(-np.sin(yaw)*horizontal+np.cos(yaw)*depth)+np.cos(elevation)*q[:,2]]
    mapping=_mapping(np.vstack([project(low),project(high)]),(100,520,2180,620))
    qlow=mapping(project(low));qhigh=mapping(project(high))
    for i in range(len(station)-1):
        c.draw.polygon([tuple(q) for q in [qlow[i],qlow[i+1],qhigh[i+1],qhigh[i]]],fill='#eaf0f2')
    for s in np.linspace(station[0],station[-1],22):
        q=np.asarray(core.board_points(board,[s,s],[0.,board[5]]))
        c.line(mapping(project(q)),'#afc1c7',1)
    c.line(qlow,'#15887f',4);c.line(qhigh,'#286fa1',4)
    _paint_shapes(c,data['paths'],lambda p:mapping(project(p['xyz'])),fill=True)
    # Mark actual course joins; these are geometric supports, not image reads.
    for j,s in enumerate(core.board_course_chart(board)):
        if station[0]<s<station[-1]:
            q=np.asarray(core.board_points(board,[s,s],[0.,board[5]]))
            pix=mapping(project(q));c.line(pix,EDGE,2)
            c.text(pix[1]+[0,-30],f'J{j} 直/弯接缝',22,EDGE,'mm',True)
    for i,path in enumerate(data['paths'],1):
        q=mapping(project(path['xyz']));at=(q.min(axis=0)+q.max(axis=0))/2
        if path['role']=='panel_outer_boundary':
            offset=_edge_offset(path,28)
        else:offset=[0,48 if _is_hole(path) else -48 if i%2 else -77]
        c.text(at+offset,f'{i:02}',19,HOLE if _is_hole(path) else INK,'mm',True)
    full_s=np.linspace(0.,core.board_perimeter(board),1025)
    ring=np.asarray(core.board_points(board,full_s,np.zeros_like(full_s)))
    locator=_mapping(ring[:,:2],(100,230,530,230))
    c.line(locator(ring[:,:2]),'#9bafb5',2)
    ad_s=np.linspace(a,b,301);ad=np.asarray(core.board_points(board,ad_s,np.zeros_like(ad_s)))
    c.line(locator(ad[:,:2]),TEAL,5)
    c.text((100,485),f'{label} 全场位置（俯视）',21)
    c.text((65,1250),'绿：围挡底沿；蓝：顶沿；紫：直/弯几何接缝；灰网格仅为绘制支持，无观测权重。',26)
    breaks=core.board_course_chart(board);P=breaks[-1]
    curved=any(max(a,start+k*P)<min(b,end+k*P) for k in range(int(np.floor(a/P))-1,int(np.floor(b/P))+2)
        for j,(start,end) in enumerate(zip(breaks[:-1],breaks[1:])) if j%2)
    c.text((65,1304),'本实例包含真实弯角段，字形逐点随围挡弯曲。' if curved else '本实例初值位于直段；旁边为同一连续围挡。',26)
    c.text((65,1358),'全部 8 个分段接缝及闭合缝已用同一 native 安装路径检查连续性；采样点不被固定在某一段。',25)
    c.text((65,1412),f'编号 01–{len(data["paths"]):02} 与展开图、源投影及数据一一对应；未建模的细节不补成材料边。',25)
    c.text((65,1466),'原源像素、实体身份与观测相关组保留；此图是安装前向审核，尚未重新拟合。',25,'#ad7225')
    c.text((65,1520),'主图为正交展示视角；源图由各自镜头的完整时变相机另行投影，未用展示视角代替标定。',24)
    c.image.save(out/'three_d.png')


def _projection(core,camera,xyz,pts):
    time=(int(pts)-int(camera['reference_pts']))*camera['time_base_num']/camera['time_base_den']
    return core.project_motion(camera['rig'],camera['controls'],camera['first_control_time'],
        camera['control_dt'],camera['image_height'],time,np.asarray(xyz))


def course_continuity(core,board,parameters):
    """Synthetic coordinate trials only; do not change any instance placement."""
    width,height=parameters[3:];eps=1e-7;records=[]
    for join in core.board_course_chart(board):
        # Raw photographed chart + pi crosses the join at u=.5.
        trial=[join+width/2,parameters[1],np.pi,width,height]
        stations=[.5-eps/width,.5,.5+eps/width]
        sz=np.asarray(core.sample_print_chart(trial,[[0.,0.],[1.,0.]],stations))
        xyz=np.asarray(core.sample_path(board,5,trial,[[0.,0.],[1.,0.]],stations))
        direct=np.asarray(core.board_points(board,sz[:,0],sz[:,1]))
        span=float(np.linalg.norm(xyz[2]-xyz[0]));error=float(np.max(abs(xyz-direct)))
        if span>2.001*eps or error>1e-12:raise ValueError('native installed print is discontinuous at course join')
        records.append({'join_station':float(join),'trial_chord':span,'chart_board_max_abs':error})
    return {'synthetic_coordinate_trials_not_instance_locations':True,'source_observations':0,
        'eps':eps,'joins':records,'actual_instance_parameters_changed':False}


def source_view(data,scene,core,camera,frame,pts,out,context_prints=(),source_verification=None):
    source_verification = source_verification or {
        'status': 'unverified_image_preview', 'native_source_verified': False}
    verified = source_verification.get('native_source_verified') is True
    image=Image.open(frame).convert('RGB')
    if image.size!=(camera['image_width'],camera['image_height']):raise ValueError('source/camera image-size mismatch')
    sizes=[len(p['xyz']) for p in data['paths']]
    projection=_projection(core,camera,np.vstack([p['xyz'] for p in data['paths']]),pts)
    uv=projection['uv'];valid=(projection['status']==0)&np.isfinite(uv).all(axis=1)
    visible=valid&(uv[:,0]>=0)&(uv[:,0]<image.width)&(uv[:,1]>=0)&(uv[:,1]<image.height)
    print_valid=valid.copy();print_visible=visible.copy()
    if not np.any(visible):raise ValueError('no visible valid projection for this print/source exposure')
    parts=np.split(uv,np.cumsum(sizes)[:-1]);valid_parts=np.split(valid,np.cumsum(sizes)[:-1])
    overlay=image.copy();pen=ImageDraw.Draw(overlay)
    font=review_font(15)
    for i,(path,q,ok) in enumerate(zip(data['paths'],parts,valid_parts),1):
        color=HOLE if _is_hole(path) else TEAL
        start=None
        for j,success in enumerate(np.r_[ok,False]):
            if success and start is None:start=j
            if not success and start is not None:
                if j-start>=2:pen.line([tuple(p) for p in q[start:j]],fill=color,width=1,joint='curve')
                start=None
        in_frame=ok&(q[:,0]>=0)&(q[:,0]<image.width)&(q[:,1]>=0)&(q[:,1]<image.height)
        shown=q[in_frame]
        if not len(shown):continue
        at=(shown.min(axis=0)+shown.max(axis=0))/2
        y=shown[:,1].max()+8 if _is_hole(path) else shown[:,1].min()-18-(i%2)*16
        if path['role']=='panel_outer_boundary':
            at=shown[len(shown)//2]+_edge_offset(path,24);y=at[1]
        pen.text((at[0],y),f'{i:02}',fill=color,font=font,anchor='mm',stroke_width=1,stroke_fill='#10262b')
    bounds=np.r_[uv[visible].min(axis=0)-[70,75],uv[visible].max(axis=0)+[65,65]]
    bounds=np.rint(bounds).astype(int);bounds[:2]=np.maximum(bounds[:2],[0,0]);bounds[2:]=np.minimum(bounds[2:],image.size)
    box=tuple(map(int,bounds));rawcrop=image.crop(box);overcrop=overlay.crop(box)
    # A nearly out-of-frame projection can leave a very narrow crop. Preserve
    # its aspect ratio without creating a many-metres-tall comparison card.
    display_scale=min(2240/rawcrop.width,1100/rawcrop.height)
    target_w=max(1,round(rawcrop.width*display_scale))
    target_h=max(1,round(rawcrop.height*display_scale))
    image_left=(2400-target_w)//2
    c=Canvas((2400,435+2*target_h))
    label=data['instance_id'].rsplit('/',1)[-1]
    seconds=pts*camera['time_base_num']/camera['time_base_den']
    mode='同源投影对照' if verified else '图像预览（源曝光未核实）'
    c.text((65,55),f'D3 · {label} {mode} · {camera["shot_id"]} / {seconds:.3f}s',43)
    clock_text=f'PTS {pts}，time_base = {camera["time_base_num"]}/{camera["time_base_den"]}'
    c.text((65,113),f'已核对原视频 SHA 与原生曝光，{clock_text}；原时变相机冻结。' if verified else
        f'图片未绑定原视频曝光；{clock_text} 仅用于相机计算，不能作同源验收。',26)
    c.text((65,178),f'上：'+('原片像素' if verified else '待核验图片')+f'（{label} 当前可见部分；裁切、遮挡不补成取点）',28)
    c.image.paste(rawcrop.resize((target_w,target_h),Image.Resampling.LANCZOS),(image_left,215))
    c.text((65,245+target_h),'下：'+('同一原片' if verified else '同一待核验图片')+' + 当前所选字形、内孔和明确边界的 native 投影',28)
    c.image.paste(overcrop.resize((target_w,target_h),Image.Resampling.LANCZOS),(image_left,285+target_h))
    footer=315+2*target_h
    c.text((65,footer),'青：所选明确轮廓及外边；橙：内孔。遮挡处仅显示模型投影，没有补造源观测。',25)
    c.text((65,footer+45),'原安装尺寸与新模板布局仍为初值，投影存在可见残差；没有在 D3 中拟合相机或广告。',26,'#ad7225')
    c.text((65,footer+90),'编号与展开图、三维图和 geometry.json 中的同一实体路径一一对应。',25)
    c.image.save(out/'source.png')
    overlay.save(out/'source_ad.png')
    # Full scene context: all current field entities and identifiers, same camera.
    full=overlay.copy();d=ImageDraw.Draw(full);field=sample_model(scene,core,samples=129)
    field_failures=0
    for shape in field['shapes']:
        xyz=shape['xyz']
        if len(xyz)==2:
            xyz=core.sample_path(scene['board'],shape['kind'],shape['parameters'],[],np.linspace(0.,1.,129))
        result=_projection(core,camera,xyz,pts);q=result['uv'];status=result['status']
        field_failures+=int(np.count_nonzero(status))
        valid=(status==0)&np.isfinite(q).all(axis=1)
        # Crop only for display. Never connect across a failed camera root.
        start=None
        for j,ok in enumerate(np.r_[valid,False]):
            if ok and start is None:start=j
            if not ok and start is not None:
                if j-start>=2:d.line([tuple(p) for p in q[start:j]],fill='#c270dd' if shape['stage']=='M5' else '#ecb74d',width=1)
                start=None
        visible=valid&(q[:,0]>=0)&(q[:,0]<image.width)&(q[:,1]>=0)&(q[:,1]<image.height)
        indices=np.flatnonzero(visible)
        if len(indices):
            at=q[indices[len(indices)//2]]
            d.text(tuple(at),shape['label'].removeprefix('ice.'),font=font,fill='#ffdd8a',
                stroke_width=1,stroke_fill='#153447',anchor='mm')
    context_instances=[]
    for other in context_prints:
        context_instances.append(other['instance_id'])
        for path in other['paths']:
            result=_projection(core,camera,path['xyz'],pts);q=result['uv']
            ok=(result['status']==0)&np.isfinite(q).all(axis=1);start=None
            for j,success in enumerate(np.r_[ok,False]):
                if success and start is None:start=j
                if not success and start is not None:
                    if j-start>=2:d.line([tuple(v) for v in q[start:j]],fill='#e7a8e3',width=1)
                    start=None
            in_frame=ok&(q[:,0]>=0)&(q[:,0]<image.width)&(q[:,1]>=0)&(q[:,1]<image.height)
            index=np.flatnonzero(in_frame)
            if len(index):
                at=q[index[len(index)//2]]
                label=other['instance_id'].rsplit('/',1)[-1]+'.'+path['id'].split('.',1)[-1]
                d.text(tuple(at),label,font=font,fill='#e7a8e3',stroke_width=1,stroke_fill='#153447',anchor='mm')
    d.text((30,30),f'D3 / {camera["shot_id"]} {seconds:.3f}s / PTS {pts} / '+
        ('native source verified' if verified else 'UNVERIFIED IMAGE')+' / initial, NOT REFITTED',
        font=font,fill='#ffffff',stroke_width=2,stroke_fill='#153447')
    full.save(out/'source_full.png')
    return {'source_pts':pts if verified else None,'source_seconds':seconds if verified else None,
        'projection_pts':pts,'projection_seconds':seconds,'crop_xyxy':box,
        'source_id':camera['source_id'] if verified else None,
        'camera_source_id':camera['source_id'],'shot_id':camera['shot_id'],
        'source_verification':source_verification,
        'time_base_num':camera['time_base_num'],'time_base_den':camera['time_base_den'],
        'projected_path_count':len(parts),'projection_failures':int(np.count_nonzero(projection['status'])),
        'visible_projected_samples':int(np.count_nonzero(print_visible)),
        'camera_root_equality_px_max':float(np.max(abs(projection['root_equality_px'][print_valid]))),
        'context_field_paths':len(field['shapes']),'context_projection_failures':field_failures,
        'context_localized_print_instances':context_instances}


def render(scene_path,assets_path,instance_id,camera_path,frame_path,pts,library_path,out_dir,*,
           source_video_path=None,scope_path=None,parameter_state=None):
    out=Path(out_dir).resolve();project=Path(__file__).resolve().parents[3]
    if out.is_relative_to(project):raise ValueError('diagnostic output must be in Data')
    if out.exists():raise FileExistsError(out)
    scene=read(scene_path);assets=read(assets_path);camera_path=Path(camera_path).resolve();camera=read(camera_path)
    registered_assets=bound_scene_assets(scene,scene_path,assets_path,instance_id)
    if parameter_state is not None:
        from ..domain.parameters import compile_scene_parameters, LOCAL_LAYOUT_SCHEMA
        layout=parameter_state.layout
        charts=({b['id']:b['local_chart'] for b in layout['blocks'] if b.get('local_chart') is not None}
                if layout['schema']==LOCAL_LAYOUT_SCHEMA else None)
        if compile_scene_parameters(scene,scene_path,local_charts=charts).layout != layout:
            raise ValueError('D3 state does not match current scene layout')
        scene=parameter_state.apply_scene(scene)
    controls=camera_path.parent/camera['controls_ref'];camera['controls']=np.load(controls)
    if not camera['first_pts']<=pts<camera['stop_pts_exclusive']:raise ValueError('PTS outside this shot camera')
    if (frame_path is None) == (source_video_path is None):
        raise ValueError('choose exactly one registered source video or unverified frame preview')
    if source_video_path is not None:
        if scope_path is None:raise ValueError('registered source scope is required with source video')
        source_image,verification=verified_source_frame(camera,source_video_path,scope_path,pts)
    else:
        if scope_path is not None:raise ValueError('scope verifies a source video, not an arbitrary frame image')
        source_image=None
        verification={'status':'unverified_image_preview','native_source_verified':False,
            'verification_meaning':'supplied image has no verified binding to the requested source PTS'}
    core=NativeCore(library_path);data=sample_instance(assets,instance_id,scene['board'],core,parameter_state=parameter_state)
    template=resolve_instance(assets,instance_id)[1]
    data['unlocalized_related_instances']=[i['id'].rsplit('/',1)[-1]
        for reference,other_assets in registered_assets for i in _selected_instances(reference,other_assets)
        if i['template_id']==template['id'] and resolve_instance(other_assets,i['id'])[3]['values'] is None]
    data['path_index']=[{'display_id':f'{i:02}','id':p['id'],'role':p['role'],
        'closed':p.get('closed',False),'parent_contour_key':p.get('parent_contour_key')} for i,p in enumerate(data['paths'],1)]
    out.mkdir(parents=True)
    if source_image is not None:
        frame_path=out/'decoded_source.png'
        source_image.save(frame_path)
    data['course_continuity']=course_continuity(core,scene['board'],data['parameters'])
    unfolded(data,out);board_view(data,scene['board'],core,out)
    context_prints=[];unlocalized=[]
    for reference,other_assets in registered_assets:
        for instance in _selected_instances(reference,other_assets):
            identity=instance['id']
            if identity==instance_id:continue
            if resolve_instance(other_assets,identity)[3]['values'] is None:
                unlocalized.append(identity);continue
            context_prints.append(sample_instance(other_assets,identity,scene['board'],core,parameter_state=parameter_state))
    source=source_view(data,scene,core,camera,frame_path,int(pts),out,context_prints,verification)
    source['unlocalized_context_print_instances']=unlocalized
    report={'schema':'rinkspace.print_installation_review.v1','instance_id':instance_id,
        'scene':ref(scene_path),'assets':ref(assets_path),'camera':ref(camera_path),'controls':ref(controls),
        'source_frame':ref(frame_path),'library':str(Path(library_path).resolve()),
        'source_binding_verified':verification['native_source_verified'],
        'purpose':'source_bound_installation_review' if verification['native_source_verified'] else 'unverified_image_preview',
        'template_id':template['id'],'native_kind':5,'coordinate_pipeline':['material_uv','native_unwrapped_sz','native_BoardCourse_XYZ','native_motion_camera'],
        'source':source,'checks':{'chart_board_max_abs':data['chart_board_max_abs'],
            'template_paths':len(data['paths']),'hole_paths':sum(_is_hole(p) for p in data['paths']),
            'closed_paths':sum(p.get('closed',False) for p in data['paths']),
            'open_paths':sum(not p.get('closed',False) for p in data['paths'])},
        'dimensions_status':'尚未重新拟合；使用保留的历史安装与模板初值，米制未验证',
        'fit_executed':False,'new_source_observations':0,
        'parameter_state_consumed':parameter_state is not None,
        'layout_revision':None if parameter_state is None else parameter_state.layout['layout_revision'],
        'artifacts':{key:str(out/name) for key,name in [('unfolded','unfolded.png'),('three_d','three_d.png'),('source','source.png'),('source_full','source_full.png'),('geometry','geometry.json')]}}
    for name,value in [('geometry.json',data),('result.json',report)]:
        (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    return report
