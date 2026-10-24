"""M1--M5 structural figures from one compiled native field model."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from PIL import Image
from ..core.native import NativeCore
from ..domain.field_model import sample_model
from .curling_preview import Canvas, Chart, write, INK, TEAL, NAVY, AMBER, GRAY, BG

RED='#b54d55'
BLUE='#3279b0'
BOARD='#547a72'
TRAP='#925e36'


def color(p):
    code=p.get('generated_code',-1)
    if p['stage']=='M5':return BOARD
    if code in (410,411):return BLUE
    if 420<=code<424:return TRAP
    if p['stage']=='M1':return TEAL if p['kind']==1 else AMBER if 200<=code<204 else NAVY
    return BLUE if code in (401,402) else RED


def path(chart,p,width=2):
    # Pixel clipping only; native source XYZ and finite support are retained.
    x,y,w,h=chart.rect
    layer=Image.new('RGBA',(w+1,h+1))
    from PIL import ImageDraw
    d=ImageDraw.Draw(layer)
    pp=chart.xy(p['xyz'])-[x,y]
    d.line([tuple(t) for t in pp],fill=color(p),width=width,joint='curve')
    chart.c.image.paste(layer,(x,y),layer)


def chart(canvas,rect,xlim,ylim):
    result=Chart(canvas,rect,xlim,ylim)
    result.rect=tuple(map(int,rect))
    return result


def draw(canvas,ax,shapes,stages=None):
    selected=[p for p in shapes if stages is None or p['stage'] in stages]
    board=next(p for p in shapes if p['label']=='board.bottom')
    # A display fill uses the very same native course; it creates no boundary.
    from PIL import ImageDraw
    x,y,w,h=ax.rect
    fill=Image.new('RGBA',(w+1,h+1));pen=ImageDraw.Draw(fill)
    pen.polygon([tuple(v) for v in ax.xy(board['xyz'])-[x,y]],fill='#f0f6f4')
    for p in selected:
        if p['kind']==6:
            pen.polygon([tuple(v) for v in ax.xy(p['xyz'])-[x,y]],fill='#cfe6f3')
    canvas.image.paste(fill,(x,y),fill)
    for p in sorted(selected,key=lambda p:(p['stage']=='M1',p['kind']==1)):
        path(ax,p,4 if p.get('generated_code') in (401,402) else 3 if p['stage'] in ('M2','M3','M4') else 2)


def centre_label(canvas,ax,p,label,size=25,offset=None):
    xyz=p['parameters'][:3];radius=p['parameters'][3]
    canvas.marker(ax.xy(xyz),'circle',color(p),2)
    ax.label(xyz,label,offset or (0,-radius*ax.scale-20),size,color(p))


def draw_equal_margins(canvas,ax,scene,by):
    samples=[np.asarray(by[103]['xyz']),np.asarray(by[115]['xyz'])]
    board=np.asarray(by[None]['xyz'])
    ylimits=(float(board[:,1].min()),float(board[:,1].max()))
    gaps=[]
    for i,points in enumerate(samples):
        at=points[np.argmin(points[:,1]) if i==0 else np.argmax(points[:,1])]
        target=np.array([at[0],ylimits[i],0.])
        gap=abs(target[1]-at[1]);gaps.append(gap)
        x=at[0]-.047
        a=ax.xy([x,at[1],0.]);b=ax.xy([x,target[1],0.])
        canvas.line([ax.xy(at),a],GRAY,1);canvas.line([ax.xy(target),b],GRAY,1)
        canvas.line([a,b],GRAY,2)
        direction=1 if b[1]>a[1] else -1
        for pt,sgn in [(a,direction),(b,-direction)]:
            canvas.line([pt+[-5,sgn*9],pt,pt+[5,sgn*9]],GRAY,2)
        canvas.text((a+b)/2+[-12,0],'d下' if i==0 else 'd上',23,GRAY,'rm',True)
    return gaps


def draw_faceoff_row_guides(canvas,ax,by):
    for left,right in ((409,406),(408,407)):
        a=ax.xy(by[left]['parameters'][:3]);b=ax.xy(by[right]['parameters'][:3])
        distance=float(np.linalg.norm(b-a))
        for start in np.arange(0.,distance,18.):
            end=min(distance,start+8.)
            canvas.line([a+(b-a)*start/distance,a+(b-a)*end/distance],'#b4c2c1',1)


def full_top(scene,data,out):
    c=Canvas((2400,1800))
    c.text((70,58),'共同场地 · M1–M5 完整几何模型',46)
    c.text((70,116),'A1 / A4、A2 / A3 各自共行，并以中场 C 线为镜面对称；既有圈、Lane 与轮廓关系保留',28)
    c.text((70,165),'同一 native 核心生成；全部尺寸与半轴数值尚未重新拟合，L = 1，米制未验证。',27,AMBER)
    ax=chart(c,(130,235,2140,1160),(-.555,.555),(-.30,.30))
    shapes=data['shapes'];by={p.get('generated_code'):p for p in shapes}
    draw(c,ax,shapes)
    draw_faceoff_row_guides(c,ax,by)
    for i in range(8):
        p=by[103+i*4];centre_label(c,ax,p,p['label'].removeprefix('ice.').split(':')[0],24)
    for code in range(405,410):
        p=by[code];centre_label(c,ax,p,p['label'].removeprefix('ice.').split(':')[0],28,(0,p['parameters'][3]*ax.scale+24))
    for code,label in [(400,'C'),(401,'B1'),(402,'B2'),(403,'G左'),(404,'G右')]:
        q=np.asarray(by[code]['xyz'])
        ax.label(q[0],label,(0,-24),28,color(by[code]))
        for xyz in q[[0,-1]]:c.marker(ax.xy(xyz),'support',GRAY,4)
    for row in range(4):
        lane=by[300+row*3];line=by[200+row]
        q=np.asarray(lane['xyz']);middle=(q[0]+q[-1])/2
        ax.label(middle,f'Lane {row+1}',(0,-14),23,NAVY)
        ax.label(np.asarray(line['xyz'])[0]+[.21,0,0],f'M{row+1}',(0,16),22,AMBER)
        endpoint=np.asarray(line['xyz'])[-1];c.marker(ax.xy(endpoint),'support',AMBER,5)
        ax.label(endpoint,f'S{row+1}',(-7,19),20,AMBER,'rm')
        for code,index in ((300+row*3,0),(302+row*3,-1)):
            c.marker(ax.xy(np.asarray(by[code]['xyz'])[index]),'support',GRAY,4)
    for code,label in [(410,'K左 半椭圆'),(411,'K右 半椭圆')]:
        p=by[code];ax.label(p['xyz'][len(p['xyz'])//2],label,(0,39),21,BLUE)
    for code,label in [(420,'T左下'),(421,'T左上'),(422,'T右下'),(423,'T右上')]:
        q=np.asarray(by[code]['xyz']);ax.label(q.mean(axis=0),label,(0,-17 if code%2 else 19),20,TRAP)
    gaps=draw_equal_margins(c,ax,scene,by)
    c.text((70,1475),'青绿：F/R 四层圈边界    深蓝：Lane 三条实体边    橙：连接中线    红：C / C0 / 球门与争球圈',27)
    c.text((70,1527),'蓝色直线：B1 / B2    淡蓝填充：两端半椭圆蓝区    棕：梯形边    灰绿：同一连续围挡',27)
    c.text((70,1579),'空心标记：绘制支持或未测连接点，不是新增源观测。中线 S1–S4 均在 N 线左侧；Lane 左侧开口。',26)
    c.text((70,1631),f'd上 = d下 = {gaps[0]:.6f} L（外圈至长边）；所有靶圈、Lane 和连接中线共用居中四行。',26)
    c.text((70,1683),'靶圈原有 center / inner / middle / outer 四层不变；它们及半椭圆蓝区取轮廓，其余线全部取宽度中线。',26)
    c.text((70,1735),'灰虚线仅示 A 圆心共行，d 为尺寸辅助；所有尺寸仍未重新拟合，辅助线不计入实体或观测。',26,GRAY)
    c.image.save(out/'full_top.png')


def midfield(scene,data,out):
    c=Canvas((1850,1370));c.text((65,55),'M2 / M3 · 共同中场骨架与成对争球关系',40)
    c.text((65,110),'A1 / A4 共下行，A2 / A3 共上行；左右圆心以 C 线为镜面对称；五个冰球圈共半径。',26)
    c.text((65,154),'共同方向固定到冰场短边；五条线与轮廓相接。所有尺寸数值尚未重新拟合。',25,AMBER)
    ax=chart(c,(80,230,1650,870),(-.49,.49),(-.25,.25))
    draw(c,ax,data['shapes'],{'M2','M3','M5'})
    by={p.get('generated_code'):p for p in data['shapes']}
    for code in range(405,410):centre_label(c,ax,by[code],by[code]['label'].removeprefix('ice.').split(':')[0],27)
    for code,label in [(400,'C'),(401,'B1'),(402,'B2'),(403,'G左'),(404,'G右')]:
        q=np.asarray(by[code]['xyz']);ax.label(q[0],label,(0,-24),28,color(by[code]))
    origin=ax.xy([0,0,0]);c.marker(origin,'cross',RED,7);c.text(origin+[10,23],'原点 O / C0 圆心',23,RED,box=True)
    c.text((65,1170),'四个争球圆心由共同左右位置与共同上下间隔生成；两门线位置仍各自独立。',26)
    c.text((65,1220),'蓝线间距、共同半径、球门位置和争球间隔均未作为固定现场尺寸；没有新增观测。',26)
    c.text((65,1270),'本图与完整图消费同一份 field/model.json，并调用相同 native 生成器。',26,GRAY)
    c.image.save(out/'midfield.png')


def ends(scene,data,out):
    c=Canvas((2400,1510));c.text((65,55),'M4 · 两端半椭圆蓝区与梯形边',43)
    c.text((65,111),'每端各有独立内向半轴 ax 与横向半轴 ay；两轴均可估，半轴数值尚未重新拟合。',26,AMBER)
    by={p.get('generated_code'):p for p in data['shapes']}
    for side,(x,limits,title) in enumerate([(75,(-.53,-.22),'左端'),(1270,(.22,.53),'右端')]):
        c.text((x,185),title,33)
        ax=chart(c,(x,250,1040,900),limits,(-.20,.20))
        # Draw on a bounded crop: offscreen entities are clipped only in pixels.
        draw(c,ax,data['shapes'],{'M3','M4','M5'})
        c.draw.rectangle((0 if side==0 else 1215,230,50 if side==0 else 1245,1170),fill=BG)
        circle_codes=(408,409) if side==0 else (406,407)
        for code in circle_codes:
            p=by[code];centre_label(c,ax,p,p['label'].removeprefix('ice.').split(':')[0],26)
        crease=by[410+side];code=410+side;goal=403+side
        q=np.asarray(crease['xyz']);middle=q[len(q)//2]
        ax.label(middle,'K左' if side==0 else 'K右',(48 if side==0 else -48,0),25,BLUE)
        c.text((x,1195),f'ax={crease["parameters"][3]:.6f}；ay={crease["parameters"][4]:.6f}（L，未重新拟合）',24,BLUE)
        ax.label(np.asarray(by[goal]['xyz'])[0]*.75+np.asarray(by[goal]['xyz'])[-1]*.25,'G左' if side==0 else 'G右',(-25 if side==0 else 25,0),25,RED)
        for p in data['shapes']:
            pc=p.get('generated_code',-1)
            if pc not in ((420,421) if side==0 else (422,423)):continue
            qq=np.asarray(p['xyz']);label=('T左' if side==0 else 'T右')+('下' if pc%2==0 else '上')
            ax.label(qq.mean(axis=0),label,(0,-24 if pc%2 else 24),22,TRAP)
        for n in data['nodes']:
            if n['display_kind']!='unmeasured_attachment':continue
            if (n['xyz'][0]<0)!=(side==0):continue
            at=ax.xy(n['xyz']);c.marker(at,'support',BLUE if n['id'].startswith('K') else TRAP,5)
            text=n['id'].replace('下','d').replace('上','u')
            offset=(-14 if side==0 else 14,-13 if n['xyz'][1]>0 else 18)
            ax.label(n['xyz'],text,offset,17,BLUE if n['id'].startswith('K') else TRAP,'rm' if side==0 else 'lm')
    c.text((65,1300),'K 的端点父实体：对应球门线 + 半椭圆。T 的 G 端：对应球门线；B 端：同一连续围挡。',26)
    c.text((65,1352),'空心节点表示几何连接 / 未测支持；没有把模型端点当作真实 UV。四条梯形边暂缺源端点读数。',26)
    c.text((65,1404),'旧轮廓包络只作半轴初值；官方源参考与历史读数分开保存，未混用时钟、未重新取点。',26,GRAY)
    c.image.save(out/'ends.png')


def source_reference(scene,out,*,scene_path):
    """Optional private source image, with metadata anchored to its scene.

    The public package includes the reference JSON but may withhold the image.
    Metadata errors remain errors; absent private media does not block the
    numeric structural preview. Neither lookup depends on the process CWD.
    """
    p=scene.get('endpoint_source_references',{}).get('path')
    if not p:return None
    scene_path=Path(scene_path).resolve()
    reference_path=Path(p)
    if not reference_path.is_absolute():reference_path=scene_path.parent/reference_path
    refs=json.loads(reference_path.read_text(encoding='utf-8-sig'))
    available=[]
    for ref in refs['official50_references']:
        image_path=Path(ref['path'])
        if not image_path.is_absolute():image_path=reference_path.parent/image_path
        if image_path.is_file():available.append((ref,image_path))
    if not available:return None
    ref,image_path=available[0]
    with Image.open(image_path) as original:im=original.convert('RGB')
    c=Canvas((2000,1510));c.text((50,55),'M4 · 已登记官方原片参考',40)
    c.text((50,110),f'源 {ref["source_id"]} · PTS {ref["pts"]} · {ref["time_base_num"]}/{ref["time_base_den"]} · {ref["seconds"]:.3f}s',25)
    im.thumbnail((1900,1090));c.image.paste(im,((2000-im.width)//2,160))
    c.text((50,1290),'原片中可见蓝色区域及红色曲边；人物遮挡与模糊仍保留。此图不证明模型端点像素已准确。',25)
    c.text((50,1340),'没有把模型投影或历史全场视频 UV 当作本片新增观测。右蓝区历史 36 个读数仍保留原时钟。',25)
    c.text((50,1390),'蓝区、梯形边端点未获得源确认，图中只标模型连接关系。',25,AMBER)
    c.text((50,1440),'完整源身份、历史读数与缺失图片登记见 scene/end_sources.json。',24,GRAY)
    c.image.save(out/'source_reference.png');return str(out/'source_reference.png')


def render(scene_path,library_path,out_dir):
    scene=json.loads(Path(scene_path).read_text(encoding='utf-8-sig'))
    out=Path(out_dir).resolve()
    if out.exists():raise FileExistsError(out)
    if out.is_relative_to(Path(__file__).resolve().parents[3]):raise ValueError('figures belong in Data')
    core=NativeCore(library_path);data=sample_model(scene,core)
    out.mkdir(parents=True)
    write(out/'geometry.json',data)
    full_top(scene,data,out);midfield(scene,data,out);ends(scene,data,out)
    reference=source_reference(scene,out,scene_path=scene_path)
    result={'schema':'rinkspace.field.review.v1','scene_id':scene['scene_id'],'scene_path':str(Path(scene_path).resolve()),
            'library':str(Path(library_path).resolve()),'top':str(out/'full_top.png'),'midfield':str(out/'midfield.png'),
            'ends':str(out/'ends.png'),'source_reference':reference,'geometry':str(out/'geometry.json'),
            'checks':data['checks'],'dimensions_status':scene['dimensions_status'],'new_source_observations':0,'fit_executed':False,'user_approval':'not_reviewed'}
    write(out/'result.json',result)
    return result
