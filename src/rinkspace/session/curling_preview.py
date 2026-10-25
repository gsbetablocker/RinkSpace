"""Native curling/Lane structural review. No fit and no source observations.

Reuses shared_geometry and sample_path; Pillow only maps their XYZ to pixels.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ..review.fonts import review_font
from ..core.native import NativeCore, NativeError
from ..domain.entity_graph import _code
from ..path_policy import guard_project_write
from ..domain.measurement import POLICY_ID, required_measurement

INK = '#243c50'
TEAL = '#087f83'
NAVY = '#285a81'
AMBER = '#c17716'
PURPLE = '#905ba4'
GRAY = '#8797a3'
BG = '#fafcfb'


def write(path, value):
    path = guard_project_write(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def import_current(directory, output):
    """Extract only curling/Lane inventory; do not resolve cameras or artwork."""
    directory = Path(directory).resolve(strict=True)
    model = json.loads((directory/'model.json').read_text(encoding='utf-8-sig'))
    family = model['scene']['structural_family']
    names = ['curling_block_id', 'lane_block_id', 'hockey_block_id', 'hockey_extra_block_id', 'hockey_end_block_id']
    with np.load(directory/'parameters.npz', allow_pickle=False) as arrays:
        blocks = [arrays[model['state']['arrays'][family[key]]].tolist() for key in names]
        b = arrays[model['state']['arrays'][family['board_block_id']]].tolist()
    blocks[0][2] = -1.5*blocks[0][3]
    blocks[2][0] = 0.
    left_shift=max(0.,blocks[3][0]-min(blocks[1][2:6]))
    blocks[1][2:6]=[x+left_shift for x in blocks[1][2:6]]
    paths = []
    for old in model['scene']['curves']:
        code = _code(old['label'])
        if code is None or not (100 <= code < 132 or 200 <= code < 204 or 300 <= code < 312):
            continue
        row = (code-100)//4 % 4+1 if code < 132 else code-200+1 if code < 204 else (code-300)//3+1
        role = 'named_ring_boundary' if code < 132 else 'physical_connection_centreline' if code < 204 else 'physical_lane_edge'
        item = {'id':old['id'], 'label':old['label'], 'entity_id':old['entity_id'], 'generated_code':code,
                'row':row, 'role':role, 'source_observation_ids':[], 'creates_source_observation':False}
        if 200 <= code < 204:
            item['endpoint_roles'] = ['support_at_F_centre_not_observed_terminal', 'independent_drawing_support_not_observed_terminal']
        if 300 <= code < 312:
            item['left_support_is_physical_endpoint'] = False
            item['right_corner_role'] = 'physical_structure_corner_not_new_UV'
        item['measurement']=required_measurement(item['label'])
        item['model_line_width']=0. if item['measurement']=='centreline' else None
        paths.append(item)
    if len(paths) != 48 or len({p['generated_code'] for p in paths}) != 48:
        raise ValueError('curling/Lane inventory must contain 32 ring paths, 4 connections and 12 physical edges')
    scene = {'schema':'rinkspace.curling_lane.v1', 'scene_id':'ankara_2026.curling_lane', 'revision':1,
             'status':'unfitted_structure_candidate', 'coordinate_unit':'rink_length_L_equals_1',
             'dimensions_status':'尚未重新拟合；米制未验证', 'source_model':str(directory/'model.json'),
             'source_parameters':str(directory/'parameters.npz'),
             'native_family':'shared_geometry_v2', 'shared_geometry_options':7, 'board':[.5,b[0]/2,*b[1:]],
             'shared_layout':sum(blocks, []), 'curling_parameters':blocks[0], 'lane_parameters':blocks[1],
             'native_layout_note':'完整33值仅匹配现有共享ABI；本子集仅求值冰壶及Lane，不导入其他实体。',
             'measurement_policy_id':POLICY_ID, 'paths':paths, 'constraints':{'F_R_columns':True,'paired_centres_same_y':True,'equal_row_pitch':True,'equal_bottom_top_long_side_clearance':True,'lane_left_not_before_left_goal_line':True,
             'common_four_ring_radii':True,'equal_lane_width':True,'collinear_lane_right_edges':True,
             'lane_left_closed':False,'equal_lane_lengths':False,'connection_end_equals_lane_right':False},
             'new_source_observations':0, 'fit_executed':False, 'user_approval':'not_reviewed'}
    write(output, scene)
    return scene


def geometry(scene, core):
    shapes, by_code = [], {}
    for p in scene['paths']:
        kind, parameters = core.shared_geometry(scene['board'],scene['shared_layout'],p['generated_code'],options=scene.get('shared_geometry_options',0))
        xyz = np.asarray(core.sample_path(scene['board'],kind,parameters,[],np.linspace(0.,1.,361 if kind==1 else 2)))
        shape = {**p,'kind':kind,'parameters':parameters,'xyz':xyz.tolist()}
        shapes.append(shape);by_code[p['generated_code']]=shape
    nodes, supports, intersections = [], [], []
    centres = []
    for side in range(2):
        for row in range(1,5):
            p=by_code[100+(side*4+row-1)*4+3]['parameters']
            n={'id':f'{"F" if side==0 else "R"}{row}','xyz':p[:3],'kind':'circle_centre',
               'source_uv':None,'source_observation_ids':[],'status':'structural_model_node_not_new_source_measurement'}
            centres.append(p[:3]);nodes.append(n)
    for row in range(1,5):
        upper,right,lower=[by_code[300+(row-1)*3+j] for j in range(3)]
        for side,edge in [('上',upper),('下',lower)]:
            xyz=core.line_line(edge['parameters'],right['parameters'])
            nodes.append({'id':f'N{row}{side}','xyz':list(xyz),'kind':'physical_lane_corner','source_uv':None,
                          'source_observation_ids':[],'status':'structural_model_node_not_new_source_measurement'})
        for side,p in [('上',upper['parameters'][:3]),('下',lower['parameters'][3:])]:
            supports.append({'id':f'L{row}{side}','xyz':p,'kind':'drawing_support','physical_endpoint':False,'source_uv':None})
        line=by_code[200+row-1]
        supports.append({'id':f'S{row}','xyz':line['parameters'][3:],'kind':'drawing_support',
                         'physical_endpoint':False,'source_uv':None,'independent_of_lane_right':True})
        try:
            point=core.line_line(line['parameters'],right['parameters'])
        except NativeError as exc:
            if exc.status != 3:
                raise
        else:
            intersections.append({'id':f'I{row}','xyz':list(point),'kind':'derived_finite_curve_intersection',
                                  'source_uv':None,'is_connection_terminal':False})
    xy=np.asarray(centres).reshape(2,4,3)
    radii=np.asarray([[by_code[100+i*4+j]['parameters'][3] for j in range(4)] for i in range(8)])
    widths=[abs(by_code[301+i*3]['parameters'][1]-by_code[301+i*3]['parameters'][4]) for i in range(4)]
    right_x=[by_code[301+i*3]['parameters'][0] for i in range(4)]
    lower_gap=float(xy[0,0,1]+scene['board'][1]);upper_gap=float(scene['board'][1]-xy[0,-1,1])
    checks={'F_column_x_spread':float(np.ptp(xy[0,:,0])),'R_column_x_spread':float(np.ptp(xy[1,:,0])),
            'paired_centre_y_difference':float(abs(xy[0,:,1]-xy[1,:,1]).max()),
            'row_pitch_spread':float(np.ptp(np.diff(xy[0,:,1]))),'ring_radius_spread':np.ptp(radii,axis=0).tolist(),
            'lane_width_spread':float(np.ptp(widths)),'lane_right_x_spread':float(np.ptp(right_x)),
            'physical_edges_per_lane':[3]*4,'left_caps':0,'equal_length_constraint':False,
            'lane_right_x':right_x[0],'connection_right_support_x':[by_code[200+i]['parameters'][3] for i in range(4)],
            'new_source_observations':0,'fit_executed':False,'bottom_long_side_centre_gap':lower_gap,'top_long_side_centre_gap':upper_gap,
            'long_side_gap_difference':abs(lower_gap-upper_gap),'outline_long_side_gap':lower_gap-float(radii[0,-1])}
    if scene.get('shared_geometry_options',0)&1 and abs(lower_gap-upper_gap)>1e-12:
        raise ValueError('centred native row constraint failed')
    errors=[checks[k] for k in ['F_column_x_spread','R_column_x_spread','paired_centre_y_difference','row_pitch_spread','lane_width_spread','lane_right_x_spread']]+checks['ring_radius_spread']
    if max(errors)>1e-12:
        raise ValueError('native curling/Lane structural relation failed: '+str(checks))
    for i in range(4):
        line=by_code[200+i]['parameters']
        if not (line[0] <= xy[0,i,0] < xy[1,i,0] < line[3] and abs(line[1]-xy[0,i,1])<1e-12):
            raise ValueError('connection does not pass both paired centres and extend right')
    if scene['constraints'].get('connection_stops_before_lane_right'):
        if not all(by_code[200+i]['parameters'][3] < right_x[i] for i in range(4)) or intersections:
            raise ValueError('connection support must stop before the Lane right edge')
    checks['connection_crosses_lane_right'] = bool(intersections)
    return {'schema':'rinkspace.curling_lane.geometry.v1','scene_id':scene['scene_id'],'native_generated':True,
            'shapes':shapes,'structural_nodes':nodes,'drawing_supports':supports,'derived_intersections':intersections,
            'checks':checks,'dimensions_status':scene['dimensions_status'],'new_source_observations':0}


class Canvas:
    def __init__(self,size):
        self.image=Image.new('RGB',size,BG);self.draw=ImageDraw.Draw(self.image);self.fonts={}
    def text(self,xy,message,size=24,color=INK,anchor='lm',box=False):
        font=review_font(size)
        if box:
            bounds=self.draw.textbbox(tuple(xy),message,font=font,anchor=anchor)
            self.draw.rectangle((bounds[0]-4,bounds[1]-2,bounds[2]+4,bounds[3]+2),fill=BG)
        self.draw.text(tuple(xy),message,font=font,fill=color,anchor=anchor)
    def line(self,xy,color,width=2):
        self.draw.line([tuple(p) for p in xy],fill=color,width=width,joint='curve')
    def marker(self,p,kind,color,size=6):
        x,y=p
        if kind=='circle':self.draw.ellipse((x-size,y-size,x+size,y+size),fill=color)
        elif kind=='square':self.draw.rectangle((x-size,y-size,x+size,y+size),fill=color)
        elif kind=='support':self.draw.ellipse((x-size,y-size,x+size,y+size),fill=BG,outline=color,width=3)
        else:self.line([(x-size,y-size),(x+size,y+size)],color,3);self.line([(x-size,y+size),(x+size,y-size)],color,3)


class Chart:
    def __init__(self,canvas,rect,xlim,ylim):
        self.c=canvas
        x,y,w,h=rect
        self.scale=min(w/(xlim[1]-xlim[0]),h/(ylim[1]-ylim[0]))
        self.center=np.array([(xlim[0]+xlim[1])/2,(ylim[0]+ylim[1])/2])
        self.origin=np.array([x+w/2,y+h/2])
    def xy(self,points):
        return (np.asarray(points)[...,:2]-self.center)*[self.scale,-self.scale]+self.origin
    def label(self,point,message,offset=(0,0),size=22,color=INK,anchor='mm'):
        self.c.text(self.xy(point)+offset,message,size,color,anchor,True)


def draw_geometry(canvas,chart,data,zoom=False):
    shapes=data['shapes'];c=canvas
    for p in sorted(shapes,key=lambda p:p['kind'] == 1):
        color=TEAL if p['kind']==1 else AMBER if p['role']=='physical_connection_centreline' else NAVY
        c.line(chart.xy(p['xyz']),color,3 if p['kind']!=1 else 2)
    for n in data['structural_nodes']:
        if zoom and n['id'].startswith('F'):continue
        is_centre=n['kind']=='circle_centre'
        c.marker(chart.xy(n['xyz']),'circle' if is_centre else 'square',TEAL if is_centre else NAVY,(3 if zoom else 2) if is_centre else 5)
        offset=(0,-78) if is_centre else (13,-12 if n['id'].endswith('上') else 14)
        anchor='mm' if is_centre else 'lm'
        if is_centre and zoom:
            outer=next(p for p in shapes if p['label']==f'ice.{n["id"]}:outer')
            offset=(-outer['parameters'][3]*chart.scale-15,-30)
            anchor='rm'
        chart.label(n['xyz'],n['id'],offset,24 if is_centre else 20,TEAL if is_centre else NAVY,anchor)
    for n in data['drawing_supports']:
        if zoom and n['id'].startswith('L'):continue
        color=AMBER if n['id'].startswith('S') else GRAY
        c.marker(chart.xy(n['xyz']),'support',color,6)
        offset=(-12,24) if n['id'].startswith('S') else (-12,-8 if n['id'].endswith('上') else 8)
        chart.label(n['xyz'],n['id'],offset,21,color,'rm')
    for n in data['derived_intersections']:
        c.marker(chart.xy(n['xyz']),'cross',PURPLE,6)
        chart.label(n['xyz'],n['id'],(13,-15),20,PURPLE,'lm')
    for row in range(1,5):
        lane=next(p for p in shapes if p['generated_code']==300+(row-1)*3)
        line=next(p for p in shapes if p['generated_code']==200+row-1)
        if not zoom:
            right=next(p for p in shapes if p['generated_code']==301+(row-1)*3)
            midpoint=(np.array(right['parameters'][:3])+right['parameters'][3:])/2
            chart.label(midpoint,f'E{row}右',(15,22),20,NAVY,'lm')
            a=np.asarray(lane['parameters'][:3]);b=np.asarray(lane['parameters'][3:])
            chart.label((a+b)/2,f'Lane {row} · E{row}上',(0,-13),21,NAVY)
            lower=next(p for p in shapes if p['generated_code']==302+(row-1)*3)
            chart.label((np.array(lower['parameters'][:3])+lower['parameters'][3:])/2,f'E{row}下',(0,14),21,NAVY)
            chart.label((np.array(line['parameters'][:3])+next(n['xyz'] for n in data['structural_nodes'] if n['id']==f'R{row}'))/2,f'M{row} 连接中线',(0,14),21,AMBER)
        else:
            right=next(p for p in shapes if p['generated_code']==301+(row-1)*3)
            midpoint=(np.array(right['parameters'][:3])+right['parameters'][3:])/2
            chart.label(midpoint,f'E{row}右',(18,24),20,NAVY,'lm')


def render(scene_path,library_path,out_dir):
    scene=json.loads(Path(scene_path).read_text(encoding='utf-8-sig'))
    if scene.get('schema')!='rinkspace.curling_lane.v1':raise ValueError('unknown curling/Lane model')
    out=Path(out_dir).resolve()
    if out.exists():raise FileExistsError(out)
    if out.is_relative_to(Path(__file__).resolve().parents[3]):raise ValueError('diagnostic figures must be in Data')
    core=NativeCore(library_path);data=geometry(scene,core)
    out.mkdir(parents=True)
    write(out/'geometry.json',data)
    c=Canvas((1900,1240))
    c.text((65,55),'冰壶 / Lane · 完整俯视模型',42)
    c.text((65,108),'同一 native 核心生成全部 32 条圈边界、12 条实体边和 4 条连接中线',25)
    c.text((65,150),'结构关系已表达；所有尺寸与支持数值尚未重新拟合，L = 1，米制未验证。',25,AMBER)
    q=scene['curling_parameters'];lane=scene['lane_parameters'];ys=[q[2]+i*q[3] for i in range(4)]
    chart=Chart(c,(80,205,1710,750),(min(lane[2:6])-.055,max(lane[6:10])+.055),(min(ys)-lane[1]/2-.045,max(ys)+lane[1]/2+.055))
    draw_geometry(c,chart,data)
    c.text((70,1005),'● F / R 圆心    ■ N 实体右角：结构节点；本步均未新增或绑定源 UV。',24,INK)
    c.text((70,1048),'○ L / S：仅绘制支持，不是真实端点；四条连接中线均止于 N 线左侧，不穿过右边线。',24,INK)
    c.text((70,1091),'Lane 左侧始终开口；四行只约束等宽和右边共线，不约束等长。M 右支持独立于 E 右。',24,INK)
    c.text((70,1134),f'当前初值：F 列 x={q[0]:.6f}；R 列 x={q[1]:.6f}；行距={q[3]:.6f}；外半径={q[4]:.6f}。',23,GRAY)
    c.text((70,1177),f'Lane 宽={lane[1]:.6f}；右边 x={lane[0]:.6f}。四层边界由八组圈共享同一组半径。',23,GRAY)
    c.image.save(out/'top.png')
    z=Canvas((1650,1390))
    z.text((65,52),'右端局部放大 · 右边线 ≠ 中线支持端',39)
    z.text((65,106),'同一份 native XYZ 等比例放大；所有数值尚未重新拟合。',25,AMBER)
    chart=Chart(z,(70,175,890,1085),(q[1]-q[4]-.025,max(lane[6:10])+.025),(min(ys)-lane[1]/2-.025,max(ys)+lane[1]/2+.03))
    draw_geometry(z,chart,data,True)
    x=1050
    lines=[('中线止于 N 线左侧',31,INK),('E1右–E4右：Lane 实体右边',25,NAVY),('N：右边与上下边的实体角点',23,NAVY),('S1–S4：独立的绘制支持端',25,AMBER),('中线不穿过 N 线，无共端约束',23,AMBER),('真实终点仍未获得源 UV 确认',24,GRAY),('左侧开口；未增加等长约束',23,GRAY)]
    for i,(t,size,color) in enumerate(lines):z.text((x,200+i*54),t,size,color)
    z.text((x,635),'保留的当前初值 / L',27)
    z.text((x,683),f'Lane 右边：{lane[0]:.6f}',25,NAVY)
    for i,v in enumerate(lane[6:10]):z.text((x,735+i*48),f'S{i+1} 支持：{v:.6f}',25,AMBER)
    for i,t in enumerate(['S1–S4 均在 N 线左侧。','端点数值各自独立，','仅用于本步绘制支持，','不代表真实终点已被证实。']):z.text((x,980+i*42),t,23,GRAY)
    z.text((65,1325),'圆心、实体角点和派生交点是模型定义；没有把它们伪装成新增源像素约束。',24)
    z.image.save(out/'right.png')
    result={'schema':'rinkspace.curling_lane.review.v1','scene_path':str(Path(scene_path).resolve()),'library':str(Path(library_path).resolve()),
            'top':str(out/'top.png'),'right':str(out/'right.png'),'geometry':str(out/'geometry.json'),
            'checks':data['checks'],'dimensions_status':scene['dimensions_status'],'fit_executed':False,'user_approval':'not_reviewed'}
    write(out/'result.json',result)
    return result