"""Native continuous board geometry and display-only structural views.

All boundary XYZ comes from NativeCore. The display transformations do not
constitute a camera calibration or additional scene/observation model.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ..review.fonts import review_font
from ..core.native import NativeCore

BG = '#fafcfb'
INK = '#213e50'
TOP = '#216eac'
BOTTOM = '#0a8c87'
GRID = '#8296a3'


def _points(core, board, stations, height):
    return np.asarray(core.board_points(board, list(map(float, stations)), [float(height)]*len(stations)), dtype=np.float64)


def build_geometry(board, core, *, boundary_samples=2049, grid_count=64):
    """Native XYZ with actual finite-course join and vertical checks."""
    board = list(map(float, board))
    if len(board) != 6 or boundary_samples < 33 or grid_count < 8:
        raise ValueError('board[6], dense boundary and at least eight stations required')
    perimeter = core.board_perimeter(board)
    station = np.linspace(0., perimeter, boundary_samples)
    low = _points(core, board, station, 0.)
    high = _points(core, board, station, board[5])
    grid_station = np.linspace(0., perimeter, grid_count, endpoint=False)
    grid_low = _points(core, board, grid_station, 0.)
    grid_high = _points(core, board, grid_station, board[5])
    segments = []
    for segment in range(8):
        points = np.asarray(core.sample_path(board, 3, [segment, 0.], [], np.linspace(0., 1., 129)), dtype=np.float64)
        points_top = np.asarray(core.sample_path(board, 3, [segment, 1.], [], np.linspace(0., 1., 129)), dtype=np.float64)
        near_start = np.asarray(core.sample_path(board, 3, [segment, 0.], [], [0., 1e-7]), dtype=np.float64)
        near_end = np.asarray(core.sample_path(board, 3, [segment, 0.], [], [1.-1e-7, 1.]), dtype=np.float64)
        t0 = near_start[1]-near_start[0]; t0 /= np.linalg.norm(t0)
        t1 = near_end[1]-near_end[0]; t1 /= np.linalg.norm(t1)
        segments.append({'id':f'board_segment_{segment}', 'display_id':f'B{segment+1}',
                         'kind':'straight' if segment%2 == 0 else 'superellipse_corner',
                         'lower':points.tolist(), 'upper':points_top.tolist(),
                         'start_tangent_secant':t0.tolist(), 'end_tangent_secant':t1.tolist()})
    joins=[]
    for i, segment in enumerate(segments):
        following=segments[(i+1)%8]
        gap=float(np.linalg.norm(np.asarray(segment['lower'][-1])-following['lower'][0]))
        tangent_gap=float(np.linalg.norm(np.asarray(segment['end_tangent_secant'])-following['start_tangent_secant']))
        joins.append({'id':f'J{i+1}', 'parents':[segment['id'],following['id']],
                      'xyz':segment['lower'][-1], 'position_gap':gap,
                      'near_join_unit_secant_difference':tangent_gap,
                      'semantic_status':'derived_course_join_not_new_source_UV', 'source_uv':None})
    checks={'same_station_top_bottom_XY_max_abs':float(np.max(abs(low[:,:2]-high[:,:2]))),
            'vertical_grid_XY_max_abs':float(np.max(abs(grid_low[:,:2]-grid_high[:,:2]))),
            'vertical_grid_height_error_max_abs':float(np.max(abs((grid_high[:,2]-grid_low[:,2])-board[5]))),
            'closed_seam_XYZ_gap':float(np.linalg.norm(low[0]-low[-1])),
            'segment_join_XYZ_gap_max':max(j['position_gap'] for j in joins),
            'near_join_unit_secant_difference_max':max(j['near_join_unit_secant_difference'] for j in joins),
            'course_parts':8, 'straight_parts':4, 'corner_parts':4,
            'boundary_samples':len(station), 'vertical_station_count':len(grid_station),
            'new_source_observations':0, 'fit_executed':False}
    if max(checks[k] for k in ['same_station_top_bottom_XY_max_abs','vertical_grid_XY_max_abs',
                              'vertical_grid_height_error_max_abs','closed_seam_XYZ_gap','segment_join_XYZ_gap_max']) > 1e-11:
        raise ValueError('native continuous-board structural check failed: '+str(checks))
    if checks['near_join_unit_secant_difference_max'] > 1e-3:
        raise ValueError('native near-join tangent continuity check failed: '+str(checks))
    return {'schema':'rinkspace.board.review_geometry.v1','board':board,'perimeter':perimeter,
            'station':station.tolist(),'bottom':low.tolist(),'top':high.tolist(),
            'verticals':[{'id':f'V{i+1:02d}','station':float(s),'xyz':[grid_low[i].tolist(),grid_high[i].tolist()],
                          'source_uv':None,'semantic_status':'model_display_station_not_new_observation'} for i,s in enumerate(grid_station)],
            'segments':segments,'joins':joins,'checks':checks,
            'coordinate_unit':'rink_length_L_equals_1','dimensions_status':'尚未重新拟合；米制未验证',
            'native_geometry_only':True,'fit_executed':False,'new_source_observations':0}


class Canvas:
    def __init__(self,size):
        self.image=Image.new('RGB',size,BG);self.draw=ImageDraw.Draw(self.image);self.fonts={}
    def text(self,xy,text,size=26,color=INK,anchor='lm',box=False):
        font=review_font(size)
        if box:
            b=self.draw.textbbox(tuple(xy),text,font=font,anchor=anchor)
            self.draw.rectangle((b[0]-4,b[1]-3,b[2]+4,b[3]+3),fill=BG)
        self.draw.text(tuple(xy),text,font=font,fill=color,anchor=anchor)
    def line(self,points,color,width=2):
        self.draw.line([tuple(p) for p in points],fill=color,width=width,joint='curve')
    def dot(self,xy,color,radius=4):
        x,y=xy;self.draw.ellipse((x-radius,y-radius,x+radius,y+radius),fill=color)


def _mapping(points,rect):
    points=np.asarray(points,dtype=np.float64)
    lo=points.min(axis=0);hi=points.max(axis=0)
    x,y,w,h=rect
    scale=min(w/max(hi[0]-lo[0],1e-12),h/max(hi[1]-lo[1],1e-12))
    middle=(lo+hi)/2;origin=np.array([x+w/2,y+h/2])
    return lambda q: (np.asarray(q,dtype=float)-middle)*[scale,-scale]+origin


def _floor_color(shape):
    label=shape.get('label','')
    if 'crease' in label: return '#298ab9'
    if 'house_connection' in label: return '#c17716'
    if '.F' in label or '.R' in label: return '#087f83'
    if 'lane' in label: return '#285a81'
    return '#c15b48'


def _project_display(xyz,z_scale=3.):
    # Orthographic display rotation only; no world/scene values are changed.
    xyz=np.asarray(xyz,dtype=np.float64)
    yaw=np.deg2rad(30.);elevation=np.deg2rad(27.)
    u=np.cos(yaw)*xyz[:,0]-np.sin(yaw)*xyz[:,1]
    v=np.sin(elevation)*(np.sin(yaw)*xyz[:,0]+np.cos(yaw)*xyz[:,1])+np.cos(elevation)*xyz[:,2]*z_scale
    return np.c_[u,v]


def render(board, library_path, out_dir, *, scene_id='board_current_candidate', floor_shapes=None):
    out=Path(out_dir).resolve()
    if out.exists():raise FileExistsError(out)
    project=Path(__file__).resolve().parents[3]
    if out == project or project in out.parents:raise ValueError('diagnostic output must be outside the OneDrive checkout')
    core=NativeCore(library_path);data=build_geometry(board,core)
    out.mkdir(parents=True)
    floor_shapes=list(floor_shapes or [])
    low=np.asarray(data['bottom']);high=np.asarray(data['top'])
    top=Canvas((2100,1220))
    top.text((65,55),'M5 · 连续围挡俯视与同 station 上下沿',42)
    top.text((65,110),'全部边界与竖格来自同一 native；顶底的 XY 完全重合，未另造围挡公式。',27)
    top.text((65,156),'结构与拓扑展示；场地长宽、围挡高和圆角曲率数值尚未重新拟合。',26,'#b57625')
    xy=_mapping(low[:,:2],(120,235,1850,765))
    if floor_shapes:
        for shape in floor_shapes:
            points=np.asarray(shape['xyz'])
            top.line(xy(points[:,:2]),_floor_color(shape),2)
    top.line(xy(high[:,:2]),TOP,7);top.line(xy(low[:,:2]),BOTTOM,3)
    for v in data['verticals']:
        p=xy(np.asarray(v['xyz'])[0,:2]);top.dot(p,GRID,4)
    for j in data['joins']:
        p=xy(np.asarray(j['xyz'])[:2]);top.dot(p,'#b15b36',5)
        top.text(p+np.array([8,-11]),j['id'],21,'#9b4c2e',box=True)
    top.text((65,1080),'蓝 / 绿：同一围挡顶部 / 底部的俯视重合线；灰点：竖格 station；J1–J8：原生分段连接。',25)
    top.text((65,1130),'网格点、分段连接点均为模型构造与绘制采样，不是新增的源 UV 或现场测量。',25)
    top.image.save(out/'top.png')
    side=Canvas((2200,800))
    side.text((65,55),'M5 · 围挡展开侧视（s–z）',42)
    side.text((65,110),'横轴为 native 连续周长 station，纵轴单独放大展示；不是空间平面广告模板。',26)
    P=data['perimeter'];height=board[5];station=np.asarray(data['station'])
    def sz(s,z):return np.c_[100+np.asarray(s)/P*1980,590-np.asarray(z)/height*330]
    side.line(sz(station,np.zeros_like(station)),BOTTOM,4)
    side.line(sz(station,np.full_like(station,height)),TOP,4)
    for i,v in enumerate(data['verticals']):
        s=v['station'];p=sz([s,s],[0.,height]);side.line(p,GRID,2)
        if i%8==0:side.text(p[0]+[0,29],v['id'],20,GRID,'mm')
    side.text((100,216),f'顶部 z = {height:.6f} L（未重新拟合）',25,TOP)
    side.text((100,685),f's = 0 与 s = P 为同一闭合接缝；P = {P:.6f} L（未重新拟合）',25)
    side.text((100,735),'每条 V 的上下端共享 XY，只改变 z。网格是模型显示辅助，没有源像素观测权重。',24)
    side.image.save(out/'side.png')
    perspective=Canvas((2200,1450))
    perspective.text((65,55),'M5 · 同一场地与连续围挡三维线框',42)
    perspective.text((65,110),'同一 native XYZ；高度显示放大 3 倍以看清顶底与竖格，参数和模型几何未被拉伸。',26)
    perspective.text((65,156),'全部具体尺寸尚未重新拟合；几何结构图不代表相机标定或像素拟合通过。',26,'#b57625')
    projected=np.vstack([_project_display(low),_project_display(high)])
    image_xy=_mapping(projected,(90,245,2020,950))
    if floor_shapes:
        for shape in floor_shapes:
            points=np.asarray(shape['xyz']);perspective.line(image_xy(_project_display(points)),_floor_color(shape),2)
    for v in data['verticals']:
        points=np.asarray(v['xyz']);perspective.line(image_xy(_project_display(points)),GRID,2)
    perspective.line(image_xy(_project_display(low)),BOTTOM,4)
    perspective.line(image_xy(_project_display(high)),TOP,4)
    for i in range(0,len(data['verticals']),8):
        v=data['verticals'][i];p=image_xy(_project_display(np.asarray(v['xyz'])))[1]
        perspective.dot(p,TOP,4);perspective.text(p+[6,-13],v['id'],20,TOP,box=True)
    perspective.text((65,1280),'蓝：围挡顶沿    绿：冰面交界底沿    灰：同 station 的竖向网格',26)
    perspective.text((65,1335),'四条直段、四段超椭圆弯角和闭合接缝均来自现有 native BoardCourse；未生成新观测。',25)
    perspective.text((65,1390),'围挡不强制为正圆弧；高度放大只用于此图的显示变换，geometry.json 保留原 XYZ。',24)
    perspective.image.save(out/'three_d.png')
    data['scene_id']=scene_id
    (out/'geometry.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    result={'schema':'rinkspace.board.review.v1','scene_id':scene_id,'library':str(Path(library_path).resolve()),
            'top':str(out/'top.png'),'side':str(out/'side.png'),'three_d':str(out/'three_d.png'),
            'geometry':str(out/'geometry.json'),'checks':data['checks'],'dimensions_status':data['dimensions_status'],
            'display_height_exaggeration':3.,'floor_shape_count':len(floor_shapes),'fit_executed':False,'new_source_observations':0}
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return result

def render_station_continuity(board, library_path, image_path):
    """Seven trial values of ONE latent station across internal and cyclic joins.

    Native kind7 alone supplies all XYZ. This is a forward-coordinate demo,
    not measured UV, not a fitted trajectory and not an LM run.
    """
    out=Path(image_path).resolve()
    if out.exists():
        return {'path':str(out),'reused':True,'fit_executed':False}
    project=Path(__file__).resolve().parents[3]
    if out.is_relative_to(project):raise ValueError('diagnostic figure belongs in Data')
    core=NativeCore(library_path)
    breaks=core.board_course_chart(board)
    c=Canvas((2100,1420))
    c.text((65,55),'围挡取点参数 · 跨接缝连续移动',43)
    c.text((65,110),'同一 feature、同一个局部 station 变量的 7 个试验值；每次均由 native kind7 求值。',27)
    c.text((65,160),'全局弧长 s 不受直段 / 弯角段边界限制；闭合接缝处允许负值，XYZ 连续。',26,'#b57625')
    records=[]
    colors=['#27568a','#2b76a5','#278e96','#be7628','#499d78','#7464a6','#a35d7a']
    for panel,(join,title) in enumerate([(breaks[1],'直段 → 弯角接缝'),(0.,'闭合接缝 s = 0')]):
        x=65+panel*1050
        c.text((x,235),title,33)
        stations=np.asarray(join)+np.linspace(-.09,.09,7)
        dense=np.asarray(core.sample_path(board,7,[0.],[],np.linspace(join-.14,join+.14,501)))
        xyz=np.asarray(core.sample_path(board,7,[0.],[],stations))
        at_join=np.asarray(core.sample_path(board,7,[0.],[],[join]))[0]
        xy=_mapping(dense[:,:2],(x+20,330,880,640))
        c.line(xy(dense[:,:2]),'#88989f',5)
        for i,point in enumerate(xyz):
            pixel=xy(point[:2]);c.dot(pixel,colors[i],9)
            c.text(pixel+[0,-27 if i%2==0 else 30],str(i+1),27,colors[i],'mm',box=True)
        pj=xy(at_join[:2])
        c.line([pj+[-11,0],pj+[11,0]],INK,2)
        c.line([pj+[0,-11],pj+[0,11]],INK,2)
        c.text(pj+[13,-56],'几何接缝 J',23,INK,box=True)
        c.text((x,1015),'同一实体：board/continuous/bottom',24)
        c.text((x,1060),'s / L：'+', '.join(f'{v:.5f}' for v in stations[:4]),23)
        c.text((x,1102),'       '+', '.join(f'{v:.5f}' for v in stations[4:]),23)
        c.text((x,1150),'1 → 7 表示同一点参数的连续变化，不是 7 个新观测。',23,GRID)
        records.append({'join_station':float(join),'stations':stations.tolist(),'xyz':xyz.tolist(),
                        'feature_id':'board/continuous/bottom','same_local_variable':True,
                        'hard_segment_bounds':False,'source_observations':0})
    c.text((65,1248),'灰线：同一 native 围挡底沿；彩色点：同一 station 变量的前向求值；J 只是几何分段连接。',26)
    c.text((65,1302),'这是参数域与连续性的演示，不是源观测或 LM 拟合结果。图像只作等比例显示，原 XYZ 未改。',26,'#b57625')
    c.text((65,1355),'旧分段身份保留为来源别名；新拟合坐标不按旧段号限制，也不在 0 / P 处截断。',25)
    out.parent.mkdir(parents=True,exist_ok=True)
    c.image.save(out)
    receipt={'schema':'rinkspace.board.station_review.v1','path':str(out),'library':str(Path(library_path).resolve()),
             'coordinate_kind':'unwrapped_global_arc_length','panels':records,'native_kind':7,
             'new_source_observations':0,'fit_executed':False,'XYZ_modified_for_display':False}
    report=out.with_suffix('.json')
    if not report.exists():report.write_text(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return receipt
