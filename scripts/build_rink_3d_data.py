"""Assemble the rink's 3D data only: no viewer or external application.

The user-PPT baseline supplies schematic feature identities and XY coordinates.
The triangulated ice and closed boards share exactly one perimeter. Regular
grid cells and board-local surface coordinates make later localization usable.
No video fitting or physical camera XYZ is introduced by this conversion.
"""
from __future__ import annotations
from rinkspace.storage import resolve_asset as _resolve_asset
import argparse, copy, hashlib, json, math, subprocess, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=_resolve_asset(ROOT, ROOT/'outputs/rink_model/ankara_2026_shared/v0.14-scene3d')
BASE=_resolve_asset(ROOT, ROOT/'outputs/rink_model/ankara_2026_shared/v0.13-women-sp/baseline/user_ppt_layout.json')

def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,data):p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def xyz(points):return [[float(x),float(y),0.] for x,y in points]
def cross(a,b):return a[0]*b[1]-a[1]*b[0]
def area(p):
    a=np.asarray(p,float)
    return abs(float(np.sum(a[:,0]*np.roll(a[:,1],-1)-a[:,1]*np.roll(a[:,0],-1))/2)) if len(a)>=3 else 0.

def clip_polygon(points,boundary):
    p=[np.array(x,float) for x in points]
    if len(p)>1 and np.linalg.norm(p[0]-p[-1])<1e-9:p.pop()
    for a,b in zip(boundary,np.roll(boundary,-1,axis=0)):
        if not p:break
        result=[];edge=b-a;prev=p[-1];dp=cross(edge,prev-a)
        for q in p:
            dq=cross(edge,q-a)
            if (dp>=-1e-9)!=(dq>=-1e-9):
                result.append(prev+(q-prev)*(dp/(dp-dq)))
            if dq>=-1e-9:result.append(q)
            prev,dp=q,dq
        p=result
    clean=[]
    for q in p:
        if not clean or np.linalg.norm(q-clean[-1])>1e-8:clean.append(q)
    if len(clean)>1 and np.linalg.norm(clean[0]-clean[-1])<1e-8:clean.pop()
    return np.array(clean).reshape(-1,2)

def clip_segment(p,q,boundary):
    p=np.asarray(p,float);q=np.asarray(q,float);v=q-p;t0,t1=0.,1.
    for a,b in zip(boundary,np.roll(boundary,-1,axis=0)):
        edge=b-a;c=cross(edge,p-a);dv=cross(edge,v)
        if abs(dv)<1e-12:
            if c < -1e-8:return None
        elif dv>0:t0=max(t0,-c/dv)
        else:t1=min(t1,-c/dv)
        if t0>t1+1e-10:return None
    return np.array([p+t0*v,p+t1*v])

def grid_data(boundary,params,ice):
    x0,y0,x1,y1=params['bounds'];step=params['grid_spacing']
    cells=[];nx=math.ceil((x1-x0)/step);ny=math.ceil((y1-y0)/step)
    for j in range(ny):
        for i in range(nx):
            xa,ya=x0+i*step,y0+j*step;xb,yb=min(xa+step,x1),min(ya+step,y1)
            p=clip_polygon([[xa,ya],[xb,ya],[xb,yb],[xa,yb]],boundary)
            ar=area(p)
            if ar<1e-9:continue
            cells.append({'id':f'grid_{i:03}_{j:03}','column':i,'row':j,'polygon_xyz':xyz(p),
                          'area':ar,'clipped_at_rink_boundary':abs(ar-(xb-xa)*(yb-ya))>1e-7})
    lookup=[]
    for k,p in enumerate(ice['positions']):
        u=(p[0]-x0)/step;v=(p[1]-y0)/step
        if abs(u-round(u))<1e-8 and abs(v-round(v))<1e-8:
            lookup.append({'column':round(u),'row':round(v),'ice_mesh_vertex_index':k})
    return {'type':'rectangular_cells_clipped_to_shared_boundary','origin_xyz':[x0,y0,0.],
      'column_axis_xyz':[1.,0.,0.],'row_axis_xyz':[0.,1.,0.],'spacing_xy':[step,step],
      'columns':nx,'rows':ny,'cells':cells,'vertex_lookup':lookup,
      'edge_cell_note':'Boundary cells follow the same polygon as ice and boards; out-of-rink cells omitted.'}

def features_data(baseline,boundary):
    features=[]
    for f in baseline['ice_features']:
        if f['id']=='rink_boundary':continue
        item={'id':f['id'],'label':f['label'],'surface_id':'ice_surface','kind':f['kind'],
              'geometry_status':'user_PPT_schematic_not_image_calibrated','source':f['ppt_evidence'],
              'style':f.get('style',{}),'plane_normal_xyz':[0.,0.,1.]}
        if f['kind']=='circle':
            item.update(center_xyz=[*f['centre'],0.],radius=f['radius'])
            if f.get('rings'):
                item['rings']=[{'role':r['role'],'center_xyz':[*r['centre'],0.],'radius':r['radius'],'style':r['style']} for r in f['rings']]
        elif f['kind']=='polygon':
            item['polygon_xyz']=xyz(clip_polygon(f['points_model'],boundary));item['closed']=True
        else:
            segments=[]
            for a,b in zip(f['points_model'][:-1],f['points_model'][1:]):
                p=clip_segment(a,b,boundary)
                if p is not None:segments.append(xyz(p))
            item['segments_xyz']=segments
        if f['kind']!='circle':
            item['source_ppt_points_xy']=f['points_model']
            item['boundary_rule']='Effective paint geometry clipped to ice; original schematic points preserved separately.'
        features.append(item)
    return features

def board_and_ad_data(boundary,baseline,height):
    edges=np.roll(boundary,-1,axis=0)-boundary;lengths=np.linalg.norm(edges,axis=1)
    cum=np.r_[0.,np.cumsum(lengths)];panels=[];ads=[]
    for i,(p,e,l) in enumerate(zip(boundary,edges,lengths)):
        t=e/l;n=[-t[1],t[0],0.]
        panels.append({'id':f'board_panel_{i:03}','mesh_id':'rink_boards',
          'bottom_vertex_indices':[i,(i+1)%len(boundary)],'top_vertex_indices':[i+len(boundary),(i+1)%len(boundary)+len(boundary)],
          's_range':[float(cum[i]),float(cum[i+1])],'tangent_xyz':[*t,0.],'normal_into_rink_xyz':n,
          'z_range':[0.,height]})
    logo_names={'image1.png':'红色人物与星圈圆徽','image2.png':'土耳其滑冰联合会圆徽','image3.png':'ANKARA盾徽','image4.png':'can红色徽标'}
    for a in baseline['scene_semantics']:
        if a['kind']!='board_advertisement':continue
        anchor=np.array(a['schematic_anchor_xy']);tt=np.clip(np.sum((anchor-boundary)*edges,axis=1)/(lengths**2),0,1)
        feet=boundary+tt[:,None]*edges;i=int(np.argmin(np.linalg.norm(feet-anchor,axis=1)))
        media=a.get('embedded_image');name=logo_names.get(Path(media['package_part']).name,a['label']) if media else a['label']
        ads.append({'id':a['id'],'label':name,'surface_id':'boards_inner_surface','board_panel_id':panels[i]['id'],
          'qualitative_side':a['qualitative_side'],'attachment_s':float(cum[i]+tt[i]*lengths[i]),
          'anchor_xyz':[float(feet[i,0]),float(feet[i,1]),None],
          'placement_status':'schematic_XY_projected_to_boundary; height_and_size_unknown',
          'height_above_ice':None,'width_along_board':None,'height':None,
          'reference_schematic_xy':a['schematic_anchor_xy'],'source':a['ppt_evidence'],
          'embedded_image_source':media,'use_as_ice_point':False})
    return {'surface_id':'boards_inner_surface','mesh_id':'rink_boards','panels':panels,
      'local_coordinates':{'s':'CCW cumulative arc length from boundary[0]','z':'height above ice; same units as scene XYZ'},
      'perimeter_length':float(cum[-1]),'closed':True},ads

def check_scene(scene,geo):
    boundary=np.array(scene['ice_boundary']['points_xyz'])[:,:2];features=scene['ice_features']
    mesh_ids={m['name'] for m in scene['meshes']};ids=[x['id'] for x in features]+[a['id'] for a in scene['advertisements']]
    assert len(ids)==len(set(ids))
    assert scene['cameras']==[] and scene['coordinate_system']['metres_per_unit'] is None
    assert 'A3' not in ids and 'A4' not in ids
    grid_area=sum(c['area'] for c in scene['ice_grid']['cells']);boundary_area=area(boundary)
    assert abs(grid_area-boundary_area)<1e-6,(grid_area,boundary_area)
    all_points=[]
    for f in features:
        assert f['surface_id'] in mesh_ids
        if 'center_xyz' in f:
            th=np.linspace(0,2*np.pi,100);c=np.array(f['center_xyz'][:2]);r=f['radius']
            all_points.extend(np.c_[c[0]+r*np.cos(th),c[1]+r*np.sin(th),np.zeros(len(th))])
        all_points.extend(f.get('polygon_xyz',[]))
        for seg in f.get('segments_xyz',[]):all_points.extend(seg)
    points=np.array(all_points);assert np.isfinite(points).all() and np.all(points[:,2]==0)
    for a,b in zip(boundary,np.roll(boundary,-1,axis=0)):
        edge=b-a;delta=points[:,:2]-a
        assert np.all(edge[0]*delta[:,1]-edge[1]*delta[:,0]>=-1e-7)
    assert all(a['anchor_xyz'][2] is None and not a['use_as_ice_point'] for a in scene['advertisements'])
    return {'all_checks_passed':True,'ice_and_boards_mesh_checks':geo['checks'],
       'grid_cell_count':len(scene['ice_grid']['cells']),'grid_cell_area_equals_shared_boundary':True,
       'ice_feature_count':len(features),'all_ice_features_at_z0_inside_rink':True,
       'board_panel_count':len(scene['board_surface']['panels']),'advertisement_instance_count':len(scene['advertisements']),
       'advertisement_heights_not_invented':True,'camera_positions_added':0,
       'note':'Data integrity and topology checks; no claim of measured rink geometry or video localization accuracy.'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--grid-spacing',type=float,default=5.);ap.add_argument('--board-height',type=float,default=4.);ap.add_argument('--board-thickness',type=float,default=.6)
    args=ap.parse_args()
    subprocess.run([sys.executable,str(ROOT/'scripts/build_rink_3d_geometry.py'),'--grid-spacing',str(args.grid_spacing),'--board-height',str(args.board_height),'--board-thickness',str(args.board_thickness)],check=True)
    geo=load(OUT/'scene_geometry.json');base=load(BASE);boundary=np.array(geo['boundary_xy'])
    meshes=copy.deepcopy(geo['meshes'])
    for m in meshes:m['group']='ice' if m['name']=='ice_surface' else 'boards'
    boards,ads=board_and_ad_data(boundary,base,geo['parameters']['board_height'])
    labels={'camera_low_end':'低位端部','camera_main_long_side':'长边主机位','camera_secondary_long_side':'长边副机位','camera_high_oblique':'高位斜俯视'}
    scene={'schema_version':'rinkspace.scene3d.v1','model_id':geo['model_id'],'revision':geo['revision'],
      'status':'complete_ice_and_board_geometry_data; schematic_dimensions_not_calibrated',
      'coordinate_system':{'id':'ankara_user_ppt_z_up','handedness':'right','origin':'user_PPT_ice_outline_center',
        'x_axis':'right in user plan','y_axis':'up in user plan','z_axis':'above ice','ice_z':0.,
        'unit':'normalized_rink_width_100','metres_per_unit':None},
      'parameters':geo['parameters'],'meshes':meshes,
      'ice_boundary':{'id':'ice_board_contact','points_xyz':xyz(boundary),'closed':True,'closure':'last connects to first',
        'ice_mesh_vertex_indices':meshes[0]['boundary_vertex_indices'],'boards_inner_bottom_vertex_indices':meshes[1]['inner_bottom_vertex_indices'],
        'identity':'one shared boundary; not independent separately drawn curves'},
      'ice_grid':grid_data(boundary,geo['parameters'],meshes[0]),
      'ice_features':features_data(base,boundary),'board_surface':boards,'advertisements':ads,
      'context':[{'id':'judges_area','side':'lower_long_side','position_xyz':None,'extent':None,'source':'user_statement'}],
      'cameras':[],'camera_identity_reference':[{'id':k,'label':v,'position_xyz':None,'position_status':'not_added_yet'} for k,v in labels.items()],
      'camera_entry_contract':{'coordinate_frame':'ankara_user_ppt_z_up','required_when_added':['id','position_xyz','position_status','evidence_refs'],
        'per_shot_properties':'view orientation, focal length and distortion are separate from physical installation XYZ'},
      'source':{**geo['source'],'user_PPT_source':base['source'],'live_ppt_read_status':'file locked during this turn; previously extracted user version used',
        'geometry_json_sha256':sha(OUT/'scene_geometry.json'),'assembly_script_sha256':sha(Path(__file__))},
      'uncertainty':{'overall_dimensions':'PPT schematic coordinates, not measured metres',
        'board_height':'temporary numeric parameter, no measured height evidence','board_thickness':'temporary numeric parameter, no measured thickness evidence',
        'ice_feature_geometry':'inherits user-edited PPT, not replaced by single-frame fit','advertisements':'identity and order inherited; approximate attachment, vertical placement and width unknown',
        'physical_accuracy_verified':False},'removed_features':['A3','A4'],
      'build_scope':{'visualization_created':False,'requires_blender':False,'video_fitting_run':False}}
    checks=check_scene(scene,geo);save(OUT/'scene.json',scene)
    save(OUT/'scene_checks.json',{'schema_version':'rinkspace.scene3d-checks.v1','scene_path':'scene.json','scene_sha256':sha(OUT/'scene.json'),'checks':checks})
    print(json.dumps({k:v for k,v in checks.items() if k!='ice_and_boards_mesh_checks'},ensure_ascii=True))

if __name__=='__main__':main()
