"""Actual partial panel edges missed by legacy cropped reference paths."""
from pathlib import Path
import sys,json,hashlib,collections
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ad_entity_measure_s01 as m
r=m.r;b=r.b;OUT=r.OUT

def read_arc(pt,name,q,n):
    frame=r.VIS[pt];im,gray,mask=r.current(pt);binding=dict(source_sha256=frame['source_video_sha256'],source_pts=pt,time_base='1/12800',shot_id='s01',event_id='ankara_2026',rig_id='main',installation_id='main',pixel_domain='native_distorted',width=1920,height=1080)
    ns=np.tile(n,(len(q),1));profiles=m.read_source_material_transition_profiles(np.ascontiguousarray(im[:,:,::-1]),binding=binding,search_points_uv=q,normals=ns,search_intervals_px=np.tile([-24.,24.],(len(q),1)),transition='dark_to_light',tangent_half_width_px=1,smoothing_sigma_px=.6,engineering_support_margin_px=.25)['readings']
    runs=[];run=[]
    for query,p in zip(q,profiles):
        uv=np.asarray(p['actual_source_uv']);ok=not any(p['source_flags'].values())
        if ok:
            grid=query+np.arange(-26.,27.)[:,None]*n
            ok=not np.any(r.sample(mask,grid[:,None,:])>0)
        if ok:run.append(uv)
        elif run:
            if len(run)>=4:runs.append(np.asarray(run))
            run=[]
    if len(run)>=4:runs.append(np.asarray(run))
    return [dict(path_id=name,pixels=q.tolist(),contour_sha256=hashlib.sha256(np.ascontiguousarray(q).tobytes()).hexdigest(),proof=dict(kind='actual_source_two_material_outer_panel_interface',source_pts=pt,source_BGR_sha256=frame['source_BGR_sha256'],partial_arc_only=True,not_a_fixed_corner_reference=True)) for q in runs]

def run():
    cv2.setNumThreads(2);extra=b.read(OUT/'additional_source_atlases.json');doc=b.read(OUT/'source_identity_charts.json');extra['atlases']=[a for a in extra['atlases'] if a['id']<82]
    specs=[(82,4,368640,6,[('Ad4.top_edge/path0',np.column_stack([np.arange(355.,960.,3.),np.full(202,404.)]),[0.,-1.]),('Ad4.bottom_edge/path0',np.column_stack([np.arange(355.,900.,3.),np.full(182,535.)]),[0.,1.])]),
    (83,7,507136,76,[('Ad7.top_edge/path0',np.column_stack([np.arange(475.,1900.,3.),np.full(475,488.)]),[0.,-1.]),('Ad7.bottom_edge/path0',np.column_stack([np.arange(475.,1900.,3.),np.full(475,709.)]),[0.,1.]),('Ad7.left_vertical/path0',np.column_stack([np.full(65,442.),np.arange(500.,695.,3.)]),[-1.,0.])])]
    for ai,ad,pt,via,queries in specs:
        paths=[]
        for name,q,n in queries:paths.extend(read_arc(pt,name,q,np.array(n)))
        if not paths:raise ValueError('No actual border evidence')
        f=r.VIS[pt];a=dict(id=ai,ad=ad,image=f['image'],image_sha256=b.sha(f['image']),paths=paths,partial_source_reference=True)
        extra['atlases'].append(a);root=np.asarray(doc['records'][str(via)][str(pt)]['H'])
        doc['records'][str(ai)]={str(p):dict(H=(np.asarray(c['H'])@np.linalg.inv(root)).tolist(),atlas_id=ai,pts=int(p),method='actual_partial_material_border_via_known_same_instance',via_atlas=via,root_source_pts=pt,source_BGR_sha256=r.VIS[int(p)]['source_BGR_sha256'],native_geometry_used=False) for p,c in doc['records'][str(via)].items()}
        print(json.dumps(dict(atlas=ai,source_pts=pt,arcs=len(paths),points=sum(len(p['pixels']) for p in paths),frames=len(doc['records'][str(ai)]))),flush=True)
    b.write(OUT/'additional_source_atlases.json',extra);b.write(OUT/'source_identity_charts.json',doc)
if __name__=='__main__':run()
