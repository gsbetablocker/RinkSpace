"""Qualified actual C interfaces using existing native paint-pair ABI.

Each source profile has its own scalar half-width. Inverse rays only initialize
that nuisance and a finite side branch; they establish no world measurement.
"""
from pathlib import Path
import copy,json,hashlib
import numpy as np
from .io_utils import read_json,read_jsonl,write_json,file_sha256
from . import tail_curve_source_append as source


def _inverse_source_plane(camera,uv,time,initial_xy):
    xy=np.asarray(initial_xy,float).copy();target=np.asarray(uv,float);times=np.asarray(time,float)
    def project(q):return camera.at(np.c_[q,np.zeros(len(q))],times)['uv']
    for iteration in range(30):
        value=project(xy);residual=value-target
        if np.max(abs(residual))<1e-8:break
        eps=1e-6;delta=[]
        for k in range(2):
            d=np.zeros_like(xy);d[:,k]=eps;delta.append((project(xy+d)-project(xy-d))/(2*eps))
        J=np.stack(delta,axis=2);step=np.linalg.solve(J,residual[...,None])[...,0]
        norms=np.linalg.norm(step,axis=1);step*=np.minimum(1,.2/np.maximum(norms,1e-100))[:,None]
        old=np.linalg.norm(residual,axis=1);accepted=np.zeros(len(xy),bool)
        for factor in 2.**-np.arange(18):
            trial=xy-factor*step;err=np.linalg.norm(project(trial)-target,axis=1);take=(~accepted)&(err<old)
            xy[take]=trial[take];accepted|=take
        if not accepted.all()and np.max(old[~accepted])>1e-7:raise ValueError('C plane inverse has no improving actual-source step')
    replay=camera.at(np.c_[xy,np.zeros(len(xy))],times)
    error=np.linalg.norm(replay['uv']-target,axis=1)
    if not np.isfinite(xy).all()or not replay['valid'].all()or error.max()>1e-6:raise ValueError('C inverse numerical initializer is outside actual camera domain '+str(dict(finite=bool(np.isfinite(xy).all()),valid=int(replay['valid'].sum()),total=len(xy),maxerror=float(error.max()),maxradius=float(replay['ray_radius'].max()))))
    return xy,error


def build_local_width_model(parent_model_path,paired_bundle_path):
    from .joint_model import as_event_view,graph_from_joint
    from .camera_graph import CameraGraph
    from rinkfit_start.native_scene import NativeScene,camera_from_event
    bp=Path(paired_bundle_path).resolve();b=read_json(bp);parent=read_json(parent_model_path);rows=list(read_jsonl(b['source_rows_ref']['file']))
    if not b['strict_paired']or len(rows)!=6328:raise ValueError('Local widths require both actual C flanks')
    view=as_event_view(parent,'s09');camera=CameraGraph(graph_from_joint(parent)).native_camera('s09');scene=NativeScene(view)
    uv=np.asarray([r['current_uv']for r in rows]);pts=np.asarray([r['native_pts']for r in rows]);times=(pts-parent['shots']['s09']['time_origin_pts'])/12800+camera.tau*(uv[:,1]/1079-.5)
    stations=np.linspace(0,1,1001);curve=scene.world(145,0,stations);initial=np.empty((len(rows),2))
    # A coarse physical forward chooses only a numerical inverse seed. Each
    # source exposure keeps its own actual UV and own sensor row-time.
    for pt in np.unique(pts):
        inds=np.flatnonzero(pts==pt);t=float((pt-parent['shots']['s09']['time_origin_pts'])/12800)
        pred=camera.at(curve,t)['uv'];dist=((uv[inds,None,:]-pred[None,:,:])**2).sum(2);initial[inds]=curve[np.argmin(dist,axis=1),:2]
    inverse,error=_inverse_source_plane(camera,uv,times,initial)
    path=parent['materials'][145]['paths'][0];q=np.asarray(parent['blocks'][path['geometry_block']]['values'],float)
    if path.get('geometry_indices'):q=q[path['geometry_indices']]
    else:q=q[path.get('geometry_offset',0):]
    tangent=q[2:4]-q[:2];normal=np.array([-tangent[1],tangent[0]])/np.linalg.norm(tangent)
    model=copy.deepcopy(parent);widths=model.setdefault('measurement_widths',[]);bindings=model.setdefault('stroke_pair_model_bindings',[]);records=[]
    for j in range(3164):
        r=rows[2*j];rid=r['derived_midpoint_source_row_id'];delta=float(np.dot(inverse[2*j+1]-inverse[2*j],normal));width=abs(delta)/2
        if not 1e-12<width<1:raise ValueError('C numerical inverse produced illegal local width')
        signs=[-1,1]if delta>0 else[1,-1];name='measurement/paint_half_width/'+hashlib.sha256(rid.encode()).hexdigest()[:24];bi=len(model['blocks'])
        model['blocks'].append(dict(name=name,values=[width],lower=[1e-12],upper=[1.],scales=[max(width,1e-5)],constant=False,fixed_indices=[]))
        widths.append(dict(width_scope_id=rid,half_width_block=bi,scope_is_source_confirmed_section_not_uniform_whole_stroke=True,positive_scalar_measurement_nuisance=True))
        bindings.append(dict(pair_id=rid+'/strict_pair',width_scope_id=rid,half_width_block=bi,material_index=145,initial_path_index=0,
            source_world_signs=signs,allowed_world_side_assignments=[[-1,1],[1,-1]],source_refs=[b['source_rows_ref']],
            finite_association_is_numerical_seed_not_material_identity=True))
        records.append(dict(source_midpoint_id=rid,native_pts=r['native_pts'],half_width_block=bi,block_name=name,actual_flank_uv=r['source_flanks_uv'],
            source_world_signs=signs,numerical_inverse_ice_xy=inverse[2*j:2*j+2].tolist(),native_forward_inverse_error_px=error[2*j:2*j+2].tolist(),
            source_UV_modified=False,world_width_is_measured_truth=False,world_signs_are_material_truth=False,
            width_shared_across_exposures=False,station_shared_between_flanks=False,source_family_id=r['source_family_id']))
    mp=bp.parent/'model.json';write_json(mp,model)
    wp=bp.parent/'local_width_qualification.json';write_json(wp,dict(schema='rinkfit.source_local_paint_half_width_allocation.v1',
        source_bundle_operator=source.PAIRED,parent_model=source.ref(parent_model_path),model=source.ref(mp),source_rows_ref=b['source_rows_ref'],
        reader_module_ref=source.ref(__file__),old_physical_blocks=len(parent['blocks']),new_scalar_halfwidth_blocks=3164,pairs=records,
        inverse_plane_initializer_is_not_source_truth=True,source_side_order_unchanged=True,source_world_side_branch_is_numerical_hypothesis=True,
        alternate_finite_world_side_branch_retained=True,maximum_native_inverse_replay_error_px=float(error.max()),
        no_world_width_calibration_claimed=True))
    b['model']=source.ref(mp);b['local_width_qualification']=source.ref(wp);b['local_width_reader_module_ref']=source.ref(__file__);write_json(bp,b)
    chunk=read_json(bp.parent/'chunk.json');chunk['source_gate_manifest']=source.ref(bp);write_json(bp.parent/'chunk.json',chunk)
    return source.ref(mp),source.ref(wp)


def install_actual_pairs(packet,source_rows,out,parent):
    packet=source.install_actual_queries(packet,source_rows,out,parent);rows=list(read_jsonl(source_rows));N=parent['row_count'];oldpairs=parent.get('stroke_pairs',[]);pairs=list(oldpairs)
    for key in ('stroke_pair_index','stroke_source_side'):
        d=packet['arrays'][key];a=np.memmap(d['file'],dtype=d['dtype'],shape=tuple(d['shape']),mode='r+')
        a[N:]=np.arange(len(rows))//2+len(oldpairs)if key=='stroke_pair_index'else np.arange(len(rows))%2
        a.flush();del a;packet['arrays'][key]={**d,'sha256':file_sha256(d['file'])}
    for j in range(len(rows)//2):
        r=rows[2*j];rid=r['derived_midpoint_source_row_id'];pairs.append(dict(id=rid+'/strict_pair',source_row_id=rid,rows=[N+2*j,N+2*j+1],
            material_index=145,target_id='ice.C:segment0',shot_id='s09',native_pts=r['native_pts'],width_scope_id=rid,
            allowed_world_side_assignments=[[-1,1],[1,-1]],actual_flank_uv=r['source_flanks_uv'],source_normal=r['normal'],
            source_family_id=r['source_family_id'],source_refs=[source.ref(source_rows)]))
    packet['stroke_pairs']=pairs
    packet['C_paired_operator_contract']=dict(actual_source_sections=len(rows)//2,actual_flank_rows=len(rows),
        independent_width_nuisance_per_section=True,each_flank_own_RS_and_station=True,image_midpoints_newly_active=False,
        source_A_B_world_sign_is_numerical_branch=True,physical_accuracy_claimed=False,
        inherited_13_midpoints_preserved_pending_separate_explicit_operator_supersession=True)
    return packet
