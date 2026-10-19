"""Two actual paint interfaces around an unchanged physical stroke centreline.
Widths belong only to the source-confirmed transverse section or visible piece.
No whole-circle constant-width assumption, new source UV, or tolerance is added.
"""
from __future__ import annotations
import copy,hashlib
import numpy as np
from .joint_model import graph_from_joint
from .camera_graph import CameraGraph
from rinkfit_start.native_camera import rotation

def _ice_ray(camera,local,uv):
    uv=np.asarray(uv,float)
    if uv.shape!=(2,) or not np.isfinite(uv).all() or not 0<=uv[1]<=1079:raise ValueError('actual flank outside native sensor row')
    tm=local+camera.tau*(uv[1]/1079-.5);a=camera.spline(np.array([tm]))[0]
    seed_certificate=copy.copy(camera);seed_certificate.controls=camera.controls.copy();seed_certificate.controls[:,2]=a[2]
    if not seed_certificate.optical_domain_certificate()['passed']:raise ValueError('actual source-row-time Brown disk has no inverse seed certificate')
    z=a[2]-np.log(float(camera.optics['f_ref']));R=rotation(camera.b,a[1])@camera.R0@rotation(camera.n,a[0]);h=np.cross(camera.n,R.T@camera.b);h/=np.linalg.norm(h);d=camera.d.copy();d[2]+=camera.head.get('pupil_linear',0)*z+camera.head.get('pupil_quadratic',0)*z*z;C=camera.P+R.T@d+camera.a*h
    o=camera.optics;f=np.exp(a[2]);eta=float(o.get('eta',0));pp=np.asarray(o['principal_point'])+z*np.asarray(o.get('principal_linear',[0,0]));target=(uv-pp)/[f*np.exp(eta/2),f*np.exp(-eta/2)];k1=np.polynomial.polynomial.polyval(z,o['k1']);k2=np.polynomial.polynomial.polyval(z,o['k2']);p1=float(o['p1']);p2=float(o['p2']);radius=float(o['radius_max']);xy=np.asarray(target,float)
    if np.linalg.norm(xy)>radius:xy*=radius/(2*np.linalg.norm(xy))
    def brown(q):
        x,y=q;r2=x*x+y*y;radial=1+k1*r2+k2*r2*r2
        return np.array([x*radial+2*p1*x*y+p2*(r2+2*x*x),y*radial+p1*(r2+2*y*y)+2*p2*x*y])
    for _ in range(60):
        x,y=xy;r2=x*x+y*y;L=1+k1*r2+k2*r2*r2;A=k1+2*k2*r2;error=brown(xy)-target
        if np.linalg.norm(error,np.inf)<1e-13:break
        J=np.array([[L+2*x*x*A+2*p1*y+6*p2*x,2*x*y*A+2*p1*x+2*p2*y],[2*x*y*A+2*p1*x+2*p2*y,L+2*y*y*A+6*p1*y+2*p2*x]])
        if np.linalg.det(J)<=1e-12:raise ValueError('uncertified inverse Brown derivative')
        step=np.linalg.solve(J,error);accepted=False
        for fraction in 2.**-np.arange(28):
            q=xy-fraction*step
            if np.linalg.norm(q)<=radius and np.linalg.norm(brown(q)-target)<np.linalg.norm(error):xy=q;accepted=True;break
        if not accepted:raise ValueError('inverse Brown did not find an in-disk step')
    if np.linalg.norm(brown(xy)-target)*f>1e-6:raise ValueError('inverse Brown residual exceeds numerical accuracy')
    direction=R.T@np.r_[xy,1.]
    if direction[2]>=-1e-12:raise ValueError('actual flank ray does not hit forward ice')
    X=C-direction*C[2]/direction[2];X[2]=0
    replay=camera.at(X,tm)
    if not bool(replay['valid']) or np.max(np.abs(replay['uv']-uv))>1e-6:raise ValueError('inverse ray forward-check failed')
    return X

def append_paint_measurement_widths(model,source_pairs):
    """Append width nuisances and two finite world-side assignments to this state.

    The inverse-ray values select numerical seeds; they are not extra evidence.
    Each pair keeps its source 0/1 order and its complete covariance column order.
    """
    result=copy.deepcopy(model);cg=CameraGraph(graph_from_joint(model));labels={m['label']:i for i,m in enumerate(model['materials'])};scopes={d['width_scope_id']:d for d in result.get('measurement_widths',[])};bindings={d['pair_id']:d for d in result.get('stroke_pair_model_bindings',[])};reports=[];seed_values={};proposals=[]
    for pair in source_pairs:
        pid=pair['id'];scope=pair['width_scope_id'];mi=int(pair['material_index']) if 'material_index' in pair else labels[pair['target_id']];paths=result['materials'][mi]['paths'];kindset={p['kind'] for p in paths}
        if not kindset<={0,1,4,5}:raise ValueError('paint width may only offset a confirmed finite ice stroke')
        label=result['materials'][mi]['label']
        if 1 in kindset and label.startswith(('ice.F','ice.R')) and len(label)>5 and label[5].isdigit():raise ValueError('F/R target-circle entity interfaces are not width-centreline paint measurements')
        allowed=pair.get('allowed_world_side_assignments',[[-1,1],[1,-1]])
        if sorted(map(tuple,allowed))!=[(-1,1),(1,-1)]:raise ValueError('a source 0/1 flank is not an assumed world inner/outer side')
        reason=None;inverse=[]
        try:
            local=float(cg.local_seconds(pair['shot_id'],np.array([pair['native_pts']],np.int64))[0]);camera=cg.native_camera(pair['shot_id']);inverse=[_ice_ray(camera,local,uv) for uv in pair['actual_flank_uv']];mid=(inverse[0]+inverse[1])/2;best=None
            for pi,path in enumerate(paths):
                q=np.asarray(result['blocks'][path['geometry_block']]['values'],float);inds=path.get('geometry_indices');q=q[inds] if inds else q[path.get('geometry_offset',0):]
                if path['kind'] in (1,4):
                    radial=np.array([np.linalg.norm(X[:2]-q[:2]) for X in inverse]);delta=float(radial[1]-radial[0]);distance=abs(np.linalg.norm(mid[:2]-q[:2])-q[path.get('radius_index',2) if path['kind']==1 else 2])
                else:
                    ends=q[:4].reshape(2,2) if path['kind']==0 else np.array([q[:2]+q[2]*np.array([np.cos(q[3]),np.sin(q[3])]),q[:2]-q[2]*np.array([np.cos(q[3]),np.sin(q[3])])]);d=ends[1]-ends[0];normal=np.array([-d[1],d[0]])/np.linalg.norm(d);delta=float(np.dot(inverse[1][:2]-inverse[0][:2],normal));distance=abs(np.dot(mid[:2]-ends[0],normal))
                candidate=(distance,pi,abs(delta)/2,[-1,1] if delta>=0 else [1,-1])
                if best is None or candidate[0]<best[0]:best=candidate
            width=float(best[2]);signs=best[3];path_index=int(best[1])
            if not 0<width<1:raise ValueError('inverse-ray halfwidth is not a finite positive numerical seed')
        except (ValueError,KeyError,FloatingPointError,np.linalg.LinAlgError) as e:
            reason=str(e);width=1e-4;signs=list(allowed[0]);path_index=0
        seed_values.setdefault(scope,[]).append(width);proposals.append((pair,mi,path_index,signs));reports.append(dict(pair_id=pid,width_scope_id=scope,material_index=mi,seed_halfwidth_L1=width,source_sides_to_world_signs=signs,source_side_order_and_UV_unchanged=True,actual_ray_initialization_passed=reason is None,numerical_seed_fallback_reason=reason,inverse_ice_points_L1=[X.tolist() for X in inverse],seed_is_not_a_new_measurement=True,inverse_seed_certificate_scope='whole_Brown_disk_at_actual_source_row_time; complete_trajectory_certificate_remains_required_for_final_fit'))
    for scope,values in seed_values.items():
        if scope in scopes:continue
        value=float(np.median(values));bi=len(result['blocks']);name='measurement/paint_half_width/'+hashlib.sha256(scope.encode()).hexdigest()[:24];result['blocks'].append(dict(name=name,values=[value],lower=[1e-12],upper=[1.],scales=[max(value,1e-5)],fixed_indices=[],constant=False));scopes[scope]=dict(width_scope_id=scope,half_width_block=bi,scope_is_source_confirmed_section_not_uniform_whole_stroke=True,positive_scalar_measurement_nuisance=True)
    for pair,mi,pi,signs in proposals:
        pid=pair['id'];old=bindings.get(pid)
        record=dict(pair_id=pid,width_scope_id=pair['width_scope_id'],half_width_block=scopes[pair['width_scope_id']]['half_width_block'],material_index=mi,initial_path_index=pi,source_world_signs=signs,allowed_world_side_assignments=pair.get('allowed_world_side_assignments',[[-1,1],[1,-1]]),source_refs=pair.get('source_refs',[dict(source_pair_id=pid,source_mean_sha256=pair.get('source_mean_sha256'),source_family_id=pair.get('source_family_id'))]),finite_association_is_numerical_seed_not_material_identity=True)
        if old is not None:
            if old['width_scope_id']!=record['width_scope_id'] or old['material_index']!=mi:raise ValueError('paired source identity changed')
        else:bindings[pid]=record
    result['measurement_widths']=list(scopes.values());result['stroke_pair_model_bindings']=list(bindings.values());report=dict(schema='rinkfit.paint_measurement_width_append.v1',pairs=reports,physical_centreline_blocks_and_materials_unchanged=result['blocks'][:len(model['blocks'])]==model['blocks'] and result['materials']==model['materials'],zero_width_midline_rendering_unchanged=True,new_source_UV_or_tolerance_added=False,whole_stroke_uniform_width_assumed=False,all_geometry_relation_migrations_disabled=not result.get('physical_equalities'))
    return result,report


def interface_world(model,material_index,path_index,station,half_width,world_side):
    """Independent NumPy world evaluator for actual flank metrics, not rendering.

    The ordinary NativeScene centreline and its zero-width display stay unchanged.
    """
    from .joint_model import as_event_view
    from rinkfit_start.native_scene import NativeScene
    if world_side not in (-1,1) or not np.isfinite(half_width) or half_width<=0:raise ValueError('positive local width and explicit world side required')
    mi=int(material_index);pi=int(path_index);pa=model['materials'][mi]['paths'][pi];kind=int(pa['kind']);source=np.asarray(model['blocks'][pa['geometry_block']]['values'],float);indices=pa.get('geometry_indices');q=source[indices] if indices else source[pa.get('geometry_offset',0):];s=np.atleast_1d(np.asarray(station,float));base=NativeScene(as_event_view(model,'s01')).world(mi,pi,s);shift=float(world_side)*float(half_width)
    if kind in (1,4):
        radius=q[pa.get('radius_index',2) if kind==1 else 2]
        if radius<=half_width:raise ValueError('paint halfwidth must leave a positive inner physical radius')
        base[:,:2]+=shift*(base[:,:2]-q[:2])/radius
    elif kind==0:
        d=q[2:4]-q[:2];normal=np.array([-d[1],d[0]])/np.linalg.norm(d);base[:,:2]+=shift*normal
    elif kind==5:base[:,:2]+=shift*np.array([np.sin(q[3]),-np.cos(q[3])])
    else:raise ValueError('only finite ice stroke primitives have paint interface offsets')
    return base


def apply_source_paint_interfaces(model,packet,world,active,material,path,station,*,pair_index=None,source_side=None):
    """Apply the actual source-side width binding before independent projection.

    Mesh paths remain their unchanged centreline. Only the two admitted material
    interfaces of each frozen source pair receive their recorded world offsets.
    """
    world=np.asarray(world);active=np.asarray(active,bool);material=np.asarray(material);path=np.asarray(path);station=np.asarray(station)
    n=len(active)
    if world.shape!=(n,3) or world.dtype.kind!='f' or world.dtype.itemsize<8 or any(v.shape!=(n,)for v in (material,path,station)):
        raise ValueError('Incomplete source paint-interface world arrays')
    widths={}
    for scope in model.get('measurement_widths',[]):
        key=scope['width_scope_id']
        if key in widths or not scope.get('scope_is_source_confirmed_section_not_uniform_whole_stroke',False):
            raise ValueError('Ambiguous source-confirmed paint width scope')
        widths[key]=int(scope['half_width_block'])
    bindings={}
    for binding in model.get('stroke_pair_model_bindings',[]):
        key=binding['pair_id']
        if key in bindings:raise ValueError('Repeated paint pair binding')
        bindings[key]=binding
    seen=set();changed=[]
    for pi,pair in enumerate(packet.get('stroke_pairs',[])):
        rows=np.asarray(pair['rows'],int)
        if rows.shape!=(2,) or rows[0]==rows[1] or np.any(rows<0) or np.any(rows>=n) or any(int(i)in seen for i in rows):
            raise ValueError('Invalid actual source paint pair rows')
        seen.update(map(int,rows))
        if pair_index is not None and not np.all(np.asarray(pair_index)[rows]==pi):raise ValueError('Source paint pair index changed')
        if source_side is not None and not np.array_equal(np.asarray(source_side)[rows],[0,1]):raise ValueError('Source paint side order changed')
        count=int(active[rows].sum())
        if count==0:continue
        if count!=2:raise ValueError('A single material flank cannot replace a stroke centreline')
        binding=bindings.get(pair['id'])
        if binding is None or binding['width_scope_id']!=pair['width_scope_id']:raise ValueError('Missing actual paint width binding')
        bi=int(binding['half_width_block']);signs=list(binding['source_world_signs'])
        if widths.get(pair['width_scope_id'])!=bi or signs not in pair['allowed_world_side_assignments'] or len(signs)!=2 or set(signs)!={-1,1}:
            raise ValueError('Source paint width or world-side identity changed')
        if not 0<=bi<len(model['blocks']):raise ValueError('Paint width block is outside the model')
        block=model['blocks'][bi]
        if len(block['values'])!=1 or not block['name'].startswith('measurement/paint_half_width/'):
            raise ValueError('Invalid scalar source width parameter')
        mi=int(pair['material_index'])
        if not 0<=mi<len(model['materials']) or mi!=int(binding['material_index']) or not np.all(material[rows]==mi):raise ValueError('Source paint material identity changed')
        for side,index in enumerate(rows):
            world[index]=interface_world(model,mi,int(path[index]),station[index],float(block['values'][0]),int(signs[side]))[0]
            changed.append(int(index))
    if pair_index is not None and any(int(i)not in seen for i in np.flatnonzero(np.asarray(pair_index)>=0)):
        raise ValueError('Unbound source paint pair index')
    return changed
