"""Frozen-inner, view-dependent boundaries of the existing rigid skate mesh.

Visibility is geometric self-occlusion of the shared lower-skate assembly.
Source occlusion by body/other objects remains in the source contract. No
anatomical inside/outside blade identity or material endpoints are invented.
"""
import numpy as np


def camera_centres(camera,frames,rows=None):
    if hasattr(camera,'camera_centres'):return camera.camera_centres(frames,rows=rows)
    times=np.asarray(camera.p['motion_times'])[frames].copy()
    if rows is not None:times+=float(camera.p['rig'][37])*(np.clip(rows,0,1079)/1079-.5)
    out=camera.core.evaluate_motion_camera(camera.p['rig'],camera.p['controls'],float(camera.p['first_control_time']),float(camera.p['control_dt']),times)
    if out['status'].any():raise ValueError('Native camera visibility query failed')
    return out['centre']*camera.scale


def structure_geometry(binding):
    packet=binding.packet();template=packet['template'];V=np.asarray(template['vertices']);F=np.asarray(template['faces']);parts=np.asarray(template['face_parts'])
    triangles=V[F];normal=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]);normal/=np.maximum(np.linalg.norm(normal,axis=1),1e-20)[:,None]
    centres=triangles.mean(1);catalog={}
    for i,tri in enumerate(F):
        for a,b in zip(tri,np.roll(tri,-1)):catalog.setdefault(tuple(sorted((int(a),int(b)))),[]).append(i)
    segments=[];adjacent=[];kinds=[];families={};side=[]
    def add(name,items):
        ids=[]
        for a,b,ff,kind,ss in items:
            if np.linalg.norm(a-b)<1e-12:continue
            ids.append(len(segments));segments.append([a,b]);adjacent.append(ff);kinds.append(kind);side.append(ss)
        families[name]=np.asarray(ids,int)
    def paths(paths):
        return [(a,b,(), 'curve',i)for i,path in enumerate(paths)for a,b in zip(path[:-1],path[1:])]
    def creases(names):
        result=[]
        for edge,ff in catalog.items():
            if not any(parts[f]in names for f in ff):continue
            if len(ff)!=2 or abs(normal[ff[0]]@normal[ff[1]])<.99999:result.append((V[edge[0]],V[edge[1]],ff,'mesh',-1))
        return result
    lower=packet['edges'];upper=lower+[0,0,packet['parameters'].blade_web_height_m]
    add('runner_lower_edge',paths(lower));add('runner_upper_edge',paths(upper))
    add('toe_pick',creases(['toe_pick_head']))
    tail=[]
    for ss,(lo,hi)in enumerate(zip(lower,upper)):
        take=lo[:,0]<=.12*packet['parameters'].sole_length_m
        for path in (lo[take],hi[take],np.stack((lo[0],hi[0]))):tail.extend([(a,b,(),'curve',ss)for a,b in zip(path[:-1],path[1:])])
    add('rear_extension',tail)
    all_parts=np.unique(parts).tolist();add('blade_assembly_outline',creases(all_parts))
    add('boot_sole',creases([v for v in all_parts if 'outsole'in v]))
    add('heel_block',creases([v for v in all_parts if 'heel_block'in v]))
    add('front_stanchion',creases(['front_post','front_nose_post']));add('rear_stanchion',creases(['rear_post']))
    q=np.asarray(segments);return dict(points=q.reshape(-1,3),starts=np.arange(len(q))*2,ends=np.arange(len(q))*2+1,feature_segments=families,
        segments=q,adjacent_faces=adjacent,kinds=np.asarray(kinds),sides=np.asarray(side),triangles=triangles,normals=normal,face_centres=centres)


def unoccluded(camera,points,triangles):
    """Finite ray/triangle depth test; points on their own surface are visible."""
    a=triangles[:,0];e1=triangles[:,1]-a;e2=triangles[:,2]-a;out=[]
    for start in range(0,len(points),32):
        target=points[start:start+32];ray=target-camera;distance=np.linalg.norm(ray,axis=1)
        h=np.cross(ray[:,None],e2[None]);det=np.einsum('fi,nfi->nf',e1,h);valid=abs(det)>1e-13
        inv=np.divide(1.,det,out=np.zeros_like(det),where=valid);s=camera-a
        u=np.einsum('fi,nfi->nf',s,h)*inv;q=np.cross(s,e1)
        v=np.einsum('ni,fi->nf',ray,q)*inv;t=np.einsum('fi,fi->f',e2,q)[None]*inv
        # 0.5mm is only mesh visibility tolerance, never an ice/fit gate.
        hit=valid&(u>=-1e-9)&(v>=-1e-9)&(u+v<=1+1e-9)&(t>0)&(t<1-.0005/np.maximum(distance[:,None],1e-12))
        out.extend(~hit.any(1))
    return np.asarray(out,bool)


def visible_segments(geometry,camera_local,feature):
    ids=geometry['feature_segments'][feature];normal=geometry['normals'];centres=geometry['face_centres'];front=np.einsum('fi,fi->f',normal,camera_local-centres)>0
    selected=[]
    near=int(camera_local[1]>=0) # lower edges are [-width/2,+width/2]
    for k in ids:
        if geometry['kinds'][k]=='curve':
            if geometry['sides'][k]==near:selected.append(k)
            continue
        ff=geometry['adjacent_faces'][k];f=front[list(ff)]
        # The whole-outline family contains silhouettes, not arbitrary creases.
        silhouette=(len(ff)==1 and f[0])or(f.any()and not f.all())
        if silhouette or(feature!='blade_assembly_outline'and f.any()):selected.append(k)
    selected=np.asarray(selected,int)
    if not len(selected):return selected,dict(visible_segments=0,occlusion_verified=True)
    segment=geometry['segments'][selected];visible=unoccluded(camera_local,segment.mean(1),geometry['triangles'])
    return selected[visible],dict(visible_segments=int(visible.sum()),tested_segments=len(selected),near_side=near,occlusion_verified=True)


def monotone_match(distance):
    """Minimum ordered path; source clicks cannot hop back along the runner."""
    best=None
    for reverse in (False,True):
        c=distance[:,::-1]if reverse else distance;cost=c[0].copy();back=[]
        for row in c[1:]:
            arg=np.empty(len(cost),int);k=0
            for j in range(len(cost)):
                if cost[j]<cost[k]:k=j
                arg[j]=k
            back.append(arg);cost=row+cost[arg]
        k=int(np.argmin(cost));choice=[k]
        for arg in back[::-1]:k=int(arg[k]);choice.append(k)
        choice=np.asarray(choice[::-1]);choice=len(cost)-1-choice if reverse else choice
        total=float(cost.min())
        if best is None or total<best[0]:best=(total,choice,reverse)
    return best[1],best[2]
