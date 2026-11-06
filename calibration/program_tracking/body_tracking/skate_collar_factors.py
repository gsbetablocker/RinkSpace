"""Vectorized, sampled real-skin collar penetration factors in solver units.

This surrogate samples triangle-section endpoints and segment midpoints. It
does not certify the continuous segment minimum or complete 3D collision;
skate_collar_review remains the independent, bounded segment check.
"""
from dataclasses import dataclass
import numpy as np

from .skate_collar_review import _section_polygon


def _readonly(value):
    value=np.array(value,copy=True);value.setflags(write=False)
    return value


@dataclass(frozen=True)
class TriangleSpec:
    edges: np.ndarray
    triangle_edges: np.ndarray
    edge_faces: np.ndarray
    edge_face_counts: np.ndarray


@dataclass(frozen=True)
class CollarSpec:
    key: tuple
    sample_ids: np.ndarray
    body_faces: tuple
    body: tuple
    inner_vertices: np.ndarray
    inner_faces: np.ndarray
    collar: np.ndarray
    inner_edge_vertices: np.ndarray
    inner_edge_fraction: np.ndarray
    height_solver: float
    tolerance_solver: float
    inner_unknown_reason: str

    def __getitem__(self,key):return getattr(self,key)


def _triangle_spec(faces):
    flat=np.sort(np.asarray(faces)[:,[[0,1],[1,2],[2,0]]],axis=-1).reshape(-1,2)
    edges,inverse=np.unique(flat,axis=0,return_inverse=True)
    counts=np.bincount(inverse,minlength=len(edges));order=np.argsort(inverse,kind='stable')
    starts=np.r_[0,np.cumsum(counts)[:-1]]
    face_ids=np.repeat(np.arange(len(faces)),3)[order]
    adjacent=np.stack((face_ids[starts],face_ids[starts+np.minimum(counts-1,1)]),axis=1)
    return TriangleSpec(*map(_readonly,(edges,inverse.reshape(-1,3),adjacent,counts)))


def prepare(binding):
    """Return immutable shape/topology data; registration columns stay live.

    sample_ids index the full SMPL-X surface. A caller supplying samples to
    evaluate must use this exact ordered selection, in body-local solver units.
    A shallow worker copy may share this immutable spec; all evaluations own
    their work buffers and never modify the binding's template arrays.
    """
    surface=binding.surface
    key=(np.asarray(binding.x[12:17],np.float64).tobytes(),float(binding.ratio),id(surface.weights),id(surface.faces))
    cached=getattr(binding,'_collar_factor_spec',None)
    if cached is not None and cached.key==key:return cached
    geometry=binding.packet();inner=np.asarray(geometry['inner']);count=len(geometry['xs'])
    collar=np.unique(geometry['collar']);faces=np.asarray(binding.faces)[np.asarray(binding.region_ids)!=3]
    if np.any(collar>=count*6):raise ValueError('Collar boundary includes tongue or invalid section vertices')
    tolerance=1e-8/binding.ratio;height=float(inner[collar,2].min()-.001/binding.ratio)
    polygon,reason=_section_polygon(inner,faces,height,tolerance)
    nodes=np.empty((0,2),int);fraction=np.empty(0)
    if reason is None:
        mesh_edges=_triangle_spec(faces).edges;distance=inner[:,2]-height
        on_plane=np.any(abs(distance[np.unique(faces)])<=tolerance)
        if on_plane:reason='inner:vertex_on_section'
        else:
            a,b=mesh_edges.T;selected=(distance[a]*distance[b])<0
            nodes=mesh_edges[selected];fraction=distance[nodes[:,0]]/(distance[nodes[:,0]]-distance[nodes[:,1]])
            xy=inner[nodes[:,0],:2]*(1-fraction[:,None])+inner[nodes[:,1],:2]*fraction[:,None]
            difference=np.linalg.norm(polygon[:,None]-xy[None],axis=-1)
            order=np.argmin(difference,axis=1)
            if len(order)!=len(nodes) or len(np.unique(order))!=len(nodes) or np.any(difference[np.arange(len(order)),order]>tolerance):
                reason='inner:ambiguous_edge_identity'
            else:nodes,fraction=nodes[order],fraction[order]
    elif not reason.startswith('inner:'):reason='inner:'+reason
    selected=[]
    for joints in ((4,7,10),(5,8,11)):
        relevant=surface.weights[:,joints].sum(axis=1)>.1
        selected.append(np.asarray(surface.faces)[relevant[np.asarray(surface.faces)].any(axis=1)])
    if any(not len(value) for value in selected):raise ValueError('Missing real shank/ankle skin triangles')
    sample_ids=np.unique(np.concatenate(selected).ravel())
    local_faces=tuple(_readonly(np.searchsorted(sample_ids,value)) for value in selected)
    spec=CollarSpec(key,_readonly(sample_ids),local_faces,tuple(_triangle_spec(value) for value in local_faces),
        _readonly(inner),_readonly(faces),_readonly(collar),_readonly(nodes),_readonly(fraction),height,tolerance,reason or '')
    binding._collar_factor_spec=spec
    return spec


def _cross(a,b):return a[...,0]*b[...,1]-a[...,1]*b[...,0]


def _simple_loops(points,counts,tolerance):
    """Batched simple-loop geometry checks, retaining every concavity."""
    n,length=points.shape[:2]
    if length<3:return np.zeros(n,bool)
    index=np.arange(length)[None];valid=index<counts[:,None]
    following=np.take_along_axis(points,((index+1)%np.maximum(counts[:,None],1))[...,None],axis=1)
    vectors=following-points
    area=np.sum(np.where(valid,_cross(points,following),0.),axis=1)/2
    good=(counts>=3)&(abs(area)>tolerance*tolerance)&~np.any(valid&(np.linalg.norm(vectors,axis=-1)<=tolerance),axis=1)
    i,j=np.triu_indices(length,1)
    use=(j[None]<counts[:,None])&(j[None]!=i[None]+1)&~((i[None]==0)&(j[None]==counts[:,None]-1))
    a,b=points[:,i],following[:,i];c,d=points[:,j],following[:,j]
    area_tolerance=tolerance*np.maximum(np.linalg.norm(b-a,axis=-1),np.linalg.norm(d-c,axis=-1))
    o1,o2=_cross(b-a,c-a),_cross(b-a,d-a);o3,o4=_cross(d-c,a-c),_cross(d-c,b-c)
    intersects=(((o1<=area_tolerance)&(o2>=-area_tolerance))|((o2<=area_tolerance)&(o1>=-area_tolerance)))
    intersects&=(((o3<=area_tolerance)&(o4>=-area_tolerance))|((o4<=area_tolerance)&(o3>=-area_tolerance)))
    intersects&=(np.maximum(np.minimum(a,b),np.minimum(c,d))<=np.minimum(np.maximum(a,b),np.maximum(c,d))+tolerance).all(axis=-1)
    return good&~np.any(use&intersects,axis=1)


def _section_samples(vertices,topology,height,tolerance,*,return_association=False):
    """Batched triangle cuts, graph traversal and endpoint/midpoint packing."""
    n=len(vertices);edges=topology.edges;tri=topology.triangle_edges
    a,b=vertices[:,edges[:,0]],vertices[:,edges[:,1]]
    za,zb=a[:,:,2]-height,b[:,:,2]-height
    crossed=(za*zb)<0.;triangle_crossed=crossed[:,tri];number=triangle_crossed.sum(axis=2)
    fraction=np.divide(za,za-zb,out=np.zeros_like(za),where=abs(za-zb)>tolerance)
    xy=a[:,:,:2]+fraction[:,:,None]*(b[:,:,:2]-a[:,:,:2])
    reason=np.full(n,'',dtype='<U48')
    reason[~crossed.any(axis=1)]='body:no_triangle_intersection'
    on_plane=np.any((abs(za)<=tolerance)|(abs(zb)<=tolerance),axis=1)
    reason[(reason=='')&on_plane]='body:vertex_on_section'
    bad_degree=np.any(crossed&(topology.edge_face_counts[None]!=2),axis=1)
    reason[(reason=='')&bad_degree]='body:open_or_branched_section'
    bad_triangle=np.any((number!=0)&(number!=2),axis=1)
    reason[(reason=='')&bad_triangle]='body:ambiguous_triangle_section'
    active=reason=='';current=np.argmax(crossed,axis=1);start=current.copy();previous=np.full(n,-1)
    visited=np.zeros_like(crossed);orders=[];row=np.arange(n)
    sums=np.where(triangle_crossed,tri[None],0).sum(axis=2)
    while np.any(active):
        orders.append(np.where(active,current,-1));visited[row[active],current[active]]=True
        adjacent=topology.edge_faces[current]
        first=sums[row,adjacent[:,0]]-current;second=sums[row,adjacent[:,1]]-current
        following=np.where(first!=previous,first,second)
        returned=following==start
        invalid=active&visited[row,following]&~returned
        reason[invalid]='body:branched_section'
        finished=active&returned
        multiple=finished&(visited.sum(axis=1)!=crossed.sum(axis=1))
        reason[multiple]='body:multiple_section_loops'
        active&=~(returned|invalid);previous,current=current,following
    if orders:
        order=np.stack(orders,axis=1);counts=(order>=0).sum(axis=1)
        loop=xy[row[:,None],np.maximum(order,0)]
        simple=_simple_loops(loop,counts,tolerance)
        reason[(reason=='')&~simple]='body:self_intersecting_or_degenerate_section'
    middle=np.sum(xy[:,tri]*triangle_crossed[:,:,:,None],axis=2)/2
    values=np.concatenate((xy,middle),axis=1);mask=np.concatenate((crossed,number==2),axis=1)
    counts=mask.sum(axis=1);width=int(counts.max(initial=0))
    packed=np.zeros((n,width,2));position=np.cumsum(mask,axis=1)-1
    r,c=np.nonzero(mask);packed[r,position[r,c]]=values[r,c]
    out=(packed,np.arange(width)[None]<counts[:,None],reason,counts)
    return out+(np.packbits(mask,axis=1),)if return_association else out


def _sampled_signed_gap(points,polygon,*,return_association=False):
    a=polygon;b=np.roll(polygon,-1,axis=1);vector=b-a
    denominator=np.sum(vector*vector,axis=-1)
    difference=points[:,:,None]-a[:,None]
    fraction=np.divide(np.einsum('npei,nei->npe',difference,vector),denominator[:,None],
        out=np.zeros(difference.shape[:-1]),where=denominator[:,None]>0)
    fraction=np.clip(fraction,0,1)
    distances=np.linalg.norm(difference-fraction[:,:,:,None]*vector[:,None],axis=-1);nearest=np.argmin(distances,axis=2)
    distance=np.take_along_axis(distances,nearest[:,:,None],axis=2)[:,:,0]
    selected=(a[:,None,:,1]>points[:,:,None,1])!=(b[:,None,:,1]>points[:,:,None,1])
    dy=b[:,:,1]-a[:,:,1];safe=np.where(dy==0,1.,dy)
    crossing=(b[:,:,0]-a[:,:,0])[:,None]*(points[:,:,None,1]-a[:,None,:,1])/safe[:,None]+a[:,None,:,0]
    inside=(np.count_nonzero(selected&(points[:,:,None,0]<crossing),axis=2)%2)==1
    gap=np.where(inside,distance,-distance)
    if not return_association:return gap
    selected_fraction=np.take_along_axis(fraction,nearest[:,:,None],axis=2)[:,:,0]
    return gap,nearest,(selected_fraction>0)&(selected_fraction<1),inside


def evaluate(binding,joints,frames,samples=None,*,hand_local_R=None,skate_relative_R=None):
    """Return nonnegative ``breach_solver[n,2]`` and explicit ``unknown``.

    samples, if supplied, must be full-surface vertices at prepare().sample_ids
    in the same native body-local solver unit as joints. Unknown rows return
    zero in the *measurement* field, with a separate finite domain penalty
    for residual assembly. Unknown sections invalidate quality selection;
    they are never valid zero-penetration observations.
    """
    spec=prepare(binding);joints,frames=np.asarray(joints),np.asarray(frames);n=len(joints)
    breach=np.zeros((n,2));unknown=np.ones((n,2),bool);reason=np.full((n,2),spec.inner_unknown_reason,dtype='<U48')
    count=np.zeros((n,2),int);minimum=np.full((n,2),np.nan)
    # A geometry-sized domain penalty prevents a lost section reducing the
    # objective. It is not a measured penetration or a new contact tolerance.
    domain_size=max(float(np.linalg.norm(np.ptp(spec.inner_vertices,axis=0))),np.finfo(float).eps)
    residual_breach=np.full((n,2),domain_size);branches=[]
    result=dict(breach_solver=breach,residual_breach_solver=residual_breach,
                unknown=unknown,unknown_reason=reason,sample_count=count,
                minimum_sample_gap_solver=minimum)
    if n==0 or spec.inner_unknown_reason:return result
    if samples is None:
        kwargs={} if hand_local_R is None else dict(hand_local_R=hand_local_R)
        samples=binding.surface.sample_vertices(joints,frames,spec.sample_ids,double_precision=True,**kwargs)
    samples=np.asarray(samples)
    if samples.shape!=(n,len(spec.sample_ids),3) or not np.isfinite(samples).all():raise ValueError('Invalid collar full-surface samples')
    R_FB,t_FB,_=binding.parameters()
    if skate_relative_R is not None:
        R_FB,t_FB=binding.relative_parameters(skate_relative_R)
        if len(R_FB)!=n:raise ValueError('skate_relative_R sample count differs')
    relative_kwargs={} if skate_relative_R is None else dict(skate_relative_R=skate_relative_R)
    alpha=binding.deformation(joints,frames,**relative_kwargs)
    inner=binding.deform(spec.inner_vertices,alpha)
    # Batch the distance tensors rather than allocating points*walls for an
    # entire programme or all exposure nodes at once.
    for first in range(0,n,128):
        last=min(first+128,n)
        for foot,ankle in enumerate((7,8)):
            if skate_relative_R is None:
                rotation=frames[first:last,ankle]@R_FB[foot]
                origin=joints[first:last,ankle]+np.einsum('nij,j->ni',frames[first:last,ankle],t_FB[foot])
            else:
                rotation=frames[first:last,ankle]@R_FB[first:last,foot]
                origin=joints[first:last,ankle]+np.einsum('nij,nj->ni',frames[first:last,ankle],t_FB[first:last,foot])
            body=np.einsum('nji,nvj->nvi',rotation,samples[first:last]-origin[:,None])
            points,mask,why,counts,section_branch=_section_samples(body,spec.body[foot],spec.height_solver,spec.tolerance_solver,return_association=True)
            vertices=inner[first:last,foot];ends=spec.inner_edge_vertices;fraction=spec.inner_edge_fraction
            polygon=vertices[:,ends[:,0],:2]*(1-fraction[None,:,None])+vertices[:,ends[:,1],:2]*fraction[None,:,None]
            simple=_simple_loops(polygon,np.full(last-first,len(ends)),spec.tolerance_solver)
            why[(why=='')&~simple]='inner:self_intersecting_or_degenerate_section'
            z_changed=np.any(abs(vertices[:,:,2]-spec.inner_vertices[None,:,2])>spec.tolerance_solver,axis=1)
            why[(why=='')&z_changed]='inner:section_height_changed'
            good=why=='';reason[first:last,foot]=why;unknown[first:last,foot]=~good;count[first:last,foot]=counts
            if points.shape[1]:
                gap,nearest,interior,inside=_sampled_signed_gap(points,polygon,return_association=True)
                values=np.where(mask,gap,np.inf).min(axis=1)
                k=np.argmin(np.where(mask,gap,np.inf),axis=1);ii=np.arange(len(k))
                branches.extend([section_branch,k,nearest[ii,k],interior[ii,k],inside[ii,k],values<0])
                minimum[first:last,foot]=np.where(good,values,np.nan)
                breach[first:last,foot]=np.where(good,np.maximum(-values,0.),0.)
                residual_breach[first:last,foot]=np.where(good,breach[first:last,foot],domain_size)
    from .weighted_blocks import signature
    result['association_signature']=signature(*branches)
    return result
