"""Native-pixel body and footwear factors with local image Jacobians.

Scalar contour distances keep the historical squared objective. Blade side is
one declared finite-edge candidate per curve, never a per-point side switch.
"""
import numpy as np
import cv2
from scipy.sparse import coo_matrix
from .footwear_fit import sample_polyline


def nearest(points,line,closed=False):
    p=np.asarray(points,float);a=np.asarray(line,float);b=np.roll(a,-1,axis=0) if closed else a[1:]
    if not closed:a=a[:-1]
    v=b-a;den=np.maximum(np.sum(v*v,1),1e-16);alpha=np.clip(np.einsum('mkc,kc->mk',p[:,None]-a[None],v)/den,0,1)
    delta=a[None]+alpha[:,:,None]*v[None]-p[:,None];squared=np.sum(delta*delta,2);which=np.argmin(squared,1);rows=np.arange(len(p));norm=np.sqrt(squared[rows,which]);unit=np.divide(delta[rows,which],norm[:,None],out=np.zeros((len(p),2)),where=norm[:,None]>1e-14)
    return norm,which,alpha[rows,which],unit



def nearest_segment(points,vertices,edges):
    """Nearest *declared finite segment* for each source point.

    Returns distance, segment row, clipped segment coordinate, and the unit
    vector from source to prediction. Its derivative is piecewise smooth away
    from nearest-edge switches, projection degeneracy, and zero distance.
    """
    p=np.asarray(points,float);vertices=np.asarray(vertices,float);edges=np.asarray(edges)
    if p.ndim!=2 or p.shape[1]!=2 or vertices.ndim!=2 or vertices.shape[1]!=2:
        raise ValueError('Expected source points and projected vertices with shape (N,2)')
    if edges.ndim!=2 or edges.shape[1]!=2 or len(edges)==0 or edges.dtype.kind not in 'iu':
        raise ValueError('Explicit boot edges require a nonempty integer (E,2) array; no convex-hull fallback')
    if np.any(edges<0) or np.any(edges>=len(vertices)) or np.any(edges[:,0]==edges[:,1]):
        raise ValueError('Explicit segment vertex indices are invalid')
    a=vertices[edges[:,0]];b=vertices[edges[:,1]]
    if not np.isfinite(p).all() or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite source or explicit-edge projection')
    vector=b-a;den=np.maximum(np.sum(vector*vector,axis=1),1e-16)
    alpha=np.clip(np.einsum('mkc,kc->mk',p[:,None]-a[None],vector)/den,0,1)
    delta=a[None]+alpha[:,:,None]*vector[None]-p[:,None]
    squared=np.sum(delta*delta,axis=2);which=np.argmin(squared,axis=1);rows=np.arange(len(p))
    distance=np.sqrt(squared[rows,which]);unit=np.divide(delta[rows,which],distance[:,None],
        out=np.zeros((len(p),2)),where=distance[:,None]>1e-14)
    return distance,which,alpha[rows,which],unit


def _boot_edge_rows(boot_edges,frame,foot):
    """Local shell indices 0..13, keyed by (frame,foot) or nested by frame/foot."""
    from collections.abc import Mapping
    try:
        if isinstance(boot_edges,Mapping):
            value=boot_edges[(frame,foot)] if (frame,foot) in boot_edges else boot_edges[frame][foot]
        else:value=boot_edges[frame][foot]
    except (KeyError,IndexError,TypeError) as error:
        raise ValueError(f'Missing explicit boot edges for frame={frame}, foot={foot}; no convex-hull fallback') from error
    if isinstance(value,Mapping):
        if 'edges' not in value:raise ValueError('Explicit visible-edge packet is missing edges; no convex-hull fallback')
        value=value['edges']
    edges=np.asarray(value)
    if edges.ndim!=2 or edges.shape[1]!=2 or not len(edges) or edges.dtype.kind not in 'iu':
        raise ValueError(f'No admissible explicit boot edges for observed frame={frame}, foot={foot}; no convex-hull fallback')
    if np.any(edges<0) or np.any(edges>=14) or np.any(edges[:,0]==edges[:,1]):
        raise ValueError('Boot edges must reference distinct local outer-shell vertices 0..13')
    return edges


def pixel_factors(uv,problem,body=True,blade_side_candidates=True,boot_edges=None):
    """uv=(N,113,2); return residual, dres/d(flat uv), parts, choices.

    boot_edges=None replays the historical projected convex hull. An explicit
    per-frame/foot edge set is mandatory once supplied; it never falls back to
    a hull, joins disconnected segments, or removes observed source samples.
    The caller owns visible-edge selection at the common exposure midpoint;
    row-time projected vertices must come from the same continuous body state.
    Heel/toe landmarks below target outer boot edges, not bare-foot landmarks
    or anatomical ankle centres.
    """
    residual=[];rr=[];cc=[];vv=[];parts={};offset=0;choices=[]
    def add(values,name,columns,derivative):
        nonlocal offset
        value=np.asarray(values,float).ravel();residual.append(value);parts[name]=parts.get(name,0.)+float(value@value/2)
        columns=np.asarray(columns,int);derivative=np.asarray(derivative,float)
        rr.extend(np.repeat(np.arange(offset,offset+len(value)),columns.shape[1]));cc.extend(columns.ravel());vv.extend(derivative.ravel());offset+=len(value)
    def curve(observed,fi,ids,scale,name,closed=False):
        distance,k,a,u=nearest(observed,uv[fi,ids],closed);end=(k+1)%len(ids);pairs=np.stack((ids[k],ids[end]),1);columns=((fi*113+pairs[:,:,None])*2+np.arange(2)[None,None]).reshape(len(k),4)
        derivative=np.concatenate(((1-a[:,None])*u,a[:,None]*u),axis=1)/scale
        add(distance/scale,name,columns,derivative)
    for row in problem.visible_rows:
        if row.get('reverse_boundary',True):raise ValueError('Joint experiment requires explicitly partial source-only contours')
        fi=row['frame'];base=row['foot']*48
        if boot_edges is None:
            ids=np.arange(14)+base;h=cv2.convexHull(uv[fi,ids].astype(np.float32),returnPoints=False).ravel()
            curve(row['observed'],fi,ids[h],row['sigma_px']*np.sqrt(48),'visible_boot_edges',True)
        else:
            edges=_boot_edge_rows(boot_edges,fi,row['foot'])+base
            distance,k,alpha,unit=nearest_segment(row['observed'],uv[fi],edges)
            pairs=edges[k];columns=((fi*113+pairs[:,:,None])*2+np.arange(2)[None,None]).reshape(len(k),4)
            scale=row['sigma_px']*np.sqrt(48)
            derivative=np.concatenate(((1-alpha[:,None])*unit,alpha[:,None]*unit),axis=1)/scale
            add(distance/scale,'visible_boot_edges',columns,derivative)
    for row in problem.blade_rows:
        fi,foot=row['frame'],row['foot'];source=sample_polyline(row['uv'],24);candidates=[np.arange(14,31)+foot*48]
        if blade_side_candidates:candidates.append(np.arange(31,48)+foot*48)
        best=min(range(len(candidates)),key=lambda i:float(np.sum(nearest(source,uv[fi,candidates[i]])[0]**2)));ids=candidates[best];choices.append(dict(pts=row['pts'],foot=foot,selected_candidate='blade_left' if best==0 else 'blade_right',source_side_confirmed=False))
        curve(source,fi,ids,row['sigma_px']*np.sqrt(24),'visible_blade_curves')
        if not row['partial']:
            # Reverse curve coverage is not admitted for the current controlled
            # joint target; current complete curves are evaluated as source
            # lower edges with finite endpoints, with this change recorded.
            raise ValueError('Declare source-only blade segments explicitly; physical endpoint coverage is not assumed')
    for row in problem.landmark_rows:
        fi,foot=row['frame'],row['foot']
        # Finite outer-boot heel/toe lower edges; never bare-foot or bony ankle points.
        for name,pair in [('heel',[0,7]),('toe',[1,8])]:
            ids=np.asarray(pair)+foot*48;a,b=uv[fi,ids];v=b-a;point=np.asarray(row[name]);den=max(float(v@v),1e-16);raw=float((point-a)@v/den);alpha=np.clip(raw,0,1);q=a+alpha*v;Ja=(1-alpha)*np.eye(2);Jb=alpha*np.eye(2)
            if 0<raw<1:
                da=(-(point-a)+(2*alpha-1)*v)/den;db=((point-a)-2*alpha*v)/den;Ja+=np.outer(v,da);Jb+=np.outer(v,db)
            columns=np.tile(((fi*113+ids[:,None])*2+np.arange(2)).ravel(),(2,1));derivative=np.c_[Ja,Jb]/row['sigma_px'];add((q-point)/row['sigma_px'],'boot_direction_endpoints',columns,derivative)
    if body:
        ids=problem.body_problem.indices;frames=problem.frames;qualified=problem.p['source_qualified'][frames][:,ids]&(~problem.body_problem.hold[frames,None])
        fi,ki=np.nonzero(qualified);point_ids=ids[ki]+96;error=(uv[fi,point_ids]-problem.p['source_uv'][frames[fi],ids[ki]])/problem.body_problem.sig[ki,None];columns=((fi*113+point_ids)[:,None]*2+np.arange(2)).reshape(-1,1);weight=np.repeat(1/problem.body_problem.sig[ki],2)[:,None];add(error,'body_pixels',columns,weight)
    vector=np.concatenate(residual) if residual else np.empty(0);A=coo_matrix((vv,(rr,cc)),shape=(len(vector),uv.size)).tocsr();return vector,A,parts,choices
