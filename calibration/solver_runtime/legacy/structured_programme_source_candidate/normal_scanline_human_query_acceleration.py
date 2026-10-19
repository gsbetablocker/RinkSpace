"""Strict-double original normal-scanline reductions; no projection/model change.

Explicit source-addressed build required. An unavailable or unsafe kernel
returns None so the calling original NumPy implementation remains authority.
The solver never compiles a binary. Each call owns every output buffer.
"""
from pathlib import Path
import ctypes,hashlib,json,os
import numpy as np
SOURCE=Path(__file__).with_name('native')/'normal_scanline_human_query.c'
SOURCE_SHA=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
BUILD=Path(os.environ.get('RINKSPACE_SOLVER_NATIVE_CACHE_ROOT','D:/CodexData/RinkSpace_Data/native/solver_runtime'))/'normal_scanline_human_query'/SOURCE_SHA
_library=_kernel=_rectangle_kernel=None;_error=None
try:
    manifest=json.loads((BUILD/'build.json').read_text(encoding='utf8'));binary=BUILD/'normal_scanline_human_query.dll'
    if manifest['source_sha256']!=SOURCE_SHA or manifest['dll_sha256']!=hashlib.sha256(binary.read_bytes()).hexdigest():raise ValueError('Strict normal query kernel source/binary identity changed')
    _library=ctypes.CDLL(str(binary));_kernel=_library.normal_scanline_human_strict
    _kernel.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int64,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int64,ctypes.c_void_p,ctypes.c_int,
                     ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]
    _kernel.restype=ctypes.c_int
    _rectangle_kernel=_library.finite_source_rectangle_strict
    _rectangle_kernel.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int64]+[ctypes.c_void_p]*3+[ctypes.c_int64,ctypes.c_void_p,ctypes.c_int]+[ctypes.c_void_p]*3
    _rectangle_kernel.restype=ctypes.c_int
except (OSError,ValueError,KeyError,AttributeError)as exc:_error=str(exc)

def normal_scanline_human_projected_queries(projected,depth,central,midpoint,uv,normals,bounds,*,tangent_bounds=None):
    if _kernel is None:return None
    projected=np.asarray(projected,float);uv=np.asarray(uv,float);normals=np.asarray(normals,float);bounds=np.asarray(bounds,float)
    start,delta=projected[:-1],np.diff(projected,axis=0);length2=np.einsum('ni,ni->n',delta,delta)
    valid=(central[:-1]&central[1:]&midpoint&np.isfinite(start).all(axis=1)&np.isfinite(delta).all(axis=1)&(depth[:-1]>0)&(depth[1:]>0)&(length2>1e-4))
    if not np.any(valid):raise ValueError('Human finite physical primitive has no positive central segment')
    ids=np.flatnonzero(valid);a=start[ids];d=delta[ids];l2=length2[ids];threshold=32*np.finfo(float).eps*np.sqrt(l2)
    # Preserve original NumPy nonfinite/overflow semantics through its fallback.
    if not all(np.isfinite(x).all()and (not x.size or np.max(abs(x))<1e140)for x in (uv,normals,a,d,l2)):return None
    inputs=[np.ascontiguousarray(x,dtype=np.float64)for x in (uv,normals,a,d,l2,threshold)]
    n=len(uv);covered=np.empty(n,np.uint8);ri=np.empty(n,np.int64);rs=np.empty(n);rv=np.empty(n);fi=np.empty(n,np.int64);fs=np.empty(n);foot=np.empty((n,2))
    threads=max(1,min(20,int(os.environ.get('RINKSPACE_HUMAN_QUERY_THREADS',str(os.cpu_count()or 1)))))
    p=lambda x:x.ctypes.data
    code=_kernel(p(inputs[0]),p(inputs[1]),n,p(inputs[2]),p(inputs[3]),p(inputs[4]),len(ids),p(inputs[5]),threads,p(covered),p(ri),p(rs),p(rv),p(fi),p(fs),p(foot))
    if code:return None
    coverage=covered.astype(bool);vector=foot-uv
    fallback_signed=np.einsum('ni,ni->n',vector,normals);tangent=np.c_[-normals[:,1],normals[:,0]];fallback_tangent=np.einsum('ni,ni->n',vector,tangent)
    selected_signed=np.where(coverage,rv,fallback_signed)
    selected_station=np.where(coverage,(ids[ri]+rs)/(len(projected)-1),(ids[fi]+fs)/(len(projected)-1))
    radius=np.max(np.abs(bounds),axis=1);tangent_bounds=np.c_[-radius,radius]if tangent_bounds is None else np.asarray(tangent_bounds,float)
    if tangent_bounds.shape!=bounds.shape or not np.isfinite(tangent_bounds).all()or np.any(tangent_bounds[:,0]>tangent_bounds[:,1]):raise ValueError('Finite authenticated drawing/context tangent envelope required')
    coverage_raw=np.where(coverage,0.,fallback_tangent);normal_violation=selected_signed-np.clip(selected_signed,bounds[:,0],bounds[:,1]);coverage_violation=coverage_raw-np.clip(coverage_raw,tangent_bounds[:,0],tangent_bounds[:,1]);violation=np.c_[normal_violation,coverage_violation].ravel()
    if not np.isfinite(violation).all():raise ValueError('Nonfinite direct human stroke query')
    return dict(residual=violation,signed_normal_px=selected_signed,coverage_tangent_px=coverage_raw,finite_scanline_intersection=coverage,
        finite_primitive_station=selected_station,normal_violation_px=normal_violation,coverage_violation_px=coverage_violation,
        nearest_finite_vector_px=vector,nearest_finite_segment_index=ids[fi],nearest_finite_segment_station=fs,
        nearest_finite_signed_normal_px=fallback_signed,nearest_finite_tangent_px=fallback_tangent,finite_source_association='normal_scanline',
        valid_projected_segment_count=len(ids),endpoint_coverage_allowance_px=radius)


RECTANGLE_OPERATOR='rinkspace.finite-curve-original-source-rectangle-Linf-distance.v1'


def _rectangle_nominal_local_stations(uv,normals,bounds,tb,a,d,l2,forced,categories,faces=None,station_branches=None):
    """Local segment derivative, retaining its nominal endpoint category.

    A coordinate stencil can leave an interior computational segment even
    when the actual small direction stays inside. Replay the local infinite
    segment branch for interior points, and the literal endpoint for clamps.
    Nonlinear values never call this derivative-only path.
    """
    a,d,l2=a[forced],d[forced],l2[forced]
    vector=a-uv;tangent=np.c_[-normals[:,1],normals[:,0]]
    an=np.einsum('ni,ni->n',vector,normals);at=np.einsum('ni,ni->n',vector,tangent)
    dn=np.einsum('ni,ni->n',d,normals);dt=np.einsum('ni,ni->n',d,tangent)
    cn,ct=abs(dn),abs(dt);ln,un=bounds.T;lt,ut=tb.T
    lnv=np.where(dn>=0,ln-an,an-un);unv=np.where(dn>=0,un-an,an-ln)
    ltv=np.where(dt>=0,lt-at,at-ut);utv=np.where(dt>=0,ut-at,at-lt)
    values=np.stack((np.zeros(len(uv)),(lnv*ct-utv*cn)/(cn+ct),(ltv*cn-unv*ct)/(cn+ct)),axis=1)
    u=np.max(values,axis=1)
    if faces is not None:
        interior=categories==0
        if np.any(interior&np.isin(faces,[1,2,3,4])):
            raise ValueError('An active finite endpoint support face cannot be replayed as interior')
        local=np.where(faces==5,1,np.where(faces==6,2,0))
        u=np.where(interior,values[np.arange(len(uv)),local],u)
    if faces is None:
        # Unqualified local replay still chooses its actual nonnegative
        # deficit. A frozen face instead needs its signed smooth extension:
        # a wide coordinate stencil may cross u=0 although the actual small
        # direction stays on the same positive face.
        u=np.maximum(u,np.where(dn==0,np.maximum(ln-an,an-un),0.))
        u=np.maximum(u,np.where(dt==0,np.maximum(lt-at,at-ut),0.))
    lows=[];highs=[]
    for delta,origin,lower,upper in ((dn,an,ln,un),(dt,at,lt,ut)):
        nonzero=delta!=0.
        low=np.divide(np.where(delta>0,lower-origin-u,upper-origin+u),delta,
            out=np.full_like(delta,-np.inf),where=nonzero)
        high=np.divide(np.where(delta>0,upper-origin+u,lower-origin-u),delta,
            out=np.full_like(delta,np.inf),where=nonzero)
        lows.append(low);highs.append(high)
    lo=np.maximum(lows[0],lows[1]);hi=np.minimum(highs[0],highs[1])
    t=np.clip(-(an*dn+at*dt)/l2,lo,hi)
    t=np.where(lo>hi,.5*(lo+hi),t)
    if station_branches is not None:
        options=np.stack((-(an*dn+at*dt)/l2,np.zeros(len(uv)),lows[0],lows[1],
            np.ones(len(uv)),highs[0],highs[1]),axis=1)
        ordinary=station_branches<7
        t[ordinary]=options[np.flatnonzero(ordinary),station_branches[ordinary]]
        unusual=~ordinary
        if np.any(unusual):
            code=station_branches[unusual]-7
            lower=np.c_[np.zeros(len(uv)),*lows];upper=np.c_[np.ones(len(uv)),*highs]
            rows=np.flatnonzero(unusual)
            t[unusual]=.5*(lower[rows,code//3]+upper[rows,code%3])
    t=np.where(categories==1,0.,np.where(categories==2,1.,t))
    return forced,t,a+t[:,None]*d


def _rectangle_selected_branches(uv,normals,bounds,tb,a,d,l2,chosen,station):
    """Exact active faces identify a clamp even if t is one ulp inside it."""
    a,d,l2=a[chosen],d[chosen],l2[chosen]
    v=a-uv;tangent=np.c_[-normals[:,1],normals[:,0]]
    an=np.einsum('ni,ni->n',v,normals);at=np.einsum('ni,ni->n',v,tangent)
    dn=np.einsum('ni,ni->n',d,normals);dt=np.einsum('ni,ni->n',d,tangent)
    cn,ct=abs(dn),abs(dt);ln,un=bounds.T;lt,ut=tb.T
    lnv=np.where(dn>=0,ln-an,an-un);unv=np.where(dn>=0,un-an,an-ln)
    ltv=np.where(dt>=0,lt-at,at-ut);utv=np.where(dt>=0,ut-at,at-lt)
    candidates=np.stack((np.zeros(len(uv)),ln-np.maximum(an,an+dn),np.minimum(an,an+dn)-un,
        lt-np.maximum(at,at+dt),np.minimum(at,at+dt)-ut,
        (lnv*ct-utv*cn)/(cn+ct),(ltv*cn-unv*ct)/(cn+ct)),axis=1)
    faces=np.argmax(candidates,axis=1);u=candidates[np.arange(len(uv)),faces]
    category=np.where(station<=0.,1,np.where(station>=1.,2,0))
    for face,at_end in ((1,dn>=0.),(2,dn<0.),(3,dt>=0.),(4,dt<0.)):
        category=np.where(faces==face,np.where(at_end,2,1),category)
    lows=[np.zeros(len(uv))];highs=[np.ones(len(uv))]
    for delta,origin,lower,upper in ((dn,an,ln,un),(dt,at,lt,ut)):
        lows.append(np.divide(np.where(delta>0,lower-origin-u,upper-origin+u),delta,
            out=np.full_like(delta,-np.inf),where=delta!=0.))
        highs.append(np.divide(np.where(delta>0,upper-origin+u,lower-origin-u),delta,
            out=np.full_like(delta,np.inf),where=delta!=0.))
    lows=np.stack(lows,axis=1);highs=np.stack(highs,axis=1)
    low_index=np.argmax(lows,axis=1);high_index=np.argmin(highs,axis=1)
    lo=lows[np.arange(len(uv)),low_index];hi=highs[np.arange(len(uv)),high_index]
    projection=-(an*dn+at*dt)/l2
    branch=np.where(projection<lo,1+low_index,np.where(projection>hi,4+high_index,0))
    branch=np.where(lo>hi,7+3*low_index+high_index,branch)
    # Cross-slab optima have one exact shared station. Floating lo/hi order
    # is not an operator branch; use the same normal face symbolically.
    branch=np.where(faces==5,np.where(dn!=0.,2,6),
        np.where(faces==6,np.where(dn!=0.,5,3),branch))
    branch=np.where(category==1,1,np.where(category==2,4,branch))
    return category,faces,branch


def _rectangle_exact_constant_support_axes(normal,transverse,normals,bounds,tb,d,chosen,categories,faces,stations):
    """Evaluate a selected zero-deficit slab coordinate by its identity.

    An interior slab station sets that source coordinate to its literal
    original bound. Reconstructing the projected foot and dotting it again
    adds cancellation noise to a mathematically constant coordinate, creating
    false weak half-spaces and nonzero finite differences. This is neither a
    tolerance nor a support expansion: the exact selected branch implies the
    equality. Positive-deficit, endpoint and free-foot branches are unchanged.
    """
    normal=normal.copy();transverse=transverse.copy()
    delta=d[chosen];tangent=np.c_[-normals[:,1],normals[:,0]]
    dn=np.einsum('ni,ni->n',delta,normals);dt=np.einsum('ni,ni->n',delta,tangent)
    inside=(categories==0)&(faces==0)
    for branch,coordinate,change,limits,low in (
        (2,normal,dn,bounds,True),(5,normal,dn,bounds,False),
        (3,transverse,dt,tb,True),(6,transverse,dt,tb,False)):
        selected=inside&(stations==branch)&(change!=0.)
        lower=np.where(change>0.,limits[:,0],limits[:,1])
        upper=np.where(change>0.,limits[:,1],limits[:,0])
        coordinate[selected]=(lower if low else upper)[selected]
    return normal,transverse


def _rectangle_reference_stations(uv,normals,bounds,tb,a,d,l2,forced):
    """Exact slab-intersection reference; each row selects an original segment."""
    chosen=np.empty(len(uv),int);station=np.empty(len(uv))
    # Chunking bounds temporary memory only, without omitting any curve face.
    for first in range(0,len(uv),128):
        last=min(first+128,len(uv));normal=normals[first:last,None,:]
        tangent=np.stack((-normal[:,:,1],normal[:,:,0]),axis=2)
        vectors=a[None,:,:]-uv[first:last,None,:]
        an=np.sum(vectors*normal,axis=2);at=np.sum(vectors*tangent,axis=2)
        dn=np.sum(d[None,:,:]*normal,axis=2);dt=np.sum(d[None,:,:]*tangent,axis=2)
        cn,ct=abs(dn),abs(dt);ln,un=bounds[first:last,0,None],bounds[first:last,1,None]
        lt,ut=tb[first:last,0,None],tb[first:last,1,None]
        u=np.maximum.reduce((np.zeros_like(an),ln-np.maximum(an,an+dn),np.minimum(an,an+dn)-un,
            lt-np.maximum(at,at+dt),np.minimum(at,at+dt)-ut))
        lnv=np.where(dn>=0,ln-an,an-un);unv=np.where(dn>=0,un-an,an-ln)
        ltv=np.where(dt>=0,lt-at,at-ut);utv=np.where(dt>=0,ut-at,at-lt)
        u=np.maximum(u,(lnv*ct-utv*cn)/(cn+ct));u=np.maximum(u,(ltv*cn-unv*ct)/(cn+ct))
        lo=np.zeros_like(an);hi=np.ones_like(an)
        for delta,origin,lower,upper in ((dn,an,ln,un),(dt,at,lt,ut)):
            nonzero=delta!=0.
            low=np.divide(np.where(delta>0,lower-origin-u,upper-origin+u),delta,
                out=np.full_like(delta,-np.inf),where=nonzero)
            high=np.divide(np.where(delta>0,upper-origin+u,lower-origin-u),delta,
                out=np.full_like(delta,np.inf),where=nonzero)
            lo=np.maximum(lo,low);hi=np.minimum(hi,high)
        t=np.clip(-(an*dn+at*dt)/l2[None,:],lo,hi)
        t=np.clip(np.where(lo>hi,.5*(lo+hi),t),0.,1.)
        f=forced[first:last];mask=f>=0
        selection=np.argmin(u,axis=1);selection[mask]=f[mask]
        chosen[first:last]=selection;station[first:last]=t[np.arange(last-first),selection]
    return chosen,station,a[chosen]+station[:,None]*d[chosen]


def finite_source_rectangle_projected_queries(projected,depth,central,midpoint,uv,normals,bounds,*,
        tangent_bounds=None,frozen_segments=None,frozen_segment_categories=None,
        frozen_support_faces=None,frozen_station_branches=None,native=True):
    """Minimum original normal×tangent support deficit on the finite curve.

    A remote scanline tangency never overrides a closer feasible source box.
    No support is widened, and every originally valid optical segment remains.
    Fixed segments are exclusively the nominal derivative replay, not guards.
    """
    projected=np.asarray(projected,float);uv=np.asarray(uv,float);normals=np.asarray(normals,float);bounds=np.asarray(bounds,float)
    radius=np.max(abs(bounds),axis=1);tb=np.c_[-radius,radius]if tangent_bounds is None else np.asarray(tangent_bounds,float)
    if(uv.ndim!=2 or uv.shape[1:]!=(2,) or normals.shape!=uv.shape or bounds.shape!=uv.shape or tb.shape!=uv.shape
        or not all(np.isfinite(x).all()for x in (uv,normals,bounds,tb)) or np.any(bounds[:,0]>bounds[:,1])or np.any(tb[:,0]>tb[:,1])):
        raise ValueError('Finite unchanged source normal and tangent support rectangles required')
    start,delta=projected[:-1],np.diff(projected,axis=0);length2=np.einsum('ni,ni->n',delta,delta)
    valid=(central[:-1]&central[1:]&midpoint&np.isfinite(start).all(axis=1)&np.isfinite(delta).all(axis=1)&(depth[:-1]>0)&(depth[1:]>0)&(length2>1e-4))
    ids=np.flatnonzero(valid)
    if not len(ids):raise ValueError('Human finite physical primitive has no positive central segment')
    a=np.ascontiguousarray(start[ids]);d=np.ascontiguousarray(delta[ids]);l2=np.ascontiguousarray(length2[ids])
    forced=np.full(len(uv),-1,np.int64)
    if frozen_segment_categories is None and(frozen_support_faces is not None or frozen_station_branches is not None):
        raise ValueError('Nominal support faces require the finite segment endpoint categories')
    if frozen_segments is not None:
        raw=np.asarray(frozen_segments)
        if raw.shape!=(len(uv),)or raw.dtype.kind not in 'iu':raise ValueError('Nominal original finite segment per source vertex required')
        forced=np.searchsorted(ids,raw).astype(np.int64)
        if np.any(forced>=len(ids))or not np.array_equal(ids[forced],raw):
            raise ValueError('Nominal support segment left the original positive central physical domain')
    if frozen_segment_categories is not None:
        categories=np.asarray(frozen_segment_categories)
        if(frozen_segments is None or categories.shape!=(len(uv),)or categories.dtype.kind not in 'iu'
            or np.any((categories<0)|(categories>2))):
            raise ValueError('Nominal finite segment endpoint category per source vertex required')
        for branch,maximum in ((frozen_support_faces,6),(frozen_station_branches,15)):
            if branch is not None:
                branch=np.asarray(branch)
                if branch.shape!=(len(uv),)or branch.dtype.kind not in 'iu'or np.any((branch<0)|(branch>maximum)):
                    raise ValueError('Authenticated nominal source support branch per vertex required')
        if frozen_support_faces is not None:frozen_support_faces=np.asarray(frozen_support_faces)
        if frozen_station_branches is not None:frozen_station_branches=np.asarray(frozen_station_branches)
        chosen,station,foot=_rectangle_nominal_local_stations(uv,normals,bounds,tb,a,d,l2,forced,categories,
            faces=frozen_support_faces,station_branches=frozen_station_branches)
    elif native and _rectangle_kernel is not None:
        inputs=[np.ascontiguousarray(x,dtype=np.float64)for x in(uv,normals,bounds,tb)]
        chosen=np.empty(len(uv),np.int64);station=np.empty(len(uv));foot=np.empty_like(uv)
        threads=max(1,min(20,int(os.environ.get('RINKSPACE_HUMAN_QUERY_THREADS',str(os.cpu_count()or 1)))))
        p=lambda x:x.ctypes.data
        code=_rectangle_kernel(*[p(x)for x in inputs],len(uv),p(a),p(d),p(l2),len(ids),p(forced),threads,p(chosen),p(station),p(foot))
        if code:raise ValueError('Original finite support rectangle native evaluation failed: '+str(code))
    else:chosen,station,foot=_rectangle_reference_stations(uv,normals,bounds,tb,a,d,l2,forced)
    vector=foot-uv;tangent=np.c_[-normals[:,1],normals[:,0]]
    normal=np.einsum('ni,ni->n',vector,normals);transverse=np.einsum('ni,ni->n',vector,tangent)
    segment=ids[chosen]
    categories,faces,station_branches=_rectangle_selected_branches(uv,normals,bounds,tb,a,d,l2,chosen,station)
    if frozen_segment_categories is not None:categories=np.asarray(frozen_segment_categories)
    if frozen_support_faces is not None:faces=np.asarray(frozen_support_faces)
    if frozen_station_branches is not None:station_branches=np.asarray(frozen_station_branches)
    normal,transverse=_rectangle_exact_constant_support_axes(normal,transverse,normals,bounds,tb,d,
        chosen,categories,faces,station_branches)
    nv=normal-np.clip(normal,bounds[:,0],bounds[:,1]);tv=transverse-np.clip(transverse,tb[:,0],tb[:,1])
    residual=np.c_[nv,tv].ravel();distance=np.maximum(abs(nv),abs(tv))
    if not np.isfinite(residual).all():raise ValueError('Nonfinite direct human source rectangle distance')
    return dict(residual=residual,signed_normal_px=normal,coverage_tangent_px=transverse,
        finite_scanline_intersection=(station>0.)&(station<1.),
        finite_primitive_station=(segment+station)/(len(projected)-1),normal_violation_px=nv,coverage_violation_px=tv,
        nearest_finite_vector_px=vector,nearest_finite_segment_index=segment,nearest_finite_segment_station=station,
        nearest_finite_signed_normal_px=normal,nearest_finite_tangent_px=transverse,
        nearest_fields_are_selected_original_support_rectangle_point_not_Euclidean_foot=True,
        finite_source_association='original_support_rectangle_Linf',source_support_operator=RECTANGLE_OPERATOR,
        source_support_distance_px=distance,source_support_window_intersection=(distance==0.),
        nominal_finite_segments=segment,frozen_segments_used_for_derivative_only=frozen_segments is not None,
        nominal_finite_segment_categories=categories,nominal_source_support_faces=faces,
        nominal_source_station_branches=station_branches,
        source_support_branch_signature=np.c_[segment,categories,faces,station_branches],
        nominal_endpoint_categories_replayed_for_derivative_only=frozen_segment_categories is not None,
        valid_projected_segment_count=len(ids),endpoint_coverage_allowance_px=radius)
