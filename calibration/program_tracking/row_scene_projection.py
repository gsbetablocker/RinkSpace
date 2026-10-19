"""Implicit world projection on the FIT native row-camera trajectory.

Only sensor-visible roots are physical row exposures. Off-screen vertices use
boundary-row cameras solely to navigate and clip finite world paths; they are
never FIT observations or exported as measured off-screen exposures.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation
from .first_shot_row_camera import prepare_frame_row_camera
from .projection import safe_project
from .adaptive_board_curve import _chord_intersects_sensor
from physical_projection import (_brown_map_jacobian, rational_minimum_denominator,
    _AXIS_STATIONS, _DENOM_FLOOR, _EIGEN_FLOOR)

class RowSurfaceDomainError(ValueError):
    """A wall polygon needs subdivision at an optical/row-domain transition."""

def clip_viewport_polygon(polygon,image_size=(1920,1080)):
    polygon=np.asarray(polygon,float).reshape(-1,2)
    if not np.isfinite(polygon).all():raise ValueError('Finite projected wall polygon required')
    for axis,bound,sign in [(0,0.,1),(0,image_size[0]-1.,-1),(1,0.,1),(1,image_size[1]-1.,-1)]:
        output=[]
        for i in range(len(polygon)):
            a,b=polygon[i-1],polygon[i];ga=sign*(a[axis]-bound)>=0;gb=sign*(b[axis]-bound)>=0
            if ga!=gb:output.append(a+(bound-a[axis])/(b[axis]-a[axis])*(b-a))
            if gb:output.append(b)
        polygon=np.asarray(output).reshape(-1,2)
        if len(polygon)<3:return np.empty((0,2))
    return polygon


def _batch_project(cameras, points):
    """The same central Brown branch guards, with one camera per world point."""
    cameras=np.asarray(cameras,float);points=np.asarray(points,float)
    if cameras.shape!=(len(points),25) or not np.isfinite(cameras).all():
        raise ValueError('Finite camera25 per world point required')
    if np.any(abs(cameras[:,23:25])>1e-12):raise ValueError('Brown tilt is fixed zero')
    rotations=Rotation.from_rotvec(cameras[:,3:6]).as_matrix()
    local=np.einsum('nij,nj->ni',rotations,points-cameras[:,:3]);depth=local[:,2]
    uv=np.full((len(points),2),np.nan);valid=np.zeros(len(points),bool)
    for distortion in np.unique(cameras[:,11:23],axis=0):
        selected=np.flatnonzero(np.all(cameras[:,11:23]==distortion,axis=1)&(depth>.05))
        if not len(selected):continue
        rays=local[selected,:2]/depth[selected,None]
        with np.errstate(all='ignore'):
            denominator=rational_minimum_denominator(rays,distortion)
            eigen=_brown_map_jacobian((rays[:,None,:]*_AXIS_STATIONS[None,:,None]).reshape(-1,2),distortion)[-1].reshape(len(rays),-1).min(1)
            good=np.isfinite(denominator)&(denominator>_DENOM_FLOOR)&np.isfinite(eigen)&(eigen>_EIGEN_FLOOR)
            selected=selected[good];mapped=_brown_map_jacobian(rays[good],distortion)[0];cam=cameras[selected]
            projected=np.c_[np.exp(cam[:,6])*mapped[:,0]+cam[:,10]*mapped[:,1]+cam[:,8],np.exp(cam[:,7])*mapped[:,1]+cam[:,9]]
        finite=np.isfinite(projected).all(1);uv[selected[finite]]=projected[finite];valid[selected[finite]]=True
    return uv,depth,valid


class RowSceneProjection:
    def __init__(self, context, reference_camera):
        self.context=context;self.reference_camera=np.asarray(reference_camera,float).copy()
        self.cache_key=('implicit-native-row-world.v1',context.cache_key)
        self.image_size=(1920,context.image_height)

    def project(self, camera, xyz):
        points=np.asarray(xyz,float).reshape(-1,3)
        if not np.isfinite(points).all():raise ValueError('Finite unchanged world points required')
        if not len(points):return np.empty((0,2)),np.empty(0),np.empty(0,bool)
        if self.context.readout_milliseconds==0.:return safe_project(camera,points)
        uv,depth,good=safe_project(camera,points)
        rows=np.where(good,uv[:,1],.5*(self.context.image_height-1))
        for _ in range(12):
            # These are bounded navigation guesses, not clamped physical roots.
            sampled=np.clip(rows,0.,self.context.image_height-1.)
            cameras=self.context.camera25_at_rows(sampled,geometry_navigation=True)
            uv,depth,good=_batch_project(cameras,points)
            updated=np.where(good,uv[:,1],sampled)
            sensor=good&(uv[:,0]>=0)&(uv[:,0]<self.image_size[0])&(updated>=0)&(updated<self.context.image_height)
            if np.all(np.abs(updated[sensor]-rows[sensor])<1e-6):break
            rows=updated
        error=np.abs(uv[:,1]-sampled)
        sensor=good&(uv[:,0]>=0)&(uv[:,0]<self.image_size[0])&(uv[:,1]>=0)&(uv[:,1]<self.context.image_height)
        retry=np.flatnonzero(sensor&(error>1e-5))
        if len(retry):
            selected=points[retry]
            grid=np.linspace(0.,self.context.image_height-1.,33)
            candidates=np.repeat(selected,len(grid),axis=0)
            cc=self.context.camera25_at_rows(np.tile(grid,len(selected)),geometry_navigation=False)
            pp,dd,vv=_batch_project(cc,candidates)
            ff=pp[:,1].reshape(-1,len(grid))-grid[None,:];vv=vv.reshape(-1,len(grid))
            brackets=vv[:,:-1]&vv[:,1:]&(ff[:,:-1]*ff[:,1:]<=0)
            supported=brackets.any(axis=1)
            absent=retry[~supported];good[absent]=False;uv[absent]=np.nan
            retry=retry[supported];selected=selected[supported];brackets=brackets[supported]
            if len(retry):
                distance=abs(.5*(grid[:-1]+grid[1:])[None,:]-sampled[retry,None])
                bracket=np.argmin(np.where(brackets,distance,np.inf),axis=1)
                lo=grid[bracket];hi=grid[bracket+1]
            def evaluate(rows):
                cc=self.context.camera25_at_rows(rows,geometry_navigation=False)
                p,d,v=_batch_project(cc,selected);return p,d,v,p[:,1]-rows
            if len(retry):
                pa,da,va,fa=evaluate(lo)
                for _ in range(32):
                    mid=.5*(lo+hi);p,d,v,f=evaluate(mid)
                    if not v.all():raise ValueError('Implicit row root crosses invalid optical branch')
                    take=fa*f>0;lo[take]=mid[take];fa[take]=f[take];hi[~take]=mid[~take]
                p,d,v,f=evaluate(.5*(lo+hi))
                if np.any(abs(f)>1e-5):raise ValueError('Implicit row root remains unresolved')
                uv[retry]=p;depth[retry]=d;good[retry]=v
        return uv,depth,good

    def centres(self, camera, xyz):
        uv,_,good=self.project(camera,xyz)
        rows=np.where(good,np.clip(uv[:,1],0.,self.context.image_height-1.),.5*(self.context.image_height-1))
        return self.context.camera25_at_rows(rows,geometry_navigation=True)[:,:3]

    def segments(self, camera, first, second, *, tolerance_px=.125, max_depth=16):
        """Sample actual world edges, retaining their valid near-plane pieces."""
        a=np.asarray(first,float).reshape(-1,3);b=np.asarray(second,float).reshape(-1,3)
        paths=[];unresolved=0;near=0
        fractions=np.array([0.,.25,.5,.75,1.])
        for level in range(max_depth+1):
            if not len(a):break
            world=a[:,None,:]+fractions[None,:,None]*(b-a)[:,None,:]
            uv,depth,good=self.project(camera,world.reshape(-1,3));uv=uv.reshape(-1,5,2);good=good.reshape(-1,5)
            active=np.any(good&(uv[:,:,0]>=0)&(uv[:,:,0]<self.image_size[0])&(uv[:,:,1]>=0)&(uv[:,:,1]<self.image_size[1]),axis=1)
            for k in np.flatnonzero(good[:,0]&good[:,-1]&~active):active[k]=_chord_intersects_sensor(uv[k,0],uv[k,-1],self.image_size)
            delta=uv[:,-1]-uv[:,0];length=np.sum(delta*delta,axis=1)
            t=np.clip(np.sum((uv[:,1:4]-uv[:,:1])*delta[:,None,:],axis=2)/np.maximum(length[:,None],1e-30),0.,1.)
            error=np.max(np.linalg.norm(uv[:,1:4]-(uv[:,:1]+t[:,:,None]*delta[:,None,:]),axis=2),axis=1)
            refine=active&(~good.all(1)|(error>tolerance_px));accept=active&good.all(1)&~refine
            if level==max_depth:
                unresolved+=int(refine.sum())
                # Keep each valid quarter chord, never bridge an invalid ray.
                accept|=refine;refine[:]=False
            for k in np.flatnonzero(accept):
                valid=good[k,:-1]&good[k,1:]
                paths.extend(zip(uv[k,:-1][valid],uv[k,1:][valid]))
                near+=int(not good[k].all())
            selected=np.flatnonzero(refine);middle=.5*(a[selected]+b[selected])
            a,b=np.vstack((a[selected],middle)),np.vstack((middle,b[selected]))
        p=np.asarray(paths,float).reshape(-1,2,2)
        return p,dict(unresolved_leaves=unresolved,near_plane_retained_leaves=near)

    def wall_polygon(self, camera, world):
        """Clip the finite 3D facet at positive row-camera depth before raster."""
        world=np.asarray(world,float);_,depth,_=self.project(camera,world)
        clipped=[]
        for i in range(len(world)):
            a,b=world[i-1],world[i];da,db=depth[i-1],depth[i];ga,gb=da>.0500001,db>.0500001
            if ga!=gb:
                lo,hi=0.,1.
                for _ in range(36):
                    m=.5*(lo+hi);_,d,_=self.project(camera,(a+m*(b-a))[None,:])
                    if (d[0]>.0500001)==ga:lo=m
                    else:hi=m
                clipped.append(a+(.5*(lo+hi))*(b-a))
            if gb:clipped.append(b)
        if len(clipped)<3:return np.empty((0,2))
        nodes=[];clipped=np.asarray(clipped)
        # Keep the ordered full polygon, including off-screen corners. Dropping
        # its outside sides would lose a facet that encloses the whole viewport.
        for a,b in zip(clipped,np.roll(clipped,-1,axis=0)):
            cache={}
            def at(t):
                if t not in cache:
                    p,_,v=self.project(camera,(a+t*(b-a))[None,:]);cache[t]=(p[0],v[0])
                return cache[t]
            def edge(lo,hi,depth):
                fractions=[lo,lo+.25*(hi-lo),.5*(lo+hi),lo+.75*(hi-lo),hi]
                samples=[at(t) for t in fractions];uv=np.array([v[0] for v in samples]);good=np.array([v[1] for v in samples])
                if not good.all():raise RowSurfaceDomainError('RS wall facet crosses an unsupported optical branch')
                active=any(0<=p[0]<1920 and 0<=p[1]<self.image_size[1] for p in uv) or _chord_intersects_sensor(uv[0],uv[-1],self.image_size)
                delta=uv[-1]-uv[0];t=np.clip((uv[1:4]-uv[0])@delta/max(delta@delta,1e-30),0.,1.)
                error=np.linalg.norm(uv[1:4]-(uv[0]+t[:,None]*delta),axis=1).max()
                if not active or error<=.125:return [uv[-1]]
                if depth>=16:raise RowSurfaceDomainError('RS wall facet curvature remains unresolved')
                mid=.5*(lo+hi);return edge(lo,mid,depth+1)+edge(mid,hi,depth+1)
            p,good=at(0.)
            if not good:raise RowSurfaceDomainError('RS near-clipped wall endpoint is unsupported')
            nodes.append(p);nodes.extend(edge(0.,1.,0))
        if len(nodes)<3:return np.empty((0,2))
        return clip_viewport_polygon(nodes,self.image_size)

    def wall_polygons(self,camera,world):
        """Retain valid parts of a fixed facet across near/optical transitions."""
        try:
            polygon=self.wall_polygon(camera,world)
            return ([polygon] if len(polygon)>=3 else []),dict(subdivided=False,unresolved_leaves=0)
        except RowSurfaceDomainError:
            pass
        world=np.asarray(world,float)
        triangles=np.array([[world[0],world[1],world[2]],[world[0],world[2],world[3]]])
        output=[];unresolved=0
        for level in range(14):
            if not len(triangles):break
            mids=.5*(triangles+np.roll(triangles,-1,axis=1));centre=triangles.mean(1)
            samples=np.concatenate((triangles,mids,centre[:,None,:]),axis=1)
            uv,_,good=self.project(camera,samples.reshape(-1,3));uv=uv.reshape(-1,7,2);good=good.reshape(-1,7)
            inside=good&(uv[:,:,0]>=0)&(uv[:,:,0]<1920)&(uv[:,:,1]>=0)&(uv[:,:,1]<self.image_size[1])
            active=inside.any(1)
            for k in np.flatnonzero(good[:,:3].all(1)&~active):
                p=uv[k,:3];active[k]=(p[:,0].min()<1920 and p[:,0].max()>=0 and p[:,1].min()<self.image_size[1] and p[:,1].max()>=0)
            # A transition may pass through the interior although every corner
            # is unsupported. Two initial world subdivisions locate that part.
            if level<2:active|=~good.all(1)
            delta=np.roll(uv[:,:3],-1,axis=1)-uv[:,:3]
            t=np.clip(np.sum((uv[:,3:6]-uv[:,:3])*delta,axis=2)/np.maximum(np.sum(delta*delta,axis=2),1e-30),0.,1.)
            error=np.max(np.linalg.norm(uv[:,3:6]-(uv[:,:3]+t[:,:,None]*delta),axis=2),axis=1)
            accept=active&good.all(1)&(error<=.125)
            output.extend(uv[accept,:3])
            refine=active&~accept
            if level==13:
                unresolved+=int(refine.sum());break
            tr=triangles[refine];mm=mids[refine]
            triangles=np.concatenate((np.stack((tr[:,0],mm[:,0],mm[:,2]),axis=1),np.stack((tr[:,1],mm[:,1],mm[:,0]),axis=1),np.stack((tr[:,2],mm[:,2],mm[:,1]),axis=1),mm))
        return output,dict(subdivided=True,unresolved_leaves=unresolved)

    def provenance(self):
        return dict(self.context.provenance(),projection='world_to_implicit_projected_sensor_row',
            offscreen_vertices='boundary-row navigation only, adaptive actual-world segments, final sensor clipping',
            screen_shear_used=False,central_Brown_domain_guard_retained=True)


def prepare_render_projection(state,q,index):
    if getattr(state,'row_readout_index',None) is None:return None
    context=prepare_frame_row_camera(state,q,index)
    if context.readout_milliseconds==0.:return None
    return RowSceneProjection(context,state.camera25(q,index))
