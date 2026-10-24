"""Independent finite-material scene evaluation from an immutable event state.

The normalized continuous superellipse perimeter is integrated independently
of the C++ quadrature. Separate paths are never connected across a gap.
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import cumulative_simpson, quad
from scipy.interpolate import PchipInterpolator
from .native_camera import NativeCamera, rotation

VERSION="native_numpy_scene_v1"


def parameter_values(model,state=None):
    blocks=model["blocks"]
    parameters=(state or {}).get("parameters",[])
    if len({p["name"] for p in parameters})!=len(parameters):raise ValueError("duplicate named state parameters")
    overrides={p["name"]:p["values"] for p in parameters}
    if state is not None and any(p["name"] not in overrides for p in blocks):
        raise ValueError("partial checkpoint cannot inherit missing physical parameters from the initializer")
    values=[np.array(overrides.get(p["name"],p["values"]),dtype=float,copy=True) for p in blocks]
    if any(v.shape!=np.asarray(p["values"]).shape for v,p in zip(values,blocks)):raise ValueError("physical state block shape mismatch")
    if any(not np.isfinite(v).all() for v in values):raise ValueError("nonfinite complete state")
    for v in values:v.setflags(write=False)
    return values


def camera_from_event(model,state=None):
    values=parameter_values(model,state);g=values[0]
    model=model.get("camera",model)
    Rseed=np.asarray(model["R0_seed"],float).reshape(3,3)
    w=g[5:8];angle=np.linalg.norm(w)
    R0=(rotation(w/angle,angle) if angle else np.eye(3))@Rseed
    n=np.asarray(model["n_seed"])+g[8]*np.asarray(model["n_tangent1"])+g[9]*np.asarray(model["n_tangent2"])
    n=n/np.linalg.norm(n);q=R0@n;bs=np.asarray(model["b_seed"])
    e=bs-q*(bs@q);e=e/np.linalg.norm(e);b=e*np.cos(g[10])+np.cross(q,e)*np.sin(g[10])
    begin=int(model.get("control_block_begin",1));nc=int(model["n_controls"])
    controls=np.stack(values[begin:begin+nc]).copy()
    dep=int(model["dependent_control"]);inds=np.asarray(model["reference_indices"],int);weights=np.asarray(model["reference_weights"])
    pivot=float(weights[inds==dep][0]);other=inds!=dep
    controls[dep,:2]=-(weights[other,None]*controls[inds[other],:2]).sum(axis=0)/pivot
    camera={"shared_head":{"R0":R0.tolist(),"n":n.tolist(),"b":b.tolist(),"P":g[11:14].tolist(),"d":g[14:17].tolist(),
            "a":g[17],"pupil_linear":g[18],"pupil_quadratic":g[19]},
       "camera_controls":{"basis":{"dt":model["dt"],"first_control_time":model["first_control_time"],"n_controls":nc},"controls":controls.tolist()},
       "optics":{"f_ref":model["f_ref"],"radius_max":model["radius_max"],"principal_point":g[20:22].tolist(),"eta":g[22],
            "principal_linear":g[23:25].tolist(),"k1":g[25:27].tolist(),"k2":g[27:29].tolist(),"p1":g[29],"p2":g[30]},
       "rolling_shutter":{"signed_readout_seconds":g[31]}}
    return NativeCamera(camera)


class NativeScene:
    def __init__(self,model,state=None):
        self.model=model;self.values=parameter_values(model,state);self.g=self.values[0]
        self.materials=model["materials"]
        g=self.g;self.width,self.rx,self.ry,self.p,self.height=g[:5]
        if not (0<self.width and 0<self.rx<.5 and 0<self.ry<self.width/2 and self.p>1 and self.height>0):
            raise ValueError("illegal shared rink dimensions/convex corner")
        # Dense numerical integration with exact endpoint speeds. This does not
        # reuse C++ 32-node quadrature or its inverse chart.
        a=np.linspace(0,np.pi/2,65537)
        speeds=self.quarter_speed(a)
        length=np.r_[0.,cumulative_simpson(speeds,x=a)]
        self.arc=float(length[-1]);self.arc_inverse=PchipInterpolator(length,a,extrapolate=False)
        # Independent adaptive integral witnesses quantify the interpolation
        # arithmetic; this is a numerical audit, not a claimed physical fit.
        checks=np.unique(np.r_[np.linspace(0,np.pi/2,129),1e-8,1e-6,np.pi/2-1e-8,np.pi/2-1e-6])
        error=[]
        interp=PchipInterpolator(a,length)
        for t in checks:
            exact,err=quad(lambda z:float(self.quarter_speed(np.array(z))),0.,float(t),epsabs=1e-13,epsrel=1e-11,limit=150)
            error.append(abs(exact-float(interp(t)))+err)
        self.arclength_numerical_max_error=float(max(error))
        self.long=1-2*self.rx;self.short=self.width-2*self.ry
        self.perimeter=2*(self.long+self.short)+4*self.arc

    def quarter(self,a):
        c=np.maximum(np.cos(a),0);s=np.maximum(np.sin(a),0)
        scale=(c**self.p+s**self.p)**(-1/self.p)
        return np.stack((self.rx*c*scale,self.ry*s*scale),axis=-1)

    def quarter_speed(self,a):
        c=np.maximum(np.cos(a),0);s=np.maximum(np.sin(a),0);den=c**self.p+s**self.p;d=den**(-1/self.p)
        v=(c**(self.p-1)*s-s**(self.p-1)*c)/den
        return np.hypot(self.rx*d*(-s+c*v),self.ry*d*(c+s*v))

    def board(self,s,h):
        s,h=np.broadcast_arrays(np.asarray(s,float),np.asarray(h,float));s=np.mod(s,self.perimeter)
        lens=np.array([self.long,self.arc,self.short,self.arc,self.long,self.arc,self.short,self.arc])
        edges=np.r_[0,np.cumsum(lens)];seg=np.searchsorted(edges[1:],s,side="left");v=s-edges[seg]
        out=np.empty(s.shape+(3,));out[...,2]=h
        for k in range(8):
            hit=seg==k
            if not np.any(hit):continue
            r=v[hit]
            if k==0:xy=np.stack((-.5+self.rx+r,np.full_like(r,-self.width/2)),axis=-1)
            elif k==2:xy=np.stack((np.full_like(r,.5),-self.width/2+self.ry+r),axis=-1)
            elif k==4:xy=np.stack((.5-self.rx-r,np.full_like(r,self.width/2)),axis=-1)
            elif k==6:xy=np.stack((np.full_like(r,-.5),self.width/2-self.ry-r),axis=-1)
            else:
                # Reverse cumulative length on bottom-right/top-left, not
                # pi/2 minus the inverse of forward length for anisotropic arcs.
                a=self.arc_inverse(np.clip(self.arc-r if k in (1,5) else r,0.,self.arc));q=self.quarter(a)
                if k==1:xy=np.stack((.5-self.rx+q[...,0],-self.width/2+self.ry-q[...,1]),axis=-1)
                elif k==3:xy=np.stack((.5-self.rx+q[...,0],self.width/2-self.ry+q[...,1]),axis=-1)
                elif k==5:xy=np.stack((-.5+self.rx-q[...,0],self.width/2-self.ry+q[...,1]),axis=-1)
                else:xy=np.stack((-.5+self.rx-q[...,0],-self.width/2+self.ry-q[...,1]),axis=-1)
            out[hit,:2]=xy
        return out

    def path(self,material,path=0):
        if not (np.isfinite(material) and np.isfinite(path) and int(material)==material and int(path)==path):raise ValueError("noninteger material/branch identity")
        if int(material)<0 or int(material)>=len(self.materials):raise IndexError("illegal material identity")
        paths=self.materials[int(material)]["paths"]
        if int(path)<0 or int(path)>=len(paths):raise IndexError("illegal material branch")
        return paths[int(path)]

    def world(self,material,path,station):
        p=self.path(material,path);s=np.asarray(station,float)
        if not np.isfinite(s).all() or np.any((s<0)|(s>1)):raise ValueError("illegal finite station")
        shared=int(p.get("shared_station_block",-1))
        if shared>=0:
            if shared>=len(self.values) or self.values[shared].shape!=(1,) or not p.get("physical_point",False) or p["kind"] not in (0,"ice_segment"):
                raise ValueError("shared finite position needs one physical-point scalar block")
            s=np.full_like(s,self.values[shared][0])
            if not np.isfinite(s).all() or np.any((s<0)|(s>1)):raise ValueError("illegal shared finite station")
        kind=p["kind"];kind={"ice_segment":0,"ice_circle":1,"board_path":2,"board_boundary":3}.get(kind,kind)
        block=int(p.get("geometry_block",-1));q=None if block<0 else self.values[block]
        if kind==0:
            return np.stack((q[0]*(1-s)+q[2]*s,q[1]*(1-s)+q[3]*s,np.zeros_like(s)),axis=-1)
        if kind==1:
            angle=p.get("start_angle",0)+s*(p.get("end_angle",2*np.pi)-p.get("start_angle",0))
            return np.stack((q[0]+q[int(p.get("radius_index",2))]*np.cos(angle),q[1]+q[int(p.get("radius_index",2))]*np.sin(angle),np.zeros_like(s)),axis=-1)
        if kind==3:
            segment=int(p.get("boundary_segment",-1))
            if segment>=0:
                lengths=np.array([self.long,self.arc,self.short,self.arc,self.long,self.arc,self.short,self.arc])
                coordinate=float(lengths[:segment].sum())+s*lengths[segment]
            else:coordinate=s*self.perimeter
            return self.board(coordinate,self.height*p.get("fixed_height_fraction",0))
        if kind!=2:raise ValueError("unknown physical material primitive")
        points=np.asarray(p["points"],float)
        if len(points)==1:uv=np.broadcast_to(points[0],s.shape+(2,)).copy()
        else:
            x=s*(len(points)-1);i=np.minimum(np.floor(x).astype(int),len(points)-2);u=x-i
            uv=points[i]*(1-u[...,None])+points[i+1]*u[...,None]
        if q is not None:uv=uv@np.array([[q[0],q[3]],[q[1],q[4]]])+q[[2,5]]
        ad=self.values[int(p["advertisement_block"])];return self.board(ad[0]+ad[1]*uv[...,0],ad[2]+(ad[3]-ad[2])*uv[...,1])

    def world_rows(self,material,path,station):
        material=np.asarray(material);path=np.asarray(path);station=np.asarray(station)
        out=np.empty((len(station),3));legal=np.ones(len(station),bool);reason=np.full(len(station),"",dtype="U64")
        # RS/GN repeatedly evaluates the same material/branch ownership at new
        # stations. Cache only its grouping, never world points or derivatives.
        # Exact array comparison also catches mutation of reused input buffers.
        cached=getattr(self,"_world_row_groups",None)
        if cached is not None and cached[0].dtype==material.dtype and cached[1].dtype==path.dtype and np.array_equal(cached[0],material) and np.array_equal(cached[1],path):
            groups=cached[2]
        else:
            groups=tuple((m,branch,np.flatnonzero((material==m)&(path==branch)))for m,branch in np.unique(np.c_[material,path],axis=0))
            self._world_row_groups=(material.copy(),path.copy(),groups)
        for m,branch,rows in groups:
            try:out[rows]=self.world(m,branch,station[rows])
            except (ValueError,IndexError,KeyError):out[rows]=np.nan;legal[rows]=False;reason[rows]="finite_material_or_branch_domain"
        return out,legal,reason


    def containment(self,xy):
        xy=np.asarray(xy,float)
        u=np.maximum((np.abs(xy[...,0])-(.5-self.rx))/self.rx,0.)
        v=np.maximum((np.abs(xy[...,1])-(self.width/2-self.ry))/self.ry,0.)
        return u**self.p+v**self.p-1.

    def circle_containment(self,center,radius,start=0.,end=2*np.pi,tolerance=1e-8):
        """Interval enclosures of all circle points, without sampled proof."""
        center=np.asarray(center,float);queue=[(float(start),float(end),0)];upper_max=-np.inf;worst_lower=-np.inf;cells=0
        while queue:
            a,b,depth=queue.pop();cells+=1
            candidates=np.r_[a,b,np.arange(np.ceil(a/(np.pi/2)),np.floor(b/(np.pi/2))+1)*np.pi/2]
            xy=center+radius*np.c_[np.cos(candidates),np.sin(candidates)]
            box=np.max(np.abs(xy),axis=0)
            upper=float(self.containment(box))+1e-12
            mid=(a+b)/2;lower=float(self.containment(center+radius*np.array([np.cos(mid),np.sin(mid)])))
            worst_lower=max(worst_lower,lower)
            if lower>tolerance:return {'certified':False,'violation_lower':lower,'upper':upper,'cells':cells}
            if upper<=tolerance:upper_max=max(upper_max,upper);continue
            if depth>=48 or cells>1000000:return {'certified':False,'violation_lower':worst_lower,'upper':upper,'cells':cells,'reason':'containment_interval_unresolved'}
            queue.append((a,mid,depth+1));queue.append((mid,b,depth+1))
        return {'certified':True,'violation_lower':worst_lower,'upper':upper_max,'cells':cells}

    def check_physics(self,tolerance=1e-8):
        violations=[];checks=[];checked=set()
        for definition,value in zip(self.model['blocks'],self.values):
            if 'lower' in definition and np.any(value<np.asarray(definition['lower'])-tolerance):violations.append(definition['name']+'/lower_bound')
            if 'upper' in definition and np.any(value>np.asarray(definition['upper'])+tolerance):violations.append(definition['name']+'/upper_bound')
            if definition['name'].startswith('ad/'):
                if not (0<value[1]<=self.perimeter+tolerance and value[2]>=-tolerance and value[3]>value[2] and value[3]<=self.height+tolerance):violations.append(definition['name']+'/finite_board_installation')
            if definition['name'].startswith('ice/circle/'):
                if np.any(value[2:]<=0) or np.any(np.diff(value[2:])<0):violations.append(definition['name']+'/positive_nested_radii')
        for mi,m in enumerate(self.materials):
            for pi,path in enumerate(m['paths']):
                kind=path['kind'];key=(kind,path.get('geometry_block'),path.get('radius_index',2),path.get('advertisement_block'))
                if kind==0 and key not in checked:
                    checked.add(key);q=self.values[int(path['geometry_block'])]
                    margin=float(np.max(self.containment(q.reshape(2,2))))
                    if margin>tolerance:violations.append(m['label']+'/ice_segment_containment')
                    checks.append({'id':m['label']+'/ice_segment','max_inequality':margin,'certified_by':'convex rink and both finite endpoints'})
                elif kind==1 and key not in checked:
                    checked.add(key);q=self.values[int(path['geometry_block'])];r=q[int(path.get('radius_index',2))]
                    result=self.circle_containment(q[:2],r,path.get('start_angle',0.),path.get('end_angle',2*np.pi),tolerance)
                    result['id']=m['label']+'/whole_circle_containment';checks.append(result)
                    if not result['certified']:violations.append(result['id'])
                elif kind==2:
                    points=np.asarray(path['points'],float);block=int(path.get('geometry_block',-1))
                    if block>=0:
                        q=self.values[block];det=q[0]*q[4]-q[1]*q[3]
                        if det<=1e-10:violations.append(m['label']+'/print_affine_orientation_or_degeneracy')
                        points=points@np.array([[q[0],q[3]],[q[1],q[4]]])+q[[2,5]]
                    ad=self.values[int(path['advertisement_block'])]
                    heights=ad[2]+(ad[3]-ad[2])*points[:,1]
                    # Every polyline segment is affine in this shared surface
                    # chart, so endpoint checks cover its complete support.
                    if np.min(heights)<-tolerance or np.max(heights)>self.height+tolerance:violations.append(m['label']+'/print_off_board_surface')
                    # A normalized internal graphic chart is not an observed
                    # outer panel boundary (notably Ad5). Enforce a canvas
                    # only when the model carries an independently sourced
                    # physical interval, never an assumed [0,1] box.
                    if 'physical_canvas_bounds' in path:
                        lo,hi=np.asarray(path['physical_canvas_bounds'])
                        if not path.get('physical_canvas_provenance'):raise ValueError('canvas bounds lack physical source')
                        if np.any(points<lo-tolerance) or np.any(points>hi+tolerance):violations.append(m['label']+'/print_outside_physical_canvas')
        return {'passed':not violations,'violations':violations,'checks':checks,'all_material_paths_checked':sum(len(m['paths']) for m in self.materials)}
