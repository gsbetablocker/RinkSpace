"""Reusable three-dimensional rink-wireframe opening; no video/audio I/O.

Scene XYZ, clipped ice-grid cells and continuous board surface are copied from
one frozen scene. Advertisements and painted rink markings are never drawn.
Presentation cameras are animated to the exact supplied source camera25.
Projection delegates to the existing Brown14/skew implementation. The renderer
uses triangle depth buffers and clips lines against them, including real board
height; it does not flatten boards through an ice-plane homography.
"""
from __future__ import annotations

import math
import cv2
import numpy as np
from scipy.spatial.transform import Rotation,Slerp

try:
    from .projection import project_fast
except ImportError:
    from projection import project_fast


def _ease(value):
    u=float(np.clip(value,0.,1.))
    return u*u*u*(u*(u*6.-15.)+10.)


def _continuous_progress(elapsed,start,end,ramp_in=.6,ramp_out=1.):
    """C2 position progress: smooth velocity ramps and a moving central span.

    Velocity uses a cubic smoothstep during each ramp. Its integral has zero
    first/second derivatives at the outer endpoints, while adjoining constant
    velocity with matching acceleration at the inner ramp boundaries.
    """
    duration=end-start
    t=float(np.clip(elapsed-start,0.,duration))
    if t<=0:return 0.
    if t>=duration:return 1.
    scale=min(1.,duration/(ramp_in+ramp_out))
    ramp_in*=scale;ramp_out*=scale
    area=duration-.5*(ramp_in+ramp_out)
    if t<ramp_in:
        q=t/ramp_in
        return ramp_in*(q*q*q-.5*q*q*q*q)/area
    if t>duration-ramp_out:
        q=(duration-t)/ramp_out
        return 1.-ramp_out*(q*q*q-.5*q*q*q*q)/area
    return (.5*ramp_in+t-ramp_in)/area


def _look_rotation(center,target):
    forward=np.asarray(target,float)-np.asarray(center,float)
    forward/=np.linalg.norm(forward)
    right=np.cross(forward,[0.,0.,1.])
    if np.linalg.norm(right)<1e-8:
        right=np.array([1.,0.,0.])
    right/=np.linalg.norm(right)
    down=np.cross(forward,right)
    return Rotation.from_matrix(np.stack((right,down,forward)))


def _segments_from_scene(scene,board_subdivisions=1,horizontal_levels=()):
    ice=[]
    seen=set()
    for cell in scene['ice_grid']['cells']:
        polygon=np.asarray(cell['polygon_xyz'],float)
        for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
            if np.linalg.norm(a-b)<1e-9:
                continue
            ka,kb=tuple(np.round(a,8)),tuple(np.round(b,8))
            key=tuple(sorted((ka,kb)))
            if key not in seen:
                seen.add(key)
                ice.append((a.copy(),b.copy()))
    board=scene['board_surface']
    bottom=np.asarray(board['bottom_edge_xyz'],float)
    top=np.asarray(board['top_edge_xyz'],float)
    if bottom.shape!=top.shape or bottom.ndim!=2 or bottom.shape[1]!=3:
        raise ValueError('Board bottom and top XYZ must align')
    if not board.get('closed',False):
        raise ValueError('Opening requires the frozen complete closed board course')
    rim=[]
    for edge in (bottom,top):
        rim.extend(zip(edge,np.roll(edge,-1,axis=0)))
    # Vertical wire spacing follows the scene grid, rather than exposing every
    # fine curve tessellation triangle as a tall decorative fence.
    spacing=float(np.mean(np.asarray(scene['ice_grid']['spacing_xy'],float)))/int(board_subdivisions)
    for level in horizontal_levels:
        edge=bottom*(1.-float(level))+top*float(level)
        rim.extend(zip(edge,np.roll(edge,-1,axis=0)))
    lengths=np.linalg.norm(np.roll(bottom,-1,axis=0)-bottom,axis=1)
    stations=np.r_[0.,np.cumsum(lengths)]
    for station in np.arange(0.,stations[-1],spacing):
        i=min(len(bottom)-1,int(np.searchsorted(stations,station,side='right')-1))
        u=(station-stations[i])/max(lengths[i],1e-12)
        rim.append((bottom[i]*(1-u)+bottom[(i+1)%len(bottom)]*u,
                    top[i]*(1-u)+top[(i+1)%len(top)]*u))
    return np.asarray(ice,float),np.asarray(rim,float)


def _clip_near_polygon(points,depths,near):
    result=[]
    for i,a in enumerate(points):
        j=(i+1)%len(points)
        b=points[j]
        da,db=float(depths[i]),float(depths[j])
        ain,bin=da>=near,db>=near
        if ain:
            result.append(a)
        if ain!=bin:
            u=(near-da)/(db-da)
            result.append(a+(b-a)*u)
    return np.asarray(result,float)


def _clip_screen(a,b,width,height):
    """Liang-Barsky clipping, retaining the screen-segment parameter range."""
    delta=b-a
    lo,hi=0.,1.
    for p,q in ((-delta[0],a[0]),(delta[0],width-1.-a[0]),
                (-delta[1],a[1]),(delta[1],height-1.-a[1])):
        if abs(p)<1e-12:
            if q<0:
                return None
        else:
            ratio=q/p
            if p<0:
                lo=max(lo,ratio)
            else:
                hi=min(hi,ratio)
            if lo>hi:
                return None
    return lo,hi


class IntroTransform:
    """Three-dimensional opening with compatible and staged presentation modes.

    profile='legacy' preserves the previous5.2-second orbit/dolly crossfade.
    profile='orbit_overlay' adds a full-rink wide orbit before approach and a
    clearly visible source-frame/grid overlap, lasting8.2seconds by default.
    frame(t,first_native_bgr,first_foreground=person_only_mask) returns BGR.
    """
    def __init__(self,scene,first_camera25,output_wh=(1920,1080),*,
                 overview_seconds=1.,travel_seconds=3.5,fade_seconds=.7,
                 fps=50,orbit_degrees=None,elevation_degrees=50.,
                 profile='legacy',orbit_seconds=2.4,approach_seconds=3.2,
                 reveal_seconds=.5,overlay_hold_seconds=.8,grid_fade_seconds=.3,
                 rotation_start_seconds=.45,approach_lead_seconds=.30,
                 motion_ramp_in_seconds=.6,motion_ramp_out_seconds=1.):
        self.profile=str(profile)
        if self.profile not in ('legacy','orbit_overlay','continuous_orbit_overlay'):
            raise ValueError('Unknown opening profile')
        self.first_camera25=np.asarray(first_camera25,float).copy()
        if self.first_camera25.shape!=(25,) or not np.isfinite(self.first_camera25).all():
            raise ValueError('first_camera25 must contain25 finite source-camera values')
        self.output_wh=tuple(map(int,output_wh))
        if len(self.output_wh)!=2 or min(self.output_wh)<2:
            raise ValueError('output_wh must be positive width,height')
        self.overview_seconds=float(overview_seconds)
        self.travel_seconds=float(travel_seconds)
        self.fade_seconds=float(fade_seconds)
        self.orbit_seconds=float(orbit_seconds) if self.profile!='legacy' else 0.
        self.approach_seconds=(float(approach_seconds) if self.profile!='legacy'
                               else self.travel_seconds)
        self.reveal_seconds=float(reveal_seconds) if self.profile!='legacy' else self.fade_seconds
        self.overlay_hold_seconds=float(overlay_hold_seconds) if self.profile!='legacy' else 0.
        self.grid_fade_seconds=float(grid_fade_seconds) if self.profile!='legacy' else 0.
        self.fps=int(fps)
        durations=[self.overview_seconds,self.travel_seconds,self.fade_seconds,self.orbit_seconds,
                   self.approach_seconds,self.reveal_seconds,self.overlay_hold_seconds,self.grid_fade_seconds]
        if (not np.isfinite(durations).all() or self.overview_seconds<0
                or self.travel_seconds<=0 or self.fade_seconds<=0 or self.approach_seconds<=0
                or self.reveal_seconds<=0 or self.overlay_hold_seconds<0 or self.fps<=0
                or (self.profile=='orbit_overlay' and (self.orbit_seconds<=0 or self.grid_fade_seconds<=0))):
            raise ValueError('Opening durations and fps are invalid')
        self.approach_start_seconds=round(self.overview_seconds+self.orbit_seconds,12)
        self.target_reached_seconds=round(self.approach_start_seconds+self.approach_seconds,12)
        self.overlay_start_seconds=round(self.target_reached_seconds+self.reveal_seconds,12)
        self.overlay_end_seconds=round(self.overlay_start_seconds+self.overlay_hold_seconds,12)
        self.total_seconds=round(self.overlay_end_seconds+self.grid_fade_seconds,12)
        self.total_frames=round(self.total_seconds*self.fps)
        self.rotation_start_seconds=float(rotation_start_seconds)
        self.approach_lead_seconds=float(approach_lead_seconds)
        self.motion_ramp_in_seconds=float(motion_ramp_in_seconds)
        self.motion_ramp_out_seconds=float(motion_ramp_out_seconds)
        if self.profile=='continuous_orbit_overlay':
            self.approach_start_seconds=round(self.rotation_start_seconds+self.approach_lead_seconds,12)
            self.approach_seconds=self.target_reached_seconds-self.approach_start_seconds
            schedule=[self.rotation_start_seconds,self.approach_lead_seconds,
                      self.motion_ramp_in_seconds,self.motion_ramp_out_seconds]
            if (not np.isfinite(schedule).all() or self.rotation_start_seconds<0.
                    or self.approach_lead_seconds<0. or self.approach_seconds<=0.
                    or min(self.motion_ramp_in_seconds,self.motion_ramp_out_seconds)<=0.):
                raise ValueError('Continuous-camera schedule is invalid')
        self.ignite_seconds=.45
        default_degrees=70. if self.profile!='legacy' else 30.
        self.orbit_degrees=float(default_degrees if orbit_degrees is None else orbit_degrees)
        self.elevation_degrees=float(elevation_degrees)
        if not 5.<self.elevation_degrees<89. or not abs(self.orbit_degrees)<120.:
            raise ValueError('Opening uses a high oblique view and a short orbit')
        boundary=np.asarray(scene['ice_boundary']['points_xyz'],float)
        top=np.asarray(scene['board_surface']['top_edge_xyz'],float)
        self._bounds=np.r_[boundary[:,:2].min(axis=0),boundary[:,:2].max(axis=0)]
        self._board_height=float(top[:,2].max())
        grid_options=({"board_subdivisions":2,"horizontal_levels":(.25,.5,.75)}
                      if self.profile!="legacy" else {})
        self._ice_segments,self._board_segments=_segments_from_scene(scene,**grid_options)
        self._meshes=[]
        for mesh in scene['meshes']:
            if mesh.get('role') in ('ice_surface','boards_inner_surface'):
                xyz=np.asarray(mesh['positions'],float).copy()
                triangles=np.asarray(mesh['triangles'],np.int32).copy()
                label=1 if mesh['role']=='ice_surface' else 2
                self._meshes.append((xyz,triangles,label))
        if {m[2] for m in self._meshes}!={1,2}:
            raise ValueError('Frozen ice and inner-board meshes are both required')
        self._scene_model_id=str(scene.get('model_id',''))
        self._near=.10
        target=self.first_camera25
        self._end_rotation=Rotation.from_rotvec(target[3:6])
        forward=self._end_rotation.as_matrix()[2]
        center=np.r_[(self._bounds[:2]+self._bounds[2:])*.5,self._board_height*.2]
        self._anchor0=center
        self._distance1=float(np.linalg.norm(target[:3]-center))
        self._anchor1=target[:3]+forward*self._distance1
        # Position polar angle matters: source close-ups do not aim at rink center.
        offset=target[:2]-center[:2]
        polar=math.atan2(offset[1],offset[0])
        if self.profile!='legacy':
            # Orbit clockwise across the long-side view, finishing on the side
            # closest to the source-camera heading before the tight approach.
            self._azimuth_start=polar+math.radians(self.orbit_degrees)*.5
            self._azimuth_end=polar-math.radians(self.orbit_degrees)*.5
        else:
            self._azimuth_start=self._azimuth_end=polar-math.radians(self.orbit_degrees)
        w,h=self.output_wh
        self._wide_template=target.copy()
        self._wide_template[6:8]=math.log(w*.86)
        self._wide_template[8:11]=[(w-1)*.5,(h-1)*.5,0.]
        vertices=np.concatenate([mesh[0] for mesh in self._meshes])
        focal=np.exp(self._wide_template[6:8]);limits=np.array([(w-1)*.40,(h-1)*.40])
        distances=[]
        angles=np.linspace(self._azimuth_start,self._azimuth_end,101 if self.orbit_seconds else 1)
        for angle in angles:
            direction=self._wide_direction(angle)
            rotation=_look_rotation(center+direction,center)
            relative=(vertices-center)@rotation.as_matrix().T
            required=np.abs(relative[:,:2])*focal/limits-relative[:,2:3]
            distances.append(float(required.max()))
        # One fixed radius for the entire wide orbit: no premature zoom/crop.
        self._distance0=max(distances)+.02*np.ptp(boundary[:,0])
        self.initial_camera25=self._wide_camera(self._azimuth_start)
        self._approach_camera25=self._wide_camera(self._azimuth_end)
        start_rotation=Rotation.from_rotvec(self._approach_camera25[3:6])
        self._slerp=Slerp([0.,1.],Rotation.from_quat([
            start_rotation.as_quat(),self._end_rotation.as_quat()]))
        self._turn_radians=float((self._end_rotation*start_rotation.inv()).magnitude())
        yy,xx=np.mgrid[0:h,0:w].astype(np.float32)
        glow=np.exp(-(((xx/(w-1)-.5)/.7)**2+((yy/(h-1)-.48)/.8)**2)*2.)
        self._wireframe_cache_key=None
        self._wireframe_cache=None
        self._background=np.empty((h,w,3),np.float32)
        for channel,(low,high) in enumerate(((10.,25.),(7.,16.),(5.,10.))):
            self._background[...,channel]=low+(high-low)*glow
        self._face_background_cache=None
        self._line_premult_cache=None
        self._line_alpha_cache=None

    def _wide_direction(self,azimuth):
        elevation=math.radians(self.elevation_degrees)
        return np.array([math.cos(azimuth)*math.cos(elevation),
                         math.sin(azimuth)*math.cos(elevation),math.sin(elevation)])

    def _wide_camera(self,azimuth):
        direction=self._wide_direction(azimuth)
        camera=self._wide_template.copy()
        camera[:3]=self._anchor0+direction*self._distance0
        camera[3:6]=_look_rotation(camera[:3],self._anchor0).as_rotvec()
        return camera

    def camera_at(self,elapsed_seconds):
        elapsed=float(elapsed_seconds)
        if not np.isfinite(elapsed):
            raise ValueError('elapsed_seconds must be finite')
        if self.profile=='continuous_orbit_overlay':
            if elapsed<=self.rotation_start_seconds:
                return self.initial_camera25.copy()
            if elapsed>=self.target_reached_seconds:
                return self.first_camera25.copy()
            yaw=_continuous_progress(elapsed,self.rotation_start_seconds,self.target_reached_seconds,
                                     self.motion_ramp_in_seconds,self.motion_ramp_out_seconds)
            approach=_continuous_progress(elapsed,self.approach_start_seconds,self.target_reached_seconds,
                                          self.motion_ramp_in_seconds,self.motion_ramp_out_seconds)
            azimuth=self._azimuth_start*(1.-yaw)+self._azimuth_end*yaw
            wide=self._wide_camera(azimuth)
            wide_rotation=Rotation.from_rotvec(wide[3:6])
            blend=Slerp([0.,1.],Rotation.from_quat([wide_rotation.as_quat(),
                                                   self._end_rotation.as_quat()]))
            rotation=blend(approach)
            anchor=self._anchor0*(1.-approach)+self._anchor1*approach
            distance=np.exp((1.-approach)*math.log(self._distance0)+approach*math.log(self._distance1))
            camera=self._wide_template*(1.-approach)+self.first_camera25*approach
            camera[:3]=anchor-rotation.as_matrix()[2]*distance
            camera[3:6]=rotation.as_rotvec()
            return camera
        if elapsed<=self.overview_seconds:
            return self.initial_camera25.copy()
        if self.orbit_seconds and elapsed<self.approach_start_seconds:
            q=float(np.clip((elapsed-self.overview_seconds)/self.orbit_seconds,0.,1.))
            q=q*q*(3.-2.*q)
            return self._wide_camera(self._azimuth_start*(1.-q)+self._azimuth_end*q)
        if elapsed>=self.target_reached_seconds:
            return self.first_camera25.copy()
        u=_ease((elapsed-self.approach_start_seconds)/self.approach_seconds)
        rotation=self._slerp(u)
        forward=rotation.as_matrix()[2]
        anchor=self._anchor0*(1.-u)+self._anchor1*u
        distance=np.exp((1.-u)*math.log(self._distance0)+u*math.log(self._distance1))
        camera=self._approach_camera25*(1.-u)+self.first_camera25*u
        camera[:3]=anchor-forward*distance
        camera[3:6]=rotation.as_rotvec()
        return camera

    @staticmethod
    def _raster_triangle(uv,depth,label,zbuffer,labels):
        if not np.isfinite(uv).all() or np.min(depth)<=0:
            return
        height,width=zbuffer.shape
        if (uv[:,0].max()<0 or uv[:,1].max()<0
                or uv[:,0].min()>width-1 or uv[:,1].min()>height-1):
            return
        # Clip projected polygons BEFORE raster work. Many long, thin triangles
        # have a full-screen bounding box while touching only a few pixels.
        polygon=np.column_stack((uv,1./depth))
        for axis,bound,direction in ((0,0.,1.),(0,width-1.,-1.),
                                     (1,0.,1.),(1,height-1.,-1.)):
            clipped=[]
            for j,a in enumerate(polygon):
                b=polygon[(j+1)%len(polygon)]
                da=direction*(a[axis]-bound);db=direction*(b[axis]-bound)
                ain,bin=da>=0.,db>=0.
                if ain:clipped.append(a)
                if ain!=bin:
                    u=da/(da-db)
                    clipped.append(a+(b-a)*u)
            if len(clipped)<3:return
            polygon=np.asarray(clipped,float)
        low=np.floor(polygon[:,:2].min(axis=0)).astype(int)
        high=np.ceil(polygon[:,:2].max(axis=0)).astype(int)
        x0,y0=max(0,int(low[0])),max(0,int(low[1]))
        x1,y1=min(width-1,int(high[0])),min(height-1,int(high[1]))
        if x1<x0 or y1<y0:return
        # Reciprocal camera depth is affine in screen coordinates. Solve once,
        # then evaluate only covered pixels rather than every bbox pixel.
        matrix=np.column_stack((uv,np.ones(3)))
        try:coefficient=np.linalg.solve(matrix,1./depth)
        except np.linalg.LinAlgError:return
        coverage=np.zeros((y1-y0+1,x1-x0+1),np.uint8)
        points=np.rint((polygon[:,:2]-[x0,y0])*16.).astype(np.int32)
        cv2.fillConvexPoly(coverage,points,1,lineType=cv2.LINE_8,shift=4)
        yy,xx=np.nonzero(coverage)
        if not len(xx):return
        yy+=y0;xx+=x0
        reciprocal=coefficient[0]*xx+coefficient[1]*yy+coefficient[2]
        good=reciprocal>0.
        yy,xx,reciprocal=yy[good],xx[good],reciprocal[good]
        z=(1./reciprocal).astype(np.float32)
        update=z<zbuffer[yy,xx]
        yy,xx,z=yy[update],xx[update],z[update]
        zbuffer[yy,xx]=z
        labels[yy,xx]=label

    def _surfaces(self,camera):
        w,h=self.output_wh
        zbuffer=np.full((h,w),np.inf,np.float32)
        labels=np.zeros((h,w),np.uint8)
        rotation=Rotation.from_rotvec(camera[3:6]).as_matrix()
        for vertices,triangles,label in self._meshes:
            uv,depth=project_fast(camera,vertices)
            for indices in triangles:
                d=depth[indices]
                if np.max(d)<self._near:
                    continue
                if np.min(d)>=self._near:
                    self._raster_triangle(uv[indices],d,label,zbuffer,labels)
                else:
                    polygon=_clip_near_polygon(vertices[indices],d,self._near)
                    if len(polygon)<3:
                        continue
                    p,z=project_fast(camera,polygon)
                    for j in range(1,len(polygon)-1):
                        fan=np.array([0,j,j+1])
                        self._raster_triangle(p[fan],z[fan],label,zbuffer,labels)
        return zbuffer,labels

    def _line_mask(self,camera,segments,zbuffer):
        w,h=self.output_wh
        mask=np.zeros((h,w),np.uint8)
        # Endpoints are actual 3D points; near-plane crossings are clipped in3D.
        points=segments.copy()
        rotation=Rotation.from_rotvec(camera[3:6]).as_matrix()
        depths=(points-camera[:3])@rotation[2]
        keep=np.max(depths,axis=1)>=self._near
        points,depths=points[keep],depths[keep]
        for endpoint in (0,1):
            bad=depths[:,endpoint]<self._near
            if np.any(bad):
                other=1-endpoint
                u=(self._near-depths[bad,endpoint])/(depths[bad,other]-depths[bad,endpoint])
                points[bad,endpoint]+=((points[bad,other]-points[bad,endpoint])*u[:,None])
                depths[bad,endpoint]=self._near
        projected,z=project_fast(camera,points.reshape(-1,3))
        projected=projected.reshape(-1,2,2);z=z.reshape(-1,2)
        for uv,d in zip(projected,z):
            if not np.isfinite(uv).all():
                continue
            clipped=_clip_screen(uv[0],uv[1],w,h)
            if clipped is None:
                continue
            t0,t1=clipped
            a=uv[0]+(uv[1]-uv[0])*t0;b=uv[0]+(uv[1]-uv[0])*t1
            count=max(2,int(math.ceil(np.max(np.abs(b-a))))+1)
            fraction=np.linspace(t0,t1,count)
            screen=uv[0]+fraction[:,None]*(uv[1]-uv[0])
            pixels=np.rint(screen).astype(np.int32)
            pixels[:,0]=np.clip(pixels[:,0],0,w-1)
            pixels[:,1]=np.clip(pixels[:,1],0,h-1)
            line_z=1./((1.-fraction)/d[0]+fraction/d[1])
            nearest=zbuffer[pixels[:,1],pixels[:,0]]
            visible=line_z<=nearest+.01+.0005*line_z
            changes=np.diff(np.r_[False,visible,False].astype(np.int8))
            for first,last in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)):
                coords=np.rint(screen[first:last]*16.).astype(np.int32)
                if len(coords)==1:
                    cv2.circle(mask,tuple(coords[0]),5,255,-1,cv2.LINE_AA,shift=4)
                else:
                    cv2.polylines(mask,[coords],False,255,
                                  2 if self.profile!="legacy" else 1,cv2.LINE_AA,shift=4)
        return mask

    def wireframe(self,elapsed_seconds):
        camera=self.camera_at(elapsed_seconds)
        key=camera.tobytes()
        if key==self._wireframe_cache_key:
            return self._wireframe_cache.copy()
        depth,labels=self._surfaces(camera)
        canvas=self._background.copy()
        for label,colour,alpha in ((1,np.array([101.,74.,37.]),.085),
                                   (2,np.array([126.,91.,48.]),.12)):
            visible=labels==label
            canvas[visible]=canvas[visible]*(1.-alpha)+colour*alpha
        face_background=canvas.copy()
        line_premult=np.zeros_like(canvas)
        line_alpha=np.zeros(canvas.shape[:2],np.float32)
        ice_colour=np.array([255.,198.,62.]) if self.profile!='legacy' else np.array([244.,154.,62.])
        board_colour=np.array([255.,215.,110.]) if self.profile!='legacy' else np.array([255.,194.,105.])
        for segments,colour,opacity in ((self._ice_segments,ice_colour,.66),
                                       (self._board_segments,board_colour,.85)):
            line=self._line_mask(camera,segments,depth)
            halo=cv2.GaussianBlur(line,(0,0),1.5).astype(np.float32)/255.*.16
            canvas+=halo[...,None]*colour
            line_premult+=halo[...,None]*colour
            alpha=line.astype(np.float32)/255.*opacity
            canvas*=1.-alpha[...,None]
            canvas+=alpha[...,None]*colour
            line_premult*=1.-alpha[...,None]
            line_premult+=alpha[...,None]*colour
            line_alpha=line_alpha*(1.-alpha)+alpha
        rendered=np.clip(canvas+.5,0,255).astype(np.uint8)
        self._wireframe_cache_key=key
        self._wireframe_cache=rendered
        self._face_background_cache=face_background
        self._line_premult_cache=line_premult
        self._line_alpha_cache=line_alpha
        return rendered.copy()

    def _person_mask(self,value):
        mask=np.asarray(value)
        w,h=self.output_wh
        if mask.shape==(h,w,1):mask=mask[...,0]
        if mask.shape!=(h,w):
            raise ValueError('first_foreground must be an output-sized person-only mask')
        divisor=255. if mask.dtype==np.uint8 else 1.
        mask=mask.astype(np.float32)/divisor
        if not np.isfinite(mask).all() or mask.min()<0 or mask.max()>1:
            raise ValueError('Person mask must lie in0..1 or uint8_0..255')
        return mask

    def frame(self,elapsed_seconds,first_native_bgr,*,first_foreground=None):
        """Draw one opening frame; foreground means the ATHLETE ONLY.

        Before source reveal, a supplied person mask has no effect. Once the
        real frame appears it protects the person from the independent cyan
        line layer. Do not pass broadcaster-title/timer protection rectangles.
        In the overlap plateau the source is fully bright, with no dark-base
        crossfade; only the cyan line layer remains above it.
        """
        native=np.asarray(first_native_bgr)
        w,h=self.output_wh
        if native.shape!=(h,w,3) or native.dtype!=np.uint8:
            raise ValueError('first_native_bgr must be output-sized uint8 BGR')
        elapsed=float(elapsed_seconds)
        if not np.isfinite(elapsed):
            raise ValueError('elapsed_seconds must be finite')
        if elapsed>=self.total_seconds:
            return native.copy()
        mesh=self.wireframe(elapsed)
        if elapsed<self.ignite_seconds:
            reveal=float(np.clip(elapsed/self.ignite_seconds,0.,1.))
            reveal=reveal*reveal*(3.-2.*reveal)
            mesh=np.clip(self._background*(1.-reveal)
                         +mesh.astype(np.float32)*reveal+.5,0,255).astype(np.uint8)
        if self.profile=='legacy':
            fade=_ease((elapsed-self.target_reached_seconds)/self.fade_seconds)
            if fade<=0:return mesh
            return np.clip(mesh.astype(np.float32)*(1.-fade)+native.astype(np.float32)*fade+.5,0,255).astype(np.uint8)
        if elapsed<self.target_reached_seconds:
            return mesh
        reveal=_ease((elapsed-self.target_reached_seconds)/self.reveal_seconds)
        grid_strength=1.-_ease((elapsed-self.overlay_end_seconds)/self.grid_fade_seconds)
        base=self._face_background_cache*(1.-reveal)+native.astype(np.float32)*reveal
        if first_foreground is not None:
            visibility=grid_strength*(1.-self._person_mask(first_foreground)*reveal)
        else:
            visibility=grid_strength
        alpha=self._line_alpha_cache*visibility
        premult=self._line_premult_cache*(visibility[...,None] if isinstance(visibility,np.ndarray) else visibility)
        result=base*(1.-alpha[...,None])+premult
        return np.clip(result+.5,0,255).astype(np.uint8)

    def parameters(self):
        parameters={'schema':'rinkspace.demo-intro.v2','profile':self.profile,'model_id':self._scene_model_id,
                'output_wh':list(self.output_wh),'fps':self.fps,
                'overview_seconds':self.overview_seconds,'travel_seconds':self.travel_seconds,
                'fade_seconds':self.fade_seconds,'ignite_seconds':self.ignite_seconds,
                'wide_orbit_seconds':self.orbit_seconds,'approach_seconds':self.approach_seconds,
                'source_reveal_seconds':self.reveal_seconds,
                'clear_overlay_hold_seconds':self.overlay_hold_seconds,
                'grid_only_fade_seconds':self.grid_fade_seconds,
                'approach_start_seconds':self.approach_start_seconds,
                'target_camera_reached_seconds':self.target_reached_seconds,
                'full_brightness_overlay_start_seconds':self.overlay_start_seconds,
                'full_brightness_overlay_end_seconds':self.overlay_end_seconds,
                'total_seconds':self.total_seconds,'total_frames':self.total_frames,
                'orbit_degrees':self.orbit_degrees,'wide_orbit_direction':'clockwise',
                'initial_elevation_degrees':self.elevation_degrees,
                'shortest_approach_rotation_degrees':float(np.degrees(self._turn_radians)),
                'ice_segments':len(self._ice_segments),'board_segments':len(self._board_segments),
                'board_height_scene_units':self._board_height,
                'board_grid_subdivision':('half-spaced verticals and25/50/75percent height rings'
                                          if self.profile!='legacy' else 'original'),
                'projection':'existing projection.project_fast Brown14 and skew',
                'surface_visibility':'triangle perspective-depth buffer;3D clipped line depth tests',
                'initial_camera25':self.initial_camera25.tolist(),
                'approach_initial_camera25':self._approach_camera25.tolist(),
                'target_source_camera25':self.first_camera25.tolist(),
                'camera_motion':'whole-rink fixed-radius orbit then shortest Slerp/log-focal approach',
                'source_overlay':'fully bright native plus independent cyan wire layer',
                'foreground_contract':'optional person-only mask, no subtitle/timer rectangles',
                'advertisements_and_painted_markings_drawn':False,'fitted_model_changed':False}
        if self.profile=='continuous_orbit_overlay':
            parameters.update(
                wide_orbit_seconds=0.,
                rotation_start_seconds=self.rotation_start_seconds,
                approach_lead_seconds=self.approach_lead_seconds,
                continuous_approach_start_seconds=self.approach_start_seconds,
                continuous_rotation_duration_seconds=self.target_reached_seconds-self.rotation_start_seconds,
                continuous_approach_duration_seconds=self.approach_seconds,
                approach_initial_camera25=self.camera_at(self.approach_start_seconds).tolist(),
                shortest_approach_rotation_degrees=float(np.degrees((
                    self._end_rotation*Rotation.from_rotvec(
                        self.camera_at(self.approach_start_seconds)[3:6]).inv()).magnitude())),
                motion_ramp_in_seconds=self.motion_ramp_in_seconds,
                motion_ramp_out_seconds=self.motion_ramp_out_seconds,
                camera_motion='continuous staggered orbit and approach;C2 velocity ramps, no internal stop')
        return parameters

    def selfcheck(self):
        terminal=self.camera_at(self.target_reached_seconds)
        w,h=self.output_wh
        source=np.full((h,w,3),127,np.uint8)
        source[::7,::11]=[7,123,249]
        exact=bool(np.array_equal(self.frame(self.total_seconds,source),source))
        xyz=np.concatenate([m[0] for m in self._meshes])
        uv,depth=project_fast(self.initial_camera25,xyz)
        inside=bool(((uv[:,0]>=0)&(uv[:,0]<=w-1)&(uv[:,1]>=0)&(uv[:,1]<=h-1)).all())
        all_wide_inside=True
        sample_start=(self.rotation_start_seconds if self.profile=='continuous_orbit_overlay'
                      else self.overview_seconds)
        sample_end=self.approach_start_seconds
        for t in np.linspace(sample_start,sample_end,
                             max(2,round((sample_end-sample_start)*self.fps)+1)):
            p,z=project_fast(self.camera_at(t),xyz)
            all_wide_inside &= bool(((p[:,0]>=0)&(p[:,0]<=w-1)&(p[:,1]>=0)&(p[:,1]<=h-1)&(z>0)).all())
        heights=[self.camera_at(t)[2] for t in np.linspace(0,self.total_seconds,101)]
        report={'terminal_camera25_max_abs_difference':float(np.max(np.abs(terminal-self.first_camera25))),
                'terminal_source_pixels_exact':exact,'overview_all_mesh_vertices_in_frame':inside,
                'wide_orbit_all_mesh_vertices_in_frame_every_output_frame':all_wide_inside,
                'overview_min_depth':float(depth.min()),'animated_min_camera_z':float(min(heights)),
                'projection_brown14_reused':True,'three_dimensional_board_height':self._board_height,
                'full_brightness_overlay_seconds':self.overlay_hold_seconds,
                'parameters':self.parameters()}
        if report['terminal_camera25_max_abs_difference']!=0 or not exact or not inside or not all_wide_inside or depth.min()<=0:
            raise AssertionError(report)
        return report
