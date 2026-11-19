"""Actual boot geometry must not protrude into clearly empty source ice.

Inside the conservative source foreground there is no attraction. A hidden boot
can remain behind either leg or the other boot without being fitted to the
foreground boot's complete silhouette.
"""
from pathlib import Path

import numpy as np
from scipy.sparse import vstack

from .programme_skate_evidence import exposure_times
from .programme_whole_skate import native_point_derivative,chain
from .rebuild_exposure_body import moving_projection
from .rebuild_prepare import digest
from .skate_physics import fk_state
from .weighted_blocks import block,full_columns


def field_sample(fields,bounds,uv,*,image_size=None):
    """Bilinear native-pixel distance with its exact piecewise derivative."""
    n,h,w=fields.shape;scale=np.array([w,h])/(bounds[:,2:]-bounds[:,:2])
    q=(uv-bounds[:,None,:2]+.5)*scale[:,None]-.5
    clipped=np.clip(q,[0.,0.],[w-1.,h-1.]);lo=np.floor(clipped).astype(int)
    lo=np.minimum(lo,[w-2,h-2]);fraction=clipped-lo
    i=np.arange(n)[:,None];x,y=lo[...,0],lo[...,1];a,b=fraction[...,0],fraction[...,1]
    q00=fields[i,y,x];q10=fields[i,y,x+1];q01=fields[i,y+1,x];q11=fields[i,y+1,x+1]
    value=(1-a)*(1-b)*q00+a*(1-b)*q10+(1-a)*b*q01+a*b*q11
    gradient=np.stack([(1-b)*(q10-q00)+b*(q11-q01),
                       (1-a)*(q01-q00)+a*(q11-q10)],axis=-1)*scale[:,None]
    gradient*=((q>=0)&(q<=np.array([w-1,h-1])))
    outside=(q-clipped)/scale[:,None];length=np.linalg.norm(outside,axis=-1)
    value+=length;gradient+=outside/np.maximum(length[...,None],1.e-12)
    if image_size is not None:
        # A cropped broadcast frame supplies no negative evidence outside its
        # sensor rectangle. Distance to the union of foreground and unseen
        # exterior stays continuous at the frame edge; it cannot pull an
        # off-screen boot up onto the visible calf.
        size=np.asarray(image_size,dtype=float)
        if size.shape!=(2,) or not np.isfinite(size).all() or np.any(size<=1):
            raise ValueError('Positive source image width and height required')
        margins=np.concatenate((uv+.5,size-.5-uv),axis=-1)
        edge=np.argmin(margins,axis=-1)
        distance=np.maximum(np.min(margins,axis=-1),0.)
        use=distance<value
        direction=np.array([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]])[edge]
        direction*= (distance>0)[...,None]
        value=np.where(use,distance,value)
        gradient=np.where(use[...,None],direction,gradient)
    return value,gradient


class NativeFootForeground:
    name='footwear_source_foreground'

    def __init__(self,problem,specification,frames):
        from experiments.skate_video_v1.body_skate_fit import Assembly
        self.problem=problem;self.assembly=Assembly(problem.binding)
        self.image_size=specification.get('image_size')
        self.geometry=specification.get('geometry','upper')
        if self.geometry not in ('upper','whole_skate'):
            raise ValueError('Unknown source footwear geometry')
        if self.geometry=='whole_skate':
            # Hand-reviewed empty background can constrain the mounted blade
            # too. Use the exact rendered template, not Assembly's coarse
            # runner approximation; foreground must include all occluders.
            packet=problem.binding.packet()
            self.assembly.vertices=np.concatenate((packet['upper'],packet['template']['vertices']))
            self.name='complete_footwear_source_foreground'
        self.fit_visible_extent=bool(specification.get('fit_visible_extent',False))
        self.fit_visible_bottom=bool(specification.get('fit_visible_bottom',False))
        self.feet=tuple(specification.get('feet',(0,1)))
        if not self.feet or len(set(self.feet))!=len(self.feet) or any(f not in (0,1) for f in self.feet):
            raise ValueError('Source footwear foreground needs distinct physical foot tracks')
        if self.fit_visible_bottom and (len(self.feet)!=1 or self.geometry!='whole_skate'):
            raise ValueError('Visible lowest metal requires one complete rendered skate')
        path=Path(specification['path'])
        if digest(path)!=specification['sha256']:raise ValueError('Source foreground changed')
        with np.load(path) as p:
            np.testing.assert_array_equal(p['pts'],problem.reference.p['pts'])
            frames=np.asarray(frames);self.frames=frames[p['active'][frames]]
            self.fields=p['distance'][self.frames].astype(float);self.bounds=p['bounds'][self.frames]
            if self.fit_visible_bottom:
                self.bottom=p['visible_bottom_region'][self.frames].astype(float)
                self.bottom_valid=p['visible_bottom_valid'][self.frames].astype(bool)
                if (self.bottom.shape!=(len(self.frames),2,2) or
                        self.bottom_valid.shape!=(len(self.frames),) or
                        not np.isfinite(self.bottom).all() or np.any(self.bottom[:,0]>self.bottom[:,1])):
                    raise ValueError('Invalid visible source lower-metal region')
            if self.fit_visible_extent:
                if len(self.feet)!=1:
                    raise ValueError('A visible boot extent must name one physical foot')
                self.extent=p['visible_extent'][self.frames].astype(float)
                self.extent_tolerance=p['extent_tolerance'][self.frames].astype(float)
                self.extent_valid=(p['visible_extent_valid'][self.frames].astype(bool)
                    if 'visible_extent_valid' in p else np.ones_like(self.extent,dtype=bool))
                if (self.extent.shape!=(len(self.frames),4) or
                        self.extent_tolerance.shape!=self.extent.shape or
                        self.extent_valid.shape!=self.extent.shape or
                        not np.isfinite(self.extent).all() or
                        not np.isfinite(self.extent_tolerance).all() or
                        np.any(self.extent_tolerance<0) or
                        np.any((self.extent[:,2:]<=self.extent[:,:2]) &
                               self.extent_valid[:,2:] & self.extent_valid[:,:2])):
                    raise ValueError('Invalid fully visible source boot extent')
        if not len(self.frames):return
        self.times=exposure_times(problem,self.frames);self.base=problem.reference.at(self.times)
        # Person/boot segmentation does not reliably resolve the thin runner.
        # Restrict only the actual upper; the mounted blade follows the same
        # articulated state and retains its own directed/outline observations.
        v=(self.assembly.vertices if self.geometry=='whole_skate'
           else self.assembly.vertices[:self.assembly.upper_count])
        selected=list(np.unique(np.r_[v.argmin(0),v.argmax(0)]))
        distance=np.min(np.linalg.norm(v[:,None]-v[selected],axis=2),axis=1)
        while len(selected)<32:
            j=int(distance.argmax());selected.append(j)
            distance=np.minimum(distance,np.linalg.norm(v-v[j],axis=1))
        # Extrema belong to the complete rendered upper. Sparse surface
        # probes can miss the toe or cuff extremum when a boot turns end-on.
        selected=(np.arange(len(v)) if self.fit_visible_extent or self.geometry=='whole_skate' or specification.get('all_vertices',False)
                  else np.asarray(selected))
        self.count=len(selected)
        self.selectors=(np.broadcast_to(np.repeat(selected[:,None],2,axis=1),
                                      (len(self.frames),len(selected),2)),
                        np.zeros((len(self.frames),len(selected))))
        self.provenance=dict(source_path=str(path),source_sha256=specification['sha256'],
            native_exposures=len(self.frames),geometry=('actual rendered upper and complete lower template'
                if self.geometry=='whole_skate' else 'actual deformed upper boot surface; blade retains separate observations'),
            semantics='outside only; hidden shoe need not coincide with visible foreground shoe',
            foot_tracks=list(self.feet),
            fit_visible_extent=self.fit_visible_extent,
            fit_visible_bottom=self.fit_visible_bottom,
            sampled_vertex_count=self.count,
            upper_vertex_count=self.assembly.upper_count if self.geometry=='whole_skate' else self.count,
            image_size=self.image_size,unseen_image_exterior_unconstrained=self.image_size is not None)

    def evaluate(self,x,jacobian=False):
        p=self.problem;delta=p.body.state.increments(x,self.times)
        pose=fk_state(p.reference,self.base,delta);values=[];matrices=[];distances=[]
        scale=np.sqrt(.02/self.count)/6.;extent_errors=[];bottom_errors=[]
        for foot in self.feet:
            def points(s):
                return self.assembly.points(s,foot,self.selectors,(s['R'],s['O'])).reshape(len(self.frames),5,-1,3)
            pr=moving_projection(p.camera,points(pose),self.frames,jacobian=jacobian)
            distance,gradient=field_sample(self.fields,self.bounds,pr['uv'],image_size=self.image_size)
            values.append((distance*scale).ravel());distances.append(distance)
            if jacobian:
                B,S=native_point_derivative(p,self.base,delta,points,foot,pr['pixel_jac_nodes'])
                metric=gradient*scale
                matrices.append(chain(p,delta,self.times,np.einsum('nkp,nkpad->nkad',metric,B),
                                      np.einsum('nkp,nkpad->nkad',metric,S)))
            if self.fit_visible_extent:
                # A fully visible source boot supplies extent evidence as well
                # as empty exterior. Outside-only evidence permits an incorrect
                # foreshortened boot to fit inside the observed boot. These four
                # extrema refer to the actual projected upper, not invented
                # material toe/heel correspondences or a completed occluded mask.
                uv=pr['uv'];axis=np.array([0,1,0,1])
                ids=np.c_[uv.argmin(axis=1),uv.argmax(axis=1)]
                n=np.arange(len(self.frames))[:,None]
                predicted=uv[n,ids,axis]
                low=self.extent-self.extent_tolerance;high=self.extent+self.extent_tolerance
                # A blurred white/gray material junction may leave one bound
                # unreadable. Such an edge supplies no extremum constraint;
                # do not invent a full rectangle or use the gray heel as white.
                error=(predicted-np.clip(predicted,low,high))*self.extent_valid
                extent_scale=np.sqrt(.02)/6.
                values.append((error*extent_scale).ravel());extent_errors.append(error)
                if jacobian:
                    active=((predicted<low)|(predicted>high))*self.extent_valid*extent_scale
                    selected_B=B[n,ids,axis]*active[...,None,None]
                    selected_S=S[n,ids,axis]*active[...,None,None]
                    matrices.append(chain(p,delta,self.times,selected_B,selected_S))
            if self.fit_visible_bottom:
                # A visible lowest metal region prevents a foreshortened blade
                # from disappearing inside the boot while satisfying an empty-
                # background constraint. A side-view region may be wide; it
                # supplies no invented correspondence to a blade endpoint.
                ids=pr['uv'][...,1].argmax(axis=1);n=np.arange(len(self.frames))
                predicted=pr['uv'][n,ids];low,high=self.bottom[:,0],self.bottom[:,1]
                error=(predicted-np.clip(predicted,low,high))*self.bottom_valid[:,None]
                bottom_scale=np.sqrt(.02)/6.
                values.append((error*bottom_scale).ravel());bottom_errors.append(error)
                if jacobian:
                    active=((predicted<low)|(predicted>high))*self.bottom_valid[:,None]*bottom_scale
                    matrices.append(chain(p,delta,self.times,B[n,ids]*active[...,None,None],
                                          S[n,ids]*active[...,None,None]))
        residual=np.concatenate(values);J=vstack(matrices,format='csr') if jacobian else None
        return dict(residual=residual,jac=J,parts={self.name:float(residual@residual/2)},
            metrics={self.name:dict(source_frames=len(self.frames),outside_px_p95=float(np.percentile(distances,95)),
                outside_px_max=float(np.max(distances)),
                visible_bottom_max_error_px=float(np.max(np.abs(bottom_errors),initial=0.)),
                visible_extent_max_error_px=float(np.max(np.abs(extent_errors),initial=0.)))},
            weighted_blocks=[block(self.name,residual,J,self.provenance)])


class RegisteredBootCuff:
    """Keep the shank inside the registered boot's deformable opening.

    This is a geometric attachment prior, not a source observation or a claim
    of complete skin collision. The finite centre term resolves a hidden
    boot's otherwise unconstrained tilt without prescribing its world yaw.
    """
    name='registered_boot_cuff'

    def __init__(self,problem,times,centre_weight=.15):
        from .skate_contact import trapezoid_time_weights
        self.problem=problem;self.times=np.asarray(times);self.base=problem.reference.at(times)
        binding=problem.binding;R,t,_=binding.parameters();packet=binding.packet()
        self.rotation=R;self.ankle=-np.einsum('fi,fij->fj',t,R)
        length=packet['parameters'].sole_length_m
        station=int(np.argmin(abs(packet['xs']/length-.28)))
        height=packet['floor'][station]+packet['rise'][station]
        radius=.027/binding.ratio
        lo=np.array([radius-.015/binding.ratio,-packet['width'][station]-.012/binding.ratio+radius])
        hi=np.array([.46*length+.020/binding.ratio-radius,packet['width'][station]+.012/binding.ratio-radius])
        self.planes=np.zeros((2,5,3));self.centre=np.zeros((2,2,3))
        for foot in (0,1):
            gap=height-self.ankle[foot,2]
            for axis in (0,1):
                self.planes[foot,2*axis,axis]=gap
                self.planes[foot,2*axis,2]=self.ankle[foot,axis]-hi[axis]
                self.planes[foot,2*axis+1,axis]=-gap
                self.planes[foot,2*axis+1,2]=lo[axis]-self.ankle[foot,axis]
                self.centre[foot,axis,axis]=gap
                self.centre[foot,axis,2]=self.ankle[foot,axis]-(lo[axis]+hi[axis])/2
            self.planes[foot,4,2]=-gap
        self.planes/=.003/binding.ratio
        self.centre/= .015/binding.ratio
        self.centre*=np.sqrt(float(centre_weight))
        J=problem.reference.body.J
        self.shank=np.stack([J[k]-J[a] for k,a in ((4,7),(5,8))])
        self.shank/=np.linalg.norm(self.shank,axis=-1)[:,None]
        self.measure=np.sqrt(trapezoid_time_weights(self.times))

    def evaluate(self,x,jacobian=False):
        from .fullbody_state import matrices
        from .fullbody_fit import right_jacobian,skew
        p=self.problem;d=p.body.state.increments(x,self.times)
        values=[];derivatives=[]
        for foot,joint in enumerate((7,8)):
            columns=6+3*(joint-1)+np.arange(3)
            increment=np.asarray(d)[:,columns]
            rotation=self.base['local_R'][:,joint-1]@matrices(increment)
            local=np.einsum('nji,j->ni',rotation,self.shank[foot])
            direction=local@self.rotation[foot]
            wall=direction@self.planes[foot].T
            value=np.c_[np.maximum(wall,0.),direction@self.centre[foot].T]
            values.append(value*self.measure[:,None])
            if jacobian:
                D=self.rotation[foot].T@skew(local)@right_jacobian(increment)
                gradient=np.concatenate((np.broadcast_to(self.planes[foot],(len(d),5,3))*(wall>0)[:,:,None],
                                         np.broadcast_to(self.centre[foot],(len(d),2,3))),axis=1)
                local_J=np.zeros((len(d),7,69))
                local_J[:,:,columns]=np.einsum('nri,nij->nrj',gradient,D)*self.measure[:,None,None]
                derivatives.append(d.chain(local_J))
        residual=np.concatenate([v.ravel() for v in values]);J=vstack(derivatives,format='csr') if jacobian else None
        result=full_columns(block(self.name,residual,J,dict(semantics='registered geometric cuff envelope and finite alignment prior; no world yaw target',
            full_surface_collision_claim=False,source_observation=False)))
        return dict(residual=result.residual,jac=result.jacobian,parts={self.name:float(residual@residual/2)},
                    metrics={self.name:dict(maximum_weighted_residual=float(np.max(np.abs(residual),initial=0.)))},weighted_blocks=[result])


class NativeHandForeground:
    """Outside-only source evidence for both hands during reviewed overlap."""
    name='hand_source_foreground'

    def __init__(self,problem,specification,frames):
        from .source_motion_refit import NativeSurfacePoints
        self.problem=problem;path=Path(specification['path'])
        if digest(path)!=specification['sha256']:raise ValueError('Hand foreground source changed')
        with np.load(path) as packet:
            np.testing.assert_array_equal(packet['pts'],problem.reference.p['pts'])
            frames=np.asarray(frames);seconds=packet['pts'][frames]/12800
            lo,hi=specification['seconds']
            self.frames=frames[packet['active'][frames]&(seconds>=lo)&(seconds<=hi)]
            self.fields=packet['distance'][self.frames].astype(float);self.bounds=packet['bounds'][self.frames]
        self.count=6
        if not len(self.frames):return
        # Existing directed-hand material points, one per actual distal digit.
        left=[[4933],[5058],[5169]];right=[[7669],[7794],[7905]]
        rows=[]
        for frame in self.frames:
            for i,ids in enumerate(left+right):
                rows.append(dict(id=f'hand_geometry:{frame}:{i}',pts=int(problem.reference.p['pts'][frame]),
                    role='fit',vertex_ids=ids,vertex_weights=[1/len(ids)]*len(ids),uv=[0.,0.],
                    normalization_px=1.,weight=1.,source_kind='internal_projection_probe_not_observation'))
        # A zero-target unit-scale probe returns the exact UV and its existing
        # analytic derivative. These virtual rows never enter a source ledger.
        self.projector=NativeSurfacePoints(problem,dict(surface_samples=rows))
        self.provenance=dict(source_path=str(path),source_sha256=specification['sha256'],
            seconds=specification['seconds'],semantics='outside only; anonymous person/boot foreground permits real occlusion',
            physical_hand_identity_observed=False,geometry_probes_are_not_source_observations=True)

    def evaluate(self,x,jacobian=False):
        from scipy.sparse import coo_matrix
        projected=self.projector.evaluate(x,jacobian)
        uv=projected['residual'].reshape(len(self.frames),self.count,2)
        distance,gradient=field_sample(self.fields,self.bounds,uv)
        scale=np.sqrt(.02/self.count)/6.;residual=(distance*scale).ravel();J=None
        if jacobian:
            ids=np.arange(distance.size)
            combine=coo_matrix(((gradient*scale).ravel(),
                (np.repeat(ids,2),(2*ids[:,None]+np.arange(2)).ravel())),shape=(len(ids),2*len(ids))).tocsr()
            J=combine@projected['jac']
        return dict(residual=residual,jac=J,parts={self.name:float(residual@residual/2)},
            metrics={self.name:dict(source_frames=len(self.frames),outside_px_max=float(distance.max()))},
            weighted_blocks=[block(self.name,residual,J,self.provenance)])
