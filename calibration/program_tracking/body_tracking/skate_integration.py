"""Joint body/shared-skate/contact refinement on frozen source exposures."""
from pathlib import Path
import argparse,json,time,copy
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix,hstack,vstack
from .rebuild_prepare import load,digest
from .fullbody_state import FrozenReference,WindowState,matrices
from .fullbody_fit import FullBodyProblem,footwear_rows
from .boot_integration import BootWindowProblem,Registration,camera_centres,edges_for
from .footwear_pixel_factors import pixel_factors
from .rebuild_exposure_body import moving_projection
from .opening_hand_polish import build as build_hand,HandWindow
from .skate_binding import SkateBinding,SHAPE_NAMES
from .skate_physics import (fk_state,constraints,body_constraint_jacobian,
    ALLOWANCE,NORMALIZATION,FEASIBILITY_TOLERANCE,SUPPORT_BAND)
from .contact_feasibility import ContactAL,physical_feasibility
from .fullbody_round import write_json
from .footwear_layout import OBSERVATION_LAYOUT


class SkateReference(FrozenReference):
    def __init__(self,parent):
        self.__dict__.update(parent.__dict__)
        self.parent=parent;self.new_refinements=[];self.layout=OBSERVATION_LAYOUT

    def __getattr__(self,name):
        return getattr(self.parent,name)

    def at(self,t):
        r=self.parent.at(t)
        r={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in r.items()}
        for ws,x in self.new_refinements:
            d=ws.increments(x,t);r['P']+=d[:,:3];r['Q']=r['Q']@matrices(d[:,3:6])
            r['local_R']=r['local_R']@matrices(d[:,6:].reshape(-1,21,3))
        return r

    def world(self,reference,delta=None):
        s=FrozenReference.world(self,reference,delta)
        if self.binding is None:return s
        local=np.einsum('nji,nkj->nki',s['Q'],s['native']-s['P'][:,None])
        g=self.binding.evaluate(s['P'],s['Q'],local,s['frames'])
        s.update(gear=g['gear'],boot_R=g['boot_R'],boot_O=g['boot_O'],footwear=g)
        s['world']=np.concatenate((s['gear'].reshape(len(local),self.layout.equipment_count,3),s['coco']),1)
        return s


def build(project,root,out,rocker_ft=7.):
    parent,camera,info,catalog=build_hand(project,root,out)
    saved=load(Path(project)/'events/ankara_2026/programmes/okada_mei_sp/opening_hand_1103.npz')
    ws=HandWindow(parent);np.testing.assert_array_equal(ws.knots,saved['knots'])
    np.testing.assert_array_equal(ws.active,saved['active']);parent.refinements.append((ws,saved['x']))
    ref=SkateReference(parent)
    ref.binding=SkateBinding(ref,parent.binding.surface,info,rocker_ft)
    return ref,camera,info,catalog


def source_evidence(root):
    """Few actual source-image reviews; no propagation of unknown labels."""
    times=(26.22,35.20,35.60,36.02,36.60,39.78,43.86)
    notes={26.22:'One blade visibly supports while the other foot is raised.',
        35.20:'Nearer skate lower edge on ice; farther foot lifted/uncertain.',
        35.60:'Deep knee flexion; at least one lower runner on ice.',
        36.02:'Both boots low; at least one runner supports.',
        36.60:'Strong bank angle, blade edge adjacent to ice; which edge unknown.',
        39.78:'Low sliding/step posture, at least one runner adjacent to ice.',
        43.86:'Visible step support, other skate moving; physical side unknown.'}
    return [dict(id='source_support_'+str(round(t*12800)),pts=round(t*12800),
        source_frame=str(Path(root)/'blade_tracking_20261030/continuous/source_crops'/f'{round(t*12800)}.png'),
        source='assistant direct review of existing unresized source crop, development',
        confidence=.8 if t in (26.22,35.60,36.02,43.86) else .65,
        mode='glide',feet=[0,1],physical_side='unknown',edge='unknown',
        temporal_scope='this exposure midpoint only',note=notes[t],human_confirmed=False) for t in times]


class SharedRegistration(Registration):
    def residual(self,x,log=False):
        b=self.binding;s=self.state;gear=b.evaluate(s['P'],s['Q'],self.local,s['frames'],x=x)
        layout=b.layout
        W=np.concatenate((gear['gear'].reshape(-1,layout.equipment_count,3),s['coco']),1).reshape(len(self.frames),5,layout.point_count,3)
        uv=moving_projection(self.camera,W,self.frames)['uv']
        edge=edges_for(W[:,2],self.frames,self.camera,b,centres=self.camera_centres)
        pixels,_,parts,_=pixel_factors(uv,self.gear,body=False,boot_edges=edge)
        neutral=b.compatibility_rows(b.neutral_native,b.neutral_frames,b.neutral_samples,x=x).ravel()
        dynamic=b.compatibility_rows(self.local[self.mid],s['frames'][self.mid],self.samples,x=x).ravel()/np.sqrt(len(self.frames))
        prior=shared_prior(x)
        result=np.r_[pixels,neutral,dynamic,prior]
        self.last_parts={**parts,'neutral_foot_fit':float(neutral@neutral/2),'dynamic_foot_fit':float(dynamic@dynamic/2),'shared_prior':float(prior@prior/2)}
        self.count+=1
        with (self.output/'fit_progress.jsonl').open('a',encoding='utf8') as stream:
            stream.write(json.dumps(dict(event='shared_registration_evaluation',evaluation=self.count,
                cost=float(result@result/2),parts=self.last_parts,seconds=time.perf_counter()-self.start))+'\n')
        return result


def shared_prior(x):
    return np.r_[x[:12].reshape(2,6)[:,:3].ravel()/.04,
        x[:12].reshape(2,6)[:,3:].ravel()/.30,x[12:]/np.array([.12,.12,.20,.20,.20])]


def shared_bounds(binding):
    p=binding.base_parameters
    lo=np.r_[np.tile(np.r_[[-.08]*3,[-.6]*3],2),np.log([.85,.80,.65,.65,.60])]
    hi=np.r_[np.tile(np.r_[[.08]*3,[.6]*3],2),np.log([1.18,1.25,1.30,1.30,1.40])]
    return lo,hi


class HaloWindow(WindowState):
    def __init__(self,reference,seconds,mode='full'):
        super().__init__(reference,seconds,mode,spacing=.20,taper=.30)


class ContinuousShotWindow(WindowState):
    """One cubic body curve over the available S01; no internal gate/taper.

    The saved HaloWindow remains readable solely to replay the earlier
    candidate. Continuity to footage omitted from this input is unknown.
    """
    def __init__(self,reference,seconds,mode='full'):
        super().__init__(reference,seconds,mode,spacing=.40,taper=.30)
        self.active=np.arange(69,dtype=int);self.d=69;self.size=self.m*self.d

    def basis(self,t,derivative=0):
        if derivative not in (0,1,2):raise ValueError(derivative)
        t=np.atleast_1d(t).astype(float);a,b=self.bounds_time
        values=self._basis_spline(np.clip(t,a,b),derivative)
        values[(t<a)|(t>b)]=0
        return values


def checkpoint_state_kind(saved):
    return str(saved.get('state_kind',np.array('halo_1103')))


def state_from_packet(reference,saved,prefix=''):
    """Reconstruct the actual control basis, without fitting another spline."""
    kind=str(saved.get(prefix+'state_kind',np.array('halo_1103')))
    factory={'halo_1103':HaloWindow,'continuous_shot':ContinuousShotWindow}[kind]
    state=factory(reference,tuple(map(float,saved[prefix+'seconds'])),'full')
    np.testing.assert_array_equal(state.knots,saved[prefix+'knots'])
    np.testing.assert_array_equal(state.active,saved[prefix+'active'])
    return state


def base_layer_packet(reference):
    """Small portable history of corrections to the same P/Q/local_R state."""
    packet={'base_layer_count':np.array(len(reference.new_refinements))}
    for i,(state,controls) in enumerate(reference.new_refinements):
        prefix=f'base_layer_{i}_'
        packet.update({prefix+'x':np.asarray(controls).copy(),prefix+'knots':state.knots.copy(),
            prefix+'active':state.active.copy(),prefix+'seconds':np.asarray(state.seconds),
            prefix+'state_kind':np.array('continuous_shot' if isinstance(state,ContinuousShotWindow) else 'halo_1103')})
    return packet


def restore_checkpoint_layers(reference,saved,include_primary=False):
    """Replay saved candidate layers exactly; there is still one body state."""
    for i in range(int(saved.get('base_layer_count',0))):
        prefix=f'base_layer_{i}_';state=state_from_packet(reference,saved,prefix)
        controls=np.asarray(saved[prefix+'x']).copy()
        if controls.shape!=(state.size,):raise ValueError('Invalid saved base body controls')
        reference.new_refinements.append((state,controls))
    if include_primary:
        state=state_from_packet(reference,saved);controls=saved['x'][17:].copy()
        if controls.shape!=(state.size,):raise ValueError('Invalid saved primary body controls')
        reference.new_refinements.append((state,controls))


class ContactBodyProblem(BootWindowProblem):
    def __init__(self,reference,camera,ledger,seconds,log=None,state_kind='halo_1103'):
        factory={'halo_1103':HaloWindow,'continuous_shot':ContinuousShotWindow}[state_kind]
        FullBodyProblem.__init__(self,reference,camera,ledger,seconds,'full',log,state_factory=factory)
        self.binding=reference.binding;self.mid_times=reference.times[self.frames]
        self.mid_ref=reference.at(self.mid_times);self.binding_key=None
        self.camera_centres=camera_centres(camera,self.frames)

    def visible_boot_edges(self,world):
        from .footwear_observations import visible_shell_edges
        wanted={(r['frame'],r['foot']) for r in self.gear.visible_rows}
        return {(i,f):visible_shell_edges(world[i,self.layout.footwear_landmarks(f,'shell')],
            self.binding.adapter_faces,self.camera_centres[i]) for i,f in wanted}


class JointContactProblem:
    def __init__(self,reference,camera,ledger,seconds,evidence,output,log,state_kind='halo_1103'):
        self.reference=reference;self.binding=reference.binding;self.camera=camera
        self.output=Path(output);self.log=log;self.start=time.perf_counter();self.outer=0;self.accepted=0
        self.state_kind=state_kind
        self.saved_base_layers=base_layer_packet(reference)
        self.compatibility_width=self.binding.compatibility_width
        self.cavity_labels=['left_foot_cavity','right_foot_cavity']
        if self.binding.collar_constraints_enabled:
            self.cavity_labels+=['left_shank_collar','right_shank_collar']
        self.cavity_rows_per_frame=len(self.cavity_labels)
        self.body=ContactBodyProblem(reference,camera,ledger,seconds,log,state_kind)
        self.size=self.body.size+17;self.cache=None;self.cache_x=None
        self.evidence=evidence;self.origin=reference.p['pts'][0]/12800.
        nodes=self.body.times
        mid=(reference.times[self.body.frames][:-1]+reference.times[self.body.frames][1:])/2
        self.set_physics_times(np.r_[nodes,mid,reference.times[self.body.frames]])

    def set_physics_times(self,times,old=None):
        self.physics_times=np.unique(np.round(times,10));self.physics_ref=self.reference.at(self.physics_times)
        self.physics_B=self.body.state.basis(self.physics_times)
        self.cache=None;self.cache_x=None
        x=np.r_[self.binding.x,np.zeros(self.body.size)]
        e=self.contact(x)
        extra_labels=[('neutral',s) for s in self.cavity_labels]
        extra_labels += [(int(p)*78125,s) for p in self.reference.p['pts'][self.body.frames] for s in self.cavity_labels]
        labels=e['labels']+extra_labels
        self.al=ContactAL.zeros(len(labels),penalty=10. if old is None else old[0].penalty)
        if len(set(labels))!=len(labels):raise ValueError('Duplicate global physical row identity')
        if old is not None:
            lookup={tuple(k):v for k,v in zip(old[1],old[0].multiplier)}
            self.al.multiplier[:]=[lookup.get(tuple(k),0.) for k in labels]
        self.row_labels=labels
        self.foot_row_labels=extra_labels

    def cavity_margins(self,rows):
        """Worst inset-cavity breach per foot, in physical metres."""
        rows=np.asarray(rows).reshape(-1,self.compatibility_width)
        outside=rows[:,12:28].reshape(-1,2,8)
        which=np.argmax(outside,axis=2)
        margin=-np.take_along_axis(outside,which[...,None],axis=2)[...,0]*.003
        if self.binding.collar_constraints_enabled:margin=np.c_[margin,-rows[:,32:34]*.003]
        return margin.ravel(),which

    def contact(self,x,jacobian=False,times=None):
        self.binding.x=x[:17].copy()
        ts=self.physics_times if times is None else np.asarray(times)
        ref=self.physics_ref if times is None else self.reference.at(ts)
        B=self.physics_B if times is None else self.body.state.basis(ts)
        delta=self.body.state.increments(x[17:],ts)
        s=fk_state(self.reference,ref,delta)
        e=constraints(self.binding,s,ts,self.origin,self.evidence)
        if jacobian:e['body_jac']=body_constraint_jacobian(self.reference,s,delta,self.body.state.active,B,e)
        e['state']=s
        return e

    def shared_replay(self,shared,body_x,packet,binding=None):
        """Recompute only shoe-dependent quantities for the 17 static columns."""
        b=self.binding if binding is None else binding
        b.x=shared.copy();s=packet['world']
        local=np.einsum('nji,nkj->nki',s['Q'],s['native']-s['P'][:,None])
        g=b.evaluate(s['P'],s['Q'],local,s['frames'])
        layout=b.layout
        W=np.concatenate((g['gear'].reshape(-1,layout.equipment_count,3),s['coco']),1).reshape(self.body.n,5,layout.point_count,3)
        nodes=W[self.body.fi,:,self.body.pi][:,:,None]
        proj=moving_projection(self.camera,nodes,self.body.frames[self.body.fi])
        uv=np.full((self.body.n,layout.point_count,2),np.nan);uv[self.body.fi,self.body.pi]=proj['uv'][:,0]
        br,_,_,_=self.body.body_factors(uv)
        gr,_,_,_=pixel_factors(uv,self.body.gear,body=False,
            boot_edges=self.body.visible_boot_edges(W[:,2]))
        mid=np.arange(self.body.n)*5+2
        samples=self._replay_samples
        compat=b.compatibility_rows(local[mid],s['frames'][mid],samples)/np.sqrt(2)
        return np.r_[br,gr,*[m@body_x for _,m in self.body.prior_matrices],compat.ravel()]

    def shared_column(self,c,x,base):
        # Each worker owns its binding and small template cache. Body/FK,
        # surface tensors and frozen camera inputs are immutable and shared.
        b=copy.copy(self.binding)
        r=object.__new__(type(self.reference));r.__dict__.update(self.reference.__dict__);r.binding=b
        delta=self.body.state.increments(x[17:],self.physics_times)
        # A collar section can cross a short skin triangle near its vertex;
        # retain a local derivative rather than averaging across that crease.
        h=1e-6 if b.collar_constraints_enabled else 1e-5
        residuals=[];guards=[]
        for sign in (1.,-1.):
            trial=x.copy();trial[c]+=sign*h;b.x=trial[:17].copy()
            replay=self.shared_replay(trial[:17],x[17:],base,binding=b)
            neutral=b.compatibility_rows(b.neutral_native,b.neutral_frames,b.neutral_samples).ravel()
            state=fk_state(r,self.physics_ref,delta)
            e=constraints(b,state,self.physics_times,self.origin,self.evidence)
            foot=np.r_[self.cavity_margins(neutral)[0],self.cavity_margins(replay[-self.body.n*self.compatibility_width:]*np.sqrt(2))[0]]/NORMALIZATION
            residuals.append(np.r_[np.sqrt(.02)*replay,neutral,shared_prior(trial[:17])])
            guards.append(np.r_[e['g'],foot])
        return (residuals[0]-residuals[1])/(2*h),(guards[0]-guards[1])/(2*h)

    def evaluate(self,x,jacobian=False):
        if self.cache_x is not None and np.array_equal(x,self.cache_x) and (not jacobian or self.cache['jac'] is not None):return self.cache
        tick=time.perf_counter();self.binding.x=x[:17].copy()
        base=self.body.evaluate(x[17:],jacobian)
        # Existing 50Hz image/reference blocks become duration integrals by a
        # common constant. Physics uses its own positive trapezoidal weights.
        data=np.sqrt(.02)*base['residual']
        neutral=self.binding.compatibility_rows(self.binding.neutral_native,self.binding.neutral_frames,self.binding.neutral_samples).ravel()
        prior=shared_prior(x[:17]);rdata=np.r_[data,neutral,prior]
        e=self.contact(x,jacobian)
        neutral_margin,neutral_which=self.cavity_margins(neutral)
        dynamic_rows=base['residual'][-self.body.n*self.compatibility_width:]*np.sqrt(2)
        dynamic_margin,dynamic_which=self.cavity_margins(dynamic_rows)
        foot_margins=np.r_[neutral_margin,dynamic_margin]
        e['margin_m']=np.r_[e['margin_m'],foot_margins]
        e['g']=np.r_[e['g'],foot_margins/NORMALIZATION]
        e['weights']=np.r_[e['weights'],np.ones(self.cavity_rows_per_frame),np.repeat(
            np.r_[.01,np.full(self.body.n-2,.02),.01] if self.body.n>1 else [.02],self.cavity_rows_per_frame)]
        e['labels']=e['labels']+self.foot_row_labels
        if e['labels']!=self.row_labels:raise ValueError('Contact row identities changed inside inner solve')
        collar_unknown=0
        if jacobian or self.binding.collar_constraints_enabled:
            s=base['world'];local=np.einsum('nji,nkj->nki',s['Q'],s['native']-s['P'][:,None]);mid=np.arange(self.body.n)*5+2
            self._replay_samples=self.binding.surface.sample_vertices(local[mid],s['frames'][mid],self.binding.sample_ids,double_precision=True)
            if self.binding.collar_constraints_enabled:
                collar_unknown=int(self.binding.collar_metrics(local[mid],s['frames'][mid],self._replay_samples)['unknown'].sum())
                collar_unknown+=int(self.binding.collar_metrics(self.binding.neutral_native,self.binding.neutral_frames,
                    self.binding.neutral_samples)['unknown'].sum())
        if jacobian:
            columns=self.body.execution.map_columns(lambda c:self.shared_column(c,x,base),range(17))
            shapeJ,shapeG=zip(*columns)
            self.binding.x=x[:17].copy()
            bodyJ=vstack((np.sqrt(.02)*base['jac'],csr_matrix((len(neutral)+len(prior),self.body.size))),format='csr')
            Jdata=hstack((csr_matrix(np.stack(shapeJ,1)),bodyJ),format='csr')
            select=[]
            for i in range(self.body.n):
                select.extend(i*self.compatibility_width+12+f*8+dynamic_which[i,f] for f in range(2))
                if self.binding.collar_constraints_enabled:select.extend([i*self.compatibility_width+32,i*self.compatibility_width+33])
            compatibility_J=base['jac'][-self.body.n*self.compatibility_width:]
            foot_body_J=vstack((csr_matrix((self.cavity_rows_per_frame,self.body.size)),
                compatibility_J[select]*(-.003*np.sqrt(2)/NORMALIZATION)),format='csr')
            full_body_J=vstack((e['body_jac'],foot_body_J),format='csr')
            Jg=hstack((csr_matrix(np.stack(shapeG,1)),full_body_J),format='csr')
        else:Jdata=Jg=None
        ral,Jal=self.al.residual_and_jacobian(e['g'],Jg,row_weights=e['weights'])
        r=np.r_[rdata,ral];J=vstack((Jdata,Jal),format='csr') if jacobian else None
        gate=physical_feasibility(e['margin_m'],tolerance_m=FEASIBILITY_TOLERANCE,row_labels=e['labels'])
        gate['collar_unknown_sections']=collar_unknown
        if collar_unknown:gate['feasible']=False
        parts=dict(data=float(data@data/2),neutral_fit=float(neutral@neutral/2),shared_prior=float(prior@prior/2),contact_AL=float(ral@ral/2))
        image_names=('body_pixels','visible_boot_edges','visible_blade_curves','boot_direction_endpoints')
        pose_names=('reference_deviation','correction_velocity','correction_acceleration')
        v=dict(residual=r,jac=J,parts=parts,contact=e,gate=gate,
            image_cost=sum(base['parts'].get(name,0.) for name in image_names),
            pose_cost=sum(base['parts'].get(name,0.) for name in pose_names),raw_data_parts=dict(base['parts']))
        self.cache_x=x.copy();self.cache=v
        self.log.write(json.dumps(dict(event='joint_evaluation',outer=self.outer,with_jacobian=jacobian,
            seconds=time.perf_counter()-self.start,evaluation_seconds=time.perf_counter()-tick,cost=float(r@r/2),
            parts=parts,physics_times=len(self.physics_times),constraints=len(e['g']),variables=self.size,
            jacobian_nnz=None if J is None else J.nnz,**gate))+'\n')
        return v

    def restoration_seed(self,x):
        e=self.contact(x);B=self.physics_B;target=np.maximum(-e['clearance'].min(1),0.)/self.binding.ratio
        for i,h in enumerate(e['support_height']):
            if np.isfinite(h):target[i]=-h/self.binding.ratio
        # Only the existing root-Z spline is initialized; cameras and separate
        # equipment poses are absent. End attachment remains unchanged.
        smooth=self.body.state.basis(self.physics_times,2)
        controls=np.linalg.lstsq(np.r_[B,.003*smooth],np.r_[target,np.zeros(len(target))],rcond=None)[0]
        out=x.copy();view=out[17:].reshape(self.body.state.m,self.body.state.d)
        view[:,np.flatnonzero(self.body.state.active==2)[0]]+=controls
        return out

    def checkpoint_fields(self,x):
        return dict(self.saved_base_layers,x=np.asarray(x).copy(),knots=self.body.state.knots,
            active=self.body.state.active,seconds=self.body.state.seconds,physics_times=self.physics_times,
            multiplier=self.al.multiplier,penalty=np.array(self.al.penalty),state_kind=np.array(self.state_kind),
            collar_constraints_enabled=np.array(self.binding.collar_constraints_enabled))

    def solve(self,initial=None,run_policy=None):
        from .skate_run_policy import RunPolicy,Monitor
        self.output.mkdir(parents=True,exist_ok=True)
        prior=self.output/'last_accepted.npz'
        if prior.exists():
            archive=self.output/'history'/('before_'+digest(prior)[:12]);archive.mkdir(parents=True,exist_ok=True)
            for name in ('last_accepted.npz','latest.json','parameter_selection.json','kernel_manifest.json'):
                path=self.output/name
                if path.exists():(archive/name).write_bytes(path.read_bytes())
            if (self.output/'source_snapshot').exists():
                dst=archive/'source_snapshot';dst.mkdir(exist_ok=True)
                for path in (self.output/'source_snapshot').glob('*.py'):(dst/path.name).write_bytes(path.read_bytes())
        monitor=Monitor(run_policy or RunPolicy(),now=time.perf_counter())
        write_json(self.output/'run_policy.json',monitor.record())
        x=np.r_[self.binding.x,np.zeros(self.body.size)] if initial is None else initial.copy()
        if initial is None:x=self.restoration_seed(x)
        lo,hi=shared_bounds(self.binding)
        angular=np.tile(self.body.state.active>=3,self.body.state.m)
        lower=np.r_[lo,np.where(angular,-np.pi,-np.inf)];upper=np.r_[hi,np.where(angular,np.pi,np.inf)]
        # All coefficients of the one body curve participate. Cubic knots
        # describe a continuous curve; they do not define independent clips.
        selection=np.arange(self.size)
        self.parameter_selection=selection
        frozen_initial=x.copy()
        def expand(z):
            full=frozen_initial.copy();full[selection]=z;return full
        write_json(self.output/'parameter_selection.json',dict(full_layout=self.size,active_variables=len(selection),
            active_state_columns=self.body.state.active.tolist(),state_kind=self.state_kind,
            body_support_seconds=list(self.body.state.seconds),selection=selection.tolist(),one_body_state=True,
            internal_temporal_freezes=False))
        implementation={}
        source_dir=self.output/'source_snapshot';source_dir.mkdir(exist_ok=True)
        for name in ('skate_integration.py','skate_binding.py','skate_physics.py','skate_template.py',
                     'skate_contact.py','contact_feasibility.py','footwear_layout.py','fullbody_fit.py',
                     'fullbody_state.py','boot_integration.py','body_execution.py','footwear_pixel_factors.py','skate_run_policy.py',
                     'skate_collar_factors.py'):
            path=Path(__file__).with_name(name);implementation[name]=digest(path)
            (source_dir/name).write_bytes(path.read_bytes())
        write_json(self.output/'kernel_manifest.json',implementation)
        class StopFit(Exception):pass
        last_x=x.copy();last_value=self.evaluate(x,False)
        monitor.observe(now=time.perf_counter(),violation_m=last_value['gate']['max_violation_m'],
            image_cost=last_value['image_cost'],pose_cost=last_value['pose_cost'])
        def save_candidate(reason):
            self.binding.x=last_x[:17].copy()
            np.savez_compressed(self.output/'candidate.pending.npz',**self.checkpoint_fields(last_x))
            (self.output/'candidate.pending.npz').replace(self.output/'candidate.npz')
            record=dict(event='candidate_saved',stop=reason,candidate_checkpoint=str(self.output/'candidate.npz'),
                source_seconds=list(self.body.state.seconds),parts=last_value['parts'],**last_value['gate'],
                raw_data_parts=last_value['raw_data_parts'],numerical_convergence=False,user_accepted=False,
                acceptance_calibrated=False,run_policy=monitor.record(),state_kind=self.state_kind)
            self.log.write(json.dumps(record)+'\n');write_json(self.output/'candidate.json',record)
            return last_x,record
        def guarded(z,jacobian=False):
            if monitor.observe(now=time.perf_counter())['stop']:raise StopFit()
            return self.evaluate(expand(z),jacobian)
        while True:
            self.cache=None;self.cache_x=None;self.outer+=1
            def accepted(reduced):
                nonlocal last_x,last_value
                z=expand(reduced)
                self.accepted+=1;v=self.evaluate(z,True)
                last_x=z.copy();last_value=v
                np.savez_compressed(self.output/'accepted.pending.npz',**self.checkpoint_fields(z),outer=np.array(self.outer))
                (self.output/'accepted.pending.npz').replace(self.output/'last_accepted.npz')
                self.log.write(json.dumps(dict(event='accepted',outer=self.outer,iteration=self.accepted,
                    cost=float(v['residual']@v['residual']/2),parts=v['parts'],**v['gate']))+'\n')
                status=monitor.observe(now=time.perf_counter(),accepted=True,violation_m=v['gate']['max_violation_m'],
                    image_cost=v['image_cost'],pose_cost=v['pose_cost'])
                write_json(self.output/'run_policy.json',monitor.record())
                if status['stop']:raise StopFit()
            try:
                fit=least_squares(lambda z:guarded(z)['residual'],x[selection],jac=lambda z:guarded(z,True)['jac'][:,selection],
                    bounds=(lower[selection],upper[selection]),method='trf',tr_solver='lsmr',x_scale='jac',ftol=1e-7,xtol=1e-7,gtol=1e-4,
                    callback=accepted,tr_options=dict(atol=1e-7,btol=1e-7))
            except StopFit:
                return save_candidate(monitor.record())
            x=expand(fit.x);value=self.evaluate(x,True)
            last_x=x.copy();last_value=value
            np.savez_compressed(self.output/f'outer_{self.outer:02d}.npz',x=x,multiplier=self.al.multiplier,
                penalty=self.al.penalty,physics_times=self.physics_times)
            record=dict(event='outer_complete',outer=self.outer,stop=str(fit.message),optimality=float(fit.optimality),
                penalty=self.al.penalty,parts=value['parts'],**value['gate'])
            self.log.write(json.dumps(record)+'\n');write_json(self.output/'latest.json',record)
            dense=np.unique(np.r_[np.arange(self.body.state.bounds_time[0],self.body.state.bounds_time[1],.0025),self.physics_times])
            check=self.contact(x,times=dense)
            gate=physical_feasibility(check['margin_m'],tolerance_m=FEASIBILITY_TOLERANCE,row_labels=check['labels'])
            if gate['feasible'] and value['gate']['feasible']:
                record.update(dense_gate=gate,temporal_check_step_s=.0025,finite_sampling_is_not_continuous_proof=True,
                    source_seconds=self.body.state.seconds,accepted_iterations=self.accepted,
                    numerical_convergence=fit.optimality<=1e-4,source_supported_exposures=[r for r in self.evidence if self.body.state.seconds[0]<=r['pts']/12800<=self.body.state.seconds[1]])
                np.savez_compressed(self.output/'result.npz',**self.checkpoint_fields(x))
                write_json(self.output/'result.json',record);return x,record
            status=monitor.observe(now=time.perf_counter(),outer_completed=self.outer)
            if status['stop']:return save_candidate(status)
            # Add only newly exposed violating physical times, without creating
            # fictitious source exposures. Existing multiplier identities stay.
            bad={check['rows'][i][0] for i,m in enumerate(check['margin_m']) if m < -FEASIBILITY_TOLERANCE}
            new=np.array([dense[i] for i in sorted(bad)])
            selected=new[~np.isin(np.round(new,10),self.physics_times)]
            previous_al=(self.al.multiplier.copy(),self.al.penalty,self.al.previous_violation)
            update=self.al.update_after_inner_solve(value['contact']['g'])
            status=monitor.observe(now=time.perf_counter(),penalty_growth_requested=update['penalty_increased'])
            if status['stop']:
                self.al.multiplier,self.al.penalty,self.al.previous_violation=previous_al
                return save_candidate(status)
            if len(selected):
                old=(self.al,self.row_labels);self.set_physics_times(np.r_[self.physics_times,selected],old)
            if not np.isfinite(self.al.penalty) or self.al.penalty>1/np.finfo(float).eps:
                record.update(failed=True,reason='AL penalty beyond floating-point conditioning; candidate retained')
                write_json(self.output/'failure.json',record);raise RuntimeError(record['reason'])


def initial_scan(ref):
    t=ref.times;s=fk_state(ref,ref.at(t),np.zeros((len(t),69)))
    return constraints(ref.binding,s,t,ref.p['pts'][0]/12800.,[])


def choose_halo(ref,scan,core=(35.2,36.8)):
    sec=ref.p['pts']/12800.;clear=scan['clearance'].min(1)
    # Fixed attachment states must already be feasible with a little reserve.
    before=np.flatnonzero((sec<=core[0]-.32)&(clear>=.003))
    after=np.flatnonzero((sec>=core[1]+.32)&(clear>=.003))
    if not len(before) or not len(after):raise ValueError('No feasible outer attachment; expand/restore shared boundary before solving')
    return (float(sec[before[-1]]),float(sec[after[0]]))


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--project',default=str(Path(__file__).resolve().parents[3]));p.add_argument('--root',required=True)
    p.add_argument('--output',required=True);p.add_argument('--stage',choices=['prepare','fit','run','export'],default='run')
    p.add_argument('--rocker-ft',type=float,choices=[7.,8.],default=7.)
    starting=p.add_mutually_exclusive_group()
    starting.add_argument('--warm-start',help='Compatibility alias for exact --base-checkpoint replay')
    starting.add_argument('--base-checkpoint',help='Preserve this candidate exactly, then optimize one whole-shot correction')
    p.add_argument('--checkpoint',help='Read-only candidate replay/export; never starts a solve')
    p.add_argument('--run-policy',help='Explicit joint-correction budget JSON; defaults are saved with each run')
    p.add_argument('--delivery-name');p.add_argument('--variant-id')
    args=p.parse_args(argv);project,root,out=map(Path,(args.project,args.root,args.output));out.mkdir(parents=True,exist_ok=True)
    print(json.dumps(dict(main_log=str(out/'fit_progress.jsonl'))),flush=True)
    ref,camera,info,catalog=build(project,root,out,args.rocker_ft)
    def delivery_options(kind):
        return dict(delivery_name=args.delivery_name or ('skate_s01_1103' if kind=='continuous_shot' else 'skate_contact_1103'),
            variant_id=args.variant_id or ('skate_s01_1103' if kind=='continuous_shot' else 'skate1103'))
    replay_path=Path(args.checkpoint) if args.checkpoint else out/'fit/result.npz'
    if args.stage in ('export','run') and replay_path.exists():
        # Export consumes the actual fitted boundary and shared parameters.
        # It must not recalibrate the input or replace its initial scan.
        saved=load(replay_path);ref.binding.x=saved['x'][:17].copy()
        restore_checkpoint_layers(ref,saved)
        if bool(saved.get('collar_constraints_enabled',False)):ref.binding.enable_collar_constraints()
        evidence=json.loads((out/'source_evidence.json').read_text(encoding='utf8'))
        halo=tuple(map(float,saved['seconds']))
        with (out/'fit_progress.jsonl').open('a',encoding='utf8',buffering=1) as log:
            problem=JointContactProblem(ref,camera,root/'boot_integration_1102/observations.npz',halo,evidence,out/'fit',log,checkpoint_state_kind(saved))
            np.testing.assert_array_equal(problem.body.state.knots,saved['knots'])
            np.testing.assert_array_equal(problem.body.state.active,saved['active'])
            if args.checkpoint:
                from .skate_diagnostics import inspect_checkpoint
                report=inspect_checkpoint(problem,saved,replay_path,out/'fit')
            else:report=json.loads((out/'fit/result.json').read_text(encoding='utf8'))
        ref.new_refinements.append((problem.body.state,saved['x'][17:].copy()))
        write_json(out/'binding.json',ref.binding.record())
        from .skate_delivery import deliver
        deliver(project,root,out,ref,camera,info,evidence,problem,report,catalog,**delivery_options(problem.state_kind))
        return
    evidence=source_evidence(root);write_json(out/'source_evidence.json',evidence)
    obs=root/'boot_integration_1102/observations.npz'
    base_path=Path(args.base_checkpoint or args.warm_start) if (args.base_checkpoint or args.warm_start) else None
    contract=dict(schema='rinkspace.body-skate-contact.v1',camera=info,parent=digest(root/'opening_timing_1103/hand_polish/fitted/result.npz'),
        observations_sha256=digest(obs),allowance_m=ALLOWANCE,support_band_m=SUPPORT_BAND,feasibility_tolerance_m=FEASIBILITY_TOLERANCE,
        quadrature='duration integral; existing 50Hz blocks times0.02; physical exact-time trapezoidal weights',
        execution_only='run_body_performance launcher',source_time_base=[1,12800],camera_frozen=True,shape_names=SHAPE_NAMES,
        preserved_candidate=None if base_path is None else dict(path=str(base_path),sha256=digest(base_path),
            replay='exact saved basis and controls, followed by one whole-S01 correction; no spline remapping'))
    write_json(out/'inputs.json',contract)
    registration_contract=dict(binding_kernel_sha256=digest(Path(__file__).with_name('skate_binding.py')),
        rocker_ft=args.rocker_ft,observations_sha256=digest(obs))
    prior_contract=json.loads((out/'registration_inputs.json').read_text(encoding='utf8')) if (out/'registration_inputs.json').exists() else None
    if base_path is None and (out/'registration.npz').exists() and prior_contract!=registration_contract:
        history=out/'history'/('registration_'+digest(out/'registration.npz')[:12]);history.mkdir(parents=True,exist_ok=True)
        for name in ('registration.npz','registration.json','registration_inputs.json','binding_initial.json','initial_scan.npz','window.json'):
            source=out/name
            if source.exists():source.replace(history/name)
    if base_path is not None:
        saved=load(base_path);ref.binding.x=saved['x'][:17].copy()
        restore_checkpoint_layers(ref,saved,include_primary=True)
    elif (out/'registration.npz').exists():ref.binding.x=load(out/'registration.npz')['x']
    else:
        reg=SharedRegistration(ref,camera,ref.binding,out)
        x0=ref.binding.x.copy();r0=reg.residual(x0);lo,hi=shared_bounds(ref.binding)
        fit=least_squares(reg.residual,x0,jac='3-point',bounds=(lo,hi),x_scale='jac',ftol=1e-8,xtol=1e-8,gtol=1e-5)
        ref.binding.x=fit.x.copy();np.savez_compressed(out/'registration.npz',x=fit.x,initial_x=x0)
        write_json(out/'registration.json',dict(initial_cost=float(r0@r0/2),final_cost=float(fit.cost),parts=reg.last_parts,
            optimality=float(fit.optimality),stopping_reason=fit.message,shared_dimensions=17))
        write_json(out/'registration_inputs.json',registration_contract)
    ref.binding.enable_collar_constraints()
    write_json(out/'binding_initial.json',ref.binding.record())
    # New correction uses the whole available S01 plus its exposure-time
    # support. The old core is a review range, not a body-state boundary.
    padding=max(abs(float(camera.p['rig'][37]))/2,float(np.min(np.diff(ref.times)))/2)
    halo=(float(ref.p['pts'][0]/12800.-padding),float(ref.p['pts'][-1]/12800.+padding))
    write_json(out/'window.json',dict(review_core=[35.2,36.8],actual=halo,
        state_kind='continuous_shot',all_source_exposures=True,internal_temporal_freezes=False,
        endpoint_policy='free at available source/exposure support boundary; continuity to omitted footage unknown'))
    print(json.dumps(dict(shared_registration=str(out/'registration.json'),halo=halo)),flush=True)
    if args.stage=='prepare':return
    with (out/'fit_progress.jsonl').open('a',encoding='utf8',buffering=1) as log:
        problem=JointContactProblem(ref,camera,obs,halo,evidence,out/'fit',log,'continuous_shot')
        if (out/'fit/result.npz').exists():x=load(out/'fit/result.npz')['x'];report=json.loads((out/'fit/result.json').read_text(encoding='utf8'))
        elif args.stage!='export':
            initial=None
            if base_path is not None:
                initial=np.r_[ref.binding.x,np.zeros(problem.body.size)]
                (out/'fit').mkdir(parents=True,exist_ok=True)
                write_json(out/'fit/warm_start_replay.json',dict(
                    source_checkpoint=str(base_path),source_sha256=digest(base_path),
                    original_candidate_retained=True,exact_control_replay=True,
                    saved_base_layers=len(ref.new_refinements),new_state_kind='continuous_shot',
                    maximum_position_increment_error_solver=0.,maximum_rotation_increment_error_rad=0.,
                    interpretation='saved layers evaluated directly in the unique body state; whole-S01 correction initially zero'))
            from .skate_run_policy import RunPolicy
            policy=RunPolicy(**json.loads(Path(args.run_policy).read_text(encoding='utf8'))) if args.run_policy else RunPolicy()
            x,report=problem.solve(initial,policy)
        else:raise FileNotFoundError(out/'fit/result.npz')
    ref.binding.x=x[:17].copy();ref.new_refinements.append((problem.body.state,x[17:].copy()))
    write_json(out/'binding.json',ref.binding.record())
    print(json.dumps(report),flush=True)
    if args.stage in ('export','run'):
        from .skate_delivery import deliver
        deliver(project,root,out,ref,camera,info,evidence,problem,report,catalog,**delivery_options(problem.state_kind))


if __name__=='__main__':main()
