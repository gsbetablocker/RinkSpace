"""Joint body/shared-skate/contact refinement on frozen source exposures."""
from pathlib import Path
import argparse,json,time,copy
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix,hstack,vstack,diags
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
from .ice_soft import IceSoftPolicy,ice_soft_rows
from .fullbody_round import write_json
from .footwear_layout import OBSERVATION_LAYOUT
from .skate_checkpoint import checkpoint_identity,validate_checkpoint_identity


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
    ref.fixed_checkpoint_identity=checkpoint_identity(ref.binding,project,info)
    return ref,camera,info,catalog


def source_evidence(root,*,shot_id='s01',source_pts=None,records=None):
    """Few actual source-image reviews; no propagation of unknown labels."""
    if records is not None:
        result=copy.deepcopy(records)
        if any('source_support_weight_s' not in r for r in result):
            raise ValueError('Generic clips require explicit source support measures')
        if source_pts is not None:
            wanted=set(map(int,source_pts));result=[r for r in result if int(r['pts']) in wanted]
        return result
    if shot_id!='s01':return []
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
        source_support_weight_s=256/12800,
        source_support_weight_basis='one native 50Hz S01 source observation; objective measure, not shutter/contact duration',
        source_time_base=[1,12800],shot_id=shot_id,
        temporal_scope='this exposure midpoint only',note=notes[t],human_confirmed=False) for t in times
        if source_pts is None or round(t*12800) in set(map(int,source_pts))]


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


class ContinuousRootWindow(ContinuousShotWindow):
    """Translation of the single body/skate state over a complete clip."""
    regularize_actual_root=True

    def __init__(self,reference,seconds,mode='full'):
        super().__init__(reference,seconds,mode)
        self.active=np.arange(3,dtype=int);self.d=3;self.size=self.m*self.d


class ContinuousPoseWindow(ContinuousShotWindow):
    """All body rotations on one curve, with the repaired root held fixed."""
    def __init__(self,reference,seconds,mode='full'):
        super().__init__(reference,seconds,mode)
        self.active=np.arange(3,69,dtype=int);self.d=66;self.size=self.m*self.d


class SupportJointWindow(WindowState):
    """One C2 correction: world P/Q and the associated support-leg chain."""
    regularize_actual_root_translation=True
    # A mixed root-acceleration/correction regularizer is not source pose
    # quality. Keep it in the objective; compare actual source groups instead.
    use_pose_cost_guard=False

    def __init__(self,reference,seconds,mode='full',*,foot=1):
        super().__init__(reference,seconds,mode,spacing=.25,taper=.50)
        if foot not in (0,1):raise ValueError('Explicit associated foot required')
        self.foot=int(foot)
        joints=(1,4,7) if foot==0 else (2,5,8)
        self.active=np.array(list(range(6))+[6+3*(j-1)+a for j in joints for a in range(3)])
        self.d=len(self.active);self.size=self.m*self.d
        self.state_kind='support_joint_left_v1' if foot==0 else 'support_joint_right_v1'


class SpinGraspWindow(WindowState):
    """One C2 correction over the action with both legs, trunk and both arms."""
    use_pose_cost_guard=False
    regularize_actual_root_translation=True
    def __init__(self,reference,seconds,mode='full'):
        super().__init__(reference,seconds,mode,spacing=.25,taper=.5)
        joints=(1,2,3,4,5,6,7,8,9,13,14,16,17,18,19,20,21)
        self.active=np.array(list(range(6))+[6+3*(j-1)+a for j in joints for a in range(3)])
        self.d=len(self.active);self.size=self.m*self.d;self.state_kind='spin_grasp_v1'



class SpinSupportWindow(SpinGraspWindow):
    """One continuous joint curve including both native toe joints."""
    def __init__(self,reference,seconds,mode='full'):
        super().__init__(reference,seconds,mode)
        joints=(1,2,3,4,5,6,7,8,9,10,11,13,14,16,17,18,19,20,21)
        self.active=np.array(list(range(6))+[6+3*(j-1)+a for j in joints for a in range(3)])
        self.d=len(self.active);self.size=self.m*self.d;self.state_kind='spin_support_v2'


def window_factory(kind):
    if kind=='absolute_motion_v1':
        from .absolute_motion_state import AbsoluteMotionState
        return AbsoluteMotionState
    return {'spin_support_v2':SpinSupportWindow,'spin_grasp_v1':SpinGraspWindow,'halo_1103':HaloWindow,'continuous_shot':ContinuousShotWindow,
        'continuous_root':ContinuousRootWindow,'continuous_pose':ContinuousPoseWindow,
        'support_joint_left_v1':lambda r,s,m:SupportJointWindow(r,s,m,foot=0),
        'support_joint_right_v1':lambda r,s,m:SupportJointWindow(r,s,m,foot=1)}[kind]


def checkpoint_state_kind(saved):
    return str(saved.get('state_kind',np.array('halo_1103')))


def state_from_packet(reference,saved,prefix=''):
    """Reconstruct the actual control basis, without fitting another spline."""
    kind=str(saved.get(prefix+'state_kind',np.array('halo_1103')))
    if kind=='absolute_motion_v1':
        from .absolute_motion_state import AbsoluteMotionState,curve_from_packet
        reference.absolute_curve=curve_from_packet(saved,seed=True)
        return AbsoluteMotionState(reference,tuple(saved['seconds']))
    factory=window_factory(kind)
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
            prefix+'state_kind':np.array(getattr(state,'state_kind','continuous_pose' if isinstance(state,ContinuousPoseWindow) else 'continuous_root' if isinstance(state,ContinuousRootWindow) else 'continuous_shot' if isinstance(state,ContinuousShotWindow) else 'halo_1103'))})
    return packet


def restore_checkpoint_layers(reference,saved,include_primary=False):
    """Replay saved candidate layers exactly; there is still one body state."""
    if str(saved.get('state_kind',''))=='absolute_motion_v1':
        raise ValueError('Absolute state requires spin_continuous_refit.restore_absolute; cannot append it as a correction layer')
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
        factory=window_factory(state_kind)
        FullBodyProblem.__init__(self,reference,camera,ledger,seconds,'full',log,state_factory=factory)
        self.binding=reference.binding;self.mid_times=reference.times[self.frames]
        self.mid_ref=reference.at(self.mid_times);self.binding_key=None
        self.camera_centres=camera_centres(camera,self.frames)
        prepare=getattr(self.execution,'prepare_compatibility',None)
        self.resident_compatibility=None if prepare is None else prepare(self.reference,self.mid_ref,self.binding)

    def visible_boot_edges(self,world):
        from .footwear_observations import visible_shell_edges
        wanted={(r['frame'],r['foot']) for r in self.gear.visible_rows}
        return {(i,f):visible_shell_edges(world[i,self.layout.footwear_landmarks(f,'shell')],
            self.binding.adapter_faces,self.camera_centres[i]) for i,f in wanted}


class JointContactProblem:
    def __init__(self,reference,camera,ledger,seconds,evidence,output,log,state_kind='halo_1103',*,optimize_shared=True,ice_policy=None,record_acceptance_before_jacobian=False):
        self.reference=reference;self.binding=reference.binding;self.camera=camera
        self.output=Path(output);self.log=log;self.start=time.perf_counter();self.outer=0;self.accepted=0
        self.state_kind=state_kind
        self.optimize_shared=bool(optimize_shared)
        self.record_acceptance_before_jacobian=bool(record_acceptance_before_jacobian)
        if ice_policy is None:
            config=Path(__file__).resolve().parents[3]/'events/ankara_2026/programmes/okada_mei_sp/ice_soft_1103.json'
            ice_policy=IceSoftPolicy(**json.loads(config.read_text(encoding='utf-8'))['policy'])
        self.ice_policy=ice_policy
        self.saved_base_layers=base_layer_packet(reference)
        self.compatibility_width=self.binding.compatibility_width
        self.cavity_labels=['left_foot_cavity','right_foot_cavity']
        if self.binding.collar_constraints_enabled:
            self.cavity_labels+=['left_shank_collar','right_shank_collar']
        self.cavity_rows_per_frame=len(self.cavity_labels)
        self.body=ContactBodyProblem(reference,camera,ledger,seconds,log,state_kind)
        from .opening_live import OpeningLiveFactors
        self.source_motion=OpeningLiveFactors(self.body)
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
        # Ice is finite objective data, never an augmented-Lagrangian row.
        labels=extra_labels
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
        return dict(residual=np.r_[br,gr,*[r for _,r in self.body.prior_rows(body_x)],compat.ravel()],
            compatibility_residual=compat.ravel())

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
            foot=np.r_[self.cavity_margins(neutral)[0],self.cavity_margins(replay['compatibility_residual']*np.sqrt(2))[0]]/NORMALIZATION
            residuals.append(np.r_[np.sqrt(.02)*replay['residual'],neutral,shared_prior(trial[:17]),
                ice_soft_rows(e,self.ice_policy)['residual']])
            guards.append(foot)
        return (residuals[0]-residuals[1])/(2*h),(guards[0]-guards[1])/(2*h)

    def evaluate(self,x,jacobian=False):
        if self.cache_x is not None and np.array_equal(x,self.cache_x) and (not jacobian or self.cache['jac'] is not None):return self.cache
        tick=time.perf_counter();self.binding.x=x[:17].copy();phases={};phase_start=tick
        base=self.body.evaluate(x[17:],jacobian)
        phases['body_with_compatibility_seconds']=time.perf_counter()-phase_start;phase_start=time.perf_counter()
        # Existing 50Hz image/reference blocks become duration integrals by a
        # common constant. Physics uses its own positive trapezoidal weights.
        data=np.sqrt(.02)*base['residual']
        neutral=self.binding.compatibility_rows(self.binding.neutral_native,self.binding.neutral_frames,self.binding.neutral_samples).ravel()
        prior=shared_prior(x[:17]);rdata=np.r_[data,neutral,prior]
        phases['neutral_and_prior_seconds']=time.perf_counter()-phase_start;phase_start=time.perf_counter()
        raw_ice=self.contact(x,jacobian)
        ice=ice_soft_rows(raw_ice,self.ice_policy)
        rdata=np.r_[rdata,ice['residual']]
        phases['ice_geometry_and_jacobian_seconds']=time.perf_counter()-phase_start;phase_start=time.perf_counter()
        e=dict(raw_ice)
        neutral_margin,neutral_which=self.cavity_margins(neutral)
        dynamic_rows=base['compatibility_residual']*np.sqrt(2)
        dynamic_margin,dynamic_which=self.cavity_margins(dynamic_rows)
        foot_margins=np.r_[neutral_margin,dynamic_margin]
        e['margin_m']=foot_margins
        e['g']=foot_margins/NORMALIZATION
        e['weights']=np.r_[np.ones(self.cavity_rows_per_frame),np.repeat(
            np.r_[.01,np.full(self.body.n-2,.02),.01] if self.body.n>1 else [.02],self.cavity_rows_per_frame)]
        e['labels']=self.foot_row_labels.copy()
        if e['labels']!=self.row_labels:raise ValueError('Contact row identities changed inside inner solve')
        collar_unknown=0
        if jacobian or self.binding.collar_constraints_enabled:
            s=base['world'];local=np.einsum('nji,nkj->nki',s['Q'],s['native']-s['P'][:,None]);mid=np.arange(self.body.n)*5+2
            self._replay_samples=self.binding.surface.sample_vertices(local[mid],s['frames'][mid],self.binding.sample_ids,double_precision=True,**({} if 'hand_local_R' not in s else {'hand_local_R':s['hand_local_R'][mid]}))
            if self.binding.collar_constraints_enabled:
                collar_unknown=int(self.binding.collar_metrics(local[mid],s['frames'][mid],self._replay_samples)['unknown'].sum())
                collar_unknown+=int(self.binding.collar_metrics(self.binding.neutral_native,self.binding.neutral_frames,
                    self.binding.neutral_samples)['unknown'].sum())
        phases['cavity_and_collar_gate_seconds']=time.perf_counter()-phase_start;phase_start=time.perf_counter()
        if jacobian:
            if getattr(self,'optimize_shared',True):
                columns=self.body.execution.map_columns(lambda c:self.shared_column(c,x,base),range(17))
                shapeJ,shapeG=zip(*columns)
                shared_data=csr_matrix(np.stack(shapeJ,1));shared_g=csr_matrix(np.stack(shapeG,1))
            else:
                shared_data=csr_matrix((len(rdata),17));shared_g=csr_matrix((len(e['g']),17))
            self.binding.x=x[:17].copy()
            ice_body_J=diags(ice['derivative_per_margin_m']*NORMALIZATION)@raw_ice['body_jac'][ice['indices']]
            bodyJ=vstack((np.sqrt(.02)*base['jac'],csr_matrix((len(neutral)+len(prior),self.body.size)),ice_body_J),format='csr')
            Jdata=hstack((shared_data,bodyJ),format='csr')
            select=[]
            for i in range(self.body.n):
                select.extend(i*self.compatibility_width+12+f*8+dynamic_which[i,f] for f in range(2))
                if self.binding.collar_constraints_enabled:select.extend([i*self.compatibility_width+32,i*self.compatibility_width+33])
            compatibility_J=base['compatibility_jacobian']
            foot_body_J=vstack((csr_matrix((self.cavity_rows_per_frame,self.body.size)),
                compatibility_J[select]*(-.003*np.sqrt(2)/NORMALIZATION)),format='csr')
            Jg=hstack((shared_g,foot_body_J),format='csr')
        else:Jdata=Jg=None
        phases['shared_columns_and_sparse_seconds']=time.perf_counter()-phase_start;phase_start=time.perf_counter()
        motion=self.source_motion.evaluate(x[17:],jacobian)
        phases['source_motion_seconds']=time.perf_counter()-phase_start;phase_start=time.perf_counter()
        action=getattr(self,'action_factors',None)
        extra=action.evaluate(x[17:],jacobian) if action is not None else dict(residual=np.empty(0),jac=csr_matrix((0,self.size)),parts={},metrics={})
        phases['action_seconds']=time.perf_counter()-phase_start;phase_start=time.perf_counter()
        rdata=np.r_[rdata,motion['residual'],extra['residual']]
        if jacobian:
            motion_J=hstack((csr_matrix((len(motion['residual']),17)),motion['jac']),format='csr')
            Jdata=vstack((Jdata,motion_J,extra['jac']),format='csr')
        phases['motion_action_assembly_seconds']=time.perf_counter()-phase_start;phase_start=time.perf_counter()
        support=getattr(self,'support_image',None)
        support_value=support.evaluate(x[17:],jacobian) if support is not None else None
        if support_value is not None:
            rdata=np.r_[rdata,support_value['residual']]
            if jacobian:Jdata=vstack((Jdata,support_value['jac']),format='csr')
        phases['support_image_seconds']=time.perf_counter()-phase_start;phase_start=time.perf_counter()
        ral,Jal=self.al.residual_and_jacobian(e['g'],Jg,row_weights=e['weights'])
        r=np.r_[rdata,ral];J=vstack((Jdata,Jal),format='csr') if jacobian else None
        gate=physical_feasibility(e['margin_m'],tolerance_m=FEASIBILITY_TOLERANCE,row_labels=e['labels'])
        gate['scope']='foot_cavity_and_collar_only; ice excluded from feasibility'
        gate['collar_unknown_sections']=collar_unknown
        gate['ice_is_delivery_gate']=False
        if collar_unknown:gate['feasible']=False
        parts=dict(**ice['parts'],data=float(data@data/2),neutral_fit=float(neutral@neutral/2),shared_prior=float(prior@prior/2),contact_AL=float(ral@ral/2),
            source_motion=float(motion['residual']@motion['residual']/2),action=float(extra['residual']@extra['residual']/2))
        image_names=('body_pixels','visible_boot_edges','visible_blade_curves','boot_direction_endpoints')
        pose_names=('reference_deviation','correction_velocity','correction_acceleration','actual_root_and_correction_acceleration')
        v=dict(residual=r,jac=J,parts=parts,contact=e,raw_ice=raw_ice,ice=ice,gate=gate,
            image_cost=(1. if support_value is None else .02)*sum(base['parts'].get(name,0.) for name in image_names)+(0. if support_value is None else support_value['cost'])+extra.get('image_cost',0.),
            pose_cost=sum(base['parts'].get(name,0.) for name in pose_names),
            source_motion_metrics=motion['metrics'],action_metrics=extra['metrics'],
            raw_data_parts={**base['parts'],**motion['parts'],**extra['parts']})
        if support_value is not None:
            v['support_image_metrics']=support_value['metrics']
            v['parts']['support_image']=support_value['cost']
            v['raw_data_parts'].update(support_value['parts'])
            v['block_slices']={'support_image':slice(len(rdata)-len(support_value['residual']),len(rdata))}
        phases['AL_and_final_assembly_seconds']=time.perf_counter()-phase_start
        v['evaluation_phases']=phases
        v['body_derivative_phases']=dict(base.get('phases',{}))
        v['compatibility_execution']=base.get('compatibility_execution')
        self.cache_x=x.copy();self.cache=v
        self.log.write(json.dumps(dict(event='joint_evaluation',outer=self.outer,with_jacobian=jacobian,
            seconds=time.perf_counter()-self.start,evaluation_seconds=time.perf_counter()-tick,cost=float(r@r/2),
            parts=parts,physics_times=len(self.physics_times),constraints=len(e['g']),variables=self.size,
            evaluation_phases=phases,body_derivative_phases=v['body_derivative_phases'],
            compatibility_execution=v['compatibility_execution'],
            jacobian_nnz=None if J is None else J.nnz,body_execution_phases=dict(base.get('phases',{})),**gate))+'\n')
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
        return dict(self.reference.fixed_checkpoint_identity,**self.saved_base_layers,**getattr(self,'checkpoint_extra',{}),
            x=np.asarray(x).copy(),knots=self.body.state.knots,
            active=self.body.state.active,seconds=self.body.state.seconds,physics_times=self.physics_times,
            multiplier=self.al.multiplier,penalty=np.array(self.al.penalty),state_kind=np.array(self.state_kind),
            collar_constraints_enabled=np.array(self.binding.collar_constraints_enabled),
            ice_mode=np.array('finite_soft_objective'),ice_is_delivery_gate=np.array(False),
            al_row_scope=np.array('foot_cavity_and_collar_only'),
            ice_soft_policy_json=np.array(json.dumps(vars(self.ice_policy),sort_keys=True)))

    def solve(self,initial=None,run_policy=None):
        from .skate_run_policy import RunPolicy,Monitor,CandidateRetention,structural_progress_is_observable
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
        # World-root motion leaves foot-cavity/collar geometry unchanged.
        # Keep those diagnostics, but do not stop a root-only solve because
        # an internal-shoe quantity outside its active variables is constant.
        structural_progress=structural_progress_is_observable(self.body.state.active,
            optimize_shared=bool(getattr(self,'optimize_shared',True)))
        monitor=Monitor(run_policy or RunPolicy(),now=time.perf_counter(),
            violation_stagnation_enabled=structural_progress)
        write_json(self.output/'run_policy.json',monitor.record())
        x=np.r_[self.binding.x,np.zeros(self.body.size)] if initial is None else initial.copy()
        # No root-height restoration: flight/unknown must not be glued to ice.
        lo,hi=shared_bounds(self.binding)
        angular=np.tile(self.body.state.active>=3,self.body.state.m)
        lower=np.r_[lo,np.where(angular,-np.pi,-np.inf)];upper=np.r_[hi,np.where(angular,np.pi,np.inf)]
        # Local repairs may activate a subset of this same global curve.
        # Every other parameter is expanded from the saved immutable state.
        selection=np.asarray(getattr(self,'active_parameter_indices',
            np.arange(0 if getattr(self,'optimize_shared',True) else 17,self.size)))
        if (selection.ndim!=1 or selection.dtype.kind not in 'iu' or not len(selection)
                or np.any(selection<0) or np.any(selection>=self.size)
                or np.any(np.diff(selection)<=0)
                or (not self.optimize_shared and np.any(selection<17))):
            raise ValueError('Invalid explicit active parameter selection')
        self.parameter_selection=selection
        frozen_initial=x.copy()
        record_before_jacobian=bool(getattr(self,'record_acceptance_before_jacobian',False))
        last_linearization=None;last_recorded=x[selection].copy()
        def expand(z):
            full=frozen_initial.copy();full[selection]=z;return full
        write_json(self.output/'parameter_selection.json',dict(full_layout=self.size,active_variables=len(selection),
            active_state_columns=self.body.state.active.tolist(),state_kind=self.state_kind,
            body_support_seconds=list(self.body.state.seconds),selection=selection.tolist(),one_body_state=True,
            internal_temporal_freezes=hasattr(self,'active_parameter_indices'),
            local_control_seconds=getattr(self,'local_control_seconds',None),
            record_acceptance_before_jacobian=record_before_jacobian))
        implementation={}
        source_dir=self.output/'source_snapshot';source_dir.mkdir(exist_ok=True)
        for name in ('skate_integration.py','skate_binding.py','skate_physics.py','skate_template.py',
                     'skate_contact.py','contact_feasibility.py','footwear_layout.py','fullbody_fit.py',
                     'fullbody_state.py','boot_integration.py','body_execution.py',
                     'compact_body_derivatives.py','selected_surface_linearization.py',
                     'body_resident_torch.py','torch_foot_compatibility.py','body_foot_cuda_bridge.py','body_foot_cuda.cu',
                     'footwear_pixel_factors.py','skate_run_policy.py',
                      'skate_collar_factors.py','skate_checkpoint.py','opening_live.py',
                      'opening_motion.py','opening_timing.py','skating_action_factors.py',
                      'ice_soft.py','action_live.py','action_integration.py','root_trajectory.py','root_repair.py',
                      'support_anchor_factors.py','support_image.py','stage_quality.py','source_support_export.py',
                      'hand_state.py','rebuild_body_geometry.py','compact_body_torch.py','spin_grasp_factors.py',
                      'grasp_temporal_factors.py','spin_grasp_refinement.py'):
            path=Path(__file__).with_name(name);implementation[name]=digest(path)
            (source_dir/name).write_bytes(path.read_bytes())
        write_json(self.output/'kernel_manifest.json',implementation)
        write_json(self.output/'execution_policy.json',self.body.execution.record())
        write_json(self.output/'ice_soft_policy.json',dict(policy=vars(self.ice_policy),
            role='finite objective only; no multiplier, feasibility threshold or continuation'))
        class StopFit(Exception):pass
        retention=CandidateRetention()
        def observe_candidate(z,value,*,accepted=False):
            metrics={name:dict(unit='deg' if name.endswith('_deg') else 'm',value=v)
                for name,v in value.get('source_motion_metrics',{}).items()}
            unknown=value['gate'].get('collar_unknown_sections',0)
            failures=[dict(code='collar_unknown_sections',category='domain',count=unknown,
                message='Collar section unavailable; finite domain residual is not a penetration measurement.')] if unknown else []
            status=monitor.observe(now=time.perf_counter(),accepted=accepted,
                violation_m=value['gate']['max_violation_m'],image_cost=value['image_cost'],
                joint_cost=float(value['residual']@value['residual']/2),
                pose_cost=value['pose_cost'] if getattr(self.body.state,'use_pose_cost_guard',True) else None,
                quality_metrics=metrics,quality_failures=failures)
            retention.observe(z,value,step=self.accepted,decision=status,metrics=monitor.latest)
            self.binding.x=z[:17].copy()
            np.savez_compressed(self.output/'attempt.pending.npz',**self.checkpoint_fields(z))
            (self.output/'attempt.pending.npz').replace(self.output/'last_attempt.npz')
            if not status['quality_failed']:
                np.savez_compressed(self.output/'good.pending.npz',**self.checkpoint_fields(z))
                (self.output/'good.pending.npz').replace(self.output/'last_good.npz')
                np.savez_compressed(self.output/'accepted.pending.npz',**self.checkpoint_fields(z))
                (self.output/'accepted.pending.npz').replace(self.output/'last_accepted.npz')
            write_json(self.output/'candidate_selection.json',retention.record())
            # Initial-domain failure and solver-return observations must also
            # persist the actual policy state, not an empty pre-solve record.
            write_json(self.output/'run_policy.json',monitor.record())
            return status
        initial_value=self.evaluate(x,False)
        if getattr(self,'stage_quality',None) is not None:
            self.stage_quality.begin(x,initial_value)
        observe_candidate(x,initial_value)
        def save_candidate(reason):
            selected=retention.selected
            if selected is None:
                write_json(self.output/'candidate.json',dict(event='candidate_unavailable',stop=reason,
                    selection=retention.record(),candidate_checkpoint=None))
                raise RuntimeError('No quality-valid candidate; last_attempt retained for diagnosis')
            last_x=selected.x
            self.binding.x=last_x[:17].copy()
            self.cache=self.cache_x=None
            self.body.cache=self.body.cache_x=None
            self.body.binding_key=None
            last_value=self.evaluate(last_x,False)
            write_json(self.output/'execution_policy.json',self.body.execution.record())
            np.savez_compressed(self.output/'candidate.pending.npz',**self.checkpoint_fields(last_x))
            (self.output/'candidate.pending.npz').replace(self.output/'candidate.npz')
            record=dict(event='candidate_saved',stop=reason,candidate_checkpoint=str(self.output/'candidate.npz'),
                source_seconds=list(self.body.state.seconds),parts=last_value['parts'],**last_value['gate'],
                raw_data_parts=last_value['raw_data_parts'],numerical_convergence=False,user_accepted=False,
                acceptance_calibrated=False,run_policy=monitor.record(),state_kind=self.state_kind,
                selection=retention.record(),source_motion_metrics=last_value.get('source_motion_metrics',{}))
            record['stage_update']='updated' if self.accepted else 'no_update'
            record['quality_baseline_scope']='stage-local monitor; cross-stage comparison is separately recorded'
            record['pose_guard_scope']='source groups, not the mixed motion regularizer' if not getattr(self.body.state,'use_pose_cost_guard',True) else 'legacy stage-local pose-cost guard'
            if getattr(self,'stage_quality',None) is not None:
                self.stage_quality.finish(last_x,last_value,accepted_steps=self.accepted)
                record['cross_stage_quality']=self.stage_quality.record()
            self.log.write(json.dumps(record)+'\n');write_json(self.output/'candidate.json',record)
            return last_x,record
        jacobian_seconds=float(getattr(self,'jacobian_seconds_hint',0.))
        def guarded(z,jacobian=False):
            nonlocal jacobian_seconds,last_linearization
            # SciPy TRF requests a new Jacobian only after accepting its cached
            # fun state. Record it before spending a J that a budget may discard.
            if jacobian and record_before_jacobian and last_linearization is not None and not np.array_equal(z,last_linearization):
                full=expand(z)
                if self.cache_x is not None and np.array_equal(full,self.cache_x) and self.cache is not None:
                    accepted(z)
            now=time.perf_counter()
            if monitor.reserve_evaluation(now=now,estimated_seconds=jacobian_seconds if jacobian else 0.)['stop']:raise StopFit()
            from .absolute_motion_curve import RotationChartDomainError
            try:
                value=self.evaluate(expand(z),jacobian)
            except RotationChartDomainError as error:
                if jacobian:raise
                # SciPy TRF treats nonfinite trial residuals as rejected and
                # contracts its trust radius. Never accept/project/rewrite a
                # branch-invalid pose or replace the last accepted candidate.
                self.log.write(json.dumps(dict(event='trial_domain_rejected',domain='absolute_so3_log_chart',message=str(error),accepted=self.accepted))+'\n')
                return dict(residual=np.full_like(initial_value['residual'],np.inf))
            if jacobian:
                jacobian_seconds=max(jacobian_seconds,time.perf_counter()-now)
                last_linearization=np.array(z,copy=True)
            return value
        while True:
            # A supplied same-state seed J is still exact before the first
            # inner solve. Later AL/association phases must invalidate it.
            if not (self.record_acceptance_before_jacobian and self.outer==0):
                self.cache=None;self.cache_x=None
            self.outer+=1
            def accepted(reduced):
                nonlocal last_recorded
                if record_before_jacobian:
                    if np.array_equal(reduced,last_recorded):return
                    last_recorded=np.array(reduced,copy=True)
                z=expand(reduced)
                self.accepted+=1;v=self.evaluate(z,False)
                self.log.write(json.dumps(dict(event='accepted',outer=self.outer,iteration=self.accepted,
                    cost=float(v['residual']@v['residual']/2),parts=v['parts'],**v['gate']))+'\n')
                status=observe_candidate(z,v,accepted=True)
                if status['stop']:raise StopFit()
            try:
                from .body_linear_solver import sparse_direct_trf
                with sparse_direct_trf(self):
                    fit=least_squares(lambda z:guarded(z)['residual'],x[selection],jac=lambda z:guarded(z,True)['jac'][:,selection],
                        bounds=(lower[selection],upper[selection]),method='trf',tr_solver='lsmr',x_scale='jac',ftol=1e-7,xtol=1e-7,gtol=1e-4,
                        callback=accepted,tr_options=dict(atol=1e-7,btol=1e-7))
            except StopFit:
                return save_candidate(monitor.record())
            x=expand(fit.x);value=self.evaluate(x,False)
            status=observe_candidate(x,value)
            if status['stop']:return save_candidate(status)
            np.savez_compressed(self.output/f'outer_{self.outer:02d}.npz',x=x,multiplier=self.al.multiplier,
                penalty=self.al.penalty,physics_times=self.physics_times)
            record=dict(event='outer_complete',outer=self.outer,stop=str(fit.message),optimality=float(fit.optimality),
                penalty=self.al.penalty,parts=value['parts'],**value['gate'])
            self.log.write(json.dumps(record)+'\n');write_json(self.output/'latest.json',record)
            # One bounded inner solve. Ice diagnostics never trigger another solve,
            # an AL update, a denser feasibility gate, or prevent delivery.
            record.update(ice_is_delivery_gate=False,ice_mode='finite_soft_objective',
                source_seconds=self.body.state.seconds,accepted_iterations=self.accepted,
                numerical_convergence=bool(fit.optimality<=1e-4))
            write_json(self.output/'result.json',record)
            return save_candidate(dict(reason='inner_solver_return',message=str(fit.message),
                ice_is_delivery_gate=False))


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
    p.add_argument('--preserve-viewer-default',action='store_true',help='Register a candidate while retaining the current viewer default')
    args=p.parse_args(argv);project,root,out=map(Path,(args.project,args.root,args.output))
    # A replay request must fail before building models or writing run records.
    # Missing/mistyped checkpoints are never permission to start a new solve.
    if args.checkpoint is not None:
        if args.stage not in ('export','run'):
            p.error('--checkpoint is read-only; use --stage export or run')
        if args.base_checkpoint or args.warm_start:
            p.error('--checkpoint cannot be combined with a fitting seed')
    replay_path=Path(args.checkpoint) if args.checkpoint is not None else out/'fit/result.npz'
    replay_requested=args.stage=='export' or args.checkpoint is not None
    if replay_requested and not replay_path.is_file():
        raise FileNotFoundError(replay_path)
    replay_mode=replay_requested or (args.stage=='run' and replay_path.exists())
    base_path=Path(args.base_checkpoint or args.warm_start) if (args.base_checkpoint or args.warm_start) else None
    saved_path=replay_path if replay_mode else base_path
    if saved_path is not None:
        validate_checkpoint_identity(load(saved_path))
    print(json.dumps(dict(main_log=str(out/'fit_progress.jsonl'))),flush=True)
    ref,camera,info,catalog=build(project,root,out,args.rocker_ft)
    saved=None
    if saved_path is not None:
        saved=load(saved_path)
        validate_checkpoint_identity(saved,ref.fixed_checkpoint_identity)
    out.mkdir(parents=True,exist_ok=True)
    def delivery_options(kind):
        return dict(delivery_name=args.delivery_name or ('skate_s01_1103' if kind=='continuous_shot' else 'skate_contact_1103'),
            variant_id=args.variant_id or ('skate_s01_1103' if kind=='continuous_shot' else 'skate1103'),
            make_default=not args.preserve_viewer_default)
    # Keep the requested mode even if a file disappears after preflight:
    # load() must report that failure, not fall through into registration.
    if replay_mode:
        # Export consumes the actual fitted boundary and shared parameters.
        # It must not recalibrate the input or replace its initial scan.
        ref.binding.x=saved['x'][:17].copy()
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
    contract=dict(schema='rinkspace.body-skate-contact.v1',camera=info,parent=digest(root/'opening_timing_1103/hand_polish/fitted/result.npz'),
        observations_sha256=digest(obs),
        ice_soft_policy=json.loads((project/'events/ankara_2026/programmes/okada_mei_sp/ice_soft_1103.json').read_text(encoding='utf-8')),
        ice_mode='finite_soft_objective',ice_is_delivery_gate=False,
        legacy_raw_diagnostics=dict(semantic_allowance_m=ALLOWANCE,support_band_m=SUPPORT_BAND,
            reference_band_is_not_a_fit_target=True),
        structural_diagnostics=dict(tolerance_m=FEASIBILITY_TOLERANCE,scope='foot_cavity_and_collar_only'),
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
        ref.binding.x=saved['x'][:17].copy()
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
