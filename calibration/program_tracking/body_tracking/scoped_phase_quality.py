"""Source phase objectives and domains for the existing metric step."""
import numpy as np
from scipy.sparse import csr_matrix, diags, hstack, vstack

from .phase_motion_quality import flight_stencil_rows
from .programme_joint_limits import NAMES
from .programme_source_step import SourceMargins
from .rebuild_prepare import digest, load
from .skate_contact import trapezoid_time_weights


class ScopedBodyQuantiles:
    """The same local source-percentile populations checked on trial pixels."""

    def __init__(self,guard,value,selected,project_pixels,*,protect_current=False,stable_membership=False):
        from .programme_guard_proposals import recover_body_pixel_jacobian

        self.problem=guard.problem
        self.project_pixels=project_pixels
        body=self.problem.body
        self.rows=body.frames[body.of]
        observed_frames=np.unique(self.rows)
        state=body.state
        knots=np.unique((selected-17)//69)
        moving=observed_frames[np.asarray(state.basis(guard.ref.times[observed_frames])[:,knots].getnnz(axis=1)>0)]
        self.points=np.flatnonzero(np.isin(self.rows,moving))
        self.frames=self.rows[self.points]
        self.ids=body.model_oj[self.points]
        self.observed=body.obs['uv'][self.frames,body.oj[self.points]]
        result=body.cache['uv'][body.of,body.layout.body_landmarks[body.model_oj]]
        self.fixed_errors=np.linalg.norm(result-body.obs['uv'][self.rows,body.oj],axis=1)
        self.derivative=recover_body_pixel_jacobian(self.problem,value,self.points)
        self.selected=selected
        self.movable=(np.diff(self.derivative[:,selected].indptr).reshape(-1,2)>0).any(axis=1)
        cfg=guard.policy
        torso=np.isin(body.oj,(5,6,11,12))
        self.groups=[]
        self.prefix=None if not stable_membership else {}
        for window,native,_,_ in guard.scope:
            if not native[moving].any():
                continue
            core=(guard.pts>=window[0])&(guard.pts<=window[1])
            for label,mask in (('',core[self.rows]),('_full_support',native[self.rows])):
                for key,keep,quantile,budget in (
                    ('body_median',mask,50,cfg['body_median_increase_px']),
                    ('body_p90',mask,90,cfg['body_p90_increase_px']),
                    ('torso_p90',mask&torso,90,cfg['torso_p90_increase_px'])):
                    active=np.flatnonzero(keep)
                    if not len(active):
                        continue
                    target=np.percentile(guard.anchor['pixel'][active],quantile)+budget-.004
                    if protect_current:
                        target=min(target,np.percentile(self.fixed_errors[active],quantile))
                    name=f'phase/source/{window[0]}-{window[1]}/{key}{label}'
                    self.groups.append((name,active,quantile,target))
                    if self.prefix is not None:
                        initial=self.fixed_errors[active]
                        order=np.argsort(initial,kind='stable')
                        threshold=percentile_prefix(initial[order],quantile,target)
                        for row,cap in zip(active[order],threshold):
                            before=self.prefix.get(row)
                            if before is None or cap<before[0]:
                                self.prefix[row]=(cap,name)
        self.local_index=np.full(len(self.rows),-1,int)
        self.local_index[self.points]=np.arange(len(self.points))

    def blocks(self,x,*,conservative=False):
        pixels=self.project_pixels(self.problem,x,self.frames,self.ids)
        delta=pixels-self.observed
        if conservative and self.prefix is not None:
            rows=np.array([row for row in self.prefix if self.local_index[row]>=0],dtype=int)
            if not len(rows):
                return SourceMargins(csr_matrix((0,len(x))),np.empty(0),[],[])
            local=self.local_index[rows]
            radius,names=zip(*(self.prefix[i] for i in rows),strict=True)
            labels=[f'{name}/prefix_row_{self.guard_pts(row)}_{row}' for row,name in zip(rows,names,strict=True)]
            J=self.derivative[(2*local[:,None]+np.arange(2)).ravel()]
            ball=dict(source_id='phase/source/rank_prefix',jacobian=J,offset=delta[local],
                      radius=np.asarray(radius),labels=labels)
            return SourceMargins(csr_matrix((0,len(x))),np.empty(0),[],[ball])
        errors=self.fixed_errors.copy()
        errors[self.points]=np.linalg.norm(delta,axis=1)
        matrix,slack,labels=[],[],[]
        for name,points,quantile,cap in self.groups:
            order=np.argsort(errors[points],kind='stable')
            index=(len(points)-1)*quantile/100
            lo,hi=int(np.floor(index)),int(np.ceil(index))
            weights=(1.-(index-lo),index-lo)
            source=points[order[[lo,hi]]]
            jac=csr_matrix((1,len(x)))
            for i,weight in zip(source,weights,strict=True):
                k=self.local_index[i]
                if weight and k>=0 and errors[i]>1.e-12:
                    jac += csr_matrix(weight*delta[k:k+1]/errors[i])@self.derivative[2*k:2*k+2]
            margin=cap-float(errors[source]@weights)
            if margin<0. and not jac[:,self.selected].nnz:
                # An untouched observation can become the percentile's active
                # sample after the other pixels move. Moving that sample is
                # impossible. Move the nearest eligible rows across the bound
                # so the actual percentile is again supported below the cap.
                hot=points[order]
                local=self.local_index[hot]
                use=(local>=0)&self.movable[np.maximum(local,0)]&(errors[hot]>cap)
                needed=hi+1-int(np.count_nonzero(errors[points]<=cap))
                candidates=hot[use][:needed]
                if len(candidates)<needed:
                    raise ArithmeticError('No affected rows can restore '+name)
                for i in candidates:
                    k=self.local_index[i]
                    matrix.append(csr_matrix(delta[k:k+1]/errors[i])@self.derivative[2*k:2*k+2])
                    slack.append(cap-errors[i]);labels.append(f'{name}/affected_row_{self.guard_pts(i)}')
                continue
            matrix.append(jac)
            slack.append(margin)
            labels.append(name)
        return SourceMargins(vstack(matrix,format='csr'),np.asarray(slack),labels)

    def guard_pts(self,index):
        return int(self.problem.reference.p['pts'][self.rows[index]])


def percentile_prefix(sorted_errors, quantile, cap):
    """Sufficient pivot bounds for the exact interpolated percentile.

    If the original first lo+1 observations stay below the lower pivot and
    the original next observation stays below the upper pivot, the actual
    percentile can only fall, regardless of intervening rank crossings.
    """
    index=(len(sorted_errors)-1)*quantile/100.
    lo,hi=int(np.floor(index)),int(np.ceil(index))
    offset=cap-np.percentile(sorted_errors,quantile)
    return np.maximum(0.,np.r_[np.full(lo+1,sorted_errors[lo]+offset),
                                np.full(hi-lo,sorted_errors[hi]+offset)])


class ScopedCavityNorm:
    """Repeat the existing per-frame cavity calculation only on active support."""

    def __init__(self,problem,value,guard,selected,x):
        from .skate_physics import NORMALIZATION

        self.problem=problem
        self.normalization=NORMALIZATION
        self.width=len(guard.kinds)
        self.base=guard._rows(guard._term(value))
        body=problem.body
        self.mask=np.where(np.asarray(body.state.basis(guard.times)[:,np.unique((selected-17)//69)].getnnz(axis=1)>0))[0]
        self.times=guard.times[self.mask]
        rows=self.width*(self.mask[:,None]+1)+np.arange(self.width)
        self.shift=problem.al.multiplier[rows]/np.sqrt(problem.al.penalty)
        self.weight=np.sqrt(value['contact']['weights'][rows])
        term=guard._term(value)
        self.derivative=[term.jacobian[rows[:,j]] for j in range(self.width)]
        actual=self.values(x)
        np.testing.assert_allclose(actual,self.base[self.mask+1],rtol=2.e-6,atol=5.e-7)
        self.constant_sum=(self.base**2).sum(axis=0)-(self.base[self.mask+1]**2).sum(axis=0)
        self.caps=np.maximum(guard.caps-2.e-6,0.)
        self.names=guard.kinds

    def values(self,x):
        from .skate_physics import fk_state

        p=self.problem
        state=p.body.state
        pose=fk_state(p.reference,p.reference.at(self.times),state.increments(x[17:],self.times))
        hand={} if 'hand_local_R' not in pose else dict(hand_local_R=pose['hand_local_R'])
        sample=p.binding.surface.sample_vertices(pose['native'],pose['frames'],p.binding.sample_ids,
                                                  double_precision=True,**hand)
        raw=p.binding.compatibility_rows(pose['native'],pose['frames'],sample,
            skate_relative_R=pose['skate_relative_R'],**hand)
        margin=p.cavity_margins(raw)[0].reshape(-1,self.width)
        return self.weight*np.maximum(0.,self.shift-np.sqrt(p.al.penalty)*margin/self.normalization)

    def cuts(self,x):
        vals=self.values(x)
        norms=np.sqrt(self.constant_sum+(vals**2).sum(axis=0))
        rows,excess,labels=[],[],[]
        for k in np.flatnonzero(norms>self.caps):
            rows.append(csr_matrix(vals[None,:,k]/norms[k])@self.derivative[k])
            excess.append(norms[k]-self.caps[k]);labels.append('phase/cavity/'+self.names[k])
        return (vstack(rows,format='csr') if rows else csr_matrix((0,len(x))),np.asarray(excess),labels)


def skate_origin_steps(problem, x, ends, *, jacobian=False):
    """Adjacent native displacement of the actual attached skate origins, in metres."""
    from .programme_whole_skate import native_point_derivative
    from .skate_physics import fk_state

    times=problem.reference.times
    samples, pairs=np.unique(ends[:,None]+[-1,0],return_inverse=True)
    pairs=pairs.reshape(-1,2)
    clock=times[samples]
    state=problem.body.state
    ref=problem.reference
    base=ref.at(clock)
    delta=state.increments(x[17:],clock)
    pose=fk_state(ref,base,delta)
    ratio=ref.binding.ratio
    steps=(pose['O'][pairs[:,1]]-pose['O'][pairs[:,0]])*ratio
    if not jacobian:
        return steps,None
    derivatives=[]
    differential=np.broadcast_to(np.eye(3),(len(clock),1,1,3,3))
    for foot in range(2):
        def origin(pose):
            return pose['O'][:,None,foot:foot+1]

        body,shoe=native_point_derivative(problem,base,delta,origin,foot,differential)
        derivative=delta.chain(body[:,0,0])+state.shoe_chain(shoe[:,0,0],x[17:],clock)
        derivatives.append(derivative)
    combined=vstack(derivatives,format='csr')
    index=3*(pairs[:,:,None]+len(clock)*np.arange(2))[...,None]+np.arange(3)
    difference=combined[index[:,1].ravel()]-combined[index[:,0].ravel()]
    return steps,hstack((csr_matrix((steps.size,17)),ratio*difference),format='csr')


class LandingTransition:
    """Transition curvature and skate speed on the normal continuous fit state."""

    def __init__(self,problem,com,x,specification):
        self.problem=problem
        self.com=com
        lo,hi=specification['acceleration_window_pts']
        pts=com.source_pts_float[1:-1]
        self.rows=np.flatnonzero((pts>=lo-1.e-7)&(pts<=hi+1.e-7))
        if not len(self.rows) or any(np.intersect1d(self.rows,rows).size for rows in com.flight_rows.values()):
            raise ValueError('Landing acceleration window must be supported outside the flight stencil interiors')
        # The measure is shared with the full objective; the finite transition
        # penalty makes no assertion of gravity, zero velocity or ice contact.
        scale=float(specification['acceleration_scale_mps2'])
        if not np.isfinite(scale) or scale<=0.:
            raise ValueError('Landing acceleration scale must be positive and finite')
        self.acceleration_weight=np.sqrt(trapezoid_time_weights(com.times)[1:-1][self.rows])/scale
        self.acceleration_cap=None
        if specification.get('protect_current_acceleration'):
            before,_=com.acceleration_rows(x[17:],self.rows)
            self.acceleration_cap=np.linalg.norm(before*self.acceleration_weight[:,None])
        lo,hi=specification['skate_step_end_pts']
        native=problem.reference.p['pts']
        self.ends=np.flatnonzero((native>=lo)&(native<=hi))
        if not len(self.ends) or not self.ends[0]:
            raise ValueError('Expected native step ends with preceding exposures')
        before,_=skate_origin_steps(problem,x,self.ends)
        step_cap=np.linalg.norm(before,axis=-1)
        # Keep each foot's observed motion available. The envelope constrains
        # only the large steps and cannot increase its phase peak.
        desired=float(specification['skate_step_cap_m'])
        floor=float(specification['skate_step_floor_m'])
        if not np.isfinite([desired,floor]).all() or not 0.<=floor<=desired:
            raise ValueError('Expected finite ordered skate-step distances')
        self.step_cap=np.maximum(floor,np.minimum(step_cap,desired))
        self.step_labels=[f'phase/landing/skate_origin_step/{native[i]}/foot_{foot}'
                          for i in self.ends for foot in range(2)]
        self.step_weight=float(specification.get('skate_step_merit_per_m',0.))
        self.step_merit_reference=floor

    def blocks(self,x):
        acceleration,j=self.com.acceleration_rows(x[17:],self.rows,jacobian=True)
        residual=[(acceleration*self.acceleration_weight[:,None]).ravel()]
        derivatives=[j.multiply(self.acceleration_weight.repeat(3)[:,None]).tocsr()]
        norms=[]
        if self.acceleration_cap is not None:
            label='phase/landing/acceleration_norm'
            norms.append(dict(source_id=label,jacobian=derivatives[0],offset=residual[0][None,:],
                              radius=np.array([self.acceleration_cap]),labels=[label]))
        steps,j=skate_origin_steps(self.problem,x,self.ends,jacobian=True)
        steps=steps.reshape(-1,3)
        length=np.linalg.norm(steps,axis=1)
        active=length>self.step_merit_reference
        unit=steps[active]/length[active,None]
        ids=np.flatnonzero(active)
        row_gradient=csr_matrix((unit.ravel(),(np.repeat(np.arange(len(ids)),3),
                                (3*ids[:,None]+np.arange(3)).ravel())),shape=(len(ids),steps.size))@j
        residual.append((length[active]-self.step_merit_reference)*self.step_weight)
        derivatives.append(row_gradient*self.step_weight)
        norms.append(dict(source_id='phase/landing/skate_origin_step',jacobian=j,offset=steps,
                          radius=self.step_cap.ravel(),labels=self.step_labels))
        return norms,np.concatenate(residual),vstack(derivatives,format='csr')


class ScopedPhaseQuality:
    """Progress toward a native reference without weakening production guards."""

    def __init__(self, guard, value, x, specification, selected, image_projection,
                 body_projection,cavity_guard):
        self.shoes = specification.get('fit_shoe_points',[])
        if any(row.get('role') != 'fit' for row in self.shoes):
            raise ValueError('Only fitted shoe evidence belongs in the direction')
        self.guard = guard
        self.com = guard.com
        self.image_projection = image_projection
        self.projection_limits=ScopedBodyQuantiles(guard,value,selected,body_projection,
            protect_current=specification.get('protect_current_body_quantiles',False),
            stable_membership=specification.get('stable_source_percentile_domains',False))
        self.cavity=(ScopedCavityNorm(guard.problem,value,cavity_guard,selected,x)
                     if cavity_guard is not None else None)
        self.flight, self.dense, self.fd_checks = [], [], []
        named = {b.name:b for b in value['weighted_blocks']}
        episodes = {e['episode_id']:e for e in guard.ref.source_document['source_episode_review']['episodes']}
        pts = guard.pts
        fraction = float(specification.get('progress_fraction',.85))
        if not 0.<fraction<1.:
            raise ValueError('Phase progress fraction must be strictly between zero and one')
        self.fraction = fraction
        for goal in specification['flight_goals']:
            name = goal['episode']
            native = flight_stencil_rows(pts, episodes[name])
            w = np.sqrt(trapezoid_time_weights(guard.ref.times)[1:-1][native])
            a,j = self.com.acceleration_rows(x[17:],native,native=True,jacobian=True)
            residual = (a-self.com.gravity)*w[:,None]
            rms = float(goal['native_rms_mps2'])
            radius = (min(rms*np.linalg.norm(w),np.linalg.norm(residual)) if goal.get('preserve_current')
                      else max(rms*np.linalg.norm(w), fraction*np.linalg.norm(residual)))
            self.flight.append(('phase/native_gravity/'+name,native,w,radius,True))
            # These are the actual main objective's weighted rows. For the
            # native comparison also check the precise selected FK derivative.
            block = named['episode/'+name+'/flight_dense_com_acceleration']
            self.dense.append((name,block))
            if goal.get('preserve_current'):
                w_dense=self.com.flight_weights[name]
                self.flight.append(('phase/dense_gravity/'+name,self.com.flight_rows[name],w_dense,
                                    np.linalg.norm(block.residual),False))
            for native_grid,rows,slope,weights in (
                    (True,native,j,w),
                    (False,self.com.flight_rows[name],block.jacobian,self.com.flight_weights[name])):
                j_selected = slope[:,selected].tocsc()
                for channel in ('root','articulation'):
                    use = ((selected-17)%69<3) == (channel=='root')
                    strength = np.asarray(j_selected.multiply(j_selected).sum(axis=0)).ravel()*use
                    if not strength.any():
                        continue
                    k = int(strength.argmax())
                    column = selected[k]; h=2.e-6
                    p = x.copy();m = x.copy();p[column]+=h;m[column]-=h
                    ap,_ = self.com.acceleration_rows(p[17:],rows,native=native_grid)
                    am,_ = self.com.acceleration_rows(m[17:],rows,native=native_grid)
                    fd = ((ap-am)/(2*h)).ravel()
                    if not native_grid:
                        fd *= weights.repeat(3)
                    analytic = j_selected[:,k].toarray().ravel()
                    np.testing.assert_allclose(fd,analytic,rtol=2.e-5,atol=3.e-4)
                    i = int(np.argmax(abs(analytic)))
                    clock = pts if native_grid else self.com.source_pts_float
                    self.fd_checks.append(dict(episode=name, native=native_grid, column=int(column),
                        channel=channel,source_pts=float(clock[rows[i//3]+1]),analytic=float(analytic[i]),
                        finite_difference=float(fd[i]),maximum_error=float(np.max(abs(fd-analytic))),
                        rows=len(analytic)))
        self.anatomy = None
        if domain := specification.get('joint_reference'):
            if digest(domain['path'])!=domain['sha256']:
                raise ValueError('Reference anatomy changed')
            reference = load(domain['path'])
            np.testing.assert_array_equal(pts,reference['source_pts'])
            np.testing.assert_array_equal(NAMES,reference['joint_names'])
            names = [NAMES.index(name) for name in domain['joints']]
            limits = np.asarray(domain['window_pts'])
            rows = np.flatnonzero((pts>=limits[0])&(pts<=limits[1]))
            original = reference['normalized_violation'][rows][:,names]
            current = guard.joints.last_values[rows][:,names]
            goal = np.maximum(original,0.)
            progress = float(domain.get('progress_fraction',fraction))
            if not 0.<=progress<1.:
                raise ValueError('Joint progress fraction outside the phase proposal domain')
            if domain.get('preserve_current'):
                goal=np.minimum(goal,current)
            caps = goal+np.maximum(current-goal,0.)*progress+3.e-6
            goal_guard = bool(domain.get('require_progress',True))
            if domain.get('preserve_current'):
                knots=np.unique((selected-17)//69)
                supported=np.asarray(guard.state.basis(guard.joints.times[rows])[:,knots].getnnz(axis=1)>0)
                self.anatomy = rows[supported],names,goal[supported],caps[supported],goal_guard
            else:
                self.anatomy = rows,names,goal,caps,goal_guard
        self.sigma = np.asarray([row['sigma_px'] for row in self.shoes],float)
        self.fit_strength = float(specification.get('shoe_fit_strength',1.))
        self.merits = tuple(float(specification.get(key,default)) for key,default in (
            ('native_flight_merit',3.), ('dense_flight_merit',3.), ('joint_merit',35.)))
        self.landing=None
        if transition:=specification.get('landing_transition'):
            self.landing=LandingTransition(guard.problem,self.com,x,transition)
        self.local_rotation_hints=None
        if hints:=specification.get('local_rotation_hints'):
            stamps=[row['pts'] for row in hints]
            rows=np.searchsorted(pts,stamps)
            np.testing.assert_array_equal(pts[rows],stamps)
            times=guard.ref.times[rows]
            tracks=np.asarray([row['track'] for row in hints],int)
            if np.any((tracks<0)|(tracks>=22)):
                raise ValueError('Rotation hints must name actual body tracks')
            ids=3+3*tracks[:,None]+np.arange(3)
            delta=guard.state.increments(x[17:],times)
            initial=delta[np.arange(len(times))[:,None],ids]
            offset=np.deg2rad([row['offset_deg'] for row in hints])
            weight=1./np.deg2rad([row['scale_deg'] for row in hints])
            self.local_rotation_hints=times,ids,initial+offset,weight

    def blocks(self,x,*,proposal_merit=False):
        matrices, limits, labels, norms = [], [], [], []
        residual, derivative = [], []
        projection=self.projection_limits.blocks(x,conservative=proposal_merit)
        P,slack,names=projection
        matrices.append(P);limits.append(slack);labels.extend(names)
        norms.extend(projection.image_balls)
        native_merit, dense_merit, joint_merit = self.merits
        for label,rows,w,radius,native in self.flight:
            a,j = self.com.acceleration_rows(x[17:],rows,native=native,jacobian=True)
            bias = ((a-self.com.gravity)*w[:,None]).ravel()
            j = diags(w.repeat(3))@j
            norms.append(dict(source_id=label,jacobian=j,offset=bias[None,:],
                              radius=np.asarray([radius]),labels=[label]))
            if proposal_merit and native:
                derivative.append(j*native_merit)
                residual.append(bias*native_merit)
        if self.landing is not None:
            balls,r,j=self.landing.blocks(x)
            norms.extend(balls)
            if proposal_merit:
                residual.append(r)
                derivative.append(j)
        if proposal_merit:
            for _,block in self.dense:
                derivative.append(block.jacobian*dense_merit)
                residual.append(block.residual*dense_merit)
        if self.anatomy is not None:
            rows,names,goal,caps,guard = self.anatomy
            raw,j = self.guard.joints.proposal_linearization(x[17:],rows=rows)
            picks = (3*(len(NAMES)*np.arange(len(rows))[:,None]+np.asarray(names))[...,None]+np.arange(3)).ravel()
            signed = raw[picks]
            j = hstack((csr_matrix((len(picks),17)),j[picks]),format='csr')
            if guard:
                matrices.append(j);limits.append(caps.ravel()-signed)
                labels.extend(f'phase/joint/{self.guard.pts[row]}/{NAMES[name]}/{axis}'
                              for row in rows for name in names for axis in range(3))
            if proposal_merit:
                active = signed>goal.ravel()
                derivative.append(j[active]*joint_merit)
                residual.append((signed[active]-goal.ravel()[active])*joint_merit)
        if proposal_merit and self.shoes:
            uv,j = self.image_projection(self.guard.problem,x,self.shoes,jacobian=True)
            centres = np.asarray([row['uv'] for row in self.shoes])
            weight = np.repeat(self.fit_strength/self.sigma,2)
            derivative.append(j.multiply(weight[:,None]))
            residual.append((uv-centres).ravel()*weight)
        if proposal_merit and self.local_rotation_hints is not None:
            times,ids,target,weight=self.local_rotation_hints
            delta=self.guard.state.increments(x[17:],times)
            result=delta[np.arange(len(times))[:,None],ids]
            slots=np.eye(69)[ids]*weight[:,None,None]
            j=delta.chain(slots)
            derivative.append(hstack((csr_matrix((result.size,17)),j),format='csr'))
            residual.append(((result-target)*weight[:,None]).ravel())
        width = len(x)
        empty = csr_matrix((0,width))
        return (SourceMargins(vstack(matrices,format='csr') if matrices else empty,
                    np.concatenate(limits) if limits else np.empty(0),labels,norms),
            np.concatenate(residual) if residual else np.empty(0),
            vstack(derivative,format='csr') if derivative else empty)

    def cuts(self,x):
        domains,_,_ = self.blocks(x)
        matrix,slack,names = domains
        rows,excess,labels = [],[],[]
        for i in np.flatnonzero(slack < -5.e-7):
            rows.append(matrix[i]);excess.append(-slack[i]);labels.append(names[i])
        for group in domains.image_balls:
            lengths=np.linalg.norm(group['offset'],axis=1)
            for i in np.flatnonzero(lengths-group['radius']>1.e-7):
                r=group['offset'][i]
                start=i*len(r)
                rows.append(csr_matrix(r[None,:]/lengths[i])@group['jacobian'][start:start+len(r)])
                excess.append(lengths[i]-group['radius'][i]);labels.append(group['labels'][i])
        if self.cavity is not None:
            j,off,names=self.cavity.cuts(x)
            if len(names):
                rows.append(j);excess.extend(off);labels.extend(names)
        return (vstack(rows,format='csr') if rows else csr_matrix((0,len(x))),np.asarray(excess),labels)
