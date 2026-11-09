"""A bounded full-state trial conditioned on all existing source guards."""
import numpy as np
from scipy.sparse import csr_matrix, vstack


class CompleteNonlinearGuardChecks(dict):
    """All original nonlinear guards have been evaluated, before any ready filtering."""


class SourceMargins(tuple):
    """Legacy three-item row tuple, plus affine norm balls with offsets (N, D).

    Each ball has N radii and labels, and N * D Jacobian rows in offset order.
    Pixel balls retain the legacy defaults for unit and guard_margin_px.
    """
    def __new__(cls, matrix, slack, labels, image_balls=()):
        value=super().__new__(cls,(matrix,slack,labels));value.image_balls=list(image_balls)
        return value


def contour_margins(value, contour, limits, contour_weight=1., *, terms=None, row_metrics=None):
    """Linearize the same outside-band p95 checked by the nonlinear guard.

    Residual radial derivatives recover the actual block scale, including an
    unassigned shoe branch. Positive margins are kept before a trial reaches
    them; no source identity or original acceptance limit is changed.
    """
    if contour is None:return None
    rows=value['action_metrics'].get('visible_runner_contours',[]) if row_metrics is None else row_metrics
    terms=next(b for b in value['weighted_blocks']if b.name=='full_source_skate_structure') if terms is None else terms
    if len(terms.residual)<2*len(contour.point_rows) or len(rows)!=len(contour.rows):
        raise ValueError('Source protection requires exact contour row order')
    gradient=[];slack=[];labels=[];balls=[]
    for i,row in enumerate(rows):
        if row['id']!=contour.rows[i]['id']:raise ValueError('Source contour/J identity differs')
        old,radius=limits[row['id']];distances=np.asarray(row['outside_visible_band_px'])
        current=float(np.percentile(distances,95));margin=old+.25*radius-current
        raw_jac=getattr(contour,'proposal_pixel_jacobian',None)
        if raw_jac is not None:
            points=np.flatnonzero(contour.point_rows==i)
            if 'model_closest_uv' in row:
                np.testing.assert_allclose(contour.proposal_pixel_prediction[points],row['model_closest_uv'],
                    rtol=0,atol=1e-5,err_msg='Raw source prediction identity changed')
            delta=contour.proposal_pixel_prediction[points]-contour.source_uv[points]
            distance=np.linalg.norm(delta,axis=1)
            balls.append(dict(source_id=row['id'],model_foot=contour.rows[i].get('foot'),guard_margin_px=margin,jacobian=raw_jac[(2*points[:,None]+np.arange(2)).ravel()].tocsr(),
                offset=delta.copy(),radius=np.maximum(distance,radius)+.8*margin,
                labels=[row['id']+'/point_'+str(int(k))for k in points]))
            np.testing.assert_allclose(np.maximum(distance-radius,0.),distances,rtol=0,atol=1e-5,
                err_msg='Raw source proposal geometry must match the guarded baseline')
            keep=distance>1e-12;chosen=points[keep]
            if len(chosen):
                radial=delta[keep]/distance[keep,None]
                projection=csr_matrix((radial.ravel(),(np.repeat(np.arange(len(chosen)),2),
                    (2*chosen[:,None]+np.arange(2)).ravel())),shape=(len(chosen),raw_jac.shape[0]))
                point_rows=projection@raw_jac;nonzero=np.diff(point_rows.indptr)>0
                if nonzero.any():
                    gradient.append(point_rows[nonzero])
                    slack.extend((.8*margin+np.maximum(radius-distance[keep],0.))[nonzero])
                    labels.extend(row['id']+'/point_'+str(int(k))for k in chosen[nonzero])
        if current<=0.:continue
        order=np.argsort(distances,kind='stable');coordinate=.95*(len(order)-1)
        lo,hi=int(np.floor(coordinate)),int(np.ceil(coordinate));fraction=coordinate-lo
        points=np.flatnonzero(contour.point_rows==i);row_jac=csr_matrix((1,terms.jacobian.shape[1]))
        for q,weight in ((lo,1-fraction),(hi,fraction)):
            index=order[q];outside=distances[index]
            if outside<=0. or weight==0.:continue
            k=points[index];pair=slice(2*k,2*k+2);norm=np.linalg.norm(terms.residual[pair])
            if norm==0.:continue
            u=np.hypot(1.,outside/(2*radius));derivative=norm/outside*(u+1)/(2*u)
            radial=terms.residual[pair]/(norm*derivative)
            row_jac+=csr_matrix(radial[None])@terms.jacobian[pair]*weight
        if row_jac.nnz:
            gradient.append(row_jac);slack.append(.8*margin);labels.append(row['id'])
    if not gradient and not balls:return None
    matrix=vstack(gradient,format='csr')if gradient else csr_matrix((0,terms.jacobian.shape[1]))
    return SourceMargins(matrix,np.asarray(slack),labels,balls)


def all_source_margins(problem,value,limits,turn_baseline,torso_baseline,observation_mask):
    """Existing assigned/unassigned contours and per-turn body image guards."""
    from types import SimpleNamespace
    from .trajectory_probe import GROUPS
    pieces=[];actions=problem.action_factors
    unassigned_rows={row['id']:row for row in value['action_metrics'].get('visible_unassigned_skate_contours',[])}
    if actions.contour is not None:pieces.append(contour_margins(value,actions.contour,limits))
    for name,factor in [('unassigned_source_skate_sets',actions.unassigned)]+[
            ('additional_unassigned_source_'+key,f)for key,f in actions.extra_unassigned_episodes]:
        if factor is None:continue
        terms=next(b for b in value['weighted_blocks']if b.name==name)
        if factor.episode:
            selected=[factor.episode_factors[factor.episode_choice]]
        else:
            selected=[branches[row['branch']]for (_,branches),row in zip(factor.groups,value['action_metrics']['source_unassigned_associations'])]
        offset=0
        for contour in selected:
            count=len(contour.last['residual'])
            piece=SimpleNamespace(residual=terms.residual[offset:offset+count],jacobian=terms.jacobian[offset:offset+count])
            pieces.append(contour_margins(value,contour,limits,terms=piece,row_metrics=[unassigned_rows[row['id']]for row in contour.rows]))
            offset+=count
        if offset!=len(terms.residual):raise ValueError('Unassigned source rows changed layout')
    body=problem.body;frames=body.frames[body.of];joints=body.oj
    model_joints=getattr(body,'model_oj',joints)
    projected=body.cache['uv'][body.of,body.layout.body_landmarks[model_joints]]
    delta=projected-body.obs['uv'][frames,joints];pixels=np.linalg.norm(delta,axis=-1)
    usable=np.isfinite(pixels)
    if observation_mask is not None:usable&=observation_mask[frames,joints]
    masks=[('torso',usable&np.isin(joints,GROUPS['torso']),torso_baseline)]
    for factor in actions.existing.turn_factors:
        episode=factor.episode['episode_id'];inside=np.isin(frames,factor.rows)&usable
        for group in ('torso','lower_limbs'):
            name=episode+'/'+group
            if name in turn_baseline:masks.append(('turn_image/'+name,inside&np.isin(joints,GROUPS[group]),turn_baseline[name]))
    terms=next(b for b in value['weighted_blocks']if b.name=='body_pixels')
    gradient=[];slack=[];labels=[];body_groups=[]
    for name,mask,baseline in masks:
        indices=np.flatnonzero(mask)
        if not len(indices):continue
        order=indices[np.argsort(pixels[indices],kind='stable')];coordinate=.95*(len(order)-1)
        lo,hi=int(np.floor(coordinate)),int(np.ceil(coordinate));fraction=coordinate-lo
        current=float(np.percentile(pixels[indices],95));old=baseline['pixel_p95']
        allowance=max(2.,.1*old);row=csr_matrix((1,terms.jacobian.shape[1]))
        body_groups.append(dict(name=name,indices=indices.copy(),margin=old+allowance-current))
        for q,weight in ((lo,1-fraction),(hi,fraction)):
            k=order[q];distance=pixels[k]
            if distance<=0 or weight==0:continue
            pair=slice(2*k,2*k+2);norm=np.linalg.norm(terms.residual[pair])
            sigma=body.obs['sigma_px'][frames[k],joints[k]]
            if norm==0:continue
            u=np.hypot(1.,distance/(2*sigma));derivative=norm/distance*(u+1)/(2*u)
            radial=terms.residual[pair]/(norm*derivative)
            row+=csr_matrix(radial[None])@terms.jacobian[pair]*weight
        if row.nnz:
            gradient.append(row);slack.append(.8*(old+allowance-current));labels.append(name)
        # Percentile identities can change during the same nonlinear trial.
        # A uniform bound on every radial increment also bounds every order
        # statistic, while retaining the original nonlinear p95 acceptance.
        # These are proposal-only half spaces, not new source measurements.
        margin=.8*(old+allowance-current)
        pairs=terms.residual.reshape(-1,2)[indices];norms=np.linalg.norm(pairs,axis=1)
        selected=(pixels[indices]>0)&(norms>0);points=indices[selected]
        if len(points):
            pairs=pairs[selected];norms=norms[selected];distance=pixels[points]
            sigma=body.obs['sigma_px'][frames[points],joints[points]]
            u=np.hypot(1.,distance/(2*sigma));derivative=norms/distance*(u+1)/(2*u)
            radial=pairs/(norms*derivative)[:,None]
            projection=csr_matrix((radial.ravel(),(np.repeat(np.arange(len(points)),2),
                (2*points[:,None]+np.arange(2)).ravel())),shape=(len(points),len(terms.residual)))
            point_rows=projection@terms.jacobian;nonzero=np.diff(point_rows.indptr)>0
            if nonzero.any():
                gradient.append(point_rows[nonzero]);slack.extend(np.full(int(nonzero.sum()),margin))
                labels.extend(name+'/observation_'+str(int(frames[k]))+'_'+str(int(joints[k])) for k in points[nonzero])
    if gradient:pieces.append((vstack(gradient,format='csr'),np.asarray(slack),labels))
    pieces=[p for p in pieces if p is not None]
    if not pieces:return None
    result=SourceMargins(vstack([p[0]for p in pieces],format='csr'),np.concatenate([p[1]for p in pieces]),sum((p[2]for p in pieces),[]),
        [ball for p in pieces for ball in getattr(p,'image_balls',[])])
    result.body_groups=body_groups
    return result


def project_box_halfspaces(point, lower, upper, A, slack):
    """Sparse Dykstra projection; local source rows never shrink all controls."""
    point=np.asarray(point,float);lower=np.asarray(lower,float);upper=np.asarray(upper,float)
    A=csr_matrix(A,dtype=float);A.sum_duplicates();A.eliminate_zeros();slack=np.asarray(slack,float)
    if A.shape!=(len(slack),len(point)) or np.any(lower>upper):raise ValueError('Incompatible source/trust domain')
    x=np.clip(point,lower,upper);box_correction=np.zeros_like(x)
    plane_correction=np.zeros(len(slack));norms=np.asarray(A.multiply(A).sum(axis=1)).ravel()
    if np.any((norms==0)&(slack<0)):raise ValueError('An empty source row excludes every step')
    cycles=0
    while len(slack):
        previous=x.copy();y=x+box_correction;x=np.clip(y,lower,upper);box_correction=y-x
        # Inactive planes have no Dykstra correction; sparse matrix products
        # find newly active rows without a Python visit to every observation.
        active=np.flatnonzero((plane_correction!=0.) | (A@x>slack))
        for k in active:
            norm,b=norms[k],slack[k]
            if norm==0:continue
            begin,end=A.indptr[k:k+2];indices=A.indices[begin:end];a=A.data[begin:end]
            y=x[indices]+plane_correction[k]*a;correction=max(float(a@y-b),0.)/norm
            x[indices]=y-correction*a;plane_correction[k]=correction
        cycles+=1
        violation=max(float(np.max(A@x-slack,initial=0.)),float(np.max(lower-x,initial=0.)),float(np.max(x-upper,initial=0.)))
        change=float(np.linalg.norm(x-previous))/max(1.,float(np.linalg.norm(x)))
        if violation<=1e-10 and change<=1e-10:break
        if change<=np.finfo(float).eps*8:break
    x=np.clip(x,lower,upper)
    return x,dict(projection_cycles=cycles,linear_source_max_excess=float(np.max(A@x-slack,initial=0.)))

def project_box_halfspaces_dual(point,lower,upper,A,slack,*,progress=None,state=None):
    import time
    from scipy.optimize import minimize
    from scipy.sparse import diags
    p=np.asarray(point,float);lower=np.asarray(lower,float);upper=np.asarray(upper,float)
    A=csr_matrix(A,dtype=float);A.sum_duplicates();A.eliminate_zeros();slack=np.asarray(slack,float)
    if A.shape!=(len(slack),len(p)) or np.any(lower>upper):raise ValueError('Incompatible source/trust domain')
    norm2=np.asarray(A.multiply(A).sum(axis=1)).ravel();keep=norm2>0
    if np.any(~keep&(slack<0)):raise ValueError('An empty source row excludes every step')
    if not keep.any():return np.clip(p,lower,upper),dict(method='box only',projection_cycles=0,linear_source_max_excess=0.,duality_gap=0.)
    indices=np.flatnonzero(keep);norm=np.sqrt(norm2[keep]);full=(diags(1/norm)@A[keep]).tocsr();s=slack[keep]/norm
    started=time.perf_counter();last=started;evaluations=0;iterations=0;cycles=0;working_updates=0
    previous_dual=None if state is None else state.get('dual');full_dual=np.zeros(len(s))
    if previous_dual is not None and len(previous_dual)<=len(full_dual):full_dual[:len(previous_dual)]=previous_dual
    active=(A[keep]@np.clip(p,lower,upper)>slack[keep])|(full_dual>0)
    previous_gap=np.inf;ftol=1e-14;attempts=0;maximum=0.
    def emit(phase,**fields):
        nonlocal last
        now=time.perf_counter()
        if progress is not None and now-last>=10.:
            progress(phase=phase,seconds=now-started,**fields);last=now
    while True:
        chosen=np.flatnonzero(active);D=full[chosen];DT=D.T.tocsr();b=s[chosen];lam=full_dual[chosen]
        if not len(chosen):x=np.clip(p,lower,upper);dual_value=-.5*float(np.dot(x-p,x-p));message='All original rows inactive at box projection';break
        def objective(lam):
            nonlocal evaluations
            x=np.clip(p-DT@lam,lower,upper);q=D@x-b;delta=x-p
            value=-.5*np.dot(delta,delta)-np.dot(lam,q);evaluations+=1
            emit('source_projection_dual',evaluations=evaluations,working_rows=len(chosen),linear_constraint_max_excess=float(np.max(q*norm[chosen],initial=0.)))
            return float(value),-q
        fit=minimize(objective,lam,jac=True,method='L-BFGS-B',bounds=[(0.,None)]*len(b),options=dict(ftol=ftol,gtol=1e-10,maxls=50))
        lam=np.maximum(np.asarray(fit.x),0.);full_dual[chosen]=lam;iterations+=int(fit.nit);attempts+=1;message=str(fit.message)
        dual_value=objective(lam)[0];x=np.clip(p-DT@lam,lower,upper)
        new=(A[keep]@x-slack[keep]>1e-10)&~active
        if new.any():active|=new;working_updates+=1;continue
        while True:
            x=np.clip(x,lower,upper);excess=A@x-slack;maximum=float(np.max(excess,initial=0.))
            if maximum<=1e-10:break
            previous=x.copy()
            for k in np.flatnonzero(excess>0.):
                begin,end=A.indptr[k:k+2];cols=A.indices[begin:end];a=A.data[begin:end];v=float(a@x[cols]-slack[k])
                if v>0:x[cols]-=(v/norm2[k])*a
            cycles+=1;emit('source_projection_polish',cycles=cycles,linear_constraint_max_excess=maximum)
            if np.array_equal(previous,x):raise ArithmeticError('Source feasibility polish stagnated')
        primal=.5*float(np.dot(x-p,x-p));gap=primal+dual_value
        if gap < -1e-9*max(1.,primal):raise ArithmeticError('Invalid source projection dual certificate')
        # This is a feasible direction proposal, not the programme objective.
        # Keep its explicit dual error bound; the caller checks full quadratic
        # descent and all original nonlinear source/structure guards.
        break
    primal=.5*float(np.dot(x-p,x-p));gap=max(primal+dual_value,0.)
    if state is not None:state['dual']=full_dual.copy()
    return x,dict(method='nonnegative working-set dual with full feasibility certificate',projection_cycles=cycles,dual_iterations=iterations,dual_evaluations=evaluations,dual_attempts=attempts,dual_message=message,seconds=time.perf_counter()-started,working_rows=int(active.sum()),original_constraint_rows=A.shape[0],working_updates=working_updates,linear_source_max_excess=maximum,linear_constraint_max_excess=maximum,projection_squared_distance=2*primal,duality_gap=gap,relative_duality_gap=gap/max(1.,primal),numerical_linear_constraint_tolerance=1e-10,original_nonlinear_limits_unchanged=True,projection_optimality_goal_met=bool(gap<=1e-8+1e-7*max(1.,primal)),full_quadratic_and_nonlinear_acceptance_required=True)


def solve_bounded_quadratic(objective, initial, lower, upper):
    """Solve the damped linearized objective inside its actual control box."""
    from scipy.optimize import minimize, Bounds
    start=np.clip(np.asarray(initial,float),lower,upper)
    fit=minimize(objective,start,jac=True,method='L-BFGS-B',bounds=Bounds(lower,upper),
                 options=dict(ftol=1e-12,gtol=1e-8,maxls=50))
    z=np.clip(np.asarray(fit.x,float),lower,upper)
    value,gradient=objective(z)
    projected=z-np.clip(z-gradient,lower,upper)
    if not np.isfinite(z).all() or not np.isfinite(value):raise ArithmeticError('Nonfinite bounded normal step')
    return z,dict(method='bounded damped normal quadratic',success=bool(fit.success),
        message=str(fit.message),iterations=int(fit.nit),evaluations=int(fit.nfev),
        predicted_quadratic_change=float(value),projected_gradient_inf=float(np.max(np.abs(projected),initial=0.)))


def contract_source_trust(margins, selection, limits, direction, check, *, progress=None):
    """Contract only sparse dependencies implicated by original nonlinear source guards.

    Probe fractions identify a numerical trust scale, not a new source tolerance.
    The caller must re-solve the complete coupled objective inside the new box.
    Unattributed domain/association failures remain for ordinary nonlinear guards.
    """
    limits=np.asarray(limits,float);selection=np.asarray(selection,int)
    groups=check(np.asarray(direction,float))
    unchanged=dict(status='domain_unattributed',original_nonlinear_limits_unchanged=True)
    if groups is None:return limits.copy(),dict(unchanged,reason='nonlinear_domain')
    rejected={key:row for key,row in groups.items()if row['rejected']}
    if not rejected:return limits.copy(),dict(status='passed',original_nonlinear_limits_unchanged=True)
    domains={ball['source_id']:ball for ball in getattr(margins,'image_balls',[])if 'source_id'in ball}
    for key,row in rejected.items():
        ball=domains.get(key)
        lookup=getattr(margins,'guard_dependency',None)
        if ball is None and lookup is not None:ball=lookup(key)
        if ball is not None:domains[key]=ball
        if ball is None:return limits.copy(),dict(unchanged,reason='missing_raw_source_dependency',source_id=key)
        if row.get('model_foot')!=ball.get('model_foot'):
            return limits.copy(),dict(unchanged,reason='source_model_association_changed',source_id=key)
    pending=set(rejected);passing={};probes=[];domain_failures=0
    for fraction in backtracking_fractions():
        if fraction==1.:continue
        trial=check(fraction*np.asarray(direction,float))
        if trial is None:
            domain_failures+=1;continue
        if any(key not in trial or trial[key].get('model_foot')!=domains[key].get('model_foot') for key in pending):
            return limits.copy(),dict(unchanged,reason='source_model_association_changed_during_probe')
        passed=[key for key in pending if not trial[key]['rejected']]
        for key in passed:passing[key]=fraction;pending.remove(key)
        probes.append(dict(fraction=fraction,remaining_source_ids=sorted(pending)))
        if progress is not None:progress(phase='source_local_trust_probe',fraction=fraction,remaining_source_groups=len(pending))
        if not pending:break
    if pending and domain_failures:return limits.copy(),dict(unchanged,reason='nonlinear_domain_during_probe')
    if pending:raise ArithmeticError('Original source guard has no resolved local trust fraction')
    result=limits.copy();rows=[]
    for key,fraction in passing.items():
        dependency=np.unique(domains[key]['jacobian'].indices)
        active=np.flatnonzero(np.isin(selection,dependency))
        if not len(active):return limits.copy(),dict(unchanged,reason='source_has_no_active_dependency',source_id=key)
        result[active]=np.minimum(result[active],fraction*limits[active])
        rows.append(dict(source_id=key,passing_fraction=fraction,global_columns=selection[active].tolist(),
            before=rejected[key].get('before'),rejected_after=rejected[key].get('after'),
            original_radius_px=rejected[key].get('original_radius_px'),unit=rejected[key].get('unit','px')))
    changed=np.flatnonzero(result<limits)
    if not len(changed)or np.any(result<=0):raise ArithmeticError('Local source trust reached floating-point resolution')
    return result,dict(status='contracted',failed_source_ids=sorted(rejected),rows=rows,probes=probes,
        changed_global_columns=selection[changed].tolist(),original_nonlinear_limits_unchanged=True,
        all_global_controls_still_coupled=True)


def source_conditioned_direction(gpu, value, bounds, margins, selection, *, control_step_limit=None, progress=None, nonlinear_source_check=None, trust_state=None, retain_affine_cuts=False, affine_guard_reserve_fraction=0., early_nonlinear_ready_feedback=False, early_complete_nonlinear_proposal=False):
    """Solve the same coupled quadratic with source-conditioned local trust boxes."""
    if not 0<=affine_guard_reserve_fraction<.2:raise ValueError('Affine tolerance must stay inside the original unused guard reserve')
    if affine_guard_reserve_fraction and nonlinear_source_check is None:raise ValueError('Guard-reserve proposal tolerance requires original nonlinear feedback')
    if early_nonlinear_ready_feedback and nonlinear_source_check is None:
        raise ValueError('Early ready-domain feedback requires the original nonlinear check')
    if early_complete_nonlinear_proposal and not early_nonlinear_ready_feedback:
        raise ValueError('Complete nonlinear proposal exit requires full early nonlinear feedback')
    scale=1./np.maximum(gpu.scale_inv,1e-6)
    if progress is not None:progress(phase='normal_solve_start')
    step,linear=gpu.solve_scaled(-value['residual'],scale,damp=.07)
    raw_scaled_norm=float(np.linalg.norm(step));raw_actual=step*scale
    if margins is None:
        A=csr_matrix((0,len(step)));slack=np.empty(0);labels=[]
    else:
        source,slack,labels=margins;A=(source[:,selection].multiply(scale)).tocsr()
    bounded=None;projection=None;feedback=[];early_checks=0;nonlinear_completed_early=False
    if control_step_limit is None:
        shrink=max(1.,raw_scaled_norm);step/=shrink
        initial=(A@step).tolist();norms=np.asarray(A.multiply(A).sum(axis=1)).ravel()
        for _ in range(8):
            for k,(norm,limit) in enumerate(zip(norms,slack)):
                begin,end=A.indptr[k:k+2];indices=A.indices[begin:end];a=A.data[begin:end]
                excess=a@step[indices]-limit
                if excess>0. and norm>0.:step[indices]-=(excess/norm)*a
        step=np.clip(step*scale,*bounds)
    else:
        limits=np.array(control_step_limit,float,copy=True)
        if limits.shape!=step.shape or not np.isfinite(limits).all() or np.any(limits<=0):
            raise ValueError('Whole-state trust limits must match every active control')
        objective=gpu.scaled_normal_objective(-value['residual'],scale,damp=.07)
        base_A,base_slack,base_labels=A,slack,labels
        norm_domains=[];domains_by_unit={}
        for ball in getattr(margins,'image_balls',[]):
            offset=np.asarray(ball['offset']);radius=np.asarray(ball['radius']);B=ball['jacobian']
            if (offset.ndim!=2 or offset.shape[1]==0 or B.shape[0]!=offset.size
                    or radius.shape!=(len(offset),) or len(ball['labels'])!=len(offset)):
                raise ValueError('Affine norm ball requires offset (N, D), N radii and labels, and N * D Jacobian rows')
            unit=ball.get('unit','px' if offset.shape[1]==2 else 'unspecified')
            if not isinstance(unit,str) or not unit:raise ValueError('Affine norm ball requires a nonempty unit label')
            margin=ball.get('guard_margin',ball.get('guard_margin_px',0.) if unit=='px' else 0.)
            tolerance=max(1e-8,affine_guard_reserve_fraction*max(float(margin),0.))
            norm_domains.append(dict(ball,offset=offset,radius=radius,unit=unit,tolerance=tolerance,
                jacobian=B[:,selection].multiply(scale).tocsr()))
            domains_by_unit[unit]=domains_by_unit.get(unit,0)+len(radius)
        has_nonpixel=any(unit!='px' for unit in domains_by_unit)
        halfspace_guard_keys=set(base_labels)
        while True:
            lo=np.maximum(bounds[0],-limits)/scale;hi=np.minimum(bounds[1],limits)/scale
            if progress is not None:progress(phase='bounded_normal_start',local_trust_round=len(feedback))
            step,bounded=solve_bounded_quadratic(objective,step,lo,hi)
            point=step.copy()
            if not retain_affine_cuts or not feedback:
                dual_state={};cut_rounds=0;pixel_cut_rounds=0;nonpixel_cut_units={}
                A,slack,labels=base_A,base_slack,base_labels
            early_contracted=False;nonlinear_completed_early=False
            while True:
                if progress is not None:progress(phase='source_projection_start',rows=A.shape[0],nnz=A.nnz,image_cut_round=cut_rounds,local_trust_round=len(feedback))
                step,projection=project_box_halfspaces_dual(point,lo,hi,A,slack,progress=progress,state=dual_state)
                projected_A,projected_slack,projected_labels=A,slack,labels
                cuts=[];cut_slack=[];cut_labels=[];pixel_cuts=False
                maximum=dict.fromkeys(domains_by_unit,0.);maximum_over_tolerance=maximum.copy();norm_ready={}
                for ball in norm_domains:
                    B=ball['jacobian'];offset=ball['offset'];dimension=offset.shape[1]
                    moved=offset+(B@step).reshape(offset.shape)
                    distance=np.linalg.norm(moved,axis=1);excess=distance-ball['radius']
                    unit=ball['unit'];tolerance=ball['tolerance']
                    maximum[unit]=max(maximum[unit],float(np.max(excess,initial=0.)))
                    maximum_over_tolerance[unit]=max(maximum_over_tolerance[unit],
                        float(np.max(excess-tolerance,initial=0.)))
                    indices=np.flatnonzero(excess>tolerance)
                    if early_nonlinear_ready_feedback:
                        norm_ready.update((label,bool(ok)) for label,ok in zip(ball['labels'],excess<=tolerance))
                        if 'source_id' in ball:norm_ready[ball['source_id']]=not len(indices)
                    if not len(indices):continue
                    radial=moved[indices]/distance[indices,None]
                    selector=csr_matrix((radial.ravel(),(np.repeat(np.arange(len(indices)),dimension),
                        (dimension*indices[:,None]+np.arange(dimension)).ravel())),shape=(len(indices),B.shape[0]))
                    cuts.append(selector@B)
                    cut_slack.extend(ball['radius'][indices]-np.sum(radial*offset[indices],axis=1))
                    added=[ball['labels'][k]+'/affine_norm_cut_'+str(cut_rounds)for k in indices]
                    cut_labels.extend(added)
                    if unit=='px':pixel_cuts=True
                    else:nonpixel_cut_units.update(dict.fromkeys(added,unit))
                if not cuts:break
                A=vstack([A,*cuts],format='csr');slack=np.r_[slack,cut_slack];labels=[*labels,*cut_labels];cut_rounds+=1
                pixel_cut_rounds+=int(pixel_cuts)
                if progress is not None:
                    if has_nonpixel:
                        progress(phase='source_affine_norm_cut',cut_round=cut_rounds,added_rows=len(cut_labels),
                            max_affine_norm_excess_by_unit=maximum)
                    else:
                        progress(phase='source_affine_pixel_norm_cut',cut_round=cut_rounds,added_rows=len(cut_labels),
                            max_affine_pixel_excess_px=maximum.get('px',0.))
                if early_nonlinear_ready_feedback:
                    # Halfspaces have a full feasibility certificate already.
                    # Only norm rows/groups whose own affine domains also pass
                    # may trigger earlier curvature feedback. Unfinished norm
                    # domains cannot shrink their dependencies at this point.
                    first_full_check=[]
                    def check_ready(direction):
                        checks=nonlinear_source_check(direction)
                        if not first_full_check:first_full_check.append(checks)
                        if checks is None:return None
                        return {key:row for key,row in checks.items()
                                if norm_ready.get(key,key in halfspace_guard_keys)}
                    revised,diagnostic=contract_source_trust(margins,selection,limits,step*scale,check_ready,progress=progress)
                    early_checks+=1
                    if progress is not None:progress(phase='source_early_nonlinear_ready_feedback',
                        check=early_checks,status=diagnostic['status'],unfinished_affine_domains=True,
                        failed_source_ids=diagnostic.get('failed_source_ids',[]),
                        full_nonlinear_check_available=bool(first_full_check and first_full_check[0] is not None),
                        full_nonlinear_failed_source_ids=([key for key,row in first_full_check[0].items()if row['rejected']]
                            if first_full_check and first_full_check[0] is not None else None))
                    if (early_complete_nonlinear_proposal and first_full_check and first_full_check[0]
                            and isinstance(first_full_check[0],CompleteNonlinearGuardChecks)
                            and not any(row['rejected']for row in first_full_check[0].values())
                            and float(objective(step)[0]) < -16*np.finfo(float).eps*max(1.,float(value['residual']@value['residual']/2))):
                        # The actual guards passed; unfinished inner norm balls
                        # are numerical proposal aids, not additional observations.
                        # Retain the certificate only for the halfspaces actually
                        # projected. Newly generated cuts were never enforced.
                        A,slack,labels=projected_A,projected_slack,projected_labels
                        cut_rounds-=1;pixel_cut_rounds-=int(pixel_cuts)
                        for label in cut_labels:nonpixel_cut_units.pop(label,None)
                        nonlinear_completed_early=True
                        if progress is not None:progress(phase='source_complete_nonlinear_proposal',
                            checked_domains=len(first_full_check[0]),affine_proxy_complete=False,
                            unneeded_affine_cut_rows=len(cut_labels),
                            max_affine_norm_excess_by_unit=maximum,
                            full_quadratic_descent=True,outer_nonlinear_cost_and_guard_required=True)
                        break
                    if diagnostic['status']=='contracted':
                        diagnostic=dict(diagnostic,affine_incomplete=True,only_affine_ready_failures_used=True)
                        feedback.append(diagnostic);limits=revised;early_contracted=True
                        if progress is not None:progress(phase='source_local_trust_feedback',local_trust_round=len(feedback),**diagnostic)
                        break
            if early_contracted:continue
            initial=(A@point).tolist()
            projection.update(affine_pixel_cut_rounds=pixel_cut_rounds,max_affine_pixel_excess_px=maximum.get('px',0.),
                max_affine_excess_over_proposal_tolerance_px=maximum_over_tolerance.get('px',0.),
                retained_affine_cuts_across_local_trust=bool(retain_affine_cuts),
                affine_proposal_tolerance_fraction_of_original_guard_margin=affine_guard_reserve_fraction,
                affine_pixel_domains=domains_by_unit.get('px',0),
                affine_proxy_complete=not nonlinear_completed_early,
                completed_by_actual_nonlinear_guards=nonlinear_completed_early)
            if has_nonpixel:
                projection.update(affine_norm_cut_rounds=cut_rounds,affine_norm_domains_by_unit=domains_by_unit,
                    max_affine_norm_excess_by_unit=maximum,
                    max_affine_norm_excess_over_proposal_tolerance_by_unit=maximum_over_tolerance,
                    affine_norm_cut_units=nonpixel_cut_units)
            bounded['after_source_projection_quadratic_change']=float(objective(step)[0])
            if bounded['after_source_projection_quadratic_change']>=-16*np.finfo(float).eps*max(1.,float(value['residual']@value['residual']/2)):
                raise ArithmeticError('Feasible source proposal has no resolved descent in the full quadratic')
            if nonlinear_source_check is None:break
            def final_check(direction):
                checks=nonlinear_source_check(direction)
                if nonlinear_completed_early and (not isinstance(checks,CompleteNonlinearGuardChecks) or not checks):return None
                return checks
            revised,diagnostic=contract_source_trust(margins,selection,limits,step*scale,final_check,progress=progress)
            if nonlinear_completed_early and diagnostic['status']!='passed':
                nonlinear_completed_early=False
                projection['completed_by_actual_nonlinear_guards']=False
                projection['early_complete_check_not_reconfirmed']=diagnostic['status']
            feedback.append(diagnostic)
            if progress is not None:progress(phase='source_local_trust_feedback',local_trust_round=len(feedback),**diagnostic)
            if diagnostic['status']!='contracted':break
            limits=revised
        if trust_state is not None:trust_state['control_step_limit']=limits.copy()
        step=step*scale;shrink=1.
    top=np.argsort(np.abs(raw_actual))[-12:][::-1]
    receipt=dict(raw_scaled_step_norm=raw_scaled_norm,initial_step_shrink=shrink,
        max_actual_control_increment=float(np.max(np.abs(step))),
        dimension_dependent_unit_norm_cap=control_step_limit is None,
        previous_global_box_shrink=(None if control_step_limit is None else max(1.,float(np.max(np.abs(raw_actual)/limits)))),
        raw_largest_controls=[dict(global_column=int(selection[k]),increment=float(raw_actual[k]),accepted_direction_increment=float(step[k])) for k in top],
        constraint_rows=labels,initial_linear_change_px=initial,
        adjusted_linear_change_px=(A@(step/scale)).tolist(),inner_margin_px=slack.tolist(),
        linear_solve=linear,bounded_quadratic=bounded,source_projection=projection,local_source_feedback=feedback,
        original_nonlinear_guard_required=True,all_active_controls_retained=True,
        proposal_constraint_units=getattr(margins,'constraint_units',dict(source='px')),
        early_nonlinear_ready_feedback_checks=early_checks,
        nonlinear_completed_early=nonlinear_completed_early)
    return step,receipt


def backtracking_fractions():
    """Finite precision backtracking, without a fixed number of trial sizes."""
    fraction=1.
    while fraction>np.finfo(float).eps:
        yield fraction
        fraction*=.5

def resolved_descent(trial_cost, baseline_cost):
    """JSON-safe finite-precision descent decision, also used in run receipts."""
    return bool(trial_cost is not None and np.isfinite(trial_cost) and
        float(trial_cost)<float(baseline_cost)-16*np.finfo(float).eps*max(1.,float(baseline_cost)))
