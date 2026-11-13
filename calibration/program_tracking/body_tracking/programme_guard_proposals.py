"""Proposal constraints for the unchanged full programme quality guards.

These are numerical domains of the common step, not independent measurements.
The original nonlinear guards and refreshed surface objective remain authoritative.
"""
import numpy as np
from scipy.sparse import csr_matrix,vstack,hstack,diags
from .programme_source_step import SourceMargins,CompleteNonlinearGuardChecks


def normalized_peak_halfspaces(terms,scales,caps,labels):
    scales=np.broadcast_to(np.asarray(scales,float),terms.residual.shape)
    caps=np.broadcast_to(np.asarray(caps,float),terms.residual.shape)
    if np.any(scales<=0) or not np.isfinite(scales).all():raise ValueError('Positive original row scales required')
    matrix=(diags(1/scales)@terms.jacobian).tocsr();matrix.eliminate_zeros()
    keep=np.diff(matrix.indptr)>0
    slack=.8*(caps-np.asarray(terms.residual)/scales)
    return SourceMargins(matrix[keep],slack[keep],[labels[k]for k in np.flatnonzero(keep)])


def recover_body_pixel_jacobian(problem,value,points):
    """Invert the actual weighted isotropic 2D robust derivative, including zero error."""
    body=problem.body;points=np.asarray(points,int);fi=body.of[points];j=body.oj[points]
    frames=body.frames[fi];model=getattr(body,'model_oj',body.oj)[points]
    uv=body.cache['uv'][fi,body.layout.body_landmarks[model]]
    sigma=body.obs['sigma_px'][frames,j];error=(uv-body.obs['uv'][frames,j])/sigma[:,None]
    u=np.sqrt(1+np.sum(error**2,axis=1)/4);k=np.sqrt(2/(u+1));dk=-k/(16*u*(u+1))
    D=(k[:,None,None]*np.eye(2)+2*dk[:,None,None]*error[:,:,None]*error[:,None,:])/sigma[:,None,None]
    terms=next(b for b in value['weighted_blocks']if b.name=='body_pixels')
    rows=(2*points[:,None]+np.arange(2)).ravel()
    weight=np.broadcast_to(np.asarray(problem.objective_block_weights.get('body_pixels',1.),float),terms.residual.shape)[rows].reshape(-1,2)
    D*=np.sqrt(problem.image_time_weight*weight)[:,:,None]
    inverse=np.linalg.inv(D)
    row=np.broadcast_to((2*np.arange(len(points)))[:,None,None]+np.arange(2)[None,:,None],inverse.shape)
    col=np.broadcast_to((2*np.arange(len(points)))[:,None,None]+np.arange(2)[None,None,:],inverse.shape)
    left=csr_matrix((inverse.ravel(),(row.ravel(),col.ravel())),shape=(2*len(points),2*len(points)))
    return (left@terms.jacobian[rows]).tocsr()



class ProgrammeCavityTrialGuard:
    """Feedback from exact model cavity/collar envelopes on the native clock."""

    prefix = '__cavity_collar__/'

    def __init__(self, problem, value):
        if problem.optimize_shared:
            raise ValueError('Cavity trial feedback requires fixed shared shoe geometry')
        self.state = problem.body.state
        self.size = problem.size
        self.times = problem.reference.times[problem.body.frames]
        self.kinds = tuple(problem.cavity_labels)
        self.labels = tuple(problem.foot_row_labels)
        self.start = self._rows(self._term(value)).copy()
        self.caps = np.linalg.norm(self.start, axis=0)
        self.failed_frames = {}

    @staticmethod
    def _term(value):
        return next(block for block in value['weighted_blocks'] if block.name == 'cavity_AL')

    def _rows(self, block):
        if tuple(block.provenance['labels']) != self.labels:
            raise ValueError('Cavity or collar row identity changed during a proposal')
        values = np.asarray(block.residual).reshape(len(self.times) + 1, len(self.kinds))
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite cavity or collar trial residual')
        return values

    def halfspaces(self, value):
        # A tangent halfspace preserves the first-order norm direction. Exact
        # evaluations also see vertices and collar sections inactive at the base.
        block = self._term(value)
        self._rows(block)
        if block.jacobian is None:
            raise ValueError('Cavity proposal requires the original AL Jacobian')
        gradients = []
        for index in range(len(self.kinds)):
            rows = block.jacobian[index::len(self.kinds)]
            gradient = rows.T @ self.start[:, index]
            gradients.append(csr_matrix(gradient.reshape(1, -1)))
        return SourceMargins(vstack(gradients, format='csr'), np.zeros(len(gradients)),
                             [self.prefix + name for name in self.kinds])

    def checks(self, value):
        current = self._rows(self._term(value))
        norms = np.linalg.norm(current, axis=0)
        self.failed_frames.clear()
        result = {}
        for index, name in enumerate(self.kinds):
            key = self.prefix + name
            before, after = float(self.caps[index]), float(norms[index])
            allowance = 1e-7 * max(1., before)
            rejected = after > before + allowance
            if rejected:
                # Find the smallest largest-first set whose positive cost
                # changes account for the net regression. Include zero-J rows.
                change = np.square(current[1:, index]) - np.square(self.start[1:, index])
                positive = np.flatnonzero(change > 0.)
                ordered = positive[np.argsort(-change[positive], kind='stable')]
                cumulative = np.cumsum(change[ordered])
                needed = (after - before) * (after + before)
                count = np.searchsorted(cumulative, needed, side='left') + 1
                self.failed_frames[key] = ordered[:count]
            result[key] = dict(rejected=rejected, before=before, after=after,
                               model_foot=index % 2, original_radius_px=None,
                               unit='weighted cavity/collar residual norm',
                               numerical_norm_allowance=allowance)
        return result

    def dependency(self, key):
        frames = self.failed_frames.get(key)
        if frames is None or not len(frames):
            return None
        foot = self.kinds.index(key.removeprefix(self.prefix)) % 2
        times = self.times[frames]
        body = np.unique(self.state.basis(times).indices)
        shoes = np.unique(self.state.shoe_curve.basis_sparse(times).indices)
        # The cavity and collar live in the shoe-relative frame. Global root
        # motion cancels. Keep all native pose columns at the implicated times.
        body_columns = (17 + body[:, None] * 69 + np.arange(6, 69)).ravel()
        shoe_columns = (17 + self.state.body_size + shoes[:, None] * 6
                        + np.arange(foot * 3, foot * 3 + 3)).ravel()
        columns = np.r_[body_columns, shoe_columns]
        return dict(jacobian=csr_matrix((np.ones(len(columns)),
                    (np.zeros(len(columns), dtype=int), columns)), shape=(1, self.size)),
                    model_foot=foot)

    def record(self):
        scope = ('relative cavity and collar model AL' if len(self.kinds) == 4 else 'relative foot cavity model AL')
        return dict(caps=dict(zip(self.kinds, self.caps.tolist())), source_frames=len(self.times),
                    numerical_allowance='1e-7 * max(1, baseline weighted norm)',
                    scope=scope+'; no ice rows or calibrated acceptance')


class ProgrammeGuardProposals:
    def __init__(self,problem,value,source_margins,guard,joint_caps,surface_peak,*,signed_joint_domains=False,flight_guard=None,dense_flight_guard=None,cavity_trial_guard=None):
        self.problem=problem;self.guard=guard;self.start=np.array(problem.cache_x,copy=True)
        self.flight_guard=flight_guard;self.dense_flight_guard=dense_flight_guard
        self.cavity_trial_guard = cavity_trial_guard
        self.groups=source_margins.body_groups;self.tracked=set();self.last_checks={}
        self.joint=next(f for f in problem.action_factors.existing.factors if type(f).__name__=='ProgrammeJointLimits')
        self.surface=next(f for f in problem.action_factors.existing.factors if type(f).__name__=='ProgrammeSurfaceFactor')
        from .programme_joint_limits import NAMES
        self.joint_caps=np.array([joint_caps[n]+1e-5 for n in NAMES]);self.surface_cap=-float(surface_peak)+1e-6
        terms={b.name:b for b in value['weighted_blocks']};joint=terms['anatomical_joint_domain'];surface=terms['surface_intersection']
        self.joint_jac=joint.jacobian;self.surface_jac=surface.jacobian
        whole_skate=terms.get('whole_skate_source')
        self.whole_skate_jac=None if whole_skate is None else whole_skate.jacobian
        self.joint_shape=(len(self.joint.times),len(self.joint.joints),3)
        joint_scale=np.broadcast_to(self.joint.weights[:,None,None],self.joint_shape).ravel()*np.sqrt(problem.objective_block_weights['anatomical_joint_domain'])
        joint_cap=np.broadcast_to(self.joint_caps[None,:,None],self.joint_shape).ravel()
        if signed_joint_domains:
            from types import SimpleNamespace
            signed,jacobian=self.joint.proposal_linearization(self.start[17:])
            self.joint_jac=hstack((csr_matrix((len(signed),17)),jacobian),format='csr')
            signed_terms=SimpleNamespace(residual=signed,jacobian=self.joint_jac)
            joint_margin=normalized_peak_halfspaces(signed_terms,1.,joint_cap,['__joint__/'+str(i)for i in range(len(signed))])
        else:
            joint_margin=normalized_peak_halfspaces(joint,joint_scale,joint_cap,['__joint__/'+str(i)for i in range(len(joint.residual))])
        surface_scale=self.surface.weight*np.sqrt(problem.objective_block_weights['surface_intersection'])
        surface_margin=normalized_peak_halfspaces(surface,surface_scale,self.surface_cap,['__surface__/'+str(i)for i in range(len(surface.residual))])
        points=np.unique(np.concatenate([g['indices']for g in self.groups]));self.points=points
        self.point_lookup=np.full(len(problem.body.of),-1,int);self.point_lookup[points]=np.arange(len(points))
        self.body_jac=recover_body_pixel_jacobian(problem,value,points)
        body=problem.body;frames=body.frames[body.of];joints=body.oj
        self.pixel_delta=body.cache['uv'][body.of,body.layout.body_landmarks[getattr(body,'model_oj',joints)]]-body.obs['uv'][frames,joints]
        self.start_pixels=np.linalg.norm(self.pixel_delta,axis=1)
        balls=list(source_margins.image_balls)
        if whole_skate is not None:
            norm=float(np.linalg.norm(whole_skate.residual))
            if norm>0.:
                radius=norm*np.sqrt(1.05)
                balls.append(dict(source_id='whole_skate_source',model_foot=None,
                    unit='weighted whole-skate residual norm',guard_margin=radius-norm,
                    jacobian=whole_skate.jacobian,offset=whole_skate.residual[None].copy(),
                    radius=np.array([radius]),labels=['whole_skate_source']))
        if flight_guard is not None:balls.extend(flight_guard.domains(value))
        if dense_flight_guard is not None:balls.extend(dense_flight_guard.domains(value))
        for i,g in enumerate(self.groups):
            take=g['indices'];q=self.point_lookup[take]
            balls.append(dict(source_id='__body_group__/'+str(i),model_foot=None,guard_margin_px=g['margin'],
                jacobian=self.body_jac[(2*q[:,None]+np.arange(2)).ravel()],offset=self.pixel_delta[take].copy(),
                radius=self.start_pixels[take]+.8*g['margin'],labels=['__body__/'+str(i)+'/'+str(k)for k in take]))
        pieces=[source_margins,joint_margin,surface_margin]
        if cavity_trial_guard is not None:
            pieces.append(cavity_trial_guard.halfspaces(value))
        self.margins=SourceMargins(vstack([p[0]for p in pieces],format='csr'),np.concatenate([p[1]for p in pieces]),sum((p[2]for p in pieces),[]),balls)
        self.margins.guard_dependency=self.dependency
        self.margins.constraint_units=dict(source='px',body='px',joint='dimensionless authored domain',surface='m oriented witness gap')

    def dependency(self,key):
        if self.cavity_trial_guard is not None and key.startswith(self.cavity_trial_guard.prefix):
            return self.cavity_trial_guard.dependency(key)
        if self.dense_flight_guard is not None and key.startswith('__flight_dense__/'):
            return self.dense_flight_guard.dependency(key)
        if self.flight_guard is not None and key.startswith('__flight__/'):
            return self.flight_guard.dependency(key)
        if key=='whole_skate_source' and self.whole_skate_jac is not None:
            return dict(jacobian=self.whole_skate_jac,model_foot=None)
        fields=key.split('/');matrix=None
        if fields[0]=='__joint__':
            row=int(fields[1]);matrix=self.joint_jac[row]
            if not matrix.nnz:
                sample,joint,_=np.unravel_index(row,self.joint_shape)
                d=self.joint.state.increments(self.start[17:],self.joint.times[[sample]])
                local=np.zeros((1,3,69));columns=6+3*(self.joint.joints[joint]-1)+np.arange(3)
                local[0,np.arange(3),columns]=1.;cols=17+np.unique(d.chain(local).indices)
                matrix=csr_matrix((np.ones(len(cols)),(np.zeros(len(cols),int),cols)),shape=(1,self.problem.size))
        elif fields[0]=='__surface__':matrix=self.surface_jac[int(fields[1])]
        elif fields[0]=='__body__':
            q=self.point_lookup[int(fields[2])];matrix=self.body_jac[2*q:2*q+2]
        if matrix is None or not matrix.nnz:return None
        return dict(jacobian=matrix,model_foot=None)

    def check(self,z):
        value=self.problem.evaluate(z,False);checks=self.guard(value,.05);self.last_checks=checks
        body=self.problem.body;frames=body.frames[body.of];joints=body.oj
        delta=body.cache['uv'][body.of,body.layout.body_landmarks[getattr(body,'model_oj',joints)]]-body.obs['uv'][frames,joints]
        pixels=np.linalg.norm(delta,axis=1)
        np.testing.assert_array_equal(self.joint.last_parameters,z[17:]);np.testing.assert_array_equal(self.surface.last_parameters,z[17:])
        joint_values=self.joint.last_values;surface_depth=-self.surface.last_gap
        for row in np.flatnonzero((joint_values>self.joint_caps[None,:,None]).ravel()):self.tracked.add('__joint__/'+str(row))
        for row in np.flatnonzero(surface_depth>self.surface_cap):self.tracked.add('__surface__/'+str(row))
        for i,g in enumerate(self.groups):
            if checks[g['name']]['rejected']:
                take=g['indices'];bad=take[pixels[take]>self.start_pixels[take]+g['margin']]
                if not len(bad):raise ArithmeticError('A body percentile failure has no pointwise dependency')
                self.tracked.update('__body__/'+str(i)+'/'+str(k)for k in bad)
        omit={g['name']for g in self.groups}|{'actual_surface_witness_peak'}
        result={k:v for k,v in checks.items()if k not in omit and not k.startswith('joint_peak_')}
        for key in self.tracked:
            f=key.split('/')
            if f[0]=='__joint__':
                row=int(f[1]);sample,joint,component=np.unravel_index(row,self.joint_shape)
                now=float(joint_values[sample,joint,component]);cap=float(self.joint_caps[joint]);unit='dimensionless authored domain'
                rejected=now>cap
            elif f[0]=='__surface__':
                now=float(surface_depth[int(f[1])]);cap=self.surface_cap;unit='m oriented witness depth';rejected=now>cap
            else:
                group=int(f[1]);point=int(f[2]);g=self.groups[group]
                now=float(pixels[point]);cap=float(self.start_pixels[point]+g['margin']);unit='px'
                rejected=bool(checks[g['name']]['rejected'] and now>cap)
            result[key]=dict(rejected=bool(rejected),before=cap,after=now,original_radius_px=None,model_foot=None,unit=unit,
                original_nonlinear_guard_unchanged=True)
        checked=CompleteNonlinearGuardChecks(result)
        checked.objective_cost=float(value['residual']@value['residual']/2)
        return checked
