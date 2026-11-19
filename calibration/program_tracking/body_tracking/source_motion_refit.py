"""Fit the complete saved motion to native body/skate evidence with finite priors.

This is an explicitly revised objective, not an exact continuation of the old
nondegradation experiment. It keeps the shared geometry and uses the existing
FK, source clock, projection, derivatives, sparse solver and result exporter.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np

from .absolute_motion_state import absolute_reference, curve_from_packet
from .actual_observation_ledger import update_actual_observation_ledger
from .body_clock import source_seconds
from .body_execution import BodyExecution, BodyExecutionConfig
from .body_gpu_linear import CuDSS
from .continuous_physics import ContinuousMassGeometry
from .fullbody_round import write_json
from .ice_soft import IceSoftPolicy
from .programme_com import export_com_diagnostics
from .programme_skate_evidence import ManualNativeBodyFactors
from .rebuild_prepare import load, digest
from .scoped_motion_step import skate_projection
from .sequence_refit import SequenceJoint, SequenceActions
from .sequence_skate_state import shoe_curve_from_packet
from .skate_delivery import export_candidate
from .skate_physics import fk_state
from .skate_run_policy import RunPolicy
from .trajectory_backend import load_reference, PrimaryActions, configure_programme_weights
from .trajectory_probe import body_groups
from .weighted_blocks import block


class NativeSurfacePoints:
    """Source pixels of fixed material patches on the actual posed body skin.

    Wrist rotations remain part of the programme's continuous body curve. The
    saved finger poses and skinning model are unchanged; a wrist centre alone
    cannot determine the direction of the palm or fingers.
    """
    name = 'reviewed_native_surface_pixels'
    def __init__(self, problem, source):
        from .programme_skate_evidence import exposure_times
        self.problem = problem
        self.source_rows = [r for r in source.get('surface_samples',[]) if r['role']=='fit']
        self.rows=[];self.material_proposals=[];self.material_tracks={}
        for i,row in enumerate(self.source_rows):
            options=row.get('material_candidates',[{}]);indices=[]
            if not options:raise ValueError('A visible surface needs at least one material candidate')
            for option in options:
                indices.append(len(self.rows));self.rows.append(dict(row,**option))
            self.material_proposals.append(indices)
            if len(indices)>1:
                self.material_tracks.setdefault(row.get('association_track',row['id']),[]).append(i)
        for track,indices in self.material_tracks.items():
            if len({len(self.material_proposals[i]) for i in indices})!=1:
                raise ValueError('Inconsistent surface candidates in one observed track')
        self.refresh_each_state=bool(self.material_tracks);self._material_choices=None
        stamps = np.array([r['pts'] for r in self.rows],dtype=np.int64)
        unique,self.owner = np.unique(stamps,return_inverse=True)
        self.frames = np.searchsorted(problem.reference.p['pts'],unique)
        np.testing.assert_array_equal(problem.reference.p['pts'][self.frames],unique)
        self.times = exposure_times(problem,self.frames)
        self.base = problem.reference.at(self.times)
        self.ids = np.unique(np.concatenate([r['vertex_ids'] for r in self.rows])).astype(int)
        self.material = np.zeros((len(self.rows),len(self.ids)))
        for i,r in enumerate(self.rows):
            weights=np.asarray(r['vertex_weights'],float)
            if len(weights)!=len(r['vertex_ids']) or np.any(weights<0):
                raise ValueError('Invalid source material patch')
            np.testing.assert_allclose(weights.sum(),1.,atol=1e-12,rtol=0.)
            self.material[i,np.searchsorted(self.ids,r['vertex_ids'])]=weights
        self.uv=np.asarray([r['uv'] for r in self.rows],float)
        self.is_region=np.array(['pixel_box' in r for r in self.rows],bool)
        self.pixel_boxes=np.array([r.get('pixel_box',[[-np.inf,-np.inf],[np.inf,np.inf]])
                                   for r in self.rows],float).reshape(-1,2,2)
        if np.any(self.pixel_boxes[:,0]>self.pixel_boxes[:,1]):
            raise ValueError('Invalid source material region')
        self.origin_joint=np.array([r.get('relative_joint',-1) for r in self.rows],int)
        self.origin_uv=np.asarray([r.get('source_origin_uv',[0.,0.]) for r in self.rows],float)
        if np.any((self.origin_joint < -1)|(self.origin_joint>16)):
            raise ValueError('Invalid source direction origin')
        self.scale=np.sqrt([r.get('weight',9.) for r in self.rows])/np.asarray(
            [r['normalization_px'] for r in self.rows])
        if not np.isfinite(self.scale).all() or np.any(self.scale<=0):
            raise ValueError('Finite positive surface point normalization required')

    def evaluate(self,x,jacobian=False):
        from scipy.sparse import csr_matrix,hstack
        from .fullbody_fit import right_jacobian,skew
        from .programme_skate_evidence import chain_exposure
        from .rebuild_exposure_body import moving_projection
        from .selected_surface_linearization import SelectedSurfaceLinearization
        p=self.problem;d=p.body.state.increments(x,self.times)
        posed=fk_state(p.reference,self.base,d)
        surface=p.reference.binding.surface
        linear=SelectedSurfaceLinearization(surface,posed['native'],posed['frames'],
            self.ids,hand_local_R=posed.get('hand_local_R'))
        local=np.einsum('rv,nvc->nrc',self.material,linear.samples)
        rotated=np.einsum('nij,nrj->nri',posed['Q'],local)
        world=rotated+posed['P'][:,None]
        pr=moving_projection(p.camera,world.reshape(len(self.frames),5,len(self.rows),3),
            self.frames,jacobian=jacobian)
        row=np.arange(len(self.rows));uv=pr['uv'][self.owner,row]
        relative=self.origin_joint>=0
        observed_frame_uv=uv.copy()
        origin=None
        if relative.any():
            from .compact_body_derivatives import CompactBodyLinearization
            compact=CompactBodyLinearization(p.reference.body,self.base['local_R'],
                np.asarray(d[:,6:]).reshape(-1,21,3),hand_local_R=self.base.get('hand_local_R'))
            body_rotated=np.einsum('nij,nkj->nki',posed['Q'],compact.coco)
            origin=moving_projection(p.camera,(body_rotated+posed['P'][:,None]).reshape(
                len(self.frames),5,17,3),self.frames,jacobian=jacobian)
            origin_ids=np.maximum(0,self.origin_joint)
            observed_frame_uv-=relative[:,None]*(origin['uv'][self.owner,origin_ids]-self.origin_uv)
        error=observed_frame_uv-self.uv;active_pixels=np.ones_like(error)
        if self.is_region.any():
            m=self.is_region;lo=self.pixel_boxes[m,0];hi=self.pixel_boxes[m,1]
            error[m]=observed_frame_uv[m]-np.clip(observed_frame_uv[m],lo,hi)
            active_pixels[m]=(observed_frame_uv[m]<lo)|(observed_frame_uv[m]>hi)
        if self._material_choices is None or self.refresh_each_state:
            choices=np.array([p[0] for p in self.material_proposals],int)
            for indices in self.material_tracks.values():
                costs=[sum(float(np.sum((error[self.material_proposals[i][candidate]]*
                    self.scale[self.material_proposals[i][candidate]])**2)) for i in indices)
                    for candidate in range(len(self.material_proposals[indices[0]]))]
                selected=int(np.argmin(costs))
                for i in indices:choices[i]=self.material_proposals[i][selected]
            self._material_choices=choices
        choices=self._material_choices
        residual=(error[choices]*self.scale[choices,None]).ravel();J=None
        if jacobian:
            sample=self.owner[:,None]*5+np.arange(5)
            X=np.zeros((len(row),5,3,69));X[...,:3]=np.eye(3)
            X[...,3:6]=-skew(rotated[sample,row[:,None]]) @ (
                posed['Q']@right_jacobian(d[:,3:6]))[sample]
            for joint in range(1,22):
                cols=6+3*(joint-1)+np.arange(3)
                axes=right_jacobian(np.asarray(d[:,cols])).swapaxes(1,2)
                dv=linear.derivatives(joint,axes)
                dl=np.einsum('rv,navc->narc',self.material,dv)
                dw=np.einsum('nij,narj->nari',posed['Q'],dl)
                for a,col in enumerate(cols):X[...,col]=dw[sample,a,row[:,None]]
            raw=pr['pixel_jac_nodes'][self.owner,row]
            local_J=np.einsum('nplc,nlck->nplk',raw,X)
            if origin is not None:
                oj=origin_ids[:,None];Y=np.zeros_like(X);Y[...,:3]=np.eye(3)
                Y[...,3:6]=-skew(body_rotated[sample,oj]) @ (
                    posed['Q']@right_jacobian(d[:,3:6]))[sample]
                for col in range(6,69):
                    dv=compact.derivative(col,kinematics=False)['coco']
                    Y[...,col]=np.einsum('nmij,nmj->nmi',posed['Q'][sample],dv[sample,oj])
                raw_origin=origin['pixel_jac_nodes'][self.owner,origin_ids]
                local_J-=relative[:,None,None,None]*np.einsum('nplc,nlck->nplk',raw_origin,Y)
            local_J*=self.scale[:,None,None,None]*active_pixels[:,:,None,None]
            J=hstack((csr_matrix((len(residual),17)),
                chain_exposure(local_J[choices],d,sample_indices=sample[choices])),format='csr')
        name='reviewed_native_surface_pixels'
        return dict(residual=residual,jac=J,parts={name:float(residual@residual/2)},
            metrics={name:[dict(id=r['id'],pts=r['pts'],prediction=u.tolist(),
                distance_px=float(np.linalg.norm(e)),absolute_pixel_error=float(np.linalg.norm(u-r['uv'])),
                material_candidate_index=self.material_proposals[i].index(int(choice)),
                material_identity_inferred=len(self.material_proposals[i])>1)
                for i,(r,u,e,choice) in enumerate(zip(self.source_rows,uv[choices],error[choices],choices))]},
            weighted_blocks=[block(name,residual,J,dict(rows=self.source_rows,
                geometry='same actual posed body skin and original camera; no display warp'))])


class NativeShoePoints:
    """Actual source heel/toe pixels enter every objective evaluation."""
    def __init__(self, problem, source):
        self.problem = problem
        self.rows = [r for r in source['samples'] if r['role'] == 'fit']
        features = {'heel', 'toe', 'sole_middle', 'runner_rear', 'runner_front', 'heel_block_rear', 'cuff_center'}
        if any(r['feature'] not in features for r in self.rows):
            raise ValueError('A source shoe point would be silently omitted')
        self.target = np.asarray([r['uv'] for r in self.rows], float)
        self.is_region = np.array(['pixel_box' in r for r in self.rows],bool)
        self.pixel_boxes = np.array([r.get('pixel_box',[[-np.inf,-np.inf],[np.inf,np.inf]])
                                     for r in self.rows],float).reshape(-1,2,2)
        if np.any(self.pixel_boxes[:,0]>self.pixel_boxes[:,1]):
            raise ValueError('Invalid source shoe exclusion region')
        # New point normalization is an explicit engineering scale, not a
        # claim that visually estimated pixels have measured covariance.
        self.sigma = np.asarray([r.get('normalization_px',r.get('sigma_px'))
                                 for r in self.rows], float)
        self.weight = np.asarray([r.get('weight',9.) for r in self.rows],float)
        if not len(self.rows) or np.any(self.sigma <= 0):
            raise ValueError('Native shoe point evidence is missing or invalid')
        # A visible image endpoint can be useful even when crossing legs make
        # its anatomical identity uncertain. Keep this as a latent association,
        # and assign both endpoints of one observed boot to the same foot.
        self.proposals=[];self.proposal_indices=[];self.groups={};self.exclusive={}
        for i,row in enumerate(self.rows):
            feet=tuple(row.get('foot_candidates',[row['foot']]))
            if not feet or len(set(feet))!=len(feet) or not set(feet)<={0,1}:
                raise ValueError('Invalid candidate feet for a source boot')
            indices={}
            for foot in feet:
                indices[foot]=len(self.proposals);self.proposals.append(dict(row,foot=foot))
            self.proposal_indices.append(indices)
            if len(feet)>1:
                group=row.get('association_group',row.get('id',f'point:{i}'))
                # An LK descendant is still the same observed physical boot.
                # Its instantaneous nearest model foot must not rename it.
                group=row.get('association_track',group.split(':flow:',1)[0])
                self.groups.setdefault(group,[]).append(i)
                exclusion=row.get('association_exclusion_group')
                if exclusion is not None:self.exclusive.setdefault(exclusion,set()).add(group)
        for group,indices in self.groups.items():
            if len({tuple(self.proposal_indices[i]) for i in indices})!=1:
                raise ValueError('One physical source boot track must have consistent candidate feet')
        if any(len(groups)>2 for groups in self.exclusive.values()):
            raise ValueError('An exclusive pair cannot contain more than two physical boots')
        self.refresh_each_state=bool(self.groups)
        self._choices=None

    def evaluate(self, x, jacobian=False):
        full = np.r_[self.problem.binding.x, x]
        proposed, J = skate_projection(self.problem, full, self.proposals, jacobian=jacobian)
        scale = np.sqrt(self.weight) / self.sigma
        def point_error(prediction,index):
            if self.is_region[index]:
                return prediction-np.clip(prediction,*self.pixel_boxes[index])
            return prediction-self.target[index]
        if self._choices is None or self.refresh_each_state:
            choices=np.array([next(iter(q.values())) for q in self.proposal_indices])
            group_costs={};assigned={}
            for group,indices in self.groups.items():
                feet=tuple(self.proposal_indices[indices[0]])
                costs=[sum(float(np.sum((point_error(proposed[self.proposal_indices[i][f]],i)*scale[i])**2))
                           for i in indices) for f in feet]
                group_costs[group]=dict(zip(feet,costs));assigned[group]=feet[int(np.argmin(costs))]
            # A track can participate in several simultaneous pairs. Solve
            # their two-colour components together so a later frame cannot
            # undo the exclusivity established at an earlier exposure.
            graph={g:set() for g in self.groups}
            for groups in self.exclusive.values():
                if len(groups)==2:
                    a,b=groups;graph[a].add(b);graph[b].add(a)
            seen=set()
            for root in graph:
                if root in seen or not graph[root]:continue
                colour={root:0};queue=[root];seen.add(root)
                while queue:
                    a=queue.pop()
                    for b in graph[a]:
                        if b in colour:
                            if colour[b]==colour[a]:raise ValueError('Contradictory source boot track identity')
                        else:colour[b]=1-colour[a];seen.add(b);queue.append(b)
                costs=[sum(group_costs[g].get(c^flip,np.inf) for g,c in colour.items()) for flip in (0,1)]
                flip=int(np.argmin(costs))
                for g,c in colour.items():assigned[g]=c^flip
            for group,indices in self.groups.items():
                foot=assigned[group]
                for i in indices:choices[i]=self.proposal_indices[i][foot]
            self._choices=choices
        uv=proposed[self._choices]
        if jacobian:J=J[(2*self._choices[:,None]+np.arange(2)).ravel()]
        error=uv-self.target;axes=np.ones_like(error)
        if self.is_region.any():
            m=self.is_region;lo=self.pixel_boxes[m,0];hi=self.pixel_boxes[m,1]
            error[m]=uv[m]-np.clip(uv[m],lo,hi)
            axes[m]=(uv[m]<lo)|(uv[m]>hi)
        r = (error*scale[:, None]).ravel()
        if jacobian:
            J = J.multiply((scale[:,None]*axes).ravel()[:, None]).tocsr()
        name = getattr(self,'name','native_heel_toe_pixels')
        return dict(residual=r, jac=J, parts={name: float(r@r/2)},
                    metrics={name: dict(points=len(self.rows),
                        pixel_error=np.linalg.norm(error, axis=1).tolist(),
                        inferred_foot_association=[dict(id=self.rows[i].get('id'),pts=self.rows[i]['pts'],
                            foot=self.proposals[self._choices[i]]['foot'],independent_anatomical_identity=False)
                            for indices in self.groups.values() for i in indices])},
                    weighted_blocks=[block(name, r, J, dict(rows=self.rows))])


class NativeBodyPoints(ManualNativeBodyFactors):
    """Reviewed body pixels, separate from unchanged detector observations."""
    name = 'reviewed_native_body_pixels'

    def __init__(self, problem, source):
        rows = [dict(r, radius_px=r['normalization_px'],
                     raw_group_id=r.get('id',f"review:{r['pts']}:{r['joint']}"),
                     foot=r.get('foot'), event_id=r.get('event_id','whole_programme_review'))
                for r in source.get('body_samples',[]) if r['role']=='fit']
        super().__init__(problem, rows)
        self.weights = np.sqrt(np.asarray([r.get('weight',9.) for r in rows]))


class NativeBodyRegions(NativeBodyPoints):
    """Visible empty-background exclusions; no hidden joint point is measured."""
    name = 'reviewed_native_body_regions'

    def __init__(self, problem, source):
        rows=[dict(r,uv=np.asarray(r['pixel_box']).mean(axis=0).tolist())
              for r in source.get('body_regions',[]) if r['role']=='fit']
        super().__init__(problem,dict(body_samples=rows))


def metrics(problem, value):
    return dict(body=body_groups(problem, value),
                objective={b.name: float(b.residual@b.residual/2)
                           for b in value['weighted_blocks']},
                structure=value['gate'], action=value.get('action_metrics', {}))


def run(args):
    project, prepared, output = args.project, args.prepared, args.output
    continuing=getattr(args,'continue_in_place',False)
    if continuing:
        candidate=output/'solve/candidate.npz'
        prior_log=output/'progress.jsonl'
        if (not candidate.is_file() or not prior_log.is_file() or
                args.parameter.parent.resolve()!=output.resolve() or
                args.parameter.resolve()==candidate.resolve() or
                digest(args.parameter)!=digest(candidate)):
            raise ValueError('In-place continuation requires an exact archived final checkpoint beside the output log')
        last=json.loads(prior_log.read_text(encoding='utf8').splitlines()[-1])
        if last.get('event')!='source_motion_fit_complete':
            raise ValueError('The previous run must finish before in-place continuation')
    if args.reuse_unstarted and any((output/'solve'/name).exists()
            for name in ('candidate.npz','last_accepted.npz','last_attempt.npz')):
        raise ValueError('Reuse is only allowed before any solver checkpoint was written')
    output.mkdir(parents=True, exist_ok=args.reuse_unstarted or continuing)
    log_path = output/'progress.jsonl'
    print(json.dumps(dict(main_log=str(log_path))), flush=True)
    started = time.perf_counter()
    parent = load(args.parameter)
    context_parameter=args.parameter
    if args.resume_objective:
        prior_objective=json.loads(str(parent['source_motion_objective_json']))
        if prior_objective['source_sha256']!=digest(args.source):
            raise ValueError('Exact objective continuation requires unchanged source evidence')
        if prior_objective['shoe_points_sha256']!=digest(args.shoe_points):
            raise ValueError('New directed shoe pixels require an explicitly revised objective')
        if prior_objective.get('refresh_shoe_matches',False)!=args.refresh_shoe_matches:
            raise ValueError('Changing outline matching is not exact objective continuation')
        if prior_objective.get('whole_skate_weight')!=args.whole_skate_weight:
            raise ValueError('Changing the outline weight revises the objective')
        if (prior_objective.get('motion_prior_substeps',1)!=args.motion_prior_substeps or
                prior_objective.get('actual_so3_angular_acceleration_weight',4.)!=args.motion_acceleration_weight):
            raise ValueError('Changing motion quadrature or its weight revises the objective')
        observation_hash=None if args.body_observations is None else digest(args.body_observations)
        if prior_objective.get('body_observations_sha256')!=observation_hash:
            raise ValueError('Changing body source correspondence revises the objective')
        context_parameter=Path(prior_objective.get('objective_warm_parameter',prior_objective['parent_parameter']))
    ref, camera, info, document, _ = load_reference(project, prepared, args.source)
    if args.body_observations is not None:
        ref.observations_path=args.body_observations
    # A continuation keeps the exact saved coordinate chart and increments;
    # rebuilding a zero chart would also reset TRF's programme-wide radius.
    ref = absolute_reference(ref, curve_from_packet(parent,seed=args.continue_chart), root=prepared)
    ref.relative_skate_curve = shoe_curve_from_packet(parent,seed=args.continue_chart)
    ref.binding = copy.copy(ref.binding)
    ref.binding.reference = ref
    ref.binding.x = parent['x'][:17].copy()
    ref.binding._parameter_key = ref.binding._packet = None
    ref.source_document = document
    policy = IceSoftPolicy(**json.loads(str(parent['ice_soft_policy_json'])))
    execution = BodyExecution(BodyExecutionConfig(workers=12, blas_threads=1,
        gpu_batch_size=1024, local_derivatives='analytic', compatibility_backend='cuda_resident'))
    execution.attach(ref)
    with log_path.open('a', encoding='utf-8', buffering=1) as log, execution.limits():
        def emit(event, **values):
            log.write(json.dumps(dict(event=event, elapsed_seconds=time.perf_counter()-started,
                                      **values), ensure_ascii=False)+'\n')
        linear = CuDSS()
        seconds = source_seconds(ref.p)
        problem = SequenceJoint(ref, camera, ref.observations_path, tuple(seconds[[0,-1]]),
            document.get('supports', []), output/'solve', log, 'sequence_skates_v1',
            optimize_shared=False, ice_policy=policy, record_acceptance_before_jacobian=True)
        problem.linear_solver_backend = 'cuda_resident'
        problem.linear_runtime = linear
        problem.release_factorization_on_relinearize = True
        x = parent['x'].copy() if args.continue_chart else np.r_[ref.binding.x, np.zeros(problem.body.size)]
        if len(x)!=problem.size:
            raise ValueError('Saved continuation chart differs from current parameter layout')
        replay = problem.body.state.current_curve(x[17:])
        np.testing.assert_allclose(replay.position_controls,parent['absolute_position_controls'],atol=1.e-12,rtol=0.)
        np.testing.assert_allclose(replay.rotation_controls,parent['absolute_rotation_controls'],atol=1.e-12,rtol=0.)
        problem.action_warm_relative_trf = not args.continue_chart
        context_x=load(context_parameter)['x'] if args.resume_objective else x
        if args.motion_prior_substeps<1 or not np.isfinite(args.motion_acceleration_weight) or args.motion_acceleration_weight<=0:
            raise ValueError('Positive motion quadrature subdivisions and weight required')
        problem.motion_prior_substeps=args.motion_prior_substeps
        mass = ContinuousMassGeometry(ref)
        base = PrimaryActions(problem, ref, mass, document, warm=context_x[17:])
        reviewed_source = json.loads(args.shoe_points.read_text(encoding='utf-8'))
        shoe_points = NativeShoePoints(problem, reviewed_source)
        base.factors.append(shoe_points)
        if any(r['role']=='fit' for r in reviewed_source.get('body_samples',[])):
            base.factors.append(NativeBodyPoints(problem, reviewed_source))
        if any(r['role']=='fit' for r in reviewed_source.get('body_regions',[])):
            base.factors.append(NativeBodyRegions(problem, reviewed_source))
        if any(r['role']=='fit' for r in reviewed_source.get('surface_samples',[])):
            base.factors.append(NativeSurfacePoints(problem, reviewed_source))
        problem.action_factors = SequenceActions(problem, ref, mass, document, context_x[17:], base_factors=base)
        whole = problem.action_factors.whole_skate
        if whole is not None:
            whole.refresh_each_state=args.refresh_shoe_matches
            same_evidence=str(parent.get('source_evidence_sha256',''))==digest(args.source)
            if args.refresh_shoe_matches or (not same_evidence and document.get('source_motion_revision',{}).get('new_source_correspondences')):
                whole.associate(x[17:])
            else:
                whole.restore_checkpoint(parent)
        modes = problem.action_factors.modes
        modes.modes = parent['sequence_contact_modes'].copy()
        modes.status = parent['sequence_contact_status'].copy()
        modes.record = json.loads(str(parent['sequence_contact_association_json']))
        if args.resume_objective:
            problem.action_factors.evaluate(context_x[17:],False)
        old_policy = configure_programme_weights(problem, base, document, x)
        body = problem.body
        ids = getattr(body, 'model_oj', body.oj)
        row_weight = np.ones(len(ids))
        row_weight[np.isin(ids, [0,1,2,3,4])] = .25
        proxy = body.obs['inside_tracked_shoe_proxy'][body.frames[body.of], body.oj]
        row_weight[np.isin(ids, [15,16]) & proxy] = .10
        problem.objective_block_weights.update(body_pixels=np.repeat(row_weight,2),
            actual_so3_angular_acceleration=args.motion_acceleration_weight, finite_parent_root_prior=.08)
        if args.whole_skate_weight is not None:
            if not np.isfinite(args.whole_skate_weight) or args.whole_skate_weight<=0:
                raise ValueError('Whole skate weight must be finite and positive')
            problem.objective_block_weights['whole_skate_source']=args.whole_skate_weight
        # These measures stay in the objective and diagnostics. They do not
        # assert that each imperfect old model statistic is a physical bound.
        problem.image_safe_trials = False
        problem.enforce_rotation_lift_continuity = True
        active_stop=problem.size if args.fit_shoe_rotation else 17+problem.body.state.body_size
        problem.active_parameter_indices = np.arange(17,active_stop)
        problem.local_control_seconds = 'complete programme; body roots and all joint curves'+(
            '; continuous ankle-centred shoe rotations' if args.fit_shoe_rotation else '')
        config = dict(schema='rinkspace.source-motion-objective.v1',
            parent_parameter=str(args.parameter), parent_sha256=digest(args.parameter),
            objective_warm_parameter=str(context_parameter),
            exact_objective_continuation=args.resume_objective,
            continued_in_existing_output=continuing,
            source=str(args.source), source_sha256=digest(args.source),
            shoe_points=str(args.shoe_points), shoe_points_sha256=digest(args.shoe_points),
            observed_pixels_unchanged=True, source_exposures=len(ref.times),
            camera=info, previous_weight_policy=old_policy,
            body_weights=dict(visible=1., face=.25, inside_shoe_proxy=.10),
            actual_so3_angular_acceleration_weight=args.motion_acceleration_weight,
            motion_prior_substeps=args.motion_prior_substeps,
            motion_quadrature_samples=len(base.motion.times),
            motion_quadrature_is_source_observation=False,
            continued_saved_coordinate_chart=args.continue_chart,
            refresh_shoe_matches=args.refresh_shoe_matches,
            whole_skate_weight=args.whole_skate_weight,
            body_observations=None if args.body_observations is None else str(args.body_observations),
            body_observations_sha256=None if args.body_observations is None else digest(args.body_observations),
            reviewed_body_points=sum(r['role']=='fit' for r in reviewed_source.get('body_samples',[])),
            reviewed_surface_points=sum(r['role']=='fit' for r in reviewed_source.get('surface_samples',[])),
            source_revision=document.get('source_motion_revision'),
            historical_proxy_nonincrease_caps=False,
            rotation_step_chart_continuity=True,
            physical_and_wear_factors='retained as finite terms; independently rechecked after fit',
            shared_geometry_fixed=True, relative_shoe_rotations_fixed=not args.fit_shoe_rotation,
            shoe_rotation_pivot='same native ankle; existing finite wear/structure terms retained',
            dense_result_export_requested=args.export_result,
            user_accepted=False)
        write_json(output/'objective.json', config)
        problem.checkpoint_extra = dict(
            source_motion_objective_json=np.array(json.dumps(config,sort_keys=True)),
            source_evidence_sha256=np.array(digest(args.source)),
            prepared_input_sha256=np.array(digest(prepared/'input.npz')),
            prepared_observations_sha256=np.array(digest(ref.observations_path)),
            source_motion_parent_sha256=np.array(digest(args.parameter)))
        if whole is not None:
            problem.checkpoint_extra.update(whole.checkpoint_fields())
        np.savez_compressed(output/'objective_block_weights.npz',**problem.objective_block_weights)
        emit('source_motion_fit_start', log=str(log_path), exposures=len(ref.times),
             active_variables=len(problem.active_parameter_indices), source_seconds=seconds[[0,-1]].tolist())
        before = problem.evaluate(x, False)
        initial_cost=float(before['residual']@before['residual']/2)
        if args.expected_initial_cost is not None:
            np.testing.assert_allclose(initial_cost,args.expected_initial_cost,rtol=1.e-8,atol=1.e-8)
        emit('initial_objective_verified',cost=initial_cost,expected=args.expected_initial_cost)
        write_json(output/'before.json', metrics(problem,before))
        result, report = problem.solve(x, RunPolicy(wall_seconds=None, max_accepted=None,
            max_outer=1, allow_penalty_growth=False))
        final = problem.evaluate(result,False)
        write_json(output/'after.json',metrics(problem,final))
        write_json(output/'solver_report.json',report)
        if not args.export_result:
            emit('source_motion_fit_complete',parameter=str(output/'solve/candidate.npz'),
                 accepted_steps=problem.accepted,visual_acceptance='pending',
                 dense_result_exported=False)
            execution.close()
            return
        for factor in base.factors:
            if type(factor).__name__ == 'ProgrammeJointLimits':
                factor.export(result[17:],output/'anatomical_frames.npz')
        emit('export_start',accepted_steps=problem.accepted)
        candidate = absolute_reference(ref,problem.body.state.current_curve(result[17:]),root=prepared)
        candidate.relative_skate_curve = problem.body.state.shoe_current_curve(result[17:])
        candidate.binding = copy.copy(ref.binding)
        candidate.binding.reference = candidate
        packet = export_candidate(prepared,output,candidate,camera,info,document.get('supports',[]),write=False)
        world = fk_state(candidate,candidate.at(candidate.times),np.zeros((len(candidate.times),69)))
        centres,rotations = mass.from_pose(world)
        packet.update(export_com_diagnostics(mass,candidate.times,centres,rotations))
        packet.update(native_body_local_R_mid=world['local_R'],
            native_body_FK_world_R_mid=world['Q'][:,None]@world['frames'],
            source_shot_ids=ref.p['shot_ids'],shared_betas=ref.p['shared_betas'],
            shared_body_scale=ref.p['body_scale'],sequence_contact_modes=modes.modes,
            sequence_contact_status=modes.status,
            source_motion_objective_json=np.array(json.dumps(config,sort_keys=True)))
        update_actual_observation_ledger(packet,problem.body)
        packet['com_projection_world_mid'] = packet['com_world_mid'].copy()
        packet['com_projection_world_mid'][:,2] = 0.
        (output/'fitted').mkdir(exist_ok=True)
        temporary = output/'fitted/result.pending.npz'
        with temporary.open('xb') as stream:
            np.savez_compressed(stream,**packet)
        temporary.replace(output/'fitted/result.npz')
        emit('source_motion_fit_complete',result=str(output/'fitted/result.npz'),
             accepted_steps=problem.accepted,visual_acceptance='pending')
    execution.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('project','prepared','source','parameter','shoe-points','output'):
        parser.add_argument('--'+name,required=True,type=lambda p:Path(p).resolve())
    parser.add_argument('--continue-chart',action='store_true')
    parser.add_argument('--resume-objective',action='store_true')
    parser.add_argument('--expected-initial-cost',type=float)
    parser.add_argument('--reuse-unstarted',action='store_true')
    parser.add_argument('--continue-in-place',action='store_true',
        help='Append a completed run using its exact archived final checkpoint; preserve one output directory')
    parser.add_argument('--fit-shoe-rotation',action='store_true')
    parser.add_argument('--motion-prior-substeps',type=int,default=1,
        help='Subdivide each native interval for actual-motion priors; does not add image observations')
    parser.add_argument('--motion-acceleration-weight',type=float,default=4.,
        help='Finite actual SO(3) acceleration objective weight')
    parser.add_argument('--whole-skate-weight',type=float,
        help='Explicit whole-programme image-outline weight; revises the saved objective')
    parser.add_argument('--body-observations',type=lambda p:Path(p).resolve(),
        help='Reviewed source correspondence ledger; original source observations stay intact')
    parser.add_argument('--refresh-shoe-matches',action='store_true',
        help='Recompute source-outline closest points at every actual trial state; checkpoint their current associations')
    parser.add_argument('--export-result',action='store_true',
        help='Explicitly export dense per-frame geometry; the default retains only solver checkpoints and diagnostics')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
