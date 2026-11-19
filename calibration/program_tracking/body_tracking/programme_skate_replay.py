"""Production replay for source-driven skate proposals on the existing sequence."""
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import lsmr

from .absolute_motion_state import absolute_reference, curve_from_packet, evaluate_curve
from .action_objective_context import (bind_action_objective_context, restore_source_masks,
                                        restore_source_episode)
from .body_clock import source_seconds, time_base
from .body_execution import BodyExecution, BodyExecutionConfig
from .continuous_physics import ContinuousMassGeometry
from .ice_soft import IceSoftPolicy
from .programme_joint_limits import ProgrammeJointLimits
from .programme_skate_evidence import (
    extend_document, load_json, merge_document, programme_observation_path,
)
from .programme_surface import CollisionSurface, ProgrammeSurfaceFactor, witnesses_packet
from .rebuild_prepare import digest, load
from .sequence_refit import SequenceActions, SequenceJoint
from .sequence_skate_state import shoe_curve_from_packet
from .skate_checkpoint import validate_checkpoint_identity
from .skate_physics import fk_state
from .trajectory_backend import PrimaryActions, configure_programme_weights, load_reference
from .trajectory_probe import body_groups
from .weighted_blocks import signature, verify


def write(path, obj):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)


def save(path, **packet):
    path=Path(path)
    temp=path.with_name(path.stem+'.pending'+path.suffix)
    with temp.open('xb') as handle:
        np.savez_compressed(handle,**packet)
    if path.exists():
        raise FileExistsError(path)
    temp.rename(path)


def load_scan(path, parameter, camera_sha256, prepared):
    path=Path(path); report=load_json(path.with_suffix('.json'))
    for k,v in dict(parameter_sha256=digest(parameter),camera_sha256=camera_sha256,
                    input_sha256=digest(prepared/'input.npz')).items():
        if report[k]!=v:raise ValueError('Surface cache binding mismatch: '+k)
    return load(path)


WITNESS=('sample','vertex','face_vertices','bary','normal','initial_gap','face_a','face_b')


def common_witnesses(scans, geometry):
    first=scans[0]
    for scan in scans[1:]:
        for k in ('times','source_pts','parts','surface_response_policy'):
            np.testing.assert_array_equal(first[k],scan[k],err_msg=k)
    merged={k:np.concatenate([a[k] for a in scans]) for k in WITNESS}
    identity=np.c_[merged['sample'],merged['vertex'],merged['face_a'],merged['face_b']]
    _,idx=np.unique(identity,axis=0,return_index=True)
    # Retain actual parent and trial witnesses, sharing each semantic pair's
    # time budget instead of giving duplicated face witnesses extra weight.
    result={k:v[idx] for k,v in merged.items()}
    a=geometry.parts[result['face_a']];b=geometry.parts[result['face_b']]
    _,ids,count=np.unique(np.c_[result['sample'],np.minimum(a,b),np.maximum(a,b)],
                          axis=0,return_inverse=True,return_counts=True)
    result['witness_share']=1./count[ids]
    result.update({k:first[k] for k in ('times','source_pts','parts','surface_response_policy')})
    return result


def refresh_surface(problem,parent_scan,parent_x,x,doc,emit,*,exact_saved_pose=False,rescan_all=False):
    """Replay native and midpoint samples changed by either spline, reusing exact support complement."""
    from concurrent.futures import ThreadPoolExecutor
    p=problem;ref=p.reference; state=p.body.state
    geometry=CollisionSurface(ref,doc)
    if geometry.broad_phase is None:
        raise RuntimeError('The existing native surface broad phase is required for this replay')
    ts=parent_scan['times']
    expected=np.sort(np.r_[ref.times,(ref.times[1:]+ref.times[:-1])/2])
    np.testing.assert_array_equal(ts,expected)
    np.testing.assert_array_equal(parent_scan['source_pts'],ref.p['pts'])
    np.testing.assert_array_equal(parent_scan['parts'],geometry.part_names)
    body_changed=np.flatnonzero(np.any((x[17:17+state.body_size]!=parent_x[17:17+state.body_size]).reshape(-1,69),axis=1))
    shoe_changed=np.flatnonzero(np.any((x[17+state.body_size:]!=parent_x[17+state.body_size:]).reshape(-1,6),axis=1))
    def affected(knots,count,changed,clock):
        spans=np.clip(np.searchsorted(knots,clock,side='right')-4,0,count-4)
        return np.isin(spans[:,None]+np.arange(4),changed).any(1)
    dirty=(affected(state.knots,state.m,body_changed,ts) |
           affected(state.shoe_curve.knots,state.shoe_m,shoe_changed,ts))
    if rescan_all:
        dirty[:]=True
    current=None
    if exact_saved_pose:
        current=absolute_reference(ref,state.current_curve(x[17:]))
        current.relative_skate_curve=state.shoe_current_curve(x[17:])
    result={k:v.copy() for k,v in parent_scan.items() if k not in (*WITNESS,'pair_counts')}
    keep=~dirty[parent_scan['sample']]
    pair_keep=~dirty[parent_scan['pair_counts'][:,0]]
    records=[];counts=[]
    def one(item):
        idx,V=item
        witnesses,detail=geometry.scan_sample(V,ts[idx]);keys={}
        for w in witnesses:
            key=tuple(sorted((int(geometry.parts[w[5]]),int(geometry.parts[w[6]]))))
            if key not in keys or w[4]<keys[key][4]:keys[key]=w
        pair=defaultdict(int)
        for a,b in detail['pairs']:
            pair[tuple(sorted((int(geometry.parts[a]),int(geometry.parts[b]))))]+=1
        return idx,keys,detail,pair
    todo=np.flatnonzero(dirty)
    # A few untouched fresh probes catch accidentally incomplete invalidation.
    off=np.flatnonzero(~dirty)
    probes=off[np.linspace(0,len(off)-1,min(12,len(off)),dtype=int)] if len(off) else off
    items=np.r_[probes,todo]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for n in range(0,len(items),32):
            ids=items[n:n+32];t=ts[ids]
            if current is None:
                posed=fk_state(ref,ref.at(t),state.increments(x[17:],t))
            else:
                # Classify the exact exported curve. Composing inverse log
                # increments onto the seed is equivalent for fitting but can
                # round the opposite way at a grazing triangle intersection.
                posed=fk_state(current,current.at(t),np.zeros((len(t),69)))
            vertices=geometry.vertices(posed)
            for i,keys,d,pair in pool.map(one,zip(ids,vertices)):
                for dest,origin in [('crossing_counts','crossing_pairs'),
                                    ('grip_crossing_counts','source_grip_crossing_pairs'),
                                    ('topology_uncertain_candidates','topology_uncertain_candidates')]:
                    if dirty[i]:result[dest][i]=d[origin]
                    elif result[dest][i]!=d[origin]:
                        raise ValueError(f'Surface cache fails unchanged sample probe {i}: {dest} '
                                         f'{result[dest][i]} != {d[origin]}')
                if dirty[i]:
                    records.extend((i,w) for w in keys.values())
                    counts.extend((i,a,b,n) for (a,b),n in pair.items())
                else:
                    expected_pairs={(int(a),int(b)):int(c) for _,a,b,c in parent_scan['pair_counts'][parent_scan['pair_counts'][:,0]==i]}
                    if pair!=expected_pairs:
                        raise ValueError(f'Surface cache differs at unchanged sample {i}: semantic pairs')
            if n%256==0 or n+32>=len(items):
                emit('actual_surface_refresh',samples=min(n+32,len(items)),total=len(items))
    additions=witnesses_packet(records)
    for k in WITNESS:result[k]=np.concatenate((parent_scan[k][keep],additions[k]))
    result['pair_counts']=np.r_[parent_scan['pair_counts'][pair_keep],np.array(counts,int).reshape(-1,4)]
    return result,dict(native_exposures=len(ref.times),checked_samples=len(ts),fresh_samples=len(todo),
                       exactly_reused_samples=int((~dirty).sum()),fresh_unchanged_probes=len(probes),
                       exact_saved_curve_geometry=bool(exact_saved_pose),
                       sampled_not_collision_free_certificate=True)


class ScopedSurfaceGuard:
    """Check actual changed surface samples and both candidates on common witnesses."""

    def __init__(self, problem, primary, document, parent_x, output, parent_costs, *, proposal_surfaces=None):
        self.problem = problem
        self.document = document
        self.parent_x = parent_x.copy()
        self.output = output
        self.reference = problem.reference
        self.surface = next(f for f in primary.factors if isinstance(f, ProgrammeSurfaceFactor))
        self.specification = document['scoped_candidate_import']
        parent_surface = self.specification.get('parent_surface')
        if parent_surface is None:
            self.scan = load(document['programme_structure']['surface_witnesses'])
        else:
            if digest(parent_surface['path']) != parent_surface['sha256']:
                raise ValueError('Saved parent surface scan changed')
            self.scan = load(parent_surface['path'])
        cfg = load_json(document['programme_structure']['policy'])
        self.allowance = float(cfg['geometry']['contact_allowance_m'])
        self.parent_cost, self.parent_surface_cost = parent_costs
        self.proposal_surfaces = {} if proposal_surfaces is None else proposal_surfaces
        self.serial = 0
        self.last_pass = None
        self.passes = {}
        self.current_scan = self.scan
        self.pending = None

    def bind_current_scan(self,x,path):
        retained = load_json(path)
        if retained.get('parameter_vector_signature') != signature(x):
            raise ValueError('Retained surface does not describe the exact resumed geometry')
        if retained['surface_sha256'] != digest(retained['surface_path']):
            raise ValueError('Resumed surface evidence changed')
        self.current_scan = load(retained['surface_path'])
        self.emit('scoped_proposal_surface_anchor',retained_surface=str(path),
                  anchor_parameter_signature=signature(x),historical_guard_baseline_retained=True)

    def emit(self, event, **values):
        self.problem.log.write(json.dumps(dict(event=event, **values)) + '\n')

    def compare_geometry(self,x,new):
        geometry = self.surface.geometry
        common = common_witnesses([self.scan,new], geometry)
        factor = ProgrammeSurfaceFactor(self.problem,self.document,common,scale_m=self.surface.scale_m)
        before = factor.evaluate(self.parent_x[17:])['residual']
        old_gap = factor.last_gap.copy()
        after = factor.evaluate(x[17:])['residual']
        new_gap = factor.last_gap
        parts = geometry.parts
        a,b = parts[common['face_a']], parts[common['face_b']]
        groups = np.c_[np.minimum(a,b),np.maximum(a,b)]
        from .programme_repair_factors import sampled_windows
        bad = []
        for window in self.specification['focus_pts']:
            active = sampled_windows(self.reference,self.problem.body.state,factor.times,[window])
            use = active[common['sample']]
            for left,right in np.unique(groups[use],axis=0):
                rows = use & (groups[:,0]==left) & (groups[:,1]==right)
                first, last = old_gap[rows].min(), new_gap[rows].min()
                if last < min(first,0.)-self.allowance:
                    indices=np.flatnonzero(rows)
                    sample=int(common['sample'][indices[np.argmin(new_gap[rows])]])
                    bad.append(dict(window_pts=window, pair=[str(geometry.part_names[left]),
                        str(geometry.part_names[right])], before_m=float(first), after_m=float(last),
                        source_seconds=float(new['source_seconds'][sample])))
        return before,after,bad

    def screen_proposal(self,x):
        """Refresh exact mesh witnesses before dispatching the same trial."""
        new,coverage=self.scan_trial(x)
        before,after,bad=self.compare_geometry(x,new)
        self.pending=(signature(x),new,coverage,before,after,bad) if not bad else None
        return bad,coverage

    def __call__(self, x, value):
        if np.array_equal(x, self.parent_x):
            return False
        if self.last_pass is not None and np.array_equal(x,self.last_pass):
            return True
        self.serial += 1
        identity = signature(x)
        reused = self.proposal_surfaces.get(identity)
        receipt = load_json(reused) if reused is not None and reused.is_file() else {}
        pending=getattr(self,'pending',None)
        if pending is not None and pending[0]==identity:
            _,new,coverage,before,after,bad=pending
            self.pending=None
            archive=None
        elif (receipt.get('parameter_vector_signature') == identity and receipt.get('rejected') is False
                and receipt.get('exact_saved_curve_geometry') is True):
            archive = Path(receipt['surface_path'])
            if digest(archive) != receipt['surface_sha256']:
                raise ValueError('Prior proposal surface cache changed')
            new = load(archive)
            coverage = {key:receipt[key] for key in (
                'native_exposures','checked_samples','fresh_samples','exactly_reused_samples',
                'fresh_unchanged_probes','exact_saved_curve_geometry','sampled_not_collision_free_certificate')}
            coverage.update(replayed_scan=str(archive.resolve()),replayed_scan_sha256=receipt['surface_sha256'],
                            recomputed_cost_and_regional_guards=True)
            self.emit('actual_surface_exact_proposal_cache',receipt=str(reused),**coverage)
            before,after,bad=self.compare_geometry(x,new)
        else:
            if receipt:
                self.emit('actual_surface_proposal_cache_mismatch',receipt=str(reused),recomputed=True)
            new,coverage = self.scan_trial(x)
            archive = None
            before,after,bad=self.compare_geometry(x,new)
        weight = float(self.problem.objective_block_weights.get('surface_intersection',1.))
        # Replace the old surface contribution on both sides of the same
        # revised objective. The resume objective keeps its original rows.
        cost0 = self.parent_cost-self.parent_surface_cost+float(before@before/2)*weight
        residual = value['residual']
        cost1 = float(residual@residual/2)-value['weighted_parts']['surface_intersection']+float(after@after/2)*weight
        rejected = bool(bad) or cost1 >= cost0
        name = f'surface_trial_{self.serial:03}'
        receipt=dict(rejected=rejected, common_witnesses=len(before), same_weighted_cost_before=cost0,
            same_weighted_cost_after=cost1, gap_allowance_m=self.allowance, regressions=bad,
            total_triangle_pairs=int(new['crossing_counts'].sum()), **coverage)
        if not rejected:
            if archive is None:
                archive=self.output/(name+'.npz')
                save(archive, **new)
            receipt['surface_sha256'] = digest(archive)
            receipt['parameter_vector_signature'] = identity
            self.last_pass = x.copy()
            self.passes[receipt['parameter_vector_signature']] = (archive,receipt)
        write(self.output/(name+'.json'),receipt)
        self.emit('scoped_surface_trial',receipt=name+'.json',**receipt)
        return not rejected

    def scan_trial(self, x):
        try:
            return refresh_surface(self.problem,self.scan,self.parent_x,x,self.document,
                                   self.emit,exact_saved_pose=True)
        except ValueError as error:
            if 'unchanged sample' not in str(error):
                raise
            self.emit('surface_cache_probe_recheck',reason=str(error))
            return refresh_surface(self.problem,self.scan,self.parent_x,x,self.document,
                                   self.emit,exact_saved_pose=True,rescan_all=True)

    def finalize(self, x, value, *, reused=None):
        """Bind the retained geometry to the successful surface verification."""
        identity=signature(x)
        if identity not in self.passes:
            if reused is not None and reused['parameter_vector_signature']==identity:
                write(self.output/'retained_surface.json',reused)
                return
            if np.array_equal(x,self.parent_x):
                return
            if not self(x,value):
                raise ValueError('Retained candidate failed its refreshed surface validation')
        archive, receipt = self.passes[identity]
        write(self.output/'retained_surface.json',dict(**receipt,surface_path=str(archive.resolve())))


def surface_events(scan, events, clock):
    parts=scan['parts'];ids=scan['pair_counts']
    out={}
    for event in events:
        e=(scan['source_seconds']>=event['start_pts']*clock)&(scan['source_seconds']<=event['end_pts']*clock)
        active=e[scan['sample']]
        grouped={}
        for i,j in np.unique(ids[e[ids[:,0]],1:3],axis=0):
            hit=ids[e[ids[:,0]]&(ids[:,1]==i)&(ids[:,2]==j)]
            witness=active & (np.minimum(event['_face_parts'][scan['face_a']],event['_face_parts'][scan['face_b']])==i) & (np.maximum(event['_face_parts'][scan['face_a']],event['_face_parts'][scan['face_b']])==j)
            gap=np.maximum(-scan['initial_gap'][witness],0)
            peak=int(np.argmax(gap)) if len(gap) else None
            grouped[f'{parts[i]}/{parts[j]}']=dict(samples=len(hit),triangle_pairs=int(hit[:,3].sum()),
                oriented_proxy_p95_m=float(np.percentile(gap,95)) if len(gap) else None,
                oriented_proxy_peak_m=float(gap[peak]) if peak is not None else None,
                witness_seconds=float(scan['source_seconds'][scan['sample'][witness][peak]]) if peak is not None else None)
        out[event['id']]=dict(samples=int(e.sum()),pairs=grouped,
            conditional_local_grip_triangle_pairs=int(scan['grip_crossing_counts'][e].sum()),
            topology_uncertain_candidates=int(scan['topology_uncertain_candidates'][e].sum()))
    return out


class SkateReplay:
    def __init__(self,args,log):
        self.args,self.log=args,log
        parent_path=args.source
        ref,camera,info,document,_=load_reference(args.project,args.prepared,parent_path)
        self.ref,self.camera,self.info=ref,camera,info
        self.latest=load(args.latest)
        self.previous=(self.latest if Path(args.latest).resolve()==Path(args.previous).resolve()
                       else load(args.previous))
        self.x,self.xold=(v['x'].copy() for v in (self.latest,self.previous))
        self.clock=source_seconds(ref.p)
        self.tb=time_base(ref.p)
        self.pts=ref.p['pts']
        if len(self.pts)!=8329 or len(set(ref.p['shot_ids']))!=9 or not np.all(np.diff(self.pts)==256):
            raise ValueError('Expected complete Ankara programme clock: nine shots, 8329 native exposures')
        parent_observations=programme_observation_path(args.prepared,document)
        packets=(self.latest,) if self.previous is self.latest else (self.latest,self.previous)
        for packet in packets:
            validate_checkpoint_identity(packet,ref.fixed_checkpoint_identity)
            for key,path in [('prepared_input_sha256',args.prepared/'input.npz'),
                             ('prepared_observations_sha256',parent_observations),
                             ('source_evidence_sha256',parent_path)]:
                if str(packet[key])!=digest(path):raise ValueError('Changed immutable input: '+key)
        for k in ('absolute_knots','absolute_seed_position_controls','absolute_seed_rotation_controls',
                  'relative_skate_knots','relative_skate_seed_rotation_controls','relative_skate_rotation_controls',
                  'sequence_contact_modes','sequence_contact_status','multiplier','penalty','ice_soft_policy_json'):
            np.testing.assert_array_equal(self.latest[k],self.previous[k],err_msg=k)
        source_contract=dict(camera_sha256=info['state_sha256'],programme_pts=self.pts,time_base_value=self.tb)
        review=getattr(args,'native_review',None)
        if review is not None:
            self.document=extend_document(parent_path,review,**source_contract)
        elif 'programme_skate_evidence' in document:
            self.document=document
        else:
            self.document=merge_document(parent_path,args.annotations,args.plane,args.contours,**source_contract)
        observations=getattr(args,'reviewed_observations',None)
        if observations is not None:
            revision=load_json(observations)
            if revision['source_video_sha256']!=document['source_sha256'] or revision['time_base']!=list(self.tb):
                raise ValueError('Detector correspondence review changed the source video or clock')
            self.document['programme_body_observation_revision']=revision
        self.observations_path=programme_observation_path(args.prepared,self.document)
        self.ev=self.document['programme_skate_evidence']
        if self.ev['source_video_sha256']!=document['source_sha256']:
            raise ValueError('Source observations come from a different video')
        self.s0=load_scan(args.latest_surface,args.latest,info['state_sha256'],args.prepared)
        same_surface=(self.previous is self.latest and
                      Path(args.latest_surface).resolve()==Path(args.previous_surface).resolve())
        self.sold=(self.s0 if same_surface else
                   load_scan(args.previous_surface,args.previous,info['state_sha256'],args.prepared))
        self.geometry=CollisionSurface(ref,self.document)
        scans=[self.s0] if same_surface else [self.s0,self.sold]
        self.source_path=self.make_document('initial',common_witnesses(scans,self.geometry))
        ref.source_document=self.document
        ref.absolute_curve=curve_from_packet(self.latest,seed=True)
        ref.relative_skate_curve=shoe_curve_from_packet(self.latest,seed=True)
        self.execution=BodyExecution(BodyExecutionConfig(workers=12,blas_threads=1,
            local_derivatives='analytic',compatibility_backend='cuda_resident',gpu_batch_size=1024))
        self.execution.attach(ref)
        policy=IceSoftPolicy(**json.loads(str(self.latest['ice_soft_policy_json'])))
        self.problem=SequenceJoint(ref,camera,self.observations_path,tuple(self.clock[[0,-1]]),
            self.document.get('supports',[]),args.output,log,'sequence_skates_v1',
            optimize_shared=False,ice_policy=policy)
        p=self.problem
        p.body.compatibility_difference_step=1e-8
        p.al.multiplier[:]=self.latest['multiplier'];p.al.penalty=float(self.latest['penalty'])
        self.mass=ContinuousMassGeometry(ref)
        self.primary=PrimaryActions(p,ref,self.mass,self.document,warm=self.x[17:])
        p.action_factors=SequenceActions(p,ref,self.mass,self.document,self.x[17:],base_factors=self.primary)
        p.checkpoint_extra={}
        # Verify the archived objective before rebinding its unchanged anchor,
        # turn branches and modes to the newly appended source contract.
        self.check_clocks()
        bind_action_objective_context(p,self.primary,parent_path,saved=self.latest,verified_chart_rebase=True)
        bind_action_objective_context(p,self.primary,self.source_path)
        self.weights=configure_programme_weights(p,self.primary,self.document,self.x)
        actions=p.action_factors
        restore_source_masks(actions.contour,self.latest,'source_association_contour_')
        restore_source_masks(actions.extent,self.latest,'source_association_runner_extent_')
        if actions.unassigned is not None and actions.unassigned.episode_factors:
            restore_source_episode(actions.unassigned,self.latest,'source_association_unassigned_',
                                   choice_key='source_association_episode_choice')
        for name,factor in actions.extra_unassigned_episodes:
            restore_source_episode(factor,self.latest,'source_association_extra_'+name+'_')
        self.joints=next(f for f in self.primary.factors if isinstance(f,ProgrammeJointLimits))
        self.ordered=actions.reviewed_skates.ordered
        ordered_saved=self.latest.get('ordered_source_association_json')
        reused=review is None and ordered_saved is not None
        if reused:
            self.ordered.restore(json.loads(str(ordered_saved)))
        else:
            self.ordered.associate(self.x[17:])
        self.emit('ready',native_frames=len(self.pts),shot_count=len(set(ref.p['shot_ids'])),
                  camera_sha256=info['state_sha256'],variables=p.size,source_sha256=digest(self.source_path),
                  inherited_contours=len(actions.contour.rows),active_ordered_rows=sum(a is not None for a in self.ordered.matches),
                  reused_saved_ordered_association=bool(reused),
                  unrepresented_ordered_rows=[r['raw_group_id'] for r,a in zip(self.ordered.rows,self.ordered.matches) if a is None])

    def check_clocks(self):
        state=self.problem.body.state
        np.testing.assert_array_equal(self.problem.body.frames,np.arange(len(self.pts)))
        np.testing.assert_array_equal(self.problem.body.obs['pts'],self.pts)
        nodes=self.problem.body.times
        saved=[(self.latest,self.x)]
        if self.previous is not self.latest:saved.append((self.previous,self.xold))
        for packet,x in saved:
            expected=curve_from_packet(packet)
            actual=state.current_curve(x[17:])
            np.testing.assert_allclose(expected.rotation_controls,actual.rotation_controls,atol=2e-11,rtol=0)
            for t in np.array_split(nodes,9):
                a,b=evaluate_curve(expected,t),evaluate_curve(actual,t)
                np.testing.assert_allclose(a['P'],b['P'],atol=1e-10,rtol=0)
                np.testing.assert_allclose(a['rotations'],b['rotations'],atol=3e-10,rtol=0)

    def make_document(self,version,packet):
        out=self.args.output
        surface=out/f'witnesses_{version}.npz';save(surface,**packet)
        self.document['programme_structure'].update(surface_witnesses=str(surface),
            surface_witnesses_sha256=digest(surface))
        # The historical numerical trust route belongs to a different
        # objective; do not replay its archived direction as a new solve.
        policy=self.document['programme_structure']['solver_step']
        for k in ('archived_source_direction','numerical_trust_seed'):
            policy.pop(k,None)
        target=out/f'canonical_source_{version}.json'
        write(target,self.document)
        return target

    def emit(self,event,**v):
        self.log.write(json.dumps(dict(event=event,**v),ensure_ascii=False)+chr(10))

    def full(self,x,label,*,jacobian=False,verify_rows=True):
        p=self.problem; t=time.perf_counter()
        self.emit('whole_objective_start',label=label,jacobian=jacobian)
        val=p.evaluate(x,jacobian)
        # Verify the actual evaluation, not a flag saying coverage is complete.
        n=len(self.pts); body=p.body
        if body.cache['uv'].shape[0]!=n or self.joints.last_values.shape!=(n,10,3):
            raise ValueError('Incomplete native projection or anatomical replay')
        registry=verify(val['weighted_blocks'],val['residual'],val['jac'] if jacobian else None) if verify_rows else None
        by={b.name:b for b in val['weighted_blocks']}
        required=('body_pixels','dynamic_foot_compatibility','cavity_AL','finite_ice',
                  'manual_native_leg','ordered_native_skate','reviewed_native_support',
                  'full_source_skate_structure','surface_intersection','anatomical_joint_domain',
                  'source_supported_grasp','programme_parent_root_trust')
        if any(k not in by or not len(by[k].residual) for k in required):
            raise ValueError('Missing production rows: '+repr([k for k in required if k not in by or not len(by[k].residual)]))
        if not any(k.endswith('/flight_integrated_com') for k in by):
            raise ValueError('Missing confirmed flight replay')
        if len(by['dynamic_foot_compatibility'].residual)!=n*p.compatibility_width or len(by['anatomical_joint_domain'].residual)!=n*30:
            raise ValueError('Incomplete full-trajectory wear or joint evaluation')
        torso=np.isin(body.oj,[5,6,11,12])
        proj=body.cache['uv'][body.of,body.layout.body_landmarks[getattr(body,'model_oj',body.oj)]]
        observed=body.obs['uv'][body.frames[body.of],body.oj]
        dist=np.linalg.norm(proj-observed,axis=1)
        torso_by_event={}
        for e in self.ev['events']:
            pts=self.pts[body.frames[body.of]]
            ix=torso&(pts>=e['start_pts'])&(pts<=e['end_pts'])
            torso_by_event[e['id']]=dict(p95_px=float(np.percentile(dist[ix],95)),
                    median_sigma_px=float(np.median(body.obs['sigma_px'][body.frames[body.of[ix]],body.oj[ix]])))
        return dict(value=val,registry=registry,cost=float(val['residual']@val['residual']/2),
            weighted=val['weighted_parts'],body=body_groups(p,val),torso_by_event=torso_by_event,
            coverage=dict(native_exposures=n,first_last_pts=self.pts[[0,-1]].tolist(),time_base=list(self.tb),
                native_body_observation_rows=len(by['body_pixels'].residual)//2,wear_exposures=n,
                joint_exposures=n,finite_ice_eval_times=len(p.physics_times),shot_count=len(set(self.ref.p['shot_ids']))),
            anatomy=self.joints.last_values.copy(),cavity=np.maximum(-val['contact']['margin_m'].reshape(-1,p.cavity_rows_per_frame)[1:],0.),
            elapsed_s=time.perf_counter()-t)

    def weights_record(self):
        return dict(policy=self.weights,weight_digest=signature(*[v for k,w in sorted(self.problem.objective_block_weights.items())
             for v in (k,np.asarray(w))]),weights={k:np.asarray(v).tolist() for k,v in self.problem.objective_block_weights.items()})

    def refresh(self,x,packet,version='refreshed'):
        p=self.problem
        self.source_path=self.make_document(version,packet)
        self.ref.source_document=self.document
        for i,old in enumerate(self.primary.factors):
            if type(old) is ProgrammeSurfaceFactor:
                self.primary.factors[i]=ProgrammeSurfaceFactor(p,self.document,packet,scale_m=old.scale_m)
        self.ordered.associate(x[17:])
        p.cache=None;p.cache_x=None
        bind_action_objective_context(p,self.primary,self.source_path)
        self.emit('association_refreshed',source_sha256=digest(self.source_path),
                  ordered_rows=sum(a is not None for a in self.ordered.matches),witnesses=len(packet['sample']))

    def selection(self):
        from .absolute_motion_curve import greville_times
        s=self.problem.body.state; origin=self.clock[0]
        bt=greville_times(s.knots)+origin;st=greville_times(s.shoe_curve.knots)+origin
        def rows(t,a,b):return np.flatnonzero((t>=a)&(t<=b))
        def controls(kn,axis):return (17+69*kn[:,None]+axis).ravel()
        def axes(j):return np.r_[np.arange(3),*[6+3*(k-1)+np.arange(3) for k in j]]
        legs=[1,2,4,5,7,8,10,11]
        narrow=np.unique(np.r_[controls(rows(bt,150.40,151.54),axes(legs)),
            controls(rows(bt,156.78,157.72),axes(legs)),
            controls(rows(bt,159.25,164.27),axes([1,4,7,10]))])
        # Q has the already reviewed lifted whole-turn path; no Q coordinates.
        # Source phases select compact support, including feet through occlusion.
        spans=[(149.82,152.),(156.42,165.82)]
        use=np.unique(np.concatenate([rows(bt,a,b) for a,b in spans]))
        shoe=np.unique(np.concatenate([rows(st,a,b) for a,b in spans]))
        broad=np.r_[controls(use,axes(legs)),17+s.body_size+(shoe[:,None]*6+np.arange(6)).ravel()]
        return narrow,np.union1d(narrow,broad)


def directions(replay, base, *, wear_seed=False):
    """One exact production linearization, then work only on the touched columns."""
    p=replay.problem; val=base['value']; J=val['jac']; r=val['residual']
    narrow,broad=replay.selection()
    sub=J[:,broad].tocsr();sub.eliminate_zeros()
    used=np.diff(sub.indptr)>0
    blocks=base['registry']['blocks']
    scale=np.where(broad>=17+p.body.state.body_size,.08,
        np.where((broad-17)%69<3,.035/p.binding.ratio,.23))
    if wear_seed:
        # Smaller joint/position trust at the cleaner attached-foot seed.
        scale*=.55
    sources={'manual_native_leg','ordered_native_skate','reviewed_native_support'}
    guards={'dynamic_foot_compatibility','cavity_AL','surface_intersection','anatomical_joint_domain'}
    identity_rows={}
    for a in blocks:
        b=next(v for v in val['weighted_blocks'] if v.name==a['name'])
        u=sub[a['start']:a['stop']]
        a['active_rows']=int(np.count_nonzero(np.diff(u.indptr)))
        a['active_columns']=int(len(np.unique(u.indices)))
        a['active_nnz']=int(u.nnz)
        a['explicit_weight']=a['name'] in p.objective_block_weights
        if isinstance(b.provenance,dict) and 'source_pts' in b.provenance:
            stamps=np.asarray(b.provenance['source_pts'])
            a['source_stamp_count']=len(np.unique(stamps))
        identity_rows[a['name']]=slice(a['start'],a['stop'])
    output={};receipt={}
    for label,take,mode in [('A_archived_activity',np.searchsorted(broad,narrow),'full'),
            ('B_attached_source',np.arange(len(broad)),'source'),
            ('B_attached_joint',np.arange(len(broad)),'full'),
            ('B_structural_correction',np.arange(len(broad)),'structural')]:
        w=np.ones(len(r))
        if mode=='source':
            w.fill(0.)
            for k in sources:w[identity_rows[k]]=1.
        if mode=='structural':
            # Proposal preconditioning only; all trials use the same
            # declared acceptance objective and source weights.
            for k in guards:w[identity_rows[k]]=2.
            if wear_seed:
                # The final replay still uses the unmodified objective weights.
                for k in sources:w[identity_rows[k]]=2.
        mask=used&(w>0)
        A=sub[mask][:,take]@diags(scale[take])
        result=lsmr(diags(w[mask])@A,-w[mask]*r[mask],damp=.20,atol=2e-5,btol=2e-5)
        m=np.zeros(len(broad)); m[take]=np.clip(result[0],-1.5,1.5)*scale[take]
        act=sub@m
        step=np.zeros(p.size);step[broad]=m
        output[label]=step
        receipt[label]=dict(columns=len(take),nonzero_controls=int(np.count_nonzero(step)),
            proposal_rows=int(mask.sum()),proposal_stop=int(result[1]),iterations=int(result[2]),
            total_rTJd=float(r@act),
            parts={a['name']:dict(rTJd=float(r[a['start']:a['stop']]@act[a['start']:a['stop']]),
                predicted_cost=float(np.linalg.norm(r[a['start']:a['stop']]+act[a['start']:a['stop']])**2/2))
                   for a in blocks})
    base['value']['jac']=None
    for b in base['value']['weighted_blocks']:b.jacobian=None
    return output,receipt
