"""Finite advertisement support as source existence, separate from localization.

No corner observation, tangential likelihood or new model coordinate is added.
Current native evidence supplies conservative station intervals.  The finite
projected polyline and observed-row camera are exactly the ordinary FIT model.
A covered station alone does not certify the normal position or visibility.
"""
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
from types import MethodType
import numpy as np
from scipy.sparse import csr_matrix, vstack
from .first_shot_source_transition import reference, read


@dataclass(frozen=True)
class AdSupportSection:
    frame: int
    native_pts: int
    target_key: str
    source_feature_id: str
    segment_index: int
    query_uv: np.ndarray
    tangent: np.ndarray
    station_uncertainty_px: float
    source_point_ids: tuple
    native_readout_indices: tuple
    source_packet: dict


def projected_station_margin(projected, segment_valid, query, tangent, uncertainty=0.):
    """Union of finite segment intervals at the fixed source section.

    The returned active endpoint and sign are the exact piecewise derivative
    of this value.  A profile's normal position remains the original W factor.
    """
    p=np.asarray(projected,float); valid=np.asarray(segment_valid,bool)
    query=np.asarray(query,float);tangent=np.asarray(tangent,float)
    if (p.ndim!=2 or p.shape[1]!=2 or valid.shape!=(len(p)-1,)
            or query.shape!=(2,) or tangent.shape!=(2,)
            or not np.isfinite(np.r_[query,tangent,uncertainty]).all()
            or not np.isclose(tangent@tangent,1.,rtol=0,atol=2e-12) or uncertainty<0):
        raise ValueError('Finite projected material and fixed unit source section required')
    station=(p-query)@tangent
    valid=valid & np.isfinite(station[:-1]) & np.isfinite(station[1:])
    if not valid.any():raise ValueError('Finite advertisement has no central forward projected segment')
    a,b=station[:-1],station[1:];forward=a<=b
    lower=np.where(forward,a,b);upper=np.where(forward,b,a)
    low_face=-lower;high_face=upper;choose_low=low_face<=high_face
    margins=np.where(valid,np.minimum(low_face,high_face)+uncertainty,-np.inf)
    edge=int(np.argmax(margins));low=bool(choose_low[edge])
    endpoint=edge+(int(not forward[edge]) if low else int(forward[edge]))
    return float(margins[edge]),endpoint,(-1. if low else 1.),edge


def compile_ad_support_sections(frames):
    """Use only retained, named ad-edge estimators with explicit native support.

    The caller supplies the actual retained formal FIT frames. Historical
    extraction allowed_for_fit flags do not override that current admission.
    Original rounded readouts have a known coordinate quantization interval.
    A nearest existing readout supplies an existence section, not a claimed
    exact material endpoint. Unsupported estimator kinds remain explicit.
    """
    packets={};sections=[];unsupported=[]
    for frame in frames:
        grouped=OrderedDict()
        for row in frame['rows']:
            target=row['target']
            if target.get('kind')!='ad_edge' or row['point_correspondence']:continue
            key=(row['_target_key'],row.get('source_feature_id'),row.get('source_segment_index'))
            grouped.setdefault(key,[]).append(row)
        if not grouped:continue
        ref=frame['source_packet'];token=(ref['path'],ref['sha256'])
        if token not in packets:packets[token]=read(ref)
        packet=packets[token];source=packet['source']
        if int(source['native_pts'])!=int(frame['native_pts']) or int(source['actual_pts'])!=int(frame['native_pts']) or source['time_base']!='1/12800':
            raise ValueError('Finite ad support belongs to another native exposure')
        features={f['id']:f for f in packet['features']}
        for (key,fid,segment_index),rows in grouped.items():
            address=dict(frame=int(frame['index']),target_key=key,source_feature_id=fid,segment_index=segment_index,source_packet=ref)
            try:
                f=features[fid];segment=f['segments'][segment_index]
                target=rows[0]['target'];expected='Ad'+str(target['advertisement_id'])+'.'+target['role']
                if (fid!=expected
                        or segment.get('measurement_origin')!='current_native_joint_straight_material_edge_fit'
                        or segment.get('source_evidence_measured_in_this_frame') is not True
                        or segment.get('pixels_measured_in_this_frame') is not False):
                    raise ValueError('This estimator has no explicit current-source straight material support contract')
                native=np.asarray(segment['source_native_readout_uv'],float);representatives=np.asarray(segment['uv'],float)
                if native.ndim!=2 or native.shape[1]!=2 or len(native)<2 or not np.isfinite(native).all():
                    raise ValueError('Actual finite native advertisement support is missing')
                tangent=representatives[-1]-representatives[0];tangent=tangent/np.linalg.norm(tangent)
                if not np.isfinite(tangent).all():raise ValueError('Source support direction is degenerate')
                ids=[];queries=[];indices=[]
                for row in rows:
                    if row['source_packet']!=ref or len(row['uv'])!=1 or len(row['source_point_ids'])!=1:
                        raise ValueError('Retained source section must preserve its original packet and point order')
                    vertex=int(row['original_vertex_indices'][0]);original=representatives[vertex]
                    if not np.array_equal(np.asarray(row['uv'])[0],original):
                        raise ValueError('Revised advertisement mean needs its own complete native-support certificate')
                    station=(native-original)@tangent;j=int(np.argmin(abs(station)))
                    queries.append(native[j]);indices.append(j);ids.append(row['source_point_ids'][0])
                # source_native_readout_uv is explicitly rounded to 3 decimals
                # by extract_first_shot_board_source.segment_records.
                tolerance=.0005*float(np.sum(abs(tangent)))
                uv=np.asarray(queries);uv.setflags(write=False);tangent.setflags(write=False)
                sections.append(AdSupportSection(**{k:address[k]for k in ('frame','target_key','source_feature_id','segment_index','source_packet')},
                    native_pts=int(frame['native_pts']),query_uv=uv,tangent=tangent,station_uncertainty_px=tolerance,
                    source_point_ids=tuple(ids),native_readout_indices=tuple(indices)))
            except (ValueError,KeyError,IndexError) as error:
                unsupported.append(dict(address,reason=str(error)))
    return tuple(sections),tuple(unsupported)


def evaluate_ad_support(state,physical,sections,worlds,derivatives=None):
    """Evaluate value and active branch from the same actual row projection."""
    from .first_shot_row_camera import prepare_frame_row_camera
    from .first_shot_source_kernel import _interfaces,_project_target
    api=_interfaces();trajectories={};margins=[];groups=[];records=[];ri=[];ci=[];vv=[]
    physical=np.asarray(physical);world_columns=np.asarray(state.world_parameter_indices,int)
    jacobian=derivatives is not None
    for group,section in enumerate(sections):
        if section.frame not in trajectories:trajectories[section.frame]=prepare_frame_row_camera(state,physical,section.frame)
        trajectory=trajectories[section.frame];world=np.asarray(worlds[section.target_key],float)
        D=np.empty(world.shape+(0,)) if not jacobian else np.asarray(derivatives[section.target_key])
        for point,query in enumerate(section.query_uv):
            camera,columns,chain=trajectory.camera15_mapping(float(query[1]),jacobian=jacobian)
            p=_project_target(camera,world,D,False,api,jacobian)
            margin,endpoint,sign,edge=projected_station_margin(p.uv,p.segment_valid,query,section.tangent,section.station_uncertainty_px)
            row=len(margins);margins.append(margin);groups.append(group)
            if jacobian:
                local=sign*section.tangent@np.c_[p.geometry_jac[endpoint],p.camera_jac[endpoint]@chain]
                cols=np.r_[world_columns,columns]
                if local.shape!=cols.shape:raise ValueError('Finite support world/camera derivative columns do not match the actual programme world chart')
                keep=local!=0.
                ri.extend([row]*int(keep.sum()));ci.extend(cols[keep].tolist());vv.extend(local[keep].tolist())
            records.append(dict(frame=section.frame,native_pts=section.native_pts,target_key=section.target_key,
                source_point_id=section.source_point_ids[point],native_readout_index=section.native_readout_indices[point],
                margin_px=margin,uncovered_px=max(0.,-margin),selected_edge=edge,endpoint=endpoint,derivative_sign=sign,
                query_uv=query.tolist(),station_uncertainty_px=section.station_uncertainty_px))
    margins=np.asarray(margins);groups=np.asarray(groups,int);violations=np.zeros(len(sections))
    np.maximum.at(violations,groups,np.maximum(0.,-margins))
    J=csr_matrix((vv,(ri,ci)),shape=(len(margins),len(physical)))if jacobian else None
    return dict(margins=margins,group_indices=groups,group_violations=violations,J=J,records=records,
        branch_signature=hashlib.sha256(str([(r['selected_edge'],r['endpoint'],r['derivative_sign'])for r in records]).encode()).hexdigest())


class FiniteAdCoveragePolicy:
    """Reuse original phase-I and truth-restoration gates on these support faces."""
    def __init__(self,owner):
        self.owner=owner;self.sections,self.unsupported=compile_ad_support_sections(owner.programme.bundle.frames)
        if not self.sections:raise ValueError('No authenticated finite advertisement support sections')
        self.cache=OrderedDict();self.context={}
    def evaluate(self,q,jacobian=False):
        for ref in getattr(self,'producer_references',()):
            if reference(ref['path']) != ref:raise ValueError('Finite advertisement support producer changed')
        q=np.asarray(q,float);key=(q.tobytes(),bool(jacobian))
        if key in self.cache:return self.cache[key]
        physical,chain=self.owner.effective_q_and_chain(q);physical=np.asarray(physical)[:self.owner.programme_count]
        # The programme world includes the five shared transverse deviations.
        # bundle.world is the unshifted 166-column constructor, not FIT's world171.
        worlds,derivatives=self.owner.programme.world(physical,bool(jacobian))
        result=evaluate_ad_support(self.owner.programme.state,physical,self.sections,worlds,derivatives)
        if jacobian:result['J']=(result['J']@csr_matrix(chain)[:len(physical)]).tocsr()
        self.cache[key]=result
        while len(self.cache)>3:self.cache.popitem(last=False)
        return result
    def directional_check(self,q,direction,*,epsilon=2e-5,confirmation_epsilon=1e-5):
        from types import SimpleNamespace
        from .first_shot_source_directional_QA import check_complete_source_direction
        current=self.evaluate(q,True);n=len(current['margins']);branch_records=[]
        def blocks(candidate,jacobian):
            value=self.evaluate(candidate,jacobian)
            branch_records.append(dict(q_sha256=hashlib.sha256(np.asarray(candidate).tobytes()).hexdigest(),branch_signature=value['branch_signature']))
            yield SimpleNamespace(r=value['margins'],J=value['J'],scales=np.ones(n),quadratic=np.ones(n,bool),
                row_ledger=dict(family='actual-finite-ad-material-support-constraints',rows=n))
        proxy=SimpleNamespace(fixed=self.owner.fixed,scales=np.ones(n),guard=self.owner.guard,
            source_blocks=blocks,programme=self.owner.programme,programme_count=self.owner.programme_count,
            effective_q_and_chain=self.owner.effective_q_and_chain,source_producers=())
        baseline=list(blocks(q,True));reports=[check_complete_source_direction(proxy,q,direction,baseline,epsilon=h)
            for h in (epsilon,confirmation_epsilon)]
        return dict(passed=all(r['passed']for r in reports),constraint_rows=n,full_camera_and_world_direction=True,
            stencils=reports,branch_records=branch_records,producer=reference(__file__),
            original_FIT_programme_world_interface_used=True,source_W_and_likelihood_unchanged=True)

    def domain(self,q):
        from .first_shot_full_physical import complete_mount_domain
        from structured_programme_source_candidate.feasibility_priority_restoration import contract_original_support_bounds
        original=complete_mount_domain(self.owner,q);current=self.evaluate(q,True)
        physical=csr_matrix(original['matrix']);matrix=vstack((physical,current['J']),format='csr')
        lower=np.r_[original['lower'],-current['margins']]
        groups=np.r_[np.full(physical.shape[0],-1),current['group_indices']].astype(int)
        free=np.setdiff1d(np.arange(len(q)),self.owner.fixed)
        contraction=contract_original_support_bounds(np.full(len(free),-np.inf),np.full(len(free),np.inf),
            matrix[:,free].toarray(),lower,groups,current['group_violations'])
        self.context=dict(q_sha256=hashlib.sha256(np.asarray(q).tobytes()).hexdigest(),contraction=contraction)
        return dict(original,matrix=matrix,lower=contraction['lower'],report=dict(original['report'],
            finite_ad_coverage=dict(sections=len(self.sections),source_sections=len(current['margins']),
                unsupported=len(self.unsupported),maximum_uncovered_px=float(current['group_violations'].max(initial=0.)),
                branch_signature=current['branch_signature'],phase_I=contraction['report'])))
    def install(self,entry_reference):
        owner=self.owner;previous_kwargs=owner.solver_kwargs;previous_identity=owner.checkpoint_identity
        refs=[reference(__file__),entry_reference]
        self.producer_references=deepcopy(refs)
        receipt=dict(schema='rinkspace.finite-ad-source-support-restoration.v1',producers=refs,
            support_sections=len(self.sections),unsupported_sections=deepcopy(self.unsupported),
            original_normal_W_cost_and_parameters_unchanged=True,new_corner_or_tangential_likelihood=False,
            source_existence_checked_by_original_observed_row_finite_polyline=True,
            full_existing_nonlinear_physical_guard_required=True,
            source_cost_may_increase_only_during_explicit_each_group_truth_restoration=True)
        for name in ('source_producers','recovery_references','_references'):
            if hasattr(owner,name):
                for ref in refs:
                    if ref not in getattr(owner,name):getattr(owner,name).append(deepcopy(ref))
        def identity(this):return dict(previous_identity(),finite_ad_support_policy=deepcopy(receipt))
        def kwargs(this):
            result=previous_kwargs();old_accept=result.get('source_trial_acceptance');old_converged=result.get('numerical_convergence_guard')
            if old_accept is not None:raise ValueError('An existing truth policy needs explicit combined per-group acceptance')
            def accept(parent,trial,*,before_cost,after_cost,default_source_cost_accepts,trial_context):
                from structured_programme_source_candidate.feasibility_priority_restoration import actual_restoration_trial_decision
                before=self.evaluate(parent);after=self.evaluate(trial);old=before['group_violations'];new=after['group_violations']
                if np.max(old,initial=0.)<=1e-8:
                    valid=bool(np.max(new,initial=0.)<=1e-8)
                    return dict(accept=bool(valid and default_source_cost_accepts),mode='source_objective',
                        finite_ad_coverage_valid=valid,maximum_uncovered_px=float(np.max(new,initial=0.)),
                        rejection_reason='finite advertisement support missing'if not valid else'original source objective not lower')
                if self.context.get('q_sha256')!=hashlib.sha256(np.asarray(parent).tobytes()).hexdigest():
                    raise ValueError('Finite support phase-I context belongs to another state')
                contraction=self.context['contraction'];decision=actual_restoration_trial_decision(old,new,
                    common_fraction=contraction['common_fraction'],step_fraction=trial_context.get('alpha')or 1.,
                    physical_guard_passed=trial_context['physical_and_complete_source_guard_passed'],
                    previous_source_cost=before_cost,trial_source_cost=after_cost)
                return dict(accept=decision['accepted'],mode='truth_restoration',
                    rejection_reason=decision['reason'],finite_ad_support=decision,
                    previous_group_violations_px=old.tolist(),trial_group_violations_px=new.tolist())
            result['source_trial_acceptance']=accept
            result['numerical_convergence_guard']=lambda q:bool(np.max(self.evaluate(q)['group_violations'],initial=0.)<=1e-8 and (old_converged is None or old_converged(q)))
            result['checkpoint_identity']['finite_ad_support_policy']=deepcopy(receipt)
            return result
        owner.solver_kwargs=MethodType(kwargs,owner);owner.checkpoint_identity=MethodType(identity,owner)
        return deepcopy(receipt)
