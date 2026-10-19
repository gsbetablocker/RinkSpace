"""Same-state full formal-source directional QA with original nonlinear guards.

No camera-only fallback, row dropping, source reweighting or frozen world is
used. All complete correlated W groups and appended factors participate.
"""
from copy import deepcopy
import hashlib
import numpy as np


def _sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def check_complete_source_direction(owner,q,direction,blocks,*,epsilon):
    q=np.asarray(q,float);d=np.asarray(direction,float);fixed=np.asarray(owner.fixed,int)
    if q.shape!=d.shape or not np.isfinite(d).all() or np.any(d[fixed]!=0.) or not np.any(d):
        raise ValueError('Nonzero complete fixed-coordinate-preserving source probe required')
    blocks=list(blocks);r=np.concatenate([b.r for b in blocks]);analytic=np.concatenate([np.asarray(b.J@d).ravel()for b in blocks])
    if len(r)!=len(owner.scales):raise ValueError('QA must contain the complete original formal source inventory')
    # Cache keys are exact model/source identities. Only their original mapping
    # is restored after read-only trials; no trial result is relabelled as q.
    saved=[]
    for obj in (owner,owner.programme.bundle):
        for name in ('_world_cache','_motion_cache','_finite_coverage_cache'):
            cache=vars(obj).get(name)
            if isinstance(cache,dict):saved.append((cache,cache.copy()))
    attempts=[]
    def values(candidate):
        owner.guard(candidate)
        current=list(owner.source_blocks(candidate,False))
        if len(current)!=len(blocks):raise ValueError('Directional QA changed the source block inventory')
        for before,after in zip(blocks,current,strict=True):
            if (before.r.shape!=after.r.shape or not np.array_equal(before.scales,after.scales)
                    or not np.array_equal(before.quadratic,after.quadratic)
                    or before.row_ledger!=after.row_ledger):
                raise ValueError('Directional QA changed original source membership, W roles, scales or masks')
        return np.concatenate([b.r for b in current])
    def trial(sign,multiple=1.):
        step=sign*multiple*epsilon;candidate=q+step*d;candidate[fixed]=q[fixed]
        try:value=values(candidate)
        except (ValueError,RuntimeError)as error:
            attempts.append(dict(signed_step=step,failure=type(error).__name__+': '+str(error)));return None
        attempts.append(dict(signed_step=step,failure=None,q_sha256=_sha(candidate),maximum_formal_displacement=float(np.max(abs(value-r)))));return value
    try:
        owner.guard(q);plus=trial(1.);minus=trial(-1.)
        if plus is not None and minus is not None:finite=(plus-minus)/(2*epsilon);method='central_original_full_domain'
        elif plus is not None or minus is not None:
            sign=1. if plus is not None else -1.;first=plus if plus is not None else minus;second=trial(sign,2.)
            if second is None:raise ValueError('Full source QA requires a measurable second-order feasible stencil')
            finite=sign*(-3*r+4*first-second)/(2*epsilon);method='second_order_one_sided_original_full_domain'
        else:raise ValueError('No original-domain side for the requested full-source direction')
        error=analytic-finite;details=[];cursor=0;frame_errors=[]
        for b in blocks:
            stop=cursor+len(b.r);err=error[cursor:stop];fd=finite[cursor:stop];ledger=b.row_ledger
            details.append(dict(family=ledger.get('family'),rows=len(err),max_absolute=float(np.max(abs(err))),
                analytic_norm=float(np.linalg.norm(analytic[cursor:stop])),finite_norm=float(np.linalg.norm(fd)),
                relative_norm=float(np.linalg.norm(err)/max(np.linalg.norm(fd),1e-15))))
            local=0
            for frame in ledger.get('frames',()):
                n=int(frame['formal_rows']);part=err[local:local+n]
                frame_errors.append(dict(frame=int(frame['programme_index']),native_pts=int(frame['native_pts']),
                    rows=n,max_absolute=float(np.max(abs(part),initial=0.)),
                    relative_norm=float(np.linalg.norm(part)/max(np.linalg.norm(fd[local:local+n]),1e-15))))
                local+=n
            cursor=stop
        association=dict(applicable=False)
        expected_refs=[ref for ref in getattr(owner,'source_producers',())if str(ref['path']).replace('\\','/').endswith(('/s1_ordered_1017.json','/s1_ordered_1017b.json','/s1_ordered_1017c.json'))]
        if expected_refs:
            from .first_shot_source_transition import read
            expected=read(expected_refs[-1]);physical,_=owner.effective_q_and_chain(q);physical_sha=_sha(np.asarray(physical)[:owner.programme_count]);frames={}
            for receipt in owner.programme.worker_receipts:
                for frame in receipt.get('ordered_contour_associations',()):
                    if frame['q_sha256']==physical_sha:frames[int(frame['frame'])]=frame
            groups=[g for f in frames.values()for g in f['groups']]
            associated=[g for g in groups if g['status']=='associated'];single=[g for g in groups if g['status']=='unqualified_single_source_vertex'];other=[g for g in groups if g['status']not in ('associated','unqualified_single_source_vertex')]
            if expected.get('association_method')is not None:
                other.extend(g for g in associated if g.get('association_method')!=expected['association_method'])
            points=sum(g['source_points']for g in associated);single_points=sum(g['source_points']for g in single)
            eligible=bool(len(frames)==1248 and len(associated)==expected['contiguous_arcs']and points==expected['qualified_points']and single_points==expected['singletons_kept']and not other)
            association=dict(applicable=True,passed=eligible,manifest=expected_refs[-1],physical_q_sha256=physical_sha,
                actual_frames=len(frames),associated_arcs=len(associated),qualified_points=points,
                singletons_retained=single_points,unexpected_groups=other,actual_source_worker_records=True)
        maxabs=float(np.max(abs(error)));relative=float(np.linalg.norm(error)/max(np.linalg.norm(finite),1e-15))
        return dict(schema='rinkspace.full-current-formal-source-directional-QA.v1',
            passed=bool(maxabs<1e-3 and relative<1e-4 and association.get('passed',True)),max_absolute=maxabs,relative_norm=relative,
            method=method,epsilon=float(epsilon),source_association_coverage=association,canonical_q_sha256=_sha(q),direction_sha256=_sha(d),
            direction_norm=float(np.linalg.norm(d)),complete_formal_rows=len(r),blocks=details,
            worst_frames=sorted(frame_errors,key=lambda x:-x['max_absolute'])[:20],attempts=attempts,
            all_current_correlated_W_groups_retained=True,camera_only_fallback=False,
            new_finite_ad_restoration_is_a_separate_acceptance_gate=True,
            independent_primitive_projection_math_certified=False)
    finally:
        for cache,old in saved:cache.clear();cache.update(old)
        actual,_=owner.effective_q_and_chain(q);owner.programme.state.scene(np.asarray(actual)[:owner.programme_count])


def checked_source_factory(owner,factory,*,direction,epsilon,write_report,confirmation_epsilon=None):
    if confirmation_epsilon is not None and not 0. < confirmation_epsilon < epsilon:
        raise ValueError("Confirmation step must be positive and smaller than the primary step")
    checked=False
    def source(q,jacobian):
        nonlocal checked
        if not jacobian or checked:
            yield from factory(q,jacobian);return
        blocks=list(factory(q,True))
        report=check_complete_source_direction(owner,q,direction,blocks,epsilon=epsilon)
        if confirmation_epsilon is not None:
            confirmation=check_complete_source_direction(owner,q,direction,blocks,epsilon=confirmation_epsilon)
            report=dict(report,passed=bool(report['passed'] and confirmation['passed']),
                max_absolute=max(report['max_absolute'],confirmation['max_absolute']),
                relative_norm=max(report['relative_norm'],confirmation['relative_norm']),
                both_original_domain_stencils_required=True,stencils=[report,confirmation])
        write_report(report)
        if not report['passed']:raise ValueError('Complete current formal-source directional check failed; all source rows retained')
        checked=True
        yield from blocks
    return source
