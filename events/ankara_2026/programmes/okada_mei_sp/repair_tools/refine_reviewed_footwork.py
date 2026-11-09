from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import lsmr

from calibration.program_tracking.body_tracking.absolute_motion_curve import greville_times
from calibration.program_tracking.body_tracking.programme_skate_evidence import ProgrammeSkateSourceFactors
from calibration.program_tracking.body_tracking.programme_skate_replay import (
    SkateReplay, common_witnesses, refresh_surface, save, surface_events, write,
)
from calibration.program_tracking.body_tracking.rebuild_prepare import digest
from calibration.program_tracking.body_tracking.weighted_blocks import signature
from scripts.refit_programme_skates import SOURCES, evaluate_guard, serial, source_derivative


def active_controls(replay, event):
    state = replay.problem.body.state
    period = replay.tb[0] / replay.tb[1]
    lower = event['start_pts'] * period - replay.clock[0]
    upper = event['end_pts'] * period - replay.clock[0]
    body_centres = greville_times(state.knots)
    skate_centres = greville_times(state.shoe_curve.knots)
    body_knots = np.flatnonzero((body_centres >= lower) & (body_centres <= upper))
    shoe_knots = np.flatnonzero((skate_centres >= lower) & (skate_centres <= upper))
    joints = [joint + foot for foot in event['active_feet'] for joint in (1, 4, 7, 10)]
    axes = np.concatenate([6 + 3 * (joint - 1) + np.arange(3) for joint in joints])
    shoe_axes = np.concatenate([3 * foot + np.arange(3) for foot in event['active_feet']])
    body_columns = (17 + 69 * body_knots[:, None] + axes).ravel()
    shoe_columns = (17 + state.body_size + 6 * shoe_knots[:, None] + shoe_axes).ravel()
    columns = np.sort(np.r_[body_columns, shoe_columns])
    if len(body_knots) == 0 or len(shoe_knots) == 0:
        raise ValueError('Missing continuous spline support for ' + event['id'])
    begin = min(state.knots[body_knots[0]], state.shoe_curve.knots[shoe_knots[0]])
    end = max(state.knots[body_knots[-1] + 4], state.shoe_curve.knots[shoe_knots[-1] + 4])
    return columns, (begin + replay.clock[0], end + replay.clock[0])


def directions(replay, base, events):
    value = base['value']
    J, residual = value['jac'], value['residual']
    blocks = {b['name']: slice(b['start'], b['stop']) for b in base['registry']['blocks']}
    rows = value['weighted_blocks']
    anchors = {}
    all_controls = {}
    direction_details = {}
    for event in events:
        name = event['id']
        indices, support = active_controls(replay, event)
        sub = J[:, indices].tocsr()
        sub.eliminate_zeros()
        used = np.flatnonzero(np.diff(sub.indptr))
        scale = np.where(indices < 17 + replay.problem.body.state.body_size, .19, .12)
        roles = np.ones(len(used))
        for k in ('dynamic_foot_compatibility', 'cavity_AL', 'anatomical_joint_domain', 'surface_intersection'):
            rr = blocks[k]
            roles[(used >= rr.start) & (used < rr.stop)] = 1.6
        for k in SOURCES:
            rr = blocks[k]
            roles[(used >= rr.start) & (used < rr.stop)] = 1.5
        design = diags(roles) @ sub[used] @ diags(scale)
        candidate = lsmr(design, -roles * residual[used], damp=.26, atol=2e-5, btol=2e-5, maxiter=160)
        update = scale * np.clip(candidate[0], -1.4, 1.4)
        action = sub @ update
        parts = {}
        for key in ('body_pixels', 'finite_ice', 'anatomical_joint_domain', 'surface_intersection',
                    'cavity_AL', 'dynamic_foot_compatibility', *SOURCES):
            s = blocks[key]
            parts[key] = dict(rTJd=float(residual[s] @ action[s]), d_norm=float(np.linalg.norm(action[s])))
        source_slope = sum(parts[k]['rTJd'] for k in SOURCES)
        overall_slope = float(residual @ action)
        action_norm = float(action @ action)
        step = min(1., max(0., -overall_slope / max(action_norm, 1e-12)))
        direction_details[name] = dict(columns=len(indices), active_rows=len(used), support_seconds=support,
            rTJd=overall_slope, source_rTJd=source_slope, parts=parts, linear_fraction=step,
            iterations=int(candidate[2]), stop=int(candidate[1]), active_controls=indices.tolist())
        all_controls[name] = (indices, support)
        if overall_slope < 0 and source_slope < 0 and step > 0:
            anchors[name] = (indices, step * update)
        replay.emit('direction', event_id=name, columns=len(indices), rows=len(used),
                    source_rTJd=source_slope, full_rTJd=overall_slope, fraction=step)
    return anchors, all_controls, direction_details


def source_outcomes(replay, before, after, events):
    result = {}
    previous, current = (state['value']['action_metrics'] for state in (before, after))
    for event in events:
        name = event['id']
        result[name] = {}
        for key in SOURCES:
            seed, fit = ([r for r in table[key] if r['event_id'] == name] for table in (previous, current))
            if not seed:
                continue
            series = []
            for a, b in zip(seed, fit, strict=True):
                if a['id'] != b['id']:
                    raise ValueError('Source row identity changed')
                u = a.get('distance_px', a.get('outside_candidate_band_px'))
                v = b.get('distance_px', b.get('outside_candidate_band_px'))
                if u is None or v is None:
                    continue
                series.append([float(np.percentile(u, 95)), float(np.percentile(v, 95))])
            if series:
                data = np.asarray(series)
                result[name][key] = dict(rows=len(series), rms_px=np.sqrt(np.mean(data**2, axis=0)).tolist(),
                    worst_px=data.max(axis=0).tolist(), maximum_source_pts=int(seed[int(np.argmax(data[:, 1]))]['pts'])
                    if len(series) == len(seed) else None)
    return result


def replay_trial(replay, x, version, controls, events, cached_geometry=None):
    output = replay.args.output
    if cached_geometry is None:
        scan, coverage = refresh_surface(replay.problem, replay.s0, replay.x, x, replay.document, replay.emit)
    else:
        from calibration.program_tracking.body_tracking.rebuild_prepare import load
        geometry_receipt = cached_geometry
        expected_parameter = replay.args.proposals_from/'candidate.npz'
        if (digest(expected_parameter)!=geometry_receipt['parameter_sha256']
                or digest(geometry_receipt['candidate_surface'])!=geometry_receipt['candidate_surface_sha256']
                or geometry_receipt['same_camera_sha256']!=replay.info['state_sha256']):
            raise ValueError('Saved geometry cannot be reused for a different continuous state')
        np.testing.assert_array_equal(load(expected_parameter)['x'],x)
        scan=load(geometry_receipt['candidate_surface'])
        for key in ('times','source_pts','parts','surface_response_policy'):
            np.testing.assert_array_equal(scan[key],replay.s0[key],err_msg=key)
        coverage=dict(geometry_receipt['trial']['coverage'],geometry_reused_for_exact_state=True,
            reused_geometry_sha256=geometry_receipt['candidate_surface_sha256'])
        replay.emit('reuse_actual_surface',samples=coverage['checked_samples'])
    save(output / ('surface_'+version+'.npz'), **scan)
    previous_rows = {r['id'] for r in replay.ordered.association_record() if r['match'] is not None}
    merged = common_witnesses([replay.s0, scan], replay.geometry)
    replay.refresh(x, merged, version)
    current_rows = {r['id'] for r in replay.ordered.association_record() if r['match'] is not None}
    missing = sorted(previous_rows - current_rows)
    base = replay.full(replay.x, 'parent_'+version)
    trial = replay.full(x, 'candidate_'+version)
    guard = evaluate_guard(replay, base, trial)
    by_source = source_outcomes(replay, base, trial, events)
    expanded = []
    for event in events:
        _, (a, b) = controls[event['id']]
        period = replay.tb[0] / replay.tb[1]
        lower, upper = int(np.floor(a/period)), int(np.ceil(b/period))
        expanded.append(dict(event, start_pts=lower, end_pts=upper, _face_parts=replay.geometry.parts))
    expanded.append(dict(id='whole_programme', start_pts=int(replay.pts[0]),
                         end_pts=int(replay.pts[-1]), _face_parts=replay.geometry.parts))
    before_surface, after_surface = (surface_events(s, expanded, replay.tb[0]/replay.tb[1])
                                     for s in (replay.s0, scan))
    surface = dict(coverage=coverage, previous=before_surface, candidate=after_surface)
    failed = {e['id']: [] for e in events}
    global_failures = []
    for item in guard['failures']:
        if item == 'source_rows_outside_finite_allowance':
            for row in guard['source_rows']:
                stamp = row['pts'] * replay.tb[0] / replay.tb[1]
                targets = [name for name, (_, (a, b)) in controls.items() if a <= stamp <= b]
                for name in targets:
                    failed[name].append('source_row')
                if not targets:
                    global_failures.append('source_change_outside_authorized_support')
        elif ':' in item and item.split(':',1)[1] in failed:
            kind, name = item.split(':',1)
            failed[name].append(kind)
        elif item.endswith(':steps_extension'):
            failed['s09_extension'].append(item)
        else:
            global_failures.append(item)
    for event in expanded:
        name=event['id']
        new = sorted(set(after_surface[name]['pairs']) - set(before_surface[name]['pairs']))
        after_surface[name]['added_pair_types'] = new
        if name in failed and new:
            failed[name].append('added_surface_pair_types')
        if name == 'whole_programme' and new and not any('added_surface_pair_types' in values for values in failed.values()):
            global_failures.append('added_surface_pair_outside_events')
        if name in failed:
            mask = (replay.pts >= event['start_pts']) & (replay.pts <= event['end_pts'])
            v0, v1 = base['anatomy'][mask], trial['anatomy'][mask]
            if np.any((v1 - v0 > .25) & (v0 < .25) & (v1 > .50)):
                failed[name].append('anatomical_excursion_in_spline_support')
            families = by_source[name].values()
            if families and not any(f['rms_px'][1] + 1e-8 < f['rms_px'][0] for f in families):
                failed[name].append('no_source_descent')
    if missing:
        global_failures.append('lost_visible_paths')
    result = dict(version=version, coverage=coverage, candidate_cost=trial['cost'], parent_cost=base['cost'],
                  event_failures=failed, global_failures=global_failures,
                  source=by_source, missing_matches=missing, guard=guard)
    write(output / ('trial_'+version+'.json'), result)
    write(output / ('surface_'+version+'.json'), surface)
    replay.emit('complete_trial', **{k: result[k] for k in ('version','candidate_cost','parent_cost','event_failures','global_failures')})
    return result, base, trial


def checkpoint(replay, x, result):
    p = replay.problem
    path = replay.args.output
    latest = replay.latest
    for k in ('prepared_input_sha256','prepared_observations_sha256','programme_context_json',
              'parent_accepted_steps','source_association_origin'):
        p.checkpoint_extra[k] = latest[k].copy()
    for k in latest:
        if k.startswith('source_association_') and k not in p.checkpoint_extra:
            p.checkpoint_extra[k] = latest[k].copy()
    p.checkpoint_extra['prepared_observations_sha256']=np.array(digest(replay.observations_path))
    associations = replay.ordered.association_record()
    write(path/'ordered_associations.json',associations)
    p.checkpoint_extra.update(objective_phase=np.array('programme_skate_evidence'),
        objective_policy_sha256=np.array(digest(path/'objective_weights.json')),
        objective_block_weights_sha256=np.array(replay.weights_record()['weight_digest']),
        programme_structure_json=np.array(json.dumps(replay.document['programme_structure'],sort_keys=True)),
        source_evidence_sha256=np.array(digest(replay.source_path)),
        ordered_source_association_sha256=np.array(digest(path/'ordered_associations.json')),
        ordered_source_association_json=np.array(json.dumps(associations,sort_keys=True)),
        source_replay_status=np.array(result), parent_parameter_sha256=np.array(digest(replay.args.latest)),
        anatomical_basis=replay.joints.chart.basis,anatomical_rest_R=replay.joints.chart.rest,
        anatomical_reference_twist=replay.joints.chart.reference_twist,
        anatomical_policy_json=np.array(json.dumps(replay.joints.configuration,sort_keys=True)))
    candidate=p.checkpoint_fields(x)
    np.testing.assert_array_equal(candidate['absolute_position_controls'],latest['absolute_position_controls'])
    np.testing.assert_allclose(candidate['absolute_rotation_controls'][:,0],
                               latest['absolute_rotation_controls'][:,0],rtol=0,atol=2e-12)
    np.testing.assert_allclose(candidate['absolute_rotation_edge_reference_quaternion'][:,0],
                               latest['absolute_rotation_edge_reference_quaternion'][:,0],rtol=0,atol=2e-12)
    for k in ('sequence_contact_modes','sequence_contact_status'):
        np.testing.assert_array_equal(candidate[k],latest[k])
    save(path/'candidate.npz',**candidate)


def revised_directions(replay, events):
    """Fast source proposals on the already existing global spline controls."""
    from copy import deepcopy
    from scipy.optimize import least_squares
    from scipy.sparse import vstack
    from calibration.program_tracking.body_tracking.fullbody_fit import FullBodyProblem

    prior = replay.args.proposals_from
    accepted = json.loads((prior/'acceptance.json').read_text(encoding='utf-8'))
    if accepted['parent_sha256'] != digest(replay.args.latest):
        raise ValueError('Previously measured directions do not share the frozen starting state')
    seeds = np.load(prior/'proposal_directions.npz', allow_pickle=False)
    np.testing.assert_array_equal(seeds['x'], replay.x)
    p = replay.problem
    shifts, all_controls, details, fallback = {}, {}, {}, {}
    for event in events:
        name = event['id']
        indices, support = active_controls(replay, event)
        all_controls[name] = (indices, support)
        previous = seeds[name] if name in seeds else None
        if previous is not None:
            allowed = np.isin(previous[:,0].astype(int), indices)
            if not allowed.all():
                raise ValueError('Saved proposal left the event control support')
        if previous is not None and name in accepted['selected_events']:
            fraction = accepted['selected_events'][name]
            fallback[name] = (previous[:,0].astype(int), previous[:,1] * fraction)
            if name != 's05_stride':
                shifts[name] = fallback[name]
                details[name] = dict(method='already measured full production direction', support_seconds=support,
                                     columns=len(indices), fraction=fraction)
                continue

        if name in ('s02_stride','s09_ccosp'):
            trial_event=dict(event,active_feet=[1])
            indices,_=active_controls(replay,trial_event)
        relative = indices >= 17 + p.body.state.body_size
        axis = (indices-17) % 69
        ids = indices[~relative & ~np.isin(axis,np.r_[33:39])]
        if not len(ids):
            raise ValueError('No attached leg controls for ' + name)
        e = deepcopy(replay.ev)
        for key in ('manual_landmarks','ordered_contours','native_support_bands'):
            e[key] = [row for row in e[key] if row['event_id']==name]
        e['plane_xy_diagnostics'] = []
        source = ProgrammeSkateSourceFactors(p,e)
        source.ordered.associate(replay.x[17:])
        frames = np.flatnonzero((replay.clock >= support[0]-1e-9) & (replay.clock <= support[1]+1e-9))
        body = FullBodyProblem(replay.ref,replay.camera,replay.observations_path,
            tuple(replay.clock[[0,-1]]),'full',state_factory=lambda *unused:p.body.state,
            source_frames=frames,execution=replay.execution)
        maximum = .52 if name=='s05_stride' else .29
        baseline = replay.x.copy()
        block_names = ['body_pixels', *SOURCES]
        fit_gain = {}
        def local(z, jacobian):
            x=baseline.copy()
            x[ids]+=z
            body_data=body.evaluate(x[17:],jacobian)['weighted_blocks'][0]
            native=source.evaluate(x[17:],jacobian)['weighted_blocks']
            residual=[body_data.residual*.10]
            jac=[body_data.jacobian[:,ids-17]*.10] if jacobian else []
            for block in native:
                scale=np.sqrt(float(p.objective_block_weights[block.name]))*1.20
                residual.append(block.residual*scale)
                if jacobian:
                    jac.append(block.jacobian[:,ids]*scale)
            residual.append(.25/maximum*z)
            if jacobian:
                jac.append(diags(np.full(len(z),.25/maximum)))
                return vstack(jac,format='csr')
            fit_gain.update({key:float(part@part/2.) for key,part in zip(block_names,residual)})
            return np.concatenate(residual)
        zero=np.zeros(len(ids))
        initial=local(zero,False)
        before_parts=fit_gain.copy()
        proposal=least_squares(lambda z:local(z,False),zero,jac=lambda z:local(z,True),
            bounds=(-maximum,maximum),method='trf',tr_solver='lsmr',max_nfev=9)
        final=local(proposal.x,False)
        source_gain=sum(before_parts.get(k,0)-fit_gain.get(k,0) for k in SOURCES)
        if final@final < initial@initial and source_gain>0:
            shifts[name]=(ids,proposal.x)
        details[name]=dict(method='native body and semantic shoe Jacobian on local support',
            source_before=before_parts,source_after=fit_gain.copy(),source_gain=source_gain,
            support_seconds=support,active_controls=ids.tolist(),local_evaluations=proposal.nfev,
            uses_corrected_detector_correspondence=True)
        replay.emit('local_proposal',event_id=name,columns=len(ids),frames=len(frames),
                    source_gain=source_gain,local_evaluations=proposal.nfev)
    seeds.close()
    return shifts,all_controls,details,fallback


def run(args,log):
    r=SkateReplay(args,log)
    a=json.loads(args.native_review.read_text(encoding='utf-8'))
    events=a['events']
    with r.execution.limits():
        write(args.output/'objective_weights.json',r.weights_record())
        fallback={}
        cached_geometry=None
        if args.replay_saved_step:
            if args.proposals_from is None:
                raise ValueError('Saved step replay requires its immutable prior receipt')
            prior=args.proposals_from
            cached_geometry=json.loads((prior/'acceptance.json').read_text(encoding='utf-8'))
            if (not cached_geometry['engineering_step_accepted']
                    or cached_geometry['parent_sha256']!=digest(args.latest)):
                raise ValueError('Expected a complete previously checked step from this same parent')
            controls={e['id']:active_controls(r,e) for e in events}
            shifts={}
            with np.load(prior/'proposal_directions.npz',allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive['x'],r.x)
                for name,scale in cached_geometry['selected_events'].items():
                    move=archive[name]
                    ids=move[:,0].astype(int)
                    if not np.isin(ids,controls[name][0]).all():
                        raise ValueError('Saved step changes undeclared controls')
                    shifts[name]=(ids,scale*move[:,1])
            details=dict(method='replay the exact saved complete step with reviewed detector assignment',
                         original_receipt_sha256=digest(prior/'acceptance.json'),events=list(shifts))
        elif args.proposals_from is None:
            base=r.full(r.x,'new_source_parent',jacobian=True)
            shifts,controls,details=directions(r,base,events)
            del base
        else:
            shifts,controls,details,fallback=revised_directions(r,events)
        write(args.output/'directions.json',details)
        if not shifts:
            raise RuntimeError('No source direction also descends on full objective')
        init={name:1. for name in shifts}
        move=np.zeros_like(r.x)
        for name,(indices,update) in shifts.items():
            move[indices]=update
        derivatives=source_derivative(r,move)
        if max(v['relative_error'] for v in derivatives.values())>1e-4:
            raise RuntimeError('Source Jacobian finite-difference mismatch')
        write(args.output/'native_source_jacobian_check.json',derivatives)
        save(args.output/'proposal_directions.npz',x=r.x,move=move,**{
            name:np.c_[ids,step] for name,(ids,step) in shifts.items()})
        r.problem.cache=r.problem.cache_x=None
        gc.collect()
        fractions=init.copy()
        trials=[]
        while fractions:
            x=r.x.copy()
            for name,scale in fractions.items():
                ids,delta=shifts[name]
                x[ids]+=scale*delta
            version=f'actual_{len(trials)+1:02d}'
            data,base,trial=replay_trial(r,x,version,controls,events,cached_geometry)
            trials.append(dict(version=version,fractions=fractions.copy(),event_failures=data['event_failures'],
                               global_failures=data['global_failures']))
            bad={name for name in fractions if data['event_failures'][name]}
            if data['global_failures']:
                bad.update(fractions)
            if not bad or cached_geometry is not None:
                break
            if len(trials)==2:
                for name in bad:
                    if name=='s05_stride' and name in fallback:
                        shifts[name]=fallback[name]
                        fractions[name]=1.
                    else:
                        del fractions[name]
            elif len(trials)>=3:
                break
            else:
                for name in bad:
                    fractions[name]*=.5
            del base,trial
            r.problem.cache=r.problem.cache_x=None
            gc.collect()
        write(args.output/'full_trials.json',trials)
        if not fractions:
            r.emit('no_step_accepted', reason='No event passed the full replay; prior candidate retained')
            return
        accepted=not data['global_failures'] and not any(data['event_failures'][name] for name in fractions)
        check=ProgrammeSkateSourceFactors(r.problem,r.ev,role='check')
        check.ordered.associate(r.x[17:])
        rows={}
        for label,z,v in [('parent',r.x,base),('candidate',x,trial)]:
            rows[label]={'fit':{k:v['value']['action_metrics'][k] for k in SOURCES},
                         'check':check.evaluate(z[17:],False)['metrics']}
        write(args.output/'source_comparison.json',rows)
        check_failures=[]
        for key in SOURCES:
            for before,after in zip(rows['parent']['check'][key],rows['candidate']['check'][key],strict=True):
                assert before['id']==after['id']
                u=before.get('distance_px',before.get('outside_candidate_band_px'))
                v=after.get('distance_px',after.get('outside_candidate_band_px'))
                allowance=after.get('radius_px',after.get('source_radius_px'))
                if u is not None and v is not None and allowance is not None:
                    pp,qq=float(np.percentile(u,95)),float(np.percentile(v,95))
                    if qq > max(pp,allowance)+.5*allowance:
                        check_failures.append(dict(id=after['id'],event=after['event_id'],pts=after['pts'],
                                                   before_px=pp,after_px=qq,radius_px=allowance))
        quality='full_replay_engineering_step' if accepted and not check_failures else 'review_only'
        checkpoint(r,x,quality)
        write(args.output/'acceptance.json',dict(schema='rinkspace.programme-footwork-replay.v1',
            acceptance=quality,engineering_step_accepted=accepted and not check_failures,user_accepted=False,
            accuracy_accepted=False,coverage=dict(trial['coverage'],surface_checked_samples=data['coverage']['checked_samples']),
            same_camera_sha256=r.info['state_sha256'],source_evidence=str(r.source_path),
            source_evidence_sha256=digest(r.source_path),parent_sha256=digest(args.latest),
            parameter_sha256=digest(args.output/'candidate.npz'),candidate_surface=str(args.output/('surface_'+version+'.npz')),
            candidate_surface_sha256=digest(args.output/('surface_'+version+'.npz')),surface_comparison=str(args.output/('surface_'+version+'.json')),
            programme_clock_sha256=signature(r.pts,np.asarray(r.tb)),selected_events=fractions,
            excluded_events=sorted(set(controls)-set(fractions)),root_P_and_lifted_Q_preserved=True,
            same_version_weighted=dict(parent=serial(base),candidate=serial(trial)),
            trial=data,check_failures=check_failures))
        write(args.output/'production_registry.json',trial['registry'])
        r.emit('candidate_saved',status=quality,events=list(fractions),main_log=str(args.output/'progress.jsonl'))


def main():
    p=argparse.ArgumentParser()
    for k in ('project','prepared','source','latest','latest-surface','native-review','output'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--reviewed-observations',type=Path)
    p.add_argument('--proposals-from',type=Path)
    p.add_argument('--replay-saved-step',action='store_true')
    a=p.parse_args()
    a.previous=a.latest
    a.previous_surface=a.latest_surface
    a.output.mkdir()
    write(a.output/'command.json',dict(command=sys.argv))
    with (a.output/'progress.jsonl').open('x',encoding='utf-8',buffering=1) as log:
        run(a,log)


if __name__=='__main__':
    main()
