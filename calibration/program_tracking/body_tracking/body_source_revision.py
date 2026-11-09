"""Explicit source-reviewed body identity/admission revisions; no new pixels."""
import copy,json
import numpy as np
from .fullbody_fit import body_joint_assignment
from .trajectory_probe import GROUPS


def revise_body_observations(observations,assignments,exclusions,provenance):
    """Keep every measured UV/sigma/confidence and all prior identity repairs.

    assignments contain exact source_pts and bilateral joint_pairs. Exclusions
    contain exact source_pts, joint_id and a source-image reason; only the
    derived use mask changes. The original parent packet remains authoritative.
    """
    old=observations;out={k:np.array(v,copy=True)for k,v in old.items()}
    pts=np.asarray(old['pts']);ids=np.array(body_joint_assignment(old),copy=True)
    initial_ids=ids.copy();initial_use=np.asarray(old['use']).copy()
    if initial_use.shape!=ids.shape or initial_use.dtype.kind!='b':raise ValueError('Boolean source use mask required')
    if not assignments and not exclusions:raise ValueError('An explicit source revision is required')
    def frame(stamp):
        if isinstance(stamp,bool)or not isinstance(stamp,(int,np.integer)):raise ValueError('Exact integer PTS required')
        index=int(np.searchsorted(pts,stamp))
        if index==len(pts)or pts[index]!=stamp:raise ValueError('Revision outside native exposures')
        return index
    seen=set();changes=[]
    for entry in assignments:
        i=frame(entry['source_pts']);pairs=entry['joint_pairs']
        if not pairs:raise ValueError('Explicit bilateral pairs required')
        for pair in pairs:
            if len(pair)!=2:raise ValueError('Bilateral joint pair required')
            a,b=pair
            if any(isinstance(q,bool)or not isinstance(q,(int,np.integer))for q in pair)or a not in range(1,17,2)or b!=a+1:
                raise ValueError('Revision must preserve the same bilateral joint kind')
            key=(i,a)
            if key in seen or not np.array_equal(ids[i,[a,b]],[a,b]):raise ValueError('Duplicate or already revised correspondence')
            seen.add(key);ids[i,[a,b]]=[b,a]
            changes.append(dict(source_pts=int(pts[i]),joint_pair=[int(a),int(b)],before=[int(a),int(b)],after=[int(b),int(a)]))
    removed=[];used=set()
    for entry in exclusions:
        i=frame(entry['source_pts']);j=entry['joint_id']
        if isinstance(j,bool)or not isinstance(j,(int,np.integer))or not 0<=j<17:raise ValueError('Invalid source joint id')
        if not entry.get('reason'):raise ValueError('Source-image exclusion reason required')
        if (i,j)in used or not out['use'][i,j]:raise ValueError('Duplicate or inactive source exclusion')
        used.add((i,j));out['use'][i,j]=False
        removed.append(dict(entry,source_pts=int(pts[i]),joint_id=int(j),raw_uv_retained=True))
    identity=np.broadcast_to(np.arange(17),ids.shape)
    prior=None
    if 'body_joint_assignment_provenance_json'in old:prior=json.loads(str(old['body_joint_assignment_provenance_json']))
    record=dict(provenance,kind='source_observation_correspondence',source_measurements_unchanged=True,
        source_pts=pts[np.any(ids!=identity,axis=1)].tolist(),previous_correspondence=prior,
        new_assignments=changes,derived_use_exclusions=removed,physical_model_identity_changed=False,
        raw_UV_sigma_confidence_unchanged=True,network_rerun=False)
    out['model_joint_ids']=ids
    out['body_joint_assignment_provenance_json']=np.array(json.dumps(record,sort_keys=True))
    if removed:
        out['body_admission_parent_use']=initial_use
        out['body_admission_revision_json']=np.array(json.dumps(dict(kind='source_reviewed_admission',
            excluded=removed,provenance=provenance,parent_admission=old.get('body_admission_revision_json',np.array('')).item()),sort_keys=True))
    body_joint_assignment(out)
    for key,value in old.items():
        if key not in ('model_joint_ids','body_joint_assignment_provenance_json','use','body_admission_parent_use','body_admission_revision_json'):
            np.testing.assert_array_equal(out[key],value,err_msg='Source array changed: '+key)
    changed_use=np.argwhere(initial_use!=out['use'])
    if {tuple(v)for v in changed_use}!={(frame(r['source_pts']),r['joint_id'])for r in removed}:
        raise AssertionError('Unrecorded source admission change')
    record['changed_correspondence_entries']=int(np.count_nonzero(ids!=initial_ids))
    record['changed_use_entries']=len(changed_use)
    return out,record


def revised_quality_population(baseline,old_observations,new_observations,new_reference,revision):
    """Change only reviewed non-guard rows; all original guard limits survive.

    The existing primary guards cover torso/lower-limb points. Excluding a
    member of those groups requires a separate explicit baseline reassessment;
    this helper refuses that case rather than silently reducing its population.
    """
    for key in ('pts','uv','sigma_px'):
        np.testing.assert_array_equal(old_observations[key],new_observations[key],err_msg='Source changed: '+key)
    old_use=np.asarray(old_observations['use']);new_use=np.asarray(new_observations['use'])
    if old_use.shape!=new_use.shape or np.any(new_use&~old_use):raise ValueError('Unexpected source admission addition')
    rows,joints=np.nonzero(old_use&~new_use)
    protected=np.array([*GROUPS['torso'],*GROUPS['lower_limbs']])
    if np.any(np.isin(joints,protected)):raise ValueError('Cannot change the protected quality population')
    declared={(int(r['source_pts']),int(r['joint_id']))for r in revision['derived_use_exclusions']}
    actual={(int(new_observations['pts'][i]),int(j))for i,j in zip(rows,joints)}
    if actual!=declared:raise ValueError('Quality population excludes undeclared evidence')
    result=copy.deepcopy(baseline)
    previous=result.get('body_observation_reference')
    result['body_observation_reference']=dict(new_reference)
    result['body_source_revision']=dict(previous_quality_population=previous,source_revision=revision,
        all_existing_source_torso_turn_limits_unchanged=True,protected_torso_lower_population_identical=True,
        excluded_rows=len(actual),reason='Original source images show duplicate/wrong-target detections; no model residual criterion')
    for key in ('source','torso','turn_episodes'):
        if key in baseline:assert result[key]==baseline[key]
    return result


def reassess_lower_limb_quality_population(baseline,old_observations,new_observations,new_reference,revision,
                                          *,reference_projection,reference_pts,turn_source_pts,reference_identity):
    """Explicit source-visibility reassessment, retaining every numeric limit.

    Replays one immutable saved pose on both populations. This is an admission
    change, not a model improvement or a reset to the current candidate's errors.
    The original non-guard helper continues to reject protected-row changes.
    """
    for key in ('pts','uv','sigma_px','confidence'):
        if key in old_observations:
            np.testing.assert_array_equal(old_observations[key],new_observations[key],err_msg='Source changed: '+key)
    old_ids=body_joint_assignment(old_observations);new_ids=body_joint_assignment(new_observations)
    np.testing.assert_array_equal(old_ids,new_ids,err_msg='Visibility reassessment cannot change identities')
    pts=np.asarray(old_observations['pts']);np.testing.assert_array_equal(reference_pts,pts)
    old_use=np.asarray(old_observations['use']);new_use=np.asarray(new_observations['use'])
    if old_use.dtype.kind!='b' or new_use.dtype.kind!='b' or old_use.shape!=new_use.shape or np.any(new_use&~old_use):
        raise ValueError('Unexpected source admission change')
    rows,joints=np.nonzero(old_use&~new_use)
    if not len(rows) or np.any(~np.isin(joints,GROUPS['lower_limbs'])):
        raise ValueError('Explicit lower-limb-only visibility reassessment required')
    declared=revision['derived_use_exclusions']
    expected={(int(r['source_pts']),int(r['joint_id'])) for r in declared}
    actual={(int(pts[i]),int(j)) for i,j in zip(rows,joints)}
    if actual!=expected or len(declared)!=len(actual) or any(not r.get('reason') for r in declared):
        raise ValueError('Visibility reassessment excludes undeclared evidence')
    if not reference_identity.get('path') or not reference_identity.get('sha256'):
        raise ValueError('Immutable reference projection identity required')
    projection=np.asarray(reference_projection,float)
    if projection.shape!=old_observations['uv'].shape:
        raise ValueError('Reference projection must contain every native body point')
    assigned=projection[np.arange(len(pts))[:,None],old_ids]
    pixels=np.linalg.norm(assigned-old_observations['uv'],axis=-1)
    protected=np.zeros_like(old_use);protected[:,[*GROUPS['torso'],*GROUPS['lower_limbs']]]=True
    if not np.isfinite(pixels[old_use&protected]).all():
        raise ValueError('Nonfinite protected reference projection')
    def metrics(mask):
        values=pixels[mask]
        if not len(values):raise ValueError('Empty reassessed body quality population')
        return dict(observations=int(len(values)),pixel_p95=float(np.percentile(values,95)))
    torso=np.zeros_like(old_use);torso[:,GROUPS['torso']]=True
    np.testing.assert_array_equal(old_use&torso,new_use&torso)
    torso_replay=metrics(old_use&torso)
    expected_episodes={name.rsplit('/',1)[0] for name in baseline['turn_episodes']}
    if set(turn_source_pts)!=expected_episodes:raise ValueError('Turn episode population differs')
    result=copy.deepcopy(baseline);replay={}
    for name,previous in baseline['turn_episodes'].items():
        episode,group=name.rsplit('/',1)
        selected=np.asarray(turn_source_pts[episode])
        if selected.dtype.kind not in 'iu' or len(np.unique(selected))!=len(selected) or not np.isin(selected,pts).all():
            raise ValueError('Turn rows must be distinct native source PTS')
        mask=np.zeros_like(old_use);mask[:,GROUPS[group]]=np.isin(pts,selected)[:,None]
        before=metrics(old_use&mask);after=metrics(new_use&mask)
        if before['observations']!=previous['observations']:
            raise ValueError('Original turn population count differs: '+name)
        result['turn_episodes'][name]['observations']=after['observations']
        replay[name]=dict(same_saved_pose_before=before,same_saved_pose_after=after,
                         retained_baseline_p95_px=previous['pixel_p95'],
                         retained_limit_px=previous['pixel_p95']+max(2.,.1*previous['pixel_p95']))
    result['body_observation_reference']=dict(new_reference)
    result['body_source_revision']=dict(previous_quality_population=baseline.get('body_observation_reference'),
        previous_revision=baseline.get('body_source_revision'),source_revision=revision,
        all_existing_source_torso_turn_numeric_limits_unchanged=True,protected_population_explicitly_reassessed=True,
        excluded_rows=len(actual),reference_projection=dict(reference_identity),same_pose_replay=replay,
        same_pose_torso_replay=torso_replay,original_torso_guard_metadata=baseline['torso'],
        reference_pose_is_not_original_threshold_replay=True,
        accuracy_improvement_claim=False,stage_baseline_reset=False,
        reason='Source-only occlusion evidence; unchanged original numeric limits and exact surviving-row counts')
    for key in ('source','torso','flight_guard','stage_reset_allowed'):
        if key in baseline:assert result[key]==baseline[key]
    assert all(result['turn_episodes'][key]['pixel_p95']==value['pixel_p95'] for key,value in baseline['turn_episodes'].items())
    return result
