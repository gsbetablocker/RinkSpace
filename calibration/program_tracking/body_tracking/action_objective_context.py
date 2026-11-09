"""Persist the fixed, warm-dependent part of one programme action objective.

An exact resume restores this context. A revised-objective warm start creates
another context. Legacy migration reads only the immediate stage's receipts;
it never replays preparation or recursively searches earlier stages.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from .weighted_blocks import signature


SCHEMA = 'rinkspace.action-objective-context.v1'
CONTEXT = 'action_objective_context_json'
ANCHOR = 'action_objective_root_anchor_solver'
ROOT_PTS = 'action_objective_root_source_pts'
SOURCE_PTS = 'action_objective_source_pts'


def resume_source_choices(choices, saved):
    """An identical resume option must use the saved branch and its masks."""
    result = dict(choices)
    if saved is None:
        return result
    for name, choice in result.items():
        value = np.asarray(saved.get('source_association_extra_'+name+'_choice'))
        if (type(choice) is not int or value.shape != () or value.dtype.kind not in 'iu'
                or int(value) != choice):
            raise ValueError('Exact resume cannot override a missing or different '
                             'source episode choice: '+name)
    return {}


def restore_source_masks(factor, saved, prefix):
    """Restore fixed correspondences before any geometry-dependent inference."""
    if factor is None:
        return
    policy = factor.association_policy
    if hasattr(factor, 'fixed_choice'):
        key = prefix+'fixed_choice'
        if key not in saved:
            if policy.get('freeze_inner') and factor.rows:
                raise ValueError('Exact resume requires saved frozen source mask: '+key)
        else:
            value = np.asarray(saved[key])
            if (value.shape != (len(factor.source_uv),) or value.dtype.kind not in 'iu'
                    or np.any(value < 0) or np.any(value >= len(factor.starts))):
                raise ValueError('Invalid saved source correspondence: '+key)
            if factor.segment_geometry is not None:
                for i,row in enumerate(factor.rows):
                    allowed = factor.segment_geometry['feature_segments'][row['structure_feature']]
                    if not np.isin(value[factor.point_rows == i],allowed).all():
                        raise ValueError('Saved correspondence changed source feature: '+key)
            factor.fixed_choice = value.copy()
    if hasattr(factor, 'restore_visible_mask'):
        key = prefix+'visible_mask'
        if key in saved:
            factor.restore_visible_mask(saved[key])
        elif policy.get('model_visibility') == 'near_visible_runner_branch' and factor.rows:
            raise ValueError('Exact resume requires saved runner visibility: '+key)


def restore_source_episode(factor, saved, prefix, *, choice_key=None):
    key = choice_key or prefix+'choice'
    value = np.asarray(saved.get(key))
    if (value.shape != () or value.dtype.kind not in 'iu'
            or not 0 <= int(value) < len(factor.episode_factors)):
        raise ValueError('Exact resume requires a valid saved source episode: '+key)
    choice = int(value)
    costs = np.asarray(saved.get(prefix+'candidate_costs',[]))
    if costs.shape not in ((0,), (len(factor.episode_factors),)) or not np.isfinite(costs).all() or np.any(costs < 0):
        raise ValueError('Invalid saved source candidate scores: '+prefix)
    restore_source_masks(factor.episode_factors[choice], saved, prefix)
    factor.episode_choice = choice
    # Older stages did not record these diagnostic scores. An empty list
    # records that absence without inventing scores at the resumed pose.
    factor.episode_costs = costs.tolist()


def validate_resume_selection(problem, stage):
    path = Path(stage)/'solve/parameter_selection.json'
    if not path.is_file():
        raise FileNotFoundError('Exact resume requires the complete active selection: '+str(path))
    previous = json.loads(path.read_text(encoding='utf-8'))
    current = np.asarray(getattr(problem,'active_parameter_indices',np.arange(17,problem.size)))
    old = np.asarray(previous.get('selection'))
    if (previous.get('full_layout') != problem.size or current.dtype.kind not in 'iu'
            or old.dtype.kind not in 'iu' or not np.array_equal(current,old)):
        raise ValueError('Exact resume changed the complete active parameter selection; use a new warm start')


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _array_hash(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(a.dtype.str.encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _text(packet, key):
    if key not in packet or np.asarray(packet[key]).shape != ():
        raise ValueError('Action objective checkpoint requires scalar '+key)
    return str(np.asarray(packet[key]))


def _json_text(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _layout(problem, root):
    state = problem.body.state
    ref = problem.reference
    arrays = dict(source_pts=ref.p['pts'], source_time_base=ref.p['time_base'],
        native_times=ref.times, root_indices=root.indices, root_times=root.times,
        body_knots=state.curve.knots, body_seed_positions=state.curve.position_controls,
        body_seed_rotations=state.curve.rotation_controls,
        shoe_knots=state.shoe_curve.knots,
        shoe_seed_rotations=state.shoe_curve.rotation_controls)
    result = {key+'_sha256': _array_hash(value) for key, value in arrays.items()}
    for name, curve in (('body', state.curve), ('shoe', state.shoe_curve)):
        chart = curve.edge_reference_quaternion
        result[name+'_edge_chart_sha256'] = None if chart is None else _array_hash(chart)
    result.update(state_size=int(state.size), body_size=int(state.body_size),
                  shoe_size=int(state.shoe_size))
    return result


def _scores(value, *, required):
    if value is None and not required:
        return None
    if not isinstance(value, dict) or {str(k) for k in value} != {'-1', '1'}:
        raise ValueError('Ambiguous turn context requires both original signed branch costs')
    result = {int(k): float(v) for k, v in value.items()}
    if not all(np.isfinite(v) and v >= 0. for v in result.values()):
        raise ValueError('Turn branch costs must be finite and nonnegative')
    return result


def _turns(primary):
    result = {}
    for factor in primary.turn_factors:
        name = factor.episode['episode_id']
        if name in result:
            raise ValueError('Duplicate action turn context: '+name)
        result[name] = dict(declared_sign=int(factor.episode['turn_direction_sign']),
            selected_sign=int(factor.sign), sign_provenance=factor.sign_provenance,
            sign_scores=factor.sign_scores,
            source_pts_sha256=_array_hash(factor.ref.p['pts'][factor.rows]))
    return result


def _validated_turns(primary, records):
    expected = _turns(primary)
    if not isinstance(records, dict) or set(records) != set(expected):
        raise ValueError('Resume changed action turn episode inventory')
    result = []
    for factor in primary.turn_factors:
        name = factor.episode['episode_id']
        row = records[name]
        for key in ('declared_sign', 'source_pts_sha256'):
            if row.get(key) != expected[name][key]:
                raise ValueError('Resume changed turn '+key+': '+name)
        sign = row.get('selected_sign')
        if isinstance(sign, bool) or not isinstance(sign, (int, np.integer)) or sign not in (-1, 1):
            raise ValueError('Invalid saved turn sign: '+name)
        declared = expected[name]['declared_sign']
        if declared and sign != declared:
            raise ValueError('Saved turn contradicts its explicit source sign: '+name)
        provenance = row.get('sign_provenance')
        if not isinstance(provenance, str) or not provenance:
            raise ValueError('Missing saved turn provenance: '+name)
        costs = _scores(row.get('sign_scores'), required=declared == 0)
        result.append((factor, int(sign), provenance, costs))
    return result


def _modes(factor, saved=None):
    if saved is None:
        modes = np.asarray(factor.modes).copy()
        status = np.asarray(factor.status).copy()
        record = json.loads(_json_text(factor.record))
    else:
        required = ('sequence_contact_modes', 'sequence_contact_status',
                    'sequence_contact_mode_names', 'sequence_contact_association_json')
        missing = [key for key in required if key not in saved]
        if missing:
            raise ValueError('Exact action resume requires original contact context: '+', '.join(missing))
        modes = np.asarray(saved['sequence_contact_modes']).copy()
        status = np.asarray(saved['sequence_contact_status']).copy()
        record = json.loads(_text(saved, 'sequence_contact_association_json'))
        if not np.array_equal(saved['sequence_contact_mode_names'], factor.record['mode_names']):
            raise ValueError('Resume changed contact mode vocabulary')
    if (modes.shape != (len(factor.times), 2) or modes.dtype.kind not in 'iu'
            or np.any(modes >= len(factor.record['mode_names'])) or np.any(modes < 0)
            or status.shape != modes.shape or status.dtype.kind not in 'US'):
        raise ValueError('Invalid saved temporal contact mode/status layout')
    if record.get('mode_names') != factor.record['mode_names']:
        raise ValueError('Resume changed temporal contact record vocabulary')
    record_hash = hashlib.sha256(_json_text(record).encode()).hexdigest()
    return modes, status, record, signature(modes, status), record_hash


def _legacy_root(primary, root, problem, saved, stage):
    if stage is None:
        raise ValueError('Legacy action resume needs its immediate original stage directory')
    stage = Path(stage)
    receipts = [stage/name for name in ('warm_start.json', 'resume.json') if (stage/name).is_file()]
    if len(receipts) != 1:
        raise ValueError('Legacy action context migration requires exactly one immediate '
            'warm_start.json or resume.json with the SHA-bound starting parameter; '
            'a missing receipt does not prove a zero root anchor')
    receipt = json.loads(receipts[0].read_text(encoding='utf-8'))
    path_text, expected = receipt.get('parameter'), receipt.get('parameter_sha256')
    if not isinstance(path_text, str) or not path_text or not isinstance(expected, str):
        raise ValueError('Legacy start receipt needs parameter and parameter_sha256')
    parameter = Path(path_text)
    if not parameter.is_file():
        raise FileNotFoundError('Legacy action anchor needs the exact directly recorded parameter: '+str(parameter))
    if _file_hash(parameter) != expected:
        raise ValueError('Legacy action anchor parameter SHA256 mismatch: '+str(parameter))
    with np.load(parameter, allow_pickle=False) as archive:
        packet = {key: archive[key].copy() for key in archive.files}
    state = problem.body.state
    identity = dict(absolute_knots=state.curve.knots,
        absolute_seed_position_controls=state.curve.position_controls,
        absolute_seed_rotation_controls=state.curve.rotation_controls,
        relative_skate_knots=state.shoe_curve.knots,
        relative_skate_seed_rotation_controls=state.shoe_curve.rotation_controls)
    for key, value in identity.items():
        if key not in packet or not np.array_equal(packet[key], value):
            raise ValueError('Legacy starting parameter changed control coordinates: '+key)
    for key in ('prepared_input_sha256', 'prepared_observations_sha256'):
        if _text(packet, key) != _text(saved, key):
            raise ValueError('Legacy starting parameter changed '+key)
    x = np.asarray(packet.get('x'))
    if x.shape != (state.size+17,) or not np.isfinite(x).all():
        raise ValueError('Legacy starting parameter has an invalid global x layout')
    root_anchor = root.position_controls(x[17:]).copy()
    records = _turns(primary)
    ambiguous = [f for f in primary.turn_factors if int(f.episode['turn_direction_sign']) == 0]
    provenance = dict(kind='immediate_legacy_stage', receipt=str(receipts[0]),
        receipt_sha256=_file_hash(receipts[0]), parameter=str(parameter), parameter_sha256=expected,
        interpretation='the fixed objective actually used by this immediate stage; no recursive historical replay')
    if ambiguous:
        path = stage/'initial_action_factors.json'
        if not path.is_file():
            raise FileNotFoundError('Legacy action resume requires original ambiguous-turn records: '+str(path))
        metrics = json.loads(path.read_text(encoding='utf-8')).get('source_and_physics', {})
        for factor in ambiguous:
            name = factor.episode['episode_id']
            row = metrics.get('episode/'+name+'/metrics')
            if not isinstance(row, dict) or not np.array_equal(row.get('path_source_pts'),
                                                              factor.ref.p['pts'][factor.rows]):
                raise ValueError('Legacy turn record lacks its exact source samples: '+name)
            records[name].update(selected_sign=row.get('selected_direction_sign'),
                sign_scores=row.get('warm_signed_branch_costs'),
                sign_provenance=row.get('direction_sign_provenance'))
        provenance.update(initial_action_factors=str(path), initial_action_factors_sha256=_file_hash(path))
    return root_anchor, records, provenance


def bind_action_objective_context(problem, primary, source_evidence, *, saved=None, legacy_stage=None):
    """Bind before the first factor evaluation/weight declaration of this run.

    ``saved=None`` means a new objective, including explicit warm starts.
    Restoring context never substitutes the resumed x for a missing anchor.
    """
    from .programme_action_factors import ProgrammeRootTrust
    roots = [factor for factor in primary.factors if isinstance(factor, ProgrammeRootTrust)]
    if len(roots) != 1:
        raise ValueError('Action context expects exactly one ProgrammeRootTrust factor')
    root = roots[0]
    modes = problem.action_factors.modes
    source_hash = _file_hash(source_evidence)
    layout = _layout(problem, root)
    source_pts = np.asarray(problem.reference.p['pts'])
    root_pts = source_pts[root.indices]
    mode_state = _modes(modes, saved)
    if saved is None:
        anchor, turns = root.parent.copy(), _turns(primary)
        provenance = dict(kind='new_objective', reason='fresh start or explicit revised-objective warm start')
        restoration = 'new_context'
    else:
        if _text(saved, 'source_evidence_sha256') != source_hash:
            raise ValueError('Exact action resume changed source_evidence SHA256; use an explicit warm start for a new objective')
        if CONTEXT in saved:
            context = json.loads(_text(saved, CONTEXT))
            if context.get('schema') != SCHEMA or context.get('source_evidence_sha256') != source_hash:
                raise ValueError('Action objective context schema/source identity mismatch')
            if context.get('layout') != layout:
                raise ValueError('Action objective context changed source clock, seed or control layout')
            for key, expected in ((SOURCE_PTS, source_pts), (ROOT_PTS, root_pts)):
                if key not in saved or not np.array_equal(saved[key], expected):
                    raise ValueError('Action objective context changed '+key)
            if ANCHOR not in saved:
                raise ValueError('Action objective checkpoint lacks its fixed root anchor')
            anchor = np.asarray(saved[ANCHOR]).copy()
            if context.get('root_anchor_sha256') != _array_hash(anchor):
                raise ValueError('Action objective root anchor hash mismatch')
            if (context.get('contact_association_signature') != mode_state[3]
                    or context.get('contact_record_sha256') != mode_state[4]):
                raise ValueError('Action objective checkpoint contact context mismatch')
            turns, provenance = context.get('turns'), context.get('origin')
            restoration = 'checkpoint_context'
        else:
            anchor, turns, provenance = _legacy_root(primary, root, problem, saved, legacy_stage)
            restoration = 'legacy_migration'
    if anchor.shape != root.parent.shape or not np.isfinite(anchor).all():
        raise ValueError('Action objective root anchor has an invalid sampled layout')
    restored_turns = _validated_turns(primary, turns)
    # Commit only after every identity and fixed branch has been checked.
    root.parent = anchor.copy()
    for factor, sign, reason, costs in restored_turns:
        factor.sign, factor.sign_provenance, factor.sign_scores = sign, reason, costs
    modes.modes, modes.status, modes.record = mode_state[:3]
    modes.association_signature = mode_state[3]
    context = dict(schema=SCHEMA, source_evidence_sha256=source_hash, layout=layout,
        root_anchor_sha256=_array_hash(anchor), turns=_turns(primary), origin=provenance,
        contact_association_signature=mode_state[3], contact_record_sha256=mode_state[4])
    problem.checkpoint_extra.update({CONTEXT: np.array(_json_text(context)), ANCHOR: anchor.copy(),
        ROOT_PTS: root_pts.copy(), SOURCE_PTS: source_pts.copy(),
        'source_evidence_sha256': np.array(source_hash)})
    return dict(schema=SCHEMA, restoration=restoration, root_samples=len(anchor),
        native_samples=len(source_pts), fixed_turns=len(restored_turns),
        contact_association_signature=mode_state[3], source_evidence_sha256=source_hash)
