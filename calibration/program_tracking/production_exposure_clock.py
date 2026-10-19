"""Evidence-bound collision-only clock derivation before native registration.

Saved phases remain immutable parent evidence.  A new programme object and a
separate effective phase carry the new spline times into every factor builder.
No pose key, source pixel, knot, state or prior target is changed here.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

import low_end_confirmed_exposure_aliases as evidence
from continuous_exposure_clock import (CLOCK_POLICIES, LEGACY_POLICY,
    COLLISION_SAFE_POLICY, candidate_clock_for_saved_phase)
from continuous_rig_rekey import _clone_model


def json_sha(value):
    def plain(item):
        if isinstance(item, np.ndarray):
            return dict(dtype=str(item.dtype), shape=list(item.shape),
                sha256=hashlib.sha256(np.ascontiguousarray(item).tobytes()).hexdigest())
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False, default=plain).encode('utf8')).hexdigest()


def strict_windows(model, pairs):
    """Exactly the canonical R/logf triples selected by base.regularize."""
    result=[]
    for shot in model.shot_splines:
        stamps=sorted({int(b.frame_key) for b in model.frames.values() if b.shot_id == shot})
        for i in range(1,len(stamps)-1):
            a,b,c=stamps[i-1:i+2]
            if (a,b) in pairs and (b,c) in pairs:
                result.append((a,b,c))
    return result


def _complete_plan(model, clock, metadata, pairs):
    # Use the real completion planner with a recorder, never construct a graph.
    from continuous_full_timeline_regularization import add_complete_motion_priors
    class Recorder:
        def add_acceleration_factor(self,*args,**kwargs):
            pass
    report=add_complete_motion_priors(None,model,clock,metadata,pairs,
                                     affine_priors=Recorder())
    report['factor_backend']='production registered affine priors; planned before construction'
    return report


def derive_saved_pose_clock(context, phase, model, *, policy=LEGACY_POLICY,
                            parent_phase_sha256, frozen_source_shots_path=None):
    """Return effective context/phase/model/binding without creating factors.

    Call only after the original reconstruction/capture gates passed.  Legacy
    returns the original three objects, preserving historical replay exactly.
    The evidence loader rechecks the actual frozen mapping and manifest bytes.
    """
    if policy not in CLOCK_POLICIES:
        raise ValueError('Unknown production exposure-clock policy')
    if policy == LEGACY_POLICY:
        return context,phase,model,None
    if not parent_phase_sha256 or len(parent_phase_sha256) != 64:
        raise ValueError('New clock requires the immutable parent phase SHA256')
    data=context['data']
    groups=json.loads((evidence.CADENCE/'exposure_groups.json').read_text(encoding='utf8'))['groups']
    wanted=set(model.frames)
    aliases={int(p):int(g['full_pts'][0]) for g in groups for p in g['full_pts'] if int(p) in wanted}
    if set(aliases) != wanted or set(data['pts']) != wanted:
        raise ValueError('Frozen official groups do not exactly cover this programme model')
    clock,metadata,pairs=candidate_clock_for_saved_phase(data,aliases,phase,
        frozen_source_shots_path=Path(frozen_source_shots_path or evidence.SOURCE_SHOTS))
    saved={int(p):float(t) for p,t in phase['frame_clock_virtual_pts'].items()}
    old_pairs={tuple(map(int,row)) for row in phase['valid_temporal_acceleration_pairs']}
    for p,b in model.frames.items():
        if b.spline_pts != saved[p]:
            raise ValueError('Parent model spline time disagrees with its saved phase at '+str(p))
    candidate=_clone_model(model,{}, {},source_clock=clock)
    old_basis={p:model.shot_splines[b.shot_id].basis_pairs(b.spline_pts) for p,b in model.frames.items()}
    new_basis={p:candidate.shot_splines[b.shot_id].basis_pairs(b.spline_pts) for p,b in candidate.frames.items()}
    changed=sorted(p for p in wanted if saved[p] != clock[p])
    for p,old in model.frames.items():
        new=candidate.frames[p]
        if (old.pts,old.frame_key,old.shot_id,old.camera_id) != (new.pts,new.frame_key,new.shot_id,new.camera_id) or not np.array_equal(old.template25,new.template25):
            raise ValueError('Clock derivation changed source/pose/template identity')
        if model.dependencies(p) != candidate.dependencies(p):
            raise ValueError('Clock derivation crosses a dependency span; explicit graph key migration required')
        if p not in changed and old_basis[p] != new_basis[p]:
            raise ValueError('Clock derivation changed a noncollision basis')
    before=strict_windows(model,old_pairs)
    after=strict_windows(candidate,pairs)
    delta=len(after)-len(before)
    effective=deepcopy(phase)
    effective.update(exposure_clock_policy=policy, exposure_clock=metadata,
        frame_clock_virtual_pts={str(p):clock[p] for p in sorted(clock)},
        valid_temporal_acceleration_pairs=sorted(map(list,pairs)))
    expected_delta=delta
    if phase.get('complete_motion_prior'):
        completion=_complete_plan(candidate,clock,metadata,pairs)
        expected_delta += completion['added_factor_count']-phase['complete_motion_prior']['added_factor_count']
        effective['complete_motion_prior']=completion
    effective['factor_counts']=dict(phase['factor_counts'])
    effective['factor_counts']['total_with_model_priors'] += delta
    effective['graph_factors_with_source_calibration'] += expected_delta
    metadata['saved_graph_clock_candidate'].update(camera_graph_rebuilt=True,
        camera_states_solved=False, clock_only=False,
        graph_reconstruction_and_real_source_validation_required=True,
        native_registration='effective programme model supplied before manager construction')
    binding=dict(schema='rinkspace.production-saved-pose-clock-derivation.v1',
        policy=policy,parent_phase_sha256=parent_phase_sha256,
        parent_phase_role='immutable historical reconstruction evidence, not the new objective',
        parent_clock_sha256=json_sha(dict(clock=phase['frame_clock_virtual_pts'],
            metadata=phase['exposure_clock'],pairs=phase['valid_temporal_acceleration_pairs'])),
        effective_clock_sha256=json_sha(dict(clock=effective['frame_clock_virtual_pts'],
            metadata=metadata,pairs=effective['valid_temporal_acceleration_pairs'])),
        source_manifest_binding=deepcopy(metadata['source_manifest_binding']),
        source_sha256=phase['source_sha256'], changed_source_pts=changed,
        changed_basis_source_pts=sorted(p for p in wanted if old_basis[p] != new_basis[p]),
        old_strict_acceleration_count=len(before),new_strict_acceleration_count=len(after),
        parent_complete_motion_count=phase.get('complete_motion_prior',{}).get('added_factor_count',0),
        new_complete_motion_count=effective.get('complete_motion_prior',{}).get('added_factor_count',0),
        source_basis_sha256=json_sha({str(p):new_basis[p] for p in sorted(wanted)}),
        parent_graph_factor_count=phase['graph_factors_with_source_calibration'],
        expected_new_graph_factor_count=effective['graph_factors_with_source_calibration'],
        expected_graph_factor_delta=expected_delta,
        pose_keys_knots_templates_preserved=True,prior_targets_modified=False,
        old_pose_sidecar_reapplied=False,physical_capture_clock_verified=False,
        new_native_clock_parity_pending=True,solve_started=False)
    effective['production_clock_derivation']=binding
    binding['effective_phase_sha256']=json_sha(effective)
    return {**context,'clock':clock,'valid_pairs':pairs},effective,candidate,binding


def ordered_inventory(prepared):
    """Bind the actual ordered factors, noise and newly copied native basis.

    Source callbacks are additionally bound by frozen source/code hashes and
    the existing family inventories. This inventory does not claim parity or
    physical accuracy; those require real evaluation outside this factory.
    """
    records=prepared.native_manager._records
    rows=[]
    for index,factor in enumerate(prepared.graph._pending):
        row=dict(index=index,type=type(factor).__name__,keys=list(map(int,factor.keys())))
        if hasattr(factor,'noiseModel'):
            noise=factor.noiseModel()
            row['noise']=str(noise)
        if index in records:
            record=records[index]
            row.update(source_pts=int(record.source_pts),native_spec_sha256=json_sha(record.spec),
                source_uv_sha256=record.observed_sha256,quality_sha256=record.quality_sha256,
                world_sha256=record.world_sha256)
        rows.append(row)
    return dict(schema='rinkspace.production-clock-ordered-factor-inventory.v1',
        ordered_factors=rows,ordered_factor_sha256=json_sha(rows),
        factor_count=len(rows), native_records=len(records))
