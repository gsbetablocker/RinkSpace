"""Restore an archived objective before diagnostics; fail closed on mismatch.

Copy to body_tracking/audit_probe_contract.py together with the small patch.
This restores explicit block weights, not a different objective or new labels.
Model/source identity checks in restore_current_reference must remain enabled.
"""
from pathlib import Path
import hashlib
import numpy as np

SOURCE_FAMILIES = (
    'full_source_skate_structure', 'complete_runner_extent',
    'source_support_location_xy', 'native_source_support_contact',
    'source_pixel_edge_profile', 'complete_visible_structure_extent')


def choose_source_names(blocks):
    names=[b.name for b in blocks]
    if len(names)!=len(set(names)):
        raise ValueError('Ambiguous duplicate weighted block names')
    selected=[n for n in names if n in SOURCE_FAMILIES]
    if not selected:
        raise ValueError('No supported source family in actual weighted registry')
    return selected


def restore_saved_objective(problem, stage, saved):
    stage=Path(stage)
    fields=(('objective_block_weights_sha256','objective_block_weights.npz'),
            ('objective_policy_sha256','objective_policy.json'))
    receipts={}
    for key,name in fields:
        if key not in saved:
            raise ValueError('Checkpoint has no archived '+key+'; do not diagnose a default objective')
        path=stage/name
        if not path.is_file():raise FileNotFoundError(path)
        expected=str(np.asarray(saved[key]).item())
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual!=expected:raise ValueError('Archived objective hash mismatch: '+name)
        receipts[name]=actual
    with np.load(stage/'objective_block_weights.npz',allow_pickle=False) as z:
        weights={k:np.array(z[k],copy=True) for k in z.files}
    if not weights:raise ValueError('Empty archived objective weights')
    for name,w in weights.items():
        if w.dtype.kind not in 'fiu' or not np.isfinite(w).all() or np.any(w<=0):
            raise ValueError('Invalid archived weight '+name)
    problem.objective_block_weights=weights
    # prepare() is expected to have no evaluations yet. Never reuse a weighted
    # residual/J cached under a different policy.
    for obj in (problem,getattr(problem,'body',None)):
        if obj is not None:
            for key in ('cache','cache_x'):
                if hasattr(obj,key):setattr(obj,key,None)
    return dict(weights_restored=True,hashes=receipts,
                remaining_requirement='Use matching current source/model/camera and factor implementation; weights alone are not full replay proof')
