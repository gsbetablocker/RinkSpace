"""Actual sealed full archive: no new cache compatibility or source fixture."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from calibration.solver_runtime import install
from calibration.solver_runtime.same_target_fresh_J_verification import (
    authenticate_qualified_complete_cache_request,restore_authenticated_complete_cache_profiles,_transform_main_source)


BASE=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003')
RUN=BASE/'okada_review86_samefe074_ICEbudget_B2_F2_Ad26_20261011_0631'
CACHE=RUN/'current_complete_linearization_cache'
SEED=RUN/'accepted_states/ccbd38b5ff3892209813ffa82736d6b64c479b42b0399c63936a8501fb003abf/complete_state_UNACCEPTED.npz'


@pytest.fixture
def actual():
    if not (CACHE/'actual_complete_cache_seal.json').exists():pytest.skip('Actual sealed diagnostic is local Data')
    install()
    from structured_programme_source_candidate import complete_same_state_replay as replay
    old=json.loads((CACHE/'input_manifest.json').read_text(encoding='utf8'))
    qual=Path(old['original_source_point_qualification']['packet_path'])
    return replay,old,qual


def test_actual_qualification_q_and_all_other_strict_source_gates(actual):
    replay,old,qual=actual
    receipt=authenticate_qualified_complete_cache_request(CACHE,qual,SEED)
    assert receipt['qualification_sha256']==old['original_source_point_qualification']['packet_sha256']
    current=deepcopy(old)
    current['native_backend_requested']['RINKSPACE_BENCHMARK_DIAGNOSTIC_DIR']='new output only'
    current['native_backend_requested']['RINKSPACE_STARTUP_DIAGNOSTIC_EVIDENCE']=str(CACHE)
    with np.load(SEED,allow_pickle=False)as z:q=z['q'].copy()
    current['joint_source_truth_restoration']['complete_initial_q_sha256']=replay._array_signature(q)['sha256']
    current['independent_pattern_new_slot_initializer']['parameter_values']=q[
        np.asarray(current['independent_pattern_placement_parameter_indices'],int)].tolist()
    current[replay.IDENTITY_KEY]['semantic_manifest_sha256']=replay._json_sha(replay._semantic_manifest(current))
    indices=old['shared_indices']
    mask=np.zeros(34340,bool);mask[indices]=True
    replay._check_strict_identity(old,current,fixed_parameter_indices=old[replay.IDENTITY_KEY]['fixed_parameter_indices'],shared_parameter_mask=mask)
    current['original_source_point_qualification']['packet_sha256']='0'*64
    current[replay.IDENTITY_KEY]['semantic_manifest_sha256']=replay._json_sha(replay._semantic_manifest(current))
    with pytest.raises(ValueError,match='Replay source/layout/physical-domain/backend differs'):
        replay._check_strict_identity(old,current,fixed_parameter_indices=old[replay.IDENTITY_KEY]['fixed_parameter_indices'],shared_parameter_mask=mask)


def test_stop_point_provenance_rejects_changed_physical_source_or_initializer(actual):
    replay,old,qual=actual
    authenticate_qualified_complete_cache_request(CACHE,qual,SEED)
    current=deepcopy(old)
    with np.load(SEED,allow_pickle=False)as z:q=z['q'].copy()
    current['joint_source_truth_restoration']['complete_initial_q_sha256']=replay._array_signature(q)['sha256']
    current['independent_pattern_new_slot_initializer']['parameter_values']=q[
        np.asarray(current['independent_pattern_placement_parameter_indices'],int)].tolist()
    mask=np.zeros(len(q),bool);mask[old['shared_indices']]=True
    current['independent_pattern_new_slot_initializer']['parameter_values'][0]+=1e-3
    current[replay.IDENTITY_KEY]['semantic_manifest_sha256']=replay._json_sha(replay._semantic_manifest(current))
    with pytest.raises(ValueError,match='initializer values are not'):
        replay._check_strict_identity(old,current,fixed_parameter_indices=old[replay.IDENTITY_KEY]['fixed_parameter_indices'],shared_parameter_mask=mask)


def test_qualification_mismatch_and_other_complete_q_refused_before_prepare(actual,tmp_path):
    _,_,qual=actual
    other=tmp_path/'different_qualification.json';other.write_text('{}')
    with pytest.raises(ValueError,match='Source point qualification differs'):
        authenticate_qualified_complete_cache_request(CACHE,other,SEED)
    otherq=tmp_path/'different_q.npz'
    with np.load(SEED,allow_pickle=False)as z:q=z['q'].copy()
    q[3]=np.nextafter(q[3],np.inf);np.savez(otherq,q=q)
    with pytest.raises(ValueError,match='another complete q'):
        authenticate_qualified_complete_cache_request(CACHE,qual,otherq)


def test_actual_parser_only_admits_full_cache_to_existing_strict_gate(actual):
    import inspect
    from calibration.solver_runtime.runtime import source_path
    import ast
    tree=ast.parse(source_path('run_current_named66_full_programme').read_text(encoding='utf8'))
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main')
    _,changes=_transform_main_source(ast.unparse(main))
    selected=[r for r in changes if r['kind']=='exact_qualified_complete_cache_parse_gate']
    assert len(selected)==1
    assert 'reuse_complete_linearization'not in selected[0]['after']
    assert all(name in selected[0]['after']for name in ('verify_source_blocks_from','reuse_wide_linearization','reuse_source_blocks_from'))


def test_actual_eighteen_cached_normal_profiles_restore_via_public_protocol(actual):
    from types import SimpleNamespace
    from z190_joint_source_solver import make_profile_update,_frozen_profile_token
    _,_,qual=actual
    authenticate_qualified_complete_cache_request(CACHE,qual,SEED)
    with np.load(SEED,allow_pickle=False)as z:
        q=z['q'].copy();initial={k[9:]:z[k].copy()for k in z.files if k.startswith('profile__')}
    # Only the profile protocol is exercised here, on the actual six saved
    # motion row shapes and complete arrays; no synthetic source is evaluated.
    bundles=[SimpleNamespace(motion_rows=2*len(initial[f'weights_{i}']),profile_weights=None,robust_phase=False)for i in range(6)]
    updater=make_profile_update([SimpleNamespace(bundle=b)for b in bundles]);updater.restore_checkpoint(initial)
    previous=_frozen_profile_token(updater)
    restore_authenticated_complete_cache_profiles(CACHE,q,SimpleNamespace(updater=updater))
    with np.load(CACHE/'actual_initial_frozen_profile_arrays.npz',allow_pickle=False)as z:
        for k,v in updater.iter_checkpoint_arrays():np.testing.assert_array_equal(v,z[k])
    assert _frozen_profile_token(updater)!=previous
