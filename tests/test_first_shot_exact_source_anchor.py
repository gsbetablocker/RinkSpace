from types import SimpleNamespace
import json
import numpy as np
import pytest
from scipy.sparse import csr_matrix,save_npz
from calibration.solver_runtime import install
install()
from calibration.program_tracking.first_shot_source_transition import reference
from calibration.program_tracking.first_shot_joint_stage_runtime import sha
from calibration.program_tracking.first_shot_exact_source_anchor import install_exact_source_anchor
from z190_streaming_source_normal import SourceRowBlock


def fixture(tmp_path):
    run=tmp_path/'closed';bank=run/'source_J_cache/generation_000000';bank.mkdir(parents=True)
    q=np.array([.2,.3]);J=csr_matrix([[1.,0.],[0.,2.],[1.,1.]]);r=np.asarray(J@q);scales=np.ones(3);mask=np.zeros(3,bool)
    evidence=tmp_path/'source_math.txt';evidence.write_text('unchanged numerical producer',encoding='utf8');producer=reference(evidence)
    save_npz(bank/'J.npz',J,compressed=False);payload=dict(q_augmented=q.copy(),aux__same=np.array([1,2]))
    np.savez(bank/'state.npz',**payload,r=r,scales=scales,quadratic=mask);np.savez(bank/'loss.npz')
    ledger=dict(family='unchanged-source',native_pts=[256256],producer=producer)
    m=dict(schema='rinkspace.complete-canonical-effective-joint-source-J.v1',qaug_sha256=sha(q),J_shape=[3,2],source_J_before_Huber_weights=True,all_active_objective_rows_present=True,source_producers=[producer],complete_current_state_r_scales_quadratic=reference(bank/'state.npz'),block_loss_weights=reference(bank/'loss.npz'),blocks=[dict(block=0,row_start=0,row_count=3,J_shape=[3,2],J_nonzeros=J.nnz,J=reference(bank/'J.npz'),row_ledger=ledger)])
    (bank/'manifest.json').write_text(json.dumps(m),encoding='utf8');(run/'owner_closed.json').write_text(json.dumps(dict(passed=True,normal_close_returned=True)),encoding='utf8');(run/'full_source_directional_QA.json').write_text(json.dumps(dict(passed=True,canonical_q_sha256=sha(q))),encoding='utf8')
    calls=[]
    def blocks(x,jacobian):
        calls.append(jacobian);yield SourceRowBlock(np.asarray(J@x),J.copy()if jacobian else None,mask,scales,ledger)
    owner=SimpleNamespace(start=q.copy(),scales=scales,quadratic=mask,source_producers=[producer],recovery_references=[],source_blocks=blocks,checkpoint_identity=lambda:{},complete_state_payload=lambda x:dict(q_augmented=x.copy(),aux__same=np.array([1,2])))
    return owner,reference(bank/'manifest.json'),calls,bank


def test_exact_anchor_checks_current_R_and_references_existing_J_only(tmp_path):
    owner,ref,calls,bank=fixture(tmp_path);receipt,checks=install_exact_source_anchor(owner,ref)
    block=list(owner.source_blocks(owner.start,True))[0]
    assert calls==[False]and len(checks)==1
    assert owner.source_J_block_reference(owner.start,0,block)==reference(bank/'J.npz')
    list(owner.source_blocks(owner.start+np.array([.01,0]),True))
    assert calls[-1]is True and receipt['changed_q_always_fresh_J']


def test_changed_complete_aux_or_current_source_is_rejected(tmp_path):
    owner,ref,calls,bank=fixture(tmp_path);owner.complete_state_payload=lambda x:dict(q_augmented=x.copy(),aux__same=np.array([9,2]))
    with pytest.raises(ValueError,match='auxiliary payload'):install_exact_source_anchor(owner,ref)
    owner,ref,calls,bank=fixture(tmp_path/'second');old=owner.source_blocks
    def changed(x,jac):
        for b in old(x,jac):yield SourceRowBlock(b.r+.01,b.J,b.quadratic,b.scales,b.row_ledger)
    owner.source_blocks=changed;install_exact_source_anchor(owner,ref)
    with pytest.raises(ValueError,match='Current raw source'):list(owner.source_blocks(owner.start,True))


def test_original_first_J_verifier_runs_and_then_allows_new_state(tmp_path):
    from types import MethodType
    owner,ref,calls,bank=fixture(tmp_path);pre_verification_blocks=owner.source_blocks;first=True
    before=dict(q=owner.start.copy());visits=[]
    def verify_first_J(this,q,jacobian):
        nonlocal first
        if jacobian and first:
            assert np.array_equal(q,before['q']);visits.append('actual-initial-J-verifier')
        yield from pre_verification_blocks(q,jacobian)
        if jacobian:first=False
    base=SimpleNamespace(scales=owner.scales)
    base.source_blocks=MethodType(verify_first_J,base)
    owner.source_blocks=lambda q,jacobian:base.source_blocks(q,jacobian)
    receipt,_=install_exact_source_anchor(owner,ref,source_prefix_owner=base)
    list(owner.source_blocks(owner.start,True));assert calls==[False]
    list(owner.source_blocks(owner.start+np.array([.01,0]),True))
    assert visits==['actual-initial-J-verifier']and calls[-1]is True
    assert receipt['original_initial_J_R_and_geometry_verifier_retained']


def test_run_provenance_moves_only_when_authenticated_JSON_content_matches(tmp_path):
    def prepare(folder,change=False):
        owner,ref,calls,bank=fixture(folder);old=bank.parents[1];new=folder/'new';(old/'construction').mkdir();(new/'construction').mkdir(parents=True)
        original=dict(source_facts='same physical source and parameters')
        (old/'execution_manifest.json').write_text(json.dumps(original),encoding='utf8')
        (new/'execution_manifest.json').write_text(json.dumps(dict(source_facts='changed'if change else original['source_facts'])),encoding='utf8')
        a=reference(old/'execution_manifest.json');b=reference(new/'execution_manifest.json')
        (old/'construction/camera_transition.json').write_text(json.dumps(dict(execution=a,camera='same')),encoding='utf8')
        (new/'construction/camera_transition.json').write_text(json.dumps(dict(execution=b,camera='same')),encoding='utf8')
        m=json.loads((bank/'manifest.json').read_text(encoding='utf8'));m['source_producers']+=[a,reference(old/'construction/camera_transition.json')]
        owner.source_producers+=[b,reference(new/'construction/camera_transition.json')]
        (bank/'manifest.json').write_text(json.dumps(m),encoding='utf8')
        return owner,reference(bank/'manifest.json')
    owner,ref=prepare(tmp_path/'valid');receipt,_=install_exact_source_anchor(owner,ref)
    assert len(receipt['run_provenance_successors'])==2
    owner,ref=prepare(tmp_path/'wrong',True)
    with pytest.raises(ValueError,match='Run-provenance successor changed'):install_exact_source_anchor(owner,ref)
