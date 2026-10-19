import json
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rinkfit_joint.qualified_scope import (EvidenceResolver, select_membership,
                                          select_group_statistics)
from rinkfit_joint.io_utils import file_sha256


def test_pending_qualification_is_not_wrong_role_or_revived():
    assert select_membership(True, False) == (False, 'pending_qualification')
    assert select_membership(True, True) == (True, 'active_qualified')
    assert select_membership(False, True, previous_inactive=True) == (False, 'previous_exclusion_preserved')
    assert select_membership(False, True, previous_inactive=True, explicit_requalification=True) == (True, 'active_qualified')
    assert select_membership(True, True, alias=True) == (False, 'exact_actual_readout_alias')
    assert select_membership(True, False, wrong_role=True) == (False, 'wrong_role')


def test_selected_covariance_preserves_cross_terms_and_rewhitens():
    F = np.array([[2.,0,0,0],[1,3,0,0],[.5,.7,1,0],[.2,.8,.3,2]])
    C = F @ F.T
    U, s, _ = np.linalg.svd(F, full_matrices=False)
    W = (U / s).T
    group = dict(id='exposure', rows=[4,7], rank=4, W='fixture',
                 statistic_kind='native_joint_readout', supported_covariance_declared=True)
    selected, M, proof = select_group_statistics(group, [7], lambda _: W)
    assert selected['rows'] == [7]
    assert np.max(abs(M @ C[2:,2:] @ M.T - np.eye(2))) < 1e-12
    assert proof['W_columns_sliced'] is False
    assert np.max(abs(W[:,2:] @ C[2:,2:] @ W[:,2:].T - np.eye(4))) > .1


def test_missing_statistics_keeps_qualified_engineering_members_and_old_declaration():
    group = dict(id='x', rows=[1,2], rank=4, W={'file':'missing'}, statistic_kind='native_joint_readout')
    def missing(_):
        raise FileNotFoundError('missing immutable W')
    result, W, proof = select_group_statistics(group, [2], missing)
    assert result['rows'] == [2] and result['rank'] == 0 and result['W'] == []
    assert result['statistical_status'] == 'statistical_unavailable'
    assert result['parent_statistical_declaration'] == group
    assert W is None and proof['precision_invented'] is False
    assert group['rank'] == 4


def test_no_covariance_is_inferred_from_undeclared_W_nullspace():
    group = dict(id='x', rows=[1,2], rank=1, W='fixture', statistic_kind='original')
    result, W, _ = select_group_statistics(group, [2], lambda _: np.ones((1,4)))
    assert result['rank'] == 0 and result['statistical_status'] == 'statistical_unavailable'


def test_rank_zero_membership_never_gains_precision():
    group = dict(id='x', rows=[1,2], rank=0, W=[], statistic_kind='constraint_only_unknown_noise')
    result, W, proof = select_group_statistics(group, [2], lambda _: pytest.fail('W loader used for rank zero'))
    assert result['rows'] == [2] and result['rank'] == 0 and result['W'] == [] and W is None
    assert proof['statistical_precision_invented'] is False


def test_dependency_relocation_requires_same_executed_bytes(tmp_path):
    original = tmp_path/'old.py'
    archive = tmp_path/'archive.py'
    original.write_text('old implementation', encoding='utf8')
    archive.write_bytes(original.read_bytes())
    ref = dict(file=str(original), sha256=file_sha256(original))
    target = dict(file=str(archive), sha256=file_sha256(archive))
    resolver = EvidenceResolver([dict(original=ref, archive=target)])
    original.write_text('new implementation', encoding='utf8')
    assert resolver(ref) == archive.resolve()
    archive.write_text('different code', encoding='utf8')
    with pytest.raises(ValueError, match='Changed qualification dependency'):
        EvidenceResolver([dict(original=ref, archive=target)])(ref)
    with pytest.raises(ValueError, match='changes bytes'):
        EvidenceResolver([dict(original=ref, archive={**target,'sha256':'0'*64})])


def test_streaming_rankzero_provenance_preserves_raw_registry_and_known_factors(tmp_path):
    from rinkfit_joint.qualified_scope_stats import repair_rankzero_source_provenance
    old = dict(schema='test', raw_ids=['same source ID', '同一原始源'],
        uv=[1.23456789012345, 9.25], padding='x'*(1024*1024-60),
        groups=[dict(id='unknown', rows=[0], rank=0, W=[], statistic_kind='constraint_only_unknown_noise'),
                dict(id='known', rows=[1], rank=2, W=[[1,0],[0,2]], statistic_kind='original')],
        native_pts=[256256, 256512], raw_ledger='preserved')
    source=tmp_path/'packet.json'
    source.write_text(json.dumps(old,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf8')
    report=repair_rankzero_source_provenance(source,tmp_path/'new')
    revised=json.loads((tmp_path/'new/packet.json').read_text(encoding='utf8'))
    assert report['repaired_groups']==1 and report['groups']==2
    assert revised['groups'][0]['unknown_noise_source_refs'][0]['sha256']==file_sha256(source)
    assert revised['groups'][0]['rank']==0 and revised['groups'][0]['W']==[]
    revised['groups'][0].pop('unknown_noise_source_refs')
    assert revised==old


def test_rankzero_provenance_never_sanitizes_an_invalid_precision(tmp_path):
    from rinkfit_joint.qualified_scope_stats import repair_rankzero_source_provenance
    source=tmp_path/'packet.json'
    source.write_text(json.dumps(dict(groups=[dict(id='invalid', rows=[0], rank=1, W=[1],
        statistic_kind='constraint_only_unknown_noise')]),separators=(',',':')),encoding='utf8')
    with pytest.raises(ValueError,match='cannot carry a precision'):
        repair_rankzero_source_provenance(source,tmp_path/'new')
