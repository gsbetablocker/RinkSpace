from copy import deepcopy
from types import SimpleNamespace
import json
import numpy as np
import pytest
from calibration.program_tracking.first_shot_fit import FrozenWhitening
from calibration.program_tracking.first_shot_ordinary_quarantine import quarantine_ordinary_rows
from calibration.program_tracking.first_shot_source_transition import reference


def fixture(tmp_path,index=4):
    evidence=tmp_path/'RGB-review.json';evidence.write_text('original RGB review',encoding='utf8');ev=reference(evidence)
    rows=[dict(uv=np.array([[i+.2,i+.7]]),sigma=np.array([1.3+i]),target=dict(kind='ad_print',contour_key=str(i)),source_point_ids=[str(i)],source_packet=ev)for i in range(5)]
    groups=[dict(point_indices=[0,2],operator=SimpleNamespace(matrix=np.array([[1.,.2,-.4,.3],[.1,1.1,.3,-.2]])))]
    f=dict(index=7,native_pts=8128,rows=rows,source_point_count=5,whitening_groups=groups)
    r=rows[index];spec=dict(native_pts=8128,source_point_id=str(index),original_UV=r['uv'][0].tolist(),original_sigma=float(r['sigma'][0]),original_target=r['target'],source_packet=ev,authority='agent_rgb_review',material_identity_confirmed=False,identity_confirmed_by_user=False,ordinary_two_pixel_components=True,reason='Original extraction mixes two actual physical contours',evidence_references=[ev])
    m=dict(schema='rinkspace.reviewed-ordinary-native-source-quarantine.v1',review_complete=True,source_UV_ID_sigma_W_approvals_and_native_PTS_unchanged=True,all_correlated_W_groups_must_be_retained_byte_exact=True,evidence_references=[ev],specifications=[spec])
    path=tmp_path/'manifest.json';path.write_text(json.dumps(m),encoding='utf8');return f,path,m


def test_ordinary_removal_preserves_complete_correlated_R_and_J(tmp_path):
    f,path,m=fixture(tmp_path);changed,proof=quarantine_ordinary_rows([f],reference(path));g=changed[0]
    keep=proof['retained_source_point_indices'];xy=np.ravel(np.column_stack((2*keep,2*keep+1)));rng=np.random.default_rng(9)
    for raw in (rng.normal(size=10),rng.normal(size=(10,7))):
        assert np.array_equal(FrozenWhitening(g).apply(raw[xy]),FrozenWhitening(f).apply(raw)[proof['retained_formal_indices']])
    assert g['rows'][0]is f['rows'][0]
    assert g['whitening_groups'][0]['operator']is f['whitening_groups'][0]['operator']
    assert g['inactive_ordinary_source_rows'][0]['original_row']is f['rows'][4]
    assert f['source_point_count']==5 and proof['removed_formal_rows']==2


@pytest.mark.parametrize('fault',['W_member','UV','sigma','unverified_evidence'])
def test_wrong_original_or_W_member_or_unverified_evidence_rejected(tmp_path,fault):
    f,path,m=fixture(tmp_path,0 if fault=='W_member'else 4)
    if fault=='UV':m['specifications'][0]['original_UV'][0]+=.01
    if fault=='sigma':m['specifications'][0]['original_sigma']+=.01
    if fault=='unverified_evidence':m['specifications'][0]['evidence_references']=[dict(path='unreviewed',sha256='no')]
    path.write_text(json.dumps(m),encoding='utf8')
    with pytest.raises(ValueError):quarantine_ordinary_rows([f],reference(path))
