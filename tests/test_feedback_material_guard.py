import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'calibration/rinkfit/src'))
from rinkfit_joint.feedback_material_guard import check_retrieved_material_order


def test_learned_material_check_can_withhold_but_never_promote():
    rows=[dict(source_row_id='bad',status='pass',material_role='board.contact',current_uv=[4.,5.],reason_codes=[],actual_material_evidence={}),
        dict(source_row_id='unknown',status='unresolved',material_role='board.contact',current_uv=[4.,5.],reason_codes=[],actual_material_evidence={})]
    calls=[]
    def support(*args,**kwargs):calls.append(args);return False
    result=check_retrieved_material_order(rows,dict(feedback_regression_checks=['recipe']),support)
    assert rows[0]['status']=='unresolved' and rows[1]['status']=='unresolved'
    assert result['counts']['unsupported_pass_withheld']==1
    assert rows[0]['current_uv']==[4.,5.]


def test_material_check_requires_current_lower_transition_evidence():
    ev=dict(same_board_material_relation=dict(first_iceward_fall_uv=[4.,5.]),
        foot_fall_support_endpoints_uv=[[4.,4.],[4.,6.]],foot_profile_normal=[0.,1.])
    row=dict(status='pass',material_role='board.contact',current_uv=[4.,5.],reason_codes=[],actual_material_evidence=ev)
    calls=[]
    def support(uv,ends,normal):calls.append((uv,ends,normal));return True
    result=check_retrieved_material_order([row],dict(feedback_regression_checks=['recipe']),support)
    assert len(calls)==1 and result['counts']==dict(current_material_order_supported=1)
    assert row['status']=='pass' and row['feedback_material_order_regression']['new_pass_created'] is False
