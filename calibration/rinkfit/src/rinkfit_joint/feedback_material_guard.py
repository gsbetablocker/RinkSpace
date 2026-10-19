"""Execute a retrieved material-order lesson against this frame's evidence.

A recipe can veto an unsupported pass, never create a pass or a replacement UV.
The original source means, windows, Query, W and raw flags are untouched.
"""
from collections import Counter


def check_retrieved_material_order(membership, authority, support_contains):
    cases = authority.get('feedback_regression_checks', [])
    if not cases:return dict(case_ids=[], executed=False, counts={})
    counts = Counter()
    for row in membership:
        evidence = row.get('actual_material_evidence', {})
        relation = evidence.get('same_board_material_relation')
        status = 'unknown_current_material_support'
        if row['status'] == 'pass':
            supported = bool(relation)
            if supported and row['material_role'] == 'board.contact':
                try:
                    supported = support_contains(row['current_uv'],
                        evidence.get('foot_fall_support_endpoints_uv', []),
                        evidence.get('foot_profile_normal', []))
                except (ValueError, TypeError, IndexError):supported = False
            elif supported and row['material_role'] == 'board.top':
                transition = evidence.get('top_entry_transition', {})
                supported = support_contains(row['current_uv'],
                    transition.get('actual_source_optical_interval_UV', []), [0.,1.], xradius=12.)
            else:supported = False
            if supported:status = 'current_material_order_supported'
            else:
                row['status'] = 'unresolved'
                row['reason_codes'].append('retrieved_material_order_regression_lacks_current_transition_support')
                status = 'unsupported_pass_withheld'
        elif row['status'] == 'wrong_role':
            status = 'current_wrong_role_retained'
        row['feedback_material_order_regression'] = dict(case_ids=list(cases), status=status,
            source_uv_or_window_changed=False, new_pass_created=False)
        counts[status] += 1
    return dict(case_ids=list(cases), executed=True, counts=dict(counts),
        current_frame_evidence_rechecked=True, can_only_withhold=True)
