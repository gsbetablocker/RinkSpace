"""Strict source-only successor entry; numerical CP binding is a separate boundary."""
from pathlib import Path
from .io_utils import read_json,file_sha256
from .board_print_current_v1 import assert_current
from .joint_source_current_v4 import assert_source_identity_current as assert_parent

def assert_source_identity_current(packet,*,return_verified_packet=False):
 path=Path(packet).resolve();p=read_json(path);b=p.get('source_only_printed_board_successor')
 if not b:raise ValueError('Missing real printed-board source successor')
 parent=Path(b['parent_packet']['file'])
 if file_sha256(parent)!=b['parent_packet']['sha256']:raise ValueError('Printed-board actual source parent changed')
 assert_parent(parent)
 report=assert_current(path)
 if 'source_capture_complete_state_binding'in p:raise ValueError('Source-only packet carries stale numerical complete-state identity')
 if return_verified_packet:return p
 return dict(source_parent_strictly_verified=True,source_only_successor=report,solver_or_numeric_eligibility_issued=False)
