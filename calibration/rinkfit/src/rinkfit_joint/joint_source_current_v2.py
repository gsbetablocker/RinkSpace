"""Versioned current-source entry; historical V5 issuer stays byte-for-byte.

The source-only path permits review/prototype qualification explicitly. A
production start requires a genuine complete capture-primal binding and a new
native consumer; this guard does not turn a prototype into an initialized state.
"""
from pathlib import Path
from .io_utils import read_json
from . import joint_solver as legacy

def assert_source_identity_current(packet,*,return_verified_packet=False,allow_unbound_source_prototype=False):
 path=Path(packet).resolve();doc=read_json(path)
 if 'source_only_board_successor'not in doc:return legacy.assert_source_identity_current(path,return_verified_packet=return_verified_packet)
 canonical_path=path
 if 'source_capture_complete_state_binding'in doc:canonical_path=Path(doc['source_capture_complete_state_binding']['canonical_source_packet']['file'])
 canonical=read_json(canonical_path);parent=Path(canonical['source_only_board_successor']['parent_packet']['file']);legacy.assert_source_identity_current(parent)
 from .board_middle_current_v2 import assert_current
 assert_current(canonical_path)
 if 'source_capture_complete_state_binding'in doc:
  from .capture_primal_prefix_v1 import assert_capture_prefix_current
  assert_capture_prefix_current(path)
 elif not allow_unbound_source_prototype:raise ValueError('Source qualification passed, but immutable accepted capture-primal parent is not bound; no production Solve is authorized by this packet')
 return doc if return_verified_packet else None
