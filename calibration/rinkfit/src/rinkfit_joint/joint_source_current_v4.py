"""Current source identity entry with separately selected capture primal import."""
from pathlib import Path
from .io_utils import read_json
from . import joint_source_current_v3 as prior

def assert_source_identity_current(packet,*,return_verified_packet=False,allow_unbound_source_prototype=False):
 path=Path(packet).resolve();doc=read_json(path)
 if 'source_capture_complete_state_binding' not in doc:return prior.assert_source_identity_current(path,return_verified_packet=return_verified_packet,allow_unbound_source_prototype=allow_unbound_source_prototype)
 from .capture_primal_prefix_v2 import SCHEMA,assert_capture_prefix_current
 if doc['source_capture_complete_state_binding']['schema']!=SCHEMA:raise ValueError('Unsupported capture-primal contract; no formal/explicit fallback')
 assert_capture_prefix_current(path)
 return doc if return_verified_packet else None
