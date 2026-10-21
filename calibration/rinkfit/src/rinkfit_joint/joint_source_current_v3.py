"""Current-source entry with immutable native readout correction successors.
The unchanged historical V5 issuer remains the recursive source authority.
"""
from pathlib import Path
import importlib
from .io_utils import read_json
from . import joint_source_current_v2 as prior

def assert_source_identity_current(packet,*,return_verified_packet=False,allow_unbound_source_prototype=False):
 path=Path(packet).resolve();doc=read_json(path)
 if 'source_current_correction_successor'not in doc:return prior.assert_source_identity_current(path,return_verified_packet=return_verified_packet,allow_unbound_source_prototype=allow_unbound_source_prototype)
 if 'source_capture_complete_state_binding'in doc:raise ValueError('Corrected-source capture prefix needs explicit affected-q/RS numeric patch consumer; no old explicit-state contract substitution')
 b=doc['source_current_correction_successor'];parent=Path(b['parent_packet']['file']);prior.assert_source_identity_current(parent,allow_unbound_source_prototype=True)
 name=Path(b['consumer']['file']).stem
 if name not in('source_correction_successor_v1','source_correction_successor_v2','source_correction_successor_v3'):raise ValueError('Unknown current source-correction successor consumer')
 module=importlib.import_module('rinkfit_joint.'+name);module.assert_current(path)
 if not allow_unbound_source_prototype:raise ValueError('Correct source generation installed, but root-selected valid capture primal and affected q/RS rebind are pending; this source-only prototype cannot start Solve')
 return doc if return_verified_packet else None
