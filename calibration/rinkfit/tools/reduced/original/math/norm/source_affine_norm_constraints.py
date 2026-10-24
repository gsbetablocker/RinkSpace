"""Frozen event header; the shared strict reader is retained in OneDrive core."""
import rinkfit_start.native_norms as _core
from rinkfit_start.native_norms import *
from fixed_entity_norm_constraints import SCOPE_CONTRACT,scope_contract_for

def load_all_source_affine_norms(qbar,transport,offset,scale,free):
 return _core.load_all_source_affine_norms(qbar,transport,offset,scale,free,scope_contract=scope_contract_for(transport))
