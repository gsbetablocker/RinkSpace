"""Frozen event header; the shared strict reader is retained in OneDrive core."""
import rinkfit_start.native_norms as _core
from rinkfit_start.native_norms import *
from fixed_entity_norm_constraints import SCOPE_CONTRACT
_EXPECTED_CORE_SHA256='c26f8eb55a4ab5fbf7890a288ffa15864df998f1445e82bf35a479253f86ec51'
if _core.CORE_SOURCE_SHA256!=_EXPECTED_CORE_SHA256 or _core.sha(_core.__file__)!=_EXPECTED_CORE_SHA256:raise ValueError('Changed frozen native norm core')

def load_all_source_affine_norms(qbar,transport,offset,scale,free):
 return _core.load_all_source_affine_norms(qbar,transport,offset,scale,free,scope_contract=SCOPE_CONTRACT)
