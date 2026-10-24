"""Frozen event header; the shared strict reader is retained in OneDrive core."""
import rinkfit_start.native_norms as _core
from rinkfit_start.native_norms import *

SCOPE_CONTRACT={"raw_rows": 182017,"packet_sha256": "9b5c3d8803a3210c1456f0f5977dc5cdde9cbe0b8d4d4b8cc6b7cb1d75e468e4","source_identity": "81d615f651e5019802ec7de0198315d68e7cde9586ee9d478a3829166a4f0794","original_formal_rank": 214112,"original_groups_asset_sha256": "f4719f1735b4bc5916dd08de4d33a824a431a8769b85107389d5a3c16172a807","original_weights_array_sha256": "42a8fec0e8eff0725e4187cc987bf65a837a043714802d64cb45cb7c6856415f"}

def scope_contract_for(transport_manifest):
 t=_core.read(transport_manifest);pk=_core.read(_core.verify(t['packet']))
 if pk.get('current_input_scope',{}).get('schema')=='rinkfit.current_advertisement_input.v1':
  return dict(current_prepared_input=True,raw_rows=int(pk['observations']),packet_sha256=t['packet']['sha256'],source_identity=pk['immutable_source_hashes']['observations.npz'])
 return SCOPE_CONTRACT

def load_fixed_entity_norms(qbar_manifest,transport_manifest,offset,scale,free):
 return _core.load_fixed_entity_norms(qbar_manifest,transport_manifest,offset,scale,free,scope_contract=scope_contract_for(transport_manifest))
