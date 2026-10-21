"""Numerical policy wrappers recurse through frozen source issuers unchanged."""
from pathlib import Path
import copy,json
from .io_utils import file_sha256
from . import r_inner_source_current_v1 as previous
SCHEMA='rinkfit.source_math_numerical_wrapper_authority.v1'
KEY='source_math_source_authority'

def checked(r,base,seen=None):
 p=Path(r.get('file',r.get('path')));p=p if p.is_absolute()else Path(base)/p;p=p.resolve();k=(str(p),r['sha256'])
 if seen is None or k not in seen:
  if file_sha256(p)!=r['sha256']:raise ValueError('Changed source mathematics dependency '+str(p))
  if seen is not None:seen.add(k)
 return p

def source_view(d,*,enriched=False):
 x=copy.deepcopy(d)
 for k in (KEY,'source_query_branch_registry'):x.pop(k,None)
 if enriched:
  x.pop('accepted_explicit_state_binding',None)
  for k in ('station','path','row','alpha'):x['arrays'].pop(k,None)
 return x

def assert_board_source_qualification_current(packet,packet_path,*,verified_dependency_cache=None):
 if KEY not in packet:return previous.assert_board_source_qualification_current(packet,packet_path,verified_dependency_cache=verified_dependency_cache)
 path=Path(packet_path).resolve();seen=set()if verified_dependency_cache is None else verified_dependency_cache;a=packet[KEY]
 if a.get('schema')!=SCHEMA or a.get('issuer_module_ref',{}).get('sha256')!=file_sha256(__file__):raise ValueError('Unrecognized source numerical-wrapper issuer')
 checked(a['issuer_module_ref'],path.parent,seen);pp=checked(a['canonical_source_packet'],path.parent,seen);mp=checked(a['canonical_source_model'],path.parent,seen)
 old=json.loads(pp.read_text(encoding='utf8'))
 if KEY in old:raise ValueError('Canonical source authority must directly name a frozen source issuer')
 from .joint_solver import assert_source_identity_current
 assert_source_identity_current(pp)
 enriched=a.get('numerical_state_kind')=='source_enriched_complete_primal'
 if a.get('numerical_state_kind')not in ('frozen_source_initialization','source_enriched_complete_primal'):raise ValueError('Unknown numerical wrapper state contract')
 if source_view(packet,enriched=enriched)!=source_view(old,enriched=enriched):raise ValueError('Numerical wrapper changed canonical non-auxiliary source fields')
 bindings=a['source_math_bindings']
 branches=[v['binding']for v in bindings if v.get('kind')=='source_query_branches']
 if len(branches)>1 or (branches and packet.get('source_query_branch_registry')!=branches[0])or(not branches and 'source_query_branch_registry'in packet):raise ValueError('Top-level typed source policy differs from ordered mathematics binding')
 from .source_math_identity_v1 import validate_source_math_bindings
 validate_source_math_bindings(bindings,packet,mp,path.parent,raw_packet_sha256=file_sha256(path),verified_dependency_cache=seen)
 if enriched:
  from .enriched_aux_prefix_v1 import assert_enriched_binding
  assert_enriched_binding(packet,path,verified_dependency_cache=seen)
 return dict(canonical_source_packet=str(pp),canonical_source_model=str(mp),source_math_bindings=bindings)
