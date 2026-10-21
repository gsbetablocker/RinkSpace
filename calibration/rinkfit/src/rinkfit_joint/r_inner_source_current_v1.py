"""Strict source dispatch for the explicitly bound V9 -> R33 append."""
from pathlib import Path
from .static_board_prefix_qualification import _json
from . import paint_operator_source_current as previous
from . import r_inner_prefix_v1 as current

def assert_board_source_qualification_current(packet,packet_path,*,verified_dependency_cache=None):
 path=Path(packet_path).resolve();seen=set()if verified_dependency_cache is None else verified_dependency_cache
 if packet.get('board_material_source_qualification'):
  cp,cert=_json(packet['board_material_source_qualification'],path.parent,seen)
  if cert.get('schema')==current.SCHEMA:return current.assert_r_inner_prefix(packet,path,certificate=(cp,cert),verified_dependency_cache=seen)
 return previous.assert_board_source_qualification_current(packet,path,verified_dependency_cache=seen)
