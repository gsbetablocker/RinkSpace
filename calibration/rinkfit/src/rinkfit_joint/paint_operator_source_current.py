"""Next-boundary source dispatch for explicit measured-operator supersession."""
from pathlib import Path
from .static_board_prefix_qualification import _json
from . import curve_source_prefix_qualification as previous
from . import paint_operator_transition as transition

def assert_board_source_qualification_current(packet,packet_path,*,verified_dependency_cache=None):
    path=Path(packet_path).resolve();seen=set()if verified_dependency_cache is None else verified_dependency_cache
    ref=packet.get('board_material_source_qualification')
    if ref:
        cp,cert=_json(ref,path.parent,seen)
        if cert.get('schema')==transition.SCHEMA:return transition.assert_operator_transition(packet,path,certificate=(cp,cert),verified_dependency_cache=seen)
    return previous.assert_board_source_qualification_current(packet,path,verified_dependency_cache=seen)
