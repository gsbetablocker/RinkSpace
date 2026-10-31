from collections import defaultdict

from qualify_registration_sources import bounded_spans, row_reason
from prepare_native_source_update import authenticate_replacement_spans
import numpy as np


def test_json_span_does_not_retire_unreviewed_hole():
    key=(256,"line")
    original={"groups":[{"pts":256,"path_id":"line","spans":[[[0.,0.],[2.,0.]]]}]}
    rows={key:[(0,{"uv":[0.,0.]}),(1,{"uv":[1.,0.]}),(2,{"uv":[2.,0.]})]}
    result,_=bounded_spans(original,rows,{key:{0,2}})
    assert result["groups"][0]["spans"]==[[[0.,0.],[0.,0.]],[[2.,0.],[2.,0.]]]
    points={"pts":np.array([256,256]),"path_indices":np.array([0,0]),"uv":np.array([[0.,0.],[2.,0.]])}
    assert len(authenticate_replacement_spans({"path_ids":["line"],"source_replacement":result},points))==1


def test_resolving_ring_boundaries_does_not_resolve_unreviewed_center():
    decisions={"pending_regions":[],"resolved_material_pending":{"shot":"s09","pts":256,"entities":["R3:inner","R3:middle","R3:outer"]}}
    assert row_reason({"entity":"R3:outer","material_pending":True},"s09",256,decisions)==(None,True)
    assert row_reason({"entity":"R3:center","material_pending":True},"s09",256,decisions)==("material_pending_without_explicit_source_resolution",False)
