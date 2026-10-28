"""An old partial glyph cannot invalidate newly measured missing sides."""
import importlib.util
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
PROG=ROOT/'events/ankara_2026/programmes/okada_mei_sp'
sys.path.insert(0,str(PROG))
from ad_entity_observability_s01 import complete_named_source_reference


def test_complete_named_photograph_requires_matching_closed_contour():
    v={'path':'Ad1.print.SKATING_G'}
    p={'path_id':v['path'],'pixels':[[0,0],[10,0],[10,10],[0,0]],
       'proof':{'kind':'visually_identified_actual_source_glyph_by_word_order'}}
    ref={'atlas':{'partial_source_reference':False,'paths':[p]}}
    assert complete_named_source_reference(v,ref)
    ref['atlas']['partial_source_reference']=True
    assert not complete_named_source_reference(v,ref)
    ref['atlas']['partial_source_reference']=False
    p['pixels']=p['pixels'][:-1]
    assert not complete_named_source_reference(v,ref)


def test_unlabelled_photo_or_other_glyph_is_not_identity_proof():
    v={'path':'Ad1.print.SKATING_G'}
    p={'path_id':v['path'],'pixels':[[0,0],[10,0],[10,10],[0,0]],'proof':{'kind':'unassigned_component'}}
    ref={'atlas':{'partial_source_reference':False,'paths':[p]}}
    assert not complete_named_source_reference(v,ref)
    p['proof']['kind']='visually_identified_actual_source_glyph_by_word_order'
    assert not complete_named_source_reference({'path':'Ad1.print.SKATING_N'},ref)
