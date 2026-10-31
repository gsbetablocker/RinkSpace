"""Native source pixels remain unchanged while audit markers are illustrated."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'events/ankara_2026/programmes/okada_mei_sp'))
from build_source_review_panels import source_crop_box,paint_case,page_rows,family,tier


def test_crop_contains_all_native_points_keeps_margin_and_minimum_without_resize():
    points=np.array([[5.25,4.],[1918.8,1078.9]])
    box=source_crop_box(points)
    assert box==[0,0,1920,1080]
    box=source_crop_box([[100.,100.]])
    assert box[2]-box[0]>=180 and box[3]-box[1]>=100
    assert box[0]<=40 and box[2]>=161 and box[1]<=40 and box[3]>=161


def test_paired_source_crop_has_exact_original_left_pixels_and_no_source_mutation():
    rng=np.random.default_rng(0);image=rng.integers(0,256,(1080,1920,3),dtype=np.uint8)
    original=image.copy();rows=[dict(uv=[100.25,100.75],local_label=1)]
    box=source_crop_box([rows[0]['uv']]);x0,y0,x1,y1=box
    canvas=paint_case(image,rows,rows,box,'source audit')
    h,w=y1-y0,x1-x0
    np.testing.assert_array_equal(canvas[52:52+h,:w],image[y0:y1,x0:x1])
    np.testing.assert_array_equal(image,original)
    assert canvas.shape==(h+52,2*w+8,3)


def test_print_family_merges_glyphs_but_board_and_ice_paths_stay_separate():
    assert family('print_instance/Ad7::Ad7.A')==family('print_instance/Ad7::Ad7.B')=='print_instance/Ad7'
    assert family('board/continuous/top')!=family('board/continuous/bottom')
    assert family('d78_material_1/path0')!=family('d78_material_2/path0')
    assert [tier(x)for x in (81,80,40)]==['01_gt80','02_gt40_to80','03_gt20_to40']
    cases=[dict(width=600),dict(width=650),dict(width=700),dict(width=1200),dict(width=500)]
    assert list(map(len,page_rows(cases)))==[3,1,1]
