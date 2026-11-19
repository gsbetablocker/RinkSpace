"""Source ambiguity is not converted into a complete, identified boot mask."""
import importlib.util
from pathlib import Path

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FILE=ROOT/'events/ankara_2026/programmes/okada_mei_sp/tools/identity_refit/source_occlusion_review.py'


def module():
    import sys
    sys.path.insert(0,str(FILE.parent))
    spec=importlib.util.spec_from_file_location('source_occlusion_review',FILE)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def rectangle(x,y,w,h):
    return np.r_[np.c_[np.linspace(x,x+w,20,endpoint=False),np.full(20,y)],
        np.c_[np.full(20,x+w),np.linspace(y,y+h,20,endpoint=False)],
        np.c_[np.linspace(x+w,x,20,endpoint=False),np.full(20,y+h)],
        np.c_[np.full(20,x),np.linspace(y+h,y,20,endpoint=False)]].astype(np.float32)


def test_duplicate_mask_does_not_supply_two_identified_boot_outlines():
    q=module();p=rectangle(80,80,36,50)
    rgb=np.full((240,240,3),120,np.uint8);cv2.fillPoly(rgb,[p.astype(int)],(230,230,230))
    valid,lines,contrast,overlap=q.qualify_arcs(rgb,np.stack([p,p]))
    assert overlap>.99
    assert not valid.any()


def test_long_rink_line_is_excluded_but_short_straight_boot_edge_remains():
    q=module();p=rectangle(100,100,30,30)
    rgb=np.full((240,240,3),160,np.uint8)
    cv2.fillPoly(rgb,[p.astype(int)],(230,230,230))
    cv2.line(rgb,(0,100),(239,100),(40,40,40),2)
    valid,lines,contrast,overlap=q.qualify_arcs(rgb,np.stack([p,np.full_like(p,np.nan)]))
    assert lines[0,:20].sum()>=15
    assert valid[0,40:60].sum()>=15
    assert not lines[0,40:60].any()
