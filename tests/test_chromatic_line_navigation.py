"""Source-search regressions: colour/material alternatives remain distinct."""
import cv2
import numpy as np
from rinkspace.session.chromatic_line_navigation import source_chromatic_line_navigation, _same_source_band,source_board_floor_limit


def test_yellow_board_band_cannot_be_a_red_source_stripe():
    image=np.full((480,640,3),220,np.uint8)
    cv2.rectangle(image,(0,190),(639,215),(235,210,85),-1)
    nav,record=source_chromatic_line_navigation(image,np.zeros((480,640),bool),[1.,0.])
    assert nav is None
    assert record['physical_identity_established'] is False


def test_opposite_hough_sides_of_one_wide_red_stripe_are_not_competitors():
    image=np.full((600,800,3),240,np.uint8)
    cv2.fillPoly(image,[np.array([[380,25],[410,25],[560,590],[220,590]],np.int32)],(230,180,180))
    nav,record=source_chromatic_line_navigation(image,np.zeros((600,800),bool),[0.,1.])
    assert nav is not None,record
    assert record['status']=='unique_long_chromatic_source_search'
    assert record['navigation_only']
    assert record['source_search_span_px']>400


def test_two_disconnected_red_stripes_remain_ambiguous():
    image=np.full((600,800,3),240,np.uint8)
    for x in (170,630):cv2.rectangle(image,(x-12,25),(x+12,590),(230,170,170),-1)
    nav,record=source_chromatic_line_navigation(image,np.zeros((600,800),bool),[0.,1.])
    assert nav is None
    assert record['status']=='competing_long_chromatic_source_branches'


def test_perspective_sides_need_actual_paint_between_them():
    active=np.zeros((600,800),np.uint8)
    left=np.array([[380.,25.],[220.,590.]])
    right=np.array([[410.,25.],[560.,590.]])
    cv2.fillPoly(active,[np.array([left[0],right[0],right[1],left[1]],np.int32)],1)
    exclusion=np.zeros(active.shape,bool);exclusion[250:350]=True
    active[exclusion]=0
    assert _same_source_band(active.astype(bool),exclusion,left,right)
    active[:]=0
    for line in (left,right):cv2.line(active,tuple(line[0].astype(int)),tuple(line[1].astype(int)),1,8)
    assert not _same_source_band(active.astype(bool),exclusion,left,right)


def test_source_floor_search_uses_long_kick_plate_not_small_scoreboard():
    rgb=np.full((600,900,3),220,np.uint8)
    cv2.rectangle(rgb,(30,20),(100,60),(230,200,70),-1)
    cv2.fillPoly(rgb,[np.array([[0,400],[899,430],[899,443],[0,413]])],(225,190,80))
    y,receipt=source_board_floor_limit(rgb,np.zeros(rgb.shape[:2],bool))
    assert y==400 and receipt['navigation_only']
