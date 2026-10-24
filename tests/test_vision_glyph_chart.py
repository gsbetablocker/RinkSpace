"""Known projected text geometry; no fitting fixtures or alternate fonts."""
from copy import deepcopy
import numpy as np
import pytest

from rinkspace.vision_glyph_chart import derive_text_chart


def transform(points, matrix):
    h = np.c_[points, np.ones(len(points))] @ matrix.T
    return h[:, :2]/h[:, 2:]


def example():
    source = np.asarray([[1.1,.07,680.],[-.03,1.3,240.],[.00007,0.,1.]])
    paths, corners = {}, {}
    for key,x,width in [('I0',0.,10.),('N',30.,36.),('I1',80.,10.),('M',110.,40.)]:
        q = np.asarray([[x,0.],[x+width,0.],[x+width,60.],[x,60.],[x,0.]])
        points = np.concatenate([np.linspace(a,b,90,endpoint=False) for a,b in zip(q[:-1],q[1:])])
        paths[key] = {'source_native_uv':transform(points,source).tolist()}
        corners[key] = transform(q,source)
    # Rounded glyph overshoot and its independently retained inner hole.
    theta = np.linspace(0,2*np.pi,160)
    outer = np.c_[180+21*np.cos(theta),30+31.5*np.sin(theta)]
    inner = np.c_[180+13*np.cos(theta),30+22*np.sin(theta)]
    paths['O'] = {'source_native_uv':transform(outer,source).tolist()}
    paths['O_inner'] = {'source_native_uv':transform(inner,source).tolist()}
    return paths,corners


def test_one_chart_recovers_projected_text_axes_and_keeps_gaps():
    paths,corners = example()
    result = derive_text_chart(paths,['I0','N','I1','M'],['I0','I1'])
    H=np.asarray(result['H'])
    recovered={k:transform(p,H) for k,p in corners.items()}
    # All caps and baselines recover globally within source localization noise.
    assert max(abs(p[:2,1]).max() for p in recovered.values()) < .001
    assert max(abs(p[2:4,1]-1).max() for p in recovered.values()) < .001
    assert max(abs(p[0,0]-p[3,0]) for p in recovered.values()) < .001
    # Two identical I stems and their source separation share one scale.
    w0=recovered['I0'][1,0]-recovered['I0'][0,0]
    w1=recovered['I1'][1,0]-recovered['I1'][0,0]
    gap=recovered['I1'][0,0]-recovered['I0'][0,0]
    assert abs(w0-w1) < .001
    assert abs(gap/w0-8) < .002
    assert result['metadata']['source_observations_created'] == 0
    assert result['metadata']['content_frame_is_observed_banner_rectangle'] is False
    assert result['metadata']['physical_aspect_verified'] is False


def test_rounded_overshoot_and_hole_are_not_clamped_or_replaced():
    paths,_=example();original=deepcopy(paths)
    result=derive_text_chart(paths,['I0','N','I1','M'],['I0','I1'])
    H=np.asarray(result['H'])
    outer=transform(np.asarray(paths['O']['source_native_uv']),H)
    inner=transform(np.asarray(paths['O_inner']['source_native_uv']),H)
    assert outer[:,1].min() < -.015 and outer[:,1].max() > 1.015
    assert inner[:,1].min() > .1 and inner[:,1].max() < .9
    assert paths==original


def test_missing_support_and_nonfinite_pixels_fail():
    paths,_=example()
    with pytest.raises(ValueError,match='missing'):
        derive_text_chart(paths,['I0','N','absent'],['I0'])
    paths['O']['source_native_uv'][0][0]=float('nan')
    with pytest.raises(ValueError,match='finite'):
        derive_text_chart(paths,['I0','N','I1'],['I0'])
