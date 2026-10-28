"""Native distorted arcs must keep their actual material order and turns."""
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'events/ankara_2026/programmes/okada_mei_sp'))
from reread_s04_house import NativeHouseReader
from ice_points_s04 import select_representatives


def test_native_distorted_curve_is_not_replaced_by_fitted_ellipse():
    reader=object.__new__(NativeHouseReader)
    theta=np.linspace(0,2*np.pi,513)
    x=600*np.cos(theta)-450;y=80*np.sin(theta)+900
    # Offscreen radial distortion makes the whole projection non-conic.
    curve=np.c_[x*(1+1e-7*x*x),y]
    points=curve[[0,2,8,16,450,480,500]]
    nearest=reader.closest_source_curve_points(curve,points)
    np.testing.assert_allclose(nearest,points,atol=1e-9)
    np.testing.assert_allclose(reader.transformed_conic_exact_distance(reader.conic(curve),points,np.zeros(6)),0,atol=1e-9)


def test_partial_circle_density_reduction_retains_supported_turn():
    theta=np.r_[np.linspace(0,.8,12),np.linspace(5.5,2*np.pi,12)]
    guide=np.c_[500*np.cos(theta),900+100*np.sin(theta)]
    raw=guide+np.c_[np.zeros(len(theta)),.2*np.sin(np.arange(len(theta)))]
    records=[dict(uv=q.tolist(),feature_id='ice.C0:centreline',pixel_support=dict(normal=[np.cos(t),np.sin(t)],query_index=i))for i,(q,t)in enumerate(zip(raw,theta))]
    keep,_=select_representatives(records,dict(kind='ice_circle'),source_curve_guide=guide,source_curve_order=theta)
    assert len(keep)>2
    assert min(np.linalg.norm(raw[i]-[500,900])for i in keep)<1
    assert min(raw[i,1]for i in keep)<840
    assert max(raw[i,1]for i in keep)>960
    np.testing.assert_array_equal(np.asarray([r['uv']for r in records]),raw)


def test_three_actual_line_reads_keep_the_two_separated_endpoints():
    raw=np.array([[262.3816,922.8902],[218.3549,1016.3507],[209.7783,1039.5971]])
    records=[dict(uv=q.tolist(),feature_id='ice.C:segment0',pixel_support=dict(normal=[1.,0.],query_index=i))for i,q in enumerate(raw)]
    keep,_=select_representatives(records,dict(kind='ice_segment'))
    assert 0 in keep and 2 in keep
    np.testing.assert_array_equal(np.asarray([r['uv']for r in records]),raw)
