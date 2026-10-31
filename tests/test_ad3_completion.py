"""Official shape completion keeps the existing affine chart and vertices."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'events/ankara_2026/programmes/okada_mei_sp'))
from build_ad3_completion import composition_affine,expanded_bounds_same_center


def test_bounds_expand_only_needed_axis_and_keep_native_center_exact():
    old=np.array([.13996585833120412,.11259450933786673,.5797795694091402,.8836591027782571])
    new=np.array(expanded_bounds_same_center(old,[[.5900220667249664,.1224633889408978],[.8604677218680195,.8557196232088574]]))
    assert new[[1,3]].tobytes()==old[[1,3]].tobytes()
    assert old[0]+(old[2]-old[0])/2==new[0]+(new[2]-new[0])/2
    assert new[0]<0 and new[2]>=.8604677218680195
    seed=np.array([.15,.31,.57]);translation=.02;logscale=.17
    before=seed+translation+np.expm1(logscale)*(seed-(old[0]+(old[2]-old[0])/2))
    after=seed+translation+np.expm1(logscale)*(seed-(new[0]+(new[2]-new[0])/2))
    assert before.tobytes()==after.tobytes()


def test_common_contours_recover_one_affine_without_mutating_material_uv():
    xy=np.array([[0.,0.],[3.,0.],[3.,2.],[0.,0.]])
    matrix=np.array([[.02,.001],[-.001,.03],[.2,.1]])
    uv=np.c_[xy,np.ones(len(xy))]@matrix
    template=dict(paths=[dict(original_id='official',material_uv=uv.tolist())])
    composition=dict(contours=[dict(id='official',xy_composition=xy.tolist())])
    result,common,error=composition_affine(template,composition)
    np.testing.assert_allclose(result,matrix,atol=1e-14)
    assert error<1e-14 and len(common)==1
    assert np.array(template['paths'][0]['material_uv']).tobytes()==uv.tobytes()
    composition['contours'][0]['xy_composition']=xy[:-1].tolist()
    with pytest.raises(ValueError,match='vertex correspondence'):composition_affine(template,composition)
