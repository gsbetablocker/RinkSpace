"""Blurred support is not the measured width of a complete material strip."""
import numpy as np
from scipy.special import erf
from calibration.program_tracking.native_ice_material_repair import whole_colour_bands


def read_band(width):
    x=np.arange(-185.,186.)
    red=12*.5*(erf((x+width/2)/(np.sqrt(2)*24))-erf((x-width/2)/(np.sqrt(2)*24)))
    return whole_colour_bands(np.array([900.,980.]),np.array([1.,0.]),x,red,np.ones(len(x),bool))


def test_diffuse_tails_do_not_discard_a_valid_measured_width():
    rows=read_band(170)
    assert len(rows)==1
    np.testing.assert_allclose(rows[0]['uv'],[900.,980.],atol=.01)
    np.testing.assert_allclose(rows[0]['width_px'],170,atol=.1)


def test_actual_overwide_material_still_fails_the_unchanged_width_gate():
    assert read_band(200)==[]
