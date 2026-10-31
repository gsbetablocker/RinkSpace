import cv2
import numpy as np
import pytest

from rinkspace.vision_programme_ice_review import _actual_profile_candidates


def read(profile, *, separate):
    offsets=np.arange(len(profile),dtype=float)
    smooth=cv2.GaussianBlur(np.asarray(profile,float)[None],(5,1),1.).ravel()
    return _actual_profile_candidates(np.zeros(2),np.array([0.,1.]),offsets,
        np.asarray(profile,float),smooth,operator='paint_center',signal='dark',
        transition=None,with_quantiles=True,separate_bands=separate)


def test_bright_gap_cannot_pair_edges_of_two_adjacent_dark_bands():
    # Native source luminance at S03 PTS720896, x1388, y992..1022.
    profile=np.array([193.13,191.53,190.38,190.78,191.18,191.18,192.21,191.41,
        184.10,175.50,172.61,172.41,172.50,172.90,179.18,193.78,206.18,208.18,
        202.04,194.24,191.18,191.18,191.58,194.18,201.00,205.40,205.79,205.39,
        205.39,203.39,201.39])
    rows=read(profile,separate=True)
    main=[r for r in rows if 8<r[0][1]<15]
    assert len(main)==1
    assert main[0][1][1,1]<17  # Main band ends before the bright separating ridge.
    assert not any(r[1][0,1]<14 and r[1][1,1]>20 for r in rows)


def test_single_band_with_darker_interior_remains_one_complete_band():
    profile=np.full(101,220.)
    profile[25:65]=150.
    profile[40:48]=110.
    rows=read(profile,separate=True)
    assert len(rows)==1
    assert 30<rows[0][0][1]<55
    assert rows[0][1][0,1]<40 and rows[0][1][1,1]>60


@pytest.mark.parametrize('transition', ['dark_to_light', 'light_to_dark'])
def test_signed_interfaces_do_not_merge_or_integrate_across_reverse_material_transition(transition):
    profile=np.full(60,40.)
    profile[20:22]=220.
    profile[24:]=220.
    if transition=='light_to_dark':profile=255.-profile
    original=profile.copy();profile.flags.writeable=False
    options=dict(operator='material_interface',signal='dark',transition=transition,with_quantiles=True)
    args=(np.zeros(2),np.array([0.,1.]),np.arange(len(profile),dtype=float),profile,profile)
    default=_actual_profile_candidates(*args,**options)
    legacy=_actual_profile_candidates(*args,separate_bands=False,**options)
    assert len(default)==len(legacy)==1 and default[0][2]
    for old,current in zip(default[0],legacy[0]):
        np.testing.assert_array_equal(old,current)
    independent=_actual_profile_candidates(*args,separate_bands=True,**options)
    assert len(independent)==2 and not any(row[2] for row in independent)
    np.testing.assert_allclose([row[0][1] for row in independent],[19.5,23.5],atol=1e-12)
    # Each integration extension stays on its own side of the reverse step;
    # separating only the candidate list would still bias these quartiles.
    assert independent[0][3][0,2,1]<21.
    assert independent[1][3][0,0,1]>22.
    np.testing.assert_array_equal(profile,original)
