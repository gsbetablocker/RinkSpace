from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest
from scipy.interpolate import make_interp_spline
from calibration.program_tracking.body_tracking.action_integration import ClipReference


def reference_packet():
    times=np.arange(0,.401,.02);P=np.c_[np.sin(2*np.pi*6*times)*.12,np.cos(2*np.pi*6*times)*.06,np.ones(len(times))]
    initial=dict(P=P,Q=np.tile(np.eye(3),(len(times),1,1)),local_R=np.tile(np.eye(3),(len(times),21,1,1)))
    p=dict(fit_time_seconds=times,pts=np.arange(len(times))*256+1000000)
    shared=SimpleNamespace(binding=SimpleNamespace(x=np.zeros(17)))
    return times,P,initial,p,shared


def test_source_clock_root_round_trip_and_exposure_nodes():
    t,P,initial,p,shared=reference_packet();curve=make_interp_spline(t,P,k=3)
    initial.update(root_curve_schema=np.array('rinkspace.source-clock-root-bspline.v1'),root_curve_knots=curve.t,root_curve_controls=curve.c,root_curve_degree=np.array(3))
    ref=ClipReference(shared,Path('.'),p,initial);query=(t[:,None]+np.linspace(-.008,.008,5)).ravel()
    np.testing.assert_allclose(ref.at(t)['P'],P,rtol=0,atol=1e-12)
    np.testing.assert_allclose(ref.at(query)['P'],curve(query),rtol=0,atol=1e-12)
    np.testing.assert_allclose(ref.root_derivative(query,2),curve(query,2),rtol=0,atol=1e-10)
    # This is the real lost-motion pattern: old 0.2s roots cannot retain it.
    legacy=ClipReference(shared,Path('.'),p,{k:v for k,v in initial.items()if not k.startswith('root_curve_')})
    assert np.max(abs(legacy.at(t)['P']-P))>.03


def test_invalid_root_archive_does_not_silently_refit():
    t,P,initial,p,shared=reference_packet();curve=make_interp_spline(t,P,k=3)
    initial.update(root_curve_schema=np.array('rinkspace.source-clock-root-bspline.v1'),root_curve_knots=curve.t,root_curve_controls=curve.c+.1,root_curve_degree=np.array(3))
    with pytest.raises(AssertionError,match='reproduce'):
        ClipReference(shared,Path('.'),p,initial)
