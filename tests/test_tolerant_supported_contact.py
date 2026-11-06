import numpy as np
from calibration.program_tracking.body_tracking.spin_grasp_factors import tolerant_support_response

def test_supported_contact_has_gentle_nonflat_interior_and_exact_derivative():
    h=np.array([-.15,-.025,-.015,0.,.015,.025,.15]);response,D=tolerant_support_response(h,.025)
    assert response[3]==D[3]==0
    assert 0<response[4]<h[4]
    np.testing.assert_allclose(response,-response[::-1])
    np.testing.assert_allclose(D,D[::-1])
    step=1e-7
    np.testing.assert_allclose((tolerant_support_response(h+step,.025)[0]-tolerant_support_response(h-step,.025)[0])/(2*step),D,atol=1e-9)
    assert response[-1]>.99*h[-1]
