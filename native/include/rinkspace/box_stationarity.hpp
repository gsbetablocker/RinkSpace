#pragma once
#include "core_types.hpp"

namespace rinkspace {
// Feasible signs for ONE independent coordinate. Only apply a physical bound
// here when its pullback has exactly one nonzero column. A coupled bound such
// as x+y>=0 must not be projected one coordinate at a time.
struct BoxTangentSigns {
    bool positive=true;
    bool negative=true;

    RS_HD void restrict(double value,double lower,double upper,double derivative=1.) {
        if(derivative==0.)return;
        const double eps=64*std::numeric_limits<double>::epsilon();
        const double lo_tol=eps*(1.+std::abs(value)+(finite_scalar(lower)?std::abs(lower):0.));
        const double hi_tol=eps*(1.+std::abs(value)+(finite_scalar(upper)?std::abs(upper):0.));
        if(finite_scalar(lower)&&value<=lower+lo_tol) {
            if(derivative>0.)negative=false;else positive=false;
        }
        if(finite_scalar(upper)&&value>=upper-hi_tol) {
            if(derivative>0.)positive=false;else negative=false;
        }
    }

    RS_HD double project_gradient(double gradient) const {
        // The descent direction is -gradient; an outward normal component
        // does not prevent constrained stationarity at an active box face.
        if(!finite_scalar(gradient))return gradient;
        return (gradient>0.&&!negative)||(gradient<0.&&!positive)?0.:gradient;
    }
};
} // namespace rinkspace
