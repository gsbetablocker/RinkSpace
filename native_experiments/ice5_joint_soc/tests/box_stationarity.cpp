#include "rinkspace/box_stationarity.hpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>
#include <iostream>

int main() {
    using rinkspace::BoxTangentSigns;
    // F(x)=.5*(x-3)^2, x<=2: true constrained optimum has raw g=-1.
    BoxTangentSigns upper;upper.restrict(2.,0.,2.);
    assert(upper.project_gradient(-1.)==0.);
    assert(upper.project_gradient(1.)==1.); // inward descent must remain visible
    assert(!std::isfinite(upper.project_gradient(-INFINITY)));
    BoxTangentSigns lower;lower.restrict(0.,0.,2.);
    assert(lower.project_gradient(1.)==0.);
    assert(lower.project_gradient(-1.)==-1.);
    // Physical x=2-.1*delta reverses the feasible chart direction.
    BoxTangentSigns reversed;reversed.restrict(2.,0.,2.,-.1);
    assert(reversed.project_gradient(.1)==0.);
    assert(reversed.project_gradient(-.1)==-.1);
    BoxTangentSigns fixed;fixed.restrict(.5,.5,.5);
    assert(fixed.project_gradient(-100.)==0.&&fixed.project_gradient(100.)==0.);
    BoxTangentSigns interior;interior.restrict(1.,0.,2.);
    assert(interior.project_gradient(7.)==7.);
    BoxTangentSigns unbounded;unbounded.restrict(1.,-INFINITY,INFINITY);
    assert(unbounded.project_gradient(-7.)==-7.);
    BoxTangentSigns roundoff;roundoff.restrict(std::nextafter(2.,0.),0.,2.);
    assert(roundoff.project_gradient(-1.)==0.);
    BoxTangentSigns clearly_interior;clearly_interior.restrict(2.-1e-9,0.,2.);
    assert(clearly_interior.project_gradient(-1.)==-1.);
    // Intersect independently sourced bounds on the SAME coordinate.
    BoxTangentSigns intersection;intersection.restrict(0.,0.,1.);
    intersection.restrict(2.,0.,2.);
    assert(intersection.project_gradient(1.)==0.&&intersection.project_gradient(-1.)==0.);
    // A coupled constraint must retain its gradient in this conservative
    // diagnostic: at x+y=0, g=(1,2), direction (1,-1) is feasible descent.
    BoxTangentSigns coupled_not_projected;
    assert(coupled_not_projected.project_gradient(2.)==2.);
    std::cout << "box stationarity regression cases passed\n";
}
