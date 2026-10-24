#pragma once
#include "rinkfit/joint_explicit.hpp"
#include <ceres/manifold.h>
namespace rinkfit {
// A closed nonnegative endpoint gap is not a full-dimensional smooth
// Manifold: clipping Plus at zero contradicts its Jacobian and Minus.  Keep
// this type only so old consumers fail before constructing a Ceres problem.
// Production uses the original Euclidean optical parameters, their ambient
// boxes, and JointCaptureOpticalDomainCost's unchanged Brown certificate.
// A future gap parameterization must use explicit bounded parameter blocks.
class JointOpticalFeasibleCoordinates final : public ceres::Manifold {
 public:
 JointOpticalFeasibleCoordinates(const JointRig&,const std::array<double,2>&,
                                 const EventParameterBlock&,std::array<bool,2> = {false,false}) {
  throw std::runtime_error("clipped optical endpoint gaps violate the Ceres Plus/Minus/Jacobian contract; use Euclidean optical parameters with the existing physical boxes and optical-domain gate");
 }
 int AmbientSize()const override{return 12;}
 int TangentSize()const override{return 12;}
 std::array<bool,2> active_endpoint_gaps()const{return {false,false};}
 bool endpoint_gaps(const double*,std::array<double,2>*)const{return false;}
 bool with_endpoint_gaps(const double*,const std::array<double,2>&,double*)const{return false;}
 std::array<std::array<double,2>,2> gap_coefficient_jacobian()const{
  throw std::runtime_error("disabled clipped optical endpoint-gap coordinates");
 }
 bool Plus(const double*,const double*,double*)const override{return false;}
 bool Minus(const double*,const double*,double*)const override{return false;}
 bool PlusJacobian(const double*,double*)const override{return false;}
 bool MinusJacobian(const double*,double*)const override{return false;}
};
}
