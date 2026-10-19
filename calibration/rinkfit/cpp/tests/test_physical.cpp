#include "rinkfit/physical.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

namespace {
bool near(double a,double b,double tolerance=1e-11) { return std::abs(a-b)<=tolerance; }
void test_mechanism() {
  rinkfit::FixedHead<double> head;
  head.P={2.,3.,-4.}; head.d={.2,-.1,.3}; head.a=.12; head.pupil_linear=.08;
  const auto pose=rinkfit::camera_pose(head,.5,-.2,.3);
  assert(pose.valid());
  const auto Rt=rinkfit::transpose(pose.R);
  auto h=rinkfit::cross(head.n,rinkfit::matvec(Rt,head.b));
  h=rinkfit::scale(h,1./std::sqrt(rinkfit::dot(h,h)));
  auto lever=head.d; lever[2]+=head.pupil_linear*.3;
  const auto recovered=rinkfit::sub(rinkfit::sub(pose.C,rinkfit::matvec(Rt,lever)),rinkfit::scale(h,head.a));
  for(int k=0;k<3;++k) assert(near(recovered[k],head.P[k]));
  assert(rinkfit::valid_rotation(pose.R));
}
void test_projection_and_finite_geometry() {
  rinkfit::Pose<double> pose;
  rinkfit::Intrinsics<double> K{900.,900.,959.5,539.5};
  rinkfit::Brown<double> D;
  const auto p=rinkfit::project(rinkfit::Vec3<double>{.2,.1,2.},pose,K,D);
  assert(p.valid()); assert(near(p.uv[0],1049.5)); assert(near(p.uv[1],584.5));
  const auto failed=rinkfit::project(rinkfit::Vec3<double>{0.,0.,-1.},pose,K,D);
  assert(!failed.valid()); assert(failed.status==rinkfit::DomainStatus::kNonPositiveDepth);
  assert(std::isnan(failed.uv[0]));
  const rinkfit::Vec3<double> A{0.,0.,0.},B{1.,0.,0.};
  const auto endpoint=rinkfit::finite_line_point(A,B,1.);
  assert(near((endpoint[0]-2.)*(endpoint[0]-2.)+.1*.1,1.01));
  bool threw=false; try { (void)rinkfit::finite_line_point(A,B,1.1); }
  catch(const std::domain_error&) { threw=true; } assert(threw);
}
void test_function_gauge_and_row_time() {
  const auto weights=rinkfit::cubic_weights(.35);
  const std::array<double,3> free{.2,-.1,.4};
  const double dependent=rinkfit::gauge_dependent_control(weights,1,free);
  const std::array<double,4> full{free[0],dependent,free[1],free[2]};
  assert(near(rinkfit::cubic_value(full,.35),0.,1e-15));
  assert(!near(dependent,0.));
  assert(near(rinkfit::row_time(.2,.012,0.,1080),.194));
  assert(near(rinkfit::row_time(.2,.012,1079.,1080),.206));
  assert(near(rinkfit::row_time(.2,-.012,0.,1080),.206));
  rinkfit::Projection<double> p;
  p.status=rinkfit::DomainStatus::kValid; p.uv={100.,20.}; p.depth=1.;
  assert(near(rinkfit::explicit_rs_equality(22.,p),2.));
}
}
int main() {
  test_mechanism(); test_projection_and_finite_geometry(); test_function_gauge_and_row_time();
  std::cout << "physical equation core: all checks passed\n";
}

// IMPLEMENTED: standalone double compile/runtime check of shared rigid head,
// failed-projection status/NaN, finite endpoint, exact cubic function gauge,
// signed row timing/equality.  NOT CLAIMED: a Ceres optimizer, implicit C++
// root solver, production all-source replay, or a completed physical fit.
