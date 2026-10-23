#include "rinkfit/joint_periodic_station.hpp"
#include <ceres/ceres.h>
#include <cmath>
#include <iostream>
#include <stdexcept>
struct CircleResidual {
 double target;
 template<class T> bool operator()(const T* s,T* r)const{using std::cos;using std::sin;const T angle=T(2*std::acos(-1.))*s[0];r[0]=cos(angle)-T(std::cos(2*std::acos(-1.)*target));r[1]=sin(angle)-T(std::sin(2*std::acos(-1.)*target));return true;}
};
int main(){
 rinkfit::JointPeriodicStationManifold m;double x=1,d=0,y=0,j=0;m.Plus(&x,&d,&y);if(y!=1)throw std::runtime_error("zero update changed saved endpoint");
 x=.999;d=.002;m.Plus(&x,&d,&y);if(std::abs(y-.001)>1e-14)throw std::runtime_error("positive seam blocked");
 x=.001;d=-.002;m.Plus(&x,&d,&y);if(std::abs(y-.999)>1e-14)throw std::runtime_error("negative seam blocked");
 for(const auto entry:{std::pair<double,double>{1.,.001},{0.,.999},{.99,.01},{.01,.99}}){
  double s=entry.first;ceres::Problem p;p.AddResidualBlock(new ceres::AutoDiffCostFunction<CircleResidual,2,1>(new CircleResidual{entry.second}),nullptr,&s);p.SetManifold(&s,new rinkfit::JointPeriodicStationManifold);
  ceres::Solver::Options o;o.linear_solver_type=ceres::DENSE_QR;o.function_tolerance=1e-14;o.gradient_tolerance=1e-14;o.parameter_tolerance=1e-14;ceres::Solver::Summary result;ceres::Solve(o,&p,&result);
  if(!result.IsSolutionUsable()||result.final_cost>1e-20||s<0||s>=1)throw std::runtime_error("seam-crossing optimization failed");std::cout<<entry.first<<" -> "<<s<<" cost "<<result.final_cost<<'\n';
 }
 std::cout<<"periodic station tests passed\n";
}
