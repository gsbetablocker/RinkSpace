#pragma once
#include "rinkfit/joint_model.hpp"
#include <ceres/evaluation_callback.h>
#include <ceres/problem.h>
namespace rinkfit {
// ABI matches the independently shadowed raw-UV CUDA output.
struct JointUVPartial {double u=0,v=0;int arithmetic=0,status=0;};
static_assert(sizeof(JointUVPartial)==24,"raw UV partial ABI");
struct JointUVCacheView {
 Vec2<double> canonical_uv{},canonical_AD_uv{};std::array<double,5>canonical_native_domains{},canonical_AD_native_domains{};std::array<bool,5>native_domain_available{};
 const JointUVPartial*partials=nullptr; // compact derivatives, primal excluded
 const std::vector<int>*partial_columns=nullptr; // ambient column -> compact offset; -1 is a proven fixed/zero column
 const JointUVPartial& partial(int column)const{static const JointUVPartial zero{0,0,1,0};if(!partial_columns)return partials[column];const int offset=partial_columns->at(column);return offset<0?zero:partials[offset];}
 const std::vector<int>*block_column_starts=nullptr;
 int station_column=-1,row_column=-1;
 bool implicit_RS_capture=false,implicit_RS_valid=false;double implicit_RS_row=0,implicit_RS_denominator=0;
 int sensor_station_active=0;double implicit_station=0,sensor_station_dv_ds=0;
 bool jacobians_ready=false;
};
class JointUVCache:public ceres::EvaluationCallback{
 public:virtual ~JointUVCache()=default;
 virtual bool lookup(std::size_t row,JointUVCacheView*view)const=0;
 // Called after the actual stage/manifolds are installed, outside evaluation.
 virtual void configure_for_problem(const ceres::Problem&){}
 virtual void set_trial_point_diagnostics(bool){}
 virtual eventio::Json report()const=0;virtual void note_native_value_hit(bool)const{};virtual void note_direct_chain_hit()const{};
};
}
