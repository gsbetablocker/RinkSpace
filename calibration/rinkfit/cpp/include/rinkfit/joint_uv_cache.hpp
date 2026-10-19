#pragma once
#include "rinkfit/joint_model.hpp"
#include <ceres/evaluation_callback.h>
namespace rinkfit {
// ABI matches the independently shadowed raw-UV CUDA output.
struct JointUVPartial {double u=0,v=0;int arithmetic=0,status=0;};
static_assert(sizeof(JointUVPartial)==24,"raw UV partial ABI");
struct JointUVCacheView {
 Vec2<double> canonical_uv{},canonical_AD_uv{};std::array<double,5>canonical_native_domains{},canonical_AD_native_domains{};std::array<bool,5>native_domain_available{};
 const JointUVPartial*partials=nullptr; // derivative column zero, primal excluded
 const std::vector<int>*block_column_starts=nullptr;
 int station_column=-1,row_column=-1;
 bool jacobians_ready=false;
};
class JointUVCache:public ceres::EvaluationCallback{
 public:virtual ~JointUVCache()=default;
 virtual bool lookup(std::size_t row,JointUVCacheView*view)const=0;
 virtual eventio::Json report()const=0;virtual void note_native_value_hit(bool)const{};virtual void note_direct_chain_hit()const{};
};
}
