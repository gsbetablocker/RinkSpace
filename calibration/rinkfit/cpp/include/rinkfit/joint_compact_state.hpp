#pragma once
#include "rinkfit/joint_al.hpp"
#include <cstring>
namespace rinkfit {
// One immutable registry owns names/offsets. Accepted and trial values never alias.
class JointCompactAcceptedState {
 std::vector<ParameterBlockBinding>registry_;
 std::vector<std::size_t>offsets_;
 std::vector<double>values_;
 public:
 explicit JointCompactAcceptedState(JointALBuilder&b):registry_(b.parameter_blocks()){
  std::size_t n=0;for(const auto&r:registry_){offsets_.push_back(n);n+=r.size;}values_.resize(n);capture();
 }
 void capture(){for(std::size_t i=0;i<registry_.size();++i)std::memcpy(values_.data()+offsets_[i],registry_[i].data,registry_[i].size*sizeof(double));}
 void restore_values()const{for(std::size_t i=0;i<registry_.size();++i)std::memcpy(registry_[i].data,values_.data()+offsets_[i],registry_[i].size*sizeof(double));}
 void restore(JointALBuilder&b)const{restore_values();for(std::size_t i=0;i<b.core.packet.rows.size();++i)if(b.core.packet.rows[i].active){b.core.packet.rows[i].row=b.core.row[i];b.core.packet.rows[i].alpha=b.core.alpha[i];}}
 std::size_t scalar_count()const{return values_.size();}
 std::size_t value_capacity_bytes()const{return values_.capacity()*sizeof(double);}
 std::size_t registry_blocks()const{return registry_.size();}
 std::string token()const{std::uint64_t h=1469598103934665603ull;const auto*data=reinterpret_cast<const unsigned char*>(values_.data());for(std::size_t i=0;i<values_.size()*sizeof(double);++i){h^=data[i];h*=1099511628211ull;}return std::to_string(h);}
 bool matches_current_values()const{for(std::size_t i=0;i<registry_.size();++i)if(std::memcmp(values_.data()+offsets_[i],registry_[i].data,registry_[i].size*sizeof(double)))return false;return true;}
};
}
