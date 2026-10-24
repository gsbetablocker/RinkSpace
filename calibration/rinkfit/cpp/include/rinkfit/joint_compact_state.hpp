#pragma once
#include "rinkfit/joint_al.hpp"
#include "rinkfit/joint_global_phr_epoch_v1.hpp"
#include <cstring>
#include "rinkfit/joint_compact_path_state_v1.hpp"
namespace rinkfit {
// One immutable registry owns names/offsets. Accepted and trial values never alias.
class JointCompactAcceptedState {
 std::vector<ParameterBlockBinding>registry_;
 std::vector<std::size_t>offsets_,active_rows_;
 std::vector<double>values_;
 std::uint64_t numeric_token_=1469598103934665603ull;
 JointCompactGlobalEpochV1 small_state_;
 JointCompactPathStateV1 path_state_;
 public:
 explicit JointCompactAcceptedState(JointALBuilder&b):registry_(b.parameter_blocks()),small_state_(b.engineering_global_phr_state_v1),path_state_(b.core.packet){
  for(std::size_t i=0;i<b.core.packet.rows.size();++i)if(b.core.packet.rows[i].active)active_rows_.push_back(i);
  std::size_t n=0;for(const auto&r:registry_){offsets_.push_back(n);n+=r.size;}values_.resize(n);capture();
 }
 void capture(){path_state_.require_current();numeric_token_=1469598103934665603ull;for(std::size_t i=0;i<registry_.size();++i){std::memcpy(values_.data()+offsets_[i],registry_[i].data,registry_[i].size*sizeof(double));const auto*data=reinterpret_cast<const unsigned char*>(values_.data()+offsets_[i]);for(std::size_t k=0;k<registry_[i].size*sizeof(double);++k){numeric_token_^=data[k];numeric_token_*=1099511628211ull;}}small_state_.capture();}
 void restore_values()const{path_state_.require_current();for(std::size_t i=0;i<registry_.size();++i)std::memcpy(registry_[i].data,values_.data()+offsets_[i],registry_[i].size*sizeof(double));small_state_.restore();}
 void restore(JointALBuilder&b)const{restore_values();for(auto i:active_rows_){b.core.packet.rows[i].row=b.core.row[i];b.core.packet.rows[i].alpha=b.core.alpha[i];}}
 void capture_rebuilt_path_generation_v1(JointALBuilder&b,bool previous_graph_destroyed){if(!path_state_.matches())path_state_.rebind(b.core.packet,previous_graph_destroyed);capture();}
 void restore_rebuilt_path_generation_v1(JointALBuilder&b,bool previous_graph_destroyed){path_state_.adopt_exact_restored_generation(b.core.packet,previous_graph_destroyed);restore(b);}
 JointCompactPathStateV1 prepare_local_path_rebind_v1(JointALBuilder&b,bool affected_closure_replaced)const{return path_state_.prepare_local_replacement(b.core.packet,affected_closure_replaced);}
 void capture_prepared_local_path_rebind_v1(JointCompactPathStateV1&&prepared){prepared.require_current();path_state_=std::move(prepared);capture();}
 void capture_after_local_path_rebind_v1(JointALBuilder&b,bool affected_closure_replaced){path_state_.rebind_after_local_replacement(b.core.packet,affected_closure_replaced);capture();}
 void restore_after_local_path_rebind_v1(JointALBuilder&b,bool affected_consumers_detached){path_state_.adopt_exact_restored_local_generation(b.core.packet,affected_consumers_detached);restore(b);}
 std::size_t path_snapshot_capacity_bytes()const{return path_state_.capacity_bytes();}
 // Used only after an actual factor failure; normal iterations do no lookup.
 bool saved_parameter_view_v1(const double*address,int size,std::string*name,const double**value)const{for(std::size_t i=0;i<registry_.size();++i)if(registry_[i].data==address){if(int(registry_[i].size)!=size)return false;*name=registry_[i].name;*value=values_.data()+offsets_[i];return true;}return false;}
 std::size_t scalar_count()const{return values_.size();}
 std::size_t value_capacity_bytes()const{return values_.capacity()*sizeof(double);}
 std::size_t registry_blocks()const{return registry_.size();}
 std::string token()const{std::uint64_t h=numeric_token_;small_state_.append_token(h);path_state_.append_token(h);return std::to_string(h);}
 bool matches_current_values(bool deep_path_audit=false)const{if(!small_state_.matches()||!path_state_.matches()||(deep_path_audit&&!path_state_.deep_matches()))return false;for(std::size_t i=0;i<registry_.size();++i)if(std::memcmp(values_.data()+offsets_[i],registry_[i].data,registry_[i].size*sizeof(double)))return false;return true;}
};
}
