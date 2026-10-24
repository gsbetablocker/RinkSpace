#pragma once
#include "rinkfit/joint_path_generation_v1.hpp"
#include <cstdint>
#include <stdexcept>
#include <vector>
namespace rinkfit {
// Paths are immutable while a graph/GPU generation is live. Snapshot them only
// once per accepted graph generation, never once per Ceres callback.
class JointCompactPathStateV1 {
 struct Run {std::size_t count;int path,cell;};
 const JointPacket* packet_=nullptr;const JointObservation* rows_data_=nullptr;std::vector<Run> runs_;std::size_t rows_=0;std::uint64_t token_=0,generation_=0;
 public:
 explicit JointCompactPathStateV1(const JointPacket&p){rebind(p,true);}
 private:
 void snapshot_paths(const JointPacket&p){
  packet_=&p;rows_=p.rows.size();rows_data_=p.rows.data();generation_=p.numerical_path_generation_v1;runs_.clear();std::uint64_t h=1469598103934665603ull;
  for(const auto&r:p.rows){if(runs_.empty()||runs_.back().path!=r.path||runs_.back().cell!=r.numerical_shape_segment)runs_.push_back({1,r.path,r.numerical_shape_segment});else ++runs_.back().count;const auto v=std::uint32_t(r.path);for(int k=0;k<4;++k){h^=(v>>(8*k))&255;h*=1099511628211ull;}}
  for(const auto&r:p.rows){const auto v=std::uint32_t(r.numerical_shape_segment);for(int k=0;k<4;++k){h^=(v>>(8*k))&255;h*=1099511628211ull;}}token_=h;
 }
 public:
 void rebind(const JointPacket&p,bool previous_graph_destroyed){if(!previous_graph_destroyed)throw std::runtime_error("path generation rebind requires destroyed previous graph/GPU layouts");snapshot_paths(p);}
 JointCompactPathStateV1 prepare_local_replacement(const JointPacket&p,bool affected_closure_replaced)const{if(!affected_closure_replaced||packet_!=&p||rows_!=p.rows.size()||rows_data_!=p.rows.data())throw std::runtime_error("local path snapshot preparation requires the same raw registry and complete affected replacement");return JointCompactPathStateV1(p);}
 void rebind_after_local_replacement(const JointPacket&p,bool affected_closure_replaced){if(!affected_closure_replaced||packet_!=&p||rows_!=p.rows.size()||rows_data_!=p.rows.data())throw std::runtime_error("local path rebind requires the same raw registry and complete affected consumer replacement");snapshot_paths(p);}
 void adopt_exact_restored_local_generation(const JointPacket&p,bool affected_consumers_detached){if(!affected_consumers_detached||packet_!=&p||!deep_matches())throw std::runtime_error("local path rollback did not restore the complete accepted path/cell snapshot");rows_data_=p.rows.data();generation_=p.numerical_path_generation_v1;}

 bool matches()const{return packet_&&packet_->rows.size()==rows_&&packet_->rows.data()==rows_data_&&packet_->numerical_path_generation_v1==generation_;}
 bool deep_matches()const{if(!packet_||packet_->rows.size()!=rows_)return false;std::size_t i=0;for(const auto&run:runs_)for(std::size_t k=0;k<run.count;++k){const auto&r=packet_->rows[i++];if(r.path!=run.path||r.numerical_shape_segment!=run.cell)return false;}return i==rows_;}
 void require_current(bool deep_audit=false)const{if(!matches()||(deep_audit&&!deep_matches()))throw std::runtime_error("accepted path generation differs; destroy/rebuild graph and explicitly rebind before numeric restore");}
 void adopt_exact_restored_generation(const JointPacket&p,bool previous_graph_destroyed){if(!previous_graph_destroyed||packet_!=&p||!deep_matches())throw std::runtime_error("path generation rollback did not restore the complete accepted path/cell snapshot");rows_data_=p.rows.data();generation_=p.numerical_path_generation_v1;}
 void append_token(std::uint64_t&h)const{for(int k=0;k<8;++k){h^=(token_>>(8*k))&255;h*=1099511628211ull;}h^=rows_;h*=1099511628211ull;}
 std::size_t capacity_bytes()const{return runs_.capacity()*sizeof(Run);}
};
}
