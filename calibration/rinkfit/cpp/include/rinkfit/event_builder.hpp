#pragma once
#include "rinkfit/event_factors.hpp"
#include <ceres/ceres.h>
namespace rinkfit {
class EventProblemBuilder final : public ProblemBuilder {
 public:
  EventProblemBuilder(EventPacket packet,EventModel model,const std::filesystem::path& model_path,const std::filesystem::path& initial_dir,const std::filesystem::path& query_manifest);
  const FrozenIdentity& identity() const override{return identity_;}
  const std::vector<ObservationSpec>& observations()const override{return packet_.observations;}
  const ConstraintSchema& schema()const override{return schema_;}
  double* budget_parameter()override{return &budget_;}
  std::vector<ParameterBlockBinding> parameter_blocks()override;
  StateSnapshot capture_state()const override;
  void restore_state(const StateSnapshot&)override;
  NativeReplay replay_native(const StateSnapshot&)const override;
  ConstraintValues evaluate_constraints(const StateSnapshot&)const override;
  GradientReport lagrangian_gradient(const StateSnapshot&,Phase,const std::vector<double>&,const std::vector<double>&)const override;
  BuildInventory build_ceres_problem(ceres::Problem&,const ObjectiveContext&)override;
  eventio::Json state_json(const StateSnapshot&)const;
  eventio::Json check_directional_derivatives();
  eventio::Json check_material_bindings();
  eventio::Json check_active_source_accounting();
  eventio::Json check_phr_equivalence();
  eventio::Json bounds_summary(const StateSnapshot&)const;
  eventio::Json domain_summary(const StateSnapshot&)const;
  eventio::Json constraint_schema_json()const;
  const EventPacket& packet()const{return packet_;}
 private:
  struct Factor {std::unique_ptr<ceres::CostFunction> cost;std::vector<std::size_t> blocks;};
  struct Frozen {EventModel model;std::vector<double> station,row,alpha;double budget;};
  EventRowEvaluation row_eval(std::size_t i,const EventModel* model=nullptr)const;
  std::vector<std::size_t> row_blocks(std::size_t i,bool alpha=false)const;
  Factor pixel_factor(std::size_t)const;Factor rs_factor(std::size_t)const;Factor query_factor(std::size_t,int)const;Factor physical_factor(std::size_t)const;Factor domain_factor(std::size_t,int)const;
  Factor statistical_factor(const EventStatisticalGroup&)const;Factor order_factor(std::size_t)const;
  Frozen decode(const StateSnapshot&)const;
  std::vector<const double*> pointers(const Factor&,const Frozen&)const;
  std::vector<double*> mutable_pointers(const Factor&);
  std::vector<int> sizes(const std::vector<std::size_t>&)const;
  double* block_pointer(std::size_t);const double* frozen_pointer(std::size_t,const Frozen&)const;
  std::size_t station_id(std::size_t i)const{return model_.blocks.size()+1+i;}
  std::size_t row_id(std::size_t i)const{return model_.blocks.size()+1+station_.size()+i;}
  std::size_t alpha_id(std::size_t i)const{return model_.blocks.size()+1+2*station_.size()+i;}
  std::size_t budget_id()const{return model_.blocks.size();}
  EventPacket packet_;EventModel model_;FrozenIdentity identity_;ConstraintSchema schema_;
  std::filesystem::path model_path_;std::string model_sha_,query_hash_;
  std::vector<double> station_,row_,alpha_;std::vector<std::uint32_t> path_;
  std::vector<EventRowLayout> layouts_;std::vector<EventPhysicalConstraint> physical_;
  struct Order{EventRunOrderConstraint functor;std::size_t left,right;std::string id;};
  std::vector<Order> orders_;std::map<std::string,std::string> discrete_;
  std::vector<std::shared_ptr<const EventQueryCurve>> queries_;std::vector<std::size_t> profile_rows_;
  double budget_=1.;
};
} // namespace rinkfit
