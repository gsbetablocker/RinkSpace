#pragma once
#include "rinkfit/joint_explicit.hpp"
#include "rinkfit/augmented_lagrangian.hpp"
namespace rinkfit {
class JointALBuilder final:public ProblemBuilder {
 public:
 JointProblemBuilder core;explicit JointALBuilder(JointEventModel,JointPacket,bool allow_initialization_surrogate=false,bool phase_one_geometry_guidance=false);
 const FrozenIdentity&identity()const override{return identity_;}const std::vector<ObservationSpec>&observations()const override{return observations_;}const ConstraintSchema&schema()const override{return schema_;}double*budget_parameter()override{return &budget_;}
 std::vector<ParameterBlockBinding>parameter_blocks()override;StateSnapshot capture_state()const override;void restore_state(const StateSnapshot&)override;NativeReplay replay_native(const StateSnapshot&)const override;ConstraintValues evaluate_constraints(const StateSnapshot&)const override;GradientReport lagrangian_gradient(const StateSnapshot&,Phase,const std::vector<double>&,const std::vector<double>&)const override;BuildInventory build_ceres_problem(ceres::Problem&,const ObjectiveContext&)override;
 BuildInventory build_advertisement_coordinate_problem(ceres::Problem&,const ObjectiveContext&,int,std::vector<std::size_t>*);
 eventio::Json linearized_actual_step_diagnostic(const StateSnapshot&candidate,const ObjectiveContext&context);
 bool initialization_geometry_guidance()const{return phase_one_geometry_guidance_;}
 BuildInventory build_engineering_problem(ceres::Problem&,const eventio::Json&,eventio::Json*);
 BuildInventory build_engineering_problem(ceres::Problem&,const eventio::Json&,eventio::Json*,const std::vector<double>& physical_lambda,const std::vector<double>& equality_nu);
 // Original physical/native/zoom g and RS/Query h, excluding pixel/budget g.
 ConstraintValues evaluate_current_engineering_constraints()const;
 void set_engineering_identity(const std::string& configuration_sha);
 void export_explicit_nuisance(const std::filesystem::path&)const;
 void enable_gpu_raw_uv(const std::string&ptx,int threads,const std::filesystem::path&log_file);ceres::EvaluationCallback*gpu_evaluation_callback()const{return gpu_uv_.get();}eventio::Json gpu_report()const{return gpu_uv_?gpu_uv_->report():eventio::Json::object();}
 private:
 struct Factor{std::unique_ptr<ceres::CostFunction>cost;std::vector<std::size_t>ids;};
 std::shared_ptr<JointUVCache>gpu_uv_;FrozenIdentity identity_;std::vector<ObservationSpec>observations_;ConstraintSchema schema_;std::vector<std::size_t>active_,station_rows_,profile_rows_;std::vector<EventPhysicalConstraint>physical_;struct Domain{std::size_t row;int kind;};std::vector<Domain>native_domains_,equalities_;struct HardEquality{std::string id;std::vector<std::size_t>ids;JointPhysicalEquality functor;};std::vector<HardEquality>hard_equalities_;struct ZoomBound{std::size_t rig;int control,offset,upper;std::string id;};std::vector<ZoomBound>zoom_bounds_;std::vector<std::array<double,2>>zoom_enclosures_,zoom_box_;EventModel domain_math_;double budget_=1;bool allow_initialization_surrogate_=false,phase_one_geometry_guidance_=false;
 std::size_t station_id(std::size_t i)const{return core.model.blocks.size()+i;}std::size_t row_id(std::size_t i)const{return core.model.blocks.size()+core.packet.rows.size()+i;}std::size_t alpha_id(std::size_t i)const{return core.model.blocks.size()+2*core.packet.rows.size()+i;}std::size_t budget_id()const{return core.model.blocks.size()+3*core.packet.rows.size();}
 std::size_t zoom_id(std::size_t rig)const{return budget_id()+1+rig;}
 double*pointer(std::size_t);std::vector<double*>pointers(const Factor&);std::vector<int>sizes(const std::vector<std::size_t>&)const;std::size_t equality_count()const{return equalities_.size()+hard_equalities_.size();}Factor pixel(std::size_t)const;Factor physical(std::size_t)const;Factor native_domain(std::size_t)const;Factor zoom_bound(std::size_t)const;Factor zoom_certificate(std::size_t)const;Factor equality(std::size_t)const;Factor statistic(const JointStatisticalGroup&)const;JointExplicitRow row(std::size_t,std::vector<std::size_t>*)const;std::unique_ptr<JointALBuilder>frozen(const StateSnapshot&)const;
};
}
