#pragma once
#include "rinkfit/joint_explicit.hpp"
#include "rinkfit/joint_motion_v1.hpp"
#include "rinkfit/joint_source_uv_capture_v1.hpp"
#include "rinkfit/augmented_lagrangian.hpp"
#include "rinkfit/joint_visible_ice_source_support.hpp"
#include "rinkfit/joint_source_query_branch_registry_v2.hpp"
namespace rinkfit {

class JointALBuilder;
class JointFormalHybridHookV1 {
 public:virtual~JointFormalHybridHookV1()=default;
 virtual const FrozenIdentity&identity()const=0;virtual const ConstraintSchema&schema()const=0;
 virtual std::vector<ParameterBlockBinding>parameter_blocks(JointALBuilder&)=0;
 virtual StateSnapshot capture_state(const JointALBuilder&)const=0;virtual void restore_state(JointALBuilder&,const StateSnapshot&)=0;
 virtual NativeReplay replay_native(JointALBuilder&,const StateSnapshot&)=0;
 virtual ConstraintValues evaluate_constraints(JointALBuilder&,const StateSnapshot&)=0;
 virtual GradientReport lagrangian_gradient(JointALBuilder&,const StateSnapshot&,Phase,const std::vector<double>&,const std::vector<double>&)=0;
 virtual BuildInventory build_ceres_problem(JointALBuilder&,ceres::Problem&,const ObjectiveContext&)=0;
 virtual ceres::EvaluationCallback*evaluation_callback()const=0;
};
#define RINKFIT_JOINT_FORMAL_HYBRID_HOOK_V1 1
class JointALBuilder final:public ProblemBuilder {
 public:
 std::vector<JointCaptureStationCellV20>capture_cells_v20;eventio::Json capture_cell_policy_ref_v20;std::string capture_cell_graph_receipt_v20,capture_cell_state_digest_v20="uninstalled";std::map<std::string,std::vector<std::uint8_t>>capture_cell_payload_v20;eventio::Json capture_cell_array_specs_v20;
 std::map<std::string,std::vector<ceres::ResidualBlockId>> engineering_residual_groups;
 std::shared_ptr<JointMotionStoreV1> static_motion;
 JointProblemBuilder core;explicit JointALBuilder(JointEventModel,JointPacket,bool allow_initialization_surrogate=false,bool phase_one_geometry_guidance=false);
 const FrozenIdentity&identity()const override{return hybrid_formal_hook_v1_?hybrid_formal_hook_v1_->identity():identity_;}const std::vector<ObservationSpec>&observations()const override{ensure_observed_catalogue();return observations_;}const ConstraintSchema&schema()const override{if(hybrid_formal_hook_v1_)return hybrid_formal_hook_v1_->schema();ensure_observed_catalogue();return schema_;}double*budget_parameter()override{return &budget_;}
 std::vector<ParameterBlockBinding>parameter_blocks()override;StateSnapshot capture_state()const override;void restore_state(const StateSnapshot&)override;NativeReplay replay_native(const StateSnapshot&)const override;ConstraintValues evaluate_constraints(const StateSnapshot&)const override;GradientReport lagrangian_gradient(const StateSnapshot&,Phase,const std::vector<double>&,const std::vector<double>&)const override;BuildInventory build_ceres_problem(ceres::Problem&,const ObjectiveContext&)override;
 BuildInventory build_advertisement_coordinate_problem(ceres::Problem&,const ObjectiveContext&,int,std::vector<std::size_t>*);
 eventio::Json linearized_actual_step_diagnostic(const StateSnapshot&candidate,const ObjectiveContext&context);
 bool initialization_geometry_guidance()const{return bool(hybrid_formal_hook_v1_)||phase_one_geometry_guidance_;}
 BuildInventory build_engineering_problem(ceres::Problem&,const eventio::Json&,eventio::Json*);
 BuildInventory build_engineering_problem(ceres::Problem&,const eventio::Json&,eventio::Json*,const std::vector<double>& physical_lambda,const std::vector<double>& equality_nu,const std::function<void()>&configure={});
 // Original physical/native/zoom g and RS/Query h, excluding pixel/budget g.
 ConstraintValues evaluate_current_engineering_constraints(int threads=1)const;
 eventio::Json restore_native_repair_extras_v1(const StateSnapshot&);
 eventio::Json select_representative_sources_v1(const eventio::Json& cfg);
 void set_engineering_identity(const std::string& configuration_sha);
 void attach_visible_ice_source_support(std::shared_ptr<const JointVisibleIceSourceSupport>);
 std::shared_ptr<const JointVisibleIceSourceSupport> visible_ice_source_support()const{return visible_ice_source_support_;}
 eventio::Json visible_ice_source_support_report()const;
 void attach_source_query_branch_registry(std::shared_ptr<const JointSourceQueryBranchRegistry>);
 std::shared_ptr<const JointSourceQueryBranchRegistry> source_query_branch_registry()const{return source_query_branch_registry_;}
 void assert_source_math_dependencies()const{if(visible_ice_source_support_)visible_ice_source_support_->assert_dependencies();if(source_query_branch_registry_)source_query_branch_registry_->assert_dependencies();}
 void export_explicit_nuisance(const std::filesystem::path&)const;
 void refresh_physical_domain_specs(bool rebuild_schema=true);
 eventio::Json attach_confirmed_tripod_outside_rink(const eventio::Json&);
 void enable_source_uv_observed_row_initialization();std::size_t refresh_observed_query_sources();void refresh_zoom_control_bindings();bool source_uv_observed_row_initialization()const{return source_uv_observed_row_initialization_;}
 void transition_observed_to_true_RS_v3(){if(!source_uv_observed_row_initialization_||gpu_uv_)throw std::runtime_error("trueRS transition requires restored observed primal before GPU");ensure_observed_catalogue();source_uv_observed_row_initialization_=false;observed_query_sources_.clear();enable_source_uv_capture_v3();}
 void enable_source_uv_capture_v1();void enable_source_uv_capture_v2();void enable_source_uv_capture_v3();void enable_source_uv_capture_v4();bool source_uv_capture_v4()const{return source_uv_capture_cells_v4_;}bool source_uv_capture_v3()const{return source_uv_capture_strict_v3_;}bool source_uv_capture_v1()const{return source_uv_capture_v1_;}bool source_uv_capture_v2()const{return source_uv_capture_sensor_station_v2_;}
 std::string RS_state_policy()const{return source_uv_observed_row_initialization_?"PROVISIONAL sourceUV measured-row initialization: source row fixed, finite station, true RS unverified; fresh v3 revalidation required":source_uv_capture_cells_v4_?"sourceUV_trueRS_capture_v4: strict sensor bracket; certified zoom hard domain; scene follows checked runtime stage; derived RS row; polyline station finite-cell active set with temporary vertex KKT; boundary-only pointwise merit rebind; original Query not imposed":source_uv_capture_strict_v3_?"sourceUV_trueRS_capture_v3: strict sensor bracket; certified zoom hard domain; scene follows checked runtime stage; derived station/row; original Query not imposed":source_uv_capture_sensor_station_v2_?"sourceUV_trueRS_capture_v2: sensor-active station/row derived; alpha retained; original Query not imposed":source_uv_capture_v1_?"sourceUV_trueRS_capture_v1: row is derived RS root; alpha retained, not optimized; original Query not imposed":"explicit bounded station/row/alpha; exact original RS and finite frozen query equalities; independent true RS on export";}
 void enable_gpu_raw_uv(const std::string&ptx,int threads,const std::filesystem::path&log_file);
 void enable_gpu_raw_uv(const std::string&ptx,int threads,const std::filesystem::path&log_file,const std::string&kernel_abi);
 void enable_gpu_raw_uv(const std::string&ptx,int threads,const std::filesystem::path&log_file,const std::string&kernel_abi,bool per_shot_batch_reuse);ceres::EvaluationCallback*gpu_evaluation_callback()const{return hybrid_formal_hook_v1_?hybrid_formal_hook_v1_->evaluation_callback():gpu_uv_.get();}eventio::Json gpu_report()const{return gpu_uv_?gpu_uv_->report():eventio::Json::object();}

 // Versioned production dispatch; native methods remain the sole original math.
 void install_formal_hybrid_hook_v1(std::shared_ptr<JointFormalHybridHookV1>h){if(!h||hybrid_formal_hook_v1_)throw std::runtime_error("null/duplicate hybrid production hook");hybrid_formal_hook_v1_=std::move(h);}
 bool has_formal_hybrid_hook_v1()const{return bool(hybrid_formal_hook_v1_);}
 const FrozenIdentity&native_identity_v1()const{return identity_;}const ConstraintSchema&native_schema_v1()const{ensure_observed_catalogue();return schema_;}
 ceres::EvaluationCallback*native_gpu_evaluation_callback_v1()const{return gpu_uv_.get();}
 std::vector<ParameterBlockBinding>native_parameter_blocks_v1();StateSnapshot native_capture_state_v1()const;void native_restore_state_v1(const StateSnapshot&);
 NativeReplay native_replay_native_v1(const StateSnapshot&)const;ConstraintValues native_evaluate_constraints_v1(const StateSnapshot&)const;GradientReport native_lagrangian_gradient_v1(const StateSnapshot&,Phase,const std::vector<double>&,const std::vector<double>&)const;
 BuildInventory native_build_ceres_problem_v1(ceres::Problem&,const ObjectiveContext&);
 using HybridNativeFactorVisitorV1=std::function<void(std::unique_ptr<ceres::CostFunction>,std::vector<double*>,double)>;
 void visit_hybrid_native_Lagrangian_constraints_v1(const std::vector<double>&,const std::vector<double>&,const HybridNativeFactorVisitorV1&);
 GradientReport project_hybrid_raw_gradient_v1(const std::map<double*,std::vector<double>>&,Phase,const std::string&);
 private:
 struct Factor{std::unique_ptr<ceres::CostFunction>cost;std::vector<std::size_t>ids;};
 std::shared_ptr<const JointVisibleIceSourceSupport>visible_ice_source_support_;std::shared_ptr<const JointSourceQueryBranchRegistry>source_query_branch_registry_;
 std::size_t visible_ice_source_inequality_count()const{return visible_ice_source_support_?visible_ice_source_support_->inequality_count():0;}
 Factor visible_ice_source_factor(std::size_t)const;
 JointSourceUVCaptureRowV1 capture_row_v1(std::size_t,std::vector<std::size_t>*)const;
 bool source_uv_observed_row_initialization_=false;std::map<std::size_t,std::shared_ptr<JointObservation>>observed_query_sources_;
 bool source_uv_capture_v1_=false,source_uv_capture_sensor_station_v2_=false,source_uv_capture_strict_v3_=false,source_uv_capture_cells_v4_=false;
 std::shared_ptr<JointUVCache>gpu_uv_;FrozenIdentity identity_;mutable std::vector<ObservationSpec>observations_;mutable ConstraintSchema schema_;mutable bool observed_catalogue_compacted_=false;void ensure_observed_catalogue()const;void compact_observed_catalogue();std::vector<std::size_t>active_,station_rows_,profile_rows_;std::vector<EventPhysicalConstraint>physical_;struct Domain{std::size_t row;int kind;};std::vector<Domain>native_domains_,equalities_;struct HardEquality{std::string id;std::vector<std::size_t>ids;JointPhysicalEquality functor;};std::vector<HardEquality>hard_equalities_;struct ZoomBound{std::size_t rig;int control,offset,upper;std::string id;};std::vector<ZoomBound>zoom_bounds_;std::vector<std::array<double,2>>zoom_enclosures_,zoom_box_;EventModel domain_math_;double budget_=1;bool allow_initialization_surrogate_=false,phase_one_geometry_guidance_=false;
 std::size_t station_id(std::size_t i)const{return core.model.blocks.size()+i;}std::size_t row_id(std::size_t i)const{return core.model.blocks.size()+core.packet.rows.size()+i;}std::size_t alpha_id(std::size_t i)const{return core.model.blocks.size()+2*core.packet.rows.size()+i;}std::size_t budget_id()const{return core.model.blocks.size()+3*core.packet.rows.size();}
 std::size_t zoom_id(std::size_t rig)const{return budget_id()+1+rig;}
 double*pointer(std::size_t);std::vector<double*>pointers(const Factor&);std::vector<int>sizes(const std::vector<std::size_t>&)const;std::size_t equality_count()const{return equalities_.size()+hard_equalities_.size();}Factor pixel(std::size_t)const;Factor physical(std::size_t)const;Factor native_domain(std::size_t)const;Factor zoom_bound(std::size_t)const;Factor zoom_certificate(std::size_t)const;Factor equality(std::size_t)const;Factor statistic(const JointStatisticalGroup&)const;JointExplicitRow row(std::size_t,std::vector<std::size_t>*)const;std::unique_ptr<JointALBuilder>frozen(const StateSnapshot&)const;
 std::shared_ptr<JointFormalHybridHookV1>hybrid_formal_hook_v1_;
};
}
