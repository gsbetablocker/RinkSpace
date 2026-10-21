#pragma once
#include "rinkfit/joint_al.hpp"
#include "rinkfit/joint_append_capture_seed.hpp"
#include "rinkfit/joint_capture.hpp"
#include <tuple>
namespace rinkfit {
namespace recovery_source_capture_detail {
inline std::set<std::string> fixed_shots(const eventio::Json&cfg,const JointEventModel&m){std::set<std::string>result;for(const auto&sid:cfg.value("temporary_runtime_fixed_nuisance_shots",std::vector<std::string>{})){if(!m.shot_map.count(sid))throw std::runtime_error("unknown recovery protected nuisance shot: "+sid);result.insert(sid);}return result;}
inline std::set<int> s01_dependencies(const JointProblemBuilder&b,std::size_t*covered=nullptr){std::set<int>blocks;std::size_t rows=0;for(const auto&r:b.packet.rows)if(r.active&&r.shot_id=="s01"){++rows;const auto l=r.layout(b.model);blocks.insert(l.block_ids.begin(),l.block_ids.end());}if(covered)*covered=rows;return blocks;}
}

// Runs before JointALBuilder constructs its domain/parameter registries.
// No accepted checkpoint, source qualification or strict Query pass is issued.
inline eventio::Json initialize_joint_recovery_source_capture(JointEventModel&model,JointPacket&packet,const eventio::Json&cfg,const std::filesystem::path&model_file,const std::filesystem::path&packet_file,const std::filesystem::path&out,const std::filesystem::path&cfg_root){
 if(!cfg.contains("source_capture_init"))return eventio::Json::object();
 const auto&spec=cfg.at("source_capture_init");
 if(spec.at("schema")!="rinkfit.engineering_append_capture_initialization.v1"||cfg.value("reinitialize_native_auxiliaries",true)||cfg.contains("resume_engineering_checkpoint")||cfg.contains("baseline_auxiliary_manifest")||cfg.contains("s01_component_rollback")||cfg.contains("source_capture_checkpoint")||cfg.contains("coarse_nuisance_seed"))throw std::runtime_error("explicit append source capture initialization cannot mix fresh-all, resume or baseline initialization");
 const auto fixed=recovery_source_capture_detail::fixed_shots(cfg,model);if(!fixed.count("s01"))throw std::runtime_error("append recovery initialization requires explicit S01 nuisance protection");
 JointProblemBuilder core(std::move(model),std::move(packet));core.fixed_nuisance_shots=fixed;
 const auto count=core.packet.rows.size(),start=core.packet.source_json.at("new_source_start").get<std::size_t>();const auto input_sha=core.packet.source_hash;
 std::vector<bool>original_constant;std::vector<std::vector<double>>original_physical;for(auto&block:core.model.blocks){original_constant.push_back(block.constant);original_physical.push_back(block.values);for(const auto&prefix:cfg.value("temporary_runtime_fixed_physical_prefixes",std::vector<std::string>{})){if(prefix.empty())throw std::runtime_error("empty recovery physical protection prefix");if(block.name.rfind(prefix,0)==0)block.constant=true;}}
 std::size_t covered=0;const auto dependencies=recovery_source_capture_detail::s01_dependencies(core,&covered);if(!covered)throw std::runtime_error("append recovery source capture omitted S01 active denominator");for(auto id:dependencies)if(!core.model.blocks.at(id).constant)throw std::runtime_error("append recovery initialization leaves a S01 physical dependency free: "+core.model.blocks.at(id).name);
 std::vector<double>station(core.station.begin(),core.station.begin()+start),row(core.row.begin(),core.row.begin()+start),alpha(core.alpha.begin(),core.alpha.begin()+start);std::vector<int>path;for(std::size_t i=0;i<start;++i){if(core.packet.rows[i].shot_id!="s01")throw std::runtime_error("append recovery original prefix is not S01");path.push_back(core.packet.rows[i].path);}
 const auto init_out=out/"source_capture_initialization";std::filesystem::create_directories(init_out);
 eventio::Json stage={{"source_capture_engineering",true},{"source_capture_feasibility",true},{"recapture_new_sources",true},{"observed_row_time_new_sources",false},{"append_capture_seed",spec.at("append_capture_seed")}};
 auto scope=install_joint_append_capture_seed(core,stage,model_file,packet_file,init_out,cfg_root);const auto requested=core.packet.source_json.at("recapture_rows").get<std::vector<std::size_t>>();eventio::Json capture;
 try{capture=recapture_joint_initial_nuisance(core,init_out,false);scope.restore(core);}catch(...){scope.restore(core);throw;}
 if(capture.at("requested_new_rows")!=requested.size()||capture.at("source_UV_changed")!=false||capture.at("material_identity_changed")!=false||capture.at("original_s01_nuisance_changed")!=false||capture.at("finite_Query_closure_required_for_success")!=false||core.packet.rows.size()!=count)throw std::runtime_error("append recovery capture changed source scope or claims strict Query closure");
 std::vector<std::size_t>RS_failed,navigation,invalid;std::set<std::size_t>seen;std::ifstream ledger(init_out/"initial_nuisance_recapture.jsonl",std::ios::binary);std::string line;while(std::getline(ledger,line)){if(line.empty())continue;const auto item=eventio::Json::parse(line);const auto i=item.at("joint_row").get<std::size_t>();if(!seen.insert(i).second)throw std::runtime_error("duplicate recovery append capture ledger row");if(!item.at("RS_valid").get<bool>())RS_failed.push_back(i);if(!item.at("numerical_initialization_seed_valid").get<bool>())invalid.push_back(i);else if(!item.at("RS_valid").get<bool>())navigation.push_back(i);}
 if(!ledger.eof()||seen!=std::set<std::size_t>(requested.begin(),requested.end())||capture.at("RS_valid_rows")!=requested.size()-RS_failed.size()||capture.at("invalid_rows")!=invalid.size())throw std::runtime_error("append recovery capture complete RS/navigation denominator differs");
 if(spec.contains("expected_true_RS_failed_rows")&&RS_failed!=spec.at("expected_true_RS_failed_rows").get<std::vector<std::size_t>>())throw std::runtime_error("append recovery actual RS failures differ; retained source ledger requires review");
 for(std::size_t i=0;i<start;++i){const auto&o=core.packet.rows[i];if(!coarse_seed_detail::same_double(station[i],core.station[i])||!coarse_seed_detail::same_double(row[i],core.row[i])||!coarse_seed_detail::same_double(alpha[i],core.alpha[i])||path[i]!=o.path)throw std::runtime_error("append recovery initialization changed S01 four-aux prefix");}
 for(std::size_t i=0;i<core.model.blocks.size();++i){if(core.model.blocks[i].values!=original_physical[i])throw std::runtime_error("append recovery initialization changed physical model values");core.model.blocks[i].constant=original_constant[i];}
 if(eventio::file_sha256(packet_file)!=input_sha)throw std::runtime_error("append recovery immutable source packet changed during initialization");
 eventio::Json receipt={{"schema","rinkfit.engineering_append_capture_initialization_receipt.v1"},{"passed",true},{"initialized_before_JointALBuilder_registration",true},{"raw_rows",count},{"source_packet_sha256",input_sha},{"requested_recapture_rows",requested.size()},{"true_RS_valid_rows",requested.size()-RS_failed.size()},{"true_RS_failed_row_indices",RS_failed},{"observed_row_navigation_only_indices",navigation},{"numerical_initialization_invalid_row_indices",invalid},{"failed_active_source_rows_retained",true},{"original_S01_four_aux_bitwise_equal",true},{"original_S01_raw_prefix_rows",start},{"S01_active_rows_covered",covered},{"S01_physical_dependency_blocks_checked",dependencies.size()},{"temporary_model_constant_flags_restored",true},{"model_values_and_source_UV_E_W_Query_unchanged",true},{"original_full_recapture_scope_restored",true},{"strict_Query_closure_claimed",false},{"formal_Phase_I_II_passed",false},{"quality_accepted",false},{"accepted_checkpoint_restored",false},{"initialization_navigation_only",true},{"remaining_failures_belong_to_original_PHR_constraints",true}};
 eventio::atomic_json(init_out/"receipt.json",receipt);model=std::move(core.model);packet=std::move(core.packet);return receipt;
}


// A new numerical initial state, never an edit of the packet or parent CP.
// Only the protected S01 sensor-row root is polished; all physical parameters,
// finite stations, paths, Query alpha and original observations remain fixed.
inline eventio::Json polish_joint_recovery_protected_RS_rows(JointALBuilder&b,const eventio::Json&cfg,const std::filesystem::path&out){
 if(!cfg.value("protected_RS_row_polish",false))return eventio::Json::object();
 if(!recovery_source_capture_detail::fixed_shots(cfg,b.core.model).count("s01"))throw std::runtime_error("protected RS row polish requires S01 nuisance protection");
 const auto target=out/"protected_RS_row_polish";std::filesystem::create_directories(target);
 const auto parent_token=b.capture_state().token();const auto physical_before=b.core.model.exact_json().dump();
 auto array_sha=[](const std::vector<double>&x){const auto*p=reinterpret_cast<const std::uint8_t*>(x.data());return eventio::sha256(std::vector<std::uint8_t>(p,p+x.size()*sizeof(double)));};
 const auto station_sha=array_sha(b.core.station),alpha_sha=array_sha(b.core.alpha),row_before_sha=array_sha(b.core.row);
 std::vector<double>proposal=b.core.row;std::vector<std::size_t>changed;
 std::ofstream ledger(target/"rows.jsonl",std::ios::binary);if(!ledger)throw std::runtime_error("cannot write protected RS polish ledger");
 std::size_t visited=0,queries=0,endpoint_roots=0;double before_h=0,after_h=0,row_delta=0,projection_delta=0,query_before=0,query_after=0;
 eventio::Json report={{"schema","rinkfit.protected_S01_RS_row_polish.v1"},{"parent_state_token",parent_token},{"source_packet_sha256",b.core.packet.source_hash},{"parent_row_sha256",row_before_sha},{"station_sha256",station_sha},{"alpha_sha256",alpha_sha},{"row_is_numerical_seed_not_source_UV",true},{"physical_station_path_alpha_source_UV_E_W_Query_unchanged",true},{"strict_RS_target_px",1e-10},{"Query_tolerance_px",1e-8},{"raw_rows",b.core.packet.rows.size()},{"rows_installed",false},{"passed",false},{"quality_accepted",false},{"whole_interval_RS_uniqueness_newly_claimed",false}};
 try{
  for(std::size_t i=0;i<b.core.packet.rows.size();++i){const auto&o=b.core.packet.rows[i];if(!o.active||o.shot_id!="s01")continue;++visited;
   const auto l=o.layout(b.core.model);const double st=o.kind==1?o.station:b.core.station[i],old=b.core.row[i];auto pointers=b.core.model.pointers(l);
   if(!std::isfinite(old)||old<0||old>1079||!o.station_allowed(b.core.model,o.path,st))throw std::runtime_error("protected RS polish initial finite/sensor domain failed at row "+std::to_string(i));
   std::vector<std::vector<ceres::Jet<double,1>>>storage;storage.reserve(l.block_ids.size());for(auto id:l.block_ids){storage.emplace_back();for(double v:b.core.model.blocks.at(id).values)storage.back().emplace_back(v);}
   std::vector<const ceres::Jet<double,1>*>jp;for(const auto&v:storage)jp.push_back(v.data());
   auto value=[&](double sensor){ceres::Jet<double,1>r(sensor);r.v[0]=1.;return b.core.model.predict_for_al(l,ceres::Jet<double,1>(st),r,jp.data());};
   auto guarded=[&](double sensor,const auto&p){if(!p.arithmetic_valid||p.physical_status!=DomainStatus::kValid||!std::isfinite(p.uv[0].a)||!std::isfinite(p.uv[1].a)||sensor<0||sensor>1079)return false;const auto vis=b.core.model.visibility(l,st,sensor,pointers.data());if(!vis.visible())return false;if(o.width_block>=0)for(int k:{5,6}){const auto g=b.core.model.measurement_domain(l,st,pointers.data(),k);if(!std::isfinite(g)||g>1e-8)return false;}return true;};
   double row=old;auto p=value(row);if(!guarded(row,p))throw std::runtime_error("protected RS polish initial physical domain failed at row "+std::to_string(i));
   const Vec2<double>old_uv={p.uv[0].a,p.uv[1].a};const double old_h=row-old_uv[1];before_h=std::max(before_h,std::abs(old_h));std::size_t updates=0;bool endpoint=false;
   while(std::abs(row-p.uv[1].a)>1e-10){const double h=row-p.uv[1].a,den=1-p.uv[1].v[0];if(!std::isfinite(den)||std::abs(den)<=1e-6)throw std::runtime_error("protected RS polish singular scalar root at row "+std::to_string(i));
    double next=row-h/den;auto q=p;
    if(next<0||next>1079){const double boundary=next<0?0.:1079.;q=value(boundary);if(!guarded(boundary,q)||std::abs(boundary-q.uv[1].a)>1e-10)throw std::runtime_error("protected RS polish has no valid sensor endpoint root at row "+std::to_string(i));next=boundary;endpoint=true;}
    else q=value(next);
    if(!guarded(next,q)||!std::isfinite(next)||next==row||std::abs(next-q.uv[1].a)>=std::abs(h))throw std::runtime_error("protected RS polish failed or stalled at row "+std::to_string(i));
    row=next;p=q;++updates;
   }
   const Vec2<double>uv={p.uv[0].a,p.uv[1].a};const double new_h=row-uv[1];after_h=std::max(after_h,std::abs(new_h));const double du=std::hypot(uv[0]-old_uv[0],uv[1]-old_uv[1]);projection_delta=std::max(projection_delta,du);row_delta=std::max(row_delta,std::abs(row-old));if(endpoint)++endpoint_roots;
   double qbefore=0,qafter=0;if(o.query){++queries;Vec2<double>q;if(!o.query->point(b.core.alpha[i],&q))throw std::runtime_error("protected RS polish frozen Query alpha invalid at row "+std::to_string(i));for(int k=0;k<2;++k){qbefore=std::max(qbefore,std::abs(old_uv[k]-q[k]));qafter=std::max(qafter,std::abs(uv[k]-q[k]));}query_before=std::max(query_before,qbefore);query_after=std::max(query_after,qafter);if(qafter>1e-8)throw std::runtime_error("protected RS polish would leave frozen Query above 1e-8 at row "+std::to_string(i));}
   proposal[i]=row;if(row!=old){changed.push_back(i);ledger<<eventio::Json({{"raw_row",i},{"native_pts",o.pts},{"path",o.path},{"station",st},{"alpha",b.core.alpha[i]},{"row_before",old},{"row_after",row},{"RS_h_before",old_h},{"RS_h_after",new_h},{"projection_delta_native_uv",{uv[0]-old_uv[0],uv[1]-old_uv[1]}},{"newton_updates",updates},{"independently_validated_sensor_endpoint",endpoint},{"Query_h_before",qbefore},{"Query_h_after",qafter},{"source_row_id",o.source_row_id}}).dump()<<'\n';}
  }
  if(!visited||b.core.model.exact_json().dump()!=physical_before||array_sha(b.core.station)!=station_sha||array_sha(b.core.alpha)!=alpha_sha)throw std::runtime_error("protected RS polish changed a protected physical/non-row state");
  ledger.close();if(!ledger)throw std::runtime_error("protected RS polish ledger write failed");
  for(auto i:changed){b.core.row[i]=proposal[i];b.core.packet.rows[i].row=proposal[i];}
  report["passed"]=true;report["rows_installed"]=true;report["new_state_token"]=b.capture_state().token();report["row_after_sha256"]=array_sha(b.core.row);
 }catch(const std::exception&e){ledger.close();report["failure"]=e.what();report["visited_S01_rows"]=visited;report["proposed_changed_rows"]=changed.size();report["max_RS_h_before_px"]=before_h;report["max_RS_h_after_px"]=after_h;report["max_Query_h_before_px"]=query_before;report["max_Query_h_after_px"]=query_after;eventio::atomic_json(target/"receipt.json",report);throw;}
 report["active_S01_rows"]=visited;report["changed_rows"]=changed.size();report["independently_validated_endpoint_roots"]=endpoint_roots;report["frozen_Query_rows"]=queries;report["max_RS_h_before_px"]=before_h;report["max_RS_h_after_px"]=after_h;report["max_Query_h_before_px"]=query_before;report["max_Query_h_after_px"]=query_after;report["max_row_change_px"]=row_delta;report["max_projection_change_native_px"]=projection_delta;report["ledger"]={{"file",(target/"rows.jsonl").string()},{"sha256",eventio::file_sha256(target/"rows.jsonl")}};
 eventio::atomic_json(target/"receipt.json",report);return report;
}

// Apply actual nuisance constants, then attest complete S01 projection closure.
// Every active source layout contributes its complete dependency block union.
inline eventio::Json protect_joint_recovery_s01(JointALBuilder&b,ceres::Problem&problem,const eventio::Json&cfg){
 const auto fixed=recovery_source_capture_detail::fixed_shots(cfg,b.core.model);std::size_t nuisance=0,protected_rows=0;for(std::size_t i=0;i<b.core.packet.rows.size();++i){const auto&r=b.core.packet.rows[i];if(!r.active||!fixed.count(r.shot_id))continue;++protected_rows;for(double*value:{&b.core.station[i],&b.core.row[i],&b.core.alpha[i]})if(problem.HasParameterBlock(value)){if(!problem.IsParameterBlockConstant(value))problem.SetParameterBlockConstant(value);++nuisance;}}
 eventio::Json receipt={{"temporary_fixed_nuisance_shots",fixed},{"protected_active_rows",protected_rows},{"actual_registered_nuisance_blocks_constant",nuisance},{"source_or_model_values_changed",false}};
 if(fixed.count("s01")){
  std::size_t covered=0;const auto blocks=recovery_source_capture_detail::s01_dependencies(b.core,&covered);if(!covered)throw std::runtime_error("recovery S01 projection protection has no active rows");for(auto id:blocks){auto*value=b.core.model.blocks.at(id).values.data();if(!problem.HasParameterBlock(value)||!problem.IsParameterBlockConstant(value))throw std::runtime_error("recovery S01 projection physical dependency is variable: "+b.core.model.blocks.at(id).name);}
  receipt["S01_active_rows_covered"]=covered;receipt["S01_unique_physical_blocks_checked"]=blocks.size();receipt["complete_S01_projection_dependencies_constant"]=true;
 }
 if(cfg.contains("source_capture_init")){
  if(!fixed.count("s01"))throw std::runtime_error("append recovery graph lost S01 nuisance protection");std::size_t latent=0;for(auto&block:b.core.model.blocks)if(block.name.rfind("texture_shared_X/",0)==0){if(!problem.HasParameterBlock(block.values.data())||problem.IsParameterBlockConstant(block.values.data())||problem.ParameterBlockSize(block.values.data())!=4||problem.ParameterBlockTangentSize(block.values.data())!=2||block.fixed_indices!=std::vector<int>{2,3})throw std::runtime_error("append recovery shared ice XY is not a free two-dimensional latent: "+block.name);++latent;}if(latent!=7)throw std::runtime_error("append recovery requires all seven shared ice XY latents");receipt["actual_free_shared_ice_XY_blocks"]=latent;
 }
 receipt["passed"]=true;return receipt;
}
} // namespace rinkfit
