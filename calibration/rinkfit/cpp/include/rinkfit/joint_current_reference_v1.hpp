#pragma once
#include "rinkfit/joint_al.hpp"
#include <fstream>
#include <unordered_set>
#include <type_traits>

namespace rinkfit {
struct JointCurrentReferenceHashV1 {eventio::FastSHA256Stream hash;void update(const void* data,std::size_t size){hash.update(static_cast<const std::uint8_t*>(data),size);}std::string finish(){return hash.finish();}};
// Explicit acceptance-only probe. Never called by normal startup or resume.
// It evaluates the already constructed graph; it does not load provenance,
// recreate qualification, mutate parameters, or decide solution quality.
inline eventio::Json run_joint_current_reference_v1(JointALBuilder& b,ceres::Problem& problem,const BuildInventory& inv,const eventio::Json& cfg,const std::filesystem::path& out,int threads){
 using Json=eventio::Json;const auto dir=out/"current_reference";std::filesystem::create_directories(dir);
 std::vector<ceres::ResidualBlockId> order;std::unordered_set<void*> seen;Json groups=Json::array();
 auto append=[&](const std::string& name,const auto& ids){const auto first=order.size();for(auto id:ids)if(seen.insert(static_cast<void*>(id)).second)order.push_back(static_cast<ceres::ResidualBlockId>(id));groups.push_back({{"name",name},{"first_block",first},{"blocks",order.size()-first}});};
 append("statistical",inv.statistical_blocks);append("measured_prior",inv.measured_prior_blocks);append("smoothness",inv.smoothness_blocks);append("inequality",inv.inequality_blocks);append("equality",inv.equality_blocks);
 for(const auto& item:b.engineering_residual_groups)append(item.first,item.second);
 if(order.size()!=static_cast<std::size_t>(problem.NumResidualBlocks()))throw std::runtime_error("explicit current reference: actual residual inventory is incomplete");
 std::size_t scalar_count=0;JointCurrentReferenceHashV1 shape_hash;for(auto id:order){const auto n=problem.GetCostFunctionForResidualBlock(id)->num_residuals();scalar_count+=static_cast<std::size_t>(n);shape_hash.update(&n,sizeof(n));}
 ceres::Problem::EvaluateOptions options;options.num_threads=threads;options.apply_loss_function=true;options.residual_blocks=order;
 double cost=0;std::vector<double> residuals;if(!problem.Evaluate(options,&cost,&residuals,nullptr,nullptr)||!std::isfinite(cost)||residuals.size()!=scalar_count)throw std::runtime_error("explicit current reference: actual target evaluation failed");
 auto write=[&](const char* name,const std::vector<double>& values){JointCurrentReferenceHashV1 hash;double maximum=0;for(double x:values){if(!std::isfinite(x))throw std::runtime_error(std::string("explicit current reference: nonfinite ")+name);maximum=std::max(maximum,std::abs(x));}const auto bytes=values.size()*sizeof(double);hash.update(values.data(),bytes);std::ofstream f(dir/name,std::ios::binary);if(!f)throw std::runtime_error("cannot write explicit reference "+(dir/name).string());f.write(reinterpret_cast<const char*>(values.data()),static_cast<std::streamsize>(bytes));f.close();if(!f)throw std::runtime_error("short explicit reference write "+(dir/name).string());return Json{{"path",name},{"dtype","little-endian float64"},{"count",values.size()},{"sha256",hash.finish()},{"max_abs",maximum}};};
 const auto residual_spec=write("target_residuals.f64",residuals);residuals.clear();residuals.shrink_to_fit();
 const auto values=b.evaluate_current_engineering_constraints(threads);const auto g_spec=write("constraint_g.f64",values.g),h_spec=write("constraint_h.f64",values.h);
 Json global=Json::array();b.visit_engineering_global_constraints_v1([&](const std::string& id,bool equality,std::unique_ptr<ceres::CostFunction> factor,std::vector<double*> parameters){std::vector<const double*> p(parameters.begin(),parameters.end());double value=0;if(!factor->Evaluate(p.data(),&value,nullptr)||!std::isfinite(value))throw std::runtime_error("explicit current reference: global constraint failed "+id);global.push_back({{"id",id},{"equality",equality},{"value",value}});});
 JointCurrentReferenceHashV1 numeric;std::size_t blocks=0,scalars=0;for(const auto& p:b.parameter_blocks()){const std::uint64_t n=p.name.size(),s=p.size;numeric.update(&n,sizeof(n));numeric.update(p.name.data(),p.name.size());numeric.update(&s,sizeof(s));numeric.update(p.data,p.size*sizeof(double));++blocks;scalars+=p.size;}
 auto array_hash=[](const auto& data){JointCurrentReferenceHashV1 hash;hash.update(data.data(),data.size()*sizeof(typename std::decay_t<decltype(data)>::value_type));return hash.finish();};
 JointCurrentReferenceHashV1 paths,fit;std::size_t fit_count=0;for(std::size_t i=0;i<b.core.packet.rows.size();++i){const auto& row=b.core.packet.rows[i];const std::int32_t path=row.path;paths.update(&path,sizeof(path));if(row.active){const std::uint64_t index=i;fit.update(&index,sizeof(index));++fit_count;}}
 Json report={{"schema","rinkfit.explicit_current_reference.v1"},{"objective_id",b.identity().objective_id},{"raw_rows",b.core.packet.rows.size()},{"active_rows",fit_count},{"active_raw_ids_sha256",fit.finish()},{"actual_cost",cost},{"actual_residual_blocks",order.size()},{"ordered_block_sizes_sha256",shape_hash.finish()},{"residual_order",groups},{"target_residuals",residual_spec},{"constraint_g",g_spec},{"constraint_h",h_spec},{"global_constraints",global},{"numerical_parameter_blocks",blocks},{"numerical_parameter_scalars",scalars},{"numerical_parameters_sha256",numeric.finish()},{"all_raw_station_sha256",array_hash(b.core.station)},{"all_raw_row_sha256",array_hash(b.core.row)},{"all_raw_alpha_sha256",array_hash(b.core.alpha)},{"all_raw_path_i32_sha256",paths.finish()},{"global_phr_state",b.engineering_global_phr_state_v1.json()},{"complete_state_token",values.state_token},{"mutated_state",false},{"quality_accepted",false},{"explicit_acceptance_only",true}};
 eventio::atomic_json(dir/"reference.json",report);return report;
}
}
