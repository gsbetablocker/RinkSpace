#pragma once
#include "rinkfit/joint_physical_feasibility_v1.hpp"
#include "rinkfit/joint_uv_cost.hpp"
#include <Eigen/Eigenvalues>
#include <unordered_map>
#include <unordered_set>
namespace rinkfit {
// A reversible numerical gauge, never a measured shape or new prior.
// Unknown chart type deliberately keeps GPU ambient null derivatives available.
class JointShapeSupportChartV1 final:public ceres::Manifold {
 Eigen::Vector2d direction_=Eigen::Vector2d::UnitX();
 public:
 void direction(const Eigen::Vector2d&v){if(!v.allFinite()||!(v.norm()>0))throw std::runtime_error("invalid numerical shape direction");direction_=v.normalized();}
 int AmbientSize()const override{return 2;}int TangentSize()const override{return 1;}
 bool Plus(const double*x,const double*d,double*y)const override{for(int a=0;a<2;++a){y[a]=x[a]+direction_[a]*d[0];if(!std::isfinite(y[a]))return false;}return true;}
 bool PlusJacobian(const double*,double*j)const override{j[0]=direction_[0];j[1]=direction_[1];return true;}
 bool Minus(const double*y,const double*x,double*d)const override{d[0]=direction_[0]*(y[0]-x[0])+direction_[1]*(y[1]-x[1]);return std::isfinite(d[0]);}
 bool MinusJacobian(const double*,double*j)const override{j[0]=direction_[0];j[1]=direction_[1];return true;}
};
struct JointShapeConditionalSupportV1 {
 Eigen::Matrix2d H=Eigen::Matrix2d::Zero();Eigen::Vector2d gradient=Eigen::Vector2d::Zero(),absolute_gradient=Eigen::Vector2d::Zero();
 bool protected_direction=false;
 // J, js, e contain actual sqrt(W)/E. Only a truly free, strictly interior
 // scalar station has a two-sided tangent that may be eliminated.
 void add(const Eigen::Matrix2d&J,const Eigen::Vector2d&js,const Eigen::Vector2d&e,int residuals,bool eliminate_station){
  gradient+=J.transpose()*e;absolute_gradient+=(J.cwiseAbs().transpose()*e.cwiseAbs()).eval();
  if(eliminate_station&&js.squaredNorm()>0.){
   if(residuals==2){const Eigen::Vector2d normal(-js[1],js[0]);const Eigen::RowVector2d q=normal.normalized().transpose()*J;H.noalias()+=q.transpose()*q;}
  }else H.noalias()+=J.transpose()*J;
 }
 struct Decision{int rank=2;Eigen::Vector2d direction=Eigen::Vector2d::UnitX();double largest=0,smallest=0;};
 Decision decision(const Eigen::Vector2d&units)const{
  Decision out;if(protected_direction||!H.allFinite()||!gradient.allFinite()||!units.allFinite()||(units.array()<=0).any())return out;
  const Eigen::Matrix2d scaled=units.asDiagonal()*H*units.asDiagonal();if(scaled.isZero(0.)){out.rank=0;return out;}
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d>e(scaled);if(e.info()!=Eigen::Success||!e.eigenvalues().allFinite())return out;
  out.smallest=e.eigenvalues()[0];out.largest=e.eigenvalues()[1];
  // Relative roundoff recognition, not an arbitrary small-support cutoff.
  if(out.largest>0&&std::abs(out.smallest)<=1024.*std::numeric_limits<double>::epsilon()*out.largest){out.rank=1;out.direction=(units.asDiagonal()*e.eigenvectors().col(1)).normalized();}
  return out;
 }
 bool excluded_gradient_nonzero(const Eigen::Vector2d&direction,int rank)const{
  if(rank>=2)return false;Eigen::Vector2d excluded;if(rank==0)excluded=Eigen::Vector2d::UnitX();else excluded=Eigen::Vector2d(-direction[1],direction[0]);
  const auto nonzero=[&](const Eigen::Vector2d&v){return std::abs(v.dot(gradient))>1024.*std::numeric_limits<double>::epsilon()*v.cwiseAbs().dot(absolute_gradient);};
  return nonzero(excluded)||(rank==0&&nonzero(Eigen::Vector2d::UnitY()));
 }
};
class JointShapeSupportWorkingSetV1 {
 using Json=eventio::Json;
 struct Row{std::size_t raw,pixel;double scale;};
 struct Vertex{std::size_t block;double*x;Eigen::Vector2d units;std::vector<std::size_t>physical_terms;JointShapeConditionalSupportV1 support;JointShapeSupportChartV1*chart=nullptr;int last_rank=2;Eigen::Vector2d last_direction=Eigen::Vector2d::UnitX();bool release_until_path_change=false;};
 struct Installed{std::size_t entry;const ceres::Manifold*original;bool constant;};
 std::vector<Row>rows_;std::vector<Vertex>vertices_;std::unordered_map<std::size_t,std::size_t>by_block_;std::vector<Installed>installed_;std::vector<signed char>physical_active_;
 const BuildInventory*inventory_;std::shared_ptr<JointPhysicalFeasibilityV1>physical_;std::uint64_t generation_=0;bool have_prepared_=false;
 static std::string schema(){return "rinkfit.conditional_print_shape_numerical_working_set.v1";}
 // Unlike the ordinary cost chain, a support proof cannot interpret a pruned
 // or absent ambient column as a measured zero. Missing J retains full 2D.
 static bool ambient_partial(const JointUVCacheView&view,int column,JointUVPartial*out){
  if(column<0||!view.partials||!view.jacobians_ready)return false;int offset=-1;
  if(view.sparse_partial_columns){const auto&v=*view.sparse_partial_columns;auto it=std::lower_bound(v.begin(),v.end(),column,[](const auto&a,int c){return a.first<c;});if(it==v.end()||it->first!=column)return false;offset=it->second;}
  else if(view.partial_columns){if(std::size_t(column)>=view.partial_columns->size())return false;offset=(*view.partial_columns)[column];}
  else return false;
  if(offset<0)return false;*out=view.partials[offset];return out->arithmetic&&std::isfinite(out->u)&&std::isfinite(out->v);
 }
 template<class Builder>void calculate(Builder&b,ceres::Problem&p){
  if(b.core.packet.numerical_path_generation_v1!=generation_){generation_=b.core.packet.numerical_path_generation_v1;for(auto&v:vertices_)v.release_until_path_change=false;}
  for(auto&v:vertices_)v.support=JointShapeConditionalSupportV1{};
  auto*cache=dynamic_cast<JointUVCache*>(b.gpu_evaluation_callback());if(cache){cache->configure_for_problem(p);cache->PrepareForEvaluation(true,false);}
  for(const auto&ref:rows_){
   const auto&source=b.core.packet.rows.at(ref.raw);const auto&path=b.core.model.materials.at(source.material).paths.at(source.path);
   auto protect_path=[&](){for(int block:path.vertex_blocks){auto it=by_block_.find(std::size_t(block));if(it!=by_block_.end())vertices_[it->second].support.protected_direction=true;}};
   const auto*cost=dynamic_cast<const JointEngineeringUVChainCost*>(p.GetCostFunctionForResidualBlock(static_cast<ceres::ResidualBlockId>(inventory_->statistical_blocks.at(ref.pixel))));
   if(!cost){protect_path();continue;}const auto&row=cost->native_domain_row();JointUVCacheView view;
   if(!row.uv_cache||!row.uv_cache->lookup(row.cache_row,&view)||!view.jacobians_ready||!view.partials||!view.block_column_starts){protect_path();continue;}
   const auto&o=*row.source;Eigen::Vector2d residual(view.canonical_AD_uv[0]-o.uv[0],view.canonical_AD_uv[1]-o.uv[1]);residual*=ref.scale;
   auto*s=&b.core.station.at(ref.raw);const bool interior=row.station_slot>=0&&p.HasParameterBlock(s)&&!p.IsParameterBlockConstant(s)&&p.ParameterBlockTangentSize(s)==1&&*s>p.GetParameterLowerBound(s,0)&&*s<p.GetParameterUpperBound(s,0);
   Eigen::Vector2d station=Eigen::Vector2d::Zero();if(interior){JointUVPartial d;if(!ambient_partial(view,view.station_column,&d)){protect_path();continue;}station={ref.scale*d.u,ref.scale*d.v};}
   bool native_protected=false;for(int k=0;k<5;++k)native_protected|=!view.native_domain_available[k]||!std::isfinite(view.canonical_AD_native_domains[k])||view.canonical_AD_native_domains[k]>=0.;
   const int residual_count=o.query?1:2;if(o.query){const Eigen::Vector2d n(o.normal[0],o.normal[1]);const double re=n.dot(residual),st=n.dot(station);residual={re,0.};station={st,0.};}
   // shape_vertex_count is the WHOLE contour's cardinality. The live row
   // contains exactly the two slots below, even on a thousand-vertex contour.
   if(row.physical_slots.size()!=row.layout.block_ids.size()){protect_path();continue;}
   for(int factor_slot:row.layout.shape_vertex_slots){
    const auto physical=std::find(row.physical_slots.begin(),row.physical_slots.end(),factor_slot);
    if(factor_slot<0||physical==row.physical_slots.end()){protect_path();continue;}
    const auto slot=std::size_t(physical-row.physical_slots.begin());
    if(slot>=row.layout.block_ids.size()||slot>=view.block_column_starts->size()){protect_path();continue;}
    const auto block=row.layout.block_ids[slot];auto it=by_block_.find(block);if(it==by_block_.end())continue;auto&v=vertices_[it->second];
    if(!p.HasParameterBlock(v.x)||p.IsParameterBlockConstant(v.x)||p.GetManifold(v.x))continue;
    if(native_protected||!residual.allFinite()||!station.allFinite()){v.support.protected_direction=true;continue;}
    const int begin=(*view.block_column_starts)[slot];Eigen::Matrix2d J;bool complete=begin>=0&&begin<std::numeric_limits<int>::max();
    for(int c=0;complete&&c<2;++c){JointUVPartial d;complete=ambient_partial(view,begin+c,&d);if(complete){J(0,c)=ref.scale*d.u;J(1,c)=ref.scale*d.v;}}
    if(!complete){v.support.protected_direction=true;continue;}
    if(o.query){const Eigen::RowVector2d first=o.normal[0]*J.row(0)+o.normal[1]*J.row(1);J.row(0)=first;J.row(1).setZero();}
    if(!J.allFinite()){v.support.protected_direction=true;continue;}v.support.add(J,station,residual,residual_count,interior);
   }
  }
  if(physical_){std::fill(physical_active_.begin(),physical_active_.end(),static_cast<signed char>(-1));for(auto&v:vertices_)if(!v.support.protected_direction)for(auto index:v.physical_terms){auto&active=physical_active_.at(index);if(active<0)active=physical_->original_term_blocks_numerical_gauge_v1(index)?1:0;if(active){v.support.protected_direction=true;break;}}}
  else for(auto&v:vertices_)v.support.protected_direction=true;
 }
 public:
 template<class Builder>JointShapeSupportWorkingSetV1(Builder&b,const BuildInventory&inventory,const Json&ledger,std::shared_ptr<JointPhysicalFeasibilityV1>physical):inventory_(&inventory),physical_(std::move(physical)),generation_(b.core.packet.numerical_path_generation_v1){
  std::unordered_set<int>blocks;for(const auto&m:b.core.model.materials)for(const auto&p:m.paths)for(int block:p.vertex_blocks)blocks.insert(block);std::vector<int>ordered(blocks.begin(),blocks.end());std::sort(ordered.begin(),ordered.end());vertices_.reserve(ordered.size());
  for(int block:ordered){auto&v=b.core.model.blocks.at(block);if(v.values.size()!=2||v.scales.size()!=2)throw std::runtime_error("split shape support needs actual two model units");by_block_.emplace(block,vertices_.size());vertices_.push_back({std::size_t(block),v.values.data(),Eigen::Vector2d(v.scales[0],v.scales[1])});}
  std::size_t pixel=0;for(const auto&g:ledger.at("groups")){if(!g.contains("actual_row_indices")||!g.contains("actual_per_row_weight"))throw std::runtime_error("conditional shape support needs consumed per-raw W ledger");const auto&indices=g.at("actual_row_indices");const auto&weights=g.at("actual_per_row_weight");if(indices.size()!=weights.size())throw std::runtime_error("conditional shape W row map differs");for(std::size_t j=0;j<indices.size();++j,++pixel){const std::size_t raw=indices[j].get<std::size_t>();const auto&r=b.core.packet.rows.at(raw);if(b.core.model.materials.at(r.material).paths.at(r.path).vertex_blocks.empty())continue;const double weight=weights[j].get<double>(),scale=std::sqrt(weight)/r.E;if(!r.active||!std::isfinite(scale)||scale<0)throw std::runtime_error("conditional shape actual W/E invalid");rows_.push_back({raw,pixel,scale});}}
  if(pixel>inventory.statistical_blocks.size())throw std::runtime_error("conditional shape ledger exceeds graph");
  if(physical_){std::unordered_map<const double*,std::size_t>pointers;for(std::size_t i=0;i<vertices_.size();++i)pointers.emplace(vertices_[i].x,i);physical_->visit_original_terms_v1([&](std::size_t ordinal,const std::vector<double*>&parameters){if(physical_active_.size()<=ordinal)physical_active_.resize(ordinal+1,-1);for(auto*x:parameters){auto it=pointers.find(x);if(it!=pointers.end())vertices_[it->second].physical_terms.push_back(ordinal);}});}
 }
 class ScopeRestoreV1{JointShapeSupportWorkingSetV1*owner_;ceres::Problem*p_;public:ScopeRestoreV1(JointShapeSupportWorkingSetV1*owner,ceres::Problem*p):owner_(owner),p_(p){}~ScopeRestoreV1(){if(owner_)owner_->restore(*p_);}void release()noexcept{owner_=nullptr;}};
 void restore(ceres::Problem&p){for(const auto&held:installed_){auto&v=vertices_.at(held.entry);if(held.constant)p.SetParameterBlockVariable(v.x);else p.SetManifold(v.x,const_cast<ceres::Manifold*>(held.original));}installed_.clear();}
 template<class Builder>Json prepare(Builder&b,ceres::Problem&p){
  if(!installed_.empty())throw std::runtime_error("shape support preparation retained an old temporary chart");
  calculate(b,p);Json local=Json::array();std::size_t rank0=0,rank1=0,physical_protected=0,original_fixed=0;
  for(auto&v:vertices_){v.last_rank=2;if(!p.HasParameterBlock(v.x)||p.IsParameterBlockConstant(v.x)||p.GetManifold(v.x)){++original_fixed;continue;}bool finite_box=false;for(int c=0;c<2;++c)finite_box|=p.GetParameterLowerBound(v.x,c)>-std::numeric_limits<double>::max()||p.GetParameterUpperBound(v.x,c)<std::numeric_limits<double>::max();if(finite_box){++original_fixed;continue;}if(v.support.protected_direction)++physical_protected;const auto d=v.support.decision(v.units);rank0+=d.rank==0;rank1+=d.rank==1;if(d.rank<2)local.push_back({{"block",v.block},{"conditional_rank",d.rank},{"direction",d.rank==1?Json::array({d.direction[0],d.direction[1]}):Json(nullptr)}});}
  // Eliminating an interior station proves only a null direction of this
  // local linearization. Its basis and rank change when cameras, neighboring
  // vertices or station bounds move. It is not an exact parameter symmetry,
  // so freezing a vertex or dropping a tangent coordinate changes later GN
  // subproblems. Existing positive LM damping handles the rank deficiency;
  // keep the original physical variables free throughout the same Solve.
  have_prepared_=true;return {{"schema",schema()},{"actual_print_rows",rows_.size()},{"rank0_numerical_gauges",0},{"rank1_numerical_gauges",0},{"conditional_rank0_vertices",rank0},{"conditional_rank1_vertices",rank1},{"conditional_support_vertices",std::move(local)},{"restricted_vertices",Json::array()},{"released_vertices",Json::array()},{"physical_or_native_domain_protected_vertices",physical_protected},{"original_fixed_chart_or_finite_box_preserved",original_fixed},{"conditional_rank_is_only_data_support_upper_bound",true},{"conditional_support_is_local_linear_information",true},{"hard_dimension_reduction_installed",false},{"full_original_vertex_freedoms_retained",true},{"existing_model_units_LM_and_Schur_retained",true},{"legal_interior_station_eliminated_for_diagnostics_only",true},{"real_per_raw_W_and_E_consumed",true},{"all_original_pixel_and_physical_factors_retained",true},{"optimizer_history_in_complete_checkpoint",false}};
 }
 template<class Builder>Json reconsider(Builder&,ceres::Problem&){
  if(!installed_.empty())throw std::runtime_error("shape reconsider retained an old temporary chart");
  return {{"schema",schema()},{"restart_required",false},{"released_vertices",Json::array()},{"all_original_vertex_freedoms_already_retained",true},{"local_linear_rank_does_not_authorize_a_hard_gauge",true}};
 }

};
}
