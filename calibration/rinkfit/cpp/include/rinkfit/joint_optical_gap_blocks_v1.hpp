#pragma once
#include "rinkfit/joint_capture_certified_zoom_v1.hpp"
#include "rinkfit/joint_gpu_cache.hpp"
#include <chrono>
namespace rinkfit {
namespace optical_gap_detail {
using V2=std::array<double,2>;
struct HalfPlane{V2 a;double b;};
inline V2 solve2(const std::array<double,4>&H,const V2&g,const std::vector<HalfPlane>&box){
 V2 best{0.,0.};long double best_cost=0;
 auto consider=[&](V2 x){if(!std::isfinite(x[0])||!std::isfinite(x[1]))return;for(const auto&q:box){double a=q.a[0]*x[0]+q.a[1]*x[1];double tol=32*std::numeric_limits<double>::epsilon()*std::max({1.,std::abs(a),std::abs(q.b)});if(a>q.b+tol)return;}long double c=(long double)g[0]*x[0]+(long double)g[1]*x[1]+.5L*((long double)H[0]*x[0]*x[0]+2.L*H[1]*x[0]*x[1]+(long double)H[3]*x[1]*x[1]);if(c<best_cost){best_cost=c;best=x;}};
 long double det=(long double)H[0]*H[3]-(long double)H[1]*H[2];if(det>0)consider({double(((long double)H[1]*g[1]-(long double)H[3]*g[0])/det),double(((long double)H[2]*g[0]-(long double)H[0]*g[1])/det)});
 if(H[0]>0)consider({-g[0]/H[0],0.});if(H[3]>0)consider({0.,-g[1]/H[3]});
 for(std::size_t i=0;i<box.size();++i){const auto&q=box[i];double aa=q.a[0]*q.a[0]+q.a[1]*q.a[1];if(!(aa>0))continue;V2 p{q.b*q.a[0]/aa,q.b*q.a[1]/aa},v{-q.a[1],q.a[0]};double den=v[0]*(H[0]*v[0]+H[1]*v[1])+v[1]*(H[2]*v[0]+H[3]*v[1]);if(den>0){double t=-(v[0]*(g[0]+H[0]*p[0]+H[1]*p[1])+v[1]*(g[1]+H[2]*p[0]+H[3]*p[1]))/den;consider({p[0]+t*v[0],p[1]+t*v[1]});}consider(p);
  for(std::size_t j=0;j<i;++j){const auto&r=box[j];double d=q.a[0]*r.a[1]-q.a[1]*r.a[0];if(d!=0)consider({(q.b*r.a[1]-q.a[1]*r.b)/d,(q.a[0]*r.b-q.b*r.a[0])/d});}
 }return best;
}
}
struct JointOpticalGapResultV1{bool accepted=false,working_changed=false;};
inline JointOpticalGapResultV1 minimize_joint_optical_gaps_v1(JointALBuilder&b,ceres::Problem&p,const JointCaptureCertifiedDomainV1&domain,const std::filesystem::path&file,int threads){
 const auto started=std::chrono::steady_clock::now();using V2=optical_gap_detail::V2;
 struct Item{const JointRig*rig;int block;std::array<double,12>base;V2 gap;std::array<bool,2>active;};std::vector<Item>items;
 ceres::Problem::EvaluateOptions options;options.num_threads=threads;
 for(const auto&r:b.core.model.rigs){auto&v=b.core.model.blocks.at(r.optics_block);double*x=v.values.data();if(!p.HasParameterBlock(x)||p.IsParameterBlockConstant(x))continue;auto*old=dynamic_cast<const JointOpticalFeasibleCoordinates*>(p.GetManifold(x));if(!old)throw std::runtime_error("optical gap block requires the existing feasible lens coordinates");Item it;it.rig=&r;it.block=r.optics_block;it.active=old->active_endpoint_gaps();std::copy(x,x+12,it.base.begin());if(!old->endpoint_gaps(x,&it.gap))throw std::runtime_error("invalid current optical gaps");p.SetManifold(x,new JointOpticalFeasibleCoordinates(r,domain.ranges.at(r.id),v));options.parameter_blocks.push_back(x);items.push_back(it);}
 JointOpticalGapResultV1 result;if(items.empty())return result;
 if(auto*c=b.gpu_evaluation_callback())c->PrepareForEvaluation(true,true);
 double before=0;std::vector<double>gradient;ceres::CRSMatrix J;if(!p.Evaluate(options,&before,nullptr,&gradient,&J))throw std::runtime_error("optical gap live Jacobian evaluation failed");const int n=int(items.size())*2;std::vector<double>H(n*n,0.),g(n),delta(n,0.);for(int i=0;i<n;++i)g[i]=gradient.at((i/2)*12+7+i%2);
 std::vector<std::pair<int,double>>v;v.reserve(n);for(int row=0;row<J.num_rows;++row){v.clear();for(int k=J.rows[row];k<J.rows[row+1];++k){int col=J.cols[k],a=col%12;if(a==7||a==8)v.push_back({(col/12)*2+a-7,J.values[k]});}for(const auto&a:v)for(const auto&c:v)H[a.first*n+c.first]+=a.second*c.second;}
 J=ceres::CRSMatrix();gradient.clear();
 for(std::size_t i=0;i<items.size();++i){const auto&it=items[i];auto&v=b.core.model.blocks[it.block];auto*chart=dynamic_cast<const JointOpticalFeasibleCoordinates*>(p.GetManifold(v.values.data()));const auto A=chart->gap_coefficient_jacobian();const int o=int(i)*2;V2 gg{g[o],g[o+1]};for(int j=0;j<n;++j){gg[0]+=H[o*n+j]*delta[j];gg[1]+=H[(o+1)*n+j]*delta[j];}std::array<double,4>hh{H[o*n+o],H[o*n+o+1],H[(o+1)*n+o],H[(o+1)*n+o+1]};std::vector<optical_gap_detail::HalfPlane>box{{{-1.,0.},it.gap[0]},{{0.,-1.},it.gap[1]}};for(int k=0;k<2;++k){box.push_back({A[k],v.upper.at(7+k)-it.base[7+k]});box.push_back({{-A[k][0],-A[k][1]},it.base[7+k]-v.lower.at(7+k)});}auto d=optical_gap_detail::solve2(hh,gg,box);V2 next{std::max(0.,it.gap[0]+d[0]),std::max(0.,it.gap[1]+d[1])};delta[o]=next[0]-it.gap[0];delta[o+1]=next[1]-it.gap[1];if(!chart->with_endpoint_gaps(it.base.data(),next,v.values.data()))throw std::runtime_error("optical gap candidate is outside its unchanged certificate");}
 if(auto*c=b.gpu_evaluation_callback())c->PrepareForEvaluation(false,true);double after=0;bool finite=p.Evaluate(options,&after,nullptr,nullptr,nullptr);double roundoff=64*std::numeric_limits<double>::epsilon()*std::max(1.,std::abs(before));result.accepted=finite&&std::isfinite(after)&&after<before-roundoff;
 if(!result.accepted){for(const auto&it:items)std::copy(it.base.begin(),it.base.end(),b.core.model.blocks[it.block].values.begin());std::fill(delta.begin(),delta.end(),0.);if(auto*c=b.gpu_evaluation_callback())c->PrepareForEvaluation(false,true);}
 eventio::Json rows=eventio::Json::array();
 for(std::size_t i=0;i<items.size();++i){const auto&it=items[i];auto&v=b.core.model.blocks[it.block];auto*chart=dynamic_cast<const JointOpticalFeasibleCoordinates*>(p.GetManifold(v.values.data()));V2 now;if(!chart->endpoint_gaps(v.values.data(),&now))throw std::runtime_error("accepted optical gaps are invalid");std::array<bool,2>active{};V2 gg{};for(int k=0;k<2;++k){int o=int(i)*2+k;gg[k]=g[o];for(int j=0;j<n;++j)gg[k]+=H[o*n+j]*delta[j];double tol=128*std::numeric_limits<double>::epsilon()*std::max({1.,std::abs(v.values[7]),std::abs(v.values[8])});active[k]=now[k]<=tol&&gg[k]>=0.;}result.working_changed=result.working_changed||active!=it.active;p.SetManifold(v.values.data(),new JointOpticalFeasibleCoordinates(*it.rig,domain.ranges.at(it.rig->id),v,active));rows.push_back({{"rig",it.rig->id},{"gap_before",it.gap},{"gap_after",now},{"normal_gradient",gg},{"active_boundary_normals",active},{"previous_active",it.active},{"tangent",p.ParameterBlockTangentSize(v.values.data())}});}
 if(auto*c=dynamic_cast<JointUVCache*>(b.gpu_evaluation_callback()))c->configure_for_problem(p);
 eventio::atomic_json(file,{{"schema","rinkfit.live_optical_gap_block_minimization.v1"},{"objective_before",before},{"proposed_objective",finite?eventio::Json(after):eventio::Json(nullptr)},{"accepted",result.accepted},{"working_set_changed",result.working_changed},{"rigs",rows},{"seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()},{"source_or_physical_parameter_bounds_changed",false},{"same_live_problem_objective",true}});
 std::cout<<"Optical boundary block: actual objective "<<before<<" -> "<<(finite?after:INFINITY)<<(result.accepted?" accepted":" restored")<<", boundary directions changed "<<result.working_changed<<std::endl;return result;
}
}
