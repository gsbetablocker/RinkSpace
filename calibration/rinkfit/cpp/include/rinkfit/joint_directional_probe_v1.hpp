#pragma once
#include "rinkfit/joint_al.hpp"
#include <iostream>
namespace rinkfit {
// A read-only finite-difference comparison against the actual fully assembled
// Ceres objective and active manifolds. No source/parameter change is retained.
inline eventio::Json joint_actual_directional_probe_v1(JointALBuilder&b,ceres::Problem&problem,int threads){
 using Json=eventio::Json;struct Probe{std::string name;int column;double step;};
 const std::vector<Probe> probes={{"scene/rink5",0,1e-6},{"scene/rink5",1,1e-6},{"rig/main/head15",6,1e-6},{"rig/secondary/head15",6,1e-6},{"rig/secondary/optics12",0,1e-3},{"rig/secondary/optics12",5,1e-6},{"shot/s02/control/15",0,1e-6},{"shot/s01/control_group/recovery_5_36",0,1e-5}};
 ceres::Problem::EvaluateOptions options;options.num_threads=threads;std::map<std::string,std::pair<int,int>>locations;int total=0;
 for(const auto&q:probes){if(locations.count(q.name))continue;auto it=std::find_if(b.core.model.blocks.begin(),b.core.model.blocks.end(),[&](const auto&x){return x.name==q.name;});if(it==b.core.model.blocks.end()||!problem.HasParameterBlock(it->values.data())||problem.IsParameterBlockConstant(it->values.data()))throw std::runtime_error("directional probe block absent or frozen: "+q.name);int index=int(it-b.core.model.blocks.begin());locations[q.name]={index,total};options.parameter_blocks.push_back(it->values.data());total+=problem.ParameterBlockTangentSize(it->values.data());}
 double base=0;std::vector<double>gradient;if(!problem.Evaluate(options,&base,nullptr,&gradient,nullptr)||gradient.size()!=std::size_t(total))throw std::runtime_error("actual directional gradient evaluation failed");
 Json rows=Json::array();
 for(const auto&q:probes){const auto where=locations.at(q.name);auto&block=b.core.model.blocks.at(where.first);const auto saved=block.values;struct Restore{std::vector<double>&v;const std::vector<double>&s;~Restore(){std::copy(s.begin(),s.end(),v.begin());}} restore{block.values,saved};const auto*manifold=problem.GetManifold(block.values.data());const int tangent=problem.ParameterBlockTangentSize(block.values.data());if(q.column>=tangent)throw std::runtime_error("directional tangent index");std::vector<double>delta(tangent,0.),trial(saved.size());double plus=0,minus=0;bool ok[2]{};
  for(int side=0;side<2;++side){delta[q.column]=(side? -1.:1.)*q.step;bool valid=true;if(manifold)valid=manifold->Plus(saved.data(),delta.data(),trial.data());else{trial=saved;trial[q.column]+=delta[q.column];}if(valid){std::copy(trial.begin(),trial.end(),block.values.begin());ok[side]=problem.Evaluate(options,side?&minus:&plus,nullptr,nullptr,nullptr);}std::copy(saved.begin(),saved.end(),block.values.begin());}
  const double exact=gradient.at(where.second+q.column);Json rec={{"block",q.name},{"tangent_column",q.column},{"step",q.step},{"J_transpose_r",exact},{"positive_trial_valid",ok[0]},{"negative_trial_valid",ok[1]}};
  if(ok[0]&&ok[1]){const double fd=(plus-minus)/(2*q.step);rec["central_difference"]=fd;rec["relative_difference"]=std::abs(fd-exact)/std::max(1.,std::max(std::abs(fd),std::abs(exact)));rec["plus_cost"]=plus;rec["minus_cost"]=minus;}
  rows.push_back(rec);std::cout<<"Directional derivative "<<rec.dump()<<std::endl;
 }
 double restored=0;if(!problem.Evaluate(options,&restored,nullptr,nullptr,nullptr)||std::abs(restored-base)>1e-10*std::max(1.,std::abs(base)))throw std::runtime_error("directional probe did not restore original objective");
 return {{"schema","rinkfit.actual_objective_directional_probe.v1"},{"initial_cost",base},{"restored_cost",restored},{"original_primal_restored",true},{"same_sources_and_full_objective",true},{"probes",rows}};
}
inline eventio::Json joint_actual_family_probe_v1(JointALBuilder&b,ceres::Problem&problem,int threads){
 struct V{double*p;std::vector<double>x,d;std::string family;const ceres::Manifold*manifold;std::size_t offset;};std::vector<V>vars;ceres::Problem::EvaluateOptions options;options.num_threads=threads;std::size_t offset=0;std::map<double*,const EventParameterBlock*>physical;for(const auto&bl:b.core.model.blocks)physical[const_cast<double*>(bl.values.data())]=&bl;
 std::uint64_t random=123456789;auto sign=[&](){random^=random<<13;random^=random>>7;random^=random<<17;return (random&1)?1.:-1.;};
 for(const auto&bl:b.parameter_blocks()){if(!problem.HasParameterBlock(bl.data)||problem.IsParameterBlockConstant(bl.data))continue;const int n=problem.ParameterBlockTangentSize(bl.data);V v{bl.data,std::vector<double>(bl.data,bl.data+bl.size),std::vector<double>(n),"",problem.GetManifold(bl.data),offset};
  if(bl.name.rfind("station/",0)==0){v.family="smooth_curve_positions";const auto i=std::stoull(bl.name.substr(8));const auto&r=b.core.packet.rows.at(i);const auto&path=b.core.model.materials.at(r.material).paths.at(r.path);int den=1;if(path.kind==2||path.kind==6){const int count=path.kind==2?int(path.points.size()):int(b.core.model.blocks.at(path.geometry_block).values.size()/2);den=path.closed?count:count-1;if(den<1||std::abs(bl.data[0]*den-std::round(bl.data[0]*den))<1e-5)continue;}v.d[0]=sign()*1e-7/den;
  }else if(bl.name.rfind("static_motion/",0)==0){v.family="shared_motion_points";for(auto&d:v.d)d=sign()*1e-7;
  }else if(physical.count(bl.data)){v.family=bl.name.find("/optics12")!=std::string::npos?"all_optics":bl.name.find("/head15")!=std::string::npos?"all_heads":bl.name.rfind("shot/",0)==0?"all_camera_motion":"scene_ads_ice_prints";const auto&sc=physical.at(bl.data)->scales;for(int k=0;k<n;++k)v.d[k]=sign()*1e-6*(n==int(sc.size())?sc[k]:.1);
  }else continue;
  if(!v.manifold){for(std::size_t k=0;k<bl.size;++k)if(v.x[k]-std::abs(v.d[k])*2<=problem.GetParameterLowerBound(v.p,int(k))||v.x[k]+std::abs(v.d[k])*2>=problem.GetParameterUpperBound(v.p,int(k)))v.d[k]=0;}
  options.parameter_blocks.push_back(v.p);offset+=n;vars.push_back(std::move(v));
 }
 double base=0;std::vector<double>g;if(!problem.Evaluate(options,&base,nullptr,&g,nullptr)||g.size()!=offset)throw std::runtime_error("family objective derivative failed");eventio::Json reports=eventio::Json::array();
 for(const std::string family:{"all_optics","all_heads","all_camera_motion","scene_ads_ice_prints","smooth_curve_positions","shared_motion_points"}){double expected=0;std::size_t cols=0;for(const auto&v:vars)if(v.family==family)for(std::size_t k=0;k<v.d.size();++k){expected+=g[v.offset+k]*v.d[k];cols+=v.d[k]!=0;}
  double costs[2]={NAN,NAN};bool ok[2]={};double maxchange=0;
  for(int side=0;side<2;++side){bool mapped=true;for(auto&v:vars)if(v.family==family){auto d=v.d;if(side)for(auto&z:d)z=-z;std::vector<double>y(v.x.size());if(v.manifold)mapped=v.manifold->Plus(v.x.data(),d.data(),y.data())&&mapped;else for(std::size_t k=0;k<y.size();++k)y[k]=v.x[k]+d[k];if(mapped){for(std::size_t k=0;k<y.size();++k)maxchange=std::max(maxchange,std::abs(y[k]-v.x[k]));std::copy(y.begin(),y.end(),v.p);}}
   if(mapped)ok[side]=problem.Evaluate(options,&costs[side],nullptr,nullptr,nullptr);for(auto&v:vars)std::copy(v.x.begin(),v.x.end(),v.p);
  }
  eventio::Json rec={{"family",family},{"perturbed_tangent_columns",cols},{"J_transpose_r_dot_delta",expected},{"plus_valid",ok[0]},{"minus_valid",ok[1]},{"max_ambient_change",maxchange}};
  if(ok[0]&&ok[1]){const double fd=(costs[0]-costs[1])/2;rec["finite_difference_dot_delta"]=fd;rec["relative_difference"]=std::abs(fd-expected)/std::max(1e-8,std::max(std::abs(fd),std::abs(expected)));rec["plus_cost_change"]=costs[0]-base;rec["minus_cost_change"]=costs[1]-base;}reports.push_back(rec);std::cout<<"Actual full-objective family derivative "<<rec.dump()<<std::endl;
 }
 double restored=0;if(!problem.Evaluate(options,&restored,nullptr,nullptr,nullptr)||std::abs(restored-base)>1e-10*std::max(1.,std::abs(base)))throw std::runtime_error("family probe failed to restore objective");return {{"schema","rinkfit.actual_full_objective_family_derivative.v1"},{"initial_cost",base},{"restored_cost",restored},{"source_and_primal_restored",true},{"probes",reports}};
}

inline eventio::Json joint_actual_gradient_locations_v1(JointALBuilder&b,ceres::Problem&problem,int threads){
 struct Block{std::string name;std::size_t offset;int size;};std::vector<Block>blocks;ceres::Problem::EvaluateOptions options;options.num_threads=threads;std::size_t offset=0;
 for(const auto&bl:b.parameter_blocks()){if(!problem.HasParameterBlock(bl.data)||problem.IsParameterBlockConstant(bl.data))continue;const int n=problem.ParameterBlockTangentSize(bl.data);options.parameter_blocks.push_back(bl.data);blocks.push_back({bl.name,offset,n});offset+=n;}
 double cost=0;std::vector<double>g;if(!problem.Evaluate(options,&cost,nullptr,&g,nullptr)||g.size()!=offset)throw std::runtime_error("actual gradient location evaluation failed");
 struct Peak{double magnitude,value;std::size_t block;int coordinate;};std::vector<Peak>peaks;
 for(std::size_t bi=0;bi<blocks.size();++bi){const auto&v=blocks[bi];Peak peak{0.,0.,bi,0};for(int k=0;k<v.size;++k){double x=g[v.offset+k];if(!std::isfinite(x))throw std::runtime_error("nonfinite actual gradient");if(std::abs(x)>peak.magnitude)peak={std::abs(x),x,bi,k};}if(peak.magnitude>0)peaks.push_back(peak);}
 const std::size_t n=std::min(std::size_t(24),peaks.size());std::partial_sort(peaks.begin(),peaks.begin()+n,peaks.end(),[](const Peak&a,const Peak&c){return a.magnitude>c.magnitude;});eventio::Json top=eventio::Json::array();for(std::size_t i=0;i<n;++i){const auto&p=peaks[i];top.push_back({{"block",blocks[p.block].name},{"tangent_coordinate",p.coordinate},{"gradient",p.value}});}
 std::cout<<"Largest actual objective derivatives: "<<top.dump()<<std::endl;return {{"objective_half_cost",cost},{"tangent_columns",offset},{"largest_by_parameter_block",top},{"source_and_primal_changed",false}};
}

}
