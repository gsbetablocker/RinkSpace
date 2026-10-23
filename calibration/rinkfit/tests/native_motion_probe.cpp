#include "rinkfit/joint_motion_v1.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <set>
int main(int argc,char**argv){try{
 if(argc!=4)throw std::runtime_error("model motion_manifest report required");using namespace rinkfit;auto m=JointEventModel::load(argv[1]);eventio::Json ref={{"file",argv[2]},{"sha256",eventio::file_sha256(argv[2])}};auto store=JointMotionStoreV1::load(ref,m);const auto init=initialize_joint_motion_domain_v1(*store,m,false);std::cout<<init.dump()<<std::endl;eventio::Json modes=eventio::Json::array();
 for(bool rs:{false,true}){std::size_t valid=0,invalid=0,fd_columns=0;double max_error=0,sumcost=0;std::set<std::pair<int,int>>covered;auto start=std::chrono::steady_clock::now();eventio::Json failures=eventio::Json::array();
  for(std::size_t i=0;i<store->rows.size();++i){auto&r=store->rows[i];auto prep=make_joint_motion_factor_v1(m,r,rs);std::unique_ptr<ceres::CostFunction>cost(prep.first);double residual[4];std::vector<const double*>p(prep.second.begin(),prep.second.end());
   if(!cost->Evaluate(p.data(),residual,nullptr)){++invalid;if(failures.size()<12)failures.push_back({{"row",i},{"shot",m.shots[r.shot].id},{"pts",r.pts0},{"latent",r.latent}});continue;}++valid;for(double v:residual){if(!std::isfinite(v))throw std::runtime_error("nonfinite actual motion residual");sumcost+=.5*v*v;}
   const auto g=m.blocks[m.scene_block].values;double per=2*(1-2*g[1]+g[0]-2*g[2])+4*EventModel::quarter_arclength(1.5707963267948966,g.data());int region=int(8*r.latent[0]/per);
   if(r.latent[1]>1e-6&&r.latent[1]<1.-1e-6&&covered.emplace(r.shot,region).second){std::vector<std::vector<double>>J;std::vector<double*>jp;for(int n:cost->parameter_block_sizes()){J.emplace_back(4*n);jp.push_back(J.back().data());}if(!cost->Evaluate(p.data(),residual,jp.data()))throw std::runtime_error("actual motion AD failed");for(const auto& block:J)for(double value:block)if(!std::isfinite(value))throw std::runtime_error("nonfinite actual motion Jacobian");
    // Latent, reference rotation and each control coordinates exercise both
    // shared-surface and camera paths. Fixed optics/scene remain audited by AD.
    for(std::size_t k=0;k<p.size();++k){const int n=cost->parameter_block_sizes()[k];for(int c=0;c<n;++c){double*value=prep.second[k]+c;const double old=*value,step=1e-6*std::max(1.,std::abs(old));double plus[4],minus[4];*value=old+step;bool hi=cost->Evaluate(p.data(),plus,nullptr);*value=old-step;bool lo=cost->Evaluate(p.data(),minus,nullptr);*value=old;if(!hi||!lo)continue;
     for(int z=0;z<4;++z){const double fd=(plus[z]-minus[z])/(2*step),ad=J[k][z*n+c];max_error=std::max(max_error,std::abs(fd-ad)/std::max({1.,std::abs(fd),std::abs(ad)}));}++fd_columns;
    }}
   }
  }
  double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();modes.push_back({{"true_RS",rs},{"selected_denominator",store->rows.size()},{"valid",valid},{"invalid",invalid},{"unweighted_half_cost",sumcost},{"finite_difference_columns",fd_columns},{"maximum_scaled_AD_FD_error",max_error},{"covered_shot_surface_regions",covered.size()},{"failures",failures},{"seconds",seconds}});std::cout<<modes.back().dump()<<std::endl;
 }
 bool passed=true;for(const auto& mode:modes)passed=passed&&mode.at("invalid")==0&&mode.at("valid")==mode.at("selected_denominator")&&mode.at("finite_difference_columns").get<int>()>0&&mode.at("maximum_scaled_AD_FD_error").get<double>()<1e-4;
 eventio::atomic_json(argv[3],{{"schema","rinkfit.actual_static_motion_factor_probe.v1"},{"model_sha256",eventio::file_sha256(argv[1])},{"motion_packet",ref},{"modes",modes},{"initialization",init},{"all_source_coordinates_unchanged",true},{"fit_performed",false},{"passed",passed},{"scaled_AD_FD_tolerance",1e-4}});return passed?0:3;
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 2;}}
