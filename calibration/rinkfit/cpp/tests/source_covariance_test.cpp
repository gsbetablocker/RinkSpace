#include <rinkfit/source_covariance.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <functional>

int main(int argc, char** argv) {
 try {
  if(argc!=2)throw std::runtime_error("fixture argument required");
  std::ifstream f(argv[1]);std::size_t r,n,k,nr,declared;double threshold;
  if(!(f>>r>>n>>k>>nr>>threshold>>declared))throw std::runtime_error("fixture header");
  std::vector<std::size_t>keep(k);for(auto& x:keep)f>>x;
  std::vector<double>old(r*n),next(nr*k),null(k);for(auto* v:{&old,&next,&null})for(auto& x:*v)f>>x;
  if(!f)throw std::runtime_error("fixture matrix payload");
  const rinkfit::SourceCovarianceRankProof p{threshold,declared};
  const auto audit=rinkfit::validate_supported_covariance_marginal(old,r,n,keep,next,nr,p);
  std::cout<<std::setprecision(17)<<"{\"real_row34524_accepted\":true,\"computed_retained_rank\":"<<audit.computed_retained_rank
   <<",\"effective_rank_threshold\":"<<audit.effective_rank_threshold<<",\"backward_error_threshold\":"<<audit.backward_error_threshold
   <<",\"accepted_min_pivot\":"<<audit.accepted_min_pivot<<",\"discarded_max_pivot\":"<<audit.discarded_max_pivot
   <<",\"whitening_max_abs_error\":"<<audit.whitening_max_abs_error<<",\"CGC_max_abs_error\":"<<audit.cgc_max_abs_error
   <<",\"GCG_max_abs_error\":"<<audit.gcg_max_abs_error<<",\"GC_symmetry_max_abs_error\":"<<audit.gc_symmetry_max_abs_error;
  auto reject=[&](const char* name,const std::function<void()>& operation){
   try{operation();}catch(const std::exception& e){std::cout<<",\""<<name<<"\":true";std::cerr<<name<<": "<<e.what()<<"\n";return;}
   throw std::runtime_error(std::string("invalid marginal accepted: ")+name);
  };
  std::vector<double>slice(r*k);for(std::size_t i=0;i<r;++i)for(std::size_t j=0;j<k;++j)slice[i*k+j]=old[i*n+keep[j]];
  reject("column_slice_rejected",[&]{rinkfit::validate_supported_covariance_marginal(old,r,n,keep,slice,r,{threshold,r});});
  auto scaled=next;for(auto& x:scaled)x*=2.;
  reject("scaled_W_rejected",[&]{rinkfit::validate_supported_covariance_marginal(old,r,n,keep,scaled,nr,p);});
  auto contaminated=next;for(std::size_t j=0;j<k;++j)contaminated[j]+=.25*null[j];
  reject("covariance_nullspace_information_rejected",[&]{rinkfit::validate_supported_covariance_marginal(old,r,n,keep,contaminated,nr,p);});
  reject("false_retained_rank_rejected",[&]{rinkfit::validate_supported_covariance_marginal(old,r,n,keep,next,nr,{threshold,nr-1});});
  auto duplicate=keep;duplicate[1]=duplicate[0];
  reject("duplicate_coordinate_rejected",[&]{rinkfit::validate_supported_covariance_marginal(old,r,n,duplicate,next,nr,p);});
  auto singular=old;for(std::size_t j=0;j<n;++j)singular[n+j]=singular[j];
  reject("singular_original_W_rejected",[&]{rinkfit::validate_supported_covariance_marginal(singular,r,n,keep,next,nr,p);});
  std::cout<<"}\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
