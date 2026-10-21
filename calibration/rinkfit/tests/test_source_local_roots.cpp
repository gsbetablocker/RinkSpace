#include "rinkfit/joint_factors.hpp"
#include <iostream>
using namespace rinkfit;
int main(int argc,char**argv){
 try{
  if(argc!=3)throw std::runtime_error("model and actual immutable packet required");
  auto model=JointEventModel::load(argv[1]);auto packet=JointPacket::load(argv[2],model);
  bool checked_query=false,checked_point=false;eventio::Json witnesses=eventio::Json::array();
  for(std::size_t i=0;i<packet.rows.size();++i){
   const auto&o=packet.rows[i];if(!o.active||(!o.query&&o.kind!=1)||(o.query?checked_query:checked_point))continue;
   JointRowEvaluation q;q.model=&model;q.observation=&o;q.layout=o.layout(model);
   for(std::size_t k=0;k<q.layout.block_ids.size();++k)q.parameter_sizes[int(k)]=int(model.blocks[q.layout.block_ids[k]].values.size());
   auto base=model.pointers(q.layout);std::array<double,3>x;Eigen::Matrix3d J;
   if(!q.local_solution(base.data(),o.station,&x,&J))continue;
   if(!o.station_allowed(model,q.layout.path,x[0])||x[1]<0||x[1]>1079)throw std::runtime_error("root outside source finite support");
   Vec2<double>error;if(!q.native_error(base.data(),&error))throw std::runtime_error("valid root failed source residual");
   using A=ceres::Jet<double,1>;std::vector<std::vector<A>>ap;std::vector<const A*>ptr;
   for(auto z:q.parameter_sizes){ap.emplace_back();for(int j=0;j<z.second;++j)ap.back().push_back(A(base[z.first][j]));ptr.push_back(ap.back().data());}
   const int slot=q.layout.optics_slot,index=3;ap[slot][index].v[0]=1;
   Vec2<A>ad;if(!q.native_error(ptr.data(),&ad))throw std::runtime_error("implicit source J failed");
   auto&parameter=model.blocks[q.layout.block_ids[slot]].values[index];const double original=parameter,h=1e-4;
   Vec2<double>plus,minus;parameter=original+h;bool vp=q.native_error(base.data(),&plus);parameter=original-h;bool vm=q.native_error(base.data(),&minus);parameter=original;
   if(!vp||!vm)continue;double worst=0;
   for(int j=0;j<2;++j){double fd=(plus[j]-minus[j])/(2*h);worst=std::max(worst,std::abs(fd-ad[j].v[0]));if(std::abs(fd-ad[j].v[0])>1e-4*(1+std::abs(fd)))throw std::runtime_error("actual-source implicit derivative mismatch");}
   auto moved=q;JointObservation unavailable=o;unavailable.row=-1;moved.observation=&unavailable;moved.cache=std::make_shared<JointRowEvaluation::RootCache>();
   std::array<double,3>bad;Eigen::Matrix3d BJ;if(moved.local_solution(base.data(),o.station,&bad,&BJ))throw std::runtime_error("out-of-native-row seed accepted");
   witnesses.push_back({{"raw_index",i},{"source_row_id",o.source_row_id},{"shot",o.shot_id},{"native_pts",o.pts},{"query",bool(o.query)},{"root",x},{"source_error_uv",error},{"Jv_FD_max_abs",worst},{"illegal_native_row_rejected",true}});
   if(o.query)checked_query=true;else checked_point=true;
   if(checked_query&&checked_point)break;
  }
  if(!checked_query||!checked_point)throw std::runtime_error("actual-source operator witness unavailable");
  std::cout<<eventio::Json({{"schema","rinkfit.actual_source_local_root_check.v1"},{"model",argv[1]},{"packet",argv[2]},{"witnesses",witnesses},{"all_original_source_rows_retained",true}}).dump(2)<<std::endl;
  return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}
}
