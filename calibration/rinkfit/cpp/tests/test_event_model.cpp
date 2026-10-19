#include "rinkfit/event_model.hpp"
#include <cassert>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
using namespace rinkfit;
template<class T>std::vector<T> binary(const std::filesystem::path& p){std::ifstream f(p,std::ios::binary|std::ios::ate);const auto size=f.tellg();if(size<0||size%sizeof(T))throw std::runtime_error("invalid diagnostic array");std::vector<T> v(static_cast<std::size_t>(size)/sizeof(T));f.seekg(0);f.read(reinterpret_cast<char*>(v.data()),size);return v;}
int main(int argc,char** argv){
  double g[kEventGlobalSize]{};g[0]=.5021836202639072;g[1]=.11134303732229413;g[2]=.12344015387994692;g[3]=1.9051596028935873;g[4]=.01665312746798789;
  const double arc=EventModel::quarter_arclength(1.5707963267948966,g);
  assert(std::abs(arc-.1827927463737)<1e-9);
  const double L=1-2*g[1],S=g[0]-2*g[2],per=2*(L+S)+4*arc;
  const double lengths[8]={L,arc,S,arc,L,arc,S,arc};double s=0;
  for(int k=0;k<8;++k){
    const auto p=EventModel::board_point(s,g[4],g);
    const auto left=EventModel::board_point(s-1e-8,g[4],g),right=EventModel::board_point(s+1e-8,g[4],g);
    assert(std::sqrt(dot(sub(left,p),sub(left,p)))<1.1e-8);
    assert(std::sqrt(dot(sub(right,p),sub(right,p)))<1.1e-8);
    const auto before=event_normalized(sub(p,left)),after=event_normalized(sub(right,p));
    assert(dot(before,after)>.999999);s+=lengths[k];
  }
  for(double x:{.01,.3,.9,1.5,2.2}){
    const auto a=EventModel::board_point(x,g[4],g),b=EventModel::board_point(x+per,g[4],g);
    assert(std::sqrt(dot(sub(a,b),sub(a,b)))<1e-12);
    const double h=1e-6;const auto pa=EventModel::board_point(x-h,g[4],g),pb=EventModel::board_point(x+h,g[4],g);
    assert(std::abs(std::sqrt(dot(sub(pb,pa),sub(pb,pa)))/(2*h)-1)<1e-6);
  }
  EventModel m;m.R0_seed={1,0,0,0,0,-1,0,1,0};m.n_seed={0,0,1};m.b_seed={1,0,0};m.n_tangent1={1,0,0};m.n_tangent2={0,1,0};
  g[5]=.02;g[6]=-.01;g[8]=.1;g[9]=-.05;g[10]=.1;g[13]=.03;
  const auto head=m.head(g);assert(std::abs(dot(head.b,matvec(head.R0,head.n)))<1e-12);assert(valid_rotation(head.R0));assert(camera_pose(head,.2,-.1,.3).valid());
  m.n_controls=6;m.blocks.resize(7);m.blocks[0].values.assign(g,g+kEventGlobalSize);
  m.materials.resize(1);m.materials[0].paths.resize(1);
  for(int j=0;j<6;++j)m.blocks[j+1].values={.03*j,-.02*j,8.+.1*j};
  m.blocks[3].values[0]=99.;m.blocks[3].values[1]=-99.; // dummy storage must not affect the function gauge
  const auto layout=m.layout(0,0,0.);const auto ptrs=m.pointers(layout);const auto tr=m.trajectory(0.,ptrs.data(),layout);
  assert(std::abs(tr[0])<1e-15&&std::abs(tr[1])<1e-15);
  std::cout<<"event_model: true superellipse arclength, anisotropic corners, all joins, periodicity, exact head chart passed\n";
  if(argc>1){const auto start=std::chrono::steady_clock::now();auto real=EventModel::load(argv[1]);const auto state_root=std::filesystem::path(argv[1]).parent_path();const auto root=std::filesystem::exists(state_root/"uv.f64")?state_root:state_root.parent_path().parent_path();
    auto stations=binary<double>(state_root/"station.f64");const auto uv=binary<double>(root/"uv.f64");const auto frames=binary<int>(root/"frame.i32");const auto mats=binary<unsigned>(root/"material.u32");auto paths=binary<unsigned>(state_root/"path.u32");
    if(stations.size()!=182017||frames.size()!=stations.size()||mats.size()!=stations.size()||paths.size()!=stations.size()||uv.size()!=2*stations.size())throw std::runtime_error("whole shot diagnostic count mismatch");
    const bool repair=argc>2&&std::string(argv[2])=="--repair-initializer";
    const auto variants=repair?binary<unsigned>(root.parent_path()/"semantic_packet"/"variant_index.u32"):std::vector<unsigned>{};
    std::vector<double> errors;std::array<int,8> status{};int exceptions=0,repaired=0,unresolved=0;double invalid_ray_max=0;
    for(std::size_t i=0;i<stations.size();++i){try{const auto l=real.layout(mats[i],paths[i],(frames[i]-1)*.02);const auto p=real.pointers(l);const auto pred=real.predict_explicit(l,stations[i],uv[2*i+1],p.data());
        ++status[static_cast<int>(pred.status)];if(pred.valid())errors.push_back(std::hypot(pred.uv[0]-uv[2*i],pred.uv[1]-uv[2*i+1]));
        else if(pred.status==DomainStatus::kOutsideDistortionDomain){const auto& g=real.blocks[0].values;const auto tr=real.trajectory(row_time(l.frame_time,g[31],uv[2*i+1]),p.data(),l);const auto oo=zoom_optics(real.optics(g.data()),tr[2]);const auto pose=camera_pose(real.head(g.data()),tr[0],tr[1],oo.z);const auto Y=matvec(pose.R,sub(real.world_point(real.materials[mats[i]].paths[paths[i]],stations[i],p.data(),l),pose.C));invalid_ray_max=std::max(invalid_ray_max,std::hypot(Y[0],Y[1])/Y[2]);}}
      catch(...){++exceptions;}
      if(repair){const auto oldlayout=real.layout(mats[i],paths[i],(frames[i]-1)*.02);const auto oldp=real.pointers(oldlayout);bool valid=false;try{valid=real.predict_explicit(oldlayout,stations[i],uv[2*i+1],oldp.data()).valid();}catch(...){}
        if(!valid){double best=INFINITY,bests=0;unsigned bestpath=0;
          for(unsigned path=0;path<real.materials[mats[i]].paths.size();++path){const auto& materialpath=real.materials[mats[i]].paths[path];if(unsigned(std::stoul(materialpath.variant_id.substr(1)))!=variants[i])continue;
            const auto layout=real.layout(mats[i],path,(frames[i]-1)*.02);const auto pointers=real.pointers(layout);
            auto cost=[&](double s)->double{try{const auto p=real.predict_explicit(layout,s,uv[2*i+1],pointers.data());return p.valid()?std::pow(p.uv[0]-uv[2*i],2)+std::pow(p.uv[1]-uv[2*i+1],2):INFINITY;}catch(...){return INFINITY;}};
            const int count=materialpath.physical_point?1:129;int pick=0;double low=INFINITY;
            for(int j=0;j<count;++j){const double s=count==1?0.:double(j)/(count-1),v=cost(s);if(v<low){low=v;pick=j;}}
            double s=count==1?0.:double(pick)/(count-1);
            if(count>1&&std::isfinite(low)){double a=std::max(0.,double(pick-1)/(count-1)),b=std::min(1.,double(pick+1)/(count-1));constexpr double ratio=.6180339887498948482;double c=b-ratio*(b-a),d=a+ratio*(b-a),fc=cost(c),fd=cost(d);
              for(int iter=0;iter<45;++iter){if(fc<fd){b=d;d=c;fd=fc;c=b-ratio*(b-a);fc=cost(c);}else{a=c;c=d;fc=fd;d=a+ratio*(b-a);fd=cost(d);}}const double candidate=(a+b)/2,v=cost(candidate);if(v<low){s=candidate;low=v;}}
            if(low<best){best=low;bests=s;bestpath=path;}}
          if(std::isfinite(best)){stations[i]=bests;paths[i]=bestpath;++repaired;}else {++unresolved;double ray=NAN;try{const auto& gg=real.blocks[0].values;const auto tr=real.trajectory(row_time(oldlayout.frame_time,gg[31],uv[2*i+1]),oldp.data(),oldlayout);const auto oo=zoom_optics(real.optics(gg.data()),tr[2]);const auto pose=camera_pose(real.head(gg.data()),tr[0],tr[1],oo.z);const auto Y=matvec(pose.R,sub(real.world_point(real.materials[mats[i]].paths[paths[i]],stations[i],oldp.data(),oldlayout),pose.C));ray=std::hypot(Y[0],Y[1])/Y[2];}catch(...){}std::cout<<"unresolved_initializer_domain row="<<i<<" frame="<<frames[i]<<" material="<<real.materials[mats[i]].label<<" path="<<paths[i]<<" station="<<stations[i]<<" ray="<<ray<<"\n";}}}
    }
    std::sort(errors.begin(),errors.end());const auto constraints=real.physical_constraints();double vmax=-INFINITY;for(double v:constraints)vmax=std::max(vmax,v);
    std::cout<<"whole_source_explicit_row_time_initializer_only rows="<<stations.size()<<" valid="<<errors.size()<<" exceptions="<<exceptions<<" statuses=";for(int v:status)std::cout<<v<<",";
    if(!errors.empty())std::cout<<" error_median_px="<<errors[errors.size()/2]<<" error_p95_px="<<errors[std::size_t(.95*(errors.size()-1))]<<" error_max_px="<<errors.back();
    std::cout<<" invalid_ray_max="<<invalid_ray_max<<" physical_constraints="<<constraints.size()<<" max_normalized_physical_g="<<vmax<<" seconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<"\n";
    if(repair){std::ofstream s(root/"station_recovered.f64",std::ios::binary);s.write(reinterpret_cast<const char*>(stations.data()),std::streamsize(stations.size()*sizeof(double)));std::ofstream p(root/"path_recovered.u32",std::ios::binary);p.write(reinterpret_cast<const char*>(paths.data()),std::streamsize(paths.size()*sizeof(unsigned)));std::cout<<"initial_domain_recovery retained_rows="<<stations.size()<<" repaired="<<repaired<<" unresolved="<<unresolved<<" (current physical model, source variant, all legal finite branches; no UV or identity change)\n";}
  }
}
