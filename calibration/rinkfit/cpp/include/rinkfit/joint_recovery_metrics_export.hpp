#pragma once
#include "rinkfit/joint_al.hpp"
#include <atomic>
#include <fstream>
#include <thread>

namespace rinkfit {
// Stored CP auxiliaries only: this function never re-solves station,row,alpha.
// Finite pixels are retained when domain/equality checks fail; flags explain
// the failure. A fresh GPU value preparation is bound to the current model.
inline eventio::Json export_joint_recovery_explicit_metrics(
    JointALBuilder& b,const std::filesystem::path& out,int threads=8) {
  std::filesystem::create_directories(out);
  auto* cache=dynamic_cast<JointUVCache*>(b.gpu_evaluation_callback());
  if(cache)cache->PrepareForEvaluation(false,true);
  const std::size_t n=b.core.packet.rows.size();
  std::vector<double> uv(2*n,NAN),error(2*n,NAN),rs(n,NAN),query(2*n,NAN);
  std::vector<std::uint8_t> valid(n),arithmetic(n),domain(n),station_ok(n),visibility(n),width(n);
  std::atomic<std::size_t> next{0},cached{0},direct{0},exceptions{0};
  auto worker=[&](){
    for(;;){auto i=next.fetch_add(1);if(i>=n)return;
      const auto&o=b.core.packet.rows[i];if(!o.active)continue;
      try {
        const auto l=o.layout(b.core.model);const double st=o.kind==1?o.station:b.core.station[i];
        JointUVCacheView v;bool hit=cache&&cache->lookup(i,&v);
        bool all_domains=hit;for(int k=0;k<5;++k)all_domains=all_domains&&v.native_domain_available[k];
        Vec2<double> px;bool ar=false,dom=false,vis=false;auto ptr=b.core.model.pointers(l);
        if(all_domains){px=v.canonical_uv;ar=finite_array(px);dom=ar&&v.canonical_native_domains[0]<=1e-10&&v.canonical_native_domains[1]<0&&v.canonical_native_domains[2]<0;vis=ar&&v.canonical_native_domains[3]<=1e-10&&v.canonical_native_domains[4]<=1e-10;++cached;}
        else {
          const auto p=b.core.model.predict_for_al(l,st,b.core.row[i],ptr.data());px=p.uv;ar=p.arithmetic_valid&&finite_array(px);dom=ar&&p.physical_status==DomainStatus::kValid;
          const auto visible=b.core.model.visibility(l,st,b.core.row[i],ptr.data());vis=visible.visible();++direct;
        }
        arithmetic[i]=ar;domain[i]=dom;visibility[i]=vis;
        station_ok[i]=o.station_allowed(b.core.model,o.path,st,1e-12);width[i]=1;
        if(o.width_block>=0)for(int k:{5,6}){const auto d=b.core.model.measurement_domain(l,st,ptr.data(),k);if(!std::isfinite(d)||d>1e-8)width[i]=0;}
        valid[i]=ar&&dom&&vis&&station_ok[i]&&width[i];
        if(ar){uv[2*i]=px[0];uv[2*i+1]=px[1];error[2*i]=px[0]-o.uv[0];error[2*i+1]=px[1]-o.uv[1];rs[i]=b.core.row[i]-px[1];
          if(o.query){Vec2<double> q;if(o.query->point(b.core.alpha[i],&q)){query[2*i]=px[0]-q[0];query[2*i+1]=px[1]-q[1];}}
        }
      }catch(...){++exceptions;}
    }
  };
  std::vector<std::thread> pool;for(int k=0;k<std::max(1,threads);++k)pool.emplace_back(worker);for(auto&t:pool)t.join();
  eventio::Json files=eventio::Json::object();
  auto write=[&](const std::string& name,const auto& values){std::ofstream f(out/name,std::ios::binary);f.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(values[0]));f.close();if(!f)throw std::runtime_error("explicit recovery metric output failed");files[name]={{"sha256",eventio::file_sha256(out/name)},{"size_bytes",std::filesystem::file_size(out/name)}};};
  write("explicit_predicted_uv.f64",uv);write("explicit_error_native_uv.f64",error);write("explicit_RS_equality.f64",rs);write("explicit_query_equality.f64",query);
  write("explicit_valid.u8",valid);write("explicit_arithmetic_valid.u8",arithmetic);write("explicit_ray_domain_valid.u8",domain);write("explicit_station_domain_valid.u8",station_ok);write("explicit_visibility_valid.u8",visibility);write("explicit_width_domain_valid.u8",width);
  eventio::Json report={{"schema","rinkfit.recovery_CP_explicit_metrics.v1"},{"packet_sha256",b.core.packet.source_hash},{"state_token",b.capture_state().token()},{"raw_rows",n},{"files",files},{"GPU_cached_value_rows",cached.load()},{"direct_native_value_rows",direct.load()},{"exceptions",exceptions.load()},{"stored_auxiliaries_reprofiled",false},{"invalid_pixels_zeroed",false},{"valid_semantics","arithmetic, ray/Brown/depth, source finite station, physical visibility and width domain; RS and Query equality errors remain separately recorded"}};
  eventio::atomic_json(out/"manifest.json",report);return report;
}
}
