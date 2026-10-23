#pragma once
#include "rinkfit/event_io.hpp"
#include <atomic>
#include <thread>
#include <exception>
namespace rinkfit {
#ifndef RINKFIT_SOURCE_VALIDATOR_SHA
#define RINKFIT_SOURCE_VALIDATOR_SHA ""
#endif
struct JointSourceValidationCache {
 static eventio::Json identity(const std::string&model,const std::string&packet,const std::string&qualification){return {{"validator",RINKFIT_SOURCE_VALIDATOR_SHA},{"model",model},{"packet",packet},{"qualification",qualification}};}
 static std::filesystem::path file(const std::filesystem::path&root,const eventio::Json&id){const auto s=id.dump();return root/(eventio::sha256(std::vector<std::uint8_t>(s.begin(),s.end()))+".json");}
 static eventio::Json read(const std::filesystem::path&root,const eventio::Json&id,int threads){
  if(root.empty()||std::string(RINKFIT_SOURCE_VALIDATOR_SHA).empty())return nullptr;
  const auto f=file(root,id);if(!std::filesystem::exists(f))return nullptr;
  const auto j=eventio::read_json(f);if(j.value("schema",std::string{})!="rinkfit.source_validation_cache.v1"||j.at("identity")!=id)return nullptr;
  const auto&refs=j.at("verified_files");std::atomic<std::size_t>next{0};std::exception_ptr error;std::mutex mutex;
  auto work=[&](){try{for(;;){const auto i=next.fetch_add(1);if(i>=refs.size())break;const auto&r=refs.at(i);eventio::verify(std::filesystem::u8path(r.at("file").get<std::string>()),r.at("sha256"));}}catch(...){std::lock_guard<std::mutex>lock(mutex);if(!error)error=std::current_exception();}};
  std::vector<std::thread>workers;for(int k=1;k<std::max(1,threads);++k)workers.emplace_back(work);work();for(auto&t:workers)t.join();if(error)std::rethrow_exception(error);
  return j;
 }
 static void write(const std::filesystem::path&root,const eventio::Json&id,const eventio::Json&report){
  if(root.empty()||std::string(RINKFIT_SOURCE_VALIDATOR_SHA).empty())return;
  eventio::Json refs=eventio::Json::array();auto&m=eventio::verification_memo();{std::lock_guard<std::mutex>lock(m.mutex);for(const auto&v:m.entries)refs.push_back({{"file",v.first},{"sha256",v.second.digest}});}
  std::filesystem::create_directories(root);eventio::atomic_json(file(root,id),{{"schema","rinkfit.source_validation_cache.v1"},{"identity",id},{"verified_files",refs},{"report",report},{"all_file_content_hashes_rechecked_on_reuse",true},{"quality_accepted",false}});
 }
};
}
