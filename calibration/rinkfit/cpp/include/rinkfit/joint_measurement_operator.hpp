#pragma once
#include <cmath>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <set>
#include <unordered_map>
#include "rinkfit/event_io.hpp"
namespace rinkfit {
// Same byte grammar and hashes as measurement_operator.canonical_tree_sha256.
// Memoization is local to one call and keyed by exact primitive content.
// Every distinct node still uses SHA256; repeated leaf values reuse that digest.
class JointCanonicalSHA256Digest {
#ifdef _WIN32
 BCRYPT_ALG_HANDLE provider_=nullptr;BCRYPT_HASH_HANDLE hash_=nullptr;
#endif
 public:
 JointCanonicalSHA256Digest(){
#ifdef _WIN32
  if(BCryptOpenAlgorithmProvider(&provider_,BCRYPT_SHA256_ALGORITHM,nullptr,0)<0)throw std::runtime_error("cannot open canonical SHA256 provider");
  if(BCryptCreateHash(provider_,&hash_,nullptr,0,nullptr,0,BCRYPT_HASH_REUSABLE_FLAG)<0){BCryptCloseAlgorithmProvider(provider_,0);provider_=nullptr;hash_=nullptr;}
#endif
 }
 ~JointCanonicalSHA256Digest(){
#ifdef _WIN32
  if(hash_)BCryptDestroyHash(hash_);if(provider_)BCryptCloseAlgorithmProvider(provider_,0);
#endif
 }
 JointCanonicalSHA256Digest(const JointCanonicalSHA256Digest&)=delete;
 JointCanonicalSHA256Digest& operator=(const JointCanonicalSHA256Digest&)=delete;
 std::string hash(const std::string& data){
#ifdef _WIN32
  if(hash_){const auto* bytes=reinterpret_cast<const unsigned char*>(data.data());std::size_t remaining=data.size();while(remaining){const ULONG n=static_cast<ULONG>(std::min<std::size_t>(remaining,1u<<30));if(BCryptHashData(hash_,const_cast<PUCHAR>(bytes),n,0)<0)throw std::runtime_error("canonical SHA256 update failed");bytes+=n;remaining-=n;}
   std::array<unsigned char,32> digest{};if(BCryptFinishHash(hash_,digest.data(),static_cast<ULONG>(digest.size()),0)<0)throw std::runtime_error("canonical SHA256 finish failed");static const char hex[]="0123456789abcdef";std::string out(64,'0');for(int i=0;i<32;++i){out[2*i]=hex[digest[i]>>4];out[2*i+1]=hex[digest[i]&15];}return out;}
#endif
  return eventio::sha256(std::vector<std::uint8_t>(data.begin(),data.end()));
 }
};
class JointCanonicalTreeDigest {
 JointCanonicalSHA256Digest digest_;

 std::unordered_map<std::uint64_t,std::string> numbers_;
 std::unordered_map<std::string,std::string> strings_;
 std::array<std::string,3> atoms_;
 std::string digest(const std::string& data){return digest_.hash(data);}
 const std::string& string_digest(const std::string& value){auto found=strings_.find(value);if(found!=strings_.end())return found->second;return strings_.emplace(value,digest("s"+std::to_string(value.size())+":"+value)).first->second;}
 public:
 JointCanonicalTreeDigest():atoms_{digest("n"),digest("b0"),digest("b1")}{}
 std::string hash(const eventio::Json& value){
  if(value.is_null())return atoms_[0];if(value.is_boolean())return atoms_[value.get<bool>()?2:1];
  if(value.is_number()){
   const double x=value.get<double>();
   if(!std::isfinite(x))throw std::runtime_error("operator identity requires finite binary64 numbers");
   if(value.is_number_unsigned()&&(x>=std::ldexp(1.,64)||static_cast<std::uint64_t>(x)!=value.get<std::uint64_t>()))throw std::runtime_error("operator identity integer is not exact binary64");
   if(value.is_number_integer()&&!value.is_number_unsigned()&&(x>=std::ldexp(1.,63)||x < -std::ldexp(1.,63)||static_cast<std::int64_t>(x)!=value.get<std::int64_t>()))throw std::runtime_error("operator identity integer is not exact binary64");
   std::uint64_t bits;std::memcpy(&bits,&x,sizeof(bits));auto found=numbers_.find(bits);if(found!=numbers_.end())return found->second;
   static const char hex[]="0123456789abcdef";std::string data(17,'0');data[0]='d';for(int i=0;i<16;++i)data[i+1]=hex[(bits>>(60-4*i))&15];return numbers_.emplace(bits,digest(data)).first->second;
  }
  if(value.is_string())return string_digest(value.get_ref<const std::string&>());
  std::string data;
  if(value.is_array()){data="a"+std::to_string(value.size())+":";data.reserve(data.size()+64*value.size());for(const auto& x:value)data+=hash(x);}
  else if(value.is_object()){data="o"+std::to_string(value.size())+":";data.reserve(data.size()+128*value.size());for(const auto& x:value.items()){data+=string_digest(x.key());data+=hash(x.value());}}
  else throw std::runtime_error("unsupported operator identity JSON node");
  return digest(data);
 }
};
inline std::string joint_canonical_tree_sha256(const eventio::Json& value){return JointCanonicalTreeDigest{}.hash(value);}
inline eventio::Json joint_dependency_identity(const eventio::Json&value) {
 if(value.is_object()){auto out=eventio::Json::object();for(const auto&x:value.items())if(!(value.contains("sha256")&&(x.key()=="file"||x.key()=="path")))out[x.key()]=joint_dependency_identity(x.value());return out;}
 if(value.is_array()){auto out=eventio::Json::array();for(const auto&x:value)out.push_back(joint_dependency_identity(x));return out;}return value;
}
inline eventio::Json joint_measurement_operator_descriptor(const eventio::Json&doc) {
 const std::string schema="rinkfit.measurement_operator_binding.v1";
 const auto n=doc.at("row_count").get<std::size_t>();auto queries=eventio::Json::array();std::set<std::size_t>seen;
 for(auto q:doc.value("queries",eventio::Json::array())){const auto i=q.at("row_index").get<std::size_t>();if(i>=n||!seen.insert(i).second)throw std::runtime_error("duplicate or out-of-range measurement Query");q.erase("alpha0");queries.push_back(std::move(q));}
 auto arrays=eventio::Json::object();for(const char*key:{"uv","native_pts","shot_index","kind","material_index","admission","fit_role","profile_normal","query_uv","alpha_bounds","stroke_pair_index","stroke_source_side"})arrays[key]=joint_dependency_identity(doc.contains("arrays")?doc.at("arrays").value(key,eventio::Json{}):eventio::Json{});
 eventio::Json out={{"schema",schema},{"row_count",n},{"arrays",arrays},{"queries",queries},{"allowed_paths",doc.value("allowed_paths",eventio::Json::object())}};
 const auto prefix=doc.contains("dense_source_binding")?doc.at("dense_source_binding").at("row_start").get<std::size_t>():n;
 for(const char*key:{"source_row_ids","source_dataset_ids","source_family_ids"}){out[key]=nullptr;if(doc.contains(key)&&!doc.at(key).is_null()){out[key]=eventio::Json::array();const auto&v=doc.at(key);for(std::size_t i=0;i<std::min(prefix,v.size());++i)out[key].push_back(v.at(i));}}
 for(const char*key:{"dense_source_binding","source_station_domains","material_binding","stroke_pairs"})out[key]=joint_dependency_identity(doc.value(key,eventio::Json{}));return out;
}
inline eventio::Json joint_measurement_operator_binding(const eventio::Json&doc) {return {{"schema","rinkfit.measurement_operator_binding.v1"},{"sha256",joint_canonical_tree_sha256(joint_measurement_operator_descriptor(doc))}};}
inline void assert_joint_measurement_operator_binding(const eventio::Json&doc,const eventio::Json&cert) {
 if(!cert.contains("measurement_operator_binding")||cert.at("measurement_operator_binding").value("schema",std::string{})!="rinkfit.measurement_operator_binding.v1")throw std::runtime_error("legacy source qualification has no frozen measurement-operator binding; explicit source measurement revision is required before new fitting");
 if(cert.at("measurement_operator_binding")!=joint_measurement_operator_binding(doc))throw std::runtime_error("source qualification measurement operator differs: Query/window/normal/source/material/finite path domain");
}
}
