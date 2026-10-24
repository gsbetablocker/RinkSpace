#include "rinkfit/joint_capture_weights_v3.hpp"
#include <fstream>
#include <iostream>
using namespace rinkfit;
void need(bool x,const char*w){if(!x)throw std::runtime_error(w);}
void put(const std::filesystem::path&p,const std::string&s){std::ofstream f(p,std::ios::binary);f<<s;}
int main(int argc,char**argv){try{
 if(argc!=2)throw std::runtime_error("test directory required");const auto dir=std::filesystem::absolute(argv[1]);std::filesystem::create_directories(dir);
 const auto live=dir/"issuer.py",archived=dir/"original.py",data=dir/"source.json";put(live,"changed producer\n");put(archived,"exact old producer\n");put(data,"actual data bytes\n");
 const auto sha=eventio::file_sha256(archived);const eventio::Json ref={{"file",live.u8string()},{"sha256",sha}};
 bool rejected=false;try{capture_weights_v3_detail::checked(ref);}catch(...){rejected=true;}need(rejected,"Unregistered changed producer accepted");
 const auto manifest=dir/"aliases.json";eventio::atomic_json(manifest,{{"schema","rinkfit.exact_dependency_aliases.v1"},{"aliases",eventio::Json::array({{{"legacy_path",live.u8string()},{"actual_file",archived.u8string()},{"sha256",sha}}})}});
 JointExactDependencyAliasesV1::install({{"file",manifest.u8string()},{"sha256",eventio::file_sha256(manifest)}});
 need(capture_weights_v3_detail::checked(ref)==archived,"Budget issuer did not read exact archived Python");need(JointExactDependencyAliasesV1::uses().size()==1,"Alias consumption not reported");
 rejected=false;try{capture_weights_v3_detail::checked({{"file",data.u8string()},{"sha256",sha}});}catch(...){rejected=true;}need(rejected,"Data SHA changed through code alias");
 put(archived,"archive tampered\n");rejected=false;try{capture_weights_v3_detail::checked(ref);}catch(...){rejected=true;}need(rejected,"Tampered archive accepted");
 std::cout<<"{\"schema\":\"rinkfit.weight_source_exact_alias_regression.v1\",\"passed\":true,\"old_code_bytes_consumed\":true,\"data_or_JSON_alias_allowed\":false,\"tampered_archive_rejected\":true}"<<std::endl;return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
