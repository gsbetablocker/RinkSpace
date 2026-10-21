#pragma once
#include "rinkfit/event_io.hpp"
#include <map>
#include <cctype>
#include <set>
namespace rinkfit {
// Exact old bytes may live in an explicitly named archive after public code or
// paths change. Ordinary file hashing and file reads retain physical semantics.
struct JointExactDependencyAliasesV1 {
 struct Entry{std::filesystem::path original,actual;std::string sha;};
 static std::map<std::pair<std::string,std::string>,Entry>&entries(){static std::map<std::pair<std::string,std::string>,Entry>x;return x;}
 static eventio::Json&uses(){static eventio::Json x=eventio::Json::array();return x;}
 static std::string key(const std::filesystem::path&p){auto s=std::filesystem::absolute(p).lexically_normal().generic_u8string();std::transform(s.begin(),s.end(),s.begin(),[](unsigned char c){return char(std::tolower(c));});return s;}
 static void install(const eventio::Json&r){const auto file=std::filesystem::u8path(r.at("file").get<std::string>());eventio::verify(file,r.at("sha256"));const auto j=eventio::read_json(file);if(j.at("schema")!="rinkfit.exact_dependency_aliases.v1")throw std::runtime_error("exact archive alias schema");entries().clear();uses()=eventio::Json::array();for(const auto&v:j.at("aliases")){Entry e{std::filesystem::u8path(v.at("legacy_path").get<std::string>()),std::filesystem::u8path(v.at("actual_file").get<std::string>()),v.at("sha256").get<std::string>()};if(e.original.extension()!=".py"||e.actual.extension()!=".py")throw std::runtime_error("exact archive resolver permits Python code only, never source UV or JSON");if(!e.original.is_absolute()||!e.actual.is_absolute()||key(e.original)==key(e.actual))throw std::runtime_error("exact archive alias must name separate absolute paths");eventio::verify(e.actual,e.sha);if(!entries().emplace(std::make_pair(key(e.original),e.sha),e).second)throw std::runtime_error("duplicate exact archive alias");}}
 static std::filesystem::path resolve(const std::filesystem::path&p,const std::string&sha){auto i=entries().find({key(p),sha});if(i==entries().end())return p;const auto&e=i->second;eventio::verify(e.actual,sha);eventio::Json use={{"legacy_path",e.original.u8string()},{"sha256",sha},{"actual_file",e.actual.u8string()},{"actual_file_sha256",eventio::file_sha256(e.actual)}};if(std::find(uses().begin(),uses().end(),use)==uses().end())uses().push_back(use);return e.actual;}
};
}
