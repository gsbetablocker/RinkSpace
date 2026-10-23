#pragma once
#include "rinkfit/event_io.hpp"
#include <map>
namespace rinkfit {
struct JointArchivedSourceMembersV1 {
 struct Entry{eventio::Json file,archive;};
 static std::map<std::pair<std::string,std::string>,Entry>&entries(){static std::map<std::pair<std::string,std::string>,Entry>v;return v;}
 static std::string key(const std::string&p){return std::filesystem::absolute(std::filesystem::u8path(p)).lexically_normal().generic_u8string();}
 static void install(const eventio::Json&ref){const auto f=std::filesystem::u8path(ref.at("file").get<std::string>());eventio::verify(f,ref.at("sha256"));const auto j=eventio::read_json(f);if(j.at("schema")!="rinkfit.exact_archived_source_members.v1")throw std::runtime_error("source archive member schema");std::map<std::string,eventio::Json>archives;for(const auto&a:j.at("archives")){eventio::verify(std::filesystem::u8path(a.at("file").get<std::string>()),a.at("sha256"));archives.emplace(key(a.at("file")),a);}entries().clear();for(const auto&r:j.at("members")){const auto a=archives.find(key(r.at("archive")));if(a==archives.end())throw std::runtime_error("source member archive missing");Entry e{{{"file",r.at("file")},{"sha256",r.at("sha256")}},a->second};if(!entries().emplace(std::make_pair(key(r.at("archive")),r.at("member").get<std::string>()),std::move(e)).second)throw std::runtime_error("duplicate archived source member");}}
 static const Entry&resolve(const eventio::Json&r){const auto i=entries().find({key(r.at("archive")),r.at("member").get<std::string>()});if(i==entries().end()||i->second.file.at("sha256")!=r.at("sha256"))throw std::runtime_error("archived native source needs exact extracted member: "+r.at("member").get<std::string>());return i->second;}
};
}
