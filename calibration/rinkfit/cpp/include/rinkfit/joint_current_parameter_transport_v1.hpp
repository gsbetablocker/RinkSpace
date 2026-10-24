#pragma once
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>
namespace rinkfit {
// Retained parameters stay in their original order. Physical block/scalar
// deletion requires an explicit model subset; scalar row auxiliaries may only
// disappear from the active registry for explicitly withdrawn fit rows. Their
// original values still agree with the preserved complete raw columns.
inline bool current_auxiliary_raw_row_v1(const std::string&name,std::size_t&row){
 for(const char*prefix:{"station/","row/","alpha/"}){const std::string p(prefix);if(name.compare(0,p.size(),p)!=0)continue;
  const auto digits=name.substr(p.size());if(digits.empty()||digits.find_first_not_of("0123456789")!=std::string::npos)return false;
  std::size_t used=0;row=std::stoull(digits,&used);return used==digits.size()&&std::to_string(row)==digits;
 }return false;
}
struct CurrentModelParameterSubsetV1 {std::string name;std::size_t old_size=0;std::vector<std::size_t> retained_coordinates;};
inline const CurrentModelParameterSubsetV1* current_model_parameter_subset_v1(const std::vector<CurrentModelParameterSubsetV1>&subsets,const std::string&name){for(const auto&s:subsets)if(s.name==name)return &s;return nullptr;}
template<class Saved,class Target,class CheckRaw>
std::vector<std::size_t> current_parameter_transport_map_v1(const Saved&saved,const Target&target,const std::vector<std::size_t>&withdrawn,CheckRaw check_raw,const std::vector<std::string>&removed_model_parameters={},const std::vector<CurrentModelParameterSubsetV1>&subsets={}){
 auto need=[](bool ok,const char*why){if(!ok)throw std::runtime_error(std::string("current parameter transport: ")+why);};
 need(std::is_sorted(withdrawn.begin(),withdrawn.end())&&std::adjacent_find(withdrawn.begin(),withdrawn.end())==withdrawn.end(),"withdrawn raw IDs must be sorted and unique");
 need(std::is_sorted(removed_model_parameters.begin(),removed_model_parameters.end())&&std::adjacent_find(removed_model_parameters.begin(),removed_model_parameters.end())==removed_model_parameters.end(),"removed model names must be sorted and unique");
 std::vector<std::string> removed_model,used_subsets;
 for(const auto&s:subsets){need(!s.name.empty()&&s.old_size>s.retained_coordinates.size()&&!s.retained_coordinates.empty(),"invalid physical coordinate subset");need(std::is_sorted(s.retained_coordinates.begin(),s.retained_coordinates.end())&&std::adjacent_find(s.retained_coordinates.begin(),s.retained_coordinates.end())==s.retained_coordinates.end()&&s.retained_coordinates.back()<s.old_size,"coordinate subset must be an ordered in-range subset");std::size_t raw=0;need(!current_auxiliary_raw_row_v1(s.name,raw)&&s.name!="budget_sq"&&s.name.rfind("static_motion/",0)!=0&&s.name.find("numerical_zoom_enclosure")==std::string::npos,"auxiliary/state coordinate removal is forbidden");}

 std::vector<std::size_t> map;map.reserve(target.size());std::vector<std::size_t>removed_rows;std::size_t t=0;
 for(std::size_t s=0;s<saved.size();++s){const auto&a=saved[s];if(t<target.size()&&a.name==target[t].name){const auto*subset=current_model_parameter_subset_v1(subsets,a.name);if(subset){need(a.values.size()==subset->old_size&&target[t].size==subset->retained_coordinates.size(),"explicit physical coordinate subset shape differs");used_subsets.push_back(a.name);}else need(a.values.size()==target[t].size,"retained parameter shape changed");map.push_back(s);++t;continue;}
  if(std::binary_search(removed_model_parameters.begin(),removed_model_parameters.end(),a.name)){std::size_t raw=0;need(!current_auxiliary_raw_row_v1(a.name,raw)&&a.name!="budget_sq"&&a.name.rfind("static_motion/",0)!=0&&a.name.find("numerical_zoom_enclosure")==std::string::npos,"only explicitly retired physical model parameters may be removed");removed_model.push_back(a.name);continue;}
  std::size_t row=0;need(current_auxiliary_raw_row_v1(a.name,row)&&std::binary_search(withdrawn.begin(),withdrawn.end(),row),"removed/reordered parameter is not an explicitly withdrawn row auxiliary");
  need(a.values.size()==1,"withdrawn auxiliary is not scalar");check_raw(a.name,row,a.values[0]);if(a.name.compare(0,4,"row/")==0)removed_rows.push_back(row);
 }
 need(t==target.size(),"new or reordered current parameter");need(used_subsets.size()==subsets.size(),"unused or duplicate physical coordinate subset");std::sort(removed_model.begin(),removed_model.end());need(removed_model==removed_model_parameters,"retired model declaration differs from actual removed parameters");std::sort(removed_rows.begin(),removed_rows.end());need(removed_rows==withdrawn,"withdrawal declaration does not match removed active rows");return map;
}
}
