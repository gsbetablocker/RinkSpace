#pragma once
#include "rinkfit/joint_current_parameter_transport_v1.hpp"
#include <limits>
#include <string_view>
#include <unordered_map>
namespace rinkfit {
// Preserve the exact serialized old prefix, checking its denominator before
// adding only the explicitly issued current tail. The ordinary RLE restorers
// subsequently validate material permissions and numerical shape cells.
inline std::string current_source_append_RLE_v1(const std::string&encoded,
    std::size_t prefix,const std::vector<std::int64_t>&tail){
 std::size_t pos=0,total=0;while(pos<encoded.size()){
  const auto colon=encoded.find(':',pos),end=encoded.find(';',pos);
  if(colon==std::string::npos||end==std::string::npos||colon>=end)throw std::runtime_error("current source append: invalid prefix RLE");
  std::size_t used=0;const auto count=std::stoull(encoded.substr(pos,colon-pos),&used);
  if(!count||used!=colon-pos||total>prefix||count>prefix-total)throw std::runtime_error("current source append: prefix RLE denominator");
  total+=count;pos=end+1;
 }
 if(total!=prefix)throw std::runtime_error("current source append: incomplete old RLE prefix");
 std::string out=encoded;for(std::size_t i=0;i<tail.size();){std::size_t j=i+1;while(j<tail.size()&&tail[j]==tail[i])++j;out+=std::to_string(j-i)+":"+std::to_string(tail[i])+";";i=j;}return out;
}
// New raw scalar parameters may appear between retained auxiliary families and
// motion/zoom/global blocks. A name map preserves the old ordered subsequence;
// it cannot create or reset a physical, motion, budget or global parameter.
template<class Saved,class Target,class CheckRaw>
std::vector<std::size_t> current_source_append_parameter_map_v1(const Saved&saved,
    const Target&target,const std::vector<std::size_t>&withdrawn,std::size_t prefix,
    std::size_t total,CheckRaw check_raw){
 auto need=[](bool ok,const char*why){if(!ok)throw std::runtime_error(std::string("current source append parameters: ")+why);};
 need(prefix<total&&std::is_sorted(withdrawn.begin(),withdrawn.end())&&std::adjacent_find(withdrawn.begin(),withdrawn.end())==withdrawn.end(),"explicit sorted withdrawal/tail domain");
 std::unordered_map<std::string_view,std::size_t>index;index.reserve(saved.size());
 for(std::size_t i=0;i<saved.size();++i)need(index.emplace(saved[i].name,i).second,"duplicate saved parameter name");
 std::vector<std::size_t>mapping;mapping.reserve(target.size());std::vector<unsigned char>used(saved.size());
 bool have_previous=false;std::size_t previous=0;
 for(const auto&z:target){const auto found=index.find(z.name);
  if(found!=index.end()){const auto i=found->second;need(!used[i]&&(!have_previous||i>previous),"retained parameters reordered");need(saved[i].values.size()==z.size,"retained parameter shape");used[i]=1;previous=i;have_previous=true;mapping.push_back(i);}
  else{std::size_t row=0;need(current_auxiliary_raw_row_v1(z.name,row)&&row>=prefix&&row<total&&z.size==1,"new parameter is not an explicitly appended tail scalar");mapping.push_back(std::numeric_limits<std::size_t>::max());}
 }
 std::vector<std::size_t>removed_rows;
 for(std::size_t i=0;i<saved.size();++i)if(!used[i]){const auto&a=saved[i];std::size_t row=0;
  need(current_auxiliary_raw_row_v1(a.name,row)&&row<prefix&&std::binary_search(withdrawn.begin(),withdrawn.end(),row)&&a.values.size()==1,"old non-withdrawn complete parameter disappeared");
  check_raw(a.name,row,a.values[0]);if(a.name.compare(0,4,"row/")==0)removed_rows.push_back(row);
 }
 std::sort(removed_rows.begin(),removed_rows.end());need(removed_rows==withdrawn,"withdrawn active row registry differs");return mapping;
}
}
