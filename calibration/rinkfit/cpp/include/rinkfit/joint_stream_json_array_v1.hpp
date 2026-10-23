#pragma once
#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>
namespace rinkfit {
// Syntax is checked by the public SAX parser. Only one top-level record is
// materialized; the caller separately verifies the immutable file SHA.
template<class Callback> std::size_t joint_stream_json_objects_v1(const std::filesystem::path& path,Callback callback){
 using Json=eventio::Json;
 struct Sax: nlohmann::json_sax<Json>{
  using string_t=Json::string_t;using binary_t=Json::binary_t;using number_integer_t=Json::number_integer_t;using number_unsigned_t=Json::number_unsigned_t;using number_float_t=Json::number_float_t;
  Callback& callback;Json record;std::vector<Json*> stack;string_t pending_key;std::size_t count=0,depth=0;bool opened=false,closed=false;
  explicit Sax(Callback& c):callback(c){}
  Json* append(Json value){if(stack.empty())throw std::runtime_error("source array requires object records");Json* p=stack.back();if(p->is_array()){p->push_back(std::move(value));return &p->back();}if(!p->is_object())throw std::runtime_error("invalid source container");Json& item=(*p)[pending_key];pending_key.clear();item=std::move(value);return &item;}
  bool null()override{append(nullptr);return true;}bool boolean(bool v)override{append(v);return true;}
  bool number_integer(number_integer_t v)override{append(v);return true;}bool number_unsigned(number_unsigned_t v)override{append(v);return true;}bool number_float(number_float_t v,const string_t&)override{append(v);return true;}
  bool string(string_t&v)override{append(v);return true;}bool binary(binary_t&v)override{append(Json::binary(v));return true;}bool key(string_t&v)override{pending_key=v;return true;}
  bool start_object(std::size_t)override{if(depth==1){record=Json::object();stack.push_back(&record);}else{if(depth<1)throw std::runtime_error("source array root required");stack.push_back(append(Json::object()));}++depth;return true;}
  bool end_object()override{if(stack.empty()||!stack.back()->is_object())throw std::runtime_error("invalid source object");stack.pop_back();--depth;if(depth==1){callback(record,count++);record=Json();}return true;}
  bool start_array(std::size_t)override{if(depth==0){if(opened)throw std::runtime_error("duplicate JSON root");opened=true;depth=1;return true;}stack.push_back(append(Json::array()));++depth;return true;}
  bool end_array()override{if(depth==1){depth=0;closed=true;return true;}if(stack.empty()||!stack.back()->is_array())throw std::runtime_error("invalid source array");stack.pop_back();--depth;return true;}
  bool parse_error(std::size_t position,const std::string&,const nlohmann::detail::exception&e)override{throw std::runtime_error("source JSON parse at byte "+std::to_string(position)+": "+e.what());}
 } sax(callback);
 std::ifstream stream(path);if(!stream)throw std::runtime_error("missing source record file "+path.u8string());
 if(!Json::sax_parse(stream,&sax)||!sax.opened||!sax.closed||sax.depth)throw std::runtime_error("incomplete source record array "+path.u8string());return sax.count;
}
}
