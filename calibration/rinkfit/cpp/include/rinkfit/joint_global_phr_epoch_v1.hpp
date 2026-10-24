#pragma once
#include "rinkfit/event_io.hpp"
#include <memory>
namespace rinkfit {
// Publish only between solver epochs. Readers and accepted snapshots share an
// immutable payload; no primal iteration copies or serializes the dual arrays.
class JointGlobalPHRStateEpochV1 {
 struct Payload {eventio::Json value;std::string serialized,digest;explicit Payload(eventio::Json v):value(std::move(v)),serialized(value.dump()),digest(eventio::sha256(std::vector<std::uint8_t>(serialized.begin(),serialized.end()))) {}};
 std::shared_ptr<const Payload> payload_;
 static const eventio::Json& null_json(){static const eventio::Json v=nullptr;return v;}
 public:
 JointGlobalPHRStateEpochV1()=default;
 explicit JointGlobalPHRStateEpochV1(eventio::Json v){publish(std::move(v));}
 JointGlobalPHRStateEpochV1&operator=(eventio::Json v){publish(std::move(v));return *this;}
 void publish(eventio::Json v){if(v.is_null())payload_.reset();else payload_=std::make_shared<const Payload>(std::move(v));}
 const eventio::Json&json()const{return payload_?payload_->value:null_json();}
 operator const eventio::Json&()const{return json();}
 bool is_null()const{return !payload_;}
 const eventio::Json&at(const char*key)const{return json().at(key);}
 const eventio::Json&at(const std::string&key)const{return json().at(key);}
 const std::string&dump()const{static const std::string nil="null";return payload_?payload_->serialized:nil;}
 const std::string&digest()const{static const std::string nil;return payload_?payload_->digest:nil;}
 const void*epoch_identity()const{return payload_.get();}
 bool same_content(const JointGlobalPHRStateEpochV1&other)const{return payload_==other.payload_||digest()==other.digest();}
};
// Compatibility for isolated legacy JSON fixtures. Production selects the
// immutable epoch constructor, and exposes no mutable JSON through this view.
class JointGlobalPHRStateAccessV1 {
 eventio::Json*legacy_=nullptr;JointGlobalPHRStateEpochV1*epoch_=nullptr;
 public:JointGlobalPHRStateAccessV1(eventio::Json&v):legacy_(&v){}
 JointGlobalPHRStateAccessV1(JointGlobalPHRStateEpochV1&v):epoch_(&v){}
 const eventio::Json&json()const{return epoch_?epoch_->json():*legacy_;}
 bool is_null()const{return json().is_null();}
 const eventio::Json&at(const char*key)const{return json().at(key);}
 void publish(eventio::Json v){if(epoch_)epoch_->publish(std::move(v));else *legacy_=std::move(v);}
};
class JointCompactGlobalEpochV1 {
 JointGlobalPHRStateEpochV1*live_;JointGlobalPHRStateEpochV1 saved_;
 public:explicit JointCompactGlobalEpochV1(JointGlobalPHRStateEpochV1&live):live_(&live){capture();}
 void capture(){saved_=*live_;}void restore()const{*live_=saved_;}
 bool matches()const{return live_->same_content(saved_);}
 void append_token(std::uint64_t&h)const{if(saved_.is_null())return;const std::string tag="global_phr_immutable_epoch_sha256_v1:";for(const auto*bytes:{&tag,&saved_.digest()})for(const unsigned char c:*bytes){h^=c;h*=1099511628211ull;}}
 const void*epoch_identity()const{return saved_.epoch_identity();}
};
}
