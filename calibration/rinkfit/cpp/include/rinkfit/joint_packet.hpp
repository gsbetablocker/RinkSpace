#pragma once
#include "rinkfit/joint_model.hpp"
#include <memory>
#include <string_view>
namespace rinkfit {
std::size_t joint_packet_raw_cache_bytes();
void release_joint_packet_raw_cache();
struct JointQuery {std::vector<double>alpha;std::vector<Vec2<double>>uv;std::vector<double>nx,ny,den;double origin=0,scale=1;template<class T>static T polynomial(const std::vector<double>&a,const T& x){T y=T(0);for(auto i=a.rbegin();i!=a.rend();++i)y=y*x+T(*i);return y;}template<class T>bool point(const T&a,Vec2<T>*p)const{if(alpha.size()<2||uv.size()!=alpha.size()||event_scalar(a)<alpha.front()||event_scalar(a)>alpha.back())return false;if(!den.empty()){T z=(a-T(origin))/T(scale),d=polynomial(den,z);if(std::abs(event_scalar(d))<1e-14)return false;*p={polynomial(nx,z)/d,polynomial(ny,z)/d};return finite_array(*p);}auto it=std::upper_bound(alpha.begin(),alpha.end(),event_scalar(a));auto i=it==alpha.end()?alpha.size()-2:std::size_t(it-alpha.begin()-1);T u=(a-T(alpha[i]))/T(alpha[i+1]-alpha[i]);*p={T(uv[i][0])*(T(1)-u)+T(uv[i+1][0])*u,T(uv[i][1])*(T(1)-u)+T(uv[i+1][1])*u};return finite_array(*p);}};
struct JointObservation {std::string source_dataset_id,source_row_id,shot_id,source_group_id;std::shared_ptr<const std::vector<int>>source_finite_paths;std::int64_t pts=0;Vec2<double>uv{},normal{};int numerical_shape_segment=-1;int material=-1,path=0,kind=0,width_block=-1,width_sign=0,stroke_pair_index=-1,stroke_source_side=-1;double E=2,station=.5,row=539.5,alpha=0;bool active=false,source_print_identity_reviewed=false;std::string fit_role;std::shared_ptr<const JointQuery>query;int station_domain_path=-1;std::vector<std::array<double,2>>source_station_intervals;std::vector<std::array<double,2>> numerical_continuous_board_intervals;
 std::vector<std::array<double,2>>station_intervals(const JointEventModel&model,int pi)const{if(!numerical_continuous_board_intervals.empty()){if(model.materials.at(material).paths.at(pi).boundary_segment!=-2)throw std::runtime_error("continuous boundary coordinate missing model binding");return numerical_continuous_board_intervals;}auto original=model.station_intervals(material,pi);if(source_station_intervals.empty())return original;std::vector<std::array<double,2>>out;if(pi!=station_domain_path)return out;for(const auto&a:original)for(const auto&b:source_station_intervals){const double lo=std::max(a[0],b[0]),hi=std::min(a[1],b[1]);if(lo<=hi)out.push_back({lo,hi});}return out;}
 template<class T>T station_domain(const JointEventModel&model,int pi,const T&s)const{T best=T(1);for(const auto&v:station_intervals(model,pi)){const T a=T(v[0])-s,b=s-T(v[1]);const T g=event_scalar(a)>event_scalar(b)?a:b;if(event_scalar(g)<event_scalar(best))best=g;}return best;}
 bool station_allowed(const JointEventModel&model,int pi,double s,double tol=0)const{return std::isfinite(s)&&station_domain(model,pi,s)<=tol;}
 JointRowLayout layout(const JointEventModel&model)const{return model.layout(model.shot_map.at(shot_id),material,path,pts,width_block,width_sign,station,numerical_shape_segment);}};
struct JointStatisticalGroup {std::string id,statistic_kind;std::vector<std::size_t>rows;std::size_t rank=0;std::vector<double>W;};
struct JointPacketColdLoadScopeV1 {std::shared_ptr<void> lifetime;std::function<eventio::Json()> statistics;eventio::Json report()const{return statistics?statistics():eventio::Json{{"enabled",false}};}};
// Current inspection rows retain immutable UTF8 columns instead of millions of
// individually allocated identity strings. Raw row indices and numeric state do
// not change. Only target rows own the expensive consumer objects.
struct JointCurrentRawColumnsV1 {
 struct Registry {std::size_t start=0,count=0;std::string dataset;std::vector<std::uint64_t>offsets;std::vector<std::uint8_t>utf8;std::map<int,std::shared_ptr<const std::vector<int>>>material_paths;};
 std::vector<std::string>prefix_datasets,prefix_ids,shot_ids,group_ids;
 std::vector<Registry>registries;std::vector<std::uint16_t>shot_indices;std::vector<std::uint8_t>roles,fit_mask,dense_path_masks;
 std::size_t dense_start=0;std::vector<std::uint32_t>group_indices;
 std::vector<std::pair<std::size_t,std::size_t>>query_indices;std::map<std::size_t,int>reviewed_print_paths;
};
struct JointPacket {std::uint64_t numerical_path_generation_v1=0;std::string schema="rinkfit.joint_packet.v1",source_hash;std::vector<JointObservation>rows;std::shared_ptr<const JointCurrentRawColumnsV1>current_raw_columns_v1;
 // Explicit inspection access: rows[i] always retains complete raw numeric
 // state; inactive current identity/Query/permission objects are decoded here.
 std::string_view raw_source_row_id_view_v1(std::size_t i)const;
 std::string_view raw_source_dataset_id_view_v1(std::size_t i)const;
 std::string_view raw_shot_id_view_v1(std::size_t i)const;
 std::string raw_source_row_id_v1(std::size_t i)const{return std::string(raw_source_row_id_view_v1(i));}
 std::string raw_source_dataset_id_v1(std::size_t i)const{return std::string(raw_source_dataset_id_view_v1(i));}
 std::string raw_shot_id_v1(std::size_t i)const{return std::string(raw_shot_id_view_v1(i));}
 // Resolve only the immutable base permission; no identity/Query/layout objects.
 // false means no base authority was declared; an explicit empty mask is true.
 bool raw_source_finite_paths_v1(std::size_t i,std::vector<int>*out)const;
 JointObservation raw_observation_v1(std::size_t i)const;
std::set<std::size_t> reviewed_native_inactive;std::vector<JointStatisticalGroup>groups;eventio::Json source_json,frozen_report_target;static JointPacket load(const std::string& path,const JointEventModel& model);static JointPacket load_current_v1(const eventio::Json&binding,const JointEventModel&model,const std::string&model_path);static JointPacket load_effective_v1(const eventio::Json&binding,const JointEventModel&model,const std::string&model_path,const eventio::Json&prepared=eventio::Json());static JointPacket load_effective_generation_v1(const std::string&path,const JointEventModel&model);static JointPacket load_effective_generation_from_parent_v1(const std::string&path,const JointEventModel&model,const JointPacket&parent,const std::filesystem::path&parent_path);static JointPacketColdLoadScopeV1 cold_load_scope_v1(const JointEventModel&model,bool reuse=true);};
}
