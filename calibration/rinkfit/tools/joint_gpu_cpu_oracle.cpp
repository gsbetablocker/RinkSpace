// Read-only CPU canonical-double / exact Jet1 oracle for joint GPU shadows.
// Shared block seeds are actual JointEventModel indices, not per-shot copies.
// C ABI consumes RAM batches; it never solves RS, changes a source, or writes J.
#include "rinkfit/joint_model.hpp"
#include <ceres/jet.h>
#include <atomic>
#include <thread>
#include <memory>
#include <limits>
#ifdef _WIN32
#define ORACLE_API extern "C" __declspec(dllexport)
#else
#define ORACLE_API extern "C"
#endif
namespace {
using namespace rinkfit;
using J=ceres::Jet<double,1>;
thread_local std::string last_error;
struct Oracle { JointEventModel model; std::vector<JointRowLayout> layouts; };
JointRowLayout read_layout(const eventio::Json& d,const JointEventModel&m){
 JointRowLayout l;
 l.block_ids=d.at("block_ids").get<std::vector<int>>();
 for(int id:l.block_ids)if(id<0||std::size_t(id)>=m.blocks.size())throw std::runtime_error("layout physical block out of range");
 l.scene_slot=d.at("scene_slot");l.rig_slot=d.at("rig_slot");l.optics_slot=d.at("optics_slot");
 l.reference_angles_slot=d.at("reference_angles_slot");l.geometry_slot=d.at("geometry_slot");
 l.advertisement_slot=d.at("advertisement_slot");l.width_slot=d.at("width_slot");l.width_sign=d.at("width_sign");
 l.control_ids=d.at("control_ids").get<std::vector<int>>();l.control_slots=d.at("control_slots").get<std::vector<int>>();
 l.control_offsets=d.at("control_offsets").get<std::vector<int>>();l.geometry_indices=d.at("geometry_indices").get<std::vector<int>>();
 l.geometry_offset=d.at("geometry_offset");l.shot_index=d.at("shot_index");l.material=d.at("material");l.path=d.at("path");
 if(l.shot_index<0||std::size_t(l.shot_index)>=m.shots.size())throw std::runtime_error("layout shot out of range");
 m.materials.at(l.material).paths.at(l.path);
 auto slot=[&](int s){if(s<0||std::size_t(s)>=l.block_ids.size())throw std::runtime_error("required layout slot out of range");};
 slot(l.scene_slot);slot(l.rig_slot);slot(l.optics_slot);slot(l.reference_angles_slot);
 if(l.block_ids.at(l.scene_slot)!=m.scene_block)throw std::runtime_error("layout scene block differs from actual shared gauge");
 if(l.control_ids.size()!=l.control_slots.size()||l.control_ids.size()!=l.control_offsets.size())throw std::runtime_error("layout control arrays differ");
 for(std::size_t i=0;i<l.control_slots.size();++i){slot(l.control_slots[i]);auto size=m.blocks.at(l.block_ids[l.control_slots[i]]).values.size();if(l.control_offsets[i]<0||std::size_t(l.control_offsets[i]+3)>size)throw std::runtime_error("layout packed control offset out of range");}
 for(int s:{l.geometry_slot,l.advertisement_slot,l.width_slot})if(s>=0)slot(s);
 if(l.geometry_slot>=0)for(int k:l.geometry_indices)if(k<0||std::size_t(k)>=m.blocks.at(l.block_ids[l.geometry_slot]).values.size())throw std::runtime_error("layout geometry index out of range");
 return l;
}
void one(const Oracle&o,std::int32_t li,double time,double s,double row,std::int32_t sb,std::int32_t sc,std::int32_t aux,double*out,std::int32_t*status){
 if(li<0||std::size_t(li)>=o.layouts.size()||!std::isfinite(time)||!std::isfinite(s)||!std::isfinite(row))throw std::runtime_error("oracle row arguments invalid");
 if(aux<0||aux>4||(aux&&sb>=0))throw std::runtime_error("physical and auxiliary seeds are mutually exclusive");
 if(sb>=0&&(std::size_t(sb)>=o.model.blocks.size()||sc<0||std::size_t(sc)>=o.model.blocks[sb].values.size()))throw std::runtime_error("actual shared block seed out of range");
 auto l=o.layouts[li];l.frame_time=time;auto ptr=o.model.pointers(l);
 auto q=o.model.predict_for_al(l,s,row,ptr.data());out[0]=q.uv[0];out[1]=q.uv[1];out[6]=q.depth;out[7]=q.ray_radius_sq;out[8]=q.brown_lower;
 status[0]=q.arithmetic_valid?1:0;status[1]=int(q.physical_status);
 std::vector<std::vector<J>> data;data.reserve(l.block_ids.size());std::vector<const J*>jp;jp.reserve(l.block_ids.size());
 for(int id:l.block_ids){data.emplace_back();auto&v=data.back();v.reserve(o.model.blocks[id].values.size());for(std::size_t k=0;k<o.model.blocks[id].values.size();++k){v.emplace_back(o.model.blocks[id].values[k]);if(id==sb&&int(k)==sc)v.back().v[0]=1.;}jp.push_back(v.data());}
 J sj(s),rj(row);if(aux==1)sj.v[0]=1.;if(aux==2)rj.v[0]=1.;
 auto j=o.model.predict_for_al(l,sj,rj,jp.data());out[2]=j.uv[0].v[0];out[3]=j.uv[1].v[0];out[4]=j.uv[0].a;out[5]=j.uv[1].a;
 status[2]=j.arithmetic_valid?1:0;status[3]=int(j.physical_status);
}
}
ORACLE_API const char*joint_oracle_last_error(){return last_error.c_str();}
ORACLE_API void*joint_oracle_create(const char*model,const char*layouts){
 try{last_error.clear();auto q=std::make_unique<Oracle>();q->model=JointEventModel::load(model);auto json=eventio::read_json(layouts);if(!json.is_array())throw std::runtime_error("unique layout table must be an array");q->layouts.reserve(json.size());for(const auto&d:json)q->layouts.push_back(read_layout(d,q->model));return q.release();}
 catch(const std::exception&e){last_error=e.what();return nullptr;}catch(...){last_error="unknown CPU oracle load error";return nullptr;}
}
ORACLE_API void joint_oracle_destroy(void*h){delete static_cast<Oracle*>(h);}
ORACLE_API std::int64_t joint_oracle_eval(void*h,std::uint64_t n,const std::int32_t*layout,const double*time,const double*station,const double*row,const std::int32_t*seed_block,const std::int32_t*seed_component,const std::int32_t*seed_aux,double*out9,std::int32_t*status4,std::int32_t threads){
 try{last_error.clear();if(!h||(!layout&&n)||(!time&&n)||(!station&&n)||(!row&&n)||(!out9&&n)||(!status4&&n)||threads<1||threads>64)throw std::runtime_error("invalid CPU oracle buffers/threads");
 const auto&o=*static_cast<Oracle*>(h);std::atomic<std::uint64_t>next{0},failed{0};
 auto worker=[&](){for(;;){const auto i=next.fetch_add(1);if(i>=n)break;auto*dst=out9+9*i;auto*st=status4+4*i;std::fill(dst,dst+9,std::numeric_limits<double>::quiet_NaN());std::fill(st,st+4,-1);try{one(o,layout[i],time[i],station[i],row[i],seed_block?seed_block[i]:-1,seed_component?seed_component[i]:-1,seed_aux?seed_aux[i]:0,dst,st);}catch(...){++failed;}}};
 std::vector<std::thread>pool;for(int k=1;k<threads;++k)pool.emplace_back(worker);worker();for(auto&t:pool)t.join();return std::int64_t(failed.load());}
 catch(const std::exception&e){last_error=e.what();return -1;}catch(...){last_error="unknown CPU oracle batch error";return -1;}
}
