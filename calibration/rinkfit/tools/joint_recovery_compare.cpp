#include "rinkfit/joint_recovery_metrics_export.hpp"
#include "rinkfit/joint_source_guard.hpp"
#include <iostream>
#include <unordered_map>

namespace {
template<class T>std::vector<T> checked_aux(const std::filesystem::path&root,const std::string&name,const rinkfit::eventio::Json&manifest){
 const auto&d=manifest.at("files").at(name);rinkfit::eventio::verify(root/name,d.at("sha256"));const auto bytes=rinkfit::eventio::bytes(root/name);
 if(bytes.size()!=d.at("size_bytes").get<std::size_t>()||bytes.size()%sizeof(T))throw std::runtime_error("old CP auxiliary byte shape changed");
 std::vector<T>v(bytes.size()/sizeof(T));std::memcpy(v.data(),bytes.data(),bytes.size());return v;
}

struct HistoricalRow {std::string source_dataset_id,source_row_id,shot_id,query_signature;std::int64_t pts=0;rinkfit::Vec2<double>uv{},normal{};int material=0,kind=0;bool has_query=false;double E=0;int stroke_pair_index=-1,stroke_source_side=-1;};
struct HistoricalRows {std::string source_hash;std::vector<HistoricalRow>rows;};
template<class T>std::vector<T> historical_array(const std::filesystem::path&base,const rinkfit::eventio::Json&d,std::size_t count){
 auto path=base/d.at("file").get<std::string>();rinkfit::eventio::verify(path,d.at("sha256"));auto bytes=rinkfit::eventio::bytes(path);const auto off=d.value("offset_bytes",std::size_t(0));std::size_t declared=1;for(const auto&v:d.at("shape"))declared*=v.get<std::size_t>();if(declared!=count||off+count*sizeof(T)>bytes.size()||off%sizeof(T))throw std::runtime_error("historical raw source array shape changed");std::vector<T>v(count);std::memcpy(v.data(),bytes.data()+off,count*sizeof(T));return v;
}
std::string query_signature(const rinkfit::JointQuery&q){
 rinkfit::eventio::Json j={{"alpha",q.alpha},{"native_uv",q.uv},{"numerator_x",q.nx},{"numerator_y",q.ny},{"denominator",q.den},{"alpha_origin",q.origin},{"alpha_scale",q.scale}};
 const auto text=j.dump();return rinkfit::eventio::sha256(std::vector<std::uint8_t>(text.begin(),text.end()));
}
HistoricalRows historical_rows(const std::string&file){
 auto j=rinkfit::eventio::read_json(file);const auto n=j.at("row_count").get<std::size_t>();const auto&a=j.at("arrays");const auto base=std::filesystem::path(file).parent_path();
 auto pts=historical_array<std::int64_t>(base,a.at("native_pts"),n);auto si=historical_array<std::uint16_t>(base,a.at("shot_index"),n);auto uv=historical_array<double>(base,a.at("uv"),2*n);auto mat=historical_array<std::uint32_t>(base,a.at("material_index"),n);auto kind=historical_array<std::uint8_t>(base,a.at("kind"),n);auto E=historical_array<double>(base,a.at("E"),n);
 std::vector<std::int32_t>pair(n,-1);std::vector<std::int8_t>side(n,-1);if(a.contains("stroke_pair_index"))pair=historical_array<std::int32_t>(base,a.at("stroke_pair_index"),n);if(a.contains("stroke_source_side"))side=historical_array<std::int8_t>(base,a.at("stroke_source_side"),n);
 HistoricalRows result;result.source_hash=rinkfit::eventio::file_sha256(file);result.rows.resize(n);
 for(std::size_t i=0;i<n;++i){auto&r=result.rows[i];r.source_row_id=j.at("source_row_ids").at(i);r.source_dataset_id=j.contains("source_dataset_ids")?j.at("source_dataset_ids").at(i).get<std::string>():"joint";r.shot_id=j.at("shot_ids").at(si[i]);r.pts=pts[i];r.uv={uv[2*i],uv[2*i+1]};r.material=mat[i];r.kind=kind[i];r.E=E[i];r.stroke_pair_index=pair[i];r.stroke_source_side=side[i];}
 if(j.contains("queries"))for(const auto&q:j.at("queries")){const auto i=q.at("row_index").get<std::size_t>();result.rows.at(i).has_query=true;result.rows[i].normal=q.at("normal").get<rinkfit::Vec2<double>>();rinkfit::JointQuery curve;curve.alpha=q.at("alpha").get<std::vector<double>>();curve.uv=q.at("native_uv").get<std::vector<rinkfit::Vec2<double>>>();curve.nx=q.value("numerator_x",std::vector<double>{});curve.ny=q.value("numerator_y",std::vector<double>{});curve.den=q.value("denominator",std::vector<double>{});curve.origin=q.value("alpha_origin",0.);curve.scale=q.value("alpha_scale",1.);result.rows[i].query_signature=query_signature(curve);}
 return result;
}
// Fresh sources have no historical CP. This is the production finite-path
// numerical seed, evaluated under the fixed historical physical model. It
// does not reclassify materials, change source UV or hide failed RS roots.
rinkfit::eventio::Json initialize_fresh_sources(rinkfit::JointProblemBuilder&b,const std::vector<std::size_t>&rows,std::vector<std::uint8_t>*status){
 using namespace rinkfit;eventio::Json ledger=eventio::Json::array();
 std::size_t fixed=0,sliding=0,rs_failed=0,no_finite=0;status->assign(b.packet.rows.size(),0);
 for(auto i:rows){auto&r=b.packet.rows.at(i);const auto&mat=b.model.materials.at(r.material);
  if(!r.active||mat.paths.size()!=1||r.path!=0||r.width_block>=0||r.stroke_pair_index>=0)throw std::runtime_error("fresh comparison requires an active unpaired source with its sole already-bound finite material path");
  if((r.kind==1)!=mat.paths.at(r.path).physical_point)throw std::runtime_error("fresh fixed entity/sliding material identity differs");
  if(r.kind!=0&&r.kind!=1)throw std::runtime_error("fresh comparison kind requires an explicit production initializer");
  const auto intervals=r.station_intervals(b.model,r.path);if(intervals.empty())throw std::runtime_error("fresh source has no finite legal station interval");
  if(r.kind==1&&!r.station_allowed(b.model,r.path,r.station))throw std::runtime_error("fixed entity station outside source/material domain");
  const auto l=r.layout(b.model);const auto ptr=b.model.pointers(l);const double observed_row=r.uv[1];double station=r.station,best=INFINITY;
  auto error=[&](double st){try{const auto p=b.model.predict_for_al(l,st,observed_row,ptr.data());return p.arithmetic_valid&&finite_array(p.uv)?std::hypot(p.uv[0]-r.uv[0],p.uv[1]-r.uv[1]):double(INFINITY);}catch(...){return double(INFINITY);}};
  if(r.kind==1){++fixed;best=error(station);}else{++sliding;for(const auto&interval:intervals){
   double local=INFINITY;int kbest=-1;for(int k=0;k<=32;++k){const double st=interval[0]+(interval[1]-interval[0])*k/32.,v=error(st);if(v<local){local=v;kbest=k;}}
   if(kbest<0||!std::isfinite(local))continue;double candidate=interval[0]+(interval[1]-interval[0])*kbest/32.;
   double lo=interval[0]+(interval[1]-interval[0])*std::max(0,kbest-1)/32.,hi=interval[0]+(interval[1]-interval[0])*std::min(32,kbest+1)/32.;
   for(int k=0;k<28;++k){const double x=lo+(hi-lo)*.3819660112501051,z=lo+(hi-lo)*.6180339887498949;if(error(x)<error(z))hi=z;else lo=x;}
   const double st=(lo+hi)*.5,v=error(st);if(v<local){local=v;candidate=st;}if(local<best){best=local;station=candidate;}
  }}
  if(!std::isfinite(best))++no_finite;const auto root=b.model.project_rs(l,station,observed_row);
  b.station[i]=r.station=station;b.row[i]=r.row=root.valid?root.row:observed_row;
  if(r.query){const auto p=b.model.predict_for_al(l,station,b.row[i],ptr.data());double value=INFINITY,alpha=r.alpha;
   if(p.arithmetic_valid)for(int k=0;k<=64;++k){const double x=r.query->alpha.front()+(r.query->alpha.back()-r.query->alpha.front())*k/64.;Vec2<double>q;if(r.query->point(x,&q)){const double v=std::hypot(q[0]-p.uv[0],q[1]-p.uv[1]);if(v<value){value=v;alpha=x;}}}b.alpha[i]=r.alpha=alpha;
  }
  (*status)[i]=root.valid?1:2;if(!root.valid)++rs_failed;
  ledger.push_back({{"row_index",i},{"dataset",r.source_dataset_id},{"source_ID",r.source_row_id},{"shot",r.shot_id},{"native_pts",r.pts},{"material",mat.label},{"kind",r.kind},{"path",r.path},{"station",station},{"row",r.row},{"alpha",r.alpha},{"has_Query",bool(r.query)},{"RS_root_valid",root.valid},{"finite_seed_available",std::isfinite(best)}});
 }
 return {{"schema","rinkfit.historical_model_new_source_numerical_seed.v1"},{"fresh_rows",rows.size()},{"fixed_entity_rows",fixed},{"finite_sliding_rows",sliding},{"failed_RS_rows",rs_failed},{"no_arithmetic_finite_seed_rows",no_finite},{"path_changed",false},{"source_UV_ID_material_operator_changed",false},{"Query_closure_claimed",false},{"policy","Same finite-path observed-row station and RS/Query-alpha seed as engineering initializer; fixed entities keep their station. Physical failures remain counted; independent replay separately attempts local closure."},{"rows",ledger}};
}
template<class T>rinkfit::eventio::Json write_provenance(const std::filesystem::path&out,const std::string&name,const std::vector<T>&v){
 std::ofstream f(out/name,std::ios::binary);f.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(T));f.close();if(!f)throw std::runtime_error("comparison provenance write failed");return {{"sha256",rinkfit::eventio::file_sha256(out/name)},{"size_bytes",std::filesystem::file_size(out/name)}};
}

}
int main(int argc,char**argv){try{
 if(argc!=6)throw std::runtime_error("usage: joint_recovery_compare MODEL CURRENT_PACKET OLD_PACKET OLD_EXPLICIT_AUX_MANIFEST CONFIG_JSON");
 const auto cfg=rinkfit::eventio::read_json(argv[5]);const std::filesystem::path out=cfg.at("output").get<std::string>();
 if(std::filesystem::exists(out))throw std::runtime_error("immutable comparison output exists");
 rinkfit::eventio::verify(argv[1],cfg.at("model_sha256"));
 if(cfg.contains("packet_sha256"))rinkfit::eventio::verify(argv[2],cfg.at("packet_sha256"));
 if(cfg.contains("old_aux_manifest_sha256"))rinkfit::eventio::verify(argv[4],cfg.at("old_aux_manifest_sha256"));
 auto phase=[](const char* name){std::cout<<"{\"comparison_phase\":\""<<name<<"\"}"<<std::endl;};
 auto model=rinkfit::JointEventModel::load(argv[1]);phase("model_loaded");auto packet=rinkfit::JointPacket::load(argv[2],model);phase("current_packet_loaded");
 rinkfit::assert_joint_board_source_qualification(packet,argv[2]);
 const auto recall=rinkfit::joint_source_recall_marker(packet.source_json,argv[2],true);if(!recall.empty())throw std::runtime_error("current source qualification recalled");
 const auto oldmeta=rinkfit::eventio::read_json(argv[4]);rinkfit::eventio::verify(argv[3],oldmeta.at("packet_sha256"));
 auto old=historical_rows(argv[3]);phase("historical_identity_arrays_loaded_without_old_W_graph");const std::filesystem::path auxroot=std::filesystem::path(argv[4]).parent_path();
 const auto station=checked_aux<double>(auxroot,"station.f64",oldmeta),row=checked_aux<double>(auxroot,"row.f64",oldmeta),alpha=checked_aux<double>(auxroot,"alpha.f64",oldmeta);const auto path=checked_aux<std::uint32_t>(auxroot,"path.u32",oldmeta);
 if(station.size()!=old.rows.size()||row.size()!=old.rows.size()||alpha.size()!=old.rows.size()||path.size()!=old.rows.size())throw std::runtime_error("old CP complete row state omitted");
 auto source_key=[](const auto&r){return r.source_dataset_id+"|"+r.shot_id+"|"+std::to_string(r.pts)+"|"+r.source_row_id;};
 std::unordered_map<std::string,std::size_t>lookup;for(std::size_t i=0;i<old.rows.size();++i)if(!lookup.emplace(source_key(old.rows[i]),i).second)throw std::runtime_error("old compound source keys duplicate");
 std::size_t identical=0,fresh=0;std::vector<std::size_t>fresh_rows;std::vector<std::uint64_t>original_row(packet.rows.size(),UINT64_MAX);std::vector<std::uint8_t>old_mask(packet.rows.size()),fresh_mask(packet.rows.size());for(std::size_t ri=0;ri<packet.rows.size();++ri){auto&o=packet.rows[ri];auto it=lookup.find(source_key(o));if(it==lookup.end()){if(o.active){++fresh;fresh_rows.push_back(ri);fresh_mask[ri]=1;}continue;}const auto i=it->second;original_row[ri]=i;const auto&a=old.rows[i];
  if(a.pts!=o.pts||a.shot_id!=o.shot_id||a.uv!=o.uv||a.material!=o.material||a.kind!=o.kind||a.has_query!=bool(o.query)||(a.has_query&&(a.normal!=o.normal||a.query_signature!=query_signature(*o.query)))||a.E!=o.E||a.stroke_pair_index!=o.stroke_pair_index||a.stroke_source_side!=o.stroke_source_side)throw std::runtime_error("same ID changed source clock/UV/material/operator");
  if(path[i]>=model.materials.at(o.material).paths.size())throw std::runtime_error("old material path absent from exact model");
  o.station=station[i];o.row=row[i];o.alpha=alpha[i];o.path=path[i];if(o.active){++identical;old_mask[ri]=1;}
 }
 // Rows absent from the old CP cannot be relabeled as old explicit state.
 phase("current_active_sources_matched_original_CP");
 if(fresh&&cfg.value("fresh_source_auxiliary_policy",std::string())!="qualified_single_path_production_seed_v1")throw std::runtime_error("current active IDs absent from old CP; explicitly declare fresh source initialization");
 lookup.clear();lookup.rehash(0);old.rows.clear();old.rows.shrink_to_fit();
 std::filesystem::create_directories(out);rinkfit::JointALBuilder b(std::move(model),std::move(packet),false,false);const int threads=cfg.value("num_threads",8);phase("explicit_CP_graph_ready");
 std::vector<std::uint8_t>fresh_status;const auto fresh_report=initialize_fresh_sources(b.core,fresh_rows,&fresh_status);
 for(std::size_t i=0;i<original_row.size();++i)if(original_row[i]!=UINT64_MAX){const auto j=original_row[i];const auto&r=b.core.packet.rows[i];if(b.core.station[i]!=station[j]||b.core.row[i]!=row[j]||b.core.alpha[i]!=alpha[j]||std::uint32_t(r.path)!=path[j])throw std::runtime_error("fresh initializer altered historical CP auxiliary");}
 rinkfit::eventio::atomic_json(out/"fresh_auxiliary_initialization.json",fresh_report);phase("fresh_single_path_numerical_seeds_ready");
 if(cfg.contains("gpu_raw_uv")){const auto&g=cfg.at("gpu_raw_uv");rinkfit::eventio::verify(g.at("ptx").get<std::string>(),g.at("ptx_sha256"));b.enable_gpu_raw_uv(g.at("ptx"),threads,out/"gpu_raw_uv_cache.jsonl");}
 phase("export_CP_explicit");
 auto explicit_report=rinkfit::export_joint_recovery_explicit_metrics(b,out/"CP_explicit_metrics",threads);
 explicit_report["files"]["historical_CP_rows.u8"]=write_provenance(out/"CP_explicit_metrics","historical_CP_rows.u8",old_mask);
 explicit_report["files"]["fresh_initializer_rows.u8"]=write_provenance(out/"CP_explicit_metrics","fresh_initializer_rows.u8",fresh_mask);
 explicit_report["files"]["fresh_initializer_status.u8"]=write_provenance(out/"CP_explicit_metrics","fresh_initializer_status.u8",fresh_status);
 explicit_report["files"]["original_CP_row_index.u64"]=write_provenance(out/"CP_explicit_metrics","original_CP_row_index.u64",original_row);
 explicit_report["historical_CP_rows"]=identical;explicit_report["fresh_numerical_seed_rows"]=fresh;
 explicit_report["auxiliary_origin_semantics"]="Historical exact ID/PTS/UV/operator rows retain original v98 CP auxiliaries; appended qualified rows have a fresh finite-path numerical seed under the unchanged v98 physical model and were never in the historical CP. Masks distinguish both origins.";
 explicit_report["valid_semantics"]=explicit_report.at("valid_semantics").get<std::string>()+"; "+explicit_report.at("auxiliary_origin_semantics").get<std::string>();
 rinkfit::eventio::atomic_json(out/"CP_explicit_metrics/manifest.json",explicit_report);
 b.export_explicit_nuisance(out/"explicit_nuisance");phase("replay_fixed_station_RS_and_legal_reprofile");const auto replay=b.core.replay(out/"independent_RS_state",true);
 rinkfit::eventio::Json report={{"schema","rinkfit.failed_v98_same_source_generation_comparison.v2"},{"model_sha256",rinkfit::eventio::file_sha256(argv[1])},{"current_packet_sha256",b.core.packet.source_hash},{"original_CP_packet_sha256",old.source_hash},{"old_aux_manifest_sha256",rinkfit::eventio::file_sha256(argv[4])},{"all_active_current_ID_UV_operator_matched_original_CP",fresh==0},{"same_original_CP_auxiliary_rows",identical},{"fresh_auxiliary_rows",fresh},{"fresh_initialization",fresh_report},{"comparison_denominator_rows",identical+fresh},{"independent_replay_scope","same fixed material path; fixed-station native RS and local finite Query nuisance profile exported separately; no chart/material reassignment or physical optimization"},{"source_UV_or_membership_modified",false},{"all_matched_original_CP_auxiliary_values_bit_identical",true},{"Ceres_problem_constructed",false},{"old_AL_duals_resumed",false},{"explicit",explicit_report},{"replay",replay}};
 rinkfit::eventio::atomic_json(out/"comparison_receipt.json",report);std::cout<<report.dump(2)<<std::endl;return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
