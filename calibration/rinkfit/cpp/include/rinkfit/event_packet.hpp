#pragma once
#include "rinkfit/event_io.hpp"
#include "rinkfit/augmented_lagrangian.hpp"
#include <set>
#include <map>
#include <functional>

namespace rinkfit {
namespace eventpacket_detail {
// NumPy <Un uses fixed-width little-endian UTF-32, not UTF-8 JSON bytes.
inline std::vector<std::uint32_t> utf32(const std::string& text) {
  std::vector<std::uint32_t> result;
  for(std::size_t i=0;i<text.size();) {
    const auto first=static_cast<unsigned char>(text[i++]);std::uint32_t cp=0;int count=0;
    if(first<0x80)cp=first;
    else if(first>=0xc2&&first<=0xdf){cp=first&31;count=1;}
    else if(first>=0xe0&&first<=0xef){cp=first&15;count=2;}
    else if(first>=0xf0&&first<=0xf4){cp=first&7;count=3;}
    else throw std::runtime_error("invalid source identity UTF-8");
    const int encoded=count;
    while(count--){if(i==text.size())throw std::runtime_error("truncated source identity UTF-8");const auto next=static_cast<unsigned char>(text[i++]);if((next&0xc0)!=0x80)throw std::runtime_error("invalid source identity UTF-8 continuation");cp=(cp<<6)|(next&63);}
    if((encoded==1&&cp<0x80)||(encoded==2&&cp<0x800)||(encoded==3&&cp<0x10000)||cp>0x10ffff||(cp>=0xd800&&cp<=0xdfff))throw std::runtime_error("noncanonical source identity UTF-8");
    result.push_back(cp);
  }
  return result;
}
inline std::string numpy_unicode_hash(const std::vector<std::string>& table,const std::vector<std::uint32_t>& indices,std::size_t width) {
  std::vector<std::vector<std::uint32_t>> decoded;decoded.reserve(table.size());
  for(const auto& value:table){decoded.push_back(utf32(value));if(decoded.back().size()>width)throw std::runtime_error("source identity exceeds immutable NumPy Unicode width");}
  std::vector<std::uint8_t> bytes(indices.size()*width*4,0);
  for(std::size_t row=0;row<indices.size();++row){if(indices[row]>=decoded.size())throw std::runtime_error("source Unicode index out of range");const auto& value=decoded[indices[row]];for(std::size_t j=0;j<value.size();++j)for(int b=0;b<4;++b)bytes[(row*width+j)*4+b]=static_cast<std::uint8_t>(value[j]>>(8*b));}
  return eventio::sha256(bytes);
}
inline std::string double_hash(const std::vector<double>& values){std::vector<std::uint8_t> bytes(values.size()*sizeof(double));if(!bytes.empty())std::memcpy(bytes.data(),values.data(),bytes.size());return eventio::sha256(bytes);}
} // namespace eventpacket_detail
struct EventStatisticalGroup {
  std::string id, provenance;
  std::vector<std::size_t> rows;
  std::size_t rank=0;
  std::vector<double> W; // row-major rank x (2 * rows.size()), frozen full group operator
};
struct EventQueryRecord {
  std::size_t source_row=0;
  std::string provenance;
  std::vector<double> alpha,numerator_u,numerator_v,denominator;
  std::vector<std::array<double,2>> native_uv;
  double lower=0.,upper=0.,initial_alpha=0.;
};
struct EventSourceRun {
  std::string id,provenance;
  std::vector<std::size_t> rows;
  std::uint32_t variant_index=0;
  int frame=0;
  bool closed=false;
};
struct EventPacket {
  std::filesystem::path root,packet_path,semantic_path;
  eventio::Json manifest, semantics;
  std::string shot_id, source_hash, packet_hash, semantic_hash,query_hash;
  std::set<std::string> semantic_ancestor_hashes;
  eventio::Json confirmed_bindings;
  eventio::Json source_exclusion,active_statistics;
  std::vector<std::uint8_t> active_mask;
  std::vector<std::uint64_t> active_rows,excluded_rows;
  std::vector<EventStatisticalGroup> excluded_original_groups;
  std::string active_mask_hash,source_exclusion_hash,active_statistics_hash;
  bool has_source_exclusion=false;
  std::vector<double> uv, budget;
  std::vector<std::uint64_t> rows;
  std::vector<std::int64_t> pts;
  std::vector<std::int32_t> frames;
  std::vector<std::uint32_t> material, point;
  std::vector<std::uint32_t> resolved_material,resolved_variant;
  std::vector<std::string> material_labels, point_ids;
  std::vector<std::uint8_t> kind;
  std::vector<double> query_uv,normal,alpha_bounds;
  std::vector<std::uint32_t> variant;
  std::vector<ObservationSpec> observations;
  std::vector<EventStatisticalGroup> groups;
  std::vector<double> sigma;
  std::vector<std::uint64_t> ordinary_rows;
  std::vector<EventQueryRecord> queries;
  std::vector<int> query_index;
  std::vector<EventSourceRun> runs,audit_runs;
  eventio::Json source_order_overlay;
  std::string source_order_overlay_hash;
  bool has_source_order_overlay=false;
  // A separate, reviewed recovery of mislabeled native generator fragments.
  // Raw labels and the earlier A3 authorities remain immutable audit facts.
  eventio::Json source_generator_recovery;
  std::string source_generator_recovery_hash,source_generator_review_hash;
  bool has_source_generator_recovery=false;
  // Native component correspondence is an algorithm/source-topology layer.
  // It does not extend the earlier human or visual-generator authorities.
  eventio::Json source_component_recovery;
  std::string source_component_recovery_hash,source_component_evidence_hash;
  bool has_source_component_recovery=false;
  std::vector<std::string> variant_ids;
  std::vector<std::string> resolved_variant_ids;
  std::uint32_t model_material(std::size_t row)const{return resolved_material.empty()?material.at(row):resolved_material.at(row);}
  const std::string& model_variant_id(std::size_t row)const{return resolved_variant.empty()?variant_ids.at(variant.at(row)):resolved_variant_ids.at(resolved_variant.at(row));}
  bool is_active(std::size_t row)const{return active_mask.at(row)!=0;}
  std::size_t original_group_count=0,formal_residual_rows=0;
  bool statistics_complete=false;
  explicit EventPacket(const std::filesystem::path& path) {
    packet_path=std::filesystem::absolute(path);root=packet_path.parent_path();manifest=eventio::read_json(packet_path);packet_hash=eventio::file_sha256(packet_path);
    if(manifest.at("schema")!="rinkfit.event_packet.v1" || manifest.at("observations")!=182017 || manifest.at("frame_count")!=1248)throw std::runtime_error("expected exact whole-shot packet, not a fixture or subset");
    shot_id=manifest.at("shot_id");source_hash=manifest.at("immutable_source_hashes").at("observations.npz");
    const auto& a=manifest.at("arrays");
    uv=eventio::read_array<double>(root,a.at("native_uv"),"<f8");rows=eventio::read_array<std::uint64_t>(root,a.at("source_rows"),"<u8");
    pts=eventio::read_array<std::int64_t>(root,a.at("native_pts"),"<i8");frames=eventio::read_array<std::int32_t>(root,a.at("frame_index"),"<i4");
    material=eventio::read_array<std::uint32_t>(root,a.at("material_index"),"<u4");point=eventio::read_array<std::uint32_t>(root,a.at("point_id_index"),"<u4");budget=eventio::read_array<double>(root,a.at("engineering_budget_px"),"<f8");
    const auto& st=manifest.at("string_tables");eventio::verify(root/st.at("file").get<std::string>(),st.at("sha256"));const auto strings=eventio::read_json(root/st.at("file").get<std::string>());
    material_labels=strings.at("material_labels").get<std::vector<std::string>>();point_ids=strings.at("source_point_ids").get<std::vector<std::string>>();
    if(material_labels.size()!=171)throw std::runtime_error("material identity count is not 171");
    for(const auto& [name,desc]:manifest.at("assets").items())eventio::verify(root/name,desc.at("sha256"));
    const auto n=rows.size();if(n!=182017 || uv.size()!=2*n || pts.size()!=n || frames.size()!=n || material.size()!=n || point.size()!=n || budget.size()!=n)throw std::runtime_error("whole-source binary dimensions mismatch");
    kind.assign(n,0);variant.assign(n,0);query_uv.assign(2*n,0.);normal.assign(2*n,0.);alpha_bounds.assign(2*n,0.);
    active_mask.assign(n,1);active_rows=rows;
    std::set<std::string> keys;std::set<int> frame_set;
    for(std::size_t i=0;i<n;++i){if(rows[i]!=i || frames[i]<1 || frames[i]>1248 || pts[i]!=256256+256*(frames[i]-1) || material[i]>=material_labels.size() || point[i]>=point_ids.size() || !std::isfinite(uv[2*i]) || !std::isfinite(uv[2*i+1]) || budget[i]<=0. || !std::isfinite(budget[i]))throw std::runtime_error("source key/index/time/UV ABI mismatch at row "+std::to_string(i));
      ObservationSpec spec;spec.key=shot_id+":"+std::to_string(pts[i])+":"+point_ids[point[i]];spec.source_uv={uv[2*i],uv[2*i+1]};spec.E={budget[i]*budget[i],0.,0.,budget[i]*budget[i]};if(!keys.insert(spec.key).second)throw std::runtime_error("duplicate composite source key");observations.push_back(std::move(spec));frame_set.insert(frames[i]);}
    if(frame_set.size()!=1248)throw std::runtime_error("not all 1248 source frames covered");
  }
  void load_semantics(const std::filesystem::path& path) {
    semantic_path=std::filesystem::absolute(path);semantics=eventio::read_json(semantic_path);semantic_hash=eventio::file_sha256(semantic_path);const auto dir=semantic_path.parent_path();
    if(semantics.contains("source_component_parent_manifest")||semantics.contains("source_component_recovery")){load_source_component_recovery();return;}
    if(semantics.contains("source_generator_parent_manifest")||semantics.contains("source_generator_recovery")){load_source_generator_recovery();return;}
    if(semantics.contains("source_order_parent_manifest")||semantics.contains("source_order_overlay")){load_source_order_evidence();return;}
    if(!groups.empty()||!runs.empty())throw std::runtime_error("semantic packet must be loaded exactly once");
    if(semantics.at("schema")!="rinkfit.semantic_packet.v1"||semantics.at("observations")!=182017||semantics.at("shot_id")!=shot_id||semantics.at("frames")!=1248||semantics.at("material_labels")!=171)throw std::runtime_error("semantic/source identity mismatch");
    if(manifest.contains("semantic_packet")&&manifest.at("semantic_packet").at("sha256")!=semantic_hash)throw std::runtime_error("semantic manifest differs from immutable packet binding");
    semantic_ancestor_hashes.insert(semantic_hash);
    if(semantics.contains("parent_semantic_manifest")){
      const auto& ref=semantics.at("parent_semantic_manifest");const auto parent_path=dir/ref.at("path").get<std::string>();eventio::verify(parent_path,ref.at("sha256"));const auto parent=eventio::read_json(parent_path);
      if(parent.contains("parent_semantic_manifest")||parent.contains("resolved_material_index"))throw std::runtime_error("nested material resolution requires an explicitly composed confirmation packet");
      for(const auto& item:parent.items()){
        if(item.key()=="arrays"||item.key()=="assets")continue;
        if(!semantics.contains(item.key())||semantics.at(item.key())!=item.value())throw std::runtime_error("semantic overlay changed original metadata: "+item.key());
      }
      for(const char* section:{"arrays","assets"}){
        if(semantics.at(section).size()!=parent.at(section).size())throw std::runtime_error("semantic overlay changed original array/asset inventory");
        for(const auto& item:parent.at(section).items()){
          const auto& current=semantics.at(section).at(item.key());
          for(const auto& field:item.value().items())if(field.key()!="file"&&(!current.contains(field.key())||current.at(field.key())!=field.value()))throw std::runtime_error("semantic overlay changed frozen source/W/query descriptor: "+item.key());
          const auto previous=parent_path.parent_path()/item.value().at("file").get<std::string>();const auto present=dir/current.at("file").get<std::string>();
          eventio::verify(previous,item.value().at("sha256"));eventio::verify(present,current.at("sha256"));
        }
      }
      semantic_ancestor_hashes.insert(ref.at("sha256").get<std::string>());
    }else if(semantics.contains("resolved_material_index")||semantics.contains("resolved_variant_index")||semantics.contains("confirmed_bindings"))throw std::runtime_error("resolved material identities require a verified immutable semantic parent");
    const auto& immutable=semantics.at("immutable_source_arrays");
    auto check_source=[&](const std::string& name,const std::string& dtype,const eventio::Json& shape,const std::string& hash){const auto& d=immutable.at(name);if(d.at("dtype_str")!=dtype||d.at("shape")!=shape||d.at("sha256_c_contiguous_array_bytes")!=hash)throw std::runtime_error("immutable source array identity mismatch: "+name);};
    const auto n=rows.size();const eventio::Json scalar_shape=eventio::Json::array({n});
    check_source("source_point_ids","<U69",scalar_shape,eventpacket_detail::numpy_unicode_hash(point_ids,point,69));
    check_source("source_target_labels","<U50",scalar_shape,eventpacket_detail::numpy_unicode_hash(material_labels,material,50));
    const auto& source_arrays=manifest.at("arrays");
    check_source("native_pts","<i8",scalar_shape,source_arrays.at("native_pts").at("sha256"));
    check_source("frame_indices","<i4",scalar_shape,source_arrays.at("frame_index").at("sha256"));
    check_source("source_uv","<f8",eventio::Json::array({n,2}),source_arrays.at("native_uv").at("sha256"));
    if(semantics.contains("arrays")){
      const auto& a=semantics.at("arrays");
      auto read=[&](const std::string& name){return a.contains(name);};
      if(read("kind"))kind=eventio::read_array<std::uint8_t>(dir,a.at("kind"),"|u1");
      if(read("variant_index"))variant=eventio::read_array<std::uint32_t>(dir,a.at("variant_index"),"<u4");
      if(read("query_uv"))query_uv=eventio::read_array<double>(dir,a.at("query_uv"),"<f8");
      if(read("profile_normal"))normal=eventio::read_array<double>(dir,a.at("profile_normal"),"<f8");
      else if(read("normal"))normal=eventio::read_array<double>(dir,a.at("normal"),"<f8");
      if(read("source_sigma"))sigma=eventio::read_array<double>(dir,a.at("source_sigma"),"<f8");
      if(read("alpha_bounds"))alpha_bounds=eventio::read_array<double>(dir,a.at("alpha_bounds"),"<f8");
      if(kind.size()!=rows.size() || variant.size()!=rows.size() || query_uv.size()!=2*rows.size() || normal.size()!=2*rows.size() || alpha_bounds.size()!=2*rows.size())throw std::runtime_error("semantic source dimensions mismatch");
      std::array<std::size_t,3> counts{};for(auto k:kind){if(k>2)throw std::runtime_error("unknown source observation semantic kind");++counts[k];}
      if(counts!=std::array<std::size_t,3>{179364,1692,961}||semantics.at("semantic_counts").at("finite_curve")!=counts[0]||semantics.at("semantic_counts").at("entity_corner")!=counts[1]||semantics.at("semantic_counts").at("true_profile")!=counts[2])throw std::runtime_error("original observation semantics inventory changed");
      for(const auto* values:{&query_uv,&normal,&alpha_bounds})for(double value:*values)if(!std::isfinite(value))throw std::runtime_error("nonfinite source profile metadata");
      for(std::size_t i=0;i<rows.size();++i)if(kind[i]==2){auto& o=observations[i];o.semantics=ObservationSemantics::kGenuineProfile;o.explicit_profile_query=true;o.profile_query_uv={query_uv[2*i],query_uv[2*i+1]};o.profile_normal={normal[2*i],normal[2*i+1]};o.profile_radius_px=budget[i];o.profile_alpha_lower_px=alpha_bounds[2*i];o.profile_alpha_upper_px=alpha_bounds[2*i+1];if(std::abs(std::hypot(o.profile_normal[0],o.profile_normal[1])-1.)>1e-8 || o.profile_alpha_lower_px>o.profile_alpha_upper_px)throw std::runtime_error("invalid genuine profile provenance/query");}
      if(read("group_rows") && read("weights")){
       const auto membership=eventio::read_array<std::uint64_t>(dir,a.at("group_rows"),"<u8");const auto W=eventio::read_array<double>(dir,a.at("weights"),"<f8");
       const auto group_path=dir/(semantics.contains("assets")?semantics.at("assets").at("groups.jsonl").at("file").get<std::string>():semantics.value("groups_file",std::string("groups.jsonl")));std::ifstream in(group_path);if(!in)throw std::runtime_error("missing original covariance groups");std::string line;std::vector<int> membership_count(rows.size(),0);std::set<std::string> group_ids;std::size_t used_members=0,used_weights=0;
       for(double value:W)if(!std::isfinite(value))throw std::runtime_error("nonfinite original full W coefficient");
       while(std::getline(in,line)){if(line.empty())continue;const auto g=eventio::Json::parse(line);EventStatisticalGroup out;out.id=g.at("group_id");out.rank=g.at("rank");out.provenance=g.value("covariance_origin",std::string());const auto s=g.at("member_start").get<std::size_t>(),m=g.at("member_count").get<std::size_t>(),w=g.at("weight_start").get<std::size_t>();
        if(!m||s!=used_members||w!=used_weights||s>membership.size()||m>membership.size()-s||w>W.size())throw std::runtime_error("original group covariance offsets mismatch");
        if(!out.rank||out.rank>2*m||out.rank>(W.size()-w)/(2*m)||out.provenance.empty()||!group_ids.insert(out.id).second||g.at("matrix_shape")!=eventio::Json::array({out.rank,2*m})||!g.at("one_statistical_measurement_per_original_group").get<bool>())throw std::runtime_error("original group covariance rank/provenance mismatch");
        const auto group_frame=g.at("frame").get<int>();const auto group_pts=g.at("native_pts").get<std::int64_t>();
        for(std::size_t j=0;j<m;++j){const auto row=membership[s+j];if(row>=rows.size()||frames[row]!=group_frame||pts[row]!=group_pts)throw std::runtime_error("group row/time identity mismatch");out.rows.push_back(row);++membership_count[row];}
        out.W.assign(W.begin()+w,W.begin()+w+out.rank*2*m);if(eventpacket_detail::double_hash(out.W)!=g.at("matrix_sha256"))throw std::runtime_error("original full group W bytes differ from frozen matrix");
        used_members+=m;used_weights+=out.rank*2*m;groups.push_back(std::move(out));}
       if(used_members!=membership.size()||used_weights!=W.size())throw std::runtime_error("unaccounted original W or group membership entries");
       original_group_count=groups.size();
       if(!read("ordinary_rows")||sigma.size()!=rows.size())throw std::runtime_error("missing original independent scalar statistics");
       ordinary_rows=eventio::read_array<std::uint64_t>(dir,a.at("ordinary_rows"),"<u8");
       for(auto row:ordinary_rows){if(row>=rows.size()||!(sigma[row]>0)||!std::isfinite(sigma[row]))throw std::runtime_error("invalid original independent row/sigma");
         ++membership_count[row];EventStatisticalGroup one;one.id="independent/"+std::to_string(row);one.provenance="original independent scalar source_sigma; immutable semantic packet "+semantic_hash;one.rows={std::size_t(row)};one.rank=2;one.W={1./sigma[row],0.,0.,1./sigma[row]};groups.push_back(std::move(one));}
       formal_residual_rows=0;for(const auto& group:groups)formal_residual_rows+=group.rank;
       statistics_complete=std::all_of(membership_count.begin(),membership_count.end(),[](int k){return k==1;});if(!statistics_complete||original_group_count!=9173||ordinary_rows.size()!=87967||formal_residual_rows!=214112)throw std::runtime_error("full original W plus independent rows must account for each source exactly once with unchanged rank");
      }
    }
    if(!statistics_complete)throw std::runtime_error("complete original statistical operator is mandatory");
    if(semantics.contains("assets"))for(const auto& item:semantics.at("assets").items())eventio::verify(dir/item.value().at("file").get<std::string>(),item.value().at("sha256"));
    variant_ids=semantics.at("variant_ids").get<std::vector<std::string>>();for(auto v:variant)if(v>=variant_ids.size())throw std::runtime_error("semantic variant index out of range");
    const auto runs_path=dir/(semantics.contains("assets")?semantics.at("assets").at("source_runs.jsonl").at("file").get<std::string>():std::string("source_runs.jsonl"));std::ifstream input(runs_path);if(!input)throw std::runtime_error("missing verified source runs");std::string line;while(std::getline(input,line)){if(line.empty())continue;const auto j=eventio::Json::parse(line);if(!j.value("source_order_verified",false))throw std::runtime_error("unverified source run");EventSourceRun run;run.rows=j.at("source_rows").get<std::vector<std::size_t>>();run.frame=j.at("frame");run.variant_index=j.at("variant_index");run.closed=j.value("source_closed",false);run.provenance=j.at("provenance");run.id=std::to_string(run.frame)+"/"+j.at("source_feature_id").get<std::string>()+"/"+std::to_string(j.at("source_segment_index").get<int>());if(run.rows.size()<2)throw std::runtime_error("empty ordered source run");for(auto row:run.rows)if(row>=rows.size()||frames[row]!=run.frame||variant[row]!=run.variant_index)throw std::runtime_error("run row/source identity mismatch");runs.push_back(std::move(run));}
    if(runs.size()!=15419)throw std::runtime_error("verified run inventory mismatch");
    for(std::size_t index=0;index<runs.size();++index){std::ostringstream id;id<<"run/"<<std::setw(6)<<std::setfill('0')<<index;runs[index].id=id.str();}
    if(semantics.contains("parent_semantic_manifest"))load_confirmed_resolution(dir);
    if(semantics.contains("source_exclusion"))load_source_exclusion(dir);
    else if(semantics.contains("source_exclusion_parent_manifest")||semantics.contains("active_source_mask")||semantics.contains("active_statistics"))throw std::runtime_error("incomplete source-exclusion layer");
  }
  // This narrow policy retires only a proven normal-query navigation registry.
  // The complete original material/exclusion/W/query authority loader runs first.
  void load_source_component_recovery(){
    const auto selected_path=semantic_path;const auto selected_hash=semantic_hash;const auto selected=semantics;const auto selected_manifest=manifest;const auto dir=selected_path.parent_path();
    if(!groups.empty()||!runs.empty()||!selected.contains("source_component_parent_manifest")||!selected.contains("source_component_recovery"))throw std::runtime_error("incomplete/repeated source-component recovery layer");
    if(manifest.at("semantic_packet").at("sha256")!=selected_hash)throw std::runtime_error("component recovery differs from immutable packet binding");
    const std::string parent_hash="cc50c4117d48e0e0875c08c1ad775682fe6198aae63edab7a71aa510135b0bb5";
    // Only exact immutable source-only evidence wrappers are admitted. A missing
    // or unknown policy never authorizes a component reassignment.
    const auto evidence_policies=eventio::Json::parse(R"({"56d67d9b96897dacc8ee0f799ff1e8d73d811dc09174550c65e6bb8a2c292186":{"changed":68,"targets":{"Ad5.print.component016":3,"Ad5.print.component009":2,"Ad5.print.component017_inner":1,"Ad5.print.component014_inner":3,"Ad5.print.component013_inner":4,"Ad5.print.component011_inner":3,"Ad5.print.component020":4,"Ad5.print.component018":14,"Ad5.print.component015":33,"Ad5.print.component010_inner":1},"facts_sha256":"606529dc5f1850a27a0463e7c2584ac86f21b4532b4729f2799088e820ffa1e3","inventory_sha256":"241c68e8302d857fbf7716d3dfc9ca4dbd2a1014b05cb359233e511199dd4d99"},"230162e0b0a95c7221d5427e3174df4a238c34dfd4daa76b87a4bbccb924ad2c":{"changed":84,"targets":{"Ad5.print.component016":3,"Ad5.print.component009":2,"Ad5.print.component017_inner":1,"Ad5.print.component014_inner":3,"Ad5.print.component013_inner":4,"Ad5.print.component011_inner":3,"Ad5.print.component020":12,"Ad5.print.component018":14,"Ad5.print.component015":33,"Ad5.print.component010_inner":1,"Ad5.print.component021_inner":8},"facts_sha256":"83e8372e9b5df5e3bc80fab8c351faa028214e0850bead6702b8389808990210","inventory_sha256":"241c68e8302d857fbf7716d3dfc9ca4dbd2a1014b05cb359233e511199dd4d99"},"9279ea527a9f04c668d62daafb65060bfe3136a9e05ae3481c408775c5320eac":{"changed":262,"targets":{"Ad5.print.component016":5,"Ad5.print.component009":2,"Ad5.print.component017_inner":19,"Ad5.print.component014_inner":22,"Ad5.print.component013_inner":20,"Ad5.print.component011_inner":10,"Ad5.print.component020":53,"Ad5.print.component018":31,"Ad5.print.component015":46,"Ad5.print.component010_inner":15,"Ad5.print.component021_inner":26,"Ad5.print.component012":2,"Ad5.print.component019":11},"facts_sha256":"9ead29a590224abe1d6f6b8309e0620098331af5da472ce73773ef371ce576f0","inventory_sha256":"241c68e8302d857fbf7716d3dfc9ca4dbd2a1014b05cb359233e511199dd4d99"}})");
    std::map<std::string,std::string> verified_files;auto file=[&](const std::filesystem::path& base,const eventio::Json& ref){const char* key=ref.contains("file")?"file":"path";const auto p=(base/std::filesystem::u8path(ref.at(key).get<std::string>())).lexically_normal();const auto k=p.u8string(),hash=ref.at("sha256").get<std::string>();const auto previous=verified_files.find(k);if(previous!=verified_files.end()){if(previous->second!=hash)throw std::runtime_error("conflicting source-component asset identity");return p;}eventio::verify(p,hash);verified_files.emplace(k,hash);return p;};
    const auto& pref=selected.at("source_component_parent_manifest");if(pref.at("sha256")!=parent_hash)throw std::runtime_error("component correspondence has another semantic parent");
    const auto proof_path=file(dir,selected.at("source_component_recovery"));const auto proof=eventio::read_json(proof_path);const auto pd=proof_path.parent_path();
    const std::set<std::string> proof_keys={"schema","rule_id","parent_semantic_sha256","observations","evidence_manifest","bindings"};
    if(proof.size()!=proof_keys.size())throw std::runtime_error("component recovery has an unrelated authority field");for(const auto& item:proof.items())if(!proof_keys.count(item.key()))throw std::runtime_error("unknown source-component recovery proof field");
    if(proof.at("schema")!="rinkfit.source_component_recovery.v1"||proof.at("rule_id")!="native_source_component_correspondence_v1"||proof.at("parent_semantic_sha256")!=parent_hash||proof.at("observations")!=rows.size())throw std::runtime_error("source-component correspondence proof scope mismatch");
    const auto evidence_hash=proof.at("evidence_manifest").at("sha256").get<std::string>();if(!evidence_policies.contains(evidence_hash))throw std::runtime_error("unpinned native component correspondence evidence");
    const auto& policy=evidence_policies.at(evidence_hash);const auto changed_count=policy.at("changed").get<std::size_t>();
    const auto evidence_path=file(pd,proof.at("evidence_manifest"));const auto evidence=eventio::read_json(evidence_path);const auto ed=evidence_path.parent_path();
    if(evidence.at("schema")!="rinkfit.native_source_component_correspondence_evidence.v1"||evidence.at("rule_id")!="native_source_component_correspondence_v1"||evidence.at("origin")!="native_source_image_and_component_topology_crosscheck"||evidence.at("parent_semantic_sha256")!=parent_hash||evidence.at("source_identity")!=source_hash||evidence.at("raw_observations")!=rows.size()||evidence.at("active_observations")!=182016||evidence.at("bindings")!=proof.at("bindings")||evidence.at("bindings").size()!=changed_count||evidence.at("changed_rows")!=changed_count||evidence.at("target_counts")!=policy.at("targets"))throw std::runtime_error("component recovery differs from exact pinned source correspondence");
    for(const char* name:{"not_a_direct_human_confirmation","no_extension_of_any_human_confirmation","no_human_anchor_used","current_model_nearest_label_is_not_identity_evidence","raw_UV_ID_labels_W_E_and_existing_authority_unchanged"})if(!evidence.at(name).is_boolean()||!evidence.at(name).get<bool>())throw std::runtime_error("component correspondence changed source-only authority scope");
    for(const char* forbidden:{"direct_human_confirmation","source_image_review","review_origin","confirmation_scope","review_scope","human_anchor"})if(evidence.contains(forbidden))throw std::runtime_error("component correspondence cannot impersonate an earlier confirmation/review authority");
    const auto parent_path=file(dir,pref);EventPacket inherited(packet_path);inherited.manifest["semantic_packet"]={{"manifest",parent_path.u8string()},{"sha256",parent_hash}};inherited.load_semantics(parent_path);
    if(inherited.has_source_component_recovery||!inherited.has_source_generator_recovery||inherited.source_generator_review_hash!="5620ec6a2519821782f4d181f840eff251c03e44baf19fb1d8cca6837b160e69"||inherited.source_generator_recovery.at("bindings").size()!=108||!inherited.has_source_order_overlay||!inherited.has_source_exclusion||inherited.active_rows.size()!=182016||inherited.audit_runs.size()!=15419||!inherited.runs.empty())throw std::runtime_error("component recovery requires the complete unchanged generator108/A3/exclusion/order parent");
    const auto parent=inherited.semantics;
    std::function<eventio::Json(const eventio::Json&)> signature=[&](const eventio::Json& v)->eventio::Json{if(v.is_array()){auto a=eventio::Json::array();for(const auto& x:v)a.push_back(signature(x));return a;}if(!v.is_object())return v;auto o=eventio::Json::object();for(const auto& item:v.items())if(!(v.contains("sha256")&&(item.key()=="file"||item.key()=="path")))o[item.key()]=signature(item.value());return o;};
    std::function<void(const eventio::Json&)> verify_refs=[&](const eventio::Json& v){if(v.is_object()){if(v.contains("sha256")&&(v.contains("file")||v.contains("path")))file(dir,v);for(const auto& x:v.items())verify_refs(x.value());}else if(v.is_array())for(const auto& x:v)verify_refs(x);};
    if(selected.size()!=parent.size()+2)throw std::runtime_error("component recovery changed unrelated field inventory");
    const std::set<std::string> changed_fields={"resolved_material_index","resolved_variant_index"};
    for(const auto& item:parent.items()){
      if(!selected.contains(item.key()))throw std::runtime_error("component recovery removed an inherited field");const auto& now=selected.at(item.key());
      if(changed_fields.count(item.key())){auto oldsig=signature(item.value()),newsig=signature(now);oldsig.erase("sha256");newsig.erase("sha256");if(oldsig!=newsig)throw std::runtime_error("component recovery changed resolved-array dtype/shape/metadata");file(dir,now);continue;}
      if(signature(now)!=signature(item.value()))throw std::runtime_error("component recovery changed raw/W/E/Q/generator108/A3/exclusion/order authority: "+item.key());
      if(item.key()=="arrays"||item.key()=="assets")verify_refs(now);
      else if(std::set<std::string>{"parent_semantic_manifest","original_material_index","confirmed_bindings","source_exclusion_parent_manifest","source_exclusion","active_source_mask","active_statistics","source_order_parent_manifest","source_order_overlay","source_generator_parent_manifest","source_generator_recovery"}.count(item.key()))file(dir,now);
    }
    if(signature(evidence.at("source_row_registry"))!=signature(parent.at("assets").at("source_rows.jsonl")))throw std::runtime_error("component correspondence changed the original row registry");
    if(evidence.at("source_correspondence").at("sha256")!=policy.at("facts_sha256")||evidence.at("source_inventory").at("sha256")!=policy.at("inventory_sha256"))throw std::runtime_error("component correspondence changed its fixed provenance snapshot");
    for(const char* name:{"model","source_row_registry","source_inventory","source_correspondence","reference_template","reference_native_image"})file(ed,evidence.at(name));
    if(!evidence.at("algorithm_sources").is_array()||evidence.at("algorithm_sources").empty())throw std::runtime_error("component correspondence lacks frozen algorithm sources");for(const auto& ref:evidence.at("algorithm_sources"))file(ed,ref);
    std::function<void(const eventio::Json&)> verify_evidence_refs=[&](const eventio::Json& v){if(v.is_object()){if(v.contains("sha256")&&(v.contains("file")||v.contains("path")))file(ed,v);for(const auto& x:v.items())verify_evidence_refs(x.value());}else if(v.is_array())for(const auto& x:v)verify_evidence_refs(x);};verify_evidence_refs(evidence);
    const auto evidence_model=eventio::read_json(file(ed,evidence.at("model")));if(evidence_model.at("source_identity")!=source_hash||evidence_model.at("materials").size()!=material_labels.size())throw std::runtime_error("component evidence model changed original source/material registry");
    for(std::size_t mi=0;mi<material_labels.size();++mi)if(evidence_model.at("materials").at(mi).at("label")!=material_labels[mi])throw std::runtime_error("component evidence reordered the raw material table");
    const auto facts=eventio::read_json(file(ed,evidence.at("source_correspondence")));if(facts.at("schema")!="rinkfit.native_component_identity_facts.v1"||!facts.at("physical_projection_used_for_identity").is_boolean()||facts.at("physical_projection_used_for_identity").get<bool>())throw std::runtime_error("component correspondence needs source-only identity facts");
    std::map<std::size_t,eventio::Json> facts_by_row;for(const auto& fact:facts.at("records"))if(!facts_by_row.emplace(fact.at("source_row").get<std::size_t>(),fact).second)throw std::runtime_error("duplicate native component fact row");if(facts_by_row.size()!=changed_count)throw std::runtime_error("component fact scope differs from pinned bindings");
    const auto inventory=eventio::read_json(file(ed,evidence.at("source_inventory")));if(inventory.at("schema")!="rinkfit.all_Ad5_print_original_fragment_inventory.v1"||inventory.at("raw_rows")!=25143||inventory.at("traceable_rows")!=25143||inventory.at("all_Ad5_print_frames")!=311||inventory.at("fragments").size()!=7234||!inventory.at("untraceable_source_rows").empty())throw std::runtime_error("component evidence omitted part of the full original Ad5 inventory");
    for(const auto& fragment:inventory.at("fragments")){file(ed,fragment.at("source_packet"));file(ed,fragment.at("board_readout_packet"));const auto& native=fragment.at("native_source");if(native.at("native_pts")!=fragment.at("native_pts")||native.at("actual_pts")!=fragment.at("native_pts")||native.at("programme_frame_index_1based")!=fragment.at("frame")||native.at("time_base")!="1/12800")throw std::runtime_error("component inventory native source clock changed");file(ed,eventio::Json{{"path",native.at("image_path")},{"sha256",native.at("png_sha256")}});file(ed,eventio::Json{{"path",native.at("mask_path")},{"sha256",native.at("mask_sha256")}});file(ed,eventio::Json{{"path",native.at("source")},{"sha256",native.at("source_video_sha256")}});}
    std::set<std::size_t> inventory_rows;std::set<int> inventory_frames;for(const auto& fragment:inventory.at("fragments")){const auto& rr=fragment.at("source_rows");const auto& vv=fragment.at("original_vertices");const auto& xy=fragment.at("raw_uv");if(rr.size()!=vv.size()||rr.size()!=xy.size())throw std::runtime_error("component inventory member dimensions changed");for(std::size_t j=0;j<rr.size();++j){const auto row=rr[j].get<std::size_t>();const auto pointid=fragment.at("source_feature_id").get<std::string>()+"/segment/"+std::to_string(fragment.at("segment_index").get<std::size_t>())+"/vertex/"+std::to_string(vv[j].get<std::size_t>());if(row>=rows.size()||!inventory_rows.insert(row).second||frames[row]!=fragment.at("frame")||pts[row]!=fragment.at("native_pts")||material_labels[material[row]]!=fragment.at("raw_label")||material_labels[material[row]].rfind("Ad5.print.",0)!=0||point_ids[point[row]]!=pointid||xy[j]!=eventio::Json::array({uv[2*row],uv[2*row+1]}))throw std::runtime_error("component inventory differs from the complete immutable original sources");inventory_frames.insert(frames[row]);}}
    if(inventory_rows.size()!=25143||inventory_frames.size()!=311)throw std::runtime_error("incomplete all-frame Ad5 inventory");for(std::size_t row=0;row<rows.size();++row)if((material_labels[material[row]].rfind("Ad5.print.",0)==0)!=(inventory_rows.count(row)!=0))throw std::runtime_error("component inventory omitted an original Ad5 obligation");
    std::set<std::size_t> wanted;for(const auto& b:proof.at("bindings"))if(!wanted.insert(b.at("source_row").get<std::size_t>()).second)throw std::runtime_error("duplicate component correspondence binding");std::map<std::size_t,eventio::Json> registry;{std::ifstream f(file(ed,evidence.at("source_row_registry")));std::string line;while(std::getline(f,line)){if(line.empty())continue;const auto r=eventio::Json::parse(line);const auto row=r.at("source_row").get<std::size_t>();if(wanted.count(row)&&!registry.emplace(row,r).second)throw std::runtime_error("duplicate component source-row provenance");}}if(registry.size()!=wanted.size())throw std::runtime_error("component source provenance is incomplete");
    std::map<std::string,eventio::Json> original_packets;
    auto material_now=eventio::read_array<std::uint32_t>(dir,selected.at("resolved_material_index"),"<u4"),variant_now=eventio::read_array<std::uint32_t>(dir,selected.at("resolved_variant_index"),"<u4");if(material_now.size()!=rows.size()||variant_now.size()!=rows.size())throw std::runtime_error("component recovery must preserve the full source ABI");
    std::set<std::size_t> protected_rows;for(const auto& b:inherited.source_generator_recovery.at("bindings"))protected_rows.insert(b.at("source_row").get<std::size_t>());for(const auto& b:inherited.confirmed_bindings.at("bindings"))protected_rows.insert(b.at("source_row").get<std::size_t>());
    const std::set<std::string> binding_keys={"source_row","native_pts","source_point_id","source_uv","original_source_label","original_material_index","original_variant_id","resolved_model_material_index","resolved_model_target_id","resolved_variant_id","allowed_path_indices","evidence_fragment_id"};
    std::map<std::size_t,eventio::Json> bindings;std::map<std::string,std::size_t> target_counts;
    for(const auto& b:proof.at("bindings")){
      if(b.size()!=binding_keys.size())throw std::runtime_error("component binding has an unknown identity/authority field");for(const auto& item:b.items())if(!binding_keys.count(item.key()))throw std::runtime_error("unknown component binding field");
      const auto row=b.at("source_row").get<std::size_t>();const auto mi=b.at("resolved_model_material_index").get<std::uint32_t>();const auto target=b.at("resolved_model_target_id").get<std::string>();const auto variant_id=b.at("resolved_variant_id").get<std::string>();const auto vi=std::find(inherited.resolved_variant_ids.begin(),inherited.resolved_variant_ids.end(),variant_id);
      if(row>=rows.size()||!bindings.emplace(row,b).second||protected_rows.count(row)||!inherited.is_active(row)||inherited.kind[row]!=0||mi>=material_labels.size()||vi==inherited.resolved_variant_ids.end()||b.at("native_pts")!=pts[row]||b.at("source_point_id")!=point_ids[point[row]]||b.at("source_uv")!=eventio::Json::array({uv[2*row],uv[2*row+1]})||b.at("original_source_label")!=material_labels[material[row]]||material_labels[material[row]].rfind("Ad5.print.",0)!=0||b.at("original_material_index")!=material[row]||b.at("original_variant_id")!=inherited.variant_ids[inherited.variant[row]]||target.rfind("Ad5.print.",0)!=0||target!=material_labels[mi]||b.at("allowed_path_indices")!=eventio::Json::array({0})||!b.at("evidence_fragment_id").is_string()||b.at("evidence_fragment_id").get<std::string>().empty()||material_now[row]!=mi||variant_now[row]!=std::uint32_t(vi-inherited.resolved_variant_ids.begin()))throw std::runtime_error("component correspondence changed a raw source, protected authority, or finite existing Ad5 target");
      const auto& paths=evidence_model.at("materials").at(mi).at("paths");if(paths.empty()||paths.at(0).at("variant_id")!=variant_id)throw std::runtime_error("component correspondence target is not the original finite model path");
      if(!facts_by_row.count(row)||!registry.count(row))throw std::runtime_error("component binding lacks a unique source-only fact/provenance record");const auto& fact=facts_by_row.at(row);if(!fact.at("identity_confirmed").is_boolean()||!fact.at("identity_confirmed").get<bool>()||fact.at("resolved_model_target_id")!=target||fact.at("evidence_fragment_id")!=b.at("evidence_fragment_id"))throw std::runtime_error("component binding is unsupported by its exact pinned source fact");
      const auto& reg=registry.at(row);const auto source_packet=file(ed,reg.at("source_packet"));const auto key=source_packet.u8string();if(!original_packets.count(key))original_packets.emplace(key,eventio::read_json(source_packet));const auto& original=original_packets.at(key);const eventio::Json* feature=nullptr;for(const auto& f:original.at("features"))if(f.at("id")==reg.at("source_feature_id")){if(feature)throw std::runtime_error("ambiguous original component feature");feature=&f;}const auto si=reg.at("source_segment_index").get<std::size_t>(),vertex=reg.at("original_vertex_index").get<std::size_t>();if(!feature||original.at("source").at("native_pts")!=b.at("native_pts")||feature->at("segments").at(si).at("uv").at(vertex)!=b.at("source_uv")||point_ids[point[row]]!=reg.at("source_feature_id").get<std::string>()+"/segment/"+std::to_string(si)+"/vertex/"+std::to_string(vertex))throw std::runtime_error("component original packet/member/PTS/UV changed");
      if(mi==inherited.model_material(row)&&variant_id==inherited.model_variant_id(row))throw std::runtime_error("redundant component correspondence override");++target_counts[target];
    }
    if(bindings.size()!=changed_count||target_counts!=policy.at("targets").get<std::map<std::string,std::size_t>>())throw std::runtime_error("component correspondence changed exact pinned counts");
    for(std::size_t row=0;row<rows.size();++row){const bool changed=material_now[row]!=inherited.model_material(row)||variant_now[row]!=inherited.resolved_variant[row];if(changed!=(bindings.count(row)!=0))throw std::runtime_error("component correspondence silently changed an unaudited row");}
    inherited.resolved_material=std::move(material_now);inherited.resolved_variant=std::move(variant_now);inherited.has_source_component_recovery=true;inherited.source_component_recovery=proof;inherited.source_component_recovery_hash=selected.at("source_component_recovery").at("sha256");inherited.source_component_evidence_hash=evidence_hash;inherited.semantic_ancestor_hashes.insert(selected_hash);inherited.semantics=selected;inherited.semantic_path=selected_path;inherited.semantic_hash=selected_hash;inherited.manifest=selected_manifest;*this=std::move(inherited);
  }
  void load_source_generator_recovery(){
    const auto selected_path=semantic_path;const auto selected_hash=semantic_hash;const auto selected=semantics;const auto selected_manifest=manifest;const auto dir=selected_path.parent_path();
    if(!groups.empty()||!runs.empty()||!selected.contains("source_generator_parent_manifest")||!selected.contains("source_generator_recovery"))throw std::runtime_error("incomplete/repeated generator-recovery overlay");
    if(manifest.at("semantic_packet").at("sha256")!=selected_hash)throw std::runtime_error("generator recovery differs from immutable packet binding");
    const std::string parent_hash="416f6f4254be483089b38fdf1d74ac64b32235e78ca87c276e5127f11bd09df4";
    const auto review_policies=eventio::Json::parse(R"({"250cf0e111410acdeea3e87465d03f3201baf8eabd089e5a8aa4235102a9ae45":{"changed":53,"retained":53,"fragments":22,"members":106,"frames":18,"target_counts":{"Ad2.print.K":39,"Ad2.print.SKATING_A_inner":14}},"5620ec6a2519821782f4d181f840eff251c03e44baf19fb1d8cca6837b160e69":{"changed":108,"retained":132,"fragments":41,"members":240,"frames":32,"target_counts":{"Ad2.print.COM_O":1,"Ad2.print.COM_O_inner":8,"Ad2.print.K":39,"Ad2.print.SKATING_A":28,"Ad2.print.SKATING_A_inner":14,"Ad2.print.SKATING_S":11,"Ad2.print.hyphen":7}}})");
    std::string review_hash;
    const auto& pref=selected.at("source_generator_parent_manifest");if(pref.at("sha256")!=parent_hash)throw std::runtime_error("generator recovery has another reviewed parent");
    auto file=[&](const std::filesystem::path& base,const eventio::Json& ref){const char* key=ref.contains("file")?"file":"path";const auto p=base/std::filesystem::u8path(ref.at(key).get<std::string>());eventio::verify(p,ref.at("sha256"));return p;};
    const auto parent_path=file(dir,pref);EventPacket inherited(packet_path);inherited.manifest["semantic_packet"]={{"manifest",parent_path.u8string()},{"sha256",parent_hash}};inherited.load_semantics(parent_path);
    if(inherited.has_source_generator_recovery||!inherited.has_source_order_overlay||!inherited.has_source_exclusion||inherited.active_rows.size()!=182016||inherited.audit_runs.size()!=15419||!inherited.runs.empty())throw std::runtime_error("generator recovery requires the complete reviewed parent");
    const auto parent=inherited.semantics;
    std::function<eventio::Json(const eventio::Json&)> signature=[&](const eventio::Json& v)->eventio::Json{if(v.is_array()){auto a=eventio::Json::array();for(const auto& x:v)a.push_back(signature(x));return a;}if(!v.is_object())return v;auto o=eventio::Json::object();for(const auto& item:v.items())if(!(v.contains("sha256")&&(item.key()=="file"||item.key()=="path")))o[item.key()]=signature(item.value());return o;};
    std::function<void(const eventio::Json&)> verify_refs=[&](const eventio::Json& v){if(v.is_object()){if(v.contains("sha256")&&(v.contains("file")||v.contains("path")))file(dir,v);for(const auto& x:v.items())verify_refs(x.value());}else if(v.is_array())for(const auto& x:v)verify_refs(x);};
    if(selected.size()!=parent.size()+2)throw std::runtime_error("generator recovery changed unrelated field inventory");
    const std::set<std::string> changed_fields={"resolved_material_index","resolved_variant_index"};
    for(const auto& item:parent.items()){
      if(!selected.contains(item.key()))throw std::runtime_error("generator recovery removed inherited authority");const auto& now=selected.at(item.key());
      if(changed_fields.count(item.key())){auto oldsig=signature(item.value()),newsig=signature(now);oldsig.erase("sha256");newsig.erase("sha256");if(oldsig!=newsig)throw std::runtime_error("generator recovery changed resolved-array dtype/shape/metadata");file(dir,now);continue;}
      if(signature(now)!=signature(item.value()))throw std::runtime_error("generator recovery changed raw/W/E/query/exclusion/order/A3 authority: "+item.key());
      if(item.key()=="arrays"||item.key()=="assets")verify_refs(now);
      else if(std::set<std::string>{"parent_semantic_manifest","original_material_index","confirmed_bindings","source_exclusion_parent_manifest","source_exclusion","active_source_mask","active_statistics","source_order_parent_manifest","source_order_overlay"}.count(item.key()))file(dir,now);
    }
    const auto proof_path=file(dir,selected.at("source_generator_recovery"));const auto proof=eventio::read_json(proof_path);const auto pd=proof_path.parent_path();
    const std::set<std::string> proof_keys={"schema","rule_id","parent_semantic_sha256","observations","review_manifest","bindings"};
    if(proof.size()!=proof_keys.size())throw std::runtime_error("generator recovery has an unreviewed authority/anchor field");for(const auto& item:proof.items())if(!proof_keys.count(item.key()))throw std::runtime_error("unknown generator recovery authority field");
    if(proof.at("schema")!="rinkfit.source_generator_recovery.v1"||proof.at("rule_id")!="native_letter_fragment_identity_recovery_v1"||proof.at("parent_semantic_sha256")!=parent_hash||proof.at("observations")!=rows.size())throw std::runtime_error("generator recovery proof scope mismatch");
    review_hash=proof.at("review_manifest").at("sha256").get<std::string>();
    if(!review_policies.contains(review_hash))throw std::runtime_error("unreviewed generator recovery evidence ledger");
    const auto& policy=review_policies.at(review_hash);
    const auto n_changed=policy.at("changed").get<std::size_t>(),n_retained=policy.at("retained").get<std::size_t>(),n_fragments=policy.at("fragments").get<std::size_t>(),n_members=policy.at("members").get<std::size_t>(),n_frames=policy.at("frames").get<std::size_t>();
    const auto review_path=file(pd,proof.at("review_manifest"));const auto review=eventio::read_json(review_path);const auto rd=review_path.parent_path();
    if(review.at("schema")!="rinkfit.native-source-generator-recovery-review.v1"||review.at("rule_id")!="native_letter_fragment_identity_recovery_v1"||review.at("review_origin")!="codex_primary_native_frame_visual_review"||!review.at("not_a_direct_human_confirmation").get<bool>()||!review.at("no_extension_of_any_human_confirmation").get<bool>()||!review.at("no_human_anchor_used").get<bool>()||!review.at("current_model_nearest_label_is_not_identity_evidence").get<bool>()||review.at("parent_semantic_sha256")!=parent_hash||review.at("source_identity")!=source_hash||review.at("raw_observations")!=rows.size()||review.at("active_observations")!=inherited.active_rows.size()||!review.at("raw_UV_ID_labels_W_E_and_existing_authority_unchanged").get<bool>())throw std::runtime_error("generator recovery cannot impersonate a human or change source identity");
    if(proof.at("bindings")!=review.at("bindings")||review.at("bindings").size()!=n_changed||review.at("changed_rows")!=n_changed||review.at("retained_outer_rows")!=n_retained||review.at("reviewed_full_fragments")!=n_fragments||review.at("reviewed_fragment_rows")!=n_members||review.at("reviewed_native_frames")!=n_frames||review.at("binding_target_counts")!=policy.at("target_counts"))throw std::runtime_error("generator recovery differs from the complete pinned native review");
    if(signature(review.at("source_row_registry"))!=signature(parent.at("assets").at("source_rows.jsonl")))throw std::runtime_error("generator recovery has another immutable source-row registry");
    file(rd,review.at("source_row_registry"));file(rd,review.at("model"));file(rd,review.at("source_video"));
    for(const auto& ref:review.at("producer_evidence"))file(rd,ref);
    for(const auto& ref:review.at("source_only_contact_sheets"))file(rd,ref);
    auto material_now=eventio::read_array<std::uint32_t>(dir,selected.at("resolved_material_index"),"<u4"),variant_now=eventio::read_array<std::uint32_t>(dir,selected.at("resolved_variant_index"),"<u4");
    if(material_now.size()!=rows.size()||variant_now.size()!=rows.size())throw std::runtime_error("generator recovery must preserve full source ABI");
    std::map<std::size_t,eventio::Json> bindings;std::map<std::string,std::size_t> target_counts;
    for(const auto& b:proof.at("bindings")){
      const auto row=b.at("source_row").get<std::size_t>();const auto mi=b.at("resolved_model_material_index").get<std::uint32_t>();const auto vi=std::find(inherited.resolved_variant_ids.begin(),inherited.resolved_variant_ids.end(),b.at("resolved_variant_id").get<std::string>());
      if(row>=rows.size()||!bindings.emplace(row,b).second||!inherited.is_active(row)||inherited.kind[row]!=0||mi>=material_labels.size()||vi==inherited.resolved_variant_ids.end()||b.at("native_pts")!=pts[row]||b.at("source_point_id")!=point_ids[point[row]]||b.at("source_uv")!=eventio::Json::array({uv[2*row],uv[2*row+1]})||b.at("original_source_label")!=material_labels[material[row]]||b.at("original_material_index")!=material[row]||b.at("original_variant_id")!=inherited.variant_ids[inherited.variant[row]]||b.at("resolved_model_target_id")!=material_labels[mi]||b.at("allowed_path_indices")!=eventio::Json::array({0})||material_now[row]!=mi||variant_now[row]!=std::uint32_t(vi-inherited.resolved_variant_ids.begin()))throw std::runtime_error("generator recovery binding differs from exact original source or reviewed target");
      if(mi==inherited.model_material(row)&&b.at("resolved_variant_id")==inherited.model_variant_id(row))throw std::runtime_error("redundant unreviewed generator override");++target_counts[material_labels[mi]];
    }
    if(target_counts!=policy.at("target_counts").get<std::map<std::string,std::size_t>>())throw std::runtime_error("generator recovery target inventory changed");
    for(std::size_t row=0;row<rows.size();++row){const bool changed=material_now[row]!=inherited.model_material(row)||variant_now[row]!=inherited.resolved_variant[row];if(changed!=(bindings.count(row)!=0))throw std::runtime_error("generator recovery silently changed an unaudited row");}
    std::map<int,eventio::Json> frame_packets,frame_records;
    for(const auto& frame:review.at("frames")){
      const int f=frame.at("frame");if(f<1||f>1248||frame.at("native_pts")!=256256+256*(f-1)||frame.at("time_base")!="1/12800"||!frame.at("native_PTS_matches_source_packet").get<bool>()||!frame.at("native_image").at("matches_source_packet").get<bool>()||frame.at("native_image").at("decoded_BGR_sha256")!=frame.at("original_decoded_BGR_sha256"))throw std::runtime_error("generator recovery native frame/time binding changed");
      const auto source=eventio::read_json(file(rd,frame.at("source_packet")));file(rd,frame.at("native_image"));file(rd,frame.at("foreground_mask"));const auto& native=source.at("source");
      if(native.at("native_pts")!=frame.at("native_pts")||native.at("actual_pts")!=frame.at("native_pts")||native.at("programme_frame_index_1based")!=f||native.at("time_base")!="1/12800"||native.at("source_video_sha256")!=review.at("source_video").at("sha256")||native.at("decoded_BGR_sha256")!=frame.at("original_decoded_BGR_sha256")||native.at("mask_sha256")!=frame.at("foreground_mask").at("sha256")||!frame_packets.emplace(f,source).second)throw std::runtime_error("generator recovery duplicate/incorrect native packet");frame_records[f]=frame;
    }
    if(frame_packets.size()!=n_frames)throw std::runtime_error("generator recovery native frame coverage changed");
    std::set<std::size_t> members,changed_members;std::set<std::string> fragments;std::size_t retained=0;
    for(const auto& fragment:review.at("fragments")){
      const auto id=fragment.at("fragment_id").get<std::string>();const int f=fragment.at("frame");if(!fragments.insert(id).second||!frame_packets.count(f)||fragment.at("native_pts")!=frame_records.at(f).at("native_pts")||fragment.at("source_packet")!=frame_records.at(f).at("source_packet")||fragment.at("native_image")!=frame_records.at(f).at("native_image")||!fragment.at("native_review_completed").get<bool>()||fragment.at("native_review_origin")!="codex_primary_native_frame_visual_review"||!fragment.at("complete_fragment_membership_reviewed").get<bool>())throw std::runtime_error("generator recovery fragment lacks exact primary review");
      const auto feature_id=fragment.at("source_feature_id").get<std::string>();const eventio::Json* feature=nullptr;for(const auto& value:frame_packets.at(f).at("features"))if(value.at("id")==feature_id){if(feature)throw std::runtime_error("ambiguous reviewed source fragment");feature=&value;}if(!feature)throw std::runtime_error("reviewed native feature missing");
      const auto canonical=feature->dump();if(eventio::sha256(std::vector<std::uint8_t>(canonical.begin(),canonical.end()))!=fragment.at("source_feature_sha256_canonical_json"))throw std::runtime_error("reviewed source generator record changed");
      std::size_t feature_members=0;for(const auto& segment:feature->at("segments"))feature_members+=segment.at("uv").size();if(feature_members!=fragment.at("members").size())throw std::runtime_error("generator recovery omitted part of a source fragment");
      for(const auto& member:fragment.at("members")){
        const auto row=member.at("source_row").get<std::size_t>(),si=member.at("segment").get<std::size_t>(),vertex=member.at("vertex").get<std::size_t>();
        if(row>=rows.size()||!members.insert(row).second||frames[row]!=f||pts[row]!=fragment.at("native_pts")||!inherited.is_active(row)||inherited.kind[row]!=0||member.at("source_uv")!=eventio::Json::array({uv[2*row],uv[2*row+1]})||feature->at("segments").at(si).at("uv").at(vertex)!=member.at("source_uv")||point_ids[point[row]]!=feature_id+"/segment/"+std::to_string(si)+"/vertex/"+std::to_string(vertex))throw std::runtime_error("generator recovery fragment membership/UV/identity changed");
        const auto found=bindings.find(row);const bool changed=found!=bindings.end();if(member.at("interpretation_changed").get<bool>()!=changed)throw std::runtime_error("generator recovery changed/retained review mismatch");
        if(changed){if(found->second.at("evidence_fragment_id")!=id||found->second.at("resolved_model_target_id")!=member.at("reviewed_model_target_id"))throw std::runtime_error("generator recovery binding has no matching reviewed native member");changed_members.insert(row);}else{if(member.at("reviewed_model_target_id")!=material_labels[inherited.model_material(row)])throw std::runtime_error("generator recovery lost an unchanged exterior member");++retained;}
      }
    }
    if(fragments.size()!=n_fragments||members.size()!=n_members||changed_members.size()!=n_changed||retained!=n_retained)throw std::runtime_error("incomplete full native-fragment recovery audit");
    inherited.resolved_material=std::move(material_now);inherited.resolved_variant=std::move(variant_now);inherited.has_source_generator_recovery=true;inherited.source_generator_recovery=proof;inherited.source_generator_recovery_hash=selected.at("source_generator_recovery").at("sha256");inherited.source_generator_review_hash=review_hash;inherited.semantic_ancestor_hashes.insert(selected_hash);inherited.semantics=selected;inherited.semantic_path=selected_path;inherited.semantic_hash=selected_hash;inherited.manifest=selected_manifest;*this=std::move(inherited);
  }
  void load_source_order_evidence(){
    const auto selected_path=semantic_path;const auto selected_hash=semantic_hash;const auto selected=semantics;const auto selected_manifest=manifest;const auto dir=selected_path.parent_path();
    if(!groups.empty()||!runs.empty()||!selected.contains("source_order_parent_manifest")||!selected.contains("source_order_overlay"))throw std::runtime_error("incomplete/repeated source-order overlay");
    if(manifest.at("semantic_packet").at("sha256")!=selected_hash)throw std::runtime_error("source-order manifest differs from immutable packet binding");
    const auto& pref=selected.at("source_order_parent_manifest");const std::string parent_hash="bb74173425444d4dbe925c50c5bf42f3739663b602d0bb839a0392f040dfb64d";
    if(pref.at("sha256")!=parent_hash)throw std::runtime_error("source-order rule has another reviewed parent");
    auto file=[&](const std::filesystem::path& base,const eventio::Json& ref){const char* key=ref.contains("file")?"file":"path";auto p=base/std::filesystem::u8path(ref.at(key).get<std::string>());eventio::verify(p,ref.at("sha256"));return p;};
    const auto parent_path=file(dir,pref);EventPacket inherited(packet_path);inherited.manifest["semantic_packet"]={{"manifest",parent_path.u8string()},{"sha256",parent_hash}};inherited.load_semantics(parent_path);
    if(inherited.has_source_order_overlay||inherited.runs.size()!=15419)throw std::runtime_error("stacked/missing original source-order audit");
    const auto parent=inherited.semantics;
    std::function<eventio::Json(const eventio::Json&)> signature=[&](const eventio::Json& v)->eventio::Json{
      if(v.is_array()){auto a=eventio::Json::array();for(const auto& x:v)a.push_back(signature(x));return a;}
      if(!v.is_object())return v;auto o=eventio::Json::object();for(const auto& item:v.items())if(!(v.contains("sha256")&&(item.key()=="file"||item.key()=="path")))o[item.key()]=signature(item.value());return o;};
    std::function<void(const eventio::Json&)> verify_refs=[&](const eventio::Json& v){if(v.is_object()){if(v.contains("sha256")&&(v.contains("file")||v.contains("path")))file(dir,v);for(const auto& x:v.items())verify_refs(x.value());}else if(v.is_array())for(const auto& x:v)verify_refs(x);};
    if(selected.size()!=parent.size()+2)throw std::runtime_error("source-order overlay changed unrelated field inventory");
    for(const auto& item:parent.items()){
      if(!selected.contains(item.key()))throw std::runtime_error("source-order overlay deleted inherited authority");
      if(item.key()!="assets"){if(signature(selected.at(item.key()))!=signature(item.value()))throw std::runtime_error("source-order overlay changed raw/W/E/query/material/exclusion authority: "+item.key());const auto& value=selected.at(item.key());if(item.key()=="arrays")verify_refs(value);else if(std::set<std::string>{"parent_semantic_manifest","original_material_index","resolved_material_index","resolved_variant_index","confirmed_bindings","source_exclusion_parent_manifest","source_exclusion","active_source_mask","active_statistics"}.count(item.key()))file(dir,value);}
    }
    if(selected.at("assets").size()!=parent.at("assets").size())throw std::runtime_error("source-order overlay changed asset inventory");
    for(const auto& item:parent.at("assets").items()){const auto& now=selected.at("assets").at(item.key());if(item.key()!="source_runs.jsonl"&&signature(now)!=signature(item.value()))throw std::runtime_error("source-order overlay changed another asset: "+item.key());file(dir,now);}
    const auto overlay_path=file(dir,selected.at("source_order_overlay"));const auto overlay=eventio::read_json(overlay_path);const auto odir=overlay_path.parent_path();
    if(overlay.at("schema")!="rinkfit.source_order_evidence_overlay.v1"||overlay.at("rule_id")!="current_native_independent_normal_queries_without_boundary_trace_v1"||overlay.at("parent_semantic_sha256")!=parent_hash||overlay.at("observations")!=rows.size()||!overlay.at("raw_arrays_W_E_and_authorities_unchanged").get<bool>()||signature(overlay.at("audit_source_runs"))!=signature(parent.at("assets").at("source_runs.jsonl"))||signature(overlay.at("active_hard_source_runs"))!=signature(selected.at("assets").at("source_runs.jsonl")))throw std::runtime_error("source-order authority/parent/audit mismatch");
    const std::map<std::string,std::string> producers={{"normal_query_producer","3d30d5d595fa7299ad53191ea5e3c2844ae96e955ce42e18b5df67d938bf4f85"},{"derived_order_stamper","00b4d293bcaa0049d17f71430f40cccf00151e22d27f07775f0ff44c9f1f6d6e"},{"semantic_run_exporter","e9072669f64e56f490f03eea23db38bbe3f90bcce6ddd580de7ebb530d6e7b17"}};
    const std::map<std::string,std::string> facts={{"spec","d085d351959380c6dbd5bd581d84cad119640cdd80818343ec6dbb11b4abac93"},{"order_counterexample","38318f642da42767cd171818bcff0a3f02b4c56966c7d06184855a75979b4e5c"},{"hole_support_diagnostic","d79abf8a483484d7b47419e152d9355d98b3c3fa3bc27d275a460725083c94e6"}};
    auto evidence=[&](const eventio::Json& list,const std::map<std::string,std::string>& expected){if(list.size()!=expected.size())throw std::runtime_error("incomplete source-order proof");std::set<std::string> seen;for(const auto& ref:list){const auto role=ref.at("role").get<std::string>();const auto it=expected.find(role);if(it==expected.end()||!seen.insert(role).second||ref.at("sha256")!=it->second)throw std::runtime_error("unreviewed source-order producer/fact");file(odir,ref);}};
    evidence(overlay.at("producer_evidence"),producers);evidence(overlay.at("evidence"),facts);
    const auto audit_path=file(odir,overlay.at("audit_source_runs")),class_path=file(odir,overlay.at("classifications")),hard_path=file(odir,overlay.at("active_hard_source_runs"));
    if(std::filesystem::file_size(hard_path)!=0||overlay.at("active_hard_run_count")!=0||overlay.at("active_hard_row_count")!=0)throw std::runtime_error("reviewed rule does not authorize arbitrary hard-order replacement");
    const std::set<std::string> fk={"allowed_for_fit","current_source_only_readout","id","label","label_zh","model_or_template_generated_observation","physical_surface","review_layer","segments","source_identity_recovery","target_id","world_correspondence_status"};
    const std::set<std::string> sk={"closed","localization_sigma_px","measurement_origin","native_connected_component_rgb_profile_support_fraction","native_simplification_tolerance_px","pixels_measured_in_this_frame","point_ids","pre_compaction_vertex_count","retained_pre_compaction_vertex_indices","source_clear_component_identity_verified_by_current_shape","source_query_count","sparse_representation_max_deviation_px","transition_iqr_px","uv","vertex_count"};
    std::map<std::string,eventio::Json> packets;std::ifstream audit(audit_path),classes(class_path);if(!audit||!classes)throw std::runtime_error("missing source-order audit/classification");std::string line,saved_line;std::size_t ordinal=0,total=0;
    while(std::getline(audit,line)){if(line.empty())continue;if(!std::getline(classes,saved_line)||saved_line.empty())throw std::runtime_error("incomplete source-order classifications");const auto source=eventio::Json::parse(line),saved=eventio::Json::parse(saved_line);const auto rr=source.at("source_rows").get<std::vector<std::size_t>>(),vv=source.at("source_vertex_indices").get<std::vector<std::size_t>>();
      if(ordinal>=inherited.runs.size()||rr!=inherited.runs[ordinal].rows||rr.size()<2||rr.size()!=vv.size()||std::set<std::size_t>(rr.begin(),rr.end()).size()!=rr.size())throw std::runtime_error("invalid original navigation run registry");
      for(std::size_t k=1;k<vv.size();++k)if(vv[k]!=vv[k-1]+1)throw std::runtime_error("nonconsecutive original navigation query indices");
      const auto& ref=source.at("source_packet");const auto key=ref.at("path").get<std::string>()+"|"+ref.at("sha256").get<std::string>();auto it=packets.find(key);
      if(it==packets.end())it=packets.emplace(key,eventio::read_json(file(audit_path.parent_path(),ref))).first;
      const auto& original=it->second;const auto id=source.at("source_feature_id").get<std::string>();const eventio::Json* feature=nullptr;for(const auto& f:original.at("features"))if(f.at("id")==id){if(feature)throw std::runtime_error("ambiguous source feature identity");feature=&f;}if(!feature)throw std::runtime_error("missing original source feature");const auto& segment=feature->at("segments").at(source.at("source_segment_index").get<std::size_t>());
      for(const auto& x:feature->items())if(!fk.count(x.key()))throw std::runtime_error("unknown trace evidence cannot be retired as navigation");for(const auto& x:segment.items())if(!sk.count(x.key()))throw std::runtime_error("unknown segment trace evidence cannot be retired as navigation");
      if(feature->value("current_source_only_readout",false)!=true||feature->value("model_or_template_generated_observation",true)!=false||feature->value("world_correspondence_status",std::string())!="named_same_physical_print_component"||segment.value("measurement_origin",std::string())!="current_native_material_normal_half_transition"||!segment.value("pixels_measured_in_this_frame",false))throw std::runtime_error("actual boundary trace lies outside the independent normal-query rule");
      for(std::size_t k=0;k<rr.size();++k){const auto row=rr[k],vertex=vv[k];if(row>=rows.size()||vertex>=segment.at("uv").size()||frames[row]!=source.at("frame")||pts[row]!=original.at("source_pts")||segment.at("uv").at(vertex)!=eventio::Json::array({uv[2*row],uv[2*row+1]}))throw std::runtime_error("source-order proof source UV/time/member mismatch");}
      std::ostringstream rid;rid<<"run/"<<std::setw(6)<<std::setfill('0')<<ordinal;const auto canonical=source.dump();const auto digest=eventio::sha256(std::vector<std::uint8_t>(canonical.begin(),canonical.end()));
      const eventio::Json expected={{"run_id",rid.str()},{"source_record_sha256",digest},{"source_rows",rr},{"source_vertex_indices",vv},{"classification","navigation_query_sequence_only"},{"measurement_origin","current_native_material_normal_half_transition"},{"independent_boundary_trace_verified",false},{"region_identity_is_not_boundary_trace",true},{"active_hard_order_factor",false}};
      if(saved!=expected)throw std::runtime_error("fabricated source-order classification ledger");total+=rr.size();++ordinal;
    }
    while(std::getline(classes,saved_line))if(!saved_line.empty())throw std::runtime_error("unaccounted source-order classifications");
    if(ordinal!=15419||total!=54948||overlay.at("audit_run_count")!=ordinal||overlay.at("audit_row_count")!=total||overlay.at("verified_source_packet_count")!=packets.size())throw std::runtime_error("source-order audit coverage mismatch");
    inherited.audit_runs=inherited.runs;inherited.runs.clear();inherited.has_source_order_overlay=true;inherited.source_order_overlay=overlay;inherited.source_order_overlay_hash=selected.at("source_order_overlay").at("sha256");inherited.semantic_ancestor_hashes.insert(selected_hash);inherited.semantics=selected;inherited.semantic_path=selected_path;inherited.semantic_hash=selected_hash;inherited.manifest=selected_manifest;*this=std::move(inherited);
  }
  void load_source_exclusion(const std::filesystem::path& dir){
    const auto n=rows.size();
    const auto& parent_ref=semantics.at("source_exclusion_parent_manifest");
    const auto parent_path=dir/std::filesystem::u8path(parent_ref.at("path").get<std::string>());
    eventio::verify(parent_path,parent_ref.at("sha256"));const auto parent=eventio::read_json(parent_path);
    if(parent.contains("source_exclusion")||parent.contains("active_source_mask")||parent.contains("active_statistics"))throw std::runtime_error("nested source exclusion requires an explicitly composed new authority ledger");
    auto descriptor_equal=[&](const eventio::Json& previous,const eventio::Json& current){
      if(previous.size()!=current.size())throw std::runtime_error("source exclusion changed original descriptor fields");
      for(const auto& field:previous.items())if(field.key()!="file"&&(!current.contains(field.key())||current.at(field.key())!=field.value()))throw std::runtime_error("source exclusion changed original descriptor metadata");
      eventio::verify(parent_path.parent_path()/std::filesystem::u8path(previous.at("file").get<std::string>()),previous.at("sha256"));
      eventio::verify(dir/std::filesystem::u8path(current.at("file").get<std::string>()),current.at("sha256"));
    };
    for(const auto& item:parent.items()){
      if(!semantics.contains(item.key()))throw std::runtime_error("source exclusion removed original semantic metadata");
      const auto& now=semantics.at(item.key());
      if(item.key()=="arrays"||item.key()=="assets"){
        if(now.size()!=item.value().size())throw std::runtime_error("source exclusion changed original source/W/query inventory");
        for(const auto& d:item.value().items())descriptor_equal(d.value(),now.at(d.key()));
      }else if(item.value().is_object()&&item.value().contains("file")&&item.value().contains("sha256"))descriptor_equal(item.value(),now);
      else if(now!=item.value())throw std::runtime_error("source exclusion changed original semantic field: "+item.key());
    }
    const auto& ref=semantics.at("source_exclusion");const auto path=dir/std::filesystem::u8path(ref.at("file").get<std::string>());
    eventio::verify(path,ref.at("sha256"));source_exclusion=eventio::read_json(path);source_exclusion_hash=ref.at("sha256");
    active_mask=eventio::read_array<std::uint8_t>(dir,semantics.at("active_source_mask"),"|u1");
    active_mask_hash=semantics.at("active_source_mask").at("sha256");
    if(active_mask.size()!=n)throw std::runtime_error("source exclusion mask must retain the entire source ABI");
    active_rows.clear();excluded_rows.clear();
    for(std::size_t i=0;i<n;++i){if(active_mask[i]>1)throw std::runtime_error("source exclusion mask is not Boolean");(active_mask[i]?active_rows:excluded_rows).push_back(i);}
    if(excluded_rows.size()!=1||excluded_rows.front()!=25537||active_rows.size()!=182016)throw std::runtime_error("this explicit source exclusion authorizes only row 25537; original ledger remains 182017");
    semantic_ancestor_hashes.insert(parent_ref.at("sha256").get<std::string>());
    validate_exclusion_authority(dir,parent_ref);
    load_active_statistics(dir,parent_ref);
    has_source_exclusion=true;
  }
  void validate_exclusion_authority(const std::filesystem::path& dir,const eventio::Json& parent_ref){
    const auto& e=source_exclusion;
    if(e.at("schema")!="rinkfit.confirmed_source_exclusion.v1"||e.at("raw_observations")!=rows.size()||e.at("active_observations")!=active_rows.size()||e.at("source_identity")!=source_hash||e.at("parent_semantic_sha256")!=parent_ref.at("sha256")||e.at("original_engineering_budget_sha256")!=manifest.at("arrays").at("engineering_budget_px").at("sha256")||!e.at("raw_arrays_W_E_and_record_bytes_remain_immutable").get<bool>()||e.at("exclusions").size()!=1)throw std::runtime_error("source exclusion source/parent/budget/authority inventory mismatch");
    const auto& record=e.at("exclusions").at(0);const auto i=record.at("source_row").get<std::size_t>();
    if(i!=25537||is_active(i)||kind[i]!=0||record.at("native_pts")!=pts[i]||record.at("source_point_id")!=point_ids[point[i]]||record.at("source_uv")!=eventio::Json::array({uv[2*i],uv[2*i+1]})||record.at("original_source_label")!=material_labels[material[i]]||record.at("reason").get<std::string>().empty())throw std::runtime_error("source exclusion exact-row key mismatch");
    for(const auto& run:runs)if(std::find(run.rows.begin(),run.rows.end(),i)!=run.rows.end())throw std::runtime_error("source exclusion would alter an existing source-order run");
    const auto& ref=record.at("direct_user_authorization");const auto path=dir/std::filesystem::u8path(ref.at("path").get<std::string>());eventio::verify(path,ref.at("sha256"));const auto human=eventio::read_json(path);
    if(human.at("schema")!="rinkfit.user_authorized_source_exclusion.v1"||human.at("authority")!="direct_current_user_instruction"||human.at("user_message").get<std::string>().empty()||!human.at("user_authorizes_excluding_only_this_source_row").get<bool>()||human.at("source_row")!=i||human.at("shot_id")!=shot_id||human.at("native_pts")!=pts[i]||human.at("native_time_base")!=manifest.at("native_time_base")||human.at("source_uv")!=record.at("source_uv")||human.at("original_source_point_id")!=point_ids[point[i]]||human.at("original_source_label")!=material_labels[material[i]]||human.at("original_npz_source_identity")!=source_hash||human.at("exclude_other_source_rows").get<bool>()||!human.at("raw_arrays_and_record_bytes_remain_immutable").get<bool>()||human.at("source_record")!=record.at("source_record"))throw std::runtime_error("source exclusion lacks direct exact-row human authorization");
    const auto& sref=record.at("source_record");const auto spath=dir/std::filesystem::u8path(sref.at("path").get<std::string>());eventio::verify(spath,sref.at("sha256"));const auto source=eventio::read_json(spath);if(source.at("source_pts")!=pts[i]||source.at(eventio::Json::json_pointer(sref.at("json_pointer").get<std::string>()))!=record.at("source_uv"))throw std::runtime_error("excluded source fact differs from the exact immutable original packet");
    observations[i].active=false;observations[i].exclusion_reason=record.at("reason");observations[i].exclusion_id=source_exclusion_hash;
  }
  void load_active_statistics(const std::filesystem::path& dir,const eventio::Json& parent_ref){
    const auto& ref=semantics.at("active_statistics");const auto path=dir/std::filesystem::u8path(ref.at("file").get<std::string>());eventio::verify(path,ref.at("sha256"));active_statistics=eventio::read_json(path);active_statistics_hash=ref.at("sha256");const auto& a=active_statistics;
    if(a.at("schema")!="rinkfit.active_statistics.v1"||a.at("parent_semantic_sha256")!=parent_ref.at("sha256")||a.at("source_exclusion_sha256")!=source_exclusion_hash||a.at("active_source_mask_sha256")!=active_mask_hash||a.at("active_observations")!=active_rows.size()||a.at("original_group_count")!=original_group_count||a.at("formal_residual_rows")!=formal_residual_rows||a.at("replacements").size()!=1)throw std::runtime_error("active statistics identity/inventory mismatch");
    std::set<std::string> replaced;
    for(const auto& replacement:a.at("replacements")){
      const auto id=replacement.at("group_id").get<std::string>();auto found=std::find_if(groups.begin(),groups.end(),[&](const EventStatisticalGroup& g){return g.id==id;});
      if(found==groups.end()||!replaced.insert(id).second||found->rows.size()!=9||found->rank!=2||replacement.at("rank")!=2||replacement.at("original_matrix_sha256")!=eventpacket_detail::double_hash(found->W))throw std::runtime_error("active statistics changed an unrelated group/rank/original operator");
      std::vector<std::size_t> expected;for(auto row:found->rows)if(is_active(row))expected.push_back(row);
      const auto retained=replacement.at("source_rows").get<std::vector<std::size_t>>();
      if(expected.size()!=8||retained!=expected||std::find(found->rows.begin(),found->rows.end(),25537)==found->rows.end())throw std::runtime_error("active group does not preserve the exact original retained member order");
      const auto& wref=replacement.at("W");if(wref.at("shape")!=eventio::Json::array({2,16}))throw std::runtime_error("active rank2 group has wrong finite dimensions");
      const auto W=eventio::read_array<double>(path.parent_path(),wref,"<f8");if(W.size()!=32)throw std::runtime_error("active group W size mismatch");for(double x:W)if(!std::isfinite(x))throw std::runtime_error("active group W contains a nonfinite coefficient");
      const auto& proof_ref=replacement.at("proof");const auto proof_path=path.parent_path()/std::filesystem::u8path(proof_ref.at("file").get<std::string>());eventio::verify(proof_path,proof_ref.at("sha256"));const auto proof=eventio::read_json(proof_path);
      const auto& src=proof.at("original_source_packet");const auto source_path=proof_path.parent_path()/std::filesystem::u8path(src.at("path").get<std::string>());eventio::verify(source_path,src.at("sha256"));const auto original=eventio::read_json(source_path);
      if(original.at("source_pts")!=pts[25537]||proof.at("feature_id")!="ice.lane3:edge2")throw std::runtime_error("active W proof source frame/material mismatch");
      const eventio::Json* feature=nullptr;for(const auto& f:original.at("features"))if(f.at("id")==proof.at("feature_id")){if(feature)throw std::runtime_error("active W proof feature is ambiguous");feature=&f;}
      if(!feature)throw std::runtime_error("active W proof feature is absent");const auto& line=feature->at("source_line_fit");const auto coefficient=line.at("coefficient_y_equals_ax_plus_b").get<std::vector<double>>();const auto cov=line.at("covariance_ab").get<std::vector<std::vector<double>>>();
      if(coefficient.size()!=2||cov.size()!=2||cov[0].size()!=2||cov[1].size()!=2)throw std::runtime_error("active W lacks the original finite line covariance");
      // Recompute the Moore-Penrose information operator from the unchanged
      // source coefficient covariance and retained native source coordinates.
      // This accepts SVD row signs/rotations, but never a slice of old W.
      auto check_marginal=[&](const std::vector<std::size_t>& members,const std::vector<double>& matrix){
        const auto count=2*members.size();const double slope=coefficient[0],norm=std::hypot(slope,1.),nx=-slope/norm,ny=1./norm;
        const double ca=cov[0][0],cb=cov[0][1],cc=cov[1][1],det=ca*cc-cb*cb;
        if(!std::isfinite(slope)||!std::isfinite(det)||!(ca>0)||!(cc>0)||!(det>0)||cov[1][0]!=cb)throw std::runtime_error("original source coefficient covariance is not positive definite");
        std::vector<std::array<double,2>> J(count);double aa=0,ab=0,bb=0;
        for(std::size_t k=0;k<members.size();++k){const auto row=members[k];if(frames[row]!=frames[25537]||material[row]!=material[25537])throw std::runtime_error("active source covariance crosses frame/material");const double x=uv[2*row];J[2*k]={nx*ny*x,nx*ny};J[2*k+1]={ny*ny*x,ny*ny};}
        for(auto v:J){aa+=v[0]*v[0];ab+=v[0]*v[1];bb+=v[1]*v[1];}const double dg=aa*bb-ab*ab;if(!(dg>0)||!std::isfinite(dg))throw std::runtime_error("retained source line design lost rank");
        const double ga=bb/dg,gb=-ab/dg,gc=aa/dg,ia=cc/det,ib=-cb/det,ic=ca/det;
        const double ma=ga*(ia*ga+ib*gb)+gb*(ib*ga+ic*gb),mb=ga*(ia*gb+ib*gc)+gb*(ib*gb+ic*gc),mc=gb*(ia*gb+ib*gc)+gc*(ib*gb+ic*gc);
        double error=0,scale=0;for(std::size_t j=0;j<count;++j)for(std::size_t k=0;k<count;++k){const double expected=J[j][0]*(ma*J[k][0]+mb*J[k][1])+J[j][1]*(mb*J[k][0]+mc*J[k][1]);const double actual=matrix[j]*matrix[k]+matrix[count+j]*matrix[count+k];error=std::max(error,std::abs(actual-expected));scale=std::max(scale,std::abs(expected));}
        if(!std::isfinite(error)||error>2e-8*std::max(1.,scale))throw std::runtime_error("active W is not the full retained covariance marginal information operator");
      };
      check_marginal(found->rows,found->W);check_marginal(retained,W);
      excluded_original_groups.push_back(*found);found->rows=retained;found->W=W;found->provenance+="; user-authorized single-row source exclusion; retained covariance marginal "+active_statistics_hash;
    }
    std::vector<unsigned> count(rows.size(),0);std::size_t rank=0;for(const auto& group:groups){rank+=group.rank;for(auto row:group.rows){if(!is_active(row))throw std::runtime_error("excluded source remains in an active statistical factor");++count[row];}}
    for(std::size_t i=0;i<rows.size();++i)if(count[i]!=(is_active(i)?1u:0u))throw std::runtime_error("active full statistics do not cover every active source exactly once");
    if(rank!=214112)throw std::runtime_error("retained rank2 group must preserve full active statistical residual rank");statistics_complete=true;
  }
  void load_confirmed_resolution(const std::filesystem::path& dir){
    const auto n=rows.size();const auto original=eventio::read_array<std::uint32_t>(dir,semantics.at("original_material_index"),"<u4");if(original!=material||semantics.at("original_material_index").at("sha256")!=manifest.at("arrays").at("material_index").at("sha256"))throw std::runtime_error("resolved binding changed raw material bytes");
    resolved_material=eventio::read_array<std::uint32_t>(dir,semantics.at("resolved_material_index"),"<u4");resolved_variant=eventio::read_array<std::uint32_t>(dir,semantics.at("resolved_variant_index"),"<u4");resolved_variant_ids=semantics.at("resolved_variant_ids").get<std::vector<std::string>>();
    if(resolved_material.size()!=n||resolved_variant.size()!=n||resolved_variant_ids!=variant_ids)throw std::runtime_error("resolved binding source/table dimensions changed");
    const auto& binding_ref=semantics.at("confirmed_bindings");const auto binding_path=dir/binding_ref.at("file").get<std::string>();eventio::verify(binding_path,binding_ref.at("sha256"));confirmed_bindings=eventio::read_json(binding_path);
    if(confirmed_bindings.at("schema")!="rinkfit.confirmed_material_resolution.v1"||confirmed_bindings.at("observations")!=n||confirmed_bindings.at("source_identity")!=source_hash||confirmed_bindings.at("parent_semantic_sha256")!=semantics.at("parent_semantic_manifest").at("sha256")||!confirmed_bindings.at("raw_five_arrays_and_original_W_E_unchanged").get<bool>())throw std::runtime_error("confirmed binding immutable source identity mismatch");
    std::set<std::size_t> in_run,confirmed;for(const auto& run:runs)in_run.insert(run.rows.begin(),run.rows.end());
    // Visual source reviews are a distinct authority. They never inherit a
    // human confirmation for another row. Cache only hashes checked in this
    // immutable packet load (not solver evaluations or later mutable loads).
    std::map<std::string,std::string> checked_review_assets;
    auto reviewed_asset=[&](const eventio::Json& ref){
      const auto path=dir/std::filesystem::u8path(ref.at("path").get<std::string>());
      const auto key=path.u8string(),expected=ref.at("sha256").get<std::string>();
      const auto found=checked_review_assets.find(key);
      if(found!=checked_review_assets.end()){
        if(found->second!=expected)throw std::runtime_error("conflicting native-review asset hashes");
      }else{
        try{eventio::verify(path,expected);}catch(const std::exception&){
          throw std::runtime_error("native-review asset hash/read failure: "+path.u8string());
        }
        checked_review_assets.emplace(key,expected);
      }
      return path;
    };

    for(const auto& binding:confirmed_bindings.at("bindings")){
      const bool direct_human=binding.contains("direct_human_confirmation"),source_review=binding.contains("source_image_review");
      if(direct_human==source_review)throw std::runtime_error("binding requires exactly one mutually exclusive same-row evidence authority");
      const char* own_scope=source_review?"review_is_not_extended_to_any_other_source_row":"confirmation_is_not_extended_to_any_other_source_row";
      const char* other_scope=source_review?"confirmation_is_not_extended_to_any_other_source_row":"review_is_not_extended_to_any_other_source_row";
      if(!binding.contains(own_scope)||!binding.at(own_scope).is_boolean()||!binding.at(own_scope).get<bool>()||binding.contains(other_scope))
        throw std::runtime_error("binding exact-row scope is missing/false or uses the other evidence authority's scope");
      const auto i=binding.at("source_row").get<std::size_t>();if(i>=n||!confirmed.insert(i).second||kind[i]!=0||in_run.count(i)||resolved_material[i]>=material_labels.size()||resolved_variant[i]>=resolved_variant_ids.size())throw std::runtime_error("confirmed material row is duplicated/profile/ordered/outside original source");
      if(binding.at("native_pts")!=pts[i]||binding.at("source_point_id")!=point_ids[point[i]]||binding.at("source_uv")!=eventio::Json::array({uv[2*i],uv[2*i+1]})||binding.at("original_source_label")!=material_labels[material[i]]||binding.at("original_material_index")!=material[i]||binding.at("original_variant_id")!=variant_ids[variant[i]]||binding.at("resolved_model_material_index")!=resolved_material[i]||binding.at("resolved_model_target_id")!=material_labels[resolved_material[i]]||binding.at("resolved_variant_id")!=resolved_variant_ids[resolved_variant[i]])throw std::runtime_error("confirmed source/resolved material key mismatch");
      if(binding.at("allowed_path_indices").empty()||binding.at("resolved_path_ids").size()!=binding.at("allowed_path_indices").size())throw std::runtime_error("confirmed material has no explicit finite allowed path");
      if(direct_human){
      const auto& ref=binding.at("direct_human_confirmation");const auto path=dir/ref.at("path").get<std::string>();eventio::verify(path,ref.at("sha256"));const auto human=eventio::read_json(path);
      if(human.at("schema")!="rinkfit.direct-human-material-confirmation.v1"||human.at("source")!="direct user message in this root conversation"||human.at("user_text").get<std::string>().empty()||human.at("source_row")!=i||human.at("shot_id")!=shot_id||human.at("native_pts")!=pts[i]||human.at("source_uv")!=binding.at("source_uv")||human.at("raw_source_point_id")!=point_ids[point[i]]||human.at("raw_source_label")!=material_labels[material[i]]||human.at("original_npz_source_identity")!=source_hash||!human.at("raw_arrays_W_E_obligations_remain_immutable").get<bool>()||human.at("confirmation_applies_to_other_source_rows").get<bool>())throw std::runtime_error("material resolution lacks direct same-row human confirmation");
      eventio::verify(human.at("annotated_source_image").get<std::string>(),human.at("image_sha256"));
      }else{
        const auto proof_path=reviewed_asset(binding.at("source_image_review"));
        const auto proof=eventio::read_json(proof_path);
        if(proof.at("schema")!="rinkfit.native-image-material-review.v1"
            ||proof.at("review_origin")!="codex_primary_native_frame_visual_review"
            ||!proof.at("not_a_direct_human_confirmation").get<bool>()
            ||!proof.at("no_extension_of_anchor_human_confirmation").get<bool>()
            ||proof.at("review_applies_to_other_source_rows").get<bool>()
            ||!proof.at("raw_arrays_W_E_obligations_remain_immutable").get<bool>()
            ||proof.at("source_row")!=i||proof.at("native_pts")!=pts[i]
            ||proof.at("native_time_base")!=manifest.at("native_time_base")
            ||proof.at("original_npz_source_identity")!=source_hash
            ||proof.at("raw_source_point_id")!=point_ids[point[i]]
            ||proof.at("raw_source_label")!=material_labels[material[i]]
            ||proof.at("source_uv")!=binding.at("source_uv")
            ||proof.at("resolved_model_target_id")!=binding.at("resolved_model_target_id"))
          throw std::runtime_error("source-image review authority/same-row identity mismatch; it is not a human confirmation");
        for(const char* key:{"exact_native_frame","native_source_video","visual_diagnostic","original_fragment_lineage"})
          reviewed_asset(proof.at(key));
        const auto& frame=proof.at("exact_native_frame");
        if(frame.at("native_pts")!=pts[i]||frame.at("width")!=manifest.at("native_image_wh").at(0)
            ||frame.at("height")!=manifest.at("native_image_wh").at(1)
            ||!proof.at("visual_diagnostic").at("green_is_immutable_source_UV").get<bool>())
          throw std::runtime_error("source-image review does not bind the exact native source frame/UV");
        const auto lineage=eventio::read_json(reviewed_asset(proof.at("original_fragment_lineage")));
        if(lineage.at("archive_npz").at("sha256")!=source_hash)
          throw std::runtime_error("native-image review lineage source archive mismatch");
        bool exact_lineage=false;
        for(const auto& point:lineage.at("points"))if(point.at("source_row")==i){
          if(exact_lineage||point.at("native_pts")!=pts[i]||point.at("source_point_id")!=point_ids[this->point[i]]
              ||point.at("raw_label")!=material_labels[material[i]]||point.at("source_uv")!=binding.at("source_uv"))
            throw std::runtime_error("native-image review lineage same-row key mismatch");
          exact_lineage=true;
        }
        if(!exact_lineage)throw std::runtime_error("native-image review row missing from original fragment lineage");
        const auto& anchor=proof.at("separate_human_confirmed_anchor");
        const auto ai=anchor.at("source_row").get<std::size_t>();
        if(ai>=n||ai==i||pts[ai]!=pts[i])throw std::runtime_error("native-image review anchor must be a different source row at the same native PTS");
        const auto human=eventio::read_json(reviewed_asset(anchor));
        if(human.at("schema")!="rinkfit.direct-human-material-confirmation.v1"
            ||human.at("source")!="direct user message in this root conversation"
            ||human.at("user_text").get<std::string>().empty()||human.at("source_row")!=ai
            ||human.at("shot_id")!=shot_id||human.at("native_pts")!=pts[ai]
            ||human.at("source_uv")!=eventio::Json::array({uv[2*ai],uv[2*ai+1]})
            ||human.at("raw_source_point_id")!=point_ids[point[ai]]
            ||human.at("raw_source_label")!=material_labels[material[ai]]
            ||human.at("original_npz_source_identity")!=source_hash
            ||!human.at("raw_arrays_W_E_obligations_remain_immutable").get<bool>()
            ||human.at("confirmation_applies_to_other_source_rows").get<bool>())
          throw std::runtime_error("native-image review anchor lacks independent exact same-row human confirmation");
        eventio::verify(std::filesystem::u8path(human.at("annotated_source_image").get<std::string>()),human.at("image_sha256"));
        bool exact_anchor_binding=false;
        for(const auto& other:confirmed_bindings.at("bindings"))if(other.at("source_row")==ai){
          if(!other.contains("direct_human_confirmation")||other.contains("source_image_review")
              ||other.at("direct_human_confirmation").at("sha256")!=anchor.at("sha256")
              ||other.at("resolved_model_target_id")!=binding.at("resolved_model_target_id"))
            throw std::runtime_error("native-image review anchor binding target/authority mismatch");
          const auto bound_path=dir/std::filesystem::u8path(other.at("direct_human_confirmation").at("path").get<std::string>());
          if(bound_path.lexically_normal()!=reviewed_asset(anchor).lexically_normal())
            throw std::runtime_error("native-image review anchor binding proof path mismatch");
          exact_anchor_binding=true;
        }
        if(!exact_anchor_binding)throw std::runtime_error("native-image review separate human anchor binding is absent");
      }
      for(const auto& proof:binding.at("exact_source_geometry_provenance")){const auto path=dir/proof.at("path").get<std::string>();eventio::verify(path,proof.at("sha256"));const auto data=eventio::read_json(path);if(data.at(eventio::Json::json_pointer(proof.at("json_pointer").get<std::string>())).is_null())throw std::runtime_error("missing exact source geometry provenance");}
    }
    for(std::size_t i=0;i<n;++i){if(resolved_material[i]>=material_labels.size()||resolved_variant[i]>=resolved_variant_ids.size())throw std::runtime_error("resolved identity outside existing full material/variant tables");const bool changed=material[i]!=resolved_material[i]||variant[i]!=resolved_variant[i];if(changed!=(confirmed.count(i)!=0))throw std::runtime_error("unconfirmed material change or confirmation silently extended to another row");}
  }
  void load_query(const std::filesystem::path& path,const std::filesystem::path& initialization) {
    const auto j=eventio::read_json(path);const auto dir=path.parent_path();query_hash=eventio::file_sha256(path);
    if(j.at("schema")!="rinkfit.immutable_native_query_curve_packet.v1"||j.at("profiles")!=961||!semantic_ancestor_hashes.count(j.at("semantic_manifest").at("sha256").get<std::string>()))throw std::runtime_error("exact query/source-semantic identity mismatch (no verified immutable parent)");
    const auto& a=j.at("arrays");
    const auto source=eventio::read_array<std::uint64_t>(dir,a.at("source_rows"),"<u8");
    const auto offsets=eventio::read_array<std::uint64_t>(dir,a.at("offsets"),"<u8");
    const auto alpha=eventio::read_array<double>(dir,a.at("alpha"),"<f8"),xy=eventio::read_array<double>(dir,a.at("uv"),"<f8"),bounds=eventio::read_array<double>(dir,a.at("bounds"),"<f8");
    const auto U=eventio::read_array<double>(dir,a.at("numerator_u"),"<f8"),V=eventio::read_array<double>(dir,a.at("numerator_v"),"<f8"),D=eventio::read_array<double>(dir,a.at("denominator"),"<f8");
    if(source.size()!=961||offsets.size()!=962||bounds.size()!=1922||U.size()!=3844||V.size()!=3844||D.size()!=3844||xy.size()!=2*alpha.size()||offsets.front()!=0||offsets.back()!=alpha.size())throw std::runtime_error("exact query dimensions mismatch");
    eventio::verify(j.at("certificates").at("path").get<std::string>(),j.at("certificates").at("sha256"));
    const auto init=eventio::read_json(initialization);if(init.at("query_packet_sha256")!=query_hash)throw std::runtime_error("native query alpha initialization identity mismatch");
    const auto initial=eventio::read_array<double>(initialization.parent_path(),init.at("alpha"),"<f8");if(initial.size()!=961)throw std::runtime_error("native query alpha seed dimensions mismatch");
    query_index.assign(rows.size(),-1);queries.clear();queries.reserve(961);
    for(std::size_t i=0;i<source.size();++i){const auto row=source[i],begin=offsets[i],end=offsets[i+1];
      if(row>=rows.size()||kind[row]!=2||query_index[row]>=0||end<=begin+1||end>alpha.size())throw std::runtime_error("query row/branch identity mismatch");
      EventQueryRecord q;q.source_row=row;q.provenance="exact immutable rational native source query "+query_hash;q.lower=bounds[2*i];q.upper=bounds[2*i+1];q.initial_alpha=initial[i];
      if(!(q.lower<=q.initial_alpha&&q.initial_alpha<=q.upper))throw std::runtime_error("query alpha seed outside original finite domain");
      q.alpha.assign(alpha.begin()+begin,alpha.begin()+end);for(std::size_t k=begin;k<end;++k){if(!std::isfinite(alpha[k])||!std::isfinite(xy[2*k])||!std::isfinite(xy[2*k+1])||(k>begin&&alpha[k]<=alpha[k-1]))throw std::runtime_error("query sample is nonfinite/nonmonotone");q.native_uv.push_back({xy[2*k],xy[2*k+1]});}
      if(std::abs(q.alpha.front()-q.lower)>1e-10||std::abs(q.alpha.back()-q.upper)>1e-10)throw std::runtime_error("query knots do not cover original finite interval");
      q.numerator_u.assign(U.begin()+4*i,U.begin()+4*i+4);q.numerator_v.assign(V.begin()+4*i,V.begin()+4*i+4);q.denominator.assign(D.begin()+4*i,D.begin()+4*i+4);
      for(double v:q.numerator_u)if(!std::isfinite(v))throw std::runtime_error("nonfinite rational numerator");for(double v:q.numerator_v)if(!std::isfinite(v))throw std::runtime_error("nonfinite rational numerator");for(double v:q.denominator)if(!std::isfinite(v))throw std::runtime_error("nonfinite rational denominator");
      query_index[row]=int(queries.size());queries.push_back(std::move(q));alpha_bounds[2*row]=bounds[2*i];alpha_bounds[2*row+1]=bounds[2*i+1];
      observations[row].original_curved_query=true;observations[row].profile_alpha_lower_px=bounds[2*i];observations[row].profile_alpha_upper_px=bounds[2*i+1];
    }
    for(std::size_t row=0;row<rows.size();++row)if((kind[row]==2)!=(query_index[row]>=0))throw std::runtime_error("not every genuine profile has one immutable query");
  }
};
} // namespace rinkfit
