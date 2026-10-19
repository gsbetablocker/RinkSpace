// Included inside EventPacket.  A distinct audited-source target, never a human authority.
  void load_source_audit_revision_layer() {
    const auto selected=semantics,selected_manifest=manifest;
    const auto selected_path=semantic_path;const auto selected_hash=semantic_hash;
    const auto dir=selected_path.parent_path();
    if(!selected.contains("source_audit_parent_manifest")||!selected.contains("source_audit_revision")||has_source_audit_revision)
      throw std::runtime_error("incomplete/repeated source audit layer");
    if(!manifest.contains("semantic_packet")||manifest.at("semantic_packet").at("sha256")!=selected_hash)
      throw std::runtime_error("audited target differs from immutable packet binding");
    auto file=[](const std::filesystem::path& base,const eventio::Json& ref){
      const char* key=ref.contains("file")?"file":"path";
      const auto p=(base/std::filesystem::u8path(ref.at(key).get<std::string>())).lexically_normal();
      eventio::verify(p,ref.at("sha256"));return p;
    };
    const auto& parent_ref=selected.at("source_audit_parent_manifest");
    const auto parent_path=file(dir,parent_ref);const auto parent_hash=parent_ref.at("sha256").get<std::string>();
    EventPacket inherited(packet_path);
    inherited.manifest["semantic_packet"]={{"manifest",parent_path.u8string()},{"sha256",parent_hash}};
    inherited.load_semantics(parent_path);
    if(!inherited.has_source_exclusion||inherited.has_source_audit_revision)
      throw std::runtime_error("source audit needs its unchanged historical authority parent");
    const auto parent=inherited.semantics;
    if(selected.size()!=parent.size()+2)throw std::runtime_error("source audit changed semantic field inventory");
    const std::set<std::string> changed={"active_source_mask","active_statistics","active_observations"};
    for(const auto& item:parent.items())if(!selected.contains(item.key())||(!changed.count(item.key())&&selected.at(item.key())!=item.value()))
      throw std::runtime_error("source audit changed unrelated inherited source field: "+item.key());
    for(const auto& item:selected.items())if(!parent.contains(item.key())&&item.key()!="source_audit_parent_manifest"&&item.key()!="source_audit_revision")
      throw std::runtime_error("unknown source audit authority field");
    const auto audit_path=file(dir,selected.at("source_audit_revision"));const auto audit=eventio::read_json(audit_path);
    const auto ad=audit_path.parent_path();const auto n=rows.size();
    const auto newly=audit.at("new_audit_exclusions").get<std::vector<std::size_t>>();
    if(newly.empty()||!std::is_sorted(newly.begin(),newly.end())||std::adjacent_find(newly.begin(),newly.end())!=newly.end())
      throw std::runtime_error("source audit exclusions are not unique sorted original rows");
    if(audit.at("schema")!="rinkfit.native-source-audit-revision.v1"||audit.at("parent_semantic_sha256")!=parent_hash||audit.at("raw_observations")!=n||audit.at("source_identity")!=source_hash||audit.at("historical_human_authorized_exclusions")!=eventio::Json(inherited.excluded_rows)||!audit.at("raw_rows_UV_ID_labels_W_E_remain_immutable").get<bool>()||!audit.at("not_a_direct_human_confirmation").get<bool>()||audit.at("exclusions").size()!=newly.size())
      throw std::runtime_error("source audit exact-parent/source authority differs");
    auto mask=eventio::read_array<std::uint8_t>(dir,selected.at("active_source_mask"),"|u1");
    if(mask.size()!=n)throw std::runtime_error("source audit changed the original row ABI");
    auto expected=inherited.active_mask;std::set<std::size_t> wanted;
    for(auto r:newly){if(r>=n||!expected[r]||inherited.kind[r]!=0)throw std::runtime_error("source audit row not a new active ordinary source");expected[r]=0;wanted.insert(r);}
    if(mask!=expected)throw std::runtime_error("source audit mask difference exceeds reviewed rows");
    std::map<std::size_t,eventio::Json> registry;
    {const auto rp=file(parent_path.parent_path(),parent.at("assets").at("source_rows.jsonl"));std::ifstream input(rp);if(!input)throw std::runtime_error("missing original source registry");std::string line;std::size_t row=0;
      while(std::getline(input,line)){if(line.empty())continue;const auto rec=eventio::Json::parse(line);if(rec.at("source_row")!=row)throw std::runtime_error("original source registry order differs");if(wanted.count(row))registry.emplace(row,rec);++row;}
      if(row!=n||registry.size()!=wanted.size())throw std::runtime_error("source audit registry coverage differs");}
    for(std::size_t k=0;k<newly.size();++k){
      const auto row=newly[k];const auto& record=audit.at("exclusions").at(k);
      if(record.at("source_row")!=row||record.at("native_pts")!=pts[row]||record.at("source_point_id")!=point_ids[point[row]]||record.at("original_source_label")!=material_labels[material[row]]||record.at("source_uv")!=eventio::Json::array({uv[2*row],uv[2*row+1]}))
        throw std::runtime_error("source audit original raw row key differs");
      const auto ep=file(ad,record.at("native_image_audit"));const auto ev=eventio::read_json(ep);const auto ed=ep.parent_path();
      if(ev.at("schema")!="rinkfit.native-image-visibility-audit.v1"||ev.at("authority")!="codex_native_source_audit_under_user_request"||!ev.at("not_a_direct_human_confirmation").get<bool>()||ev.at("user_personally_confirmed_pixel").get<bool>()||ev.at("review_applies_to_other_source_rows").get<bool>()||ev.at("classification")!="foreground_occlusion_not_static_board_material"||!ev.at("native_video_PTS_redecode_BGR_bitexact").get<bool>()||ev.at("current_user_request_scope").get<std::string>().empty()||ev.at("project_rule_applied").get<std::string>().empty()||ev.at("source_row")!=row||ev.at("native_pts")!=pts[row]||ev.at("source_uv")!=record.at("source_uv")||ev.at("original_source_point_id")!=record.at("source_point_id")||ev.at("original_source_label")!=record.at("original_source_label"))
        throw std::runtime_error("source audit lacks exact native visibility evidence");
      for(const char* key:{"exact_native_frame","visual_diagnostic","source_packet","source_registry_record","project_rules"})file(ed,ev.at(key));
      const auto canonical=registry.at(row);
      if(eventio::read_json(file(ad,record.at("source_record")))!=canonical||eventio::read_json(file(ed,ev.at("source_registry_record")))!=canonical)
        throw std::runtime_error("source audit registry copy is not its immutable original row");
      const auto original=eventio::read_json(file(parent_path.parent_path(),canonical.at("source_packet")));
      if(eventio::read_json(file(ed,ev.at("source_packet")))!=original)throw std::runtime_error("source audit packet differs from original registry");
      const auto feature_id=canonical.at("source_feature_id").get<std::string>();const eventio::Json* feature=nullptr;
      for(const auto& f:original.at("features"))if(f.at("id")==feature_id){if(feature)throw std::runtime_error("ambiguous audited original feature");feature=&f;}
      const auto si=canonical.at("source_segment_index").get<std::size_t>(),vi=canonical.at("original_vertex_index").get<std::size_t>();
      const auto pid=feature_id+"/segment/"+std::to_string(si)+"/vertex/"+std::to_string(vi);
      const auto& source=original.at("source");const auto& image=ev.at("exact_native_frame");
      if(!feature||!canonical.at("native_query_id").is_null()||feature->at("segments").at(si).at("uv").at(vi)!=record.at("source_uv")||pid!=point_ids[point[row]]||feature_id!=material_labels[material[row]]||canonical.at("snapshot_frame")!=frames[row]||original.at("source_pts")!=pts[row]||source.at("native_pts")!=pts[row]||source.at("actual_pts")!=pts[row]||source.at("time_base")!="1/12800"||image.at("native_pts")!=pts[row]||image.at("sha256")!=source.at("png_sha256")||eventio::Json::array({image.at("height"),image.at("width")})!=source.at("native_hw"))
        throw std::runtime_error("source audit original material/vertex/PTS/PNG differs");
      for(const auto& run:inherited.runs)if(std::find(run.rows.begin(),run.rows.end(),row)!=run.rows.end())throw std::runtime_error("audited row needs an explicit active source topology revision");
    }
    const auto stat_path=file(dir,selected.at("active_statistics"));const auto stats=eventio::read_json(stat_path);const auto sd=stat_path.parent_path();
    const auto mask_hash=selected.at("active_source_mask").at("sha256").get<std::string>(),audit_hash=selected.at("source_audit_revision").at("sha256").get<std::string>();
    const auto active_count=std::count(mask.begin(),mask.end(),std::uint8_t(1));
    if(stats.at("schema")!="rinkfit.source-audit-active-statistics.v1"||stats.at("parent_semantic_sha256")!=parent_hash||stats.at("source_audit_revision_sha256")!=audit_hash||stats.at("active_source_mask_sha256")!=mask_hash||stats.at("active_observations")!=active_count||audit.at("active_observations")!=active_count||selected.at("active_observations")!=active_count||stats.at("original_group_count")!=inherited.original_group_count||stats.at("original_groups_asset_sha256")!=parent.at("assets").at("groups.jsonl").at("sha256")||stats.at("original_weights_array_sha256")!=parent.at("arrays").at("weights").at("sha256"))
      throw std::runtime_error("source audit active covariance ledger identity differs");
    std::vector<eventio::Json> raw_groups;
    {std::ifstream f(file(parent_path.parent_path(),parent.at("assets").at("groups.jsonl")));std::string line;while(std::getline(f,line))if(!line.empty())raw_groups.push_back(eventio::Json::parse(line));}
    if(raw_groups.size()!=inherited.original_group_count)throw std::runtime_error("source audit original covariance inventory differs");
    std::set<std::string> affected,replaced;std::map<std::string,eventio::Json> canonical_replacements;
    for(const auto& r:inherited.active_statistics.at("replacements"))canonical_replacements.emplace(r.at("group_id").get<std::string>(),r);
    for(const auto& g:inherited.groups)for(auto row:g.rows)if(wanted.count(row)&&g.id.rfind("independent/",0)!=0)affected.insert(g.id);
    for(const auto& rep:stats.at("new_replacements")){
      const auto id=rep.at("group_id").get<std::string>();const auto ordinal=rep.at("original_group_ordinal").get<std::size_t>();
      if(!affected.count(id)||!replaced.insert(id).second||ordinal>=raw_groups.size()||raw_groups[ordinal].at("group_id")!=id)throw std::runtime_error("unexpected/duplicate audited covariance replacement");
      auto found=std::find_if(inherited.groups.begin(),inherited.groups.end(),[&](const EventStatisticalGroup& g){return g.id==id;});
      if(found==inherited.groups.end())throw std::runtime_error("missing audited covariance parent");
      EventStatisticalGroup original=*found;
      for(const auto& g:inherited.excluded_original_groups)if(g.id==id)original=g;
      std::vector<std::size_t> retained,keep,excluded;
      for(std::size_t j=0;j<original.rows.size();++j){const auto row=original.rows[j];if(mask[row]){retained.push_back(row);keep.push_back(2*j);keep.push_back(2*j+1);}else excluded.push_back(row);}
      if(rep.at("source_rows")!=eventio::Json(retained)||retained.empty())throw std::runtime_error("audit covariance retained original order differs");
      const auto nr=rep.at("rank").get<std::size_t>();const auto& wd=rep.at("W");
      if(wd.at("shape")!=eventio::Json::array({nr,2*retained.size()}))throw std::runtime_error("audit W dimensions differ");
      const auto W=eventio::read_array<double>(sd,wd,"<f8");
      const auto proof_path=file(sd,rep.at("proof"));const auto proof=eventio::read_json(proof_path);
      if(proof.at("schema")!="rinkfit.source-audit-covariance-marginal-proof.v1"||proof.at("method")!="retained_original_supported_covariance_marginal"||proof.at("group_id")!=id||proof.at("original_group_ordinal")!=ordinal||proof.at("original_source_rows")!=eventio::Json(original.rows)||proof.at("source_rows")!=eventio::Json(retained)||proof.at("excluded_source_rows")!=eventio::Json(excluded)||proof.at("original_W_sha256")!=eventpacket_detail::double_hash(original.W)||proof.at("original_matrix_shape")!=eventio::Json::array({original.rank,2*original.rows.size()})||proof.at("marginal_matrix_shape")!=eventio::Json::array({nr,2*retained.size()})||proof.at("old_W_column_slice_used").get<bool>()||!proof.at("original_native_noise_and_retained_source_covariance_not_refitted").get<bool>())
        throw std::runtime_error("audit covariance proof/source dimensions differ");
      for(std::size_t j=0;j<original.rows.size();++j){const auto r=original.rows[j];if(proof.at("original_uv_native").at(j)!=eventio::Json::array({uv[2*r],uv[2*r+1]})||proof.at("original_source_point_ids").at(j)!=point_ids[point[r]])throw std::runtime_error("audit covariance proof changes original source pixels");}
      const auto threshold=raw_groups[ordinal].at("numerical_rank_evidence").at("numerical_rank_threshold").get<double>();
      if(proof.at("rank_proof").at("source_rank_threshold")!=threshold||proof.at("rank_proof").at("retained_rank")!=nr)throw std::runtime_error("audit covariance source rank threshold differs");
      validate_supported_covariance_marginal(original.W,original.rank,2*original.rows.size(),keep,W,nr,SourceCovarianceRankProof{threshold,nr});
      if(proof.contains("physical_source_noise_provenance"))file(proof_path.parent_path(),proof.at("physical_source_noise_provenance"));
      if(std::none_of(inherited.excluded_original_groups.begin(),inherited.excluded_original_groups.end(),[&](const EventStatisticalGroup& g){return g.id==id;}))inherited.excluded_original_groups.push_back(original);
      found->rows=retained;found->rank=nr;found->W=W;found->provenance+="; distinct native source audit covariance marginal "+audit_hash;
      canonical_replacements[id]=rep;
    }
    if(replaced!=affected)throw std::runtime_error("incomplete audited covariance marginalization");
    inherited.groups.erase(std::remove_if(inherited.groups.begin(),inherited.groups.end(),[&](const EventStatisticalGroup& g){return g.id.rfind("independent/",0)==0&&g.rows.size()==1&&!mask[g.rows[0]];}),inherited.groups.end());
    inherited.ordinary_rows.erase(std::remove_if(inherited.ordinary_rows.begin(),inherited.ordinary_rows.end(),[&](std::uint64_t r){return !mask[r];}),inherited.ordinary_rows.end());
    std::set<std::string> flattened;
    for(const auto& r:stats.at("replacements")){
      const auto id=r.at("group_id").get<std::string>();
      if(!canonical_replacements.count(id)||!flattened.insert(id).second)throw std::runtime_error("unvalidated/duplicate flattened audited W");
      const auto found=std::find_if(inherited.groups.begin(),inherited.groups.end(),[&](const EventStatisticalGroup& g){return g.id==id;});
      if(found==inherited.groups.end()||r.at("source_rows")!=eventio::Json(found->rows)||r.at("rank")!=found->rank||r.at("W").at("shape")!=eventio::Json::array({found->rank,2*found->rows.size()})||eventio::read_array<double>(sd,r.at("W"),"<f8")!=found->W||r.at("proof").at("sha256")!=canonical_replacements.at(id).at("proof").at("sha256"))throw std::runtime_error("flattened audited W differs from validated covariance");
      file(sd,r.at("proof"));
    }
    if(flattened.size()!=canonical_replacements.size())throw std::runtime_error("missing flattened audited W");
    std::vector<unsigned> partition(n,0);std::size_t rank=0;
    for(const auto& g:inherited.groups){rank+=g.rank;for(auto r:g.rows){if(r>=n||!mask[r])throw std::runtime_error("audited inactive source remains in W");++partition[r];}}
    for(std::size_t r=0;r<n;++r)if(partition[r]!=(mask[r]?1u:0u))throw std::runtime_error("audited W does not count every selected source exactly once");
    if(stats.at("formal_residual_rows")!=rank)throw std::runtime_error("audited W formal rank differs");
    *this=std::move(inherited);manifest=selected_manifest;semantics=selected;semantic_path=selected_path;semantic_hash=selected_hash;
    semantic_ancestor_hashes.insert(parent_hash);semantic_ancestor_hashes.insert(selected_hash);
    active_mask=std::move(mask);active_rows.clear();excluded_rows.clear();
    for(std::size_t r=0;r<n;++r)(active_mask[r]?active_rows:excluded_rows).push_back(r);
    for(auto r:newly){observations[r].active=false;observations[r].exclusion_reason="native source image audit: foreground occlusion";observations[r].exclusion_id=audit_hash;}
    active_mask_hash=mask_hash;active_statistics=stats;active_statistics_hash=selected.at("active_statistics").at("sha256");formal_residual_rows=rank;statistics_complete=true;
    source_audit_revision=audit;source_audit_revision_hash=audit_hash;has_source_audit_revision=true;
  }
