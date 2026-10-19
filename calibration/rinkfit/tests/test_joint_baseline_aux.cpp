#include "rinkfit/joint_baseline_aux.hpp"
#include <iostream>
using namespace rinkfit;
int main(int argc,char**argv){try{
 if(argc!=2)throw std::runtime_error("test output directory required");std::filesystem::path out=argv[1];std::filesystem::create_directories(out);
 std::ofstream(out/"model.json")<<"{}";std::ofstream(out/"packet.json")<<"{}";
 JointEventModel model;EventParameterBlock block;block.name="test";block.values={1.};model.blocks={block};model.source_json={{"blocks",eventio::Json::array({{{"values",{1.}}}})}};
 EventPath path;path.physical_point=true;path.id="fixed/corner";EventMaterial material;material.label="Ad2.left_top";material.paths={path};model.materials={material};model.path_station_intervals[{0,0}]={{{0.,1.}}};
 JointPacket packet;packet.source_hash=eventio::file_sha256(out/"packet.json");for(int i=0;i<2;++i){JointObservation r;r.source_row_id="same_vertex";r.source_dataset_id="s01_original";r.shot_id="s01";r.pts=256256+256*i;r.material=0;r.kind=1;r.active=true;packet.rows.push_back(r);}
 eventio::Json files;auto write=[&](const std::string&name,const auto&v,const std::string&dtype){std::ofstream f(out/name,std::ios::binary);f.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(v[0]));f.close();files[name]={{"file",(out/name).string()},{"sha256",eventio::file_sha256(out/name)},{"dtype",dtype},{"shape",{v.size()}}};};
 write("rows.u64",std::vector<std::uint64_t>{0,1},"<u8");write("station.f64",std::vector<double>{.25,.75},"<f8");write("path.u32",std::vector<std::uint32_t>{0,0},"<u4");write("row.f64",std::vector<double>{111.,222.},"<f8");write("alpha.f64",std::vector<double>{0.,0.},"<f8");
 eventio::Json manifest={{"schema","rinkfit.exact_native_baseline_auxiliary_import.v1"},{"all_original_UV_ID_PTS_material_kind_normal_Query_exact",true},{"v98_auxiliary_state_loaded",false},{"source_membership_UV_W_E_or_qualification_changed",false},{"current_packet",{{"sha256",packet.source_hash}}},{"model",{{"sha256",eventio::file_sha256(out/"model.json")}}},{"files",files},{"preserved_active_rows",2},{"Ad2_same623",{{"n",2}}},{"maximum_saved_d78_prediction_change_px",0.}};
 eventio::atomic_json(out/"manifest.json",manifest);auto ref=eventio::Json{{"file",(out/"manifest.json").string()},{"sha256",eventio::file_sha256(out/"manifest.json")}};
 int passed=0;
 {JointProblemBuilder core(model,packet);auto p=restore_joint_baseline_auxiliaries(core,ref,out/"model.json",out/"packet.json");if(core.station!=std::vector<double>({.25,.75})||core.row!=std::vector<double>({111.,222.})||p.preserved!=std::vector<std::uint8_t>({1,1}))throw std::runtime_error("exact aux restore failed");++passed;}
 auto rejected=[&](JointProblemBuilder&core){bool failed=false;try{restore_joint_baseline_auxiliaries(core,ref,out/"model.json",out/"packet.json");}catch(const std::exception&){failed=true;}if(!failed||core.station[0]!=.5||core.station[1]!=.5)throw std::runtime_error("fail-closed atomic restore violated");++passed;};
 {JointProblemBuilder core(model,packet);core.packet.source_hash="different_g5_generation";rejected(core);}
 {JointProblemBuilder core(model,packet);core.model.blocks[0].values[0]=2.;rejected(core);}
 {JointProblemBuilder core(model,packet);core.packet.rows[1].active=false;rejected(core);}
 {JointProblemBuilder core(model,packet);core.packet.rows[1].source_station_intervals={{{0.,.5}}};core.packet.rows[1].station_domain_path=0;rejected(core);}
 {JointProblemBuilder core(model,packet);std::ofstream(out/"station.f64",std::ios::binary|std::ios::trunc)<<"corrupt";rejected(core);}
 eventio::atomic_json(out/"receipt.json",{{"schema","rinkfit.native_baseline_auxiliary_targeted_tests.v1"},{"passed",true},{"cases",passed},{"test_only_not_source_qualification",true}});std::cout<<"passed "<<passed<<" baseline auxiliary native cases"<<std::endl;return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
