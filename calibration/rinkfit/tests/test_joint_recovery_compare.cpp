#include <numeric>
#include <functional>
#define main comparison_fullscope_main
#include "../tools/joint_recovery_compare.cpp"
#undef main
// The fixture contains only the 239 actual g5 appended sources. No full-scope
// graph, fit or source qualification is performed by this targeted test.
int main(int argc,char**argv){try{
 using namespace rinkfit;if(argc!=4)throw std::runtime_error("MODEL TINY_PACKET OUT required");
 auto model=JointEventModel::load(argv[1]);auto packet=JointPacket::load(argv[2],model);if(packet.rows.size()!=239)throw std::runtime_error("targeted actual g5 fixture changed");
 JointProblemBuilder b(std::move(model),std::move(packet));const auto original=b.packet.rows;
 std::vector<std::vector<double>>physical;for(const auto&block:b.model.blocks)physical.push_back(block.values);
 std::vector<std::size_t>rows(b.packet.rows.size());std::iota(rows.begin(),rows.end(),0);std::vector<std::uint8_t>status;const auto report=initialize_fresh_sources(b,rows,&status);
 if(report.at("fixed_entity_rows")!=197||report.at("finite_sliding_rows")!=42||status.size()!=239)throw std::runtime_error("actual kind denominator changed");
 for(std::size_t i=0;i<rows.size();++i){const auto&a=original[i];const auto&r=b.packet.rows[i];if(a.uv!=r.uv||a.pts!=r.pts||a.source_row_id!=r.source_row_id||a.material!=r.material||a.kind!=r.kind||a.path!=r.path||a.active!=r.active||a.query!=r.query||a.normal!=r.normal||a.E!=r.E)throw std::runtime_error("fresh seed changed physical source or operator");if(r.kind==1&&a.station!=r.station)throw std::runtime_error("fixed entity became sliding");if(status[i]!=1&&status[i]!=2)throw std::runtime_error("active failed source lost");}
 for(std::size_t i=0;i<physical.size();++i)if(physical[i]!=b.model.blocks[i].values)throw std::runtime_error("fresh seed changed historical physical model");
 const std::filesystem::path out=argv[3];std::filesystem::create_directories(out);auto replay=b.replay(out/"tiny_independent_RS_state",true);
 if(replay.at("active_rows")!=239||replay.at("valid_rows").get<std::size_t>()+replay.at("invalid_rows").get<std::size_t>()!=239)throw std::runtime_error("independent replay discarded a source");
 int rejects=0;auto reject=[&](const std::function<void(JointProblemBuilder&)>&change){auto m=JointEventModel::load(argv[1]);auto p=JointPacket::load(argv[2],m);JointProblemBuilder x(std::move(m),std::move(p));change(x);bool threw=false;try{std::vector<std::uint8_t>s;initialize_fresh_sources(x,{0},&s);}catch(...){threw=true;}if(!threw)throw std::runtime_error("fresh source identity guard did not reject");++rejects;};
 reject([](auto&x){auto&r=x.packet.rows[0];x.model.materials[r.material].paths.push_back(x.model.materials[r.material].paths[0]);});
 reject([](auto&x){x.packet.rows[0].kind=0;});reject([](auto&x){x.packet.rows[0].active=false;});
 eventio::atomic_json(out/"receipt.json",{{"schema","rinkfit.historical_comparison_targeted_native_test.v1"},{"passed",true},{"actual_g5_fresh_rows",239},{"fixed_entity_rows",197},{"finite_ice_rows",42},{"identity_guards_rejected_cases",rejects},{"historical_physical_values_unchanged",true},{"fresh_initialization",report},{"independent_replay",replay},{"full_scope_comparison_started",false},{"fit_started",false},{"source_qualification_granted",false}});std::cout<<"actual 239 g5 source fresh initializer and independent replay passed; 3 identity guards rejected"<<std::endl;return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
