#pragma once
#include "rinkfit/joint_factors.hpp"
namespace rinkfit {
class JointProblemBuilder {
 public:
 JointEventModel model;JointPacket packet;std::vector<double>station,row,alpha;std::set<std::string>fixed_nuisance_shots;bool observed_row_time_new_sources=false;explicit JointProblemBuilder(JointEventModel m,JointPacket p):model(std::move(m)),packet(std::move(p)){for(const auto&r:packet.rows){station.push_back(r.station);row.push_back(r.row);alpha.push_back(r.alpha);}}
 JointRowEvaluation row_evaluation(std::size_t i)const;
 eventio::Json replay(const std::filesystem::path& out,bool write_arrays=true)const;
 eventio::Json solve(int iterations,int threads=8,const std::string& mode="full_W_initialization");
};
}
