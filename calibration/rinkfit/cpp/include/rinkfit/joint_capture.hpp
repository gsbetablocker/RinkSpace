#pragma once
#include "rinkfit/joint_builder.hpp"
namespace rinkfit {eventio::Json recapture_joint_initial_nuisance(JointProblemBuilder&,const std::filesystem::path&);eventio::Json initialize_joint_query_witnesses(JointProblemBuilder&,const std::vector<std::size_t>&,const std::filesystem::path&,int threads);}
