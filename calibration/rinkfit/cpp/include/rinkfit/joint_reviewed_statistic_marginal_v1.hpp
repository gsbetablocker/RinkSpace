#pragma once
#include <Eigen/SVD>
#include <limits>
#include "rinkfit/joint_packet.hpp"
namespace rinkfit {
// Eliminate only reviewed inactive residual coordinates from the original
// quadratic statistic. This is the exact minimum over the missing coordinates;
// it keeps the original source operator in source_json for provenance.
inline eventio::Json joint_reviewed_statistic_marginal_v1(JointStatisticalGroup& g,const std::vector<JointObservation>& rows){
 std::vector<std::size_t> keep,drop;for(std::size_t j=0;j<g.rows.size();++j)(rows.at(g.rows[j]).active?keep:drop).push_back(j);
 if(drop.empty()||!g.rank)return eventio::Json();
 const auto original_rows=g.rows;const auto original_rank=g.rank;const auto n=g.rows.size();
 Eigen::MatrixXd a(g.rank,2*keep.size()),d(g.rank,2*drop.size());
 for(std::size_t r=0;r<g.rank;++r){for(std::size_t j=0;j<keep.size();++j)for(int c=0;c<2;++c)a(r,2*j+c)=g.W[r*2*n+2*keep[j]+c];for(std::size_t j=0;j<drop.size();++j)for(int c=0;c<2;++c)d(r,2*j+c)=g.W[r*2*n+2*drop[j]+c];}
 Eigen::JacobiSVD<Eigen::MatrixXd> missing(d,Eigen::ComputeFullU);missing.setThreshold(64*std::numeric_limits<double>::epsilon()*std::max(d.rows(),d.cols()));
 const Eigen::Index lost=missing.rank(),remaining=Eigen::Index(original_rank)-lost;
 Eigen::MatrixXd reduced=missing.matrixU().rightCols(remaining).transpose()*a;
 g.rows.clear();for(auto j:keep)g.rows.push_back(original_rows[j]);g.W.clear();g.rank=0;
 if(remaining&&a.cols()){
  Eigen::JacobiSVD<Eigen::MatrixXd> retained(reduced,Eigen::ComputeThinV);retained.setThreshold(64*std::numeric_limits<double>::epsilon()*std::max(reduced.rows(),reduced.cols()));g.rank=retained.rank();g.W.resize(g.rank*2*g.rows.size());
  for(std::size_t r=0;r<g.rank;++r)for(std::size_t c=0;c<2*g.rows.size();++c)g.W[r*2*g.rows.size()+c]=retained.singularValues()[r]*retained.matrixV()(c,r);
 }
 g.statistic_kind=g.rank?"propagated_source_covariance":"constraint_only_unknown_noise";
 std::vector<std::size_t> inactive;for(auto j:drop)inactive.push_back(original_rows[j]);
 return {{"group_id",g.id},{"reviewed_inactive_rows",inactive},{"retained_active_rows",g.rows},{"original_rank",original_rank},{"remaining_rank",g.rank},{"operation","exact_minimum_over_reviewed_inactive_residual_coordinates"},{"original_source_operator_retained",true}};
}
}
