#pragma once
#include "rinkfit/joint_model.hpp"
#include <ceres/ceres.h>
#include <Eigen/Dense>
namespace rinkfit {
class JointLinearControlManifold final:public ceres::Manifold {
 Eigen::MatrixXd N_;
 public:explicit JointLinearControlManifold(const eventio::Json&d){int n=d.at("ambient_size"),k=d.at("tangent_size");N_.resize(n,k);if(int(d.at("basis").size())!=n)throw std::runtime_error("linear control basis row shape");for(int i=0;i<n;++i){if(int(d.at("basis")[i].size())!=k)throw std::runtime_error("linear control basis column shape");for(int j=0;j<k;++j)N_(i,j)=d.at("basis")[i][j];}if((N_.transpose()*N_-Eigen::MatrixXd::Identity(k,k)).cwiseAbs().maxCoeff()>1e-10)throw std::runtime_error("linear control tangent not orthonormal");}
 int AmbientSize()const override{return int(N_.rows());}int TangentSize()const override{return int(N_.cols());}
 bool Plus(const double*x,const double*d,double*y)const override{Eigen::Map<Eigen::VectorXd>out(y,N_.rows());out=Eigen::Map<const Eigen::VectorXd>(x,N_.rows())+N_*Eigen::Map<const Eigen::VectorXd>(d,N_.cols());return out.allFinite();}
 bool PlusJacobian(const double*,double*j)const override{for(int a=0;a<N_.rows();++a)for(int b=0;b<N_.cols();++b)j[a*N_.cols()+b]=N_(a,b);return true;}
 bool Minus(const double*y,const double*x,double*d)const override{Eigen::Map<Eigen::VectorXd>out(d,N_.cols());out=N_.transpose()*(Eigen::Map<const Eigen::VectorXd>(y,N_.rows())-Eigen::Map<const Eigen::VectorXd>(x,N_.rows()));return out.allFinite();}
 bool MinusJacobian(const double*,double*j)const override{for(int a=0;a<N_.rows();++a)for(int b=0;b<N_.cols();++b)j[b*N_.rows()+a]=N_(a,b);return true;}
};
inline void apply_joint_control_manifolds(ceres::Problem&p,JointEventModel&m){for(const auto&d:m.control_manifolds()){auto id=d.at("block").get<std::size_t>();auto&b=m.blocks.at(id);if(int(b.values.size())!=d.at("ambient_size").get<int>())throw std::runtime_error("linear control ambient shape");p.SetManifold(b.values.data(),new JointLinearControlManifold(d));}}
inline double joint_control_membership_error(const JointEventModel&m){double maxe=0;for(const auto&d:m.control_manifolds()){const auto&x=m.blocks.at(d.at("block").get<std::size_t>()).values;for(const auto&row:d.at("constraint_matrix")){if(row.size()!=x.size())throw std::runtime_error("local control constraint shape");double r=0;for(std::size_t j=0;j<x.size();++j)r+=row[j].get<double>()*x[j];maxe=std::max(maxe,std::abs(r));}}return maxe;}
}
