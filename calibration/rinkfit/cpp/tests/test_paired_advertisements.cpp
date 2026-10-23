#include "rinkfit/joint_model.hpp"
#include "rinkfit/joint_manifold.hpp"
#include <iostream>
#include <set>
using namespace rinkfit;
Vec3<double> world(JointEventModel& m,int mi,int pi,double s){auto l=m.layout(0,mi,pi,m.shots[0].first_pts);auto ps=m.pointers(l);auto g=m.global(l,ps.data());std::vector<const double*>p;std::vector<double>geometry;auto e=m.legacy_layout(l,p,g,ps.data(),geometry);return m.world_point(l,s,p.data(),e,m.shots[0].math);}
int main(int argc,char**argv){if(argc!=4)return 2;auto old=JointEventModel::load(argv[1]);auto m=JointEventModel::load(argv[2]);auto patch=eventio::read_json(argv[3]);double worst=0;std::size_t points=0;std::set<int> primary={1,2,3,4,5,6,7,8,10};
 for(std::size_t mi=0;mi<old.materials.size();++mi){const auto& mat=old.materials[mi];if(mat.label.rfind("Ad",0)!=0)continue;int number=std::stoi(mat.label.substr(2));if(!primary.count(number))continue;for(std::size_t pi=0;pi<mat.paths.size();++pi)for(double s:{0.,.137,.5,.883,1.}){auto a=world(old,int(mi),int(pi),s),b=world(m,int(mi),int(pi),s);for(int k=0;k<3;++k)worst=std::max(worst,std::abs(a[k]-b[k]));++points;}}
 if(worst>2e-12)throw std::runtime_error("first-instance world geometry changed while reencoding");double membership=0;
 for(const auto& pair:patch.at("pairs")){const auto&d=pair.at("manifold");JointLinearControlManifold N(d);auto x=m.blocks.at(pair.at("block_index").get<int>()).values;std::vector<double>delta(N.TangentSize(),1e-5),y(8);if(!N.Plus(x.data(),delta.data(),y.data()))return 3;for(const auto&row:d.at("constraint_matrix")){double e=0;for(int i=0;i<8;++i)e+=row.at(i).get<double>()*y[i];membership=std::max(membership,std::abs(e));}if(membership>1e-12)throw std::runtime_error("shared print dimensions separated");}
 std::cout<<eventio::Json({{"primary_world_samples",points},{"primary_world_max_abs_delta",worst},{"shared_width_height_tangent_error",membership},{"pair_count",patch.at("pairs").size()}}).dump()<<std::endl;
}
