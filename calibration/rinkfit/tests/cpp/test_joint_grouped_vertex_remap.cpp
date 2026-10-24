#include "rinkfit/joint_factors.hpp"
#include <iostream>
using namespace rinkfit;
int main(int argc,char**argv){try{
 if(argc!=4)throw std::runtime_error("model rows result required");auto model=JointEventModel::load(argv[1]);const auto source=eventio::read_json(argv[2]);
 std::array<JointObservation,2> observations;std::array<JointRowEvaluation,2> rows;std::array<double,2> station;
 std::array<std::vector<const double*>,2> local;std::vector<const double*> joined;std::vector<int> sizes;std::array<std::vector<int>,2> maps;
 for(int i=0;i<2;++i){auto&o=observations[i];const auto&s=source.at(i);o.shot_id=s.at("shot_id");o.pts=s.at("native_pts");o.material=s.at("material");o.path=s.at("path");o.kind=s.at("kind");o.uv=s.at("uv").get<Vec2<double>>();o.station=s.at("station");o.row=s.at("row");o.alpha=s.at("alpha");o.active=true;station[i]=o.station;
  auto&q=rows[i];q.model=&model;q.observation=&o;q.layout=model.layout(model.shot_map.at(o.shot_id),o.material,o.path,o.pts,-1,0,o.station);if(q.layout.shape_segment<0)throw std::runtime_error("actual split vertex path required");
  local[i]=model.pointers(q.layout);for(std::size_t k=0;k<q.layout.block_ids.size();++k)q.parameter_sizes[int(k)]=int(model.blocks.at(q.layout.block_ids[k]).values.size());q.station_slot=int(local[i].size());q.parameter_sizes[q.station_slot]=1;local[i].push_back(&station[i]);
  for(std::size_t k=0;k<local[i].size();++k){auto p=std::find(joined.begin(),joined.end(),local[i][k]);int slot=int(p-joined.begin());if(p==joined.end()){joined.push_back(local[i][k]);sizes.push_back(q.parameter_sizes.at(int(k)));}maps[i].push_back(slot);}
 }
 JointStatisticalGroup g;g.rows={0,1};g.rank=3;g.W={1.,.2,-.3,.1,0.,.8,.4,-.2,.1,0.,.7,1.};JointGroupedStatistic group;group.group=&g;
 for(int i=0;i<2;++i){auto q=rows[i].remap(maps[i]);for(int k=0;k<2;++k)if(q.layout.shape_vertex_slots[k]!=maps[i].at(rows[i].layout.shape_vertex_slots[k]))throw std::runtime_error("shared vertex not remapped");group.rows.push_back(std::move(q));}
 double max_residual=0,max_derivative=0;int derivatives=0;
 for(int slot=0;slot<int(joined.size());++slot)for(int component=0;component<sizes[slot];++component){using J=ceres::Jet<double,1>;std::vector<std::vector<J>>data;std::vector<const J*>p;for(std::size_t k=0;k<joined.size();++k){data.emplace_back();for(int j=0;j<sizes[k];++j){J x(joined[k][j]);if(int(k)==slot&&j==component)x.v[0]=1;data.back().push_back(x);}p.push_back(data.back().data());}
  std::array<J,4> error;for(int i=0;i<2;++i){std::vector<const J*>one;for(int k:maps[i])one.push_back(p[k]);Vec2<J>x;if(!rows[i].native_error(one.data(),&x))throw std::runtime_error("actual standalone formal row failed");error[2*i]=x[0];error[2*i+1]=x[1];}
  J actual[3];if(!group(p.data(),actual))throw std::runtime_error("actual merged formal residual failed");for(int k=0;k<3;++k){J expected(0);for(int j=0;j<4;++j)expected+=J(g.W[4*k+j])*error[j];max_residual=std::max(max_residual,std::abs(expected.a-actual[k].a));max_derivative=std::max(max_derivative,std::abs(expected.v[0]-actual[k].v[0]));}++derivatives;
 }
 if(max_residual>1e-10||max_derivative>1e-9)throw std::runtime_error("merged/standalone formal residual or derivative differs");eventio::atomic_json(argv[3],{{"passed",true},{"source_rows",source},{"model_sha256",eventio::file_sha256(argv[1])},{"actual_RS_residual_max_difference",max_residual},{"actual_RS_derivative_max_difference",max_derivative},{"scalar_derivative_directions",derivatives},{"second_row_original_vertex_slots",rows[1].layout.shape_vertex_slots},{"second_row_merged_vertex_slots",group.rows[1].layout.shape_vertex_slots},{"quality_accepted",false}});std::cout<<"formal merged shared-vertex residual/derivatives match, "<<derivatives<<" scalars"<<std::endl;return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
