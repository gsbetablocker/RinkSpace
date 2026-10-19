#include "rinkfit/event_factors.hpp"
#include <cassert>
#include <iostream>
using namespace rinkfit;
int main(){
  EventModel model;model.n_controls=8;model.f_ref=1000.;model.radius_max=2.;model.R0_seed={1,0,0,0,0,-1,0,1,0};model.n_seed={0,0,1};model.b_seed={1,0,0};model.n_tangent1={1,0,0};model.n_tangent2={0,1,0};model.blocks.resize(10);
  auto& g=model.blocks[0].values;g.assign(32,0.);g[0]=.5;g[1]=.1;g[2]=.12;g[3]=1.9;g[4]=.02;g[12]=-.3;g[13]=.03;g[20]=959.5;g[21]=539.5;g[31]=.02;
  for(int i=0;i<8;++i)model.blocks[i+1].values={.001*i,.0005*i,std::log(1000.)};
  model.blocks[9].values={-.1,.2,.1,.2};model.materials.resize(1);model.materials[0].paths.resize(1);model.materials[0].paths[0].geometry_block=9;
  auto l=model.layout(0,0,.02);std::vector<double*> params;std::vector<int> sizes;for(int id:l.block_ids){params.push_back(model.blocks[id].values.data());sizes.push_back(int(model.blocks[id].values.size()));}
  double station=.9,rowvalue=600.,budget=3.;EventRowEvaluation row;row.model=&model;row.layout=l;row.station_slot=int(params.size());params.push_back(&station);sizes.push_back(1);row.row_slot=int(params.size());params.push_back(&rowvalue);sizes.push_back(1);
  Projection<double> pred;assert(row.predict(params.data(),&pred));row.source_uv={pred.uv[0]+1.2,pred.uv[1]-.7};
  const int budgetslot=int(params.size());params.push_back(&budget);sizes.push_back(1);auto pixel=event_autodiff_cost(EventPixelConstraint{row,budgetslot},sizes);
  std::vector<std::vector<double>> J;std::vector<double*> jptr;for(int size:sizes){J.emplace_back(size,0.);jptr.push_back(J.back().data());}
  // reserve prevents vector relocation of vector objects from affecting their
  // owned buffers; each allocation itself remains stable.
  double residual=0;assert(pixel->Evaluate(params.data(),&residual,jptr.data()));
  auto check=[&](int block,int col,double h){double* x=params[block]+col;const double save=*x;double plus,minus;*x=save+h;assert(pixel->Evaluate(params.data(),&plus,nullptr));*x=save-h;assert(pixel->Evaluate(params.data(),&minus,nullptr));*x=save;const double fd=(plus-minus)/(2*h),ad=J[block][col];assert(std::abs(fd-ad)<2e-5*(1+std::abs(fd)));};
  check(0,5,1e-7);check(0,11,1e-7);check(0,31,1e-7);check(row.station_slot,0,1e-7);check(row.row_slot,0,1e-4);check(budgetslot,0,1e-6);
  EventQueryCurve query;query.alpha={-2.,2.};query.native_uv={{0,0},{0,0}};query.numerator_x={pred.uv[0],.8,.02};query.numerator_y={pred.uv[1],.6,-.01};query.denominator={1.};query.provenance="curved rational query derivative fixture";
  double alpha=.2;row.genuine_profile=true;row.source_normal={.8,.6};row.query=std::make_shared<const EventQueryCurve>(query);row.alpha_slot=int(params.size());params.push_back(&alpha);sizes.push_back(1);
  auto qcost=event_autodiff_cost(EventQueryEquality{row,0},sizes);std::vector<std::vector<double>> qJ;std::vector<double*> qp;for(int size:sizes){qJ.emplace_back(size,0.);qp.push_back(qJ.back().data());}assert(qcost->Evaluate(params.data(),&residual,qp.data()));assert(std::abs(qJ.back()[0]+.808)<1e-12);
  EventPhysicalConstraint shape;shape.id="nonzero_short";shape.type=0;shape.scale=.5;shape.block_ids={0};auto phys=event_autodiff_cost(EventPhysicalInequality{&model,shape},{32});double* gp[]={g.data()};double jac[32]{};double* gj[]={jac};assert(phys->Evaluate(gp,&residual,gj));assert(std::abs(jac[0]+2)<1e-12&&std::abs(jac[2]-4)<1e-12);
  EventALProjection<double> raw;assert(row.predict_for_al(params.data(),&raw));assert(raw.physical_valid());
  assert(std::abs(raw.uv[0]-pred.uv[0])<1e-10&&std::abs(raw.uv[1]-pred.uv[1])<1e-10);
  const double original_radius=model.radius_max;model.radius_max=.5*std::sqrt(raw.ray_radius_sq);
  assert(!row.predict(params.data(),&pred));assert(row.predict_for_al(params.data(),&raw)&&!raw.physical_valid());
  assert(raw.physical_status==DomainStatus::kOutsideDistortionDomain);
  auto ray=event_autodiff_cost(EventRowDomainInequality{row,0,1.},sizes);assert(ray->Evaluate(params.data(),&residual,nullptr));assert(std::abs(residual-3.)<1e-12);
  // A trial outside R remains visible to every AL obligation. It never
  // becomes a physical projection and R itself is never enlarged.
  assert(pixel->Evaluate(params.data(),&residual,nullptr));assert(qcost->Evaluate(params.data(),&residual,nullptr));
  auto rs=event_autodiff_cost(EventRSEquality{row},sizes);assert(rs->Evaluate(params.data(),&residual,nullptr));
  model.radius_max=original_radius;g[25]=-.1;g[27]=0.;g[29]=.001;g[30]=.002;
  for(int kind=0;kind<3;++kind){
    auto cost=event_autodiff_cost(EventRowDomainInequality{row,kind,1.},sizes);std::vector<std::vector<double>> D;std::vector<double*> dp;for(int size:sizes){D.emplace_back(size,0.);dp.push_back(D.back().data());}
    assert(cost->Evaluate(params.data(),&residual,dp.data()));
    std::vector<std::pair<int,int>> columns={{0,5},{0,11},{0,13},{0,31},{row.station_slot,0},{row.row_slot,0},{0,25},{0,27},{0,29},{0,30}};
    for(auto column:columns){double* x=params[column.first]+column.second;const double saved=*x,step=column.first==row.row_slot?1e-4:1e-7;double plus,minus;*x=saved+step;assert(cost->Evaluate(params.data(),&plus,nullptr));*x=saved-step;assert(cost->Evaluate(params.data(),&minus,nullptr));*x=saved;const double fd=(plus-minus)/(2*step),ad=D[column.first][column.second];assert(std::abs(fd-ad)<2e-5*(1+std::abs(fd)));}
  }
  g[25]=-10.;assert(!row.predict(params.data(),&pred));assert(row.predict_for_al(params.data(),&raw)&&!raw.physical_valid());assert(raw.physical_status==DomainStatus::kDistortionUncertified);
  auto fold=event_autodiff_cost(EventRowDomainInequality{row,2,1.},sizes);assert(fold->Evaluate(params.data(),&residual,nullptr)&&residual>0.);
  // Positive depth is a true perspective arithmetic guard, not a fabricated
  // finite residual at the camera plane.
  const double save_y=g[12];g[12]=.3;assert(!row.predict_for_al(params.data(),&raw));g[12]=save_y;
  std::cout<<"event_factors: fresh AutoDiff/FD for native, curved query, all three domain constraints; raw AL violations and strict physical rejection passed\n";
}
