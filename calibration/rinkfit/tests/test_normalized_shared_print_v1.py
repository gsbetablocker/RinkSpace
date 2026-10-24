"""Shared normalized print geometry, affine-gauge, and physical domain regressions."""
from pathlib import Path
import os
import shutil
import subprocess
import unittest
import numpy as np
from rinkfit_start.native_scene import NativeScene

ROOT=Path(__file__).resolve().parents[3]
SOURCE=r'''
#include "rinkfit/joint_normalized_print_domain_v1.hpp"
#include "rinkfit/joint_manifold.hpp"
#include "rinkfit/joint_explicit.hpp"
#include <ceres/jet.h>
#include <iostream>
using namespace rinkfit;
void need(bool yes,const char*why){if(!yes)throw std::runtime_error(why);}
EventParameterBlock block(std::string name,std::vector<double>x){EventParameterBlock b;b.name=name;b.values=x;b.scales.assign(x.size(),1);b.lower.assign(x.size(),-100);b.upper.assign(x.size(),100);return b;}
int main(){try{
 JointEventModel m;m.blocks={block("scene",{.5,.1,.1,2.,.02}),block("print_template/shape",{-99,-99,-.1,.1,.6,.8,1.2,.2}),block("ad/pair",{.05,.2,.002,.018,.75,.4,.001,.015})};m.scene_block=0;
 EventPath p;p.kind=6;p.geometry_block=1;p.advertisement_block=2;p.advertisement_offset=4;p.closed=true;p.id="shape";p.variant_id="shape";EventMaterial mat;mat.label="Ad1.print.shape";mat.paths={p,p};mat.paths[1].advertisement_offset=0;m.materials={mat};
 m.path_geometry_offsets[{0,0}]=2;m.path_geometry_offsets[{0,1}]=2;m.path_geometry_indices[{0,0}]={};m.path_geometry_indices[{0,1}]={};
 std::array<double,32>g{};for(int i=0;i<5;++i)g[i]=m.blocks[0].values[i];const double*pars[]={g.data(),m.blocks[1].values.data(),m.blocks[2].values.data()};
 for(int branch:{0,1})for(bool closed:{false,true})for(bool indexed:{false,true}){m.materials[0].paths[branch].closed=closed;JointRowLayout l;l.material=0;l.path=branch;l.geometry_slot=1;l.advertisement_slot=2;l.geometry_offset=2;if(indexed)l.geometry_indices={6,7,2,3,4,5};
  std::vector<const double*>local;std::vector<double>geometry;const auto e=m.legacy_layout(l,local,g,pars,geometry);const double*selected=local[e.geometry_slot];const double*a=m.blocks[2].values.data()+m.materials[0].paths[branch].advertisement_offset;
  for(double s:{0.,.11,.48,.99,1.}){const int edges=closed?3:2;const double z=s*edges;const int k=std::min(edges-1,int(std::floor(z))),next=(k+1)%3;const double u=z-k,x=selected[2*k]*(1-u)+selected[2*next]*u,y=selected[2*k+1]*(1-u)+selected[2*next+1]*u;const auto expected=EventModel::board_point(a[0]+a[1]*x,a[2]+(a[3]-a[2])*y,g.data());const auto actual=m.world_point(l,s,local.data(),e,EventModel{});for(int j=0;j<3;++j)need(std::abs(actual[j]-expected[j])<1e-12,"normalized shared shape changed continuous board geometry");}
 }
 auto specs=joint_normalized_print_domain_specs_v1(m);need(specs.size()==6,"normalized shape domains must deduplicate by actual installation");const auto&shape=m.blocks[1].values;const auto&ad=m.blocks[2].values;need(joint_normalized_print_height_envelope_v1(g.data(),shape.data(),ad.data(),std::vector<int>{3,5,7},4,false)<0,"valid lower board envelope rejected outside guessed unit rectangle");need(joint_normalized_print_height_envelope_v1(g.data(),shape.data(),ad.data(),std::vector<int>{3,5,7},4,true)<0,"valid upper board envelope rejected");auto bad=shape;bad[5]=3;need(joint_normalized_print_height_envelope_v1(g.data(),bad.data(),ad.data(),std::vector<int>{3,5,7},4,true)>0,"out of board vertex escaped exact envelope");need(joint_normalized_print_course_span_v1(g.data(),shape.data(),ad.data(),std::vector<int>{2,4,6},4)<0,"valid finite course span rejected");bad[6]=20;need(joint_normalized_print_course_span_v1(g.data(),bad.data(),ad.data(),std::vector<int>{2,4,6},4)>0,"whole-perimeter print span escaped");
 using J=ceres::Jet<double,1>;J gj[32],qj[8],aj[8];for(int i=0;i<32;++i)gj[i]=J(g[i]);for(int i=0;i<8;++i){qj[i]=J(shape[i]);aj[i]=J(ad[i]);}qj[5].v[0]=1;const auto jet=joint_normalized_print_height_envelope_v1(gj,qj,aj,std::vector<int>{3,5,7},4,true);need(std::abs(jet.v[0]-(ad[7]-ad[6]))<1e-14,"domain AD lost normalized installation chain");
 JointEventModel split;split.scene_block=0;split.blocks={block("scene",{.5,.1,.1,2.,.02}),block("vertex0",{-.1,.1}),block("vertex1",{.6,.8}),block("vertex2",{1.2,.2}),block("ad/pair",{.05,.2,.002,.018,.75,.4,.001,.015}),block("numeric_course_envelope",{-.1,1.2})};EventPath sp;sp.kind=6;sp.vertex_blocks={1,2,3};sp.geometry_block=-1;sp.advertisement_block=4;sp.advertisement_offset=4;sp.course_envelope_block=5;sp.closed=true;EventMaterial sm;sm.label="Ad1.print.split";sm.paths={sp};split.materials={sm};split.path_geometry_offsets[{0,0}]=0;split.path_geometry_indices[{0,0}]={};
 JointRowLayout sl;sl.material=0;sl.path=0;sl.shape_vertex_count=3;sl.shape_segment=0;sl.shape_vertex_slots={1,2};sl.advertisement_slot=3;const double*spars[]={g.data(),split.blocks[1].values.data(),split.blocks[2].values.data(),split.blocks[4].values.data()};std::vector<const double*>slocal;std::vector<double>sgeometry;const auto se=split.legacy_layout(sl,slocal,g,spars,sgeometry);const auto sw=split.world_point(sl,1./6,slocal.data(),se,EventModel{}),sr=split.split_print_world_at(0,0,1./6,g.data());for(int j=0;j<3;++j)need(std::abs(sw[j]-sr[j])<1e-12,"split fixed-cell residual and dynamic full-curve reader differ");bool escaped=false;try{split.world_point(sl,.8,slocal.data(),se,EventModel{});}catch(const std::domain_error&){escaped=true;}need(escaped,"split residual silently crossed undeclared vertex parameters");const auto distant=split.split_print_world_at(0,0,.8,g.data());need(finite_array(distant),"read-only full curve wrongly retained segment0");sl.shape_segment=1;sl.shape_vertex_slots={1,2};const double*leftpars[]={g.data(),split.blocks[2].values.data(),split.blocks[3].values.data(),split.blocks[4].values.data()};slocal.clear();const auto le=split.legacy_layout(sl,slocal,g,leftpars,sgeometry);const auto boundary=split.world_point(sl,2./3,slocal.data(),le,EventModel{});const auto exact=split.split_print_world_at(0,0,2./3,g.data());for(int j=0;j<3;++j)need(std::abs(boundary[j]-exact[j])<1e-12,"upper cell boundary discarded one-sided segment");auto domains=joint_normalized_print_domain_specs_v1(split);need(domains.size()==13,"split static vertex domain factor graph is not linear in vertex count");need(joint_normalized_print_envelope_bound_v1(split.blocks[1].values.data(),split.blocks[5].values.data(),0)==0,"numeric lower envelope changed vertex truth");need(joint_normalized_print_envelope_span_v1(g.data(),split.blocks[5].values.data(),split.blocks[4].values.data(),4)<0,"numeric envelope does not preserve finite physical span");
 JointEventModel gauge;gauge.blocks={block("print_template/gauge",{2,-3,10})};const double v=1/std::sqrt(2.);eventio::Json d={{"block",0},{"ambient_size",3},{"tangent_size",1},{"basis",{{v},{v},{0.}}},{"constraint_matrix",eventio::Json::array()},{"affine_origin",{2.,-3.,10.}},{"affine_subspace_role","source_supported_static_shape_with_initialization_nullspace_gauge_v1"}};gauge.source_json["linear_control_manifolds"]=eventio::Json::array({d});need(joint_control_membership_error(gauge)<1e-14,"nonzero static template baseline rejected as homogeneous constraint");gauge.blocks[0].values[0]+=v;gauge.blocks[0].values[1]+=v;need(joint_control_membership_error(gauge)<1e-14,"supported static shape direction rejected");gauge.blocks[0].values[2]+=.1;need(joint_control_membership_error(gauge)>.09,"initialization nullspace gauge drift undetected");
 JointExplicitRow remapped;remapped.layout.shape_vertex_slots={{1,2}};remapped.layout.scene_slot=0;remapped.layout.block_ids={0,1,2};const auto mapped=remapped.remap({2,0,1});need(mapped.layout.shape_vertex_slots==std::array<int,2>{{0,1}}&&mapped.layout.scene_slot==2,"W union lost static vertex parameter slot remap");
 std::cout<<"normalized paired kind6 CPU geometry, exact board domain/AD, affine baseline gauge passed"<<std::endl;return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 2;}}
'''

class NormalizedSharedPrintTests(unittest.TestCase):
 def test_python_normalized_closed_and_offset(self):
  g=[.5,.1,.1,2.,.02]+[0.]*27
  shape=[-99.,-99.,-.1,.1,.6,.8,1.2,.2]
  ad=[.05,.2,.002,.018,.75,.4,.001,.015]
  for closed in (False,True):
   for indices in (None,[6,7,2,3,4,5]):
    path=dict(kind=6,geometry_block=1,geometry_offset=2,advertisement_block=2,advertisement_offset=4,closed=closed)
    if indices:path['geometry_indices']=indices
    model=dict(blocks=[dict(name='global',values=g),dict(name='shape',values=shape),dict(name='ad',values=ad)],materials=[dict(label='Ad1.print.shape',paths=[path])])
    scene=NativeScene(model);s=np.array([0.,.11,.48,.99,1.]);q=np.array(shape)[indices]if indices else np.array(shape)[2:];q=q.reshape(-1,2);edges=3 if closed else 2;z=s*edges;k=np.minimum(np.floor(z).astype(int),edges-1);u=z-k;uv=q[k]*(1-u[:,None])+q[(k+1)%3]*u[:,None];expected=scene.board(ad[4]+ad[5]*uv[:,0],ad[6]+(ad[7]-ad[6])*uv[:,1]);np.testing.assert_allclose(scene.world(0,0,s),expected,atol=1e-12,rtol=0)
    path['advertisement_block']=-1
    np.testing.assert_allclose(NativeScene(model).world(0,0,s),scene.board(uv[:,0],uv[:,1]),atol=1e-12,rtol=0)

 def test_split_vertices_replay_all_cells_and_physical_envelope(self):
  g=[.5,.1,.1,2.,.02]+[0.]*27
  blocks=[dict(name='global',values=g),dict(name='v0',values=[-.1,.1]),dict(name='v1',values=[.6,.8]),dict(name='v2',values=[1.2,.2]),dict(name='ad/pair',values=[.75,.4,.001,.015]),dict(name='numeric_envelope',values=[-.1,1.2])]
  path=dict(kind=6,geometry_block=-1,vertex_blocks=[1,2,3],course_envelope_block=5,advertisement_block=4,closed=True)
  model=dict(blocks=blocks,materials=[dict(label='Ad1.print.split',paths=[path])]);scene=NativeScene(model);s=np.linspace(0,1,71);points=np.array([x['values']for x in blocks[1:4]]);z=s*3;i=np.minimum(np.floor(z).astype(int),2);u=z-i;q=points[i]*(1-u[:,None])+points[(i+1)%3]*u[:,None];expected=scene.board(.75+.4*q[:,0],.001+.014*q[:,1]);np.testing.assert_allclose(scene.world(0,0,s),expected,atol=1e-12,rtol=0);self.assertTrue(scene.check_physics()['passed']);blocks[2]['values'][1]=3.;self.assertFalse(NativeScene(model).check_physics()['passed'])

 def test_kind2_direct_replay_preserves_second_installation_offset(self):
  blocks=[dict(name='global',values=[.5,.1,.1,2.,.02]+[0.]*27),dict(name='ad/pair',values=[.05,.2,.002,.018,.75,.4,.001,.015])]
  p=dict(kind=2,geometry_block=-1,advertisement_block=1,advertisement_offset=4,points=[[.1,.2],[.7,.8]])
  model=dict(blocks=blocks,materials=[dict(label='Ad23.print.shape',paths=[p])]);scene=NativeScene(model);s=np.linspace(0,1,31);u=.1+.6*s;v=.2+.6*s;np.testing.assert_allclose(scene.world(0,0,s),scene.board(.75+.4*u,.001+.014*v),atol=1e-12,rtol=0);self.assertTrue(scene.check_physics()['passed'])
  blocks[1]['values'][7]=.03;self.assertFalse(NativeScene(model).check_physics()['passed'])

 def test_display_initializer_is_explicitly_unverified_and_has_no_shape_domain(self):
  model=dict(blocks=[dict(name='global',values=[.5,.1,.1,2.,.02]+[0.]*27),dict(name='ad/x',values=[.05,.2,.002,.018])],materials=[dict(label='Ad1.print.unsupported',paths=[dict(kind=2,geometry_block=-1,advertisement_block=1,display_only_initializer=True,points=[[.1,3.],[.2,4.]])])]);r=NativeScene(model).check_physics();self.assertTrue(r['passed']);self.assertEqual(len(r['unverified_display_only_paths']),1);self.assertFalse(r['unverified_display_shapes_are_accepted'])

 def test_native_geometry_domain_and_affine_gauge(self):
  out=Path('D:/CodexData/RinkSpace_Data/tests/normalized_shared_print_v1');out.mkdir(parents=True,exist_ok=True);source=out/'check.cpp';source.write_text(SOURCE,encoding='utf8')
  vc=Path('D:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat');r=subprocess.run('cmd.exe /d /s /c '+chr(34)+'call '+chr(34)+str(vc)+chr(34)+' >nul && set'+chr(34),capture_output=True,text=True,check=True)
  env={k.upper():v for k,v in os.environ.items()};env.update({k.upper():v for k,v in(x.split('=',1)for x in r.stdout.splitlines()if'='in x and not x.startswith('='))});compiler=shutil.which('cl.exe',path=env['PATH']);self.assertIsNotNone(compiler)
  exe=out/'check.exe';command=[compiler,'/nologo','/std:c++17','/EHsc','/O1','/MD','/utf-8','/bigobj','/DGLOG_USE_GLOG_EXPORT','/DGLOG_USE_GFLAGS','/DGLOG_NO_ABBREVIATED_SEVERITIES','/I'+str(ROOT/'calibration/rinkfit/cpp/include'),'/I'+str(ROOT/'environments/cg22p/include')]
  for folder in ('nlohmann_json','eigen','glog','gflags'):command+=['/ID:/CodexData/RinkSpace_Data/runtime_tools/rf_deps/'+folder+'/Library/include'+('/eigen3'if folder=='eigen'else'')]
  command += [str(source),'/Fo:'+str(out/'check.obj'),'/Fe:'+str(exe)]
  r=subprocess.run(command,cwd=out,env=env,capture_output=True,text=True,errors='replace');(out/'build.log').write_text(r.stdout+r.stderr,encoding='utf8');self.assertEqual(r.returncode,0,r.stdout+r.stderr)
  r=subprocess.run([str(exe)],cwd=out,capture_output=True,text=True,errors='replace');(out/'test.log').write_text(r.stdout+r.stderr,encoding='utf8');self.assertEqual(r.returncode,0,r.stdout+r.stderr)

if __name__=='__main__':unittest.main()
