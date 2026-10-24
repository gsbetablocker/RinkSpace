"""Native source-role, exact identity and thin-domain boundary checks."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from rinkfit_joint import native_print_append_v3 as append

ROOT=Path(__file__).resolve().parents[3]
NATIVE_SOURCE=r'''
#include "rinkfit/joint_representative_eligibility_v2.hpp"
#include <iostream>
struct Row{std::string shot_id,source_row_id;std::int64_t pts;int material=0,path=0;std::array<double,2>uv{};};
struct Rows{
 std::size_t count=0;std::vector<Row>plain;std::vector<std::int64_t>pts;std::vector<std::uint16_t>shots;std::vector<std::string>shot_ids;std::map<std::size_t,Row>special;
 std::size_t size()const{return count;}
 Row operator[](std::size_t i)const{if(!plain.empty())return plain.at(i);Row r;const auto p=special.find(i);if(p!=special.end())r=p->second;r.pts=pts.at(i);r.shot_id=shot_ids.at(shots.at(i));return r;}
};
Row row(const rinkfit::eventio::Json&j){return {j.at("shot_id"),j.at("source_row_id"),j.at("native_pts"),j.at("material_index"),j.at("path"),j.at("uv").get<std::array<double,2>>()};}
int main(int argc,char**argv){using namespace rinkfit;try{
 if(argc!=2)throw std::runtime_error("request path required");const auto q=eventio::read_json(argv[1]);const auto m=eventio::read_json(q.at("manifest").get<std::string>());Rows rows;rows.shot_ids=q.at("shot_ids").get<std::vector<std::string>>();
 if(q.contains("source_packet")){eventio::verify(q.at("source_packet").at("file").get<std::string>(),q.at("source_packet").at("sha256"));if(m.at("source_packet").at("sha256")!=q.at("source_packet").at("sha256"))throw std::runtime_error("test request actual packet mismatch");}
 if(q.contains("native_columns")){const auto&c=q.at("native_columns");rows.pts=eventio::read_array<std::int64_t>({},c.at("native_pts"),"<i8");rows.shots=eventio::read_array<std::uint16_t>({},c.at("shot_index"),"<u2");rows.count=rows.pts.size();if(rows.shots.size()!=rows.count)throw std::runtime_error("native clock row denominator");for(const auto&v:q.at("rows"))rows.special.emplace(v.at("raw_row").get<std::size_t>(),row(v));}
 else{for(const auto&v:q.at("rows"))rows.plain.push_back(row(v));rows.count=rows.plain.size();}
 std::vector<unsigned char>physical=q.contains("physical_ref")?eventio::read_array<unsigned char>({},q.at("physical_ref"),"|u1"):q.at("physical").get<std::vector<unsigned char>>();
 const auto eligibility=JointRepresentativeEligibilityV2::load(m,m.at("source_packet").at("sha256"),physical);eligibility.validate_source_rows(rows,rows.shot_ids);
 std::vector<unsigned char>keep(physical.size());if(q.contains("keep"))keep=q.at("keep").get<std::vector<unsigned char>>();else{for(const auto i:eventio::read_array<std::uint64_t>({},m.at("fit_rows"),"<u8")){if(i>=keep.size()||keep[i])throw std::runtime_error("duplicate native fit row");keep[i]=1;}}
 auto groups=q.value("groups",eventio::Json::array());if(q.contains("groups_ref")){const auto&r=q.at("groups_ref");eventio::verify(r.at("file").get<std::string>(),r.at("sha256"));groups=eventio::read_json(r.at("file").get<std::string>()).at("groups");}
 for(const auto&g:groups)if(g.at("rank").get<std::size_t>())eligibility.validate_correlated_group(g.at("rows").get<std::vector<std::size_t>>(),physical,keep);
 eligibility.validate_inspection(m,physical,keep);std::cout<<eventio::Json({{"passed",true},{"physical",eligibility.physical_count},{"training",eligibility.training_count},{"fit",m.at("fit_count")},{"inspection",m.at("inspection_count")},{"groups",groups.size()}}).dump()<<std::endl;return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 2;}}
'''

class NativeEligibilityTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cpp=ROOT/'calibration/rinkfit/cpp';spec=importlib.util.spec_from_file_location('eligibility_native_environment',cpp/'gpu_rows/build.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  cls.output=Path('D:/CodexData/RinkSpace_Data/tests/training_eligibility_v2');cls.output.mkdir(parents=True,exist_ok=True)
  source=cls.output/'check.cpp';source.write_text(NATIVE_SOURCE,encoding='utf8')
  vc=Path('D:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat');result=subprocess.run('cmd.exe /d /s /c '+chr(34)+'call '+chr(34)+str(vc)+chr(34)+' >nul && set'+chr(34),capture_output=True,text=True,check=True)
  env={k.upper():v for k,v in os.environ.items()};env.update({k.upper():v for k,v in(x.split('=',1)for x in result.stdout.splitlines()if'='in x and not x.startswith('='))});compiler=shutil.which('cl.exe',path=env['PATH'])
  if not compiler:raise RuntimeError('Existing MSVC unavailable for native eligibility boundary check')
  cls.executable=cls.output/'check.exe';command=[compiler,'/nologo','/std:c++17','/EHsc','/O1','/MD','/utf-8','/I'+str(cpp/'include'),'/I'+str(module.D/'nlohmann_json/Library/include'),str(source),'/Fo:'+str(cls.output/'check.obj'),'/Fe:'+str(cls.executable)]
  result=subprocess.run(command,cwd=cls.output,env=env,capture_output=True,text=True,errors='replace');(cls.output/'build.log').write_text(result.stdout+result.stderr,encoding='utf8')
  if result.returncode:raise RuntimeError('Native eligibility compile failed: '+result.stdout+result.stderr)
 def fixture(self,root):
  def raw(name,data,dtype=None,count=None):
   p=root/name;p.write_bytes(data);r=dict(file=str(p),sha256=hashlib.sha256(data).hexdigest())
   if dtype:r.update(dtype=dtype,shape=[count])
   return r
  def doc(name,data):return raw(name,json.dumps(data).encode())
  roles={str(i*256):dict(shot_id='s01'if i<2 else's02',role='validation_guard'if i==2 else'development_candidate')for i in range(6)}
  role_ref=doc('roles.json',dict(schema='rinkfit.joint_source_frame_protection.v1',source_video_sha256='a'*64,frame_roles=roles))
  rows=[dict(raw_row=i,source_row_id='original/'+str(i),native_pts=i*256,shot_id='s01'if i<2 else's02',shot_index=0 if i<2 else 1,material_index=10+i,path=0,uv=[float(i),20.])for i in range(6)]
  verdict=copy.deepcopy(rows[3]);verdict.update(decision='rejected',uses_fitted_model_projection=False)
  proof=raw('identity.txt',b'independent source identity fixture')
  verdict_ref=doc('verdict.json',dict(schema='rinkfit.source_identity_verdicts.v1',source_packet_sha256='packet-current',scope='listed_row_only',rows=[verdict],evidence_files=[proof],all_source_identity_proven=False))
  original_fit=raw('original.u64',struct.pack('<4Q',0,1,2,3),'<u8',4)
  original=doc('original.json',dict(schema='rinkfit.per_frame_representative_source_selection.v1',source_packet=dict(sha256='packet-current'),source_row_count=6,fit_rows=original_fit,fit_count=4))
  training=raw('training.u8',bytes([1,1,0,0,0,1]),'|u1',6)
  fit=raw('fit.u64',struct.pack('<2Q',0,1),'<u8',2);inspection=raw('inspection.u64',struct.pack('<3Q',2,3,5),'<u8',3)
  m=dict(schema='rinkfit.per_frame_representative_source_selection.v2',source_packet=dict(sha256='packet-current'),source_row_count=6,physical_admitted_rows=5,eligible_rows=3,fit_count=2,fit_cap=4,fit_rows=fit,inspection_rows=inspection,inspection_count=3,training_eligibility=dict(schema='rinkfit.source_training_eligibility_mask.v1',source_packet_sha256='packet-current',row_count=6,eligible_count=3,mask=training),original_selection=original,original_fit_rows=original_fit,eligibility_provenance=dict(frame_roles=role_ref,source_identity_verdicts=verdict_ref,rejected_source_rows=[3]),method=dict(no_backfill=True,selection_uses_fit_residual=False,all_physically_admitted_nonfit_rows_remain_inspection=True))
  request=dict(rows=rows,shot_ids=['s01','s02'],physical=[1,1,1,1,0,1],keep=[1,1,0,0,0,0],groups=[dict(rank=4,rows=[0,1])])
  return m,request
 def execute(self,root,m,request):
  mf=root/'manifest.json';mf.write_text(json.dumps(m));request['manifest']=str(mf);rq=root/'request.json';rq.write_text(json.dumps(request));return subprocess.run([str(self.executable),str(rq)],capture_output=True,text=True)
 def rewrite(self,ref,document):
  p=Path(ref['file']);p.write_text(json.dumps(document));ref['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
 def mask(self,m,values):
  ref=m['training_eligibility']['mask'];p=Path(ref['file']);p.write_bytes(bytes(values));ref['sha256']=hashlib.sha256(p.read_bytes()).hexdigest();m['eligible_rows']=m['training_eligibility']['eligible_count']=sum(values)
 def test_current_roles_identity_groups_and_inspection_pass(self):
  with tempfile.TemporaryDirectory(dir=self.output)as tmp:
   root=Path(tmp);m,r=self.fixture(root);result=self.execute(root,m,r);self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(json.loads(result.stdout)['inspection'],3)
 def test_protected_renamed_unknown_and_cross_shot_roles_fail(self):
  for change in ('protected','renamed','missing','cross_shot'):
   with self.subTest(change=change),tempfile.TemporaryDirectory(dir=self.output)as tmp:
    root=Path(tmp);m,r=self.fixture(root);ref=m['eligibility_provenance']['frame_roles'];d=json.loads(Path(ref['file']).read_text())
    if change=='protected':d['frame_roles']['0']['role']='reserved_pending_lineage'
    if change=='renamed':d['frame_roles']['0']['role']='native_material_repair_training'
    if change=='missing':del d['frame_roles']['0']
    if change=='cross_shot':d['frame_roles']['0']['shot_id']='s02'
    self.rewrite(ref,d);self.assertNotEqual(self.execute(root,m,r).returncode,0)
 def test_mask_cannot_resurrect_current_physical_withdrawal_or_bad_identity(self):
  for index in (2,3,4):
   with self.subTest(index=index),tempfile.TemporaryDirectory(dir=self.output)as tmp:
    root=Path(tmp);m,r=self.fixture(root);values=[1,1,0,0,0,1];values[index]=1;self.mask(m,values);self.assertNotEqual(self.execute(root,m,r).returncode,0)
 def test_bad_verdict_ID_UV_material_path_shot_and_SHA_fail(self):
  for field in ('source_row_id','uv','material_index','path','shot_index','sha256'):
   with self.subTest(field=field),tempfile.TemporaryDirectory(dir=self.output)as tmp:
    root=Path(tmp);m,r=self.fixture(root);ref=m['eligibility_provenance']['source_identity_verdicts'];d=json.loads(Path(ref['file']).read_text())
    if field=='sha256':ref['sha256']='0'*64
    else:
     d['rows'][0][field]={'source_row_id':'different','uv':[3.,21.],'material_index':1,'path':1,'shot_index':0}[field];self.rewrite(ref,d)
    self.assertNotEqual(self.execute(root,m,r).returncode,0)
 def test_covariance_partial_training_inspection_omission_and_backfill_fail(self):
  for case in ('group','inspection','backfill'):
   with self.subTest(case=case),tempfile.TemporaryDirectory(dir=self.output)as tmp:
    root=Path(tmp);m,r=self.fixture(root)
    if case=='group':self.mask(m,[1,0,0,0,0,1]);r['keep'][1]=0;m['fit_count']=1
    elif case=='inspection':m['inspection_count']=2;m['inspection_rows']['shape']=[2];p=Path(m['inspection_rows']['file']);p.write_bytes(struct.pack('<2Q',2,5));m['inspection_rows']['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    else:r['keep'][5]=1;m['fit_count']=3
    self.assertNotEqual(self.execute(root,m,r).returncode,0)

class PrintIssuanceTests(unittest.TestCase):
 def test_original_role_cannot_be_renamed_or_cross_cut(self):
  roles=SimpleNamespace(roles={256:dict(shot_id='s01',role='validation_guard')})
  self.assertFalse(append.frame_training_role(roles,256,'s01')[1])
  with self.assertRaises(ValueError):append.frame_training_role(roles,256,'s02')
  roles.roles[256]['role']='native_material_repair_training'
  with self.assertRaises(ValueError):append.frame_training_role(roles,256,'s01')
 def test_failed_seed_status_is_preserved_and_false_success_rejected(self):
  rows=[dict(allowed_paths=[1])]
  with patch.object(append,'_fast_seed_frame',return_value=(np.array([0]),np.array([.5]),np.array([0]),np.array([np.inf]))):
   _,_,status,error=append.seed_readout(None,None,0,rows);self.assertEqual(int(status[0]),0);self.assertTrue(np.isinf(error[0]))
  for values in (([1],[.5],[1],[np.inf]),([0],[.5],[1],[1.])):
   with patch.object(append,'_fast_seed_frame',return_value=tuple(np.asarray(x)for x in values)),self.assertRaises(ValueError):append.seed_readout(None,None,0,rows)

if __name__=='__main__':unittest.main()
