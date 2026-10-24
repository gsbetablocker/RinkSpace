import ast
import collections
import unittest
from pathlib import Path
import numpy as np
from rinkfit_joint.finite_ring_source import unique_source_interface_candidate

ROOT=Path(__file__).resolve().parents[3]

class InterfaceIdentityTest(unittest.TestCase):
 def test_unique_keeps_original_measurement(self):
  point={'uv':[17.,23.],'source_id':'immutable_source'}
  selected,verdict=unique_source_interface_candidate([point])
  self.assertIs(selected,point)
  self.assertEqual(verdict['actual_interface_candidates'],1)

 def test_competing_and_missing_stay_unknown(self):
  points=[{'uv':[17.,23.]},{'uv':[17.,30.]}]
  for candidates in ([],points):
   selected,verdict=unique_source_interface_candidate(candidates)
   self.assertIsNone(selected)
   self.assertEqual(verdict['source_identity_decision'],'unknown')
   self.assertFalse(verdict['qualification_issued'])
  self.assertEqual(points,[{'uv':[17.,23.]},{'uv':[17.,30.]}])

 def test_actual_event_entry_never_resolves_two_interfaces_by_shift(self):
  file=ROOT/'events/ankara_2026/programmes/okada_mei_sp/source_datasets/dense56_20261020/ice/reader56_exact.py'
  tree=ast.parse(file.read_text('utf8'))
  fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=='coherent_branch')
  scope={'np':np,'collections':collections,
         'unique_source_interface_candidate':unique_source_interface_candidate}
  exec(compile(ast.Module(body=[fn],type_ignores=[]),str(file),'exec'),scope)
  options=[dict(query=i,shift=0.,uv=np.array([float(i),20.]),p={'operator':'material_interface'})for i in range(8)]
  ambiguous=dict(query=3,shift=1.,uv=np.array([3.,24.]),p={'operator':'material_interface'})
  failures=[]
  selected,_=scope['coherent_branch'](options+[ambiguous],None,None,'F2:middle',1306368,failures)
  self.assertEqual([o['query']for o in selected],[0,1,2,4,5,6,7])
  self.assertEqual(failures[0]['candidate_uv'],[[3.,20.],[3.,24.]])
  self.assertFalse(failures[0]['qualification_issued'])
  self.assertEqual(failures[0]['actual_interface_candidates'],2)

if __name__=='__main__':unittest.main()
