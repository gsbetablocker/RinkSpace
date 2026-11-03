"""Numerical contact-kernel regressions; no models, native DLL or fit needed.

Loads the actual pure geometry/constraint functions from the source AST so
these tests do not import the unrelated full-body/native-camera pipeline.
This verifies contact assembly and ContactAL, not FK or the complete solve.
"""
import ast
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest

import numpy as np


def load_kernel(directory):
    directory=Path(directory)
    contact=ast.parse((directory/'skate_contact.py').read_text(encoding='utf8'))
    weight=next(n for n in contact.body if isinstance(n,ast.FunctionDef) and n.name=='trapezoid_time_weights')
    namespace={'np':np}
    exec(compile(ast.Module(body=[weight],type_ignores=[]),str(directory/'skate_contact.py'),'exec'),namespace)
    source=ast.parse((directory/'skate_physics.py').read_text(encoding='utf8'))
    pure=[n for n in source.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
    exec(compile(ast.Module(body=pure,type_ignores=[]),str(directory/'skate_physics.py'),'exec'),namespace)
    name='_rinkspace_contact_al_weight_test'
    spec=importlib.util.spec_from_file_location(name,directory/'contact_feasibility.py')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return types.SimpleNamespace(**namespace),module.ContactAL


class RigidPointBinding:
    """One exact rigid support vertex per foot, sufficient for row semantics."""
    ratio=1.0

    def packet(self):
        return {'template':{
            'contact_arcs':[],
            'contact_point_groups':[{'id':'runner_point','points':np.zeros((1,3)),
                                     'semantic':'runner','support_eligible':True}],
            'support_feature_ids':{'glide':['runner_point'],'pick':['runner_point']}}}


def evaluate(kernel,times,heights,evidence):
    times=np.asarray(times,float)
    R=np.broadcast_to(np.eye(3),(len(times),2,3,3)).copy()
    O=np.zeros((len(times),2,3));O[:,:,2]=np.asarray(heights)
    return kernel.constraints(RigidPointBinding(),{'R':R,'O':O},times,0.,evidence)


def support_evidence(**extra):
    return dict(id='source_support_256',pts=256,feet=[0],mode='glide',**extra)


class SourceSupportWeightTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fallback=Path(__file__).resolve().parents[1]/'calibration/program_tracking/body_tracking'
        cls.kernel,cls.al=load_kernel(os.environ.get('RINKSPACE_CONTACT_TEST_ROOT',fallback))
        cls.coarse=np.array([0.,.02,.04])
        cls.dense=np.arange(17)*.0025

    def support_rows(self,result):
        return np.array([i for i,label in enumerate(result['labels']) if 'source_support_' in label[1]])

    def test_source_penalty_and_jacobian_do_not_change_with_collision_grid(self):
        packets=[evaluate(self.kernel,t,[.012,.018],[support_evidence()]) for t in (self.coarse,self.dense)]
        signatures=[]
        for result in packets:
            chosen=self.support_rows(result)
            al=self.al.zeros(len(result['g']),penalty=10.)
            J=np.ones((len(result['g']),1))
            residual,jac=al.residual_and_jacobian(result['g'],J,row_weights=result['weights'])
            signatures.append((result['margin_m'][chosen],result['weights'][chosen],residual[chosen],jac[chosen]))
        for a,b in zip(*signatures):np.testing.assert_allclose(a,b,rtol=0,atol=1e-14)
        np.testing.assert_allclose(signatures[0][1],[.02,.02])

    def test_constant_nonpenetration_integral_stays_grid_invariant(self):
        contributions=[]
        for times in (self.coarse,self.dense):
            result=evaluate(self.kernel,times,[-.010,-.010],[])
            residual,_=self.al.zeros(len(result['g']),penalty=10.).residual_and_jacobian(
                result['g'],row_weights=result['weights'])
            contributions.append(float(residual@residual))
        np.testing.assert_allclose(contributions,[2.048,2.048],rtol=0,atol=1e-13)

    def test_two_independent_support_groups_at_one_pts_are_both_kept(self):
        left=support_evidence();right=dict(left,id='source_support_256_right',feet=[1])
        result=evaluate(self.kernel,self.coarse,[.012,.018],[left,right])
        chosen=self.support_rows(result)
        self.assertEqual(len(chosen),4)
        self.assertEqual(len(set(result['labels'])),len(result['labels']))
        by_label={label[1]:result['margin_m'][i] for i,label in enumerate(result['labels'])}
        self.assertAlmostEqual(by_label['source_support_256:upper'],-.009)
        self.assertAlmostEqual(by_label['source_support_256_right:upper'],-.015)
        self.assertEqual(len(result['support_groups']),2)

    def test_unknown_and_airborne_add_no_support_band(self):
        for mode in ('unknown','airborne'):
            row=dict(support_evidence(),mode=mode)
            result=evaluate(self.kernel,self.coarse,[.012,.018],[row])
            self.assertEqual(len(result['labels']),2*len(self.coarse))
            self.assertTrue(np.isnan(result['support_height']).all())

    def test_explicit_weight_is_positive_and_is_not_confidence(self):
        row=support_evidence(source_support_weight_s=.04,confidence=.01)
        result=evaluate(self.kernel,self.dense,[.012,.018],[row])
        np.testing.assert_allclose(result['weights'][self.support_rows(result)],[.04,.04])
        for value in (0.,-1.,np.nan,np.inf,True):
            with self.assertRaises(ValueError):
                evaluate(self.kernel,self.coarse,[.012,.018],[support_evidence(source_support_weight_s=value)])


if __name__=='__main__':unittest.main()
