"""Pure image-limit checks and a tiny real TRF trial rejection smoke."""
import ast
import io
import json
from pathlib import Path
from types import SimpleNamespace
import time
import unittest

import numpy as np
from scipy.optimize import least_squares

from calibration.program_tracking.body_tracking.skate_run_policy import image_trial_quality


def production_guarded(problem, monitor, initial):
    """Execute the actual trial wrapper without loading body assets or solving them."""
    path=Path(__file__).resolve().parents[1]/'calibration/program_tracking/body_tracking/skate_integration.py'
    tree=ast.parse(path.read_text(encoding='utf8'))
    joint=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='JointContactProblem')
    solve=next(node for node in joint.body if isinstance(node,ast.FunctionDef) and node.name=='solve')
    guarded=next(node for node in solve.body if isinstance(node,ast.FunctionDef) and node.name=='guarded')
    factory=ast.parse('def factory():\n    jacobian_seconds=0.\n    last_linearization=None\n    return guarded\n').body[0]
    factory.body.insert(-1,guarded)
    namespace=dict(self=problem,monitor=monitor,initial_value=initial,np=np,time=time,json=json,
                   expand=lambda x:x,record_before_jacobian=False,StopFit=RuntimeError,
                   image_trial_quality=image_trial_quality,
                   __package__='calibration.program_tracking.body_tracking')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[factory],type_ignores=[])),str(path),'exec'),namespace)
    return namespace['factory']()


class ImageSafeTrialTests(unittest.TestCase):
    def test_relative_guard_has_exact_original_boundary(self):
        self.assertEqual(image_trial_quality(100.,105.,.05)['status'],'allowed')
        self.assertTrue(image_trial_quality(100.,105.000001,.05)['rejected'])
        self.assertEqual(image_trial_quality(100.,0.,.05)['status'],'allowed')
        self.assertEqual(image_trial_quality(100.,100.,0.)['status'],'allowed')

    def test_missing_zero_or_nonfinite_baseline_is_unavailable(self):
        for baseline in (None,0.,-1.,np.nan,np.inf,True):
            with self.subTest(baseline=baseline):
                result=image_trial_quality(baseline,100.,.05)
                self.assertEqual(result['status'],'unavailable')
                self.assertFalse(result['rejected'])
        for current,limit in ((None,.05),(np.inf,.05),(100.,None),(100.,-1.)):
            self.assertEqual(image_trial_quality(100.,current,limit)['status'],'unavailable')

    def test_actual_guarded_contracts_trf_trials_and_preserves_objective_and_cache(self):
        J=np.array([[1.],[10.]])
        def residual(x):return np.array([1.+x[0],10.*(x[0]-3.)])
        initial=dict(residual=residual([0.]),image_cost=.5)
        baseline=dict(image_cost=.5)
        monitor=SimpleNamespace(baseline=baseline,policy=SimpleNamespace(max_image_cost_increase=.05),
                                reserve_evaluation=lambda **kwargs:dict(stop=False))
        problem=SimpleNamespace(image_safe_trials=False,accepted=0,start=time.perf_counter(),log=io.StringIO())
        def evaluate(x,jacobian=False):
            problem.cache_x=np.array(x,copy=True)
            problem.cache=dict(residual=residual(x),image_cost=.5*(1.+x[0])**2,
                               jac=J.copy() if jacobian else None)
            return problem.cache
        problem.evaluate=evaluate
        guarded=production_guarded(problem,monitor,initial)
        np.testing.assert_array_equal(guarded(np.array([1.]))['residual'],residual([1.]))
        problem.image_safe_trials=True
        self.assertTrue(np.isinf(guarded(np.array([1.]))['residual']).all())
        np.testing.assert_array_equal(problem.cache['residual'],residual([1.]))
        np.testing.assert_array_equal(guarded(np.array([1.]),True)['jac'],J)
        accepted=[]
        def callback(x):
            accepted.append(x.copy())
            if len(accepted)==4:raise StopIteration
        fit=least_squares(lambda x:guarded(x)['residual'],[0.],jac=lambda x:evaluate(x,True)['jac'],
                          method='trf',x_scale='jac',callback=callback,max_nfev=80)
        self.assertGreater(len(accepted),0)
        self.assertGreater(fit.x[0],0.)
        for x in accepted:self.assertLessEqual(.5*(1.+x[0])**2,.5*1.05)
        np.testing.assert_allclose(fit.fun,residual(fit.x),rtol=0,atol=0)
        np.testing.assert_array_equal(fit.jac,J)
        events=[json.loads(line) for line in problem.log.getvalue().splitlines()]
        self.assertGreater(len(events),1)
        self.assertTrue(all(event['event']=='trial_image_quality_rejected' for event in events))
        self.assertTrue(all(event['image_cost']>event['maximum_image_cost'] for event in events))
        self.assertEqual(problem.accepted,0)
        self.assertEqual(baseline,{'image_cost':.5})
