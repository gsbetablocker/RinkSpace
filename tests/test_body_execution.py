"""Execution-policy tests using NumPy only.

The CUDA-mode cases exercise perturbation batching/slicing with a fake world
function. They do not exercise Torch, CUDA kernels, a device, or GPU speed.
"""
import importlib.util
import os
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch

import numpy as np

PATH=Path(__file__).resolve().parents[1]/'calibration/program_tracking/body_tracking/body_execution.py'
SPEC=importlib.util.spec_from_file_location('_test_body_execution',PATH)
EXECUTION=importlib.util.module_from_spec(SPEC);sys.modules[SPEC.name]=EXECUTION;SPEC.loader.exec_module(EXECUTION)
BodyExecution=EXECUTION.BodyExecution
BodyExecutionConfig=EXECUTION.BodyExecutionConfig


class FakeReference:
    def __init__(self):
        self.body=object();self.calls=[]
        self.lower_vertices=np.arange(42,dtype=float).reshape(14,3)
    def world(self,reference,delta):
        n=len(delta);self.calls.append(n)
        weights=np.arange(1.,70.)
        signal=np.sum((delta+.03*delta**2)*weights,axis=1)+reference['times']
        xyz=reference['P']+signal[:,None]*np.array([.3,-.2,.4])
        native=xyz[:,None]+np.arange(22)[None,:,None]*np.array([.1,.2,.3])
        world=xyz[:,None]+np.arange(113)[None,:,None]*np.array([.03,.04,.05])
        frames=np.broadcast_to(reference['Q'][:,None],(n,22,3,3)).copy()
        gear=world[:,:96].reshape(n,2,48,3)
        return dict(P=xyz,Q=reference['Q'],local_R=reference['local_R'],native=native,
            coco=world[:,96:],world=world,frames=frames,gear=gear,
            footwear=dict(gear=gear,upper_deformation=np.repeat(signal[:,None,None],6,axis=1).reshape(n,2,3),
                lower_boot_vertices=self.lower_vertices),constant=17)


def inputs(n=7):
    rng=np.random.default_rng(15)
    reference=dict(P=rng.normal(size=(n,3)),Q=np.broadcast_to(np.eye(3),(n,3,3)).copy(),
        local_R=np.broadcast_to(np.eye(3),(n,21,3,3)).copy(),times=np.arange(n)*.07)
    delta=rng.normal(scale=.01,size=(n,69))
    return reference,delta


def consume(state):
    if state['footwear']['lower_boot_vertices'].shape!=(14,3):
        raise AssertionError('Metadata array was sliced as sample data')
    if state['constant']!=17:raise AssertionError('Metadata changed')
    return np.concatenate((state['P'],state['native'][:,5],state['world'][:,87],
        state['footwear']['upper_deformation'].reshape(-1,6)),axis=1)


class BodyExecutionTests(unittest.TestCase):
    def test_explicit_resident_compatibility_config_keeps_cpu_body_and_validates_environment(self):
        with patch.dict(os.environ, {}, clear=True):
            ordinary=BodyExecution(BodyExecutionConfig.from_environment())
            self.assertEqual(ordinary.record()['compatibility_backend'],'cpu')
            self.assertIsNone(ordinary.prepare_compatibility(None,None,None))
        with patch.dict(os.environ, dict(RINKSPACE_BODY_BACKEND='CPU',
                RINKSPACE_BODY_WORKERS='4',RINKSPACE_BODY_LOCAL_DERIVATIVES='ANALYTIC',
                RINKSPACE_BODY_COMPATIBILITY_BACKEND='CUDA_RESIDENT'), clear=True):
            config=BodyExecutionConfig.from_environment()
            self.assertEqual((config.backend,config.workers,config.local_derivatives,
                              config.compatibility_backend),('cpu',4,'analytic','cuda_resident'))
            # Configuration and reporting must not initialize CUDA on their own.
            execution=BodyExecution(config)
            self.assertIsNone(execution._body_backend)
            self.assertIsNone(execution._compatibility_backend)
            self.assertEqual(execution.record()['compatibility_backend'],'cuda_resident')
        for settings in (dict(compatibility_backend='auto'),
                         dict(compatibility_backend='cuda_resident',local_derivatives='finite_difference'),
                         dict(compatibility_backend='cuda_resident',backend='cuda')):
            with self.subTest(settings=settings),self.assertRaises(ValueError):
                BodyExecutionConfig(**settings)

    def test_all_69_columns_batch_vs_serial_with_partial_and_capped_groups(self):
        reference,delta=inputs();original_delta=delta.copy();original={k:v.copy() for k,v in reference.items()}
        columns=list(range(68,-1,-1))+[6,68]
        with self.subTest(mode='serial'):
            expected=BodyExecution(BodyExecutionConfig()).state_differences(FakeReference(),reference,delta,columns,consume)
        # 71 columns: both 4-column and 2-column groups finish with a partial group.
        for batch,cap,expected_width in ((4,1000,4),(8,28,2),(4,1,1)):
            with self.subTest(batch=batch,cap=cap):
                fake=FakeReference();execution=BodyExecution(BodyExecutionConfig(backend='cuda',workers=1,
                    perturbation_batch=batch,max_batch_samples=cap))
                result=execution.state_differences(fake,reference,delta,columns,consume)
                np.testing.assert_allclose(result,expected,rtol=0,atol=1e-9)
                self.assertEqual(len(fake.calls),(len(columns)+expected_width-1)//expected_width)
                self.assertLessEqual(max(fake.calls),2*len(delta)*expected_width)
        np.testing.assert_array_equal(delta,original_delta)
        for key,value in original.items():np.testing.assert_array_equal(reference[key],value)

    def test_batch_world_derivatives_preserve_column_and_sample_order(self):
        reference,delta=inputs(5);columns=[68,0,16,1,67]
        serial=BodyExecution(BodyExecutionConfig())
        batch=BodyExecution(BodyExecutionConfig(backend='cuda',perturbation_batch=3))
        expected=list(serial.world_derivatives(FakeReference(),reference,delta,columns))
        actual=list(batch.world_derivatives(FakeReference(),reference,delta,columns))
        self.assertEqual([c for c,_ in actual],columns)
        for (_,a),(_,b) in zip(actual,expected):np.testing.assert_array_equal(a,b)

    def test_threaded_columns_equal_serial_and_ordered(self):
        reference,delta=inputs(4);columns=[68,0,15,4,7,14,6]
        serial=BodyExecution(BodyExecutionConfig());parallel=BodyExecution(BodyExecutionConfig(workers=3))
        try:
            expected=serial.state_differences(FakeReference(),reference,delta,columns,consume)
            actual=parallel.state_differences(FakeReference(),reference,delta,columns,consume)
            np.testing.assert_array_equal(actual,expected)
        finally:serial.close();parallel.close()

    def test_worker_failure_is_drained_before_control_returns(self):
        execution=BodyExecution(BodyExecutionConfig(workers=2));second_started=threading.Event();second_done=threading.Event()
        def work(column):
            if column==0:
                if not second_started.wait(1):raise AssertionError('Second worker did not start')
                raise ValueError('expected worker failure')
            second_started.set()
            # The failing caller must wait for this task before returning to
            # any code that could mutate shared body/binding state.
            threading.Event().wait(.03);second_done.set();return column
        with self.assertRaisesRegex(ValueError,'expected worker failure'):
            execution.map_columns(work,[0,1,2,3])
        self.assertTrue(second_done.is_set());self.assertIsNone(execution._pool)
        try:self.assertEqual(execution.map_columns(lambda x:x*x,[3,1,2]),[9,1,4])
        finally:execution.close()

    def test_opening_proxy_attaches_to_body_owner_and_reuses_execution(self):
        class Proxy:
            def __init__(self,parent):self.parent=parent
            def __getattr__(self,key):return getattr(self.parent,key)
        owner=FakeReference();proxy=Proxy(Proxy(owner));execution=BodyExecution(BodyExecutionConfig())
        self.assertIs(execution.attach(proxy),owner)
        self.assertNotIn('_body_execution',vars(proxy))
        self.assertIs(EXECUTION.get_body_execution(proxy),execution)
        self.assertIs(EXECUTION.get_body_execution(owner),execution)
        self.assertIs(execution.attach(proxy),owner)
        with self.assertRaisesRegex(ValueError,'already has an execution owner'):
            BodyExecution(BodyExecutionConfig(workers=2)).attach(proxy)
        execution.close()

    def test_detached_torch_adapter_cannot_be_misreported_as_cpu(self):
        class TorchCompactBody:pass
        owner=FakeReference();owner.body=TorchCompactBody()
        with self.assertRaisesRegex(ValueError,'already a Torch adapter'):
            BodyExecution(BodyExecutionConfig()).attach(owner)

    def test_owner_cycles_and_invalid_inputs_are_explicit(self):
        class Wrapper:pass
        a=Wrapper();b=Wrapper();a.parent=b;b.parent=a
        with self.assertRaisesRegex(TypeError,'state owner'):BodyExecution().attach(a)
        reference,delta=inputs()
        for columns in ([-1],[69]):
            with self.assertRaisesRegex(ValueError,'outside'):BodyExecution().state_differences(FakeReference(),reference,delta,columns,consume)
        with self.assertRaisesRegex(ValueError,'positive'):
            BodyExecution().state_differences(FakeReference(),reference,delta,[0],consume,step=0)
        with self.assertRaisesRegex(ValueError,'one body CUDA owner'):
            BodyExecutionConfig(backend='cuda',workers=2)


if __name__=='__main__':unittest.main()
