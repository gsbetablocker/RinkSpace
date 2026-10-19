from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import pickle
import numpy as np
import pytest
from calibration.solver_runtime import install
install()
from calibration.program_tracking.first_shot_source_kernel import _interfaces
from calibration.program_tracking.reference_board_pair import BoardIntersectionError as PublicDomain
from calibration.program_tracking.ordered_contour_transport import source_chunk

REASON='ordered_print_arc_no_directed_monotone_path'


def _raise_ordered(task):
    error=_interfaces()[3](REASON,[1,4]);error.add_note('qualified source arc, trial must be rejected')
    raise error


def _raise_public(task):raise PublicDomain('ice_curve_no_central_segment',[2])


def _return_worker(task):return task,dict(status='unchanged')


def _raise_other(task):
    error=ValueError('unrelated');error.reason=REASON;error.indices=[7];raise error


def test_actual_domain_aliases_are_unpickleable_but_all_reasons_transport_safely():
    for cls in set((_interfaces()[3],PublicDomain)):
        with pytest.raises(TypeError):pickle.loads(pickle.dumps(cls(REASON,[1])))
    for producer,reason,indices in [(_raise_ordered,REASON,[1,4]),(_raise_public,'ice_curve_no_central_segment',[2])]:
        with pytest.raises(ValueError)as caught:source_chunk(None,original_sampler=producer)
        value=pickle.loads(pickle.dumps(caught.value))
        assert type(value)is ValueError
        assert value.reason==reason and value.worker_domain_rejection
        assert value.indices.tolist()==indices
    assert value.original_exception_type=='BoardIntersectionError'
    with pytest.raises(ValueError,match='unrelated')as caught:source_chunk(None,original_sampler=_raise_other)
    assert not hasattr(caught.value,'worker_domain_rejection')


def test_actual_spawn_rejects_trial_without_breaking_source_pool():
    with ProcessPoolExecutor(1,mp_context=multiprocessing.get_context('spawn'))as pool:
        for producer,reason in [(_raise_ordered,REASON),(_raise_public,'ice_curve_no_central_segment')]:
            with pytest.raises(ValueError,match=reason)as caught:
                pool.submit(source_chunk,None,original_sampler=producer).result(timeout=30)
            assert caught.value.reason==reason
        result=pool.submit(source_chunk,7,original_sampler=_return_worker).result(timeout=30)
        assert result[0]==7 and result[-1]['status']=='unchanged'
        assert result[-1]['board_domain_error_transport']['all_reasons']
