from collections import OrderedDict
from types import SimpleNamespace
import numpy as np
from scipy.sparse import csr_matrix
from calibration.program_tracking.first_shot_source_directional_QA import check_complete_source_direction


def test_full_correlated_source_detects_omitted_world_chain_and_restores_caches():
    from calibration.solver_runtime import install
    install()
    from z190_streaming_source_normal import SourceRowBlock
    W=np.array([[1.,.4],[-.2,.8]])
    cache=OrderedDict([(b'actual-current',object())]);motion=OrderedDict([(b'motion-current',object())]);oldworld=cache.copy();oldmotion=motion.copy();seen=[]
    class Owner:
        fixed=np.array([2]);scales=np.ones(2);programme_count=3
        programme=SimpleNamespace(bundle=SimpleNamespace(_world_cache=cache),state=SimpleNamespace(scene=lambda q:seen.append(q.copy())))
        def __init__(self):self._motion_cache=motion
        def effective_q_and_chain(self,q):return q,csr_matrix(np.eye(3))
        def guard(self,q):
            if q[0]<.3 or q[2]!=0.:raise ValueError('original active physical inequality')
            return True
        def source_blocks(self,q,jacobian):
            cache.clear();cache[q.tobytes()]=object();motion.clear();motion[q.tobytes()]=object()
            raw=np.array([q[0]+q[1]**2,np.sin(q[1])+q[0]*q[1]])
            J=np.array([[1.,2*q[1],0.],[q[1],np.cos(q[1])+q[0],0.]])
            yield SourceRowBlock(W@raw,csr_matrix(W@J)if jacobian else None,np.zeros(2,bool),np.ones(2),dict(family='same-complete-correlated-W'))
    owner=Owner();q=np.array([.3,.2,0.]);d=np.array([.5,.7,0.]);blocks=list(owner.source_blocks(q,True));cache.clear();cache.update(oldworld);motion.clear();motion.update(oldmotion)
    report=check_complete_source_direction(owner,q,d,blocks,epsilon=1e-5)
    assert report['passed']and report['method']=='second_order_one_sided_original_full_domain'
    assert list(cache.items())==list(oldworld.items())and list(motion.items())==list(oldmotion.items())
    np.testing.assert_array_equal(seen[-1],q)
    broken=blocks[0].J.copy().tolil();broken[:,1]=0.
    invalid=[SourceRowBlock(blocks[0].r,broken.tocsr(),blocks[0].quadratic,blocks[0].scales,blocks[0].row_ledger)]
    failure=check_complete_source_direction(owner,q,d,invalid,epsilon=1e-5)
    assert not failure['passed']and failure['max_absolute']>.1


def test_checked_factory_requires_both_full_source_stencils_and_rejects_bad_J():
    import pytest
    from calibration.solver_runtime import install
    install()
    from z190_streaming_source_normal import SourceRowBlock
    from calibration.program_tracking.first_shot_source_directional_QA import checked_source_factory
    class Owner:
        fixed=np.array([],int);scales=np.ones(2);programme_count=2
        programme=SimpleNamespace(bundle=SimpleNamespace(),state=SimpleNamespace(scene=lambda q:None))
        def guard(self,q):return True
        def effective_q_and_chain(self,q):return q,csr_matrix(np.eye(2))
        def source_blocks(self,q,jacobian):
            yield SourceRowBlock(np.array([q[0]**2,q[0]+np.sin(q[1])]),
                csr_matrix([[2*q[0],0.],[1.,np.cos(q[1])]])if jacobian else None,
                np.zeros(2,bool),self.scales,dict(family='all-original-rows'))
    owner=Owner();q=np.array([.3,.2]);d=np.array([.7,.4]);reports=[]
    source=checked_source_factory(owner,owner.source_blocks,direction=d,epsilon=2e-5,
        confirmation_epsilon=1e-5,write_report=reports.append)
    assert len(list(source(q,True)))==1
    assert reports[0]['passed']and reports[0]['both_original_domain_stencils_required']
    assert [r['epsilon']for r in reports[0]['stencils']]==[2e-5,1e-5]
    def broken(q,jacobian):
        for b in owner.source_blocks(q,jacobian):
            yield SourceRowBlock(b.r,csr_matrix((2,2))if jacobian else None,b.quadratic,b.scales,b.row_ledger)
    source=checked_source_factory(owner,broken,direction=d,epsilon=2e-5,
        confirmation_epsilon=1e-5,write_report=reports.append)
    with pytest.raises(ValueError,match='Complete current formal-source'):list(source(q,True))
    assert not reports[-1]['passed']
