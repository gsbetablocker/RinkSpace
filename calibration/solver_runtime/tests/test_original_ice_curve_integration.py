from pathlib import Path
from types import SimpleNamespace
import copy,json,pickle,sys
import numpy as np
import pytest
from scipy.sparse import csr_matrix
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from calibration.solver_runtime import install
install()
from calibration.solver_runtime.original_ice_curve_integration import (
    SCHEMA,SCAN_DECLARATIONS,wrap_original_ice_curve_integration,original_working_sigma)
from calibration.solver_runtime.original_board_curve_integration import file_reference
from calibration.solver_runtime.source_loss import SourceLossWeights,weighted_source_cost,source_irls_weights
from z190_streaming_source_normal import SourceRowBlock,accumulate_source_normal,replay_source_cost

def fixture(tmp_path,duplicate=False):
    def row(name,uv,identity,sigma,text=None):
        r=dict(id=name,pts=10,split='fit',origin='same_original_reader',uv=np.asarray(uv,float),
            sigma=np.broadcast_to(sigma,(len(uv),)).copy(),world=np.array([[0.,0.,0.],[1.,0.,0.]]),_identity=identity)
        if text is not None:r['sigma_modelling_decision']=text
        return r
    rows=[row('upper',[[1,2],[3,2]],['fixed_circle','R2','outer'],2*np.sqrt(4),SCAN_DECLARATIONS[0]+' original producer'),
          row('lower',[[1,4],[3,4]],['fixed_circle','R2','outer'],2*np.sqrt(4),SCAN_DECLARATIONS[0]+' original producer'),
          row('extent',[[5,3]],['fixed_circle','R2','outer'],2.),
          row('inner',[[2,2],[3,3]],['fixed_circle','R2','inner'],3.),
          row('B2',[[1,1],[2,1]]+([[1,1],[2,1]] if duplicate else []),['line','B2'],3.)]
    other=row('outer_other_exposure',[[10,10],[20,10]],['fixed_circle','R2','outer'],4.);other['pts']=20
    groups=[(10,rows),(20,[other])]
    entries=[dict(block='main7',group_order=gi,row_order=ri,group_actual_pts=actual,collection_role='FIT',source_row=copy.deepcopy(r))
             for gi,(actual,rr)in enumerate(groups) for ri,r in enumerate(rr)]
    parent=tmp_path/'original.pkl';parent.write_bytes(pickle.dumps(dict(programme_absolute_rows=entries)))
    policy=tmp_path/'policy.json';policy.write_text(json.dumps(dict(schema=SCHEMA,curve_budget=1.,enable_point_compression=False,
        support_basis='isolated_authenticated_source_anchors',source_video_sha256='a'*64,original_packet=file_reference(parent),
        original_packet_row_indices=list(range(len(entries))))),encoding='utf8')
    sig=np.r_[1.,1.,np.concatenate([np.repeat(e['source_row']['sigma'],2)for e in entries]),1.]
    # Duplicated observations repeat exactly their original R/J, including an
    # outlier beyond Huber's working-space threshold.
    errors=np.array([.2,10.,-1.,-9.,5.,.3,-10.,3.,20.,-1.,4.,-5.,6.,8.])
    native=[];J=[]
    for e in entries:
        for uv in e['source_row']['uv']:
            v=np.array([uv[0]*3.-8.,uv[1]*2.-9.]);native.extend(v);J.extend([[uv[0]+.1,uv[1]*.2],[uv[1]+.2,-uv[0]*.3]])
    r=np.r_[.3,-3.,np.asarray(native)/sig[2:-1],.4]
    matrix=csr_matrix(np.vstack(([1.,2.],[3.,4.],np.asarray(J)/sig[2:-1,None],[.5,.6])))
    oldw=np.ones(len(sig));oldw[:2]=[.2,.8];old=SourceLossWeights(oldw)
    block=SourceRowBlock(r,matrix,np.zeros(len(sig),bool),sig,dict(family='programme:main7',source_sha256='a'*64),old)
    base=SimpleNamespace(count=2,scales=sig,quadratic=block.quadratic,contract={},blocks=lambda q,jacobian=True:iter((block,)))
    actor=SimpleNamespace(name='main7',groups=groups,quadratic=block.quadratic,bundle=SimpleNamespace(motion_rows=2,source_residual_rows=len(sig)-1))
    geo=SimpleNamespace(_binding=lambda r:('original',dict(kind='circle' if r['_identity'][0]=='fixed_circle' else r['_identity'][0],
        contract=dict(physical_identity=r['_identity']))))
    state=SimpleNamespace(blocks=[actor],geometry=geo,original_absolute_input_compaction_loss_weights={'programme:main7':old})
    return base,state,policy,block

def test_actual_SourceRowBlock_Huber_cost_gradient_and_H_preserve_original_layout(tmp_path):
    base,state,policy,old=fixture(tmp_path);owner=wrap_original_ice_curve_integration(base,state,policy)
    new=next(owner.blocks(np.zeros(2)))
    assert new.r is old.r and new.J is old.J and new.scales is old.scales and new.quadratic is old.quadratic
    assert wrap_original_ice_curve_integration(base,state,None)is base
    assert new.loss_weights.integration_weights[:2].tolist()==[.2,.8]
    w=new.loss_weights.integration_weights;c=new.loss_weights.working_normalization_multiplier
    assert np.allclose(c[2:10],np.sqrt(4)) and np.all(c[10:]==1.)
    assert np.isclose(w[2:12].sum()/2,1.) # upper/lower/extent one outer curve budget
    assert np.isclose(w[12:16].sum()/2,1.) # inner is a separate layer
    normal=accumulate_source_normal(lambda:iter((new,)),2,expected_row_count=len(new.r),frozen_profile_token='actual-frozen')
    wr=new.r*c;wj=new.J.toarray()*c[:,None];robust=np.minimum(1.,2./np.maximum(abs(wr),1e-12))
    assert np.any(abs(wr)>2.) and np.any(abs(wr)<2.)
    assert normal.cost==weighted_source_cost(new.r,new.quadratic,new.loss_weights)
    assert np.allclose(normal.gradient,wj.T@((w*robust)*wr),atol=1e-13)
    assert np.allclose(normal.normal.toarray(),wj.T@((w*robust)[:,None]*wj),atol=1e-13)
    h=1e-6;direction=np.array([.3,-.2]);jv=new.J@direction
    fd=(weighted_source_cost(new.r+h*jv,new.quadratic,new.loss_weights)-weighted_source_cost(new.r-h*jv,new.quadratic,new.loss_weights))/(2*h)
    assert abs(fd-normal.gradient@direction)<2e-8
    with pytest.raises(ValueError):replay_source_cost(lambda:iter((old,)),normal.contract,frozen_profile_token='actual-frozen')
    with pytest.raises(ValueError,match='already applied'):owner.apply_block(new)

def test_exact_duplicate_density_does_not_add_curve_budget_or_Huber_information(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b';a.mkdir();b.mkdir()
    values=[]
    for path,duplicate in ((a,False),(b,True)):
        source,state,policy,_=fixture(path,duplicate)
        block=next(wrap_original_ice_curve_integration(source,state,policy).blocks(np.zeros(2)))
        values.append(accumulate_source_normal(lambda:iter((block,)),2,expected_row_count=len(block.r),frozen_profile_token='frozen'))
    assert abs(values[0].cost-values[1].cost)<1e-13
    assert np.allclose(values[0].gradient,values[1].gradient,rtol=0,atol=1e-13)
    assert np.allclose(values[0].normal.toarray(),values[1].normal.toarray(),rtol=0,atol=1e-13)

def test_exact_row_sigma_identity_and_HELD_qualification_fail_closed(tmp_path):
    source,state,policy,_=fixture(tmp_path);state.blocks[0].groups[0][1][0]['sigma'][0]+=.1
    with pytest.raises(ValueError,match='sigma'):wrap_original_ice_curve_integration(source,state,policy)
    source,state,policy,_=fixture(tmp_path);d=json.loads(policy.read_text());d['original_packet']['sha256']='f'*64;policy.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='packet changed'):wrap_original_ice_curve_integration(source,state,policy)
    source,state,policy,_=fixture(tmp_path);entry=pickle.loads(Path(json.loads(policy.read_text())['original_packet']['path']).read_bytes())
    entry['programme_absolute_rows'][0]['collection_role']='HELD'
    p=tmp_path/'original.pkl';p.write_bytes(pickle.dumps(entry));d=json.loads(policy.read_text());d['original_packet']=file_reference(p);policy.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='HELD'):wrap_original_ice_curve_integration(source,state,policy)

def test_only_declared_original_complete_scan_N_is_removed():
    stored=np.full(3,2*np.sqrt(12));row=dict(sigma_modelling_decision=SCAN_DECLARATIONS[0]+' original',uv=np.zeros((3,2)))
    working,prior,_=original_working_sigma(row,stored,12)
    assert np.allclose(working,2.) and np.allclose(prior,np.sqrt(12))
    with pytest.raises(ValueError,match='changed'):original_working_sigma(row,stored,3)
    w,c,_=original_working_sigma({},stored,0)
    assert np.array_equal(w,stored) and np.all(c==1.)
    with pytest.raises(ValueError,match='Unrecognized'):original_working_sigma(dict(sigma_modelling_decision='sqrt(N) assumed'),stored,12)


def test_fresh_family_retains_original_board_payload_and_ICE_budget(tmp_path,monkeypatch):
    import structured_programme_source_candidate.activated_small_source_blocks as refresh
    base,state,policy,old=fixture(tmp_path)
    ownership=np.zeros(len(old.r),bool);ownership[:2]=True
    board=SimpleNamespace(base=base,original_board_curve_integration={},
        _payloads={'programme:main7':old.loss_weights},_new_slots={'programme:main7':ownership},
        _old_slots={'programme:main7':np.zeros(len(old.r),bool)},
        count=base.count,scales=base.scales,quadratic=base.quadratic,contract=base.contract)
    owner=wrap_original_ice_curve_integration(board,state,policy)
    def apply(raw):return SourceRowBlock(raw.r,raw.J,raw.quadratic,raw.scales,raw.row_ledger,old.loss_weights)
    board.apply_block=apply
    class Programme:
        actors=None
        def iter_source_blocks(self,q,jacobian):return iter((old,))
    programme=Programme();programme.state=state;programme.scales=old.scales;programme.quadratic=old.quadratic
    class Bias:
        def __init__(self):
            self.cap={'iter_source_blocks':programme.iter_source_blocks};self.programme={'main7':state.blocks[0]}
            self.validated_programme=set();self.count=2
        def unpack(self,value):return value,None
        def blocks(self,value,jacobian):yield from self.cap['iter_source_blocks'](value,jacobian)
    bias=Bias();monkeypatch.setattr(refresh,'ExplicitBiasSource',Bias);monkeypatch.setattr(refresh,'CurrentProgrammeSource',Programme)
    monkeypatch.setattr(refresh,'SOURCE_SHA','a'*64)
    monkeypatch.setattr(refresh,'ordered_block_evaluations',lambda owners,value,jacobian,threads:iter(((owners[0],(old.r,old.J)),)))
    ledger=dict(old.row_ledger,original_start=0,original_stop=len(old.r))
    actual=refresh._fresh_complete_programme_family([owner,board,bias,programme],np.zeros(2),ledger)
    expected=owner.apply_block(old)
    assert actual.r is old.r and actual.J is old.J
    assert actual.loss_weights.identity_digest==expected.loss_weights.identity_digest
    assert actual.loss_weights.integration_weights[:2].tolist()==[.2,.8]
    assert 'original_ice_curve_integration'in actual.row_ledger
