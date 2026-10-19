from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import pickle
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from calibration.solver_runtime import install
install()
from calibration.solver_runtime.original_wide_curve_integration import (
    SCHEMA, wrap_original_wide_curve_integration, checked_wide_curve_integration_reuse)
from calibration.solver_runtime.source_loss import (
    SourceLossWeights, weighted_source_cost, source_irls_weights,
    source_loss_checkpoint_arrays, source_loss_from_checkpoint)
from rinkspace.source_sampling import compile_curve_sampling
from z190_streaming_source_normal import SourceRowBlock, accumulate_source_normal, replay_source_cost
from wide_anchor_candidate.complete_wide_free_optics_source import CompleteWideSource


def ref(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def numeric_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value, dtype='<f8').tobytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item()), encoding='utf-8')


def fixture(tmp_path, *, duplicate=False):
    # Same actual45/2144/5808/23776 layout, but independent synthetic pixels.
    counts=np.full(45,48);counts[26]=29;counts[44]=51
    observations=[];uvs=[];sigmas=[];records=[];rows=[];evidence=[]
    parent=tmp_path/'original.json';parent.write_text('{}',encoding='utf-8');authority=ref(parent)
    source_files={key:authority for key in ('v5_problem','v5_additional','step02','annotation_inventory','user_ad_digitization')}
    packet=dict(source_pts=56218880,source_time_base='1/15360',source_video_sha256='a'*64,
        native_image=dict(path=str(tmp_path/'not_decoded.png'),sha256='b'*64),source_files=source_files)
    for i,n in enumerate(counts):
        n=int(n);group=i//2 if i<10 else i-5
        uv=np.c_[np.arange(n)+i*100,np.full(n,i+3.)]
        if duplicate and i==0:uv[1]=uv[0]
        sigma=np.full(n,3. if i!=26 else 4.)
        target=dict(kind='circle_outline',feature_id='C'+str(group))
        literal=4. if i==26 else 1.
        original=dict(id='curve'+str(i),pixels_uv=uv.tolist(),sigma_px=float(sigma[0]),target=target,weight=literal)
        operator='wide_full_Brown_native_xy_nearest_finite_curve:'+json.dumps(target,sort_keys=True,separators=(',',':'))
        row=dict(original_v5_index=i,id=original['id'],points=n,target=target,
            original_point_indices=list(range(n)),source_UV_float64_sha256=numeric_sha(uv),
            source_sigma_float64_sha256=numeric_sha(sigma),original_sigma_px=float(sigma[0]),
            stored_sigma_px=sigma,working_sigma_px=sigma,prior_sampling_multiplier=np.ones(n),
            point_integration_mass=np.ones(n),sampling_kind=['measured']*n,actual_pts=56218880,
            source_video_sha256='a'*64,physical_curve_id='physical'+str(group),operator_id=operator,
            physical_identity_status='authenticated_synthetic_fixture',source_support_basis='isolated_authenticated_source_anchors',
            source_row_reference=authority,literal_source_weight=literal,global_engineering_multiplier=4.)
        ev=dict(id=original['id'],original_v5_index=i,target=target,points=n,native_uv_sha256=numeric_sha(uv),
            physical_curve_id=row['physical_curve_id'],physical_identity_status=row['physical_identity_status'],
            source_row_reference=authority,source_support_proof=dict(kind='isolated_authenticated_original_anchor_support',continuous_support_claim=False))
        record=dict(id=original['id'],split='fit',actual_pts=56218880,source_video_sha256='a'*64,
            physical_curve_id=row['physical_curve_id'],operator_id=operator,uv=uv,stored_sigma_px=sigma,
            working_sigma_px=sigma,prior_sampling_multiplier=np.ones(n),point_integration_mass=np.ones(n),
            sampling_kind=['measured']*n,source_anchor_identity_authority=authority)
        observations.append(original);uvs.append(uv);sigmas.append(sigma);records.append(record);rows.append(row);evidence.append(ev)
    duplicate_policy=dict(kind='same_source_actual_physical_curve_operator_UV_share_budget_preserve_pointwise_sigma',source_anchor_identity_authority=authority)
    plan=compile_curve_sampling(records,duplicate_anchor_policy=duplicate_policy)
    for row in rows:row['point_integration_weights']=plan['rows'][row['id']]['point_integration_weights']
    audit=tmp_path/'audit.json';dump(audit,dict(schema='rinkspace.current-wide45-exact-original-support-audit.v1',source_packet_contract=packet,source_rows=evidence))
    import rinkspace.source_sampling as compiler
    policy=dict(schema=SCHEMA,curve_budget=1.,support_basis='isolated_authenticated_source_anchors',normalization=None,
        enable_point_compression=False,preserve_literal_source_weights=True,preserve_global_engineering_multiplier=True,
        global_engineering_multiplier=4.,working_sigma_unchanged=True,
        original_observation_indices=list(range(45)),original_UV_count=2144,
        source_files=source_files,source_packet_reference=authority,source_video_sha256='a'*64,
        source_pts=56218880,native_pts=56218880,actual_pts=56218880,time_base='1/15360',native_image=packet['native_image'],
        source_support_audit_reference=ref(audit),duplicate_anchor_policy=duplicate_policy,
        public_compiler_reference=ref(Path(compiler.__file__)),rows=rows,groups=plan['groups'])
    path=tmp_path/'policy.json';dump(path,policy)
    literal=np.concatenate([np.full(2*len(uv),r['weight']) for uv,r in zip(uvs,observations)])
    cost_weights=np.r_[literal,np.full(4,.5),np.full(48,.25),np.ones(2+1466)]
    scales=np.r_[np.concatenate([np.repeat(s,2)for s in sigmas]),np.full(54,4.),np.full(1466,2*np.sqrt(733))]
    mask=np.r_[np.zeros(4342,bool),np.ones(1466,bool)]
    residual=np.resize(np.array([.3,-3.,1.9,-2.1,10.,-10.]),5808)
    jac=csr_matrix(np.c_[np.resize([1.,2.,3.],5808),np.resize([-1.,.5,2.],5808),np.ones(5808)])
    factor=SimpleNamespace(packet=SimpleNamespace(contract=packet),observations=observations,uvs=uvs,sigmas=sigmas,
        cost_weights=cost_weights,scales=scales,quadratic=mask,private_camera_parameter_count=25,contract={},
        evaluate=lambda core,private,jacobian=True:(residual,jac if jacobian else None,np.zeros((5808,25)) if jacobian else None))
    before=SourceRowBlock(np.array([.4,4.]),csr_matrix([[1.,0.,0.,0.,0.,0.],[0.,1.,0.,0.,0.,0.]]),np.zeros(2,bool),np.ones(2),dict(family='existing'),SourceLossWeights([.2,.8]))
    base=SimpleNamespace(count=6,core_count=3,scales=before.scales,quadratic=before.quadratic,updater=None,
        blocks=lambda q,jacobian=True:iter([before]))
    source=CompleteWideSource(base,factor,4)
    return SimpleNamespace(source=source,factor=factor,path=path,policy=policy,residual=residual,jac=jac,before=before)


def test_default_and_existing_base_loss_payload_survive_wide_wrapper(tmp_path):
    f=fixture(tmp_path)
    assert wrap_original_wide_curve_integration(f.source,f.factor,None)is f.source
    block=next(f.source.blocks(np.zeros(31)))
    assert block.loss_weights is f.before.loss_weights
    assert block.r is f.before.r and block.scales is f.before.scales


def test_expanded_Huber_objective_normal_and_raw_layout_preserved(tmp_path):
    f=fixture(tmp_path);owner=wrap_original_wide_curve_integration(f.source,f.factor,f.path)
    old=list(f.source.blocks(np.zeros(31)))[1];new=owner.apply_block(old)
    assert len(new.r)==23776 and len(f.factor.scales)==5808
    assert new.r is old.r and new.J is old.J and new.scales is old.scales and new.quadratic is old.quadratic
    assert new.loss_weights.working_normalization_multiplier is None
    weights=np.concatenate([np.repeat(r['point_integration_weights'],2)for r in f.policy['rows']]+[np.ones(1520)])
    assert np.array_equal(new.loss_weights.integration_weights,weights[f.source.row_indices])
    assert np.all(weights>0) and np.array_equal(weights[4288:],np.ones(1520))
    for group in owner.original_wide_curve_integration['groups']:assert group['point_integration_weight_sum']==1.
    unique_loss=SourceLossWeights(4*f.factor.cost_weights*weights)
    expected=weighted_source_cost(f.residual,f.factor.quadratic,unique_loss)
    assert weighted_source_cost(new.r,new.quadratic,new.loss_weights)==pytest.approx(expected,abs=1e-9)
    normal=accumulate_source_normal(lambda:iter([new]),31,expected_row_count=23776,frozen_profile_token='same-original-profile')
    irls=source_irls_weights(f.residual,f.factor.quadratic,unique_loss)
    assert np.allclose(normal.gradient[:3],f.jac.T@(irls*f.residual),atol=1e-10)
    assert np.allclose(normal.normal.toarray()[:3,:3],f.jac.toarray().T@(irls[:,None]*f.jac.toarray()),atol=1e-10)
    assert np.all(normal.gradient[3:]==0)
    with pytest.raises(ValueError):replay_source_cost(lambda:iter([old]),normal.contract,frozen_profile_token='same-original-profile')
    with pytest.raises(ValueError,match='twice'):owner.apply_block(new)
    payload=source_loss_from_checkpoint(source_loss_checkpoint_arrays(new.loss_weights,len(new.r)))
    assert payload.identity_digest==new.loss_weights.identity_digest
    assert pickle.loads(pickle.dumps(payload)).identity_digest==payload.identity_digest
    # A true derivative of the weighted objective, on both sides of Huber2.
    direction=np.array([.2,-.3,.1]);jv=f.jac@direction;h=1e-4
    fd=(weighted_source_cost(f.residual+h*jv,f.factor.quadratic,unique_loss)-weighted_source_cost(f.residual-h*jv,f.factor.quadratic,unique_loss))/(2*h)
    assert fd==pytest.approx(normal.gradient[:3]@direction,abs=2e-6)


@pytest.mark.parametrize('change', ['sigma','UV','literal','target','operator','mass','interpolated','normalization','groups','source_sha','exposure'])
def test_policy_does_not_change_source_or_self_certify_weights(tmp_path,change):
    f=fixture(tmp_path);p=json.loads(f.path.read_text())
    if change=='sigma':p['rows'][0]['working_sigma_px'][0]+=.1
    elif change=='UV':p['rows'][0]['source_UV_float64_sha256']='0'*64
    elif change=='literal':p['rows'][26]['literal_source_weight']=1.
    elif change=='target':p['rows'][0]['target']['feature_id']='wrong'
    elif change=='operator':p['rows'][0]['operator_id']='new_duplicate_identity'
    elif change=='mass':p['rows'][0]['point_integration_mass'][0]=2.
    elif change=='interpolated':p['rows'][0]['sampling_kind'][0]='interpolated'
    elif change=='normalization':p['normalization']=2.
    elif change=='groups':p['groups'][0]['engineering_curve_budget']=2.
    elif change=='source_sha':p['source_files']['v5_problem']['sha256']='0'*64
    elif change=='exposure':p['actual_pts']+=1
    dump(f.path,p)
    with pytest.raises(ValueError):wrap_original_wide_curve_integration(f.source,f.factor,f.path)


@pytest.mark.parametrize('option', ['reuse_complete_linearization','reuse_source_blocks_from','reuse_wide_linearization','verify_source_blocks_from'])
def test_new_loss_target_rejects_every_persistent_old_cache(tmp_path,option):
    args=SimpleNamespace(original_wide_curve_integration_policy=tmp_path/'policy.json',reuse_source_jacobian=True)
    checked_wide_curve_integration_reuse(args) # RAM reuse begins with fresh normal.
    setattr(args,option,tmp_path/'old')
    with pytest.raises(ValueError,match='fresh source'):checked_wide_curve_integration_reuse(args)
    args.original_wide_curve_integration_policy=None;checked_wide_curve_integration_reuse(args)


def test_isolated_exact_duplicates_share_positive_anchor_mass(tmp_path):
    f=fixture(tmp_path,duplicate=True);owner=wrap_original_wide_curve_integration(f.source,f.factor,f.path)
    weights=f.policy['rows'][0]['point_integration_weights']
    assert weights[0]==weights[1] and 2*weights[0]==weights[2]
    assert len(owner.original_wide_curve_integration['groups'])==40
    # Pair coordinates each retain the SAME alpha, rather than halve it.
    unique=f.source.row_indices
    expanded=owner._payload.integration_weights
    assert expanded[np.flatnonzero(unique==0)[0]]==expanded[np.flatnonzero(unique==1)[0]]


def test_optional_wide_policy_is_authenticated_by_existing_same_q_restore_gate(tmp_path):
    from scripts.diagnostics.verify_review85_startup_binding_from_receipts import (
        BINDING,actual_receipt_carriers,invoke)
    if not BINDING.exists():pytest.skip('Actual saved same-q cap receipts unavailable')
    f=actual_receipt_carriers(tmp_path)
    directory=tmp_path/'wide';directory.mkdir();wide=fixture(directory)
    owner=wrap_original_wide_curve_integration(wide.source,wide.factor,wide.path)
    contract=owner.original_wide_curve_integration
    f.manifest['original_wide_curve_integration']=deepcopy(contract)
    f.manifest['source_contract'].setdefault('evaluation_source_contract',{})['original_wide_curve_integration']=deepcopy(contract)
    f.scope['admitted_operator_sources']['original_wide_curve_integration']=dict(reference=ref(wide.path))
    f.binding=tmp_path/'optional_startup_binding.json';dump(f.binding,f.scope)
    packet,_=invoke(f)
    for family,values in f.parent['families'].items():
        n=len(values['group_ids'])
        for key in ('group_ids','limits','uncertainties'):assert packet['families'][family][key][:n]==values[key]
    f.manifest['original_wide_curve_integration']['fresh_source_J_and_profiles_required']=False
    with pytest.raises(ValueError,match='curve integration policy differs'):invoke(f)
