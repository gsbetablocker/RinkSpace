from copy import deepcopy
import numpy as np
import pytest

from rinkspace.source_sampling import source_support_masses, source_support_integration_weights, compile_curve_sampling


def row(identity='curve', uv=None, *, sigma=6., prior=1., physical='C0', actual=100):
    uv = [[0., 0.], [10., 0.], [20., 0.]] if uv is None else uv
    n = len(uv)
    return dict(id=identity, split='fit', source_video_sha256='a'*64, actual_pts=actual,
        physical_curve_id=physical, operator_id='finite-curve', uv=uv,
        stored_sigma_px=np.full(n, sigma*prior), working_sigma_px=np.full(n, sigma),
        prior_sampling_multiplier=np.full(n, prior), sampling_kind=['measured']*n,
        point_integration_mass=source_support_masses(uv, [list(range(n))]),
        interpolation_parents=[None]*n)


def huber_components(value):
    value = np.asarray(value)
    return np.where(abs(value)<=2., .5*value**2, 2.*abs(value)-2.)


def grouped_cost(records, native_errors):
    plan = compile_curve_sampling(records)
    return sum(float(np.repeat(plan['rows'][r['id']]['point_integration_weights'], 2)
        @ huber_components(np.asarray(native_errors[r['id']])/np.repeat(r['working_sigma_px'], 2)))
        for r in records)


def test_source_paths_repeat_or_reverse_without_new_support_and_do_not_bridge_gaps():
    uv = [[0., 0.], [10., 0.], [20., 0.], [100., 0.], [110., 0.]]
    expected = [5., 10., 5., 5., 5.]
    assert np.array_equal(source_support_masses(uv, [[0, 1, 2], [3, 4]]), expected)
    assert np.array_equal(source_support_masses(uv, [[0, 1, 2], [2, 1, 0], [3, 4]]), expected)
    with pytest.raises(ValueError, match='coverage'):
        source_support_masses(uv, [[0, 1, 2]])


def test_direct_neural_integration_divides_duplicate_mass_across_annotations():
    uv=[[0.,0.],[10.,0.],[20.,0.],[0.,0.],[10.,0.],[20.,0.]]
    weights=source_support_integration_weights(uv,[[0,1,2],[3,4,5]])
    assert np.array_equal(weights,[.125,.25,.125,.125,.25,.125])
    assert weights.sum()==1.
    original=source_support_integration_weights(uv[:3],[[0,1,2]])
    assert np.array_equal(weights[:3]+weights[3:],original)


def test_duplicate_raw_curve_rows_have_identical_cost_gradient_and_normal():
    original = row()
    duplicate = deepcopy(original); duplicate['id']='copy'
    errors = np.array([3., -4., 18., -20., 120., -130.])
    baseline = grouped_cost([original], {'curve': errors})
    assert grouped_cost([original, duplicate], {'curve': errors, 'copy': errors}) == baseline
    old = compile_curve_sampling([original])
    new = compile_curve_sampling([original, duplicate])
    J = np.arange(12.).reshape(6, 2)+1.
    def normal(plan, ids):
        H = np.zeros((2,2)); g=np.zeros(2)
        z=errors/6.; robust=np.minimum(1., 2./np.maximum(abs(z), 1e-12))
        for identity in ids:
            weight=np.repeat(plan['rows'][identity]['point_integration_weights'],2)*robust
            H += (J/6.).T @ (weight[:,None]*(J/6.))
            g += (J/6.).T @ (weight*z)
        return H,g
    a,b=normal(old,['curve']),normal(new,['curve','copy'])
    assert np.array_equal(a[0],b[0]) and np.array_equal(a[1],b[1])
    assert new['groups'][0]['engineering_curve_budget']==1.
    assert new['groups'][0]['stored_measured_copies']==6
    assert new['groups'][0]['unique_source_operator_UV_anchors']==3


def test_interpolation_density_keeps_original_cost_and_source_budget():
    original=row(uv=[[0.,0.],[20.,0.]])
    dense=deepcopy(original); dense['id']='dense'
    t=np.linspace(0.,1.,1001)
    dense['uv']=np.c_[20*t,np.zeros(len(t))]
    dense['sampling_kind']=['interpolated']*len(t)
    dense['sampling_kind'][0]=dense['sampling_kind'][-1]='measured'
    measured=np.array([kind=='measured'for kind in dense['sampling_kind']])
    dense['point_integration_mass']=source_support_masses(dense['uv'],[list(range(len(t)))],measured_mask=measured)
    dense['interpolation_parents']=[None]+[[dict(row_id='dense',point_index=0,weight=float(1-x)),
        dict(row_id='dense',point_index=len(t)-1,weight=float(x))] for x in t[1:-1]]+[None]
    for name in ('stored_sigma_px','working_sigma_px','prior_sampling_multiplier'):
        dense[name]=np.full(len(t),6. if name!='prior_sampling_multiplier' else 1.)
    base_error=np.array([120.,-6.,60.,3.])
    # Large arbitrary residuals at derived points cannot add source evidence.
    dense_error=np.full(2*len(t),1e6);dense_error[:2]=base_error[:2];dense_error[-2:]=base_error[-2:]
    assert grouped_cost([original],{'curve':base_error})==grouped_cost([dense],{'dense':dense_error})
    plan=compile_curve_sampling([dense])
    assert np.count_nonzero(plan['rows']['dense']['point_integration_weights'])==2
    assert plan['groups'][0]['effective_sum_inverse_working_sigma_squared']==1/36.


def test_explicit_embedded_sqrt_N_is_undone_once_and_uniform_mean_retained():
    source=row(prior=np.sqrt(3.))
    source['point_integration_mass']=np.ones(3)
    result=compile_curve_sampling([source])['rows']['curve']
    assert np.allclose(result['quadratic_residual_J_multiplier'],np.ones(3),rtol=0.,atol=1e-15)
    assert np.allclose(result['working_normalization_multiplier'],np.sqrt(3.))
    assert np.allclose(result['point_integration_weights'],1/3.)
    assert np.array_equal(source['stored_sigma_px'],np.full(3,6*np.sqrt(3.)))


def test_split_curve_rows_share_one_budget_but_distinct_curves_and_exposures_do_not():
    first=row('part7',[[float(i),0.]for i in range(7)],sigma=4.)
    second=row('part1',[[10.,0.]],sigma=4.)
    first['point_integration_mass']=np.ones(7)
    second['point_integration_mass']=np.ones(1)
    third=row('other_physical',physical='C',sigma=4.)
    fourth=row('other_exposure',actual=200,sigma=4.)
    plan=compile_curve_sampling([first,second,third,fourth])
    assert len(plan['groups'])==3
    assert np.array_equal(plan['rows']['part7']['point_integration_weights'],np.full(7,1/8.))
    assert plan['rows']['part1']['point_integration_weights'][0]==1/8.
    assert all(g['engineering_curve_budget']==1. for g in plan['groups'])


@pytest.mark.parametrize('invalid',['held','scale_decomposition','duplicate_mass','duplicate_sigma','derived_mass','derived_lineage'])
def test_invalid_sampling_policy_fails_closed(invalid):
    source=row();records=[source]
    if invalid=='held':source['split']='holdout'
    elif invalid=='scale_decomposition':source['prior_sampling_multiplier'][1]=10.
    elif invalid.startswith('duplicate'):
        other=deepcopy(source);other['id']='copy';records.append(other)
        if invalid=='duplicate_mass':other['point_integration_mass'][1]+=1.
        else:
            other['working_sigma_px'][1]*=2.;other['stored_sigma_px'][1]*=2.
    else:
        source['sampling_kind'][1]='interpolated'
        if invalid=='derived_lineage':source['point_integration_mass'][1]=0.
    with pytest.raises(ValueError):compile_curve_sampling(records)


def test_arc_length_density_does_not_multiply_engineering_precision():
    sparse=row(uv=[[0.,0.],[20.,0.]])
    dense=row(uv=np.c_[np.linspace(0.,20.,201),np.zeros(201)].tolist())
    a,b=compile_curve_sampling([sparse]),compile_curve_sampling([dense])
    assert np.allclose([a['groups'][0]['source_support_mass'],b['groups'][0]['source_support_mass']],20.,rtol=0.,atol=1e-12)
    assert np.isclose(a['groups'][0]['effective_sum_inverse_working_sigma_squared'],
        b['groups'][0]['effective_sum_inverse_working_sigma_squared'],rtol=1e-14)
    # Varying true raw RGB support improves quadrature, not an independence claim.
    assert a['measured_statistical_independence_claimed'] is False


def test_authenticated_duplicate_budget_preserves_original_pointwise_sigma():
    original=row('first',sigma=3.)
    other=row('second',sigma=4.)
    authority=dict(path='authenticated_original_pixel_packet.pkl',sha256='b'*64)
    for source in (original,other):source['source_anchor_identity_authority']=authority
    policy=dict(kind='same_source_actual_physical_curve_operator_UV_share_budget_preserve_pointwise_sigma',
        source_anchor_identity_authority=authority)
    with pytest.raises(ValueError,match='conflicts'):compile_curve_sampling([original,other])
    plan=compile_curve_sampling([original,other],duplicate_anchor_policy=policy)
    weights=compile_curve_sampling([original])['rows']['first']['point_integration_weights']
    assert np.array_equal(plan['rows']['first']['point_integration_weights'],weights/2.)
    assert np.array_equal(plan['rows']['second']['point_integration_weights'],weights/2.)
    assert np.array_equal(original['stored_sigma_px'],np.full(3,3.))
    assert np.array_equal(other['stored_sigma_px'],np.full(3,4.))
    assert np.isclose(plan['groups'][0]['effective_sum_inverse_working_sigma_squared'],.5/9.+.5/16.)
    changed=deepcopy(other);changed['source_anchor_identity_authority']={'path':'other_packet','sha256':'c'*64}
    with pytest.raises(ValueError,match='authority'):compile_curve_sampling([original,changed],duplicate_anchor_policy=policy)
    changed=deepcopy(other);changed['point_integration_mass'][1]+=1.
    with pytest.raises(ValueError,match='conflicts'):compile_curve_sampling([original,changed],duplicate_anchor_policy=policy)
    distinct=deepcopy(other);distinct['physical_curve_id']='different-curve'
    assert len(compile_curve_sampling([original,distinct],duplicate_anchor_policy=policy)['groups'])==2
