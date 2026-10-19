"""Observe already-produced programme value rows without source reevaluation."""
from pathlib import Path
import hashlib,json
import numpy as np
from calibration.solver_runtime.source_loss import (
    SourceLossWeights, loss_weight_arrays, source_loss_from_checkpoint,
    weighted_source_cost,
)

def _sha_array(a):return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()

def _loss_identity(loss, count):
    weights,normalization=loss_weight_arrays(loss,count)
    return tuple(None if a is None else (a.shape,_sha_array(a)) for a in (weights,normalization))

def _cost_rows(r,mask,loss):
    """Vectorized ranking payload, checked against the public source cost."""
    r=np.asarray(r,float);mask=np.asarray(mask,bool)
    total=weighted_source_cost(r,mask,loss)
    weights,normalization=loss_weight_arrays(loss,len(r))
    working=r if normalization is None else r*normalization
    a=np.abs(working)
    values=np.where(mask,.5*working*working,np.where(a<=2.,.5*working*working,2.*a-2.))
    if weights is not None:values*=weights
    if not np.isclose(values.sum(),total,rtol=1e-12,atol=1e-10):
        raise ValueError('Row-cost ranking differs from the public current source objective')
    return values

def _cost_span(r,mask,weights,normalization,start,stop):
    loss=None if weights is None else weights[start:stop]
    if normalization is not None:loss=SourceLossWeights(loss,normalization[start:stop])
    return weighted_source_cost(r[start:stop],mask[start:stop],loss)

def _main7_partition(block,state):
    """Bind to the actual current native layout, including compacted inputs."""
    family=block.row_ledger.get('family','')
    if not family.startswith('programme:'):
        raise ValueError('Programme row-cost diagnostic not applicable: native owner absent')
    name=family.split(':',1)[1]
    owners=[b for b in state.blocks if b.name==name]
    if len(owners)!=1:raise ValueError('Current main7 row-cost diagnostic not applicable: unique owner absent')
    owner=owners[0];bundle=owner.bundle;ledger=block.row_ledger
    ordinal=next(i for i,b in enumerate(state.blocks) if b is owner)
    start=sum(len(b.quadratic) for b in state.blocks[:ordinal])
    count=len(block.r);n=len(bundle.a);motion=int(bundle.motion_rows)
    absolute=2*sum(len(r['uv']) for _,rows in owner.groups for r in rows)
    native=int(bundle.source_residual_rows);priors=count-native
    if (ledger.get('family')!='programme:'+owner.name or ledger.get('original_start')!=start
            or ledger.get('original_stop')!=start+count or count!=len(owner.quadratic)
            or motion!=2*n or len(bundle.b)!=n or len(bundle.ia)!=n or len(bundle.ib)!=n
            or np.asarray(bundle.motion_sigma).shape!=(motion,) or native!=motion+absolute
            or priors!=len(bundle.stamps) or np.asarray(block.quadratic).shape!=(count,)
            or np.asarray(block.scales).shape!=(count,) or not np.all(block.quadratic[native:])
            or not np.all(block.scales[native:]==1.)):
        raise ValueError('Current main7 row-cost diagnostic not applicable: complete native/roll layout is not bound')
    return owner,motion,absolute,priors

def capture_existing_programme_baseline(value,profile,block,state,source_manifest_sha256):
    """Keep only the residual baseline; never retain a Jacobian or call a source."""
    _main7_partition(block,state)
    return dict(q_sha256=_sha_array(value),frozen_profile_token=profile,
        source_manifest_sha256=source_manifest_sha256,r=np.asarray(block.r).copy(),
        quadratic=np.asarray(block.quadratic).copy(),scales=np.asarray(block.scales).copy(),
        ledger=json.loads(json.dumps(block.row_ledger)),loss=getattr(block,'loss_weights',None))


def write_existing_main7_trial_row_cost(out,value,profile,block,state,pass_number,*,baseline=None):
    """Called only after an original value block already exists; no evaluator."""
    family=block.row_ledger['family']
    if baseline is None and family!='programme:main7':return
    owner,motion,absolute,priors=_main7_partition(block,state)
    out=Path(out)
    prefix=baseline if baseline is not None else json.loads((out/'initial_normal_source_blocks/completed_prefix.json').read_bytes())
    if prefix['frozen_profile_token']!=profile:
        raise ValueError('Row-cost baseline profile differs; do not compare different profiled targets')
    if baseline is not None:
        initial=baseline['r'];quadratic=baseline['quadratic'];scales=baseline['scales']
        baseline_ledger=baseline['ledger'];baseline_loss=baseline['loss']
    else:
        ordinal=next(i for i,b in enumerate(prefix['completed_blocks'])if b['family']=='programme:main7')
        baseline_path=out/'initial_normal_source_blocks'/(str(ordinal).zfill(3)+'.npz')
        with np.load(baseline_path,allow_pickle=False)as z:
            if str(z['frozen_profile_token'].item())!=profile or str(z['q_sha256'].item())!=prefix['q_sha256']:
                raise ValueError('Saved baseline source identity differs')
            initial=z['r'].copy();quadratic=z['quadratic'].copy();scales=z['scales'].copy()
            baseline_ledger=json.loads(str(z['ledger_json'].item()))
            baseline_loss=source_loss_from_checkpoint(z)
    residual=np.asarray(block.r,float)
    loss=getattr(block,'loss_weights',None)
    if (residual.shape!=initial.shape or not np.array_equal(block.quadratic,quadratic)
            or not np.array_equal(block.scales,scales) or baseline_ledger!=block.row_ledger
            or _loss_identity(baseline_loss,len(initial))!=_loss_identity(loss,len(residual))):
        raise ValueError('Source row or loss inventory changed')
    weights,normalization=loss_weight_arrays(loss,len(residual))
    old_cost=_cost_rows(initial,quadratic,loss);new_cost=_cost_rows(residual,quadratic,loss)
    delta=new_cost-old_cost
    parts=[]
    for name,lo,hi in(('motion',0,motion),('absolute_original_FIT',motion,motion+absolute),('physical_roll_prior',motion+absolute,len(residual))):
        before=_cost_span(initial,quadratic,weights,normalization,lo,hi)
        after=_cost_span(residual,quadratic,weights,normalization,lo,hi)
        parts.append(dict(family=name,scalar_rows=hi-lo,actual_before_cost=before,
            actual_after_cost=after,actual_cost_delta=after-before))
    absolute_rows=[];cursor=motion;row_index=0
    for pts,rows in owner.groups:
        for row in rows:
            n=2*len(row['uv'])
            before=_cost_span(initial,quadratic,weights,normalization,cursor,cursor+n)
            after=_cost_span(residual,quadratic,weights,normalization,cursor,cursor+n)
            absolute_rows.append(dict(original_programme_FIT_row_index=row_index,actual_pts=int(pts),
                source_row_pts=int(row['pts']),id=row['id'],original_split=row.get('split'),
                collection_role='FIT',UV_count=len(row['uv']),sigma=np.asarray(row['sigma'],float).tolist(),
                source_uv=np.asarray(row['uv'],float).tolist(),original_normal_interval=None if 'normal_interval'not in row
                else np.asarray(row['normal_interval'],float).tolist(),
                source_corner_roles=row.get('corner_roles'),source_ad_ids=row.get('ad_ids'),
                scanline_fixed_axis=row.get('scanline_fixed_axis'),scanline_branch=row.get('scanline_branch'),
                actual_before_cost=before,actual_after_cost=after,
                actual_cost_delta=after-before,scalar_start=cursor,scalar_stop=cursor+n))
            cursor+=n;row_index+=1
    motion_delta=delta[:motion].reshape(-1,2).sum(axis=1)
    strongest_motion=[]
    for index in np.argsort(motion_delta)[-10:][::-1]:
        strongest_motion.append(dict(active_runtime_motion_point_index=int(index),
            source_actual_pts_a=int(owner.bundle.stamps[int(owner.bundle.ia[index])]),
            source_actual_pts_b=int(owner.bundle.stamps[int(owner.bundle.ib[index])]),
            source_uv_a=np.asarray(owner.bundle.a[index],float).tolist(),source_uv_b=np.asarray(owner.bundle.b[index],float).tolist(),
            original_motion_sigma=np.asarray(owner.bundle.motion_sigma[2*index:2*index+2],float).tolist(),
            actual_cost_delta=float(motion_delta[index])))
    result=dict(schema='rinkspace.existing-programme-actual-trial-source-row-cost.v2',
        complete_q_sha256=_sha_array(value),initial_complete_q_sha256=prefix['q_sha256'],
        frozen_profile_token=profile,source_manifest_sha256=prefix['source_manifest_sha256'],
        original_source_row_ledger=block.row_ledger,actual_trial_r_sha256=_sha_array(residual),
        actual_baseline_r_sha256=_sha_array(initial),original_quadratic_mask_sha256=_sha_array(quadratic),
        actual_source_partition=parts,
        current_native_layout=dict(motion_scalar_rows=motion,absolute_scalar_rows=absolute,
            physical_roll_prior_scalar_rows=priors,complete_scalar_rows=len(residual),
            row_binding='actual block ledger and current complete native bundle/groups'),
        observed_programme_family=family,
        residual_baseline_only_no_J_retained=baseline is not None,
        cost_rule='public weighted_source_cost; original component Huber2/quadratic then source integration',
        source_loss_policy_identity=_loss_identity(loss,len(residual)),
        actual_before_cost=weighted_source_cost(initial,quadratic,loss),
        actual_after_cost=weighted_source_cost(residual,quadratic,loss),
        largest_actual_absolute_source_cost_increases=sorted(absolute_rows,key=lambda x:x['actual_cost_delta'],reverse=True)[:10],
        largest_actual_motion_pair_cost_increases=strongest_motion,
        source_arrays_not_modified=True,original_operator_and_source_roles_preserved=True,
        extra_source_calls=0,actual_nonlinear_values_not_J_predictions=True,accuracy_accepted=False)
    path=out/('actual_trial_'+owner.name+'_source_row_cost_pass_'+str(pass_number).zfill(4)+'.json')
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    return path
