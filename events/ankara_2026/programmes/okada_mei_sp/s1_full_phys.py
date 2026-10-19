"""Check or install the explicit complete-physical numerical successor.

This entry never constructs an owner, replaces a resume state or starts a fit.
The existing authenticated owner factory supplies its unchanged complete parent.
"""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
DEFAULT_RECIPE=Path(__file__).parent/'source_datasets/s1_wide/fullphys.json'


def reference(path):
    path=Path(path);return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def same_array(a,b):
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()


def authenticate(value):
    count=0
    if isinstance(value,dict):
        if {'path','sha256'}<=set(value):
            if reference(value['path'])!={key:value[key]for key in ('path','sha256')}:
                raise ValueError('Frozen complete-physical recipe reference changed: '+value['path'])
            count+=1
        count+=sum(authenticate(item)for item in value.values())
    elif isinstance(value,list):count+=sum(authenticate(item)for item in value)
    return count


def load_recipe(path=DEFAULT_RECIPE):
    recipe=json.loads(Path(path).read_text(encoding='utf8'))
    if recipe.get('schema')!='rinkspace.first-shot-full-physical-entry.v1' or recipe.get('entry')!=reference(__file__):
        raise ValueError('Exact new complete-physical entry and recipe required')
    count=authenticate(recipe)
    numeric=json.loads(Path(recipe['frozen_numerical_recipe']['path']).read_text(encoding='utf8'))
    count+=authenticate(numeric)
    with np.load(recipe['complete_parent_references']['accepted_parent_checkpoint']['path'],allow_pickle=False)as saved:
        q=saved['q'];aux_count=sum(key.startswith('aux__')for key in saved.files)
        if (hashlib.sha256(q.tobytes()).hexdigest()!=recipe['original_parent_q_sha256']
                or float(saved['damping'])!=recipe['original_parent_damping']
                or float(saved['next_line_search_fraction'])!=recipe['original_parent_fraction']):
            raise ValueError('Original complete numerical parent changed')
    point=_read_current_point(recipe)
    return recipe,dict(passed=True,status=recipe['status'],authenticated_references=count,
        construction_q_sha256=recipe['original_parent_q_sha256'],construction_aux_count=aux_count,
        current_numerical_q_sha256=recipe['current_numerical_q_sha256'],current_aux_count=len(point['aux']),
        retained_damping=point['damping'],retained_fraction=point['fraction'],
        owner_constructed=False,accepted_state_written=False,solver_started=False)


def _read_current_point(recipe):
    with np.load(recipe['current_complete_numerical_payload']['path'],allow_pickle=False)as saved:
        q=saved['q_augmented'].copy()
        if hashlib.sha256(q.tobytes()).hexdigest()!=recipe['current_numerical_q_sha256']:
            raise ValueError('Current numerical q changed; construction q must not replace it')
        aux={key[5:]:saved[key].copy()for key in saved.files if key.startswith('aux__')}
        if len(aux)!=recipe['current_aux_count']:raise ValueError('Complete current auxiliary inventory changed')
        return dict(q=q,aux=aux,physical_q=saved['physical_q_augmented'].copy(),
            expanded_q=saved['native_time_expanded_base_canonical_q'].copy(),
            damping=recipe['retained_numerical_damping'],fraction=recipe['retained_numerical_fraction'],
            checkpoint_is_accuracy_accepted=False,payload=recipe['current_complete_numerical_payload'])


def load_current_point(recipe_path=DEFAULT_RECIPE):
    """Explicit actual solver q/ALL auxiliary seed; never returns q6 or a probe."""
    recipe,_=load_recipe(recipe_path)
    return _read_current_point(recipe)


def install_public_policy(owner, *, driver, cache_directory, recipe_path=DEFAULT_RECIPE):
    """Install on the authenticated construction owner, without selecting solve q.

    The actual numerical point is explicitly returned by load_current_point().
    Its q and ALL current auxiliary arrays must be used by the caller; the
    owner's historical construction start is never the latest numerical point.
    """
    from calibration.program_tracking.first_shot_jacobian_reuse import install_source_jacobian_reuse
    from calibration.program_tracking.first_shot_prediction_refresh import install_prediction_refresh
    from calibration.program_tracking.first_shot_full_physical import install as install_physical
    recipe,_=load_recipe(recipe_path)
    if recipe['status']!='ready':raise ValueError('Real-owner complete-physical integration QA is not registered yet')
    initial=np.asarray(owner.start).copy();aux=owner.checkpoint_arrays();before=owner.solver_kwargs()
    with np.load(recipe['complete_parent_references']['accepted_parent_checkpoint']['path'],allow_pickle=False)as saved:
        if not np.array_equal(initial.view(np.uint64),saved['q'].view(np.uint64)):
            raise ValueError('Original canonical start must remain byte exact')
        for key in saved.files:
            if key.startswith('aux__'):
                name=key[5:]
                if name not in aux or not same_array(aux[name],saved[key]):
                    raise ValueError('Every complete original parent auxiliary must be retained: '+name)
    if (before['initial_damping']!=recipe['original_parent_damping']
            or before['initial_line_search_fraction']!=recipe['original_parent_fraction']):
        raise ValueError('Original saved lambda and line-search fraction required')
    entry=reference(__file__)
    install_source_jacobian_reuse(owner,cache_directory=cache_directory,entry_reference=entry)
    refresh=install_prediction_refresh(owner,driver=driver,entry_reference=entry,
        minimum_prediction_ratio=.25,reset_initial_fresh_source_fraction=False)
    physical=install_physical(owner,driver=driver,entry_reference=entry)
    after=owner.solver_kwargs();current_aux=owner.checkpoint_arrays()
    if (not np.array_equal(owner.start.view(np.uint64),initial.view(np.uint64))
            or after['initial_damping']!=before['initial_damping']
            or after['initial_line_search_fraction']!=before['initial_line_search_fraction']):
        raise ValueError('Numerical policy changed the original complete parent seed')
    for key,value in aux.items():
        if key not in current_aux or not same_array(value,current_aux[key]):
            raise ValueError('Numerical policy changed original auxiliary: '+key)
    return dict(recipe=reference(recipe_path),prediction_refresh=refresh,complete_physical=physical,
        original_q_all_aux_lambda_fraction_retained=True,
        owner_start_is_construction_baseline_only=True,
        actual_solver_start_must_be_current_numerical_q=recipe['current_numerical_q_sha256'],
        current_complete_numerical_payload=recipe['current_complete_numerical_payload'],
        explicit_solver_seed_api='load_current_point',solver_started=False)


def main():
    parser=argparse.ArgumentParser(description='只读检查新完整物理数值入口；不构建owner，不启动求解。')
    parser.add_argument('--recipe',type=Path,default=DEFAULT_RECIPE)
    args=parser.parse_args();_,result=load_recipe(args.recipe)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
