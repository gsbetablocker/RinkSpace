"""Prepare a named runtime shared-parameter stage, without changing source data.

The native consumer requires active known-local board material support for
advertisement charts, and fixed-local-corner support for rig/rink geometry.
It checks the complete scalar whitelist. Incidence is not a rank test.
Unmeasured print chart alignment and temporary camera subsets may be released
explicitly. The camera release requires the model's recorded initialization
provenance; coordinate gauge blocks remain unchanged.
"""
from copy import deepcopy
import argparse,json
from .observed_primal_restart import read,write_new


def build(config,model,names,*,release_camera_initializers=False):
    cfg=deepcopy(config)
    blocks={b['name']:(i,b) for i,b in enumerate(model['blocks'])}
    if not names or len(set(names))!=len(names):
        raise ValueError('Explicit, distinct block names required')
    release={r['block_name']:r for r in cfg.get('runtime_shared_parameter_stage_v2',{}).get('blocks',[])}
    expected={r['block_name']:r for r in cfg['audit_expected_free_physical_blocks']}
    for name in names:
        i,b=blocks[name]
        print_affine=name.startswith('print/') and len(b['values'])==6
        if not (print_affine or (name.startswith('rig/') and name.endswith('/optics12')) or name.startswith('ad/') or name.startswith('ice/circle/') or name=='scene/rink5' or (name.startswith('rig/') and name.endswith('/head15'))):
            raise ValueError('Unsupported shared physical block')
        if any(name.startswith(prefix)for prefix in cfg.get('temporary_runtime_fixed_physical_prefixes',[])):
            raise ValueError('Requested block conflicts with an explicit frozen prefix')
        release_alignment=name.startswith('print/group/') and name.endswith('/new_native')
        camera=name.startswith('rig/') and name.endswith(('/head15','/optics12'))
        release_camera=release_camera_initializers and camera
        if release_camera and not model.get('camera_bootstrap_rebase_v24',{}).get('all_shared_parameters_must_be_released_after_initialization',False):
            raise ValueError('Camera release requires explicit temporary-initialization provenance')
        free=[k for k in range(len(b['values'])) if k not in b.get('fixed_indices',[]) or (release_alignment and k in (1,3)) or release_camera]
        if not free:raise ValueError('No original non-gauge scalar remains')
        release[name]=dict(block_index=i,block_name=name,free_ambient_indices=free)
        if name.startswith('ad/'):release[name]['known_local_board_material_support']=True
        if release_alignment:release[name]['release_unmeasured_chart_alignment']=True
        if release_camera:release[name]['release_initialization_camera_indices']=True
        expected[name]=dict(block_index=i,block_name=name,ambient_size=len(b['values']),expected_tangent_size=len(free),free_ambient_indices=free)
    cfg['runtime_shared_parameter_stage_v2']=dict(schema='rinkfit.explicit_shared_parameter_stage.v2',
        source_packet_sha256=cfg['engineering_support_budget_v2']['source_packet_sha256'],
        world_values_measured=False,blocks=sorted(release.values(),key=lambda r:r['block_index']))
    cfg['temporary_runtime_fixed_physical_blocks']=[name for name in cfg['temporary_runtime_fixed_physical_blocks']if name not in release]
    cfg['temporary_runtime_fixed_nuisance_shots']=[]
    cfg['audit_expected_free_physical_blocks']=sorted(expected.values(),key=lambda r:r['block_index'])
    for key in ('protected_S01_scalar_prefixes','source_capture_init'):
        if key in cfg:raise ValueError('An older S01-specific initializer cannot become a shared stage')
    return cfg


def release_existing_s01_trajectory(config, model):
    """Release the already-defined source-supported S01 motion subspace."""
    cfg=deepcopy(config)
    if cfg.get('source_uv_capture_objective')!='rinkfit.sourceUV_observed_row_finite_initialization.v1':
        raise ValueError('S01 release currently requires the explicit measured-row initializer')
    provenance=model.get('s01_affine_recovery_corrections',{})
    if not provenance.get('original_exact_subspace_preserved') or provenance.get('baseline_function_resampled') is not False:
        raise ValueError('S01 trajectory must retain its original fine baseline and exact subspace')
    authoritative={v['block'] for v in model['shots']['s01']['control_coordinates']}
    release={r['block_name']:r for r in cfg['runtime_shared_parameter_stage_v2']['blocks']}
    expected={r['block_name']:r for r in cfg['audit_expected_free_physical_blocks']}
    count=0
    for manifold in model['linear_control_manifolds']:
        i=manifold['block']; block=model['blocks'][i]; name=block['name']
        if i not in authoritative or not name.startswith('shot/s01/control_group/'):
            continue
        size=len(block['values']); tangent=manifold['tangent_size']; basis=manifold['basis']
        if block.get('fixed_indices') or len(basis)!=size or any(len(row)!=tangent for row in basis):
            raise ValueError('Unexpected S01 affine coordinate layout')
        free=[k for k,row in enumerate(basis) if any(v!=0 for v in row)]
        if not free or tangent<=0:raise ValueError('No existing S01 motion directions')
        if any(name.startswith(p) for p in cfg.get('temporary_runtime_fixed_physical_prefixes',[])):
            raise ValueError('S01 release contradicts a runtime prefix restriction')
        release[name]=dict(block_index=i,block_name=name,free_ambient_indices=free,
                           expected_tangent_size=tangent,release_initialization_s01_trajectory=True)
        expected[name]=dict(block_index=i,block_name=name,ambient_size=size,
                            expected_tangent_size=tangent,free_ambient_indices=free)
        count+=1
    if count!=provenance['groups']:raise ValueError('Incomplete existing S01 trajectory release')
    cfg['runtime_shared_parameter_stage_v2']['blocks']=sorted(release.values(),key=lambda r:r['block_index'])
    cfg['runtime_shared_parameter_stage_v2']['release_s01_trajectory']=True
    cfg['audit_expected_free_physical_blocks']=sorted(expected.values(),key=lambda r:r['block_index'])
    cfg['temporary_runtime_fixed_physical_blocks']=[n for n in cfg['temporary_runtime_fixed_physical_blocks'] if n not in release]
    return cfg


def restore_original_print_vertices(config, model, converter_file):
    """Restore original shared vertex freedoms, retaining the current curve as seed."""
    from .observed_primal_restart import reference
    cfg=deepcopy(config); overlay=read(converter_file)
    if overlay['schema']!='rinkfit.prototype.shared_print_group_chart_reencoding.v1':
        raise ValueError('Original shared print converter required')
    stage=cfg['runtime_shared_parameter_stage_v2']
    if 'restore_original_print_vertices' in stage:raise ValueError('Print restoration already configured')
    release={r['block_name']:r for r in stage['blocks']}
    expected={r['block_name']:r for r in cfg['audit_expected_free_physical_blocks']}
    restored=set(); unused=set()
    for item in overlay['material_path_updates']:
        mi,pi=item['material_index'],item['path_index'];path=model['materials'][mi]['paths'][pi]
        if path!=item['path']:raise ValueError('Current print path differs from its original converter')
        bi=item['original_geometry_block'];b=model['blocks'][bi];name=b['name']
        if bi in restored or not name.startswith('print_component/polyline/') or b.get('fixed_indices'):
            raise ValueError('Original shared print registry differs')
        if 'numerical initialization' not in item['expected_path']['provenance']:
            raise ValueError('Cannot reinterpret a measured world shape')
        restored.add(bi); unused.add(model['blocks'][path['geometry_block']]['name'])
        free=list(range(len(b['values'])))
        release[name]=dict(block_index=bi,block_name=name,free_ambient_indices=free,restore_original_print_vertices=True)
        expected[name]=dict(block_index=bi,block_name=name,ambient_size=len(free),expected_tangent_size=len(free),free_ambient_indices=free)
    if len(restored)!=len(overlay['orphaned_vertex_block_updates']):raise ValueError('Incomplete original shape restoration')
    for name in unused:release.pop(name,None);expected.pop(name,None)
    stage['restore_original_print_vertices']=reference(converter_file)
    stage['blocks']=sorted(release.values(),key=lambda r:r['block_index'])
    cfg['audit_expected_free_physical_blocks']=sorted(expected.values(),key=lambda r:r['block_index'])
    frozen=set(cfg['temporary_runtime_fixed_physical_blocks'])
    frozen.difference_update(model['blocks'][i]['name'] for i in restored)
    frozen.update(unused)
    cfg['temporary_runtime_fixed_physical_blocks']=sorted(frozen)
    return cfg


def restore_full_original_s01_subspace(config, model):
    """Remove the two-bump initializer, retaining the original cubic constraints."""
    import numpy as np
    cfg=deepcopy(config);stage=cfg['runtime_shared_parameter_stage_v2']
    if not stage.get('release_s01_trajectory'):
        raise ValueError('Existing S01 trajectory must already be explicitly variable')
    if model.get('s01_full_original_subspace_v1'):
        raise ValueError('Original S01 motion space is already restored')
    release={r['block_name']:r for r in stage['blocks']}
    expected={r['block_name']:r for r in cfg['audit_expected_free_physical_blocks']}
    count=0;before=0;after=0
    for m in model['linear_control_manifolds']:
        i=m['block'];b=model['blocks'][i];name=b['name'];n=len(b['values'])
        if not name.startswith('shot/s01/control_group/'):continue
        if b.get('fixed_indices') or not release[name].get('release_initialization_s01_trajectory'):
            raise ValueError('Original motion group was not explicitly released')
        C=np.asarray(m['constraint_matrix'],float).reshape(-1,n)
        rank=int(np.linalg.matrix_rank(C)) if len(C) else 0
        if rank!=len(C):raise ValueError('Original cubic constraints must be independent')
        tangent=n-rank;free=list(range(n))
        release[name].update(free_ambient_indices=free,expected_tangent_size=tangent)
        expected[name].update(free_ambient_indices=free,expected_tangent_size=tangent)
        count+=1;before+=m['tangent_size'];after+=tangent
    if count!=model['s01_affine_recovery_corrections']['groups'] or after<=before:
        raise ValueError('Original S01 source-supported motion scope differs')
    stage['restore_full_original_s01_subspace']=True
    stage['blocks']=sorted(release.values(),key=lambda r:r['block_index'])
    cfg['audit_expected_free_physical_blocks']=sorted(expected.values(),key=lambda r:r['block_index'])
    return cfg


def release_original_s01_end_controls(config, model):
    """Release the remaining original motion controls; retain the dependent-angle gauge."""
    cfg=deepcopy(config);stage=cfg['runtime_shared_parameter_stage_v2']
    if not stage.get('restore_full_original_s01_subspace'):
        raise ValueError('Complete original grouped motion space required')
    release={r['block_name']:r for r in stage['blocks']}
    expected={r['block_name']:r for r in cfg['audit_expected_free_physical_blocks']}
    shot=model['shots']['s01'];added=[]
    for i in sorted({c['block']for c in shot['control_coordinates']}):
        b=model['blocks'][i];name=b['name']
        if not name.startswith('shot/s01/control/'):continue
        if len(b['values'])!=3 or i not in shot['control_blocks']:
            raise ValueError('Not an authoritative original fine control')
        free=[j for j in range(3)if j not in b.get('fixed_indices',[])]
        release[name]=dict(block_index=i,block_name=name,free_ambient_indices=free,expected_tangent_size=len(free),release_initialization_s01_trajectory=True)
        expected[name]=dict(block_index=i,block_name=name,ambient_size=3,free_ambient_indices=free,expected_tangent_size=len(free))
        added.append(name)
    total=sum(r['expected_tangent_size']for r in expected.values()if r['block_name'].startswith(('shot/s01/control/','shot/s01/control_group/')))
    if total!=model['s01_time_subspace_provenance']['coarse_free_time_scalars']:
        raise ValueError('S01 freedoms do not equal the original complete motion model')
    stage['release_original_s01_end_controls']=True
    stage['blocks']=sorted(release.values(),key=lambda r:r['block_index'])
    cfg['audit_expected_free_physical_blocks']=sorted(expected.values(),key=lambda r:r['block_index'])
    cfg['temporary_runtime_fixed_physical_blocks']=[n for n in cfg['temporary_runtime_fixed_physical_blocks']if n not in added]
    return cfg


def restore_original_s01_sparse_storage(config, model):
    """Reuse original 3-scalar blocks with exactly the same current trajectory."""
    cfg=deepcopy(config);stage=cfg['runtime_shared_parameter_stage_v2']
    if not stage.get('restore_full_original_s01_subspace') or not stage.get('release_original_s01_end_controls'):
        raise ValueError('The complete original motion space must already be released')
    release={r['block_name']:r for r in stage['blocks']}
    expected={r['block_name']:r for r in cfg['audit_expected_free_physical_blocks']}
    constrained={m['block']for m in model['linear_control_manifolds']}
    shot=model['shots']['s01'];unused=set();restored=set()
    for k,c in enumerate(shot['control_coordinates']):
        original=shot['control_blocks'][k];old=c['block']
        if old==original or old in constrained:continue
        b=model['blocks'][original];name=b['name'];prior=model['blocks'][old]['name']
        if not prior.startswith('shot/s01/control_group/') or name!='shot/s01/control/'+str(k) or b.get('fixed_indices'):
            raise ValueError('Not an original unconstrained fine control')
        unused.add(prior);restored.add(name);free=[0,1,2]
        release[name]=dict(block_index=original,block_name=name,free_ambient_indices=free,expected_tangent_size=3,release_initialization_s01_trajectory=True)
        expected[name]=dict(block_index=original,block_name=name,ambient_size=3,free_ambient_indices=free,expected_tangent_size=3)
    if not unused:raise ValueError('Sparse conversion is empty')
    for name in unused:release.pop(name,None);expected.pop(name,None)
    total=sum(r['expected_tangent_size']for r in expected.values()if r['block_name'].startswith(('shot/s01/control/','shot/s01/control_group/')))
    if total!=model['s01_time_subspace_provenance']['coarse_free_time_scalars']:
        raise ValueError('Sparse storage changed the complete motion space')
    stage['restore_original_s01_sparse_storage']=True
    stage['blocks']=sorted(release.values(),key=lambda r:r['block_index'])
    cfg['audit_expected_free_physical_blocks']=sorted(expected.values(),key=lambda r:r['block_index'])
    frozen=set(cfg['temporary_runtime_fixed_physical_blocks']);frozen.difference_update(restored);frozen.update(unused)
    cfg['temporary_runtime_fixed_physical_blocks']=sorted(frozen)
    return cfg


def release_supported_ice_initializers(config, model, packet_file):
    """Release unmeasured ice primitives supported by the admitted source packet."""
    import numpy as np
    from .joint_metrics import PacketArrays
    packet=PacketArrays(packet_file)
    cfg=deepcopy(config);stage=cfg['runtime_shared_parameter_stage_v2']
    from .io_utils import file_sha256
    if file_sha256(packet_file)!=stage['source_packet_sha256']:
        raise ValueError('The ice support must use the same admitted source packet')
    admitted=(packet.load('admission')==1)&(packet.load('fit_role')==1)
    used=set(np.unique(packet.load('material_index')[admitted]).tolist());supported={};unmeasured=set()
    for mi,mat in enumerate(model['materials']):
        if not mat['label'].startswith('ice.'):continue
        for p in mat['paths']:
            bi=p.get('geometry_block',-1)
            if bi<0 or p.get('physical_point') or p['kind']not in (0,1,4,5):continue
            b=model['blocks'][bi]
            if not b['name'].startswith('ice/'):continue
            provenance=p.get('provenance','')
            if any(k in provenance for k in ('initializer','initialization','legacy_static_seed_only_not_world_truth','not a physical measurement')):unmeasured.add(bi)
            if mi in used:supported.setdefault(bi,set()).add(mi)
    release={r['block_name']:r for r in stage['blocks']};expected={r['block_name']:r for r in cfg['audit_expected_free_physical_blocks']}
    for bi in sorted(set(supported)&unmeasured):
        b=model['blocks'][bi];name=b['name'];free=[i for i in range(len(b['values']))if i not in b.get('fixed_indices',[])]
        if not free:continue
        release[name]=dict(block_index=bi,block_name=name,free_ambient_indices=free,release_unmeasured_ice_geometry=True)
        expected[name]=dict(block_index=bi,block_name=name,ambient_size=len(b['values']),expected_tangent_size=len(free),free_ambient_indices=free)
        if name.startswith('ice/line/') and len(b['values'])==6:
            q=np.asarray(b['values'],dtype=float).reshape(3,2);v=q[1]-q[0];w=q[2]-q[0]
            if float(v@v)<=0 or abs(float(v[0]*w[1]-v[1]*w[0]))>128*np.finfo(float).eps*max(1.,float(np.linalg.norm(v)*np.linalg.norm(w))):
                raise ValueError('Shared line coverage points must remain collinear')
            release[name]['release_collinear_ice_carrier']=True
            expected[name]['expected_tangent_size']=5
    stage['release_unmeasured_ice_geometry']=True;stage['blocks']=sorted(release.values(),key=lambda r:r['block_index'])
    cfg['audit_expected_free_physical_blocks']=sorted(expected.values(),key=lambda r:r['block_index'])
    cfg['temporary_runtime_fixed_physical_blocks']=[n for n in cfg['temporary_runtime_fixed_physical_blocks']if n not in release]
    return cfg


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--model',required=True)
    p.add_argument('--release',nargs='+',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();print(json.dumps(write_new(a.out,build(read(a.config),read(a.model),a.release))))

if __name__=='__main__':main()
