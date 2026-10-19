"""Small existing-format cache verification; never construct the full graph."""
from pathlib import Path
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace
from collections import OrderedDict
import argparse
import hashlib
import json
import sys
import numpy as np
from scipy.sparse import csr_matrix

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from calibration.solver_runtime import install
install()
from structured_programme_source_candidate import complete_same_state_replay as replay
from structured_programme_source_candidate import source_block_target_extension_replay as extension
from z190_streaming_source_normal import SourceRowBlock,accumulate_source_normal
from z190_joint_source_solver import _frozen_profile_token


class Profiles:
    def __init__(self):self.arrays={f'fixture_profile_{k:02d}':np.array([k+.25,k+.75])for k in range(18)}
    def iter_checkpoint_arrays(self):yield from self.arrays.items()


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')


def main(output):
    output=Path(output).resolve()
    if output.exists():raise ValueError('Fresh separate verifier output required')
    output.mkdir(parents=True)
    q=np.array([.25,.75,1.25]);fixed=[2];shared=np.array([True,False,False]);profiles=Profiles();token=_frozen_profile_token(profiles)
    source=SimpleNamespace(updater=profiles,quadratic=np.array([True,False,True,True]),scales=np.array([2.,3.,4.,4.]))
    row=dict(id='actual-source-field-fixture-parent',split='fit',pts=229618944,actual_pts=229618944,
        uv=[[225.7935494859925,674.4662017358396]],sigma_px=[4.],feature_id='goal_line_right',
        curve={'kind':'current_ice_segment','feature_id':'goal_line_right','segment_index':0})
    packet=output/'fixture_parent_ice.json';write(packet,dict(schema='strict-native-packet-field-fixture',source_video_sha256='fixture',time_base='1/15360',rows=[row]))
    bound=dict(schema='fixture-native-ice-factor',packet_path=str(packet),packet_sha256=replay._sha(packet),
        new_fit_observation_ids=[row['id']],held_observation_ids=[],new_scalar_rows=2,parameter_count=3,
        observation_operator='fixture exact linear source value/J',group_sqrt_N_multiplier_applied=False)
    manifest=dict(full_parameters=3,shared_indices=[0],complete_parameter_layout={'count':3},geometry_world_indices=[0],
        current_geometry={'profile_source_token':token,'physical_geometry':'fixture'},source_rows=4,
        source_contract={'schema':extension.ICE_SCHEMA,'original_source_contract':{'schema':'fixture-base'},
            'additional_real_ice_curve_contract':bound,'additional_scalar_rows':2,'total_rows':4},
        additional_real_native_ice_curves=bound,additional_real_native_board_edges=None,
        latest9a8c_reference18_preserved={'schema':'fixture'},user_trace_sha256='fixture',native_backend_requested={'RINKSPACE_MOTION_BACKEND':'native'},
        camera_mount_working_constraints={'delta_lower_m':[0.],'delta_upper_m':[1.]},camera_pivot_exterior_constraints={'outside_margin_positive':True},
        original_source_role_retirement=None,static_motion_source_role_retirement=None,original_motion_far_board_role_correction=None,
        original_motion_input_compaction=None,original_ring_source_qualification=None,original_ice_pixel_correction=None)
    inputs={'source_uv_a':np.array([[1.,2.]]),'source_uv_b':np.array([[3.,4.]]),'source_sigma':np.array([4.]),'finite_mask':np.array([False])}
    records={'clock':{229618944:0},'source_row_ids':['fixture-parent']}
    def identity(m):return replay.build_persistent_source_replay_identity(m,fixed_parameter_indices=fixed,
        shared_parameter_mask=shared,source_input_arrays=inputs,source_input_records=records)
    manifest[replay.IDENTITY_KEY]=identity(manifest);write(output/'input_manifest.json',manifest)
    blocks=[SourceRowBlock(np.array([.1,.2]),csr_matrix(np.array([[1.,2.,0.],[0.,1.,1.]])),source.quadratic[:2],source.scales[:2],
        {'family':'programme:fixture','original_start':0,'original_stop':2}),
        SourceRowBlock(np.array([.3,.4]),csr_matrix(np.array([[2.,1.,0.],[1.,0.,1.]])),source.quadratic[2:],source.scales[2:],
        {'family':'additional_real_native_ice_curves','original_start':2,'original_stop':4,'packet_sha256':bound['packet_sha256'],
         'fit_observation_ids':[row['id']],'held_observation_ids':[],'native_pts':[229618944],'actual_pts':[229618944],'feature_ids':['goal_line_right']})]
    normal=accumulate_source_normal(lambda:iter(blocks),len(q),frozen_profile_token=token,fixed_parameter_indices=fixed,
        expected_row_count=4,retain_native_replay=True,source_state=q)
    np.savez_compressed(output/'actual_initial_frozen_profile_arrays.npz',**profiles.arrays)
    np.savez_compressed(output/'actual_initial_normal_state.npz',q=q,frozen_profile_token=token,source_manifest_sha256=replay._sha(output/'input_manifest.json'))
    prefix=output/'initial_normal_source_blocks';prefix.mkdir()
    for i,b in enumerate(blocks):
        np.savez(prefix/(f'{i:03d}.npz'),r=b.r,J_data=b.J.data,J_indices=b.J.indices,J_indptr=b.J.indptr,J_shape=b.J.shape,
            quadratic=b.quadratic,scales=b.scales,ledger_json=json.dumps(b.row_ledger),q_sha256=extension._qtoken(q),frozen_profile_token=token)
    write(prefix/'completed_prefix.json',dict(source_manifest_sha256=replay._sha(output/'input_manifest.json'),q_sha256=extension._qtoken(q),
        frozen_profile_token=token,completed_blocks=[b.row_ledger for b in blocks],completed_original_rows=4))
    np.savez_compressed(output/'first_complete_source_normal_numeric.npz',q=q,normal_data=normal.normal.data,normal_indices=normal.normal.indices,
        normal_indptr=normal.normal.indptr,normal_shape=normal.normal.shape,gradient=normal.gradient,cost=normal.cost,frozen_profile_token=token,
        source_contract_json=json.dumps(asdict(normal.contract)),report_json=json.dumps(normal.report))
    replay.seal_complete_same_state_cache(output,q,source,manifest,token,fixed_parameter_indices=fixed,shared_parameter_mask=shared)
    restored=replay.load_complete_same_state_normal(output,q,source,manifest,token,fixed_parameter_indices=fixed,shared_parameter_mask=shared)
    assert (restored.normal!=normal.normal).nnz==0 and np.array_equal(restored.gradient,normal.gradient)and restored.cost==normal.cost
    assert all(np.array_equal(a.r,b.r)and(a.J!=b.J).nnz==0 for a,b in zip(blocks,restored.native_replay_cache.blocks))
    failures=[]
    def denied(label,fn):
        try:fn()
        except (OSError,KeyError,TypeError,ValueError):failures.append(label);return
        raise AssertionError('Unsafe negative accepted: '+label)
    def load(m=manifest,value=q,f=fixed,s=shared):return replay.load_complete_same_state_normal(output,value,source,m,token,fixed_parameter_indices=f,shared_parameter_mask=s)
    denied('changed_q',lambda:load(value=q+np.array([0.,1e-12,0.])))
    denied('changed_fixed_indices',lambda:load(f=[1]))
    denied('changed_shared_mask',lambda:load(s=np.array([False,True,False])))
    for key,new in [('complete_parameter_layout',{'count':3,'permutation':[1,0,2]}),('camera_mount_working_constraints',{'delta_upper_m':[2.]}),
        ('camera_pivot_exterior_constraints',None),('original_motion_far_board_role_correction',{'changed':True}),
        ('original_ice_pixel_correction',{'changed':True}),('original_motion_input_compaction',{'changed':True}),
        ('native_backend_requested',{'RINKSPACE_MOTION_BACKEND':'numpy'})]:
        bad=deepcopy(manifest);bad[key]=new;bad[replay.IDENTITY_KEY]=identity(bad);denied(key,lambda m=bad:load(m))
    old=profiles.arrays['fixture_profile_00'].copy();profiles.arrays['fixture_profile_00'][0]+=1e-12
    denied('changed_actual_profile_array',load);profiles.arrays['fixture_profile_00'][:]=old
    source.scales[0]+=1e-12;denied('changed_sigma_scale',load);source.scales[0]-=1e-12;source.scales[0]=2.
    source.quadratic[0]=False;denied('changed_quadratic_mask',load);source.quadratic[0]=True
    bad=deepcopy(manifest);bad[replay.IDENTITY_KEY]['implementation_files'][0]['sha256']='0'*64;denied('changed_code_hash',lambda:load(bad))
    inputs['source_uv_a'][0,0]+=1.;bad=deepcopy(manifest);bad[replay.IDENTITY_KEY]=identity(bad);denied('changed_actual_source_UV',lambda:load(bad));inputs['source_uv_a'][0,0]-=1.
    fresh=deepcopy(row);fresh['id']='actual-source-field-fixture-append';fresh['uv']=[[255.47005308204731,679.6286012663338]]
    npacket=output/'fixture_append_ice.json';write(npacket,dict(schema='strict-native-packet-field-fixture',source_video_sha256='fixture',time_base='1/15360',rows=[row,fresh]))
    newbound=dict(bound,packet_path=str(npacket),packet_sha256=replay._sha(npacket),new_fit_observation_ids=[row['id'],fresh['id']],new_scalar_rows=4)
    newer=deepcopy(manifest);newer['additional_real_native_ice_curves']=newbound;newer['source_rows']=6
    newer['source_contract'].update(additional_real_ice_curve_contract=newbound,additional_scalar_rows=4,total_rows=6)
    newer[replay.IDENTITY_KEY]=identity(newer)
    plan=extension._native_append_plan(manifest,newer,('additional_real_native_ice_curves',))
    replay._check_strict_identity(manifest,newer,append_normalized_pair=extension._append_normalized_manifests(manifest,newer,plan),fixed_parameter_indices=fixed,shared_parameter_mask=shared)
    calls=[]
    class AppendedFactor:
        parameter_count=3
        def __init__(self):
            self.rows=[row,fresh];self.held=[];self.scales=np.full(4,4.);self.quadratic=np.ones(4,bool)
            self.row_count=4;self.blocks={229618944:None};self.contract=newbound;self._cache=OrderedDict()
        def block(self,value,jacobian=True,*,original_start=0):
            calls.append([r['id']for r in self.rows])
            assert [r['id']for r in self.rows]==[fresh['id']],'Parent factor must never be recomputed'
            matrix=csr_matrix(np.array([[1.,1.,0.],[0.,2.,-1.]]))
            return SourceRowBlock(matrix@value,matrix,self.quadratic,self.scales,
                {'family':'additional_real_native_ice_curves','original_start':original_start,'original_stop':original_start+2})
    appended_source=SimpleNamespace(updater=profiles,scales=np.r_[source.scales,[4.,4.]],quadratic=np.r_[source.quadratic,[True,True]],native_ice_factor=AppendedFactor())
    factory,append_report=extension.prepare_same_state_source_extension_blocks(output,q,appended_source,newer,token,
        fixed_parameter_indices=fixed,shared_parameter_mask=shared,permitted_families=('additional_real_native_ice_curves',))
    replayed=list(factory());assert calls==[[fresh['id']]]
    assert np.array_equal(replayed[0].r,blocks[0].r)and(replayed[0].J!=blocks[0].J).nnz==0
    assert np.array_equal(replayed[1].r[:2],blocks[1].r)and(replayed[1].J[:2]!=blocks[1].J).nnz==0
    assert replayed[1].row_ledger['original_stop']==6 and replayed[1].row_ledger['fit_observation_ids']==[row['id'],fresh['id']]
    tampered=deepcopy(row);tampered['sigma_px']=[5.];write(npacket,dict(schema='strict-native-packet-field-fixture',source_video_sha256='fixture',time_base='1/15360',rows=[tampered,fresh]))
    bad=deepcopy(newer);bad['additional_real_native_ice_curves']['packet_sha256']=replay._sha(npacket)
    denied('append_original_sigma_changed',lambda:extension._native_append_plan(manifest,bad,('additional_real_native_ice_curves',)))
    report=dict(schema='rinkspace.small-existing-cache-strict-replay-verifier.v1',passed=True,scope='3 parameters / 4 actual fixture CSR source scalars / 18 profile arrays; no full programme graph/J or historical-cache recovery claim',
        exact_restored_H_g_cost_r_J=True,append_exact_parent_prefix_field_gate_passed=True,
        append_only_new_FIT_rows_evaluated=True,retained_parent_r_J_exact=True,negative_cases_rejected=failures,
        current_programme_large_cache_restoration_not_tested=True,legacy_104825_123200_cache_files_absent=True)
    write(output/'verifier_result.json',report);print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args();main(args.output)
