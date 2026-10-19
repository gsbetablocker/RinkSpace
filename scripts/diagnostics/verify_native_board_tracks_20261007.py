"""Coupled world and physical-camera FD check using real frozen native tracks."""
from pathlib import Path
import os,sys,json,argparse
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
from copy import deepcopy,copy
from collections import OrderedDict
from scipy.sparse import csr_matrix
from scripts.diagnostics.verify_current_native_primitives_20261007 import state_camera_adapter
from rinkspace.vision_camera_initialization import current_complete_source_geometry

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--state',required=True,type=Path);ap.add_argument('--auxiliary',required=True,type=Path)
    ap.add_argument('--packet',required=True,type=Path);ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--parent-packet',type=Path)
    ap.add_argument('--new-fit-groups-only',action='store_true',help='Verify only newly appended actual source groups; retain exact old source r/scales/order and reuse historical source-J proof')
    a=ap.parse_args()
    arrays=dict(np.load(a.state,allow_pickle=False));identity=json.loads((a.state.parent/'state_identity.json').read_text(encoding='utf8'))
    cameras=dict(np.load(identity['camera_path'],allow_pickle=False))
    provider,world,proof=current_complete_source_geometry(a.state,a.auxiliary)
    state,error=state_camera_adapter(arrays,cameras,provider);q=state.evaluation_q(arrays['q']).copy()
    from structured_programme_source_candidate.native_board_track_source import NativeBoardTrackFactor
    factor=NativeBoardTrackFactor(state,a.packet,len(q));factor.guard(q)
    old_prefix=None
    if a.new_fit_groups_only:
        if a.parent_packet is None:raise ValueError('Explicit immutable source parent required')
        parent=NativeBoardTrackFactor(state,a.parent_packet,len(q));prior=parent.block(q,False);complete=factor.block(q,False)
        assert np.array_equal(prior.r,complete.r[:len(prior.r)])
        assert np.array_equal(parent.scales,factor.scales[:len(prior.r)])
        assert np.array_equal(parent.quadratic,factor.quadratic[:len(prior.r)])
        assert [r['id']for r in factor.fit_evaluation_rows[:len(parent.fit_evaluation_rows)]]==[r['id']for r in parent.fit_evaluation_rows]
        for before,after in zip(parent.fit_evaluation_rows,factor.fit_evaluation_rows):
            assert np.array_equal(before['_columns'],after['_columns'])and before['_steps']==after['_steps']
        old_prefix=dict(original_scalar_rows=len(prior.r),original_actual_residual_scales_quadratic_and_row_order_bitexact=True,
                        old_source_activity_and_derivative_steps_identical=True,old_complete_J_not_rebuilt=True)
        component=copy(factor);component.fit_evaluation_rows=factor.fit_evaluation_rows[len(parent.fit_evaluation_rows):]
        component.scales=factor.scales[len(prior.r):];component.quadratic=factor.quadratic[len(prior.r):]
        component.row_count=len(component.scales);component._cache=OrderedDict();factor_for_fd=component
    else:factor_for_fd=factor
    block=factor_for_fd.block(q,True)
    # Probe the entire actual parameter vector, including declared zero
    # columns, so an accidentally missing dependency cannot pass unnoticed.
    d=np.random.default_rng(229694208).normal(size=len(q))
    if a.new_fit_groups_only:
        active=np.unique(np.concatenate([r['_columns']for r in factor_for_fd.fit_evaluation_rows]));mask=np.zeros(len(q),bool);mask[active]=True;d[~mask]=0.
    d/=max(float(abs(block.J@d).max()),1.)
    h=1e-4
    stencil_report=None
    if a.new_fit_groups_only:
        from structured_programme_source_candidate.fixed_pivot_positive_domain_fd import print_domain_directional_stencil
        def validate(value):
            provider.validate_parameters(value[state.geometry_parameter_indices])
            if np.any(value[factor.height_columns]<=0):raise ValueError('Strict actual positive fixed-pivot height domain')
        stencil=print_domain_directional_stencil(q,d,h,validate);stencil_report=stencil
        nominal=block.r
        values=[factor_for_fd.block(q+offset*d,False).r for offset in stencil['offsets']]
        if len(values)==2:fd=(values[1]-values[0])/(stencil['offsets'][1]-stencil['offsets'][0])
        else:fd=stencil['weights'][1]*(values[1]-nominal)+stencil['weights'][2]*(values[2]-nominal)
    else:fd=(factor_for_fd.block(q+h*d,False).r-factor_for_fd.block(q-h*d,False).r)/(2*h)
    errorJ=float(abs(fd-block.J@d).max());assert errorJ<2e-3,errorJ
    review=factor.review(q)
    report=dict(schema='rinkspace.actual-native-board-track-verification.v1',state=str(a.state),source_packet=str(a.packet),
        physical_camera_archive_max_error=error,J_shape=list(block.J.shape),coupled_world_camera_J_max_error=errorJ,
        source_role_and_raw_UV_preserved=True,review=review,caller_chart='explicit complete canonical -> evaluation',
        canonical_parameter_count=len(arrays['q']),evaluation_parameter_count=len(q),new_source_group_only=a.new_fit_groups_only,
        old_source_prefix_verification=old_prefix,actual_physical_domain_directional_stencil=stencil_report)
    if a.parent_packet and not a.new_fit_groups_only:
        parent=json.loads(a.parent_packet.read_text(encoding='utf8'))
        packet=json.loads(a.packet.read_text(encoding='utf8'))
        assert packet['rows'][:len(parent['rows'])]==parent['rows'],'Original row prefix changed'
        for key in ('schema','source_sha256','time_base','model_projection_used_to_generate_or_select_uv'):
            assert parent[key]==packet[key],key
        original=NativeBoardTrackFactor(state,a.parent_packet,len(q));ob=original.block(q,True)
        assert np.array_equal(ob.r,block.r[:len(ob.r)]),'Original residual changed'
        delta=csr_matrix(block.J[:len(ob.r)]-ob.J)
        assert delta.nnz==0 or np.max(abs(delta.data))==0.,'Original J changed'
        report.update(parent_packet=str(a.parent_packet),original_rows_bitexact=True,
            original_FIT_residual_and_J_bitexact=True,original_FIT_scalar_rows=len(ob.r),
            new_source_scalar_rows=len(block.r)-len(ob.r),new_parameters=0)
    if a.new_fit_groups_only:
        a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        print(json.dumps(dict(J_shape=list(block.J.shape),coupled_J_error=errorJ,old_prefix=old_prefix,report=str(a.output))),flush=True)
        return
    scene=json.loads((a.state.parent/'scene_UNACCEPTED.json').read_text(encoding='utf8'))
    exported=factor.export_print_scene(deepcopy(scene),q)
    if factor.print_contours:
        patterns={r['advertisement_id']:r for r in exported['source_conditioned_board_prints']['patterns']}
        for number,item in patterns.items():
            assert item['anchor_actual_pts']==state.original.aliases[item['anchor_pts']],number
            assert item['contour_source_anchor_actual_pts']==state.original.aliases[item['contour_source_anchor_pts']],number
        for pair in factor.shared_print_templates.values():
            source=patterns[pair['canonical_advertisement_id']];target=patterns[pair['counterpart_advertisement_id']]
            assert target['contour_source_anchor_pts']==pair['canonical_anchor_pts']
            assert target['anchor_pts']==pair['counterpart_anchor_pts']
            assert len(source['external_contours'])==len(target['external_contours'])
            for left,right in zip(source['external_contours'],target['external_contours']):
                expected=factor._place_counterpart(q,np.asarray(left['xyz_scene_units']),pair)
                assert np.array_equal(expected,np.asarray(right['xyz_scene_units']))
        report.update(export_print_ids=sorted(patterns),actual_instance_and_canonical_contour_clocks_distinct=True,
            counterpart_same_developed_print_no_new_scale_or_reflection=True)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(J_shape=list(block.J.shape),coupled_J_error=errorJ,rows=review['rows'])),flush=True)

if __name__=='__main__':main()
