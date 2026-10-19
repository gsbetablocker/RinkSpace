"""Verify new source primitives on an immutable actual state and actual UV.

The small physical-camera adapter uses the archived calibration and the same
global pivot/mount/lens/RF chart. No full motion graph is loaded for this check.
"""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
import argparse
import json
import os
import sys

for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import numpy as np
from rinkspace.vision_camera_initialization import current_complete_source_geometry
from calibration.program_tracking.z190_physical_camera import SharedCalibration,ViewState


def state_camera_adapter(arrays,cameras,provider):
    from z190_fixed_lever_xyz_source_bundle import FixedLeverXYZSourceBundle
    from structured_programme_source_candidate.paired_programme_chart import PairedProgrammeCoordinateChart,CompleteProgrammeParameterLayout
    reference=arrays['q'].copy()
    layout=None
    if 'aux__current_complete_parameter_layout_json'in arrays:
        saved=json.loads(str(arrays['aux__current_complete_parameter_layout_json'].item()))
        layout=CompleteProgrammeParameterLayout(int(saved['original_complete_parameter_count']),int(saved['additional_world_parameter_count']))
        reference=layout.evaluation_q(reference)
    chart=PairedProgrammeCoordinateChart(provider.parameter_count-66)
    metadata=json.loads(str(cameras['metadata_json']))
    canonical=arrays['aux__canonical_pts']
    lengths=arrays['aux__block_lengths']//4
    cumulative=np.r_[0,np.cumsum(lengths)]
    pivots=arrays['aux__installation_xyz_slots']
    mounts=arrays['aux__current_shared_mount_parameter_indices'].reshape(3,3)
    optics=arrays['aux__current_all_installation_optics_parameter_indices'].reshape(3,3)
    calibrations={k:SharedCalibration.from_manifest(v)for k,v in metadata['calibrations'].items()}
    installation_order=json.loads(str(arrays['aux__correlated_roll_prior_contract_json']))['installation_order']
    aliases=dict(zip(map(int,cameras['pts']),map(int,cameras['exposure_representative_pts'])))
    camera_by_actual={int(p):i for i,p in enumerate(cameras['exposure_representative_pts'])}
    blocks=[]
    for bi,(start,end)in enumerate(zip(cumulative[:-1],cumulative[1:])):
        stamps=canonical[start:end]
        identity=str(cameras['installation_id'][camera_by_actual[int(stamps[0])]])
        ii=installation_order.index(identity)
        index={int(p):i for i,p in enumerate(stamps)}
        begin=int(arrays['aux__block_start'][bi])
        calibration=calibrations[identity]
        def camera(q,p,*,ii=ii,begin=begin,index=index,calibration=calibration):
            q=np.asarray(q);dx=q[pivots[ii]]-reference[pivots[ii]]
            dm=q[mounts[ii]]-reference[mounts[ii]]
            do=q[optics[ii]]-reference[optics[ii]]
            cal=replace(calibration,pivot_xyz=tuple(np.asarray(calibration.pivot_xyz)+dx/provider.scene_unit_m),
                lever_xyz=tuple(np.asarray(calibration.lever_xyz)+dm/provider.scene_unit_m),
                principal_x_fraction=(calibration.principal_x_fraction[0]+do[0]/1920.,),
                principal_y_fraction=(calibration.principal_y_fraction[0]+do[1]/1080.,),
                k1=(calibration.k1[0]+.001*do[2],))
            v=q[begin+4*index[int(p)]:begin+4*index[int(p)]+4]
            return cal.camera25(ViewState(tuple(v[:3]),float(v[3]-np.log(1920.))))
        bundle=FixedLeverXYZSourceBundle.__new__(FixedLeverXYZSourceBundle)
        bundle.index=index;bundle.stamps=stamps;bundle.calibration=calibration;bundle.camera=camera
        block=SimpleNamespace(name=str(arrays['aux__block_names'][bi]),start=begin,
            xyz_slots=pivots[ii],local=lambda q:q,
            optics_chart=SimpleNamespace(indices=optics[ii],mount_indices=mounts[ii]),
            bundle=bundle)
        blocks.append(block)
    bypts={int(p):b for b in blocks for p in b.bundle.stamps}
    maximum=0.
    for i,p in enumerate(cameras['pts']):
        at=aliases[int(p)]
        maximum=max(maximum,float(abs(bypts[at].bundle.camera(reference,at)-cameras['camera25'][i]).max()))
    assert maximum<1e-11,maximum
    return SimpleNamespace(geometry=provider,original=SimpleNamespace(aliases=aliases),bypts=bypts,blocks=blocks,
        structured_programme_coordinate_chart=chart,geometry_parameter_indices=np.asarray(chart.world_indices,int),
        canonical_parameter_layout=layout,evaluation_q=(lambda q:np.asarray(q,float))if layout is None else layout.evaluation_q),maximum


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--state',required=True,type=Path)
    ap.add_argument('--auxiliary',required=True,type=Path)
    ap.add_argument('--ice-packet',required=True,type=Path)
    ap.add_argument('--board-packet',type=Path)
    ap.add_argument('--output',type=Path)
    args=ap.parse_args()
    arrays=dict(np.load(args.state,allow_pickle=False))
    identity=json.loads((args.state.parent/'state_identity.json').read_text(encoding='utf8'))
    cameras=dict(np.load(identity['camera_path'],allow_pickle=False))
    provider,world,proof=current_complete_source_geometry(args.state,args.auxiliary)
    state,camera_error=state_camera_adapter(arrays,cameras,provider)
    q=state.evaluation_q(arrays['q']).copy()
    from structured_programme_source_candidate.native_ice_curve_source import NativeIceCurveFactor
    ice=NativeIceCurveFactor(state,args.ice_packet,len(q))
    block=ice.block(q,True)
    direction=np.zeros(len(q));direction[np.unique(block.J.indices)]=np.random.default_rng(230955776).normal(size=len(np.unique(block.J.indices)))
    # Physical-domain camera/core increments only; scale the mixed direction
    # to a small native-pixel displacement before checking true values.
    direction/=max(float(abs(block.J@direction).max()),1.)
    h=1e-4
    fd=(ice.block(q+h*direction,False).r-ice.block(q-h*direction,False).r)/(2*h)
    err=float(abs(fd-block.J@direction).max())
    assert err<2e-3,(err,float(abs(fd).max()))
    crease=[r for r in ice.rows+ice.held if r['curve']['kind']=='current_goal_crease']
    scene=json.loads((args.state.parent/'scene_UNACCEPTED.json').read_text(encoding='utf8'))
    fs={f['id']:f for f in scene['ice_features']}
    for row in crease:
        original=np.asarray(fs[row['feature_id']]['polygon_xyz']);original=np.vstack((original,original[0]))if not np.array_equal(original[0],original[-1])else original
        assert np.array_equal(original,ice._world(q,row['curve']))
    report=dict(schema='rinkspace.actual-native-source-primitive-verification.v1',complete_state=str(args.state),
        world_proof=proof,physical_camera_archive_reproduce_max=camera_error,
        ice_packet=str(args.ice_packet),ice_FIT_scalar_rows=ice.row_count,ice_J_shape=list(block.J.shape),
        ice_coupled_directional_max_error=err,goal_crease_rows=len(crease),
        goal_crease_world_equals_existing_complete_polygon=True,
        motion_graph_not_loaded_or_modified=True,source_rows_UV_sigma_and_roles_not_modified=True)
    if args.board_packet:
        from z190_shared_board_pattern_source import SharedPerimeterBoardPatternGeometry,SharedBoardPatternSourceFactor
        from shared_board_course_candidate.course_pattern_geometry import CourseBoardPatternGeometry
        from structured_programme_source_candidate.current_full_source import CurrentPatterns
        from structured_programme_source_candidate.native_board_material_point_source import NativeBoardMaterialPointFactor,SourceWithAdditionalNativeBoardMaterialPoints
        from structured_programme_source_candidate.course_patterns_independent_placements import SourceWithIndependentPatternPlacements
        base=SharedPerimeterBoardPatternGeometry(provider.current.base,provider.scene_unit_m,facet_authority=scene['shared_board_patterns']['facet_authority'])
        original=SharedBoardPatternSourceFactor(base)
        owner=SimpleNamespace(start=int(arrays['aux__shared_pattern_start']),state=state,geometry=base,factor=original,external_factor=original,
            _original_course_pattern_geometry=base,_original_course_pattern_factor=original,_original_course_external_pattern_factor=original,
            course_world_geometry=provider,fit=[],held=[],bind_current_geometry=None)
        from types import MethodType
        owner.bind_current_geometry=MethodType(CurrentPatterns.bind_current_geometry,owner)
        state.patterns=owner
        prefix_count=len(q)-len(arrays['aux__current_independent_pattern_placement_parameter_indices'])
        material=NativeBoardMaterialPointFactor(state,args.board_packet,prefix_count)
        empty=SimpleNamespace(count=prefix_count,quadratic=np.empty(0,bool),scales=np.empty(0),contract={},updater=None,
            blocks=lambda q,jacobian:iter(()),guard=lambda q:provider.validate_parameters(q[proof['actual_world_parameter_indices']]))
        base_source=SourceWithAdditionalNativeBoardMaterialPoints(empty,material)
        authority=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/model_constraints/user_advertisement_pairs_20261005.json')
        source=SourceWithIndependentPatternPlacements(base_source,state,authority)
        full=source.extend_initial(q[:prefix_count],suffix=q[prefix_count:])
        assert np.array_equal(full[:len(q)],q)
        nb=next(source.blocks(full,True))
        d=np.zeros(len(full));cols=np.unique(nb.J.indices);d[cols]=np.random.default_rng(26).normal(size=len(cols));d/=max(float(abs(nb.J@d).max()),1.)
        fb=(next(source.blocks(full+1e-4*d,False)).r-next(source.blocks(full-1e-4*d,False)).r)/(2e-4)
        be=float(abs(fb-nb.J@d).max());assert be<2e-3,be
        if source.placement_count==4:
            assert nb.J[:,source.placement_indices[-2:]].nnz>0
        report.update(board_packet=str(args.board_packet),board_source_scalar_rows=len(nb.r),board_J_shape=list(nb.J.shape),
            inherited_complete_prefix_bitexact=True,placement_columns=source.placement_indices.tolist(),
            actual_Ad26_source_present_adds_only_two_columns=source.placement_count==4,board_coupled_directional_max_error=be)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
