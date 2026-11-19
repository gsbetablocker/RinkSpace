"""Export reviewed full-source repair without replacing the rejected sparse state."""
from argparse import ArgumentParser
import numpy as np
from shared import (EVENT,LIBRARY,OUT,PREPARED,PROGRAMMES,REGISTRATIONS,arrays,
    numeric_sha,prepared_packet_path,read,registration,save,sha,write,programme_state_sha)
from review import camera_state,native_for,sample
from solve import stats
from deliver import file_ref,geometry,camera_export
from pathlib import Path

VERSION='joint_dense_1118'
ORIGINAL_EVALUATION=OUT/'original/repair_full_v7'


def compare(candidate):
    current=arrays(OUT/candidate/'full_qa/curve_errors.npz')
    original_binding=read(ORIGINAL_EVALUATION/'metrics_binding.json')
    original=arrays(original_binding.get('errors_path',str(ORIGINAL_EVALUATION/'curve_errors.npz')))
    setup=read(PREPARED/'setup.json');programmes={};shots={};coverage={}
    for programme in PROGRAMMES:
        before=[];after=[];total=active=0
        for name,s in setup['shots'].items():
            if s['program']!=programme:continue
            a=arrays(prepared_packet_path(name));use=a['fold']<5
            x=original[name+'_error'];y=current[name+'_error']
            assert len(x)==len(y)==len(a['pts'])
            before.append(x[use]);after.append(y[use]);total+=len(use);active+=int(use.sum())
            shots[name]=dict(original=stats(x[use]),candidate=stats(y[use]),
                active_source_rows=int(use.sum()),source_exposures=len(np.unique(a['pts'][use])))
        programmes[programme]=dict(original=stats(np.concatenate(before)),candidate=stats(np.concatenate(after)))
        coverage[programme]=dict(retained_source_rows=total,active_source_rows=active)
    return dict(programmes=programmes,shots=shots,source_coverage=coverage,
        prepared=str(PREPARED),metric='original source pixels; finite native curve distance with exact fixed material stations',
        role='same full admitted development rows; previous checks reused during repair; not independent final accuracy')


def export(candidate):
    root=OUT/candidate;setup=read(PREPARED/'setup.json');base=camera_state(candidate)
    native=native_for(candidate,'nayeon');values=native.parameter_state.values
    digest=numeric_sha(values);revision=native.parameter_state.layout['layout_revision']
    visual=read(root/'visual_review.json')
    assert visual['state_sha256']==sha(root/'state.npz')
    assert read(root/'repair_motion/binding.json')['state_sha256']==visual['state_sha256']
    if visual.get('blocking_regressions'):raise ValueError('Unresolved blocking source regressions')
    for directory in [root/'full_qa',ORIGINAL_EVALUATION]:
        binding=read(directory/'metrics_binding.json')
        assert set(binding['packets'])==set(setup['shots'])
        for name,value in binding['packets'].items():assert value==sha(prepared_packet_path(name))
        if directory==root/'full_qa':assert binding['state_sha256']==sha(root/'state.npz')
        if 'errors_path' in binding:assert sha(binding['errors_path'])==binding['errors_sha256']
    reference=sample(native)
    for programme in PROGRAMMES:
        other=sample(native_for(candidate,programme))
        assert [c['id'] for c in other['cuts']]==[c['id'] for c in reference['cuts']]
        np.testing.assert_array_equal(other['xyz'],reference['xyz'])
    report=compare(candidate);write(root/'comparison.json',report)
    snapshot=EVENT/'scene/field'/f'{VERSION}.npz'
    save(snapshot,scene_values=values,scene_values_sha256=digest,layout_revision=revision)
    coordinates=EVENT/'scene'/f'{VERSION}_coordinates.json';write(coordinates,geometry(native))
    binding=dict(version_id=VERSION,status='full_source_repair_candidate',snapshot=file_ref(snapshot),
        structure=file_ref(native.scene_path),native_library=file_ref(LIBRARY),value_sha256=digest,
        layout_revision=revision,scene_parameters_active=False,coordinate_scale='rink length = 1; not verified metres')
    programmes={};mounts={};board=np.asarray(native.scene['board'])
    render_sources=read(root/'render_sources.json') if (root/'render_sources.json').exists() else {}
    for programme,directory in PROGRAMMES.items():
        names=[name for name,s in setup['shots'].items() if s['program']==programme]
        folder=Path(render_sources.get(programme,str(root/'review'/programme)))
        render=read(folder/'binding.json');coverage=read(folder/'coverage.json')
        assert render['state_sha256']==sha(render['state'])
        assert render['programme_state_sha256']==programme_state_sha(base,names)
        assert render['scene_sha256']==sha(native.scene_path)
        assert coverage['validation']['decoded_frames']==coverage['frames']==sum(len(base[n+'_pts']) for n in names)
        for name in names:
            assert render['observation_packets'][name]==sha(prepared_packet_path(name))
            rig=base[name+'_rigs'];pivot=rig[:,15:18]
            u=np.maximum(0.,(np.abs(pivot[:,0])-(board[0]-board[2]))/board[2])
            v=np.maximum(0.,(np.abs(pivot[:,1])-(board[1]-board[3]))/board[3])
            margin=u**board[4]+v**board[4]-1
            valid=(margin>=-1.e-10)&(pivot[:,2]>0)&(np.linalg.norm(rig[:,18:21],axis=1)>1.e-12)
            valid&=(pivot[:,0]<0) if setup['shots'][name]['camera_index']==2 else (pivot[:,1]<=-board[1]+1.e-10)
            if not valid.all():raise ValueError((name,'invalid physical camera installation'))
            mounts[name]=dict(exposures=len(rig),invalid=0,minimum_board_margin=float(margin.min()),minimum_height=float(pivot[:,2].min()))
        camera=camera_export(programme,base,digest,revision,version=VERSION,prepared=PREPARED)
        record=dict(schema='rinkspace.programme_joint_refinement.v1',event_id='ankara_2026',programme_id=directory,
            version_id=VERSION,status='full_source_repair_candidate',user_visual_acceptance=False,
            previous=file_ref(REGISTRATIONS[programme]),source=registration(programme)['source'],
            geometry_binding=binding,camera=file_ref(camera),shots=names,
            fit_observations={name:file_ref(prepared_packet_path(name)) for name in names},
            fit_roles=[0,1,2,3,4],role_meaning='previously reused development observations; full original temporal coverage',
            full_review=dict(video=file_ref(coverage['video']),coverage=file_ref(folder/'coverage.json'),
                render_state=file_ref(render['state']),selected_geometry_and_programme_camera_exactly_equal=True,
                source_seconds=coverage['source_seconds'],frames=coverage['frames'],native_time_base=[1,12800],fps=50,
                source_audio=coverage['source_audio']),comparison=report['programmes'][programme],
            source_motion=file_ref(root/'repair_motion/summary.json'),visual_review=file_ref(root/'visual_review.json'),
            exclusions='Nayeon S01 remains outside assigned-camera scope; every assigned shot retained')
        target=EVENT/'programmes'/directory/f'{VERSION}.json';write(target,record);programmes[programme]=file_ref(target)
    write(root/'physical_mounts.json',mounts)
    target=EVENT/'scene'/f'{VERSION}.json'
    write(target,dict(schema='rinkspace.event_joint_refinement.v1',event_id='ankara_2026',version_id=VERSION,
        status='full_source_repair_candidate',user_visual_acceptance=False,geometry_binding=binding,
        coordinates=file_ref(coordinates),programmes=programmes,selected_run=candidate,checkpoint=file_ref(root/'state.npz'),
        comparison=file_ref(root/'comparison.json'),prepared=file_ref(PREPARED/'setup.json'),
        source_motion=file_ref(root/'repair_motion/summary.json'),visual_review=file_ref(root/'visual_review.json'),
        physical_mounts=file_ref(root/'physical_mounts.json'),limitations=visual.get('limitations',[]),
        rejected_sparse_version='joint_free_1117: source thinning and severe Okada temporal regression',
        user_request='Improve all four programmes while preserving earlier dense source points and local repairs'))
    print('EXPORTED',target,flush=True)


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__);parser.add_argument('--candidate',required=True)
    export(parser.parse_args().candidate)
