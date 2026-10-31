"""A retry reuses source results and stored searches without native projection."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

PROGRAMME=Path(__file__).resolve().parents[1]/'events/ankara_2026/programmes/okada_mei_sp'
sys.path.insert(0,str(PROGRAMME))
import capture_native_print_patterns as capture


def note(index,status='source_pattern_registered',accepted=1):
    H=np.eye(3);H[:2,2]=[index*7,index*8]
    return dict(atlas_index=index,instance='Ad7',initial_H=H.tolist(),
        registration=dict(status=status,correlation=.8,H=H.tolist()),source_readout=dict(accepted=accepted))


@pytest.mark.parametrize('status,accepted,expected',[
    ('source_pattern_registered',1,False),('source_pattern_registered',0,True),('registration_failed',0,True)])
def test_only_unresolved_or_empty_original_patterns_may_need_registration_recovery(status,accepted,expected):
    assert capture.needs_retry(note(0,status,accepted)) is expected


def test_original_paths_keep_unthinned_rows_and_selected_uv_when_retry_competes():
    row=dict(path_id='A',uv=[10.,20.],source_registration_label='0/Ad7')
    extra=dict(row,uv=[11.,21.])
    original=dict(patterns=[note(0)],all_actual_readouts=[row,extra])
    incoming=dict(label='1/Ad7',origin='retry',registration=dict(correlation=.99,status='source_pattern_registered'),
        rows=[dict(path_id='A',uv=[90.,90.]),dict(path_id='B',uv=[30.,40.])])
    retained,unthinned,choices=capture.choose_readouts([incoming],dict(rows=[row]),original)
    assert retained[0]==row and unthinned[:2]==[row,extra]
    assert not any(r['uv']==[90.,90.]for r in unthinned)
    assert choices['A']['origin']=='original_capture' and choices['B']['origin']=='retry'


@pytest.fixture
def retry_run(tmp_path,monkeypatch):
    prepared=tmp_path/'prepared';prepared.mkdir()
    (prepared/'mapping.json').write_text('{}',encoding='utf8')
    np.savez(prepared/'point_observations.npz',test_only=np.array([1]))
    scene=tmp_path/'scene.json';scene.write_text('{}',encoding='utf8')
    library=tmp_path/'native.dll';library.write_bytes(b'fixture: no native library is loaded')
    shot=dict(id='s08',first_pts=1000,last_pts=1256,stop_pts_exclusive=1512,camera_index=0)
    manifest=dict(shots=[shot],native_library=str(library),scene_path=str(scene),scene_file_sha256=capture.sha(scene),
        point_input_sha256=capture.sha(prepared/'point_observations.npz'))
    capture.write(prepared/'joint_manifest.json',manifest)
    state=tmp_path/'state.npz'
    np.savez(state,native_library_provenance_sha256=np.array(capture.sha(library)),
        point_input_sha256=np.array(manifest['point_input_sha256']))
    reference=tmp_path/'atlas.png';cv2.imwrite(str(reference),np.zeros((20,20,3),np.uint8))
    atlases=tmp_path/'atlases.json'
    docs=[dict(ad=7,image=str(reference),image_sha256=capture.sha(reference),
        paths=[dict(path_id='Ad7.'+name,pixels=[[1,1],[10,1],[10,10],[1,1]])])for name in ('A','B')]
    capture.write(atlases,dict(atlases=docs))
    original=tmp_path/'original';(original/'s08').mkdir(parents=True)
    sources=[];registrations=[];frames=[]
    for pts in (1000,1256):
        row=dict(path_id='print_instance/Ad7::Ad7.A',uv=[10.,20.],source_registration_label='0/Ad7')
        source=dict(pts=pts,source_BGR_sha256='b'*64,source_video_sha256='a'*64,
            source_exclusion_sha256='c'*64,mask_sha256='d'*64,rows=[row],fit_eligible=False)
        sources.append(source)
        registrations.append(dict(pts=pts,patterns=[note(0),note(1,'registration_failed',0)],all_actual_readouts=[row]))
        frames.append(dict(source,image=np.zeros((100,100,3),np.uint8),
            source_exclusion_mask=np.zeros((100,100),bool),mask_path=tmp_path/'mask.png',mask_sources=[]))
    for name,records in [('source_rows',sources),('registration',registrations)]:
        (original/'s08'/(name+'.jsonl')).write_text(''.join(__import__('json').dumps(r)+'\n'for r in records),encoding='utf8')
    def actual_frames(sid,*,bootstrap_root,selected_pts):
        assert sid=='s08' and selected_pts==[1256]
        yield copy.deepcopy(frames[1])
    monkeypatch.setattr(capture,'iter_registration_source_frames',actual_frames)
    bindings=[dict(atlas_index=i,instance='Ad7',path_ids=['print_instance/Ad7::Ad7.'+name],
        reference_image=str(reference),reference_image_sha256=capture.sha(reference),ink_polarity='light',
        excluded_partial_reference_paths=[],reference_photometric_transform=None)for i,name in enumerate(('A','B'))]
    binding=dict(schema='rinkspace.current_native_whole_print_RGB_capture.v1',prepared=str(prepared),state=str(state),
        state_sha256=capture.sha(state),native_library=str(library),native_library_sha256=capture.sha(library),
        source_atlases=str(atlases),source_atlases_sha256=capture.sha(atlases),atlas_ids=[0,1],patterns=bindings,
        source_adapter_sha256=capture.sha(actual_frames.__code__.co_filename),reader_sha256='e'*64,
        query_spacing_px=8.,retained_arc_spacing_px=48.,shots=['s08'])
    capture.write(original/'binding.json',binding)
    def forbidden(*args,**kwargs):pytest.fail('Retry initialized/projected a native camera or scene')
    for name in ('NativeCore','SceneParameterModel','ShotMotionTimeline','sample_native_review_scene'):
        monkeypatch.setattr(capture,name,forbidden)
    args=SimpleNamespace(prepared=prepared,state=state,bootstrap=tmp_path/'bootstrap',atlases=atlases,
        output=tmp_path/'recovered',atlas_ids=[0,1],shots=['s08'],pts=[1256],query_spacing=8.,
        include_paired=False,coarse_navigation=True,retry_capture=original)
    return args,original,sources,registrations,frames


@pytest.mark.parametrize('recover',[True,False])
def test_retry_uses_stored_H_keeps_full_ledger_and_preserves_successful_paths(retry_run,monkeypatch,recover):
    args,original,sources,registrations,_=retry_run
    calls=[];reads=[]
    class Pattern:
        def __init__(self,atlas,image):self.path=atlas['paths'][0]['path_id']
        def read_paths(self,image,exclusion,registration,binding,spacing):
            reads.append(self.path)
            if self.path=='Ad7.A':return [],dict(accepted=0,reason='no_new_resolved_material')
            if not recover:return [],dict(accepted=0,reason='no_resolved_visible_glyphs')
            return [dict(path_id='print_instance/Ad7::Ad7.B',uv=[33.,44.])],dict(accepted=1)
    def coarse(pattern,image,exclusion,H):
        calls.append(H.copy())
        return dict(status='source_pattern_registered'if recover else'registration_failed',correlation=.97,H=H.tolist())
    monkeypatch.setattr(capture,'PrintPattern',Pattern)
    monkeypatch.setattr(capture,'coarse_print_registration',coarse)
    before={p:capture.sha(p)for p in original.rglob('*')if p.is_file()}
    capture.run(args)
    assert len(calls)==1
    assert reads==['Ad7.A','Ad7.B']
    np.testing.assert_array_equal(calls[0],registrations[1]['patterns'][1]['initial_H'])
    import json
    output=[json.loads(line)for line in (args.output/'s08/source_rows.jsonl').read_text().splitlines()]
    assert [r['pts']for r in output]==[1000,1256]
    assert output[0]==sources[0] and output[1]['rows'][0]==sources[1]['rows'][0]
    assert len(output[1]['rows'])==(2 if recover else 1)
    registrations_out=[json.loads(line)for line in (args.output/'s08/registration.jsonl').read_text().splitlines()]
    assert registrations_out[1]['patterns'][0]['registration']==registrations[1]['patterns'][0]['registration']
    assert registrations_out[1]['patterns'][0]['original_pattern_result']==registrations[1]['patterns'][0]
    assert registrations_out[1]['patterns'][0]['navigation_attempt_performed'] is False
    assert registrations_out[1]['patterns'][1]['original_pattern_result']==registrations[1]['patterns'][1]
    assert registrations_out[1]['all_actual_readouts'][0]==registrations[1]['all_actual_readouts'][0]
    result_binding=capture.read(args.output/'binding.json')
    assert result_binding['retry_capture']['binding']['sha256']==before[original/'binding.json']
    assert result_binding['coarse_navigation_helper']['sha256']==capture.sha(__file__)
    assert all(capture.sha(p)==value for p,value in before.items())


def test_original_source_hash_mismatch_is_rejected(retry_run):
    _,_,sources,_,frames=retry_run
    frames[1]['source_BGR_sha256']='f'*64
    with pytest.raises(ValueError,match='source_BGR_sha256'):
        capture.validate_retry_frame(frames[1],sources[1])


def test_registered_zero_readout_retries_material_gate_before_any_ecc(retry_run,monkeypatch):
    args,_,_,_,_=retry_run
    class Pattern:
        def read_paths(self,image,exclusion,registration,binding,spacing):
            assert registration==original_registration
            return [dict(path_id='print_instance/Ad7::Ad7.B',uv=[33.,44.])],dict(accepted=1)
        def register(self,*args,**kwargs):pytest.fail('Repeated local ECC despite recovered actual material')
    def no_coarse(*args,**kwargs):pytest.fail('Ran coarse navigation despite recovered actual material')
    monkeypatch.setattr(capture,'coarse_print_registration',no_coarse)
    original_registration=note(1)['registration']
    item=dict(atlas_index=1,instance='Ad7',pattern=Pattern())
    result,candidate=capture.read_pattern_attempt(item,np.zeros((20,20,3),np.uint8),
        np.zeros((20,20),np.uint8),np.eye(3),{},args,'retry',original_registration)
    assert result['registration_reused_for_material_retry'] is True
    assert result['navigation_attempt_performed'] is False
    assert result['registration']==original_registration
    assert candidate['rows'][0]['uv']==[33.,44.]


def test_registered_partial_readout_fills_missing_path_without_replacing_original_or_ecc(retry_run,monkeypatch):
    args,_,_,_,_=retry_run
    old=dict(path_id='print_instance/Ad7::Ad7.A',uv=[10.,20.],source_registration_label='1/Ad7')
    prior=note(1)
    class Pattern:
        def read_paths(self,*args,**kwargs):
            return [dict(path_id=old['path_id'],uv=[90.,90.]),
                dict(path_id='print_instance/Ad7::Ad7.B',uv=[33.,44.])],dict(accepted=2)
        def register(self,*args,**kwargs):pytest.fail('Repeated ECC for original registered partial reading')
    monkeypatch.setattr(capture,'coarse_print_registration',lambda *a,**kw:pytest.fail('Unnecessary coarse navigation'))
    item=dict(atlas_index=1,instance='Ad7',pattern=Pattern())
    result,candidate=capture.read_pattern_attempt(item,np.zeros((20,20,3),np.uint8),
        np.zeros((20,20),np.uint8),np.eye(3),{},args,'retry',prior['registration'],True)
    kept,actual,choices=capture.choose_readouts([candidate],dict(rows=[old]),
        dict(patterns=[prior],all_actual_readouts=[old]))
    assert kept[0]==old and actual[0]==old
    assert [row['uv']for row in kept]==[[10.,20.],[33.,44.]]
    assert result['navigation_attempt_performed'] is False
    assert choices['print_instance/Ad7::Ad7.B']['origin']=='retry'


def test_original_state_hash_mismatch_is_rejected_before_output(retry_run,monkeypatch):
    args,original,_,_,_=retry_run
    binding=capture.read(original/'binding.json');binding['state_sha256']='f'*64
    capture.write(original/'binding.json',binding)
    class Pattern:
        def __init__(self,atlas,image):pass
        def read_paths(self,*args,**kwargs):pytest.fail('Read before original binding validation')
    monkeypatch.setattr(capture,'PrintPattern',Pattern)
    with pytest.raises(ValueError,match='state_sha256'):capture.run(args)
    assert not args.output.exists()
