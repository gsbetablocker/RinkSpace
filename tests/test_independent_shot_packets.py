"""Synthetic source-only packet selection; no historical data, GPU or fitting."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

PROGRAMME=Path(__file__).resolve().parents[1]/'events/ankara_2026/programmes/okada_mei_sp'
sys.path.insert(0,str(PROGRAMME))
import build_independent_shot_packets as packets


@pytest.fixture
def bootstrap(tmp_path):
    root=tmp_path/'bootstrap';root.mkdir()
    capture=tmp_path/'capture';capture.mkdir()
    baseline_manifest=tmp_path/'baseline_manifest.json'
    packets.write(baseline_manifest,dict(shots=[dict(id='s01',camera_index=0)]))
    shots=[]
    for i,sid in enumerate(packets.DEFAULT_SHOT_IDS):
        first=1000+i*512
        shots.append(dict(shot_id=sid,known_rig_id='main',processing=dict(
            first_pts=first,last_pts=first+256,stop_pts_exclusive=first+512,native_frame_count=2)))
    scope_path=tmp_path/'scope.json'
    source_hash='a'*64
    clock=dict(time_base_num=1,time_base_den=12800,native_pts_step=256)
    packets.write(scope_path,dict(source=dict(sha256=source_hash),source_clock=clock,
        shots=[dict(id='s01',rig_id='main')]+[
            dict(id=s['shot_id'],rig_id='main',processing=s['processing']) for s in shots]))
    index=dict(source=dict(sha256=source_hash,actual_sha256=source_hash),source_clock=clock,shots=shots)
    packets.write(root/'index.json',index)
    packets.write(root.parent/'baseline.json',dict(source_scope=str(scope_path),
        inputs=dict(manifest=dict(path=str(baseline_manifest),sha256=packets.digest(baseline_manifest)))))
    return root,capture,index


def write_capture(capture,shot,*,eligible=True):
    sid=shot['shot_id'];folder=capture/sid;folder.mkdir()
    pts=packets.native_pts(shot['processing'])
    qualified=dict(pts=int(pts[0]),fit_eligible=eligible,source_BGR_sha256='b'*64,
        rows=[dict(path_id='ice.A',uv=[123.125,456.75],measurement_operator='actual_RGB_boundary')])
    pending=dict(pts=int(pts[1]),fit_eligible=False,reason='source boundary unresolved',rows=[])
    raw=(json.dumps(qualified)+'\n'+json.dumps(pending)+'\n').encode()
    (folder/'source_rows.jsonl').write_bytes(raw)
    np.savez(folder/'poses.npz',pts=pts,poses=np.tile([.01,.02,7.],(len(pts),1)))
    return folder,qualified,(json.dumps(pending)+'\n').encode()


@pytest.mark.parametrize('requested,expected',[
    (None,['s07','s08','s09']),('s08',['s08']),('s09,s07',['s07','s09']),
    (['s08','s09'],['s08','s09']),
])
def test_selection_preserves_bootstrap_order(bootstrap,requested,expected):
    _,_,index=bootstrap
    assert [s['shot_id'] for s in packets.selected_shots(index,requested)]==expected


@pytest.mark.parametrize('requested',['',',','s07,s07',[],['s07','s07']])
def test_empty_or_duplicate_selection_fails(bootstrap,requested):
    with pytest.raises(ValueError,match='nonempty and distinct'):
        packets.selected_shots(bootstrap[2],requested)


def test_missing_selected_bootstrap_shot_is_explicit(bootstrap):
    root,capture,index=bootstrap
    index['shots']=[s for s in index['shots'] if s['shot_id']!='s08']
    packets.write(root/'index.json',index)
    with pytest.raises(ValueError,match='missing from bootstrap index: s08'):
        packets.build(root,capture,capture.parent/'packets')
    assert [s['shot_id'] for s in packets.selected_shots(index,'s07')]==['s07']
    with pytest.raises(ValueError,match='missing from bootstrap index: s10'):
        packets.selected_shots(index,'s10')


def test_one_shot_build_ignores_other_capture_files_and_preserves_source_uv(bootstrap):
    root,capture,index=bootstrap
    selected=index['shots'][1]
    folder,qualified,pending_bytes=write_capture(capture,selected)
    # Unselected metadata is not a dependency of the selected shot's identity.
    index['shots'][0]['processing']={}
    packets.write(root/'index.json',index)
    inputs={p:packets.digest(p) for p in folder.iterdir()}
    output=capture.parent/'packets'
    receipt=packets.build(root,capture,output,shot_ids='s08')
    assert receipt['selected_shot_ids']==['s08']
    assert [s['shot_id'] for s in receipt['shots']]==['s08']
    assert not (capture/'s07').exists() and not (capture/'s09').exists()
    assert {p.name for p in output.iterdir()}=={'s08','packets_manifest.json'}
    with np.load(output/'s08'/'point_observations.npz',allow_pickle=False) as z:
        assert set(z.files)=={'pts','observation_ids','uv','kind','stations','source_rows','path_indices'}
        np.testing.assert_array_equal(z['uv'],[qualified['rows'][0]['uv']])
        np.testing.assert_array_equal(z['pts'],[qualified['pts']])
    locations=[json.loads(line) for line in (output/'s08'/'source_row_manifest.jsonl').read_text().splitlines()]
    assert locations[0]['uv']==qualified['rows'][0]['uv']
    assert locations[0]['source_BGR_sha256']==qualified['source_BGR_sha256']
    assert locations[0]['source_jsonl_sha256']==inputs[folder/'source_rows.jsonl']
    assert (output/'s08'/'pending_source_rows.jsonl').read_bytes()==pending_bytes
    assert set(str(p) for p in folder.iterdir()).issubset(receipt['input_sha256'])
    assert not any('/s07/' in p.replace('\\','/') or '/s09/' in p.replace('\\','/') for p in receipt['input_sha256'])
    assert all(packets.digest(p)==sha for p,sha in inputs.items())
    assert receipt['prepare_executed'] is False and receipt['pixel_fit_executed'] is False


def test_default_build_still_requires_and_builds_all_three(bootstrap):
    root,capture,index=bootstrap
    for shot in index['shots']:write_capture(capture,shot)
    receipt=packets.build(root,capture,capture.parent/'all_packets')
    assert receipt['selected_shot_ids']==['s07','s08','s09']
    assert [s['shot_id'] for s in receipt['shots']]==['s07','s08','s09']
    rows=[]
    for shot in receipt['shots']:
        with np.load(Path(shot['packet'])/'point_observations.npz',allow_pickle=False) as z:
            rows.extend(z['source_rows'])
    np.testing.assert_array_equal(rows,[0,1,2])


@pytest.mark.parametrize('missing',['source_rows.jsonl','poses.npz'])
def test_selected_missing_capture_input_fails_before_read_or_output(bootstrap,monkeypatch,missing):
    root,capture,index=bootstrap
    write_capture(capture,index['shots'][0])
    folder,_,_=write_capture(capture,index['shots'][1]);(folder/missing).unlink()
    output=capture.parent/'missing_packets'
    monkeypatch.setattr(packets,'capture_rows',lambda *a,**kw: pytest.fail('Read before selected-input preflight'))
    with pytest.raises(FileNotFoundError,match='s08: missing selected-shot capture input: .*'+missing):
        packets.build(root,capture,output,shot_ids='s07,s08')
    assert not output.exists()


@pytest.mark.parametrize('invalid',['ineligible','missing_rgb_proof','invalid_uv'])
def test_selected_rows_keep_fit_eligibility_rgb_proof_and_exact_uv_gates(bootstrap,invalid):
    root,capture,index=bootstrap
    folder,qualified,_=write_capture(capture,index['shots'][0])
    if invalid=='ineligible':qualified['fit_eligible']=False
    elif invalid=='missing_rgb_proof':del qualified['source_BGR_sha256']
    else:qualified['rows'][0]['uv']=[1920.,100.]
    (folder/'source_rows.jsonl').write_text(json.dumps(qualified)+'\n',encoding='utf8')
    output=capture.parent/'invalid_packets'
    with pytest.raises(ValueError):packets.build(root,capture,output,shot_ids='s07')
    assert not output.exists()


def test_cli_forwards_explicit_shot_selection(monkeypatch):
    calls=[]
    monkeypatch.setattr(packets,'build',lambda *a,**kw: calls.append((a,kw)))
    monkeypatch.setattr(sys,'argv',['build_independent_shot_packets.py','--root','bootstrap',
        '--capture','capture','--output','packets','--shot-ids','s08,s09'])
    packets.main()
    assert calls==[((Path('bootstrap'),Path('capture'),Path('packets')),dict(shot_ids='s08,s09'))]
