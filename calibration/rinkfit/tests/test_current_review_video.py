import json
from pathlib import Path
import pytest
from rinkfit_joint import joint_review_video as video
from rinkfit_joint.io_utils import file_sha256


def fixture(tmp_path, monkeypatch):
    candidate=tmp_path/'candidate';candidate.mkdir()
    current=tmp_path/'current';current.mkdir()
    def save(path, value):
        path.write_text(json.dumps(value),encoding='utf8');return path
    def ref(path): return dict(file=str(path),sha256=file_sha256(path))
    model=save(candidate/'model.json',dict(blocks=[]))
    checkpoint=save(candidate/'engineering_checkpoint.json',dict(token='saved'))
    packet=save(current/'packet.json',dict(row_count=3))
    manifest=save(current/'manifest.json',dict(schema='rinkfit.current_joint_input.v1',logical_source_id='c'*64,
        raw_rows=3,packet=ref(packet)))
    transport=dict(schema='rinkfit.current_complete_primal_transport.v1',model=ref(model),checkpoint=ref(checkpoint),
        current_input_v1=ref(manifest),from_state_token='saved',raw_rows=3)
    save(candidate/'current_checkpoint.json',transport)
    binding=dict(source_packet_sha256='c'*64,state_token='saved')
    from rinkfit_joint import joint_review_packet
    class Reader:
        def __init__(self, path, *, current_input):
            assert Path(current_input['file'])==manifest
            assert current_input['sha256']==file_sha256(manifest)
            self.packet_sha=file_sha256(path)
    monkeypatch.setattr(joint_review_packet,'CurrentReviewPacket',Reader)
    return candidate,packet,manifest,binding,transport,ref


def test_current_video_distinguishes_logical_source_and_actual_packet(tmp_path,monkeypatch):
    candidate,packet,manifest,binding,transport,ref=fixture(tmp_path,monkeypatch)
    assert binding['source_packet_sha256']!=file_sha256(packet)
    reader,receipt=video._current_review_inputs(candidate,packet,binding,file_sha256(candidate/'model.json'),manifest)
    assert receipt['logical_source_id']=='c'*64
    assert receipt['actual_source_packet']==ref(packet)
    assert receipt['state_token']=='saved' and receipt['raw_rows']==3
    assert not receipt['historical_source_replay']
    video._current_review_inputs(candidate,packet,binding,file_sha256(candidate/'model.json'))


@pytest.mark.parametrize('changed', ['logical_source','state_token','model_bytes','packet_bytes','manifest_bytes','raw_rows','current_input'])
def test_current_video_rejects_mismatched_current_bindings(tmp_path,monkeypatch,changed):
    candidate,packet,manifest,binding,transport,ref=fixture(tmp_path,monkeypatch)
    model_sha=file_sha256(candidate/'model.json'); supplied=manifest
    if changed=='logical_source':binding['source_packet_sha256']='d'*64
    elif changed=='state_token':binding['state_token']='other'
    elif changed=='model_bytes':(candidate/'model.json').write_text('{}')
    elif changed=='packet_bytes':packet.write_text('{}')
    elif changed=='manifest_bytes':manifest.write_text('{}')
    elif changed=='raw_rows':
        transport['raw_rows']=4;(candidate/'current_checkpoint.json').write_text(json.dumps(transport))
    elif changed=='current_input':
        supplied=tmp_path/'other_manifest.json';supplied.write_bytes(manifest.read_bytes())
    with pytest.raises(ValueError):video._current_review_inputs(candidate,packet,binding,model_sha,supplied)


def test_current_video_has_no_history_fallback(tmp_path,monkeypatch):
    candidate,packet,manifest,binding,transport,ref=fixture(tmp_path,monkeypatch)
    (candidate/'current_checkpoint.json').unlink()
    with pytest.raises(FileNotFoundError):
        video._current_review_inputs(candidate,packet,binding,file_sha256(candidate/'model.json'),manifest)
