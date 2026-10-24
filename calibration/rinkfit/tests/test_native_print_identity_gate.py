import hashlib
import json

import numpy as np
import pytest

from rinkfit_joint.native_print_identity_gate_v1 import (
    CurrentExposureIdentityGateV1, canonical_allowed_paths, numerical_rows)


def make_gate(tmp_path, monkeypatch, decision='confirmed'):
    monkeypatch.setattr('rinkfit_joint.native_print_identity_gate_v1.np', np)
    def save(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value), encoding='utf8')
        return dict(file=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    cfg = dict(source_video_sha256='video', seeds=[dict(label='Ad2.print.G', native_pts=256, uv=[8., 9.])])
    config = save('config.json', cfg)
    manifest = save('candidates.json', dict(source_model_projection_used=False))
    paths = save('paths.json', dict(original_image_canonical=True))
    evidence = save('original_image_review.json', dict(reviewed=True))
    readout = save('512.npz', dict(actual_immutable_source=True))
    row = dict(seed_index=0, shot_id='s04', native_pts=512,
               source_label='Ad2.print.G', canonical_label='Ad2.print.G', path=0,
               source_row_id='native_named_print_1022/s04/512/seed0',
               source_uv=[10., 11.], source_BGR_sha256='rgb', source_readout=readout,
               uses_fitted_model_projection=False, qualification_uses_model_residual=False,
               raw_UV_PTS_ID_preserved=True, decision=decision, reason='Original current image review',
               physical_entity_path_source_confirmed=True, source_identity_evidence=[evidence])
    review = dict(schema='rinkfit.native_print_current_identity_review.v1',
                  source_original_images_reviewed=True, source_model_projection_used=False,
                  qualification_uses_model_residual=False, source_pixels_changed=False,
                  source_video_sha256='video', source_config=config, candidate_manifest=manifest,
                  source_seed_path_review=paths, rows=[row])
    filename = save('current_review.json', review)['file']
    args = dict(source_config=config['file'], candidate_manifest=manifest['file'],
                path_review=paths['file'], current_review=filename, config=cfg,
                materials=np.array([0]), paths=np.array([0]),
                model=dict(materials=[dict(label='Ad2.print.G')]))
    return args, row, review, dict(file=readout['file'], sha256=readout['sha256'], source_BGR_sha256='rgb')


def test_low_fb_or_named_anchor_cannot_qualify_missing_current_exposure(tmp_path, monkeypatch):
    args, row, _, frame = make_gate(tmp_path, monkeypatch)
    gate = CurrentExposureIdentityGateV1(**args)
    assert gate.mask(shot_id='s04', native_pts=512, seed_indices=[0], uv=[[10., 11.]], frame=frame).tolist() == [True]
    # Same signed edge and perfect flow still do not grant another exposure/seed.
    assert gate.mask(shot_id='s04', native_pts=768, seed_indices=[0], uv=[[10., 11.]], frame=frame).tolist() == [False]
    assert gate.last_reasons == ['current_exposure_entity_identity_unknown']


@pytest.mark.parametrize('decision', ['unknown', 'revoked'])
def test_known_wrong_or_unknown_current_identity_stays_out(tmp_path, monkeypatch, decision):
    args, row, _, frame = make_gate(tmp_path, monkeypatch, decision)
    gate = CurrentExposureIdentityGateV1(**args)
    assert not gate.mask(shot_id='s04', native_pts=512, seed_indices=[0], uv=[[10., 11.]], frame=frame)[0]


@pytest.mark.parametrize('changed', ['uv', 'rgb', 'readout'])
def test_current_verdict_cannot_bind_another_actual_reading(tmp_path, monkeypatch, changed):
    args, _, _, frame = make_gate(tmp_path, monkeypatch)
    gate = CurrentExposureIdentityGateV1(**args)
    uv = [[10., 11.]]
    if changed == 'uv': uv = [[10.000000001, 11.]]
    elif changed == 'rgb': frame['source_BGR_sha256'] = 'other'
    else: frame['sha256'] = 'other'
    with pytest.raises(ValueError, match='actual immutable'):
        gate.mask(shot_id='s04', native_pts=512, seed_indices=[0], uv=uv, frame=frame)


def test_issuance_requires_current_identity_authority_and_exact_canonical_path(tmp_path, monkeypatch):
    args, _, review, _ = make_gate(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='requires original'):
        CurrentExposureIdentityGateV1(**dict(args, current_review=None))
    review['rows'][0]['path'] = 1
    with open(args['current_review'], 'w', encoding='utf8') as f: json.dump(review, f)
    with pytest.raises(ValueError, match='canonical seed/path'):
        CurrentExposureIdentityGateV1(**args)
    rows = numerical_rows(seed_indices=[0], uv=np.array([[10., 11.]]), materials=[0], paths=[1], model=args['model'])
    assert rows[0]['allowed_paths'] == [1] and rows[0]['path_index'] == 1
    with pytest.raises(ValueError, match='cannot union identities'):
        canonical_allowed_paths(seed_indices=[0, 1], materials=[0, 0], paths=[0, 1])


def test_real_append_requires_identity_before_rgb_or_numerical_initialization(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from rinkfit_joint import native_print_append_v3 as issuer, native_print_identity_gate_v1 as gate_module, role_guard
    def save(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value), encoding='utf8')
        return dict(file=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    cfg = save('config.json', dict(source_video_sha256='video', seeds=[dict(label='Ad2.print.G')]))
    reader = save('reader.py', dict(immutable=True))
    manifest = save('manifest.json', dict(config_file=cfg['file'], config_sha256=cfg['sha256'],
        reader_file=reader['file'], reader_sha256=reader['sha256'], source_model_projection_used=False))
    save('frame_index.json', [])
    model = save('model.json', dict(materials=[dict(label='Ad2.print.G', paths=[dict(id='G')])]))
    review = save('entity_review.json', dict(source_model_residual_used=False, source_pixels_changed=False,
        source_config=cfg, eligible_materials=[dict(label='Ad2.print.G', material_index=0, physical_surface='board', allowed_paths=[0])]))
    monkeypatch.setattr(role_guard, 'FrameRoleGuard', lambda p: SimpleNamespace(document=dict(
        schema='rinkfit.joint_source_frame_protection.v1', source_video_sha256='video')))
    monkeypatch.setattr(gate_module, 'seed_bindings', lambda **k: (np.array([0]), np.array([0]), {}))
    def forbidden(*args, **kwargs): raise AssertionError('Must not decode or seed before source identity gate')
    monkeypatch.setattr(issuer, 'actual_frames', forbidden)
    monkeypatch.setattr(issuer, 'seed_readout', forbidden)
    output = tmp_path / 'not_issued'
    with pytest.raises(ValueError, match='requires original'):
        issuer.append(parent_packet='unused', parent_qualification='unused', parent_weights='unused',
            model=model['file'], candidate_manifest=manifest['file'], source_review=review['file'],
            output=output, frame_roles='unused', source_path_review='declared_but_no_current_review')
    assert not output.exists()
