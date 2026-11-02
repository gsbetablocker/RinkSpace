import hashlib
import json

import cv2
import numpy as np
import pytest

from native_source_mask_selection import (SCHEMA, local_hint_exclusion,
    load_selections, reference, selected_frame)


def fixture_selection(tmp_path):
    root = tmp_path/'bootstrap'
    folder = root/'person_masks_refined/full'
    folder.mkdir(parents=True)
    refined_path, guard_path = folder/'refined.png', root/'guard.png'
    refined = np.zeros((1080,1920), np.uint8)
    refined[400:500,700:800] = 255
    cv2.imwrite(str(refined_path), refined)
    cv2.imwrite(str(guard_path), refined)
    image = np.zeros((1080,1920,3), np.uint8)
    source_sha = 'a'*64
    binding = dict(shot_id='s07', source_pts=1450240, source_sha256=source_sha,
        time_base='1/12800', native_hw=[1080,1920], image_domain='native_brown_pixels')
    rgb_sha = hashlib.sha256(image.tobytes()).hexdigest()
    diagnostic = dict(errors=[], empty_object_ids=[], candidate_present=True,
        uncovered_positive_hints=[dict(object_id=1,uv=[900.,450.])])
    record = dict(source_binding=binding, decoded_native_RGB_sha256=rgb_sha,
        mask_path=str(refined_path), mask_sha256=reference(refined_path)['sha256'],
        diagnostic=diagnostic)
    ledger = folder/'frames.jsonl'
    ledger.write_text(json.dumps(record)+'\n', encoding='utf8')
    (root/'index.json').write_text(json.dumps(dict(source=dict(sha256=source_sha))), encoding='utf8')
    mask, construction = local_hint_exclusion(refined, diagnostic)
    selected_path = root/'selected.png'
    cv2.imwrite(str(selected_path), mask.astype(np.uint8)*255)
    item = dict(approved=True, review_conclusion='Explicit source review',
        source_binding=binding, source_RGB_sha256=rgb_sha, source_BGR_sha256=rgb_sha,
        refined_mask=reference(refined_path), refined_ledger=reference(ledger),
        selected_mask=reference(selected_path), replaced_guard=reference(guard_path),
        construction=construction)
    document = dict(schema=SCHEMA, time_base='1/12800', source_video_sha256=source_sha,
        selections=[item])
    path = root/'selections.json'
    path.write_text(json.dumps(document), encoding='utf8')
    frame = dict(image=image, shot_id='s07', pts=1450240,
        source_video_sha256=source_sha, source_BGR_sha256=rgb_sha,
        mask_path=str(guard_path), mask_sha256=reference(guard_path)['sha256'],
        source_exclusion_mask=refined>0, mask_sources=[])
    return root,path,document,frame


def test_local_hint_exclusion_keeps_person_safety_envelope_and_HUD():
    refined = np.zeros((1080,1920), np.uint8)
    refined[400:500,700:800] = 255
    mask, receipt = local_hint_exclusion(refined, dict(errors=[], empty_object_ids=[],
        candidate_present=True, uncovered_positive_hints=[dict(uv=[900.,450.])]))
    assert mask[400,697] and mask[450,924] and mask[450,927]
    assert not mask[450,928] and not mask[600,900]
    assert mask[:290,:570].all() and mask[940:,1580:].all()
    assert receipt['source_material_UV_created'] is False


@pytest.mark.parametrize('failure', [dict(errors=['SAM2 failed']),
    dict(empty_object_ids=[1]), dict(candidate_present=False)])
def test_failed_refinement_keeps_old_guard(failure):
    diagnostic = dict(errors=[], empty_object_ids=[], candidate_present=True,
        uncovered_positive_hints=[])
    diagnostic.update(failure)
    with pytest.raises(ValueError, match='retain the original guard'):
        local_hint_exclusion(np.full((1080,1920),255,np.uint8), diagnostic)


def test_approved_selection_is_bound_to_exact_RGB_and_historical_guard(tmp_path):
    root,path,document,frame = fixture_selection(tmp_path)
    selections,_ = load_selections(path,root)
    result = selected_frame(frame,selections['s07',1450240])
    assert result['source_exclusion_mask'][450,900]
    assert not frame['source_exclusion_mask'][450,900]
    assert selected_frame(frame,None) is frame
    assert result['source_mask_selection']['default_mask_adapter_changed'] is False
    changed = dict(frame,image=frame['image'].copy())
    changed['image'][0,0,0] = 1
    with pytest.raises(ValueError,match='decoded exposure'):
        selected_frame(changed,selections['s07',1450240])
    with pytest.raises(ValueError,match='historical guard'):
        selected_frame(dict(frame,mask_sha256='b'*64),selections['s07',1450240])


@pytest.mark.parametrize('change', ['pending','ledger','selected_png','source_pts'])
def test_selection_rejects_unapproved_or_changed_dependencies(tmp_path,change):
    root,path,document,frame = fixture_selection(tmp_path)
    item = document['selections'][0]
    if change == 'pending':
        item['approved'] = False
    elif change == 'ledger':
        item['refined_ledger']['sha256'] = 'b'*64
    elif change == 'source_pts':
        item['source_binding']['source_pts'] += 256
    else:
        mask = cv2.imread(item['selected_mask']['path'],cv2.IMREAD_GRAYSCALE)
        mask[600,900] = 255
        cv2.imwrite(item['selected_mask']['path'],mask)
        item['selected_mask'] = reference(item['selected_mask']['path'])
    path.write_text(json.dumps(document),encoding='utf8')
    with pytest.raises(ValueError):
        load_selections(path,root)
