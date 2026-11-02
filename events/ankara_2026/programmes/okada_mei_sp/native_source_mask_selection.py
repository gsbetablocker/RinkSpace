"""Explicit, reviewed same-RGB exclusions for native source capture only.

The default mask adapter and its historical guards are untouched. A selected
mask must reproduce the refined SAM2 silhouette plus local uncovered-hint
safety envelopes, the existing dilation and HUD exclusions. These masks only
restrict source searches; they never provide measured material pixels.
"""
from pathlib import Path
import hashlib
import json

import cv2
import numpy as np

from capture_independent_materials import (_source_mask_document,
    _source_mask_record, _source_mask_path, _read_source_mask)


SCHEMA = 'rinkspace.explicit_same_RGB_ice_mask_selections.v1'
METHOD = 'refined_SAM2_plus_local_uncovered_hint_safety_envelopes'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def reference(path):
    return dict(path=str(Path(path).resolve()), sha256=digest(path))


def local_hint_exclusion(refined, diagnostic, *, hint_radius_px=24):
    """Build a conservative search mask, keeping failed SAM2 frames guarded."""
    refined = np.asarray(refined)
    if (refined.shape != (1080,1920) or not np.isin(refined, [0,255]).all()
            or not np.any(refined)):
        raise ValueError('A nonempty native binary refined SAM2 mask is required')
    if (diagnostic.get('errors') or diagnostic.get('empty_object_ids')
            or diagnostic.get('candidate_present') is not True):
        raise ValueError('Failed or empty SAM2 support must retain the original guard')
    if isinstance(hint_radius_px, bool) or hint_radius_px != 24:
        raise ValueError('This explicit mask construction uses 24px hint safety envelopes')
    mask = refined.copy()
    hints = diagnostic.get('uncovered_positive_hints')
    if not isinstance(hints, list):
        raise ValueError('Refined SAM2 uncovered-hint diagnostics are required')
    for hint in hints:
        uv = np.asarray(hint.get('uv'), float)
        if (uv.shape != (2,) or not np.isfinite(uv).all()
                or np.any(uv < 0) or np.any(uv > [1919,1079])):
            raise ValueError('Invalid same-RGB uncovered positive hint')
        cv2.circle(mask, tuple(np.rint(uv).astype(int)), hint_radius_px, 255, -1)
    mask = cv2.dilate(mask, np.ones((7,7), np.uint8)).astype(bool)
    mask[:290,:570] = True
    mask[940:,1580:] = True
    return mask, dict(method=METHOD, hint_radius_px=24, dilation_kernel_px=[7,7],
        HUD_xyxy=[[0,0,570,290],[1580,940,1920,1080]],
        uncovered_positive_hints=hints, hints_are_pixel_ground_truth=False,
        purpose='conservative search exclusion safety boundary only',
        source_material_UV_created=False)


def load_selections(path, bootstrap):
    """Authenticate the explicit approved document and its immutable inputs."""
    if path is None:
        return {}, None
    path, bootstrap = Path(path).resolve(), Path(bootstrap).resolve()
    data = json.loads(path.read_text(encoding='utf8'))
    if (data.get('schema') != SCHEMA or data.get('time_base') != '1/12800'
            or not isinstance(data.get('selections'), list)):
        raise ValueError('Invalid explicit same-RGB mask selection document')
    index = json.loads((bootstrap/'index.json').read_text(encoding='utf8'))
    source_sha = index['source']['sha256']
    if data.get('source_video_sha256') != source_sha:
        raise ValueError('Explicit mask selection belongs to another source video')
    result = {}
    for item in data['selections']:
        if item.get('approved') is not True or not item.get('review_conclusion', '').strip():
            raise ValueError('Pending source-mask candidates cannot be selected for capture')
        binding = item.get('source_binding', {})
        sid, pts = binding.get('shot_id'), binding.get('source_pts')
        if (not isinstance(sid, str) or isinstance(pts, bool) or not isinstance(pts, int)
                or binding.get('source_sha256') != source_sha
                or binding.get('time_base') != '1/12800'
                or binding.get('native_hw') != [1080,1920]):
            raise ValueError('Explicit mask source exposure binding differs')
        key = sid, pts
        if key in result:
            raise ValueError('Duplicate explicit source-mask exposure')
        ledger = item['refined_ledger']; ledger_path = Path(ledger['path']).resolve()
        if ledger_path != bootstrap/'person_masks_refined/full/frames.jsonl':
            raise ValueError('Explicit mask refined ledger belongs to another bootstrap')
        records, _ = _source_mask_document(ledger_path, ledger['sha256'])
        record = _source_mask_record(records, sid, pts)
        if (record['source_binding'] != binding
                or record['decoded_native_RGB_sha256'] != item['source_RGB_sha256']):
            raise ValueError('Explicit mask and refined ledger source RGB differ')
        refined_ref = item['refined_mask']
        refined_path = _source_mask_path(record['mask_path'], ledger_path.parent)
        if (Path(refined_ref['path']).resolve() != refined_path
                or refined_ref['sha256'] != record['mask_sha256']):
            raise ValueError('Explicit mask refined PNG does not match its ledger')
        refined = _read_source_mask(refined_path, refined_ref['sha256'])
        expected, construction = local_hint_exclusion(refined, record['diagnostic'])
        if item.get('construction') != construction:
            raise ValueError('Explicit mask construction receipt differs')
        selected_ref = item['selected_mask']
        selected = _read_source_mask(selected_ref['path'], selected_ref['sha256'])
        if not np.isin(selected, [0,255]).all() or not np.array_equal(selected > 0, expected):
            raise ValueError('Selected mask differs from its declared safe construction')
        guard = item['replaced_guard']
        if digest(guard['path']) != guard['sha256']:
            raise ValueError('Historical guard bytes changed')
        result[key] = dict(item=item, mask=expected, document=reference(path))
    return result, reference(path)


def selected_frame(frame, selection):
    """Apply an authenticated mask only to its exact decoded source exposure."""
    if selection is None:
        return frame
    item = selection['item']; binding = item['source_binding']
    image = frame['image']
    rgb_sha = hashlib.sha256(np.ascontiguousarray(image[:,:,::-1]).tobytes()).hexdigest()
    if (frame['shot_id'] != binding['shot_id'] or frame['pts'] != binding['source_pts']
            or frame['source_video_sha256'] != binding['source_sha256']
            or frame['source_BGR_sha256'] != item['source_BGR_sha256']
            or rgb_sha != item['source_RGB_sha256']):
        raise ValueError('Explicit mask does not belong to this actual decoded exposure')
    prior = item['replaced_guard']
    if (Path(frame['mask_path']).resolve() != Path(prior['path']).resolve()
            or frame['mask_sha256'] != prior['sha256']):
        raise ValueError('Explicit mask does not replace the authenticated historical guard')
    mask_ref = item['selected_mask']; applied = selection['mask']
    receipt = dict(document=selection['document'], source_binding=binding,
        adapter=reference(__file__),
        selected_mask=mask_ref, refined_mask=item['refined_mask'],
        refined_ledger=item['refined_ledger'], replaced_guard=prior,
        construction=item['construction'], approved=True,
        review_conclusion=item['review_conclusion'], default_mask_adapter_changed=False)
    dependencies = [*frame['mask_sources'],
        dict(reference(__file__), role='explicit_same_RGB_mask_selection_adapter'),
        dict(selection['document'], role='explicit_reviewed_same_RGB_mask_selection'),
        dict(mask_ref, role='actual_explicit_same_RGB_search_exclusion')]
    return dict(frame, source_exclusion_mask=applied, mask_path=mask_ref['path'],
        mask_sha256=mask_ref['sha256'], mask_sources=dependencies,
        source_exclusion_sha256=hashlib.sha256(applied.tobytes()).hexdigest(),
        source_mask_selection=receipt)
