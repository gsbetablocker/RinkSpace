"""Re-read independently registered source prints into pending repair evidence.

This adapter does not infer identity, change source registrations, ingest points,
load camera state, or run a fit. Cached whole-print registration must still pass
its current source-patch check. Every candidate remains pending for source review.
"""
from pathlib import Path
import argparse
import hashlib
import json
from collections import Counter

import cv2
import numpy as np

from rinkspace.geometry import project
from rinkspace.session.registered_panel_edges import read_registered_panel_edges
from capture_independent_badge import exact_frames
from capture_independent_prints import material_guard, source_patch_check
from capture_independent_materials import source_exclusion, spaced_indices, plain, write


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): digest.update(chunk)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--registration', type=Path)
    p.add_argument('--shot', default='s09')
    p.add_argument('--ad', default='Ad3')
    p.add_argument('--pts', type=int, nargs='+', default=[2132224, 2136832, 2139136, 2139648])
    args = p.parse_args()
    cv2.setNumThreads(2)
    root = args.root.resolve(); output = args.output.resolve()
    if output.exists(): raise FileExistsError(output)
    registration = (args.registration or root/'registered_full').resolve()
    index = json.loads((root/'index.json').read_text('utf8'))
    source = Path(index['source']['resolved_path'])
    video_sha = sha(source)
    if video_sha != index['source']['actual_sha256']: raise ValueError('Original source hash differs from index')
    cache_binding = json.loads((registration/'binding.json').read_text('utf8'))
    if cache_binding['source_sha256'] != video_sha: raise ValueError('Cached registration belongs to another original')
    shot = next(s for s in index['shots'] if s['shot_id'] == args.shot)
    requested = sorted(set(args.pts))
    first, last = shot['processing']['first_pts'], shot['processing']['last_pts']
    if any(not first <= pt <= last or (pt-first) % 256 for pt in requested):
        raise ValueError('Requested PTS are outside the registered native shot grid')
    cached = {}
    registration_file = registration/args.shot/'registration.jsonl'
    with registration_file.open(encoding='utf8') as stream:
        for line_number, line in enumerate(stream, 1):
            record = json.loads(line)
            if record['pts'] in requested:
                cached[record['pts']] = (record, line_number)
    if set(cached) != set(requested): raise ValueError('Missing cached source registration frame')
    reference_bank = root/'print_references/references.json'
    bank = json.loads(reference_bank.read_text('utf8'))
    references = {}
    for meta in bank['references']:
        if meta['ad_id'] != args.ad: continue
        z = dict(np.load(meta['descriptor_npz'], allow_pickle=False))
        reference_path = root/meta['shot_id']/'originals'/f'{meta["shot_id"]}_pts{meta["pts"]}.png'
        z['source_image'] = cv2.imread(str(reference_path))
        z['reference_mask'] = cv2.imread(meta['descriptor_mask'], cv2.IMREAD_GRAYSCALE) > 0
        if z['source_image'] is None: raise ValueError('Original reference RGB missing')
        if hashlib.sha256(z['source_image'].tobytes()).hexdigest() != meta['source_BGR_sha256']:
            raise ValueError('Reference original source pixels differ from bank')
        if sha(meta['descriptor_npz']) != meta['descriptor_npz_sha256']:
            raise ValueError('Reference descriptor content differs from bank')
        references[meta['pts']] = (meta, z)
    images = exact_frames(source, requested)
    output.mkdir(parents=True)
    folder = output/args.shot; folder.mkdir()
    originals = folder/'originals'; originals.mkdir()
    proofs = folder/'proof'; proofs.mkdir()
    source_root = Path(__file__).resolve().parents[4]
    files = [Path(__file__), source_root/'src/rinkspace/session/registered_panel_edges.py',
        source_root/'src/rinkspace/session/panel_boundaries.py',
        source_root/'src/rinkspace/session/material_tracks.py',
        Path(__file__).with_name('capture_independent_prints.py'),
        Path(__file__).with_name('capture_independent_materials.py'),
        Path(__file__).with_name('capture_independent_badge.py')]
    implementation = {str(f.resolve()): sha(f) for f in files}
    binding = dict(source_sha256=video_sha, source_video=str(source),
        source_pts_time_base='1/12800', pixel_domain='native_distorted',
        source_registration=str(registration_file), source_registration_sha256=sha(registration_file),
        reference_bank=str(reference_bank), reference_bank_sha256=sha(reference_bank),
        implementation_sha256=implementation, independent_source_capture=True,
        legacy_new_shot_inputs_read=False, native_state_loaded=False, solver_started=False,
        source_feature_warp_is_boundary_measurement=False, source_component_is_boundary_measurement=False,
        fit_eligible=False, review_state='pending_independent_original_RGB_exterior_review')
    write(output/'binding.json', binding)
    summaries = []
    with (folder/'source_rows.jsonl').open('w', encoding='utf8') as rows_out, (folder/'registration.jsonl').open('w', encoding='utf8') as audit:
        for pt in requested:
            image = images[pt]; rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask, mask_path = source_exclusion(root, args.shot, pt)
            original = originals/f'{args.shot}_pts{pt}.png'
            cv2.imwrite(str(original), image)
            applied_mask = proofs/f'{pt}_applied_exclusion.png'
            cv2.imwrite(str(applied_mask), mask.astype('uint8')*255)
            parent, line_number = cached[pt]
            items = [n for n in parent['prints'] if n.get('ad') == args.ad
                and n.get('status') == 'source_content_registered'
                and n.get('source_pattern_check', {}).get('supported')]
            notes = []; rows = []; pending = []; all_records = []
            proof_paths = {}
            for item in items:
                meta, z = references[item['reference_pts']]
                h = np.asarray(item['H_current_from_reference'], float)
                check = source_patch_check(image, mask, meta, z, h)
                note = dict(ad=args.ad, reference_pts=meta['pts'], H_current_from_reference=h.tolist(),
                    source_registration_line=line_number, cached_pattern_check=item['source_pattern_check'],
                    source_pattern_check=check, registration_is_navigation_only=True,
                    physical_identity_determined_by_repair_reader=False)
                if not check['supported']:
                    note['status'] = 'source_pattern_pending'; notes.append(note); continue
                uv = project(h, z['projected_uv']); offsets = z['edge_offsets']
                paths = [uv[offsets[j]:offsets[j+1]] for j in range(4)]
                def save_proof(stage, arrays):
                    dest = proofs/f'{pt}_{stage}.npz'
                    np.savez_compressed(dest, **arrays)
                    proof_paths[stage] = dict(path=str(dest), sha256=sha(dest))
                local, record = read_registered_panel_edges(rgb, paths, mask,
                    binding=dict(source_sha256=video_sha, source_pts=pt, time_base='1/12800',
                        shot_id=args.shot, independent_source_pattern_match_verified=True),
                    search_radius=20., spacing=24.,
                    candidate_material_validator=lambda row: material_guard(rgb, row, args.ad)[0],
                    proof_sink=save_proof)
                for candidate in record['candidate_records']:
                    q = dict(candidate, path_id=meta['native_edge_path_ids'][candidate['curve']],
                        entity=args.ad, fit_eligible=False, source_proof=proof_paths.get(candidate['stage']),
                        profile_query_index=candidate['stage_query'], source_reference_pts=meta['pts'])
                    q['source_colour'] = material_guard(rgb, q, args.ad)[1]
                    pending.append(q)
                for j, path_id in enumerate(meta['native_edge_path_ids']):
                    group = [q for q in local if q['curve'] == j]
                    for k in spaced_indices([q['uv'] for q in group]):
                        row = group[k]
                        candidate_id = next(c['candidate_id'] for c in record['candidate_records']
                            if c['disposition'] == 'source_line_organized_actual_measurement'
                            and c['curve'] == j and c['uv'] == row['uv'])
                        trace = record['candidate_records'][candidate_id]
                        rows.append(dict(row, path_id=path_id, entity=args.ad,
                            source_candidate_id=candidate_id, source_profile_query=trace['stage_query'],
                            source_topology_probes=trace['source_topology_probes'],
                            actual_source_sides=trace['actual_source_sides'],
                            source_colour=material_guard(rgb, row, args.ad)[1],
                            source_profile_proof=proof_paths['actual_source_envelope_search'],
                            source_reference_pts=meta['pts'], capture_family='source_registered_exterior_repair',
                            fit_eligible=False))
                note.update(status='source_exterior_repair_pending', source_readout=record, profile_proofs=proof_paths)
                all_records.append(record); notes.append(note)
            source_row = dict(pts=pt, source_BGR_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                source_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(), source_video_sha256=video_sha,
                source_image=str(original), source_image_file_sha256=sha(original),
                mask_path=str(mask_path), mask_sha256=sha(mask_path), applied_exclusion=str(applied_mask),
                applied_exclusion_sha256=sha(applied_mask), fit_eligible=False,
                review_state='pending_independent_original_RGB_exterior_review', rows=rows,
                pending_rows=pending, source_registration_record=dict(path=str(registration_file), line=line_number),
                profile_proofs=proof_paths)
            rows_out.write(json.dumps(plain(source_row), ensure_ascii=False)+'\n')
            audit.write(json.dumps(plain(dict(pts=pt, prints=notes, fit_eligible=False)), ensure_ascii=False)+'\n')
            overlay = image.copy()
            for record in all_records:
                topology = record.get('source_search_chart')
                if topology:
                    for path in topology['paths']:
                        cv2.polylines(overlay, [np.rint(path).astype('int32')], False, (0, 180, 180), 1)
                for q in record['candidate_records']:
                    if q['stage'] != 'actual_source_envelope_search': continue
                    c = (0, 180, 0) if q['candidate_flags'] == 0 else (0, 130, 255)
                    cv2.circle(overlay, tuple(np.rint(q['uv']).astype(int)), 2, c, -1)
            for row in rows:
                qq = np.array(row['actual_source_quantiles'])[0]
                cv2.line(overlay, tuple(np.rint(qq[0]).astype(int)), tuple(np.rint(qq[2]).astype(int)), (255, 0, 255), 1)
                cv2.circle(overlay, tuple(np.rint(row['uv']).astype(int)), 3, (0, 0, 255), 1)
            proof_image = proofs/f'{pt}_actual_exterior_pending.jpg'
            cv2.imwrite(str(proof_image), overlay)
            reasons = Counter(reason for q in pending for reason in q['pending_reasons'])
            summary = dict(pts=pt, source_time_seconds=pt/12800., ad=args.ad,
                pattern_supported=any(n.get('source_pattern_check', {}).get('supported') for n in notes),
                all_actual_RGB_candidates=len(pending), organized_actual_RGB_candidates=sum(len(r['selected_candidate_ids']) for r in all_records),
                spaced_pending_rows=len(rows), rows_by_path=dict(Counter(q['path_id'] for q in rows)),
                candidate_rejection_reasons=dict(reasons), proof_image=str(proof_image),
                source_profiles=proof_paths, fit_eligible=False)
            summaries.append(summary); print(json.dumps(plain(summary), ensure_ascii=True), flush=True)
    report = dict(binding, frames=summaries, total_frames=len(summaries),
        total_pending_rows=sum(s['spaced_pending_rows'] for s in summaries),
        total_actual_RGB_candidates=sum(s['all_actual_RGB_candidates'] for s in summaries),
        source_rows=str(folder/'source_rows.jsonl'), registration_audit=str(folder/'registration.jsonl'),
        note='All candidates pending. Source envelope supplies topology/search only; actual q50 UV remains unchanged. No camera state or fit touched.')
    write(output/'report.json', report)


if __name__ == '__main__': main()