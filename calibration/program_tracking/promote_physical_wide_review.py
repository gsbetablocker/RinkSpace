"""Apply reviewed source width measurements without changing other targets."""
from __future__ import annotations
import hashlib
import json
import shutil
import zipfile
from .extract_first_shot_ice import DATA, read, write, sha
from .repair_first_shot_wide_line_identity import apply

RD = DATA / 'repair_20261014'
WIDE = {'ice.B1:segment0', 'ice.B2:segment0', 'ice.C:segment0'}

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode('utf8')).hexdigest()

def preserved(packet):
    return dict(source=packet['source'],
                features=[f for f in packet['features'] if f['target_id'] not in WIDE],
                coverage=[r for r in packet['per_feature_coverage'] if r['target_id'] not in WIDE],
                groups=[r for r in packet['source_correlated_groups'] if r['target_id'] not in WIDE])

def main():
    proposals = RD / 'physical_wide_proposals'
    records = []
    # Complete preflight before any candidate is changed.
    for i in range(201, 1249):
        path = proposals / f'{i:06d}.json'
        proposal = read(path)
        current = read(RD / f'ice_candidates/{i:06d}.json')
        assert proposal['programme_frame_index_1based'] == i
        assert proposal['source']['native_pts'] == current['source']['native_pts']
        assert proposal['source']['decoded_BGR_sha256'] == current['source']['decoded_BGR_sha256']
        assert all(f['target_id'] in WIDE for f in proposal['features'])
        assert sum(f['point_count'] for f in proposal['features']) == proposal['point_count']
        count = sum(f['point_count'] for f in current['features'] if f['target_id'] not in WIDE) + proposal['point_count']
        board = read(RD / f'board_candidates/{i:06d}.json')
        assert count + board['point_count'] <= 500, (i, count, board['point_count'])
        records.append(dict(index=i, proposal_path=str(path), proposal_sha256=sha(path),
                            nonwide_and_source_sha256=digest(preserved(current))))
    archive = RD / 'wide_line_diagnostics/wide_before_physical_width.zip'
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
            for i in range(201, 1249):
                source = RD / f'wide_line_candidates/{i:06d}.json'
                z.write(source, f'wide_line_candidates/{i:06d}.json')
            previous = RD / 'wide_line_identity_repair_receipt.json'
            if previous.exists(): z.write(previous, previous.name)
    for record in records:
        source = record['proposal_path']
        assert sha(source) == record['proposal_sha256'], 'Proposal changed during promotion'
        shutil.copyfile(source, RD / f"wide_line_candidates/{record['index']:06d}.json")
    apply()
    for record in records:
        path = RD / f"ice_candidates/{record['index']:06d}.json"
        assert digest(preserved(read(path))) == record['nonwide_and_source_sha256']
        record['applied_ice_sha256'] = sha(path)
    write(RD / 'wide_line_diagnostics/physical_width_applied_receipt.json', dict(
        schema='rinkspace.physical-width-target-only-promotion.v1', frames=1048,
        target_ids=sorted(WIDE), nonwide_features_coverage_groups_and_source_preserved=True,
        prior_wide_archive=dict(path=str(archive), sha256=sha(archive)), records=records))
    print('Applied 1048 physical-width packets; every non-wide target and source preserved.', flush=True)

if __name__ == '__main__': main()
