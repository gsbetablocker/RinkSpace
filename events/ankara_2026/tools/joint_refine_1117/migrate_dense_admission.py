"""Carry exact historical source decisions into the restored dense packets."""
from collections import Counter
from pathlib import Path
import shutil
import numpy as np
from shared import OUT, arrays, read, save, sha, write


def run():
    previous = OUT / 'repair/prepared_v6'
    target = OUT / 'repair/prepared_v7'
    if (target / 'setup.json').exists():
        raise FileExistsError(target)
    target.mkdir(parents=True, exist_ok=True)
    setup = read(previous / 'setup.json')
    audits = [OUT / 'source_admission_final/audit.json',
              OUT / 'source_admission_material/audit.json']
    decisions = {str(p): read(p) for p in audits}
    report = {}; hashes = {}; masses = Counter()
    for name, shot in setup['shots'].items():
        path = Path(shot['observations']); a = arrays(path)
        before = a['fold'].copy(); changes = []; supersession = []
        if shot['program'] in ('chen', 'nayeon'):
            red_path = OUT / 'source_admission_red_only' / f'{name}.npz'
            red = arrays(red_path)
            requested = []
            for i in range(len(red['pts'])):
                if red['reason'][i] == 'superseded_by_source_reviewed_two_flank_trapezoid':
                    # This was the sparse workflow's replacement instruction,
                    # not a source identity rejection. The restored packet is
                    # the programme's full corrected original, with its own
                    # bracket/eligibility record. Do not reapply sparse-wide
                    # replacement to just its old sampled subset.
                    supersession.append(int(red['origin_row'][i]))
                    continue
                if red['fold'][i] >= 5:
                    requested.append((int(red['pts'][i]), str(red['path'][i]),
                        red['old_uv'][i], str(red['reason'][i]), str(red_path)))
            for audit_path, audit in decisions.items():
                source = None
                for item in audit['changes']:
                    if item['shot'] != name or item['new_fold'] < 5:
                        continue
                    entity = item.get('path')
                    if entity is None:
                        if source is None:
                            source = arrays(Path(audit['previous_packets']) / f'{name}.npz')
                        rows = np.flatnonzero((source['pts'] == item['pts']) &
                            (source['origin_row'] == item['origin_row']) &
                            (source['uv'] == item['original_uv']).all(axis=1))
                        assert len(rows) == 1
                        entity = str(source['path'][rows[0]])
                    requested.append((item['pts'], entity, item['original_uv'], item['reason'], audit_path))
            candidates = np.flatnonzero(np.isin(a['pts'], [r[0] for r in requested]))
            lookup = {}
            for i in candidates:
                lookup.setdefault((int(a['pts'][i]), str(a['path'][i]), *a['uv'][i]), []).append(int(i))
            reason = np.full(len(before), '', dtype='<U160')
            for pts, entity, uv, why, source in requested:
                rows = lookup.get((int(pts), str(entity), *uv), [])
                for i in rows:
                    assert not a['station_fixed'][i], (name, i, 'confirmed anchor would be excluded')
                    changes.append(dict(row=i, pts=int(pts), path=entity, original_uv=list(uv),
                        previous_fold=int(before[i]), reason=why, source=source))
                    a['fold'][i] = 9
                    reason[i] = why
            if supersession:
                original_path = Path(shot['source_packets']['0']['path'])
                original = arrays(original_path)
                ids = np.asarray(supersession)
                assert original['eligible'][ids].all()
                assert np.isfinite(original['source_bracket'][ids]).all()
                np.testing.assert_allclose(original['uv'][ids],
                    original['source_bracket'][ids].mean(axis=1), atol=2e-8, rtol=0)
            if changes:
                a['restored_admission_reason'] = reason
                destination = target / f'{name}.npz'
                save(destination, **a)
                shot['observations'] = str(destination)
        active_before = np.unique(a['pts'][before < 5])
        active_after = np.unique(a['pts'][a['fold'] < 5])
        np.testing.assert_array_equal(active_before, active_after)
        changed = (before < 5) & (a['fold'] >= 5)
        report[name] = dict(previous_packet=str(path), previous_sha256=sha(path),
            current_packet=shot['observations'], rows=len(before),
            newly_inactive=int(changed.sum()), active_rows=int((a['fold'] < 5).sum()),
            active_frames=len(active_after), original_fields_unchanged_except_fold=True,
            decisions=changes, sparse_supersession_not_reapplied=supersession)
        hashes[name] = sha(shot['observations'])
        active = a['fold'] < 5
        family = a['family'][active]
        # Match solve.weights' full source group normalization exactly.
        from solve import weights
        rows = dict(program=np.repeat(shot['program'], int(active.sum())),
            layer=np.where(np.char.startswith(family, 'Ad'), 'ad',
                np.where(np.char.startswith(family, 'board.'), 'board', 'ice')),
            shot=np.repeat(name, int(active.sum())), pts=a['pts'][active],
            family=family, path=a['path'][active])
        _, mass = weights(rows)
        masses.update(mass)
    shutil.copy2(previous / 'baseline.npz', target / 'baseline.npz')
    write(target / 'full_source_masses.json', dict(masses=dict(masses), source_packets=hashes,
        source_roles='all active full source packets', prepared=str(target)))
    write(target / 'source_decisions.json', dict(previous=str(previous), shots=report,
        source_reviews={str(p): sha(p) for p in audits},
        policy='Exact source identity decisions only. No frame/space thinning; original UV, PTS, rows and confirmed anchors retained.',
        sparse_supersession='86 Chen rows retain the current original corrected packet with finite source brackets; historical blanket sparse replacement is not an identity rejection.'))
    write(target / 'setup.json', setup)
    print({p: sum(r['newly_inactive'] for n, r in report.items() if n.startswith(p + '_'))
           for p in ('jin', 'okada', 'chen', 'nayeon')}, flush=True)


if __name__ == '__main__':
    run()
