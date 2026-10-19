"""Compare immutable full-cut motion reviews without decoding or fitting.

All source-supported pairs are retained, including regressions. This is a
development diagnostic of relative motion, not an absolute accuracy verdict.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    a = np.asarray(values, float)
    return dict(count=len(a), median=float(np.median(a)),
                p95=float(np.percentile(a, 95)), minimum=float(a.min()),
                maximum=float(a.max())) if len(a) else dict(count=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--before', type=Path, required=True)
    ap.add_argument('--after', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    old, new = [json.loads(p.read_text(encoding='utf8')) for p in (args.before, args.after)]
    for key in ('schema', 'source_tracks_sha256', 'source_people_box_qualification',
                'source_people_box_padding_native_px'):
        if old[key] != new[key]:
            raise ValueError(f'Comparison source/domain differs: {key}')
    if len(old['pairs']) != len(new['pairs']):
        raise ValueError('Comparison pair coverage differs')
    rows = []
    for a, b in zip(old['pairs'], new['pairs']):
        for key in ('source_pts', 'target_pts', 'shot_index', 'source_tracks',
                    'original_source_tracks', 'source_people_box_excluded'):
            if a.get(key) != b.get(key):
                raise ValueError(f'Original evidence differs: {key}')
        # The review records source motion only after a successful model-domain
        # query. A new domain failure must remain a reported regression, rather
        # than being mistaken for a change to the immutable source archive.
        for key in ('actual_motion_energy_px2', 'actual_motion_median_xy_px'):
            if key in a and key in b:
                if a[key] != b[key]:
                    raise ValueError(f'Original evidence differs: {key}')
            elif (key in a) != (key in b):
                missing = b if key in a else a
                if not missing.get('current_model_domain_failure'):
                    raise ValueError(f'Unexplained absent source evidence: {key}')
        row = {k: a[k] for k in ('source_pts', 'target_pts', 'shot_index', 'source_tracks')}
        row['supported'] = a['source_tracks'] >= 6 and all('error_native_px' in r for r in (a, b))
        row['domain_failure_before'] = a.get('current_model_domain_failure')
        row['domain_failure_after'] = b.get('current_model_domain_failure')
        if row['supported']:
            row.update(before_median_px=a['error_native_px']['median'],
                       after_median_px=b['error_native_px']['median'],
                       delta_median_px=b['error_native_px']['median'] - a['error_native_px']['median'],
                       before_relative_excess=a['relative_excess_along_actual_motion'],
                       after_relative_excess=b['relative_excess_along_actual_motion'],
                       before_signed_xy=a['signed_median_xy_px'], after_signed_xy=b['signed_median_xy_px'],
                       energy=a['actual_motion_energy_px2'],
                       before_dot=a['residual_dot_actual_motion_px2'],
                       after_dot=b['residual_dot_actual_motion_px2'])
        rows.append(row)

    def summary(rr):
        supported = [r for r in rr if r['supported']]
        energy = sum(r['energy'] for r in supported)
        return dict(all_pairs=len(rr), supported_pairs=len(supported),
                    source_supported_pairs=sum(r['source_tracks'] >= 6 for r in rr),
                    missing_or_sparse_pairs=sum(r['source_tracks'] < 6 for r in rr),
                    model_domain_failures_before=[r for r in rr if r['domain_failure_before']],
                    model_domain_failures_after=[r for r in rr if r['domain_failure_after']],
                    before_pair_median_px=stats([r['before_median_px'] for r in supported]),
                    after_pair_median_px=stats([r['after_median_px'] for r in supported]),
                    change_in_pair_median_px=stats([r['delta_median_px'] for r in supported]),
                    energy_weighted_excess_before=sum(r['before_dot'] for r in supported) / energy if energy else None,
                    energy_weighted_excess_after=sum(r['after_dot'] for r in supported) / energy if energy else None,
                    pairs_decreased=sum(r['delta_median_px'] < 0 for r in supported),
                    pairs_increased=sum(r['delta_median_px'] > 0 for r in supported),
                    largest_reductions=sorted(supported, key=lambda r: r['delta_median_px'])[:5],
                    largest_increases=sorted(supported, key=lambda r: r['delta_median_px'], reverse=True)[:5])

    per_shot = [dict(shot_index=n, **summary([r for r in rows if r['shot_index'] == n])) for n in range(1, 10)]
    result = dict(schema='rinkspace.same-source-all-cut-motion-change.v1',
                  before=dict(path=str(args.before.resolve()), sha256=sha(args.before), q=old['state_identity']['q_sha256']),
                  after=dict(path=str(args.after.resolve()), sha256=sha(args.after), q=new['state_identity']['q_sha256']),
                  source_tracks_sha256=old['source_tracks_sha256'], all_original_pairs_unchanged=True,
                  source_extraction_or_fitting_performed=False, source_images_written=0,
                  aggregate=summary(rows), per_shot=per_shot,
                  limitations=['Source-only navigation does not individually prove material identity.',
                               'Relative motion is not absolute line/advertisement alignment.',
                               'Previously inspected source pairs are development evidence, not unused validation.',
                               'Missing and sparse pairs are included in coverage, not assigned zero error.',
                               'Two accepted states differ in several parameter groups; comparison alone is not attribution.'],
                  accuracy_accepted=False)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(dict(output=str(args.output), all_pairs=len(rows),
                         summary=[{k: r[k] for k in ('shot_index', 'supported_pairs', 'before_pair_median_px',
                                                   'after_pair_median_px', 'energy_weighted_excess_before',
                                                   'energy_weighted_excess_after')} for r in per_shot])))


if __name__ == '__main__':
    main()
