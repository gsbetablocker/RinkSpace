"""Version the existing packets, then apply narrowly scoped source-identity QA."""
from pathlib import Path
from argparse import ArgumentParser
import shutil
import numpy as np
from shared import OUT, arrays, read, save, sha, write

DECISIONS = {
    'nayeon_s05': (1201664, 66380, 'point_inside_red_paint_not_a_material_boundary'),
    'nayeon_s10': (2124288, 53230, 'paint_flanks_and_A3_identity_unresolved_in_original'),
    'okada_s04': (924160, 4827, 'clipped_blurred_G_boundary_unresolved'),
    'nayeon_s04': (879360, 112165, 'named_gray_ring_layer_unresolved_in_original'),
    'nayeon_s06': (1544960, 205863, 'white_streak_cannot_be_identified_as_material_boundary'),
    'nayeon_s03': (612096, 45169, 'named_gray_ring_layer_unresolved_in_original'),
    'chen_s13': (1587200, 313093, 'A4_identity_unresolved_against_red_house_paint'),
    'jin_s14': (1558016, 3420, 'point_inside_red_paint_not_a_material_boundary'),
}


def run():
    target = OUT/'source_admission_final'
    if target.exists():
        receipt = read(target/'audit.json')
        for name, digest in receipt['current_packets'].items():
            assert sha(OUT/'prepared'/(name+'.npz')) == digest
        print('Existing source-identity revision verified; no changes.')
        return
    backup = OUT/'prepared_before_final_source_qa'
    backup.mkdir(exist_ok=False)
    for src in (OUT/'prepared').iterdir():
        if src.is_file():
            shutil.copy2(src, backup/src.name)
    shutil.copytree(OUT/'source_admission', OUT/'source_admission_red_only')
    cards = {x['shot']: x for x in read(OUT/'remaining_source_audit/manifest.json')}
    changes = []
    for name, (pts, origin, reason) in DECISIONS.items():
        path = OUT/'prepared'/(name+'.npz')
        a = arrays(path)
        row = np.flatnonzero((a['pts']==pts)&(a['origin_row']==origin))
        assert len(row)==1
        i = int(row[0]); old_fold = int(a['fold'][i])
        a['fold'][i] = 5
        save(path, **a)
        changes.append(dict(shot=name, pts=pts, origin_row=origin, old_fold=old_fold,
            new_fold=5, original_uv=a['uv'][i].tolist(), reason=reason,
            image=cards[name]['image'], image_sha256=sha(cards[name]['image']),
            scope='Only this exact source point; no group or temporal extrapolation'))
    write(target/'audit.json', dict(changes=changes, previous_packets=str(backup),
        previous_admission=str(OUT/'source_admission_red_only'),
        current_packets={p.stem:sha(p) for p in (OUT/'prepared').glob('*_s*.npz')},
        raw_source_uv_unchanged=True, candidate_selection='large residuals prompted source review; decisions use original material identity',
        blue_source_audit=dict(path=str(OUT/'blue_paint_source/audit.json'),
            visual_cards=str(OUT/'blue_source_cards/manifest.json'), proposals_admitted=0,
            conclusion='Large cross-section proposals captured unrelated shadows/colours. No systematic old blue edge definition proven; existing UV retained.'),
        user_visual_acceptance=False))
    print('Preserved previous packet version; excluded', len(changes), 'source-identity-unresolved points.')


def material_review():
    """Exact rows from the four later source-only cards; never a family filter."""
    target=OUT/'source_admission_material'
    if (target/'audit.json').exists():
        for name,digest in read(target/'audit.json')['current_packets'].items():
            assert sha(OUT/'prepared'/(name+'.npz'))==digest
        print('Existing material revision verified; no changes.')
        return
    groups=[
        ('nayeon_s05',1170944,[37697,37698,37720,37708,37712,37710,37713,37716],
         'middle_label_on_other_edge_or_material_interior'),
        ('nayeon_s05',1170944,[37723,37728],
         'outer_label_inside_coloured_ring'),
        ('nayeon_s05',1170944,[37734,37738],
         'outer_label_near_inner_edge_rebinding_not_confirmed'),
        ('nayeon_s06',1573120,[227679,227681,227683,227685,227682],
         'middle_label_on_outer_region_or_material_interior'),
        ('nayeon_s06',1573120,[227694,227687,227688],
         'named_material_layer_unresolved_in_low_contrast_source'),
        ('nayeon_s03',612096,[45229,45230,45231,45232,45233,45234],
         'lane4_identity_not_resolved_from_actual_short_blurred_streak'),
    ]
    backup=OUT/'prepared_before_material_review';backup.mkdir(exist_ok=False)
    cards={(c['shot'],c['pts']):c for c in read(OUT/'nayeon_material_audit/manifest.json')}
    changes=[]
    for name in sorted({g[0] for g in groups}):
        path=OUT/'prepared'/(name+'.npz');shutil.copy2(path,backup/path.name)
        a=arrays(path)
        for shot,pt,origins,reason in groups:
            if shot!=name:continue
            card=cards[name,pt]
            for origin in origins:
                row=np.flatnonzero((a['pts']==pt)&(a['origin_row']==origin))
                assert len(row)==1
                i=int(row[0]);prior=int(a['fold'][i]);assert prior<5
                a['fold'][i]=5
                changes.append(dict(shot=name,pts=pt,origin_row=origin,
                    path=str(a['path'][i]),original_uv=a['uv'][i].tolist(),
                    old_fold=prior,new_fold=5,reason=reason,
                    image=card['image'],image_sha256=sha(card['image']),
                    scope='Only this exact source row; no cross-frame or group extrapolation'))
        save(path,**a)
    write(target/'audit.json',dict(changes=changes,previous_packets=str(backup),
        previous_admission=str(OUT/'source_admission_final/audit.json'),
        current_packets={p.stem:sha(p) for p in (OUT/'prepared').glob('*_s*.npz')},
        raw_source_uv_unchanged=True,proposed_rebindings_admitted=0,
        candidate_selection='Large residual groups prompted source-only material review, not error-based rejection',
        card_legend_correction='F3 d78_material_151 is middle and 152 is outer; both were drawn blue by the generic card helper',
        retained_uncertainty='Unlisted rows were not visually certified. The lane3 card supports a faint source line, not its complete physical identity.',
        user_visual_acceptance=False))
    print('Preserved packets; isolated',len(changes),'specific source rows')


if __name__=='__main__':
    ap=ArgumentParser(description=__doc__);ap.add_argument('--material-review',action='store_true')
    if ap.parse_args().material_review:material_review()
    else:run()
