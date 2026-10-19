"""Bind the newly reviewed twelve B1 pixels to the existing whole-train shot.

This receipt adds no implicit roles to other unlisted source points and does
not rewrite the immutable observation packet.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from calibration.program_tracking.strict_absolute_source import StrictAbsoluteSourcePack
from calibration.program_tracking.strict_model_scope import authenticate_source_reference, file_sha
from rinkspace.vision import configure, load_training_manifest

WORK = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010/full_programme_strict_source_review_20261012')
DELTA = WORK/'current_B1_gap_reviewed_source_delta_20261013_v2'
SOURCE = dict(path=str(DELTA/'B1_missing_three_exposures_reviewed_source_delta.npz'),
    sha256='e365e1bc2eca8907200d35d690279f2094c6883ccdd368328bb431d0faa3a9b8')
WITNESSES = [dict(path=str(DELTA/'exact_B1_source_review_facts.json'),
    sha256='2d29f8b69a78e1de053c685e095278f84eda45c64618bfe7f3c8e706b4899f33'),
    dict(path=str(DELTA/'exact_B1_source_scope_review.json'),
    sha256='93f55d70f335b177467a1726b6b895497e8a25fba83200c36a75a2e20393127f'),
    dict(path=str(WORK/'current_source_pool_joint_qualified_20261013_v1/original_terminal_plus_current_gray_board_B1_merge_operator_audit_20261013_v2.json'),
    sha256='b74a47f795d6b6874408cf816be96f22d05f75f5079841e132f45672af03f3a9')]


def prepare(config_path, dataset_path, output_path):
    for ref in (SOURCE,*WITNESSES): authenticate_source_reference(ref)
    dataset_ref = dict(path=str(dataset_path.resolve()),sha256=file_sha(dataset_path))
    dataset = json.loads(dataset_path.read_text(encoding='utf8'))
    packets = [p for p in dataset['cuts'] if p['source_pack']['sha256']==SOURCE['sha256']
        and Path(p['source_pack']['path']).resolve()==Path(SOURCE['path']).resolve()]
    if len(packets)!=1: raise ValueError('The final FIT dataset must contain exactly this reviewed B1 delta')
    packet = packets[0];authenticate_source_reference(packet['support_index'])
    with np.load(packet['support_index']['path'],allow_pickle=False) as index:
        fit = index['fit_point_indices']
    if not np.array_equal(fit,np.arange(12,dtype=np.int64)):
        raise ValueError('Only all twelve exact reviewed, finite non-HELD B1 points may receive this admission')
    config,data = configure(config_path)
    _,plan = load_training_manifest(config,data)
    pack = StrictAbsoluteSourcePack(SOURCE['path'],expected_sha256=SOURCE['sha256'])
    if len(pack.values['uv'])!=12 or len(pack.rows)!=3:
        raise ValueError('The exact B1 source review scope changed')
    entries=[]
    for ri,row in enumerate(pack.rows):
        shots = [s for s in plan['shots'] if s['id']==row['shot_id'] and s['whole_shot'] is True
            and s['split']=='train' and s['source_sha256']==pack.metadata['source']['video_sha256']
            and s['time_base']==pack.metadata['source']['time_base']
            and s['start_pts']<=row['native_pts']<s['end_pts']]
        if len(shots)!=1 or row.get('vision_training_role') is not None or row['split']!='fit':
            raise ValueError('New review admission cannot change the existing whole-shot training split or source role')
        lo,hi=map(int,pack.values['row_offsets'][ri:ri+2]);ids=list(range(lo,hi))
        entries.append(dict(pack_row_index=ri,pack_point_indices=ids,
            **{k:row[k] for k in ('physical_material_id','feature_id','actual_pts','native_pts','decoded_BGR_sha256')},
            original_candidate_indices=pack.values['original_point_indices'][ids].tolist(),
            existing_whole_train_shot=shots[0]))
    receipt=dict(schema='rinkspace.current-reviewed-exact-source-vision-training-admission.v1',
        source_pack_reference=SOURCE,source_review_witnesses=WITNESSES,fit_dataset_reference=dataset_ref,
        entries=entries,points=12,vision_training_role='train',source_pack_roles_unchanged=True,
        admission_scope='Only the twelve newly source-reviewed B1 delta points within the existing 034 whole-train shot',
        source_labels_from_original_RGB_q50_and_measured_support=True,model_projection_used_as_labels=False,
        unknown_pending_used_as_positive_or_negative=False,other_unlisted_source_roles_promoted=False,
        original_source_UV_sigma_clock_and_review_bytes_unchanged=True)
    if output_path.exists():raise FileExistsError('Preserve earlier immutable exact training admission')
    output_path.parent.mkdir(parents=True,exist_ok=True)
    output_path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return dict(path=str(output_path.resolve()),sha256=file_sha(output_path),points=12)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--fit-dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();print(json.dumps(prepare(args.config,args.fit_dataset,args.output)))
