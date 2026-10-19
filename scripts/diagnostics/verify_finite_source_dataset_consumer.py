"""Exercise the public FIT view on actual ice, board and mixed-support sources."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from calibration.program_tracking.strict_source_fit_dataset import reference
from calibration.program_tracking.strict_source_fit_dataset_stream import iter_dataset_rows
from calibration.program_tracking.strict_model_scope import authenticate_source_reference


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--source-index',type=int,action='append',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    dataset_ref=reference(args.dataset);authenticate_source_reference(dataset_ref)
    dataset=json.loads(args.dataset.read_text(encoding='utf8'))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    records=[];started=time.perf_counter()
    for index in args.source_index:
        selected=dataset['cuts'][index]
        subset=deepcopy(dataset);subset['cuts']=[selected]
        subset['diagnostic_subset_only_not_complete_dataset']=True
        path=args.output_dir/f'consumer_source_{index:02d}.json'
        with path.open('x',encoding='utf8')as f:json.dump(subset,f,separators=(',',':'))
        with np.load(selected['source_pack']['path'],allow_pickle=False)as z:
            uv=z['uv'];sigma=z['sigma_px'];offsets=z['row_offsets']
        count=rows=0;scopes=set();targets=set()
        for row in iter_dataset_rows(reference(path),split='fit'):
            locator=row['source_dataset_locator'];ri=locator['source_row_index']
            ids=int(offsets[ri])+np.asarray(locator['local_point_indices'],np.int64)
            if (not np.isfinite(row['sigma_px']).all() or np.any(row['sigma_px']<=0)
                    or row['split']!='fit' or row['uv'].tobytes()!=uv[ids].tobytes()
                    or row['sigma_px'].tobytes()!=sigma[ids].tobytes()):
                raise ValueError('Public view changed original source UV, sigma or finite FIT role')
            rows+=1;count+=len(ids);scopes.add(row['scope']);targets.add(row['model_target']['kind'])
        if count!=selected['fit_points']:raise ValueError('Public consumer lost selected FIT stations')
        records.append(dict(source_index=index,source_pack_reference=selected['source_pack'],
            consumed_fit_points=count,consumed_fit_rows=rows,scopes=sorted(scopes),target_kinds=sorted(targets),
            source_UV_and_sigma_bitexact=True,all_fit_sigma_finite_positive=True))
        print(json.dumps(dict(source_index=index,fit_points=count,rows=rows)),flush=True)
    result=dict(schema='rinkspace.actual-finite-source-dataset-public-consumer-smoke.v1',
        dataset_reference=dataset_ref,records=records,passed=True,seconds=time.perf_counter()-started,
        selected_actual_sources_only_not_full_solver_route=True,solver_started=False,
        independent_pixel_accuracy_accepted=False)
    path=args.output_dir/'public_consumer_smoke.json'
    with path.open('x',encoding='utf8')as f:json.dump(result,f,separators=(',',':'))
    print(json.dumps(dict(report_reference=reference(path),passed=True,seconds=result['seconds'])),flush=True)


if __name__=='__main__':main()
