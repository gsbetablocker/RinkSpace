"""Consume frozen source indices without copying a whole decoded row graph."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from .strict_source_fit_dataset import (
    SCHEMA, POINT_FIELDS, partition_support, finite_source_row, _named_pending_indices,
)
from .strict_absolute_source import StrictAbsoluteSourcePack
from .strict_model_scope import authenticate_source_reference


def iter_dataset_rows(manifest_reference, *, split='fit'):
    """Keep one authenticated source graph and its small role index in memory."""
    if split not in ('fit','held'):
        raise ValueError('Consume FIT and HELD separately')
    authenticate_source_reference(manifest_reference)
    manifest=json.loads(Path(manifest_reference['path']).read_text(encoding='utf8'))
    if manifest['schema']!=SCHEMA:raise ValueError('Frozen finite-support source dataset required')
    for cut in manifest['cuts']:
        authenticate_source_reference(cut['support_index'])
        ref=cut['source_pack']
        pack=StrictAbsoluteSourcePack(ref['path'],expected_sha256=ref['sha256'])
        # Full row evidence remains in pack.rows once. Only role/count scalars
        # are needed to independently check the stored support partition.
        roles=[dict(split=row['split'],point_count=row['point_count']) for row in pack.rows]
        with np.load(cut['support_index']['path'],allow_pickle=False)as saved:
            selection=saved[split+'_point_indices']
            if 'promoted_held_rows' in saved:
                for ri in saved['promoted_held_rows']:
                    if roles[int(ri)]['split']!='fit':raise ValueError('Only FIT rows can acquire HELD protection')
                    roles[int(ri)]['split']='held'
            expected=partition_support(roles,pack.values['row_offsets'],pack.values['sigma_px'])
            if 'pending_named_point_indices' in saved:
                named=saved['pending_named_point_indices']
                bound={p:(b['native_pts'],b['decoded_BGR_sha256']) for p,b in pack.frame_bindings.items()}
                actual_named=_named_pending_indices(pack.rows,pack.values['row_offsets'],ref,
                    pack.metadata['source'],bound,cut.get('pending_filter_references',[]))
                if not np.array_equal(named,actual_named):raise ValueError('Exact named pending selection changed')
                expected={k:np.setdiff1d(v,named,assume_unique=True) for k,v in expected.items()}
            for name,indices in expected.items():
                if not np.array_equal(saved[name],indices):raise ValueError('Stored source role/support selection changed')
            if not np.array_equal(saved['row_offsets'],pack.values['row_offsets']):
                raise ValueError('Stored source row boundaries changed')
        for ri,header in enumerate(pack.rows):
            lo,hi=map(int,pack.values['row_offsets'][ri:ri+2])
            a,b=np.searchsorted(selection,[lo,hi]);local=selection[a:b]-lo
            if len(local):
                row=dict(header,**{name:pack.values[name][lo:hi] for name in POINT_FIELDS})
                yield finite_source_row(row,local,ref,ri,output_split=split)
        # Loop locals otherwise retain the last graph while the next one loads.
        if pack.rows:del header
        del pack,roles,expected
        if 'row' in locals():del row
