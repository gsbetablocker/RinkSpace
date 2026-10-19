"""Exercise actual child dispatch while keeping original videos untouched."""
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts import admit_programme_source_cuts as scheduler


def test_independent_cut_batch_executes_frozen_graph_successor(tmp_path, monkeypatch):
    scripts = tmp_path/'scripts'
    scripts.mkdir()
    variant = scripts/'admit_programme_source_cut_variant.py'
    variant.write_text('''import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser()
p.add_argument('--manifest',required=True)
p.add_argument('--shot-id',required=True)
p.add_argument('--output',required=True)
a=p.parse_args()
m=json.loads(Path(a.manifest).read_text())
c=next(c for c in m['cuts'] if c['shot_id']==a.shot_id)
np.savez(a.output,graph=m['ice_graph_class'],pts=np.arange(c['start_pts'],c['stop_pts']))
''', encoding='utf8')
    # The public base driver cannot satisfy a manifest requiring a successor.
    (scripts/'admit_programme_source_inputs.py').write_text(
        "raise RuntimeError('Required graph successor was not dispatched')\n", encoding='utf8')
    reference = dict(path=str(variant),sha256=hashlib.sha256(variant.read_bytes()).hexdigest())
    manifest = tmp_path/'frozen.json'
    cuts = [dict(shot_id='camera:cut1',start_pts=100,stop_pts=103),
        dict(shot_id='camera:cut2',start_pts=200,stop_pts=202)]
    manifest.write_text(json.dumps(dict(schema='rinkspace.exact-source-cut-batch.v1',
        references=[reference],common_arguments=[],cuts=cuts,
        ice_graph_class='confirmed_source_graph:Successor')), encoding='utf8')
    monkeypatch.setattr(scheduler,'ROOT',tmp_path)
    receipt = scheduler.run_source_cut_manifest(manifest,tmp_path/'outputs',workers=2)
    assert receipt['all_source_workers_succeeded'] is True
    assert [c['shot_id'] for c in receipt['cuts']] == [c['shot_id'] for c in cuts]
    for cut in receipt['cuts']:
        with np.load(cut['output_path'],allow_pickle=False) as saved:
            assert str(saved['graph'].item()) == 'confirmed_source_graph:Successor'
            assert np.array_equal(saved['pts'],np.arange(cut['start_pts'],cut['stop_pts']))
        assert Path(cut['log_path']).exists()
    assert receipt['solver_started'] is False
    assert receipt['source_processing_is_not_manual_audit_completion'] is True
