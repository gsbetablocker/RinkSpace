import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile

import numpy as np

from calibration.solver_runtime.checkpoint_history import (
    HistorySegmentWriter, _iter_json_records, _text, iter_checkpoint_history, iter_history_reference,
    migrate_legacy_checkpoint_history, read_checkpoint_history,
    read_checkpoint_reviews, read_solver_history,
)


class CheckpointHistoryTests(unittest.TestCase):
    def test_tail_versions_append_and_integrity(self):
        with tempfile.TemporaryDirectory() as folder:
            writer = HistorySegmentWriter(Path(folder) / 'segments')
            history = [dict(iteration=0, accepted=True, timing=dict(save=1))]
            first = writer.commit(history)
            history[0]['timing']['review'] = 2
            history[0] = copy.deepcopy(history[0])  # production seals the tail
            history.append(dict(iteration=1, accepted=False, report=dict(rows=[1, 2])))
            latest = writer.commit(history)
            self.assertEqual(list(iter_history_reference(first)),
                             [dict(iteration=0, accepted=True, timing=dict(save=1))])
            self.assertEqual(list(iter_history_reference(latest)), history)
            before = list((Path(folder) / 'segments').iterdir())
            self.assertEqual(writer.commit(history), latest)
            self.assertEqual(before, list((Path(folder) / 'segments').iterdir()))
            bad = dict(latest, record_count=99)
            with self.assertRaises(ValueError):
                list(iter_history_reference(bad))
            Path(latest['head']['path']).write_text('{}', encoding='utf8')
            with self.assertRaises(ValueError):
                list(iter_history_reference(latest))

    def test_legacy_stream_migration_and_copied_archive(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old, new = root / 'old.npz', root / 'recovery' / 'new.npz'
            history = [dict(iteration=i, accepted=i != 1,
                            nested=dict(values=list(range(8))), unicode='芽衣') for i in range(3)]
            reviews = [dict(event=history[-1], result=dict(stop_branch=True))]
            np.savez_compressed(old, q=np.array([1., 2.]), damping=np.array(.001),
                                profile__weights_0=np.array([[.2, .8]]),
                                history_json=np.asarray(_text(history)),
                                reviews_json=np.asarray(_text(reviews)))
            with np.load(old, allow_pickle=False) as stored:
                self.assertEqual(read_checkpoint_history(stored), history)
            self.assertEqual(list(iter_checkpoint_history(old)), history)
            receipt = migrate_legacy_checkpoint_history(old, new)
            with zipfile.ZipFile(old) as a, zipfile.ZipFile(new) as b:
                self.assertEqual(a.namelist(), b.namelist())
                for name in a.namelist():
                    if name not in ('history_json.npy', 'reviews_json.npy'):
                        self.assertEqual(a.read(name), b.read(name))
            archive = root / 'accepted' / 'complete_state.npz'
            archive.parent.mkdir()
            shutil.copy2(new, archive)
            with np.load(archive, allow_pickle=False) as stored:
                self.assertEqual(read_checkpoint_history(stored, archive), history)
                self.assertEqual(read_checkpoint_reviews(stored, archive), reviews)
            for field, records in (('history_json', history), ('reviews_json', reviews)):
                digest = hashlib.sha256()
                for record in iter_checkpoint_history(archive, field=field):
                    digest.update((_text(record) + '\n').encode('utf8'))
                self.assertEqual(receipt['record_semantic_digests'][field]['records'], len(records))
                self.assertEqual(receipt['record_semantic_digests'][field]['sha256'], digest.hexdigest())
            iterations = root / 'iterations.json'
            iterations.write_text(_text(receipt['history_reference']), encoding='utf8')
            self.assertEqual(read_solver_history(iterations), history)
            iterations.write_text(_text(history), encoding='utf8')
            self.assertEqual(read_solver_history(iterations), history)

    def test_truncated_legacy_history_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'iterations.json'
            for value in ('[{"iteration": 0}', '[{"iteration": 0},]', '[1]'):
                path.write_text(value, encoding='utf8')
                with self.assertRaises((ValueError, json.JSONDecodeError)):
                    read_solver_history(path)

    def test_fragmented_large_record_then_multiple_records(self):
        records = [dict(text='芽衣' * 3000, nested=dict(values=list(range(300)))),
                   dict(iteration=2), dict(iteration=3)]
        text = _text(records)
        chunks = (text[index:index + 7] for index in range(0, len(text), 7))
        self.assertEqual(list(_iter_json_records(chunks)), records)


if __name__ == '__main__':
    unittest.main()
