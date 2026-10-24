from pathlib import Path
import hashlib
import json
import tempfile
import unittest
import numpy as np
from rinkfit_joint.effective_input_audit_v1 import CheckedBytes, digest, resolve_packet, strict_json


class EffectiveInputInspectionTests(unittest.TestCase):
    def write(self, path, value):
        path.write_bytes(json.dumps(value).encode('utf8'))
        return dict(file=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def test_native_parent_append_and_forbidden_override(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parent = self.write(root/'base.json', dict(schema='rinkfit.joint_packet.v1', arrays={'old':1},
                groups=[1], source_row_ids=['raw'], native_material_identity_revision={'file':'r','sha256':'a'}))
            head = self.write(root/'head.json', dict(schema='rinkfit.joint_packet.v1', dense_parent_packet=parent,
                arrays={'new':2}, dense_parent_append={'groups':[2]}))
            packet, ancestry = resolve_packet(head, CheckedBytes())
            self.assertEqual(packet['arrays'], {'new':2})
            self.assertEqual(packet['groups'], [1,2])
            self.assertEqual(len(ancestry), 2)
            self.assertEqual(packet['resolved_native_identity_revision_chain'], [{'file':'r','sha256':'a'}])
            bad = self.write(root/'bad.json', dict(schema='rinkfit.joint_packet.v1',dense_parent_packet=parent,source_row_ids=['fake']))
            with self.assertRaisesRegex(ValueError,'immutable prefix'): resolve_packet(bad,CheckedBytes())
            (root/'base.json').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'SHA mismatch'): resolve_packet(head,CheckedBytes())

    def test_strict_json_and_portable_cross_type_digest(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'): strict_json(b'{"x":1,"x":2}')
        with self.assertRaisesRegex(ValueError,'Nonfinite'): strict_json(b'{"x":NaN}')
        self.assertEqual(digest({'b':2,'a':1}),digest({'a':1,'b':2}))
        self.assertNotEqual(digest(1),digest(1.0))
        self.assertNotEqual(digest(0.0),digest(-0.0))
        self.assertEqual(digest({'file':'D:/one','sha256':'f'*64}),digest({'file':'C:/other','sha256':'f'*64}))
        self.assertNotEqual(digest({'file':'D:/one','sha256':'f'*64}),digest({'file':'D:/one','sha256':'e'*64}))

    def test_backing_sha_is_distinct_from_slice_sha_and_bundle_cold_read(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'x'; raw=np.array([4.,5.,6.],dtype='<f8').tobytes(); p.write_bytes(raw)
            ref=dict(file=str(p),sha256=hashlib.sha256(raw).hexdigest(),dtype='<f8',shape=[1],offset_bytes=8)
            a,check=CheckedBytes().array(ref,p.parent)
            self.assertEqual(a.tolist(),[5.])
            self.assertNotEqual(check['sha256'],check['slice_sha256'])
            a._mmap.close(); del a
            with self.assertRaisesRegex(ValueError,'slice SHA'): CheckedBytes().array(dict(ref,slice_sha256=ref['sha256']),p.parent)
            class Bundle:
                def read(self,ref): return raw
            p.write_bytes(b'bad local bytes')
            a,check=CheckedBytes(Bundle(),bundle_only=True).array(ref,p.parent)
            self.assertEqual(a.tolist(),[5.]); self.assertEqual(check['location'],'complete_zip')
            with self.assertRaisesRegex(ValueError,'Array SHA'): CheckedBytes().array(ref,p.parent)


if __name__=='__main__': unittest.main()
