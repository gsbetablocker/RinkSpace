"""Source-authority regressions using actual SourceInstanceGate and file hashes.

The small registry proves only its two test rows. No pixels, world coordinates,
noise model, or production source permission is invented by these fixtures.
"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from rinkfit_joint.dense_source_factory import source_reader_kind
from rinkfit_joint.dense_typed_source_fit import SourceInstanceGate


def pin(path):
    return {'file':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


class SourceFactoryContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name)
        p=root/'packet.json';p.write_text('{}')
        q=root/'qualification.json';q.write_text('{}')
        consumer=root/'consumer.py';consumer.write_text('# frozen issuer fixture\n')
        self.row={'selected_row_id':'dense/s02/600320/17','shot_id':'s02','native_pts':600320,
                  'uv':[100.25,200.5],'physical_instance_qualification':'eligible'}
        positive={'selected_row_id':self.row['selected_row_id'],'shot_id':'s02','native_pts':600320,
                  'source_uv':[100.25,200.5],'material_index':144,'instance_state':1,
                  'instance_support_kind':1,'evidence':pin(q)}
        negative=dict(positive,selected_row_id='dense/s02/600320/18',instance_state=0,instance_support_kind=0)
        self.registry=root/'instance_registry.json'
        self.registry.write_text(json.dumps({'schema':'rinkfit.dense_positive_finite_source_instance_registry.v1',
            'unreviewed_rows_default_to_confirmed':False,'paired_profile_or_repeated_pixels_grant_identity':False,
            'actual_consumer':pin(consumer),'source_packet':pin(p),'source_geometry_qualification':pin(q),
            'evidence_refs':[pin(q)],'current_source_parts':[],'rows':[positive,negative],'reviewed_rows':2}))
        self.gate=SourceInstanceGate(self.registry)
        self.legacy={'schema':'rinkfit.dense_source_fit_input.v1'}
        self.typed=dict(self.legacy,positive_finite_instance_required=True,
                        instance_support_gate=pin(self.registry),legacy_material_geometry_reader=pin(consumer),frame_role_authority=pin(q))
        self.qual_path=q

    def test_new_legacy_packet_rejected_but_historical_identity_remains_readable(self):
        with self.assertRaises(ValueError):source_reader_kind(self.legacy,purpose='new_packet')
        self.assertEqual(source_reader_kind(self.legacy,purpose='historical_audit'),'legacy')

    def test_typed_input_always_dispatches_to_typed_reader(self):
        for purpose in ('new_packet','historical_audit'):
            self.assertEqual(source_reader_kind(self.typed,purpose=purpose),'typed')

    def test_flag_without_pinned_registry_cannot_grant_fresh_admission(self):
        broken=dict(self.typed);del broken['instance_support_gate']
        with self.assertRaises(ValueError):source_reader_kind(broken,purpose='new_packet')

    def test_missing_frame_roles_cannot_issue_fresh_packet(self):
        old=dict(self.typed);del old['frame_role_authority']
        with self.assertRaisesRegex(ValueError,'frame roles'):source_reader_kind(old,purpose='new_packet')
        self.assertEqual(source_reader_kind(old,purpose='historical_audit'),'typed')

    def test_actual_gate_accepts_only_its_exact_positive_source_identity(self):
        self.assertEqual(self.gate.check(self.row,144),0)
        with self.assertRaises(ValueError):self.gate.check(dict(self.row,uv=[100.5,200.5]),144)
        with self.assertRaises(ValueError):self.gate.check(self.row,145)

    def test_repeated_pixels_do_not_promote_unreviewed_identity(self):
        self.assertEqual(self.gate.check(dict(self.row,selected_row_id='unreviewed'),144),11)

    def test_explicit_unknown_source_is_rejected_even_with_a_former_positive_record(self):
        self.assertEqual(self.gate.check(dict(self.row,physical_instance_qualification='unknown'),144),10)
        self.assertEqual(self.gate.check(dict(self.row,source_fit_eligible=False),144),10)
        self.assertEqual(self.gate.check(dict(self.row,selected_row_id='dense/s02/600320/18'),144),10)

    def test_changed_source_evidence_is_rejected_by_actual_hash_check(self):
        self.qual_path.write_text('{"changed":true}')
        with self.assertRaises(ValueError):SourceInstanceGate(self.registry)


if __name__=='__main__':unittest.main()
