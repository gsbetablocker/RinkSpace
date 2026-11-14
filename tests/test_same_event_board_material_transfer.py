"""Same-event role transfer cannot transfer old pixels or human approval."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rinkspace.vision_source_topology import (
    same_event_board_foot_material_authority, read_board_foot_material_profiles,
    _validate_same_event_board_foot_authority,
)


class SameEventBoardMaterialTransferTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        folder = Path(self.directory.name)
        semantics = dict(event_id='test_event',authority=dict(kind='direct_user_confirmation'),
                         definitions=dict(yellow_bottom_strip_belongs_to='board'))
        semantic_path = folder/'event_semantics.json'
        semantic_path.write_text(json.dumps(semantics),encoding='utf8')
        self.original_source = dict(source_sha256='a'*64,source_pts=123,time_base='1/12800',
                                    decoded_native_RGB_sha256='b'*64)
        packet = dict(schema='rinkspace.ice-contact-material-role-confirmation.v1',event_id='test_event',
            authority=dict(kind='direct_user_confirmation',human_authority=True,user_text='reference event semantics'),
            authority_scope=dict(semantic_role_authoritative=True,automatically_confirms_new_UV=False),
            definition=dict(bottom_strip_region_belongs_to='board',bottom_strip_is_ice=False,
                bottom_strip_material_in_reference='very_narrow_pale_yellow_board_bottom_strip',
                lower_strip_edge='ice_contact_board_bottom_strip_to_ice',
                yellow_reflection_is_not_board_material=True),reference_source=deepcopy(self.original_source),
            existing_event_semantics=dict(path=str(semantic_path),sha256=hashlib.sha256(semantic_path.read_bytes()).hexdigest()))
        contract = folder/'board_source_roles.json'
        contract.write_text(json.dumps(packet),encoding='utf8')
        self.reference = dict(path=str(contract),sha256=hashlib.sha256(contract.read_bytes()).hexdigest())
        self.rgb = np.full((96,120,3),235,np.uint8)
        self.rgb[:10] = [35,80,180]
        self.rgb[45:55] = [235,220,180]
        self.binding = dict(event_id='test_event',source_sha256='c'*64,source_pts=256,
                            time_base='1/12800',native_hw=[96,120],shot_id='new_shot')
        self.rgb_sha = hashlib.sha256(self.rgb.tobytes()).hexdigest()
        uv = np.array([[[20,5],[20,20],[20,50],[20,75]],[[80,5],[80,20],[80,50],[80,75]]])
        self.evidence = dict(source_binding=deepcopy(self.binding),RGB_sha256=self.rgb_sha,
            material_order=['blue_cap','white_board_face','yellow_board_foot','white_ice'],
            actual_current_RGB_read=True,yellow_reflection_is_not_board_material=True,
            sample_uv=uv.tolist(),sample_RGB=self.rgb[uv[...,1],uv[...,0]].tolist())

    def authority(self, evidence=None):
        return same_event_board_foot_material_authority(self.reference,self.binding,
            material_order_evidence=self.evidence if evidence is None else evidence,rgb_sha256=self.rgb_sha)

    def test_transfer_retains_reference_and_enters_actual_profile_reader(self):
        before = Path(self.reference['path']).read_bytes()
        authority = self.authority()
        self.assertEqual(authority['kind'],'same_event_material_role_transfer')
        self.assertEqual(authority['reference_source'],self.original_source)
        self.assertFalse(authority['human_authority'])
        self.assertFalse(authority['human_pixel_confirmation'])
        result = read_board_foot_material_profiles(self.rgb,
            dict(source_binding=self.binding,instances=[]),dict(source_binding=self.binding),
            material_role_authority=authority,device='cpu',
            source_query_points_uv=np.column_stack((np.arange(20,101,8),np.full(11,54.))))
        self.assertEqual(len(result['readings']),2)
        self.assertIn('original_RGB_profiles',result)
        self.assertFalse(result['direct_user_material_role_confirmed'])
        self.assertTrue(result['same_event_material_role_transferred'])
        self.assertFalse(result['human_pixel_confirmation'])
        self.assertTrue(all(not r['source_pixel_accuracy_accepted'] for r in result['readings']))
        self.assertEqual(Path(self.reference['path']).read_bytes(),before)

    def test_changed_query_exposure_or_rgb_is_rejected(self):
        authority = self.authority()
        for key,value in dict(event_id='other_event',source_sha256='d'*64,source_pts=512,
                              time_base='1/6400',native_hw=[97,120],shot_id='other_shot').items():
            with self.subTest(key=key),self.assertRaises(ValueError):
                _validate_same_event_board_foot_authority(authority,{**self.binding,key:value},self.rgb)
        altered = self.rgb.copy();altered[0,0,0] += 1
        with self.assertRaises(ValueError):
            _validate_same_event_board_foot_authority(authority,self.binding,altered)
        evidence = deepcopy(self.evidence);evidence['sample_RGB'][0][0][0] += 1
        with self.assertRaises(ValueError):
            _validate_same_event_board_foot_authority(self.authority(evidence),self.binding,self.rgb)

    def test_reference_role_and_claim_tampering_is_rejected(self):
        with self.assertRaises(ValueError):
            same_event_board_foot_material_authority({**self.reference,'sha256':'0'*64},self.binding,
                material_order_evidence=self.evidence,rgb_sha256=self.rgb_sha)
        for field,value in (('material_order',list(reversed(self.evidence['material_order']))),
                            ('yellow_reflection_is_not_board_material',False)):
            with self.subTest(field=field),self.assertRaises(ValueError):
                self.authority({**self.evidence,field:value})
        for field in ('human_authority','human_pixel_confirmation','automatically_confirms_new_UV'):
            changed=self.authority();changed[field]=True
            with self.subTest(field=field),self.assertRaises(ValueError):
                _validate_same_event_board_foot_authority(changed,self.binding,self.rgb)
        changed=self.authority();changed['reference_source']['source_sha256']=self.binding['source_sha256']
        with self.assertRaises(ValueError):
            _validate_same_event_board_foot_authority(changed,self.binding,self.rgb)

    def test_default_direct_authority_still_rejects_another_source(self):
        direct=dict(kind='direct_user_confirmation',reference=self.reference['path'],source_sha256='a'*64,
                    board_foot_material='warm_yellow_strip_belongs_to_board')
        with self.assertRaisesRegex(ValueError,'source-bound'):
            read_board_foot_material_profiles(self.rgb,dict(source_binding=self.binding,instances=[]),
                dict(source_binding=self.binding),material_role_authority=direct,device='cpu')


if __name__ == '__main__':
    unittest.main()
