import unittest
from unittest.mock import patch
import numpy as np
from rinkspace.vision_board_known_interface_profiles import read_known_board_interface_profiles
from rinkspace.vision_board_context_seed_reader import ContextTemplateBoardSourceReader
from rinkspace.vision_board_material_point_profiles import BoardMaterialPointSourceReader


class KnownBoardInterfaceTests(unittest.TestCase):
    def scene(self):
        x=np.arange(1920,dtype=float)
        progress=np.clip((x-894.)/28.,0.,1.)
        colors=np.array([160.,45.,45.])[None]*(1-progress[:,None])+np.array([220.,220.,220.])[None]*progress[:,None]
        image=np.broadcast_to(colors[None],(1080,1920,3)).astype(np.float32).copy()
        guess=np.array([[911.,500.]])
        normal=np.array([[-1.,0.]])
        direction=np.array([[160.,45.,45.]])-np.array([[220.,220.,220.]])
        direction/=np.linalg.norm(direction,axis=1,keepdims=True)
        return image,np.zeros((1080,1920),bool),guess,normal,direction

    def test_wide_current_transition_reads_its_own_midpoint_and_blur(self):
        args=self.scene();rows=read_known_board_interface_profiles(*args)
        self.assertEqual(len(rows),1)
        self.assertAlmostEqual(rows[0][1][0],908.,places=3)
        self.assertAlmostEqual(rows[0][4][0],-7.,places=3)
        self.assertAlmostEqual(rows[0][4][1],7.,places=3)
        self.assertNotEqual(rows[0][1][0],args[2][0,0])

    def test_competing_actual_material_excursion_stays_missing(self):
        image,mask,guess,normal,direction=self.scene()
        image[:,916:922]=[40.,40.,40.]
        self.assertEqual(read_known_board_interface_profiles(image,mask,guess,normal,direction),[])

    def test_measured_person_context_is_not_cleared(self):
        image,mask,guess,normal,direction=self.scene();mask[498:503,907:910]=True
        self.assertEqual(read_known_board_interface_profiles(image,mask,guess,normal,direction),[])

    def test_unknown_membership_cannot_enable_context_operator(self):
        reader=ContextTemplateBoardSourceReader.__new__(ContextTemplateBoardSourceReader)
        track={'template':{'anchor':{'physical_material_id':'Ad1.right_vertical','edge':'right_vertical',
            'curve':{'kind':'advertisement_boundary'},'template_descriptors_from_confirmed_original_entity_context':True}}}
        with patch.object(BoardMaterialPointSourceReader,'_read_edge',return_value=[]),patch(
                'rinkspace.vision_board_context_seed_reader.read_known_board_interface_profiles') as operator:
            self.assertEqual(reader._read_edge(None,None,None,track,known_source_board_material=None),[])
            operator.assert_not_called()


if __name__=='__main__':unittest.main()
