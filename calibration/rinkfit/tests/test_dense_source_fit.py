from copy import deepcopy
import unittest
from rinkfit_joint.dense_source_fit import _decision, _key


class DenseMaterialAdmissionGuards(unittest.TestCase):
    def setUp(self):
        self.row=dict(uv=[100.,200.],time_base='1/12800',shot_id='s06',native_pts=1301248,physical_surface='ice',source_UV_is_direct_measurement=True,feature_id='ice.C0:outer',display_class='ice',measurement_definition='width_midpoint_between_both_material_edges')
        self.mapping={'ice.C0:outer':dict(material_index=144,physical_point=False,measurement_definitions=['width_midpoint_between_both_material_edges'])}

    def test_inferred_curve_is_not_pixel_observation(self):
        row=dict(self.row,source_UV_is_direct_measurement=False)
        self.assertEqual(_decision(row,self.mapping)[0],1)

    def test_wide_paint_is_not_target_ring_boundary(self):
        row=dict(self.row,measurement_definition='declared_target_ring_material_boundary')
        self.assertEqual(_decision(row,self.mapping)[0],5)

    def test_image_texture_does_not_become_world_point_by_name(self):
        row=dict(self.row,display_class='print')
        self.assertEqual(_decision(row,self.mapping)[0],7)

    def test_unknown_material_never_uses_model_nearest(self):
        row=dict(self.row,feature_id='unbound.ice.test')
        self.assertEqual(_decision(row,self.mapping)[0],3)

    def test_static_surface_and_native_clock_required(self):
        self.assertEqual(_decision(dict(self.row,physical_surface='stands'),self.mapping)[0],2)
        self.assertEqual(_decision(dict(self.row,time_base='1/60'),self.mapping)[0],6)

    def test_same_exposure_parent_is_not_counted_twice(self):
        keys={_key('s06',1301248,144,[100.,200.])}
        self.assertEqual(_decision(self.row,self.mapping,keys)[0],9)
        self.assertEqual(_decision(dict(self.row,native_pts=1301504),self.mapping,keys)[0],0)

    def test_no_fixed_world_point_without_correspondence(self):
        m=deepcopy(self.mapping);m['ice.C0:outer']['physical_point']=True
        self.assertEqual(_decision(self.row,m)[0],8)

if __name__=='__main__':unittest.main()
