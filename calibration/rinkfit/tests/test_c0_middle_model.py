"""Small topology/parameter tests for the explicit current circle transition."""
import copy
import unittest
from rinkfit_joint.camera_graph import GraphError
from rinkfit_joint.joint_model import revise_c0_midline_without_a,validate_current_ice_definition


def fixture():
    def block(name,values):
        return dict(name=name,values=values,lower=[-1.]*len(values),upper=[1.]*len(values),scales=[.1]*len(values),fixed_indices=[],constant=False)
    def material(label,bi,ri=2):
        return dict(label=label,paths=[dict(id=label,variant_id=label,kind=1,geometry_block=bi,advertisement_block=-1,radius_index=ri,closed=True)])
    return dict(schema='rinkfit.joint_event_model.v1',blocks=[block('scene/rink5',[.5]*5),block('ice/circle/C0',[.01,.02,.07,.08]),block('ice/circle/A2',[.2,.1,.06,.07]),block('measurement/paint_half_width/test',[.001]),block('ice/circle/F1',[.2,.1,.02,.04,.06])],materials=[material('ice.C0:outer',1),material('ice.C0:paint_outer_boundary',1,3),material('ice.A2:outer',2),material('ice.A2:paint_outer_boundary',2,3),material('ice.F1:outer',4,4)],measurement_widths=[dict(width_scope_id='c0section',half_width_block=3)],stroke_pair_model_bindings=[dict(pair_id='c0pair',width_scope_id='c0section',half_width_block=3,material_index=0)],scene_block=0,shots={},rigs={},linear_control_manifolds=[dict(block=4)],physical_equalities=[],physical_inequalities=[])


class CurrentCircleDefinition(unittest.TestCase):
    def test_real_deletion_and_exact_retained_subset(self):
        old=fixture();before=copy.deepcopy(old);new,report=revise_c0_midline_without_a(old)
        self.assertEqual(old,before)
        self.assertEqual(report['old_block_to_new'],[0,1,-1,-1,2])
        self.assertEqual(report['old_material_to_new'],[0,-1,-1,-1,1])
        self.assertEqual(report['old_path_to_new'],[[0],[-1],[-1],[-1],[0]])
        self.assertEqual(new['blocks'][1]['values'],old['blocks'][1]['values'][:3])
        self.assertEqual(new['blocks'][2],old['blocks'][4])
        self.assertEqual(new['materials'][1]['paths'][0]['geometry_block'],2)
        self.assertEqual(new['linear_control_manifolds'][0]['block'],2)
        self.assertEqual(new['materials'][0]['label'],'ice.C0:middle')
        self.assertEqual(report['removed_stroke_pair_ids'],['c0pair'])
        self.assertEqual(new['measurement_widths'],[])
        self.assertEqual(new['stroke_pair_model_bindings'],[])
        self.assertEqual(revise_c0_midline_without_a(new)[0],new)
        validate_current_ice_definition(new)

    def test_no_deleting_shared_live_geometry_or_width(self):
        old=fixture();old['materials'][4]['paths'][0]['geometry_block']=2
        with self.assertRaises(GraphError):revise_c0_midline_without_a(old)
        old=fixture();old['stroke_pair_model_bindings'].append(dict(pair_id='F1pair',width_scope_id='c0section',half_width_block=3,material_index=4))
        with self.assertRaises(GraphError):revise_c0_midline_without_a(old)

    def test_current_display_rejects_stale_extra_circle(self):
        new,_=revise_c0_midline_without_a(fixture())
        new['materials'].append(copy.deepcopy(new['materials'][0]))
        with self.assertRaises(GraphError):validate_current_ice_definition(new)


if __name__=='__main__':unittest.main()
