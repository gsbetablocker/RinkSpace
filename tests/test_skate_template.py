"""Small geometric checks for the new template, no motion fitting or GUI."""
from collections import Counter
from dataclasses import replace
import unittest

import numpy as np

from calibration.program_tracking.body_tracking.skate_template import SkateParameters, build_skate, profile_height, transform_contact_primitives, FT, _arc_point


class SkateTemplateTests(unittest.TestCase):
    def test_metric_default_has_closed_outward_solid_parts_and_open_upper_contract(self):
        t=build_skate();v,f=t['vertices'],t['faces']
        self.assertEqual(len(t['contact_arcs']),4)
        self.assertEqual(t['invariants']['upper'],'not generated')
        self.assertGreaterEqual(v[:,2].min(),-1e-12)
        for part in t['part_ranges']:
            triangles=f[t['face_parts']==part]
            edges=Counter(tuple(sorted((int(a),int(b)))) for tri in triangles for a,b in zip(tri,np.roll(tri,-1)))
            self.assertTrue(all(n==2 for n in edges.values()),part)
            p=v[triangles]
            self.assertTrue(np.all(np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)>1e-14),part)
            volume=np.einsum('ij,ij->i',p[:,0],np.cross(p[:,1],p[:,2])).sum()/6
            self.assertGreater(volume,0.,part)

    def test_two_official_primary_candidates_have_tangent_continuity(self):
        for r in (7*FT,8*FT):
            t=build_skate(SkateParameters(main_rocker_m=r))
            for main,secondary in zip(t['contact_arcs'][::2],t['contact_arcs'][1::2]):
                theta=main['theta_end_rad']
                np.testing.assert_allclose(_arc_point(main,theta),_arc_point(secondary,secondary['theta_start_rad']),atol=1e-12)
                def tangent(a,s):
                    d=-np.array(a['axis_cos'])*np.sin(s)+np.array(a['axis_sin'])*np.cos(s)
                    return d/np.linalg.norm(d)
                np.testing.assert_allclose(tangent(main,theta),tangent(secondary,secondary['theta_start_rad']),atol=1e-12)
            self.assertAlmostEqual(float(profile_height(.4*.25,SkateParameters(main_rocker_m=r))),0.,places=12)

    def test_mounts_derive_from_sole_and_posts_stay_connected(self):
        t=build_skate();v=t['vertices']
        for patch in t['mounting_patches']:
            plate=t['part_ranges'][patch['id']+'_mount_plate']
            z=v[slice(*plate),2]
            self.assertAlmostEqual(z.max(),patch['plane_z_m'],places=12)
            self.assertEqual(patch['top_gap_m'],0.)
        for post in t['post_connections']:
            points=v[slice(*t['part_ranges'][post['id']])]
            self.assertAlmostEqual(points[:,2].max(),post['plate_bottom_z_m'],places=12)
            self.assertGreater(post['minimum_web_overlap_m'],.00099)

    def test_display_density_does_not_change_exact_contact(self):
        a=build_skate();b=build_skate(replace(SkateParameters(),render_step_m=.002))
        self.assertNotEqual(len(a['vertices']),len(b['vertices']))
        self.assertEqual(a['contact_arcs'],b['contact_arcs'])
        np.testing.assert_array_equal(a['contact_point_groups'][0]['points'],b['contact_point_groups'][0]['points'])

    def test_rigid_transform_and_input_guards(self):
        t=build_skate();angle=.7
        R=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
        shift=np.array([2.,-1.,.3]);out=transform_contact_primitives(t,R,shift)
        for a,b in zip(t['contact_arcs'],out['arcs']):
            theta=(a['theta_start_rad']+a['theta_end_rad'])/2
            np.testing.assert_allclose(_arc_point(b,theta),R@_arc_point(a,theta)+shift,atol=1e-12)
        for kwargs in (dict(sole_length_m=250),dict(blade_width_m=.10),dict(front_mount_height_m=.006),dict(main_rocker_m=.1),dict(secondary_rocker_m=.005)):
            with self.assertRaises(ValueError,msg=str(kwargs)):build_skate(replace(SkateParameters(),**kwargs))
        with self.assertRaises(ValueError):transform_contact_primitives(t,np.diag([-1.,1.,1.]),np.zeros(3))


if __name__=='__main__':unittest.main()
