import unittest
import numpy as np
from rinkspace.domain.motion_chart import cubic_samples
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline,refine_uniform_controls


class JointTimelineTest(unittest.TestCase):
    def test_exact_refinement_and_independent_clock_support(self):
        rng=np.random.default_rng(41);old=rng.normal(size=(44,3))
        fine,first=refine_uniform_controls(old,-.8,.4,.02)
        t=np.linspace(-.39,15.99,1333)
        a,w=cubic_samples(44,-.8,.4,t);b,v=cubic_samples(len(fine),first,.02,t)
        np.testing.assert_allclose(np.einsum('ij,ijk->ik',w,old[a]),np.einsum('ij,ijk->ik',v,fine[b]),atol=2e-13,rtol=0)
        shots=[]
        for i in range(2):
            shots.append(dict(id=f's{i}',control_begin=i*len(fine),control_stop=(i+1)*len(fine),row_begin=2*i,row_stop=2*i+2,
                reference_pts=1000000*i,first_pts=1000000*i,last_pts=1000000*i+198000,stop_pts_exclusive=1000000*i+198001,
                time_base_num=1,time_base_den=12800,first_control_time=first,knot_stride=10,motion_reference_time=0.))
        tl=ShotMotionTimeline(shots,.02);pts=np.array([0,198000,1000000,1198000]);times=tl.row_times(pts)
        np.testing.assert_allclose(np.diff(times.reshape(2,2),axis=1),198000/12800,atol=1e-14)
        tl.assert_support_separation(pts,.04)
        stacked=np.r_[fine,fine];a,w=cubic_samples(len(stacked),0.,.02,times)
        b,v=cubic_samples(len(fine),first,.02,np.tile([0.,198000/12800],2))
        np.testing.assert_allclose(np.einsum('ij,ijk->ik',w,stacked[a]),np.einsum('ij,ijk->ik',v,fine[b]),atol=3e-13,rtol=0)
        directions,labels=tl.directions();second=[j for j,l in enumerate(labels) if l[0]=='s1']
        self.assertEqual(np.count_nonzero(directions[:len(fine),:,second]),0)
        first_cols=[j for j,l in enumerate(labels) if l[0]=='s0']
        self.assertEqual(len(second),len(first_cols)+2)
        with self.assertRaises(ValueError):tl.assert_support_separation(pts,2.)

    def test_each_physical_camera_has_one_reference_angle_gauge(self):
        shots=[]
        for i,camera in enumerate([0,1,0,1]):
            shots.append(dict(id=f's{i}',camera_index=camera,control_begin=i*23,control_stop=(i+1)*23,
                first_control_time=-.4,knot_stride=10,motion_reference_time=0.))
        timeline=ShotMotionTimeline(shots,.02);directions,labels=timeline.directions()
        counts=[sum(label[0]==shot['id'] for label in labels) for shot in shots]
        self.assertEqual(counts[0],counts[1]);self.assertEqual(counts[2],counts[0]+2);self.assertEqual(counts[3],counts[1]+2)
        for shot in shots:
            outside=np.ones(len(directions),bool);outside[shot['control_begin']:shot['control_stop']]=False
            columns=[j for j,label in enumerate(labels) if label[0]==shot['id']]
            np.testing.assert_array_equal(directions[outside][:,:,columns],0.)

if __name__=='__main__':unittest.main()
