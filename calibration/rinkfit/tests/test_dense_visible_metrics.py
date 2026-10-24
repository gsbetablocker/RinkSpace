"""Finite-domain and integrity checks for same-source visual diagnostics."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from rinkfit_joint import dense_visible_metrics as d

class DenseVisibleMetricsTests(unittest.TestCase):
    def test_restricted_station_domain_cannot_use_nearer_illegal_segment(self):
        dist,uv,s=d.nearest_segments(np.array([[2.,.5]]),np.array([0.,.5,1.]),
                                    np.array([[0.,0.],[1.,0.],[2.,0.]]),
                                    np.ones(3,bool),[[[0.,.25]]])
        self.assertAlmostEqual(dist[0],np.hypot(1.5,.5))
        np.testing.assert_allclose(uv,[[.5,0.]])
        self.assertEqual(s[0],.25)
    def test_long_segment_not_lost_by_nearest_vertex_shortcut(self):
        s=np.arange(4)/3
        uv=np.array([[-100.,0.],[100.,0.],[.1,2.],[.2,2.]])
        dist,_,_=d.nearest_segments(np.array([[0.,.9]]),s,uv,np.ones(4,bool),[[[0.,1.]]])
        self.assertAlmostEqual(dist[0],.9)
    def test_unresolved_sampling_is_not_visual_valid(self):
        good=np.array([d.VALID_BITS,d.VALID_BITS|d.SAMPLING_UNRESOLVED,d.VALID_BITS|d.QUERY_PENDING],np.uint8)
        np.testing.assert_array_equal(d.valid_status(good),[True,False,True])
        # Query's visual diagnostic is allowed; no formal certificate follows.
        self.assertTrue(bool(good[2]&d.QUERY_PENDING))
    def test_current_width_rechart_and_polyline_closure(self):
        def block(values):return {'name':'test','values':values,'lower':[-10.]*len(values),'upper':[10.]*len(values),'scales':[1.]*len(values),'constant':False}
        model={'scene_block':0,'blocks':[block([.5,.1,.1,4.,.02]),block([.1,.2,.3,.4]),block([0.,0.,.1,.2]),block([.1,.1,0.,.02])],
               'materials':[{'label':'ice.B1','paths':[{'id':'line','kind':5,'geometry_block':1}]},
                            {'label':'ice.C0','paths':[{'id':'circle','kind':4,'geometry_block':2}]},
                            {'label':'Ad','paths':[{'id':'open','kind':2,'geometry_block':-1,'advertisement_block':3,'closed':True,'points':[[0,0],[1,0],[1,1]]}]}]}
        # NativeScene validates unique block names, just as complete state does.
        for i,b in enumerate(model['blocks']):b['name']='b'+str(i)
        original=copy.deepcopy(model);scene=d.ReviewScene(model)
        x=scene.world(0,0,np.array([.5]),width=.01,sign=1)
        np.testing.assert_allclose(x,[[.1+.01*np.sin(.4),.2-.01*np.cos(.4),0.]])
        c=scene.world(1,0,np.array([0.]),width=.01,sign=1)
        np.testing.assert_allclose(c,[[.11*np.cos(.2),.11*np.sin(.2),0.]])
        np.testing.assert_array_equal(scene.knots(2,0),[0.,.5,1.])
        self.assertNotEqual(scene.world(2,0,np.array([0.])).tolist(),scene.world(2,0,np.array([1.])).tolist())
        self.assertEqual(model,original)
    def _fixture(self,root,name,status,dist):
        out=root/name;out.mkdir()
        arrays=d._open_outputs(out,len(status))
        for k,v in arrays.items():
            if k=='status':v[:]=status
            elif k=='distance':v[:]=dist
            elif k=='domain':v[:]=d.PHYSICAL|d.TRAINING|d.FIT
            elif k=='operator':v[:]=2
            elif k=='station':v[:]=.5
            elif k=='path':v[:]=0
            elif k=='nearest_uv':v[:]=0
            v.flush()
        common={'schema':'rinkfit.dense_visible_metrics.v1','source_packet':d.file_ref(root/'packet.json'),
                'selection':d.file_ref(root/'selection.json'),'model':d.file_ref(root/'model.json'),
                'model_source_clock_sha256':'clock','operator_source_files':{'test':d.file_ref(root/'operator.py')},
                'sampling':{'visibility':True,'tolerance':.05},'operator_codes':{'2':'visual'},
                'raw_rows':len(status),'processed_rows':len(status),'complete_raw_domain_processed':True,
                'arrays':{k:dict(d.file_ref(v.filename),dtype=v.dtype.str,shape=list(v.shape)) for k,v in arrays.items()}}
        file=out/'report.json';d._write(file,common);return file,common
    def test_complete_failure_differences_and_changed_array_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for file in ['packet.json','selection.json','model.json','operator.py']:(root/file).write_text('{}',encoding='utf-8')
            valid=d.VALID_BITS;invalid=d.PROCESSED|d.DOMAIN_FAILURE
            a,ra=self._fixture(root,'before',[valid,invalid,invalid,valid,valid,valid],[1,np.nan,np.nan,2,5,1])
            b,rb=self._fixture(root,'after',[invalid,valid,invalid,valid,valid,valid],[np.nan,2,np.nan,5,2,1])
            report=d.compare(a,b,root/'compare')
            self.assertEqual(report['counts']['newly_invalid'],1)
            self.assertEqual(report['counts']['recovered'],1)
            self.assertEqual(report['counts']['still_invalid'],1)
            np.testing.assert_array_equal(np.fromfile(report['sets']['worsened']['file'],dtype='<u8'),[3])
            np.testing.assert_array_equal(np.fromfile(report['sets']['improved']['file'],dtype='<u8'),[4])
            self.assertEqual(report['same_valid_sum_squared_delta_px'],0.)
            self.assertEqual(report['counts_by_domain']['fit']['newly_invalid'],1)
            class Inputs:
                n=6;packet_ref=ra['source_packet']
                packet={'shot_ids':['s01','s02'],'material_binding':{'materials':[{'index':0,'label':'Ad.edge'}]}}
                arrays={'shot_index':np.array([0,0,1,1,1,1]),'native_pts':np.array([1,1,2,2,2,2]),'material_index':np.zeros(6,int),'uv':np.arange(12).reshape(6,2).astype(float)}
                def source_id(self,row):return 'immutable/'+str(row)
            detailed=d.compare(a,b,root/'detailed',inputs=Inputs())
            automatic=json.loads((root/'detailed/automatic_frames.json').read_text())
            self.assertEqual(automatic['comparison']['sha256'],d.file_ref(root/'detailed/comparison.json')['sha256'])
            self.assertEqual({f['shot_id'] for f in automatic['frames']},{'s01','s02'})
            self.assertEqual(automatic['frames'][0]['raw_row'],0)
            self.assertEqual(automatic['frames'][0]['source_row_id'],'immutable/0')
            self.assertEqual(automatic['frames'][0]['source_uv'],[0.,1.])
            # A producer cannot silently edit a binary payload beneath its report.
            data=np.memmap(rb['arrays']['distance']['file'],mode='r+',dtype='<f4',shape=(6,))
            data[3]=3.;data.flush();del data
            with self.assertRaisesRegex(ValueError,'artifact changed'):d.compare(a,b,root/'tampered')
    def test_incomplete_scope_and_valid_nonfinite_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for file in ['packet.json','selection.json','model.json','operator.py']:(root/file).write_text('{}',encoding='utf-8')
            a,ra=self._fixture(root,'a',[d.VALID_BITS],[1.])
            b,rb=self._fixture(root,'b',[d.VALID_BITS],[np.nan])
            with self.assertRaisesRegex(ValueError,'nonfinite'):d.compare(a,b,root/'bad')
            rb['complete_raw_domain_processed']=False;d._write(b,rb)
            with self.assertRaisesRegex(ValueError,'Incomplete raw domain'):d.compare(a,b,root/'partial')

if __name__=='__main__':unittest.main()
