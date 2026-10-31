"""Independent extension preserves frozen arrays and the true native PTS span."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

PROGRAMME = Path(__file__).resolve().parents[1]/'events/ankara_2026/programmes/okada_mei_sp'
sys.path.insert(0, str(PROGRAMME))
spec = importlib.util.spec_from_file_location('native_extension_prepare', PROGRAMME/'prepare_native_extension.py')
extension = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extension)


class NativeExtensionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='rink_extension_')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.parent = self.root/'baseline'; self.parent.mkdir()
        self.layout = dict(size=10, layout_revision='layout', scene_content_revision='scene')
        self.model = SimpleNamespace(initial_state=SimpleNamespace(layout=self.layout))
        self.scene, self.library = self.root/'scene.json', self.root/'native.dll'
        self.scene.write_text('{}', encoding='utf-8')
        self.library.write_bytes(b'no library is loaded in synthetic tests')
        profile, coupling = self.root/'profile.json', self.root/'coupling.json'
        profile.write_text('{"profile":"frozen"}', encoding='utf-8')
        coupling.write_text('{"coupling":"frozen"}', encoding='utf-8')
        self.parameterization = dict(profile_path=str(profile), profile_sha256=extension.prepared.digest(profile),
            coupling_path=str(coupling), coupling_sha256=extension.prepared.digest(coupling), stage='pupil')
        self.mapping = dict(path_ids=['old/path0', 'old/path1'], scene_revision='scene')
        self.points = self.points_for(1000+np.arange(12)*256, 'old')
        self.points['path_indices'] = np.arange(12, dtype=np.int32)%2
        self.shots = [dict(id=f's{i+1:02d}', row_begin=2*i, row_stop=2*i+2,
            control_begin=8*i, control_stop=8*i+8, reference_pts=1000+512*i,
            first_pts=1000+512*i, last_pts=1256+512*i, stop_pts_exclusive=1512+512*i,
            time_base_num=1, time_base_den=12800, first_control_time=-.06,
            knot_stride=1, motion_reference_time=0., camera_index=[0,1,2,0,2,1][i]) for i in range(6)]
        rig = np.arange(129, dtype=float).reshape(3,43)/100
        rig[:,26:28] = [959.5,539.5]; rig[:,[28,29,30]] = 0
        self.state = dict(rig=rig, controls=np.arange(144, dtype=float).reshape(48,3)/100,
            stations=np.where(self.points['kind']==0,self.points['stations'],.61),
            source_rows=self.points['source_rows'].copy(), scene_values=np.arange(10,dtype=float),
            board=np.arange(6,dtype=float), layout=np.arange(146,dtype=float))
        np.savez(self.parent/'point_observations.npz', **self.points)
        self.manifest = dict(schema='rinkspace.multicamera_disjoint_shots.v1', shots=self.shots,
            cameras=[dict(id=x) for x in ('main','secondary','end')], control_dt=.02,
            first_control_time=0., max_readout=.04,
            point_input_sha256=extension.prepared.digest(self.parent/'point_observations.npz'),
            scene_path=str(self.scene), scene_file_sha256=extension.prepared.digest(self.scene), scene_revision='scene',
            native_library=str(self.library), native_library_sha256=extension.prepared.digest(self.library),
            camera_parameterization=self.parameterization, source_inputs={})
        extension.prepared.write(self.parent/'mapping.json', self.mapping)
        extension.prepared.write(self.parent/'joint_manifest.json', self.manifest)
        extension.prepared.bind_snapshot(self.state,self.layout,self.mapping,self.manifest)
        self.resume=self.parent/'frozen_state.npz'; np.savez(self.resume,**self.state)
        self.baseline_record=self.root/'baseline.json'
        files=dict(manifest=self.parent/'joint_manifest.json', mapping=self.parent/'mapping.json',
            points=self.parent/'point_observations.npz', snapshot=self.resume, scene=self.scene, library=self.library)
        extension.prepared.write(self.baseline_record, dict(inputs={k:dict(path=str(p),sha256=extension.prepared.digest(p))
            for k,p in files.items()}, source_shots=[s['id'] for s in self.shots], camera_parameterization=self.parameterization))
        self.scope_path=self.root/'scope.json'
        self.scope=dict(source=dict(sha256='source-video-sha'), source_clock=dict(native_pts_step=256),
            shots=[dict(id=s['id'],rig_id=('main','secondary','end')[s['camera_index']]) for s in self.shots])
        for i, rig_id in [(7,'main'),(8,'end')]:
            first=4072+(i-7)*768
            self.scope['shots'].append(dict(id=f's{i:02d}',rig_id=rig_id,
                processing=dict(first_pts=first,last_pts=first+512,stop_pts_exclusive=first+768,native_frame_count=3)))
        extension.prepared.write(self.scope_path,self.scope)
        self.packet=self.make_packet('s07',0,4072)

    def points_for(self, pts, prefix):
        n=len(pts)
        return dict(pts=np.array(pts,np.int64), observation_ids=np.array([f'{prefix}-{i}' for i in range(n)],dtype='U96'),
            uv=np.column_stack([np.arange(n)+20.,np.arange(n)+30.]), kind=np.arange(n,dtype=np.int64)%2,
            stations=np.full(n,.25), source_rows=np.arange(n,dtype=np.int64)+100,
            path_indices=np.zeros(n,np.int32))

    def make_packet(self, shot_id, camera_index, first):
        folder=self.root/shot_id; folder.mkdir()
        z=self.points_for(first+np.arange(3)*256,shot_id)
        z['path_indices']=np.array([1,0,1],np.int32)
        np.savez(folder/'point_observations.npz',**z)
        np.save(folder/'controls.npy',np.arange(27,dtype=float).reshape(9,3)/100)
        extension.prepared.write(folder/'mapping.json',dict(path_ids=['old/path1','new/path'],
            source_video_sha256='source-video-sha', pixel_domain='native_distorted', image_size=[1920,1080]))
        extension.prepared.write(folder/'shot_seed.json',dict(id=shot_id,camera_index=camera_index,reference_pts=first,
            time_base_num=1,time_base_den=12800,first_control_time=-.06,control_dt=.02,knot_stride=1,
            motion_reference_time=0.,controls_ref='controls.npy',independent_source_capture=True,
            source_video_sha256='source-video-sha'))
        return folder

    def run_prepare(self, packets=None, sparse=False):
        native=SimpleNamespace(paths={p:{} for p in ['old/path0','old/path1','new/path']},scene_revision='scene')
        def initialize(native_arg,**kw):
            self.assertTrue(np.all(kw['kinds']==1))
            self.assertEqual(kw['camera']['rig'],self.state['rig'][kw['camera']['camera_index']].tolist())
            return np.full(len(kw['stations']),.73),dict(unresolved=[],groups=1)
        args=SimpleNamespace(run_root=self.parent,resume=self.resume,output=self.root/'output',
                             shot_packets=packets or [self.packet],allow_sparse_bootstrap=sparse)
        with patch.object(extension,'BASELINE_RECORD',self.baseline_record), patch.object(extension,'SCOPE_PATH',self.scope_path), \
             patch.object(extension,'SceneParameterModel',return_value=self.model), \
             patch.object(extension,'ParameterState',return_value=SimpleNamespace()), \
             patch.object(extension,'NativeCore'), patch.object(extension,'NativeSceneGPU',return_value=native), \
             patch.object(extension,'initialize_curve_stations',side_effect=initialize):
            return extension.prepare(args)

    def test_append_preserves_frozen_bytes_paths_and_current_model_dependencies(self):
        second=self.make_packet('s08',2,4840)
        protected={str(p):extension.prepared.digest(p) for p in self.parent.iterdir()}
        receipt=self.run_prepare([second,self.packet])
        out=self.root/'output'; points=extension.prepared.npz(out/'point_observations.npz')
        state=extension.prepared.npz(out/'initial_pixels.npz'); mapping=extension.prepared.read(out/'mapping.json')
        manifest=extension.prepared.read(out/'joint_manifest.json')
        for key in self.points:
            self.assertEqual(points[key][:12].dtype,self.points[key].dtype)
            self.assertEqual(points[key][:12].tobytes(),self.points[key].tobytes())
        for key in ('rig','scene_values','board','layout'):
            self.assertEqual(state[key].tobytes(),self.state[key].tobytes())
        self.assertEqual(state['controls'][:48].tobytes(),self.state['controls'].tobytes())
        self.assertEqual(state['stations'][:12].tobytes(),self.state['stations'].tobytes())
        np.testing.assert_array_equal(state['stations'][12:],[.25,.73,.25,.25,.73,.25])
        np.testing.assert_array_equal(points['stations'][12:],np.full(6,.25))
        np.testing.assert_array_equal(points['path_indices'][12:],[2,1,2,2,1,2])
        np.testing.assert_array_equal(points['pts'][12:],[4072,4328,4584,4840,5096,5352])
        self.assertEqual(mapping['path_ids'],['old/path0','old/path1','new/path'])
        self.assertEqual(manifest['shots'][:6],self.shots)
        self.assertEqual(manifest['camera_parameterization'],self.parameterization)
        self.assertEqual(manifest['scene_path'],str(self.scene))
        self.assertEqual(manifest['max_readout'],self.manifest['max_readout'])
        self.assertFalse(receipt['partial_source_coverage']); self.assertFalse(receipt['fit_executed'])
        extension.prepared.validate_snapshot(state,self.layout,mapping,manifest,points)
        for path,digest in protected.items():self.assertEqual(extension.prepared.digest(path),digest)

    def test_sparse_bootstrap_keeps_complete_scope_without_filling_pts(self):
        z=extension.prepared.npz(self.packet/'point_observations.npz')
        np.savez(self.packet/'point_observations.npz',**{k:v[1:2] for k,v in z.items()})
        with self.assertRaisesRegex(ValueError,'Incomplete shot exposure'):
            self.run_prepare()
        self.assertFalse((self.root/'output').exists())
        receipt=self.run_prepare(sparse=True)
        out=self.root/'output'; m=extension.prepared.read(out/'joint_manifest.json')
        z=extension.prepared.npz(out/'point_observations.npz'); s=m['shots'][-1]
        self.assertEqual((s['first_pts'],s['last_pts'],s['stop_pts_exclusive']),(4072,4584,4840))
        np.testing.assert_array_equal(z['pts'][12:],[4328])
        self.assertEqual(s['source_coverage']['missing_exposures'],2)
        self.assertTrue(m['partial_source_coverage']); self.assertTrue(receipt['partial_source_coverage'])

    def test_seed_source_rig_and_pts_are_validated(self):
        seed_path=self.packet/'shot_seed.json'; seed=extension.prepared.read(seed_path)
        for key,value,pattern in [('independent_source_capture',False,'independent capture'),
                                  ('camera_index',1,'physical rig'),('source_video_sha256','other','SHA256'),
                                  ('control_dt',.04,'without resampling')]:
            changed=dict(seed);changed[key]=value;extension.prepared.write(seed_path,changed)
            with self.assertRaisesRegex(ValueError,pattern):self.run_prepare()
            self.assertFalse((self.root/'output').exists())
        extension.prepared.write(seed_path,seed)
        z=extension.prepared.npz(self.packet/'point_observations.npz');z['pts'][1]+=1
        np.savez(self.packet/'point_observations.npz',**z)
        with self.assertRaisesRegex(ValueError,'registered shot exposure clock'):self.run_prepare(sparse=True)

    def test_lossy_new_ids_do_not_widen_old_source_array(self):
        z=extension.prepared.npz(self.packet/'point_observations.npz')
        z['observation_ids']=np.array(['x'*97,'y','z'])
        np.savez(self.packet/'point_observations.npz',**z)
        with self.assertRaisesRegex(ValueError,'baseline dtype: observation_ids'):self.run_prepare()


if __name__=='__main__':
    unittest.main()
