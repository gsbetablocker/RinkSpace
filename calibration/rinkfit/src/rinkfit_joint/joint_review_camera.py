"""Read-only cameras derived from the complete current model and native clock.

This adapter never reloads the historical initialization graph. It preserves
current reference angles, spline controls, optics, and the shared physical rig.
"""
import copy
import numpy as np
from .joint_model import native_camera_from_joint

class CurrentJointCameras:
    def __init__(self, model):
        self.model = model
        self._cameras = {}
        self.graph = {'source_clock': copy.deepcopy(model['source_clock']),
                      'shots': copy.deepcopy(model['shots']),
                      'rigs': copy.deepcopy(model['rigs'])}
        for rig in self.graph['rigs'].values():
            block = model['blocks'][rig['optics_block']]
            rig['tau_bound'] = max(abs(block['lower'][11]), abs(block['upper'][11]))

    def native_camera(self, shot_id):
        if shot_id not in self._cameras:
            self._cameras[shot_id] = native_camera_from_joint(self.model, shot_id)
        return self._cameras[shot_id]

    def local_seconds(self, shot_id, native_pts):
        from .dense_fixed_anchor_guard import source_local_seconds
        return source_local_seconds(self.model, shot_id, np.asarray(native_pts))

    def validate(self, require_ready=False):
        records = []
        for sid, shot in self.model['shots'].items():
            scope = shot['scope']
            stamps = np.asarray([scope['first_pts'], scope['last_pts']], dtype=np.int64)
            local = self.local_seconds(sid, stamps)
            camera = self.native_camera(sid)
            controls = camera.spline(local)
            if not np.isfinite(controls).all():
                raise ValueError('Current camera endpoint controls are nonfinite: ' + sid)
            records.append({'shot_id': sid, 'native_pts': stamps.tolist(),
                            'local_seconds': local.tolist()})
        return {'schema': 'rinkfit.current_model_camera_adapter.v1',
                'current_complete_model_consumed': True,
                'historical_initializer_graph_loaded': False,
                'source_clock_checked': True, 'shots': records,
                'geometry_or_physical_quality_certified': False}
