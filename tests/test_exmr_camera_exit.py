"""Real CPU torch regression of the ExMR exit patch; no weights or inference.

Requires an external pinned ExMR checkout and torch; otherwise skips. Override its
location with RINKSPACE_EXMR_ROOT. Neural modules and all world helpers are tiny
fakes; camera geometry, rotation conversions and smoothing use the upstream code.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest import mock

try:
    import torch
except (ImportError, OSError):
    torch = None

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from rinkspace.storage import AssetStore

spec = importlib.util.spec_from_file_location(
    "_exmr_camera_patch", PROJECT / "environments/apply_exmr_camera_patch.py")
patch_entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch_entry)


def _load(name, source, filename):
    module = types.ModuleType(name)
    module.__file__ = str(filename)
    sys.modules[name] = module
    exec(compile(source, str(filename), "exec"), module.__dict__)
    return module


@unittest.skipIf(torch is None, "requires optional torch runtime")
class ExMRCameraExitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(os.environ.get("RINKSPACE_EXMR_ROOT") or
                        AssetStore(PROJECT).data / "runtime_tools/body/ExMR")
        cls.source = cls.root / patch_entry.RELATIVE_SOURCE
        if not cls.source.is_file():
            raise unittest.SkipTest("requires external pinned ExMR source; no downloads performed")
        try:
            cls.original = subprocess.check_output([
                "git", "-C", str(cls.root), "show",
                f"{patch_entry.COMMIT}:{patch_entry.RELATIVE_SOURCE}"], text=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise unittest.SkipTest(f"requires pinned upstream Git source: {exc}")
        if patch_entry._source_sha(cls.source) != patch_entry.PATCHED_SHA256:
            raise AssertionError("apply the reviewed environments/exmr_camera.patch first")

    def setUp(self):
        modules = {}
        for name in ("exmr", "exmr.geometry", "exmr.model", "exmr.body_models",
                     "safetensors", "safetensors.torch", "exmr.model.global_traj",
                     "exmr.body_models.smplx_model", "exmr.model.body_net",
                     "exmr.model.decoder", "exmr.model.refiners"):
            modules[name] = types.ModuleType(name)
            modules[name].__path__ = []
        modules["safetensors.torch"].load_file = self.forbidden
        modules["exmr.body_models.smplx_model"].SmplxModel = object
        modules["exmr.model.body_net"].BodyNet = object
        modules["exmr.model.body_net"].HAND_POSE = slice(0, 180)
        modules["exmr.model.decoder"].Decoder = object
        modules["exmr.model.refiners"].FootRefiner = object
        modules["exmr.model.refiners"].HandRefiner = object
        imports = mock.patch.dict(sys.modules, modules)
        self.addCleanup(imports.stop)
        imports.start()
        self.rot = self.load_geometry("rotation_conversions")
        self.load_geometry("camera")
        self.load_geometry("smoothing")
        self.world = modules["exmr.model.global_traj"]
        self.world_calls = []

        def world_fn(name, operation):
            def run(*args):
                self.world_calls.append(name)
                return operation(*args)
            setattr(self.world, name, run)

        world_fn("rollout_global_params", lambda gv, vel, *args:
                 dict(global_orient=gv.clone() + .07, transl=vel.clone() + 2))
        world_fn("static_cam_transl", lambda incam, glob, *args: glob["transl"] + 3)
        world_fn("static_joint_transl", lambda glob, *args: glob["transl"] + 4)
        world_fn("snap_to_ground", lambda glob, *args: glob["transl"] + 5)
        world_fn("static_joint_ik", lambda glob, *args: glob["body_pose"].clone() + .06)
        world_fn("put_on_ground", lambda glob, *args: glob["transl"] + 7)
        world_fn("face_forward", lambda glob, *args:
                 (glob["global_orient"] + .08, glob["transl"] + 9))
        self.patched = _load("_patched_exmr", self.source.read_text(encoding="utf-8"), self.source)
        self.upstream = _load("_upstream_exmr", self.original, "pinned_upstream_exmr.py")
        self.inputs = dict(
            kp2d=torch.zeros(1, 3, 23, 3),
            bbx_xys=torch.tensor([[[12., 10., 20.]]]).expand(1, 3, 3).clone(),
            K_fullimg=torch.tensor([[100., 0., 10.], [0., 100., 10.], [0., 0., 1.]])
                .expand(1, 3, 3, 3).clone(),
            cam_angvel=torch.zeros(1, 3, 6), body_feat=torch.zeros(1, 3, 2),
            hand_feat=torch.zeros(1, 3, 2, 2))
        self.disabled = dict(postprocess=False, snap_to_ground=False,
                             smooth_incam=False, smooth_global=False)

    def load_geometry(self, name):
        path = self.root / "exmr/geometry" / f"{name}.py"
        return _load(f"exmr.geometry.{name}", path.read_text(encoding="utf-8"), path)

    @staticmethod
    def forbidden(*args, **kwargs):
        raise AssertionError("world, smoothing, model construction or checkpoint loading was called")

    def model(self, module):
        rot = self.rot

        class TinyBody(torch.nn.Module):
            def forward(self, length, obs, *args):
                b, l = obs.shape[:2]
                return dict(x=torch.zeros(b, l, 180),
                            pred_cam=torch.tensor([1., .25, -.125]).expand(b, l, 3).clone(),
                            static_conf_logits=torch.arange(6.).expand(b, l, 6).clone())

        class TinyDecoder(torch.nn.Module):
            def decode(self, x):
                b, l = x.shape[:2]
                return dict(body_pose=torch.zeros(b, l, 63), betas=torch.ones(b, l, 10),
                            global_orient=torch.tensor([.1, .2, .3]).expand(b, l, 3).clone(),
                            left_hand_pose=torch.zeros(b, l, 45),
                            right_hand_pose=torch.zeros(b, l, 45),
                            global_orient_gv=torch.zeros(b, l, 3),
                            local_transl_vel=torch.ones(b, l, 3))

            def fk_rot(self, pose, orient):
                return torch.eye(3).expand(*pose.shape[:2], 22, 3, 3).clone()

            def denormalize(self, x, which):
                return x

        class TinyFoot(torch.nn.Module):
            def forward(self, length, obs, cliff, rotations):
                aa = torch.zeros(*obs.shape[:2], 2, 3)
                aa[..., 2] = .2
                return rot.matrix_to_rotation_6d(rot.axis_angle_to_matrix(aa)).flatten(-2)

        class TinyHand(torch.nn.Module):
            def forward(self, length, feat, cliff, rotations):
                aa = torch.zeros(*feat.shape[:2], 34, 3)
                aa[..., :2, 2], aa[..., 2:4, 2], aa[..., 4:, 2] = .3, .8, .4
                return rot.matrix_to_rotation_6d(rot.axis_angle_to_matrix(aa)).flatten(-2)

        model = module.ExMR.__new__(module.ExMR)
        torch.nn.Module.__init__(model)
        model.body_net, model.decoder = TinyBody(), TinyDecoder()
        model.foot_refiner, model.hand_refiner = TinyFoot(), TinyHand()
        model.body_model = self.forbidden
        model.register_buffer("cam_angvel_mean", torch.zeros(6))
        model.register_buffer("cam_angvel_std", torch.ones(6))
        return model

    def test_camera_exit_preserves_independent_refinement_stages(self):
        for name in vars(self.world):
            if not name.startswith("_"):
                setattr(self.world, name, self.forbidden)
        self.patched.gaussian_smooth = self.forbidden
        self.patched.gaussian_smooth_rotations = self.forbidden
        result = self.model(self.patched)(**self.inputs, **self.disabled, camera_only=True,
                                          length=torch.tensor([2]))
        self.assertEqual(set(result), {"smplx_incam_prefoot", "smplx_incam_postfoot",
                                       "smplx_incam", "static_conf_logits"})
        pre, post, final = (result[key] for key in
                           ("smplx_incam_prefoot", "smplx_incam_postfoot", "smplx_incam"))
        torch.testing.assert_close(pre["body_pose"], torch.zeros(1, 3, 63))
        torch.testing.assert_close(post["body_pose"][..., 18:24],
                                   torch.tensor([0., 0., .2, 0., 0., .2]).expand(1, 3, 6))
        torch.testing.assert_close(final["body_pose"][..., 18:24], post["body_pose"][..., 18:24])
        torch.testing.assert_close(post["body_pose"][..., 51:63], torch.zeros(1, 3, 12))
        torch.testing.assert_close(final["body_pose"][..., 51:63],
                                   torch.tensor([0., 0., .3, 0., 0., .3,
                                                 0., 0., .5, 0., 0., .5]).expand(1, 3, 12))
        for key in ("left_hand_pose", "right_hand_pose"):
            torch.testing.assert_close(post[key], torch.zeros(1, 3, 45))
            self.assertGreater(final[key].abs().sum().item(), 0)
        for key in ("betas", "global_orient", "transl"):
            torch.testing.assert_close(pre[key], post[key], rtol=0, atol=0)
            torch.testing.assert_close(pre[key], final[key], rtol=0, atol=0)
        torch.testing.assert_close(final["global_orient"],
                                   torch.tensor([.1, .2, .3]).expand(1, 3, 3))
        torch.testing.assert_close(final["transl"],
                                   torch.tensor([.45, -.125, 10.]).expand(1, 3, 3))
        with torch.inference_mode():
            for key in final:
                self.assertEqual(len({pre[key].data_ptr(), post[key].data_ptr(),
                                      final[key].data_ptr()}), 3)
                before, after = pre[key].clone(), post[key].clone()
                final[key].add_(100)
                torch.testing.assert_close(pre[key], before, rtol=0, atol=0)
                torch.testing.assert_close(post[key], after, rtol=0, atol=0)
        self.assertEqual(self.world_calls, [])

    def test_camera_only_rejects_each_processing_flag_before_network(self):
        model = self.model(self.patched)
        model.body_net.forward = self.forbidden
        for flag in self.disabled:
            with self.subTest(flag=flag):
                flags = {**self.disabled, flag: True}
                with self.assertRaisesRegex(ValueError, "camera_only=True requires"):
                    model(**self.inputs, **flags, camera_only=True)
        with self.assertRaisesRegex(ValueError, "camera_only=True requires"):
            model(**self.inputs, camera_only=True)

    def test_normal_exit_matches_pinned_upstream_in_all_world_branches(self):
        for flags in (self.disabled, {}, dict(static_cam=True, snap_to_ground=True,
                                              smooth_incam=True)):
            with self.subTest(flags=flags):
                self.world_calls.clear()
                expected = self.model(self.upstream)(**self.inputs, **flags)
                calls = self.world_calls.copy()
                self.world_calls.clear()
                actual = self.model(self.patched)(**self.inputs, **flags, camera_only=False)
                self.assertEqual(self.world_calls, calls)
                self.assertIn("rollout_global_params", calls)
                self.assertEqual(set(actual), set(expected))
                for key, value in expected.items():
                    if isinstance(value, dict):
                        self.assertEqual(set(actual[key]), set(value))
                        for field in value:
                            torch.testing.assert_close(actual[key][field], value[field], rtol=0, atol=0)
                    else:
                        torch.testing.assert_close(actual[key], value, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
