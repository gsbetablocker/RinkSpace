"""Focused numerical and semantic tests; no production shot fitting."""
import math
import unittest

import numpy as np

from rinkfit_start.registration import (
    LocalHomographyChart, MaterialSupport, RegistrationError, RegistrationObservations,
    RegistrationOptions, build_quadrature, fit_registration, native_replay,
    nearest_on_material, normalized_dlt, normalized_positive_softmin,
    pixel_center_transform, project_points, rank_diagnostics,
    scale_atlas_to_native, scale_frame_i_to_j, validate_material_domain,
)
from rinkfit_start.graph import GraphAnchor, PairwiseEdge, RegistrationGraph


class RegistrationGeometryTests(unittest.TestCase):
    def setUp(self):
        self.H = np.array([[380., 22., 960.], [7., 350., 540.], [.025, -.012, 1.]])

    def test_pixel_centers_and_direction(self):
        points = np.array([[0., 0.], [959.5, 539.5], [1919., 1079.]])
        s = .125
        np.testing.assert_allclose(project_points(pixel_center_transform(s), points), s*(points+.5)-.5)
        atlas = np.array([[-.6, -.4], [.2, .5], [.7, -.1]])
        np.testing.assert_allclose(project_points(scale_atlas_to_native(self.H, s), atlas),
                                   s*(project_points(self.H, atlas)+.5)-.5, atol=1e-10)
        G = np.array([[1.1, .02, 15.], [-.01, .95, -8.], [1e-5, 2e-5, 1.]])
        si, sj = .25, .5
        scaled_input = si*(points+.5)-.5
        expected = sj*(project_points(G, points)+.5)-.5
        np.testing.assert_allclose(project_points(scale_frame_i_to_j(G, si, sj), scaled_input), expected, atol=1e-9)

    def test_entity_dlt_and_h33_zero_chart(self):
        src = np.array([[1., 1.], [3., 1.], [3., 2.], [1., 2.], [2., 1.5]])
        H = np.array([[100., 4., 20.], [5., 90., 12.], [.2, .01, 0.]])
        dst = project_points(H, src)
        seed = normalized_dlt(src, dst, entity_evidence="synthetic physical marker identities")
        np.testing.assert_allclose(project_points(seed.matrix, src), dst, atol=1e-8)
        chart = LocalHomographyChart.from_homography(H, src)
        np.testing.assert_allclose(project_points(chart.matrix(np.zeros(8)), src), dst, atol=1e-8)
        with self.assertRaises(RegistrationError):
            normalized_dlt(src, dst, entity_evidence="")
        line = np.column_stack((np.arange(5.), np.zeros(5)))
        with self.assertRaises(RegistrationError):
            normalized_dlt(line, line+1, entity_evidence="physical but collinear entities")

    def test_finite_endpoint_counterexample_and_fixed_station(self):
        line = MaterialSupport.segment("L", [0., 0.], [10., 0.], provenance="declared finite material")
        predicted, station, boundary, _ = nearest_on_material(np.eye(3), [[20., 3.]], line)
        np.testing.assert_allclose(predicted, [[10., 0.]])
        np.testing.assert_allclose(station, [1.])
        self.assertTrue(boundary[0])
        self.assertAlmostEqual(np.linalg.norm(predicted[0]-[20., 3.]), math.sqrt(109.))
        fixed, station, _, _ = nearest_on_material(np.eye(3), [[9., 0.]], line, [0.])
        np.testing.assert_allclose(fixed, [[0., 0.]])
        with self.assertRaises(RegistrationError):
            RegistrationObservations([[9., 0.]], ["L"], [(256256, 12)], fixed_stations=[0.])

    def test_no_array_order_connectivity_and_no_id_deduplication(self):
        points = MaterialSupport.points("M", [[0., 0.], [10., 0.]], provenance="unordered point support")
        predicted, _, _, _ = nearest_on_material(np.eye(3), [[5., 0.]], points)
        self.assertAlmostEqual(float(np.linalg.norm(predicted[0]-[5., 0.])), 5.)
        with self.assertRaises(RegistrationError):
            MaterialSupport("M", "polyline", [[0., 0.], [10., 0.]], "array order alone is insufficient")
        observations = RegistrationObservations([[0., 0.], [10., 0.]], ["M", "M"], [(256256, 42), (256512, 42)])
        self.assertEqual(len(native_replay(np.eye(3), [points], observations).stations), 2)

    def test_normalized_softmin_positive_bias_and_scale(self):
        observed = np.array([[0., 0.], [2., 1.]])
        samples = np.array([[0., 0.], [10., 0.]])
        weights = np.array([.5, .5])
        energy = normalized_positive_softmin(observed, samples, weights, 2.)
        self.assertGreater(energy[0], 0.)
        s = .125
        scaled = normalized_positive_softmin(s*(observed+.5)-.5, s*(samples+.5)-.5, weights, s*2.)
        np.testing.assert_allclose(scaled, s*s*energy, rtol=1e-12)
        duplicated = normalized_positive_softmin(observed, np.repeat(samples, 2, axis=0), np.full(4, .25), 2.)
        np.testing.assert_allclose(duplicated, energy, atol=1e-12)
        with self.assertRaises(RegistrationError):
            normalized_positive_softmin(observed, samples, [1., 0.], 2.)

    def test_continuous_curves_and_unknown_circle_rank(self):
        circle = MaterialSupport.arc("circle", [0., 0.], 1., 0., 2*math.pi, provenance="known Euclidean circle")
        stations = np.linspace(.017, .987, 30)
        uv = project_points(self.H, circle.atlas_points(stations))
        predicted, recovered, _, _ = nearest_on_material(self.H, uv, circle)
        np.testing.assert_allclose(predicted, uv, atol=2e-7)
        observations = RegistrationObservations(uv, ["circle"]*len(uv), [(256256, i) for i in range(len(uv))])
        rank = rank_diagnostics(self.H, [circle], observations)
        self.assertLessEqual(rank.rank, 5)
        self.assertGreaterEqual(len(rank.weak_directions), 3)
        bezier = MaterialSupport.cubic_bezier("B", [[-1., 0.], [-.4, 1.], [.3, -1.], [1., .2]],
                                              provenance="declared continuous cubic branch")
        bezier_uv = project_points(self.H, bezier.atlas_points(np.linspace(.01, .99, 15)))
        pred, _, _, _ = nearest_on_material(self.H, bezier_uv, bezier)
        np.testing.assert_allclose(pred, bezier_uv, atol=2e-6)

    def test_sampling_fill_and_explicit_horizon_limit(self):
        arc = MaterialSupport.arc("arc", [0., 0.], 1., 0., math.pi, provenance="finite semicircle")
        quadrature = build_quadrature(arc, self.H, 8., max_samples=4097)
        self.assertLessEqual(quadrature.native_fill_bound(self.H), 1.+1e-10)
        self.assertTrue(np.all(quadrature.weights > 0))
        self.assertAlmostEqual(quadrature.weights.sum(), 1.)
        H_horizon = np.array([[100., 0., 1.], [0., 100., 2.], [1., 0., .1]])
        with self.assertRaisesRegex(RegistrationError, "not an observation-error"):
            validate_material_domain(H_horizon, [arc])

    def test_small_multiscale_refinement_and_immutable_snapshot(self):
        corners = np.array([[-1., -.6], [1., -.6], [1., .6], [-1., .6]])
        materials, all_uv, mids, keys = [], [], [], []
        stations = np.linspace(.1, .9, 10)
        for i in range(4):
            material = MaterialSupport.segment(str(i), corners[i], corners[(i+1)%4], provenance="synthetic finite rectangle edge")
            materials.append(material)
            all_uv.extend(project_points(self.H, material.atlas_points(stations)))
            mids.extend([str(i)]*len(stations))
            keys.extend((256256, 100*i+j) for j in range(len(stations)))
        uv = np.array(all_uv)
        metadata = {"nested": {"caller_value": [1, 2]}}
        observations = RegistrationObservations(uv, mids, keys, metadata=metadata)
        initial = np.array([[1., -.015, 20.], [.01, 1., -15.], [0., 0., 1.]]) @ self.H
        result = fit_registration(materials, observations, initial, frame_id=1, plane_id="ice",
                                  options=RegistrationOptions(capture_iterations_per_stage=2, native_iterations=20,
                                                               max_sampling_rebuilds_per_stage=0))
        self.assertLess(result.best_native.replay.max_error_native_px, 1e-3)
        self.assertEqual(result.best_native.rank.rank, 8)
        self.assertEqual({entry["scale"] for entry in result.optimization_log if "scale" in entry}, {.125, .25, .5, 1.})
        remembered = result.best_native.observations.native_uv.copy()
        uv[:] = -999.
        metadata["nested"]["caller_value"][0] = 999
        np.testing.assert_array_equal(result.best_native.observations.native_uv, remembered)
        self.assertEqual(result.best_native.observations.metadata["nested"]["caller_value"][0], 1)
        with self.assertRaises(ValueError):
            result.best_native.H_atlas_to_native.setflags(write=True)
        with self.assertRaises(TypeError):
            result.best_native.objective_settings["new"] = True


class RegistrationGraphTests(unittest.TestCase):
    def setUp(self):
        self.witness = np.array([[100., 100.], [600., 100.], [600., 500.], [100., 500.], [300., 250.]])
        self.H = {1: np.eye(3),
                  2: np.array([[1.05, .02, 30.], [-.01, 1.02, -5.], [2e-5, -1e-5, 1.]]),
                  3: np.array([[.95, -.01, -10.], [.02, .98, 15.], [-1e-5, 1e-5, 1.]])}

    def edge(self, i, j, suffix="", matrix=None, original=True):
        G = self.H[j] @ np.linalg.inv(self.H[i]) if matrix is None else matrix
        return PairwiseEdge(i, j, G, "ice", f"independent synthetic physical evidence {i},{j} {suffix}",
                            self.witness, original_evidence=original, data_rank=8,
                            edge_id=f"e{i}{j}{suffix}", derived_from_H=not original)

    def test_full_1248_ledger_and_multi_anchor_bidirectional_propagation(self):
        graph = RegistrationGraph(range(1, 1249), "ice", [self.edge(1, 2), self.edge(3, 2)],
                                  frame_information_scores={2: 10.})
        anchors = [GraphAnchor(1, self.H[1], "shared_atlas", "synthetic anchor 1"),
                   GraphAnchor(3, self.H[3], "shared_atlas", "synthetic anchor 3")]
        initialized = graph.initialize(anchors)
        self.assertEqual(len(initialized.frame_states), 1248)
        self.assertEqual(len(initialized.frame_ledger[2].candidates), 2)
        self.assertIsNone(initialized.frame_ledger[1248].H_atlas_to_native)
        candidate_from_1_to_3 = [c for c in initialized.frame_ledger[3].candidates if c.origin_anchor_frame == 1][0]
        self.assertEqual(candidate_from_1_to_3.edge_path, (("e12", True), ("e32", False)))
        comp = initialized.components[0]
        np.testing.assert_allclose(project_points(initialized.frame_ledger[comp.gauge_frame].H_atlas_to_native, self.witness),
                                   self.witness, atol=1e-9)
        empty = RegistrationGraph(range(1, 1249), "ice").initialize()
        self.assertEqual(len(empty.frame_states), 1248)
        self.assertTrue(all(s.H_atlas_to_native is None for s in empty.frame_states))

    def test_real_edge_cycle_displacement_and_synthetic_rejection(self):
        e12, e23 = self.edge(1, 2), self.edge(2, 3)
        perfect31 = self.H[1] @ np.linalg.inv(self.H[3])
        offset = np.array([[1., 0., 3.], [0., 1., 0.], [0., 0., 1.]])
        e31 = self.edge(3, 1, matrix=offset @ perfect31)
        synthetic = self.edge(3, 1, suffix="synth", original=False)
        graph = RegistrationGraph([1, 2, 3, 4], "ice", [e12, e23, e31, synthetic])
        loop = graph.check_cycle([("e12", True), ("e23", True), ("e31", True)])
        self.assertAlmostEqual(loop.max_displacement_native_px, 3., places=7)
        self.assertEqual(len(graph.fundamental_cycles()), 1)
        with self.assertRaises(RegistrationError):
            graph.check_cycle([("e12", True), ("e23", True), ("e31synth", True)])
        synthetic_only = RegistrationGraph([1, 2, 3, 4], "ice", [synthetic])
        self.assertEqual(len(synthetic_only.connected_components()), 4)

    def test_joint_sparse_graph_refinement(self):
        graph = RegistrationGraph([1, 2, 3, 4], "ice", [self.edge(1, 2), self.edge(2, 3), self.edge(3, 1)],
                                  frame_information_scores={1: 100.})
        shift2 = np.array([[1., .005, 12.], [0., 1., -7.], [0., 0., 1.]])
        shift3 = np.array([[1., -.004, -9.], [.003, 1., 6.], [0., 0., 1.]])
        anchors = [GraphAnchor(1, self.H[1], "shared", "synthetic reference"),
                   GraphAnchor(2, shift2 @ self.H[2], "shared", "perturbed synthetic seed 2"),
                   GraphAnchor(3, shift3 @ self.H[3], "shared", "perturbed synthetic seed 3")]
        initialized = graph.initialize(anchors, gauge_mode="external_fixed_atlas")
        result = graph.refine(initialized, max_iterations=20)
        self.assertGreater(result.initial_native_witness_rms_px, 1.)
        self.assertLess(result.final_native_witness_rms_px, 1e-5)
        self.assertEqual(result.actual_residual_rank, 16)
        self.assertEqual(len(result.frame_states), 4)
        self.assertIsNone(result.frame_ledger[4].H_atlas_to_native)
        np.testing.assert_allclose(result.frame_ledger[1].H_atlas_to_native,
                                   initialized.frame_ledger[1].H_atlas_to_native, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
