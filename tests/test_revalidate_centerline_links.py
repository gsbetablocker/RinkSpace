"""Cached RGB edges cannot keep a removed source-shape bridge alive."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

PROGRAMME = Path(__file__).resolve().parents[1]/'events/ankara_2026/programmes/okada_mei_sp'
sys.path.insert(0, str(PROGRAMME))
spec = importlib.util.spec_from_file_location('revalidate_links', PROGRAMME/'revalidate_centerline_links.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RevalidatedLinkTests(unittest.TestCase):
    def setUp(self):
        self.nodes = [('s02', pts, 0) for pts in (256, 512, 768)]
        ids = np.array([0, 2, 4, 6])
        uv = np.zeros((7, 2))
        uv[ids, 0] = [10., 40., 70., 100.]
        query = np.zeros(7, int)
        query[ids] = np.arange(4)
        self.frames = {node[:2]: dict(ids=ids, uv=uv.copy(), query=query.copy(), fragments=[dict(candidate_ids=ids.tolist())])
                       for node in self.nodes}
        self.qualified = {node: set(ids) for node in self.nodes}
        self.edge = dict(unambiguous=True, pairs=[[0, 0], [1, 1], [2, 2]], support=3, previous_support=3)

    def edges(self):
        kept = []
        for first, last in zip(self.nodes[:-1], self.nodes[1:]):
            edge, _ = module.revalidate_edge(self.edge, first, last, self.qualified, self.frames.__getitem__)
            if edge is not None:
                kept.append(edge)
        return kept

    def seed(self):
        node = self.nodes[0]
        rows = [dict(candidate=cid, uv=self.frames[node[:2]]['uv'][cid].tolist()) for cid in self.qualified[node]]
        support = module.seed_support(rows, self.frames[node[:2]]['uv'][[0, 2, 4]], self.frames[node[:2]]['query'])
        return {node} if support['qualified'] else set()

    def test_removed_bridge_candidate_disconnects_original_chain(self):
        self.qualified[self.nodes[1]].remove(4)
        edges = self.edges()
        self.assertEqual(edges, [])
        self.assertEqual(module.anchored_nodes(self.qualified, edges, self.seed()), {self.nodes[0]})

    def test_still_valid_chain_keeps_original_node_order_and_reaches_seed(self):
        edges = self.edges()
        self.assertEqual(len(edges), 2)
        self.assertEqual(edges[0]['pairs'], [[0, 0], [1, 1], [2, 2]])
        self.assertEqual(edges[0]['qualified_target_span_px'], 60.)
        self.assertEqual(module.anchored_nodes(self.qualified, edges, self.seed()), set(self.nodes))
        ambiguous = dict(self.edge, unambiguous=False)
        edge, reason = module.revalidate_edge(ambiguous, self.nodes[0], self.nodes[1], self.qualified, self.frames.__getitem__)
        self.assertIsNone(edge)
        self.assertEqual(reason, 'original_edge_not_unambiguous')

    def test_no_requalified_seed_admits_no_point_despite_valid_chain(self):
        rows = [dict(candidate=cid, uv=[500.+index*30., 500.]) for index, cid in enumerate((0, 2, 4))]
        support = module.seed_support(rows, self.frames[self.nodes[0][:2]]['uv'][[0, 2, 4]], self.frames[self.nodes[0][:2]]['query'])
        self.assertEqual(support['count'], 0)
        self.assertEqual(module.anchored_nodes(self.qualified, self.edges(), set()), set())

    def test_offset_sampling_phase_still_seeds_the_same_finite_source_line(self):
        rows = [dict(candidate=i, uv=[float(x), 0.]) for i, x in enumerate((0, 30, 60, 90))]
        old_uv = np.array([[15., 0.], [45., 0.], [75., 0.]])
        support = module.seed_support(rows, old_uv, np.arange(4))
        self.assertEqual(support['count'], 3)
        self.assertEqual(support['span_px'], 60.)
        self.assertTrue(support['qualified'])

    def test_missing_query_gap_cannot_create_a_source_seed(self):
        rows = [dict(candidate=i, uv=[float(x), 0.]) for i, x in enumerate((0, 30, 150, 180))]
        old_uv = np.array([[45., 0.], [90., 0.], [135., 0.]])
        support = module.seed_support(rows, old_uv, np.array([0, 1, 4, 5]))
        self.assertEqual(support['count'], 0)
        self.assertFalse(support['qualified'])

    def anchor_rows(self):
        return {node: {cid: dict(candidate=int(cid), pts=node[1], identity='lane2:edge0',
                                path_id='native/lane2', uv=self.frames[node[:2]]['uv'][cid].tolist())
                       for cid in ids}
                for node, ids in self.qualified.items()}

    def test_valid_reviewed_source_anchor_connects_previously_seedless_chain(self):
        with tempfile.TemporaryDirectory(prefix='rink_link_anchor_') as directory:
            root = Path(directory)
            image = root/'reviewed_source.png'
            image.write_bytes(b'unchanged independently reviewed source evidence fixture')
            path = root/'anchors.json'
            document = dict(anchors=[dict(shot='s02', pts=768, identity='lane2:edge0',
                path_id='native/lane2', candidate_ids=[0, 2, 4], evidence_image=image.name,
                reason='Independent source image and known camera/topology identify this long lane')])
            path.write_text(json.dumps(document), encoding='utf-8')
            seeds, checks = module.reviewed_anchor_seeds(document, self.anchor_rows(), path)
            self.assertEqual(seeds, {self.nodes[2]})
            self.assertEqual(module.anchored_nodes(self.qualified, self.edges(), seeds), set(self.nodes))
            self.assertEqual(checks[0]['original_fragment_hits'][0]['fragment'], list(self.nodes[2]))
            self.assertEqual(checks[0]['original_fragment_hits'][0]['qualified_source_uv_span_px'], 60.)
            self.assertEqual(checks[0]['evidence_image_sha256'], module.prepared.digest(image))

    def test_reviewed_anchor_rejects_candidate_removed_from_qualified_source(self):
        with tempfile.TemporaryDirectory(prefix='rink_link_anchor_') as directory:
            root = Path(directory)
            image = root/'reviewed_source.png'
            image.write_bytes(b'original evidence fixture')
            document = dict(anchors=[dict(shot='s02', pts=768, identity='lane2:edge0',
                path_id='native/lane2', candidate_ids=[0, 2, 5], evidence_image=image.name,
                reason='Independent source review')])
            with self.assertRaisesRegex(ValueError, 'not in qualified accepted'):
                module.reviewed_anchor_seeds(document, self.anchor_rows(), root/'anchors.json')


if __name__ == '__main__':
    unittest.main()
