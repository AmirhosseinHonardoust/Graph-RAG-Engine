import tempfile
import unittest
from pathlib import Path

from graph.graph_store import GraphStore


class GraphStoreTests(unittest.TestCase):
    def _build_graph(self):
        graph = GraphStore()
        graph.add_doc("doc_a", "Doc A", "file://doc_a.md")
        graph.add_doc("doc_b", "Doc B", "file://doc_b.md")
        graph.add_chunk("chunk_a", "Graph retrieval text", "doc_a")
        graph.add_chunk("chunk_b", "Graph search text", "doc_b")
        for concept in ["graph", "retrieval"]:
            graph.add_concept(concept)
            graph.link_mentions("chunk_a", concept)
        graph.add_concept("graph")
        graph.link_mentions("chunk_b", "graph")
        graph.compute_doc_pagerank()
        return graph

    def test_neighbor_chunks_by_shared_concepts(self):
        graph = self._build_graph()

        neighbors = graph.neighbor_chunks_by_concepts("chunk_a")

        self.assertEqual(neighbors, ["chunk_b"])

    def test_explain_paths_returns_doc_and_concepts(self):
        graph = self._build_graph()

        paths = graph.explain_paths(["chunk_a"])

        self.assertEqual(paths[0]["doc_id"], "doc_a")
        self.assertEqual(paths[0]["doc_title"], "Doc A")
        self.assertIn("graph", paths[0]["concepts"])

    def test_save_and_load_roundtrip(self):
        graph = self._build_graph()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "graph.pkl"
            graph.save(path)
            loaded = GraphStore.load(path)

        self.assertEqual(loaded.get_doc_info("doc_a")["title"], "Doc A")


if __name__ == "__main__":
    unittest.main()
