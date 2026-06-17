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
            path = Path(tmp) / "graph.json"
            graph.save(path)
            loaded = GraphStore.load(path)

        self.assertEqual(loaded.get_doc_info("doc_a")["title"], "Doc A")

    def test_pagerank_links_docs_sharing_a_concept(self):
        graph = self._build_graph()  # doc_a and doc_b both mention "graph"

        related_edges = {
            (u[1], v[1])
            for u, v, data in graph.G.edges(data=True)
            if data.get("type") == "RELATED_DOC"
        }
        self.assertIn(("doc_a", "doc_b"), related_edges)
        self.assertIn(("doc_b", "doc_a"), related_edges)
        self.assertGreater(graph.get_doc_info("doc_a")["pagerank"], 0.0)

    def test_pagerank_leaves_unrelated_docs_unconnected(self):
        graph = GraphStore()
        graph.add_doc("doc_a", "Doc A", "data/docs/a.md")
        graph.add_doc("doc_c", "Doc C", "data/docs/c.md")
        graph.add_chunk("chunk_a", "graph text", "doc_a")
        graph.add_chunk("chunk_c", "streamlit text", "doc_c")
        for chunk_id, concept in [("chunk_a", "graph"), ("chunk_c", "streamlit")]:
            graph.add_concept(concept)
            graph.link_mentions(chunk_id, concept)
        graph.compute_doc_pagerank()

        related_edges = {
            (u[1], v[1])
            for u, v, data in graph.G.edges(data=True)
            if data.get("type") == "RELATED_DOC"
        }
        self.assertEqual(related_edges, set())


if __name__ == "__main__":
    unittest.main()
