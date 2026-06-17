import compileall
import pickle
import unittest
from pathlib import Path

import numpy as np

from graph.graph_store import GraphStore

ROOT = Path(__file__).resolve().parents[1]

INDEX_DIR = ROOT / "data" / "index"
INDEX_AVAILABLE = all(
    (INDEX_DIR / name).exists() for name in ("chunks.pkl", "docs.pkl", "vectors.npy", "graph.pkl")
)
INDEX_SKIP_REASON = "Index artifacts not built; run `python -m ingest.ingest_docs`."


class ProjectIntegrityTests(unittest.TestCase):
    def test_source_files_compile(self):
        for folder in ["backend", "graph", "ingest", "ui"]:
            self.assertTrue(
                compileall.compile_dir(ROOT / folder, quiet=1),
                msg=f"Python files in {folder} failed to compile",
            )

    @unittest.skipUnless(INDEX_AVAILABLE, INDEX_SKIP_REASON)
    def test_saved_index_artifacts_are_consistent(self):
        index_dir = ROOT / "data" / "index"
        with (index_dir / "chunks.pkl").open("rb") as f:
            chunks = pickle.load(f)
        with (index_dir / "docs.pkl").open("rb") as f:
            docs = pickle.load(f)
        vectors = np.load(index_dir / "vectors.npy")

        self.assertGreater(len(docs), 0)
        self.assertGreater(len(chunks), 0)
        self.assertEqual(len(chunks), vectors.shape[0])
        self.assertTrue(all("id" in chunk and "text" in chunk for chunk in chunks))

    @unittest.skipUnless(INDEX_AVAILABLE, INDEX_SKIP_REASON)
    def test_index_urls_are_repo_relative(self):
        index_dir = ROOT / "data" / "index"
        with (index_dir / "docs.pkl").open("rb") as f:
            docs = pickle.load(f)
        with (index_dir / "chunks.pkl").open("rb") as f:
            chunks = pickle.load(f)

        urls = [doc.get("url", "") for doc in docs]
        urls.extend(chunk.get("url", "") for chunk in chunks)

        self.assertTrue(urls)
        for url in urls:
            self.assertTrue(
                url.startswith("data/docs/"),
                msg=f"Index URL should be repo-relative, got {url!r}",
            )
            self.assertNotIn("C:\\", url)
            self.assertNotIn("file://", url)
            self.assertNotIn("\\", url)

        graph = GraphStore.load(index_dir / "graph.pkl")
        for doc in docs:
            info = graph.get_doc_info(doc["id"])
            self.assertEqual(info["url"], doc["url"])


if __name__ == "__main__":
    unittest.main()
