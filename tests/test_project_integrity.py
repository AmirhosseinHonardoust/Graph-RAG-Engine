import compileall
import pickle
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class ProjectIntegrityTests(unittest.TestCase):
    def test_source_files_compile(self):
        for folder in ["backend", "graph", "ingest", "ui"]:
            self.assertTrue(
                compileall.compile_dir(ROOT / folder, quiet=1),
                msg=f"Python files in {folder} failed to compile",
            )

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


if __name__ == "__main__":
    unittest.main()
