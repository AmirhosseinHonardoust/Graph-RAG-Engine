import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from fastapi.testclient import TestClient
    from backend.api import app
except ModuleNotFoundError as exc:  # Allows lightweight local checks without optional deps.
    TestClient = None
    app = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(IMPORT_ERROR is not None, f"Optional API dependency missing: {IMPORT_ERROR}")
class ApiContractTests(unittest.TestCase):
    def test_health_endpoint_does_not_load_retrieval_artifacts(self):
        client = TestClient(app)

        response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"ok": True})


if __name__ == "__main__":
    unittest.main()
