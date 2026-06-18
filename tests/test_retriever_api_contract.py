import unittest

try:
    from fastapi.testclient import TestClient

    from backend.api import app, parse_cors_origins
except ModuleNotFoundError as exc:  # Allows lightweight local checks without optional deps.
    TestClient = None
    app = None
    parse_cors_origins = None
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

    def test_cors_origins_default_to_wildcard(self):
        self.assertEqual(parse_cors_origins(None), ["*"])
        self.assertEqual(parse_cors_origins(""), ["*"])

    def test_cors_origins_parsed_from_csv(self):
        self.assertEqual(
            parse_cors_origins("https://a.example, https://b.example"),
            ["https://a.example", "https://b.example"],
        )


if __name__ == "__main__":
    unittest.main()
