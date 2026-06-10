import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UI_APP = ROOT / "ui" / "app.py"


class TestStreamlitUiContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = UI_APP.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_ui_exposes_answer_mode_selector(self):
        self.assertIn('"Answer mode"', self.source)
        self.assertIn('options=["extractive", "llm"]', self.source)
        self.assertIn('index=0', self.source)

    def test_ask_payload_includes_selected_mode(self):
        self.assertIn('{"question": q, "mode": answer_mode}', self.source)
        self.assertIn('post_json(f"{api_url}/ask"', self.source)

    def test_ui_displays_answer_mode_and_llm_fallback_error(self):
        self.assertRegex(self.source, r"data\.get\([\'\"]answer_mode[\'\"], answer_mode\)")
        self.assertIn('data.get("llm_error")', self.source)
        self.assertIn('fell back to extractive mode', self.source)

    def test_ui_has_no_unused_json_import(self):
        imports = [node.names[0].name for node in self.tree.body if isinstance(node, ast.Import)]
        self.assertNotIn("json", imports)


if __name__ == "__main__":
    unittest.main()
