"""Dependency-free wiring checks for the mandatory vLLM preflight gate."""

import ast
from pathlib import Path
import unittest


APP = Path(__file__).resolve().parents[1] / "app.py"


class VllmStartGateWiringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(APP.read_text())

    def test_validated_start_wrapper_exists(self):
        names = {
            node.name for node in ast.walk(self.tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertIn("_validated_engine_start", names)

    def test_generated_start_route_uses_validated_wrapper(self):
        route_factory = next(
            node for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_make_engine_routes"
        )
        calls = {
            node.func.id for node in ast.walk(route_factory)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("_validated_engine_start", calls)
        self.assertNotIn("_engine_start", calls)


if __name__ == "__main__":
    unittest.main()
