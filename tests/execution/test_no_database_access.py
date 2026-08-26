import ast
import os
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXCLUDED_DIRECTORIES = {
    ".git",
    ".llama.cpp",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "tests",
}
FORBIDDEN_MODULES = {
    "age",
    "asyncpg",
    "neo4j",
    "pgvector",
    "psycopg",
    "psycopg2",
    "sqlalchemy",
}
FORBIDDEN_CONFIG_KEYS = {"DATABASE_URL", "POSTGRES_"}
LEGACY_COORDINATOR_PACKAGES = {"agent", "agents", "database"}


def production_files(suffixes):
    for root, directories, files in os.walk(PROJECT_ROOT):
        directories[:] = [
            name for name in directories if name not in EXCLUDED_DIRECTORIES
        ]
        for name in files:
            if name.endswith(suffixes):
                yield Path(root) / name


class NoDatabaseAccessTest(unittest.TestCase):
    def test_models_has_no_legacy_coordinator_package(self):
        existing = sorted(
            name
            for name in LEGACY_COORDINATOR_PACKAGES
            if (PROJECT_ROOT / name).exists()
        )

        self.assertEqual([], existing)

    def test_production_code_has_no_database_client_or_configuration(self):
        violations = []
        for path in production_files((".py", ".json")):
            source = path.read_text(encoding="utf-8")
            relative = path.relative_to(PROJECT_ROOT)
            for key in FORBIDDEN_CONFIG_KEYS:
                if key in source:
                    violations.append(f"{relative}: configuration {key}")
            if path.suffix != ".py":
                continue
            tree = ast.parse(source, filename=str(relative))
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module]
                for module in modules:
                    root_module = module.split(".", 1)[0]
                    if root_module in FORBIDDEN_MODULES or root_module == "database":
                        violations.append(f"{relative}: import {module}")

        self.assertEqual([], violations)

    def test_dependency_locks_have_no_database_driver(self):
        violations = []
        for name in ("requirements.txt", "requirements-gpu.txt"):
            path = PROJECT_ROOT / name
            for line in path.read_text(encoding="utf-8").splitlines():
                package = (
                    line.split("==", 1)[0]
                    .split("[", 1)[0]
                    .strip()
                    .lower()
                )
                if package in FORBIDDEN_MODULES:
                    violations.append(f"{name}: {package}")

        self.assertEqual([], violations)


if __name__ == "__main__":
    unittest.main()
