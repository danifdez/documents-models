import copy
import io
import shutil
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from contracts.execution.v1 import validate

FIXTURES = Path(__file__).parent / "contracts" / "execution" / "v1" / "fixtures"


class ExecutionContractTests(unittest.TestCase):
    def run_validator(self, *arguments):
        output = io.StringIO()
        with patch("sys.argv", ["validate.py", *map(str, arguments)]), redirect_stdout(output):
            validate.main()
        return output.getvalue()

    def test_validates_the_shipped_bundle_and_protocol_fixtures(self):
        output = self.run_validator("--fixtures", FIXTURES)
        bundles = len(list((FIXTURES / "valid").glob("*-bundle.json")))
        invalid = len(list((FIXTURES / "invalid").glob("*.json")))
        protocol = len(list((FIXTURES / "protocol" / "valid").glob("*.json")))
        invalid_protocol = len(list((FIXTURES / "protocol" / "invalid").glob("*.json")))
        self.assertIn(f"{bundles} valid bundles, {invalid} invalid bundle fixtures", output)
        self.assertIn(f"{protocol} valid protocol fixtures, and {invalid_protocol} invalid protocol fixtures", output)

    def test_regenerates_fixture_hashes_and_detects_a_stale_manifest(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fixtures = root / "fixtures"
            shutil.copytree(FIXTURES, fixtures)
            manifest = root / "schema-manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            with patch.object(validate, "SCHEMA_MANIFEST", manifest):
                with self.assertRaisesRegex(validate.ContractError, "canonical schemas"):
                    self.run_validator("--fixtures", fixtures)

                output = self.run_validator("--fixtures", fixtures, "--update-hashes")

            self.assertIn("valid bundles", output)
            self.assertEqual(validate.read_json(manifest), validate.read_json(validate.SCHEMA_MANIFEST))

    def test_reports_invalid_fixture_directory_and_broken_valid_fixture(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            errors = io.StringIO()
            with redirect_stderr(errors), self.assertRaises(SystemExit) as invalid_args:
                self.run_validator("--fixtures", root)
            self.assertEqual(invalid_args.exception.code, 2)
            self.assertIn("bundle and protocol", errors.getvalue())

            fixtures = root / "fixtures"
            shutil.copytree(FIXTURES, fixtures)
            valid = fixtures / "valid"
            affected_bundle = valid / "ia-browser-bundle.json"
            bundle = validate.read_json(affected_bundle)
            bundle["manifestHash"] = "sha256:" + "0" * 64
            validate.write_json(affected_bundle, bundle)
            (valid / "invalid-shape-bundle.json").write_text("{}", encoding="utf-8")

            with redirect_stderr(errors), self.assertRaises(SystemExit) as failure:
                self.run_validator("--fixtures", fixtures)
            self.assertEqual(failure.exception.code, 1)
            self.assertIn("ia-browser-bundle.json: bundle manifestHash mismatch", errors.getvalue())
            self.assertIn("invalid-shape-bundle.json", errors.getvalue())

    def test_event_sequence_producer_and_causes_must_preserve_order(self):
        fixture = FIXTURES / "valid" / "documents-bundle.json"
        bundle = validate.read_json(fixture)
        manifest_hash = validate.verify_schema_manifest()
        cases = (
            ({"sequence": 1000}, "contiguous canonical sequence"),
            ({"causedByEventId": bundle["events"][-1]["eventId"]}, "cause is missing or does not precede"),
            ({"eventId": bundle["events"][0]["eventId"]}, "duplicate eventId"),
            ({"rootExecutionId": "different-root"}, "rootExecutionId differs"),
        )
        for updates, message in cases:
            with self.subTest(updates=updates):
                mutated = copy.deepcopy(bundle)
                mutated["events"][1].update(updates)
                with self.assertRaisesRegex(validate.ContractError, message):
                    validate.validate_bundle_invariants(mutated, fixture, manifest_hash, fixture.parent)

        mutated = copy.deepcopy(bundle)
        mutated["events"][1].update(
            producer=mutated["events"][0]["producer"],
            producerSequence=mutated["events"][0]["producerSequence"],
        )
        with self.assertRaisesRegex(validate.ContractError, "producerSequence is not strictly monotonic"):
            validate.validate_bundle_invariants(mutated, fixture, manifest_hash, fixture.parent)

    def test_rejects_sensitive_nested_fields_and_noncanonical_numbers(self):
        with self.assertRaisesRegex(validate.ContractError, r"\$\.events\[0\]\.access_token: forbidden"):
            validate.check_forbidden_data({"events": [{"access_token": "synthetic"}]})
        with self.assertRaisesRegex(validate.ContractError, r"\$\.nested\[0\]: floating-point"):
            validate.canonical_bytes({"nested": [1.5]})

    def test_applies_fixture_mutations_to_a_copy_and_decodes_pointer_escapes(self):
        initial = {"items": [{"a/b~c": "before"}, "removed"]}
        operations = [
            {"op": "replace", "path": "/items/0/a~1b~0c", "value": "after"},
            {"op": "replace", "path": "/items/1", "value": "changed"},
            {"op": "add", "path": "/items/1", "value": "middle"},
            {"op": "add", "path": "/extra", "value": 7},
            {"op": "remove", "path": "/items/2"},
        ]
        self.assertEqual(
            validate.apply_mutations(initial, operations),
            {"items": [{"a/b~c": "after"}, "middle"], "extra": 7},
        )
        self.assertEqual(initial, {"items": [{"a/b~c": "before"}, "removed"]})
        with self.assertRaisesRegex(validate.ContractError, "unsupported fixture mutation"):
            validate.apply_mutations(initial, [{"op": "move", "path": "/items"}])

    def test_artifact_paths_cannot_leave_the_fixture_directory(self):
        root = FIXTURES / "valid"
        bundle = root / "documents-bundle.json"
        with self.assertRaisesRegex(validate.ContractError, "artifact path escapes fixture root"):
            validate.safe_artifact_path(bundle, "../invalid/missing-artifact.json", root)
