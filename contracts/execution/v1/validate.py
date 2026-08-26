#!/usr/bin/env python3

import argparse
import base64
import binascii
import copy
import hashlib
import json
import sys
from pathlib import Path

try:
    from jsonschema import Draft202012Validator, FormatChecker
    from referencing import Registry, Resource
except ImportError:
    print("jsonschema is required; use models/.venv/bin/python", file=sys.stderr)
    raise SystemExit(2)


ROOT = Path(__file__).resolve().parent
SCHEMAS = ROOT / "schemas"
SCHEMA_MANIFEST = ROOT / "schema-manifest.json"
HASH_PREFIX = "sha256:"
FORBIDDEN_KEYS = {
    "accesstoken",
    "apikey",
    "authtoken",
    "authorization",
    "chainofthought",
    "cookie",
    "credential",
    "idtoken",
    "password",
    "refreshtoken",
    "secret",
    "sessiontoken",
    "thoughts",
}


class ContractError(Exception):
    pass


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def reject_floats(value, path="$"):
    if isinstance(value, float):
        raise ContractError(f"{path}: floating-point numbers are outside the v1 canonical profile")
    if isinstance(value, dict):
        for key, child in value.items():
            reject_floats(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_floats(child, f"{path}[{index}]")


def canonical_bytes(value):
    reject_floats(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes):
    return HASH_PREFIX + hashlib.sha256(value).hexdigest()


def canonical_hash(value):
    return sha256_bytes(canonical_bytes(value))


def load_schemas():
    schemas = {}
    for path in sorted(SCHEMAS.rglob("*.json")):
        schema = read_json(path)
        Draft202012Validator.check_schema(schema)
        schemas[schema["$id"]] = schema
    return schemas


def validator_for(schema_name, schemas):
    schema = read_json(SCHEMAS / schema_name)
    return validator_for_schema(schema, schemas)


def validator_for_schema(schema, schemas):
    registry = Registry().with_resources(
        (schema_id, Resource.from_contents(value))
        for schema_id, value in schemas.items()
    )
    return Draft202012Validator(
        schema,
        registry=registry,
        format_checker=FormatChecker(),
    )


def validate_protocol_fixture(fixture, schemas):
    records = {}
    for record in fixture.get("records", []):
        schema_id = record.get("schemaId")
        schema = schemas.get(schema_id)
        if schema is None:
            return "invalid_contract"
        validator = validator_for_schema(schema, schemas)
        errors = sorted(
            validator.iter_errors(record.get("instance")),
            key=lambda item: list(item.absolute_path),
        )
        if errors:
            unsupported_version = any(
                list(error.absolute_path) == ["schemaVersion"]
                and error.validator == "const"
                for error in errors
            )
            return (
                "unsupported_schema_version"
                if unsupported_version
                else "invalid_contract"
            )
        records[schema_id] = record["instance"]

    def schema(name):
        return records.get(f"https://documents.local/harness/v1/schemas/{name}")

    execution = schema("execution.schema.json")
    step = schema("step.schema.json")
    attempt = schema("step-attempt.schema.json")
    assignment = schema("step-assignment.schema.json")
    result = schema("step-result.schema.json")
    ack = schema("step-result-ack.schema.json")
    tool_invocation = schema("tool-invocation.schema.json")
    tool_plan = schema("tool-plan.schema.json")
    tool_result = schema("tool-result.schema.json")
    if not all((execution, step, attempt, assignment, result, ack)):
        return "invalid_contract"

    if (
        (
            not execution.get("parentExecutionId")
            and execution["rootExecutionId"] != execution["executionId"]
        )
        or step["executionId"] != execution["executionId"]
        or step.get("currentAttemptId") != attempt["attemptId"]
    ):
        return "invalid_protocol_identity"

    for field in ("executionId", "stepId", "operationId", "attemptId"):
        expected = assignment[field]
        if any(record[field] != expected for record in (attempt, result, ack)):
            return "invalid_protocol_identity"

    if (
        step["stepId"] != assignment["stepId"]
        or step.get("operationId") != assignment["operationId"]
        or step["stepKind"] != assignment["stepKind"]
        or result["stepKind"] != assignment["stepKind"]
    ):
        return "invalid_protocol_identity"

    tool_records = (tool_invocation, tool_plan, tool_result)
    if any(tool_records) and not all(tool_records):
        return "invalid_contract"
    if tool_invocation:
        if (
            tool_invocation["executionContext"]["executionId"]
            != execution["executionId"]
            or tool_invocation["toolCallId"] != tool_plan["toolCallId"]
            or tool_plan["toolCallId"] != tool_result["toolCallId"]
            or tool_plan["operationId"] != tool_result["operationId"]
            or tool_invocation["name"] != tool_plan["toolName"]
        ):
            return "invalid_protocol_identity"

    return None


def schema_errors(instance, validator):
    errors = sorted(validator.iter_errors(instance), key=lambda item: list(item.absolute_path))
    return [f"/{'/'.join(map(str, error.absolute_path))}: {error.message}" for error in errors]


def schema_manifest_value():
    entries = []
    for path in sorted(SCHEMAS.rglob("*.json")):
        relative = path.relative_to(ROOT).as_posix()
        entries.append({"path": relative, "sha256": sha256_bytes(path.read_bytes())})
    lines = "".join(f"{entry['path']}\0{entry['sha256']}\n" for entry in entries)
    return {
        "manifestSchema": "execution-contract-manifest/1",
        "contractVersion": "v1",
        "contractSetHash": sha256_bytes(lines.encode("utf-8")),
        "schemas": entries,
    }


def verify_schema_manifest():
    expected = schema_manifest_value()
    actual = read_json(SCHEMA_MANIFEST)
    if actual != expected:
        raise ContractError("schema-manifest.json does not match the canonical schemas")
    return actual["contractSetHash"]


def safe_artifact_path(bundle_path: Path, relative: str, valid_fixtures: Path):
    fixture_root = valid_fixtures.resolve()
    resolved = (bundle_path.parent / relative).resolve()
    if not resolved.is_relative_to(fixture_root):
        raise ContractError(f"artifact path escapes fixture root: {relative}")
    return resolved


def update_bundle_hashes(bundle_path: Path, schema_manifest_hash: str, valid_fixtures: Path):
    bundle = read_json(bundle_path)
    artifact_by_id = {artifact["artifactId"]: artifact for artifact in bundle["artifacts"]}
    for artifact in bundle["artifacts"]:
        if "bundlePath" not in artifact:
            continue
        body = safe_artifact_path(bundle_path, artifact["bundlePath"], valid_fixtures).read_bytes()
        artifact["size"] = len(body)
        artifact["contentHash"] = sha256_bytes(body)

    for event in bundle["events"]:
        payload = event.get("payload", {})
        snapshot_id = payload.get("snapshotArtifactId")
        if snapshot_id in artifact_by_id:
            payload["contentHash"] = artifact_by_id[snapshot_id]["contentHash"]
        event_without_hash = {key: value for key, value in event.items() if key != "contentHash"}
        event["contentHash"] = canonical_hash(event_without_hash)

    bundle["integrity"]["eventsHash"] = canonical_hash(bundle["events"])
    bundle["integrity"]["schemaManifestHash"] = schema_manifest_hash
    bundle_without_hash = {key: value for key, value in bundle.items() if key != "manifestHash"}
    bundle["manifestHash"] = canonical_hash(bundle_without_hash)
    write_json(bundle_path, bundle)


def check_forbidden_data(value, path="$"):
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = "".join(character for character in key.lower() if character.isalnum())
            if normalized in FORBIDDEN_KEYS:
                raise ContractError(f"{path}.{key}: forbidden sensitive or private-reasoning field")
            check_forbidden_data(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            check_forbidden_data(child, f"{path}[{index}]")


def validate_bundle_invariants(
    bundle,
    bundle_path: Path,
    schema_manifest_hash: str,
    valid_fixtures: Path | None = None,
):
    reject_floats(bundle)
    check_forbidden_data(bundle)
    events = bundle["events"]
    skill_activations = bundle["skillActivations"]
    artifacts = bundle["artifacts"]
    root_execution_id = bundle["rootExecutionId"]
    event_ids = {}
    producer_positions = {}
    operation_starts = {}
    operation_finishes = set()
    event_types = set()
    source_events = {}
    policy = bundle["policySummary"]
    policy_purpose = policy["purpose"]
    policy_destinations = set(policy["allowedDestinations"])
    policy_retention = policy["retentionClass"]

    sequences = [event["sequence"] for event in events]
    expected_sequences = list(range(bundle["eventRange"]["firstSequence"], bundle["eventRange"]["lastSequence"] + 1))
    if sequences != expected_sequences:
        raise ContractError("events are not in a contiguous canonical sequence matching eventRange")

    artifact_by_id = {artifact["artifactId"]: artifact for artifact in artifacts}
    if len(artifact_by_id) != len(artifacts):
        raise ContractError("duplicate artifactId")

    activation_ids = {
        activation["activationId"] for activation in skill_activations
    }
    if len(activation_ids) != len(skill_activations):
        raise ContractError("duplicate skill activationId")
    activation_identities = {
        (activation["executionId"], activation["skillId"])
        for activation in skill_activations
    }
    if len(activation_identities) != len(skill_activations):
        raise ContractError("duplicate skill activation identity")

    embedded = bundle.get("embeddedArtifacts") or {}
    for artifact_id, entry in embedded.items():
        artifact = artifact_by_id.get(artifact_id)
        if artifact is None:
            raise ContractError(f"embedded body has no artifact manifest: {artifact_id}")
        try:
            body = base64.b64decode(entry["data"], validate=True)
        except (KeyError, TypeError, ValueError, binascii.Error) as error:
            raise ContractError(f"embedded artifact is not canonical base64: {artifact_id}") from error
        if base64.b64encode(body).decode("ascii") != entry["data"]:
            raise ContractError(f"embedded artifact is not canonical base64: {artifact_id}")
        if artifact["size"] != len(body) or artifact["contentHash"] != sha256_bytes(body):
            raise ContractError(f"embedded artifact integrity mismatch: {artifact_id}")
    for artifact_id, artifact in artifact_by_id.items():
        if str(artifact.get("storageRef", "")).startswith("bundle:") and artifact_id not in embedded:
            raise ContractError(f"embedded artifact body is absent: {artifact_id}")
        if artifact["retentionClass"] != policy_retention:
            raise ContractError(
                f"artifact retention differs from bundle policy: {artifact_id}"
            )

    for event in events:
        event_id = event["eventId"]
        if event_id in event_ids:
            raise ContractError(f"duplicate eventId: {event_id}")
        event_ids[event_id] = event
        event_types.add(event["eventType"])
        if event["rootExecutionId"] != root_execution_id:
            raise ContractError(f"{event_id}: rootExecutionId differs from bundle")
        if not event.get("parentExecutionId") and event["rootExecutionId"] != event["executionId"]:
            raise ContractError(f"{event_id}: root execution identity mismatch")
        cause = event.get("causedByEventId")
        if cause and cause not in event_ids:
            raise ContractError(f"{event_id}: cause is missing or does not precede the event")
        producer = event["producer"]
        producer_key = (producer["component"], producer["instanceId"])
        previous = producer_positions.get(producer_key, 0)
        if event["producerSequence"] <= previous:
            raise ContractError(f"{event_id}: producerSequence is not strictly monotonic")
        producer_positions[producer_key] = event["producerSequence"]
        for artifact_id in event["artifactRefs"]:
            if artifact_id not in artifact_by_id:
                raise ContractError(f"{event_id}: referenced artifact is absent: {artifact_id}")
        expected_hash = canonical_hash({key: value for key, value in event.items() if key != "contentHash"})
        if event["contentHash"] != expected_hash:
            raise ContractError(f"{event_id}: contentHash mismatch")
        security = event["security"]
        if security["purpose"] != policy_purpose:
            raise ContractError(f"{event_id}: purpose differs from bundle policy")
        if not policy_destinations.issubset(security["allowedDestinations"]):
            raise ContractError(
                f"{event_id}: destination differs from bundle policy"
            )

        if event["eventType"] == "source.observed":
            payload = event["payload"]
            source_id = payload["sourceId"]
            if event.get("sourceId") != source_id:
                raise ContractError(f"{event_id}: source identity mismatch")
            if source_id in source_events:
                raise ContractError(f"duplicate sourceId: {source_id}")
            source_events[source_id] = event
            snapshot_id = payload.get("snapshotArtifactId")
            if snapshot_id:
                artifact = artifact_by_id.get(snapshot_id)
                if not artifact or source_id not in artifact.get("inputSourceIds", []):
                    raise ContractError(
                        f"{event_id}: source snapshot lacks provenance link"
                    )
        elif event["eventType"] == "operation.started":
            key = (event["operationId"], event["attemptId"])
            if key in operation_starts:
                raise ContractError(f"duplicate operation start: {key}")
            operation_starts[key] = event
        elif event["eventType"] == "operation.finished":
            key = (event["operationId"], event["attemptId"])
            start = operation_starts.get(key)
            if not start:
                raise ContractError(f"operation finish without start: {key}")
            if key in operation_finishes:
                raise ContractError(f"duplicate operation finish: {key}")
            if start["payload"]["operationKind"] != event["payload"]["operationKind"]:
                raise ContractError(f"operation kind changed: {key}")
            operation_finishes.add(key)

    for artifact in artifacts:
        for source_id in artifact.get("inputSourceIds", []):
            if source_id not in source_events:
                raise ContractError(
                    f"artifact input source is absent: {artifact['artifactId']}: {source_id}"
                )

    required_types = {
        "execution.created",
        "execution.state_changed",
        "operation.started",
        "operation.finished",
        "message.recorded",
        "source.observed",
    }
    missing_types = sorted(required_types - event_types)
    if missing_types:
        raise ContractError(f"required event types absent: {', '.join(missing_types)}")
    open_operations = set(operation_starts) - operation_finishes
    if open_operations and bundle["bundleCompleteness"]["status"] == "reproducible":
        raise ContractError(f"reproducible bundle has open operations: {sorted(open_operations)}")

    terminal = [
        event for event in events
        if event["eventType"] == "execution.state_changed"
        and event["payload"].get("to") in {"completed", "failed", "cancelled"}
    ]
    if len(terminal) != 1 or terminal[0] is not events[-1]:
        raise ContractError("bundle must end in exactly one terminal execution state")

    event_execution_ids = {event["executionId"] for event in events}
    terminal_by_execution = {
        event["executionId"]: event["payload"]["to"]
        for event in events
        if event["eventType"] == "execution.state_changed"
        and event["payload"].get("to") in {"completed", "failed", "cancelled"}
    }
    for activation in skill_activations:
        execution_id = activation["executionId"]
        if execution_id not in event_execution_ids:
            raise ContractError(
                f"skill activation execution is absent: {activation['activationId']}"
            )
        terminal_status = terminal_by_execution.get(execution_id)
        if activation["status"] == "superseded":
            continue
        if terminal_status is None or activation["status"] != terminal_status:
            raise ContractError(
                f"skill activation status differs from execution: {activation['activationId']}"
            )

    for artifact in artifacts:
        if artifact["dataClassification"] == "secret":
            raise ContractError(f"secret artifact cannot be exported: {artifact['artifactId']}")
        created_by = artifact.get("createdByEventId")
        if created_by and created_by not in event_ids:
            raise ContractError(f"artifact creator event is absent: {artifact['artifactId']}")
        if "bundlePath" in artifact and valid_fixtures is not None:
            path = safe_artifact_path(bundle_path, artifact["bundlePath"], valid_fixtures)
            if not path.is_file():
                raise ContractError(f"artifact body is absent: {artifact['artifactId']}")
            body = path.read_bytes()
            if artifact["size"] != len(body) or artifact["contentHash"] != sha256_bytes(body):
                raise ContractError(f"artifact integrity mismatch: {artifact['artifactId']}")

    for event in events:
        payload = event["payload"]
        snapshot_id = payload.get("snapshotArtifactId")
        if snapshot_id:
            artifact = artifact_by_id.get(snapshot_id)
            if not artifact or payload.get("contentHash") != artifact["contentHash"]:
                raise ContractError(f"source snapshot integrity mismatch: {event['eventId']}")

    if bundle["integrity"]["schemaManifestHash"] != schema_manifest_hash:
        raise ContractError("bundle schemaManifestHash mismatch")
    if bundle["integrity"]["eventsHash"] != canonical_hash(events):
        raise ContractError("bundle eventsHash mismatch")
    expected_manifest_hash = canonical_hash({key: value for key, value in bundle.items() if key != "manifestHash"})
    if bundle["manifestHash"] != expected_manifest_hash:
        raise ContractError("bundle manifestHash mismatch")


def pointer_parent(value, pointer):
    parts = [part.replace("~1", "/").replace("~0", "~") for part in pointer.lstrip("/").split("/")]
    current = value
    for part in parts[:-1]:
        current = current[int(part)] if isinstance(current, list) else current[part]
    return current, parts[-1]


def apply_mutations(value, mutations):
    mutated = copy.deepcopy(value)
    for mutation in mutations:
        parent, key = pointer_parent(mutated, mutation["path"])
        if mutation["op"] == "remove":
            parent.pop(int(key)) if isinstance(parent, list) else parent.pop(key)
        elif mutation["op"] == "replace":
            if isinstance(parent, list):
                parent[int(key)] = mutation["value"]
            else:
                parent[key] = mutation["value"]
        elif mutation["op"] == "add":
            if isinstance(parent, list):
                parent.insert(int(key), mutation["value"])
            else:
                parent[key] = mutation["value"]
        else:
            raise ContractError(f"unsupported fixture mutation: {mutation['op']}")
    return mutated


def validate_invalid_fixtures(bundle_validator, schema_manifest_hash, valid_fixtures, invalid_fixtures):
    failures = []
    for path in sorted(invalid_fixtures.glob("*.json")):
        fixture = read_json(path)
        base_path = (path.parent / fixture["base"]).resolve()
        instance = apply_mutations(read_json(base_path), fixture["mutations"])
        errors = schema_errors(instance, bundle_validator)
        category = "schema" if errors else None
        if not errors:
            try:
                validate_bundle_invariants(instance, base_path, schema_manifest_hash, valid_fixtures)
            except ContractError:
                category = "invariant"
        if category != fixture["expectedFailure"]:
            failures.append(f"{path.name}: expected {fixture['expectedFailure']} failure, got {category or 'valid'}")
    return failures


def validate_protocol_fixtures(protocol_fixtures, schemas):
    failures = []
    valid_paths = sorted((protocol_fixtures / "valid").glob("*.json"))
    invalid_paths = sorted((protocol_fixtures / "invalid").glob("*.json"))
    for path in valid_paths:
        error = validate_protocol_fixture(read_json(path), schemas)
        if error:
            failures.append(f"{path.name}: expected valid protocol fixture, got {error}")
    for path in invalid_paths:
        fixture = read_json(path)
        base_path = (path.parent / fixture["base"]).resolve()
        instance = apply_mutations(read_json(base_path), fixture["mutations"])
        error = validate_protocol_fixture(instance, schemas)
        if error != fixture["expectedError"]:
            failures.append(
                f"{path.name}: expected {fixture['expectedError']} failure, got {error or 'valid'}"
            )
    return failures, len(valid_paths), len(invalid_paths)


def main():
    parser = argparse.ArgumentParser(description="Validate canonical execution v1 contracts")
    parser.add_argument(
        "--fixtures",
        type=Path,
        required=True,
        help="path to the test fixture root containing valid/ and invalid/",
    )
    parser.add_argument("--update-hashes", action="store_true")
    args = parser.parse_args()
    fixtures = args.fixtures.resolve()
    valid_fixtures = fixtures / "valid"
    invalid_fixtures = fixtures / "invalid"
    protocol_fixtures = fixtures / "protocol"
    if (
        not valid_fixtures.is_dir()
        or not invalid_fixtures.is_dir()
        or not (protocol_fixtures / "valid").is_dir()
        or not (protocol_fixtures / "invalid").is_dir()
    ):
        parser.error(
            "--fixtures must contain bundle and protocol valid/ and invalid/ directories"
        )

    schemas = load_schemas()
    expected_manifest = schema_manifest_value()
    if args.update_hashes:
        write_json(SCHEMA_MANIFEST, expected_manifest)
        for path in sorted(valid_fixtures.glob("*-bundle.json")):
            update_bundle_hashes(path, expected_manifest["contractSetHash"], valid_fixtures)

    schema_manifest_hash = verify_schema_manifest()
    bundle_validator = validator_for("execution-bundle.schema.json", schemas)
    failures = []
    valid_count = 0
    for path in sorted(valid_fixtures.glob("*-bundle.json")):
        bundle = read_json(path)
        errors = schema_errors(bundle, bundle_validator)
        if errors:
            failures.extend(f"{path.name}{error}" for error in errors)
            continue
        try:
            validate_bundle_invariants(bundle, path, schema_manifest_hash, valid_fixtures)
        except ContractError as error:
            failures.append(f"{path.name}: {error}")
            continue
        valid_count += 1

    failures.extend(
        validate_invalid_fixtures(
            bundle_validator,
            schema_manifest_hash,
            valid_fixtures,
            invalid_fixtures,
        )
    )
    protocol_failures, valid_protocol_count, invalid_protocol_count = (
        validate_protocol_fixtures(protocol_fixtures, schemas)
    )
    failures.extend(protocol_failures)
    if failures:
        for failure in failures:
            print(f"FAIL {failure}", file=sys.stderr)
        raise SystemExit(1)

    invalid_count = len(list(invalid_fixtures.glob("*.json")))
    print(
        f"Validated {len(schemas)} schemas, {valid_count} valid bundles, "
        f"{invalid_count} invalid bundle fixtures, {valid_protocol_count} valid protocol fixtures, "
        f"and {invalid_protocol_count} invalid protocol fixtures"
    )


if __name__ == "__main__":
    main()
