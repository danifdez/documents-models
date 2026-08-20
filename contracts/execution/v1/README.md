# Canonical execution contract v1

This directory contains the models worker's pinned copy of the
language-neutral contract for durable executions.

## Layout

- `schemas/`: JSON Schema 2020-12 contracts.
- `schema-manifest.json`: SHA-256 inventory of every schema.
- `validate.py`: schema, integrity, and cross-record invariant validator.

Conformance fixtures are test data and live in
`tests/contracts/execution/v1/fixtures/`.

## Integrity rules

Hashes use lowercase SHA-256 with the `sha256:` prefix.

- Artifact hashes cover the exact bytes at `bundlePath`.
- Event hashes cover the event object without `contentHash`.
- `eventsHash` covers the canonical event array, including event hashes.
- `manifestHash` covers the complete bundle object without `manifestHash`.
- `contractSetHash` covers sorted lines of `<path>\0<sha256>\n` from the
  schema manifest.

Canonical JSON is UTF-8 JSON with object keys sorted lexicographically, no
insignificant whitespace, and non-ASCII characters preserved. The v1 hashed
surface rejects floating-point numbers; durations and usage values use integer
units or the string `unknown`. This constrained profile is intentionally easy
to reproduce in TypeScript, Python, and C++.

Run the conformance suite from the models repository:

```bash
.venv/bin/python -B contracts/execution/v1/validate.py \
  --fixtures tests/contracts/execution/v1/fixtures
```

Regenerate fixture and schema hashes after an intentional contract edit:

```bash
.venv/bin/python -B contracts/execution/v1/validate.py \
  --fixtures tests/contracts/execution/v1/fixtures \
  --update-hashes
```

Unknown optional fields are accepted. Unknown schema versions and unknown
`payloadSchema` values are rejected explicitly.
