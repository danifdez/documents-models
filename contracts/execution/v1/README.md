# Canonical execution contract v1

This directory is the canonical, language-neutral source for durable
executions. Product repositories consume versioned copies of these files;
they must not redefine the logical schema.

## Layout

- `schemas/`: JSON Schema 2020-12 contracts.
- `schema-manifest.json`: SHA-256 inventory of every schema.
- `validate.py`: schema, integrity, and cross-record invariant validator.

Conformance fixtures are test data and live outside the distributed contract,
under each consumer's test tree. The development reference set lives at
`development/harness/tests/contracts/v1/fixtures/`.

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

Run the conformance suite from the monorepo root:

```bash
models/.venv/bin/python development/harness/contracts/v1/validate.py \
  --fixtures development/harness/tests/contracts/v1/fixtures
```

Regenerate fixture and schema hashes after an intentional contract edit:

```bash
models/.venv/bin/python development/harness/contracts/v1/validate.py \
  --fixtures development/harness/tests/contracts/v1/fixtures \
  --update-hashes
```

Unknown optional fields are accepted. Unknown schema versions and unknown
`payloadSchema` values are rejected explicitly.
