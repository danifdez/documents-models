# Data access

Models has no PostgreSQL credentials and no application-domain database
adapter. Backend is the sole owner of relational data, pgvector tables,
Apache AGE and execution state.

Every handler receives all required data through its step assignment:

- small structured inputs live in `work.payload`;
- larger snapshots use attempt-scoped input artifacts;
- Models returns small calculated values inline and large calculated bodies as
  attempt-scoped output artifacts. Backend validates both before persisting any
  resulting domain effect during finalization.

Vector retrieval uses the `vector_candidates` artifact. It contains the
bounded, scope-checked candidate snapshot selected by Backend. Models embeds
the query and ranks those candidates locally without opening a datastore.

Input file bodies follow the same rule. Backend returns artifact references in
the assignment, Models downloads each body through the attempt-scoped artifact
endpoint, and the executor exposes them by role:

```python
payload["_input_artifacts"]["document"]
payload["_input_artifacts"]["vector_candidates"]
```

A missing or malformed required artifact fails the step. Handlers must not
fall back to a filesystem path, database query or productive service.
