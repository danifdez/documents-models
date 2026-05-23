"""GBNF grammar builder for dataset extraction.

Given a dataset schema (a subset of fields to extract), produces a GBNF
that constrains the LLM output to:

    {
      "<key1>": { "value": <typed value or null>, "_quote": <string>, "_page": <int or null> },
      "<key2>": { ... },
      ...
    }

`null` is admitted everywhere a value can go:
the model must declare "not found in source" with null, not invent).
"""

from typing import Iterable, List


def _escape_literal(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _value_rule(field: dict) -> str:
    """Return a GBNF expression matching the value union for this field."""
    ftype = field.get("type", "text")
    if ftype == "number":
        return "number | null-lit"
    if ftype == "boolean":
        return "boolean | null-lit"
    if ftype == "select":
        options = field.get("options") or []
        if not options:
            return "string | null-lit"
        alternatives = " | ".join(
            f"\"\\\"{_escape_literal(o)}\\\"\"" for o in options)
        return f"({alternatives}) | null-lit"
    # text, date, datetime, time, and any unknown type -> free string with null sentinel
    return "string | null-lit"


def build_grammar(fields_to_extract: List[dict]) -> str:
    """Build a GBNF that validates a flat object with one entry per field.

    `fields_to_extract` is a list of DatasetField dicts. The order defines the
    order of keys in the grammar's object production — llama.cpp respects it.
    """
    if not fields_to_extract:
        # Empty schema -> trivial grammar that matches "{}" only.
        return 'root ::= "{}"\n'

    field_rules: List[str] = []
    for idx, field in enumerate(fields_to_extract):
        key = field["key"]
        rule_name = f"field-{idx}"
        value_rule = _value_rule(field)
        field_rules.append(
            f"{rule_name} ::= "
            f"\"\\\"{_escape_literal(key)}\\\":\" ws \"{{\" ws "
            f"\"\\\"value\\\":\" ws ({value_rule}) ws \",\" ws "
            f"\"\\\"_quote\\\":\" ws string ws \",\" ws "
            f"\"\\\"_page\\\":\" ws (integer | null-lit) ws \"}}\""
        )

    body = ' ws "," ws '.join(f"{rname.split(' ')[0]}" for rname in (
        f"field-{i}" for i in range(len(fields_to_extract))))
    root = f"root ::= \"{{\" ws {body} ws \"}}\""

    primitives = """
ws ::= ([ \\t\\n] ws)?
null-lit ::= "null"
boolean ::= "true" | "false"
integer ::= "-"? ("0" | [1-9] [0-9]*)
number ::= integer ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
string ::= "\\"" char* "\\""
char ::= [^"\\\\] | "\\\\" (["\\\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
""".strip()

    return root + "\n" + "\n".join(field_rules) + "\n" + primitives + "\n"
