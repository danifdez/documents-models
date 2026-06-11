"""Shared GBNF grammars for structured LLM output.

Passing one of these to `LLMService.chat(grammar=...)` constrains sampling so
the model *cannot* produce malformed output — no markdown fences, no prose,
no truncated JSON. Prefer this over regex-parsing free-form responses.

The `_JSON_COMMON` block defines generic JSON rules (value/object/array/
string/number/ws) adapted from llama.cpp's grammars/json.gbnf; task grammars
compose a stricter `root` on top of it.
"""

_JSON_COMMON = r"""
value  ::= object | array | string | number | ("true" | "false" | "null") ws
object ::= "{" ws ( string ":" ws value ("," ws string ":" ws value)* )? "}" ws
array  ::= "[" ws ( value ("," ws value)* )? "]" ws
string ::= "\"" ( [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) )* "\"" ws
number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws
ws     ::= ([ \t\n] ws)?
"""

# Any single JSON value (the equivalent of "just give me valid JSON").
JSON_VALUE_GBNF = "root ::= ws value\n" + _JSON_COMMON

# relationship-extraction: array of {subject, predicate, object} triples.
RELATIONSHIPS_GBNF = (
    r"""
root ::= ws "[" ws ( rel ( "," ws rel )* )? "]" ws
rel  ::= "{" ws "\"subject\"" ws ":" ws string "," ws "\"predicate\"" ws ":" ws string "," ws "\"object\"" ws ":" ws string "}" ws
"""
    + _JSON_COMMON
)

# date-extraction LLM fallback: either a resolved date or an unresolved marker.
DATE_RESOLUTION_GBNF = (
    r"""
root       ::= ws ( resolved | unresolved )
resolved   ::= "{" ws "\"date\"" ws ":" ws datestr "," ws "\"endDate\"" ws ":" ws ( datestr | "null" ws ) "," ws "\"precision\"" ws ":" ws ( "\"day\"" | "\"month\"" | "\"year\"" ) ws "}" ws
unresolved ::= "{" ws "\"unresolved\"" ws ":" ws "true" ws "," ws "\"reason\"" ws ":" ws string "}" ws
datestr    ::= "\"" [0-9] [0-9] [0-9] [0-9] "-" [0-9] [0-9] "-" [0-9] [0-9] "\"" ws
"""
    + _JSON_COMMON
)

# Agent step decision: {"thought": ..., "tool": ..., "args": {...}}
# or {"thought": ..., "finish": <value>}.
AGENT_DECISION_GBNF = (
    r"""
root     ::= ws "{" ws "\"thought\"" ws ":" ws string "," ws ( toolcall | finish ) "}" ws
toolcall ::= "\"tool\"" ws ":" ws string "," ws "\"args\"" ws ":" ws object
finish   ::= "\"finish\"" ws ":" ws value
"""
    + _JSON_COMMON
)

# Forced final synthesis when an agent runs out of steps: finish only.
AGENT_FINISH_GBNF = (
    r"""
root   ::= ws "{" ws "\"thought\"" ws ":" ws string "," ws "\"finish\"" ws ":" ws value "}" ws
"""
    + _JSON_COMMON
)
