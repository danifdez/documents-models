"""Shared GBNF grammars for structured LLM output.

Passing one of these to `LLMService.chat(grammar=...)` constrains sampling so
the model *cannot* produce malformed output — no markdown fences, no prose,
no truncated JSON. Prefer this over regex-parsing free-form responses.

The `_JSON_COMMON` block defines generic JSON rules (value/object/array/
string/number/ws) adapted from llama.cpp's grammars/json.gbnf; task grammars
compose a stricter `root` on top of it.

`ws` is BOUNDED and no `root` starts with it, on purpose. An unbounded
`ws ::= ([ \t\n] ws)?` lets the sampler emit whitespace forever, and a
thinking model whose `<think>` opener the grammar masks out falls straight
into that hole: it burns the whole `max_tokens` on spaces and returns an
empty string, which the caller then fails to parse.

Array cardinality is bounded for the same reason: tasks decode greedily
(`temperature=0.0`) and our sampling profiles leave `repetition_penalty` at
1.0, so a model that starts repeating an item has nothing to stop it and
burns `max_tokens` mid-string. The cap turns that into a parseable array
whose duplicates the callers already drop. Keep any new rule from
reintroducing an unbounded repetition.
"""

_JSON_COMMON = r"""
value  ::= object | array | string | number | ("true" | "false" | "null") ws
object ::= "{" ws ( string ":" ws value ("," ws string ":" ws value)* )? "}" ws
array  ::= "[" ws ( value ("," ws value)* )? "]" ws
string ::= "\"" ( [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) )* "\"" ws
number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws
ws     ::= | " " | "\n" [ \t]{0,20}
"""

# A string that cannot run away. `string` above is unbounded on purpose (an
# agent's `thought` or a summary legitimately needs the room), but extraction
# grammars quote short spans copied from the document, and there the same
# runaway shows up *inside* the quotes: the model loops, `max_tokens` ends the
# generation mid-string and the whole array fails to parse — one unterminated
# string silently costs every date in that chunk.
_BOUNDED_STRING = r"""
sstring ::= "\"" ( [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) ){0,200} "\"" ws
"""

# Any single JSON value (the equivalent of "just give me valid JSON").
JSON_VALUE_GBNF = "root ::= value\n" + _JSON_COMMON

# A JSON array of strings (e.g. verbatim date expressions / candidate spans).
STRING_ARRAY_GBNF = (
    r"""
root ::= "[" ws ( sstring ( "," ws sstring ){0,63} )? "]"
"""
    + _BOUNDED_STRING + _JSON_COMMON
)

# relationship-extraction: array of {subject, predicate, object} triples.
RELATIONSHIPS_GBNF = (
    r"""
root ::= "[" ws ( rel ( "," ws rel ){0,127} )? "]"
rel  ::= "{" ws "\"subject\"" ws ":" ws sstring "," ws "\"predicate\"" ws ":" ws sstring "," ws "\"object\"" ws ":" ws sstring "}" ws
"""
    + _BOUNDED_STRING + _JSON_COMMON
)

# Agent step decision: {"thought": ..., "tool": ..., "args": {...}}
# or {"thought": ..., "finish": <value>}.
AGENT_DECISION_GBNF = (
    r"""
root     ::= "{" ws "\"thought\"" ws ":" ws string "," ws ( toolcall | finish ) "}"
toolcall ::= "\"tool\"" ws ":" ws string "," ws "\"args\"" ws ":" ws object
finish   ::= "\"finish\"" ws ":" ws value
"""
    + _JSON_COMMON
)

# Forced final synthesis when an agent runs out of steps: finish only.
AGENT_FINISH_GBNF = (
    r"""
root   ::= "{" ws "\"thought\"" ws ":" ws string "," ws "\"finish\"" ws ":" ws value "}"
"""
    + _JSON_COMMON
)
