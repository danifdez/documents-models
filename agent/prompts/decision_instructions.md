At every step, emit ONE JSON object and nothing else.
Two valid shapes:

  {"thought": "<short rationale>", "tool": "<tool_name>", "args": { ... }}

  {"thought": "<short rationale>", "finish": { ...your final result matching the output_schema... }}

Do NOT wrap the JSON in markdown fences. Do NOT add prose around it.
Pick exactly one tool from the list of available tools, or use "finish" when done.
