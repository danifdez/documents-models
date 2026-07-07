Anchor date: {anchor_str}. Text language: {language}.
Expression: "{expression}"
Context: "{context}"
Respond ONLY with JSON. If the expression refers to a resolvable date or range, respond with {{"date":"YYYY-MM-DD","endDate":"YYYY-MM-DD" or null,"precision":"day" or "month" or "year"}}. If it is a relative expression (like "yesterday") and the anchor is UNKNOWN, respond with {{"unresolved":true,"reason":"missing_anchor"}}. If it cannot be resolved, respond with {{"unresolved":true,"reason":"ambiguous"}}. Do not include any other text.