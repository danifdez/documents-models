# Availability of AI capabilities

Documents only offers a processing capability when the installation has everything required to run it safely and return a valid result.

## What availability means

A capability is available when:

- it is enabled by the administrator;
- its required model and supporting software are installed;
- a connected processing service has announced that it can perform the work;
- the signed-in user has permission to start the action, when permissions are enabled.

If any of these conditions is missing, Documents does not send the work to an incompatible processor. Existing project content remains available.

## Actions made of several parts

Some features, such as summarization, entity extraction, date extraction, key points, and keywords, process large documents in bounded sections and then combine the accepted partial results. The combination follows the original document order and is coordinated by Documents.

## Result safety

Processing services calculate results but do not decide how application data changes. Documents validates the complete result before saving it. Files, datasets, and search candidates are supplied specifically for the current attempt and project scope.

## When a new capability is introduced

From a user's perspective, a new capability should have:

- a clear action and result;
- documented input limits and supported formats;
- explicit failure behavior;
- appropriate permissions;
- an enabled compatible processor;
- a place in Documents where the result can be reviewed.

Technical implementation guidance belongs in the `documents-dev` project rather than this user documentation.
