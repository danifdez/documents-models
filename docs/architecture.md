# How AI processing works

The Models service provides the processing capabilities used by Documents for extraction, language work, semantic search, question answering, and dataset analysis. It works in the background and does not provide a separate user interface.

## From an action to a result

When a user starts an AI-assisted action:

1. Documents records the action and determines the work it requires.
2. An available processing service accepts a compatible part of that work.
3. The service uses only the content and project-scoped material supplied for that action.
4. Documents validates and stores the result.
5. The desktop application receives a completion or failure notification.

Large documents may be processed in several parts. Documents combines them in their original order and does not present a partial result as complete.

## Available capacity

An installation can have one or more processing services, each with a limited number of simultaneous actions. When all capacity is in use, new actions remain queued. Different services may provide different capabilities, so one feature can be available while another is temporarily unavailable.

## Data boundaries

The processing service does not browse the application database or expand a search to other projects. Documents supplies the exact text, file, dataset snapshot, search candidates, or entity relationships needed for the current action.

Calculated results return to Documents, which decides whether and where to save them. See [Data and privacy](./database.md).

## Interruptions and recovery

Documents keeps the authoritative action state. If processing is interrupted, unfinished work can be retried. A result from an expired attempt is ignored, and delivering the same accepted result twice does not apply it twice.

Each processing slot owns an isolated, reusable handler process. Models polls control while work is active and terminates that process when Documents reports cancellation, the lease is rejected or the local lease window expires. The slot starts a clean replacement process before accepting more work. A healthy process stays alive between assignments so loaded handler state can be reused.
