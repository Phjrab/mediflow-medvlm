# Architecture

## Boundary

Mediflow MedVLM is a research-oriented orchestration layer, not a diagnostic
device. It accepts only de-identified, validated imaging references and returns
structured AI drafts for human review.

```text
Approved ingest/de-identification
              |
              v
InferenceRequest --> validation --> workflow --> MedicalVLM adapter
                                      |                 |
                                      v                 v
                              output-schema check   local/API model
                                      |
                                      v
                            reviewable InferenceResult
```

## Module responsibilities

| Module | Responsibility | Must not do |
| --- | --- | --- |
| `domain` | Immutable cross-provider request/result contracts | Store PHI or provider-specific payloads |
| `validation` | Enforce task requirements and public input policy | Replace approved de-identification |
| `adapters` | Define the local/remote model integration boundary | Embed workflow or clinical policy |
| `workflows` | Supply safe task instruction and validate schema | Make final clinical decisions |

## Adapter implementation rules

An adapter must implement `generate(request, system_instruction)` and return an
`InferenceResult`. It must set the model identifier, model revision when known,
the exact request prompt version, and measured latency. The adapter must not log
image bytes, request questions, or raw model output by default.

External API adapters require an explicit data-transfer review before use. They
must apply timeouts, payload limits, credential injection from a secret manager,
and an allowlist of endpoint hosts.

## Adding a task

1. Add the task to `TaskType`.
2. Define its minimum validated input in `validate_request`.
3. Define its required non-empty output fields in `_REQUIRED_FIELDS`.
4. Add task instruction and tests for success and schema rejection.
5. Update the evaluation protocol before exposing it to users.
