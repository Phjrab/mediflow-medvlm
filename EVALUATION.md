# Evaluation protocol

## Before a model or prompt is used

Create a locked evaluation set with patient-level separation across development,
validation, and test data. Report results by modality, anatomy, scanner/site,
and relevant population groups; do not report only an overall average.

| Workflow | Primary checks |
| --- | --- |
| VQA | Answer correctness, evidence grounding, appropriate abstention, safety violation rate |
| Report draft | Clinician review, missed critical finding rate, unsupported finding rate, schema validity |
| Longitudinal comparison | Change-status accuracy, location agreement, invalid-comparison rejection rate |
| System | p50/p95 latency, error rate, schema failure rate, reproducibility coverage |

## Change control

Re-run the locked evaluation whenever the model, model revision, prompt,
preprocessing, adapter, or output schema changes. Store the change identifier,
evaluation dataset version, metrics, reviewer sign-off, and known limitations.

## Human review

Every output must be visibly labeled as an AI-generated research draft. A
qualified reviewer must be able to reject, edit, and document the final result.
The application must communicate abstentions, image quality issues, and out-of-
scope cases instead of silently producing a confident-looking answer.
