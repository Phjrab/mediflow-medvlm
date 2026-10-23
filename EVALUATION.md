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

## Public research protocol v1

`python -m mediflow_medvlm evaluation freeze` locks an answer-independent sample
selection, source file digests, input/reference digests and generation/parser
policy. The evaluator passes only question/context/options/verified images to
the model; scoring joins references afterwards. Test is used after validation
selection. Data patient grouping is recorded at its actual available level.

Scheduled, attempted, completed, valid, abstained, invalid and technical failure
counts remain visible. Classification accuracy uses either all scheduled rows or
valid answers with explicit denominators. Free QA/VQA exact match and token F1
are lexical metrics, not clinical correctness. Paired group bootstrap reflects
evaluation-sample uncertainty, not variation across training seeds. Benchmark
expert review remains `not_performed` unless a real reviewer completes it.
