from datetime import date
import unittest

from mediflow_medvlm import InferenceRequest, InferenceResult, RequestValidationError, TaskType
from mediflow_medvlm.domain import ImageReference
from mediflow_medvlm.workflows import run_workflow


class FakeAdapter:
    def __init__(self, content: dict[str, object]) -> None:
        self.content = content
        self.instruction = ""

    def generate(self, request: InferenceRequest, system_instruction: str) -> InferenceResult:
        self.instruction = system_instruction
        return InferenceResult(
            content=self.content,
            model_id="test-model",
            model_revision="test-revision",
            prompt_version=request.prompt_version,
            latency_ms=10,
        )


def vqa_request() -> InferenceRequest:
    return InferenceRequest(
        study_id="research-study-002",
        task=TaskType.VQA,
        current_images=(ImageReference(uri="file:///deidentified/image.png", acquired_at=date(2026, 1, 1)),),
        question="What finding is visible?",
        prompt_version="vqa-1",
    )


class WorkflowTests(unittest.TestCase):
    def test_vqa_workflow_returns_validated_result(self) -> None:
        adapter = FakeAdapter({"answer": "Indeterminate", "evidence": "Image observation", "limitations": "Research only"})

        result = run_workflow(adapter, vqa_request())

        self.assertEqual(result.model_id, "test-model")
        self.assertIn("research-only", adapter.instruction)
        self.assertIn("answer, evidence, and limitations", adapter.instruction)

    def test_workflow_rejects_incomplete_model_result(self) -> None:
        adapter = FakeAdapter({"answer": "Possible finding", "evidence": "Image observation"})

        with self.assertRaisesRegex(RequestValidationError, "missing required fields: limitations"):
            run_workflow(adapter, vqa_request())
