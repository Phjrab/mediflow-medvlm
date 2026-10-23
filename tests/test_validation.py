from datetime import date
import unittest

from mediflow_medvlm import InferenceRequest, RequestValidationError, TaskType, validate_request
from mediflow_medvlm.domain import ImageReference


def request_for(task: TaskType = TaskType.VQA, **changes: object) -> InferenceRequest:
    values: dict[str, object] = {
        "study_id": "research-study-001",
        "task": task,
        "current_images": (ImageReference(uri="file:///deidentified/image.png", acquired_at=date(2026, 1, 1)),),
        "question": "What finding is visible?",
    }
    values.update(changes)
    return InferenceRequest(**values)  # type: ignore[arg-type]


class RequestValidationTests(unittest.TestCase):
    def test_text_qa_accepts_question_without_dummy_image(self):
        validate_request(request_for(task=TaskType.TEXT_QA, current_images=(), context="Public educational context", options={"A": "yes", "B": "no"}))

    def test_image_tasks_still_reject_missing_current_image(self):
        for task in (TaskType.VQA, TaskType.REPORT, TaskType.LONGITUDINAL_COMPARISON):
            with self.subTest(task=task), self.assertRaisesRegex(RequestValidationError, "current image"):
                validate_request(request_for(task=task, current_images=()))

    def test_text_requires_question_and_rejects_images(self):
        with self.assertRaisesRegex(RequestValidationError, "requires a question"):
            validate_request(request_for(task=TaskType.TEXT_QA, current_images=(), question="  "))
        with self.assertRaisesRegex(RequestValidationError, "does not accept images"):
            validate_request(request_for(task=TaskType.TEXT_QA))

    def test_valid_vqa_request_passes(self) -> None:
        validate_request(request_for())

    def test_vqa_requires_question(self) -> None:
        with self.assertRaisesRegex(RequestValidationError, "requires a question"):
            validate_request(request_for(question=None))

    def test_comparison_requires_prior_image(self) -> None:
        with self.assertRaisesRegex(RequestValidationError, "requires at least one prior"):
            validate_request(request_for(task=TaskType.LONGITUDINAL_COMPARISON))

    def test_rejects_identifier_parameter(self) -> None:
        with self.assertRaisesRegex(RequestValidationError, "direct identifiers"):
            validate_request(request_for(model_parameters={"mrn": "12345"}))

    def test_rejects_mrn_in_question(self) -> None:
        with self.assertRaisesRegex(RequestValidationError, "direct patient identifier"):
            validate_request(request_for(question="MRN: 12345의 소견은?"))
